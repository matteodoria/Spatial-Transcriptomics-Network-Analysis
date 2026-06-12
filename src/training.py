"""
Training and evaluation for Part 3 GNN models.

Generic over model class — same loop trains MLP and GraphSAGE.
NeighborLoader is used for any model with edge dependencies (GNN); otherwise
a plain DataLoader on node features.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score, classification_report

try:
    from torch_geometric.loader import NeighborLoader
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int = 8) -> np.ndarray:
    """Inverse-frequency weights, normalized to sum to n_classes."""
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(np.float32)
    w = 1.0 / counts
    w = w * (n_classes / w.sum())
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mlp(
    model, loader, device, y_true=None,
) -> dict:
    """Predict on a TensorDataset DataLoader. Returns metrics + predictions."""
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true_arr = np.concatenate(trues) if y_true is None else y_true
    return {
        "y_pred": y_pred,
        "y_true": y_true_arr,
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true_arr, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }


@torch.no_grad()
def evaluate_gnn(
    model, data, device, eval_mask, batch_size=4096, num_neighbors=(20, 20),
) -> dict:
    """
    Predict for all nodes where eval_mask is True. Uses NeighborLoader to
    sample the receptive field around each target node.

    Note: for a 2-layer GNN we sample 2 hops; num_neighbors needs len == layers.
    """
    if not _HAS_PYG:
        raise ImportError("torch_geometric required for evaluate_gnn")
    model.eval()
    eval_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        input_nodes=eval_mask,
        shuffle=False,
    )
    preds = []
    trues = []
    target_idxs = []
    for batch in eval_loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch.x, batch.edge_index)
        # The first `batch.batch_size` nodes are the targets; the rest are sampled
        # neighbors. We only score the targets.
        target_logits = logits[: batch.batch_size]
        target_y = batch.y[: batch.batch_size]
        target_global = batch.n_id[: batch.batch_size]  # global node ids in original Data
        preds.append(target_logits.argmax(dim=1).cpu().numpy())
        trues.append(target_y.cpu().numpy())
        target_idxs.append(target_global.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "target_global_idxs": np.concatenate(target_idxs),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnn(
    model: nn.Module,
    data,
    device: torch.device,
    train_input_mask=None,           # NEW: defaults to data.train_mask if None
    val_input_mask=None,             # NEW: defaults to data.val_mask if None
    train_batch_size: int = 1024,
    val_batch_size: int = 4096,
    num_neighbors=(20, 20),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 20,
    patience: int = 3,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> dict:
    """
    Train a GNN model with NeighborLoader on (data, train_mask) and eval on (data, val_mask).
    Returns dict with best model state, training history, and final val metrics.
    """
    if not _HAS_PYG:
        raise ImportError("torch_geometric required for train_gnn")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    train_nodes = train_input_mask if train_input_mask is not None else data.train_mask
    train_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=train_batch_size,
        input_nodes=train_nodes,
        shuffle=True,
    )

    history = []
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            # only loss on the target nodes (first batch.batch_size)
            target_logits = logits[: batch.batch_size]
            target_y = batch.y[: batch.batch_size]
            loss = loss_fn(target_logits, target_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # Val
        val_nodes = val_input_mask if val_input_mask is not None else data.val_mask
        val_metrics = evaluate_gnn(
            model, data, device, eval_mask=val_nodes,
            batch_size=val_batch_size, num_neighbors=num_neighbors,
        )
        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_acc": val_metrics["accuracy"],
            "time_s": elapsed,
        }
        history.append(record)
        flag = ""
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            flag = " *best*"
        else:
            patience_counter += 1
        if verbose:
            print(f"epoch {epoch:>2d} | train_loss {train_loss:.4f} | "
                  f"val_f1 {val_metrics['macro_f1']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
                  f"{elapsed:.1f}s{flag}")
        if patience_counter >= patience:
            if verbose:
                print(f"early stop after {patience} epochs without improvement")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval with best model
    val_nodes = val_input_mask if val_input_mask is not None else data.val_mask
    final = evaluate_gnn(
        model, data, device, eval_mask=val_nodes,
        batch_size=val_batch_size, num_neighbors=num_neighbors,
    )

    return {
        "best_state": best_state,
        "history": history,
        "best_val_macro_f1": best_val_f1,
        "final_val_metrics": final,
    }