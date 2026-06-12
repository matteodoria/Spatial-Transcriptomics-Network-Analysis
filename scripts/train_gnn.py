"""
Train a GNN cell-type classifier on the cohort Data. Designed to be invoked
from SLURM via slurm/train_gnn.sbatch.

Usage:
    python scripts/train_gnn.py --model graphsage --rung 3 --modalities hvg \
        --max-epochs 20 --batch-size 2048 --neighbors 20 20 \
        --run-tag rung3_graphsage_hvg

Saves:
    cache/models/{run_tag}.pkl   — full results dict (state, history, val metrics)
    cache/models/{run_tag}.log   — training log (epoch lines + final metrics)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data_io
import preprocessing
import datasets as gnn_datasets
import models as gnn_models
import training as gnn_training
import splits as splits_mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "graphsage", "gat"], required=True)
    p.add_argument("--rung", type=int, required=True, help="Rung number for logging")
    p.add_argument("--modalities", default="hvg",
                   help="Comma-separated: hvg,he,morph (e.g. 'hvg' or 'hvg,he,morph')")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--val-batch-size", type=int, default=4096)
    p.add_argument("--neighbors", type=int, nargs="+", default=[20, 20])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--sage-hidden", type=int, default=256)
    p.add_argument("--gat-hidden", type=int, default=64)
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--run-tag", required=True, help="Identifier for saved outputs")
    return p.parse_args()

# pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cu130.html
def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    print(f"=== Rung {args.rung} ({args.model}) ===")
    print(f"args: {vars(args)}")
    print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve modalities
    modality_set = set(args.modalities.split(","))
    include_he = "he" in modality_set
    include_morph = "morph" in modality_set
    print(f"modalities: hvg={'hvg' in modality_set}, he={include_he}, morph={include_morph}")

    # Load or build cohort Data
    suffix = "_".join(sorted(modality_set))
    cohort_path = cache_dir / "gnn" / f"cohort_{suffix}.pt"
    if cohort_path.exists():
        print(f"loading cached cohort Data from {cohort_path}...")
        t0 = time.time()
        data = gnn_datasets.load_cohort_data(cohort_path)
        print(f"  loaded in {time.time()-t0:.1f}s")
    else:
        print(f"building cohort Data (cache miss)...")
        G = data_io.load_global(cache_dir)
        split = splits_mod.load_splits(str(cache_dir / "splits" / "patient_split.json"))
        prep_cache = cache_dir / "preprocessing"
        hvg_idx = preprocessing.load_hvg(prep_cache)
        morph_scaler = preprocessing.load_morphology_scaler(prep_cache) if include_morph else None
        # H&E scaler: fit on training cells. Done lazily here for full reproducibility.
        if include_he:
            from scipy.sparse import vstack as sparse_vstack
            print("  fitting H&E scaler on training samples...")
            train_sample_ids = sorted(set(int(s) for s in np.unique(G.sample[np.isin(G.patient, split.train)])))
            t_he = time.time()
            sum_he = np.zeros(1280, dtype=np.float64)
            sum_he2 = np.zeros(1280, dtype=np.float64)
            n_he = 0
            train_cell_mask_full = np.isin(G.patient, split.train) & G.train_mask
            for s in train_sample_ids:
                sd = data_io.load_sample(int(s), cache_dir)
                gi = np.where(G.sample == int(s))[0]
                keep = train_cell_mask_full[gi]
                if keep.any():
                    block = sd.HE_embeddings[keep].astype(np.float64)
                    sum_he += block.sum(axis=0)
                    sum_he2 += (block ** 2).sum(axis=0)
                    n_he += block.shape[0]
                    del block
            he_mean = (sum_he / n_he).astype(np.float32)
            he_var = (sum_he2 / n_he) - (sum_he / n_he) ** 2
            he_std = np.sqrt(np.maximum(he_var, 0.0)).astype(np.float32)
            he_std = np.where(he_std < 1e-6, 1.0, he_std)
            print(f"  H&E scaler fit in {time.time() - t_he:.1f}s ({n_he:,} cells)")
        else:
            he_mean = he_std = None
        data = gnn_datasets.build_cohort_data(
            G=G,
            hvg_indices=hvg_idx,
            cache_dir=cache_dir,
            include_he=include_he,
            include_morph=include_morph,
            he_scaler_mean=he_mean,
            he_scaler_std=he_std,
            morph_scaler=morph_scaler,
            train_patients=np.array(split.train),
            val_patients=np.array(split.val),
            test_patients=np.array(split.test),
        )
        gnn_datasets.save_cohort_data(data, cohort_path)

    print(f"\nData summary:")
    print(f"  n_nodes:    {data.num_nodes:,}")
    print(f"  n_edges:    {data.num_edges:,}")
    print(f"  n_features: {data.num_node_features}")
    print(f"  train: {data.train_mask.sum().item():,}, "
          f"val: {data.val_mask.sum().item():,}, "
          f"test: {data.test_mask.sum().item():,}")

    # Class weights from full training set
    y_train = data.y[data.train_mask].numpy()
    class_weights = gnn_training.compute_class_weights(y_train, n_classes=8)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\nclass weights: {class_weights.round(3).tolist()}")

    # Build model
    n_hvg = 1000
    n_he = 1280 if include_he else 0
    n_morph = 4 if include_morph else 0

    if args.model == "mlp":
        model = gnn_models.MLPCellTyper(
            hvg_dim=n_hvg, he_dim=n_he, morph_dim=n_morph,
            dropout=args.dropout,
        )
    elif args.model == "graphsage":
        model = gnn_models.GraphSAGECellTyper(
            hvg_dim=n_hvg, he_dim=n_he, morph_dim=n_morph,
            sage_hidden=args.sage_hidden, sage_layers=len(args.neighbors),
            dropout=args.dropout,
        )
    elif args.model == "gat":
        model = gnn_models.GATCellTyper(
            hvg_dim=n_hvg, he_dim=n_he, morph_dim=n_morph,
            gat_hidden=args.gat_hidden, gat_heads=args.gat_heads,
            gat_layers=len(args.neighbors),
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {args.model}, {n_params:,} params")

    # Train
    t_train = time.time()
    result = gnn_training.train_gnn(
        model=model,
        data=data,
        device=device,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_neighbors=tuple(args.neighbors),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        class_weights=class_weights_t,
        verbose=True,
    )
    train_time = time.time() - t_train
    print(f"\ntotal training time: {train_time/60:.1f} min")
    print(f"best val macro-F1: {result['best_val_macro_f1']:.4f}")

    # Save
    out_dir = cache_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_tag}.pkl"
    payload = {
        "name": args.run_tag,
        "rung": args.rung,
        "model_type": args.model,
        "modalities": sorted(modality_set),
        "args": vars(args),
        "best_state": result["best_state"],
        "history": result["history"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "final_val_metrics": result["final_val_metrics"],
        "n_params": n_params,
        "train_time_s": train_time,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nsaved results to {out_path}")


if __name__ == "__main__":
    main()