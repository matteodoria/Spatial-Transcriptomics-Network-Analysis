"""
Cohort-wide PyG Data construction for GNN training.

Builds one big graph with:
  - all 8M cells as nodes
  - HVG expression as node features (n_cells, 1000)
  - all spatial KNN edges (across all 112 samples, ~70M directed edges)
  - cell-type labels and fold masks

Cross-sample edges don't exist by construction, so the cohort graph is the
disjoint union of 112 sample-level graphs. PyG NeighborLoader sees this
correctly and never crosses sample boundaries during sampling.

The Data object is cached as a single .pt file (~33 GB) for fast reload.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from . import data_io, graph_construction, preprocessing
except ImportError:
    import data_io
    import graph_construction
    import preprocessing


def build_cohort_data(
    G,                                  # GlobalArrays
    hvg_indices: np.ndarray,            # (K,) HVG selection
    cache_dir: str | Path = "cache",
    he_scaler_mean: np.ndarray | None = None,
    he_scaler_std: np.ndarray | None = None,
    morph_scaler: dict | None = None,
    include_he: bool = False,
    include_morph: bool = False,
    train_patients: np.ndarray | None = None,
    val_patients: np.ndarray | None = None,
    test_patients: np.ndarray | None = None,
) -> Data:
    """
    Build a cohort-wide PyG Data object.

    Parameters
    ----------
    G : GlobalArrays from data_io.load_global
    hvg_indices : indices into the original gene dimension
    cache_dir : root of the cache
    include_he, include_morph : whether to concatenate those modalities into x
    he_scaler_mean, he_scaler_std : per-dim mean/std for H&E (fit on train)
    morph_scaler : as returned by preprocessing.load_morphology_scaler
    train_patients, val_patients, test_patients : arrays of patient IDs for the folds

    Returns
    -------
    Data with attributes: x, edge_index, y, train_mask, val_mask, test_mask,
    sample_id (per-node), niche_id (per-node, set externally if needed).
    """
    cache_dir = Path(cache_dir)
    n_cells = G.n_cells
    n_hvg = len(hvg_indices)

    # 1. Build the feature matrix
    feature_blocks = []
    print(f"Allocating feature matrix...")
    if include_he:
        n_features = n_hvg + 1280
    else:
        n_features = n_hvg
    if include_morph:
        n_features += 4
    print(f"  total dim per cell: {n_features}")
    print(f"  total tensor size: {n_cells * n_features * 4 / 1e9:.1f} GB (float32)")
    x = np.zeros((n_cells, n_features), dtype=np.float32)

    t0 = time.time()
    sample_ids_sorted = sorted(int(s) for s in np.unique(G.sample))
    for i, s in enumerate(sample_ids_sorted):
        sd = data_io.load_sample(int(s), cache_dir)
        global_idxs_s = sd.global_idxs
        # HVG block
        hvg_block = sd.expression[:, hvg_indices].toarray().astype(np.float32)
        x[global_idxs_s, :n_hvg] = hvg_block
        # H&E block
        if include_he:
            he_block = sd.HE_embeddings.astype(np.float32)
            if he_scaler_mean is not None:
                he_block = (he_block - he_scaler_mean) / he_scaler_std
            x[global_idxs_s, n_hvg:n_hvg + 1280] = he_block
        # Morph block (from G.morphology, applied via scaler)
        if include_morph and morph_scaler is not None:
            morph_block = preprocessing.apply_morphology_scaler(
                G.morphology[global_idxs_s], morph_scaler
            )
            start = n_hvg + (1280 if include_he else 0)
            x[global_idxs_s, start:start + 4] = morph_block
        if (i + 1) % 20 == 0 or i == len(sample_ids_sorted) - 1:
            print(f"  loaded {i+1}/{len(sample_ids_sorted)} samples in {time.time() - t0:.0f}s")

    # 2. Build edges
    print(f"\nbuilding edge index across {len(sample_ids_sorted)} samples...")
    t0 = time.time()
    edges_list = []
    for s in sample_ids_sorted:
        sg = graph_construction.build_for_sample(int(s), global_arrays=G, weighted=False)
        # sg.edge_index uses LOCAL node ids; remap to global
        local2global = sg.global_idxs.astype(np.int64)
        src_global = local2global[sg.edge_index[0]]
        dst_global = local2global[sg.edge_index[1]]
        edges_list.append(np.stack([src_global, dst_global], axis=0))
    edge_index_np = np.concatenate(edges_list, axis=1)
    print(f"  total edges (directed, symmetric): {edge_index_np.shape[1]:,}")
    print(f"  time: {time.time() - t0:.0f}s")

    # 3. Standardize HVG block per modality on train cells (post-load standardization)
    # We do this AFTER assembly to ensure consistency across train/val/test.
    # Chunked to avoid creating a full-size temporary array.
    print(f"\nstandardizing HVG features on training cells...")
    if train_patients is not None:
        train_cell_mask = np.isin(G.patient, train_patients) & G.train_mask
    else:
        train_cell_mask = np.ones(n_cells, dtype=bool)
    # Compute mean/std from training cells. Chunk this too to be safe.
    sum_x = np.zeros(n_hvg, dtype=np.float64)
    sum_x2 = np.zeros(n_hvg, dtype=np.float64)
    n_train = 0
    chunk = 200_000
    train_idxs = np.where(train_cell_mask)[0]
    for start in range(0, len(train_idxs), chunk):
        idx = train_idxs[start:start + chunk]
        block = x[idx, :n_hvg]
        sum_x += block.sum(axis=0)
        sum_x2 += (block.astype(np.float64) ** 2).sum(axis=0)
        n_train += len(idx)
    hvg_mean = (sum_x / n_train).astype(np.float32)
    hvg_var = (sum_x2 / n_train) - (sum_x / n_train) ** 2
    hvg_std = np.sqrt(np.maximum(hvg_var, 0.0)).astype(np.float32)
    hvg_std = np.where(hvg_std < 1e-6, 1.0, hvg_std)

    # Apply in chunks (in place, no full-size temporary)
    chunk = 200_000
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        x[start:end, :n_hvg] = (x[start:end, :n_hvg] - hvg_mean) / hvg_std
    print(f"  HVG post-std (sampled): "
          f"mean={x[train_idxs[:10_000], :n_hvg].mean():.4f}, "
          f"std={x[train_idxs[:10_000], :n_hvg].std():.4f}")

    # 4. Construct PyG Data
    print(f"\nbuilding PyG Data object...")
    train_mask = np.isin(G.patient, train_patients) & G.train_mask if train_patients is not None else np.zeros(n_cells, dtype=bool)
    val_mask = np.isin(G.patient, val_patients) if val_patients is not None else np.zeros(n_cells, dtype=bool)
    test_mask = np.isin(G.patient, test_patients) if test_patients is not None else np.zeros(n_cells, dtype=bool)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index_np.astype(np.int64)),
        y=torch.from_numpy(G.cell_type.astype(np.int64)),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        sample_id=torch.from_numpy(G.sample.astype(np.int64)),
    )
    print(f"  Data summary:")
    print(f"    n_nodes:  {data.num_nodes:,}")
    print(f"    n_edges:  {data.num_edges:,}")
    print(f"    n_features: {data.num_node_features}")
    print(f"    train cells: {train_mask.sum():,}")
    print(f"    val cells:   {val_mask.sum():,}")
    print(f"    test cells:  {test_mask.sum():,}")
    return data


def save_cohort_data(data: Data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"saved cohort data to {path} ({path.stat().st_size / 1e9:.1f} GB)")


def load_cohort_data(path: str | Path) -> Data:
    return torch.load(path, weights_only=False)