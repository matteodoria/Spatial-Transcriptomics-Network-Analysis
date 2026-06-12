"""
Feature preprocessing for Part 3 (and Part 4) — fit on training cells, save artifacts.

Fits:
  - HVG selection on training-patient cells (top K genes via Scanpy seurat flavor)
  - Morphology z-score standardizer on training-mask cells (in_mask=True, train patients)

Outputs to cache/preprocessing/:
  - hvg_indices.npy        : int32 array of selected gene indices (length K)
  - hvg_metadata.json      : selection parameters + fit-set size
  - morphology_scaler.npz  : mean, std (length-4 arrays); plus column names
  - silent_genes.npy       : int32 indices of genes dropped pre-selection
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import vstack as sparse_vstack

try:
    from . import data_io
    from .splits import load_splits
except ImportError:
    import data_io
    from splits import load_splits


# ---------------------------------------------------------------------------
# Gene-level helpers
# ---------------------------------------------------------------------------

def identify_silent_genes(
    sample_ids: Sequence[int],
    cache_dir: str | Path = "cache",
) -> np.ndarray:
    """Find genes with zero expression across all given samples (full data)."""
    cum_max = None
    for s in sample_ids:
        sd = data_io.load_sample(int(s), cache_dir)
        col_max = np.asarray(sd.expression.max(axis=0).todense()).ravel()
        cum_max = col_max if cum_max is None else np.maximum(cum_max, col_max)
    silent = np.where(cum_max == 0)[0].astype(np.int32)
    return silent


def select_hvgs(
    sample_ids: Sequence[int],
    n_top_genes: int = 1000,
    cache_dir: str | Path = "cache",
    flavor: str = "seurat",
    silent_genes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Stack expression for the given samples and select highly-variable genes.

    Returns
    -------
    hvg_indices : (K,) int32 array of selected gene indices into the original
                  16,497-gene matrix.
    metadata    : dict describing what was fitted.
    """
    import anndata as ad
    import scanpy as sc

    print(f"Loading and stacking expression for {len(sample_ids)} samples...")
    matrices = []
    n_cells_total = 0
    for s in sample_ids:
        sd = data_io.load_sample(int(s), cache_dir)
        matrices.append(sd.expression)
        n_cells_total += sd.n_cells
    X = sparse_vstack(matrices, format="csr")
    print(f"  stacked shape: {X.shape}, nnz: {X.nnz:,}, "
          f"density: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")

    # Wrap in AnnData. Scanpy operates on the .X slot.
    adata = ad.AnnData(X=X)

    # Mark silent genes so HVG ranking ignores them
    if silent_genes is not None and len(silent_genes):
        keep_mask = np.ones(adata.shape[1], dtype=bool)
        keep_mask[silent_genes] = False
        adata.var["pass_qc"] = keep_mask

    print(f"running scanpy.pp.highly_variable_genes (flavor={flavor}, n_top_genes={n_top_genes})...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
    )

    hvg_mask = adata.var["highly_variable"].values
    # If we marked silent genes, exclude them post-hoc from selection
    if silent_genes is not None and len(silent_genes):
        hvg_mask = hvg_mask & adata.var["pass_qc"].values

    hvg_indices = np.where(hvg_mask)[0].astype(np.int32)

    metadata = {
        "n_top_genes_requested": int(n_top_genes),
        "n_selected": int(len(hvg_indices)),
        "flavor": flavor,
        "n_fit_samples": int(len(sample_ids)),
        "n_fit_cells": int(n_cells_total),
        "n_silent_excluded": int(len(silent_genes)) if silent_genes is not None else 0,
    }
    return hvg_indices, metadata


# ---------------------------------------------------------------------------
# Morphology standardizer
# ---------------------------------------------------------------------------

def fit_morphology_scaler(
    fit_cell_indices: np.ndarray,
    morphology: np.ndarray,
) -> dict:
    """
    Compute (mean, std) per morphology feature on the given cells.

    Parameters
    ----------
    fit_cell_indices : (n_fit,) global cell indices to use for fitting.
    morphology       : (N, 4) global morphology array.

    Returns
    -------
    dict with keys: mean (4,), std (4,), n_fit_cells, columns (list of names).
    """
    morph_fit = morphology[fit_cell_indices]
    mean = morph_fit.mean(axis=0)
    std = morph_fit.std(axis=0)
    # Guard against tiny std
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_fit_cells": int(len(fit_cell_indices)),
        "columns": ["cell_area", "cell_eccentricity", "cell_extent", "cell_perimeter"],
    }


def apply_morphology_scaler(
    morphology: np.ndarray,
    scaler: dict,
) -> np.ndarray:
    """Apply a previously-fitted scaler to a morphology matrix."""
    return ((morphology - scaler["mean"]) / scaler["std"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_hvg(out_dir: Path, hvg_indices: np.ndarray, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "hvg_indices.npy", hvg_indices)
    with open(out_dir / "hvg_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_silent_genes(out_dir: Path, silent: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "silent_genes.npy", silent)


def save_morphology_scaler(out_dir: Path, scaler: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "morphology_scaler.npz",
        mean=scaler["mean"],
        std=scaler["std"],
        n_fit_cells=scaler["n_fit_cells"],
        columns=np.array(scaler["columns"]),
    )


def load_hvg(in_dir: str | Path = "cache/preprocessing") -> np.ndarray:
    return np.load(Path(in_dir) / "hvg_indices.npy")


def load_morphology_scaler(in_dir: str | Path = "cache/preprocessing") -> dict:
    f = np.load(Path(in_dir) / "morphology_scaler.npz", allow_pickle=False)
    return {
        "mean": f["mean"],
        "std": f["std"],
        "n_fit_cells": int(f["n_fit_cells"]),
        "columns": f["columns"].tolist(),
    }