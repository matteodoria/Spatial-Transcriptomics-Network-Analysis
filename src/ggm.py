"""
Per-cell-type Gaussian Graphical Models on HVG expression.

For each cell type:
  - Subsample cells (class-balanced, patient-stratified within class).
  - Standardize HVG expression in-subsample.
  - Fit a Graphical Lasso → sparse precision matrix.

Provides utilities for the subsampling, fitting, and downstream
network analysis (hubs, edge overlap, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.covariance import GraphicalLasso

try:
    from . import data_io
except ImportError:
    import data_io


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_cells_per_label(
    G,
    labels: np.ndarray,
    train_patients: Sequence[int],
    in_mask: np.ndarray,
    n_per_label: int = 5000,
    seed: int = 42,
    label_values: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Patient-stratified subsampling within each label group.

    Parameters
    ----------
    G : GlobalArrays (used for G.patient only)
    labels : (n_cells,) integer array, one entry per cell. May include -1
        for "unassigned" (those cells are excluded).
    train_patients : list of patient IDs to draw from
    in_mask : bool array of length n_cells (typically G.train_mask)
    n_per_label : target cells per label value
    seed : RNG seed
    label_values : if None, uses sorted unique labels >= 0. If specified, only
        these label values are included.

    Returns
    -------
    dict mapping label_value -> array of global cell indices.
    """
    rng = np.random.default_rng(seed)
    is_train = np.isin(G.patient, train_patients) & in_mask & (labels >= 0)
    if label_values is None:
        label_values = sorted(int(v) for v in np.unique(labels) if v >= 0)

    out: dict[int, np.ndarray] = {}
    for v in label_values:
        candidates = np.where(is_train & (labels == v))[0]
        if len(candidates) == 0:
            out[v] = np.array([], dtype=np.int64)
            continue

        cand_patients = G.patient[candidates]
        unique_pat, counts = np.unique(cand_patients, return_counts=True)
        n_patients = len(unique_pat)

        if n_patients * 1 >= n_per_label:
            per_patient = np.ones(n_patients, dtype=int)
        else:
            base_alloc = (counts / counts.sum() * n_per_label).astype(int)
            base_alloc = np.maximum(base_alloc, 1)
            while base_alloc.sum() > n_per_label:
                idx = np.argmax(base_alloc)
                base_alloc[idx] -= 1
            while base_alloc.sum() < n_per_label:
                idx = np.argmin(base_alloc / counts)
                if base_alloc[idx] < counts[idx]:
                    base_alloc[idx] += 1
                else:
                    break
            per_patient = base_alloc

        per_patient = np.minimum(per_patient, counts)

        chosen = []
        for p, n in zip(unique_pat, per_patient):
            p_mask = cand_patients == p
            p_idxs = candidates[p_mask]
            picked = rng.choice(p_idxs, size=n, replace=False)
            chosen.append(picked)
        out[v] = np.concatenate(chosen)
        rng.shuffle(out[v])
    return out


# Backwards-compatible alias for Part 4 code
def subsample_cells_per_class(
    G,
    train_patients: Sequence[int],
    in_mask: np.ndarray,
    n_per_class: int = 5000,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Subsample by G.cell_type. Wrapper around subsample_cells_per_label."""
    return subsample_cells_per_label(
        G=G,
        labels=G.cell_type,
        train_patients=train_patients,
        in_mask=in_mask,
        n_per_label=n_per_class,
        seed=seed,
        label_values=list(range(8)),
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_glasso(
    X: np.ndarray,
    alpha: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    min_std: float = 0.05,
    drop_low_var_genes: bool = True,
    silent_gene_threshold: float = 1e-6,
) -> dict:
    """
    Fit a Graphical Lasso. Two-tier gene handling:
    - `silent_gene_threshold` genes (literally constant or near-constant) are
      ALWAYS dropped from the fit then re-expanded to all-zero rows/cols.
      This is needed for numerical stability.
    - If `drop_low_var_genes`, genes with std < min_std are also dropped
      (and reported in `dropped_low_var`) for the fit. Their rows/cols
      are also re-expanded to all-zero.

    Returns the precision matrix in the ORIGINAL gene space (n_genes, n_genes),
    with all-zero entries for silent and dropped genes.
    """
    gene_std = X.std(axis=0)
    n_genes_orig = X.shape[1]

    # Silent genes — ALWAYS dropped for numerical safety
    silent_mask = gene_std < silent_gene_threshold

    # Low-variance genes — dropped if drop_low_var_genes
    if drop_low_var_genes:
        lowvar_mask = (gene_std < min_std) & ~silent_mask
    else:
        lowvar_mask = np.zeros(n_genes_orig, dtype=bool)

    excluded_mask = silent_mask | lowvar_mask
    kept_mask = ~excluded_mask
    kept_idxs = np.where(kept_mask)[0]
    excluded_idxs = np.where(excluded_mask)[0]
    silent_idxs = np.where(silent_mask)[0]
    lowvar_idxs = np.where(lowvar_mask)[0]

    if len(kept_idxs) < 2:
        raise ValueError(
            f"Only {len(kept_idxs)} genes survive filtering; cannot fit GLasso"
        )

    X_keep = X[:, kept_idxs]

    # Standardize with std floor
    g_mean = X_keep.mean(axis=0, keepdims=True)
    g_std = np.maximum(X_keep.std(axis=0, keepdims=True), min_std)
    X_std = ((X_keep - g_mean) / g_std).astype(np.float64)

    model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol, assume_centered=True)
    model.fit(X_std)

    # Re-expand precision and covariance to (n_genes_orig, n_genes_orig)
    precision_full = np.zeros((n_genes_orig, n_genes_orig), dtype=np.float64)
    covariance_full = np.zeros((n_genes_orig, n_genes_orig), dtype=np.float64)
    ix_grid = np.ix_(kept_idxs, kept_idxs)
    precision_full[ix_grid] = model.precision_
    covariance_full[ix_grid] = model.covariance_
    # Excluded genes get a 1.0 on the diagonal (so they're identifiable, not all-zero)
    for i in excluded_idxs:
        precision_full[i, i] = 1.0
        covariance_full[i, i] = 1.0

    p = precision_full.shape[0]
    off_diag_mask = ~np.eye(p, dtype=bool)
    off_diag = precision_full[off_diag_mask]
    n_edges = int((np.abs(off_diag) > 1e-10).sum() / 2)
    n_possible_edges = p * (p - 1) // 2
    sparsity = n_edges / n_possible_edges

    return {
        "precision_": precision_full,
        "covariance_": covariance_full,
        "alpha": alpha,
        "n_edges": n_edges,
        "n_possible_edges": n_possible_edges,
        "sparsity": sparsity,
        "n_iter": int(model.n_iter_),
        "converged": bool(model.n_iter_ < max_iter),
        "kept_genes": kept_idxs,
        "excluded_genes": excluded_idxs,
        "silent_genes": silent_idxs,
        "dropped_low_var": lowvar_idxs,
        "n_kept_genes": int(len(kept_idxs)),
        "n_silent_genes": int(len(silent_idxs)),
        "n_lowvar_genes": int(len(lowvar_idxs)),
    }

def sweep_alpha_for_target_sparsity(
    X: np.ndarray,
    target_sparsity: float = 0.04,
    alpha_min: float = 0.01,
    alpha_max: float = 1.0,
    n_alphas: int = 10,
    max_iter: int = 200,
    verbose: bool = True,
) -> list[dict]:
    """
    Sweep alpha logarithmically, return fits sorted by alpha (ascending).
    Each fit reports its achieved sparsity. The caller picks the one closest
    to target_sparsity.

    Higher alpha → sparser network. Lower alpha → denser.
    """
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
    results = []
    for i, a in enumerate(alphas):
        if verbose:
            print(f"  [{i+1}/{n_alphas}] alpha={a:.4f}...", end=" ", flush=True)
        try:
            r = fit_glasso(X, alpha=a, max_iter=max_iter)
            results.append(r)
            if verbose:
                print(f"sparsity={r['sparsity']:.4f}, "
                      f"n_edges={r['n_edges']}, "
                      f"iters={r['n_iter']}{'' if r['converged'] else ' (NOT converged)'}")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            continue
    return results


# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------

def precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """Convert precision matrix to partial correlation matrix.
    Partial corr (i, j) = -P[i,j] / sqrt(P[i,i] * P[j,j])."""
    d = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial


def network_degree(precision: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Per-gene degree: count of nonzero off-diagonal entries in the row."""
    abs_p = np.abs(precision)
    np.fill_diagonal(abs_p, 0)
    return (abs_p > threshold).sum(axis=1)


def jaccard_edge_overlap(p1: np.ndarray, p2: np.ndarray, threshold: float = 1e-10) -> float:
    """
    Jaccard overlap of edge sets between two precision matrices.
    Edges are defined as |P[i,j]| > threshold for i < j.
    """
    n = p1.shape[0]
    abs_p1 = np.abs(p1); np.fill_diagonal(abs_p1, 0)
    abs_p2 = np.abs(p2); np.fill_diagonal(abs_p2, 0)
    # Upper triangular only
    triu = np.triu(np.ones(n, dtype=bool), k=1)
    e1 = abs_p1[triu] > threshold
    e2 = abs_p2[triu] > threshold
    intersect = int((e1 & e2).sum())
    union = int((e1 | e2).sum())
    return intersect / union if union > 0 else 0.0


def hub_genes(precision: np.ndarray, top_k: int = 20, threshold: float = 1e-10) -> np.ndarray:
    """Return gene indices sorted by degree (descending), top-k."""
    degree = network_degree(precision, threshold)
    return degree.argsort()[::-1][:top_k]

# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def bootstrap_class_ggm(
    X: np.ndarray,
    alpha: float,
    n_bootstraps: int = 10,
    subsample_frac: float = 1.0,
    seed: int = 42,
    max_iter: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Refit GLasso on different subsamples of the same class data.

    Parameters
    ----------
    X : (n_cells, n_genes) full-class subsample matrix.
    alpha : penalty (same as headline fit).
    n_bootstraps : number of bootstrap subsamples.
    subsample_frac : fraction of X to use per bootstrap (default 1.0 = bootstrap
        with replacement at the full size, equivalent to n_cells sampled WITH replacement).
    seed : root RNG seed.
    max_iter : iterations per fit.

    Returns
    -------
    dict with:
        - edge_count : (n_genes, n_genes) int — how many bootstraps had this edge
        - degree_per_bootstrap : (n_bootstraps, n_genes) int
        - n_edges_per_bootstrap : (n_bootstraps,) int
        - sparsity_per_bootstrap : (n_bootstraps,) float
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = X.shape
    sample_size = int(n_cells * subsample_frac)

    edge_count = np.zeros((n_genes, n_genes), dtype=np.int32)
    degree_per_bootstrap = np.zeros((n_bootstraps, n_genes), dtype=np.int32)
    n_edges_per_bootstrap = np.zeros(n_bootstraps, dtype=np.int32)
    sparsity_per_bootstrap = np.zeros(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idxs = rng.choice(n_cells, size=sample_size, replace=True)  # WITH replacement
        Xb = X[idxs]
        try:
            fit = fit_glasso(Xb, alpha=alpha, max_iter=max_iter,
                             drop_low_var_genes=False, min_std=0.05)
        except Exception as e:
            if verbose:
                print(f"  bootstrap {b}: FAILED ({e})")
            continue
        # Mark edges (|precision| > tiny threshold on off-diagonal)
        prec = fit["precision_"].copy()
        np.fill_diagonal(prec, 0)
        edge_mask = np.abs(prec) > 1e-10
        edge_count += edge_mask.astype(np.int32)
        degree_per_bootstrap[b] = edge_mask.sum(axis=1)
        n_edges_per_bootstrap[b] = int(edge_mask.sum() / 2)
        sparsity_per_bootstrap[b] = fit["sparsity"]
        if verbose:
            print(f"  bootstrap {b}: sparsity {fit['sparsity']:.4f}, "
                  f"n_edges {n_edges_per_bootstrap[b]}, "
                  f"iters {fit['n_iter']}{' (NOT conv)' if not fit['converged'] else ''}")

    return {
        "edge_count": edge_count,
        "degree_per_bootstrap": degree_per_bootstrap,
        "n_edges_per_bootstrap": n_edges_per_bootstrap,
        "sparsity_per_bootstrap": sparsity_per_bootstrap,
        "n_bootstraps": n_bootstraps,
    }