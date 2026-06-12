"""
Loaders for the per-sample cache. Centralizes the file format so callers don't
need to know how expression is stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


DEFAULT_CACHE_DIR = Path("cache")

@dataclass
class SampleData:
    """All per-sample data for one tissue section."""
    sample_id: int
    expression: csr_matrix          # (n_cells, 16497) — sparse
    HE_embeddings: np.ndarray       # (n_cells, 1280) — dense float32
    global_idxs: np.ndarray         # (n_cells,) int32 — index into global arrays

    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    def expression_dense(self) -> np.ndarray:
        """Densify expression. ~3 GB for the largest sample — use carefully."""
        return self.expression.toarray()


def load_sample(
    sample_id: int,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> SampleData:
    """Load one sample's expression + H&E + global indices from the cache."""
    path = Path(cache_dir) / "per_sample" / f"sample_{sample_id:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No cache file for sample {sample_id} at {path}")

    with np.load(path) as f:
        expr = csr_matrix(
            (f["expr_data"], f["expr_indices"], f["expr_indptr"]),
            shape=tuple(f["expr_shape"]),
        )
        he = f["HE_embeddings"]
        gidx = f["global_idxs"]

    return SampleData(
        sample_id=sample_id,
        expression=expr,
        HE_embeddings=he,
        global_idxs=gidx,
    )


@dataclass
class GlobalArrays:
    """Small global arrays loaded once and shared across the project."""
    cell_type: np.ndarray         # (N,) int8
    patient: np.ndarray           # (N,) int8
    sample: np.ndarray            # (N,) int8
    train_mask: np.ndarray        # (N,) bool
    positions: np.ndarray         # (N, 2) float32
    neighbors_idxs: np.ndarray    # (N, 8) int32 — -1 padding
    neighbors_radius: np.ndarray  # (N, 8) float32 — [0, 1]
    neighbors_angle: np.ndarray   # (N, 8) float32
    morphology: np.ndarray        # (N, 4) float32

    @property
    def n_cells(self) -> int:
        return len(self.cell_type)


def load_global(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> GlobalArrays:
    """Load all small global arrays. Cheap — ~600 MB total in RAM."""
    g = Path(cache_dir) / "global"
    return GlobalArrays(
        cell_type=np.load(g / "cell_type.npy"),
        patient=np.load(g / "patient.npy"),
        sample=np.load(g / "sample.npy"),
        train_mask=np.load(g / "train_mask.npy"),
        positions=np.load(g / "positions.npy"),
        neighbors_idxs=np.load(g / "neighbors_idxs.npy"),
        neighbors_radius=np.load(g / "neighbors_radius.npy"),
        neighbors_angle=np.load(g / "neighbors_angle.npy"),
        morphology=np.load(g / "morphology.npy"),
    )


def load_sample_offsets(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict[int, np.ndarray]:
    """Return {sample_id: array of global indices belonging to that sample}."""
    path = Path(cache_dir) / "global" / "sample_offsets.npz"
    with np.load(path) as f:
        return {int(key.split("_")[1]): f[key] for key in f.files}