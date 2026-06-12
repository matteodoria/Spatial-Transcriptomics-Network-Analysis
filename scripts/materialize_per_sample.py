"""
One-time preprocessing: materialize the zarr into per-sample contiguous files
plus small global arrays. Run once; everything downstream uses the cache.

Usage:
    python scripts/materialize_per_sample.py \
        --zarr-path /path/to/data.zarr \
        --cache-dir cache/ \
        --compress  # optional

Cost: ~15-30 minutes I/O. Output size: ~340 GB uncompressed for per-sample files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Global arrays — everything except expression and H&E
# ---------------------------------------------------------------------------

def save_global_arrays(store: zarr.Group, out_dir: Path) -> dict[str, np.ndarray]:
    """
    Load and save the small global arrays. Returns a dict of the loaded arrays
    so the caller doesn't have to re-load them.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    arrays = {}

    print("Loading global arrays...")

    # Labels
    print("  cell_types (one-hot) -> cell_type (int8)")
    ct_onehot = store["labels/cell_types"][:]
    arrays["cell_type"] = ct_onehot.argmax(axis=1).astype(np.int8)
    np.save(out_dir / "cell_type.npy", arrays["cell_type"])

    # Patients and samples
    for name in ["patient", "sample"]:
        print(f"  {name}")
        arrays[name] = store[f"labels/{name}"][:]
        np.save(out_dir / f"{name}.npy", arrays[name])

    print("  train_mask")
    arrays["train_mask"] = store["labels/idx_to_take_into_training_based_on_grid"][:]
    np.save(out_dir / "train_mask.npy", arrays["train_mask"])

    # Geometry & graph structure
    print("  positions")
    arrays["positions"] = store["numerical/positions"][:]
    np.save(out_dir / "positions.npy", arrays["positions"])

    for name in ["neighbors_idxs", "neighbors_radius", "neighbors_angle"]:
        print(f"  {name}")
        arrays[name] = store[f"numerical/{name}"][:]
        np.save(out_dir / f"{name}.npy", arrays[name])

    # Morphology — stack the four 1D arrays into (N, 4) in a fixed order
    print("  morphology (stacked)")
    morph_names = ["cell_area", "cell_eccentricity", "cell_extent", "cell_perimeter"]
    morph = np.stack(
        [store[f"numerical/{n}"][:] for n in morph_names],
        axis=1,
    ).astype(np.float32)
    arrays["morphology"] = morph
    np.save(out_dir / "morphology.npy", morph)
    with open(out_dir / "morphology_columns.json", "w") as f:
        json.dump(morph_names, f, indent=2)

    print(f"Global arrays saved to {out_dir}")
    return arrays

# ---------------------------------------------------------------------------
# Per-sample index mapping
# ---------------------------------------------------------------------------

# def compute_sample_offsets(sample: np.ndarray) -> dict[int, list[int]]:
#     """
#     For each sample ID, return the (sorted) list of global indices belonging
#     to that sample. We saw earlier that samples are mostly NOT contiguous in
#     the array, so we have to enumerate.
#     """
#     sample_ids = np.unique(sample).tolist()
#     offsets = {}
#     for s in sample_ids:
#         idx = np.where(sample == s)[0]
#         offsets[int(s)] = idx.tolist()
#     return offsets

def compute_sample_offsets(sample: np.ndarray) -> dict[int, np.ndarray]:
    """For each sample ID, return the (sorted) array of global indices."""
    return {int(s): np.where(sample == s)[0] for s in np.unique(sample)}


# def save_sample_offsets(offsets: dict[int, list[int]], out_dir: Path) -> None:
#     """
#     Save offsets in two formats:
#     - JSON: human-readable, lists each sample's indices.
#       (CAUTION: 8M ints in JSON is ~80 MB — kept anyway for transparency.)
#     - NPZ: compact, fast to load — one array per sample.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # NPZ form: keys are 'sample_NNN', values are int32 arrays
#     npz_payload = {
#         f"sample_{s:03d}": np.array(idxs, dtype=np.int32)
#         for s, idxs in offsets.items()
#     }
#     np.savez(out_dir / "sample_offsets.npz", **npz_payload)
#
#     # Also keep a small summary JSON: just sample_id -> count
#     summary = {str(s): len(idxs) for s, idxs in offsets.items()}
#     with open(out_dir / "sample_offsets_summary.json", "w") as f:
#         json.dump(summary, f, indent=2)
def save_sample_offsets(offsets: dict[int, np.ndarray], out_dir: Path) -> None:
    """Save offsets as NPZ (compact) + a summary JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        f"sample_{s:03d}": np.asarray(idxs, dtype=np.int32)
        for s, idxs in offsets.items()
    }
    np.savez(out_dir / "sample_offsets.npz", **npz_payload)
    summary = {str(s): int(len(idxs)) for s, idxs in offsets.items()}
    with open(out_dir / "sample_offsets_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------------
# Per-sample materialization (the heavy part)
# ---------------------------------------------------------------------------

def materialize_one_sample(
        store: zarr.Group,
        sample_id: int,
        global_idxs: np.ndarray,
        out_dir: Path,
        compress: bool = False,
) -> dict[str, int]:
    """
    Load expression and H&E for the given global indices, save to a .npz file.
    Returns a small dict with timing/size info.
    """
    out_path = out_dir / f"sample_{sample_id:03d}.npz"

    t0 = time.time()
    expr_dense = store["numerical/expression"].oindex[global_idxs, :]
    t1 = time.time()
    he = store["numerical/HE_embeddings"].oindex[global_idxs, :]
    t2 = time.time()

    # save_fn = np.savez_compressed if compress else np.savez
    # save_fn(out_path, expression=expr, HE_embeddings=he, global_idxs=global_idxs.astype(np.int32))
    # t3 = time.time()
    #
    # size_mb = out_path.stat().st_size / 1e6
    # return {
    #     "n_cells": int(len(global_idxs)),
    #     "expr_seconds": round(t1 - t0, 2),
    #     "he_seconds": round(t2 - t1, 2),
    #     "save_seconds": round(t3 - t2, 2),
    #     "total_seconds": round(t3 - t0, 2),
    #     "file_size_mb": round(size_mb, 1),
    # }
    # Convert to CSR. This is fast (single pass), and the cost is recouped
    # many times over in the storage and load-time savings.
    expr_csr = csr_matrix(expr_dense)
    t3 = time.time()

    np.savez(
        out_path,
        expr_data=expr_csr.data,
        expr_indices=expr_csr.indices,
        expr_indptr=expr_csr.indptr,
        expr_shape=np.asarray(expr_csr.shape, dtype=np.int64),
        HE_embeddings=he,
        global_idxs=global_idxs.astype(np.int32),
    )
    t4 = time.time()

    size_mb = out_path.stat().st_size / 1e6
    return {
        "n_cells": int(len(global_idxs)),
        "nnz": int(expr_csr.nnz),
        "sparsity": float(1 - expr_csr.nnz / (expr_dense.shape[0] * expr_dense.shape[1])),
        "expr_read_seconds": round(t1 - t0, 2),
        "he_read_seconds": round(t2 - t1, 2),
        "csr_convert_seconds": round(t3 - t2, 2),
        "save_seconds": round(t4 - t3, 2),
        "total_seconds": round(t4 - t0, 2),
        "file_size_mb": round(size_mb, 1),
    }


# def materialize_all_samples(
#         store: zarr.Group,
#         sample: np.ndarray,
#         out_dir: Path,
#         compress: bool = False,
# ) -> dict:
#     """Iterate over every sample and materialize. Returns aggregate stats."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     sample_ids = np.unique(sample).tolist()
#     print(f"\nMaterializing {len(sample_ids)} samples to {out_dir}")
#     print(f"  compression: {'on (zip)' if compress else 'off (raw npz)'}")
#
#     stats = {}
#     t_start = time.time()
#     for i, s in enumerate(sample_ids):
#         global_idxs = np.where(sample == s)[0]
#         info = materialize_one_sample(store, int(s), global_idxs, out_dir, compress)
#         stats[int(s)] = info
#         elapsed = time.time() - t_start
#         eta = elapsed / (i + 1) * (len(sample_ids) - i - 1)
#         print(
#             f"  [{i + 1:3d}/{len(sample_ids)}] sample {s:3d}: "
#             f"n={info['n_cells']:>7,}  "
#             f"expr={info['expr_seconds']:>5.1f}s  "
#             f"he={info['he_seconds']:>4.1f}s  "
#             f"save={info['save_seconds']:>5.1f}s  "
#             f"size={info['file_size_mb']:>6.1f}MB  "
#             f"| elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m"
#         )
#
#     total_time = time.time() - t_start
#     total_size = sum(s["file_size_mb"] for s in stats.values())
#     print(f"\nDone. Total: {total_time / 60:.1f} min, {total_size / 1024:.1f} GB")
#     return stats

def materialize_all_samples(
        store: zarr.Group,
        sample: np.ndarray,
        out_dir: Path,
) -> dict:
    """Iterate over every sample and materialize. Returns aggregate stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted(int(s) for s in np.unique(sample))
    print(f"\nMaterializing {len(sample_ids)} samples to {out_dir}")
    print(f"  format: sparse CSR expression + dense H&E (single .npz per sample)")

    stats = {}
    t_start = time.time()
    for i, s in enumerate(sample_ids):
        global_idxs = np.where(sample == s)[0]
        info = materialize_one_sample(store, s, global_idxs, out_dir)
        stats[s] = info
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(sample_ids) - i - 1)
        print(
            f"  [{i + 1:3d}/{len(sample_ids)}] sample {s:3d}: "
            f"n={info['n_cells']:>7,}  "
            f"read={info['expr_read_seconds'] + info['he_read_seconds']:>5.1f}s  "
            f"csr={info['csr_convert_seconds']:>4.1f}s  "
            f"save={info['save_seconds']:>4.1f}s  "
            f"size={info['file_size_mb']:>6.1f}MB  "
            f"| elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m"
        )

    total_time = time.time() - t_start
    total_size = sum(s["file_size_mb"] for s in stats.values())
    print(f"\nDone. Total: {total_time / 60:.1f} min, {total_size / 1024:.1f} GB")

    with open(out_dir / "_materialization_stats.json", "w") as f:
        json.dump({"per_sample": stats,
                   "total_time_seconds": total_time,
                   "total_size_gb": total_size / 1024}, f, indent=2)
    return stats

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", default="/group/sottoriva/andrey.tyshevich/Foundational_Spatial_Model/foundational_model_data.zarr/")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--skip-per-sample", action="store_true")
    parser.add_argument("--only-sample", type=int, default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    global_dir = cache_dir / "global"
    per_sample_dir = cache_dir / "per_sample"
    splits_dir = cache_dir / "splits"

    for d in [global_dir, per_sample_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Opening zarr at {args.zarr_path}")
    store = zarr.open(args.zarr_path, mode="r")

    if args.skip_global:
        print("Skipping global arrays (--skip-global).")
        sample = store["labels/sample"][:]
    else:
        arrays = save_global_arrays(store, global_dir)
        sample = arrays["sample"]

        print("\nComputing sample offsets...")
        offsets = compute_sample_offsets(sample)
        save_sample_offsets(offsets, global_dir)
        print(f"  saved to {global_dir / 'sample_offsets.npz'}")

    if args.skip_per_sample:
        print("Skipping per-sample materialization (--skip-per-sample).")
        return

    if args.only_sample is not None:
        print(f"\nMaterializing only sample {args.only_sample} (test mode)")
        idxs = np.where(sample == args.only_sample)[0]
        info = materialize_one_sample(store, args.only_sample, idxs, per_sample_dir)
        print(f"  done: {info}")
    else:
        materialize_all_samples(store, sample, per_sample_dir)


if __name__ == "__main__":
    main()


