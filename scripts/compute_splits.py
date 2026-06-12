"""
Compute the patient-level stratified split and save it to cache/splits/.

Run once after materialization (needs cache/global/patient.npy and
cache/global/cell_type.npy). Reproducible: seed=42, n_trials=100_000.

Usage:
    python scripts/compute_splits.py --cache-dir cache/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from splits import compute_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100_000)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    patient = np.load(cache / "global" / "patient.npy")
    cell_type = np.load(cache / "global" / "cell_type.npy")

    print(f"Computing split: seed={args.seed}, n_trials={args.n_trials:,}")
    split = compute_splits(patient, cell_type, seed=args.seed, n_trials=args.n_trials)

    out_path = cache / "splits" / "patient_split.json"
    split.save(out_path)

    print(f"\nbalance score: {split.balance_score:.4f}")
    print(f"train ({len(split.train)}): {split.train}")
    print(f"val   ({len(split.val)}): {split.val}")
    print(f"test  ({len(split.test)}): {split.test}")
    print(f"\nsaved to {out_path}")

    # sanity
    all_p = set(split.train) | set(split.val) | set(split.test)
    assert len(all_p) == 56, "missing/duplicate patients"
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))
    print("sanity checks passed: 56 unique patients, disjoint folds")


if __name__ == "__main__":
    main()
