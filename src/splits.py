"""
Patient-level train/val/test split, stratified by cell-type composition.

Used by all parts of the project. Run once via `compute_splits()`, then load
via `load_splits()` everywhere else.
"""
from __future__ import annotations


import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise


@dataclass
class PatientSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    seed: int

    n_trials: int
    balance_score: float

    train_composition: list[float]
    val_composition: list[float]
    test_composition: list[float]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PatientSplit":
        with open(path, "r") as f:
            return cls(**json.load(f))

def _compute_patient_composition(
        patient: np.ndarray,
        cell_type: np.ndarray,
        n_classes: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (patient_ids, compositions) where compositions[i] is the
    cell-type proportion vector for patient_ids[i].
    """
    patient_ids = np.unique(patient)
    comp = np.zeros((len(patient_ids), n_classes), dtype=np.float64)

    for i, p in enumerate(patient_ids):
        mask = patient == p
        counts = np.bincount(cell_type[mask], minlength=n_classes)
        comp[i] = counts / counts.sum()
    return patient_ids, comp

def _balance_score(
        composition: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
) -> float:
    """Max pairwise L1 distance between mean composition of the three folds"""
    means = np.stack([
        composition[train_idx].mean(axis=0),
        composition[val_idx].mean(axis=0),
        composition[test_idx].mean(axis=0),
    ])
    pairwise = [
        np.abs(means[i] - means[j]).sum()
        for i in range(3) for j in range(i + 1, 3)
    ]
    return float(max(pairwise))

def compute_splits(
        patient: np.ndarray,
        cell_type: np.ndarray,
        n_train: int = 40,
        n_val: int = 8,
        n_test: int = 8,
        seed: int = 42,
        n_trials: int = 100_000
) -> PatientSplit:
    """
    Compute a stratified patient-level train/val/test split.

    Approach: random-restart search over n_trials partitions; keep the
    partition minimizing the max L1 distance between fold mean compositions.
    """
    patient_ids, composition = _compute_patient_composition(patient, cell_type)
    n_patients = len(patient_ids)
    assert n_train + n_val + n_test == n_patients, (
        f"sum of split sizes ({n_train} + {n_val} + {n_test}) != n_patients ({n_patients})"
    )

    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_perm = None

    for _ in range(n_trials):
        perm = rng.permutation(n_patients)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        score = _balance_score(composition, train_idx, val_idx, test_idx)
        if score < best_score:
            best_score = score
            best_perm = perm

    train_idx = best_perm[:n_train]
    val_idx = best_perm[n_train:n_train+n_val]
    test_idx = best_perm[n_train+n_val:]

    return PatientSplit(
        train = sorted(int(p) for p in patient_ids[train_idx]),
        val   = sorted(int(p) for p in patient_ids[val_idx]),
        test  = sorted(int(p) for p in patient_ids[test_idx]),
        seed  = seed,
        n_trials = n_trials,
        balance_score = best_score,
        train_composition = composition[train_idx].mean(axis=0).tolist(),
        val_composition   = composition[val_idx].mean(axis=0).tolist(),
        test_composition  = composition[test_idx].mean(axis=0).tolist(),
    )

def load_splits(path: str | Path = "cache/splits/patient_split.json") -> PatientSplit:
    return PatientSplit.load(path)














