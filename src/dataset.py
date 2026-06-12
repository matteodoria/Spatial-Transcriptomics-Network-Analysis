import numpy as np
import pandas as pd
import zarr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from src.utils import get_logger, load_config

log = get_logger("dataset")


def open_zarr(path: str) -> zarr.Group:
    store = zarr.open(path, mode="r")
    log.info(f"Zarr aperto: {path}")
    log.info(f"Chiavi: {list(store['numerical'].keys()) + list(store['labels'].keys())}")
    n = store["numerical/HE_embeddings"].shape[0]
    log.info(f"N celle totali: {n:,}")
    return store

def load_subset_indices(store: zarr.Group, cfg: dict) -> np.ndarray:
    """
    Restituisce gli indici delle celle da usare.
    Se subset_n è impostato → campiona casualmente subset_n celle.
    Se subset_n è None → usa tutte le 8M celle.
    """
    n_total  = store[cfg["data"]["zarr_keys"]["he_emb"]].shape[0]
    subset_n = cfg["data"]["subset_n"]
    seed     = cfg["data"]["random_seed"]

    if subset_n and subset_n < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=subset_n, replace=False)
        idx.sort()   # ordinati → lettura zarr più veloce
        log.info(f"Subset: {subset_n:,} / {n_total:,} celle")
    else:
        idx = np.arange(n_total)
        log.info(f"Tutte le celle: {n_total:,}")

    return idx

def load_features(
    store: zarr.Group,
    idx: np.ndarray,
    cfg: dict,
    neighbors_npz_path: str | None = None,
) -> dict:
    """
    Carica le feature per gli indici selezionati.

    Se `neighbors_npz_path` è fornito, usa i neighbors già corretti
    (same patient + same sample) salvati in .npz.
    Altrimenti usa i neighbors presenti nel file zarr.

    Returns
    -------
    feats : dict
        Dizionario con:
        - he          : (N, 1280) float32
        - rna         : (N, G) float32
        - morph       : (N, 4) float32
        - positions   : (N, 2) float32
        - cell_types  : (N, 8) bool
        - train_mask  : (N,) bool
        - patient     : (N,) int
        - sample      : (N,) int
        - nn_ids      : (N, 8) int32
        - nn_radius   : (N, 8) float32
        - nn_angles   : (N, 8) float32
    """
    zk = cfg["data"]["zarr_keys"]
    log.info("Caricamento feature dal zarr ...")

    feats = {
        "he": store[zk["he_emb"]].oindex[idx].astype(np.float32),
        "rna": store[zk["rna"]].oindex[idx].astype(np.float32),
        "morph": np.column_stack([
            store[zk["area"]].oindex[idx],
            store[zk["eccentricity"]].oindex[idx],
            store[zk["extent"]].oindex[idx],
            store[zk["perimeter"]].oindex[idx],
        ]).astype(np.float32),
        "positions": store[zk["positions"]].oindex[idx].astype(np.float32),
        "cell_types": store[zk["cell_types"]].oindex[idx],   # (N, 8) bool
        "train_mask": store[zk["train_mask"]].oindex[idx],   # (N,) bool
        "patient": store[zk["patient"]].oindex[idx],
        "sample": store[zk["sample"]].oindex[idx],
    }

    # ---------------------------
    # Neighbors: cache .npz oppure zarr grezzo
    # ---------------------------
    if neighbors_npz_path is not None:
        log.info(f"Caricamento neighbors corretti da cache: {neighbors_npz_path}")
        nn_data = np.load(neighbors_npz_path)

        # usa i nomi espliciti nuovi
        feats["nn_ids"] = nn_data["neighbors_idxs"][idx].astype(np.int32)
        feats["nn_radius"] = nn_data["neighbors_radius"][idx].astype(np.float32)
        feats["nn_angles"] = nn_data["neighbors_angle"][idx].astype(np.float32)

        # sanity check opzionale: i metadati devono matchare
        if "patient_ids" in nn_data:
            cached_patient = nn_data["patient_ids"][idx]
            assert np.array_equal(cached_patient, feats["patient"]), \
                "Mismatch tra patient nel cache neighbors e patient nel zarr"

        if "sample_ids" in nn_data:
            cached_sample = nn_data["sample_ids"][idx]
            assert np.array_equal(cached_sample, feats["sample"]), \
                "Mismatch tra sample nel cache neighbors e sample nel zarr"

    else:
        log.warning("Usando neighbors grezzi dal zarr (non necessariamente corretti same-sample)")
        feats["nn_ids"] = store[zk["nn_ids"]].oindex[idx].astype(np.int32)
        feats["nn_radius"] = store[zk["nn_dists"]].oindex[idx].astype(np.float32)
        feats["nn_angles"] = store[zk["nn_angles"]].oindex[idx].astype(np.float32)

    log.info(f"  he:         {feats['he'].shape}")
    log.info(f"  rna:        {feats['rna'].shape}")
    log.info(f"  morph:      {feats['morph'].shape}")
    log.info(f"  positions:  {feats['positions'].shape}")
    log.info(f"  cell_types: {feats['cell_types'].shape}")
    log.info(f"  train_mask: {feats['train_mask'].shape}")
    log.info(f"  patient:    {feats['patient'].shape}")
    log.info(f"  sample:     {feats['sample'].shape}")
    log.info(f"  nn_ids:     {feats['nn_ids'].shape}")
    log.info(f"  nn_radius:  {feats['nn_radius'].shape}")
    log.info(f"  nn_angles:  {feats['nn_angles'].shape}")

    return feats

def remap_nn_ids(nn_ids: np.ndarray, global_idx: np.ndarray, padding: int = -1) -> np.ndarray:
    """
    Rimappa gli indici globali dei vicini agli indici locali del subset.
    I vicini non presenti nel subset diventano padding (-1).
    """
    global_to_local = {int(g): i for i, g in enumerate(global_idx)}
    local = np.full_like(nn_ids, fill_value=padding, dtype=np.int32)

    n_valid = 0
    for i in range(nn_ids.shape[0]):
        for k in range(nn_ids.shape[1]):
            g = int(nn_ids[i, k])
            if g == padding:
                continue
            l = global_to_local.get(g, padding)
            local[i, k] = l
            if l != padding:
                n_valid += 1

    log.info(f"nn_ids rimappati: {n_valid:,} vicini validi")
    return local


def prepare_cell_types(cell_types: np.ndarray) -> np.ndarray:
    """
    Converte one-hot (N, 8) bool → label integers (N,) int8.
    cell_types.argmax(axis=1).
    """
    labels = cell_types.argmax(axis=1).astype(np.int8)

    # Verifica one-hot
    assert np.all(cell_types.sum(axis=1) == 1), "Non one-hot encoding!"

    n_types = len(np.unique(labels))
    log.info(f"Cell types estratti: {n_types} tipi unici")
    return labels


def prepare_target(rna: np.ndarray, hvg_path: str,
                   top_k: int = 2000) -> tuple:
    """
    Filtra RNA ai top-K HVG e restituisce target + gene names.
    """
    hvg_df = pd.read_csv(hvg_path, sep="\t")
    hvg_df = hvg_df.rename(columns={"Unnamed: 0": "gene_name"})

    # Top-K per highly_variable_rank più basso (più variabili)
    top_hvg = hvg_df.nsmallest(top_k, "highly_variable_rank")
    hvg_idx = top_hvg.index.tolist()

    target = rna[:, hvg_idx].astype(np.float32)

    gene_names = top_hvg["gene_name"].tolist()
    log.info(f"Target preparato: {target.shape} (top-{top_k} HVG)")
    log.info(f"Top 5 geni: {gene_names[:5]}")

    return target, gene_names