"""
Build per-sample spatial cell graphs from the precomputed KNN arrays.

Handles the data quirks pre-committed in DESIGN.md §4:
    - -1 padding in neighbors_idxs  -> no edge
    - 454 cross-sample edges        -> filtered
    - duplicate neighbors           -> none in valid entries (verified)
    - isolated cells (66,796)       -> kept as degree-0 nodes
    - KNN asymmetry                 -> symmetrize (union of directed edges)

The core function is library-agnostic: returns numpy arrays in COO edge
format. Thin adapters convert to igraph / PyG / networkx as needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# data_io is in the same package; allow flat-import too for ad-hoc use
try:
    from . import data_io
except ImportError:
    import data_io  # type: ignore

# Core Builder (numpy only)
@dataclass
class SpatialGraph:
    """A per-sample spatial cell graph in library-agnostic form"""
    sample_id:   int
    n_nodes:     int
    edge_index:  np.ndarray         # (2, num_edges) int32, undirected (both dirs)
    edge_weight: np.ndarray | None  # (num_edges,) float32, or None if unweighted
    global_idxs: np.ndarray         # (n_nodes,) int32 - local node i = global cell global_idxs[i]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1] // 2

    @property
    def n_directed_entries(self) -> int:
        return self.edge_index.shape[1]

def build_spatial_graph(
    sample_id: int,
    global_idxs: np.ndarray,
    neighbors_idxs: np.ndarray,
    sample_array: np.ndarray,
    neighbors_radius: np.ndarray | None = None,
    weighted: bool = False,
) -> SpatialGraph:
    """
    Core graph builder. Takes raw global arrays + the indices for one sample,
    returns a SpatialGraph.

    Parameters
    ----------
    sample_id : int
        Sample ID (stored on the returned object; also used for cross-sample
        filtering via `sample_array`).
    global_idxs : (n_cells,) int
        Global indices of the cells in this sample. Local node id i
        corresponds to global cell global_idxs[i].
    neighbors_idxs : (N, 8) int
        Global neighbors array (full dataset). -1 entries are padding.
    sample_array : (N,) int
        Global sample-id array, used to filter cross-sample edges.
    neighbors_radius : (N, 8) float, optional
        Global radius array. Required if weighted=True.
    weighted : bool
        If True, attach edge weights = 1 - mean(radius_forward, radius_reverse)
        where both directions are available; just `1 - r` otherwise.
    """
    if weighted and neighbors_radius is None:
        raise ValueError("Weighted graph requires neighbors_radius")

    n_cells = len(global_idxs)
    n_global = len(sample_array)

    # 1. Build a global -> local index map for THIS sample
    #    gmap[g] = local if for global cell g, or -1 if g is not in the sample
    gmap = np.full(n_global, -1, dtype=np.int32)
    gmap[global_idxs] = np.arange(n_cells, dtype=np.int32)

    # 2. For each cell in this sample, look up its 8 neighbor slots
    nbr = neighbors_idxs[global_idxs]           # (n_cells, 8) int32, may contain -1
    if weighted:
        rad = neighbors_radius[global_idxs]     # (n_cells, 8) float32

    # 3. Filter: (a) drop -1 padding, (b) drop cross-sample edges (neighbor not in this sample)
    src_local = np.broadcast_to(
        np.arange(n_cells, dtype=np.int32)[:, None], nbr.shape
    )
    valid_idx = nbr >= 0
    # Clamp to 0 for the gmap lookup
    dst_local_candidate = gmap[np.where(valid_idx, nbr, 0)]
    valid_in_sample = dst_local_candidate >= 0
    valid = valid_idx & valid_in_sample

    src_directed = src_local[valid]             # (E_dir,)
    dst_directed = dst_local_candidate[valid]   # (E_dir,)
    if weighted:
        rad_directed = rad[valid]               # (E_dir,)

    # 4. Symmetrize: stack (src, dst) and (dst, src).
    #    Some edges appear twice (KNN-mutual), some once (KNN-asymmetric)
    #    De-duplicate tget exactly 2 entries per undirected entries
    src_all = np.concatenate([src_directed, dst_directed])
    dst_all = np.concatenate([dst_directed, src_directed])

    if not weighted:
        # Unique (src, dst) pairs - represented as a single int64 key for speed
        keys = src_all.astype(np.int64) * np.int64(n_cells) + dst_all.astype(np.int64)
        unique_keys = np.unique(keys)
        src_final = (unique_keys // n_cells).astype(np.int32)
        dst_final = (unique_keys %  n_cells).astype(np.int32)

        edge_index = np.stack([src_final, dst_final], axis=0)
        return SpatialGraph(
            sample_id   = sample_id,
            n_nodes     = int(n_cells),
            edge_index  = edge_index,
            edge_weight = None,
            global_idxs = global_idxs.astype(np.int32),
        )

    # Weighted path: each directed entry has a radius; symmetrized entry should
    # combine the forward and reverse radii (when both exist) via the mean.
    # Strategy: build a (src, dst) -> list of radii via groupby on the key.
    rad_all = np.concatenate([rad_directed, rad_directed])
    keys = src_all.astype(np.int64) * np.int64(n_cells) + dst_all.astype(np.int64)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    rad_sorted = rad_all[order]
    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    # mean of radii per unique key: np.add.reduceat
    sums = np.add.reduceat(rad_sorted, starts)
    counts = np.diff(np.concatenate([starts, [len(rad_sorted)]]))
    mean_radii = (sums / counts).astype(np.float32)

    src_final = (unique_keys // n_cells).astype(np.int32)
    dst_final = (unique_keys %  n_cells).astype(np.int32)
    edge_index = np.stack([src_final, dst_final], axis=0)
    edge_weight = (1.0 - mean_radii).astype(np.float32)

    return SpatialGraph(
        sample_id   = int(sample_id),
        n_nodes     = int(n_cells),
        edge_index  = edge_index,
        edge_weight = edge_weight,
        global_idxs = global_idxs.astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Convenience wrapper: build from sample_id + cached global arrays
# ---------------------------------------------------------------------------

def build_for_sample(
        sample_id: int,
        global_arrays: "data_io.GlobalArrays | None" = None,
        cache_dir: str = "cache",
        weighted: bool = False,
) -> SpatialGraph:
    """Convenience: load the data, find the sample, build the graph."""
    if global_arrays is None:
        global_arrays = data_io.load_global(cache_dir)
    global_idxs = np.where(global_arrays.sample == sample_id)[0]
    return build_spatial_graph(
        sample_id=sample_id,
        global_idxs=global_idxs,
        neighbors_idxs=global_arrays.neighbors_idxs,
        sample_array=global_arrays.sample,
        neighbors_radius=global_arrays.neighbors_radius if weighted else None,
        weighted=weighted,
    )


# ---------------------------------------------------------------------------
# Adapter: igraph
# ---------------------------------------------------------------------------

def to_igraph(g: SpatialGraph):
    """
    Convert a SpatialGraph to an igraph.Graph (undirected, simple).

    The igraph object carries vertex attribute `global_idx` so it remains
    self-describing for cross-referencing with global arrays.
    """
    import igraph as ig

    # igraph wants a list of (u, v) tuples for undirected edges, each once.
    # Our edge_index has both directions take only entries with src < dst
    src, dst = g.edge_index
    keep = src < dst
    edges = list(zip(src[keep].tolist(), dst[keep].tolist()))

    graph = ig.Graph(n=g.n_nodes, edges=edges, directed=False)
    graph.vs["global_idx"] = g.global_idxs.tolist()

    if g.edge_weight is not None:
        graph.es["weight"] = g.edge_weight[keep].tolist()


    return graph









