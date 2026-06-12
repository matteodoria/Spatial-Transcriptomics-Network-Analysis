"""
Part 1 — spatial graph topological metrics.

Free-tier: cheap per-sample summary statistics. Computed for all 112 samples.
Expensive-tier (betweenness, closeness): separate functions, sampled subset.

All functions take a SpatialGraph (from graph_construction) plus, for
biology-aware metrics, the cell-type vector for the graph's nodes.
"""
from __future__ import annotations

import numpy as np

try:
    from .graph_construction import SpatialGraph, to_igraph
except ImportError:
    from graph_construction import SpatialGraph, to_igraph  # type: ignore

def compute_free_tier_metrics(
    sg: SpatialGraph,
    cell_type: np.ndarray | None = None,
) -> dict:
    """
    Compute the free-tier per-graph topology metrics.

    Parameters
    ----------
    sg : SpatialGraph
        From graph_construction.build_for_sample(...)
    cell_type : (n_nodes,) int array, optional
        Cell-type label per local node. If provided, celltype_assortativity
        is computed. Must align with sg.global_idxs ordering.

    Returns
    -------
    dict with one entry per metric.
    """
    g = to_igraph(sg)
    out = {
        "sample_id": int(sg.sample_id),
        "n_nodes": int(g.vcount()),
        "n_edges": int(g.ecount()),
    }

    # --- degree distribution ---
    deg = np.asarray(g.degree())
    out["mean_degree"] = float(deg.mean())
    out["median_degree"] = float(np.median(deg))
    out["max_degree"] = int(deg.max())
    out["degree_std"] = float(deg.std())
    out["n_isolated"] = int((deg == 0).sum())
    out["frac_isolated"] = float(out["n_isolated"] / out["n_nodes"])

    # --- connected components ---
    comps = g.connected_components(mode="weak")
    sizes = np.array(comps.sizes())
    out["n_components"] = int(len(sizes))
    out["largest_component_size"] = int(sizes.max())
    out["largest_component_frac"] = float(sizes.max() / out["n_nodes"])
    out["n_components_excluding_isolates"] = int((sizes > 1).sum())
    # "Large component" structure — captures fragmentation pattern across samples.
    out["n_components_ge_100"] = int((sizes >= 100).sum())
    out["frac_in_components_ge_100"] = float(sizes[sizes >= 100].sum() / out["n_nodes"])
    out["n_components_ge_500"] = int((sizes >= 500).sum())
    out["frac_in_components_ge_500"] = float(sizes[sizes >= 500].sum() / out["n_nodes"])

    # --- clustering ---
    # Global transitivity: 3 * triangles / connected_triples. One scalar.
    out["transitivity_global"] = float(g.transitivity_undirected(mode="zero"))
    # Average local clustering coefficient (per-node, then averaged).
    # mode='zero' assigns 0 to nodes with degree < 2 (otherwise NaN).
    out["mean_local_clustering"] = float(
        np.mean(g.transitivity_local_undirected(mode="zero"))
    )

    # --- assoratativity ---
    # Degree assortativity: do high-degree nodes connect to high-degree?
    # igraph returns NaN if undefined (e.g. all degrees identical).
    try:
        out["degree_assortativity"] = float(g.assortativity_degree())
    except Exception:
        out["degree_assortativity"] = float("nan")

    # Cell-type assortativity: nominal assortativity on the categorical label.
    if cell_type is not None:
        assert len(cell_type) == out["n_nodes"], (
            f"cell_type length {len(cell_type)} != n_nodes {out['n_nodes']}"
        )
        try:
            out["celltype_assortativity"] = float(
                g.assortativity_nominal(types=cell_type.astype(int).tolist(), directed=False)
            )
        except Exception:
            out["celltype_assortativity"] = float("nan")
    else:
        out["celltype_assortativity"] = float("nan")

    return out


def compute_expensive_tier_metrics(
    sg: "SpatialGraph",
    cell_type: np.ndarray | None = None,
    n_samples_for_centrality: int = 5000,
    sampling_seed: int = 42,
    distance_chunk_size: int = 500,
) -> dict:
    """
    Compute centrality metrics on the giant connected component via
    source/target sampling (Brandes-Pich style). No path cutoff.

    Betweenness: source-sampled, raw counts (paths from sampled sources to
    all targets that pass through each node).
    Closeness: target-sampled, defined as 1 / mean-shortest-path to sampled
    target nodes. Computed in chunks to bound memory.

    Returns scalar summaries plus the full per-node centrality arrays.
    """
    import igraph as ig

    g = to_igraph(sg)
    comps = g.connected_components(mode="weak")
    sizes = np.array(comps.sizes())
    giant_idx = int(sizes.argmax())
    giant_size = int(sizes[giant_idx])
    membership = np.array(comps.membership)
    giant_node_ids = np.where(membership == giant_idx)[0]
    sub = g.subgraph(giant_node_ids.tolist())
    sub_n = sub.vcount()

    rng = np.random.default_rng(sampling_seed)
    n_anchors = max(1, min(sub_n // 10, n_samples_for_centrality))
    anchors = rng.choice(sub_n, size=n_anchors, replace=False).tolist()

    # Sampled betweenness
    bw = np.asarray(sub.betweenness(sources=anchors, directed=False))

    # Sampled closeness, computed in chunks to bound memory
    sum_dist = np.zeros(sub_n, dtype=np.float64)
    count_finite = np.zeros(sub_n, dtype=np.int32)
    for start in range(0, n_anchors, distance_chunk_size):
        chunk = anchors[start:start + distance_chunk_size]
        dm = np.asarray(sub.distances(source=chunk, mode="all"))
        dm = np.where(np.isinf(dm), np.nan, dm)
        mask = (dm > 0) & ~np.isnan(dm)
        sum_dist += np.where(mask, dm, 0.0).sum(axis=0)
        count_finite += mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dist = sum_dist / np.where(count_finite > 0, count_finite, 1)
        cl = np.where(count_finite > 0, 1.0 / mean_dist, 0.0)

    bw_full = np.full(sg.n_nodes, np.nan, dtype=np.float64)
    bw_full[giant_node_ids] = bw
    cl_full = np.full(sg.n_nodes, np.nan, dtype=np.float64)
    cl_full[giant_node_ids] = cl

    return {
        "sample_id": int(sg.sample_id),
        "giant_size": giant_size,
        "giant_frac": giant_size / sg.n_nodes,
        "n_anchors_used": n_anchors,
        "betweenness_max": float(bw.max()),
        "betweenness_p99": float(np.percentile(bw, 99)),
        "betweenness_p95": float(np.percentile(bw, 95)),
        "betweenness_median": float(np.median(bw)),
        "betweenness_mean": float(bw.mean()),
        "closeness_max": float(cl.max()),
        "closeness_median": float(np.median(cl)),
        "closeness_mean": float(cl.mean()),
        "closeness_std": float(cl.std()),
        "betweenness_per_node": bw_full,
        "closeness_per_node": cl_full,
        "giant_node_ids": giant_node_ids,
    }



















