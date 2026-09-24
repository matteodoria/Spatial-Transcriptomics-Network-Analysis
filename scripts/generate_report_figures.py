"""
Generate publication-quality figures for the ASLCD project report.

Run from the project root:
    python scripts/generate_report_figures.py

Each figure is produced by an independent function. To regenerate a single
figure, comment out the others in main() or call the function directly.

Outputs to report/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA_DIR = ROOT / "data"
FIGDIR = ROOT / "report" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Add src to path so we can use data_io if needed
sys.path.insert(0, str(ROOT / "src"))


# -----------------------------------------------------------------------------
# Style: clean, print-friendly matplotlib defaults
# -----------------------------------------------------------------------------
def set_style():
    """
    Matplotlib style for report figures.

    Modes:
    - TRANSPARENT = False:
        white paper/PDF, dark ink.
    - TRANSPARENT = True:
        transparent figures for dark deck/background, light ink.

    Important for notebooks:
    rcParams are global and persistent across cells, so we reset them first.
    """
    mpl.rcdefaults()

    # -------------------------------------------------------------------------
    # Ink/background palette
    # -------------------------------------------------------------------------
    if TRANSPARENT:
        ink = LIGHT_INK
        subtle_ink = LIGHT_INK
        bg = "none"
        axes_bg = "none"
        grid_alpha = 0.20
        grid_color = LIGHT_INK
        legend_face = "none"
        legend_edge = "none"
    else:
        ink = "#333333"
        subtle_ink = "#555555"
        bg = "white"
        axes_bg = "white"
        grid_alpha = 0.70
        grid_color = "#dddddd"
        legend_face = "white"
        legend_edge = "#cccccc"

    mpl.rcParams.update({
        # ---------------------------------------------------------------------
        # Figure / save
        # ---------------------------------------------------------------------
        "figure.facecolor": bg,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": bg,
        "savefig.edgecolor": bg,

        # ---------------------------------------------------------------------
        # Fonts
        # ---------------------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,

        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Math text: keeps LaTeX-like labels consistent without requiring LaTeX
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",

        # ---------------------------------------------------------------------
        # Text colors
        # ---------------------------------------------------------------------
        "text.color": ink,
        "axes.titlecolor": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,

        # ---------------------------------------------------------------------
        # Axes
        # ---------------------------------------------------------------------
        "axes.facecolor": axes_bg,
        "axes.edgecolor": ink,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # ---------------------------------------------------------------------
        # Grid
        # ---------------------------------------------------------------------
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.linewidth": 0.5,
        "grid.alpha": grid_alpha,

        # ---------------------------------------------------------------------
        # Ticks
        # ---------------------------------------------------------------------
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # ---------------------------------------------------------------------
        # Lines / patches
        # ---------------------------------------------------------------------
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,

        # ---------------------------------------------------------------------
        # Legend
        # ---------------------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": legend_face,
        "legend.edgecolor": legend_edge,
        "legend.labelcolor": ink,

        # ---------------------------------------------------------------------
        # PDF/SVG text handling
        # Keeps text editable/searchable in vector outputs.
        # ---------------------------------------------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

# -----------------------------------------------------------------------------
# Color palette
# -----------------------------------------------------------------------------
PRIMARY = "#1F4E79"          # matches the report's linkcolor
SECONDARY = "#C45A4F"        # warm red, complementary
ACCENT_GREEN = "#3F8E5C"
ACCENT_GRAY = "#7A7A7A"
ANOMALY_COLOR = "#E07A5F"    # warm orange for highlighting anomalies
CLASS_COLORS = plt.cm.tab10.colors[:8]  # 8 cell-type classes

# Figure a sfondo trasparente per il deck scuro (inchiostro chiaro)
TRANSPARENT = True
LIGHT_INK   = "#E8E0EF"   # testo/assi/tick: chiaro, legge sul prugna

# Gene-family annotation used in Part 4 network figures. Each family lists
# canonical HUGO symbols; genes not in any family fall into "Other".
FAMILY_GROUPS = {
    "SMC / smooth muscle": [
        "DES", "ACTG2", "MYH11", "ACTA2", "MYLK", "CNN1", "SMTN", "SYNM", "TAGLN",
    ],
    "Inflammatory / immune": [
        "CXCL8", "CXCL5", "S100A8", "S100A9", "IL1B", "IL1RN", "PTGS2", "G0S2",
        "CSF3R", "BCL2A1", "FOS", "FOSB", "DUSP1", "SOCS3", "TREM1", "SELL",
        "EGR1", "CXCR2", "FCAR", "MT2A", "CCL3", "CCL5", "MMP1", "MMP3",
        "MMP12", "MMP25", "PLEK", "ITGAX", "SCG2", "SIGLEC5",
    ],
    "ECM / matrix": [
        "COL1A2", "COL7A1", "TIMP3", "ELN", "SFRP4", "SFRP2", "MGP", "MFAP4",
        "TNXB", "IGFBP5",
    ],
    "Epithelial / Paneth": [
        "SLC26A3", "FXYD6", "DEFA6", "ALPI", "EPCAM", "KRT8",
    ],
    "Adaptive immune (B/T/plasma)": [
        "MS4A1", "IGHM", "IGLC1", "CD79A", "CD79B",
    ],
}

FAMILY_COLORS = {
    "SMC / smooth muscle":             "#C45A4F",  # red
    "Inflammatory / immune":            "#E89B3F",  # orange
    "ECM / matrix":                     "#3F8E5C",  # green
    "Epithelial / Paneth":              "#9467BD",  # purple
    "Adaptive immune (B/T/plasma)":     "#1F4E79",  # dark blue
    "Other":                             "#cccccc",  # light gray
}

_SYMBOL_TO_FAMILY = {sym: fam for fam, syms in FAMILY_GROUPS.items() for sym in syms}


def _gene_family(symbol: str) -> str:
    return _SYMBOL_TO_FAMILY.get(symbol, "Other")


def save(fig, name: str, png: bool = False, dpi = 200):
    """PDF (report) o, in modalità trasparente, PNG trasparente in figures/transparent/."""
    ext = "png" if (png or TRANSPARENT) else "pdf"
    outdir = (FIGDIR / "transparent") if TRANSPARENT else FIGDIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=TRANSPARENT)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


# =============================================================================
# INTRO FIGURES
# =============================================================================

def fig_intro_data_to_graph(sample_id: int = 47, seed: int = 0):
    """
    Three-panel teaching figure: (a) full sample, cells colored by cell type;
    (b) a zoomed-in region of the sample; (c) graph close-up with KNN edges
    drawn. Communicates 'from tissue to graph'.

    The region for (b) and the close-up window for (c) are picked deterministically
    from the seed; the seed can be tuned if the chosen region is uninformative.
    """
    print("[intro] data-to-graph figure (PNG)")
    import data_io
    import graph_construction

    G = data_io.load_global(str(CACHE))
    sg = graph_construction.build_for_sample(sample_id, global_arrays=G, weighted=False)
    pos = G.positions[sg.global_idxs]
    ct = G.cell_type[sg.global_idxs]
    src, dst = sg.edge_index

    # Span of the sample
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()

    # Region for panel (b): a 25% x 25% window picked from a non-empty area
    # We pick by sampling several candidate centres, keeping the one with
    # the most cells (avoids landing in tissue gaps).
    rng = np.random.default_rng(seed)
    region_w = (xmax - xmin) * 0.25
    region_h = (ymax - ymin) * 0.25
    best_count = -1
    best_centre = None
    for _ in range(50):
        cx = rng.uniform(xmin + region_w / 2, xmax - region_w / 2)
        cy = rng.uniform(ymin + region_h / 2, ymax - region_h / 2)
        in_window = ((np.abs(pos[:, 0] - cx) < region_w / 2) &
                     (np.abs(pos[:, 1] - cy) < region_h / 2))
        cnt = int(in_window.sum())
        if cnt > best_count:
            best_count = cnt
            best_centre = (cx, cy)

    cx, cy = best_centre
    region_mask = ((np.abs(pos[:, 0] - cx) < region_w / 2) &
                   (np.abs(pos[:, 1] - cy) < region_h / 2))
    print(f"  region centre = ({cx:.0f}, {cy:.0f}), {region_mask.sum():,} cells")

    # Close-up for panel (c): a ~6x smaller window inside the region
    closeup_w = region_w / 6
    closeup_h = region_h / 6
    # Pick the densest sub-window inside the region by trying several centres
    best_cnt = -1
    best_cu = None
    cells_in_region = np.where(region_mask)[0]
    for _ in range(50):
        sub_cx = rng.uniform(cx - region_w / 2 + closeup_w / 2,
                              cx + region_w / 2 - closeup_w / 2)
        sub_cy = rng.uniform(cy - region_h / 2 + closeup_h / 2,
                              cy + region_h / 2 - closeup_h / 2)
        in_cu = ((np.abs(pos[cells_in_region, 0] - sub_cx) < closeup_w / 2) &
                 (np.abs(pos[cells_in_region, 1] - sub_cy) < closeup_h / 2))
        cnt = int(in_cu.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_cu = (sub_cx, sub_cy)
    sub_cx, sub_cy = best_cu
    closeup_mask = ((np.abs(pos[:, 0] - sub_cx) < closeup_w / 2) &
                    (np.abs(pos[:, 1] - sub_cy) < closeup_h / 2))
    print(f"  close-up centre = ({sub_cx:.0f}, {sub_cy:.0f}), {closeup_mask.sum():,} cells")

    # Edges in the close-up: both endpoints must be inside it.
    closeup_local_ids = np.where(closeup_mask)[0]
    closeup_set = set(closeup_local_ids.tolist())
    # We want each edge once. Use src < dst.
    edge_mask = (src < dst)
    src_e, dst_e = src[edge_mask], dst[edge_mask]
    edge_in_cu = (np.isin(src_e, closeup_local_ids) &
                  np.isin(dst_e, closeup_local_ids))
    src_cu = src_e[edge_in_cu]
    dst_cu = dst_e[edge_in_cu]
    print(f"  close-up edges: {len(src_cu):,}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))

    # Panel (a): full sample, colored by cell type
    ax = axes[0]
    ax.scatter(pos[:, 0], pos[:, 1], s=0.3, c=[CLASS_COLORS[c] for c in ct],
               alpha=0.7, rasterized=True)
    # Outline the region
    rect_x = [cx - region_w/2, cx + region_w/2, cx + region_w/2,
              cx - region_w/2, cx - region_w/2]
    rect_y = [cy - region_h/2, cy - region_h/2, cy + region_h/2,
              cy + region_h/2, cy - region_h/2]
    ax.plot(rect_x, rect_y, color="black", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) Sample {sample_id}: cells colored by type\n"
                 f"{len(pos):,} cells, 8 cell types")
    ax.grid(False)

    # Panel (b): the region zoomed in
    ax = axes[1]
    in_region = pos[region_mask]
    ct_region = ct[region_mask]
    ax.scatter(in_region[:, 0], in_region[:, 1], s=2.0,
               c=[CLASS_COLORS[c] for c in ct_region],
               alpha=0.85, rasterized=True)
    # Outline the close-up window
    cux = [sub_cx - closeup_w/2, sub_cx + closeup_w/2, sub_cx + closeup_w/2,
           sub_cx - closeup_w/2, sub_cx - closeup_w/2]
    cuy = [sub_cy - closeup_h/2, sub_cy - closeup_h/2, sub_cy + closeup_h/2,
           sub_cy + closeup_h/2, sub_cy - closeup_h/2]
    ax.plot(cux, cuy, color="black", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(b) Zoomed region\n{region_mask.sum():,} cells")
    ax.grid(False)

    # Panel (c): close-up with edges
    ax = axes[2]
    in_cu = pos[closeup_mask]
    ct_cu = ct[closeup_mask]
    # Draw edges first so they sit underneath the nodes
    for s_local, d_local in zip(src_cu, dst_cu):
        ax.plot([pos[s_local, 0], pos[d_local, 0]],
                [pos[s_local, 1], pos[d_local, 1]],
                color="#888888", linewidth=0.4, alpha=0.6,
                zorder=1, rasterized=True)
    ax.scatter(in_cu[:, 0], in_cu[:, 1], s=12,
               c=[CLASS_COLORS[c] for c in ct_cu],
               edgecolor="black", linewidth=0.3, zorder=2, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(c) Close-up with KNN edges\n"
                 f"{closeup_mask.sum():,} cells, {len(src_cu):,} edges")
    ax.grid(False)

    plt.tight_layout()
    save(fig, "intro_data_to_graph", png=True)


# =============================================================================
# PART 1 FIGURES
# =============================================================================

def fig_part1_degree_distribution():
    """
    Mean degree per sample (histogram), with the two anomalies annotated.
    Headline: local topology is consistent across the cohort.
    """
    print("[part1] degree distribution")
    free = pd.read_parquet(CACHE / "topology" / "free_tier.parquet")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    clean = free[~free["is_anomaly"]]
    anomalies = free[free["is_anomaly"]]

    ax.hist(clean["mean_degree"], bins=30, color=PRIMARY, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(clean["mean_degree"].median(), color="black",
               linestyle="--", linewidth=1.0,
               label=f"median = {clean['mean_degree'].median():.2f}")

    # Mark anomalies as small markers below the histogram
    ymax = ax.get_ylim()[1]
    for _, row in anomalies.iterrows():
        ax.plot(row["mean_degree"], ymax * 0.05, marker="v",
                color=ANOMALY_COLOR, markersize=8)
        ax.annotate(f"s{int(row.name)}",
                    xy=(row["mean_degree"], ymax * 0.05),
                    xytext=(row["mean_degree"], ymax * 0.18),
                    ha="center", fontsize=8, color=ANOMALY_COLOR)

    ax.set_xlabel("Mean degree of the spatial graph")
    ax.set_ylabel("Number of samples")
    ax.set_title("Per-sample mean degree (n = 112 samples)")
    ax.legend(loc="upper left", frameon=True)
    save(fig, "part1_mean_degree_distribution")


def fig_part1_bimodality():
    """
    Histogram of largest_component_frac across the cohort.
    Headline: connectivity is a *continuous gradient* from fragmented to compact,
    NOT two distinct modes (Hartigan dip test: dip = 0.025, p = 0.93).
    Filename kept as part1_lcf_bimodality for backward-compat with the report .tex.
    """
    print("[part1] connectivity gradient")
    free = pd.read_parquet(CACHE / "topology" / "free_tier.parquet")
    clean = free[~free["is_anomaly"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(clean["largest_component_frac"], bins=25, color=PRIMARY, alpha=0.85,
            edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Largest connected component fraction")
    ax.set_ylabel("Number of samples")
    ax.set_title("A continuous connectivity gradient across the cohort "
                 "(n = 110, anomalies excluded)")

    ymax = ax.get_ylim()[1]
    # Descriptive labels for the two ENDS of the continuum (not modes).
    ax.text(0.155, ymax * 0.74, "fragmented", ha="center", va="top",
            fontsize=9, style="italic", color=SECONDARY)
    ax.text(0.90, ymax * 0.74, "compact", ha="center", va="top",
            fontsize=9, style="italic", color=ACCENT_GREEN)

    save(fig, "part1_lcf_bimodality")


def fig_part1_composition_topology_heatmap():
    """
    Spearman correlation between topology metrics (rows) and per-sample
    cell-type composition fractions (columns), across 110 non-anomaly samples.
    """
    print("[part1] composition x topology heatmap")
    free = pd.read_parquet(CACHE / "topology" / "free_tier.parquet")
    clean = free[~free["is_anomaly"]].copy()

    # If composition columns aren't already in free_tier, compute them
    needed_comp_cols = [f"frac_class_{i}" for i in range(8)]
    have_comp = all(c in clean.columns for c in needed_comp_cols)

    if not have_comp:
        # Compute composition from global arrays
        import data_io
        G = data_io.load_global(str(CACHE))
        for s in clean.index:
            mask = G.sample == s
            counts = np.bincount(G.cell_type[mask], minlength=8)
            for i in range(8):
                clean.loc[s, f"frac_class_{i}"] = counts[i] / counts.sum()

    topo_cols = ["mean_degree", "frac_isolated", "largest_component_frac",
                 "n_components_ge_100", "transitivity_global",
                 "mean_local_clustering", "degree_assortativity",
                 "celltype_assortativity"]
    topo_labels = ["mean degree", "frac. isolated", "largest comp. frac.",
                   "n comp. $\\geq$100", "transitivity",
                   "mean local cluster.", "deg. assortativity",
                   "cell-type assortativity"]
    comp_cols = [f"frac_class_{i}" for i in range(8)]
    comp_labels = [
        "B cells",  # c0
        "Endothelial",  # c1
        "Fibroblasts",  # c2
        "Myeloid",  # c3
        "Normal epi.",  # c4
        "SMC",  # c5
        "T cells",  # c6
        "Tumor",  # c7
    ]

    corr = clean[topo_cols + comp_cols].corr(method="spearman")
    block = corr.loc[topo_cols, comp_cols].values

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(block, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(comp_cols)))
    ax.set_xticklabels(comp_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(topo_cols)))
    ax.set_yticklabels(topo_labels)
    ax.set_xlabel("Cell-type fraction in sample")
    ax.set_title("Spearman correlation: topology metrics $\\times$ cell-type composition (n = 110)")
    ax.grid(False)
    for i in range(len(topo_cols)):
        for j in range(len(comp_cols)):
            v = block[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=8)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, label="Spearman $\\rho$")
    save(fig, "part1_composition_topology_heatmap")


def fig_part1_mask_vs_degree():
    """
    Scatter: train_mask keep fraction vs. mean degree per sample.
    Headline: the mask is essentially a function of mean degree (rho ~ -0.91).
    """
    print("[part1] mask vs degree scatter")
    free = pd.read_parquet(CACHE / "topology" / "free_tier.parquet")
    clean = free[~free["is_anomaly"]].copy()

    if "mask_frac" not in clean.columns:
        import data_io
        G = data_io.load_global(str(CACHE))
        for s in clean.index:
            mask = G.sample == s
            clean.loc[s, "mask_frac"] = float(G.train_mask[mask].mean())

    rho, _ = spearmanr(clean["mask_frac"], clean["mean_degree"])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(clean["mean_degree"], clean["mask_frac"],
               s=18, color=PRIMARY, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean degree of the spatial graph")
    ax.set_ylabel("Train-mask keep fraction")
    ax.set_title(f"Train mask is a function of local density\nSpearman $\\rho = {rho:.2f}$")
    save(fig, "part1_mask_vs_degree")


def _load_betweenness_sample(sample_id: int = 2):
    """Carica (o ricalcola) positions+betweenness della giant component."""
    expensive_per_sample = CACHE / "topology" / "expensive_per_sample"
    if expensive_per_sample.exists():
        sample_file = expensive_per_sample / f"sample_{sample_id}.npz"
        if sample_file.exists():
            data = np.load(sample_file)
            pos, bw = data["positions"], data["betweenness"]
        else:
            print(f"  (sample_{sample_id}.npz not found; recomputing)")
            pos, bw = _recompute_betweenness_sample(sample_id)
    else:
        pos, bw = _recompute_betweenness_sample(sample_id)

    mask = ~np.isnan(bw)          # drop cells outside the giant component
    return pos[mask], bw[mask]


def _plot_betweenness_panel(ax, pos_g, bw_g, pct: float, sample_id: int = 2):
    """Disegna un pannello: sfondo grigio + top-(100-pct)% evidenziato."""
    threshold = np.percentile(bw_g, pct)
    is_top = bw_g >= threshold
    top_frac = 100 - pct

    bg = LIGHT_INK if TRANSPARENT else "#d8d8d8"
    ax.scatter(pos_g[~is_top, 0], pos_g[~is_top, 1],
               s=0.3, c=bg, alpha=0.45 if TRANSPARENT else 0.6, rasterized=True)
    ax.scatter(pos_g[is_top, 0], pos_g[is_top, 1],
               s=3, c=SECONDARY, alpha=0.95, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"Top {top_frac:g}% betweenness "
                 f"({is_top.sum():,} of {len(bw_g):,} cells)")
    ax.grid(False)


def fig_part1_betweenness_skeleton(mode: str = "single", sample_id: int = 2):
    """
    Sample spatial layout with high-betweenness cells highlighted (PNG).

    mode="single"  -> un solo pannello, top 1% (default)
    mode="compare" -> due pannelli affiancati, top 1% e top 5%
    """
    print(f"[part1] betweenness skeleton (PNG, mode={mode})")
    pos_g, bw_g = _load_betweenness_sample(sample_id)

    if mode == "compare":
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.0))
        _plot_betweenness_panel(axes[0], pos_g, bw_g, pct=99, sample_id=sample_id)
        _plot_betweenness_panel(axes[1], pos_g, bw_g, pct=95, sample_id=sample_id)
        fig.suptitle(f"Sample {sample_id} -- betweenness skeleton at two thresholds",
                     y=0.98)
        save(fig, "part1_betweenness_skeleton_compare", png=True)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        _plot_betweenness_panel(ax, pos_g, bw_g, pct=99, sample_id=sample_id)
        ax.set_title(f"Sample {sample_id} -- top 1% betweenness cells form a "
                     f"tissue skeleton")
        save(fig, "part1_betweenness_skeleton", png=True)


def _recompute_betweenness_sample(sample_id: int):
    """Recompute betweenness for one sample, returning (positions, betweenness)."""
    import data_io
    import graph_construction
    import topology
    print(f"  recomputing betweenness for sample {sample_id}...")
    G = data_io.load_global(str(CACHE))
    sg = graph_construction.build_for_sample(sample_id, global_arrays=G,
                                              weighted=False)
    ct = G.cell_type[sg.global_idxs]
    result = topology.compute_expensive_tier_metrics(sg, ct)
    pos = G.positions[sg.global_idxs]
    return pos, result["betweenness_per_node"]


def fig_part1_anomaly_handling():
    """
    3-panel comparison of a normal sample (#2) against the two anomalies
    (#87, #107), all at a common spatial scale of 30k x 30k units,
    centered on each sample's centroid. Cells colored by cell type if in
    the giant component, light gray otherwise. Includes a 5k-unit scalebar.
    PNG (large point count).
    """
    print("[part1] anomaly handling (PNG)")
    import data_io
    import graph_construction
    import igraph as ig

    G = data_io.load_global(str(CACHE))
    free = pd.read_parquet(CACHE / "topology" / "free_tier.parquet")

    span = 30000.0          # spatial extent shown per panel (axis-equal)
    bar_length = 5000.0      # scalebar length in units

    def plot_panel(ax, sample_id, title_prefix):
        sg = graph_construction.build_for_sample(sample_id, global_arrays=G,
                                                  weighted=False)
        pos = G.positions[sg.global_idxs]
        ct = G.cell_type[sg.global_idxs]

        # Giant connected component via igraph
        src, dst = sg.edge_index
        n_nodes = len(pos)
        g_ig = ig.Graph(n=n_nodes,
                        edges=list(zip(src.tolist(), dst.tolist())),
                        directed=False)
        components = g_ig.connected_components()
        sizes = components.sizes()
        if len(sizes) > 0:
            giant_id = int(np.argmax(sizes))
            membership = np.array(components.membership)
            in_giant = membership == giant_id
        else:
            in_giant = np.zeros(n_nodes, dtype=bool)

        not_giant = ~in_giant
        if not_giant.any():
            ax.scatter(pos[not_giant, 0], pos[not_giant, 1],
                       s=0.5, c="#dddddd", alpha=0.6, rasterized=True)
        if in_giant.any():
            ax.scatter(pos[in_giant, 0], pos[in_giant, 1],
                       s=0.5, c=[CLASS_COLORS[c] for c in ct[in_giant]],
                       alpha=0.85, rasterized=True)

        # Common spatial scale: fixed span centered on sample centroid
        cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
        half = span / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

        # Scalebar in bottom-left
        bar_x_start = cx - half + span * 0.05
        bar_y = cy - half + span * 0.05
        ax.plot([bar_x_start, bar_x_start + bar_length], [bar_y, bar_y],
                "k-", linewidth=2)
        ax.text(bar_x_start + bar_length / 2, bar_y + span * 0.02,
                "5 kunits", ha="center", va="bottom", fontsize=8)

        row = free.loc[sample_id]
        md = row["mean_degree"]
        fi = row["frac_isolated"]
        lcf = row["largest_component_frac"]

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{title_prefix}Sample {sample_id}\n"
            f"mean degree = {md:.2f} | isolated = {fi:.2f} | LCF = {lcf:.2f}",
            fontsize=10
        )
        ax.grid(False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    plot_panel(axes[0], 2, "Normal -- ")
    plot_panel(axes[1], 87, "Anomaly -- ")
    plot_panel(axes[2], 107, "Anomaly -- ")
    plt.tight_layout()
    save(fig, "part1_anomaly_handling", png=True)


# =============================================================================
# PART 2 FIGURES
# =============================================================================

def fig_part2_niche_centroid_heatmap():
    """
    Heatmap of niche centroids: 20 niches (rows) x 8 cell-type fractions
    (columns). Headline figure for Part 2 -- defines what each niche IS.
    """
    print("[part2] niche centroid heatmap")
    centroids = pd.read_parquet(CACHE / "communities" / "niche_centroids.parquet")

    # The parquet should have columns frac_class_0...frac_class_7 and possibly
    # dominant_class. We'll be defensive about column names.
    frac_cols = [c for c in centroids.columns if c.startswith("frac_class_")]
    if not frac_cols:
        # Try alternative naming
        frac_cols = [c for c in centroids.columns if c.startswith("class_")]
    frac_cols = sorted(frac_cols, key=lambda c: int(c.split("_")[-1]))

    n_niches = len(centroids)
    n_classes = len(frac_cols)

    # Biological cell-type names matching c0..c7 mapping
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    # Sort niches by dominant_class (primary) and dominant_frac descending
    # (secondary, so the purest niches of each family appear first within
    # their block). Use .loc[] for correct niche_id -> row mapping.
    if "dominant_class" in centroids.columns and "dominant_frac" in centroids.columns:
        order = centroids.sort_values(
            ["dominant_class", "dominant_frac"],
            ascending=[True, False],
        ).index.tolist()
    elif "dominant_class" in centroids.columns:
        order = centroids.sort_values(["dominant_class"]).index.tolist()
    else:
        # Fallback: sort by argmax of the row
        matrix_tmp = centroids[frac_cols].values
        order_pos = list(np.argsort(matrix_tmp.argmax(axis=1)))
        order = [centroids.index[i] for i in order_pos]

    matrix_sorted = centroids.loc[order, frac_cols].values
    niche_labels = [f"niche {i}" for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(matrix_sorted, cmap="viridis", aspect="auto",
                   vmin=0, vmax=matrix_sorted.max())
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names[:n_classes], rotation=30, ha="right")
    ax.set_yticks(range(n_niches))
    ax.set_yticklabels(niche_labels)
    ax.set_xlabel("Cell-type composition")
    ax.set_title(f"Niche centroids: composition of each of {n_niches} niches")
    ax.grid(False)

    # Annotate cells with values >= 0.1
    for i in range(n_niches):
        for j in range(n_classes):
            v = matrix_sorted[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.7, label="class fraction")
    save(fig, "part2_niche_centroid_heatmap")


def fig_part2_spatial_niches(samples=(2, 47)):
    """
    Two-panel spatial visualization: each sample's cells colored by the
    niche they belong to. Shows that niches form coherent tissue regions.
    """
    print("[part2] spatial niches (PNG)")
    import data_io

    G = data_io.load_global(str(CACHE))
    niche_per_cell = np.load(CACHE / "communities" / "niche_per_cell.npy")

    n_niches = int(niche_per_cell[niche_per_cell >= 0].max()) + 1

    # Color palette for 20 niches: tab20 has exactly 20 colors
    cmap = plt.cm.tab20
    niche_colors = [cmap(i % 20) for i in range(n_niches)]

    fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 5.5))
    if len(samples) == 1:
        axes = [axes]

    for ax, s in zip(axes, samples):
        mask = G.sample == s
        pos = G.positions[mask]
        niches = niche_per_cell[mask]

        # Cells unassigned to any niche (niche < 0) plotted in light gray
        unassigned = niches < 0
        ax.scatter(pos[unassigned, 0], pos[unassigned, 1],
                   s=0.3, c="#dddddd", alpha=0.5, rasterized=True)
        # Cells assigned to niches plotted by niche color
        assigned = ~unassigned
        if assigned.any():
            colors = [niche_colors[n] for n in niches[assigned]]
            ax.scatter(pos[assigned, 0], pos[assigned, 1],
                       s=0.5, c=colors, alpha=0.8, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        n_assigned = int(assigned.sum())
        n_total = int(mask.sum())
        ax.set_title(f"Sample {s} ({n_assigned:,} / {n_total:,} cells assigned)")
        ax.grid(False)

    plt.tight_layout()
    save(fig, "part2_spatial_niches", png=True)


def fig_part2_niche_recurrence():
    """
    Bar chart: for each niche, in how many of the 56 patients does it appear?
    Captures the universal/common/selective/rare structure.
    A niche appears in a patient if any cell of that patient has been assigned
    to that niche.
    """
    print("[part2] niche recurrence")
    import data_io

    G = data_io.load_global(str(CACHE))
    niche_per_cell = np.load(CACHE / "communities" / "niche_per_cell.npy")
    n_niches = int(niche_per_cell[niche_per_cell >= 0].max()) + 1

    # For each niche, count distinct patients it appears in
    n_patients_per_niche = np.zeros(n_niches, dtype=int)
    for n in range(n_niches):
        cells_in_niche = niche_per_cell == n
        patients = np.unique(G.patient[cells_in_niche])
        n_patients_per_niche[n] = len(patients)

    n_total_patients = int(len(np.unique(G.patient)))

    # Sort niches by recurrence
    order = np.argsort(-n_patients_per_niche)  # descending
    sorted_counts = n_patients_per_niche[order]
    sorted_ids = order

    # Define category thresholds (consistent with the project summary)
    UNIVERSAL = int(n_total_patients * 0.85)   # >=48 of 56
    COMMON = int(n_total_patients * 0.50)       # >=28 of 56
    SELECTIVE = int(n_total_patients * 0.25)    # >=14 of 56
    # below SELECTIVE = rare

    def category_color(c):
        if c >= UNIVERSAL:
            return PRIMARY
        if c >= COMMON:
            return "#4F8EC4"          # lighter blue
        if c >= SELECTIVE:
            return ACCENT_GREEN
        return ANOMALY_COLOR

    colors = [category_color(c) for c in sorted_counts]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(range(n_niches), sorted_counts, color=colors,
                  edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(n_niches))
    ax.set_xticklabels([f"{i}" for i in sorted_ids], rotation=0, fontsize=8)
    ax.set_xlabel("Niche id (sorted by recurrence)")
    ax.set_ylabel(f"Patients with niche present  (of {n_total_patients})")
    ax.set_title(f"Niche recurrence across the cohort  ({n_niches} niches, {n_total_patients} patients)")
    ax.grid(False)

    # Reference lines for the category thresholds
    ax.axhline(UNIVERSAL, color="#888888", linestyle="--", linewidth=0.6)
    ax.axhline(COMMON, color="#888888", linestyle="--", linewidth=0.6)
    ax.axhline(SELECTIVE, color="#888888", linestyle="--", linewidth=0.6)

    # Place category labels to the right of the bars, each at the vertical
    # center of its category band. Extend the x-axis slightly to make room.
    label_x = n_niches + 0.1
    label_specs = [
        ("UNIVERSAL",  (UNIVERSAL + n_total_patients) / 2),
        ("COMMON",     (COMMON + UNIVERSAL) / 2),
        ("SELECTIVE",  (SELECTIVE + COMMON) / 2),
        ("RARE",       SELECTIVE / 2),
    ]
    for label, y in label_specs:
        ax.text(label_x, y, label, ha="left", va="center",
                fontsize=8)
    ax.set_xlim(-0.6, n_niches + .0)

    ax.set_ylim(0, n_total_patients + 4)
    save(fig, "part2_niche_recurrence")

def fig_part2_niche_recurrence_celltype():
    """
    Same recurrence bar chart as fig_part2_niche_recurrence, but each bar is
    colored by the DOMINANT cell type of the niche (most frequent cell_type
    among the cells assigned to that niche), using the shared CLASS_COLORS
    palette. The universal/common/selective/rare threshold lines and labels
    are kept; a legend maps colors -> cell-type names.
    """
    print("[part2] niche recurrence (colored by dominant cell type)")
    import data_io
    from matplotlib.patches import Patch

    G = data_io.load_global(str(CACHE))
    niche_per_cell = np.load(CACHE / "communities" / "niche_per_cell.npy")
    n_niches = int(niche_per_cell[niche_per_cell >= 0].max()) + 1

    # Recurrence: distinct patients per niche
    n_patients_per_niche = np.zeros(n_niches, dtype=int)
    # Dominant cell type per niche
    dom_class_per_niche = np.zeros(n_niches, dtype=int)
    for n in range(n_niches):
        cells_in_niche = niche_per_cell == n
        patients = np.unique(G.patient[cells_in_niche])
        n_patients_per_niche[n] = len(patients)
        # most frequent cell_type among cells assigned to this niche
        ct_counts = np.bincount(G.cell_type[cells_in_niche], minlength=8)
        dom_class_per_niche[n] = int(np.argmax(ct_counts))

    n_total_patients = int(len(np.unique(G.patient)))

    # Sort niches by recurrence (descending), same as the original
    order = np.argsort(-n_patients_per_niche)
    sorted_counts = n_patients_per_niche[order]
    sorted_ids = order
    sorted_dom = dom_class_per_niche[order]

    # Category thresholds (kept identical to the original)
    UNIVERSAL = int(n_total_patients * 0.85)
    COMMON = int(n_total_patients * 0.50)
    SELECTIVE = int(n_total_patients * 0.25)

    # Biological cell-type names matching c0..c7 mapping (shared with Part 3)
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    # Color each bar by its dominant cell type
    colors = [CLASS_COLORS[c] for c in sorted_dom]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(range(n_niches), sorted_counts, color=colors,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(n_niches))
    ax.set_xticklabels([f"{i}" for i in sorted_ids], rotation=0, fontsize=8)
    ax.set_xlabel("Niche id (sorted by recurrence)")
    ax.set_ylabel(f"Patients with niche present  (of {n_total_patients})")
    ax.set_title(f"Niche recurrence across the cohort  ({n_niches} niches, "
                 f"{n_total_patients} patients)")
    ax.grid(False)

    # Reference lines for the category thresholds
    ax.axhline(UNIVERSAL, color="#888888", linestyle="--", linewidth=0.6)
    ax.axhline(COMMON, color="#888888", linestyle="--", linewidth=0.6)
    ax.axhline(SELECTIVE, color="#888888", linestyle="--", linewidth=0.6)

    # Category labels to the right of the bars
    label_x = n_niches + 0.1
    label_specs = [
        ("UNIVERSAL",  (UNIVERSAL + n_total_patients) / 2),
        ("COMMON",     (COMMON + UNIVERSAL) / 2),
        ("SELECTIVE",  (SELECTIVE + COMMON) / 2),
        ("RARE",       SELECTIVE / 2),
    ]
    for label, y in label_specs:
        ax.text(label_x, y, label, ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.6, n_niches + .0)
    ax.set_ylim(0, n_total_patients + 4)

    # Legend: one swatch per dominant cell type actually present
    present_classes = sorted(set(int(c) for c in sorted_dom))
    legend_handles = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="white",
              label=class_names[c])
        for c in present_classes
    ]
    # leg = ax.legend(
    #     handles=legend_handles,
    #     loc="upper right",
    #     fontsize=8,
    #     ncol=2,
    #     frameon=True,
    #     framealpha=1.0,
    #     facecolor="white",
    #     edgecolor="#cccccc",
    #     fancybox=False,
    #     title="Dominant cell type",
    #     title_fontsize=8,
    # )
    # leg.set_zorder(10)

    save(fig, "part2_niche_recurrence_celltype")

# =============================================================================
# PART 3 FIGURES
# =============================================================================

def _load_rung_pickles():
    """Load all available rung pickles, returning a dict {rung_id: data}."""
    import pickle
    models_dir = CACHE / "models"
    rungs = {}
    for i in range(1, 6):
        candidates = list(models_dir.glob(f"rung{i}_*.pkl"))
        if not candidates:
            print(f"  warning: no pickle for rung {i}")
            continue
        # If multiple, pick the most recent
        p = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        with open(p, "rb") as f:
            rungs[i] = pickle.load(f)
        print(f"  loaded rung {i}: {p.name}")
    return rungs


def _extract_val_macro_f1(data):
    """Get the validation macro-F1 from a rung pickle.
    Rung 1/2 use flat schema: data['macro_f1'].
    Rung 3/4/5 use nested schema: data['best_val_macro_f1'] (best across epochs)
    or data['final_val_metrics']['macro_f1'] (final epoch).
    """
    if not isinstance(data, dict):
        return None
    # GNN-style first (these are usually present and best)
    for key in ("best_val_macro_f1", "val_macro_f1", "best_val_f1"):
        if key in data:
            return float(data[key])
    # Nested final_val_metrics
    fv = data.get("final_val_metrics")
    if isinstance(fv, dict) and "macro_f1" in fv:
        return float(fv["macro_f1"])
    # Flat schema (rung 1/2)
    if "macro_f1" in data:
        return float(data["macro_f1"])
    return None


def _extract_val_predictions(data):
    """Return (y_true, y_pred) from a rung pickle, or (None, None).
    Rung 1/2: top-level y_true / y_pred.
    Rung 3/4/5: data['final_val_metrics']['y_true' / 'y_pred'].
    """
    if not isinstance(data, dict):
        return None, None
    # Nested first
    fv = data.get("final_val_metrics")
    if isinstance(fv, dict) and "y_true" in fv and "y_pred" in fv:
        return np.asarray(fv["y_true"]), np.asarray(fv["y_pred"])
    # Flat (rung 1/2)
    if "y_true" in data and "y_pred" in data:
        return np.asarray(data["y_true"]), np.asarray(data["y_pred"])
    return None, None


def fig_part3_rung_comparison():
    """
    Bar chart of validation macro-F1 across the 5 rungs.
    Headline figure for Part 3.
    """
    print("[part3] rung comparison")
    rungs = _load_rung_pickles()
    if not rungs:
        print("  skipping (no rung pickles)")
        return

    rung_ids = sorted(rungs.keys())
    labels, scores = [], []
    for r in rung_ids:
        score = _extract_val_macro_f1(rungs[r])
        if score is None:
            print(f"  could not extract val F1 for rung {r}, skipping")
            continue
        labels.append(f"Rung {r}")
        scores.append(score)

    colors = [PRIMARY if s == max(scores) else "#7AA6CB" for s in scores]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.bar(range(len(scores)), scores, color=colors,
                  edgecolor="white", linewidth=0.6)
    for i, (b, s) in enumerate(zip(bars, scores)):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.005,
                f"{s:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Validation $\mathrm{macro\text{-}}F_1$")
    ax.set_title("5-rung ablation: best validation macro-$F_1$ per model")
    ax.set_ylim(0.45, max(scores) * 1.06)
    ax.grid(False)
    save(fig, "part3_rung_comparison")


def fig_part3_confusion_matrix():
    """
    Row-normalized confusion matrix for Rung 3 on the test set.
    """
    print("[part3] confusion matrix (Rung 3)")
    rungs = _load_rung_pickles()
    if 3 not in rungs:
        print("  skipping (no rung 3 pickle)")
        return

    data = rungs[3]
    tm = data.get("test_metrics")
    if not (tm and "y_pred" in tm and "y_true" in tm):
        print("  no test_metrics.y_pred/y_true, skipping")
        return

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tm["y_true"], tm["y_pred"], labels=list(range(8)))

    # Row-normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Biological cell-type names matching c0..c7 mapping
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(class_names[:cm.shape[1]], rotation=30, ha="right")
    ax.set_yticklabels(class_names[:cm.shape[0]])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(r"Rung 3 test set: row-normalized confusion matrix")
    ax.grid(False)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Fraction (row-normalized)")
    save(fig, "part3_confusion_matrix")


def fig_part3_per_class_progression():
    """
    Per-class F1 across the 5 rungs (one line per class). Computed from each
    rung's predictions via sklearn. Handles both flat (rung 1/2) and nested
    (rung 3/4/5) pickle schemas.
    """
    print("[part3] per-class F1 progression")
    rungs = _load_rung_pickles()
    if not rungs:
        print("  skipping (no rung pickles)")
        return

    from sklearn.metrics import f1_score

    n_classes = 8
    rung_ids = sorted(rungs.keys())
    f1_matrix = np.full((n_classes, len(rung_ids)), np.nan)

    for j, r in enumerate(rung_ids):
        y_true, y_pred = _extract_val_predictions(rungs[r])
        if y_true is None:
            print(f"  rung {r}: no predictions found, skipping")
            continue
        f1_per = f1_score(y_true, y_pred, labels=list(range(n_classes)),
                          average=None, zero_division=0)
        f1_matrix[:, j] = f1_per

    if np.isnan(f1_matrix).all():
        print("  no per-class data found in any rung, skipping")
        return

    # Biological cell-type names matching c0..c7 mapping
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    # Opacity of all lines EXCEPT T cells (c6). Set to 1.0 to disable dimming.
    OTHER_LINES_ALPHA = 0.25
    TCELL_CLASS = 6

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for c in range(n_classes):
        alpha = 1.0 if c == TCELL_CLASS else OTHER_LINES_ALPHA
        ax.plot(rung_ids, f1_matrix[c], marker="o", markersize=5,
                color=CLASS_COLORS[c], linewidth=1.4, alpha=alpha,
                label=class_names[c])
    ax.set_xlabel("Rung")
    ax.set_ylabel(r"Validation $F_1$ per class")
    ax.set_xticks(rung_ids)
    ax.set_xticklabels([f"R{r}" for r in rung_ids])
    ax.set_title(r"Per-class validation $F_1$ across the 5 rungs")
    leg = ax.legend(
        ncol=4,
        fontsize=8,
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="#15273F" if TRANSPARENT else "white",
        edgecolor="white" if TRANSPARENT else "#cccccc",
        fancybox=False,
    )
    leg.set_zorder(10)
    ax.set_ylim(0, 1)
    save(fig, "part3_per_class_progression")


def fig_part3_training_curves():
    """
    Training curves for Rung 3: train loss and val macro-F1 per epoch.
    """
    print("[part3] training curves (Rung 3)")
    rungs = _load_rung_pickles()
    if 3 not in rungs:
        print("  skipping (no rung 3 pickle)")
        return

    data = rungs[3]
    history = None
    for key in ("history", "train_history", "log", "epoch_log"):
        if isinstance(data, dict) and key in data:
            history = data[key]
            break

    if history is None:
        print("  no training history found, skipping")
        return

    # history can be a list of dicts or a dict of lists. Normalize.
    if isinstance(history, list):
        train_loss = [h.get("train_loss") for h in history]
        val_f1 = [h.get("val_macro_f1", h.get("val_f1")) for h in history]
    else:
        train_loss = history.get("train_loss", [])
        val_f1 = history.get("val_macro_f1", history.get("val_f1", []))

    train_loss = [x for x in train_loss if x is not None]
    val_f1 = [x for x in val_f1 if x is not None]

    if not train_loss and not val_f1:
        print("  empty history, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    if train_loss:
        epochs_l = range(1, len(train_loss) + 1)
        axes[0].plot(epochs_l, train_loss, color=PRIMARY, marker="o",
                     markersize=3, linewidth=1.2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Train loss")
        axes[0].set_title("Train loss (Rung 3)")

    if val_f1:
        epochs_f = range(1, len(val_f1) + 1)
        axes[1].plot(epochs_f, val_f1, color=ACCENT_GREEN, marker="o",
                     markersize=3, linewidth=1.2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(r"Validation $\mathrm{macro\text{-}}F_1$")
        axes[1].set_title("Validation $F_1$ (Rung 3)")
        # Use a fixed-ish y range so small wiggles don't look dramatic
        ymin = min(0.55, min(val_f1) - 0.02)
        ymax = max(0.70, max(val_f1) + 0.02)
        axes[1].set_ylim(ymin, ymax)
        # Mark best
        best_idx = int(np.argmax(val_f1))
        axes[1].axvline(best_idx + 1, color=SECONDARY, linestyle="--",
                        linewidth=0.8, label=f"best @ epoch {best_idx + 1}")
        axes[1].legend(loc="lower right", fontsize=8, frameon=False)

    plt.tight_layout()
    save(fig, "part3_training_curves")


def fig_part3_per_niche_test_f1():
    """
    Bar chart: Rung 3 test macro-F1 broken down by niche.
    Uses the pre-computed `test_per_niche` list in the rung 3 pickle.
    """
    print("[part3] per-niche test F1")
    rungs = _load_rung_pickles()
    if 3 not in rungs:
        print("  skipping (no rung 3 pickle)")
        return

    data = rungs[3]
    per_niche_list = data.get("test_per_niche")
    if not per_niche_list:
        print("  no test_per_niche field, skipping")
        return

    # per_niche_list is a list of dicts: niche_id, dom_class, n_cells_test, test_f1
    niche_ids = [int(d["niche_id"]) for d in per_niche_list]
    dom_classes = [int(d["dom_class"]) for d in per_niche_list]
    f1s = [float(d["test_f1"]) for d in per_niche_list]
    n_cells = [int(d["n_cells_test"]) for d in per_niche_list]

    f1s_arr = np.array(f1s)
    # Sort by F1 descending
    order = np.argsort(-f1s_arr)
    sorted_f1 = f1s_arr[order]
    sorted_ids = [niche_ids[i] for i in order]
    sorted_dom = [dom_classes[i] for i in order]
    sorted_n = [n_cells[i] for i in order]

    # Color each bar by its dominant class
    bar_colors = [CLASS_COLORS[c] for c in sorted_dom]

    # Biological cell-type names matching c0..c7 mapping
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    xs = np.arange(len(sorted_f1))
    ax.bar(xs, sorted_f1, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"n{nid}" for nid in sorted_ids], fontsize=8)
    ax.set_xlabel("Niche id (sorted by test $F_1$)")
    ax.set_ylabel(r"Rung 3 test $\mathrm{macro\text{-}}F_1$")

    median = float(np.median(f1s_arr))
    ax.axhline(median, color=LIGHT_INK if TRANSPARENT else "gray",
               linestyle="--", linewidth=0.6)
    # Annotate the median value at the right end of the dashed line
    ax.text(len(xs) - 0.5, median + 0.005, f"median = {median:.3f}",
            ha="right", va="bottom", fontsize=8,
            color=LIGHT_INK if TRANSPARENT else "#555555")

    # Bar above showing cell count (small text on top of each bar)
    for x, v, n in zip(xs, sorted_f1, sorted_n):
        ax.text(x, v + 0.015, f"{n // 1000}k" if n >= 1000 else f"{n}",
                ha="center", va="bottom", fontsize=6,
                color=LIGHT_INK if TRANSPARENT else "black")

    ax.set_title("Rung 3 test $F_1$ broken down by niche, colored by dominant class")

    # Build a custom legend: one swatch per dominant class actually present
    # in the data, mapping color -> biological cell-type name.
    from matplotlib.patches import Patch
    present_classes = sorted(set(sorted_dom))
    legend_handles = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="white",
              label=class_names[c])
        for c in present_classes
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        ncol=2,
        frameon=True,
        framealpha=1.0,
        facecolor="#15273F" if TRANSPARENT else "white",
        edgecolor="white" if TRANSPARENT else "#cccccc",
        fancybox=False,
    )
    leg.set_zorder(10)

    ax.set_ylim(0, max(1.0, sorted_f1.max() + 0.1))
    save(fig, "part3_per_niche_test_f1")


# =============================================================================
# PART 4 FIGURES
# =============================================================================

def _load_part4_artifacts():
    """Load the cached Part 4 GGM artifacts."""
    import pickle
    ggm_dir = CACHE / "ggm"
    out = {}
    for name in ["ggms_alpha_007_common378.pkl",
                 "bootstraps_alpha_007_common378.pkl"]:
        p = ggm_dir / name
        if p.exists():
            with open(p, "rb") as f:
                out[name.replace(".pkl", "")] = pickle.load(f)
            print(f"  loaded {name}")
    return out


def _extract_class_ggms(arts):
    """Pull the per-class GGMs out of the loaded artifacts. Returns a dict
    {class_int: {'precision', 'degree', 'edges'}} or None if structure unknown."""
    if "ggms_alpha_007_common378" not in arts:
        return None
    root = arts["ggms_alpha_007_common378"]
    if not isinstance(root, dict):
        return None
    # Real structure: root["per_class"] is a dict with string keys "0".."7"
    per_class = root.get("per_class")
    if not isinstance(per_class, dict):
        return None

    out = {}
    for k, data in per_class.items():
        try:
            cid = int(k)
        except (ValueError, TypeError):
            continue
        P = np.array(data["precision_"]).copy()
        np.fill_diagonal(P, 0)
        deg = np.array(data["degree"])
        # Edges from upper triangle of non-zero precision
        mask = (P != 0) & np.triu(np.ones_like(P, dtype=bool), k=1)
        ii, jj = np.where(mask)
        edges = set(zip(ii.tolist(), jj.tolist()))
        out[cid] = {"precision": P, "degree": deg, "edges": edges,
                    "n_edges": int(data.get("n_edges", len(edges)))}
    return out


def fig_part4_hub_jaccard_heatmap():
    """
    8x8 heatmap of hub Jaccard (top-15) between class GGMs, side-by-side
    with edge Jaccard.
    """
    print("[part4] hub & edge Jaccard heatmaps")
    arts = _load_part4_artifacts()
    ggms = _extract_class_ggms(arts)
    if ggms is None:
        print("  GGM structure not recognized, skipping")
        return

    class_ids = sorted(ggms.keys())
    n = len(class_ids)

    hubs = {c: set(np.argsort(-ggms[c]["degree"])[:15].tolist()) for c in class_ids}
    edges = {c: ggms[c]["edges"] for c in class_ids}

    hub_jacc = np.zeros((n, n))
    edge_jacc = np.zeros((n, n))
    for i, a in enumerate(class_ids):
        for j, b in enumerate(class_ids):
            if i == j:
                hub_jacc[i, j] = 1.0
                edge_jacc[i, j] = 1.0
                continue
            ha, hb = hubs[a], hubs[b]
            ea, eb = edges[a], edges[b]
            hub_jacc[i, j] = len(ha & hb) / max(1, len(ha | hb))
            edge_jacc[i, j] = len(ea & eb) / max(1, len(ea | eb))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    for ax, mat, title in [
        (axes[0], hub_jacc, "Hub Jaccard (top-15)"),
        (axes[1], edge_jacc, "Edge Jaccard"),
    ]:
        # Mask diagonal (always 1) so the off-diagonal contrast is visible
        mat_show = mat.copy()
        np.fill_diagonal(mat_show, np.nan)
        masked = np.ma.masked_invalid(mat_show)
        offdiag_max = np.nanmax(mat_show)
        im = ax.imshow(masked, cmap="Blues", vmin=0, vmax=offdiag_max,
                       aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"c{c}" for c in class_ids])
        ax.set_yticklabels([f"c{c}" for c in class_ids])
        ax.set_title(f"{title}  (max off-diag = {offdiag_max:.2f})")
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > offdiag_max * 0.5 else "black",
                        fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    save(fig, "part4_jaccard_heatmaps")


def fig_part4_degree_distributions():
    """
    Per-class degree distributions in the fitted GGMs, as small multiples.
    """
    print("[part4] degree distributions")
    arts = _load_part4_artifacts()
    ggms = _extract_class_ggms(arts)
    if ggms is None:
        print("  GGM structure not recognized, skipping")
        return

    class_ids = sorted(ggms.keys())

    # Biological cell-type names matching c0..c7 mapping
    class_names = [
        "B cells",          # c0
        "Endothelial",      # c1
        "Fibroblasts",      # c2
        "Myeloid",          # c3
        "Normal epi.",      # c4
        "SMC",              # c5
        "T cells",          # c6
        "Tumor",            # c7
    ]

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.0), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, class_ids):
        deg = ggms[c]["degree"]
        n_edges = ggms[c]["n_edges"]
        ax.hist(deg, bins=40, color=CLASS_COLORS[c % 8], alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{class_names[c]}  (n_edges = {n_edges:,})", fontsize=10)
        ax.grid(alpha=0.4)

    for ax in axes[1]:
        ax.set_xlabel("Node degree")
    for ax in axes[:, 0]:
        ax.set_ylabel("Gene count")

    fig.suptitle("Per-class GGM degree distributions (378 common genes, $\\alpha$=0.07)",
                 y=1.02)
    plt.tight_layout()
    save(fig, "part4_degree_distributions")

def _load_gene_symbols():
    """Load HUGO gene symbols. Returns a numpy array indexed by global gene id."""
    gene_names = pd.read_csv(DATA_DIR / "gene_names.csv")
    # The CSV has columns "Unnamed: 0" (index) and "0" (the actual symbol).
    return gene_names["0"].values


def _build_gene_network(class_id: int, ggm_root, common_symbols):
    """Build a networkx Graph for a per-class GGM, with biological metadata.

    Each node carries: symbol (str), family (str), degree (int).
    Each edge carries: weight (|precision|).
    """
    import networkx as nx
    data = ggm_root["per_class"][class_id]
    precision = data["precision_"]
    degree = data["degree"]
    n = precision.shape[0]

    G = nx.Graph()
    for i in range(n):
        sym = str(common_symbols[i])
        G.add_node(
            i,
            symbol=sym,
            family=_gene_family(sym),
            degree=int(degree[i]),
        )
    # Off-diagonal upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(precision[i, j])
            if w > 0:
                G.add_edge(i, j, weight=float(w))
    return G


def _plot_full_network_panel(ax, G, ggm_data, title, seed: int = 42,
                                top_n_labels: int = 15):
    """Top-row panel: all nodes of the GGM with log-weighted edges."""
    import networkx as nx
    pos = nx.spring_layout(G, seed=seed, k=0.18, iterations=80)

    precision = ggm_data["precision_"]
    edge_data = [(u, v, abs(precision[u, v])) for u, v in G.edges()]

    if edge_data:
        log_weights = np.log10([w for _, _, w in edge_data])
        lw_lo, lw_hi = log_weights.min(), log_weights.max()
        if lw_hi > lw_lo:
            normed = (log_weights - lw_lo) / (lw_hi - lw_lo)
        else:
            normed = np.zeros_like(log_weights)
        linewidths = 0.01 + 1.1 * normed
        alphas = 0.05 + 0.50 * normed

        for (u, v, _), lw, al in zip(edge_data, linewidths, alphas):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="green", alpha=al,
                    linewidth=lw, zorder=1)

    # Group nodes by family for legend
    families_present = sorted({G.nodes[n]["family"] for n in G.nodes()})
    if "Other" in families_present:
        families_present.remove("Other")
        families_present = ["Other"] + families_present
    for fam in families_present:
        nodes_in_fam = [n for n in G.nodes() if G.nodes[n]["family"] == fam]
        sizes_in_fam = [25 + G.nodes[n]["degree"] * 0.5 for n in nodes_in_fam]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes_in_fam,
                                node_size=sizes_in_fam,
                                node_color=FAMILY_COLORS[fam],
                                edgecolors="white", linewidths=0.3,
                                label=fam)

    nodes_sorted = sorted(G.nodes(), key=lambda n: -G.nodes[n]["degree"])
    top_hubs = nodes_sorted[:top_n_labels]
    labels = {n: G.nodes[n]["symbol"] for n in top_hubs}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7,
                             font_weight="bold", font_color="black")

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


def _plot_chord_panel(ax, G_full, ggm_data, title, top_n: int = 35):
    """Bottom-row panel: circular layout of top-n hubs with log-weighted edges."""
    nodes_sorted = sorted(G_full.nodes(), key=lambda n: -G_full.nodes[n]["degree"])
    top_hubs = nodes_sorted[:top_n]
    G_sub = G_full.subgraph(top_hubs).copy()

    family_order = [
        "SMC / smooth muscle",
        "Inflammatory / immune",
        "ECM / matrix",
        "Adaptive immune (B/T/plasma)",
        "Epithelial / Paneth",
        "Other",
    ]

    def family_sort_key(n):
        fam = G_sub.nodes[n]["family"]
        fam_idx = family_order.index(fam) if fam in family_order else 999
        return (fam_idx, -G_sub.nodes[n]["degree"])

    sorted_hubs = sorted(G_sub.nodes(), key=family_sort_key)
    n_nodes = len(sorted_hubs)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_nodes, endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(sorted_hubs, angles)}

    precision = ggm_data["precision_"]
    edge_data = [(u, v, abs(precision[u, v])) for u, v in G_sub.edges()]

    if edge_data:
        log_weights = np.log10([w for _, _, w in edge_data])
        lw_lo, lw_hi = log_weights.min(), log_weights.max()
        if lw_hi > lw_lo:
            normed = (log_weights - lw_lo) / (lw_hi - lw_lo)
        else:
            normed = np.zeros_like(log_weights)
        linewidths = normed     # range [0, 1], thin edges very thin
        alphas = 0.15 + 0.55 * normed

        for (u, v, _), lw, al in zip(edge_data, linewidths, alphas):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="purple", alpha=al,
                    linewidth=lw, zorder=1)

    for n_id in sorted_hubs:
        fam = G_sub.nodes[n_id]["family"]
        deg = G_sub.nodes[n_id]["degree"]
        ax.scatter([pos[n_id][0]], [pos[n_id][1]], s=70 + 8 * deg,
                   c=FAMILY_COLORS[fam], edgecolors="black", linewidths=0.6,
                   zorder=3)

    for n_id, angle in zip(sorted_hubs, angles):
        symbol = G_sub.nodes[n_id]["symbol"]
        x = 1.15 * np.cos(angle)
        y = 1.15 * np.sin(angle)
        rot_deg = np.degrees(angle)
        if -90 < rot_deg < 90:
            ha = "left"
        else:
            ha = "right"
            rot_deg += 180
        ax.text(x, y, symbol, ha=ha, va="center", fontsize=8,
                rotation=rot_deg, rotation_mode="anchor",
                family="monospace")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def fig_part4_gene_networks():
    """
    Two-cell-type comparison of GGM gene networks:
    Myeloid cells (c3, cell-intrinsic identity preserved) vs.
    B cells (c0, dominated by neighbourhood signal under diffusion).

    Layout: 2x2 figure.
      - Top row: full GGM (378 nodes) with log-weighted edges, top-15 hubs labelled.
      - Bottom row: chord diagram of top-35 hubs, sorted by family,
        with log-weighted edges and outside-circle gene labels.
    Single legend at the bottom mapping family -> color.
    """
    print("[part4] gene networks (Myeloid vs B cells)")
    import networkx  # noqa: F401  (imported by sub-helpers, fail early if missing)
    from matplotlib.lines import Line2D

    arts = _load_part4_artifacts()
    if "ggms_alpha_007_common378" not in arts:
        print("  GGM artifacts not found, skipping")
        return
    ggm_root = arts["ggms_alpha_007_common378"]
    common_idx = ggm_root.get("common_gene_global")
    if common_idx is None:
        print("  common_gene_global not in artifacts, skipping")
        return

    try:
        all_symbols = _load_gene_symbols()
    except FileNotFoundError:
        print("  data/gene_names.csv not found, skipping")
        return
    common_symbols = all_symbols[common_idx]

    G_myeloid = _build_gene_network(3, ggm_root, common_symbols)
    G_bcell = _build_gene_network(0, ggm_root, common_symbols)

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1],
                            hspace=0.10, wspace=0.05)
    ax_full_m = fig.add_subplot(gs[0, 0])
    ax_full_b = fig.add_subplot(gs[0, 1])
    ax_chord_m = fig.add_subplot(gs[1, 0])
    ax_chord_b = fig.add_subplot(gs[1, 1])

    _plot_full_network_panel(
        ax_full_m, G_myeloid, ggm_root["per_class"][3],
        "Myeloid cells (c3): full GGM (378 nodes, log-weighted edges)",
        top_n_labels=15,
    )
    _plot_full_network_panel(
        ax_full_b, G_bcell, ggm_root["per_class"][0],
        "B cells (c0): full GGM (378 nodes, log-weighted edges)",
        top_n_labels=15,
    )
    _plot_chord_panel(
        ax_chord_m, G_myeloid, ggm_root["per_class"][3],
        "Myeloid cells (c3): top 35 hubs", top_n=35,
    )
    _plot_chord_panel(
        ax_chord_b, G_bcell, ggm_root["per_class"][0],
        "B cells (c0): top 35 hubs", top_n=35,
    )

    # Common family legend at the bottom of the figure
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=fam,
               markerfacecolor=col, markeredgecolor="black",
               markeredgewidth=0.5, markersize=10)
        for fam, col in FAMILY_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               frameon=True, framealpha=1.0, facecolor="white",
               edgecolor="#cccccc", fontsize=9,
               bbox_to_anchor=(0.5, 0.03))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "part4_gene_networks", png=True, dpi=300)

# =============================================================================
# PART 4 — gene network di una SINGOLA classe, in due immagini separate hi-res
# =============================================================================
def fig_part4_gene_network_single(
    class_id: int,
    *,
    top_n_labels: int = 15,
    top_n_chord: int = 35,
    seed: int = 42,
    dpi: int = 400,
    full_figsize=(10, 10),
    chord_figsize=(10, 10),
    font_scale: float = 1.0,
    legend_fontsize: float = 9,
    chord_edge_color: str | None = None,
    chord_edge_alpha_boost: float = 1.0,
    full_edge_color: str | None = None,
    bare: bool = False,
    name_prefix: str = "part4_gene_network",
):
    """Genera DUE immagini separate ad alta risoluzione per la GGM di una classe.

      1) <prefix>_full_c{id}_{slug}   -> full network (tutti i nodi, archi log-pesati)
      2) <prefix>_chord_c{id}_{slug}  -> chord dei top-{top_n_chord} hub

    Rispetta la modalità trasparente globale (TRANSPARENT): se attiva, salva PNG
    a sfondo trasparente in report/figures/transparent/ con inchiostro chiaro.
    Riusa gli helper _plot_full_network_panel / _plot_chord_panel: il look e' identico
    ai pannelli della figura 2x2, ma ogni pannello vive in un file proprio.
    """
    print(f"[part4] gene network singolo (classe c{class_id})")
    import networkx  # noqa: F401  (usato dagli helper; fallisce subito se assente)
    from matplotlib.lines import Line2D

    set_style()  # idempotente; applica lo stile, incl. inchiostro chiaro se TRANSPARENT

    CLASS_NAMES = {
        0: "B cells", 1: "Endothelial", 2: "Fibroblasts", 3: "Myeloid",
        4: "Normal epi.", 5: "SMC", 6: "T cells", 7: "Tumor",
    }
    cell_name = CLASS_NAMES.get(class_id, f"class {class_id}")
    slug = (cell_name.lower().replace(" ", "_").replace(".", "")
            .replace("/", "_"))

    # --- carico gli artefatti GGM ---
    arts = _load_part4_artifacts()
    if "ggms_alpha_007_common378" not in arts:
        print("  GGM artifacts non trovati, skip"); return
    ggm_root = arts["ggms_alpha_007_common378"]
    common_idx = ggm_root.get("common_gene_global")
    if common_idx is None:
        print("  common_gene_global assente, skip"); return
    if class_id not in ggm_root["per_class"]:
        print(f"  classe {class_id} assente in per_class, skip"); return
    try:
        all_symbols = _load_gene_symbols()
    except FileNotFoundError:
        print("  data/gene_names.csv non trovato, skip"); return
    common_symbols = all_symbols[common_idx]

    G = _build_gene_network(class_id, ggm_root, common_symbols)
    ggm_data = ggm_root["per_class"][class_id]

    # legenda famiglie (riusata su entrambe le figure)
    edge_ink = LIGHT_INK if TRANSPARENT else "black"
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=fam,
               markerfacecolor=col, markeredgecolor=edge_ink,
               markeredgewidth=0.5, markersize=10)
        for fam, col in FAMILY_COLORS.items()
    ]

    def _add_legend(fig):
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   frameon=not TRANSPARENT, framealpha=1.0,
                   facecolor=("none" if TRANSPARENT else "white"),
                   edgecolor=("none" if TRANSPARENT else "#cccccc"),
                   fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.0))

    def _scale_text(ax):
        # gli helper fissano i font (label, titolo) a valori hardcoded:
        # qui li riscalo a posteriori senza toccare gli helper.
        if font_scale != 1.0:
            for t in list(ax.texts) + [ax.title]:
                t.set_fontsize(t.get_fontsize() * font_scale)

    def _restyle_edges(ax, color, alpha_boost):
        # gli archi sono Line2D in ax.lines (i nodi sono scatter/collections,
        # le label sono testo): li ricoloro/rinforzo mantenendo il gradiente
        # di alpha gia' calcolato dall'helper.
        for ln in ax.lines:
            if color is not None:
                ln.set_color(color)
            if alpha_boost != 1.0:
                a = ln.get_alpha()
                if a is not None:
                    ln.set_alpha(min(1.0, a * alpha_boost))

    # ---------- 1) FULL NETWORK ----------
    fig_full = plt.figure(figsize=full_figsize)
    ax_full = fig_full.add_subplot(111)
    _plot_full_network_panel(
        ax_full, G, ggm_data,
        f"{cell_name} (c{class_id}): full GGM "
        f"({G.number_of_nodes()} nodi, archi log-pesati)",
        seed=seed, top_n_labels=top_n_labels,
    )
    _restyle_edges(ax_full, full_edge_color, 1.0)
    if bare:
        ax_full.set_title("")
    if TRANSPARENT:
        # i top-hub label sono hardcoded "black": su deck scuro non si leggono
        for t in ax_full.texts:
            t.set_color(LIGHT_INK)
    _scale_text(ax_full)
    if not bare:
        _add_legend(fig_full)
    fig_full.tight_layout(rect=[0, 0, 1, 1] if bare else [0, 0.05, 1, 1])
    save(fig_full, f"{name_prefix}_full_c{class_id}_{slug}", png=True, dpi=dpi)

    # ---------- 2) CHORD ----------
    fig_chord = plt.figure(figsize=chord_figsize)
    ax_chord = fig_chord.add_subplot(111)
    _plot_chord_panel(
        ax_chord, G, ggm_data,
        f"{cell_name} (c{class_id}): top {top_n_chord} hub",
        top_n=top_n_chord,
    )
    _restyle_edges(ax_chord, chord_edge_color, chord_edge_alpha_boost)
    if bare:
        ax_chord.set_title("")
    _scale_text(ax_chord)
    if not bare:
        _add_legend(fig_chord)
    fig_chord.tight_layout(rect=[0, 0, 1, 1] if bare else [0, 0.05, 1, 1])
    save(fig_chord, f"{name_prefix}_chord_c{class_id}_{slug}", png=True, dpi=dpi)

# =============================================================================
# PART 5 FIGURES
# =============================================================================

def _load_part5_artifacts():
    """Load the cached Part 5 niche GGM artifacts."""
    import pickle
    ggm_dir = CACHE / "ggm_niches"
    out = {}
    for name in ["niche_ggms_alpha_007_common378.pkl", "part5_summary.pkl"]:
        p = ggm_dir / name
        if p.exists():
            with open(p, "rb") as f:
                out[name.replace(".pkl", "")] = pickle.load(f)
            print(f"  loaded {name}")
    return out


def fig_part5_triple_jaccard():
    """
    Three side-by-side heatmaps: edge Jaccard, hub Jaccard, composition cosine
    between the 20 niches. Headline figure of Part 5.
    """
    print("[part5] triple Jaccard heatmap")
    arts = _load_part5_artifacts()
    if "part5_summary" not in arts:
        print("  skipping (no part5_summary)")
        return

    summary = arts["part5_summary"]
    edge_j = np.asarray(summary.get("edge_jaccard"))
    hub_j = np.asarray(summary.get("hub_jaccard"))
    comp_c = np.asarray(summary.get("comp_cos"))

    if edge_j is None or hub_j is None or comp_c is None:
        print("  missing one of edge/hub/comp matrices, skipping")
        return

    n = edge_j.shape[0]

    # Per-niche dominant class for labels
    niche_summary = summary.get("niche_summary", [])
    if niche_summary:
        labels = []
        for i in range(n):
            row = niche_summary[i] if i < len(niche_summary) else {}
            dc = row.get("dom_class", row.get("dominant_class", "?"))
            labels.append(f"{i}(c{dc})")
    else:
        labels = [str(i) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    for ax, mat, title in [
        (axes[1], hub_j, "Hub Jaccard (top-15)"),
        (axes[2], edge_j, "Edge Jaccard"),
        (axes[0], comp_c, "Composition cosine"),
    ]:
        mat_show = mat.copy().astype(float)
        np.fill_diagonal(mat_show, np.nan)
        masked = np.ma.masked_invalid(mat_show)
        # Use vmax = max off-diagonal so each panel's internal contrast is visible
        offdiag_max = float(np.nanmax(mat_show))
        im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=offdiag_max,
                       aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{title}  (max off-diag = {offdiag_max:.2f})")
        ax.grid(False)
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    save(fig, "part5_triple_jaccard", dpi=300)


def fig_part5_dendrograms():
    """
    Three dendrograms (edge / hub / composition), side-by-side.
    """
    print("[part5] three dendrograms")
    from scipy.cluster.hierarchy import linkage, dendrogram

    arts = _load_part5_artifacts()
    if "part5_summary" not in arts:
        print("  skipping (no part5_summary)")
        return
    summary = arts["part5_summary"]

    matrices = {
        "Composition cosine": np.asarray(summary["comp_cos"]),
        "Hub Jaccard": np.asarray(summary["hub_jaccard"]),
        "Edge Jaccard":          np.asarray(summary["edge_jaccard"]),
    }

    niche_summary = summary.get("niche_summary", [])
    if niche_summary:
        labels = []
        for i in range(len(niche_summary)):
            row = niche_summary[i]
            dc = row.get("dom_class", row.get("dominant_class", "?"))
            labels.append(f"{i}(c{dc})")
    else:
        n = matrices["Edge Jaccard"].shape[0]
        labels = [str(i) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for ax, (title, mat) in zip(axes, matrices.items()):
        # Distance = 1 - similarity
        dist = 1 - mat
        np.fill_diagonal(dist, 0)
        # Symmetrize and pull out the upper triangle as condensed
        from scipy.spatial.distance import squareform
        cond = squareform(dist, checks=False)
        Z = linkage(cond, method="average")
        dendrogram(Z, labels=labels, ax=ax, leaf_font_size=8,
                   leaf_rotation=90, color_threshold=0)
        ax.set_title(f"{title} -- avg linkage")
        ax.set_ylabel("distance")
        ax.grid(False)

    plt.tight_layout()
    save(fig, "part5_dendrograms")


def fig_part5_patient_jaccard():
    """
    Patient-set Jaccard heatmap between niches.
    """
    print("[part5] patient-set Jaccard")
    arts = _load_part5_artifacts()
    if "part5_summary" not in arts:
        print("  skipping")
        return
    summary = arts["part5_summary"]
    pat = np.asarray(summary.get("pat_jaccard"))
    if pat is None:
        print("  missing pat_jaccard, skipping")
        return

    n = pat.shape[0]
    niche_summary = summary.get("niche_summary", [])
    if niche_summary:
        labels = []
        for i in range(len(niche_summary)):
            row = niche_summary[i]
            dc = row.get("dom_class", row.get("dominant_class", "?"))
            labels.append(f"{i}(c{dc})")
    else:
        labels = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    mat = pat.copy().astype(float)
    np.fill_diagonal(mat, np.nan)
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Patient-set Jaccard between niches  (cohort-wide)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.85, label="patient-set Jaccard")
    plt.tight_layout()
    save(fig, "part5_patient_jaccard")


# =============================================================================
# Main entry point
# =============================================================================
def main(transparent: bool = False):
    global TRANSPARENT
    TRANSPARENT = transparent
    set_style()
    print(f"output directory: {FIGDIR}")
    print()

    print("=" * 60)
    print("Intro figures")
    print("=" * 60)
    fig_intro_data_to_graph(sample_id=47, seed=0)
    print()

    print("=" * 60)
    print("Part 1 figures")
    print("=" * 60)
    fig_part1_degree_distribution()
    fig_part1_bimodality()
    fig_part1_composition_topology_heatmap()
    fig_part1_mask_vs_degree()
    fig_part1_betweenness_skeleton()
    fig_part1_anomaly_handling()

    print()
    print("=" * 60)
    print("Part 2 figures")
    print("=" * 60)
    fig_part2_niche_centroid_heatmap()
    fig_part2_spatial_niches(samples=(2, 20))
    fig_part2_niche_recurrence()

    print()
    print("=" * 60)
    print("Part 3 figures")
    print("=" * 60)
    fig_part3_rung_comparison()
    fig_part3_confusion_matrix()
    fig_part3_per_class_progression()
    fig_part3_training_curves()
    fig_part3_per_niche_test_f1()

    print()
    print("=" * 60)
    print("Part 4 figures")
    print("=" * 60)
    fig_part4_hub_jaccard_heatmap()
    fig_part4_degree_distributions()
    fig_part4_gene_networks()

    print()
    print("=" * 60)
    print("Part 5 figures")
    print("=" * 60)
    fig_part5_triple_jaccard()
    fig_part5_dendrograms()
    fig_part5_patient_jaccard()

    print()
    print("done.")


if __name__ == "__main__":
    main(transparent=True)