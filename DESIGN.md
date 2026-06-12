# Project Design Document

**Course.** Advanced Statistical Learning for Complex Data — PhD Data Analytics and Decision Sciences, PoliMi / Human Technopole.
**Topic.** Graph Theory & Complex Networks (Lesson 10.1, Ieva) and Network Theory for Multi-Modal Data (Lesson 10.2, Mapelli).
**Status.** Part 0 (data exploration) complete. Infrastructure & Part 1 next.

---

## 1. Dataset overview

Per-cell spatial transcriptomics from **Visium HD** paired with H&E imaging, preprocessed by a pipeline upstream of us. All data lives in a single Zarr store on the cluster.

### 1.1 Scale and structure

- **8,065,561 cells** total.
- **56 patients**, **2 samples per patient** (uniformly — every patient has exactly 2), **112 samples** with globally-unique IDs.
- Cells per sample: 981 (min) — 50,268 (median) — 231,682 (max). Wide spread; one tiny sample (981 cells) worth flagging in downstream analyses.
- Cells per patient: 42,564 — 139,437 — 273,007.

### 1.2 Modalities and labels per cell

| Path | Shape | Dtype | Notes |
|---|---|---|---|
| `numerical/expression` | (8.07M, 16,497) | float32 | Gene expression (16,497 genes). Pre-normalized; see §1.4. |
| `numerical/HE_embeddings` | (8.07M, 1,280) | float32 | Foundation-model patch embeddings (likely UNI/ViT-L family). |
| `numerical/positions` | (8.07M, 2) | float32 | (x, y) in image pixel coordinates. |
| `numerical/cell_area` | (8.07M,) | float32 | Morphology: segmentation-derived. |
| `numerical/cell_eccentricity` | (8.07M,) | float32 | ∈ [0, 1]. |
| `numerical/cell_extent` | (8.07M,) | float32 | Area / bounding-box area, ∈ (0, 1]. |
| `numerical/cell_perimeter` | (8.07M,) | float32 | In pixels. |
| `numerical/neighbors_idxs` | (8.07M, 8) | int32 | Spatial KNN with k=8; -1 padding for missing neighbors. |
| `numerical/neighbors_radius` | (8.07M, 8) | float32 | Edge distances, normalized to [0, 1]. |
| `numerical/neighbors_angle` | (8.07M, 8) | float32 | Edge angles. |
| `labels/cell_types` | (8.07M, 8) | bool | One-hot, strict (verified). 8 classes. |
| `labels/patient` | (8.07M,) | int8 | 0–55. |
| `labels/sample` | (8.07M,) | int8 | 0–111, globally unique. |
| `labels/idx_to_take_into_training_based_on_grid` | (8.07M,) | bool | Provided training-set mask. See §1.5. |

### 1.3 Spatial graph integrity (verified on full dataset)

- **No cross-patient edges** (0 / 64M edges).
- **454 cross-sample edges** within the *same patient* (0.0008% of valid edges) — small bug from preprocessing; we'll filter in graph construction.
- **0 self-loops, 0 real duplicates.**
- **66,796 cells (0.83%) have zero valid neighbors** after the radius threshold — isolated nodes.
- **11.4% of neighbor slots are -1** (smooth gradient from 0 to 8 effective neighbors per cell — signature of radius-thresholded KNN, not noise).
- Neighbor radius is **normalized to [0, 1]** in every sample (max is exactly 1.0 — distances were divided by the threshold). Distributions overlap perfectly across samples → no per-sample calibration needed.

### 1.4 Expression — pre-normalized

- **Log-normalized**, not raw counts. 0% integer-valued, no negatives.
- **Capped at 5.0** (max value 4.999984, sharp histogram cutoff).
- **98.3% sparse**. ~280 expressed genes per cell on average.
- Library sizes in log space: median 124.5, std 31 — cells are well-equalized.
- Decision: **use expression as-is**. No additional normalization. Z-score per gene only when a model needs it (e.g., GGM in Part 4).

### 1.5 The training mask — class-aware spatial decorrelation

- Keeps 79.08% of cells overall.
- Per-sample keep fraction varies 0.58–1.00 (std 0.085).
- **Class-aware**: class 6 keeps 62.7%, class 5 keeps 94.6% (32-point gap).
- Visual inspection on sample 47: exclusion concentrates in dense same-type regions (one clear tumor/aggregate visible as a uniform mass), not in a regular grid pattern.
- **Interpretation.** The mask thins dense same-type regions to spatially-decorrelate training data, preventing the GNN from trivially learning "you are what your dense same-type neighborhood is."
- **Policy:** use the mask **for Part 3 training only**. Ignore it everywhere else (we want the *real* graph, not a subsample, for topology, communities, GGM, and multi-layer analysis).
- **Evaluation policy.** Train on `in_mask=True` within training patients. Validate on held-out patients with **no mask filtering** (natural distribution).

### 1.6 Array layout caveats

- Cells of the same patient are blocked together in the array (no cross-patient interleaving).
- Within a patient, the two samples' cells are **interleaved**. Only 2 of 112 samples are contiguous in their index range.
- This makes per-sample slicing of large arrays (expression, H&E) expensive via fancy indexing. **One-time materialization pass** will write per-sample contiguous artifacts (see §3.1).

---

## 2. Project scope — 5-part arc

We commit to all five themes, with Part 5 explicitly exploratory.

### Part 1 — Spatial graph topology
*Question.* Are spatial topological properties consistent across patients/samples? Where they differ, do differences track with biology?
*Methods.* Per-sample graphs from `neighbors_idxs`. Degree distribution, clustering coefficient, betweenness/closeness centrality (sampled), assortativity by cell type, sample-comparative statistics. Use full data (no mask).

### Part 2 — Community detection → tissue niches
*Question.* Does unsupervised community detection on the spatial graph recover biologically meaningful tissue niches?
*Methods.* Leiden per-sample with resolution sweep. Characterize communities by cell-type composition and gene expression. Cross-sample alignment via composition fingerprints (Hungarian assignment).

### Part 3 — Multi-modal GNN for cell typing
*Question.* Beyond what each cell knows about itself, how much does spatial context contribute to cell-type prediction?
*Methods.* Ablation ladder: MLP(expr) → MLP(expr+H&E+morph) → GraphSAGE(expr) → GraphSAGE(full) → GAT/Graph-Transformer(full). PyG `NeighborLoader`. Use the training mask. Macro-F1, per-class F1, confusion matrices, per-patient breakdown. GAT attention as interpretability.

### Part 4 — Gaussian graphical models on gene expression
*Question.* Are gene-gene regulatory relationships shared across cell types, or type-specific? Hub genes per type?
*Methods.* Restrict to top ~500 HVGs (genes with >5% non-zero rate). Per-cell-type graphical lasso via `sklearn.covariance.GraphicalLassoCV`. Per-type network analysis: hub identification, gene-module communities, cross-type comparison (Jaccard / edge overlap).

### Part 5 — Multi-layer integration (exploratory)
*Question.* Can spatial cell graph (Part 1) + gene network (Part 4) combine into a useful multi-layer representation?
*Methods.*
- **(a) Descriptive.** Define cell layer + gene layer + cross-layer edges. Compute multi-layer centrality on a representative subset.
- **(b) Predictive (ambitious).** Use Part 4's gene network as a prior in Part 3's GNN: propagate expression features through the gene graph before message-passing on cells. Compare to Part 3 baselines.

---

## 3. Infrastructure decisions

### 3.1 Per-sample materialization

Write a one-time preprocessing pass (`scripts/materialize_per_sample.py`) that produces:

```
cache/
├── per_sample/
│   ├── sample_000.npz   # expression, HE_embeddings (contiguous per sample)
│   ├── sample_001.npz
│   └── ...
└── global/
    ├── cell_type.npy        # int8, 0-7
    ├── patient.npy          # int8
    ├── sample.npy           # int8
    ├── train_mask.npy       # bool
    ├── positions.npy        # float32 (N, 2)
    ├── neighbors_idxs.npy   # int32 (N, 8)
    ├── neighbors_radius.npy # float32 (N, 8)
    └── morphology.npy       # float32 (N, 4)
```

Rationale: heavy modalities (expression, H&E) materialized per-sample for cheap contiguous loading; small/global arrays stay global since loading them in full is trivial. Cost: ~15 min one-shot. Benefit: every downstream per-sample read is fast.

### 3.2 Train/val/test split

**Fixed once, used everywhere.** Patient-level 5-fold cross-validation, plus a held-out test set.
- **40 patients training**, **8 patients validation**, **8 patients test**, fixed via seeded random partition.
- For the GNN: use 5-fold CV within the 40+8 = 48 train/val patients for hyperparameter selection; final results on the 8 test patients.
- Saved to `cache/splits/patient_split.json` once, never regenerated.
- For Part 3, an additional **within-patient cross-sample** sanity check: train on sample-1 cells of training patients, test on sample-2 cells of same patients.

### 3.3 Repo skeleton

```
project/
├── README.md
├── environment.yml
├── data/
│   └── zarr_path.txt              # cluster path, gitignored
├── cache/                          # gitignored
│   ├── per_sample/
│   ├── global/
│   └── splits/
├── src/
│   ├── __init__.py
│   ├── io.py                       # loaders (zarr + cache)
│   ├── preprocessing.py            # HVG selection, standardization
│   ├── graph_construction.py       # neighbors -> PyG Data, edge filtering
│   ├── splits.py                   # patient-level CV
│   ├── topology.py                 # Part 1
│   ├── communities.py              # Part 2
│   ├── models.py                   # Part 3 GNN architectures
│   ├── ggm.py                      # Part 4
│   └── multilayer.py               # Part 5
├── scripts/                        # SLURM-runnable
│   ├── materialize_per_sample.py
│   ├── compute_topology.py
│   ├── detect_niches.py
│   ├── train_gnn.py
│   ├── fit_ggm.py
│   └── multilayer_analysis.py
├── slurm/
│   └── *.sbatch
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_spatial_graph_topology.ipynb
│   ├── 02_niche_detection.ipynb
│   ├── 03_gnn_cell_typing.ipynb
│   ├── 04_gene_networks.ipynb
│   └── 05_multilayer.ipynb
└── results/                        # figures, tables, models
```

### 3.4 Environment

Python 3.12. Key dependencies: `numpy`, `scipy`, `pandas`, `zarr>=3`, `networkx`, `python-igraph`, `leidenalg`, `scikit-learn` (for graphical lasso), `torch`, `torch-geometric`, `scanpy` (optional, for HVG selection), `matplotlib`, `seaborn`. CUDA build of PyTorch matching cluster setup.

---

## 4. Pre-committed handling of data quirks

These are decisions made during exploration; capturing them so we apply them consistently.

| Quirk | Resolution |
|---|---|
| -1 padding in `neighbors_idxs` | Treat as "no edge"; filter before PyG `Data` construction. |
| 454 cross-sample edges | Filter on `sample[src] == sample[dst]` in graph construction. |
| 66,796 isolated cells | Keep in feature-only baselines; exclude from spatial models (or pass through GraphSAGE with empty neighborhoods). |
| 4+ globally silent genes | Drop in HVG selection step. |
| Tiny cells (area < 100 px) | Keep; mention in report as a known noise source. |
| Sample 107 (981 cells) | Flag in any per-sample comparison; omit from per-sample averages where it would skew. |
| H&E asymmetric range (-43 to 28) | Ignore (outliers from segmentation edge cases). Per-dim z-score in models that need it. |
| Morphology scale heterogeneity | Z-score within `preprocessing.py`. |

---

## 5. Timeline (4 weeks)

| Week | Deliverable |
|---|---|
| 1 | Infrastructure (materialization, splits, repo). Part 1 notebook. |
| 2 | Part 2 (niches), start Part 3 (baselines + GraphSAGE). |
| 3 | Finish Part 3 (GAT, evaluation). Part 4 (GGM). |
| 4 | Part 5 (multi-layer). README, results consolidation. |

---

*Last updated: end of Part 0 — data exploration.*

---

## Status of analyses

| Part | Status | Notebook | Key cache outputs |
|---|---|---|---|
| Part 0 | Complete | `00_data_exploration.ipynb` | `cache/global/*.npy`, `cache/per_sample/sample_NNN.npz`, `cache/splits/patient_split.json` |
| Part 1 | Complete | `01_spatial_graph_topology.ipynb` | `cache/topology/free_tier.parquet`, `cache/topology/expensive_tier.parquet` |
| Part 2 | Complete | `02_niche_detection.ipynb` | `cache/communities/leiden_labels.npy`, `cache/communities/per_sample_summary.parquet`, `cache/communities/communities_table.parquet`, `cache/communities/niche_assignments.parquet`, `cache/communities/niche_centroids.parquet`, `cache/communities/patient_niche_matrix.parquet`, `cache/communities/niche_per_cell.npy` |
| Part 3 | Not started | `03_gnn_cell_typing.ipynb` | — |
| Part 4 | Not started | `04_gene_networks.ipynb` | — |
| Part 5 | Not started | `05_multilayer.ipynb` | — |

## Part 2 findings (headline)

- Per-sample Leiden at γ=0.5 on weighted giant-component subgraphs produces median 76 communities per sample, scaling with giant size (Spearman ρ = 0.83).
- 11,496 non-anomaly communities; 9,457 pass the ≥100-cell filter for niche alignment.
- Hierarchical clustering (Ward, Euclidean, K=20) over composition vectors yields 20 recurrent niche types arranged into 8 cell-class families, plus a class-6-as-secondary niche (niche 17).
- Niche recurrence forms a graded continuum: 4 universal niches (≥80% of patients, all class-7 dominated), 9 common niches, 5 selective niches, 2 rare niches (pure class 4 and pure class 5 enclaves).
- 77.0% of cells (6.21M) received a niche assignment; the remainder are in non-giant components, small communities, or anomaly samples.

## Update to §3.3 (repo skeleton)

Notebook list now includes `02_niche_detection.ipynb` (complete). Add a `cache/communities/` directory entry analogous to `cache/topology/`.

### Part 3 progress notes

**Rungs 1-2 establish a per-cell-feature ceiling of macro-F1 ≈ 0.57 on val patients.**

Rung 1 (multinomial LogReg on 1000 HVG expression features, class-weighted) reaches macro-F1 = 0.552. Rung 2's flat MLP on concatenated features required a critical feature-scale fix (H&E embeddings have per-cell L2 norm ~100 vs HVG ~1.5, requiring per-modality standardization) and reached only 0.557. Modality-specific encoders (HVG → 256, H&E → 256, morph → 16; then fused MLP head) lifted Rung 2's macro-F1 to 0.570 — a +1.4pp improvement over the flat architecture.

To quantify each modality's actual contribution, we zeroed each modality at inference time on the encoder-based model: removing HVG dropped macro-F1 by **39.0pp** (to 0.181), removing H&E dropped it by **8.5pp** (to 0.486), and removing morphology dropped it by **0.1pp** (to 0.570). The interpretation: HVG dominates the per-cell signal; H&E provides a real but smaller secondary contribution that is only accessible with modality-aware architectures; morphology features in their current form (cell area, eccentricity, extent, perimeter — all derived from the segmentation pipeline) carry essentially no class-discriminative information.

The Rung 1 → Rung 2 per-class breakdown shows the gains concentrate in some classes (notably class 0, +6.7pp from 0.452 to 0.519) but the class-0-vs-class-6 confusion (~24% of class-0 cells predicted as class 6 in Rung 1) persists in Rung 2. From Part 2, class 6 lives spatially as a minority component within class-0-dominated niches; the persistence of this confusion across all per-cell variants is the most concrete prediction that Rung 3 should test — namely, that spatial neighborhood context, not additional per-cell modalities, is what disambiguates these classes.

**Rungs 3-5: The spatial graph adds 6-8 percentage points of macro-F1; nothing beyond that helps.**

Rung 3 (GraphSAGE on HVG + spatial graph, 2 layers, hidden=256) reaches macro-F1 = 0.635 on val patients, a substantial +6.5pp gain over Rung 2's per-cell-feature MLP (0.570). The per-class breakdown shows uniform improvement: class 0 F1 jumps from 0.519 to 0.615 (+9.6pp), class 6 F1 from 0.387 to 0.416 (+2.9pp), and middle-abundance classes 1-3 all gain 6-10pp. Per-patient, all 8 validation patients improved (mean +5.6pp, range +0.8pp to +10.3pp). Per-niche, the largest improvements concentrate on niches dominated by classes 1-4 (gains of +6 to +8pp on niches 9, 11, 12); niche 17 (the class-0 + class-6 mediator niche from Part 2) gains only +2.9pp, indicating the 2-hop graph context identifies the niche but doesn't fully disambiguate cells within it. One small niche (niche 15, pure class-5 enclave, n=1,336 val cells) gets slightly worse (-2.9pp), likely due to local class imbalance noise.

Rung 4 (GraphSAGE on fused HVG + H&E + morphology) reaches macro-F1 = 0.631 — slightly *worse* than Rung 3, with nearly twice the parameters (988k vs 521k). The graph context already captures whatever per-cell signal H&E and morphology might have contributed, and the additional capacity introduces mild overfitting. This is a sharper version of the Rung 2 finding: H&E adds minimal per-cell signal for cell-type prediction, and once spatial context is present, H&E adds nothing.

Rung 5 (GAT on fused features) reaches macro-F1 = 0.624, the worst of the three GNN configurations. Attention-weighted aggregation does not help on this graph — in a KNN graph with mean degree 8 where all neighbors are spatially close and roughly equally relevant, mean aggregation is sufficient. The attention layer adds parameter complexity without meaningful information gain.

**Summary of Part 3.** The headline finding is that spatial graph context contributes substantially more to cell-type classification than any per-cell modality: a 2-layer GraphSAGE on the spatial KNN graph using HVG expression alone outperforms a fused multi-modal MLP by 6.5pp. Adding modalities (Rung 4) or attention (Rung 5) on top of the graph does not improve performance. The graph is the answer; the architectural choices around the graph are largely neutral.

### Part 4 progress notes

**Question.** Do sparse precision-matrix estimates on HVG expression recover biologically interpretable, cell-type-specific gene co-expression structure?

**Approach.** For each of the 8 cell types, fit a Graphical Lasso on a class-balanced, patient-stratified subsample of 5,000 training cells. Restrict to 378 "common genes" — HVGs with std ≥ 0.05 in every cell type — to enable direct cross-class comparison. Tune α to 0.07 (achieving ~1% sparsity, biologically reasonable). Validate with 10 bootstrap resamples per class.

**Key findings.**

1. **Networks are class-specific and biologically coherent.** Each class's top-15 hub genes form a recognizable cell-type signature: class 0 (DES, MYH11, ACTA2, MYLK, CNN1 — smooth muscle); class 2 (CXCL8, CXCL5, IL1B, S100A8/A9 — neutrophil/acute inflammatory); class 3 (G0S2, CSF3R, BCL2A1, FOS, DUSP1 — activated granulocytes); class 4 (COL1A2, MFAP4, TNXB — fibroblast/stromal); class 6 (CTSG, MMP25, F13A1, DEFA6, IGHM, IGLC1 — plasma cells / mast cells / mature granulocytes); class 7 (COL1A2, TIMP3, ELN, MGP, SFRP4 — mature ECM-producing fibroblasts).

2. **Cell types share hub regulators but rewire their connections.** Hub-set Jaccard within a "muscle-inflammatory" cluster (classes 1, 2, 3, 5) is 0.25-0.30; cross-cluster Jaccard 0.07-0.15. Edge-set Jaccard, by contrast, is always below 0.09 — even among classes sharing 30% of hubs, only ~5-10% of specific gene-gene edges are shared. The biological interpretation: the same regulator genes (CXCL8, S100A8/A9, COL1A2) are active across multiple classes but connect to different downstream genes depending on cell-type context.

3. **Class 6 is mathematically and biologically distinct from all others.** Hub-set Jaccard with the other 7 classes is 0.000 (one shared gene with class 4: DEFA6). Network density is the lowest (494 edges, mean degree 2.6, max degree 10 — flat distribution with no dominant hubs). The hub signature (CTSG + MMP25 + F13A1 + DEFA6 + IGHM + IGLC1 + CHIT1) corresponds to a heterogeneous mix of mast cells, plasma cells, and tissue granulocytes. This explains the persistent classification difficulty observed in Part 3: class 6 cells use a fundamentally different transcriptional toolkit than the other classes, and likely represent a heterogeneous label combining several rare cell types.

4. **Hub identities are reproducible; specific edges less so.** Bootstrap stability shows top-15 hubs have CV < 0.20 for canonical markers, confirming biological identification. Network sparsity (n_edges per bootstrap) is highly reproducible (±3-5% within each class). Specific edges show moderate stability: 22-34% of headline edges appear in ≥8/10 bootstraps; the remainder are weaker conditional dependencies near the L1 threshold.

**Summary.** Per-class Gaussian Graphical Models on HVG expression recover known cell-type marker programs and reveal a "shared regulators, rewired connections" architecture for multi-cell-type co-expression. Class 6 emerges as biologically distinct — flat network, unique hub set, weak conditional dependencies — consistent with its role in Parts 2-3 as a minority/mediator population. The 378 common genes are mostly broadly-expressed regulatory genes; the ~622 dropped HVGs (variable in only some classes) are likely class-specific markers and would be the focus of a Track B class-specific analysis.

### Part 5 progress notes

**Question.** Do gene networks fitted per-niche (rather than per-cell-type as in Part 4) recover the same biological structure that Part 2's composition-based niche family analysis identified?

**Approach.** Apply Part 4's GGM machinery to the 20 niches identified in Part 2. Subsample 5,000 training-mask cells per niche (patient-stratified), fit Graphical Lasso at α=0.07 on the same 378 common genes used in Part 4. Compare cross-niche network similarity (edge Jaccard, top-15 hub Jaccard) with composition similarity from Part 2 and with patient cohort overlap.

**Key findings.**

1. **Niche networks vary in density but converge on a comparable per-gene network architecture.** All 20 networks have similar density per kept gene (edges-per-kept-gene 1.77-3.00 — a 1.7× spread). Variation in raw edge count (629-1087) is largely driven by silent-gene count differences across niches (5-32 silent genes per niche). Rare or pure niches (5, 14, 15) have more silent genes (their dominant cell types don't express many of the broadly-variable HVGs); broad niches (4, 1) have fewer.

2. **Gene network similarity partially recovers composition-based niche relationships.** Across all 190 niche pairs, edge Jaccard correlates with composition cosine at Spearman ρ = 0.615. Composition similarity is also more predictive of network similarity than patient overlap is (ρ = 0.575 patient↔composition; ρ = 0.42 patient↔edge). Gene networks track cell-type composition substantially better than sample-level cohort effects.

3. **At cluster-resolution, gene networks do NOT robustly recover Part 2's 8 cell-class families.** Hierarchical clustering on edge Jaccard at K=8 gives adjusted Rand Index of only 0.250 against Part 2's family assignments (composition-cosine clustering sanity-checks at ARI = 0.802). Hub Jaccard clustering performs essentially at chance (ARI = 0.054). The headline correlation captures pairwise concordance but the global cluster structure of gene-network space does not align with composition-space families.

4. **Niche 17 (the c0+c6 mediator) is a network outlier consistent with its Part 2/3 mediator role.** In edge-Jaccard clustering it splits off into its own singleton cluster. This is biologically meaningful: niche 17's gene network is dissimilar from other c0-dominant niches because it carries a heterogeneous cell composition (29% class 6 secondary) that gives it a different transcriptional architecture from pure-c0 niches (16, 18, 19). Gene-network analysis independently identifies the same niche as anomalous in a manner consistent with the spatial heterogeneity finding from Part 2.

**Summary.** Per-niche Gaussian Graphical Models reveal niche-specific gene regulatory architecture that partially echoes — but does not perfectly mirror — Part 2's composition-based niche structure. Niches share dominant regulators (CXCL8, S100A8/A9, COL1A2 as universal hubs in Part 4) but the specific gene-gene edge sets are largely niche-specific. The "shared regulators, rewired connections" architecture observed in Part 4 between cell types is recapitulated at the niche level. The independence of network structure from patient cohort effects (controlling for composition) confirms that the niche-level networks reflect biological microenvironment rather than sample-level artifacts.
