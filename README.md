# Spatial Graph Learning on Multi-Modal Visium HD Data

Course project for **Advanced Statistical Learning for Complex Data** (PhD DADS, PoliMi / Human Technopole), covering Lessons 10.1 (Graph Theory & Complex Networks) and 10.2 (Network Theory for Multi-Modal Data).

We analyse a large multi-modal spatial-transcriptomics dataset (Visium HD + H&E, ~8M cells, 56 patients) as a collection of **spatial cell graphs**, working through five connected themes from the two lessons.

## Dataset

Per-cell data for 8,065,561 cells from 56 patients (2 tissue sections each, 112 samples). Each cell has:
- **Gene expression** (16,497 genes, log-normalized, sparse)
- **H&E embeddings** (1,280-d foundation-model features)
- **Morphology** (area, eccentricity, extent, perimeter)
- **Spatial position** + a precomputed **k=8 spatial neighbor graph**
- A **cell-type label** (8 mutually exclusive classes)

See `DESIGN.md` for the full data exploration and all design decisions.

## Project structure

```
.
├── DESIGN.md                # data findings + all design decisions
├── environment.yml          # conda environment
├── data/
│   └── zarr_path.txt        # path to the source zarr (gitignored)
├── cache/                   # materialized data (gitignored, ~60 GB)
│   ├── global/              # small global arrays (.npy)
│   ├── per_sample/          # per-sample expression (sparse) + H&E (dense)
│   └── splits/              # patient_split.json
├── src/                     # reusable modules
│   ├── data_io.py           # cache loaders
│   ├── splits.py            # patient-level stratified split
│   ├── preprocessing.py     # HVG selection, standardization
│   ├── graph_construction.py# neighbor arrays -> PyG graphs
│   ├── topology.py          # Part 1
│   ├── communities.py       # Part 2
│   ├── models.py            # Part 3
│   ├── ggm.py               # Part 4
│   └── multilayer.py        # Part 5
├── scripts/                 # command-line entry points (SLURM-friendly)
│   ├── materialize_per_sample.py
│   ├── compute_splits.py
│   └── ...
├── slurm/                   # job submission scripts
├── notebooks/               # analysis + figures, one per part
└── results/                 # figures, tables, trained models
```

## Setup & run order

```bash
# 1. Environment
conda env create -f environment.yml
conda activate aslcd-spatial

# 2. Point to the source data
echo "/path/to/foundational_model_data.zarr" > data/zarr_path.txt

# 3. One-time materialization (~30-45 min, ~60 GB output)
python scripts/materialize_per_sample.py \
    --zarr-path "$(cat data/zarr_path.txt)" --cache-dir cache/

# 4. Compute the patient split (seconds)
python scripts/compute_splits.py --cache-dir cache/

# 5. Run the analyses (notebooks/ 01..05)
```

## The five parts

| Part | Theme | Notebook |
|------|-------|----------|
| 1 | Spatial graph topology | `01_spatial_graph_topology.ipynb` |
| 2 | Community detection → tissue niches | `02_niche_detection.ipynb` |
| 3 | Multi-modal GNN for cell typing | `03_gnn_cell_typing.ipynb` |
| 4 | Gene–gene Gaussian graphical models | `04_gene_networks.ipynb` |
| 5 | Multi-layer cell + gene graph | `05_multilayer.ipynb` |

## Key methodological notes

- **Patient-level CV** (40 train / 8 val / 8 test), stratified by cell-type composition (seed 42). The only scientifically valid split for this cohort.
- **Training mask** (`idx_to_take_into_training_based_on_grid`) is used **only** for GNN training (Part 3), where it spatially decorrelates dense same-type regions. All other parts use the full cell population.
- **Expression is pre-normalized** (log1p, capped at 5.0); used as-is, z-scored per gene only where a model requires it.
