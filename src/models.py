"""
Models for Part 3 (cell-type classification).

All models share the signature:
    forward(x, edge_index) -> (n_nodes, n_classes) logits

So the training loop is generic and only the model class changes between rungs.

Architectures:
  - MLPCellTyper:      MLP on per-cell features (no graph). Rung 2.
  - GraphSAGECellTyper: GraphSAGE on features + graph. Rungs 3 (HVG only) and 4 (fused).
  - GATCellTyper:      GAT on features + graph (attention). Rung 5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GATConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# Shared modality encoder
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Per-modality projection: HVG / H&E / morph each get their own linear+ReLU.
    All produce per-modality embeddings that get concatenated.
    """
    def __init__(
        self,
        hvg_dim=1000,
        he_dim=0,                # 0 disables H&E
        morph_dim=0,             # 0 disables morph
        hvg_hidden=256,
        he_hidden=256,
        morph_hidden=16,
        dropout=0.2,
    ):
        super().__init__()
        self.hvg_dim = hvg_dim
        self.he_dim = he_dim
        self.morph_dim = morph_dim

        # Slices into the input feature vector
        self.hvg_slice = slice(0, hvg_dim)
        self.he_slice = slice(hvg_dim, hvg_dim + he_dim) if he_dim else None
        self.morph_slice = (
            slice(hvg_dim + he_dim, hvg_dim + he_dim + morph_dim) if morph_dim else None
        )

        self.hvg_enc = nn.Sequential(
            nn.Linear(hvg_dim, hvg_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        out_dim = hvg_hidden
        if he_dim:
            self.he_enc = nn.Sequential(
                nn.Linear(he_dim, he_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            out_dim += he_hidden
        if morph_dim:
            self.morph_enc = nn.Sequential(
                nn.Linear(morph_dim, morph_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            out_dim += morph_hidden
        self.out_dim = out_dim

    def forward(self, x):
        out = [self.hvg_enc(x[:, self.hvg_slice])]
        if self.he_slice is not None:
            out.append(self.he_enc(x[:, self.he_slice]))
        if self.morph_slice is not None:
            out.append(self.morph_enc(x[:, self.morph_slice]))
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# MLP (no graph)
# ---------------------------------------------------------------------------

class MLPCellTyper(nn.Module):
    """Rung 2 reference model. Doesn't use edge_index but accepts it for API compat."""
    def __init__(
        self,
        hvg_dim=1000, he_dim=0, morph_dim=0,
        head_hidden=128, n_classes=8, dropout=0.2,
    ):
        super().__init__()
        self.encoder = ModalityEncoder(
            hvg_dim=hvg_dim, he_dim=he_dim, morph_dim=morph_dim, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x, edge_index=None):
        h = self.encoder(x)
        return self.head(h)


# ---------------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------------

class GraphSAGECellTyper(nn.Module):
    """
    Rungs 3 (HVG-only) and 4 (fused). Two GraphSAGE layers on top of the
    modality encoder.

    Mean aggregator (default for SAGEConv) — robust to varying neighborhood
    sizes, including degree-0 nodes (which get only their own features).
    """
    def __init__(
        self,
        hvg_dim=1000, he_dim=0, morph_dim=0,
        sage_hidden=256, sage_layers=2,
        n_classes=8, dropout=0.2,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GraphSAGECellTyper")
        self.encoder = ModalityEncoder(
            hvg_dim=hvg_dim, he_dim=he_dim, morph_dim=morph_dim, dropout=dropout,
        )
        # GraphSAGE stack
        self.convs = nn.ModuleList()
        in_dim = self.encoder.out_dim
        for i in range(sage_layers):
            out_dim = sage_hidden
            self.convs.append(SAGEConv(in_dim, out_dim, aggr="mean"))
            in_dim = out_dim
        self.dropout = dropout
        self.head = nn.Linear(sage_hidden, n_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)


# ---------------------------------------------------------------------------
# GAT (Rung 5)
# ---------------------------------------------------------------------------

class GATCellTyper(nn.Module):
    """
    Rung 5. Same shape as GraphSAGECellTyper but with attention-weighted
    aggregation. Multi-head attention; concatenation across heads.
    """
    def __init__(
        self,
        hvg_dim=1000, he_dim=0, morph_dim=0,
        gat_hidden=64, gat_heads=4, gat_layers=2,
        n_classes=8, dropout=0.2,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GATCellTyper")
        self.encoder = ModalityEncoder(
            hvg_dim=hvg_dim, he_dim=he_dim, morph_dim=morph_dim, dropout=dropout,
        )
        self.convs = nn.ModuleList()
        in_dim = self.encoder.out_dim
        for i in range(gat_layers):
            self.convs.append(
                GATConv(in_dim, gat_hidden, heads=gat_heads,
                        concat=True, dropout=dropout)
            )
            in_dim = gat_hidden * gat_heads
        self.dropout = dropout
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)