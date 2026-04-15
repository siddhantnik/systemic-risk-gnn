"""
gnn_model.py — Graph Attention Network v2 for Systemic Risk

CHANGES:
  OLD: GCNConv — treats all neighbors equally, doesn't handle directed/weighted graphs
  NEW: GATv2Conv — dynamic attention scores per-edge, handles asymmetric interbank flows
  OLD: Single regression head → MSE loss → poor PR-AUC
  NEW: Dual output heads: regression (SRISK score) + classification (critical/not).
       This resolves the fundamental MSE-vs-PR-AUC goal conflict.
  NEW: Jumping Knowledge (JK) aggregation — concatenates all layer outputs
       so shallow and deep neighborhood info both survive. Critical for detecting
       contagion that propagates across 2-3 hops.
  NEW: Edge dropout during training — prevents the GNN from memorizing specific
       interbank links that may not exist next quarter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, JumpingKnowledge
from torch_geometric.data import Data


class SystemicRiskGNN(nn.Module):
    """
    GATv2-based systemic risk predictor with:
      - Multi-head attention per layer (captures diverse contagion pathways)
      - Jumping Knowledge network (preserves both local and global neighborhood info)
      - Dual output: continuous risk score + binary critical probability
      - Edge dropout for training regularization

    Args:
        input_dim:   Latent dimension from autoencoder (e.g. 16)
        hidden_dim:  Per-layer hidden size (e.g. 64)
        num_layers:  Number of GATv2 layers (2–4 recommended)
        heads:       Attention heads per layer (4 recommended)
        dropout:     Feature + attention dropout probability
        edge_dropout: Probability of masking a random edge during training
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.edge_dropout = edge_dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 0: input_dim → hidden_dim * heads
        self.convs.append(GATv2Conv(
            input_dim, hidden_dim, heads=heads, dropout=dropout,
            add_self_loops=True, share_weights=False, edge_dim=1
        ))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Layers 1..N-1: hidden_dim*heads → hidden_dim*heads
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(
                hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout,
                add_self_loops=True, share_weights=False, edge_dim=1
            ))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.jk = JumpingKnowledge(mode="max")

        jk_out_dim = hidden_dim * heads  # max JK output dim

        # ── Regression head: continuous SRISK score ──────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(jk_out_dim, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)   # raw logit — no sigmoid, loss handles it
        )

        # ── Classification head: P(node is systemic/critical) ────────────
        self.cls_head = nn.Sequential(
            nn.Linear(jk_out_dim, 32),
            nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1)   # raw logit for BCEWithLogitsLoss
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_edge_dropout(self, edge_index, edge_weight):
        """Randomly drop edges during training to prevent memorization."""
        if not self.training or self.edge_dropout == 0.0:
            return edge_index, edge_weight
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
        return edge_index[:, mask], edge_weight[mask] if edge_weight is not None else None

    def forward(self, data: Data):
        x          = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)

        edge_index, edge_weight = self._apply_edge_dropout(edge_index, edge_weight)

        # GATv2Conv expects edge_attr as [num_edges, 1] when edge_dim=1
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)

        layer_outs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_weight)
            x = self.norms[i](x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)

        reg_out = self.reg_head(x).squeeze(-1)
        cls_out = self.cls_head(x).squeeze(-1)

        return reg_out, cls_out


# Keep a legacy alias so routes/inference that call gnn(data) and expect one
# output can be migrated gracefully (returns regression score only).
class SystemicRiskGCN(SystemicRiskGNN):
    """Legacy alias — wraps dual-head GNN, returns only regression score."""
    def forward(self, data):
        reg, _ = super().forward(data)
        return reg