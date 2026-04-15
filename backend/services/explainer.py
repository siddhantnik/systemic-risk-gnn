import torch
import os
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from backend.services.inference import load_models, get_active_data


class _ExplainerProxy(torch.nn.Module):
    """
    Wraps SystemicRiskGCN so it accepts the (x, edge_index, **kwargs) signature
    that PyG's Explainer framework requires, while preserving edge_attr.
    """
    def __init__(self, gnn, edge_attr=None):
        super().__init__()
        self.model = gnn
        self._edge_attr = edge_attr

    def forward(self, x, edge_index):
        d = Data(x=x, edge_index=edge_index, edge_attr=self._edge_attr)
        out = self.model(d)
        # GNNExplainer requires a plain tensor — return only regression head
        return out[0] if isinstance(out, tuple) else out


def explain_bank_risk(bank_id_str: str):
    """
    Uses GNNExplainer to identify which edges and node features
    contribute most to a bank's predicted systemic risk score.

    Now uses get_active_data() so it operates on the synthetic 75-node
    graph when one has been generated, instead of always falling back
    to the massive historical dataset.

    Returns dict with 'target', 'toxic_edges', and 'weak_features'.
    """
    # GNNExplainer runs gradient optimisation — it does NOT support MPS.
    # Force everything to CPU for this call only.
    cpu = torch.device("cpu")

    # 1. Load the ACTIVE graph then move to CPU
    data = get_active_data()
    data = data.clone()
    data.x = data.x.to(cpu)
    data.edge_index = data.edge_index.to(cpu)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(cpu)

    try:
        target_idx = int(bank_id_str.split("_")[1])
    except (IndexError, ValueError):
        target_idx = 0

    if target_idx >= data.num_nodes:
        return {"error": f"Bank index {target_idx} exceeds graph size ({data.num_nodes})."}

    gnn, encoder = load_models()

    # Move model copies to CPU (lru_cache keeps originals intact on MPS)
    import copy
    gnn_cpu     = copy.deepcopy(gnn).to(cpu)
    encoder_cpu = copy.deepcopy(encoder).to(cpu)
    gnn_cpu.eval()
    encoder_cpu.eval()

    # 2. Encode features on CPU
    with torch.no_grad():
        x_encoded = encoder_cpu.encode(data.x)

    # 3. Build proxy that passes edge_attr to the GNN (all on CPU)
    ea = data.edge_attr
    if ea is not None and ea.dim() > 1:
        ea = ea.squeeze(-1)

    proxy = _ExplainerProxy(gnn_cpu, edge_attr=ea)
    proxy.eval()

    explainer = Explainer(
        model=proxy,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )

    # 4. Generate explanation
    explanation = explainer(x_encoded, data.edge_index, index=target_idx)

    # 5. Extract top toxic edges
    toxic_edges = []
    edge_mask = explanation.edge_mask
    if edge_mask is not None and edge_mask.numel() > 0:
        k = min(5, edge_mask.size(0))
        top_indices = torch.topk(edge_mask, k=k).indices
        for e_idx in top_indices:
            src = data.edge_index[0, e_idx].item()
            dst = data.edge_index[1, e_idx].item()
            toxic_edges.append({
                "source": f"Bank_{src:04d}",
                "target": f"Bank_{dst:04d}",
                "toxicity_weight": round(edge_mask[e_idx].item(), 4)
            })

    # 6. Extract top weak features at the target node
    weak_features = []
    node_mask = explanation.node_mask
    if node_mask is not None and node_mask.numel() > 0:
        features_at_target = node_mask[target_idx]
        k = min(3, features_at_target.size(0))
        top_feat_indices = torch.topk(features_at_target, k=k).indices
        for f_idx in top_feat_indices:
            weak_features.append({
                "encoded_feature_index": f_idx.item(),
                "importance_weight": round(features_at_target[f_idx].item(), 4)
            })

    return {
        "target": bank_id_str,
        "toxic_edges": toxic_edges,
        "weak_features": weak_features
    }
