"""
inference.py — Inference pipeline fixes

CHANGES:
  OLD: _hub_indices_cache global never resets between synthetic/historical graphs
       → stale hub labels after generate_network() is called
  NEW: Cache is keyed by (num_nodes, edge_count) so it auto-invalidates on graph change
  OLD: encoder.encoder(x) — wrong attribute name for new VAE (.encode() method)
  NEW: encoder.encode(x) — deterministic mu, no sampling noise
  OLD: run_inference() loads DATA_PATH always, ignores active synthetic graph
  NEW: run_inference() respects get_active_data() like all other routes do
  OLD: Predictions sorted but no normalization — raw logits shown as "scores"
  NEW: Scores normalized to [0,1] range for display consistency
"""

import torch, json, os, sys
import numpy as np
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.config import DATA_PATH, PARAM_JSON_PATH, GNN_MODEL_PATH, AUTOENCODER_MODEL_PATH
from backend.models.autoencoder import FinancialAutoencoder
from backend.models.gnn_model import SystemicRiskGNN
from backend.utils.bank_names import generate_bank_name

SAFE_THRESHOLD = 0.30

_pwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_DATA_PATH = os.path.join(_pwd, "data", "processed", "synthetic_scenario.pt")

# Cache keyed by graph signature to auto-invalidate
_hub_cache: dict = {}


def _get_hubs(edge_index, num_nodes, top_k=20):
    key = (num_nodes, int(edge_index.size(1)))
    if key not in _hub_cache:
        degrees = torch.bincount(edge_index[1], minlength=num_nodes)
        _, top_idx = torch.topk(degrees, min(top_k, num_nodes))
        _hub_cache[key] = set(top_idx.tolist())
    return _hub_cache[key]


def invalidate_hub_cache():
    _hub_cache.clear()


def get_active_data():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if os.path.exists(SYNTHETIC_DATA_PATH):
        return torch.load(SYNTHETIC_DATA_PATH, map_location=device, weights_only=False)
    return torch.load(DATA_PATH, map_location=device, weights_only=False)


@lru_cache(maxsize=1)
def load_models():
    with open(PARAM_JSON_PATH, "r") as f:
        saved = json.load(f)

    best_params = saved["best_params"]
    latent_dim  = saved.get("latent_dim", best_params.get("latent_dim", 16))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data        = torch.load(DATA_PATH, map_location=device, weights_only=False)
    features_in = data.x.size(1)

    encoder = FinancialAutoencoder(input_dim=features_in, latent_dim=latent_dim)
    encoder.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()

    
    from backend.models.gnn_model import SystemicRiskGNN
    gnn = SystemicRiskGNN(
        input_dim    = latent_dim,
        hidden_dim   = best_params["hidden_dim"],
        num_layers   = min(best_params["num_layers"], 2),
        heads        = best_params.get("heads", 4),
        dropout      = best_params["dropout"],
        edge_dropout = 0.0,
    )
    gnn.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device, weights_only=True))
    gnn.eval()
    return gnn, encoder


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return np.zeros_like(scores)


def run_inference():
    data = get_active_data()   # BUG FIX: was always loading DATA_PATH
    gnn, encoder = load_models()

    with torch.no_grad():
        x_enc        = encoder.encode(data.x)
        data_enc     = data.clone()
        data_enc.x   = x_enc
        gnn_out      = gnn(data_enc)
        # Handle both old single-output and new dual-head models
        reg_scores   = gnn_out[0] if isinstance(gnn_out, tuple) else gnn_out

    raw = reg_scores.cpu().numpy().flatten()
    scores_norm = _normalize_scores(raw)

    critical_threshold = float(np.percentile(scores_norm, 95))
    hub_set = _get_hubs(data.edge_index, data.num_nodes, top_k=20)

    nodes = []
    for i, (s_raw, s_norm) in enumerate(zip(raw, scores_norm)):
        is_hub   = i in hub_set
        bank_name = generate_bank_name(i, data.num_nodes, is_hub=is_hub)
        nodes.append({
            "bank_id":    f"Bank_{i:04d} // {bank_name}",
            "score":      round(float(s_norm), 6),
            "score_raw":  round(float(s_raw), 6),
            "raw_idx":    i,
        })

    nodes.sort(key=lambda d: d["score"], reverse=True)
    critical_count = sum(1 for n in nodes if n["score"] >= critical_threshold)
    safe_count     = sum(1 for n in nodes if n["score"] < SAFE_THRESHOLD)

    return {
        "total_banks":        len(nodes),
        "critical_node_count": critical_count,
        "safe_node_count":    safe_count,
        "critical_threshold": round(critical_threshold, 6),
        "output_type":        "normalized_gnn_regression",
        "nodes":              nodes,
    }