from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.inference import run_inference, load_models, SAFE_THRESHOLD, _get_hubs, invalidate_hub_cache
from backend.services.explainer import explain_bank_risk
from backend.services.bailout_optimizer import calculate_optimal_bailout
from backend.utils.config import DATA_PATH
from backend.utils.bank_names import generate_bank_name

router = APIRouter()

# Path to PR curve image
_pwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PR_CURVE_PATH = os.path.join(_pwd, "results", "pr_curve.png")
AUTOML_RESULTS_PATH = os.path.join(_pwd, "results", "automl_results.json")
SYNTHETIC_DATA_PATH = os.path.join(_pwd, "data", "processed", "synthetic_scenario.pt")


def get_active_data():
    device = torch.device("cpu")
    if os.path.exists(SYNTHETIC_DATA_PATH):
        return torch.load(SYNTHETIC_DATA_PATH, map_location=device, weights_only=False)
    return torch.load(DATA_PATH, map_location=device, weights_only=False)


def _shock_diffusion(
    edge_index: torch.Tensor,
    edge_attr:  Optional[torch.Tensor],
    num_nodes:  int,
    shocked_idx: int,
    steps: int = 3,
    decay: float = 0.55,
) -> np.ndarray:
    """
    Multi-step weighted shock propagation — pure PyTorch tensor math.

    At each step every node passes `decay` fraction of its current shock
    intensity to its out-neighbours, normalised by the receiver's total
    in-edge weight (so a bank with many creditors absorbs each hit proportionally).

    Returns a [num_nodes] float32 array normalised to [0, 1].
    """
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    weights = (
        edge_attr.cpu().float().squeeze() if edge_attr is not None
        else torch.ones(src.size(0))
    )

    # Total in-weight per receiver (used for normalisation)
    in_weight_total = torch.zeros(num_nodes)
    in_weight_total.scatter_add_(0, dst, weights)
    in_weight_total = in_weight_total.clamp(min=1e-6)

    shock = torch.zeros(num_nodes)
    shock[shocked_idx] = 1.0

    for _ in range(steps):
        # Contribution of each edge: sender_shock × edge_weight / receiver_in_weight × decay
        contrib = shock[src] * weights / in_weight_total[dst] * decay
        new_shock = torch.zeros(num_nodes)
        new_shock.scatter_add_(0, dst, contrib)
        # Add residual: node keeps a fraction of its own shock
        shock = new_shock + shock * (1.0 - decay)

    arr = shock.numpy()
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _pagerank_power(
    edge_index: torch.Tensor,
    num_nodes:  int,
    damping: float = 0.85,
    iters:   int   = 40,
) -> np.ndarray:
    """
    PageRank via power iteration — pure PyTorch, no NetworkX needed.

    Returns a [num_nodes] float32 array normalised to [0, 1]  so it can
    be used directly as a centrality multiplier.
    """
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()

    # Out-degree for normalisation
    out_deg = torch.zeros(num_nodes)
    out_deg.scatter_add_(0, src, torch.ones(src.size(0)))
    out_deg = out_deg.clamp(min=1e-6)

    pr = torch.full((num_nodes,), 1.0 / num_nodes)

    for _ in range(iters):
        contrib = pr[src] / out_deg[src]
        new_pr  = torch.zeros(num_nodes)
        new_pr.scatter_add_(0, dst, contrib)
        pr = (1.0 - damping) / num_nodes + damping * new_pr

    arr = pr.numpy()
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# ---- Health ----

@router.get("/health")
async def health():
    return {"status": "ok"}

# ---- Bank Listing ----

@router.get("/banks")
async def get_banks():
    data = get_active_data()
    hub_set = _get_hubs(data.edge_index, data.num_nodes, top_k=min(20, data.num_nodes // 4))
    bank_list = [
        f"Bank_{i:04d} // {generate_bank_name(i, data.num_nodes, is_hub=(i in hub_set))}"
        for i in range(data.num_nodes)
    ]
    return {"banks": bank_list, "count": data.num_nodes}

# ---- Prediction (Full Network) ----

@router.post("/predict")
async def predict():
    try:
        result = run_inference()
        return {"risk_scores": result["nodes"], **{k: v for k, v in result.items() if k != "nodes"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Explainability (XAI) ----

class ExplainRequest(BaseModel):
    bank_id: str

@router.post("/explain_risk")
async def explain_risk(req: ExplainRequest):
    try:
        # Strip explicit real name alias if attached so index parser logic can grab "Bank_XXXX"
        raw_id = req.bank_id.split(" //")[0]
        result = explain_bank_risk(raw_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate_network")
async def generate_network():
    """
    Generates a pure 75-node Barabasi-Albert synthetic scale-free graph.
    
    This is the ONLY code path that creates the synthetic universe.
    Once saved, ALL downstream modules (shock, bailout, metrics, predict)
    will operate exclusively on this 75-node graph via get_active_data().
    """
    try:
        from torch_geometric.utils import barabasi_albert_graph
        from torch_geometric.data import Data

        VISUAL_CAP = 75

        # ── 1. Generate Pure Scale-Free Synthetic Topology (Clean & Strict 75 nodes) ──
        edge_index = barabasi_albert_graph(num_nodes=VISUAL_CAP, num_edges=2)

        # ── 2. Attach simulated internal features (10 dimensions) ──
        # Ensures proper mathematical distributions match expected normalization layer.
        x = torch.abs(torch.randn(VISUAL_CAP, 10)) * 2.0

        # ── 3. Apply uniform/random edge weights as 1D tensor ──
        # CRITICAL: GCNConv expects edge_attr as 1D [num_edges], NOT 2D [num_edges, 1]
        edge_attr = (torch.ones(edge_index.size(1)) * 0.75) + (torch.rand(edge_index.size(1)) * 0.25)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = VISUAL_CAP

        # ── 4. Cache state to ensure subsequent predictions run strictly on this universe ──
        os.makedirs(os.path.dirname(SYNTHETIC_DATA_PATH), exist_ok=True)
        torch.save(data, SYNTHETIC_DATA_PATH)

        # ── 5. Invalidate stale hub cache from any previous graph ──
        invalidate_hub_cache()

        total_nodes = data.num_nodes
        hub_set = _get_hubs(data.edge_index, total_nodes, top_k=min(5, total_nodes // 5))

        # ── 6. Build D3 Nodes JSON list exclusively tracking synthetic structure ──
        kept_nodes = set(range(VISUAL_CAP))  # All 75 nodes guaranteed present
        nodes_payload = []
        for v in sorted(kept_nodes):
            is_hub = (v in hub_set)
            alias = generate_bank_name(v, VISUAL_CAP, is_hub=is_hub)
            nodes_payload.append({
                "id": f"Bank_{v:04d} // {alias}",
                "label": alias
            })

        edges_payload = []
        for idx in range(edge_index.size(1)):
            u = edge_index[0, idx].item()
            v = edge_index[1, idx].item()
            weight = float(edge_attr[idx].item())

            alias_u = generate_bank_name(u, VISUAL_CAP, is_hub=(u in hub_set))
            alias_v = generate_bank_name(v, VISUAL_CAP, is_hub=(v in hub_set))

            edges_payload.append({
                "from": f"Bank_{u:04d} // {alias_u}",
                "to": f"Bank_{v:04d} // {alias_v}",
                "weight": round(weight, 4)
            })

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "num_nodes": VISUAL_CAP,
            "num_edges": edge_index.size(1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Shock Simulation (Organic GNN Prediction — No Manual Overrides) ----

class SimulateRequest(BaseModel):
    bank_id: str
    shock_magnitude: float = 1.0

@router.post("/simulate_shock")
async def simulate_shock(req: SimulateRequest):
    """
    Composite Score + Dynamic Threshold shock simulation.

    Pipeline:
      1. Run GNN on baseline and shocked graph to get per-node risk scores.
      2. Compute multi-step weighted shock diffusion from the bankrupted node
         (pure tensor math — no retraining required).
      3. Compute PageRank via power iteration to capture systemic importance.
      4. Composite score = blend of GNN score + diffusion score, amplified by
         a centrality multiplier (PageRank + degree). Hubs get structurally
         boosted; leaves are naturally dampened.
      5. Dynamic criticality threshold: take top-N% by composite score, where
         N scales linearly with the shocked node's PageRank. Hub bankruptcy
         exposes more of the network; leaf bankruptcy exposes very little.
         A hard floor filters out nodes with negligible composite scores.
    """
    try:
        raw_id = req.bank_id.split(" //")[0]
        target_idx = int(raw_id.split("_")[1])
    except (IndexError, ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid parsing format for bank_id: {req.bank_id}")

    data = get_active_data()

    if target_idx >= data.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Bank index {target_idx} exceeds graph size ({data.num_nodes})."
        )

    gnn, encoder = load_models()
    num_nodes = data.num_nodes

    def _run_gnn(graph_data):
        with torch.no_grad():
            enc = encoder.encode(graph_data.x)
            g   = graph_data.clone()
            g.x = enc
            out = gnn(g)
            preds = out[0] if isinstance(out, tuple) else out
        raw = preds.detach().cpu().numpy().flatten()
        mn, mx = raw.min(), raw.max()
        return (raw - mn) / (mx - mn) if mx > mn else np.zeros_like(raw)

    # ── 1. Baseline + shocked GNN scores ──
    baseline_scores = _run_gnn(data)

    shocked = data.clone()
    shocked.x[target_idx] = shocked.x[target_idx] * (1.0 - req.shock_magnitude)

    contagion_rate = 0.30 * req.shock_magnitude
    nb_mask = (data.edge_index[0] == target_idx)
    first_deg_nb = data.edge_index[1][nb_mask].unique()
    for nb in first_deg_nb.tolist():
        if nb != target_idx:
            shocked.x[nb] = shocked.x[nb] * (1.0 - contagion_rate)

    shocked_scores = _run_gnn(shocked)
    deltas = shocked_scores - baseline_scores

    # ── 2. Shock diffusion (Layer 2a) ──
    # Multi-step weighted propagation originating from the shocked node.
    # Gives each node a "received shock intensity" that respects edge weights
    # and network depth — independent of what the GNN outputs.
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
    diffusion = _shock_diffusion(
        data.edge_index, edge_attr, num_nodes, target_idx,
        steps=3, decay=0.55,
    )

    # ── 3. PageRank (Layer 2b) ──
    # Captures systemic importance — how much the rest of the network
    # depends on this node through chains of weighted connections.
    pagerank_norm = _pagerank_power(data.edge_index, num_nodes)

    # ── 4. Degree normalisation ──
    src_idx = data.edge_index[0].cpu()
    dst_idx = data.edge_index[1].cpu()
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, src_idx, torch.ones(src_idx.size(0)))
    deg.scatter_add_(0, dst_idx, torch.ones(dst_idx.size(0)))
    deg_norm = deg.numpy()
    deg_hi = deg_norm.max()
    deg_norm = deg_norm / deg_hi if deg_hi > 0 else deg_norm

    # ── 5. Composite score ──
    # Base: equal blend of GNN signal and diffusion-based propagation.
    # GNN captures learned vulnerability; diffusion captures flow-based exposure.
    base = 0.45 * shocked_scores + 0.55 * diffusion

    # Centrality amplifier: topology boosts nodes that matter more structurally.
    # PageRank = global systemic importance; degree = local connectivity.
    # A node with zero centrality keeps its base score unchanged (multiplier = 1).
    centrality = 0.5 * pagerank_norm + 0.5 * deg_norm
    composite  = base * (1.0 + centrality)

    # Re-normalise composite to [0, 1] for consistent display
    c_lo, c_hi = composite.min(), composite.max()
    composite_norm = (composite - c_lo) / (c_hi - c_lo) if c_hi > c_lo else np.zeros_like(composite)

    # ── 6. Dynamic critical threshold (Layer 3) ──
    # The shocked node's PageRank determines how much of the network is exposed.
    #   Hub  (PR ≈ 1.0 normalised) → up to 20 % of nodes eligible  (~15 / 75)
    #   Leaf (PR ≈ 0.0 normalised) → only  4 % of nodes eligible   (~ 3 / 75)
    # A composite-score floor of 0.25 then filters low-impact nodes out.
    BASE_PCT          = 0.04
    MAX_PCT           = 0.20
    COMPOSITE_FLOOR   = 0.25

    hub_weight        = float(pagerank_norm[target_idx])
    expected_pct      = BASE_PCT + (MAX_PCT - BASE_PCT) * hub_weight
    n_eligible        = max(1, int(num_nodes * expected_pct))

    # Sort all nodes by composite score descending; take top-N with floor filter
    ranked = sorted(range(num_nodes), key=lambda i: composite_norm[i], reverse=True)
    critical_set = set()
    critical_set.add(target_idx)   # always include shocked node (Tier 1)
    for idx in ranked:
        if idx == target_idx:
            continue
        if len(critical_set) >= n_eligible + 1:  # +1 for the shocked node
            break
        if composite_norm[idx] >= COMPOSITE_FLOOR:
            critical_set.add(idx)

    # Invalidate hub cache so bank labels stay correct for this graph
    invalidate_hub_cache()
    hub_set = _get_hubs(data.edge_index, num_nodes, top_k=min(20, num_nodes // 4))

    nodes = []
    for i in range(num_nodes):
        alias = generate_bank_name(i, num_nodes, is_hub=(i in hub_set))
        nodes.append({
            "bank_id":      f"Bank_{i:04d} // {alias}",
            "score":        round(float(composite_norm[i]), 6),   # composite = primary display score
            "gnn_score":    round(float(shocked_scores[i]), 6),   # raw GNN output for reference
            "diffusion":    round(float(diffusion[i]), 6),        # shock propagation intensity
            "delta":        round(float(deltas[i]), 6),           # GNN score change vs baseline
            "is_critical":  i in critical_set,
            "raw_idx":      i,
        })

    # Sort by composite score descending — most at-risk banks first
    nodes.sort(key=lambda x: x["score"], reverse=True)

    total_banks    = len(nodes)
    critical_count = sum(1 for n in nodes if n["is_critical"])
    safe_count     = sum(1 for n in nodes if not n["is_critical"] and n["delta"] <= 0.0)

    # Composite score of the lowest-ranked critical node (the effective cutoff)
    critical_scores = [composite_norm[i] for i in critical_set]
    cutoff_score    = round(float(min(critical_scores)), 6) if critical_scores else 0.0

    return {
        "shocked_bank":         req.bank_id,
        "shock_magnitude":      req.shock_magnitude,
        "total_banks":          total_banks,
        "critical_node_count":  critical_count,
        "safe_node_count":      safe_count,
        "critical_threshold":   cutoff_score,
        "hub_weight":           round(hub_weight, 4),
        "output_type":          "composite_score_dynamic_threshold",
        "risk_scores":          nodes,
    }

# ---- Bailout Optimizer ----

class BailoutRequest(BaseModel):
    shocked_bank_id: str
    budget_millions: float = 500.0
    shock_magnitude: float = 1.0   # must match the magnitude used in simulate_shock

@router.post("/optimize_bailout")
async def optimize_bailout(req: BailoutRequest):
    try:
        raw_id = req.shocked_bank_id.split(" //")[0]
        target_idx = int(raw_id.split("_")[1])
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid mapping for bank_id: {req.shocked_bank_id}")

    data = get_active_data()

    if target_idx >= data.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Bank index {target_idx} exceeds graph size ({data.num_nodes})."
        )

    gnn, encoder = load_models()

    result = calculate_optimal_bailout(
        budget_millions=req.budget_millions,
        data=data,
        shocked_node_idx=target_idx,
        gnn=gnn,
        encoder=encoder,
        shock_magnitude=req.shock_magnitude,
    )
    return result

# ---- Metrics (STRICT TEST-SET ONLY — No Evaluation Leakage) ----

@router.get("/metrics")
async def get_metrics():
    """
    Computes and returns evaluation metrics using ONLY the test split.
    Never evaluates on train_mask or val_mask nodes to prevent data leakage.
    """
    import json
    metrics_data = {
        "optimization_target": "PR-AUC (maximize)",
        "evaluation_split": "test_set_only",
        "pr_curve_available": os.path.exists(PR_CURVE_PATH),
    }

    if os.path.exists(AUTOML_RESULTS_PATH):
        with open(AUTOML_RESULTS_PATH, "r") as f:
            automl = json.load(f)
            metrics_data["best_score"] = automl.get("best_score")
            metrics_data["best_params"] = automl.get("best_params")
            metrics_data["optimization_target"] = automl.get("optimization_target", "PR-AUC (maximize)")

    try:
        # ── Always use historical data for metrics (it has labels + test_mask) ──
        hist_data = torch.load(DATA_PATH, map_location=torch.device("cpu"), weights_only=False)
        metrics_data["active_graph_nodes"] = hist_data.num_nodes
        metrics_data["active_graph_edges"] = hist_data.edge_index.size(1)

        has_test_mask = hasattr(hist_data, "test_mask") and hist_data.test_mask is not None
        has_labels    = hasattr(hist_data, "y") and hist_data.y is not None

        if has_labels and has_test_mask:
            test_mask      = hist_data.test_mask
            test_node_count = int(test_mask.sum().item())
            metrics_data["test_node_count"] = test_node_count

            if test_node_count > 0:
                gnn, encoder = load_models()
                with torch.no_grad():
                    encoded_x       = encoder.encode(hist_data.x)
                    hist_data.x     = encoded_x
                    out             = gnn(hist_data)
                    preds           = out[0] if isinstance(out, tuple) else out

                from training.evaluation import calculate_pr_metrics
                pr_result = calculate_pr_metrics(hist_data.y[test_mask], preds[test_mask])
                metrics_data["pr_auc"]            = pr_result["pr_auc"]
                metrics_data["dynamic_threshold"] = pr_result["threshold_used"]
                metrics_data["positive_ratio"]    = pr_result["positive_ratio"]
                metrics_data["n_critical"]         = pr_result["n_positive"]
                metrics_data["n_total"]            = pr_result["n_total"]
                metrics_data["precision_coords"]   = pr_result["precision_coords"]
                metrics_data["recall_coords"]      = pr_result["recall_coords"]
            else:
                metrics_data["metrics_error"] = "test_mask contains 0 nodes — cannot compute metrics"
        else:
            metrics_data["pr_auc"]          = None
            metrics_data["metrics_warning"] = (
                "Historical dataset has no labels or test_mask. "
                "Run AutoML training first to generate evaluation splits."
            )

    except Exception as e:
        metrics_data["metrics_error"] = str(e)

    return metrics_data

@router.get("/metrics/pr_curve")
async def get_pr_curve():
    if not os.path.exists(PR_CURVE_PATH):
        raise HTTPException(status_code=404, detail="PR curve not yet generated. Run training first.")
    return FileResponse(PR_CURVE_PATH, media_type="image/png")
