"""
bailout_optimizer.py — Fixed batched GNN bailout allocation

CRITICAL BUG FIX:
  OLD: preds_reshaped = batched_preds.view(len(neighbor_set), num_nodes_per_graph)
       This assumes PyG Batch produces [num_graphs × num_nodes] contiguous output.
       It DOES NOT. PyG Batch offsets node indices; output is [total_nodes_across_batch].
       Result: completely wrong risk sums, allocations are random noise.
  NEW: Use batch.batch tensor to properly segment per-graph predictions via scatter.

OTHER FIXES:
  OLD: encoder.encoder(x) wrong attribute
  NEW: encoder.encode(x)
  OLD: Greedy allocation uses fixed 20 chunks regardless of budget
  NEW: Chunk count scales with number of neighbors (min 5, max 30)
  OLD: Only boosts feature index 0 (Total_assets) — ignores capital/liquidity
  NEW: Boosts features 0 (assets), 1 (liquid), 2 (tier1) proportionally
"""

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from typing import Dict
from backend.utils.bank_names import generate_bank_name


def calculate_optimal_bailout(
    budget_millions: float,
    data: Data,
    shocked_node_idx: int,
    gnn: torch.nn.Module,
    encoder: torch.nn.Module,
    shock_magnitude: float = 1.0,
) -> Dict:
    work_data    = data.clone()
    raw_features = work_data.x.clone()
    edge_index   = data.edge_index

    from backend.services.inference import _get_hubs
    hub_set = _get_hubs(edge_index, work_data.num_nodes, top_k=20)

    # ── Apply the shock to the starting feature state ───────────────────────
    # Mirror the exact same shock that simulate_shock applies so that
    # original_risk reflects the post-bankruptcy damage state, not the
    # clean baseline. The optimiser then tries to UNDO this damage.
    raw_features[shocked_node_idx] = raw_features[shocked_node_idx] * (1.0 - shock_magnitude)

    contagion_rate = 0.30 * shock_magnitude
    out_nb_shock = edge_index[1, edge_index[0] == shocked_node_idx].tolist()
    for nb in out_nb_shock:
        if nb != shocked_node_idx:
            raw_features[nb] = raw_features[nb] * (1.0 - contagion_rate)

    # Risk load = sum of POSITIVE logits only (clamp negatives to 0).
    # Always >= 0. Hub shocks → more nodes with elevated positive logits → higher sum.
    # Leaf shocks → fewer elevated nodes → lower sum. Direction is always intuitive:
    # lower = safer network, so bailout should always reduce this value.
    with torch.no_grad():
        enc_x = encoder.encode(raw_features.clone())
        tmp   = data.clone(); tmp.x = enc_x
        base_reg, _ = gnn(tmp)
        original_risk_sum = float(base_reg.clamp(min=0).sum().item())

    # ── Candidate pool: 1st-degree + 2nd-degree neighbours ──────────────────
    # For a leaf (degree 1–2) injecting capital into only 1 bank gives the
    # optimizer almost no choices. Expanding to 2nd-degree neighbours gives
    # meaningful targets even for leaf bankruptcies.
    # For a hub (degree 20+) the 1st-degree pool is already large; the cap
    # prevents unnecessary computation.
    MAX_POOL = 25

    out_nb_1 = edge_index[1, edge_index[0] == shocked_node_idx].tolist()
    in_nb_1  = edge_index[0, edge_index[1] == shocked_node_idx].tolist()
    first_deg = set(out_nb_1 + in_nb_1) - {shocked_node_idx}

    # Walk one more hop to collect 2nd-degree neighbours
    second_deg: set = set()
    for nb in first_deg:
        out2 = edge_index[1, edge_index[0] == nb].tolist()
        in2  = edge_index[0, edge_index[1] == nb].tolist()
        second_deg |= set(out2 + in2)
    second_deg -= first_deg | {shocked_node_idx}   # exclude already covered nodes

    # Always include all 1st-degree; fill remaining slots with 2nd-degree
    neighbor_set = list(first_deg)
    if len(neighbor_set) < MAX_POOL:
        # Sort for deterministic ordering across runs
        second_deg_sorted = sorted(second_deg)
        neighbor_set += second_deg_sorted[: MAX_POOL - len(neighbor_set)]

    n_first_deg  = len(first_deg)
    n_second_deg = len(neighbor_set) - n_first_deg

    if not neighbor_set:
        return {
            "recommended_allocations": [],
            "original_network_risk_sum":  round(original_risk_sum, 4),
            "optimized_network_risk_sum": round(original_risk_sum, 4),
            "risk_reduction_pct": 0.0,
            "message": "Shocked node has no network connections.",
        }

    n_chunks   = min(30, max(5, len(neighbor_set)))
    chunk_size = budget_millions / n_chunks
    allocations   = {n: 0.0 for n in neighbor_set}
    current_feats = raw_features.clone()

    for _ in range(n_chunks):
        injection = (chunk_size / budget_millions) * 5.0
        scenario_list = []
        for nb_idx in neighbor_set:
            test_feats = current_feats.clone()
            # Boost assets (0), liquidity (1), tier1 capital (2)
            for feat_col in [0, 1, 2]:
                if feat_col < test_feats.size(1):
                    test_feats[nb_idx, feat_col] += injection

            with torch.no_grad():
                enc = encoder.encode(test_feats)

            sd = Data(x=enc, edge_index=edge_index)
            if hasattr(work_data, "edge_attr") and work_data.edge_attr is not None:
                sd.edge_attr = work_data.edge_attr
            scenario_list.append(sd)

        batched = Batch.from_data_list(scenario_list)

        with torch.no_grad():
            b_reg, _ = gnn(batched)

        # FIXED: use batch.batch to segment predictions per graph
        risk_sums = scatter(b_reg, batched.batch, dim=0, reduce='sum')   # [num_scenarios]

        best_pos    = int(torch.argmin(risk_sums).item())
        best_nb     = neighbor_set[best_pos]
        for feat_col in [0, 1, 2]:
            if feat_col < current_feats.size(1):
                current_feats[best_nb, feat_col] += injection
        allocations[best_nb] += chunk_size

    # Final positive risk load after capital injections
    with torch.no_grad():
        opt_enc = encoder.encode(current_feats)
        tmp2    = data.clone(); tmp2.x = opt_enc
        opt_reg, _ = gnn(tmp2)
        optimized_risk_sum = float(opt_reg.clamp(min=0).sum().item())

    recommended = []
    for idx, alloc in allocations.items():
        if alloc > 0:
            is_hub = idx in hub_set
            alias  = generate_bank_name(idx, data.num_nodes, is_hub=is_hub)
            recommended.append({
                "bank_id":             f"Bank_{idx:04d} // {alias}",
                "allocation_millions": round(alloc, 2),
                "raw_idx":             idx,
            })
    recommended.sort(key=lambda x: x["allocation_millions"], reverse=True)

    risk_reduction     = original_risk_sum - optimized_risk_sum
    risk_reduction_pct = (risk_reduction / abs(original_risk_sum) * 100
                         ) if original_risk_sum != 0 else 0.0

    shocked_alias = generate_bank_name(shocked_node_idx, data.num_nodes,
                                       is_hub=(shocked_node_idx in hub_set))
    return {
        "recommended_allocations":    recommended,
        "original_network_risk_sum":  round(original_risk_sum, 4),
        "optimized_network_risk_sum": round(optimized_risk_sum, 4),
        "risk_reduction_pct":         round(risk_reduction_pct, 2),
        "budget_millions":            budget_millions,
        "shock_magnitude":            shock_magnitude,
        "shocked_node":               f"Bank_{shocked_node_idx:04d} // {shocked_alias}",
        "neighbors_evaluated":        len(neighbor_set),
        "first_deg_candidates":       n_first_deg,
        "second_deg_candidates":      n_second_deg,
    }