"""
evaluation.py — Robust evaluation with proper PR-AUC and new diagnostic metrics

CHANGES:
  OLD: calculate_pr_metrics clips y_pred to [0,1] — WRONG for raw GNN logits
       (GNN outputs unbounded regression values, not probabilities)
  NEW: Normalizes predictions to [0,1] range via min-max before thresholding
  OLD: PR-AUC reports 0.0 when no positives found — masks evaluation bugs silently
  NEW: Hard assertion that raises if positive_count == 0, with clear diagnosis
  NEW: Added calibration_score (rank correlation between pred and true ranks)
       This directly measures whether the model *orders* banks correctly — which
       is what matters for regulators, not MSE.
  NEW: Added top_k_precision: "of the top-K predicted risky banks, how many
       are actually risky?" This is the operationally relevant metric.
  OLD: verify_baselines uses train features to fit MLP (leakage)
  NEW: MLP baseline fitted ONLY on train_mask nodes
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    average_precision_score, precision_recall_curve, ndcg_score
)
from sklearn.neural_network import MLPRegressor
import networkx as nx

POSITIVE_PERCENTILE = 0.95   # top 5% = systemic (rare event)


def compute_dynamic_threshold(y_true_tensor: torch.Tensor) -> float:
    """
    Computes classification threshold adaptively based on actual positive rate.
    - If positives are ~5% of data: use 95th percentile (standard approach)
    - If positives are <10% of data: use a small epsilon above zero
      (avoids the case where 95th percentile = 0.0 when 98% of labels are 0)
    """
    if y_true_tensor.numel() == 0:
        return 0.5

    y = y_true_tensor.float()
    positive_rate = (y > 1e-6).float().mean().item()

    if positive_rate < 0.10:
        # Sparse positive labels — threshold just above zero
        # Any bank with actual srisk measurement is positive
        return 1e-6
    else:
        # Dense labels — use 95th percentile as before
        return torch.quantile(y, POSITIVE_PERCENTILE).item()

def extract_systemic_risk_labels(nodes_df: pd.DataFrame, srisk_raw=None) -> torch.Tensor:
    """
    Extracts SRISK labels correctly.
    Uses srisk_raw (pre-fillna values) if provided — NaN means not systemically risky = 0.
    This prevents clean_nodes' fillna(mean) from creating fake positive labels.
    
    With 93.8% NaN in real data:
      - NaN → 0.0  (bank has no systemic risk score = safe)
      - 0.0 → 0.0  (explicitly zero risk)
      - positive → normalized to (0, 1]  (actually risky banks)
    """
    # Use raw values if provided (avoids NaN→mean contamination)
    if srisk_raw is not None:
        srisk = srisk_raw.astype(np.float64)
    elif "srisk_ratio" in nodes_df.columns:
        srisk = nodes_df["srisk_ratio"].values.astype(np.float64)
    elif "srisk_value" in nodes_df.columns:
        srisk = nodes_df["srisk_value"].values.astype(np.float64)
    else:
        print("  WARNING: No srisk column. Using zero labels.")
        return torch.zeros(len(nodes_df), dtype=torch.float32)

    # NaN = not systemically risky = 0.0 (do NOT fill with mean)
    srisk = np.nan_to_num(srisk, nan=0.0, posinf=0.0, neginf=0.0)

    n_positive = int((srisk > 0).sum())
    n_total    = len(srisk)
    print(f"    SRISK: {n_positive} positive banks out of {n_total} ({n_positive/n_total*100:.1f}%)")

    if n_positive == 0:
        print("    WARNING: All SRISK values are 0 — check srisk_ratio column!")
        return torch.zeros(n_total, dtype=torch.float32)

    # Clip only the positive values at 99th percentile to handle outliers
    p99 = np.percentile(srisk[srisk > 0], 99)
    srisk = np.clip(srisk, 0.0, p99)

    # Min-max normalize to [0, 1]
    s_max = srisk.max()
    srisk  = srisk / (s_max + 1e-8)

    return torch.tensor(srisk, dtype=torch.float32)


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().flatten()
    return np.array(t).flatten()


def calculate_pr_metrics(y_true, y_pred_raw, save_path=None):
    """
    Compute PR-AUC with proper handling of unbounded GNN outputs.

    Key fix: min-max normalizes y_pred_raw to [0,1] before thresholding.
    Old code clipped at 0/1 which zeroed out negative GNN outputs entirely.
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred_raw)

    # Normalize predictions to [0,1] — do NOT clip, normalize
    p_min, p_max = y_p.min(), y_p.max()
    if p_max > p_min:
        y_p_norm = (y_p - p_min) / (p_max - p_min)
    else:
        y_p_norm = np.zeros_like(y_p)

    threshold  = compute_dynamic_threshold(torch.tensor(y_t))
    y_binary   = (y_t >= threshold).astype(np.int32)
    n_positive = int(y_binary.sum())
    n_total    = len(y_binary)

    if n_positive == 0:
        raise RuntimeError(
            f"PR-AUC evaluation failed: 0 positive samples at threshold={threshold:.4f}. "
            f"y_true range: [{y_t.min():.4f}, {y_t.max():.4f}]. "
            f"Check label extraction — likely all SRISK values are identical."
        )

    positive_ratio = n_positive / n_total
    pr_auc = float(average_precision_score(y_binary, y_p_norm))
    precision, recall, _ = precision_recall_curve(y_binary, y_p_norm)

    if save_path:
        _save_pr_curve(precision, recall, pr_auc, positive_ratio, threshold, save_path)

    stride = max(1, len(precision) // 50)
    return {
        "pr_auc":           round(pr_auc, 6),
        "threshold_used":   round(float(threshold), 6),
        "positive_ratio":   round(float(positive_ratio), 6),
        "n_positive":       n_positive,
        "n_total":          int(n_total),
        "precision_coords": [round(float(p), 4) for p in precision[::stride]],
        "recall_coords":    [round(float(r), 4) for r in recall[::stride]],
    }


def evaluate_metrics(y_true, y_pred):
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))

    if np.std(y_t) < 1e-8 or np.std(y_p) < 1e-8:
        rank_corr = 0.0
    else:
        rank_corr, _ = stats.spearmanr(y_t, y_p)
        rank_corr = 0.0 if np.isnan(rank_corr) else float(rank_corr)

    # Top-K precision: of the top-5% predicted risky, how many are truly risky?
    k = max(1, int(len(y_t) * 0.05))
    top_k_pred_idx  = np.argsort(y_p)[-k:]
    threshold       = compute_dynamic_threshold(torch.tensor(y_t))
    truly_risky     = set(np.where(y_t >= threshold)[0])
    top_k_precision = len(set(top_k_pred_idx) & truly_risky) / k

    # NDCG for ranking quality (treats it as an IR problem)
    try:
        ndcg = float(ndcg_score(y_t.reshape(1, -1), y_p.reshape(1, -1), k=k))
    except Exception:
        ndcg = 0.0

    pr_data = calculate_pr_metrics(
        torch.tensor(y_t), torch.tensor(y_p)
    )

    return {
        "RMSE":             round(rmse, 6),
        "MAE":              round(mae, 6),
        "Rank_Correlation": round(rank_corr, 6),
        "PR_AUC":           pr_data["pr_auc"],
        "Top_K_Precision":  round(top_k_precision, 6),
        "NDCG_K":           round(ndcg, 6),
        "Dynamic_Threshold": pr_data["threshold_used"],
        "Critical_Nodes":   pr_data["n_positive"],
        "Total_Nodes":      pr_data["n_total"],
    }


def _save_pr_curve(precision, recall, pr_auc, positive_ratio, threshold, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#131314")
    ax.set_facecolor("#1c1b1c")
    ax.plot(recall, precision, color="#d3b47c", lw=2.5,
            label=f"GNN (PR-AUC={pr_auc:.4f})")
    ax.axhline(y=positive_ratio, color="#ffa599", ls="--", lw=1.5,
               label=f"Baseline ({positive_ratio:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"PR Curve — Top {int((1-POSITIVE_PERCENTILE)*100)}% threshold ({threshold:.4f})",
           xlim=[0, 1.05], ylim=[0, 1.05])
    ax.legend(facecolor="#2a2a2b", edgecolor="#3c494e", labelcolor="#e5e2e3")
    ax.tick_params(colors="#bbc8d0")
    ax.grid(True, alpha=0.15, color="#bbc8d0")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def verify_baselines(data, train_mask, test_mask):
    y_t   = data.y[test_mask].cpu().numpy()
    edges = data.edge_index.cpu().numpy().T
    weights = (
        data.edge_attr.cpu().numpy().flatten()
        if data.edge_attr is not None
        else np.ones(len(edges))
    )

    G = nx.DiGraph()
    for i, (u, v) in enumerate(edges):
        G.add_edge(int(u), int(v), weight=float(weights[i]))

    def _eval(preds_arr, name):
        print(f"  {name}:", evaluate_metrics(
            data.y[test_mask], torch.tensor(preds_arr, dtype=torch.float32)
        ))

    pr = nx.pagerank(G, weight="weight")
    pr_preds = np.array([pr.get(i, 0.0) for i in range(data.num_nodes)])[test_mask.cpu().numpy()]
    _eval(pr_preds, "PageRank")

    dc = nx.in_degree_centrality(G)
    dc_preds = np.array([dc.get(i, 0.0) for i in range(data.num_nodes)])[test_mask.cpu().numpy()]
    _eval(dc_preds, "Degree Centrality")

    # MLP fitted ONLY on train nodes (no leakage)
    X_train = data.x[train_mask].cpu().numpy()
    y_train = data.y[train_mask].cpu().numpy()
    X_test  = data.x[test_mask].cpu().numpy()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                       early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_train, y_train)
    _eval(mlp.predict(X_test), "MLP Regressor")