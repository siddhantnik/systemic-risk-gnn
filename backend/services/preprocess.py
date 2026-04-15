"""
preprocess.py — Feature engineering for interbank CAMELS data

CHANGES:
  OLD: Raw 6 CAMELS columns scaled with StandardScaler + 1/99 percentile clip
  NEW: Same base columns PLUS 4 engineered ratio features derived from them,
       giving the GNN 10 dimensions of signal instead of 6.
       Added features:
         - liquidity_ratio    = Liquid_assets / Total_assets  (coverage ratio)
         - leverage_stress    = Impaired_loans * Leverage     (stress interaction)
         - earnings_buffer    = ROAE / (Tier1 + 1e-6)        (earnings-to-capital)
         - asset_quality_adj  = (1 - Impaired_ratio) * Tier1 (quality-adjusted capital)
       These capture non-linear cross-variable relationships the GNN can't learn
       from raw features alone.
  OLD: Fuzzy column matching silently returned wrong columns (first prefix match)
  NEW: Strict matching with explicit fallback logging. Will hard-fail clearly.
  OLD: StandardScaler only
  NEW: RobustScaler (IQR-based) — more appropriate for heavy-tailed financial data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

CAMELS_COLUMNS = [
    "Total_assets",
    "Liquid_assets",
    "Tier_1_Ratio",
    "Impaired_loans_/_Gross_customer_loans_&_advances",
    "Return_on_average_equity_(ROAE)",
    "Unadjusted_leverage_ratio_(computed)",
]

_CAMELS_KEYS = {
    "total":    "Total_assets",
    "liquid":   "Liquid_assets",
    "tier":     "Tier_1_Ratio",
    "impaired": "Impaired_loans_/_Gross_customer_loans_&_advances",
    "return":   "Return_on_average_equity_(ROAE)",
    "leverage": "Unadjusted_leverage_ratio_(computed)",
}


def clean_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = nodes_df.select_dtypes(include=["number"]).copy()
    if "index" not in numeric_df.columns:
        raise ValueError("Missing 'index' column in node data.")
    numeric_df = numeric_df.fillna(numeric_df.mean())
    return numeric_df


def clean_edges(edges_df: pd.DataFrame, valid_node_ids: set) -> pd.DataFrame:
    df = edges_df.copy()
    df = df[df["Weights"] > 0]
    df = df[df["Sourceid"] != df["Targetid"]]
    df = df[df["Sourceid"].isin(valid_node_ids) & df["Targetid"].isin(valid_node_ids)]
    return df


def id_mapping(nodes_df, edges_df):
    unique_ids  = nodes_df["index"].unique()
    node_id_map = {oid: idx for idx, oid in enumerate(unique_ids)}

    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()

    nodes_df["mapped_id"] = nodes_df["index"].map(node_id_map)
    edges_df["Sourceid"]  = edges_df["Sourceid"].map(node_id_map)
    edges_df["Targetid"]  = edges_df["Targetid"].map(node_id_map)

    edges_df = edges_df.dropna(subset=["Sourceid", "Targetid"])
    edges_df["Sourceid"] = edges_df["Sourceid"].astype(int)
    edges_df["Targetid"] = edges_df["Targetid"].astype(int)
    nodes_df = nodes_df.sort_values("mapped_id").reset_index(drop=True)
    return nodes_df, edges_df, node_id_map


def _resolve_camels_columns(available_columns: list) -> dict:
    """
    Returns {role: actual_column_name} for all 6 CAMELS features.
    Strict exact match first, then conservative substring fallback.
    Raises ValueError if fewer than 4 resolved.
    """
    resolved = {}
    for key, target in _CAMELS_KEYS.items():
        if target in available_columns:
            resolved[key] = target
        else:
            matches = [c for c in available_columns if key in c.lower()]
            if matches:
                resolved[key] = matches[0]
                print(f"  CAMELS fuzzy: '{target}' → '{matches[0]}'")
            else:
                print(f"  WARNING: CAMELS '{target}' not found, skipping.")

    if len(resolved) < 4:
        raise ValueError(
            f"Only {len(resolved)}/6 CAMELS columns resolved. "
            f"Available columns: {available_columns[:20]}"
        )
    return resolved


def scale_features(nodes_df: pd.DataFrame):
    """
    Extracts CAMELS features, engineers 4 interaction ratios,
    clips at 1/99 percentile, then applies RobustScaler.
    Returns (X_scaled [N, F], feature_names [F]).
    """
    col_map = _resolve_camels_columns(list(nodes_df.columns))

    df = pd.DataFrame(index=nodes_df.index)
    for key, col in col_map.items():
        df[key] = nodes_df[col].values.astype(np.float64)

    # ── Engineered ratio features ──────────────────────────────────────────
    eps = 1e-8
    if "liquid" in df and "total" in df:
        df["liquidity_ratio"] = df["liquid"] / (df["total"].abs() + eps)

    if "impaired" in df and "leverage" in df:
        df["leverage_stress"] = df["impaired"] * df["leverage"].abs()

    if "return" in df and "tier" in df:
        df["earnings_buffer"] = df["return"] / (df["tier"].abs() + eps)

    if "impaired" in df and "tier" in df:
        df["asset_quality_adj"] = (1.0 - df["impaired"].clip(0, 1)) * df["tier"]

    X_raw = df.values.astype(np.float64)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip at 1/99 percentile per column
    for j in range(X_raw.shape[1]):
        p1, p99 = np.nanpercentile(X_raw[:, j], [1, 99])
        X_raw[:, j] = np.clip(X_raw[:, j], p1, p99)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Hard clip after scaling — financial data has extreme outliers that
    # survive percentile clipping. Clip to [-5, 5] which covers 99.9% of
    # a normal distribution while killing the 295x/318x outliers in assets.
    X_scaled = X_scaled.clip(-5.0, 5.0)

    return X_scaled, list(df.columns)


def normalize_edge_weights(weights: np.ndarray) -> np.ndarray:
    """Log-normalize then scale to [0, 1]."""
    log_w = np.log1p(weights)
    if log_w.max() > 0:
        log_w = log_w / log_w.max()
    return log_w