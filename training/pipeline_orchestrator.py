import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_loader import load_dataset, load_all_quarters
from backend.services.preprocess import clean_nodes, clean_edges, id_mapping, scale_features
from backend.services.graph_builder import build_graph
from training.evaluation import extract_systemic_risk_labels


def generate_masks(num_nodes: int, train_ratio=0.7, val_ratio=0.15):
    """70% train / 15% val / 15% test split, fixed seed for reproducibility."""
    gen     = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_nodes, generator=gen)

    train_end = int(train_ratio * num_nodes)
    val_end   = train_end + int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]]      = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]]         = True

    return train_mask, val_mask, test_mask


def run_pipeline(quarter=None, save_path=None, use_all_quarters=False):
    """
    Full pipeline: load → clean → CAMELS features → scale → graph → labels → save.

    Args:
        quarter:          Single quarter string e.g. '2019Q3'. Used if use_all_quarters=False.
        save_path:        Where to save the .pt file. Defaults to data/processed/graph_data.pt
        use_all_quarters: If True, loads and concatenates all available quarters.
    """
    # Resolve save path relative to project root
    pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_path is None:
        save_path = os.path.join(pwd, "data", "processed", "graph_data.pt")

    if use_all_quarters:
        print("--- Pipeline: ALL QUARTERS ---")
        print("[1] Loading all quarterly datasets...")
        nodes_df, edges_df, quarters = load_all_quarters()
    else:
        quarter = quarter or "2019Q3"
        print(f"--- Pipeline: {quarter} ---")
        print("[1] Loading raw data...")
        nodes_df, edges_df = load_dataset(quarter)

    # Extract SRISK labels from ORIGINAL nodes_df BEFORE clean_nodes fills NaN with mean
    # NaN = bank has no systemic risk measurement = safe = 0.0
    if "srisk_ratio" in nodes_df.columns:
        srisk_raw_series = nodes_df.set_index("index")["srisk_ratio"]
    elif "srisk_value" in nodes_df.columns:
        srisk_raw_series = nodes_df.set_index("index")["srisk_value"]
    else:
        srisk_raw_series = None

    cleaned_nodes = clean_nodes(nodes_df)
    valid_ids     = set(cleaned_nodes["index"].values)
    cleaned_edges = clean_edges(edges_df, valid_ids)

    mapped_nodes, mapped_edges, node_id_map = id_mapping(cleaned_nodes, cleaned_edges)

    # Align raw srisk to the mapped/sorted node order
    if srisk_raw_series is not None:
        # node_id_map: original_id -> mapped_id
        # We need mapped_id -> original srisk value
        reverse_map = {v: k for k, v in node_id_map.items()}
        srisk_raw = np.array([
            srisk_raw_series.get(reverse_map[i], np.nan)
            for i in range(len(mapped_nodes))
        ])
    else:
        srisk_raw = None
        
    # [3] CAMELS features
    print("[3] Extracting and scaling CAMELS features...")
    X_scaled, chosen_cols = scale_features(mapped_nodes)
    print(f"    Using {len(chosen_cols)} features: {chosen_cols}")

    # [4] Build graph
    print("[4] Building PyG graph...")
    data = build_graph(mapped_nodes, mapped_edges, X_scaled)

    # [5] Labels — extract from RAW srisk (NaN = 0.0, not filled with mean)
    print("[5] Extracting SRISK labels...")
    data.y = extract_systemic_risk_labels(mapped_nodes, srisk_raw=srisk_raw)

    # Check labels are non-trivial
    if data.y.std() < 0.01:
        print("  WARNING: Labels have near-zero variance — check srisk_ratio column!")

    # [6] Train/val/test masks
    data.train_mask, data.val_mask, data.test_mask = generate_masks(data.num_nodes)

    # [7] Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)

    print(f"\n[DONE] Saved → {save_path}")
    print(f"  Nodes:    {data.num_nodes:,}")
    print(f"  Edges:    {data.num_edges:,}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Train / Val / Test: "
          f"{data.train_mask.sum()} / {data.val_mask.sum()} / {data.test_mask.sum()}")
    return data


if __name__ == "__main__":
    run_pipeline(use_all_quarters=True)