import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from backend.services.preprocess import normalize_edge_weights


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, X_scaled) -> Data:
    """
    Constructs a PyTorch Geometric Data object from preprocessed data.
    
    Args:
        nodes_df: DataFrame with 'mapped_id' column, ordered by mapped_id.
        edges_df: DataFrame with 'Sourceid', 'Targetid', 'Weights' (mapped IDs).
        X_scaled: numpy array of shape [num_nodes, num_features], scaled features.
    
    Returns:
        PyG Data object with x, edge_index, edge_attr (1D).
    """
    # Node features
    node_features = torch.tensor(X_scaled, dtype=torch.float)

    # Edge index: [2, num_edges] long tensor
    edge_index = torch.tensor(
        edges_df[["Sourceid", "Targetid"]].values.T, dtype=torch.long
    )

    # Edge weights: log-normalize then store as 1D tensor
    raw_weights = edges_df["Weights"].values.astype(np.float64)
    norm_weights = normalize_edge_weights(raw_weights)
    edge_attr = torch.tensor(norm_weights, dtype=torch.float)

    # Ensure edge_attr is always 1D — GCNConv expects [num_edges]
    if edge_attr.dim() > 1:
        edge_attr = edge_attr.squeeze(-1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data
