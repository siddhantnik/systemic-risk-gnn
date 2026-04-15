import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.autoencoder import FinancialAutoencoder
from backend.models.gnn_model import SystemicRiskGCN

pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(pwd, "data", "processed", "graph_data.pt")
save_params_path = os.path.join(pwd, "results", "automl_results.json")
save_gnn_path = os.path.join(pwd, "models", "best_gnn_model.pt")
save_ae_path = os.path.join(pwd, "models", "autoencoder_model.pt")


def retrain_optimal():
    """
    Retrains both models using saved best params from AutoML.
    Saves both autoencoder and GNN weights.
    """
    with open(save_params_path, "r") as f:
        params = json.load(f)["best_params"]

    print(f"Retraining with params: {params}")
    data = torch.load(data_path, weights_only=False)

    # 1. Train Autoencoder
    print("[1/2] Training Autoencoder...")
    encoder = FinancialAutoencoder(
        input_dim=data.x.size(1), latent_dim=params["latent_dim"]
    )
    opt_ae = optim.Adam(encoder.parameters(), lr=0.005)
    crit_ae = nn.MSELoss()

    best_ae_loss = float("inf")
    best_ae_state = None
    encoder.train()

    for epoch in range(150):
        opt_ae.zero_grad()
        loss = crit_ae(encoder(data.x), data.x)
        loss.backward()
        opt_ae.step()
        if loss.item() < best_ae_loss:
            best_ae_loss = loss.item()
            best_ae_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | AE MSE: {loss.item():.6f}")

    encoder.load_state_dict(best_ae_state)
    torch.save(best_ae_state, save_ae_path)
    print(f"  OK Autoencoder saved (loss: {best_ae_loss:.6f})")

    encoder.eval()
    with torch.no_grad():
        data.x = encoder.encoder(data.x)

    # 2. Train GNN
    print("[2/2] Training GNN...")
    model = SystemicRiskGCN(
        input_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    )
    opt_gnn = optim.Adam(model.parameters(), lr=params["lr"])
    crit_gnn = nn.MSELoss()

    best_gnn_loss = float("inf")
    best_gnn_state = None

    for epoch in range(200):
        model.train()
        opt_gnn.zero_grad()
        train_loss = crit_gnn(model(data)[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        opt_gnn.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit_gnn(model(data)[data.val_mask], data.y[data.val_mask]).item()
            if val_loss < best_gnn_loss:
                best_gnn_loss = val_loss
                best_gnn_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Train: {train_loss.item():.6f} | Val: {val_loss:.6f}")

    torch.save(best_gnn_state, save_gnn_path)
    print(f"  OK GNN saved (val loss: {best_gnn_loss:.6f})")


if __name__ == "__main__":
    retrain_optimal()
