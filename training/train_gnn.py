"""
train_gnn.py — Two-stage training with dual-head GNN

CHANGES:
  OLD: model(data) returns single tensor (regression only)
  NEW: model(data) returns (reg_out, cls_out) — dual head
  OLD: MSE on val used for model selection despite PR-AUC being the metric
  NEW: Best model selected by val PR-AUC
  NEW: Gradient clipping, AdamW, LR scheduling
"""

import os, sys, json, torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.autoencoder import FinancialAutoencoder, vae_loss
from backend.models.gnn_model import SystemicRiskGNN
from training.evaluation import evaluate_metrics, verify_baselines, calculate_pr_metrics
def dual_head_loss(reg_out, cls_out, y_true, alpha=0.5):
    from training.evaluation import compute_dynamic_threshold
    import torch.nn.functional as F
    threshold = compute_dynamic_threshold(y_true)
    y_bin = (y_true >= threshold).float()
    mse = F.mse_loss(reg_out, y_true)
    # Keep pos_weight on same device as the tensors
    pos_weight = torch.tensor(
        [(y_bin == 0).sum() / (y_bin.sum() + 1e-8)]
    ).to(reg_out.device)
    bce = F.binary_cross_entropy_with_logits(cls_out, y_bin, pos_weight=pos_weight)
    return alpha * mse + (1 - alpha) * bce


def train_gnn():
    from torch_geometric.loader import NeighborLoader

    pwd         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(pwd, "data", "processed", "graph_data.pt")
    ae_path     = os.path.join(pwd, "models", "autoencoder_model.pt")
    params_path = os.path.join(pwd, "results", "automl_results.json")

    print("[1] Loading graph data...")
    data = torch.load(data_path, weights_only=False)
    input_features = data.x.size(1)
    print(f"    Shape: {data.x.shape}, Edges: {data.edge_index.shape[1]}")

    params = {"latent_dim": 32, "hidden_dim": 48, "num_layers": 2,
              "heads": 4, "dropout": 0.2, "edge_dropout": 0.1,
              "lr": 1e-3, "alpha": 0.5}
    if os.path.exists(params_path):
        with open(params_path) as f:
            saved = json.load(f)
        params.update(saved["best_params"])
        params["latent_dim"] = saved.get("latent_dim", params["latent_dim"])
        # Cap num_layers at 2 for memory safety on large graphs
        params["num_layers"] = min(params.get("num_layers", 2), 2)
        print(f"    Params: {params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    # [2] Encoder
    print("[2] Loading autoencoder...")
    enc = FinancialAutoencoder(input_dim=input_features,
                               latent_dim=params["latent_dim"], hidden_dim=64)
    if os.path.exists(ae_path):
        enc.load_state_dict(torch.load(ae_path, weights_only=True))
        print("    Loaded pre-trained encoder")
    enc = enc.to(device)
    enc.eval()
    with torch.no_grad():
        data.x = enc.encode(data.x.to(device)).cpu()
    print(f"    Encoded shape: {data.x.shape}")

    # [3] Build GNN
    print("[3] Training GNN with neighbor sampling...")
    model = SystemicRiskGNN(
        input_dim    = params["latent_dim"],
        hidden_dim   = params["hidden_dim"],
        num_layers   = params["num_layers"],
        heads        = params.get("heads", 4),
        dropout      = params["dropout"],
        edge_dropout = params.get("edge_dropout", 0.1),
    ).to(device)
    opt   = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    sch   = ReduceLROnPlateau(opt, patience=20, factor=0.5)
    alpha = params.get("alpha", 0.5)

    # Neighbor sampler — 2 hops, 20 neighbors each
    # Keeps memory bounded regardless of graph size
    train_loader = NeighborLoader(
        data,
        num_neighbors=[20, 10],
        batch_size=512,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[20, 10],
        batch_size=512,
        input_nodes=data.val_mask,
        shuffle=False,
    )

    best_pr, best_state = 0.0, None

    for epoch in range(300):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            reg, cls = model(batch)
            # Only compute loss on seed nodes (first batch_size nodes)
            n = batch.batch_size
            loss = dual_head_loss(
                reg[:n], cls[:n], batch.y[:n], alpha=alpha
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reg, _ = model(batch)
                n = batch.batch_size
                val_preds.append(reg[:n].cpu())
                val_true.append(batch.y[:n].cpu())

        val_preds = torch.cat(val_preds)
        val_true  = torch.cat(val_true)

        try:
            pr    = calculate_pr_metrics(val_true, val_preds)
            vp    = pr["pr_auc"]
        except Exception:
            vp = 0.0

        sch.step(1 - vp)
        if vp > best_pr:
            best_pr    = vp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"  Ep {epoch:03d} | loss={avg_loss:.5f} | val PR-AUC={vp:.5f}")

    # [4] Save
    save_dir = os.path.join(pwd, "models")
    os.makedirs(save_dir, exist_ok=True)
    gnn_path = os.path.join(save_dir, "best_gnn_model.pt")
    torch.save(best_state, gnn_path)
    print(f"  GNN saved (best val PR-AUC={best_pr:.5f})")

    # [5] Full test evaluation using test loader
    print("\n[4] Test Metrics:")
    model.load_state_dict(best_state)
    model.eval()

    test_loader = NeighborLoader(
        data,
        num_neighbors=[20, 10],
        batch_size=512,
        input_nodes=data.test_mask,
        shuffle=False,
    )
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reg, _ = model(batch)
            n = batch.batch_size
            test_preds.append(reg[:n].cpu())
            test_true.append(batch.y[:n].cpu())

    test_preds = torch.cat(test_preds)
    test_true  = torch.cat(test_true)
    metrics    = evaluate_metrics(test_true, test_preds)
    print(f"  GNN: {metrics}")

    verify_baselines(data, data.train_mask, data.test_mask)

    # [6] Export top predictions
    print("\n[5] Exporting predictions...")
    all_preds = []
    all_loader = NeighborLoader(
        data,
        num_neighbors=[20, 10],
        batch_size=512,
        input_nodes=torch.ones(data.num_nodes, dtype=torch.bool),
        shuffle=False,
    )
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            reg, _ = model(batch)
            n = batch.batch_size
            all_preds.append(reg[:n].cpu())

    all_scores = torch.cat(all_preds).numpy()
    res_df = pd.DataFrame({
        "Bank_ID":    [f"Bank_{i:04d}" for i in range(len(all_scores))],
        "Risk_Score": all_scores,
    }).sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    res_df["Rank"] = res_df.index + 1

    res_path = os.path.join(pwd, "results", "predictions.csv")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    res_df.to_csv(res_path, index=False)
    print(f"  Top 5:\n{res_df.head().to_string()}")

    # [3] GNN
    print("[3] Training GNN...")
    model = SystemicRiskGNN(
        input_dim    = params["latent_dim"],
        hidden_dim   = params["hidden_dim"],
        num_layers   = params["num_layers"],
        heads        = params.get("heads", 4),
        dropout      = params["dropout"],
        edge_dropout = params.get("edge_dropout", 0.1),
    )
    opt = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, patience=20, factor=0.5)
    alpha = params.get("alpha", 0.5)

    best_pr, best_state = 0.0, None

    for epoch in range(300):
        model.train()
        opt.zero_grad()
        reg, cls = model(data)
        loss = dual_head_loss(reg[data.train_mask], cls[data.train_mask],
                              data.y[data.train_mask], alpha=alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            v_reg, _ = model(data)
            pr = calculate_pr_metrics(data.y[data.val_mask], v_reg[data.val_mask])
            vp = pr["pr_auc"]
        sch.step(1 - vp)

        if vp > best_pr:
            best_pr  = vp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  Ep {epoch:03d} | loss={loss.item():.5f} | val PR-AUC={vp:.5f}")

    save_dir = os.path.join(pwd, "models")
    os.makedirs(save_dir, exist_ok=True)
    gnn_path = os.path.join(save_dir, "best_gnn_model.pt")
    torch.save(best_state, gnn_path)
    print(f"  GNN saved (best val PR-AUC={best_pr:.5f})")

    # [4] Evaluate
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t_reg, _ = model(data)
        metrics  = evaluate_metrics(data.y[data.test_mask], t_reg[data.test_mask])
    print(f"\n[TEST METRICS] {metrics}")
    verify_baselines(data, data.train_mask, data.test_mask)

    # [5] Export
    scores  = t_reg.detach().cpu().numpy()
    res_df  = pd.DataFrame({
        "Bank_ID":    [f"Bank_{i:04d}" for i in range(len(scores))],
        "Risk_Score": scores,
    }).sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    res_df["Rank"] = res_df.index + 1
    res_path = os.path.join(pwd, "results", "predictions.csv")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    res_df.to_csv(res_path, index=False)
    print(f"\n  Top 5:\n{res_df.head().to_string()}")


if __name__ == "__main__":
    train_gnn()