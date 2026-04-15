import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

# After imports, before anything else
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% VRAM max
    torch.backends.cudnn.benchmark = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.autoencoder import FinancialAutoencoder
from backend.models.gnn_model import SystemicRiskGCN
from training.evaluation import evaluate_metrics, calculate_pr_metrics

# Paths
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(pwd, "data", "processed", "graph_data.pt")
save_params_path = os.path.join(pwd, "results", "automl_results.json")
save_gnn_path = os.path.join(pwd, "models", "best_gnn_model.pt")
save_ae_path = os.path.join(pwd, "models", "autoencoder_model.pt")
pr_curve_path = os.path.join(pwd, "results", "pr_curve.png")


def train_and_evaluate(trial, params, data_base):
    """
    Single Optuna trial: trains autoencoder + GNN on a subsampled dataset.
    Uses 25% node mask and 50 epochs for fast search.
    Returns PR-AUC (higher is better) for Optuna to maximize.
    """
    data = data_base.clone()

    # Subsample: keep only 25% of train nodes for quick evaluation
    subset_mask = torch.rand(data.num_nodes) < 0.25
    data.train_mask = data.train_mask & subset_mask

    active_train = data.train_mask.sum().item()
    active_val = data.val_mask.sum().item()
    if active_train < 10 or active_val < 10:
        raise optuna.TrialPruned()

    # [A] Train Autoencoder (short)
    input_features = data.x.size(1)
    encoder = FinancialAutoencoder(
        input_dim=input_features, latent_dim=params["latent_dim"]
    )
    optimizer_ae = optim.Adam(encoder.parameters(), lr=0.01)
    criterion_ae = nn.MSELoss()

    encoder.train()
    for _ in range(30):
        optimizer_ae.zero_grad()
        loss = criterion_ae(encoder(data.x), data.x)
        loss.backward()
        optimizer_ae.step()

    encoder.eval()

    # [B] Encode features
    with torch.no_grad():
        data.x = encoder.encoder(data.x)

    # [C] Train GNN
    model = SystemicRiskGCN(
        input_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    )
    optimizer_gnn = optim.Adam(model.parameters(), lr=params["lr"])
    criterion_gnn = nn.MSELoss()

    best_pr_auc = 0.0

    for epoch in range(50):
        model.train()
        optimizer_gnn.zero_grad()
        preds = model(data)
        train_loss = criterion_gnn(preds[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer_gnn.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(data)

            # Compute PR-AUC on validation set
            pr_data = calculate_pr_metrics(data.y[data.val_mask], val_preds[data.val_mask])
            epoch_pr_auc = pr_data["pr_auc"]

            # Strict bounds validation: PR-AUC must be in [0, 1]
            if not (0.0 <= epoch_pr_auc <= 1.0):
                raise ValueError(
                    f"PR-AUC out of bounds: {epoch_pr_auc:.6f}. "
                    f"Valid range is [0.0, 1.0]. This indicates stale data or an evaluation bug."
                )

            if epoch_pr_auc > best_pr_auc:
                best_pr_auc = epoch_pr_auc

            # Report for pruning (Optuna maximizes, so report PR-AUC)
            trial.report(epoch_pr_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(best_pr_auc)


def objective(trial):
    """Defines Optuna search space and runs a single trial."""
    params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "latent_dim": trial.suggest_int("latent_dim", 8, 32),
    }

    data = torch.load(data_path, weights_only=False)
    return train_and_evaluate(trial, params, data)


def finalize_best_model(study):
    """
    Retrains BOTH autoencoder and GNN using the best hyperparameters
    on the FULL dataset for 200 epochs, then saves BOTH model weights.
    Model selection is based on best PR-AUC on validation set.
    """
    print("\n" + "=" * 60)
    print("[FINAL TRAINING] Full dataset, 200 epochs, best params (PR-AUC optimized)")
    print("=" * 60)

    params = study.best_params
    data = torch.load(data_path, weights_only=False)

    # ---- 1. Train Autoencoder (full) ----
    print("\n[1/2] Training Autoencoder...")
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
            print(f"  Epoch {epoch:03d} | AE Reconstruction MSE: {loss.item():.6f}")

    # Save autoencoder
    encoder.load_state_dict(best_ae_state)
    torch.save(best_ae_state, save_ae_path)
    print(f"  OK Autoencoder saved -> {save_ae_path} (loss: {best_ae_loss:.6f})")

    encoder.eval()
    with torch.no_grad():
        data.x = encoder.encoder(data.x)

    # ---- 2. Train GNN (full, PR-AUC optimized) ----
    print("\n[2/2] Training GNN (PR-AUC optimized)...")
    model = SystemicRiskGCN(
        input_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    )
    opt_gnn = optim.Adam(model.parameters(), lr=params["lr"])
    crit_gnn = nn.MSELoss()

    best_gnn_pr_auc = 0.0
    best_gnn_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(200):
        model.train()
        opt_gnn.zero_grad()
        train_loss = crit_gnn(model(data)[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        opt_gnn.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(data)
            pr_data = calculate_pr_metrics(data.y[data.val_mask], val_preds[data.val_mask])
            val_pr_auc = pr_data["pr_auc"]

            if val_pr_auc > best_gnn_pr_auc:
                best_gnn_pr_auc = val_pr_auc
                best_gnn_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Train MSE: {train_loss.item():.6f} | Val PR-AUC: {val_pr_auc:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (no PR-AUC improvement for {patience} epochs)")
            break

    # Save GNN
    torch.save(best_gnn_state, save_gnn_path)
    print(f"  OK GNN saved -> {save_gnn_path} (best val PR-AUC: {best_gnn_pr_auc:.6f})")

    # ---- 3. Test metrics + PR Curve ----
    model.load_state_dict(best_gnn_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(data)
        metrics = evaluate_metrics(data.y[data.test_mask], final_preds[data.test_mask])
        print(f"\n[TEST METRICS] {metrics}")

        # Save PR curve
        calculate_pr_metrics(data.y[data.test_mask], final_preds[data.test_mask],
                             save_path=pr_curve_path)


def _clean_stale_artifacts():
    """Deletes stale model weights and results to guarantee a clean slate."""
    stale_paths = [save_params_path, save_gnn_path, save_ae_path]
    for p in stale_paths:
        if os.path.exists(p):
            os.remove(p)
            print(f"  CLEANED stale artifact: {p}")
    # Also remove PR curve if it exists
    if os.path.exists(pr_curve_path):
        os.remove(pr_curve_path)
        print(f"  CLEANED stale artifact: {pr_curve_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Optuna AutoML Search (PR-AUC Maximization, 30 trials)")
    print("=" * 60)

    # ── CLEAN SLATE: Remove all stale artifacts before starting ──
    print("\n[CLEANUP] Removing stale artifacts...")
    _clean_stale_artifacts()
    print("[CLEANUP] Done. Starting fresh Optuna study.\n")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=30)

    # ── POST-STUDY VALIDATION: best_score must be in [0.0, 1.0] ──
    if not (0.0 <= study.best_value <= 1.0):
        raise ValueError(
            f"PR-AUC out of bounds: {study.best_value:.6f}. "
            f"Valid range is [0.0, 1.0]. Refusing to save stale/impossible results."
        )

    print(f"\n[BEST] PR-AUC={study.best_value:.6f}, Params={study.best_params}")

    # Save results — filter out trials with out-of-bounds values
    os.makedirs(os.path.dirname(save_params_path), exist_ok=True)
    valid_trials = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
        if t.value is not None and 0.0 <= t.value <= 1.0
    ]
    with open(save_params_path, "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_score": study.best_value,
            "optimization_target": "PR-AUC (maximize)",
            "all_trials": valid_trials,
            "total_trials_run": len(study.trials),
            "valid_trials_saved": len(valid_trials),
        }, f, indent=4)

    finalize_best_model(study)
