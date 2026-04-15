"""
train_autoencoder.py — Standalone VAE training script

CHANGES:
  OLD: Plain AE forward(x) → reconstructed, MSELoss
  NEW: VAE forward(x) → (x_hat, mu, logvar), vae_loss with beta annealing
  NEW: Gradient clipping (prevents exploding grads in early epochs)
  NEW: AdamW with weight decay instead of Adam
  NEW: LR scheduling with ReduceLROnPlateau
"""

import os, sys, torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.autoencoder import FinancialAutoencoder, vae_loss


def train():
    pwd       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(pwd, "data", "processed", "graph_data.pt")

    print("[1] Loading processed graph data...")
    data = torch.load(data_path, weights_only=False)
    num_features = data.x.size(1)
    print(f"    Input features: {num_features}")

    model     = FinancialAutoencoder(input_dim=num_features, latent_dim=32,
                                     hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

    best_loss, best_state = float("inf"), None
    model.train()

    for epoch in range(300):
        # Beta annealing: ramp KL weight from 0 → 0.5 over first 100 epochs
        beta = min(0.5, epoch / 100 * 0.5)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(data.x)
        loss, recon, kl   = vae_loss(data.x, x_hat, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  Ep {epoch:03d} | total={loss.item():.5f} "
                  f"recon={recon:.5f} kl={kl:.5f} beta={beta:.3f}")

    save_path = os.path.join(pwd, "models", "autoencoder_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    print(f"[DONE] Best loss={best_loss:.6f} → {save_path}")


if __name__ == "__main__":
    train()