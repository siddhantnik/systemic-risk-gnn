"""
autoencoder.py — Variational Financial Autoencoder

CHANGES:
  OLD: Plain deterministic AE, BatchNorm1d (crashes batch_size=1), no latent regularization
  NEW: Beta-VAE with reparameterization, LayerNorm (stable at any batch size),
       residual blocks for gradient flow, Kaiming init, separate encode() that
       returns deterministic mu — no sampling noise during GNN inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FinancialAutoencoder(nn.Module):
    """
    Beta-VAE for CAMELS feature compression.
    Use encode(x) → mu for deterministic GNN input (no sampling noise at inference).
    forward(x) → (x_hat, mu, logvar) for training with vae_loss().
    """
    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.enc_res1 = ResidualBlock(hidden_dim, dropout)
        self.enc_res2 = ResidualBlock(hidden_dim, dropout)
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.dec_res1 = ResidualBlock(hidden_dim, dropout)
        self.dec_res2 = ResidualBlock(hidden_dim, dropout)
        self.dec_out  = nn.Linear(hidden_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_trunk(self, x):
        return self.enc_res2(self.enc_res1(self.enc_proj(x)))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar).clamp(max=10.0)
            return mu + torch.randn_like(std) * std
        return mu  # deterministic at inference

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent mean — use this for GNN, not forward()."""
        h = self._encode_trunk(x)
        return self.mu_head(h)

    def decode(self, z):
        h = self.dec_proj(z)
        return self.dec_out(self.dec_res2(self.dec_res1(h)))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h      = self._encode_trunk(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 2.0)
        z      = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta: float = 0.5):
    """
    Beta-VAE loss. beta=0.5 balances reconstruction vs latent regularity.
    Returns (total, recon_mse_scalar, kl_scalar) for logging.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()