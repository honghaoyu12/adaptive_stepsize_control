"""Toy demo for PI-controlled Muon.

This trains a small MLP on a synthetic regression problem. It is intentionally
small and CPU-friendly. The purpose is to demonstrate the closure pattern and
controller diagnostics, not to claim benchmark performance.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pi_muon import PIMuon, default_muon_param_groups


def make_data(n: int = 256, d: int = 16, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    true_w = torch.randn(d, 1, generator=gen)
    y = torch.sin(X @ true_w) + 0.05 * torch.randn(n, 1, generator=gen)
    return X, y


class TinyMLP(nn.Module):
    def __init__(self, d: int = 16, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(1)
    X, y = make_data()

    model = TinyMLP(d=X.shape[1], width=32)
    loss_fn = nn.MSELoss()

    # For large transformer-like models, you usually want Muon only on hidden
    # matrix weights, and AdamW on embeddings, heads, norms, biases, etc.
    # This helper is conservative by name. For this toy MLP, it will use Muon
    # for the hidden matrix weights and AdamW for biases / output head.
    param_groups = default_muon_param_groups(model.named_parameters())

    optimizer = PIMuon(
        param_groups,
        alpha0=3e-3,
        alpha_min=1e-5,
        alpha_max=5e-2,
        rho_star=0.8,
        kp=0.05,
        ki=0.001,
        beta_rho=0.9,
        integral_decay=0.95,
        integral_clip=5.0,
        multiplier_min=0.85,
        multiplier_max=1.15,
        muon_momentum=0.95,
        ns_steps=5,
        weight_decay=0.0,
        reject_bad_steps=False,
    )

    batch_size = 64
    steps = 45

    for step in range(1, steps + 1):
        idx = torch.randint(0, X.shape[0], (batch_size,))
        xb = X[idx]
        yb = y[idx]

        def closure(backward: bool = True):
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if backward:
                loss.backward()
            return loss

        optimizer.step(closure)

        if step % 5 == 0 or step == 1:
            stats = optimizer.last_stats
            assert stats is not None
            print(
                f"step={step:03d} "
                f"loss={stats.loss_after:.5f} "
                f"alpha={stats.alpha:.3e} "
                f"rho={stats.rho:+.3f} "
                f"rho_bar={stats.rho_bar:+.3f} "
                f"err={stats.error:+.3f} "
                f"I={stats.integral:+.3f} "
                f"accepted={stats.accepted} "
                f"fallback={stats.used_fallback_direction}"
            )

    with torch.no_grad():
        full_loss = loss_fn(model(X), y).item()
    print(f"\nFinal full-data loss: {full_loss:.6f}")


if __name__ == "__main__":
    main()
