"""Compare DelayedFeedbackAdam with torch.optim.Adam on a tiny problem.

This is not a benchmark. It is a smoke/demo script showing usage and diagnostics.

Run from the project root:

    python examples/compare_with_torch_adam.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy

import torch

from delayed_feedback_adam import DelayedFeedbackAdam


def make_data(n: int = 512, d: int = 20, seed: int = 123):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    true_w = torch.randn(d, 1, generator=generator)
    y = x @ true_w + 0.2 * torch.randn(n, 1, generator=generator)
    return x, y


def make_model(d: int):
    return torch.nn.Sequential(
        torch.nn.Linear(d, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


def train_one_epoch(model, optimizer, x, y, batch_size: int = 64):
    criterion = torch.nn.MSELoss()
    perm = torch.randperm(len(x))
    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(x), batch_size):
        idx = perm[start : start + batch_size]
        xb = x[idx]
        yb = y[idx]

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()

        if isinstance(optimizer, DelayedFeedbackAdam):
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main() -> None:
    torch.manual_seed(123)
    x, y = make_data()

    model_a = make_model(x.shape[1])
    model_b = copy.deepcopy(model_a)

    delayed = DelayedFeedbackAdam(
        model_a.parameters(),
        lr=1e-3,
        alpha_init=1.0,
        alpha_bounds=(0.1, 10.0),
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.95,
    )
    adam = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    print("epoch delayed_loss adam_loss delayed_alpha delayed_rho_bar")
    for epoch in range(15):
        delayed_loss = train_one_epoch(model_a, delayed, x, y)
        adam_loss = train_one_epoch(model_b, adam, x, y)
        diagnostics = delayed.get_diagnostics()
        print(
            f"{epoch:02d} "
            f"{delayed_loss:.6f} "
            f"{adam_loss:.6f} "
            f"{diagnostics['alpha']:.4f} "
            f"{diagnostics['rho_bar']}"
        )


if __name__ == "__main__":
    main()
