"""Train a tiny regression model with DelayedFeedbackAdam.

Run from the project root:

    python examples/train_toy_regression.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from delayed_feedback_adam import DelayedFeedbackAdam


def make_data(n: int = 512, d: int = 10, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    true_w = torch.randn(d, 1, generator=generator)
    y = x @ true_w + 0.1 * torch.randn(n, 1, generator=generator)
    return x, y


def main() -> None:
    torch.manual_seed(0)
    x, y = make_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )
    criterion = torch.nn.MSELoss()

    optimizer = DelayedFeedbackAdam(
        model.parameters(),
        lr=1e-3,
        alpha_init=1.0,
        alpha_bounds=(0.1, 10.0),
        rho_star=0.5,
        kp=0.05,
        ki=0.0,
        kd=0.0,
        rho_beta=0.0,
        rho_clip=(-1.0, 2.0),
        multiplier_bounds=(0.8, 1.25),
    )

    batch_size = len(x)  # full-batch demo: delayed feedback is accurate but one-step delayed
    num_epochs = 20

    for epoch in range(num_epochs):
        permutation = torch.randperm(len(x))
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(x), batch_size):
            idx = permutation[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step(loss=loss.item())

            epoch_loss += loss.item()
            num_batches += 1

        diagnostics = optimizer.get_diagnostics()
        print(
            f"epoch={epoch:02d} "
            f"loss={epoch_loss / num_batches:.6f} "
            f"alpha={diagnostics['alpha']:.4f} "
            f"rho_bar={diagnostics['rho_bar']} "
            f"rho_raw={diagnostics['last_rho_raw']}"
        )


if __name__ == "__main__":
    main()
