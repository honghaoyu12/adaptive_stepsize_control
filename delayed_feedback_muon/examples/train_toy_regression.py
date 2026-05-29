"""Toy regression demo for DelayedFeedbackMuon.

Run from the project root:

    python examples/train_toy_regression.py
"""

from __future__ import annotations

import torch

from delayed_feedback_muon import DelayedFeedbackMuon


def make_data(n: int = 512, d: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(n, d)
    true_w = torch.randn(d, 1)
    y = x @ true_w + 0.1 * torch.randn(n, 1)
    return x, y


def main() -> None:
    torch.manual_seed(1)
    x, y = make_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )
    criterion = torch.nn.MSELoss()

    optimizer = DelayedFeedbackMuon(
        model.parameters(),
        lr=2e-2,
        weight_decay=0.01,
        alpha_init=1.0,
        alpha_bounds=(0.1, 10.0),
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.95,
        multiplier_bounds=(0.8, 1.25),
    )

    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step(loss=loss.item())

        if step % 10 == 0 or step == 99:
            diag = optimizer.get_diagnostics()
            print(
                f"step={step:03d} "
                f"loss={loss.item():.6f} "
                f"alpha={diag['alpha']:.4f} "
                f"rho_bar={diag['rho_bar']} "
                f"muon_tensors={diag['last_muon_tensors']} "
                f"aux_tensors={diag['last_aux_adamw_tensors']}"
            )


if __name__ == "__main__":
    main()
