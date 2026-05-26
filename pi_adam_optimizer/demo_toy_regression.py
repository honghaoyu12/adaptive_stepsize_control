"""Small demo for PIAdam on a synthetic regression task.

Run:

    python demo_toy_regression.py

The demo intentionally uses a full batch so that the actual-vs-predicted
loss decrease is deterministic. For minibatch training, make sure the closure
uses the same batch for both the backward=True and backward=False calls inside
one optimizer step.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from pi_adam import PIAdam


def make_data(n: int = 128, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
    noise = 0.05 * torch.randn(n, 1, generator=generator)
    y = torch.sin(3.0 * x) + 0.3 * x**2 + noise
    return x, y


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)

    x, y = make_data()
    model = TinyMLP()
    criterion = nn.MSELoss()

    optimizer = PIAdam(
        model.parameters(),
        alpha0=1e-3,
        rho_star=0.8,
        kp=0.05,
        ki=0.001,
        integral_decay=0.95,
        integral_clip=(-2.0, 2.0),
        rho_smoothing=0.9,
        alpha_min=1e-6,
        alpha_max=1e-2,
        multiplicative_clip=(0.8, 1.25),
        reject_bad_steps=False,
    )

    losses = []
    alphas = []
    rhos = []
    rho_bars = []
    errors = []
    integrals = []

    def closure(backward: bool = True):
        # Important: this closure uses the same x, y each time it is called.
        # In minibatch training, capture one minibatch here and reuse it for
        # both before/after evaluations inside optimizer.step(closure).
        if backward:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        if backward:
            loss.backward()
        return loss

    for step in range(60):
        diag = optimizer.step(closure)
        losses.append(diag.loss_after if diag.loss_after is not None else diag.loss_before)
        alphas.append(diag.alpha)
        rhos.append(float("nan") if diag.rho is None else diag.rho)
        rho_bars.append(float("nan") if diag.rho_bar is None else diag.rho_bar)
        errors.append(float("nan") if diag.error is None else diag.error)
        integrals.append(diag.integral)

        if step % 10 == 0:
            print(
                f"step={step:03d} "
                f"loss={losses[-1]:.6f} "
                f"alpha={diag.alpha:.3e} "
                f"rho={diag.rho if diag.rho is not None else math.nan:.3f} "
                f"rho_bar={diag.rho_bar if diag.rho_bar is not None else math.nan:.3f} "
                f"I={diag.integral:.3f}"
            )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.semilogy(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(alphas)
    plt.xlabel("Step")
    plt.ylabel("alpha")
    plt.title("PI-controlled Adam global step size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "alpha.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(rhos, label="rho")
    plt.plot(rho_bars, label="rho_bar")
    plt.axhline(optimizer.rho_star, linestyle="--", label="rho_star")
    plt.xlabel("Step")
    plt.ylabel("actual / predicted decrease")
    plt.title("Controller signal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "rho.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(errors, label="error")
    plt.plot(integrals, label="integral")
    plt.xlabel("Step")
    plt.title("PI controller states")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "controller_states.png", dpi=160)
    plt.close()

    print("\nSaved plots to:")
    for name in ["loss.png", "alpha.png", "rho.png", "controller_states.png"]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
