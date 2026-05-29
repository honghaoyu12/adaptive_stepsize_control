"""Example showing explicit Muon and auxiliary AdamW parameter groups."""

from __future__ import annotations

import torch

from delayed_feedback_muon import DelayedFeedbackMuon


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def main() -> None:
    model = TinyModel()
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not name.startswith("head"):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = DelayedFeedbackMuon(
        [
            {"params": muon_params, "use_muon": True, "lr": 2e-2},
            {"params": adamw_params, "use_muon": False, "lr": 1e-3},
        ],
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.95,
    )

    x = torch.randn(64, 8)
    y = torch.randn(64, 1)
    criterion = torch.nn.MSELoss()

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(loss=loss.item())

    print(optimizer.get_diagnostics())


if __name__ == "__main__":
    main()
