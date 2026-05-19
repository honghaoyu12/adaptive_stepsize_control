"""Tests for PyTorch controlled Adam minibatch mechanics."""

import torch
import torch.nn as nn

from controlled_adam.torch_optimizers import TorchControlledAdam


def test_controlled_adam_accepts_same_batch_loss_reduction() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    x = torch.randn(16, 2)
    y = torch.randn(16, 1)
    criterion = nn.MSELoss()

    optimizer = TorchControlledAdam(model.parameters(), alpha0=1e-2, alpha_max=1e-1)
    optimizer.zero_grad()
    loss_before = criterion(model(x), y)
    loss_before.backward()

    def same_batch_loss() -> torch.Tensor:
        return criterion(model(x), y)

    step = optimizer.step(loss_before, same_batch_loss)

    assert step.accepted
    assert step.loss_after < step.loss_before
    assert step.actual_decrease > 0.0
    assert step.predicted_decrease > 0.0


def test_controlled_adam_rejects_and_restores_bad_trial_step() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    x = torch.randn(16, 2)
    y = torch.randn(16, 1)
    criterion = nn.MSELoss()

    optimizer = TorchControlledAdam(
        model.parameters(),
        alpha0=10.0,
        alpha_max=10.0,
        max_backtracks=0,
    )
    params_before = [param.detach().clone() for param in model.parameters()]
    optimizer.zero_grad()
    loss_before = criterion(model(x), y)
    loss_before.backward()

    def same_batch_loss() -> torch.Tensor:
        return criterion(model(x), y)

    step = optimizer.step(loss_before, same_batch_loss)

    assert not step.accepted
    for param, before in zip(model.parameters(), params_before):
        torch.testing.assert_close(param, before)


def test_trust_region_expands_tiny_high_quality_steps() -> None:
    model = nn.Linear(1, 1)
    optimizer = TorchControlledAdam(
        model.parameters(),
        alpha0=1e-5,
        alpha_min=1e-8,
        alpha_max=1e-1,
        kp=0.0,
        max_alpha_factor=1.05,
        trust_region_expand=True,
        trust_region_rho_threshold=0.9,
        trust_region_alpha_threshold=1e-4,
        trust_region_expand_factor=2.0,
    )

    alpha_next, alpha_update_factor, trust_region_expanded = (
        optimizer._next_alpha_after_trial(
            alpha_used=1e-5,
            rho_control=0.95,
            backtracks=0,
        )
    )

    assert trust_region_expanded
    assert alpha_update_factor == 2.0
    assert alpha_next == 2e-5


def test_trust_region_does_not_expand_after_backtracking() -> None:
    model = nn.Linear(1, 1)
    optimizer = TorchControlledAdam(
        model.parameters(),
        alpha0=1e-5,
        kp=0.0,
        max_alpha_factor=1.05,
        trust_region_expand=True,
        trust_region_rho_threshold=0.9,
        trust_region_alpha_threshold=1e-4,
        trust_region_expand_factor=2.0,
    )

    alpha_next, alpha_update_factor, trust_region_expanded = (
        optimizer._next_alpha_after_trial(
            alpha_used=1e-5,
            rho_control=0.95,
            backtracks=1,
        )
    )

    assert not trust_region_expanded
    assert alpha_update_factor == 1.0
    assert alpha_next == 1e-5
