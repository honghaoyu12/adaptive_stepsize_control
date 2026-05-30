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
    assert step.predicted_decrease_safe > 0.0


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


def test_controlled_adam_uses_scaled_gradient_fallback_for_non_descent_direction() -> None:
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = TorchControlledAdam(
        [param],
        alpha0=0.1,
        alpha_max=1.0,
        kp=0.0,
        max_backtracks=0,
    )
    optimizer.m[0].fill_(-10.0)
    optimizer.v[0].fill_(1.0)

    loss_before = 0.5 * param.square().sum()
    loss_before.backward()

    def same_batch_loss() -> torch.Tensor:
        return 0.5 * param.square().sum()

    step = optimizer.step(loss_before, same_batch_loss)

    assert step.used_gradient_fallback
    assert step.accepted
    assert step.descent_score > 0.0
    assert param.item() < 1.0
    assert step.loss_after < step.loss_before


def test_rho_clipping_controls_alpha_without_changing_measured_acceptance() -> None:
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = TorchControlledAdam(
        [param],
        alpha0=0.1,
        alpha_max=10.0,
        kp=1.0,
        rho_star=0.0,
        max_alpha_factor=10.0,
        trust_region_expand=False,
        use_rho_ema=False,
        rho_clip_max=0.1,
        max_backtracks=0,
    )

    loss_before = 0.5 * param.square().sum()
    loss_before.backward()

    def same_batch_loss() -> torch.Tensor:
        return 0.5 * param.square().sum()

    step = optimizer.step(loss_before, same_batch_loss)

    assert step.accepted
    assert step.rho > 0.1
    assert step.rho_clipped == 0.1
    assert step.rho_was_clipped
    assert abs(step.alpha_next - step.alpha * 1.1051709180756477) < 1e-12


def test_predicted_decrease_floor_stabilizes_tiny_denominator() -> None:
    model = nn.Linear(1, 1)
    optimizer = TorchControlledAdam(
        model.parameters(),
        absolute_predicted_floor=1e-6,
        relative_predicted_floor=0.0,
    )

    rho, rho_clipped, predicted_safe, was_floored, was_clipped = optimizer._measure_rho(
        actual_decrease=1e-9,
        predicted_decrease_raw=1e-12,
        loss_before_value=1.0,
    )

    assert predicted_safe == 1e-6
    assert was_floored
    assert rho == 1e-3
    assert rho_clipped == 1e-3
    assert not was_clipped


def test_rejected_trial_sequence_cannot_increase_next_alpha() -> None:
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = TorchControlledAdam(
        [param],
        alpha0=0.1,
        alpha_max=10.0,
        kp=1.0,
        rho_star=0.0,
        max_alpha_factor=10.0,
        trust_region_expand=False,
        use_rho_ema=False,
        rho_min=2.0,
        max_backtracks=1,
        backtrack_shrink=0.5,
    )
    loss_before = 0.5 * param.square().sum()
    loss_before.backward()

    def same_batch_loss() -> torch.Tensor:
        return 0.5 * param.square().sum()

    step = optimizer.step(loss_before, same_batch_loss)

    assert not step.accepted
    assert step.alpha == 0.05
    assert step.alpha_next <= step.alpha
    assert optimizer.alpha <= 0.05


def test_trust_region_expands_after_patience() -> None:
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
        trust_region_max_factor=2.0,
        trust_region_patience=2,
    )

    first_alpha_next, first_factor, first_expanded = (
        optimizer._next_alpha_after_trial(
            alpha_used=1e-5,
            rho_control=0.95,
            backtracks=0,
        )
    )
    alpha_next, alpha_update_factor, trust_region_expanded = (
        optimizer._next_alpha_after_trial(
            alpha_used=1e-5,
            rho_control=0.95,
            backtracks=0,
        )
    )

    assert not first_expanded
    assert first_factor == 1.0
    assert first_alpha_next == 1e-5
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
        trust_region_max_factor=2.0,
        trust_region_patience=1,
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


def test_trust_region_expansion_obeys_hard_factor_bound() -> None:
    model = nn.Linear(1, 1)
    optimizer = TorchControlledAdam(
        model.parameters(),
        alpha0=1e-5,
        kp=0.0,
        max_alpha_factor=1.05,
        trust_region_expand=True,
        trust_region_rho_threshold=0.9,
        trust_region_alpha_threshold=1e-4,
        trust_region_expand_factor=3.0,
        trust_region_max_factor=2.0,
        trust_region_patience=1,
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


def test_default_backtracking_depth_is_one() -> None:
    model = nn.Linear(1, 1)
    optimizer = TorchControlledAdam(model.parameters())

    assert optimizer.max_backtracks == 1
