"""Smoke tests for DelayedFeedbackAdam."""

from __future__ import annotations

import torch

from delayed_feedback_adam import DelayedFeedbackAdam


def test_optimizer_reduces_simple_quadratic_loss() -> None:
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = DelayedFeedbackAdam(
        [parameter],
        lr=0.05,
        alpha_init=1.0,
        alpha_bounds=(0.01, 100.0),
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.0,
    )

    initial_loss = None
    final_loss = None
    for _ in range(80):
        optimizer.zero_grad()
        loss = 0.5 * parameter.pow(2).sum()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step(loss=loss.item())
        final_loss = loss.item()

    assert final_loss is not None
    assert initial_loss is not None
    assert final_loss < initial_loss
    assert abs(parameter.item()) < 5.0


def test_diagnostics_contains_expected_keys() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DelayedFeedbackAdam([parameter], lr=0.01)

    optimizer.zero_grad()
    loss = 0.5 * parameter.pow(2).sum()
    loss.backward()
    optimizer.step(loss=loss.item())

    diagnostics = optimizer.get_diagnostics()
    assert "alpha" in diagnostics
    assert "last_predicted_decrease" in diagnostics
    assert diagnostics["alpha"] > 0


def test_state_dict_round_trip_preserves_controller() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DelayedFeedbackAdam([parameter], lr=0.01, alpha_init=2.0)
    state = optimizer.state_dict()

    new_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    new_optimizer = DelayedFeedbackAdam([new_parameter], lr=0.01, alpha_init=1.0)
    new_optimizer.load_state_dict(state)

    assert abs(new_optimizer.get_diagnostics()["alpha"] - 2.0) < 1e-12


def test_decoupled_adamw_direction_matches_torch_adamw() -> None:
    torch.manual_seed(123)
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    reference = torch.nn.Parameter(parameter.detach().clone())
    gradients = [torch.randn_like(parameter) for _ in range(3)]

    optimizer = DelayedFeedbackAdam(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        adam_eps=1e-8,
        weight_decay=0.01,
        decoupled_weight_decay=True,
        alpha_init=1.0,
        fallback_to_gradient=False,
    )
    torch_optimizer = torch.optim.AdamW(
        [reference],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    for grad in gradients:
        parameter.grad = grad.clone()
        reference.grad = grad.clone()
        optimizer.step(loss=None)
        torch_optimizer.step()

    assert torch.allclose(parameter, reference, atol=1e-7, rtol=1e-6)
