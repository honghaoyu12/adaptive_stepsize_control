import math

import torch

from pi_adam import PIAdam


def test_alpha0_is_clipped_to_bounds():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIAdam([param], alpha0=1.0, alpha_min=1e-5, alpha_max=1e-2)

    assert math.isclose(optimizer.alpha, 1e-2)


def test_no_gradient_step_is_skipped():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIAdam([param])

    def closure(backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
        return param.detach().sum() * 0.0

    diagnostics = optimizer.step(closure)

    assert diagnostics.accepted is False
    assert diagnostics.skipped_reason == "no gradients available"


def test_rejection_backtracks_and_restores_parameters():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIAdam(
        [param],
        alpha0=8.0,
        alpha_max=10.0,
        reject_bad_steps=True,
        max_backtracks=2,
        backtrack_shrink=0.5,
        fallback_to_gradient=False,
        use_rho_ema=False,
    )

    def closure(backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * param.pow(2).sum()
        if backward:
            loss.backward()
        return loss

    diagnostics = optimizer.step(closure)

    assert diagnostics.accepted is False
    assert diagnostics.backtracks == 2
    assert torch.allclose(param.detach(), torch.tensor([1.0]))
    assert diagnostics.rho is not None and diagnostics.rho <= 0.0


def test_trust_region_expansion_can_raise_alpha_factor():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIAdam(
        [param],
        alpha0=1e-5,
        alpha_min=1e-8,
        alpha_max=1.0,
        kp=0.0,
        ki=0.0,
        reject_bad_steps=False,
        trust_region_expand=True,
        trust_region_rho_threshold=0.5,
        trust_region_alpha_threshold=1e-4,
        trust_region_expand_factor=2.0,
        use_rho_ema=False,
    )

    def closure(backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * param.pow(2).sum()
        if backward:
            loss.backward()
        return loss

    diagnostics = optimizer.step(closure)

    assert diagnostics.trust_region_expanded is True
    assert diagnostics.alpha_update_factor is not None
    assert math.isclose(diagnostics.alpha_update_factor, 2.0, rel_tol=1e-6)


def test_weight_decay_matches_torch_adamw_one_step_when_alpha_is_fixed():
    ours = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(ours.detach().clone())
    grad = torch.tensor([0.25, -0.5])

    optimizer = PIAdam(
        [ours],
        alpha0=1e-3,
        alpha_min=1e-3,
        alpha_max=1e-3,
        weight_decay=0.1,
        kp=0.0,
        ki=0.0,
        reject_bad_steps=False,
        use_rho_ema=False,
        fallback_to_gradient=False,
    )
    torch_optimizer = torch.optim.AdamW([reference], lr=1e-3, weight_decay=0.1)

    def closure(backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
            ours.grad = grad.clone()
        return (ours.detach() * grad).sum()

    optimizer.step(closure)
    reference.grad = grad.clone()
    torch_optimizer.step()

    assert torch.allclose(ours.detach(), reference.detach(), atol=1e-7)
