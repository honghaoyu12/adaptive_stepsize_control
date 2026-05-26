import math

import torch

from pi_muon import PIMuon, default_muon_param_groups


def test_alpha0_is_clipped_to_bounds():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIMuon([param], alpha0=1.0, alpha_min=1e-5, alpha_max=1e-2)

    assert math.isclose(optimizer.alpha, 1e-2)


def test_no_gradient_step_is_skipped():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIMuon([param])

    def closure(backward=True):
        optimizer.zero_grad(set_to_none=True)
        return param.detach().sum() * 0.0

    optimizer.step(closure)

    assert optimizer.last_stats is not None
    assert optimizer.last_stats.accepted is False
    assert optimizer.last_stats.skipped_reason == "no gradients available"


def test_non_descent_after_fallback_is_skipped_and_shrinks_alpha():
    param = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = PIMuon(
        [param],
        alpha0=1e-3,
        fallback_to_sgd_if_not_descent=True,
        non_descent_shrink=0.5,
    )

    def closure(backward=True):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * param.pow(2).sum()
        if backward:
            loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.last_stats is not None
    assert optimizer.last_stats.accepted is False
    assert optimizer.last_stats.skipped_reason == "non-descent direction and fallback unavailable"
    assert math.isclose(optimizer.alpha, 5e-4)


def test_rejection_backtracks_and_restores_parameters():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIMuon(
        [param],
        alpha0=16.0,
        alpha_max=20.0,
        reject_bad_steps=True,
        max_backtracks=2,
        backtrack_shrink=0.5,
        fallback_to_sgd_if_not_descent=True,
        use_rho_ema=False,
    )

    def closure(backward=True):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * param.pow(2).sum()
        if backward:
            loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.last_stats is not None
    assert optimizer.last_stats.accepted is False
    assert optimizer.last_stats.backtracks == 2
    assert torch.allclose(param.detach(), torch.tensor([1.0]))
    assert optimizer.last_stats.rho is not None and optimizer.last_stats.rho <= 0.0


def test_trust_region_expansion_can_raise_alpha_factor():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PIMuon(
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
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * param.pow(2).sum()
        if backward:
            loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.last_stats is not None
    assert optimizer.last_stats.trust_region_expanded is True
    assert optimizer.last_stats.alpha_update_factor is not None
    assert math.isclose(optimizer.last_stats.alpha_update_factor, 2.0, rel_tol=1e-6)


def test_default_param_groups_match_official_2d_muon_scope():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.Conv2d(1, 2, kernel_size=3),
        torch.nn.LayerNorm(3),
        torch.nn.Linear(3, 2, bias=False),
    )

    groups = default_muon_param_groups(model.named_parameters())
    muon_params = set(groups[0]["params"])
    adamw_params = set(groups[1]["params"])
    names_by_param = {param: name for name, param in model.named_parameters()}

    assert {names_by_param[param] for param in muon_params} == {"0.weight", "3.weight"}
    assert {names_by_param[param] for param in adamw_params} == {
        "0.bias",
        "1.weight",
        "1.bias",
        "2.weight",
        "2.bias",
    }


def test_muon_momentum_matches_torch_lerp_convention():
    param = torch.nn.Parameter(torch.eye(2))
    optimizer = PIMuon([{"params": [param], "use_muon": True}])
    state = optimizer.state[param]
    grad = torch.ones_like(param)

    direction = optimizer._build_muon_direction(param, grad, state)

    assert torch.allclose(state["momentum_buffer"], torch.full_like(param, 0.05))
    assert direction.shape == param.shape


def test_muon_shape_scaling_matches_official_original_adjustment():
    param = torch.nn.Parameter(torch.zeros(4, 2))
    grad = torch.ones_like(param)
    optimizer_scaled = PIMuon(
        [{"params": [param], "use_muon": True}],
        muon_shape_scale=True,
    )
    optimizer_unscaled = PIMuon(
        [{"params": [param], "use_muon": True}],
        muon_shape_scale=False,
    )

    direction_scaled = optimizer_scaled._build_muon_direction(param, grad, optimizer_scaled.state[param])
    direction_unscaled = optimizer_unscaled._build_muon_direction(param, grad, optimizer_unscaled.state[param])

    assert torch.allclose(direction_scaled, direction_unscaled * math.sqrt(2.0), atol=1e-6)


def test_adamw_fallback_matches_torch_adamw_one_step():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(param.detach().clone())
    grad = torch.tensor([0.25, -0.5])
    optimizer = PIMuon(
        [{"params": [param], "use_muon": False}],
        alpha0=1e-3,
        alpha_min=1e-3,
        alpha_max=1e-3,
        weight_decay=0.1,
        reject_bad_steps=False,
        kp=0.0,
        ki=0.0,
        use_rho_ema=False,
    )
    torch_optimizer = torch.optim.AdamW([reference], lr=1e-3, weight_decay=0.1)

    def closure(backward=True):
        optimizer.zero_grad(set_to_none=True)
        if backward:
            param.grad = grad.clone()
        return (param.detach() * grad).sum()

    optimizer.step(closure)
    reference.grad = grad.clone()
    torch_optimizer.step()

    assert torch.allclose(param.detach(), reference.detach(), atol=1e-7)


def test_muon_group_with_non_2d_parameter_uses_adamw_fallback_not_muon():
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = PIMuon([{"params": [param], "use_muon": True}], alpha0=1e-3)
    grad = torch.tensor([0.2, -0.3])

    param.grad = grad.clone()
    directions, _ = optimizer._build_directions()

    assert len(directions) == 1
    assert "exp_avg" in optimizer.state[param]
    assert "momentum_buffer" not in optimizer.state[param]
