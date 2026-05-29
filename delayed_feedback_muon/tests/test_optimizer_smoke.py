import torch

from delayed_feedback_muon import DelayedFeedbackMuon
from delayed_feedback_muon.optimizer import _zeropower_newton_schulz


def test_newton_schulz_shape() -> None:
    x = torch.randn(7, 3)
    y = _zeropower_newton_schulz(x, steps=2)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_delayed_feedback_muon_runs_and_updates_controller() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )
    opt = DelayedFeedbackMuon(
        model.parameters(),
        lr=1e-2,
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.5,
        alpha_bounds=(0.1, 10.0),
    )
    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    criterion = torch.nn.MSELoss()

    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = criterion(model(x), y)
        losses.append(loss.item())
        loss.backward()
        opt.step(loss=loss.item())

    diag = opt.get_diagnostics()
    assert diag["alpha"] > 0
    assert diag["last_muon_tensors"] >= 1
    assert diag["last_aux_adamw_tensors"] >= 1
    assert diag["last_controller_applied"] is True
    assert torch.isfinite(torch.tensor(losses)).all()


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 2)
    opt = DelayedFeedbackMuon(model.parameters(), lr=1e-2)

    x = torch.randn(8, 3)
    y = torch.randn(8, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step(loss=loss.item())

    sd = opt.state_dict()
    opt2 = DelayedFeedbackMuon(model.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)
    assert opt2.get_diagnostics()["alpha"] == opt.get_diagnostics()["alpha"]


def test_muon_momentum_matches_torch_lerp_convention() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))
    optimizer = DelayedFeedbackMuon(
        [{"params": [parameter], "use_muon": True}],
        momentum=0.95,
    )

    direction, _ = optimizer._muon_direction(
        parameter,
        torch.ones_like(parameter),
        optimizer.param_groups[0],
    )

    assert torch.allclose(
        optimizer.state[parameter]["momentum_buffer"],
        torch.full_like(parameter, 0.05),
    )
    assert direction.shape == parameter.shape


def test_muon_core_matches_torch_muon_one_step_without_controller_feedback() -> None:
    torch.manual_seed(123)
    parameter = torch.nn.Parameter(torch.randn(5, 3))
    reference = torch.nn.Parameter(parameter.detach().clone())
    grad = torch.randn_like(parameter)

    optimizer = DelayedFeedbackMuon(
        [{"params": [parameter], "use_muon": True}],
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="original",
        alpha_init=1.0,
        fallback_to_gradient=False,
    )
    torch_optimizer = torch.optim.Muon(
        [reference],
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="original",
    )

    parameter.grad = grad.clone()
    reference.grad = grad.clone()
    optimizer.step(loss=None)
    torch_optimizer.step()

    assert torch.allclose(parameter, reference, atol=1e-4, rtol=1e-4)


def test_aux_adamw_direction_matches_torch_adamw() -> None:
    torch.manual_seed(456)
    parameter = torch.nn.Parameter(torch.randn(4))
    reference = torch.nn.Parameter(parameter.detach().clone())
    gradients = [torch.randn_like(parameter) for _ in range(3)]

    optimizer = DelayedFeedbackMuon(
        [{"params": [parameter], "use_muon": False}],
        lr=1e-3,
        weight_decay=0.01,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
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
