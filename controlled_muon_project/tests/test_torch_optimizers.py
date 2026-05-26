"""Tests for PyTorch Muon optimizer utilities."""

import math

import torch
import torch.nn as nn

from controlled_muon.torch_optimizers import MuonConfig, TorchControlledMuon, default_muon_param_groups


def test_default_muon_param_groups_use_adamw_for_biases_and_batchnorm() -> None:
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Conv2d(1, 2, kernel_size=3),
        nn.BatchNorm1d(3),
        nn.Linear(3, 2, bias=False),
    )

    muon_params, adamw_params = default_muon_param_groups(model.named_parameters())

    names_by_param = {param: name for name, param in model.named_parameters()}
    muon_names = {names_by_param[param] for param in muon_params}
    adamw_names = {names_by_param[param] for param in adamw_params}

    assert muon_names == {"0.weight", "3.weight"}
    assert adamw_names == {"0.bias", "1.weight", "1.bias", "2.weight", "2.bias"}


def test_torch_controlled_muon_splits_matrix_and_vector_params() -> None:
    model = nn.Linear(4, 2)
    muon_params, adamw_params = default_muon_param_groups(model.named_parameters())

    optimizer = TorchControlledMuon((muon_params, adamw_params))

    assert set(optimizer.muon_params) == {model.weight}
    assert set(optimizer.adamw_params) == {model.bias}
    assert model.weight in optimizer.momentum
    assert model.bias in optimizer.adam_state


def test_torch_controlled_muon_uses_official_momentum_update() -> None:
    param = nn.Parameter(torch.eye(2))
    optimizer = TorchControlledMuon(([param], []), alpha0=1e-2)
    grad = torch.ones_like(param)
    direction, new_momentum = optimizer._direction(grad, torch.zeros_like(param))

    assert torch.allclose(new_momentum, torch.full_like(param, 0.05))
    assert direction.shape == param.shape


def test_torch_controlled_muon_shape_scaling_matches_official_original_adjustment() -> None:
    param = nn.Parameter(torch.zeros(4, 2))
    grad = torch.ones_like(param)
    scaled = TorchControlledMuon(([param], []), config=MuonConfig(shape_scale=True))
    unscaled = TorchControlledMuon(([param], []), config=MuonConfig(shape_scale=False))

    direction_scaled, _ = scaled._direction(grad, torch.zeros_like(param))
    direction_unscaled, _ = unscaled._direction(grad, torch.zeros_like(param))

    torch.testing.assert_close(direction_scaled, direction_unscaled * math.sqrt(2.0), atol=1e-6, rtol=1e-6)
