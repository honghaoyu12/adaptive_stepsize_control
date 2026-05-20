"""Controlled Muon demo package."""

from controlled_muon.objectives import (
    Ackley,
    AnisotropicQuadratic,
    Beale,
    Easom,
    GoldsteinPrice,
    Himmelblau,
    Objective,
    MatrixQuadraticObjective,
    Rastrigin,
    Rosenbrock,
    SixHumpCamel,
)
from controlled_muon.optimizers import OptimizationHistory, controlled_muon, vanilla_muon
from controlled_muon.orthogonalization import orthogonalize
from controlled_muon.torch_optimizers import ControlledMuonStep, MuonConfig, TorchControlledMuon

__all__ = [
    "Ackley",
    "AnisotropicQuadratic",
    "Beale",
    "ControlledMuonStep",
    "Easom",
    "GoldsteinPrice",
    "Himmelblau",
    "MuonConfig",
    "Objective",
    "OptimizationHistory",
    "MatrixQuadraticObjective",
    "Rastrigin",
    "Rosenbrock",
    "SixHumpCamel",
    "TorchControlledMuon",
    "controlled_muon",
    "orthogonalize",
    "vanilla_muon",
]
