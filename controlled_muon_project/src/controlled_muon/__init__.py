"""Controlled Muon demo package."""

from controlled_muon.objectives import MatrixQuadraticObjective
from controlled_muon.optimizers import (
    MuonConfig,
    OptimizationHistory,
    controlled_muon,
    vanilla_muon,
)
from controlled_muon.orthogonalization import orthogonalize

__all__ = [
    "MatrixQuadraticObjective",
    "MuonConfig",
    "OptimizationHistory",
    "controlled_muon",
    "vanilla_muon",
    "orthogonalize",
]
