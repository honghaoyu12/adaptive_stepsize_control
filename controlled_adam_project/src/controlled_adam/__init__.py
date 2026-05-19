"""Vanilla Adam and outer-loop controlled Adam demos."""

from controlled_adam.objectives import (
    Ackley,
    AnisotropicQuadratic,
    Beale,
    Easom,
    GoldsteinPrice,
    Himmelblau,
    Objective,
    Rastrigin,
    Rosenbrock,
    SixHumpCamel,
)
from controlled_adam.optimizers import OptimizationHistory, controlled_adam, vanilla_adam
from controlled_adam.torch_optimizers import ControlledAdamStep, TorchControlledAdam

__all__ = [
    "Objective",
    "Ackley",
    "AnisotropicQuadratic",
    "Beale",
    "Easom",
    "GoldsteinPrice",
    "Himmelblau",
    "Rastrigin",
    "Rosenbrock",
    "SixHumpCamel",
    "OptimizationHistory",
    "ControlledAdamStep",
    "TorchControlledAdam",
    "vanilla_adam",
    "controlled_adam",
]
