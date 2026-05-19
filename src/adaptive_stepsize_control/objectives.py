"""Objective functions used in the demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ObjectiveFunction(Protocol):
    """Interface required by the optimizers."""

    name: str

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""


@dataclass(frozen=True)
class QuadraticObjective:
    """A positive-definite quadratic objective.

    The objective is

        f(x) = 1/2 x^T A x.

    Parameters
    ----------
    A:
        Symmetric positive-definite matrix.
    """

    A: np.ndarray
    name: str = "anisotropic_quadratic"

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")
        if not np.allclose(A, A.T):
            raise ValueError("A must be symmetric.")
        if np.any(np.linalg.eigvalsh(A) <= 0):
            raise ValueError("A must be positive definite.")
        object.__setattr__(self, "A", A)

    @classmethod
    def anisotropic_2d(cls) -> "QuadraticObjective":
        """Return f(x, y) = 1/2 * (50 x^2 + y^2)."""
        return cls(A=np.diag([50.0, 1.0]))

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        x = np.asarray(x, dtype=float)
        return float(0.5 * x.T @ self.A @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""
        x = np.asarray(x, dtype=float)
        return self.A @ x

    @property
    def global_minima(self) -> np.ndarray:
        """Known global minimizer locations."""
        return np.zeros((1, self.A.shape[0]))


@dataclass(frozen=True)
class RosenbrockObjective:
    """The Rosenbrock banana function.

    f(x, y) = (a - x)^2 + b (y - x^2)^2.
    """

    a: float = 1.0
    b: float = 100.0
    name: str = "rosenbrock"

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        x = np.asarray(x, dtype=float)
        return float((self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""
        x = np.asarray(x, dtype=float)
        dx = -2.0 * (self.a - x[0]) - 4.0 * self.b * x[0] * (x[1] - x[0] ** 2)
        dy = 2.0 * self.b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    @property
    def global_minima(self) -> np.ndarray:
        """Known global minimizer locations."""
        return np.array([[self.a, self.a**2]])


@dataclass(frozen=True)
class HimmelblauObjective:
    """Himmelblau's function with four equivalent minima."""

    name: str = "himmelblau"

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        x = np.asarray(x, dtype=float)
        u = x[0] ** 2 + x[1] - 11.0
        v = x[0] + x[1] ** 2 - 7.0
        return float(u**2 + v**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""
        x = np.asarray(x, dtype=float)
        u = x[0] ** 2 + x[1] - 11.0
        v = x[0] + x[1] ** 2 - 7.0
        dx = 4.0 * x[0] * u + 2.0 * v
        dy = 2.0 * u + 4.0 * x[1] * v
        return np.array([dx, dy])

    @property
    def global_minima(self) -> np.ndarray:
        """Known global minimizer locations."""
        return np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )


@dataclass(frozen=True)
class RastriginObjective:
    """Two-dimensional Rastrigin function with many local wells."""

    amplitude: float = 10.0
    name: str = "rastrigin"

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        x = np.asarray(x, dtype=float)
        n = x.size
        return float(
            self.amplitude * n
            + np.sum(x**2 - self.amplitude * np.cos(2.0 * np.pi * x))
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""
        x = np.asarray(x, dtype=float)
        return 2.0 * x + 2.0 * np.pi * self.amplitude * np.sin(2.0 * np.pi * x)

    @property
    def global_minima(self) -> np.ndarray:
        """Known global minimizer locations."""
        return np.zeros((1, 2))


@dataclass(frozen=True)
class BealeObjective:
    """Beale function with curved valleys and a sharp minimum."""

    name: str = "beale"

    def value(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        x = np.asarray(x, dtype=float)
        term1 = 1.5 - x[0] + x[0] * x[1]
        term2 = 2.25 - x[0] + x[0] * x[1] ** 2
        term3 = 2.625 - x[0] + x[0] * x[1] ** 3
        return float(term1**2 + term2**2 + term3**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate grad f(x)."""
        x = np.asarray(x, dtype=float)
        term1 = 1.5 - x[0] + x[0] * x[1]
        term2 = 2.25 - x[0] + x[0] * x[1] ** 2
        term3 = 2.625 - x[0] + x[0] * x[1] ** 3
        dx = (
            2.0 * term1 * (-1.0 + x[1])
            + 2.0 * term2 * (-1.0 + x[1] ** 2)
            + 2.0 * term3 * (-1.0 + x[1] ** 3)
        )
        dy = (
            2.0 * term1 * x[0]
            + 4.0 * term2 * x[0] * x[1]
            + 6.0 * term3 * x[0] * x[1] ** 2
        )
        return np.array([dx, dy])

    @property
    def global_minima(self) -> np.ndarray:
        """Known global minimizer locations."""
        return np.array([[3.0, 0.5]])
