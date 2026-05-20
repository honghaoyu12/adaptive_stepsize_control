"""Objective functions for controlled Muon comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MatrixQuadraticObjective:
    """An anisotropic matrix quadratic objective."""

    target: np.ndarray
    curvature: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=float)
        curvature = np.asarray(self.curvature, dtype=float)

        if target.ndim != 2:
            raise ValueError("target must be a 2D matrix.")
        if curvature.shape != target.shape:
            raise ValueError("curvature must have the same shape as target.")
        if np.any(curvature <= 0):
            raise ValueError("all curvature entries must be positive.")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "curvature", curvature)

    @classmethod
    def random_anisotropic(
        cls,
        shape: tuple[int, int] = (12, 8),
        seed: int = 7,
        curvature_min: float = 0.2,
        curvature_max: float = 60.0,
    ) -> "MatrixQuadraticObjective":
        rng = np.random.default_rng(seed)
        target = rng.normal(size=shape)
        values = np.geomspace(curvature_min, curvature_max, num=shape[0] * shape[1])
        curvature = values.reshape(shape)
        flat = curvature.ravel()
        rng.shuffle(flat)
        curvature = flat.reshape(shape)
        return cls(target=target, curvature=curvature)

    def value(self, W: np.ndarray) -> float:
        W = np.asarray(W, dtype=float)
        diff = W - self.target
        return float(0.5 * np.sum(self.curvature * diff * diff))

    def gradient(self, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        return self.curvature * (W - self.target)

    def distance_to_target(self, W: np.ndarray) -> float:
        W = np.asarray(W, dtype=float)
        return float(np.linalg.norm(W - self.target, ord="fro"))


class Objective(Protocol):
    """Protocol for deterministic objectives."""

    name: str

    def value(self, x: np.ndarray) -> float:
        """Return f(x)."""

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Return grad f(x)."""


@dataclass(frozen=True)
class AnisotropicQuadratic:
    curvature_x: float = 50.0
    curvature_y: float = 1.0
    name: str = "quadratic"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(
            0.5 * (self.curvature_x * x[0] ** 2 + self.curvature_y * x[1] ** 2)
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.array(
            [self.curvature_x * x[0], self.curvature_y * x[1]],
            dtype=float,
        )

    @property
    def global_minima(self) -> np.ndarray:
        return np.zeros((1, 2))


@dataclass(frozen=True)
class Rosenbrock:
    a: float = 1.0
    b: float = 100.0
    name: str = "rosenbrock"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        return float((self.a - x0) ** 2 + self.b * (x1 - x0**2) ** 2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        dfdx = -2.0 * (self.a - x0) - 4.0 * self.b * x0 * (x1 - x0**2)
        dfdy = 2.0 * self.b * (x1 - x0**2)
        return np.array([dfdx, dfdy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array([[self.a, self.a**2]])


@dataclass(frozen=True)
class Himmelblau:
    name: str = "himmelblau"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        u = x[0] ** 2 + x[1] - 11.0
        v = x[0] + x[1] ** 2 - 7.0
        return float(u**2 + v**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = x[0] ** 2 + x[1] - 11.0
        v = x[0] + x[1] ** 2 - 7.0
        dfdx = 4.0 * x[0] * u + 2.0 * v
        dfdy = 2.0 * u + 4.0 * x[1] * v
        return np.array([dfdx, dfdy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )


@dataclass(frozen=True)
class Rastrigin:
    amplitude: float = 10.0
    name: str = "rastrigin"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(
            self.amplitude * x.size
            + np.sum(x**2 - self.amplitude * np.cos(2.0 * np.pi * x))
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 2.0 * x + 2.0 * np.pi * self.amplitude * np.sin(2.0 * np.pi * x)

    @property
    def global_minima(self) -> np.ndarray:
        return np.zeros((1, 2))


@dataclass(frozen=True)
class Beale:
    name: str = "beale"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        term1 = 1.5 - x[0] + x[0] * x[1]
        term2 = 2.25 - x[0] + x[0] * x[1] ** 2
        term3 = 2.625 - x[0] + x[0] * x[1] ** 3
        return float(term1**2 + term2**2 + term3**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        term1 = 1.5 - x[0] + x[0] * x[1]
        term2 = 2.25 - x[0] + x[0] * x[1] ** 2
        term3 = 2.625 - x[0] + x[0] * x[1] ** 3
        dfdx = (
            2.0 * term1 * (-1.0 + x[1])
            + 2.0 * term2 * (-1.0 + x[1] ** 2)
            + 2.0 * term3 * (-1.0 + x[1] ** 3)
        )
        dfdy = (
            2.0 * term1 * x[0]
            + 4.0 * term2 * x[0] * x[1]
            + 6.0 * term3 * x[0] * x[1] ** 2
        )
        return np.array([dfdx, dfdy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array([[3.0, 0.5]])


@dataclass(frozen=True)
class Ackley:
    a: float = 20.0
    b: float = 0.2
    c: float = 2.0 * np.pi
    name: str = "ackley"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        mean_square = 0.5 * np.dot(x, x)
        mean_cos = 0.5 * np.sum(np.cos(self.c * x))
        return float(
            -self.a * np.exp(-self.b * np.sqrt(mean_square))
            - np.exp(mean_cos)
            + self.a
            + np.e
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mean_square = 0.5 * np.dot(x, x)
        root_mean_square = np.sqrt(mean_square)
        first = np.zeros_like(x)
        if root_mean_square > 0.0:
            first = (
                self.a
                * self.b
                * np.exp(-self.b * root_mean_square)
                * x
                / (2.0 * root_mean_square)
            )
        mean_cos = 0.5 * np.sum(np.cos(self.c * x))
        second = 0.5 * self.c * np.exp(mean_cos) * np.sin(self.c * x)
        return first + second

    @property
    def global_minima(self) -> np.ndarray:
        return np.zeros((1, 2))


@dataclass(frozen=True)
class SixHumpCamel:
    name: str = "six_hump_camel"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        return float(
            (4.0 - 2.1 * x0**2 + (x0**4) / 3.0) * x0**2
            + x0 * x1
            + (-4.0 + 4.0 * x1**2) * x1**2
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        dfdx = 8.0 * x0 - 8.4 * x0**3 + 2.0 * x0**5 + x1
        dfdy = x0 - 8.0 * x1 + 16.0 * x1**3
        return np.array([dfdx, dfdy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array([[0.089842, -0.712656], [-0.089842, 0.712656]])


@dataclass(frozen=True)
class GoldsteinPrice:
    name: str = "goldstein_price"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        s = x0 + x1 + 1.0
        q = 2.0 * x0 - 3.0 * x1
        a = 1.0 + s**2 * (
            19.0 - 14.0 * x0 + 3.0 * x0**2 - 14.0 * x1
            + 6.0 * x0 * x1 + 3.0 * x1**2
        )
        b = 30.0 + q**2 * (
            18.0 - 32.0 * x0 + 12.0 * x0**2 + 48.0 * x1
            - 36.0 * x0 * x1 + 27.0 * x1**2
        )
        return float(a * b)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        s = x0 + x1 + 1.0
        q = 2.0 * x0 - 3.0 * x1
        c = (
            19.0 - 14.0 * x0 + 3.0 * x0**2 - 14.0 * x1
            + 6.0 * x0 * x1 + 3.0 * x1**2
        )
        d = (
            18.0 - 32.0 * x0 + 12.0 * x0**2 + 48.0 * x1
            - 36.0 * x0 * x1 + 27.0 * x1**2
        )
        a = 1.0 + s**2 * c
        b = 30.0 + q**2 * d
        dc_dx = -14.0 + 6.0 * x0 + 6.0 * x1
        dc_dy = -14.0 + 6.0 * x0 + 6.0 * x1
        dd_dx = -32.0 + 24.0 * x0 - 36.0 * x1
        dd_dy = 48.0 - 36.0 * x0 + 54.0 * x1
        da_dx = 2.0 * s * c + s**2 * dc_dx
        da_dy = 2.0 * s * c + s**2 * dc_dy
        db_dx = 4.0 * q * d + q**2 * dd_dx
        db_dy = -6.0 * q * d + q**2 * dd_dy
        return np.array([da_dx * b + a * db_dx, da_dy * b + a * db_dy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array([[0.0, -1.0]])


@dataclass(frozen=True)
class Easom:
    name: str = "easom"

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        radius = (x0 - np.pi) ** 2 + (x1 - np.pi) ** 2
        return float(-np.cos(x0) * np.cos(x1) * np.exp(-radius))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[0], x[1]
        radius = (x0 - np.pi) ** 2 + (x1 - np.pi) ** 2
        exp_term = np.exp(-radius)
        dfdx = exp_term * np.cos(x1) * (
            np.sin(x0) + 2.0 * (x0 - np.pi) * np.cos(x0)
        )
        dfdy = exp_term * np.cos(x0) * (
            np.sin(x1) + 2.0 * (x1 - np.pi) * np.cos(x1)
        )
        return np.array([dfdx, dfdy], dtype=float)

    @property
    def global_minima(self) -> np.ndarray:
        return np.array([[np.pi, np.pi]])
