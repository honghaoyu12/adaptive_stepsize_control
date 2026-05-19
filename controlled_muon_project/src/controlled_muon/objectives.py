"""Matrix-valued toy objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixQuadraticObjective:
    """An anisotropic matrix quadratic objective.

    The objective is

        f(W) = 1/2 * sum_ij curvature_ij * (W_ij - target_ij)^2.

    This is intentionally simple, but it is matrix-valued, so it is a useful
    toy problem for optimizers such as Muon that operate on matrix-shaped
    updates.
    """

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
        """Create a reproducible anisotropic matrix quadratic."""
        rng = np.random.default_rng(seed)
        target = rng.normal(size=shape)

        # Smoothly varying positive curvature across matrix entries.
        values = np.geomspace(curvature_min, curvature_max, num=shape[0] * shape[1])
        curvature = values.reshape(shape)

        # Shuffle so the high-curvature entries are not all in one corner.
        flat = curvature.ravel()
        rng.shuffle(flat)
        curvature = flat.reshape(shape)

        return cls(target=target, curvature=curvature)

    def value(self, W: np.ndarray) -> float:
        """Evaluate f(W)."""
        W = np.asarray(W, dtype=float)
        diff = W - self.target
        return float(0.5 * np.sum(self.curvature * diff * diff))

    def gradient(self, W: np.ndarray) -> np.ndarray:
        """Evaluate grad f(W)."""
        W = np.asarray(W, dtype=float)
        return self.curvature * (W - self.target)

    def distance_to_target(self, W: np.ndarray) -> float:
        """Return Frobenius distance to the minimizer."""
        W = np.asarray(W, dtype=float)
        return float(np.linalg.norm(W - self.target, ord="fro"))
