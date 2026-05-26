"""Orthogonalization utilities for Muon-style matrix updates."""

from __future__ import annotations

import numpy as np


def orthogonalize(
    matrix: np.ndarray,
    method: str = "newton_schulz",
    ns_steps: int = 5,
    eps: float = 1e-7,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> np.ndarray:
    """Return an approximate polar factor of a 2D matrix.

    Parameters
    ----------
    matrix:
        Matrix to orthogonalize.
    method:
        Either ``"newton_schulz"`` or ``"svd"``.
    ns_steps:
        Number of Newton-Schulz quintic iterations.
    eps:
        Small value for numerical stability.
    ns_coefficients:
        Quintic Newton-Schulz coefficients. The defaults match PyTorch Muon's
        implementation.

    Notes
    -----
    The SVD version returns the exact polar factor ``U @ Vt``. The
    Newton-Schulz version uses the same quintic iteration and default
    coefficients as PyTorch Muon, implemented here in NumPy for the educational
    demos.
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if np.linalg.norm(X, ord="fro") <= eps:
        return np.zeros_like(X)

    method = method.lower()
    if method == "svd":
        return _orthogonalize_svd(X)
    if method == "newton_schulz":
        return _orthogonalize_newton_schulz(
            X,
            ns_steps=ns_steps,
            eps=eps,
            coefficients=ns_coefficients,
        )
    raise ValueError("method must be either 'newton_schulz' or 'svd'.")


def _orthogonalize_svd(X: np.ndarray) -> np.ndarray:
    """Exact polar factor via SVD."""
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def _orthogonalize_newton_schulz(
    X: np.ndarray,
    ns_steps: int,
    eps: float,
    coefficients: tuple[float, float, float],
) -> np.ndarray:
    """Approximate the Muon zeroth-power update using quintic Newton-Schulz."""
    if ns_steps <= 0:
        raise ValueError("ns_steps must be positive.")
    if len(coefficients) != 3:
        raise ValueError("coefficients must contain exactly three values.")

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    Y = X / max(np.linalg.norm(X, ord="fro"), eps)
    a, b, c = coefficients

    for _ in range(ns_steps):
        A = Y @ Y.T
        B = b * A + c * (A @ A)
        Y = a * Y + B @ Y

    if transposed:
        Y = Y.T
    return Y
