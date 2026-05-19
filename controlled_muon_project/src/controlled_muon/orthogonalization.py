"""Orthogonalization utilities for Muon-style matrix updates."""

from __future__ import annotations

import numpy as np


def orthogonalize(
    matrix: np.ndarray,
    method: str = "newton_schulz",
    ns_steps: int = 8,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return an approximate polar factor of a 2D matrix.

    Parameters
    ----------
    matrix:
        Matrix to orthogonalize.
    method:
        Either ``"newton_schulz"`` or ``"svd"``.
    ns_steps:
        Number of Newton-Schulz polar iterations.
    eps:
        Small value for numerical stability.

    Notes
    -----
    The SVD version returns the exact polar factor ``U @ Vt``.
    The Newton-Schulz version is a compact educational implementation of the
    polar iteration

        X <- 1/2 X (3I - X^T X),

    after normalization. It is not intended to be the fastest possible GPU
    implementation.
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
        return _orthogonalize_newton_schulz(X, ns_steps=ns_steps, eps=eps)
    raise ValueError("method must be either 'newton_schulz' or 'svd'.")


def _orthogonalize_svd(X: np.ndarray) -> np.ndarray:
    """Exact polar factor via SVD."""
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def _orthogonalize_newton_schulz(
    X: np.ndarray,
    ns_steps: int,
    eps: float,
) -> np.ndarray:
    """Approximate polar factor using Newton-Schulz iteration."""
    if ns_steps <= 0:
        raise ValueError("ns_steps must be positive.")

    # The simple polar iteration below is easiest when rows >= columns.
    transposed = False
    if X.shape[0] < X.shape[1]:
        X = X.T
        transposed = True

    # Frobenius normalization keeps the largest singular value <= 1.
    Y = X / (np.linalg.norm(X, ord="fro") + eps)

    n_cols = Y.shape[1]
    I = np.eye(n_cols)
    for _ in range(ns_steps):
        Y = 0.5 * Y @ (3.0 * I - Y.T @ Y)

    if transposed:
        Y = Y.T
    return Y
