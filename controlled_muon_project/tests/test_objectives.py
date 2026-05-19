"""Tests for toy objectives."""

import numpy as np

from controlled_muon.objectives import MatrixQuadraticObjective


def test_matrix_quadratic_value_and_gradient() -> None:
    target = np.array([[1.0, -2.0], [0.5, 3.0]])
    curvature = np.array([[2.0, 4.0], [1.0, 3.0]])
    objective = MatrixQuadraticObjective(target=target, curvature=curvature)

    W = np.zeros_like(target)
    diff = W - target

    expected_value = 0.5 * np.sum(curvature * diff * diff)
    expected_gradient = curvature * diff

    assert np.isclose(objective.value(W), expected_value)
    np.testing.assert_allclose(objective.gradient(W), expected_gradient)


def test_random_anisotropic_is_reproducible() -> None:
    a = MatrixQuadraticObjective.random_anisotropic(seed=123)
    b = MatrixQuadraticObjective.random_anisotropic(seed=123)

    np.testing.assert_allclose(a.target, b.target)
    np.testing.assert_allclose(a.curvature, b.curvature)
