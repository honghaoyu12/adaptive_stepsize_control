"""Tests for Muon optimizers."""

import numpy as np

from controlled_muon.objectives import MatrixQuadraticObjective
from controlled_muon.optimizers import MuonConfig, controlled_muon, vanilla_muon
from controlled_muon.orthogonalization import orthogonalize


def test_svd_orthogonalization_shape() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    Q = orthogonalize(A, method="svd")

    assert Q.shape == A.shape
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)


def test_vanilla_muon_runs() -> None:
    objective = MatrixQuadraticObjective.random_anisotropic(shape=(6, 4), seed=1)
    W0 = np.zeros_like(objective.target)
    config = MuonConfig(momentum=0.8, orthogonalizer="svd")

    history = vanilla_muon(objective, W0, eta=0.02, steps=10, config=config)

    assert history.Ws.shape == (10, *W0.shape)
    assert history.fs[-1] < objective.value(W0)


def test_controlled_muon_reduces_objective() -> None:
    objective = MatrixQuadraticObjective.random_anisotropic(shape=(6, 4), seed=2)
    W0 = np.zeros_like(objective.target)
    config = MuonConfig(momentum=0.8, orthogonalizer="svd")

    history = controlled_muon(
        objective=objective,
        W0=W0,
        alpha0=0.08,
        steps=40,
        config=config,
        kp=0.4,
        rho_star=0.7,
        alpha_max=0.3,
    )

    assert history.fs[-1] < objective.value(W0)
    assert np.all(history.step_sizes > 0)
    assert history.rhos is not None
    assert history.accepted is not None
