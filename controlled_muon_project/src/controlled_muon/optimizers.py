"""Vanilla Muon and outer-loop controlled Muon implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from controlled_muon.objectives import MatrixQuadraticObjective
from controlled_muon.orthogonalization import orthogonalize


@dataclass(frozen=True)
class MuonConfig:
    """Configuration for the educational Muon implementation."""

    momentum: float = 0.95
    nesterov: bool = True
    orthogonalizer: str = "newton_schulz"
    ns_steps: int = 8
    update_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1.")
        if self.ns_steps <= 0:
            raise ValueError("ns_steps must be positive.")
        if self.update_scale <= 0:
            raise ValueError("update_scale must be positive.")


@dataclass
class OptimizationHistory:
    """Stores trajectories and diagnostics."""

    Ws: np.ndarray
    fs: np.ndarray
    distances: np.ndarray
    step_sizes: np.ndarray
    rhos: Optional[np.ndarray] = None
    predicted_decreases: Optional[np.ndarray] = None
    actual_decreases: Optional[np.ndarray] = None
    directional_derivatives: Optional[np.ndarray] = None
    accepted: Optional[np.ndarray] = None


def _muon_direction(
    gradient: np.ndarray,
    momentum_buffer: np.ndarray,
    config: MuonConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Muon's proposed descent direction and updated momentum buffer."""
    new_momentum = config.momentum * momentum_buffer + gradient

    if config.nesterov:
        matrix_to_orthogonalize = config.momentum * new_momentum + gradient
    else:
        matrix_to_orthogonalize = new_momentum

    ortho_update = orthogonalize(
        matrix_to_orthogonalize,
        method=config.orthogonalizer,
        ns_steps=config.ns_steps,
    )

    direction = -config.update_scale * ortho_update
    return direction, new_momentum


def vanilla_muon(
    objective: MatrixQuadraticObjective,
    W0: np.ndarray,
    eta: float,
    steps: int,
    config: Optional[MuonConfig] = None,
) -> OptimizationHistory:
    """Run vanilla Muon with a fixed global learning rate."""
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    config = config or MuonConfig()
    W = np.asarray(W0, dtype=float).copy()
    momentum_buffer = np.zeros_like(W)

    Ws = []
    fs = []
    distances = []
    step_sizes = []

    for _ in range(steps):
        G = objective.gradient(W)
        P, momentum_buffer = _muon_direction(G, momentum_buffer, config)

        W = W + eta * P

        Ws.append(W.copy())
        fs.append(objective.value(W))
        distances.append(objective.distance_to_target(W))
        step_sizes.append(eta)

    return OptimizationHistory(
        Ws=np.asarray(Ws),
        fs=np.asarray(fs),
        distances=np.asarray(distances),
        step_sizes=np.asarray(step_sizes),
    )


def controlled_muon(
    objective: MatrixQuadraticObjective,
    W0: np.ndarray,
    alpha0: float,
    steps: int,
    config: Optional[MuonConfig] = None,
    kp: float = 0.4,
    rho_star: float = 0.7,
    rho_min: float = 0.0,
    alpha_min: float = 1e-8,
    alpha_max: float = 1.0,
    descent_fail_shrink: float = 0.25,
    max_log_change: float = 2.0,
    rollback_state_on_reject: bool = True,
    eps: float = 1e-12,
) -> OptimizationHistory:
    """Run Muon with an outer-loop global step-size controller.

    Muon proposes a matrix-valued direction ``P``. The controller chooses the
    global multiplier ``alpha`` using the actual-over-predicted decrease ratio.
    """
    if alpha0 <= 0:
        raise ValueError("alpha0 must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if kp < 0:
        raise ValueError("kp must be non-negative.")
    if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")
    if not 0 < descent_fail_shrink < 1:
        raise ValueError("descent_fail_shrink must be between 0 and 1.")

    config = config or MuonConfig()
    W = np.asarray(W0, dtype=float).copy()
    momentum_buffer = np.zeros_like(W)
    alpha = float(np.clip(alpha0, alpha_min, alpha_max))

    Ws = []
    fs = []
    distances = []
    alphas = []
    rhos = []
    predicted_decreases = []
    actual_decreases = []
    directional_derivatives = []
    accepted = []

    for _ in range(steps):
        f_t = objective.value(W)
        G = objective.gradient(W)
        P, candidate_momentum = _muon_direction(G, momentum_buffer, config)

        directional_derivative = float(np.sum(G * P))
        predicted_decrease = -alpha * directional_derivative

        if predicted_decrease <= eps:
            # The proposed Muon direction is not descent-like under the current
            # gradient. Reject and shrink the global step size.
            rho = -np.inf
            actual_decrease = 0.0
            step_accepted = False
            alpha = max(alpha_min, alpha * descent_fail_shrink)
        else:
            W_trial = W + alpha * P
            f_trial = objective.value(W_trial)
            actual_decrease = f_t - f_trial
            rho = actual_decrease / (predicted_decrease + eps)
            step_accepted = rho > rho_min

            if step_accepted:
                W = W_trial
                momentum_buffer = candidate_momentum
            elif not rollback_state_on_reject:
                momentum_buffer = candidate_momentum

            log_change = kp * (rho - rho_star)
            log_change = float(np.clip(log_change, -max_log_change, max_log_change))
            alpha = float(np.clip(alpha * np.exp(log_change), alpha_min, alpha_max))

        Ws.append(W.copy())
        fs.append(objective.value(W))
        distances.append(objective.distance_to_target(W))
        alphas.append(alpha)
        rhos.append(rho)
        predicted_decreases.append(predicted_decrease)
        actual_decreases.append(actual_decrease)
        directional_derivatives.append(directional_derivative)
        accepted.append(step_accepted)

    return OptimizationHistory(
        Ws=np.asarray(Ws),
        fs=np.asarray(fs),
        distances=np.asarray(distances),
        step_sizes=np.asarray(alphas),
        rhos=np.asarray(rhos),
        predicted_decreases=np.asarray(predicted_decreases),
        actual_decreases=np.asarray(actual_decreases),
        directional_derivatives=np.asarray(directional_derivatives),
        accepted=np.asarray(accepted),
    )
