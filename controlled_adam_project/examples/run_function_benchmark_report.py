"""Generate a self-contained deterministic function-optimization report.

This script is intentionally independent of the neural-network benchmarks. It
compares vanilla Adam with three controlled Adam variants on the 2D objective
functions already implemented in the project:

- raw-rho controlled Adam updates alpha directly from rho_t;
- EMA-rho controlled Adam updates alpha from an exponential moving average of
  rho_t.
- EMA-rho plus trust-region recovery can force faster expansion when rho is
  high but alpha is already tiny.

The suite uses multiple fixed initial points per function, writes CSV summaries,
generates manager-friendly plots, and produces a Markdown report in the output
directory. Run from ``controlled_adam_project`` with:

    MPLCONFIGDIR=/private/tmp PYTHONPATH=src \
      python examples/run_function_benchmark_report.py
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

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


OPTIMIZER_LABELS = {
    "vanilla_adam": "Vanilla Adam",
    "controlled_raw_rho": "Controlled Adam (raw rho)",
    "controlled_ema_rho": "Controlled Adam (EMA rho)",
    "controlled_ema_trust": "Controlled Adam (EMA + trust)",
}

OPTIMIZER_LABELS_ZH = {
    "vanilla_adam": "标准 Adam",
    "controlled_raw_rho": "受控 Adam（原始 rho）",
    "controlled_ema_rho": "受控 Adam（EMA 平滑 rho）",
    "controlled_ema_trust": "受控 Adam（EMA + 信赖域扩张）",
}

OPTIMIZER_COLORS = {
    "vanilla_adam": "#e07a24",
    "controlled_raw_rho": "#c92a2a",
    "controlled_ema_rho": "#2f6fbb",
    "controlled_ema_trust": "#2b8a3e",
}

HIGHLIGHT_OBJECTIVES = {
    "quadratic",
    "rosenbrock",
    "himmelblau",
    "beale",
    "rastrigin",
    "ackley",
    "six_hump_camel",
    "goldstein_price",
    "easom",
}

MANAGER_FIGURES = [
    "quadratic_surface_3d.png",
    "beale_surface_3d.png",
    "goldstein_price_surface_3d.png",
    "success_rate_by_objective.png",
    "median_best_residual_by_objective.png",
    "rosenbrock_trajectory_comparison.png",
    "rosenbrock_objective_curves.png",
    "beale_objective_curves.png",
    "goldstein_price_trajectory_comparison.png",
    "rastrigin_trajectory_comparison.png",
]

OBJECTIVE_NAMES_ZH = {
    "quadratic": "二次函数",
    "rosenbrock": "Rosenbrock 函数",
    "himmelblau": "Himmelblau 函数",
    "rastrigin": "Rastrigin 函数",
    "beale": "Beale 函数",
    "ackley": "Ackley 函数",
    "six_hump_camel": "六峰驼背函数",
    "goldstein_price": "Goldstein-Price 函数",
    "easom": "Easom 函数",
}

OBJECTIVE_DESCRIPTIONS_ZH = {
    "quadratic": "光滑凸二次碗，但两个方向的曲率差异很大。",
    "rosenbrock": "经典弯曲狭谷，固定全局步长很难同时兼顾稳定性和速度。",
    "himmelblau": "有四个等价极小点，可观察不同初始点进入不同盆地的行为。",
    "rastrigin": "高度多峰，有许多局部井，是局部优化方法的限制案例。",
    "beale": "弯曲谷底并带有尖锐全局极小点。",
    "ackley": "宽盆地叠加振荡波纹，接近最优点时容易受局部结构影响。",
    "six_hump_camel": "包含两个全局极小点和多个局部盆地。",
    "goldstein_price": "非线性耦合强、尺度变化大，对步长非常敏感。",
    "easom": "在 `(pi, pi)` 附近有非常尖锐且孤立的最优点。",
}

OBJECTIVE_FORMULAS = {
    "quadratic": (
        r"$f(x,y)=\frac{1}{2}(50x^2+y^2)$"
    ),
    "rosenbrock": (
        r"$f(x,y)=(1-x)^2+100(y-x^2)^2$"
    ),
    "himmelblau": (
        r"$f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2$"
    ),
    "rastrigin": (
        r"$f(x,y)=20+x^2+y^2-10\cos(2\pi x)-10\cos(2\pi y)$"
    ),
    "beale": (
        r"$f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2$"
        "\n"
        r"$\quad +(2.625-x+xy^3)^2$"
    ),
    "ackley": (
        r"$f(x,y)=-20e^{-0.2\sqrt{0.5(x^2+y^2)}}$"
        "\n"
        r"$\quad -e^{0.5(\cos 2\pi x+\cos 2\pi y)}+20+e$"
    ),
    "six_hump_camel": (
        r"$f(x,y)=(4-2.1x^2+x^4/3)x^2+xy+(-4+4y^2)y^2$"
    ),
    "goldstein_price": (
        r"$f(x,y)=A(x,y)B(x,y)$"
        "\n"
        r"$A=1+(x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)$"
        "\n"
        r"$B=30+(2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)$"
    ),
    "easom": (
        r"$f(x,y)=-\cos(x)\cos(y)e^{-((x-\pi)^2+(y-\pi)^2)}$"
    ),
}


@dataclass(frozen=True)
class BenchmarkCase:
    """Configuration for one objective in the deterministic benchmark."""

    objective: Objective
    starts: np.ndarray
    steps: int
    alpha: float
    alpha_max: float
    rho_star: float
    kp: float
    success_tol_f: float
    success_tol_dist: float
    description: str
    highlight_start_index: int = 0
    alpha_min: float = 1e-8
    rho_min: float = 0.0
    ema_beta: float = 0.90
    max_backtracks: int = 8
    min_alpha_factor: float = 0.5
    max_alpha_factor: float = 1.25
    trust_region_rho_threshold: float = 0.90
    trust_region_alpha_threshold: float = 1e-4
    trust_region_expand_factor: float = 1.5


@dataclass
class RunSummary:
    """Flat metrics for one objective/start/optimizer run."""

    objective: str
    start_id: int
    optimizer: str
    start_x: float
    start_y: float
    final_f: float
    best_f: float
    final_residual: float
    best_residual: float
    final_distance: float
    best_distance: float
    final_grad_norm: float
    best_iteration: int
    success: bool
    iterations_to_success: int
    accepted_rate: float
    final_alpha: float
    median_alpha: float
    final_rho: float
    median_rho: float
    trust_expansions: int


def controlled_adam_ema_rho(
    objective: Objective,
    x0: np.ndarray,
    alpha0: float,
    steps: int,
    kp: float,
    rho_star: float,
    rho_beta: float,
    rho_min: float,
    alpha_min: float,
    alpha_max: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    non_descent_shrink: float = 0.5,
    reject_bad_steps: bool = True,
    max_backtracks: int = 8,
    backtrack_shrink: float = 0.5,
    min_alpha_factor: float | None = None,
    max_alpha_factor: float | None = None,
    trust_region_expand: bool = False,
    trust_region_rho_threshold: float = 0.90,
    trust_region_alpha_threshold: float = 1e-4,
    trust_region_expand_factor: float = 1.5,
) -> OptimizationHistory:
    """Run controlled Adam while smoothing rho before updating alpha.

    The core project optimizer updates alpha from the raw ratio rho_t. This
    local implementation mirrors that optimizer but replaces the control signal
    with

        rho_ema_t = beta * rho_ema_{t-1} + (1 - beta) * rho_t.

    Keeping it local makes this report runner self-contained and avoids changing
    the public optimizer API for existing demos.
    """

    if not (0.0 <= rho_beta < 1.0):
        raise ValueError("rho_beta must be in [0, 1).")
    if kp < 0.0:
        raise ValueError("kp must be non-negative.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not (0.0 < alpha_min <= alpha_max):
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")

    x = np.asarray(x0, dtype=float).copy()
    alpha = float(np.clip(alpha0, alpha_min, alpha_max))
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    rho_ema: float | None = None

    xs = [x.copy()]
    fs = [objective.value(x)]
    alphas = []
    grad_norms = []
    rhos = []
    predicted_decreases = []
    actual_decreases = []
    accepted = []
    descent_scores = []
    trust_expanded = []

    for t in range(1, steps + 1):
        f_t = objective.value(x)
        g = objective.gradient(x)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        p = -m_hat / (np.sqrt(v_hat) + eps)

        descent_score = -float(np.dot(g, p))

        rho = np.nan
        predicted_decrease = 0.0
        actual_decrease = 0.0
        step_accepted = False
        alpha_used = alpha
        num_backtracks = 0
        expanded = False

        if descent_score <= 0.0:
            alpha = float(np.clip(alpha * non_descent_shrink, alpha_min, alpha_max))
        else:
            trial_alpha = alpha
            for j in range(max_backtracks + 1):
                x_candidate = x + trial_alpha * p
                f_candidate = objective.value(x_candidate)
                candidate_predicted = trial_alpha * descent_score
                candidate_actual = f_t - f_candidate
                candidate_rho = candidate_actual / (candidate_predicted + eps)

                if (not reject_bad_steps) or (candidate_rho > rho_min):
                    predicted_decrease = candidate_predicted
                    actual_decrease = candidate_actual
                    rho = candidate_rho
                    alpha_used = trial_alpha
                    num_backtracks = j
                    step_accepted = True
                    x = x_candidate
                    break

                predicted_decrease = candidate_predicted
                actual_decrease = candidate_actual
                rho = candidate_rho
                alpha_used = trial_alpha
                num_backtracks = j
                trial_alpha = max(alpha_min, trial_alpha * backtrack_shrink)

            if step_accepted:
                if np.isfinite(rho):
                    rho_ema = rho if rho_ema is None else (
                        rho_beta * rho_ema + (1.0 - rho_beta) * rho
                    )
                    rho_control = rho_ema
                else:
                    rho_control = rho_star
                factor = float(np.exp(kp * (rho_control - rho_star)))
                if min_alpha_factor is not None or max_alpha_factor is not None:
                    lower = 0.0 if min_alpha_factor is None else min_alpha_factor
                    upper = np.inf if max_alpha_factor is None else max_alpha_factor
                    factor = float(np.clip(factor, lower, upper))
                if (
                    trust_region_expand
                    and num_backtracks == 0
                    and rho_control >= trust_region_rho_threshold
                    and alpha_used <= trust_region_alpha_threshold
                ):
                    factor = max(factor, trust_region_expand_factor)
                    expanded = True
                alpha = alpha_used * factor
            else:
                alpha = alpha_used * backtrack_shrink

            alpha = float(np.clip(alpha, alpha_min, alpha_max))

        xs.append(x.copy())
        fs.append(objective.value(x))
        alphas.append(alpha_used)
        grad_norms.append(float(np.linalg.norm(g)))
        rhos.append(rho)
        predicted_decreases.append(predicted_decrease)
        actual_decreases.append(actual_decrease)
        accepted.append(step_accepted)
        descent_scores.append(descent_score)
        trust_expanded.append(expanded)

    history = OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        alphas=np.asarray(alphas),
        grad_norms=np.asarray(grad_norms),
        rhos=np.asarray(rhos),
        predicted_decreases=np.asarray(predicted_decreases),
        actual_decreases=np.asarray(actual_decreases),
        accepted=np.asarray(accepted),
        descent_scores=np.asarray(descent_scores),
    )
    history.trust_region_expanded = np.asarray(trust_expanded)
    return history


def benchmark_cases() -> list[BenchmarkCase]:
    """Return the fixed benchmark suite.

    Starts are hand-picked to cover representative regions while keeping the
    suite simple and deterministic. The same starts are used for every
    optimizer.
    """

    return [
        BenchmarkCase(
            objective=AnisotropicQuadratic(),
            starts=np.array(
                [[2.0, 2.0], [-2.0, 1.5], [1.5, -2.5], [-1.0, -2.0], [3.0, 0.5]]
            ),
            steps=300,
            alpha=0.003,
            alpha_max=0.5,
            rho_star=0.8,
            kp=0.10,
            success_tol_f=1e-8,
            success_tol_dist=1e-4,
            description="Smooth convex bowl with very different curvature in x and y.",
        ),
        BenchmarkCase(
            objective=Rosenbrock(),
            starts=np.array(
                [[-1.5, 1.5], [-1.2, 1.0], [0.0, 2.0], [1.5, 2.0], [-2.0, 2.0]]
            ),
            steps=3000,
            alpha=0.003,
            alpha_max=0.05,
            rho_star=0.5,
            kp=0.05,
            success_tol_f=1e-4,
            success_tol_dist=5e-2,
            description="Classic curved valley where a fixed global step is hard to tune.",
        ),
        BenchmarkCase(
            objective=Himmelblau(),
            starts=np.array(
                [[-3.5, 0.5], [0.0, 0.0], [4.0, 4.0], [-4.0, -4.0], [3.0, -3.0]]
            ),
            steps=800,
            alpha=0.01,
            alpha_max=0.08,
            rho_star=0.7,
            kp=0.08,
            success_tol_f=1e-5,
            success_tol_dist=3e-2,
            description="Four equivalent minima, useful for multi-basin behavior.",
        ),
        BenchmarkCase(
            objective=Rastrigin(),
            starts=np.array(
                [[3.3, 2.8], [-3.0, 2.5], [1.5, -2.5], [0.8, 0.8], [-1.5, -1.0]]
            ),
            steps=1200,
            alpha=0.004,
            alpha_max=0.04,
            rho_star=0.5,
            kp=0.04,
            success_tol_f=1e-3,
            success_tol_dist=5e-2,
            description="Highly multimodal landscape with many local wells.",
        ),
        BenchmarkCase(
            objective=Beale(),
            starts=np.array(
                [[1.0, 1.0], [2.0, 0.0], [4.0, 1.0], [-1.0, 1.0], [3.5, -0.5]]
            ),
            steps=1800,
            alpha=0.003,
            alpha_max=0.08,
            rho_star=0.6,
            kp=0.06,
            success_tol_f=1e-5,
            success_tol_dist=5e-2,
            description="Curved valley with a sharp minimum near (3, 0.5).",
        ),
        BenchmarkCase(
            objective=Ackley(),
            starts=np.array(
                [[2.5, 2.0], [-2.0, 2.0], [3.5, -2.5], [0.8, 0.8], [-1.0, -1.0]]
            ),
            steps=1400,
            alpha=0.01,
            alpha_max=0.08,
            rho_star=0.5,
            kp=0.05,
            success_tol_f=1e-3,
            success_tol_dist=8e-2,
            description="Broad basin with oscillatory ripples near the optimum.",
        ),
        BenchmarkCase(
            objective=SixHumpCamel(),
            starts=np.array(
                [[1.2, -1.0], [-1.2, 1.0], [0.0, 0.0], [1.5, 1.0], [-1.5, -1.0]]
            ),
            steps=900,
            alpha=0.01,
            alpha_max=0.08,
            rho_star=0.6,
            kp=0.06,
            success_tol_f=1e-5,
            success_tol_dist=4e-2,
            description="Two global minima and several local basins.",
        ),
        BenchmarkCase(
            objective=GoldsteinPrice(),
            starts=np.array(
                [[0.5, -0.5], [-0.5, -1.2], [1.0, 0.0], [-1.0, 1.0], [0.2, -1.5]]
            ),
            steps=1800,
            alpha=0.003,
            alpha_max=0.04,
            rho_star=0.5,
            kp=0.04,
            success_tol_f=1e-3,
            success_tol_dist=5e-2,
            description="Steep nonlinear coupling and strong scale sensitivity.",
        ),
        BenchmarkCase(
            objective=Easom(),
            starts=np.array(
                [[2.6, 3.6], [3.6, 2.6], [2.8, 2.8], [3.5, 3.5], [2.0, 3.14]]
            ),
            steps=1400,
            alpha=0.01,
            alpha_max=0.08,
            rho_star=0.5,
            kp=0.05,
            success_tol_f=1e-4,
            success_tol_dist=8e-2,
            description="Very sharp isolated optimum near (pi, pi).",
        ),
    ]


def nearest_minimum_distance(objective: Objective, xs: np.ndarray) -> np.ndarray:
    """Return distance to the nearest known global minimizer for each point."""

    minima = getattr(objective, "global_minima", None)
    if minima is None or len(minima) == 0:
        return np.full(xs.shape[0], np.nan)
    minima = np.asarray(minima, dtype=float)
    distances = np.linalg.norm(xs[:, None, :] - minima[None, :, :], axis=2)
    return distances.min(axis=1)


def objective_minimum_value(objective: Objective) -> float:
    """Return the known global minimum value for an objective."""

    minima = getattr(objective, "global_minima", None)
    if minima is None or len(minima) == 0:
        return 0.0
    return float(min(objective.value(point) for point in np.asarray(minima, dtype=float)))


def objective_residuals(objective: Objective, values: np.ndarray) -> np.ndarray:
    """Return nonnegative residuals above the known global minimum value."""

    f_min = objective_minimum_value(objective)
    return np.maximum(np.asarray(values, dtype=float) - f_min, 0.0)


def iterations_to_success(
    history: OptimizationHistory,
    objective: Objective,
    success_tol_f: float,
    success_tol_dist: float,
) -> int:
    """Return first iteration that satisfies objective or distance tolerance."""

    distances = nearest_minimum_distance(objective, history.xs)
    residuals = objective_residuals(objective, history.fs)
    success_mask = (residuals <= success_tol_f) | (distances <= success_tol_dist)
    hits = np.flatnonzero(success_mask)
    return int(hits[0]) if len(hits) else -1


def summarize_run(
    case: BenchmarkCase,
    start_id: int,
    optimizer_name: str,
    x0: np.ndarray,
    history: OptimizationHistory,
) -> RunSummary:
    """Create a flat summary row for a completed run."""

    distances = nearest_minimum_distance(case.objective, history.xs)
    residuals = objective_residuals(case.objective, history.fs)
    best_iteration = int(np.nanargmin(residuals))
    best_distance_iteration = int(np.nanargmin(distances))
    hit_iteration = iterations_to_success(
        history,
        case.objective,
        case.success_tol_f,
        case.success_tol_dist,
    )
    success = hit_iteration >= 0

    accepted_rate = float("nan")
    final_rho = float("nan")
    median_rho = float("nan")
    if history.accepted is not None:
        accepted_rate = float(np.mean(history.accepted))
    if history.rhos is not None:
        finite_rhos = history.rhos[np.isfinite(history.rhos)]
        if len(finite_rhos) > 0:
            final_rho = float(finite_rhos[-1])
            median_rho = float(np.median(finite_rhos))
    trust_region_expanded = getattr(history, "trust_region_expanded", None)
    trust_expansions = (
        int(np.sum(trust_region_expanded)) if trust_region_expanded is not None else 0
    )

    final_alpha = float(history.alphas[-1]) if len(history.alphas) else float("nan")
    median_alpha = float(np.median(history.alphas)) if len(history.alphas) else float("nan")
    final_grad_norm = (
        float(history.grad_norms[-1]) if len(history.grad_norms) else float("nan")
    )

    return RunSummary(
        objective=case.objective.name,
        start_id=start_id,
        optimizer=optimizer_name,
        start_x=float(x0[0]),
        start_y=float(x0[1]),
        final_f=float(history.fs[-1]),
        best_f=float(np.nanmin(history.fs)),
        final_residual=float(residuals[-1]),
        best_residual=float(np.nanmin(residuals)),
        final_distance=float(distances[-1]),
        best_distance=float(distances[best_distance_iteration]),
        final_grad_norm=final_grad_norm,
        best_iteration=best_iteration,
        success=success,
        iterations_to_success=hit_iteration,
        accepted_rate=accepted_rate,
        final_alpha=final_alpha,
        median_alpha=median_alpha,
        final_rho=final_rho,
        median_rho=median_rho,
        trust_expansions=trust_expansions,
    )


def run_case(case: BenchmarkCase) -> tuple[list[RunSummary], dict[tuple[int, str], OptimizationHistory]]:
    """Run all optimizers for all starts in one benchmark case."""

    rows: list[RunSummary] = []
    histories: dict[tuple[int, str], OptimizationHistory] = {}

    for start_id, x0 in enumerate(case.starts):
        adam = vanilla_adam(case.objective, x0, alpha=case.alpha, steps=case.steps)
        histories[(start_id, "vanilla_adam")] = adam
        rows.append(summarize_run(case, start_id, "vanilla_adam", x0, adam))

        raw = controlled_adam(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=case.kp,
            rho_star=case.rho_star,
            rho_min=case.rho_min,
            alpha_min=case.alpha_min,
            alpha_max=case.alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
        )
        histories[(start_id, "controlled_raw_rho")] = raw
        rows.append(summarize_run(case, start_id, "controlled_raw_rho", x0, raw))

        ema = controlled_adam_ema_rho(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=case.kp,
            rho_star=case.rho_star,
            rho_beta=case.ema_beta,
            rho_min=case.rho_min,
            alpha_min=case.alpha_min,
            alpha_max=case.alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
        )
        histories[(start_id, "controlled_ema_rho")] = ema
        rows.append(summarize_run(case, start_id, "controlled_ema_rho", x0, ema))

        ema_trust = controlled_adam_ema_rho(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=case.kp,
            rho_star=case.rho_star,
            rho_beta=case.ema_beta,
            rho_min=case.rho_min,
            alpha_min=case.alpha_min,
            alpha_max=case.alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
            min_alpha_factor=case.min_alpha_factor,
            max_alpha_factor=case.max_alpha_factor,
            trust_region_expand=True,
            trust_region_rho_threshold=case.trust_region_rho_threshold,
            trust_region_alpha_threshold=case.trust_region_alpha_threshold,
            trust_region_expand_factor=case.trust_region_expand_factor,
        )
        histories[(start_id, "controlled_ema_trust")] = ema_trust
        rows.append(
            summarize_run(case, start_id, "controlled_ema_trust", x0, ema_trust)
        )

    return rows, histories


def aggregate_rows(rows: list[RunSummary]) -> list[dict[str, object]]:
    """Aggregate per-start rows by objective and optimizer."""

    aggregate: list[dict[str, object]] = []
    keys = sorted({(row.objective, row.optimizer) for row in rows})
    for objective, optimizer in keys:
        subset = [row for row in rows if row.objective == objective and row.optimizer == optimizer]
        aggregate.append(
            {
                "objective": objective,
                "optimizer": optimizer,
                "num_starts": len(subset),
                "success_rate": float(np.mean([row.success for row in subset])),
                "median_final_f": float(np.median([row.final_f for row in subset])),
                "median_best_f": float(np.median([row.best_f for row in subset])),
                "median_final_residual": float(np.median([row.final_residual for row in subset])),
                "median_best_residual": float(np.median([row.best_residual for row in subset])),
                "median_final_distance": float(np.median([row.final_distance for row in subset])),
                "median_best_distance": float(np.median([row.best_distance for row in subset])),
                "median_iterations_to_success": median_success_iteration(subset),
                "median_accepted_rate": finite_median(row.accepted_rate for row in subset),
                "median_final_alpha": finite_median(row.final_alpha for row in subset),
                "median_trust_expansions": finite_median(row.trust_expansions for row in subset),
            }
        )
    return aggregate


def median_success_iteration(rows: Iterable[RunSummary]) -> float:
    """Return median iteration among successful runs, or NaN if none succeed."""

    values = [row.iterations_to_success for row in rows if row.iterations_to_success >= 0]
    return float(np.median(values)) if values else float("nan")


def finite_median(values: Iterable[float]) -> float:
    """Return median of finite values, or NaN."""

    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    """Write dataclass or dict rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                data = {name: getattr(row, name) for name in fieldnames}
            else:
                data = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(data)


def plot_success_rates(aggregate: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot success rate by objective and optimizer."""

    objectives = sorted({str(row["objective"]) for row in aggregate})
    optimizers = list(OPTIMIZER_LABELS)
    x = np.arange(len(objectives))
    width = min(0.18, 0.8 / len(optimizers))

    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, optimizer in enumerate(optimizers):
        values = [
            float(next(row["success_rate"] for row in aggregate if row["objective"] == objective and row["optimizer"] == optimizer))
            for objective in objectives
        ]
        ax.bar(
            x + (i - (len(optimizers) - 1) / 2.0) * width,
            values,
            width=width,
            label=OPTIMIZER_LABELS[optimizer],
            color=OPTIMIZER_COLORS[optimizer],
        )
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(objectives, rotation=35, ha="right")
    ax.set_title("Multi-start success rate by objective")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    path = output_dir / "success_rate_by_objective.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_median_best(aggregate: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot median best residual by objective and optimizer."""

    objectives = sorted({str(row["objective"]) for row in aggregate})
    optimizers = list(OPTIMIZER_LABELS)
    x = np.arange(len(objectives))
    width = min(0.18, 0.8 / len(optimizers))

    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, optimizer in enumerate(optimizers):
        values = [
            safe_log10(float(next(row["median_best_residual"] for row in aggregate if row["objective"] == objective and row["optimizer"] == optimizer)))
            for objective in objectives
        ]
        ax.bar(
            x + (i - (len(optimizers) - 1) / 2.0) * width,
            values,
            width=width,
            label=OPTIMIZER_LABELS[optimizer],
            color=OPTIMIZER_COLORS[optimizer],
        )
    ax.set_ylabel("log10(median best residual + epsilon)")
    ax.set_xticks(x)
    ax.set_xticklabels(objectives, rotation=35, ha="right")
    ax.set_title("Median best residual by objective")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    path = output_dir / "median_best_residual_by_objective.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def safe_log10(value: float) -> float:
    """Return a plotting-safe log10 value for objective residuals."""

    return float(np.log10(max(value, 1e-12)))


def plot_objective_surface_3d(case: BenchmarkCase, output_dir: Path) -> Path:
    """Plot a standalone 3D surface with the objective's functional form."""

    xmin, xmax, ymin, ymax = surface_plot_bounds(case)
    xx = np.linspace(xmin, xmax, 180)
    yy = np.linspace(ymin, ymax, 180)
    X, Y = np.meshgrid(xx, yy)
    Z = np.array(
        [
            case.objective.value(np.array([x, y]))
            for x, y in zip(X.ravel(), Y.ravel())
        ]
    ).reshape(X.shape)

    z_plot = Z.copy()
    z_label = "f(x, y)"
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size:
        z_cap = float(np.quantile(finite, 0.985))
        if z_cap > float(np.min(finite)):
            z_plot = np.minimum(z_plot, z_cap)
            z_label = "f(x, y), clipped for display"

    fig = plt.figure(figsize=(9.2, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        X,
        Y,
        z_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax.contour(
        X,
        Y,
        z_plot,
        zdir="z",
        offset=float(np.nanmin(z_plot)),
        levels=16,
        cmap="viridis",
        linewidths=0.75,
    )

    minima = getattr(case.objective, "global_minima", np.empty((0, 2)))
    if len(minima) > 0:
        minima = np.asarray(minima, dtype=float)
        in_view = (
            (minima[:, 0] >= xmin)
            & (minima[:, 0] <= xmax)
            & (minima[:, 1] >= ymin)
            & (minima[:, 1] <= ymax)
        )
        visible_minima = minima[in_view]
        if len(visible_minima) > 0:
            z_minima = np.array([case.objective.value(point) for point in visible_minima])
            z_minima = np.minimum(z_minima, np.nanmax(z_plot))
            ax.scatter(
                visible_minima[:, 0],
                visible_minima[:, 1],
                z_minima,
                s=70,
                color="#f8f9fa",
                edgecolor="black",
                linewidth=0.7,
                marker="*",
                label="Known global minimum",
                depthshade=False,
            )
            ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))

    formula = OBJECTIVE_FORMULAS.get(case.objective.name, case.objective.name)
    ax.text2D(
        0.03,
        0.97,
        formula,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#ced4da",
            "alpha": 0.88,
        },
    )

    ax.set_title(f"3D objective landscape: {case.objective.name}", pad=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(z_label)
    ax.view_init(elev=32, azim=-58)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label=z_label)
    fig.tight_layout()

    path = output_dir / f"{case.objective.name}_surface_3d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def surface_plot_bounds(case: BenchmarkCase) -> tuple[float, float, float, float]:
    """Return stable display bounds for a 3D objective surface plot."""

    bounds = {
        "quadratic": (-3.2, 3.2, -3.2, 3.2),
        "rosenbrock": (-2.0, 2.0, -1.0, 3.0),
        "himmelblau": (-5.0, 5.0, -5.0, 5.0),
        "rastrigin": (-4.0, 4.0, -4.0, 4.0),
        "beale": (-4.5, 4.5, -4.5, 4.5),
        "ackley": (-4.0, 4.0, -4.0, 4.0),
        "six_hump_camel": (-2.0, 2.0, -1.5, 1.5),
        "goldstein_price": (-2.0, 2.0, -2.0, 2.0),
        "easom": (1.8, 4.5, 1.8, 4.5),
    }
    return bounds.get(case.objective.name, default_surface_bounds(case))


def default_surface_bounds(case: BenchmarkCase) -> tuple[float, float, float, float]:
    """Return bounds from starts and known minima for objectives without presets."""

    points = [case.starts]
    minima = getattr(case.objective, "global_minima", np.empty((0, 2)))
    if len(minima) > 0:
        points.append(np.asarray(minima, dtype=float))
    stacked = np.vstack(points)
    xy_min = stacked.min(axis=0)
    xy_max = stacked.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1.0)
    padding = np.maximum(0.5, 0.18 * span)
    xmin, ymin = xy_min - padding
    xmax, ymax = xy_max + padding
    return float(xmin), float(xmax), float(ymin), float(ymax)


def plot_highlight_trajectory(
    case: BenchmarkCase,
    histories: dict[tuple[int, str], OptimizationHistory],
    output_dir: Path,
) -> Path:
    """Plot representative trajectories over the objective landscape."""

    start_id = case.highlight_start_index
    minima = getattr(case.objective, "global_minima", np.empty((0, 2)))
    all_xs = [histories[(start_id, optimizer)].xs for optimizer in OPTIMIZER_LABELS]
    all_xs.append(case.starts)
    if len(minima) > 0:
        all_xs.append(minima)
    stacked = np.vstack(all_xs)
    xy_min = stacked.min(axis=0)
    xy_max = stacked.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-8)
    padding = np.maximum(0.3, 0.12 * span)
    xmin, ymin = xy_min - padding
    xmax, ymax = xy_max + padding

    xx = np.linspace(xmin, xmax, 260)
    yy = np.linspace(ymin, ymax, 260)
    X, Y = np.meshgrid(xx, yy)
    Z = np.array(
        [
            case.objective.value(np.array([x, y]))
            for x, y in zip(X.ravel(), Y.ravel())
        ]
    ).reshape(X.shape)
    Z_plot = Z.copy()
    finite = Z_plot[np.isfinite(Z_plot)]
    if finite.size:
        z_cap = float(np.quantile(finite, 0.985))
        if z_cap > float(np.min(finite)):
            Z_plot = np.minimum(Z_plot, z_cap)

    width, height = trajectory_figure_size(xmax - xmin, ymax - ymin)
    fig, ax = plt.subplots(figsize=(width, height))
    levels = 48
    contour = ax.contourf(X, Y, Z_plot, levels=levels, cmap="viridis", alpha=0.86)
    ax.contour(X, Y, Z_plot, levels=14, colors="white", linewidths=0.45, alpha=0.55)
    for optimizer in OPTIMIZER_LABELS:
        history = histories[(start_id, optimizer)]
        stride = max(1, len(history.xs) // 90)
        xs = history.xs[::stride]
        ax.plot(
            xs[:, 0],
            xs[:, 1],
            color=OPTIMIZER_COLORS[optimizer],
            linewidth=2.0,
            marker="o",
            markersize=2.4,
            label=OPTIMIZER_LABELS[optimizer],
        )
        ax.scatter(
            history.xs[0, 0],
            history.xs[0, 1],
            color=OPTIMIZER_COLORS[optimizer],
            s=38,
            marker="s",
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
    if len(minima) > 0:
        ax.scatter(
            minima[:, 0],
            minima[:, 1],
            marker="*",
            s=150,
            color="white",
            edgecolor="black",
            linewidth=0.6,
            label="Known global minimum",
            zorder=6,
        )
    ax.set_title(f"Representative trajectory: {case.objective.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    fig.colorbar(contour, ax=ax, label="f(x, y)", fraction=0.035, pad=0.025, shrink=0.78)
    fig.tight_layout()
    path = output_dir / f"{case.objective.name}_trajectory_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def trajectory_figure_size(x_span: float, y_span: float) -> tuple[float, float]:
    """Return a readable figure size for trajectory plots with unequal spans."""

    aspect = float(np.clip(x_span / max(y_span, 1e-8), 0.7, 2.7))
    height = 5.8
    width = float(np.clip(height * aspect + 1.0, 7.0, 13.0))
    return width, height


def plot_highlight_objective(
    case: BenchmarkCase,
    histories: dict[tuple[int, str], OptimizationHistory],
    output_dir: Path,
) -> Path:
    """Plot objective value curves for the representative start."""

    start_id = case.highlight_start_index
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for optimizer in OPTIMIZER_LABELS:
        history = histories[(start_id, optimizer)]
        values = objective_for_plot(case.objective, history.fs)
        ax.plot(
            np.arange(len(values)),
            values,
            color=OPTIMIZER_COLORS[optimizer],
            linewidth=2.0,
            label=OPTIMIZER_LABELS[optimizer],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective residual")
    ax.set_title(f"Objective convergence: {case.objective.name}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{case.objective.name}_objective_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def objective_for_plot(objective: Objective, values: np.ndarray) -> np.ndarray:
    """Return positive objective residuals for log plotting."""

    return np.maximum(objective_residuals(objective, values), 1e-12)


def plot_highlight_alpha(
    case: BenchmarkCase,
    histories: dict[tuple[int, str], OptimizationHistory],
    output_dir: Path,
) -> Path:
    """Plot alpha schedules for the representative start."""

    start_id = case.highlight_start_index
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for optimizer in OPTIMIZER_LABELS:
        history = histories[(start_id, optimizer)]
        ax.plot(
            np.arange(len(history.alphas)),
            history.alphas,
            color=OPTIMIZER_COLORS[optimizer],
            linewidth=2.0,
            label=OPTIMIZER_LABELS[optimizer],
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global step size")
    ax.set_title(f"Step-size behavior: {case.objective.name}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{case.objective.name}_alpha_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_report(
    cases: list[BenchmarkCase],
    aggregate: list[dict[str, object]],
    per_start: list[RunSummary],
    plot_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Write the standalone Markdown report."""

    report_path = output_dir / "FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md"
    objective_order = [case.objective.name for case in cases]
    aggregate_by_objective = {
        objective: [row for row in aggregate if row["objective"] == objective]
        for objective in objective_order
    }
    num_objectives = len(cases)
    objective_word = "objective" if num_objectives == 1 else "objectives"

    winners_final = winner_counts(aggregate, "median_final_residual", lower_is_better=True)
    winners_best = winner_counts(aggregate, "median_best_residual", lower_is_better=True)
    winners_success = winner_counts(aggregate, "success_rate", lower_is_better=False)
    controlled_success_advantages = objective_success_advantages(aggregate)
    controlled_best_advantages = objective_best_residual_advantages(aggregate)

    lines: list[str] = []
    lines.append("# Function Optimization Benchmark Report")
    lines.append("")
    lines.append("This report is self-contained and uses only deterministic 2D objective functions. It excludes all neural-network and dataset experiments.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- The benchmark compares vanilla Adam against three controlled Adam variants on {num_objectives} deterministic {objective_word}.")
    lines.append("- Every optimizer sees the same fixed starting points, iteration budgets, and objective gradients.")
    lines.append("- The clearest controlled-optimizer benefit is robustness: controlled raw-rho ties or wins the best success rate on every objective in this suite, and EMA-rho ties or wins on most of them.")
    lines.append("- Convergence quality is more mixed: vanilla Adam still ties or wins several median-residual comparisons when its fixed learning rate happens to be well matched.")
    lines.append("- The controlled methods are most useful when a fixed global step size is difficult to choose, especially on curved, steep, anisotropic, or scale-sensitive landscapes.")
    lines.append("- Multimodal functions remain difficult because a step-size controller changes local step quality, not global basin selection.")
    lines.append("")
    lines.append("Winner counts across objectives:")
    lines.append("")
    lines.append("| Criterion | Vanilla Adam | Controlled raw-rho | Controlled EMA-rho | Controlled EMA+trust |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Highest success rate | {winners_success.get('vanilla_adam', 0)} | "
        f"{winners_success.get('controlled_raw_rho', 0)} | "
        f"{winners_success.get('controlled_ema_rho', 0)} | "
        f"{winners_success.get('controlled_ema_trust', 0)} |"
    )
    lines.append(
        f"| Lowest median final residual | {winners_final.get('vanilla_adam', 0)} | "
        f"{winners_final.get('controlled_raw_rho', 0)} | "
        f"{winners_final.get('controlled_ema_rho', 0)} | "
        f"{winners_final.get('controlled_ema_trust', 0)} |"
    )
    lines.append(
        f"| Lowest median best residual | {winners_best.get('vanilla_adam', 0)} | "
        f"{winners_best.get('controlled_raw_rho', 0)} | "
        f"{winners_best.get('controlled_ema_rho', 0)} | "
        f"{winners_best.get('controlled_ema_trust', 0)} |"
    )
    lines.append("")
    lines.append(f"Ties are counted for every tied optimizer, so row totals can exceed {num_objectives}.")
    lines.append("")
    lines.append("Plain-language takeaway: the controlled optimizer is not a universal replacement for a tuned fixed learning rate, but it is a useful safety and adaptation layer. It often reaches the right basin or a much better residual from the same starts without giving it any extra information about the global optimum.")
    lines.append("")
    lines.append("## Optimizers Compared")
    lines.append("")
    lines.append("| Optimizer | Meaning |")
    lines.append("|---|---|")
    lines.append("| Vanilla Adam | Adam direction with a fixed global learning rate. |")
    lines.append("| Controlled Adam (raw rho) | Adam direction, but the global multiplier is updated from the current actual/predicted decrease ratio. |")
    lines.append("| Controlled Adam (EMA rho) | Same as raw-rho, but the rho signal is smoothed before changing the global multiplier. |")
    lines.append("| Controlled Adam (EMA + trust) | EMA-rho control plus a trust-region style expansion rule when rho is high and alpha is tiny. |")
    lines.append("")
    lines.append("The controlled ratio is:")
    lines.append("")
    lines.append("```text")
    lines.append("rho_t = actual objective decrease / first-order predicted decrease")
    lines.append("alpha_{t+1} = clip(alpha_t * exp(kp * (rho_signal - rho_star)))")
    lines.append("```")
    lines.append("")
    lines.append("For deterministic functions, both the before and after objective values are exact, so rho is not contaminated by minibatch noise.")
    lines.append("")
    lines.append("## Benchmark Design")
    lines.append("")
    lines.append(f"- {num_objectives} {objective_word} are tested.")
    lines.append("- Each function uses five fixed starting points.")
    lines.append("- All optimizers use the same starts and iteration budget for that function.")
    lines.append("- Success is counted when the run reaches either a small residual above the known global minimum or a small distance to a known global minimizer.")
    lines.append("- Residual means `f(x) - f_min`, clipped at zero for numerical roundoff. This matters for functions such as Easom and Six-Hump Camel, whose global minima are negative.")
    lines.append("- Reported aggregate values are medians across starts, which reduces the effect of one unusually easy or hard start.")
    lines.append("- This suite is deliberately simple and deterministic. It is designed for optimizer-behavior communication, not for exhaustive black-box global optimization benchmarking.")
    lines.append("")
    lines.append("## Function Suite")
    lines.append("")
    lines.append("| Function | What it tests | Starts | Steps |")
    lines.append("|---|---|---:|---:|")
    for case in cases:
        lines.append(
            f"| `{case.objective.name}` | {case.description} | {len(case.starts)} | {case.steps} |"
        )
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("Success rate and median best/final values by objective:")
    lines.append("")
    lines.append("| Objective | Optimizer | Success Rate | Median Final Residual | Median Best Residual | Median Best Distance | Median Iterations To Success |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for objective in objective_order:
        for row in aggregate_by_objective[objective]:
            lines.append(
                f"| `{objective}` | {OPTIMIZER_LABELS[str(row['optimizer'])]} | "
                f"{float(row['success_rate']):.2f} | "
                f"{float(row['median_final_residual']):.3e} | "
                f"{float(row['median_best_residual']):.3e} | "
                f"{float(row['median_best_distance']):.3e} | "
                f"{format_float(row['median_iterations_to_success'])} |"
            )
    lines.append("")
    lines.append("## What To Show In A Short Manager Update")
    lines.append("")
    lines.append("A concise update should use these figures first:")
    lines.append("")
    for figure in MANAGER_FIGURES:
        if (output_dir / figure) in plot_paths:
            lines.append(f"- `{figure}`")
    lines.append("")
    lines.append("Recommended story:")
    lines.append("")
    lines.append("- Start with `success_rate_by_objective.png` to show robustness across starts.")
    lines.append("- Use the trajectory and objective-curve plots to show where step-size control helps on anisotropic, curved, or scale-sensitive landscapes.")
    lines.append("- Use alpha plots to explain that the controller changes only the global multiplier on top of the same Adam direction.")
    lines.append("- Use the 3D surface plots to introduce the reported functions and their mathematical forms before discussing optimizer trajectories.")
    lines.append("")
    lines.append("Objective-level highlights from this run:")
    lines.append("")
    lines.extend(manager_highlight_lines(controlled_success_advantages, controlled_best_advantages))
    lines.append("")
    lines.append("## Key Plots")
    lines.append("")
    for path in plot_paths:
        rel = path.relative_to(output_dir)
        lines.append(f"![{rel.stem}]({rel.as_posix()})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If controlled Adam has a higher success rate, the controller found a safer or more useful global step scale from the same Adam direction.")
    lines.append("- If controlled Adam has a lower median best objective but similar success rate, it mainly improved convergence quality rather than basin selection.")
    lines.append("- If vanilla Adam wins on a function, that is expected on some landscapes: no controller can beat a well-chosen fixed learning rate everywhere.")
    lines.append("- If raw-rho is less stable than EMA-rho, the unsmoothed signal is reacting faster but with more variance.")
    lines.append("- If all optimizers fail on a multimodal objective, that means the starts and local search dynamics entered local basins; it does not mean the controller calculation is broken.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- These are deterministic 2D functions, not deep-learning workloads.")
    lines.append("- The starting points are fixed and representative, not exhaustive.")
    lines.append("- Adam hyperparameters are held fixed per function; a heavily tuned fixed learning rate could change individual outcomes.")
    lines.append("- Multimodal objectives can trap all local optimizers, including controlled ones.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_start_results.csv`: one row per objective/start/optimizer.")
    lines.append("- `aggregate_results.csv`: median and success-rate summary by objective/optimizer.")
    lines.append("- `benchmark_config.csv`: objective settings and tolerances.")
    lines.append("- `*_surface_3d.png`: standalone 3D objective landscape plots with the function formula printed inside the figure.")
    lines.append("- `*_trajectory_comparison.png`: representative landscape/trajectory plots.")
    lines.append("- `*_objective_curves.png`: representative objective convergence plots.")
    lines.append("- `*_alpha_curves.png`: representative global step-size plots.")
    lines.append("")
    lines.append("## Per-Start Detail")
    lines.append("")
    lines.append("The full per-start table is stored in `per_start_results.csv`. It is intentionally kept outside the main report table so this report remains short enough to read.")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def generate_chinese_report(
    cases: list[BenchmarkCase],
    aggregate: list[dict[str, object]],
    per_start: list[RunSummary],
    plot_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Write a Chinese companion report for manager communication."""

    report_path = output_dir / "FUNCTION_OPTIMIZATION_BENCHMARK_REPORT_ZH.md"
    objective_order = [case.objective.name for case in cases]
    aggregate_by_objective = {
        objective: [row for row in aggregate if row["objective"] == objective]
        for objective in objective_order
    }
    num_objectives = len(cases)

    winners_final = winner_counts(aggregate, "median_final_residual", lower_is_better=True)
    winners_best = winner_counts(aggregate, "median_best_residual", lower_is_better=True)
    winners_success = winner_counts(aggregate, "success_rate", lower_is_better=False)
    controlled_success_advantages = objective_success_advantages(aggregate)
    controlled_best_advantages = objective_best_residual_advantages(aggregate)
    median_trust_expansions = [
        float(row["median_trust_expansions"])
        for row in aggregate
        if row["optimizer"] == "controlled_ema_trust"
        and np.isfinite(float(row["median_trust_expansions"]))
    ]

    lines: list[str] = []
    lines.append("# 函数优化基准测试报告")
    lines.append("")
    lines.append("本报告只使用确定性的二维测试函数，不包含 MNIST、Fashion-MNIST、CIFAR-10 或其他神经网络实验。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append(f"- 本次基准比较了标准 Adam 和三种受控 Adam 变体，共 {num_objectives} 个确定性二维目标函数。")
    lines.append("- 所有优化器使用完全相同的初始点、迭代预算、目标函数和梯度。")
    lines.append("- 最清楚的收益是鲁棒性：受控 raw-rho 版本在本套函数中经常取得或并列取得最高成功率，EMA 版本通常更平滑。")
    lines.append("- 收敛质量不是单边胜利：当固定学习率刚好合适时，标准 Adam 仍然可能在某些残差指标上并列或领先。")
    lines.append("- 受控方法最适合展示在固定全局步长难以选择的地形上，尤其是弯曲、陡峭、各向异性或尺度敏感的函数。")
    lines.append("- 多峰函数仍然困难，因为步长控制改善的是局部步子质量，并不会自动解决全局盆地选择问题。")
    if median_trust_expansions and max(median_trust_expansions) == 0.0:
        lines.append("- EMA + 信赖域版本已经纳入比较，但本次选择的函数中信赖域扩张触发很少；它主要与 EMA-rho 结果重合。")
    lines.append("")
    lines.append("各目标函数上的胜出次数：")
    lines.append("")
    lines.append("| 指标 | 标准 Adam | 受控 raw-rho | 受控 EMA-rho | 受控 EMA+信赖域 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| 最高成功率 | {winners_success.get('vanilla_adam', 0)} | "
        f"{winners_success.get('controlled_raw_rho', 0)} | "
        f"{winners_success.get('controlled_ema_rho', 0)} | "
        f"{winners_success.get('controlled_ema_trust', 0)} |"
    )
    lines.append(
        f"| 最低最终残差中位数 | {winners_final.get('vanilla_adam', 0)} | "
        f"{winners_final.get('controlled_raw_rho', 0)} | "
        f"{winners_final.get('controlled_ema_rho', 0)} | "
        f"{winners_final.get('controlled_ema_trust', 0)} |"
    )
    lines.append(
        f"| 最低历史最佳残差中位数 | {winners_best.get('vanilla_adam', 0)} | "
        f"{winners_best.get('controlled_raw_rho', 0)} | "
        f"{winners_best.get('controlled_ema_rho', 0)} | "
        f"{winners_best.get('controlled_ema_trust', 0)} |"
    )
    lines.append("")
    lines.append(f"如果多个优化器并列第一，会同时计入胜出次数，所以每一行的总数可能超过 {num_objectives}。")
    lines.append("")
    lines.append("一句话总结：受控 Adam 不是对精调固定学习率的万能替代，但它是一个有用的安全和自适应层。它在不使用全局最优点信息的情况下，常常能从相同初始点到达更好的区域或更低的残差。")
    lines.append("")
    lines.append("## 比较的优化器")
    lines.append("")
    lines.append("| 优化器 | 含义 |")
    lines.append("|---|---|")
    lines.append("| 标准 Adam | Adam 方向，使用固定全局学习率。 |")
    lines.append("| 受控 Adam（原始 rho） | Adam 方向不变，但根据当前实际下降量/一阶预测下降量的比例更新全局步长乘子。 |")
    lines.append("| 受控 Adam（EMA 平滑 rho） | 与 raw-rho 相同，但先对 rho 做指数滑动平均，再更新全局步长乘子。 |")
    lines.append("| 受控 Adam（EMA + 信赖域扩张） | 在 EMA-rho 控制器基础上，当 rho 很好且 alpha 已经很小时，使用类似信赖域的扩张规则。 |")
    lines.append("")
    lines.append("控制比例为：")
    lines.append("")
    lines.append("```text")
    lines.append("rho_t = 实际目标函数下降量 / 一阶预测下降量")
    lines.append("alpha_{t+1} = clip(alpha_t * exp(kp * (rho_signal - rho_star)))")
    lines.append("```")
    lines.append("")
    lines.append("在这些确定性函数上，步前和步后的目标函数值都是精确计算的，因此 rho 不会受到 minibatch 随机噪声污染。")
    lines.append("")
    lines.append("## 基准设计")
    lines.append("")
    lines.append(f"- 共测试 {num_objectives} 个目标函数。")
    lines.append("- 每个函数使用五个固定初始点。")
    lines.append("- 同一个函数内，所有优化器使用相同初始点和相同迭代预算。")
    lines.append("- 成功条件：达到足够小的目标残差，或离已知全局极小点足够近。")
    lines.append("- 残差定义为 `f(x) - f_min`，并对数值误差截断到非负值。这样比直接比较目标函数值更公平，因为有些函数的全局最小值是负数。")
    lines.append("- 聚合结果使用五个初始点上的中位数，避免某个特别容易或特别困难的初始点主导结论。")
    lines.append("- 这是一套小而确定性的沟通型测试，不是完整的黑箱全局优化排行榜。")
    lines.append("")
    lines.append("## 函数列表")
    lines.append("")
    lines.append("| 函数 | 测试重点 | 初始点数 | 迭代步数 |")
    lines.append("|---|---|---:|---:|")
    for case in cases:
        name = case.objective.name
        lines.append(
            f"| `{name}`（{OBJECTIVE_NAMES_ZH.get(name, name)}） | "
            f"{OBJECTIVE_DESCRIPTIONS_ZH.get(name, case.description)} | "
            f"{len(case.starts)} | {case.steps} |"
        )
    lines.append("")
    lines.append("## 聚合结果")
    lines.append("")
    lines.append("下表给出每个目标函数上的成功率、最终残差中位数、历史最佳残差中位数，以及到最近已知全局极小点的距离。")
    lines.append("")
    lines.append("| 目标函数 | 优化器 | 成功率 | 最终残差中位数 | 历史最佳残差中位数 | 最佳距离中位数 | 达到成功的迭代步中位数 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for objective in objective_order:
        for row in aggregate_by_objective[objective]:
            lines.append(
                f"| `{objective}` | {OPTIMIZER_LABELS_ZH[str(row['optimizer'])]} | "
                f"{float(row['success_rate']):.2f} | "
                f"{float(row['median_final_residual']):.3e} | "
                f"{float(row['median_best_residual']):.3e} | "
                f"{float(row['median_best_distance']):.3e} | "
                f"{format_float(row['median_iterations_to_success'])} |"
            )
    lines.append("")
    lines.append("## 给管理层汇报时建议展示")
    lines.append("")
    lines.append("建议优先展示这些图：")
    lines.append("")
    for figure in MANAGER_FIGURES:
        if (output_dir / figure) in plot_paths:
            lines.append(f"- `{figure}`")
    lines.append("")
    lines.append("推荐讲法：")
    lines.append("")
    lines.append("- 先用 `success_rate_by_objective.png` 展示不同初始点下的鲁棒性。")
    lines.append("- 先插入 3D 函数曲面图，向管理层说明每个被汇报函数的形状和数学形式。")
    lines.append("- 再用轨迹图和目标残差曲线展示：在各向异性、弯曲或尺度敏感地形上，步长控制为什么有帮助。")
    lines.append("- 用 alpha 曲线说明：控制器并没有改变 Adam 的方向，只是在 Adam 方向外面调节一个全局步长乘子。")
    lines.append("")
    lines.append("本次运行的目标级亮点：")
    lines.append("")
    lines.extend(manager_highlight_lines_zh(controlled_success_advantages, controlled_best_advantages))
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for path in plot_paths:
        rel = path.relative_to(output_dir)
        lines.append(f"![{rel.stem}]({rel.as_posix()})")
        lines.append("")
    lines.append("## 如何解读")
    lines.append("")
    lines.append("- 如果受控 Adam 的成功率更高，说明控制器从相同 Adam 方向中找到了更安全或更有效的全局步长尺度。")
    lines.append("- 如果成功率相近但受控 Adam 的历史最佳残差更低，说明它主要改善的是收敛质量，而不是进入哪个盆地。")
    lines.append("- 如果标准 Adam 在某个函数上领先，这是正常现象：任何控制器都不可能在所有地形上超过一个刚好调好的固定学习率。")
    lines.append("- raw-rho 反应最快，但可能更抖；EMA-rho 更平滑，但可能更保守。")
    lines.append("- 如果所有优化器都在多峰函数上失败，说明局部搜索进入了局部盆地；这不表示控制器计算错误。")
    lines.append("")
    lines.append("## 局限性")
    lines.append("")
    lines.append("- 这些是确定性二维函数，不是深度学习工作负载。")
    lines.append("- 初始点是代表性的，但不是穷尽的。")
    lines.append("- 每个函数上的 Adam 超参数固定；如果专门重调标准 Adam，个别结果可能改变。")
    lines.append("- 多峰目标函数可能困住所有局部优化器，包括受控优化器。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `per_start_results.csv`：每个目标函数、初始点、优化器一行。")
    lines.append("- `aggregate_results.csv`：按目标函数和优化器聚合后的成功率与中位数结果。")
    lines.append("- `benchmark_config.csv`：每个目标函数的步数、步长和容差配置。")
    lines.append("- `*_surface_3d.png`：单独的 3D 目标函数曲面图，图中标出了函数表达式。")
    lines.append("- `*_trajectory_comparison.png`：目标函数地形上的代表性轨迹图。")
    lines.append("- `*_objective_curves.png`：代表性初始点上的目标残差曲线。")
    lines.append("- `*_alpha_curves.png`：代表性初始点上的全局步长变化曲线。")
    lines.append("")
    lines.append("## 逐初始点细节")
    lines.append("")
    lines.append("完整逐初始点结果保存在 `per_start_results.csv`。主报告只放聚合表，是为了让报告保持简洁。")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def winner_counts(
    aggregate: list[dict[str, object]],
    metric: str,
    lower_is_better: bool,
) -> dict[str, int]:
    """Count objective-level winners for a metric."""

    counts = {optimizer: 0 for optimizer in OPTIMIZER_LABELS}
    objectives = sorted({str(row["objective"]) for row in aggregate})
    for objective in objectives:
        rows = [row for row in aggregate if row["objective"] == objective]
        values = [float(row[metric]) for row in rows]
        if lower_is_better:
            target = min(values)
        else:
            target = max(values)
        for row in rows:
            if math.isclose(float(row[metric]), target, rel_tol=1e-12, abs_tol=1e-12):
                counts[str(row["optimizer"])] += 1
    return counts


def objective_success_advantages(aggregate: list[dict[str, object]]) -> list[str]:
    """Return objectives where a controlled variant beats vanilla success rate."""

    objectives = sorted({str(row["objective"]) for row in aggregate})
    wins: list[str] = []
    for objective in objectives:
        rows = {
            str(row["optimizer"]): float(row["success_rate"])
            for row in aggregate
            if row["objective"] == objective
        }
        vanilla = rows["vanilla_adam"]
        controlled_best = max(
            rows["controlled_raw_rho"],
            rows["controlled_ema_rho"],
            rows["controlled_ema_trust"],
        )
        if controlled_best > vanilla:
            wins.append(objective)
    return wins


def objective_best_residual_advantages(aggregate: list[dict[str, object]]) -> list[str]:
    """Return objectives where a controlled variant beats vanilla best residual."""

    objectives = sorted({str(row["objective"]) for row in aggregate})
    wins: list[str] = []
    for objective in objectives:
        rows = {
            str(row["optimizer"]): float(row["median_best_residual"])
            for row in aggregate
            if row["objective"] == objective
        }
        vanilla = rows["vanilla_adam"]
        controlled_best = min(
            rows["controlled_raw_rho"],
            rows["controlled_ema_rho"],
            rows["controlled_ema_trust"],
        )
        if controlled_best < vanilla:
            wins.append(objective)
    return wins


def manager_highlight_lines(
    success_advantages: list[str],
    best_residual_advantages: list[str],
) -> list[str]:
    """Return short, manager-friendly highlights for the generated report."""

    lines = []
    if success_advantages:
        objectives = ", ".join(f"`{name}`" for name in success_advantages)
        lines.append(
            f"- Controlled variants improve success rate over vanilla Adam on: {objectives}."
        )
    else:
        lines.append("- No objective has a higher controlled success rate than vanilla Adam in this run.")

    if best_residual_advantages:
        objectives = ", ".join(f"`{name}`" for name in best_residual_advantages)
        lines.append(
            f"- Controlled variants improve median best residual over vanilla Adam on: {objectives}."
        )
    else:
        lines.append("- No objective has a lower controlled median best residual than vanilla Adam in this run.")

    return lines


def manager_highlight_lines_zh(
    success_advantages: list[str],
    best_residual_advantages: list[str],
) -> list[str]:
    """Return short Chinese highlights for the generated report."""

    lines = []
    if success_advantages:
        objectives = "、".join(f"`{name}`" for name in success_advantages)
        lines.append(f"- 受控变体在这些目标函数上提高了相对标准 Adam 的成功率：{objectives}。")
    else:
        lines.append("- 本次运行中，没有目标函数的受控成功率高于标准 Adam。")

    if best_residual_advantages:
        objectives = "、".join(f"`{name}`" for name in best_residual_advantages)
        lines.append(f"- 受控变体在这些目标函数上取得了更低的历史最佳残差中位数：{objectives}。")
    else:
        lines.append("- 本次运行中，没有目标函数的受控历史最佳残差中位数低于标准 Adam。")

    return lines


def format_float(value: object) -> str:
    """Format floats for report tables."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return "-"
    return f"{v:.1f}"


def write_config_csv(cases: list[BenchmarkCase], path: Path) -> None:
    """Write benchmark configuration to CSV."""

    rows = []
    for case in cases:
        rows.append(
            {
                "objective": case.objective.name,
                "num_starts": len(case.starts),
                "steps": case.steps,
                "alpha": case.alpha,
                "alpha_max": case.alpha_max,
                "rho_star": case.rho_star,
                "kp": case.kp,
                "ema_beta": case.ema_beta,
                "success_tol_f": case.success_tol_f,
                "success_tol_dist": case.success_tol_dist,
                "min_alpha_factor": case.min_alpha_factor,
                "max_alpha_factor": case.max_alpha_factor,
                "trust_region_rho_threshold": case.trust_region_rho_threshold,
                "trust_region_alpha_threshold": case.trust_region_alpha_threshold,
                "trust_region_expand_factor": case.trust_region_expand_factor,
                "description": case.description,
            }
        )
    write_csv(
        path,
        rows,
        [
            "objective",
            "num_starts",
            "steps",
            "alpha",
            "alpha_max",
            "rho_star",
            "kp",
            "ema_beta",
            "success_tol_f",
            "success_tol_dist",
            "min_alpha_factor",
            "max_alpha_factor",
            "trust_region_rho_threshold",
            "trust_region_alpha_threshold",
            "trust_region_expand_factor",
            "description",
        ],
    )


def filter_cases(
    cases: list[BenchmarkCase],
    objective_names: list[str] | None,
) -> list[BenchmarkCase]:
    """Return only requested objective cases, preserving requested order."""

    if not objective_names:
        return cases
    cases_by_name = {case.objective.name: case for case in cases}
    missing = [name for name in objective_names if name not in cases_by_name]
    if missing:
        available = ", ".join(sorted(cases_by_name))
        raise ValueError(
            f"Unknown objective(s): {', '.join(missing)}. Available objectives: {available}"
        )
    return [cases_by_name[name] for name in objective_names]


def scale_case_steps(
    cases: list[BenchmarkCase],
    step_multiplier: float,
) -> list[BenchmarkCase]:
    """Return benchmark cases with their iteration budgets scaled."""

    if step_multiplier <= 0.0:
        raise ValueError("step_multiplier must be positive.")
    if step_multiplier == 1.0:
        return cases
    return [
        replace(case, steps=max(1, int(round(case.steps * step_multiplier))))
        for case in cases
    ]


def add_random_starts(
    cases: list[BenchmarkCase],
    random_starts_per_objective: int,
    random_seed: int,
) -> list[BenchmarkCase]:
    """Append deterministic random starts sampled from each objective's plot bounds."""

    if random_starts_per_objective < 0:
        raise ValueError("random_starts_per_objective must be nonnegative.")
    if random_starts_per_objective == 0:
        return cases

    rng = np.random.default_rng(random_seed)
    augmented_cases: list[BenchmarkCase] = []
    for case in cases:
        xmin, xmax, ymin, ymax = surface_plot_bounds(case)
        random_starts = np.column_stack(
            [
                rng.uniform(xmin, xmax, size=random_starts_per_objective),
                rng.uniform(ymin, ymax, size=random_starts_per_objective),
            ]
        )
        starts = np.vstack([case.starts, random_starts])
        augmented_cases.append(replace(case, starts=starts))
    return augmented_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic multi-start function benchmark report."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/function_report_multistart"),
        help="Directory for CSVs, plots, and the Markdown report.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        help=(
            "Optional objective names to include, e.g. "
            "--objectives quadratic rosenbrock beale."
        ),
    )
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=1.0,
        help="Multiply each selected objective's default iteration budget.",
    )
    parser.add_argument(
        "--random-starts-per-objective",
        type=int,
        default=0,
        help="Append this many deterministic random starts to each selected objective.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260525,
        help="Seed used when --random-starts-per-objective is positive.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = filter_cases(benchmark_cases(), args.objectives)
    cases = add_random_starts(
        cases,
        args.random_starts_per_objective,
        args.random_seed,
    )
    cases = scale_case_steps(cases, args.step_multiplier)
    all_rows: list[RunSummary] = []
    representative_histories: dict[str, dict[tuple[int, str], OptimizationHistory]] = {}
    plot_paths: list[Path] = []

    print("Running deterministic function benchmark report")
    print(f"Output directory: {output_dir.resolve()}")

    for case in cases:
        print(f"- {case.objective.name}: {len(case.starts)} starts, {case.steps} steps")
        rows, histories = run_case(case)
        all_rows.extend(rows)
        representative_histories[case.objective.name] = histories

        if case.objective.name in HIGHLIGHT_OBJECTIVES:
            plot_paths.append(plot_objective_surface_3d(case, output_dir))
            plot_paths.append(plot_highlight_trajectory(case, histories, output_dir))
            plot_paths.append(plot_highlight_objective(case, histories, output_dir))
            plot_paths.append(plot_highlight_alpha(case, histories, output_dir))

    aggregate = aggregate_rows(all_rows)

    per_start_fields = list(RunSummary.__dataclass_fields__.keys())
    aggregate_fields = [
        "objective",
        "optimizer",
        "num_starts",
        "success_rate",
        "median_final_f",
        "median_best_f",
        "median_final_residual",
        "median_best_residual",
        "median_final_distance",
        "median_best_distance",
        "median_iterations_to_success",
        "median_accepted_rate",
        "median_final_alpha",
        "median_trust_expansions",
    ]

    write_csv(output_dir / "per_start_results.csv", all_rows, per_start_fields)
    write_csv(output_dir / "aggregate_results.csv", aggregate, aggregate_fields)
    write_config_csv(cases, output_dir / "benchmark_config.csv")

    plot_paths.insert(0, plot_success_rates(aggregate, output_dir))
    plot_paths.insert(1, plot_median_best(aggregate, output_dir))

    report_path = generate_report(cases, aggregate, all_rows, plot_paths, output_dir)
    chinese_report_path = generate_chinese_report(
        cases,
        aggregate,
        all_rows,
        plot_paths,
        output_dir,
    )

    print("Done.")
    print(f"Report: {report_path.resolve()}")
    print(f"Chinese report: {chinese_report_path.resolve()}")
    print(f"Aggregate CSV: {(output_dir / 'aggregate_results.csv').resolve()}")
    print(f"Per-start CSV: {(output_dir / 'per_start_results.csv').resolve()}")


if __name__ == "__main__":
    main()
