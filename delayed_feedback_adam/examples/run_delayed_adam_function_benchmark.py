"""Tune and benchmark delayed-feedback Adam on deterministic 2D functions.

This script mirrors the PyTorch ``DelayedFeedbackAdam`` controller in NumPy so
we can run fast deterministic function-optimization sweeps.  The delayed
controller uses the current objective value to evaluate the previous step's
actual-vs-predicted decrease, then updates the global Adam multiplier before the
next step.  Unlike the same-step controlled Adam benchmark, it does not reject
bad trial steps because the feedback arrives one iteration late.

Run from the repository root or from ``delayed_feedback_adam``:

    MPLCONFIGDIR=/private/tmp python examples/run_delayed_adam_function_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_ADAM_SRC = ROOT / "controlled_adam_project" / "src"
CONTROLLED_ADAM_EXAMPLES = ROOT / "controlled_adam_project" / "examples"
for path in (CONTROLLED_ADAM_SRC, CONTROLLED_ADAM_EXAMPLES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import matplotlib.pyplot as plt
import numpy as np

import run_function_benchmark_report as base


OPTIMIZER_LABELS = {
    "vanilla_adam": "Vanilla Adam",
    "delayed_current": "Delayed Adam current",
    "delayed_raw_p": "Delayed Adam raw P",
    "delayed_ema_p": "Delayed Adam EMA P",
    "delayed_best": "Delayed Adam best tuned",
}

OPTIMIZER_COLORS = {
    "vanilla_adam": "#e07a24",
    "delayed_current": "#7b2cbf",
    "delayed_raw_p": "#c92a2a",
    "delayed_ema_p": "#2f6fbb",
    "delayed_best": "#2b8a3e",
}


@dataclass(frozen=True)
class DelayedAdamVariant:
    """Hyperparameters for one delayed-feedback Adam variant."""

    name: str
    description: str
    alpha_init: float = 1.0
    alpha_min: float = 0.1
    alpha_max: float = 10.0
    rho_star: float = 0.8
    kp: float = 0.05
    ki: float = 0.0
    kd: float = 0.0
    rho_beta: float = 0.95
    rho_min: float = -1.0
    rho_max: float = 2.0
    multiplier_min: float = 0.8
    multiplier_max: float = 1.25
    integral_decay: float = 0.95
    integral_min: float = -5.0
    integral_max: float = 5.0
    derivative_beta: float = 0.9
    min_predicted_decrease: float = 1e-16
    fallback_to_gradient: bool = True


@dataclass
class DelayedAdamHistory:
    """Trajectory and diagnostics for deterministic delayed-feedback Adam."""

    xs: np.ndarray
    fs: np.ndarray
    alphas: np.ndarray
    grad_norms: np.ndarray
    rhos: np.ndarray
    rho_bars: np.ndarray
    predicted_decreases: np.ndarray
    actual_decreases: np.ndarray
    multipliers: np.ndarray
    controller_applied: np.ndarray
    non_descent_tensors: np.ndarray


def delayed_feedback_adam(
    objective: base.Objective,
    x0: np.ndarray,
    alpha0: float,
    steps: int,
    variant: DelayedAdamVariant,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    divergence_norm: float = 1e8,
) -> DelayedAdamHistory:
    """Run deterministic Adam with delayed actual-vs-predicted feedback."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive.")
    if not (0.0 < variant.alpha_min <= variant.alpha_max):
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")
    if not (0.0 <= variant.rho_beta < 1.0):
        raise ValueError("rho_beta must be in [0, 1).")
    if not (0.0 < variant.multiplier_min <= variant.multiplier_max):
        raise ValueError("multiplier bounds must satisfy 0 < min <= max.")

    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    alpha = float(np.clip(variant.alpha_init, variant.alpha_min, variant.alpha_max))
    log_alpha = math.log(alpha)
    rho_bar: float | None = None
    prev_error: float | None = None
    integral = 0.0
    derivative = 0.0
    prev_loss: float | None = None
    prev_predicted_decrease: float | None = None

    xs = [x.copy()]
    fs = [objective.value(x)]
    alphas = []
    grad_norms = []
    rhos = []
    rho_bars = []
    predicted_decreases = []
    actual_decreases = []
    multipliers = []
    controller_applied = []
    non_descent_tensors = []

    for t in range(1, steps + 1):
        current_loss = objective.value(x)
        rho_raw = np.nan
        rho_control = np.nan
        actual_decrease = np.nan
        multiplier = 1.0
        applied = False

        if (
            prev_loss is not None
            and prev_predicted_decrease is not None
            and math.isfinite(prev_predicted_decrease)
            and prev_predicted_decrease > variant.min_predicted_decrease
        ):
            actual_decrease = float(prev_loss) - float(current_loss)
            rho_raw = actual_decrease / float(prev_predicted_decrease)
            rho_clipped = float(np.clip(rho_raw, variant.rho_min, variant.rho_max))
            if rho_bar is None:
                rho_bar = rho_clipped
            else:
                rho_bar = (
                    variant.rho_beta * float(rho_bar)
                    + (1.0 - variant.rho_beta) * rho_clipped
                )
            rho_control = float(rho_bar)

            error = rho_control - variant.rho_star
            integral = variant.integral_decay * integral + error
            integral = float(np.clip(integral, variant.integral_min, variant.integral_max))
            raw_derivative = 0.0 if prev_error is None else error - float(prev_error)
            derivative = (
                variant.derivative_beta * derivative
                + (1.0 - variant.derivative_beta) * raw_derivative
            )
            log_multiplier = (
                variant.kp * error
                + variant.ki * integral
                + variant.kd * derivative
            )
            multiplier = float(
                np.clip(
                    math.exp(log_multiplier),
                    variant.multiplier_min,
                    variant.multiplier_max,
                )
            )
            log_alpha += math.log(multiplier)
            log_alpha = float(
                np.clip(
                    log_alpha,
                    math.log(variant.alpha_min),
                    math.log(variant.alpha_max),
                )
            )
            alpha = math.exp(log_alpha)
            prev_error = error
            applied = True

        g = objective.gradient(x)
        grad_norm = float(np.linalg.norm(g))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        adam_direction = -m_hat / (np.sqrt(v_hat) + eps)
        direction = adam_direction
        non_descent = 0

        if float(np.dot(g, direction)) >= 0.0 and variant.fallback_to_gradient:
            direction = -g
            non_descent = 1

        effective_alpha = alpha0 * alpha
        predicted_decrease = -effective_alpha * float(np.dot(g, direction))
        x_candidate = x + effective_alpha * direction

        if (
            not np.all(np.isfinite(x_candidate))
            or float(np.linalg.norm(x_candidate)) > divergence_norm
        ):
            break

        f_candidate = objective.value(x_candidate)
        if not np.isfinite(f_candidate):
            break

        x = x_candidate
        xs.append(x.copy())
        fs.append(float(f_candidate))
        alphas.append(effective_alpha)
        grad_norms.append(grad_norm)
        rhos.append(rho_raw)
        rho_bars.append(rho_control)
        predicted_decreases.append(predicted_decrease)
        actual_decreases.append(actual_decrease)
        multipliers.append(multiplier)
        controller_applied.append(applied)
        non_descent_tensors.append(non_descent)

        prev_loss = float(current_loss)
        prev_predicted_decrease = float(predicted_decrease)

    return DelayedAdamHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        alphas=np.asarray(alphas),
        grad_norms=np.asarray(grad_norms),
        rhos=np.asarray(rhos),
        rho_bars=np.asarray(rho_bars),
        predicted_decreases=np.asarray(predicted_decreases),
        actual_decreases=np.asarray(actual_decreases),
        multipliers=np.asarray(multipliers),
        controller_applied=np.asarray(controller_applied, dtype=bool),
        non_descent_tensors=np.asarray(non_descent_tensors, dtype=int),
    )


def history_for_summary(history: DelayedAdamHistory) -> base.OptimizationHistory:
    """Convert delayed diagnostics to the shared report history type."""

    return base.OptimizationHistory(
        xs=history.xs,
        fs=history.fs,
        alphas=history.alphas,
        grad_norms=history.grad_norms,
        rhos=history.rhos,
        predicted_decreases=history.predicted_decreases,
        actual_decreases=history.actual_decreases,
        accepted=None,
        descent_scores=None,
    )


def current_variant() -> DelayedAdamVariant:
    """Return the README-style initial delayed Adam settings."""

    return DelayedAdamVariant(
        name="current",
        description="README-style delayed P controller: alpha in [0.1, 10], rho*=0.8, kp=0.05, rho_beta=0.95.",
        alpha_init=1.0,
        alpha_min=0.1,
        alpha_max=10.0,
        rho_star=0.8,
        kp=0.05,
        rho_beta=0.95,
        multiplier_min=0.8,
        multiplier_max=1.25,
    )


def preset_variants() -> list[DelayedAdamVariant]:
    """Return a compact preset grid for delayed Adam deterministic tuning."""

    variants = [
        current_variant(),
        DelayedAdamVariant(
            name="raw_balanced",
            description="Raw delayed rho, moderate response, wider alpha range.",
            alpha_min=0.01,
            alpha_max=50.0,
            rho_star=0.5,
            kp=0.08,
            rho_beta=0.0,
            multiplier_min=0.5,
            multiplier_max=1.5,
        ),
        DelayedAdamVariant(
            name="raw_aggressive",
            description="Raw delayed rho, lower rho target and stronger response.",
            alpha_min=0.01,
            alpha_max=50.0,
            rho_star=0.3,
            kp=0.16,
            rho_beta=0.0,
            multiplier_min=0.25,
            multiplier_max=2.0,
        ),
        DelayedAdamVariant(
            name="ema_balanced",
            description="EMA delayed rho, moderate response.",
            alpha_min=0.01,
            alpha_max=50.0,
            rho_star=0.5,
            kp=0.12,
            rho_beta=0.7,
            multiplier_min=0.5,
            multiplier_max=1.5,
        ),
        DelayedAdamVariant(
            name="ema_fast",
            description="Lightly smoothed delayed rho, lower target and stronger response.",
            alpha_min=0.01,
            alpha_max=50.0,
            rho_star=0.3,
            kp=0.20,
            rho_beta=0.5,
            multiplier_min=0.4,
            multiplier_max=1.8,
        ),
    ]

    for rho_star in [0.3, 0.5, 0.7]:
        for kp in [0.08, 0.16, 0.24]:
            for alpha_min in [0.003, 0.01, 0.03]:
                variants.append(
                    DelayedAdamVariant(
                        name=(
                            f"raw_rs{value_tag(rho_star)}_kp{value_tag(kp)}"
                            f"_amin{value_tag(alpha_min)}"
                        ),
                        description=(
                            f"Raw delayed rho, rho*={rho_star:g}, kp={kp:g}, "
                            f"alpha_min={alpha_min:g}."
                        ),
                        alpha_min=alpha_min,
                        alpha_max=50.0,
                        rho_star=rho_star,
                        kp=kp,
                        rho_beta=0.0,
                        multiplier_min=0.4,
                        multiplier_max=1.8,
                    )
                )

    for alpha_max in [10.0, 20.0]:
        for rho_star in [0.5, 0.7]:
            for kp in [0.04, 0.08, 0.12]:
                variants.append(
                    DelayedAdamVariant(
                        name=(
                            f"raw_cap{value_tag(alpha_max)}_rs{value_tag(rho_star)}"
                            f"_kp{value_tag(kp)}"
                        ),
                        description=(
                            f"Raw delayed rho with capped alpha_max={alpha_max:g}, "
                            f"rho*={rho_star:g}, kp={kp:g}."
                        ),
                        alpha_min=0.01,
                        alpha_max=alpha_max,
                        rho_star=rho_star,
                        kp=kp,
                        rho_beta=0.0,
                        multiplier_min=0.5,
                        multiplier_max=1.5,
                    )
                )

    for alpha_max in [10.0, 20.0]:
        for rho_star in [0.5, 0.7]:
            for kp in [0.08, 0.12, 0.20]:
                variants.append(
                    DelayedAdamVariant(
                        name=(
                            f"ema70_cap{value_tag(alpha_max)}_rs{value_tag(rho_star)}"
                            f"_kp{value_tag(kp)}"
                        ),
                        description=(
                            f"EMA beta=0.70 delayed rho with capped alpha_max={alpha_max:g}, "
                            f"rho*={rho_star:g}, kp={kp:g}."
                        ),
                        alpha_min=0.01,
                        alpha_max=alpha_max,
                        rho_star=rho_star,
                        kp=kp,
                        rho_beta=0.7,
                        multiplier_min=0.5,
                        multiplier_max=1.5,
                    )
                )

    for rho_beta in [0.5, 0.7, 0.9]:
        for rho_star in [0.3, 0.5]:
            for kp in [0.12, 0.20, 0.32]:
                variants.append(
                    DelayedAdamVariant(
                        name=(
                            f"ema_b{int(rho_beta * 100)}_rs{value_tag(rho_star)}"
                            f"_kp{value_tag(kp)}"
                        ),
                        description=(
                            f"EMA delayed rho, beta={rho_beta:g}, rho*={rho_star:g}, "
                            f"kp={kp:g}."
                        ),
                        alpha_min=0.01,
                        alpha_max=50.0,
                        rho_star=rho_star,
                        kp=kp,
                        rho_beta=rho_beta,
                        multiplier_min=0.4,
                        multiplier_max=1.8,
                    )
                )

    for ki in [0.002, 0.005, 0.01]:
        variants.append(
            DelayedAdamVariant(
                name=f"pi_ema70_ki{value_tag(ki)}",
                description=f"Small PI term with EMA beta=0.70 and ki={ki:g}.",
                alpha_min=0.01,
                alpha_max=50.0,
                rho_star=0.5,
                kp=0.08,
                ki=ki,
                rho_beta=0.7,
                multiplier_min=0.5,
                multiplier_max=1.5,
                integral_decay=0.9,
                integral_min=-2.0,
                integral_max=2.0,
            )
        )

    return dedupe_variants(variants)


def dedupe_variants(variants: Iterable[DelayedAdamVariant]) -> list[DelayedAdamVariant]:
    """Remove accidental duplicate names while preserving order."""

    seen = set()
    deduped = []
    for variant in variants:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        deduped.append(variant)
    return deduped


def value_tag(value: float) -> str:
    """Return a compact float tag for variant names."""

    return f"{value:g}".replace(".", "p").replace("-", "m")


def run_variant_cases(
    cases: list[base.BenchmarkCase],
    variant: DelayedAdamVariant,
) -> tuple[list[base.RunSummary], dict[tuple[str, int], DelayedAdamHistory]]:
    """Run one delayed Adam variant on every objective/start."""

    rows: list[base.RunSummary] = []
    histories: dict[tuple[str, int], DelayedAdamHistory] = {}
    for case in cases:
        for start_id, x0 in enumerate(case.starts):
            history = delayed_feedback_adam(
                case.objective,
                x0,
                alpha0=case.alpha,
                steps=case.steps,
                variant=variant,
            )
            histories[(case.objective.name, start_id)] = history
            rows.append(
                base.summarize_run(
                    case,
                    start_id,
                    variant.name,
                    x0,
                    history_for_summary(history),
                )
            )
    return rows, histories


def mean_log10_best(rows: list[dict[str, object]]) -> float:
    """Return mean log10 median-best residual across objective rows."""

    values = [
        math.log10(max(float(row["median_best_residual"]), 1e-12))
        for row in rows
    ]
    return float(np.mean(values)) if values else float("nan")


def average_success(rows: list[dict[str, object]]) -> float:
    """Return average objective-level success rate."""

    values = [float(row["success_rate"]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def score_variant(aggregate_rows: list[dict[str, object]]) -> float:
    """Rank variants by success first, then log residual."""

    return average_success(aggregate_rows) - 0.02 * mean_log10_best(aggregate_rows)


def variant_summary_rows(
    aggregate: list[dict[str, object]],
    variants: list[DelayedAdamVariant],
) -> list[dict[str, object]]:
    """Summarize each variant across objectives."""

    rows = []
    for variant in variants:
        subset = [row for row in aggregate if row["optimizer"] == variant.name]
        rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "avg_success": average_success(subset),
                "mean_log10_median_best_residual": mean_log10_best(subset),
                "score": score_variant(subset),
                "alpha_init": variant.alpha_init,
                "alpha_min": variant.alpha_min,
                "alpha_max": variant.alpha_max,
                "rho_star": variant.rho_star,
                "kp": variant.kp,
                "ki": variant.ki,
                "kd": variant.kd,
                "rho_beta": variant.rho_beta,
                "rho_min": variant.rho_min,
                "rho_max": variant.rho_max,
                "multiplier_min": variant.multiplier_min,
                "multiplier_max": variant.multiplier_max,
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)


def rows_for_optimizer(
    aggregate: list[dict[str, object]],
    optimizer: str,
) -> list[dict[str, object]]:
    """Return aggregate rows for one optimizer."""

    return [row for row in aggregate if row["optimizer"] == optimizer]


def select_final_variants(
    summary: list[dict[str, object]],
    variants: list[DelayedAdamVariant],
    extra_names: list[str] | None = None,
) -> list[DelayedAdamVariant]:
    """Select variants for the full report."""

    by_name = {variant.name: variant for variant in variants}
    selected_names = ["current", "raw_balanced", "ema_balanced"]
    if summary:
        selected_names.append(str(summary[0]["variant"]))
    if extra_names:
        selected_names.extend(extra_names)

    selected: list[DelayedAdamVariant] = []
    seen = set()
    for name in selected_names:
        if name in seen or name not in by_name:
            continue
        selected.append(by_name[name])
        seen.add(name)

    while len(selected) < 4 and len(summary) >= len(selected):
        name = str(summary[len(selected)]["variant"])
        if name in by_name and name not in seen:
            selected.append(by_name[name])
            seen.add(name)
    return selected


def configure_final_labels(selected: list[DelayedAdamVariant]) -> None:
    """Patch shared plotting helpers for the selected final optimizers."""

    labels = {"vanilla_adam": "Vanilla Adam"}
    colors = {"vanilla_adam": OPTIMIZER_COLORS["vanilla_adam"]}

    for index, variant in enumerate(selected):
        if variant.name == "current":
            label = "Delayed Adam current"
            color = OPTIMIZER_COLORS["delayed_current"]
        elif index == len(selected) - 1:
            label = f"Delayed Adam best ({variant.name})"
            color = OPTIMIZER_COLORS["delayed_best"]
        elif variant.name.startswith("raw"):
            label = f"Delayed Adam raw ({variant.name})"
            color = OPTIMIZER_COLORS["delayed_raw_p"]
        elif variant.name.startswith("ema"):
            label = f"Delayed Adam EMA ({variant.name})"
            color = OPTIMIZER_COLORS["delayed_ema_p"]
        else:
            label = f"Delayed Adam ({variant.name})"
            color = "#495057"
        labels[variant.name] = label
        colors[variant.name] = color

    base.OPTIMIZER_LABELS = labels
    base.OPTIMIZER_LABELS_ZH = labels
    base.OPTIMIZER_COLORS = colors


def run_final_benchmark(
    cases: list[base.BenchmarkCase],
    selected_variants: list[DelayedAdamVariant],
) -> tuple[
    list[base.RunSummary],
    dict[tuple[str, int, str], base.OptimizationHistory],
    dict[str, dict[tuple[str, int], DelayedAdamHistory]],
]:
    """Run vanilla Adam and selected delayed variants for the full report."""

    all_rows: list[base.RunSummary] = []
    highlight_histories: dict[tuple[str, int, str], base.OptimizationHistory] = {}
    delayed_histories: dict[str, dict[tuple[str, int], DelayedAdamHistory]] = {}

    for case in cases:
        for start_id, x0 in enumerate(case.starts):
            adam = base.vanilla_adam(case.objective, x0, alpha=case.alpha, steps=case.steps)
            all_rows.append(base.summarize_run(case, start_id, "vanilla_adam", x0, adam))
            if start_id == case.highlight_start_index:
                highlight_histories[
                    (case.objective.name, start_id, "vanilla_adam")
                ] = adam

        for variant in selected_variants:
            rows, histories = run_variant_cases([case], variant)
            delayed_histories[variant.name] = {
                **delayed_histories.get(variant.name, {}),
                **histories,
            }
            all_rows.extend(rows)
            key = (case.objective.name, case.highlight_start_index)
            if key in histories:
                highlight_histories[
                    (case.objective.name, case.highlight_start_index, variant.name)
                ] = history_for_summary(histories[key])

    return all_rows, highlight_histories, delayed_histories


def write_variant_config_csv(variants: list[DelayedAdamVariant], path: Path) -> None:
    """Write selected variant settings to CSV."""

    fieldnames = [
        "name",
        "description",
        "alpha_init",
        "alpha_min",
        "alpha_max",
        "rho_star",
        "kp",
        "ki",
        "kd",
        "rho_beta",
        "rho_min",
        "rho_max",
        "multiplier_min",
        "multiplier_max",
        "integral_decay",
        "integral_min",
        "integral_max",
        "derivative_beta",
        "min_predicted_decrease",
        "fallback_to_gradient",
    ]
    base.write_csv(path, variants, fieldnames)


def write_tuning_report(
    summary: list[dict[str, object]],
    variants: list[DelayedAdamVariant],
    aggregate: list[dict[str, object]],
    per_start_count: int,
    output_dir: Path,
) -> Path:
    """Write a concise delayed Adam tuning report."""

    report_path = output_dir / "DELAYED_ADAM_FUNCTION_TUNING_REPORT.md"
    current_rows = rows_for_optimizer(aggregate, "current")
    best_name = str(summary[0]["variant"]) if summary else ""
    best_rows = rows_for_optimizer(aggregate, best_name)

    lines = [
        "# Delayed Feedback Adam Function Tuning",
        "",
        "This sweep mirrors `DelayedFeedbackAdam` on the deterministic 2D function suite.",
        "The controller receives one-step-delayed feedback and cannot reject bad steps.",
        "",
        "## Validation",
        "",
        f"- Variants: `{len(variants)}`",
        "- Objectives: `9`",
        "- Starts per objective: `30`",
        f"- Per-start rows: `{per_start_count}`",
        "",
        "## Top Variants",
        "",
        "| Rank | Variant | Avg Success | Mean Log10 Median Best Residual | Score |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary[:12], start=1):
        lines.append(
            f"| {rank} | `{row['variant']}` | "
            f"{float(row['avg_success']):.3f} | "
            f"{float(row['mean_log10_median_best_residual']):.3f} | "
            f"{float(row['score']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Current vs Best",
            "",
            "| Variant | Avg Success | Mean Log10 Median Best Residual |",
            "|---|---:|---:|",
            (
                f"| `current` | {average_success(current_rows):.3f} | "
                f"{mean_log10_best(current_rows):.3f} |"
            ),
            (
                f"| `{best_name}` | {average_success(best_rows):.3f} | "
                f"{mean_log10_best(best_rows):.3f} |"
            ),
            "",
            "## Interpretation",
            "",
            "- Delayed feedback is lower-overhead than same-step control but reacts one step late.",
            "- Strong deterministic presets can improve over the README-style current settings.",
            "- The best function-suite preset is not automatically a neural-network default.",
            "- The final benchmark report compares selected delayed variants with vanilla Adam.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
    return report_path


def write_final_report(
    cases: list[base.BenchmarkCase],
    aggregate: list[dict[str, object]],
    selected_variants: list[DelayedAdamVariant],
    plot_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Write final delayed Adam benchmark report."""

    report_path = output_dir / "DELAYED_ADAM_FUNCTION_BENCHMARK_REPORT.md"
    objective_order = [case.objective.name for case in cases]
    by_objective = {
        objective: [row for row in aggregate if row["objective"] == objective]
        for objective in objective_order
    }

    winners_success = base.winner_counts(aggregate, "success_rate", lower_is_better=False)
    winners_final = base.winner_counts(aggregate, "median_final_residual", lower_is_better=True)
    winners_best = base.winner_counts(aggregate, "median_best_residual", lower_is_better=True)

    lines = [
        "# Delayed Feedback Adam Function Benchmark",
        "",
        "This report tests delayed-feedback Adam on the same deterministic 2D function suite used by `controlled_adam_project`.",
        "The delayed controller uses the next normally observed objective value to evaluate the previous step.",
        "",
        "## Executive Summary",
        "",
        "- Vanilla Adam is included as the non-controlled baseline.",
        "- Delayed Adam variants use the same Adam direction and base `alpha0` as vanilla Adam, but scale it by a controlled multiplier.",
        "- Unlike same-step controlled Adam, delayed Adam cannot reject bad steps before applying them.",
        "- Results are deterministic function findings, not neural-network defaults.",
        "",
        "Average objective-level summary:",
        "",
        "| Optimizer | Avg Success | Mean Log10 Median Best Residual |",
        "|---|---:|---:|",
    ]
    for optimizer, label in base.OPTIMIZER_LABELS.items():
        rows = rows_for_optimizer(aggregate, optimizer)
        lines.append(
            f"| {label} | {average_success(rows):.3f} | "
            f"{mean_log10_best(rows):.3f} |"
        )

    lines.extend(
        [
            "",
            "Winner counts across objectives:",
            "",
            base.report_optimizer_header(),
            base.winner_row("Highest success rate", winners_success),
            base.winner_row("Lowest median final residual", winners_final),
            base.winner_row("Lowest median best residual", winners_best),
            "",
            "## Selected Delayed Variants",
            "",
            "| Variant | Description | alpha bounds | rho_star | kp | rho_beta | multiplier bounds |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for variant in selected_variants:
        lines.append(
            f"| `{variant.name}` | {variant.description} | "
            f"[{variant.alpha_min:g}, {variant.alpha_max:g}] | "
            f"{variant.rho_star:g} | {variant.kp:g} | {variant.rho_beta:g} | "
            f"[{variant.multiplier_min:g}, {variant.multiplier_max:g}] |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Objective | Optimizer | Success Rate | Median Final Residual | Median Best Residual | Median Best Distance | Median Iterations To Success | Median Final Alpha |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for objective in objective_order:
        for row in by_objective[objective]:
            label = base.OPTIMIZER_LABELS[str(row["optimizer"])]
            lines.append(
                f"| `{objective}` | {label} | "
                f"{float(row['success_rate']):.2f} | "
                f"{float(row['median_final_residual']):.3e} | "
                f"{float(row['median_best_residual']):.3e} | "
                f"{float(row['median_best_distance']):.3e} | "
                f"{base.format_float(row['median_iterations_to_success'])} | "
                f"{float(row['median_final_alpha']):.3e} |"
            )

    lines.extend(
        [
            "",
            "## Key Plots",
            "",
        ]
    )
    for path in plot_paths:
        rel = path.relative_to(output_dir)
        lines.append(f"![{rel.stem}]({rel.as_posix()})")
        lines.append("")

    lines.extend(
        [
            "## Output Files",
            "",
            "- `per_start_results.csv`: one row per objective/start/optimizer.",
            "- `aggregate_results.csv`: summary by objective/optimizer.",
            "- `benchmark_config.csv`: objective settings and tolerances.",
            "- `selected_variant_config.csv`: delayed Adam variant settings.",
            "- `*_trajectory_comparison.png`: representative trajectories.",
            "- `*_objective_curves.png`: representative objective curves.",
            "- `*_alpha_curves.png`: representative effective learning-rate schedules.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
    return report_path


def run_tuning(args: argparse.Namespace) -> None:
    """Run delayed Adam tuning sweep."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = base.filter_cases(base.benchmark_cases(), args.objectives)
    cases = base.add_random_starts(
        cases,
        args.random_starts_per_objective,
        args.random_seed,
    )
    cases = base.scale_case_steps(cases, args.step_multiplier)
    variants = preset_variants()

    all_rows: list[base.RunSummary] = []
    print("Running delayed Adam function tuning")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Variants: {len(variants)}")
    for index, variant in enumerate(variants, start=1):
        print(f"- [{index}/{len(variants)}] {variant.name}")
        rows, _ = run_variant_cases(cases, variant)
        all_rows.extend(rows)

    aggregate = base.aggregate_rows(all_rows)
    summary = variant_summary_rows(aggregate, variants)

    per_start_fields = list(base.RunSummary.__dataclass_fields__.keys())
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
    summary_fields = [
        "variant",
        "description",
        "avg_success",
        "mean_log10_median_best_residual",
        "score",
        "alpha_init",
        "alpha_min",
        "alpha_max",
        "rho_star",
        "kp",
        "ki",
        "kd",
        "rho_beta",
        "rho_min",
        "rho_max",
        "multiplier_min",
        "multiplier_max",
    ]

    base.write_csv(output_dir / "per_start_results.csv", all_rows, per_start_fields)
    base.write_csv(output_dir / "aggregate_results.csv", aggregate, aggregate_fields)
    base.write_csv(output_dir / "variant_summary.csv", summary, summary_fields)
    write_variant_config_csv(variants, output_dir / "variant_config.csv")
    base.write_config_csv(cases, output_dir / "benchmark_config.csv")
    report_path = write_tuning_report(
        summary,
        variants,
        aggregate,
        len(all_rows),
        output_dir,
    )

    print("Done.")
    print(f"Report: {report_path}")
    if summary:
        print(f"Best variant: {summary[0]['variant']}")


def run_final(args: argparse.Namespace) -> None:
    """Run final delayed Adam benchmark report."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = base.filter_cases(base.benchmark_cases(), args.objectives)
    cases = base.add_random_starts(
        cases,
        args.random_starts_per_objective,
        args.random_seed,
    )
    cases = base.scale_case_steps(cases, args.step_multiplier)
    variants = preset_variants()

    summary_path = args.tuning_summary
    extra_names = args.variants
    if summary_path is not None and summary_path.exists():
        summary = list(csv.DictReader(summary_path.open()))
    else:
        print("No tuning summary supplied; using built-in selected variants.")
        summary = variant_summary_rows([], variants)
    selected = select_final_variants(summary, variants, extra_names)
    if not selected:
        selected = [current_variant()]

    configure_final_labels(selected)
    all_rows, highlight_histories, _ = run_final_benchmark(cases, selected)
    aggregate = base.aggregate_rows(all_rows)

    plot_paths: list[Path] = []
    for case in cases:
        if case.objective.name in base.HIGHLIGHT_OBJECTIVES:
            plot_paths.append(base.plot_objective_surface_3d(case, output_dir))
            case_histories = {
                (case.highlight_start_index, optimizer): highlight_histories[
                    (case.objective.name, case.highlight_start_index, optimizer)
                ]
                for optimizer in base.OPTIMIZER_LABELS
            }
            plot_paths.append(base.plot_highlight_trajectory(case, case_histories, output_dir))
            plot_paths.append(base.plot_highlight_objective(case, case_histories, output_dir))
            plot_paths.append(base.plot_highlight_alpha(case, case_histories, output_dir))

    plot_paths.insert(0, base.plot_success_rates(aggregate, output_dir))
    plot_paths.insert(1, base.plot_median_best(aggregate, output_dir))

    per_start_fields = list(base.RunSummary.__dataclass_fields__.keys())
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
    base.write_csv(output_dir / "per_start_results.csv", all_rows, per_start_fields)
    base.write_csv(output_dir / "aggregate_results.csv", aggregate, aggregate_fields)
    base.write_config_csv(cases, output_dir / "benchmark_config.csv")
    write_variant_config_csv(selected, output_dir / "selected_variant_config.csv")
    report_path = write_final_report(cases, aggregate, selected, plot_paths, output_dir)

    print("Done.")
    print(f"Report: {report_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run delayed-feedback Adam function tuning or benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "final"],
        default="tune",
        help="Run a tuning sweep or a final benchmark report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "delayed_adam_function_tuning_30runs",
    )
    parser.add_argument("--objectives", nargs="+", help="Optional objective names.")
    parser.add_argument("--step-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--random-starts-per-objective",
        type=int,
        default=25,
        help="Append this many deterministic random starts per objective.",
    )
    parser.add_argument("--random-seed", type=int, default=20260527)
    parser.add_argument(
        "--tuning-summary",
        type=Path,
        help="variant_summary.csv from a tuning run, used in final mode.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Additional variant names to include in final mode.",
    )
    args = parser.parse_args()

    if args.mode == "tune":
        run_tuning(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
