"""Run the deterministic function benchmark with simplified tuned Adam controls.

This report is a follow-up to ``run_controlled_adam_simplified_tuning_sweep.py``.
It uses the best simplified preset found there, ``aggressive_high_floor``, for
all three controlled Adam families:

- ``kp_multiplier = 2``
- ``rho_star_delta = -0.2``
- ``alpha_min = 0.01 * alpha0``
- ``alpha_max = 50 * alpha0``
- trust uses ``rho_beta = 0.90``, ``rho >= 0.60``, ``alpha <= 3 * alpha0``,
  and expansion factor ``3``.

The comparison intentionally excludes SGD with momentum. It keeps plain fixed
gradient descent, vanilla Adam, and the three tuned controlled Adam variants.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import run_function_benchmark_report as base


TUNED_PRESET = {
    "name": "aggressive_high_floor",
    "kp_multiplier": 2.0,
    "rho_star_delta": -0.2,
    "rho_beta": 0.90,
    "alpha_min_factor": 0.01,
    "alpha_max_factor": 50.0,
    "trust_region_rho_threshold": 0.60,
    "trust_alpha_threshold_factor": 3.0,
    "trust_region_expand_factor": 3.0,
    "trust_region_max_factor": 3.0,
}

OPTIMIZER_LABELS = {
    "gradient_descent": "Gradient descent",
    "vanilla_adam": "Vanilla Adam",
    "controlled_raw_rho": "Tuned controlled Adam (raw rho)",
    "controlled_ema_rho": "Tuned controlled Adam (EMA rho)",
    "controlled_ema_trust": "Tuned controlled Adam (EMA + trust)",
}

OPTIMIZER_LABELS_ZH = {
    "gradient_descent": "梯度下降",
    "vanilla_adam": "标准 Adam",
    "controlled_raw_rho": "调参后受控 Adam（原始 rho）",
    "controlled_ema_rho": "调参后受控 Adam（EMA 平滑 rho）",
    "controlled_ema_trust": "调参后受控 Adam（EMA + 信赖域扩张）",
}

OPTIMIZER_COLORS = {
    "gradient_descent": "#6f42c1",
    "vanilla_adam": "#e07a24",
    "controlled_raw_rho": "#c92a2a",
    "controlled_ema_rho": "#2f6fbb",
    "controlled_ema_trust": "#2b8a3e",
}


def configure_report_labels() -> None:
    """Patch imported plotting/report helpers to use this benchmark's optimizers."""

    base.OPTIMIZER_LABELS = OPTIMIZER_LABELS
    base.OPTIMIZER_LABELS_ZH = OPTIMIZER_LABELS_ZH
    base.OPTIMIZER_COLORS = OPTIMIZER_COLORS


def tuned_alpha_min(case: base.BenchmarkCase) -> float:
    """Return alpha_min for the simplified tuned preset."""

    return float(case.alpha * TUNED_PRESET["alpha_min_factor"])


def tuned_alpha_max(case: base.BenchmarkCase, alpha_min: float) -> float:
    """Return alpha_max for the simplified tuned preset."""

    return max(alpha_min, float(case.alpha * TUNED_PRESET["alpha_max_factor"]))


def tuned_rho_star(case: base.BenchmarkCase) -> float:
    """Return rho_star for the simplified tuned preset."""

    return float(np.clip(case.rho_star + TUNED_PRESET["rho_star_delta"], 0.05, 0.95))


def tuned_kp(case: base.BenchmarkCase) -> float:
    """Return kp for the simplified tuned preset."""

    return float(case.kp * TUNED_PRESET["kp_multiplier"])


def tuned_trust_alpha_threshold(
    case: base.BenchmarkCase,
    alpha_min: float,
) -> float:
    """Return trust alpha threshold for the simplified tuned preset."""

    return max(alpha_min, float(case.alpha * TUNED_PRESET["trust_alpha_threshold_factor"]))


def run_case(
    case: base.BenchmarkCase,
    gradient_descent_alpha_multiplier: float,
) -> tuple[list[base.RunSummary], dict[tuple[int, str], base.OptimizationHistory]]:
    """Run no-momentum benchmark variants for one objective."""

    rows: list[base.RunSummary] = []
    histories: dict[tuple[int, str], base.OptimizationHistory] = {}

    alpha_min = tuned_alpha_min(case)
    alpha_max = tuned_alpha_max(case, alpha_min)
    rho_star = tuned_rho_star(case)
    kp = tuned_kp(case)
    trust_alpha_threshold = tuned_trust_alpha_threshold(case, alpha_min)
    gd_alpha = case.alpha * gradient_descent_alpha_multiplier

    for start_id, x0 in enumerate(case.starts):
        gd = base.run_gradient_descent(
            case.objective,
            x0,
            alpha=gd_alpha,
            steps=case.steps,
        )
        histories[(start_id, "gradient_descent")] = gd
        rows.append(base.summarize_run(case, start_id, "gradient_descent", x0, gd))

        adam = base.vanilla_adam(
            case.objective,
            x0,
            alpha=case.alpha,
            steps=case.steps,
        )
        histories[(start_id, "vanilla_adam")] = adam
        rows.append(base.summarize_run(case, start_id, "vanilla_adam", x0, adam))

        raw = base.controlled_adam(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=kp,
            rho_star=rho_star,
            rho_min=case.rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
        )
        histories[(start_id, "controlled_raw_rho")] = raw
        rows.append(base.summarize_run(case, start_id, "controlled_raw_rho", x0, raw))

        ema = base.controlled_adam_ema_rho(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=kp,
            rho_star=rho_star,
            rho_beta=TUNED_PRESET["rho_beta"],
            rho_min=case.rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
        )
        histories[(start_id, "controlled_ema_rho")] = ema
        rows.append(base.summarize_run(case, start_id, "controlled_ema_rho", x0, ema))

        trust = base.controlled_adam_ema_rho(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=kp,
            rho_star=rho_star,
            rho_beta=TUNED_PRESET["rho_beta"],
            rho_min=case.rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
            trust_region_expand=True,
            trust_region_rho_threshold=TUNED_PRESET["trust_region_rho_threshold"],
            trust_region_alpha_threshold=trust_alpha_threshold,
            trust_region_expand_factor=TUNED_PRESET["trust_region_expand_factor"],
            trust_region_max_factor=TUNED_PRESET["trust_region_max_factor"],
        )
        histories[(start_id, "controlled_ema_trust")] = trust
        rows.append(
            base.summarize_run(case, start_id, "controlled_ema_trust", x0, trust)
        )

    return rows, histories


def write_tuned_config_csv(cases: list[base.BenchmarkCase], path: Path) -> None:
    """Write absolute tuned settings by objective."""

    fieldnames = [
        "objective",
        "preset",
        "alpha0",
        "alpha_min",
        "alpha_max",
        "rho_star",
        "kp",
        "rho_beta",
        "trust_region_rho_threshold",
        "trust_region_alpha_threshold",
        "trust_region_expand_factor",
        "trust_region_max_factor",
    ]
    rows = []
    for case in cases:
        alpha_min = tuned_alpha_min(case)
        rows.append(
            {
                "objective": case.objective.name,
                "preset": TUNED_PRESET["name"],
                "alpha0": case.alpha,
                "alpha_min": alpha_min,
                "alpha_max": tuned_alpha_max(case, alpha_min),
                "rho_star": tuned_rho_star(case),
                "kp": tuned_kp(case),
                "rho_beta": TUNED_PRESET["rho_beta"],
                "trust_region_rho_threshold": TUNED_PRESET[
                    "trust_region_rho_threshold"
                ],
                "trust_region_alpha_threshold": tuned_trust_alpha_threshold(
                    case,
                    alpha_min,
                ),
                "trust_region_expand_factor": TUNED_PRESET[
                    "trust_region_expand_factor"
                ],
                "trust_region_max_factor": TUNED_PRESET[
                    "trust_region_max_factor"
                ],
            }
        )
    base.write_csv(path, rows, fieldnames)


def report_optimizer_header() -> str:
    """Return a Markdown table header for winner counts."""

    columns = " | ".join(OPTIMIZER_LABELS.values())
    separators = " | ".join(["---", *[":---:" for _ in OPTIMIZER_LABELS]])
    return f"| Criterion | {columns} |\n|{separators}|"


def winner_row(label: str, counts: dict[str, int]) -> str:
    """Format an optimizer winner-count row."""

    values = " | ".join(str(counts.get(optimizer, 0)) for optimizer in OPTIMIZER_LABELS)
    return f"| {label} | {values} |"


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


def median_trust_expansions(rows: list[dict[str, object]]) -> float:
    """Return median objective-level trust expansion count."""

    values = [
        float(row["median_trust_expansions"])
        for row in rows
        if np.isfinite(float(row["median_trust_expansions"]))
    ]
    return float(np.median(values)) if values else float("nan")


def rows_for_optimizer(
    aggregate: list[dict[str, object]],
    optimizer: str,
) -> list[dict[str, object]]:
    """Return aggregate rows for one optimizer."""

    return [row for row in aggregate if row["optimizer"] == optimizer]


def generate_report(
    cases: list[base.BenchmarkCase],
    aggregate: list[dict[str, object]],
    plot_paths: list[Path],
    output_dir: Path,
    gradient_descent_alpha_multiplier: float,
) -> Path:
    """Write the tuned no-momentum benchmark report."""

    report_path = output_dir / "FUNCTION_OPTIMIZATION_TUNED_BENCHMARK_REPORT.md"
    objective_order = [case.objective.name for case in cases]
    aggregate_by_objective = {
        objective: [row for row in aggregate if row["objective"] == objective]
        for objective in objective_order
    }
    num_objectives = len(cases)
    objective_word = "objective" if num_objectives == 1 else "objectives"

    winners_final = base.winner_counts(
        aggregate,
        "median_final_residual",
        lower_is_better=True,
    )
    winners_best = base.winner_counts(
        aggregate,
        "median_best_residual",
        lower_is_better=True,
    )
    winners_success = base.winner_counts(
        aggregate,
        "success_rate",
        lower_is_better=False,
    )
    controlled_success_advantages = base.objective_success_advantages(aggregate)
    controlled_best_advantages = base.objective_best_residual_advantages(aggregate)

    lines: list[str] = []
    lines.append("# Tuned Function Optimization Benchmark Report")
    lines.append("")
    lines.append(
        "This report uses the simplified tuned `aggressive_high_floor` controlled Adam preset and intentionally excludes SGD with momentum."
    )
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- The benchmark compares fixed gradient descent, vanilla Adam, and three tuned controlled Adam variants on {num_objectives} deterministic {objective_word}."
    )
    lines.append("- Momentum gradient descent is omitted in this run by request.")
    lines.append(
        "- Every optimizer sees the same starting points, iteration budgets, objective values, and gradients."
    )
    lines.append(
        "- The tuned controlled variants use scale-relative alpha bounds derived from each objective's base `alpha0`."
    )
    lines.append(
        f"- The gradient descent baseline uses `alpha = {gradient_descent_alpha_multiplier:g} * alpha0`; Adam and controlled Adam still use the original `alpha0`."
    )
    lines.append(
        "- This is still a local deterministic function benchmark, not a neural-network or global-optimization claim."
    )
    lines.append("")
    lines.append("Average objective-level summary:")
    lines.append("")
    lines.append(
        "| Optimizer | Avg Success | Mean Log10 Median Best Residual | Median Trust Expansions |"
    )
    lines.append("|---|---:|---:|---:|")
    for optimizer, label in OPTIMIZER_LABELS.items():
        rows = rows_for_optimizer(aggregate, optimizer)
        lines.append(
            f"| {label} | {average_success(rows):.3f} | "
            f"{mean_log10_best(rows):.3f} | "
            f"{base.format_float(median_trust_expansions(rows))} |"
        )
    lines.append("")
    lines.append("Winner counts across objectives:")
    lines.append("")
    lines.extend(report_optimizer_header().splitlines())
    lines.append(winner_row("Highest success rate", winners_success))
    lines.append(winner_row("Lowest median final residual", winners_final))
    lines.append(winner_row("Lowest median best residual", winners_best))
    lines.append("")
    lines.append(
        f"Ties are counted for every tied optimizer, so row totals can exceed {num_objectives}."
    )
    lines.append("")
    lines.append("## Tuned Preset")
    lines.append("")
    lines.append("The controlled variants use:")
    lines.append("")
    lines.append("```text")
    lines.append("preset = aggressive_high_floor")
    lines.append("kp_multiplier = 2")
    lines.append("rho_star_delta = -0.2")
    lines.append("rho_beta = 0.90")
    lines.append("alpha_min = 0.01 * alpha0")
    lines.append("alpha_max = 50 * alpha0")
    lines.append("trust_region_rho_threshold = 0.60")
    lines.append("trust_region_alpha_threshold = 3 * alpha0")
    lines.append("trust_region_expand_factor = 3")
    lines.append("trust_region_max_factor = 3")
    lines.append("```")
    lines.append("")
    lines.append(
        "The absolute per-objective settings are written to `tuned_preset_config.csv`."
    )
    lines.append("")
    lines.append("## Optimizers Compared")
    lines.append("")
    lines.append("| Optimizer | Meaning |")
    lines.append("|---|---|")
    lines.append(
        f"| Gradient descent | Raw gradient direction with fixed learning rate `{gradient_descent_alpha_multiplier:g} * alpha0`. |"
    )
    lines.append("| Vanilla Adam | Adam direction with a fixed global learning rate. |")
    lines.append(
        "| Tuned controlled Adam (raw rho) | Adam direction, tuned scale-relative alpha bounds, and alpha updates from current rho. |"
    )
    lines.append(
        "| Tuned controlled Adam (EMA rho) | Same tuned alpha bounds, but alpha updates use EMA-smoothed rho. |"
    )
    lines.append(
        "| Tuned controlled Adam (EMA + trust) | EMA-rho control plus reachable trust expansion under the tuned preset. |"
    )
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
    lines.append(
        "| Objective | Optimizer | Success Rate | Median Final Residual | Median Best Residual | Median Best Distance | Median Iterations To Success | Median Final Alpha | Median Trust Expansions |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for objective in objective_order:
        for row in aggregate_by_objective[objective]:
            lines.append(
                f"| `{objective}` | {OPTIMIZER_LABELS[str(row['optimizer'])]} | "
                f"{float(row['success_rate']):.2f} | "
                f"{float(row['median_final_residual']):.3e} | "
                f"{float(row['median_best_residual']):.3e} | "
                f"{float(row['median_best_distance']):.3e} | "
                f"{base.format_float(row['median_iterations_to_success'])} | "
                f"{float(row['median_final_alpha']):.3e} | "
                f"{base.format_float(row['median_trust_expansions'])} |"
            )
    lines.append("")
    lines.append("## Objective-Level Highlights")
    lines.append("")
    lines.extend(
        base.manager_highlight_lines(
            controlled_success_advantages,
            controlled_best_advantages,
        )
    )
    lines.append("")
    lines.append("## Key Plots")
    lines.append("")
    for path in plot_paths:
        rel = path.relative_to(output_dir)
        lines.append(f"![{rel.stem}]({rel.as_posix()})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If a tuned controlled variant beats vanilla Adam, the preset found a more useful scalar step scale on top of Adam-style directions, with fallback events preventing non-descent momentum skips."
    )
    lines.append(
        "- If gradient descent wins on a simple convex objective, that is a reminder that no adaptive direction is universally best under every finite budget."
    )
    lines.append(
        "- The tuned trust variant should be interpreted separately from earlier dormant trust reports because the threshold is now tied to `alpha0`."
    )
    lines.append(
        "- Rastrigin and Ackley remain basin-selection limitation cases; this preset improves local step behavior, not global exploration."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_start_results.csv`: one row per objective/start/optimizer.")
    lines.append("- `aggregate_results.csv`: summary by objective/optimizer.")
    lines.append("- `benchmark_config.csv`: objective settings and tolerances.")
    lines.append("- `tuned_preset_config.csv`: absolute tuned settings by objective.")
    lines.append("- `*_surface_3d.png`: 3D objective landscapes.")
    lines.append("- `*_trajectory_comparison.png`: representative trajectories.")
    lines.append("- `*_objective_curves.png`: representative objective curves.")
    lines.append("- `*_alpha_curves.png`: representative alpha schedules.")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the tuned simplified no-momentum function benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum"),
        help="Directory for CSVs, plots, and the Markdown report.",
    )
    parser.add_argument("--objectives", nargs="+", help="Optional objective names.")
    parser.add_argument("--step-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--gradient-descent-alpha-multiplier",
        type=float,
        default=1.0,
        help="Multiply each objective's base alpha0 for the fixed GD baseline only.",
    )
    parser.add_argument(
        "--random-starts-per-objective",
        type=int,
        default=25,
        help="Append this many deterministic random starts per objective.",
    )
    parser.add_argument("--random-seed", type=int, default=20260527)
    args = parser.parse_args()

    configure_report_labels()

    if args.gradient_descent_alpha_multiplier <= 0:
        raise ValueError("--gradient-descent-alpha-multiplier must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = base.filter_cases(base.benchmark_cases(), args.objectives)
    cases = base.add_random_starts(
        cases,
        args.random_starts_per_objective,
        args.random_seed,
    )
    cases = base.scale_case_steps(cases, args.step_multiplier)

    all_rows: list[base.RunSummary] = []
    plot_paths: list[Path] = []

    print("Running tuned simplified no-momentum function benchmark")
    print(f"Output directory: {output_dir.resolve()}")

    for case in cases:
        print(f"- {case.objective.name}: {len(case.starts)} starts, {case.steps} steps")
        rows, histories = run_case(case, args.gradient_descent_alpha_multiplier)
        all_rows.extend(rows)

        if case.objective.name in base.HIGHLIGHT_OBJECTIVES:
            plot_paths.append(base.plot_objective_surface_3d(case, output_dir))
            plot_paths.append(base.plot_highlight_trajectory(case, histories, output_dir))
            plot_paths.append(base.plot_highlight_objective(case, histories, output_dir))
            plot_paths.append(base.plot_highlight_alpha(case, histories, output_dir))

    aggregate = base.aggregate_rows(all_rows)

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
    write_tuned_config_csv(cases, output_dir / "tuned_preset_config.csv")

    plot_paths.insert(0, base.plot_success_rates(aggregate, output_dir))
    plot_paths.insert(1, base.plot_median_best(aggregate, output_dir))

    report_path = generate_report(
        cases,
        aggregate,
        plot_paths,
        output_dir,
        args.gradient_descent_alpha_multiplier,
    )

    print("Done.")
    print(f"Report: {report_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


if __name__ == "__main__":
    main()
