"""Sweep controlled Adam parameter variants on the deterministic function suite.

The main function report compares a fixed set of optimizers and generates many
plots. This companion script is intentionally lighter: it reuses the same
objectives, starts, success criteria, and optimizer implementation, then writes
CSV/Markdown summaries for controlled-optimizer parameter experiments.

Run from ``controlled_adam_project`` with:

    MPLCONFIGDIR=/private/tmp PYTHONPATH=src \
      python examples/run_controlled_adam_parameter_sweep.py \
      --output-dir ../outputs/controlled_adam_parameter_sweep_30runs
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from run_function_benchmark_report import (
    BenchmarkCase,
    RunSummary,
    add_random_starts,
    aggregate_rows,
    benchmark_cases,
    controlled_adam,
    controlled_adam_ema_rho,
    filter_cases,
    format_float,
    scale_case_steps,
    summarize_run,
    write_config_csv,
    write_csv,
)


@dataclass(frozen=True)
class SweepVariant:
    """One controlled Adam parameter variant."""

    name: str
    mode: str
    description: str
    kp_multiplier: float = 1.0
    rho_beta: float | None = None
    min_alpha_factor: float | None = None
    max_alpha_factor: float | None = None
    trust_region_expand: bool = False
    trust_region_rho_threshold: float | None = None
    trust_region_alpha_threshold: float | None = None
    trust_region_expand_factor: float | None = None


def sweep_variants() -> list[SweepVariant]:
    """Return controlled optimizer variants for the first trust-region sweep."""

    return [
        SweepVariant(
            name="raw_current",
            mode="raw",
            description="Current raw-rho controlled Adam.",
        ),
        SweepVariant(
            name="ema_current",
            mode="ema",
            description="Current EMA-rho controlled Adam, without per-step factor clipping.",
        ),
        SweepVariant(
            name="ema_clipped_current",
            mode="ema",
            description="EMA-rho with the same [0.5, 1.25] factor clipping used by current trust.",
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
        ),
        SweepVariant(
            name="trust_current_clipped",
            mode="ema",
            description="Current EMA+trust: rho>=0.90, alpha<=1e-4, expand x1.5, factor clipped.",
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
            trust_region_expand=True,
            trust_region_rho_threshold=0.90,
            trust_region_alpha_threshold=1e-4,
            trust_region_expand_factor=1.5,
        ),
        SweepVariant(
            name="trust_current_unclipped",
            mode="ema",
            description="Current trust thresholds without ordinary factor clipping.",
            trust_region_expand=True,
            trust_region_rho_threshold=0.90,
            trust_region_alpha_threshold=1e-4,
            trust_region_expand_factor=1.5,
        ),
        SweepVariant(
            name="trust_rho70_a1e3_x2",
            mode="ema",
            description="More permissive trust: rho>=0.70, alpha<=1e-3, expand x2.",
            trust_region_expand=True,
            trust_region_rho_threshold=0.70,
            trust_region_alpha_threshold=1e-3,
            trust_region_expand_factor=2.0,
        ),
        SweepVariant(
            name="trust_rho70_a1e3_x3",
            mode="ema",
            description="More aggressive trust: rho>=0.70, alpha<=1e-3, expand x3.",
            trust_region_expand=True,
            trust_region_rho_threshold=0.70,
            trust_region_alpha_threshold=1e-3,
            trust_region_expand_factor=3.0,
        ),
        SweepVariant(
            name="trust_rho60_a1e3_x3",
            mode="ema",
            description="Very permissive trust: rho>=0.60, alpha<=1e-3, expand x3.",
            trust_region_expand=True,
            trust_region_rho_threshold=0.60,
            trust_region_alpha_threshold=1e-3,
            trust_region_expand_factor=3.0,
        ),
        SweepVariant(
            name="trust_rho70_a1e2_x2",
            mode="ema",
            description="Wide alpha gate trust: rho>=0.70, alpha<=1e-2, expand x2.",
            trust_region_expand=True,
            trust_region_rho_threshold=0.70,
            trust_region_alpha_threshold=1e-2,
            trust_region_expand_factor=2.0,
        ),
        SweepVariant(
            name="ema_beta80",
            mode="ema",
            description="Faster rho EMA response, beta=0.80, no trust.",
            rho_beta=0.80,
        ),
        SweepVariant(
            name="ema_beta80_trust_rho70_a1e3_x2",
            mode="ema",
            description="Beta=0.80 plus permissive trust: rho>=0.70, alpha<=1e-3, expand x2.",
            rho_beta=0.80,
            trust_region_expand=True,
            trust_region_rho_threshold=0.70,
            trust_region_alpha_threshold=1e-3,
            trust_region_expand_factor=2.0,
        ),
        SweepVariant(
            name="ema_gain2",
            mode="ema",
            description="Double the proportional gain kp, no trust.",
            kp_multiplier=2.0,
        ),
        SweepVariant(
            name="ema_gain2_trust_rho70_a1e3_x2",
            mode="ema",
            description="Double kp plus permissive trust: rho>=0.70, alpha<=1e-3, expand x2.",
            kp_multiplier=2.0,
            trust_region_expand=True,
            trust_region_rho_threshold=0.70,
            trust_region_alpha_threshold=1e-3,
            trust_region_expand_factor=2.0,
        ),
    ]


def run_variant(
    case: BenchmarkCase,
    x0: np.ndarray,
    variant: SweepVariant,
):
    """Run one controlled Adam variant."""

    kp = case.kp * variant.kp_multiplier
    if variant.mode == "raw":
        return controlled_adam(
            case.objective,
            x0,
            alpha0=case.alpha,
            steps=case.steps,
            kp=kp,
            rho_star=case.rho_star,
            rho_min=case.rho_min,
            alpha_min=case.alpha_min,
            alpha_max=case.alpha_max,
            reject_bad_steps=True,
            max_backtracks=case.max_backtracks,
        )

    if variant.mode != "ema":
        raise ValueError(f"Unknown variant mode: {variant.mode}")

    return controlled_adam_ema_rho(
        case.objective,
        x0,
        alpha0=case.alpha,
        steps=case.steps,
        kp=kp,
        rho_star=case.rho_star,
        rho_beta=case.ema_beta if variant.rho_beta is None else variant.rho_beta,
        rho_min=case.rho_min,
        alpha_min=case.alpha_min,
        alpha_max=case.alpha_max,
        reject_bad_steps=True,
        max_backtracks=case.max_backtracks,
        min_alpha_factor=variant.min_alpha_factor,
        max_alpha_factor=variant.max_alpha_factor,
        trust_region_expand=variant.trust_region_expand,
        trust_region_rho_threshold=(
            case.trust_region_rho_threshold
            if variant.trust_region_rho_threshold is None
            else variant.trust_region_rho_threshold
        ),
        trust_region_alpha_threshold=(
            case.trust_region_alpha_threshold
            if variant.trust_region_alpha_threshold is None
            else variant.trust_region_alpha_threshold
        ),
        trust_region_expand_factor=(
            case.trust_region_expand_factor
            if variant.trust_region_expand_factor is None
            else variant.trust_region_expand_factor
        ),
        trust_region_max_factor=(
            case.trust_region_expand_factor
            if variant.trust_region_expand_factor is None
            else variant.trust_region_expand_factor
        ),
    )


def run_sweep(
    cases: list[BenchmarkCase],
    variants: list[SweepVariant],
) -> list[RunSummary]:
    """Run every variant on every start in every case."""

    rows: list[RunSummary] = []
    for case in cases:
        print(f"- {case.objective.name}: {len(case.starts)} starts, {case.steps} steps")
        for start_id, x0 in enumerate(case.starts):
            for variant in variants:
                history = run_variant(case, x0, variant)
                rows.append(summarize_run(case, start_id, variant.name, x0, history))
    return rows


def write_variant_config(variants: list[SweepVariant], path: Path) -> None:
    """Write variant settings to CSV."""

    fieldnames = list(SweepVariant.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            writer.writerow({field: getattr(variant, field) for field in fieldnames})


def finite_median(values: Iterable[float]) -> float:
    """Return median of finite values, or NaN."""

    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def winner_counts(
    aggregate: list[dict[str, object]],
    metric: str,
    lower_is_better: bool,
) -> dict[str, int]:
    """Count objective-level winners, counting ties for every tied variant."""

    counts = {str(row["optimizer"]): 0 for row in aggregate}
    for objective in sorted({str(row["objective"]) for row in aggregate}):
        rows = [row for row in aggregate if row["objective"] == objective]
        values = [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))]
        if not values:
            continue
        target = min(values) if lower_is_better else max(values)
        for row in rows:
            value = float(row[metric])
            if np.isfinite(value) and math.isclose(
                value, target, rel_tol=1e-12, abs_tol=1e-12
            ):
                counts[str(row["optimizer"])] += 1
    return counts


def format_metric(value: object) -> str:
    """Format numeric metrics for Markdown."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "-"
    if v == 0.0:
        return "0"
    if abs(v) >= 1000.0 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.4g}"


def write_summary(
    cases: list[BenchmarkCase],
    variants: list[SweepVariant],
    aggregate: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    """Write a compact Markdown summary of the sweep."""

    path = output_dir / "CONTROLLED_ADAM_PARAMETER_SWEEP.md"
    variant_order = [variant.name for variant in variants]
    objective_order = [case.objective.name for case in cases]
    rows_by_objective = {
        objective: [
            row for row in aggregate if str(row["objective"]) == objective
        ]
        for objective in objective_order
    }
    for rows in rows_by_objective.values():
        rows.sort(key=lambda row: variant_order.index(str(row["optimizer"])))

    success_winners = winner_counts(aggregate, "success_rate", lower_is_better=False)
    final_winners = winner_counts(
        aggregate, "median_final_residual", lower_is_better=True
    )
    best_winners = winner_counts(
        aggregate, "median_best_residual", lower_is_better=True
    )

    lines: list[str] = []
    lines.append("# Controlled Adam Parameter Sweep")
    lines.append("")
    lines.append("This sweep reuses the deterministic function benchmark cases and 30-start setup.")
    lines.append("It tests whether the current controlled Adam parameters, especially EMA+trust, are under-tuned.")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| Variant | Description |")
    lines.append("|---|---|")
    for variant in variants:
        lines.append(f"| `{variant.name}` | {variant.description} |")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Variant | Avg Success | Success Wins | Final Residual Wins | Best Residual Wins | Median Trust Expansions |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in variants:
        rows = [row for row in aggregate if row["optimizer"] == variant.name]
        avg_success = float(np.mean([float(row["success_rate"]) for row in rows]))
        median_expansions = finite_median(row["median_trust_expansions"] for row in rows)
        lines.append(
            f"| `{variant.name}` | {avg_success:.3f} | "
            f"{success_winners.get(variant.name, 0)} | "
            f"{final_winners.get(variant.name, 0)} | "
            f"{best_winners.get(variant.name, 0)} | "
            f"{format_metric(median_expansions)} |"
        )
    lines.append("")
    lines.append("## Per-Objective Results")
    lines.append("")
    lines.append("Each cell is `success rate / median final residual / median trust expansions`.")
    lines.append("")
    lines.append("| Function | " + " | ".join(f"`{variant.name}`" for variant in variants) + " |")
    lines.append("|---|" + "|".join("---:" for _ in variants) + "|")
    for objective in objective_order:
        cells = []
        for row in rows_by_objective[objective]:
            cells.append(
                f"{100.0 * float(row['success_rate']):.0f}% / "
                f"{format_metric(row['median_final_residual'])} / "
                f"{format_metric(row['median_trust_expansions'])}"
            )
        lines.append(f"| `{objective}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_start_results.csv`: one row per objective/start/variant.")
    lines.append("- `aggregate_results.csv`: aggregate metrics by objective/variant.")
    lines.append("- `benchmark_config.csv`: objective settings and success tolerances.")
    lines.append("- `variant_config.csv`: tested controlled-optimizer parameter variants.")
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled Adam parameter sweep on deterministic functions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/controlled_adam_parameter_sweep_30runs"),
        help="Directory for sweep CSVs and Markdown summary.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        help="Optional objective names to include.",
    )
    parser.add_argument(
        "--random-starts-per-objective",
        type=int,
        default=25,
        help="Append this many deterministic random starts to each selected objective.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260527,
        help="Seed used when --random-starts-per-objective is positive.",
    )
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=1.0,
        help="Multiply each selected objective's default iteration budget.",
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
    variants = sweep_variants()

    print("Running controlled Adam parameter sweep")
    print(f"Output directory: {output_dir.resolve()}")
    rows = run_sweep(cases, variants)
    aggregate = aggregate_rows(rows)

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

    write_csv(output_dir / "per_start_results.csv", rows, per_start_fields)
    write_csv(output_dir / "aggregate_results.csv", aggregate, aggregate_fields)
    write_config_csv(cases, output_dir / "benchmark_config.csv")
    write_variant_config(variants, output_dir / "variant_config.csv")
    summary_path = write_summary(cases, variants, aggregate, output_dir)
    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


if __name__ == "__main__":
    main()
