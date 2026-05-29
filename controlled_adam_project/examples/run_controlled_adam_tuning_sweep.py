"""Tune controlled Adam variants on the deterministic function benchmark.

This script is a broader follow-up to ``run_controlled_adam_parameter_sweep.py``.
It tunes the three controlled Adam families separately:

- raw-rho controlled Adam;
- EMA-rho controlled Adam;
- EMA-rho with trust-style expansion.

It deliberately skips plot generation so a larger parameter grid can be run on
the same 30-start deterministic function suite.
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
    scale_case_steps,
    summarize_run,
    write_config_csv,
    write_csv,
)


@dataclass(frozen=True)
class TuningVariant:
    """One controlled Adam tuning candidate."""

    name: str
    family: str
    description: str
    kp_multiplier: float = 1.0
    rho_star_delta: float = 0.0
    rho_beta: float | None = None
    alpha_min: float | None = None
    alpha_max_multiplier: float = 1.0
    min_alpha_factor: float | None = None
    max_alpha_factor: float | None = None
    trust_region_rho_threshold: float | None = None
    trust_region_alpha_threshold: float | None = None
    trust_region_expand_factor: float | None = None


def tuning_variants() -> list[TuningVariant]:
    """Return a moderate grid for the three controlled Adam families."""

    variants: list[TuningVariant] = [
        TuningVariant("raw_current", "raw", "Current raw-rho controlled Adam."),
        TuningVariant("ema_current", "ema", "Current EMA-rho controlled Adam."),
        TuningVariant(
            "trust_current",
            "trust",
            "Current EMA+trust settings: rho>=0.90, alpha<=1e-4, expand x1.5.",
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
            trust_region_rho_threshold=0.90,
            trust_region_alpha_threshold=1e-4,
            trust_region_expand_factor=1.5,
        ),
    ]

    for kp in [0.5, 1.5, 2.0, 3.0]:
        variants.append(
            TuningVariant(
                f"raw_kp{kp:g}",
                "raw",
                f"Raw-rho with kp multiplier {kp:g}.",
                kp_multiplier=kp,
            )
        )
    for delta in [-0.2, -0.1, 0.1]:
        variants.append(
            TuningVariant(
                f"raw_rhostar{delta:+.1f}",
                "raw",
                f"Raw-rho with rho_star shifted by {delta:+.1f}.",
                rho_star_delta=delta,
            )
        )
    for alpha_min in [1e-7, 1e-6, 1e-5]:
        variants.append(
            TuningVariant(
                f"raw_amin{alpha_min:g}",
                "raw",
                f"Raw-rho with alpha_min={alpha_min:g}.",
                alpha_min=alpha_min,
            )
        )
    variants.extend(
        [
            TuningVariant(
                "raw_kp1.5_rhostar-0.1",
                "raw",
                "Raw-rho with kp x1.5 and rho_star shifted down by 0.1.",
                kp_multiplier=1.5,
                rho_star_delta=-0.1,
            ),
            TuningVariant(
                "raw_kp2_amin1e-6",
                "raw",
                "Raw-rho with kp x2 and alpha_min=1e-6.",
                kp_multiplier=2.0,
                alpha_min=1e-6,
            ),
            TuningVariant(
                "raw_alpha_max2",
                "raw",
                "Raw-rho with alpha_max doubled.",
                alpha_max_multiplier=2.0,
            ),
        ]
    )

    for beta in [0.95, 0.80, 0.70, 0.50]:
        variants.append(
            TuningVariant(
                f"ema_beta{int(beta * 100)}",
                "ema",
                f"EMA-rho with rho_beta={beta:.2f}.",
                rho_beta=beta,
            )
        )
    for kp in [1.5, 2.0, 3.0]:
        variants.append(
            TuningVariant(
                f"ema_kp{kp:g}",
                "ema",
                f"EMA-rho with kp multiplier {kp:g}.",
                kp_multiplier=kp,
            )
        )
    for delta in [-0.2, -0.1, 0.1]:
        variants.append(
            TuningVariant(
                f"ema_rhostar{delta:+.1f}",
                "ema",
                f"EMA-rho with rho_star shifted by {delta:+.1f}.",
                rho_star_delta=delta,
            )
        )
    for alpha_min in [1e-7, 1e-6, 1e-5]:
        variants.append(
            TuningVariant(
                f"ema_amin{alpha_min:g}",
                "ema",
                f"EMA-rho with alpha_min={alpha_min:g}.",
                alpha_min=alpha_min,
            )
        )
    variants.extend(
        [
            TuningVariant(
                "ema_beta80_kp1.5",
                "ema",
                "EMA-rho with beta=0.80 and kp x1.5.",
                rho_beta=0.80,
                kp_multiplier=1.5,
            ),
            TuningVariant(
                "ema_beta80_rhostar-0.1",
                "ema",
                "EMA-rho with beta=0.80 and rho_star shifted down by 0.1.",
                rho_beta=0.80,
                rho_star_delta=-0.1,
            ),
            TuningVariant(
                "ema_kp2_amin1e-6",
                "ema",
                "EMA-rho with kp x2 and alpha_min=1e-6.",
                kp_multiplier=2.0,
                alpha_min=1e-6,
            ),
        ]
    )

    trust_grid = [
        (0.70, 1e-3, 2.0),
        (0.70, 1e-3, 3.0),
        (0.60, 1e-3, 2.0),
        (0.60, 1e-3, 3.0),
        (0.80, 1e-3, 2.0),
        (0.70, 3e-3, 2.0),
        (0.70, 5e-3, 2.0),
        (0.70, 1e-2, 1.5),
        (0.70, 1e-2, 2.0),
        (0.70, 1e-2, 3.0),
        (0.60, 1e-2, 2.0),
        (0.80, 1e-2, 2.0),
    ]
    for rho_threshold, alpha_threshold, expand_factor in trust_grid:
        variants.append(
            TuningVariant(
                (
                    f"trust_rho{int(rho_threshold * 100)}_"
                    f"a{alpha_threshold:g}_x{expand_factor:g}"
                ),
                "trust",
                (
                    f"Trust with rho>={rho_threshold:.2f}, "
                    f"alpha<={alpha_threshold:g}, expand x{expand_factor:g}."
                ),
                trust_region_rho_threshold=rho_threshold,
                trust_region_alpha_threshold=alpha_threshold,
                trust_region_expand_factor=expand_factor,
            )
        )
    variants.extend(
        [
            TuningVariant(
                "trust_beta80_rho70_a0.003_x2",
                "trust",
                "Trust with beta=0.80, rho>=0.70, alpha<=0.003, expand x2.",
                rho_beta=0.80,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=3e-3,
                trust_region_expand_factor=2.0,
            ),
            TuningVariant(
                "trust_beta80_rho70_a0.01_x2",
                "trust",
                "Trust with beta=0.80, rho>=0.70, alpha<=0.01, expand x2.",
                rho_beta=0.80,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=1e-2,
                trust_region_expand_factor=2.0,
            ),
            TuningVariant(
                "trust_kp1.5_rho70_a0.003_x2",
                "trust",
                "Trust with kp x1.5, rho>=0.70, alpha<=0.003, expand x2.",
                kp_multiplier=1.5,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=3e-3,
                trust_region_expand_factor=2.0,
            ),
            TuningVariant(
                "trust_kp1.5_rho70_a0.01_x2",
                "trust",
                "Trust with kp x1.5, rho>=0.70, alpha<=0.01, expand x2.",
                kp_multiplier=1.5,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=1e-2,
                trust_region_expand_factor=2.0,
            ),
            TuningVariant(
                "trust_rho70_a0.01_x2_amin1e-7",
                "trust",
                "Wide trust with alpha_min=1e-7.",
                alpha_min=1e-7,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=1e-2,
                trust_region_expand_factor=2.0,
            ),
            TuningVariant(
                "trust_rho70_a0.01_x2_amin1e-6",
                "trust",
                "Wide trust with alpha_min=1e-6.",
                alpha_min=1e-6,
                trust_region_rho_threshold=0.70,
                trust_region_alpha_threshold=1e-2,
                trust_region_expand_factor=2.0,
            ),
        ]
    )
    return variants


def bounded_rho_star(case: BenchmarkCase, delta: float) -> float:
    """Return a rho_star shifted but kept in a reasonable positive range."""

    return float(np.clip(case.rho_star + delta, 0.05, 0.95))


def run_variant(case: BenchmarkCase, x0: np.ndarray, variant: TuningVariant):
    """Run one tuning variant."""

    kp = case.kp * variant.kp_multiplier
    rho_star = bounded_rho_star(case, variant.rho_star_delta)
    alpha_min = case.alpha_min if variant.alpha_min is None else variant.alpha_min
    alpha_max = case.alpha_max * variant.alpha_max_multiplier

    if variant.family == "raw":
        return controlled_adam(
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

    if variant.family not in {"ema", "trust"}:
        raise ValueError(f"Unknown variant family: {variant.family}")

    trust_region_expand = variant.family == "trust"
    return controlled_adam_ema_rho(
        case.objective,
        x0,
        alpha0=case.alpha,
        steps=case.steps,
        kp=kp,
        rho_star=rho_star,
        rho_beta=case.ema_beta if variant.rho_beta is None else variant.rho_beta,
        rho_min=case.rho_min,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        reject_bad_steps=True,
        max_backtracks=case.max_backtracks,
        min_alpha_factor=variant.min_alpha_factor,
        max_alpha_factor=variant.max_alpha_factor,
        trust_region_expand=trust_region_expand,
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
    )


def run_sweep(cases: list[BenchmarkCase], variants: list[TuningVariant]) -> list[RunSummary]:
    """Run the tuning grid."""

    rows: list[RunSummary] = []
    for case in cases:
        print(f"- {case.objective.name}: {len(case.starts)} starts, {case.steps} steps")
        for start_id, x0 in enumerate(case.starts):
            for variant in variants:
                history = run_variant(case, x0, variant)
                rows.append(summarize_run(case, start_id, variant.name, x0, history))
    return rows


def write_variant_config(variants: list[TuningVariant], path: Path) -> None:
    """Write variant settings to CSV."""

    fieldnames = list(TuningVariant.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            writer.writerow({field: getattr(variant, field) for field in fieldnames})


def finite_median(values: Iterable[object]) -> float:
    """Return median of finite values, or NaN."""

    finite = []
    for value in values:
        v = float(value)
        if np.isfinite(v):
            finite.append(v)
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
    variants: list[TuningVariant],
    aggregate: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    """Write a compact tuning summary."""

    path = output_dir / "CONTROLLED_ADAM_TUNING_SWEEP.md"
    success_winners = winner_counts(aggregate, "success_rate", lower_is_better=False)
    final_winners = winner_counts(
        aggregate, "median_final_residual", lower_is_better=True
    )
    best_winners = winner_counts(
        aggregate, "median_best_residual", lower_is_better=True
    )

    lines: list[str] = []
    lines.append("# Controlled Adam Tuning Sweep")
    lines.append("")
    lines.append("This sweep tunes raw-rho, EMA-rho, and EMA+trust controlled Adam variants on the 30-start deterministic function suite.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Variant | Family | Avg Success | Success Wins | Final Residual Wins | Best Residual Wins | Mean Log10 Best Residual | Median Trust Expansions |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        rows = [row for row in aggregate if row["optimizer"] == variant.name]
        avg_success = float(np.mean([float(row["success_rate"]) for row in rows]))
        mean_log_best = float(
            np.mean(
                [
                    np.log10(max(float(row["median_best_residual"]), 1e-12))
                    for row in rows
                ]
            )
        )
        median_expansions = finite_median(row["median_trust_expansions"] for row in rows)
        lines.append(
            f"| `{variant.name}` | {variant.family} | {avg_success:.3f} | "
            f"{success_winners.get(variant.name, 0)} | "
            f"{final_winners.get(variant.name, 0)} | "
            f"{best_winners.get(variant.name, 0)} | "
            f"{mean_log_best:.3f} | {format_metric(median_expansions)} |"
        )
    lines.append("")
    lines.append("## Variant Config")
    lines.append("")
    lines.append("| Variant | Description |")
    lines.append("|---|---|")
    for variant in variants:
        lines.append(f"| `{variant.name}` | {variant.description} |")
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
        description="Run a broader controlled Adam tuning sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/controlled_adam_tuning_sweep_30runs"),
        help="Directory for tuning CSVs and Markdown summary.",
    )
    parser.add_argument("--objectives", nargs="+", help="Optional objective names.")
    parser.add_argument(
        "--random-starts-per-objective",
        type=int,
        default=25,
        help="Append this many deterministic random starts per selected objective.",
    )
    parser.add_argument("--random-seed", type=int, default=20260527)
    parser.add_argument("--step-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = filter_cases(benchmark_cases(), args.objectives)
    cases = add_random_starts(cases, args.random_starts_per_objective, args.random_seed)
    cases = scale_case_steps(cases, args.step_multiplier)
    variants = tuning_variants()

    print("Running controlled Adam tuning sweep")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Variants: {len(variants)}")
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
    summary_path = write_summary(variants, aggregate, output_dir)
    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


if __name__ == "__main__":
    main()
