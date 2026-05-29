"""Refine controlled Adam tuning around the best coarse-sweep regions.

This script follows ``run_controlled_adam_tuning_sweep.py`` but concentrates on
the neighborhoods that improved the deterministic function benchmark:

- larger ``alpha_min`` floors and stronger proportional gain for raw-rho;
- larger ``alpha_min`` floors and stronger proportional gain for EMA-rho;
- wider trust expansion gates for EMA-rho with trust-style expansion.

It writes CSV tables plus a compact Markdown summary that compares each family
against its current parameterization.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from run_controlled_adam_tuning_sweep import (
    TuningVariant,
    finite_median,
    format_metric,
    run_sweep,
    write_variant_config,
)
from run_function_benchmark_report import (
    RunSummary,
    add_random_starts,
    aggregate_rows,
    benchmark_cases,
    filter_cases,
    scale_case_steps,
    write_config_csv,
    write_csv,
)


CURRENT_BY_FAMILY = {
    "raw": "raw_current",
    "ema": "ema_current",
    "trust": "trust_current",
}


def value_tag(value: float) -> str:
    """Return a stable short token for parameterized variant names."""

    return f"{value:g}".replace("-", "m").replace("+", "p").replace(".", "p")


def add_unique(
    variants: list[TuningVariant],
    seen: set[str],
    variant: TuningVariant,
) -> None:
    """Append a variant unless a previous generated name already used it."""

    if variant.name in seen:
        return
    seen.add(variant.name)
    variants.append(variant)


def refined_tuning_variants() -> list[TuningVariant]:
    """Return a focused tuning grid around the coarse-sweep winners."""

    variants: list[TuningVariant] = []
    seen: set[str] = set()

    for variant in [
        TuningVariant("raw_current", "raw", "Current raw-rho controlled Adam."),
        TuningVariant("ema_current", "ema", "Current EMA-rho controlled Adam."),
        TuningVariant(
            "trust_current",
            "trust",
            "Current EMA+trust: rho>=0.90, alpha<=1e-4, expand x1.5.",
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
            trust_region_rho_threshold=0.90,
            trust_region_alpha_threshold=1e-4,
            trust_region_expand_factor=1.5,
        ),
    ]:
        add_unique(variants, seen, variant)

    alpha_floors = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]
    focused_alpha_floors = [1e-6, 3e-6, 1e-5]

    for alpha_min in alpha_floors:
        add_unique(
            variants,
            seen,
            TuningVariant(
                f"raw_amin{value_tag(alpha_min)}",
                "raw",
                f"Raw-rho with alpha_min={alpha_min:g}.",
                alpha_min=alpha_min,
            ),
        )
    for kp in [1.5, 2.0, 2.5]:
        for alpha_min in focused_alpha_floors:
            add_unique(
                variants,
                seen,
                TuningVariant(
                    f"raw_kp{value_tag(kp)}_amin{value_tag(alpha_min)}",
                    "raw",
                    f"Raw-rho with kp x{kp:g} and alpha_min={alpha_min:g}.",
                    kp_multiplier=kp,
                    alpha_min=alpha_min,
                ),
            )
    for delta in [-0.2, -0.1]:
        for kp in [1.0, 2.0]:
            for alpha_min in [1e-6, 1e-5]:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"raw_rhostar{value_tag(delta)}_kp{value_tag(kp)}_"
                            f"amin{value_tag(alpha_min)}"
                        ),
                        "raw",
                        (
                            f"Raw-rho with rho_star shifted by {delta:+.1f}, "
                            f"kp x{kp:g}, alpha_min={alpha_min:g}."
                        ),
                        rho_star_delta=delta,
                        kp_multiplier=kp,
                        alpha_min=alpha_min,
                    ),
                )

    for alpha_min in alpha_floors:
        add_unique(
            variants,
            seen,
            TuningVariant(
                f"ema_amin{value_tag(alpha_min)}",
                "ema",
                f"EMA-rho with alpha_min={alpha_min:g}.",
                alpha_min=alpha_min,
            ),
        )
    for kp in [1.5, 2.0, 2.5]:
        for alpha_min in focused_alpha_floors:
            add_unique(
                variants,
                seen,
                TuningVariant(
                    f"ema_kp{value_tag(kp)}_amin{value_tag(alpha_min)}",
                    "ema",
                    f"EMA-rho with kp x{kp:g} and alpha_min={alpha_min:g}.",
                    kp_multiplier=kp,
                    alpha_min=alpha_min,
                ),
            )
    for beta in [0.80, 0.95]:
        for kp in [1.0, 2.0]:
            for alpha_min in [1e-6, 1e-5]:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"ema_beta{int(beta * 100)}_kp{value_tag(kp)}_"
                            f"amin{value_tag(alpha_min)}"
                        ),
                        "ema",
                        (
                            f"EMA-rho with rho_beta={beta:.2f}, kp x{kp:g}, "
                            f"alpha_min={alpha_min:g}."
                        ),
                        rho_beta=beta,
                        kp_multiplier=kp,
                        alpha_min=alpha_min,
                    ),
                )
    for delta in [-0.2, -0.1]:
        for kp in [1.0, 2.0]:
            for alpha_min in [1e-6, 1e-5]:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"ema_rhostar{value_tag(delta)}_kp{value_tag(kp)}_"
                            f"amin{value_tag(alpha_min)}"
                        ),
                        "ema",
                        (
                            f"EMA-rho with rho_star shifted by {delta:+.1f}, "
                            f"kp x{kp:g}, alpha_min={alpha_min:g}."
                        ),
                        rho_star_delta=delta,
                        kp_multiplier=kp,
                        alpha_min=alpha_min,
                    ),
                )

    for alpha_threshold in [5e-3, 1e-2, 2e-2]:
        for rho_threshold in [0.60, 0.70, 0.80]:
            for expand_factor in [1.5, 2.0, 2.5, 3.0]:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"trust_rho{int(rho_threshold * 100)}_"
                            f"a{value_tag(alpha_threshold)}_x{value_tag(expand_factor)}"
                        ),
                        "trust",
                        (
                            f"Trust with rho>={rho_threshold:.2f}, "
                            f"alpha<={alpha_threshold:g}, expand x{expand_factor:g}."
                        ),
                        trust_region_rho_threshold=rho_threshold,
                        trust_region_alpha_threshold=alpha_threshold,
                        trust_region_expand_factor=expand_factor,
                    ),
                )
    for rho_threshold in [0.60, 0.70]:
        for expand_factor in [2.0, 2.5, 3.0]:
            for alpha_min in focused_alpha_floors:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"trust_rho{int(rho_threshold * 100)}_a0p01_"
                            f"x{value_tag(expand_factor)}_amin{value_tag(alpha_min)}"
                        ),
                        "trust",
                        (
                            f"Trust with rho>={rho_threshold:.2f}, alpha<=0.01, "
                            f"expand x{expand_factor:g}, alpha_min={alpha_min:g}."
                        ),
                        alpha_min=alpha_min,
                        trust_region_rho_threshold=rho_threshold,
                        trust_region_alpha_threshold=1e-2,
                        trust_region_expand_factor=expand_factor,
                    ),
                )
    for rho_threshold in [0.60, 0.70]:
        for kp in [1.5, 2.0]:
            for alpha_min in [1e-6, 1e-5]:
                add_unique(
                    variants,
                    seen,
                    TuningVariant(
                        (
                            f"trust_rho{int(rho_threshold * 100)}_a0p01_x2_"
                            f"kp{value_tag(kp)}_amin{value_tag(alpha_min)}"
                        ),
                        "trust",
                        (
                            f"Trust with rho>={rho_threshold:.2f}, alpha<=0.01, "
                            f"expand x2, kp x{kp:g}, alpha_min={alpha_min:g}."
                        ),
                        kp_multiplier=kp,
                        alpha_min=alpha_min,
                        trust_region_rho_threshold=rho_threshold,
                        trust_region_alpha_threshold=1e-2,
                        trust_region_expand_factor=2.0,
                    ),
                )
    for beta in [0.80, 0.95]:
        for alpha_min in [1e-6, 1e-5]:
            add_unique(
                variants,
                seen,
                TuningVariant(
                    f"trust_beta{int(beta * 100)}_rho70_a0p01_x2_amin{value_tag(alpha_min)}",
                    "trust",
                    (
                        f"Trust with beta={beta:.2f}, rho>=0.70, alpha<=0.01, "
                        f"expand x2, alpha_min={alpha_min:g}."
                    ),
                    rho_beta=beta,
                    alpha_min=alpha_min,
                    trust_region_rho_threshold=0.70,
                    trust_region_alpha_threshold=1e-2,
                    trust_region_expand_factor=2.0,
                ),
            )

    return variants


def mean_log10_best(rows: Iterable[dict[str, object]]) -> float:
    """Return the mean log10 median-best residual across objective rows."""

    values = [
        math.log10(max(float(row["median_best_residual"]), 1e-12))
        for row in rows
    ]
    return float(np.mean(values)) if values else float("nan")


def average_success(rows: Iterable[dict[str, object]]) -> float:
    """Return average objective-level success rate."""

    values = [float(row["success_rate"]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def median_trust_expansions(rows: Iterable[dict[str, object]]) -> float:
    """Return median of objective-level median trust expansion counts."""

    return finite_median(row["median_trust_expansions"] for row in rows)


def variant_score(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Rank by success first, then by lower residual."""

    return (round(average_success(rows), 12), mean_log10_best(rows))


def rows_for_variant(
    aggregate: list[dict[str, object]],
    variant_name: str,
) -> list[dict[str, object]]:
    """Return aggregate rows for one variant."""

    return [row for row in aggregate if row["optimizer"] == variant_name]


def best_by_family(
    variants: list[TuningVariant],
    aggregate: list[dict[str, object]],
) -> dict[str, tuple[TuningVariant, list[dict[str, object]]]]:
    """Return the best average-success variant for each controlled family."""

    best: dict[str, tuple[TuningVariant, list[dict[str, object]]]] = {}
    for family in sorted({variant.family for variant in variants}):
        candidates = []
        for variant in variants:
            if variant.family != family:
                continue
            rows = rows_for_variant(aggregate, variant.name)
            candidates.append((variant_score(rows), variant, rows))
        _, variant, rows = min(candidates, key=lambda item: (-item[0][0], item[0][1]))
        best[family] = (variant, rows)
    return best


def family_rankings(
    variants: list[TuningVariant],
    aggregate: list[dict[str, object]],
    family: str,
    limit: int = 8,
) -> list[tuple[TuningVariant, list[dict[str, object]]]]:
    """Return top variants within a family."""

    candidates = [
        (variant_score(rows_for_variant(aggregate, variant.name)), variant)
        for variant in variants
        if variant.family == family
    ]
    ranked = sorted(candidates, key=lambda item: (-item[0][0], item[0][1]))
    return [(variant, rows_for_variant(aggregate, variant.name)) for _, variant in ranked[:limit]]


def objective_best_rows(
    variants: list[TuningVariant],
    aggregate: list[dict[str, object]],
) -> list[tuple[str, str, str, dict[str, object]]]:
    """Return the best variant row for each objective and family."""

    result = []
    objectives = sorted({str(row["objective"]) for row in aggregate})
    for objective in objectives:
        for family in ["raw", "ema", "trust"]:
            family_names = {variant.name for variant in variants if variant.family == family}
            rows = [
                row
                for row in aggregate
                if row["objective"] == objective and row["optimizer"] in family_names
            ]
            best = max(
                rows,
                key=lambda row: (
                    float(row["success_rate"]),
                    -float(row["median_best_residual"]),
                ),
            )
            result.append((objective, family, str(best["optimizer"]), best))
    return result


def write_refined_summary(
    variants: list[TuningVariant],
    aggregate: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    """Write the refinement summary."""

    path = output_dir / "CONTROLLED_ADAM_REFINED_TUNING_SWEEP.md"
    best = best_by_family(variants, aggregate)

    lines: list[str] = []
    lines.append("# Controlled Adam Refined Tuning Sweep")
    lines.append("")
    lines.append(
        "This refinement sweep tests parameter neighborhoods suggested by the coarse controlled Adam tuning sweep."
    )
    lines.append("")
    lines.append("## Best Tuned vs Current")
    lines.append("")
    lines.append(
        "| Family | Current Variant | Current Avg Success | Best Tuned Variant | Tuned Avg Success | Delta | Current Mean Log10 Best Residual | Tuned Mean Log10 Best Residual | Tuned Median Trust Expansions |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for family in ["raw", "ema", "trust"]:
        current_name = CURRENT_BY_FAMILY[family]
        current_rows = rows_for_variant(aggregate, current_name)
        tuned_variant, tuned_rows = best[family]
        current_success = average_success(current_rows)
        tuned_success = average_success(tuned_rows)
        lines.append(
            f"| {family} | `{current_name}` | {current_success:.3f} | "
            f"`{tuned_variant.name}` | {tuned_success:.3f} | "
            f"{tuned_success - current_success:+.3f} | "
            f"{mean_log10_best(current_rows):.3f} | "
            f"{mean_log10_best(tuned_rows):.3f} | "
            f"{format_metric(median_trust_expansions(tuned_rows))} |"
        )

    lines.append("")
    lines.append("## Top Variants by Family")
    for family in ["raw", "ema", "trust"]:
        lines.append("")
        lines.append(f"### {family}")
        lines.append("")
        lines.append(
            "| Variant | Avg Success | Mean Log10 Best Residual | Median Trust Expansions |"
        )
        lines.append("|---|---:|---:|---:|")
        for variant, rows in family_rankings(variants, aggregate, family):
            lines.append(
                f"| `{variant.name}` | {average_success(rows):.3f} | "
                f"{mean_log10_best(rows):.3f} | "
                f"{format_metric(median_trust_expansions(rows))} |"
            )

    lines.append("")
    lines.append("## Per-Objective Best by Family")
    lines.append("")
    lines.append(
        "| Objective | Family | Best Variant | Success Rate | Median Best Residual | Median Final Alpha | Median Trust Expansions |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for objective, family, variant_name, row in objective_best_rows(variants, aggregate):
        lines.append(
            f"| {objective} | {family} | `{variant_name}` | "
            f"{float(row['success_rate']):.3f} | "
            f"{format_metric(row['median_best_residual'])} | "
            f"{format_metric(row['median_final_alpha'])} | "
            f"{format_metric(row['median_trust_expansions'])} |"
        )

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
        description="Run a refined controlled Adam tuning sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/controlled_adam_refined_tuning_sweep_30runs"),
        help="Directory for refined tuning CSVs and Markdown summary.",
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
    variants = refined_tuning_variants()

    print("Running refined controlled Adam tuning sweep")
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
    summary_path = write_refined_summary(variants, aggregate, output_dir)
    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


if __name__ == "__main__":
    main()
