"""Tune a simplified controlled Adam parameterization.

This sweep is intentionally lower-dimensional than the broad/refined raw
parameter sweeps. It tests a practical interface made from:

- optimizer family: raw-rho, EMA-rho, or EMA+trust;
- response preset: conservative, balanced, or aggressive;
- alpha-range preset: alpha_min, alpha_max, and trust threshold derived from
  the objective's base alpha0.

The goal is to see whether most of the tuned benefit can be recovered without
exposing many independent hyperparameters to the user.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from run_controlled_adam_tuning_sweep import (
    bounded_rho_star,
    finite_median,
    format_metric,
)
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


CURRENT_BY_FAMILY = {
    "raw": "raw_current",
    "ema": "ema_current",
    "trust": "trust_current",
}


@dataclass(frozen=True)
class ResponsePreset:
    """Low-dimensional response setting for the controller."""

    name: str
    kp_multiplier: float
    rho_star_delta: float
    rho_beta: float
    trust_region_rho_threshold: float
    trust_region_expand_factor: float
    description: str


@dataclass(frozen=True)
class AlphaRangePreset:
    """Alpha bounds and trust threshold as factors of alpha0."""

    name: str
    alpha_min_factor: float
    alpha_max_factor: float
    trust_alpha_threshold_factor: float
    description: str


@dataclass(frozen=True)
class SimplifiedVariant:
    """One simplified controlled Adam preset variant."""

    name: str
    family: str
    response_preset: str
    alpha_preset: str
    description: str
    kp_multiplier: float = 1.0
    rho_star_delta: float = 0.0
    rho_beta: float | None = None
    alpha_min_factor: float | None = None
    alpha_max_factor: float | None = None
    trust_region_rho_threshold: float | None = None
    trust_alpha_threshold_factor: float | None = None
    trust_region_expand_factor: float | None = None
    min_alpha_factor: float | None = None
    max_alpha_factor: float | None = None


def response_presets() -> list[ResponsePreset]:
    """Return the small response-preset grid."""

    return [
        ResponsePreset(
            name="conservative",
            kp_multiplier=1.0,
            rho_star_delta=0.0,
            rho_beta=0.95,
            trust_region_rho_threshold=0.80,
            trust_region_expand_factor=1.5,
            description="Original target, original gain, slower rho EMA, gentle trust expansion.",
        ),
        ResponsePreset(
            name="balanced",
            kp_multiplier=1.5,
            rho_star_delta=-0.1,
            rho_beta=0.90,
            trust_region_rho_threshold=0.70,
            trust_region_expand_factor=2.0,
            description="Moderately lower target and stronger response.",
        ),
        ResponsePreset(
            name="aggressive",
            kp_multiplier=2.0,
            rho_star_delta=-0.2,
            rho_beta=0.90,
            trust_region_rho_threshold=0.60,
            trust_region_expand_factor=3.0,
            description="Lower target, stronger response, and more permissive trust expansion.",
        ),
    ]


def alpha_range_presets() -> list[AlphaRangePreset]:
    """Return alpha-range presets expressed relative to each case's alpha0."""

    return [
        AlphaRangePreset(
            name="low_floor",
            alpha_min_factor=1e-3,
            alpha_max_factor=25.0,
            trust_alpha_threshold_factor=1.0,
            description="alpha_min=0.001*alpha0, alpha_max=25*alpha0, trust threshold=alpha0.",
        ),
        AlphaRangePreset(
            name="mid_floor",
            alpha_min_factor=3e-3,
            alpha_max_factor=25.0,
            trust_alpha_threshold_factor=2.0,
            description="alpha_min=0.003*alpha0, alpha_max=25*alpha0, trust threshold=2*alpha0.",
        ),
        AlphaRangePreset(
            name="wide_cap",
            alpha_min_factor=3e-3,
            alpha_max_factor=50.0,
            trust_alpha_threshold_factor=3.0,
            description="alpha_min=0.003*alpha0, alpha_max=50*alpha0, trust threshold=3*alpha0.",
        ),
        AlphaRangePreset(
            name="high_floor",
            alpha_min_factor=1e-2,
            alpha_max_factor=50.0,
            trust_alpha_threshold_factor=3.0,
            description="alpha_min=0.01*alpha0, alpha_max=50*alpha0, trust threshold=3*alpha0.",
        ),
    ]


def simplified_variants() -> list[SimplifiedVariant]:
    """Return current baselines plus simplified preset variants."""

    variants = [
        SimplifiedVariant(
            "raw_current",
            "raw",
            "current",
            "current",
            "Current raw-rho controlled Adam.",
        ),
        SimplifiedVariant(
            "ema_current",
            "ema",
            "current",
            "current",
            "Current EMA-rho controlled Adam.",
        ),
        SimplifiedVariant(
            "trust_current",
            "trust",
            "current",
            "current",
            "Current EMA+trust: rho>=0.90, alpha<=1e-4, expand x1.5.",
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
            trust_region_rho_threshold=0.90,
            trust_alpha_threshold_factor=None,
            trust_region_expand_factor=1.5,
        ),
    ]

    for family in ["raw", "ema", "trust"]:
        for response in response_presets():
            for alpha_range in alpha_range_presets():
                variants.append(
                    SimplifiedVariant(
                        name=f"{family}_{response.name}_{alpha_range.name}",
                        family=family,
                        response_preset=response.name,
                        alpha_preset=alpha_range.name,
                        description=(
                            f"{family} with {response.name} response and "
                            f"{alpha_range.name} alpha range."
                        ),
                        kp_multiplier=response.kp_multiplier,
                        rho_star_delta=response.rho_star_delta,
                        rho_beta=response.rho_beta,
                        alpha_min_factor=alpha_range.alpha_min_factor,
                        alpha_max_factor=alpha_range.alpha_max_factor,
                        trust_region_rho_threshold=response.trust_region_rho_threshold,
                        trust_alpha_threshold_factor=(
                            alpha_range.trust_alpha_threshold_factor
                        ),
                        trust_region_expand_factor=response.trust_region_expand_factor,
                    )
                )
    return variants


def alpha_min_for(case: BenchmarkCase, variant: SimplifiedVariant) -> float:
    """Return absolute alpha_min for a variant."""

    if variant.alpha_min_factor is None:
        return case.alpha_min
    return float(case.alpha * variant.alpha_min_factor)


def alpha_max_for(
    case: BenchmarkCase,
    variant: SimplifiedVariant,
    alpha_min: float,
) -> float:
    """Return absolute alpha_max for a variant."""

    if variant.alpha_max_factor is None:
        return case.alpha_max
    return max(alpha_min, float(case.alpha * variant.alpha_max_factor))


def trust_alpha_threshold_for(
    case: BenchmarkCase,
    variant: SimplifiedVariant,
    alpha_min: float,
) -> float:
    """Return absolute trust alpha threshold for a variant."""

    if variant.trust_alpha_threshold_factor is None:
        return case.trust_region_alpha_threshold
    return max(alpha_min, float(case.alpha * variant.trust_alpha_threshold_factor))


def run_variant(
    case: BenchmarkCase,
    x0: np.ndarray,
    variant: SimplifiedVariant,
):
    """Run one simplified preset variant."""

    kp = case.kp * variant.kp_multiplier
    rho_star = bounded_rho_star(case, variant.rho_star_delta)
    alpha_min = alpha_min_for(case, variant)
    alpha_max = alpha_max_for(case, variant, alpha_min)

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
        trust_region_alpha_threshold=trust_alpha_threshold_for(
            case,
            variant,
            alpha_min,
        ),
        trust_region_expand_factor=(
            case.trust_region_expand_factor
            if variant.trust_region_expand_factor is None
            else variant.trust_region_expand_factor
        ),
    )


def run_sweep(
    cases: list[BenchmarkCase],
    variants: list[SimplifiedVariant],
) -> list[RunSummary]:
    """Run every simplified variant on every start in every case."""

    rows: list[RunSummary] = []
    for case in cases:
        print(f"- {case.objective.name}: {len(case.starts)} starts, {case.steps} steps")
        for start_id, x0 in enumerate(case.starts):
            for variant in variants:
                history = run_variant(case, x0, variant)
                rows.append(summarize_run(case, start_id, variant.name, x0, history))
    return rows


def write_variant_config(
    variants: list[SimplifiedVariant],
    cases: list[BenchmarkCase],
    path: Path,
) -> None:
    """Write variant settings and alpha factors to CSV."""

    fieldnames = list(SimplifiedVariant.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            writer.writerow({field: getattr(variant, field) for field in fieldnames})

    expanded_path = path.with_name("variant_expanded_by_objective.csv")
    expanded_fields = [
        "objective",
        "name",
        "family",
        "response_preset",
        "alpha_preset",
        "alpha0",
        "alpha_min",
        "alpha_max",
        "rho_star",
        "kp",
        "rho_beta",
        "trust_region_rho_threshold",
        "trust_region_alpha_threshold",
        "trust_region_expand_factor",
    ]
    with expanded_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=expanded_fields)
        writer.writeheader()
        for case in cases:
            for variant in variants:
                alpha_min = alpha_min_for(case, variant)
                alpha_max = alpha_max_for(case, variant, alpha_min)
                uses_ema = variant.family in {"ema", "trust"}
                uses_trust = variant.family == "trust"
                writer.writerow(
                    {
                        "objective": case.objective.name,
                        "name": variant.name,
                        "family": variant.family,
                        "response_preset": variant.response_preset,
                        "alpha_preset": variant.alpha_preset,
                        "alpha0": case.alpha,
                        "alpha_min": alpha_min,
                        "alpha_max": alpha_max,
                        "rho_star": bounded_rho_star(case, variant.rho_star_delta),
                        "kp": case.kp * variant.kp_multiplier,
                        "rho_beta": (
                            (
                                case.ema_beta
                                if variant.rho_beta is None
                                else variant.rho_beta
                            )
                            if uses_ema
                            else ""
                        ),
                        "trust_region_rho_threshold": (
                            (
                                case.trust_region_rho_threshold
                                if variant.trust_region_rho_threshold is None
                                else variant.trust_region_rho_threshold
                            )
                            if uses_trust
                            else ""
                        ),
                        "trust_region_alpha_threshold": (
                            trust_alpha_threshold_for(
                                case,
                                variant,
                                alpha_min,
                            )
                            if uses_trust
                            else ""
                        ),
                        "trust_region_expand_factor": (
                            (
                                case.trust_region_expand_factor
                                if variant.trust_region_expand_factor is None
                                else variant.trust_region_expand_factor
                            )
                            if uses_trust
                            else ""
                        ),
                    }
                )


def mean_log10_best(rows: Iterable[dict[str, object]]) -> float:
    """Return mean log10 median-best residual across objective rows."""

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


def rows_for_variant(
    aggregate: list[dict[str, object]],
    variant_name: str,
) -> list[dict[str, object]]:
    """Return aggregate rows for one variant."""

    return [row for row in aggregate if row["optimizer"] == variant_name]


def variant_score(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Rank by success first, then lower residual."""

    return (round(average_success(rows), 12), mean_log10_best(rows))


def best_by_family(
    variants: list[SimplifiedVariant],
    aggregate: list[dict[str, object]],
) -> dict[str, tuple[SimplifiedVariant, list[dict[str, object]]]]:
    """Return the best simplified variant for each family."""

    best: dict[str, tuple[SimplifiedVariant, list[dict[str, object]]]] = {}
    for family in ["raw", "ema", "trust"]:
        candidates = []
        for variant in variants:
            if variant.family != family or variant.response_preset == "current":
                continue
            rows = rows_for_variant(aggregate, variant.name)
            candidates.append((variant_score(rows), variant, rows))
        _, variant, rows = min(candidates, key=lambda item: (-item[0][0], item[0][1]))
        best[family] = (variant, rows)
    return best


def family_rankings(
    variants: list[SimplifiedVariant],
    aggregate: list[dict[str, object]],
    family: str,
    limit: int = 8,
) -> list[tuple[SimplifiedVariant, list[dict[str, object]]]]:
    """Return top simplified variants within a family."""

    candidates = [
        (variant_score(rows_for_variant(aggregate, variant.name)), variant)
        for variant in variants
        if variant.family == family and variant.response_preset != "current"
    ]
    ranked = sorted(candidates, key=lambda item: (-item[0][0], item[0][1]))
    return [(variant, rows_for_variant(aggregate, variant.name)) for _, variant in ranked[:limit]]


def write_summary(
    variants: list[SimplifiedVariant],
    aggregate: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    """Write a compact simplified-tuning summary."""

    path = output_dir / "CONTROLLED_ADAM_SIMPLIFIED_TUNING_SWEEP.md"
    best = best_by_family(variants, aggregate)

    lines: list[str] = []
    lines.append("# Controlled Adam Simplified Tuning Sweep")
    lines.append("")
    lines.append(
        "This sweep tests a low-dimensional preset interface instead of independent raw hyperparameters."
    )
    lines.append("")
    lines.append("## Simplified Interface")
    lines.append("")
    lines.append("- `family`: raw-rho, EMA-rho, or EMA+trust.")
    lines.append(
        "- `response_preset`: conservative, balanced, or aggressive. This jointly sets gain, rho target shift, EMA beta, and trust expansion aggressiveness."
    )
    lines.append(
        "- `alpha_preset`: low_floor, mid_floor, wide_cap, or high_floor. This derives alpha_min, alpha_max, and trust alpha threshold from the objective's base alpha0."
    )
    lines.append("")
    lines.append("## Preset Definitions")
    lines.append("")
    lines.append(
        "| Response Preset | kp Multiplier | rho_star Shift | rho_beta | Trust rho Threshold | Trust Expand Factor |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for preset in response_presets():
        lines.append(
            f"| {preset.name} | {preset.kp_multiplier:g} | "
            f"{preset.rho_star_delta:+.1f} | {preset.rho_beta:.2f} | "
            f"{preset.trust_region_rho_threshold:.2f} | "
            f"{preset.trust_region_expand_factor:g} |"
        )
    lines.append("")
    lines.append(
        "| Alpha Preset | alpha_min | alpha_max | Trust Alpha Threshold |"
    )
    lines.append("|---|---|---|---|")
    for preset in alpha_range_presets():
        lines.append(
            f"| {preset.name} | {preset.alpha_min_factor:g} * alpha0 | "
            f"{preset.alpha_max_factor:g} * alpha0 | "
            f"{preset.trust_alpha_threshold_factor:g} * alpha0 |"
        )
    lines.append("")
    lines.append("## Best Simplified vs Current")
    lines.append("")
    lines.append(
        "| Family | Current Variant | Current Avg Success | Best Simplified Variant | Simplified Avg Success | Delta | Current Mean Log10 Best Residual | Simplified Mean Log10 Best Residual | Simplified Median Trust Expansions |"
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
    lines.append("## Top Simplified Variants by Family")
    for family in ["raw", "ema", "trust"]:
        lines.append("")
        lines.append(f"### {family}")
        lines.append("")
        lines.append(
            "| Variant | Response Preset | Alpha Preset | Avg Success | Mean Log10 Best Residual | Median Trust Expansions |"
        )
        lines.append("|---|---|---|---:|---:|---:|")
        for variant, rows in family_rankings(variants, aggregate, family):
            lines.append(
                f"| `{variant.name}` | {variant.response_preset} | "
                f"{variant.alpha_preset} | {average_success(rows):.3f} | "
                f"{mean_log10_best(rows):.3f} | "
                f"{format_metric(median_trust_expansions(rows))} |"
            )

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_start_results.csv`: one row per objective/start/variant.")
    lines.append("- `aggregate_results.csv`: aggregate metrics by objective/variant.")
    lines.append("- `benchmark_config.csv`: objective settings and success tolerances.")
    lines.append("- `variant_config.csv`: compact tested preset variants.")
    lines.append(
        "- `variant_expanded_by_objective.csv`: absolute alpha/rho settings after expanding preset factors for each objective."
    )
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simplified controlled Adam preset tuning sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/controlled_adam_simplified_tuning_sweep_30runs"),
        help="Directory for simplified tuning CSVs and Markdown summary.",
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
    variants = simplified_variants()

    print("Running simplified controlled Adam tuning sweep")
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
    write_variant_config(variants, cases, output_dir / "variant_config.csv")
    summary_path = write_summary(variants, aggregate, output_dir)
    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Aggregate CSV: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-start CSV: {output_dir / 'per_start_results.csv'}")


if __name__ == "__main__":
    main()
