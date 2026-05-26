"""Focused Rastrigin basin-of-attraction benchmark for Adam variants.

This benchmark asks a narrower question than the general function report:

    If every optimizer starts from the same random points around the true
    Rastrigin minimizer at (0, 0), how large is the basin where it reaches the
    real global minimum?

Run from ``controlled_adam_project`` with:

    MPLCONFIGDIR=/private/tmp PYTHONPATH=src \
      python examples/run_rastrigin_basin_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controlled_adam.objectives import Rastrigin
from controlled_adam.optimizers import OptimizationHistory, controlled_adam, vanilla_adam
from run_function_benchmark_report import controlled_adam_ema_rho


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


@dataclass(frozen=True)
class BasinRun:
    """Metrics for one optimizer from one Rastrigin start."""

    radius: float
    start_id: int
    optimizer: str
    start_x: float
    start_y: float
    final_f: float
    best_f: float
    final_distance: float
    best_distance: float
    success: bool
    iterations_to_success: int
    final_alpha: float
    median_alpha: float
    accepted_rate: float


def iterations_to_success(
    history: OptimizationHistory,
    success_tol_f: float,
    success_tol_dist: float,
) -> int:
    """Return first iteration that reaches the global Rastrigin minimum."""

    distances = np.linalg.norm(history.xs, axis=1)
    success_mask = (history.fs <= success_tol_f) | (distances <= success_tol_dist)
    hits = np.flatnonzero(success_mask)
    return int(hits[0]) if len(hits) else -1


def summarize_history(
    radius: float,
    start_id: int,
    optimizer: str,
    x0: np.ndarray,
    history: OptimizationHistory,
    success_tol_f: float,
    success_tol_dist: float,
) -> BasinRun:
    """Summarize one trajectory."""

    distances = np.linalg.norm(history.xs, axis=1)
    hit_iteration = iterations_to_success(history, success_tol_f, success_tol_dist)
    accepted_rate = float("nan")
    if history.accepted is not None:
        accepted_rate = float(np.mean(history.accepted))
    final_alpha = float(history.alphas[-1]) if len(history.alphas) else float("nan")
    median_alpha = float(np.median(history.alphas)) if len(history.alphas) else float("nan")
    return BasinRun(
        radius=radius,
        start_id=start_id,
        optimizer=optimizer,
        start_x=float(x0[0]),
        start_y=float(x0[1]),
        final_f=float(history.fs[-1]),
        best_f=float(np.min(history.fs)),
        final_distance=float(distances[-1]),
        best_distance=float(np.min(distances)),
        success=hit_iteration >= 0,
        iterations_to_success=hit_iteration,
        final_alpha=final_alpha,
        median_alpha=median_alpha,
        accepted_rate=accepted_rate,
    )


def run_optimizer(
    optimizer: str,
    objective: Rastrigin,
    x0: np.ndarray,
    steps: int,
    alpha: float,
    alpha_max: float,
    rho_star: float,
    kp: float,
    alpha_min: float,
    rho_min: float,
    ema_beta: float,
    max_backtracks: int,
) -> OptimizationHistory:
    """Run one optimizer variant with the Rastrigin report settings."""

    if optimizer == "vanilla_adam":
        return vanilla_adam(objective, x0, alpha=alpha, steps=steps)
    if optimizer == "controlled_raw_rho":
        return controlled_adam(
            objective,
            x0,
            alpha0=alpha,
            steps=steps,
            kp=kp,
            rho_star=rho_star,
            rho_min=rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=max_backtracks,
        )
    if optimizer == "controlled_ema_rho":
        return controlled_adam_ema_rho(
            objective,
            x0,
            alpha0=alpha,
            steps=steps,
            kp=kp,
            rho_star=rho_star,
            rho_beta=ema_beta,
            rho_min=rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=max_backtracks,
        )
    if optimizer == "controlled_ema_trust":
        return controlled_adam_ema_rho(
            objective,
            x0,
            alpha0=alpha,
            steps=steps,
            kp=kp,
            rho_star=rho_star,
            rho_beta=ema_beta,
            rho_min=rho_min,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            reject_bad_steps=True,
            max_backtracks=max_backtracks,
            min_alpha_factor=0.5,
            max_alpha_factor=1.25,
            trust_region_expand=True,
            trust_region_rho_threshold=0.90,
            trust_region_alpha_threshold=1e-4,
            trust_region_expand_factor=1.5,
        )
    raise ValueError(f"Unknown optimizer: {optimizer}")


def generate_starts(radii: list[float], starts_per_radius: int, seed: int) -> dict[float, np.ndarray]:
    """Generate deterministic box-uniform starts for each radius."""

    if starts_per_radius <= 0:
        raise ValueError("starts_per_radius must be positive.")
    rng = np.random.default_rng(seed)
    return {
        radius: rng.uniform(-radius, radius, size=(starts_per_radius, 2))
        for radius in radii
    }


def aggregate_rows(rows: list[BasinRun]) -> list[dict[str, object]]:
    """Aggregate metrics by radius and optimizer."""

    aggregate: list[dict[str, object]] = []
    keys = sorted({(row.radius, row.optimizer) for row in rows})
    for radius, optimizer in keys:
        subset = [row for row in rows if row.radius == radius and row.optimizer == optimizer]
        successful_iters = [
            row.iterations_to_success
            for row in subset
            if row.iterations_to_success >= 0
        ]
        aggregate.append(
            {
                "radius": radius,
                "optimizer": optimizer,
                "num_starts": len(subset),
                "success_rate": float(np.mean([row.success for row in subset])),
                "median_best_f": float(np.median([row.best_f for row in subset])),
                "median_final_f": float(np.median([row.final_f for row in subset])),
                "median_best_distance": float(np.median([row.best_distance for row in subset])),
                "median_final_distance": float(np.median([row.final_distance for row in subset])),
                "median_iterations_to_success": (
                    float(np.median(successful_iters)) if successful_iters else float("nan")
                ),
                "median_final_alpha": finite_median(row.final_alpha for row in subset),
                "median_accepted_rate": finite_median(row.accepted_rate for row in subset),
            }
        )
    return aggregate


def finite_median(values: object) -> float:
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


def plot_success_rate(aggregate: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot convergence success rate versus initialization radius."""

    radii = sorted({float(row["radius"]) for row in aggregate})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for optimizer, label in OPTIMIZER_LABELS.items():
        values = [
            float(
                next(
                    row["success_rate"]
                    for row in aggregate
                    if float(row["radius"]) == radius and row["optimizer"] == optimizer
                )
            )
            for radius in radii
        ]
        ax.plot(
            radii,
            values,
            marker="o",
            linewidth=2.2,
            label=label,
            color=OPTIMIZER_COLORS[optimizer],
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Initialization box radius r for x0 ~ Uniform([-r, r]^2)")
    ax.set_ylabel("Success rate")
    ax.set_title("Rastrigin basin-of-attraction success")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "rastrigin_success_rate_by_radius.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_median_best(aggregate: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot median best objective value versus initialization radius."""

    radii = sorted({float(row["radius"]) for row in aggregate})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for optimizer, label in OPTIMIZER_LABELS.items():
        values = [
            float(
                next(
                    row["median_best_f"]
                    for row in aggregate
                    if float(row["radius"]) == radius and row["optimizer"] == optimizer
                )
            )
            for radius in radii
        ]
        ax.plot(
            radii,
            values,
            marker="o",
            linewidth=2.2,
            label=label,
            color=OPTIMIZER_COLORS[optimizer],
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("Initialization box radius r for x0 ~ Uniform([-r, r]^2)")
    ax.set_ylabel("Median best objective value")
    ax.set_title("Rastrigin median best objective by basin radius")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "rastrigin_median_best_by_radius.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_successful_iterations(aggregate: list[dict[str, object]], output_dir: Path) -> Path:
    """Plot median iterations among successful starts."""

    radii = sorted({float(row["radius"]) for row in aggregate})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for optimizer, label in OPTIMIZER_LABELS.items():
        values = [
            float(
                next(
                    row["median_iterations_to_success"]
                    for row in aggregate
                    if float(row["radius"]) == radius and row["optimizer"] == optimizer
                )
            )
            for radius in radii
        ]
        ax.plot(
            radii,
            values,
            marker="o",
            linewidth=2.2,
            label=label,
            color=OPTIMIZER_COLORS[optimizer],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Initialization box radius r for x0 ~ Uniform([-r, r]^2)")
    ax.set_ylabel("Median iterations to success among successful starts")
    ax.set_title("Rastrigin successful-run speed")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "rastrigin_iterations_to_success_by_radius.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_report(
    aggregate: list[dict[str, object]],
    output_dir: Path,
    radii: list[float],
    starts_per_radius: int,
    steps: int,
    seed: int,
    alpha: float,
    alpha_max: float,
    success_tol_f: float,
    success_tol_dist: float,
) -> tuple[Path, Path]:
    """Generate English and Chinese Markdown reports."""

    english_path = output_dir / "RASTRIGIN_BASIN_BENCHMARK_REPORT.md"
    chinese_path = output_dir / "RASTRIGIN_BASIN_BENCHMARK_REPORT_ZH.md"

    table_lines = [
        "| Radius | Optimizer | Success rate | Median best f | Median iterations to success |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    table_lines_zh = [
        "| 初始半径 | 优化器 | 成功率 | 中位最佳函数值 | 成功样本的中位迭代数 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        optimizer = str(row["optimizer"])
        med_iters = float(row["median_iterations_to_success"])
        med_iters_text = "n/a" if not np.isfinite(med_iters) else f"{med_iters:.0f}"
        table_lines.append(
            f"| {float(row['radius']):.3g} | {OPTIMIZER_LABELS[optimizer]} | "
            f"{float(row['success_rate']):.2f} | {float(row['median_best_f']):.3e} | "
            f"{med_iters_text} |"
        )
        table_lines_zh.append(
            f"| {float(row['radius']):.3g} | {OPTIMIZER_LABELS_ZH[optimizer]} | "
            f"{float(row['success_rate']):.2f} | {float(row['median_best_f']):.3e} | "
            f"{med_iters_text} |"
        )

    english_path.write_text(
        "\n".join(
            [
                "# Rastrigin Basin-of-Attraction Benchmark",
                "",
                "This focused benchmark samples initial points from boxes centered on the true Rastrigin global minimizer `(0, 0)`.",
                "Each optimizer receives exactly the same starts at each radius.",
                "",
                "## Configuration",
                "",
                f"- Radii: {', '.join(str(radius) for radius in radii)}",
                f"- Starts per radius: {starts_per_radius}",
                f"- Steps per run: {steps}",
                f"- Random seed: {seed}",
                f"- Adam learning rate / initial alpha: {alpha}",
                f"- Controlled alpha max: {alpha_max}",
                f"- Success criterion: `f(x) <= {success_tol_f}` or `||x|| <= {success_tol_dist}`",
                "",
                "## Plots",
                "",
                "- `rastrigin_success_rate_by_radius.png`",
                "- `rastrigin_median_best_by_radius.png`",
                "- `rastrigin_iterations_to_success_by_radius.png`",
                "",
                "## Aggregate Results",
                "",
                *table_lines,
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    chinese_path.write_text(
        "\n".join(
            [
                "# Rastrigin 吸引域专项 benchmark",
                "",
                "这个实验专门回答一个问题：如果初始点从全局最优点 `(0, 0)` 附近采样，各个优化器能在多大的初始范围内收敛到真正的全局最小值？",
                "每个优化器在每个半径下使用完全相同的初始点，因此比较的是优化器本身，而不是随机初始点差异。",
                "",
                "## 实验设置",
                "",
                f"- 初始半径: {', '.join(str(radius) for radius in radii)}",
                f"- 每个半径的初始点数量: {starts_per_radius}",
                f"- 每次运行步数: {steps}",
                f"- 随机种子: {seed}",
                f"- Adam 学习率 / 受控 Adam 初始 alpha: {alpha}",
                f"- 受控 Adam alpha 上限: {alpha_max}",
                f"- 成功标准: `f(x) <= {success_tol_f}` 或 `||x|| <= {success_tol_dist}`",
                "",
                "## 图表",
                "",
                "- `rastrigin_success_rate_by_radius.png`: 不同初始半径下的成功率。",
                "- `rastrigin_median_best_by_radius.png`: 不同初始半径下的中位最佳函数值。",
                "- `rastrigin_iterations_to_success_by_radius.png`: 成功样本中达到成功标准所需的中位迭代数。",
                "",
                "## 汇总结果",
                "",
                *table_lines_zh,
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return english_path, chinese_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Rastrigin basin-of-attraction benchmark for Adam variants."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rastrigin_basin_benchmark"),
        help="Directory for CSVs, plots, and reports.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        help="Initialization box radii around (0, 0).",
    )
    parser.add_argument(
        "--starts-per-radius",
        type=int,
        default=50,
        help="Number of random starts for each radius.",
    )
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--alpha", type=float, default=0.004)
    parser.add_argument("--alpha-max", type=float, default=0.04)
    parser.add_argument("--rho-star", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=0.04)
    parser.add_argument("--alpha-min", type=float, default=1e-8)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--ema-beta", type=float, default=0.90)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--success-tol-f", type=float, default=1e-3)
    parser.add_argument("--success-tol-dist", type=float, default=5e-2)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = Rastrigin()
    starts_by_radius = generate_starts(args.radii, args.starts_per_radius, args.seed)

    rows: list[BasinRun] = []
    print("Running Rastrigin basin-of-attraction benchmark")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Radii: {args.radii}")
    print(f"Starts per radius: {args.starts_per_radius}")
    print(f"Steps: {args.steps}")

    for radius in args.radii:
        starts = starts_by_radius[radius]
        print(f"- radius {radius}: {len(starts)} starts")
        for start_id, x0 in enumerate(starts):
            for optimizer in OPTIMIZER_LABELS:
                history = run_optimizer(
                    optimizer=optimizer,
                    objective=objective,
                    x0=x0,
                    steps=args.steps,
                    alpha=args.alpha,
                    alpha_max=args.alpha_max,
                    rho_star=args.rho_star,
                    kp=args.kp,
                    alpha_min=args.alpha_min,
                    rho_min=args.rho_min,
                    ema_beta=args.ema_beta,
                    max_backtracks=args.max_backtracks,
                )
                rows.append(
                    summarize_history(
                        radius=radius,
                        start_id=start_id,
                        optimizer=optimizer,
                        x0=x0,
                        history=history,
                        success_tol_f=args.success_tol_f,
                        success_tol_dist=args.success_tol_dist,
                    )
                )

    aggregate = aggregate_rows(rows)
    per_run_fields = list(BasinRun.__dataclass_fields__.keys())
    aggregate_fields = [
        "radius",
        "optimizer",
        "num_starts",
        "success_rate",
        "median_best_f",
        "median_final_f",
        "median_best_distance",
        "median_final_distance",
        "median_iterations_to_success",
        "median_final_alpha",
        "median_accepted_rate",
    ]
    write_csv(output_dir / "per_run_results.csv", rows, per_run_fields)
    write_csv(output_dir / "aggregate_results.csv", aggregate, aggregate_fields)
    plot_success_rate(aggregate, output_dir)
    plot_median_best(aggregate, output_dir)
    plot_successful_iterations(aggregate, output_dir)
    report_path, chinese_report_path = generate_report(
        aggregate=aggregate,
        output_dir=output_dir,
        radii=args.radii,
        starts_per_radius=args.starts_per_radius,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
        alpha_max=args.alpha_max,
        success_tol_f=args.success_tol_f,
        success_tol_dist=args.success_tol_dist,
    )

    print("Done.")
    print(f"Report: {report_path.resolve()}")
    print(f"Chinese report: {chinese_report_path.resolve()}")
    print(f"Aggregate CSV: {(output_dir / 'aggregate_results.csv').resolve()}")
    print(f"Per-run CSV: {(output_dir / 'per_run_results.csv').resolve()}")


if __name__ == "__main__":
    main()
