"""Plot Fashion-MNIST PI optimizer tuning results.

The script consumes the CSV files produced by ``run_pi_fashion_mnist_multiseed.py``
plus the earlier P-controller Fashion-MNIST runs, then writes comparison plots
for loss, accuracy, alpha, rho, acceptance, and final aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "pi_fashion_mnist_plots_5seed_3epoch_4k_1k"
SEEDS = [101, 202, 303, 404, 505]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    path: Path
    source_optimizer: str
    seed: int | None = None


PI_RUNS = [
    SeriesSpec("Vanilla Adam", ROOT / "outputs" / "pi_fashion_mnist_all4_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "vanilla_adam"),
    SeriesSpec("PI Adam A", ROOT / "outputs" / "pi_fashion_mnist_adam_A_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_adam"),
    SeriesSpec("PI Adam B", ROOT / "outputs" / "pi_fashion_mnist_adam_B_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_adam"),
    SeriesSpec("PI Adam C", ROOT / "outputs" / "pi_fashion_mnist_adam_C_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_adam"),
    SeriesSpec("Vanilla Muon", ROOT / "outputs" / "pi_fashion_mnist_all4_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "vanilla_muon"),
    SeriesSpec("PI Muon A", ROOT / "outputs" / "pi_fashion_mnist_muon_A_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
    SeriesSpec("PI Muon B", ROOT / "outputs" / "pi_fashion_mnist_muon_B_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
    SeriesSpec("PI Muon C", ROOT / "outputs" / "pi_fashion_mnist_muon_C_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
    SeriesSpec("PI Muon D", ROOT / "outputs" / "pi_fashion_mnist_muon_D_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
    SeriesSpec("PI Muon E", ROOT / "outputs" / "pi_fashion_mnist_muon_E_5seed_3epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
]

PI_20_EPOCH_RUNS = [
    SeriesSpec("Vanilla Adam", ROOT / "outputs" / "pi_fashion_mnist_vanilla_adam_muon_5seed_20epoch_4k_1k" / "epoch_metrics.csv", "vanilla_adam"),
    SeriesSpec("Vanilla Muon lr=1e-2", ROOT / "outputs" / "pi_official_muon_20epoch_vanilla_adam_muon_lr1e2_5seed_4k_1k" / "epoch_metrics.csv", "vanilla_muon"),
    SeriesSpec("PI Adam C", ROOT / "outputs" / "pi_fashion_mnist_adam_C_5seed_20epoch_4k_1k" / "epoch_metrics.csv", "pi_adam"),
    SeriesSpec("PI Muon D", ROOT / "outputs" / "pi_official_muon_D_5seed_20epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
    SeriesSpec("PI Muon E", ROOT / "outputs" / "pi_official_muon_E_5seed_20epoch_4k_1k" / "epoch_metrics.csv", "pi_muon"),
]

P_RUNS = [
    *[
        SeriesSpec(
            "P Adam EMA+trust",
            ROOT / "outputs" / f"p_adam_fashion_mnist_seed_{seed}_3epoch_4k_1k" / "fashion_mnist_epoch_metrics.csv",
            "controlled_ema_trust",
            seed,
        )
        for seed in SEEDS
    ],
    *[
        SeriesSpec(
            "P Muon EMA+trust",
            ROOT / "outputs" / f"p_muon_fashion_mnist_seed_{seed}_3epoch_4k_1k" / "fashion_mnist_epoch_metrics.csv",
            "controlled_ema_trust",
            seed,
        )
        for seed in SEEDS
    ],
]

ADAM_ORDER = ["Vanilla Adam", "P Adam EMA+trust", "PI Adam A", "PI Adam B", "PI Adam C"]
MUON_ORDER = ["Vanilla Muon", "P Muon EMA+trust", "PI Muon A", "PI Muon B", "PI Muon C", "PI Muon D", "PI Muon E"]
MUON_TUNING_ORDER = ["PI Muon C", "PI Muon D", "PI Muon E"]
COMBINED_20_EPOCH_ORDER = [
    "Vanilla Adam",
    "PI Adam C",
    "Vanilla Muon lr=1e-2",
    "PI Muon D",
    "PI Muon E",
]

COLORS = {
    "Vanilla Adam": "#4C78A8",
    "P Adam EMA+trust": "#F58518",
    "PI Adam A": "#54A24B",
    "PI Adam B": "#B279A2",
    "PI Adam C": "#E45756",
    "Vanilla Muon": "#4C78A8",
    "Vanilla Muon lr=1e-3": "#72B7B2",
    "Vanilla Muon lr=1e-2": "#4C78A8",
    "P Muon EMA+trust": "#F58518",
    "PI Muon A": "#72B7B2",
    "PI Muon B": "#54A24B",
    "PI Muon C": "#B279A2",
    "PI Muon D": "#E45756",
    "PI Muon E": "#9D755D",
}


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_rows(specs: Iterable[SeriesSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        if not spec.path.exists():
            print(f"Skipping missing input: {spec.path}")
            continue
        with spec.path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("optimizer") != spec.source_optimizer:
                    continue
                seed = spec.seed if spec.seed is not None else int(row.get("seed", 0))
                rows.append(
                    {
                        "label": spec.label,
                        "seed": seed,
                        "epoch": int(row["epoch"]),
                        "train_loss": parse_float(row.get("train_loss")),
                        "test_loss": parse_float(row.get("test_loss")),
                        "train_accuracy": parse_float(row.get("train_accuracy")),
                        "test_accuracy": parse_float(row.get("test_accuracy")),
                        "elapsed_seconds": parse_float(row.get("elapsed_seconds")),
                        "optimizer_steps": parse_float(row.get("optimizer_steps")),
                        "mean_alpha": parse_float(row.get("mean_alpha")),
                        "mean_rho": parse_float(row.get("mean_rho")),
                        "accepted_rate": parse_float(row.get("accepted_rate")),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, object]], x_field: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        x_value = parse_float(str(row[x_field]))
        if not math.isfinite(x_value):
            continue
        grouped[(str(row["label"]), x_value)].append(row)

    summary = []
    metrics = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "elapsed_seconds",
        "optimizer_steps",
        "mean_alpha",
        "mean_rho",
        "accepted_rate",
    ]
    for (label, x_value), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, object] = {"label": label, x_field: x_value, "n": len(group)}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            values = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(out)
    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows_for_labels(rows: list[dict[str, object]], labels: list[str]) -> list[dict[str, object]]:
    allowed = set(labels)
    return [row for row in rows if row["label"] in allowed]


def plot_metric_grid(rows: list[dict[str, object]], labels: list[str], x_field: str, output_path: Path, title: str) -> None:
    metric_titles = [
        ("train_loss", "Train Loss"),
        ("test_loss", "Test Loss"),
        ("train_accuracy", "Train Accuracy"),
        ("test_accuracy", "Test Accuracy"),
    ]
    summary = summarize_rows(rows_for_labels(rows, labels), x_field)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    for ax, (metric, metric_title) in zip(axes.ravel(), metric_titles):
        for label in labels:
            series = [row for row in summary if row["label"] == label and math.isfinite(float(row[f"{metric}_mean"]))]
            if not series:
                continue
            x = np.array([float(row[x_field]) for row in series], dtype=float)
            y = np.array([float(row[f"{metric}_mean"]) for row in series], dtype=float)
            std = np.array([float(row[f"{metric}_std"]) for row in series], dtype=float)
            color = COLORS.get(label)
            ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
            if np.any(std > 0):
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.12)
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        if "accuracy" in metric:
            ax.set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel(x_field.replace("_", " ").title())
    axes[1, 1].set_xlabel(x_field.replace("_", " ").title())
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_diagnostics(rows: list[dict[str, object]], labels: list[str], x_field: str, output_path: Path, title: str) -> None:
    metric_titles = [
        ("mean_alpha", "Mean Alpha"),
        ("mean_rho", "Mean Rho"),
        ("accepted_rate", "Accepted Rate"),
    ]
    summary = summarize_rows(rows_for_labels(rows, labels), x_field)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, (metric, metric_title) in zip(axes, metric_titles):
        for label in labels:
            series = [row for row in summary if row["label"] == label and math.isfinite(float(row[f"{metric}_mean"]))]
            if not series:
                continue
            x = np.array([float(row[x_field]) for row in series], dtype=float)
            y = np.array([float(row[f"{metric}_mean"]) for row in series], dtype=float)
            color = COLORS.get(label)
            ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
        ax.set_title(metric_title)
        ax.set_xlabel(x_field.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        if metric == "accepted_rate":
            ax.set_ylim(0.0, 1.05)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def final_rows(rows: list[dict[str, object]], labels: list[str]) -> list[dict[str, object]]:
    by_seed_label: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows_for_labels(rows, labels):
        key = (str(row["label"]), int(row["seed"]))
        previous = by_seed_label.get(key)
        if previous is None or int(row["epoch"]) > int(previous["epoch"]):
            by_seed_label[key] = row
    return list(by_seed_label.values())


def plot_final_bars(rows: list[dict[str, object]], labels: list[str], output_path: Path, title: str) -> None:
    finals = final_rows(rows, labels)
    metrics = [("test_accuracy", "Final Test Accuracy"), ("test_loss", "Final Test Loss")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (metric, metric_title) in zip(axes, metrics):
        means = []
        stds = []
        for label in labels:
            values = np.array([float(row[metric]) for row in finals if row["label"] == label], dtype=float)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=[COLORS.get(label, "#999999") for label in labels], alpha=0.9, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(metric_title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(bottom=0.0)
        if metric == "test_accuracy":
            ax.set_ylim(0.0, min(1.0, max(means) + 0.05))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_specs = [
        ("adam", ADAM_ORDER, "Adam: Vanilla vs P Control vs PI Tuning"),
        ("muon", MUON_ORDER, "Muon: Vanilla vs P Control vs PI Tuning"),
        ("muon_cde", MUON_TUNING_ORDER, "Muon PI C/D/E Tuning"),
    ]
    for prefix, labels, title in plot_specs:
        plot_metric_grid(rows, labels, "epoch", output_dir / f"{prefix}_learning_curves_by_epoch.png", title)
        plot_metric_grid(rows, labels, "optimizer_steps", output_dir / f"{prefix}_learning_curves_by_steps.png", title)
        plot_metric_grid(rows, labels, "elapsed_seconds", output_dir / f"{prefix}_learning_curves_by_time.png", title)
        plot_diagnostics(rows, labels, "optimizer_steps", output_dir / f"{prefix}_alpha_rho_acceptance_by_steps.png", title)
        plot_diagnostics(rows, labels, "epoch", output_dir / f"{prefix}_alpha_rho_acceptance_by_epoch.png", title)
        plot_final_bars(rows, labels, output_dir / f"{prefix}_final_bars.png", title)


def make_20_epoch_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    labels = COMBINED_20_EPOCH_ORDER
    title = "Fashion-MNIST 20 Epochs: Adam and Corrected Muon Variants"
    plot_metric_grid(rows, labels, "epoch", output_dir / "combined_learning_curves_by_epoch.png", title)
    plot_metric_grid(rows, labels, "optimizer_steps", output_dir / "combined_learning_curves_by_steps.png", title)
    plot_metric_grid(rows, labels, "elapsed_seconds", output_dir / "combined_learning_curves_by_time.png", title)
    plot_diagnostics(rows, labels, "optimizer_steps", output_dir / "combined_alpha_rho_acceptance_by_steps.png", title)
    plot_diagnostics(rows, labels, "epoch", output_dir / "combined_alpha_rho_acceptance_by_epoch.png", title)
    plot_final_bars(rows, labels, output_dir / "combined_final_bars.png", title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-set",
        choices=["3epoch", "20epoch"],
        default="3epoch",
        help="Select the hard-coded Fashion-MNIST result family to plot.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = PI_20_EPOCH_RUNS if args.run_set == "20epoch" else [*PI_RUNS, *P_RUNS]
    rows = read_rows(specs)
    if not rows:
        raise SystemExit("No input rows found.")

    write_summary(args.output_dir / "combined_epoch_rows.csv", rows)
    write_summary(args.output_dir / "combined_epoch_summary_by_epoch.csv", summarize_rows(rows, "epoch"))
    write_summary(args.output_dir / "combined_epoch_summary_by_steps.csv", summarize_rows(rows, "optimizer_steps"))
    write_summary(args.output_dir / "combined_epoch_summary_by_time.csv", summarize_rows(rows, "elapsed_seconds"))
    if args.run_set == "20epoch":
        make_20_epoch_plots(rows, args.output_dir)
    else:
        make_plots(rows, args.output_dir)

    print(f"Wrote plots and summaries to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
