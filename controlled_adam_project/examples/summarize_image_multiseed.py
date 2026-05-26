"""Aggregate image benchmark metrics across multiple seed output folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_SPECS = {
    "test_accuracy": ("Test accuracy", "test_accuracy"),
    "test_loss": ("Test cross-entropy", "test_loss"),
    "train_accuracy": ("Train accuracy", "train_accuracy"),
    "train_loss": ("Train cross-entropy", "train_loss"),
}

DEFAULT_VANILLA_ADAM_LR = 1e-3

OPTIMIZER_COLORS = {
    "vanilla_adam": "#e07a24",
    "controlled_raw_rho": "#c92a2a",
    "controlled_ema": "#2f6fbb",
    "controlled_ema_trust": "#2b8a3e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Seed output directories containing *_epoch_metrics.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional optimizer subset/order to include in the aggregate report.",
    )
    parser.add_argument(
        "--copy-metadata",
        action="store_true",
        help="Copy compact run metadata into the aggregate JSON.",
    )
    return parser.parse_args()


def read_seed(path: Path, dataset: str) -> tuple[int | None, list[dict[str, str]]]:
    metrics_path = path / f"{dataset}_epoch_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")
    metadata_path = path / "run_metadata.json"
    seed = None
    if metadata_path.exists():
        with metadata_path.open() as handle:
            seed = json.load(handle).get("seed")
    with metrics_path.open() as handle:
        rows = list(csv.DictReader(handle))
    return seed, rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    array = np.array(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def stderr(std: float, n: int) -> float:
    if n <= 0 or not math.isfinite(std):
        return float("nan")
    return float(std / math.sqrt(n))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_epoch_rows(
    seed_rows: list[tuple[int | None, Path, list[dict[str, str]]]],
    variants: list[str] | None,
) -> tuple[list[dict[str, object]], list[str]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    variant_order: list[str] = []
    for _, _, rows in seed_rows:
        for row in rows:
            optimizer = row["optimizer"]
            if variants is not None and optimizer not in variants:
                continue
            if optimizer not in variant_order:
                variant_order.append(optimizer)
            epoch = int(row["epoch"])
            for metric in METRIC_SPECS:
                grouped[(optimizer, epoch)][metric].append(float(row[metric]))
            grouped[(optimizer, epoch)]["elapsed_seconds"].append(float(row["elapsed_seconds"]))
            grouped[(optimizer, epoch)]["optimizer_steps"].append(float(row["optimizer_steps"]))
            if row.get("mean_alpha"):
                grouped[(optimizer, epoch)]["mean_alpha"].append(float(row["mean_alpha"]))
            if row.get("mean_rho"):
                grouped[(optimizer, epoch)]["mean_rho"].append(float(row["mean_rho"]))
            if row.get("accepted_rate"):
                grouped[(optimizer, epoch)]["accepted_rate"].append(float(row["accepted_rate"]))

    if variants is not None:
        variant_order = [name for name in variants if name in variant_order]

    aggregate = []
    for optimizer in variant_order:
        epochs = sorted(epoch for opt, epoch in grouped if opt == optimizer)
        for epoch in epochs:
            values_by_metric = grouped[(optimizer, epoch)]
            row: dict[str, object] = {
                "optimizer": optimizer,
                "epoch": epoch,
                "n_seeds": len(values_by_metric["test_accuracy"]),
            }
            for metric in [
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
                "elapsed_seconds",
                "optimizer_steps",
                "mean_alpha",
                "mean_rho",
                "accepted_rate",
            ]:
                values = values_by_metric.get(metric, [])
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            aggregate.append(row)
    return aggregate, variant_order


def summarize_by_seed(
    seed_rows: list[tuple[int | None, Path, list[dict[str, str]]]],
    variants: list[str] | None,
) -> list[dict[str, object]]:
    summaries = []
    for seed, path, rows in seed_rows:
        by_optimizer: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            optimizer = row["optimizer"]
            if variants is None or optimizer in variants:
                by_optimizer[optimizer].append(row)
        for optimizer, opt_rows in by_optimizer.items():
            opt_rows = sorted(opt_rows, key=lambda row: int(row["epoch"]))
            test_acc = [float(row["test_accuracy"]) for row in opt_rows]
            best_index = int(np.argmax(test_acc))
            final = opt_rows[-1]
            best = opt_rows[best_index]
            summaries.append(
                {
                    "seed": seed,
                    "source_dir": str(path),
                    "optimizer": optimizer,
                    "final_epoch": int(final["epoch"]),
                    "final_train_loss": float(final["train_loss"]),
                    "final_train_accuracy": float(final["train_accuracy"]),
                    "final_test_loss": float(final["test_loss"]),
                    "final_test_accuracy": float(final["test_accuracy"]),
                    "best_epoch": int(best["epoch"]),
                    "best_test_accuracy": float(best["test_accuracy"]),
                    "best_test_loss": float(best["test_loss"]),
                    "final_elapsed_seconds": float(final["elapsed_seconds"]),
                    "final_optimizer_steps": int(float(final["optimizer_steps"])),
                    "final_mean_alpha": final.get("mean_alpha", ""),
                    "final_mean_rho": final.get("mean_rho", ""),
                    "final_accepted_rate": final.get("accepted_rate", ""),
                }
            )
    return summaries


def aggregate_final_summary(seed_summary: list[dict[str, object]], variant_order: list[str]) -> list[dict[str, object]]:
    rows = []
    by_optimizer: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in seed_summary:
        by_optimizer[str(row["optimizer"])].append(row)
    for optimizer in variant_order:
        rows_for_optimizer = by_optimizer[optimizer]
        final_acc = [float(row["final_test_accuracy"]) for row in rows_for_optimizer]
        best_acc = [float(row["best_test_accuracy"]) for row in rows_for_optimizer]
        final_loss = [float(row["final_test_loss"]) for row in rows_for_optimizer]
        elapsed = [float(row["final_elapsed_seconds"]) for row in rows_for_optimizer]
        final_acc_mean, final_acc_std = mean_std(final_acc)
        best_acc_mean, best_acc_std = mean_std(best_acc)
        final_loss_mean, final_loss_std = mean_std(final_loss)
        elapsed_mean, elapsed_std = mean_std(elapsed)
        rows.append(
            {
                "optimizer": optimizer,
                "n_seeds": len(rows_for_optimizer),
                "final_test_accuracy_mean": final_acc_mean,
                "final_test_accuracy_std": final_acc_std,
                "final_test_accuracy_stderr": stderr(final_acc_std, len(rows_for_optimizer)),
                "best_test_accuracy_mean": best_acc_mean,
                "best_test_accuracy_std": best_acc_std,
                "best_test_accuracy_stderr": stderr(best_acc_std, len(rows_for_optimizer)),
                "final_test_loss_mean": final_loss_mean,
                "final_test_loss_std": final_loss_std,
                "final_elapsed_seconds_mean": elapsed_mean,
                "final_elapsed_seconds_std": elapsed_std,
            }
        )
    return rows


def plot_epoch_metric(
    aggregate_rows: list[dict[str, object]],
    variant_order: list[str],
    output_dir: Path,
    metric: str,
) -> None:
    ylabel, slug = METRIC_SPECS[metric]
    plt.figure(figsize=(8, 4.8))
    for optimizer in variant_order:
        rows = [row for row in aggregate_rows if row["optimizer"] == optimizer]
        rows = sorted(rows, key=lambda row: int(row["epoch"]))
        x = np.array([int(row["epoch"]) for row in rows])
        y = np.array([float(row[f"{metric}_mean"]) for row in rows])
        std = np.array([float(row[f"{metric}_std"]) for row in rows])
        plt.plot(x, y, label=optimizer, color=OPTIMIZER_COLORS.get(optimizer))
        plt.fill_between(x, y - std, y + std, alpha=0.12)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Multi-seed {ylabel.lower()} by epoch")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{slug}_by_epoch_mean_std.png", dpi=170)
    plt.close()


def plot_epoch_diagnostic(
    aggregate_rows: list[dict[str, object]],
    variant_order: list[str],
    output_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    """Plot optimizer diagnostic curves when finite values are available."""

    plt.figure(figsize=(8, 4.8))
    plotted = False
    for optimizer in variant_order:
        rows = [row for row in aggregate_rows if row["optimizer"] == optimizer]
        rows = sorted(rows, key=lambda row: int(row["epoch"]))
        x = np.array([int(row["epoch"]) for row in rows])
        y = np.array([float(row[f"{metric}_mean"]) for row in rows])
        std = np.array([float(row[f"{metric}_std"]) for row in rows])
        if metric == "mean_alpha" and optimizer == "vanilla_adam" and not np.any(np.isfinite(y)):
            y = np.full_like(x, DEFAULT_VANILLA_ADAM_LR, dtype=float)
            std = np.zeros_like(x, dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        plotted = True
        color = OPTIMIZER_COLORS.get(optimizer)
        plt.plot(x[finite], y[finite], label=optimizer, color=color)
        lower = y[finite] - std[finite]
        upper = y[finite] + std[finite]
        plt.fill_between(x[finite], lower, upper, alpha=0.12, color=color)
    if not plotted:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Multi-seed {ylabel.lower()} by epoch")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=170)
    plt.close()


def plot_final_bars(final_rows: list[dict[str, object]], output_dir: Path) -> None:
    labels = [str(row["optimizer"]) for row in final_rows]
    x = np.arange(len(labels))
    width = 0.36
    final_means = np.array([float(row["final_test_accuracy_mean"]) for row in final_rows])
    best_means = np.array([float(row["best_test_accuracy_mean"]) for row in final_rows])
    final_err = np.array([float(row["final_test_accuracy_stderr"]) for row in final_rows])
    best_err = np.array([float(row["best_test_accuracy_stderr"]) for row in final_rows])
    colors = [OPTIMIZER_COLORS.get(label) for label in labels]

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(
        x - width / 2,
        final_means,
        width,
        yerr=final_err,
        label="final",
        capsize=4,
        color=colors,
        alpha=0.72,
    )
    plt.bar(
        x + width / 2,
        best_means,
        width,
        yerr=best_err,
        label="best",
        capsize=4,
        color=colors,
        alpha=1.0,
    )
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Final and best test accuracy across seeds")
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_best_test_accuracy_mean_stderr.png", dpi=170)
    plt.close()


def write_report(
    output_dir: Path,
    seed_summary: list[dict[str, object]],
    final_summary: list[dict[str, object]],
    run_dirs: list[Path],
) -> None:
    lines = [
        "# Multi-Seed Image Benchmark Summary",
        "",
        "This report aggregates epoch metrics from completed benchmark output folders.",
        "",
        "## Source Runs",
        "",
    ]
    for path in run_dirs:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Aggregate Final/Best Test Accuracy", ""])
    lines.append("| Optimizer | Seeds | Final Acc Mean +/- Std | Best Acc Mean +/- Std | Mean Time (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in final_summary:
        lines.append(
            "| {optimizer} | {n_seeds} | {final:.4f} +/- {final_std:.4f} | "
            "{best:.4f} +/- {best_std:.4f} | {elapsed:.1f} |".format(
                optimizer=row["optimizer"],
                n_seeds=row["n_seeds"],
                final=float(row["final_test_accuracy_mean"]),
                final_std=float(row["final_test_accuracy_std"]),
                best=float(row["best_test_accuracy_mean"]),
                best_std=float(row["best_test_accuracy_std"]),
                elapsed=float(row["final_elapsed_seconds_mean"]),
            )
        )
    lines.extend(["", "## Per-Seed Final/Best Test Accuracy", ""])
    lines.append("| Seed | Optimizer | Final Acc | Best Acc | Best Epoch | Final Time (s) |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in sorted(seed_summary, key=lambda item: (str(item["optimizer"]), int(item["seed"] or -1))):
        lines.append(
            "| {seed} | {optimizer} | {final:.4f} | {best:.4f} | {epoch} | {elapsed:.1f} |".format(
                seed=row["seed"],
                optimizer=row["optimizer"],
                final=float(row["final_test_accuracy"]),
                best=float(row["best_test_accuracy"]),
                epoch=row["best_epoch"],
                elapsed=float(row["final_elapsed_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `seed_summary.csv`: final and best metrics per seed and optimizer.",
            "- `final_summary.csv`: mean/std/stderr aggregate at the final and best epoch.",
            "- `epoch_aggregate.csv`: mean/std metrics for each epoch.",
            "- `*_by_epoch_mean_std.png`: mean curves with +/- one standard deviation bands.",
            "- `mean_alpha_by_epoch_mean_std.png`: controlled-optimizer alpha curve with +/- one standard deviation bands.",
            "- `final_best_test_accuracy_mean_stderr.png`: compact final/best accuracy comparison.",
        ]
    )
    (output_dir / "MULTISEED_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for run_dir in args.runs:
        seed, rows = read_seed(run_dir, args.dataset)
        seed_rows.append((seed, run_dir, rows))

    epoch_rows, variant_order = aggregate_epoch_rows(seed_rows, args.variants)
    seed_summary = summarize_by_seed(seed_rows, args.variants)
    final_summary = aggregate_final_summary(seed_summary, variant_order)

    epoch_fields = [
        "optimizer",
        "epoch",
        "n_seeds",
        "train_loss_mean",
        "train_loss_std",
        "train_accuracy_mean",
        "train_accuracy_std",
        "test_loss_mean",
        "test_loss_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "elapsed_seconds_mean",
        "elapsed_seconds_std",
        "optimizer_steps_mean",
        "optimizer_steps_std",
        "mean_alpha_mean",
        "mean_alpha_std",
        "mean_rho_mean",
        "mean_rho_std",
        "accepted_rate_mean",
        "accepted_rate_std",
    ]
    seed_fields = [
        "seed",
        "source_dir",
        "optimizer",
        "final_epoch",
        "final_train_loss",
        "final_train_accuracy",
        "final_test_loss",
        "final_test_accuracy",
        "best_epoch",
        "best_test_accuracy",
        "best_test_loss",
        "final_elapsed_seconds",
        "final_optimizer_steps",
        "final_mean_alpha",
        "final_mean_rho",
        "final_accepted_rate",
    ]
    final_fields = [
        "optimizer",
        "n_seeds",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "final_test_accuracy_stderr",
        "best_test_accuracy_mean",
        "best_test_accuracy_std",
        "best_test_accuracy_stderr",
        "final_test_loss_mean",
        "final_test_loss_std",
        "final_elapsed_seconds_mean",
        "final_elapsed_seconds_std",
    ]
    write_csv(args.output_dir / "epoch_aggregate.csv", epoch_rows, epoch_fields)
    write_csv(args.output_dir / "seed_summary.csv", seed_summary, seed_fields)
    write_csv(args.output_dir / "final_summary.csv", final_summary, final_fields)

    for metric in METRIC_SPECS:
        plot_epoch_metric(epoch_rows, variant_order, args.output_dir, metric)
    plot_epoch_diagnostic(
        epoch_rows,
        variant_order,
        args.output_dir,
        metric="mean_alpha",
        ylabel="Mean alpha",
        filename="mean_alpha_by_epoch_mean_std.png",
    )
    plot_final_bars(final_summary, args.output_dir)
    write_report(args.output_dir, seed_summary, final_summary, args.runs)

    if args.copy_metadata:
        metadata = []
        for seed, path, _ in seed_rows:
            metadata_path = path / "run_metadata.json"
            if metadata_path.exists():
                with metadata_path.open() as handle:
                    item = json.load(handle)
                item["source_dir"] = str(path)
                item["seed"] = seed
                metadata.append(item)
        with (args.output_dir / "source_run_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

    print(f"Aggregated {len(seed_rows)} runs into {args.output_dir.resolve()}")
    for row in final_summary:
        print(
            f"{row['optimizer']}: final={float(row['final_test_accuracy_mean']):.4f} "
            f"+/- {float(row['final_test_accuracy_std']):.4f}, "
            f"best={float(row['best_test_accuracy_mean']):.4f} "
            f"+/- {float(row['best_test_accuracy_std']):.4f}"
        )


if __name__ == "__main__":
    main()
