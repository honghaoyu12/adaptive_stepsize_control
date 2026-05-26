"""Plot checkpoint training trajectories in a PCA plane.

This post-processing script follows the PCA trajectory idea from
"Visualizing the Loss Landscape of Neural Nets": collect checkpoints along a
training run, flatten the model weights, fit a two-dimensional PCA basis to
the checkpoint displacements, and project each checkpoint into that plane.

The script expects checkpoints produced by ``run_mnist_demo.py`` with
``--checkpoint-every`` enabled. It writes:

* ``pca_trajectory_coordinates.csv``
* ``pca_explained_variance.csv``
* ``pca_training_trajectory.png``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_mnist_demo import make_model


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    run_name: str
    dataset: str
    epoch: int
    state_dict: dict[str, torch.Tensor]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Metadata:
    dataset_name: str | None
    model_name: str | None


def load_metadata(run_dir: Path) -> Metadata:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return Metadata(dataset_name=None, model_name=None)

    with metadata_path.open() as handle:
        raw = json.load(handle)

    dataset_info = raw.get("dataset", {})
    model_info = raw.get("model", {})
    return Metadata(
        dataset_name=dataset_info.get("resolved"),
        model_name=model_info.get("resolved") or model_info.get("requested"),
    )


def resolve_checkpoint_dir(path: Path) -> tuple[Path, Path]:
    """Return ``(run_dir, checkpoint_dir)`` for a run or checkpoint path."""
    path = path.expanduser().resolve()
    if (path / "checkpoints").is_dir():
        return path, path / "checkpoints"
    if path.is_dir() and path.name == "checkpoints":
        return path.parent, path
    if path.is_dir():
        return path, path
    raise FileNotFoundError(f"Could not find checkpoint directory from {path}")


def load_checkpoint(path: Path) -> CheckpointRecord:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} does not contain a model_state_dict.")

    metrics = payload.get("metrics", [])
    metric_row: dict[str, Any] = {}
    if isinstance(metrics, list) and metrics:
        last = metrics[-1]
        if isinstance(last, dict):
            metric_row = last

    run_name = str(payload.get("run_name", infer_run_name_from_filename(path)))
    dataset = str(payload.get("dataset", "unknown"))
    epoch = int(payload.get("epoch", infer_epoch_from_filename(path)))
    return CheckpointRecord(
        path=path,
        run_name=run_name,
        dataset=dataset,
        epoch=epoch,
        state_dict=state_dict,
        metrics=metric_row,
    )


def infer_run_name_from_filename(path: Path) -> str:
    match = re.match(r".+?_(.+)_epoch_\d+\.pt$", path.name)
    return match.group(1) if match else path.stem


def infer_epoch_from_filename(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"Could not infer epoch from {path.name}.")
    return int(match.group(1))


def discover_checkpoints(
    checkpoint_dir: Path,
    glob_pattern: str,
    selected_runs: list[str] | None,
) -> list[CheckpointRecord]:
    selected_run_set = set(selected_runs) if selected_runs is not None else None
    selected_run_order = (
        {run_name: index for index, run_name in enumerate(selected_runs)}
        if selected_runs is not None
        else None
    )
    records = []
    for path in sorted(checkpoint_dir.glob(glob_pattern)):
        if not path.is_file():
            continue
        record = load_checkpoint(path)
        if selected_run_set is not None and record.run_name not in selected_run_set:
            continue
        records.append(record)

    if selected_run_set is not None:
        found_runs = {record.run_name for record in records}
        missing_runs = [run_name for run_name in selected_runs if run_name not in found_runs]
        if missing_runs:
            available_runs = sorted(
                {
                    load_checkpoint(path).run_name
                    for path in checkpoint_dir.glob(glob_pattern)
                    if path.is_file()
                }
            )
            raise ValueError(
                f"No checkpoints found for requested runs: {missing_runs}. "
                f"Available runs: {available_runs}"
            )

    if len(records) < 2:
        raise ValueError(
            "Need at least two checkpoints after filtering to build a PCA trajectory."
        )

    if selected_run_order is None:
        records.sort(key=lambda item: (item.run_name, item.epoch, str(item.path)))
    else:
        records.sort(
            key=lambda item: (
                selected_run_order[item.run_name],
                item.epoch,
                str(item.path),
            )
        )
    return records


def resolve_parameter_names(
    records: list[CheckpointRecord],
    dataset_name: str | None,
    model_name: str | None,
    include_buffers: bool,
) -> list[str]:
    first_state = records[0].state_dict
    if include_buffers:
        return [
            name
            for name, value in first_state.items()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        ]

    if dataset_name is not None and model_name is not None:
        model = make_model(model_name, dataset_name)
        parameter_names = [name for name, _param in model.named_parameters()]
        missing = [name for name in parameter_names if name not in first_state]
        if missing:
            raise ValueError(
                "Model parameter names do not match checkpoint state_dict. "
                f"Missing examples: {missing[:5]}"
            )
        return parameter_names

    return infer_parameter_like_names(first_state)


def infer_parameter_like_names(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Fallback for bare checkpoint folders with no model metadata.

    This excludes common BatchNorm running-stat buffers while keeping floating
    weights and biases. Supplying metadata is preferred because it lets the
    script use ``model.named_parameters()`` exactly.
    """
    excluded_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    names = []
    for name, value in state_dict.items():
        if any(name.endswith(suffix) for suffix in excluded_suffixes):
            continue
        if torch.is_tensor(value) and torch.is_floating_point(value):
            names.append(name)
    return names


def flatten_state(
    record: CheckpointRecord,
    parameter_names: list[str],
) -> np.ndarray:
    missing = [name for name in parameter_names if name not in record.state_dict]
    if missing:
        raise ValueError(
            f"{record.path} is missing parameters present in other checkpoints: "
            f"{missing[:5]}"
        )

    tensors = []
    for name in parameter_names:
        value = record.state_dict[name]
        if not torch.is_tensor(value):
            raise ValueError(f"{record.path}:{name} is not a tensor.")
        tensors.append(value.detach().cpu().reshape(-1).float())
    return torch.cat(tensors).numpy()


def fit_pca(
    weights: np.ndarray,
    reference_index: int,
    center_for_pca: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates, components, and explained variance ratios."""
    reference = weights[reference_index]
    displacements = weights - reference
    fit_matrix = displacements
    if center_for_pca:
        fit_matrix = fit_matrix - fit_matrix.mean(axis=0, keepdims=True)

    _u, singular_values, vt = np.linalg.svd(fit_matrix, full_matrices=False)
    if vt.shape[0] < 2:
        raise ValueError("Need at least two PCA components; provide more checkpoints.")

    components = vt[:2].copy()
    orient_components(components)
    coordinates = displacements @ components.T

    variances = singular_values**2
    total_variance = float(variances.sum())
    if total_variance <= 0.0:
        explained = np.zeros(2, dtype=float)
    else:
        explained = variances[:2] / total_variance
    return coordinates, components, explained


def orient_components(components: np.ndarray) -> None:
    """Choose deterministic component signs for stable plots across runs."""
    for row in components:
        index = int(np.argmax(np.abs(row)))
        if row[index] < 0:
            row *= -1.0


def reference_record_index(
    records: list[CheckpointRecord],
    reference_run: str | None,
    reference_epoch: int | None,
) -> int:
    run_order = list(dict.fromkeys(record.run_name for record in records))
    selected_run = reference_run or run_order[0]
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.run_name == selected_run
    ]
    if not candidates:
        raise ValueError(f"No checkpoints found for reference run {selected_run!r}.")

    if reference_epoch is not None:
        for index, record in candidates:
            if record.epoch == reference_epoch:
                return index
        raise ValueError(
            f"No checkpoint found for run {selected_run!r} at epoch {reference_epoch}."
        )

    return max(candidates, key=lambda item: item[1].epoch)[0]


def metric_value(record: CheckpointRecord, name: str) -> Any:
    value = record.metrics.get(name)
    return "" if value is None else value


def write_coordinates_csv(
    output_path: Path,
    records: list[CheckpointRecord],
    coordinates: np.ndarray,
) -> None:
    fields = [
        "run_name",
        "dataset",
        "epoch",
        "pc1",
        "pc2",
        "radius",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "optimizer_steps",
        "elapsed_seconds",
        "checkpoint_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, (pc1, pc2) in zip(records, coordinates):
            writer.writerow(
                {
                    "run_name": record.run_name,
                    "dataset": record.dataset,
                    "epoch": record.epoch,
                    "pc1": float(pc1),
                    "pc2": float(pc2),
                    "radius": float(math.hypot(float(pc1), float(pc2))),
                    "train_loss": metric_value(record, "train_loss"),
                    "train_accuracy": metric_value(record, "train_accuracy"),
                    "test_loss": metric_value(record, "test_loss"),
                    "test_accuracy": metric_value(record, "test_accuracy"),
                    "optimizer_steps": metric_value(record, "optimizer_steps"),
                    "elapsed_seconds": metric_value(record, "elapsed_seconds"),
                    "checkpoint_path": str(record.path),
                }
            )


def write_explained_variance_csv(
    output_path: Path,
    explained: np.ndarray,
    n_checkpoints: int,
    n_parameters: int,
    reference: CheckpointRecord,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "explained_variance_ratio",
                "n_checkpoints",
                "n_parameters",
                "reference_run",
                "reference_epoch",
                "reference_checkpoint",
            ],
        )
        writer.writeheader()
        for component, ratio in enumerate(explained, start=1):
            writer.writerow(
                {
                    "component": component,
                    "explained_variance_ratio": float(ratio),
                    "n_checkpoints": n_checkpoints,
                    "n_parameters": n_parameters,
                    "reference_run": reference.run_name,
                    "reference_epoch": reference.epoch,
                    "reference_checkpoint": str(reference.path),
                }
            )


def plot_trajectory(
    output_path: Path,
    records: list[CheckpointRecord],
    coordinates: np.ndarray,
    explained: np.ndarray,
    reference: CheckpointRecord,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    run_names = list(dict.fromkeys(record.run_name for record in records))
    colors = plt.get_cmap("tab10")

    for run_index, run_name in enumerate(run_names):
        indices = [
            index
            for index, record in enumerate(records)
            if record.run_name == run_name
        ]
        indices.sort(key=lambda index: records[index].epoch)
        xy = coordinates[indices]
        epochs = [records[index].epoch for index in indices]
        color = colors(run_index % 10)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=run_name,
            color=color,
        )
        ax.scatter(xy[0, 0], xy[0, 1], marker="s", s=55, color=color)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="*", s=95, color=color)
        ax.annotate(
            f"e{epochs[0]}",
            (xy[0, 0], xy[0, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )
        ax.annotate(
            f"e{epochs[-1]}",
            (xy[-1, 0], xy[-1, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )

    ax.axhline(0.0, color="0.82", linewidth=0.8)
    ax.axvline(0.0, color="0.82", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(
        "Training Trajectory PCA\n"
        f"reference: {reference.run_name} epoch {reference.epoch}"
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory containing checkpoints/, or the checkpoint directory itself.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="*.pt",
        help="Glob pattern inside the checkpoint directory.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional run_name values to include, e.g. vanilla_adam controlled_raw_rho.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name for model reconstruction. Defaults to run_metadata.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for model reconstruction. Defaults to run_metadata.json.",
    )
    parser.add_argument(
        "--include-buffers",
        action="store_true",
        help="Include floating state_dict buffers instead of trainable parameters only.",
    )
    parser.add_argument(
        "--reference-run",
        default=None,
        help="Run whose final checkpoint is used as the origin. Defaults to the first selected run.",
    )
    parser.add_argument(
        "--reference-epoch",
        type=int,
        default=None,
        help="Use a specific reference epoch instead of the selected run's final epoch.",
    )
    parser.add_argument(
        "--no-center-for-pca",
        action="store_true",
        help="Fit PCA directly on final-relative displacements without mean-centering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and PNG outputs. Defaults to <run_dir>/pca_trajectory.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, checkpoint_dir = resolve_checkpoint_dir(args.path)
    metadata = load_metadata(run_dir)
    dataset_name = args.dataset or metadata.dataset_name
    model_name = args.model or metadata.model_name
    selected_runs = list(dict.fromkeys(args.runs)) if args.runs is not None else None

    records = discover_checkpoints(
        checkpoint_dir,
        args.checkpoint_glob,
        selected_runs,
    )
    parameter_names = resolve_parameter_names(
        records,
        dataset_name,
        model_name,
        args.include_buffers,
    )
    if not parameter_names:
        raise ValueError("No parameters found to project.")

    weights = np.stack([flatten_state(record, parameter_names) for record in records])
    reference_index = reference_record_index(
        records,
        args.reference_run,
        args.reference_epoch,
    )
    coordinates, _components, explained = fit_pca(
        weights,
        reference_index,
        center_for_pca=not args.no_center_for_pca,
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "pca_trajectory"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = records[reference_index]
    write_coordinates_csv(
        output_dir / "pca_trajectory_coordinates.csv",
        records,
        coordinates,
    )
    write_explained_variance_csv(
        output_dir / "pca_explained_variance.csv",
        explained,
        n_checkpoints=len(records),
        n_parameters=weights.shape[1],
        reference=reference,
    )
    plot_trajectory(
        output_dir / "pca_training_trajectory.png",
        records,
        coordinates,
        explained,
        reference,
        dpi=args.dpi,
    )

    print(f"Loaded checkpoints: {len(records)}")
    print(f"Projected parameters: {weights.shape[1]}")
    print(
        "Explained variance: "
        f"PC1={explained[0] * 100:.2f}%, PC2={explained[1] * 100:.2f}%"
    )
    print(f"Reference: {reference.run_name} epoch {reference.epoch}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
