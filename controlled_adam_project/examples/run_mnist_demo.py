"""Compare vanilla Adam and controlled Adam on small image classifiers."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from controlled_adam.torch_optimizers import ControlledAdamStep, TorchControlledAdam


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SmallMLP(nn.Module):
    """Small fully connected classifier for 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FashionCNN(nn.Module):
    """Small convolutional classifier for 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SmallCIFARCNN(nn.Module):
    """Batch-normalized convolutional classifier for CIFAR-10 RGB images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LeNetCIFAR(nn.Module):
    """Classic LeNet-style CIFAR-10 classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(model_name: str, dataset_name: str) -> nn.Module:
    """Build a classifier compatible with the resolved dataset."""
    if model_name == "auto":
        model_name = "cnn" if dataset_name == "cifar10" else "mlp"
    if model_name == "mlp":
        if dataset_name == "cifar10":
            raise ValueError("The MLP model expects 28x28 grayscale inputs, not CIFAR-10.")
        return SmallMLP()
    if model_name == "fashion_cnn":
        if dataset_name not in {"mnist", "fashion_mnist", "sklearn_digits"}:
            raise ValueError("The Fashion CNN model expects 28x28 grayscale inputs.")
        return FashionCNN()
    if model_name == "cnn":
        if dataset_name != "cifar10":
            raise ValueError("The CNN model is currently configured for CIFAR-10 RGB inputs.")
        return SmallCIFARCNN()
    if model_name == "lenet_cifar":
        if dataset_name != "cifar10":
            raise ValueError("The LeNet CIFAR model is currently configured for CIFAR-10 RGB inputs.")
        return LeNetCIFAR()
    raise ValueError(f"Unknown model: {model_name}")


@dataclass
class EpochMetrics:
    """Metrics collected after one training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float = 0.0
    optimizer_steps: int = 0
    mean_alpha: float | None = None
    mean_rho: float | None = None
    accepted_rate: float | None = None


@dataclass
class OptimizerRun:
    """Metrics and optional step diagnostics for one optimizer variant."""

    name: str
    metrics: list[EpochMetrics]
    step_logs: list[ControlledAdamStep] | None = None


@dataclass
class CheckpointConfig:
    """Per-epoch checkpoint and progress reporting settings."""

    output_dir: Path
    dataset_name: str
    save_every: int
    print_every: int


@dataclass
class DatasetBundle:
    """Train/eval datasets with shared deterministic subset indices."""

    train_data: torch.utils.data.Dataset
    eval_train_data: torch.utils.data.Dataset
    test_data: torch.utils.data.Dataset
    name: str
    train_transform: str
    eval_transform: str
    train_size_full: int
    test_size_full: int
    train_size_used: int
    test_size_used: int


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_image_dataset_or_digits(
    data_dir: Path,
    dataset: str,
    train_subset: int,
    test_subset: int,
    seed: int,
    allow_download: bool,
    fashion_folder: Path | None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, str]:
    """Load train/test datasets, falling back to sklearn digits."""
    bundle = load_image_dataset_bundle(
        data_dir,
        dataset,
        train_subset,
        test_subset,
        seed,
        allow_download,
        fashion_folder,
    )
    return bundle.train_data, bundle.test_data, bundle.name


def load_image_dataset_bundle(
    data_dir: Path,
    dataset: str,
    train_subset: int,
    test_subset: int,
    seed: int,
    allow_download: bool,
    fashion_folder: Path | None,
) -> DatasetBundle:
    """Load train/eval/test datasets with deterministic subset alignment."""
    if dataset == "fashion_mnist" and fashion_folder is not None:
        train = load_idx_image_dataset(fashion_folder, train=True)
        test = load_idx_image_dataset(fashion_folder, train=False)
        train = deterministic_subset(train, train_subset, seed)
        test = deterministic_subset(test, test_subset, seed + 1)
        return DatasetBundle(
            train_data=train,
            eval_train_data=train,
            test_data=test,
            name=dataset,
            train_transform="ToTensor() from flat IDX gzip Fashion-MNIST files",
            eval_transform="same TensorDataset as training data",
            train_size_full=60000,
            test_size_full=10000,
            train_size_used=len(train),
            test_size_used=len(test),
        )

    try:
        from torchvision import datasets, transforms

        if dataset == "cifar10":
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
            eval_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
            train = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=allow_download,
                transform=train_transform,
            )
            eval_train = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=False,
                transform=eval_transform,
            )
            test = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=allow_download,
                transform=eval_transform,
            )
            train_indices = deterministic_indices(len(train), train_subset, seed)
            test_indices = deterministic_indices(len(test), test_subset, seed + 1)
            train_data = subset_from_indices(train, train_indices)
            eval_train_data = subset_from_indices(eval_train, train_indices)
            test_data = subset_from_indices(test, test_indices)
            return DatasetBundle(
                train_data=train_data,
                eval_train_data=eval_train_data,
                test_data=test_data,
                name=dataset,
                train_transform=(
                    "RandomCrop(32, padding=4) -> RandomHorizontalFlip() -> "
                    f"ToTensor() -> Normalize(mean={CIFAR10_MEAN}, std={CIFAR10_STD})"
                ),
                eval_transform=(
                    f"ToTensor() -> Normalize(mean={CIFAR10_MEAN}, std={CIFAR10_STD})"
                ),
                train_size_full=len(train),
                test_size_full=len(test),
                train_size_used=len(train_data),
                test_size_used=len(test_data),
            )

        transform = transforms.ToTensor()
        dataset_cls = {
            "mnist": datasets.MNIST,
            "fashion_mnist": datasets.FashionMNIST,
        }[dataset]
        train = dataset_cls(
            root=data_dir,
            train=True,
            download=allow_download,
            transform=transform,
        )
        test = dataset_cls(
            root=data_dir,
            train=False,
            download=allow_download,
            transform=transform,
        )
        train = deterministic_subset(train, train_subset, seed)
        test = deterministic_subset(test, test_subset, seed + 1)
        return DatasetBundle(
            train_data=train,
            eval_train_data=train,
            test_data=test,
            name=dataset,
            train_transform="ToTensor()",
            eval_transform="same dataset as training data",
            train_size_full=len(train),
            test_size_full=len(test),
            train_size_used=len(train),
            test_size_used=len(test),
        )
    except Exception as exc:
        if dataset == "cifar10":
            raise RuntimeError(
                "Could not load CIFAR-10. Re-run with --download or place the "
                "torchvision CIFAR-10 files under the selected --data-dir."
            ) from exc

        try:
            from sklearn.datasets import load_digits
        except Exception as sklearn_exc:
            raise RuntimeError(
                f"Could not load {dataset} and sklearn digits fallback is unavailable."
            ) from sklearn_exc

        print(f"{dataset} unavailable ({exc}). Falling back to sklearn digits.")
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
        images = torch.nn.functional.interpolate(
            images,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        labels = torch.tensor(digits.target, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(labels), generator=generator)
        split = int(0.8 * len(labels))
        train = TensorDataset(images[indices[:split]], labels[indices[:split]])
        test = TensorDataset(images[indices[split:]], labels[indices[split:]])
        train = deterministic_subset(train, train_subset, seed)
        test = deterministic_subset(test, test_subset, seed + 1)
        return DatasetBundle(
            train_data=train,
            eval_train_data=train,
            test_data=test,
            name="sklearn_digits",
            train_transform="sklearn digits scaled to [0, 1] and resized to 28x28",
            eval_transform="same TensorDataset as training data",
            train_size_full=len(train),
            test_size_full=len(test),
            train_size_used=len(train),
            test_size_used=len(test),
        )


def load_idx_image_dataset(folder: Path, train: bool) -> TensorDataset:
    """Load Fashion-MNIST style IDX gzip files from a flat folder."""
    image_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
    label_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"
    image_path = folder / image_name
    label_path = folder / label_name
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing IDX files in {folder}")

    with gzip.open(image_path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"Unexpected image IDX header in {image_path}")
        image_data = handle.read()
    with gzip.open(label_path, "rb") as handle:
        magic, label_count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX header in {label_path}")
        label_data = handle.read()
    if count != label_count:
        raise ValueError("Image and label counts do not match.")

    images = torch.frombuffer(bytearray(image_data), dtype=torch.uint8)
    images = images.reshape(count, 1, rows, cols).float() / 255.0
    labels = torch.frombuffer(bytearray(label_data), dtype=torch.uint8).long()
    return TensorDataset(images, labels)


def deterministic_subset(dataset, size: int, seed: int):
    """Return a deterministic subset, or the original dataset if size <= 0."""
    indices = deterministic_indices(len(dataset), size, seed)
    return subset_from_indices(dataset, indices)


def deterministic_indices(length: int, size: int, seed: int) -> list[int] | None:
    """Return deterministic subset indices, or None to keep the full dataset."""
    if size <= 0 or size >= length:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:size].tolist()


def subset_from_indices(dataset, indices: list[int] | None):
    """Apply optional subset indices to a dataset."""
    if indices is None:
        return dataset
    return Subset(dataset, indices)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic dataloader."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size
    return total_loss / total, total_correct / total


def format_epoch_metrics(
    run_name: str,
    row: EpochMetrics,
    elapsed_seconds: float,
) -> str:
    """Return a compact one-line epoch progress report."""
    message = (
        f"[{run_name}] epoch {row.epoch}: "
        f"train_loss={row.train_loss:.4f}, train_acc={row.train_accuracy:.4f}, "
        f"test_loss={row.test_loss:.4f}, test_acc={row.test_accuracy:.4f}, "
        f"epoch_elapsed={elapsed_seconds:.1f}s, "
        f"total_elapsed={row.elapsed_seconds:.1f}s, steps={row.optimizer_steps}"
    )
    if row.mean_alpha is not None:
        message += (
            f", mean_alpha={row.mean_alpha:.3e}, "
            f"mean_rho={row.mean_rho:.3f}, accepted={row.accepted_rate:.3f}"
        )
    return message


def maybe_print_epoch(
    run_name: str,
    row: EpochMetrics,
    elapsed_seconds: float,
    checkpoint: CheckpointConfig | None,
) -> None:
    """Print progress on the configured cadence."""
    if checkpoint is None or checkpoint.print_every <= 0:
        return
    if row.epoch == 1 or row.epoch % checkpoint.print_every == 0:
        print(format_epoch_metrics(run_name, row, elapsed_seconds), flush=True)


def maybe_save_checkpoint(
    run_name: str,
    model: nn.Module,
    epoch: int,
    metrics: list[EpochMetrics],
    checkpoint: CheckpointConfig | None,
    optimizer_state: dict[str, object] | None = None,
) -> None:
    """Save a lightweight checkpoint on the configured cadence."""
    if checkpoint is None or checkpoint.save_every <= 0:
        return
    if epoch % checkpoint.save_every != 0:
        return

    checkpoint_dir = checkpoint.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{checkpoint.dataset_name}_{slugify(run_name)}_epoch_{epoch:04d}.pt"
    payload: dict[str, object] = {
        "run_name": run_name,
        "dataset": checkpoint.dataset_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": [row.__dict__ for row in metrics],
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    torch.save(payload, path)
    print(f"[{run_name}] checkpoint saved: {path}", flush=True)


def train_vanilla_adam(
    model: nn.Module,
    train_loader: DataLoader,
    eval_train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    run_name: str = "vanilla_adam",
    checkpoint: CheckpointConfig | None = None,
) -> list[EpochMetrics]:
    """Train a model with PyTorch Adam."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []
    run_start_time = time.perf_counter()
    optimizer_steps = 0

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        set_seed(seed + epoch)
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        elapsed = time.perf_counter() - start_time
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                elapsed_seconds=time.perf_counter() - run_start_time,
                optimizer_steps=optimizer_steps,
            )
        )
        maybe_print_epoch(run_name, metrics[-1], elapsed, checkpoint)
        maybe_save_checkpoint(
            run_name,
            model,
            epoch,
            metrics,
            checkpoint,
            optimizer_state=optimizer.state_dict(),
        )

    return metrics


def train_controlled_adam(
    model: nn.Module,
    train_loader: DataLoader,
    eval_train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    alpha0: float,
    kp: float,
    rho_star: float,
    rho_beta: float,
    alpha_min: float,
    alpha_max: float,
    min_alpha_factor: float,
    max_alpha_factor: float,
    trust_region_expand: bool,
    trust_region_rho_threshold: float,
    trust_region_alpha_threshold: float,
    trust_region_expand_factor: float,
    seed: int,
    use_rho_ema: bool = True,
    reject_bad_steps: bool = True,
    run_name: str = "controlled_adam",
    checkpoint: CheckpointConfig | None = None,
) -> tuple[list[EpochMetrics], list[ControlledAdamStep]]:
    """Train a model with same-minibatch controlled Adam."""
    optimizer = TorchControlledAdam(
        model.parameters(),
        alpha0=alpha0,
        kp=kp,
        rho_star=rho_star,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        rho_beta=rho_beta,
        use_rho_ema=use_rho_ema,
        min_alpha_factor=min_alpha_factor,
        max_alpha_factor=max_alpha_factor,
        trust_region_expand=trust_region_expand,
        trust_region_rho_threshold=trust_region_rho_threshold,
        trust_region_alpha_threshold=trust_region_alpha_threshold,
        trust_region_expand_factor=trust_region_expand_factor,
        reject_bad_steps=reject_bad_steps,
        max_backtracks=3,
    )
    metrics = []
    step_logs = []
    run_start_time = time.perf_counter()
    optimizer_steps = 0

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        set_seed(seed + epoch)
        model.train()
        epoch_logs = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss_before = criterion(model(x), y)
            loss_before.backward()

            def same_batch_loss() -> torch.Tensor:
                return criterion(model(x), y)

            step_log = optimizer.step(loss_before, same_batch_loss)
            step_logs.append(step_log)
            epoch_logs.append(step_log)
            optimizer_steps += 1

        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        finite_rhos = [log.rho for log in epoch_logs if np.isfinite(log.rho)]
        elapsed = time.perf_counter() - start_time
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                elapsed_seconds=time.perf_counter() - run_start_time,
                optimizer_steps=optimizer_steps,
                mean_alpha=float(np.mean([log.alpha for log in epoch_logs])),
                mean_rho=float(np.mean(finite_rhos)) if finite_rhos else float("nan"),
                accepted_rate=float(np.mean([log.accepted for log in epoch_logs])),
            )
        )
        maybe_print_epoch(run_name, metrics[-1], elapsed, checkpoint)
        maybe_save_checkpoint(
            run_name,
            model,
            epoch,
            metrics,
            checkpoint,
            optimizer_state={
                "alpha": optimizer.alpha,
                "rho_ema": optimizer.rho_ema,
                "step_count": optimizer.step_count,
                "m": optimizer.m,
                "v": optimizer.v,
            },
        )

    return metrics, step_logs


def save_epoch_metrics(
    path: Path,
    runs: list[OptimizerRun],
) -> None:
    """Save epoch-level metrics as CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "optimizer",
                "epoch",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
                "elapsed_seconds",
                "optimizer_steps",
                "mean_alpha",
                "mean_rho",
                "accepted_rate",
            ]
        )
        for run in runs:
            for row in run.metrics:
                writer.writerow(
                    [
                        run.name,
                        row.epoch,
                        row.train_loss,
                        row.train_accuracy,
                        row.test_loss,
                        row.test_accuracy,
                        row.elapsed_seconds,
                        row.optimizer_steps,
                        "" if row.mean_alpha is None else row.mean_alpha,
                        "" if row.mean_rho is None else row.mean_rho,
                        "" if row.accepted_rate is None else row.accepted_rate,
                    ]
            )


def save_step_logs(
    path: Path,
    step_logs: list[ControlledAdamStep],
    optimizer_name: str | None = None,
) -> None:
    """Save controlled Adam step diagnostics as CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "step",
            "loss_before",
            "loss_after",
            "alpha",
            "rho",
            "predicted_decrease",
            "actual_decrease",
            "accepted",
            "descent_score",
            "backtracks",
            "rho_ema",
            "alpha_next",
            "alpha_update_factor",
            "trust_region_expanded",
        ]
        if optimizer_name is not None:
            header.insert(0, "optimizer")
        writer.writerow(header)
        for i, log in enumerate(step_logs):
            row = [
                i,
                log.loss_before,
                log.loss_after,
                log.alpha,
                log.rho,
                log.predicted_decrease,
                log.actual_decrease,
                int(log.accepted),
                log.descent_score,
                log.backtracks,
                log.rho_ema,
                log.alpha_next,
                log.alpha_update_factor,
                int(log.trust_region_expanded),
            ]
            if optimizer_name is not None:
                row.insert(0, optimizer_name)
            writer.writerow(row)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def optimizer_variant_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Describe optimizer variants used by this run."""
    variants: list[dict[str, object]] = [
        {
            "name": "vanilla_adam",
            "type": "torch.optim.Adam",
            "learning_rate": args.lr,
        }
    ]
    controlled_common = {
        "direction": "Adam moments with same-minibatch actual/predicted controller",
        "alpha0": args.lr,
        "kp": args.controlled_kp,
        "rho_star": args.controlled_rho_star,
        "rho_beta": args.controlled_rho_beta,
        "alpha_min": args.controlled_alpha_min,
        "alpha_max": args.controlled_alpha_max,
        "min_alpha_factor": args.controlled_min_alpha_factor,
        "max_alpha_factor": args.controlled_max_alpha_factor,
        "trust_region_expand": args.controlled_trust_region_expand,
        "trust_region_rho_threshold": args.controlled_trust_rho_threshold,
        "trust_region_alpha_threshold": args.controlled_trust_alpha_threshold,
        "trust_region_expand_factor": args.controlled_trust_expand_factor,
        "max_backtracks": 3,
        "same_minibatch_trial_loss": True,
    }

    if args.ablation:
        variants.extend(
            [
                {
                    "name": "fixed_adam_direction",
                    "type": "TorchControlledAdam direction with fixed alpha",
                    "alpha": args.lr,
                    "reject_bad_steps": False,
                },
                {
                    **controlled_common,
                    "name": "controlled_raw_rho",
                    "use_rho_ema": False,
                    "trust_region_expand": False,
                    "reject_bad_steps": True,
                },
                {
                    **controlled_common,
                    "name": "controlled_ema",
                    "use_rho_ema": True,
                    "trust_region_expand": False,
                    "reject_bad_steps": True,
                },
                {
                    **controlled_common,
                    "name": "controlled_ema_trust",
                    "use_rho_ema": True,
                    "reject_bad_steps": True,
                },
            ]
        )
    else:
        variants.append(
            {
                **controlled_common,
                "name": "controlled_adam",
                "use_rho_ema": True,
                "reject_bad_steps": True,
            }
        )
    return variants


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_bundle: DatasetBundle,
    model: nn.Module,
    dataset_name: str,
) -> None:
    """Write detailed benchmark metadata to JSON and text files."""
    resolved_model = "cnn" if dataset_name == "cifar10" and args.model == "auto" else args.model
    if resolved_model == "auto":
        resolved_model = "mlp"
    metadata = {
        "script": Path(__file__).name,
        "python": sys.version,
        "torch_version": torch.__version__,
        "device": args.device,
        "seed": args.seed,
        "dataset": {
            "requested": args.dataset,
            "resolved": dataset_name,
            "data_dir": str(args.data_dir),
            "fashion_folder": None if args.fashion_folder is None else str(args.fashion_folder),
            "download": args.download,
            "train_size_full": dataset_bundle.train_size_full,
            "test_size_full": dataset_bundle.test_size_full,
            "train_size_used": dataset_bundle.train_size_used,
            "test_size_used": dataset_bundle.test_size_used,
            "train_transform": dataset_bundle.train_transform,
            "eval_transform": dataset_bundle.eval_transform,
        },
        "model": {
            "requested": args.model,
            "resolved": resolved_model,
            "class": model.__class__.__name__,
            "trainable_parameters": count_parameters(model),
            "architecture": str(model),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "criterion": "CrossEntropyLoss",
            "eval_train_transform_is_deterministic": dataset_name == "cifar10",
            "epoch_seed_rule": "set_seed(seed + epoch) for each optimizer variant",
            "epoch_metrics_include": [
                "epoch",
                "elapsed_seconds",
                "optimizer_steps",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            ],
            "checkpoint_every": args.checkpoint_every,
            "print_every": args.print_every,
        },
        "optimizers": optimizer_variant_specs(args),
    }

    json_path = output_dir / "run_metadata.json"
    txt_path = output_dir / "run_metadata.txt"
    with json_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    with txt_path.open("w") as handle:
        handle.write("Benchmark metadata\n")
        handle.write("==================\n\n")
        handle.write(json.dumps(metadata, indent=2))
        handle.write("\n")


def slugify(name: str) -> str:
    """Return a filesystem-friendly optimizer name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_metrics(
    output_dir: Path,
    dataset_name: str,
    runs: list[OptimizerRun],
) -> None:
    """Save comparison plots."""
    epochs = [row.epoch for row in runs[0].metrics]

    plt.figure(figsize=(7, 4))
    for run in runs:
        plt.plot(epochs, [row.test_loss for row in run.metrics], label=run.name)
    plt.xlabel("Epoch")
    plt.ylabel("Test cross-entropy")
    plt.title(f"{dataset_name} test loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for run in runs:
        plt.plot(epochs, [row.train_loss for row in run.metrics], label=run.name)
    plt.xlabel("Epoch")
    plt.ylabel("Train cross-entropy")
    plt.title(f"{dataset_name} train loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_train_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for run in runs:
        line = plt.plot(epochs, [row.test_loss for row in run.metrics], label=f"{run.name} test")[0]
        plt.plot(
            epochs,
            [row.train_loss for row in run.metrics],
            linestyle="--",
            color=line.get_color(),
            label=f"{run.name} train",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.title(f"{dataset_name} train vs test loss")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_train_test_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for run in runs:
        plt.plot(epochs, [row.test_accuracy for row in run.metrics], label=run.name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset_name} test accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_accuracy.png", dpi=160)
    plt.close()

    plot_metric_by_axis(
        output_dir,
        dataset_name,
        runs,
        axis_attr="optimizer_steps",
        axis_label="Optimizer steps",
        axis_slug="steps",
    )
    if all(any(row.elapsed_seconds > 0.0 for row in run.metrics) for run in runs):
        plot_metric_by_axis(
            output_dir,
            dataset_name,
            runs,
            axis_attr="elapsed_seconds",
            axis_label="Wall-clock seconds",
            axis_slug="time",
        )

    runs_with_steps = [run for run in runs if run.step_logs]
    if runs_with_steps:
        plt.figure(figsize=(7, 4))
        for run in runs_with_steps:
            assert run.step_logs is not None
            plt.plot([log.alpha for log in run.step_logs], label=run.name)
            expanded_steps = [
                (i, log.alpha)
                for i, log in enumerate(run.step_logs)
                if log.trust_region_expanded
            ]
            if expanded_steps:
                plt.scatter(
                    [i for i, _ in expanded_steps],
                    [alpha for _, alpha in expanded_steps],
                    s=12,
                    zorder=3,
                )
        plt.xlabel("Minibatch step")
        plt.ylabel("alpha")
        plt.title("Adam-direction global step size")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_controlled_alpha.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        for run in runs_with_steps:
            assert run.step_logs is not None
            finite = [
                (i, log.rho_ema)
                for i, log in enumerate(run.step_logs)
                if np.isfinite(log.rho_ema)
            ]
            if finite:
                plt.plot([i for i, _ in finite], [rho for _, rho in finite], label=run.name)
        plt.xlabel("Minibatch step")
        plt.ylabel("rho control signal")
        plt.title("Same-minibatch rho control signal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_controlled_rho.png", dpi=160)
        plt.close()


def plot_metric_by_axis(
    output_dir: Path,
    dataset_name: str,
    runs: list[OptimizerRun],
    *,
    axis_attr: str,
    axis_label: str,
    axis_slug: str,
) -> None:
    """Save loss and accuracy plots against a non-epoch x-axis."""
    series = [(run, [float(getattr(row, axis_attr)) for row in run.metrics]) for run in runs]
    if not all(any(value > 0.0 for value in values) for _, values in series):
        return

    metric_specs = [
        ("test_loss", "Test cross-entropy", "loss"),
        ("train_loss", "Train cross-entropy", "train_loss"),
        ("test_accuracy", "Test accuracy", "accuracy"),
    ]
    for metric_attr, ylabel, metric_slug in metric_specs:
        plt.figure(figsize=(7, 4))
        for run, x_values in series:
            plt.plot(x_values, [getattr(row, metric_attr) for row in run.metrics], label=run.name)
        plt.xlabel(axis_label)
        plt.ylabel(ylabel)
        plt.title(f"{dataset_name} {ylabel.lower()} vs {axis_label.lower()}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_{metric_slug}_vs_{axis_slug}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    for run, x_values in series:
        line = plt.plot(x_values, [row.test_loss for row in run.metrics], label=f"{run.name} test")[0]
        plt.plot(
            x_values,
            [row.train_loss for row in run.metrics],
            linestyle="--",
            color=line.get_color(),
            label=f"{run.name} train",
        )
    plt.xlabel(axis_label)
    plt.ylabel("Cross-entropy")
    plt.title(f"{dataset_name} train vs test loss vs {axis_label.lower()}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_train_test_loss_vs_{axis_slug}.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10"],
        default="fashion_mnist",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "mlp", "cnn", "lenet_cifar", "fashion_cnn"],
        default="auto",
        help="Classifier architecture. Auto uses CNN for CIFAR-10 and MLP otherwise.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=4096)
    parser.add_argument("--test-subset", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--controlled-kp", type=float, default=0.05)
    parser.add_argument("--controlled-rho-star", type=float, default=0.7)
    parser.add_argument("--controlled-rho-beta", type=float, default=0.9)
    parser.add_argument("--controlled-alpha-min", type=float, default=1e-5)
    parser.add_argument("--controlled-alpha-max", type=float, default=5e-2)
    parser.add_argument("--controlled-min-alpha-factor", type=float, default=0.8)
    parser.add_argument("--controlled-max-alpha-factor", type=float, default=1.05)
    parser.add_argument(
        "--controlled-trust-region-expand",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--controlled-trust-rho-threshold", type=float, default=0.9)
    parser.add_argument("--controlled-trust-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--controlled-trust-expand-factor", type=float, default=1.5)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run Adam-direction ablations instead of only vanilla vs controlled.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--fashion-folder",
        type=Path,
        default=None,
        help="Flat folder containing Fashion-MNIST IDX gzip files.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a model/optimizer checkpoint every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print epoch metrics every N epochs. Use 0 to disable progress output.",
    )
    return parser.parse_args()


def run_controlled_variant(
    name: str,
    base_state: dict[str, torch.Tensor],
    train_data,
    eval_train_loader: DataLoader,
    eval_test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    dataset_name: str,
    *,
    alpha0: float,
    kp: float,
    rho_beta: float,
    min_alpha_factor: float,
    max_alpha_factor: float,
    trust_region_expand: bool,
    use_rho_ema: bool,
    reject_bad_steps: bool,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    checkpoint: CheckpointConfig | None = None,
) -> OptimizerRun:
    """Train one Adam-direction variant from the shared initialization."""
    model = make_model(args.model, dataset_name).to(device)
    model.load_state_dict(base_state)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed)
    metrics, step_logs = train_controlled_adam(
        model,
        train_loader,
        eval_train_loader,
        eval_test_loader,
        criterion,
        device,
        args.epochs,
        alpha0,
        kp,
        args.controlled_rho_star,
        rho_beta,
        args.controlled_alpha_min if alpha_min is None else alpha_min,
        args.controlled_alpha_max if alpha_max is None else alpha_max,
        min_alpha_factor,
        max_alpha_factor,
        trust_region_expand,
        args.controlled_trust_rho_threshold,
        args.controlled_trust_alpha_threshold,
        args.controlled_trust_expand_factor,
        args.seed,
        use_rho_ema=use_rho_ema,
        reject_bad_steps=reject_bad_steps,
        run_name=name,
        checkpoint=checkpoint,
    )
    return OptimizerRun(name, metrics, step_logs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path("outputs") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_image_dataset_bundle(
        args.data_dir,
        args.dataset,
        args.train_subset,
        args.test_subset,
        args.seed,
        args.download,
        args.fashion_folder,
    )
    train_data = datasets.train_data
    eval_train_data = datasets.eval_train_data
    test_data = datasets.test_data
    dataset_name = datasets.name
    checkpoint = CheckpointConfig(
        output_dir=output_dir,
        dataset_name=dataset_name,
        save_every=args.checkpoint_every,
        print_every=args.print_every,
    )
    eval_train_loader = make_loader(eval_train_data, args.batch_size, False, args.seed)
    eval_test_loader = make_loader(test_data, args.batch_size, False, args.seed)

    base_model = make_model(args.model, dataset_name).to(device)
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base_model.state_dict().items()
    }
    save_run_metadata(output_dir, args, datasets, base_model, dataset_name)

    criterion = nn.CrossEntropyLoss()
    runs = []

    vanilla_model = make_model(args.model, dataset_name).to(device)
    vanilla_model.load_state_dict(base_state)
    vanilla_metrics = train_vanilla_adam(
        vanilla_model,
        make_loader(train_data, args.batch_size, True, args.seed),
        eval_train_loader,
        eval_test_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
        args.seed,
        run_name="vanilla_adam",
        checkpoint=checkpoint,
    )
    runs.append(OptimizerRun("vanilla_adam", vanilla_metrics))

    if args.ablation:
        runs.append(
            run_controlled_variant(
                "fixed_adam_direction",
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                dataset_name,
                alpha0=args.lr,
                kp=0.0,
                rho_beta=0.0,
                min_alpha_factor=1.0,
                max_alpha_factor=1.0,
                trust_region_expand=False,
                use_rho_ema=False,
                reject_bad_steps=False,
                alpha_min=args.lr,
                alpha_max=args.lr,
                checkpoint=checkpoint,
            )
        )
        runs.append(
            run_controlled_variant(
                "controlled_raw_rho",
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                dataset_name,
                alpha0=args.lr,
                kp=args.controlled_kp,
                rho_beta=0.0,
                min_alpha_factor=args.controlled_min_alpha_factor,
                max_alpha_factor=args.controlled_max_alpha_factor,
                trust_region_expand=False,
                use_rho_ema=False,
                reject_bad_steps=True,
                checkpoint=checkpoint,
            )
        )
        runs.append(
            run_controlled_variant(
                "controlled_ema",
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                dataset_name,
                alpha0=args.lr,
                kp=args.controlled_kp,
                rho_beta=args.controlled_rho_beta,
                min_alpha_factor=args.controlled_min_alpha_factor,
                max_alpha_factor=args.controlled_max_alpha_factor,
                trust_region_expand=False,
                use_rho_ema=True,
                reject_bad_steps=True,
                checkpoint=checkpoint,
            )
        )
        runs.append(
            run_controlled_variant(
                "controlled_ema_trust",
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                dataset_name,
                alpha0=args.lr,
                kp=args.controlled_kp,
                rho_beta=args.controlled_rho_beta,
                min_alpha_factor=args.controlled_min_alpha_factor,
                max_alpha_factor=args.controlled_max_alpha_factor,
                trust_region_expand=args.controlled_trust_region_expand,
                use_rho_ema=True,
                reject_bad_steps=True,
                checkpoint=checkpoint,
            )
        )
    else:
        runs.append(
            run_controlled_variant(
                "controlled_adam",
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                dataset_name,
                alpha0=args.lr,
                kp=args.controlled_kp,
                rho_beta=args.controlled_rho_beta,
                min_alpha_factor=args.controlled_min_alpha_factor,
                max_alpha_factor=args.controlled_max_alpha_factor,
                trust_region_expand=args.controlled_trust_region_expand,
                use_rho_ema=True,
                reject_bad_steps=True,
                checkpoint=checkpoint,
            )
        )

    prefix = dataset_name
    save_epoch_metrics(
        output_dir / f"{prefix}_epoch_metrics.csv",
        runs,
    )
    for run in runs:
        if run.step_logs is None:
            continue
        diagnostics_name = (
            f"{prefix}_controlled_step_diagnostics.csv"
            if not args.ablation and run.name == "controlled_adam"
            else f"{prefix}_{slugify(run.name)}_step_diagnostics.csv"
        )
        save_step_logs(
            output_dir / diagnostics_name,
            run.step_logs,
            optimizer_name=run.name,
        )
    plot_metrics(output_dir, dataset_name, runs)

    print(f"Dataset: {dataset_name}")
    print(f"Outputs written to: {output_dir.resolve()}")
    for run in runs:
        final = run.metrics[-1]
        print(f"{run.name} final test accuracy: {final.test_accuracy:.4f}")
        if final.accepted_rate is not None:
            print(f"{run.name} accepted rate: {final.accepted_rate:.4f}")


if __name__ == "__main__":
    main()
