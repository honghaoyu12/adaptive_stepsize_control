"""Compare vanilla Adam and controlled Adam on a small image-classification MLP."""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from controlled_adam.torch_optimizers import ControlledAdamStep, TorchControlledAdam


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


@dataclass
class EpochMetrics:
    """Metrics collected after one training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    mean_alpha: float | None = None
    mean_rho: float | None = None
    accepted_rate: float | None = None


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
    """Load a torchvision image dataset, falling back to sklearn digits."""
    if dataset == "fashion_mnist" and fashion_folder is not None:
        train = load_idx_image_dataset(fashion_folder, train=True)
        test = load_idx_image_dataset(fashion_folder, train=False)
        return (
            deterministic_subset(train, train_subset, seed),
            deterministic_subset(test, test_subset, seed + 1),
            dataset,
        )

    try:
        from torchvision import datasets, transforms

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
        return (
            deterministic_subset(train, train_subset, seed),
            deterministic_subset(test, test_subset, seed + 1),
            dataset,
        )
    except Exception as exc:
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
        return (
            deterministic_subset(train, train_subset, seed),
            deterministic_subset(test, test_subset, seed + 1),
            "sklearn_digits",
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
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
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


def train_vanilla_adam(
    model: nn.Module,
    train_loader: DataLoader,
    eval_train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[EpochMetrics]:
    """Train a model with PyTorch Adam."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        metrics.append(
            EpochMetrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
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
        min_alpha_factor=min_alpha_factor,
        max_alpha_factor=max_alpha_factor,
        max_backtracks=3,
    )
    metrics = []
    step_logs = []

    for epoch in range(1, epochs + 1):
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

        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        finite_rhos = [log.rho for log in epoch_logs if np.isfinite(log.rho)]
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                mean_alpha=float(np.mean([log.alpha for log in epoch_logs])),
                mean_rho=float(np.mean(finite_rhos)) if finite_rhos else float("nan"),
                accepted_rate=float(np.mean([log.accepted for log in epoch_logs])),
            )
        )

    return metrics, step_logs


def save_epoch_metrics(
    path: Path,
    vanilla: list[EpochMetrics],
    controlled: list[EpochMetrics],
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
                "mean_alpha",
                "mean_rho",
                "accepted_rate",
            ]
        )
        for name, rows in [("vanilla_adam", vanilla), ("controlled_adam", controlled)]:
            for row in rows:
                writer.writerow(
                    [
                        name,
                        row.epoch,
                        row.train_loss,
                        row.train_accuracy,
                        row.test_loss,
                        row.test_accuracy,
                        "" if row.mean_alpha is None else row.mean_alpha,
                        "" if row.mean_rho is None else row.mean_rho,
                        "" if row.accepted_rate is None else row.accepted_rate,
                    ]
                )


def save_step_logs(path: Path, step_logs: list[ControlledAdamStep]) -> None:
    """Save controlled Adam step diagnostics as CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ]
        )
        for i, log in enumerate(step_logs):
            writer.writerow(
                [
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
                ]
            )


def plot_metrics(
    output_dir: Path,
    dataset_name: str,
    vanilla: list[EpochMetrics],
    controlled: list[EpochMetrics],
    step_logs: list[ControlledAdamStep],
) -> None:
    """Save comparison plots."""
    epochs = [row.epoch for row in vanilla]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row.train_loss for row in vanilla], label="Adam train")
    plt.plot(epochs, [row.test_loss for row in vanilla], label="Adam test")
    plt.plot(epochs, [row.train_loss for row in controlled], label="Controlled train")
    plt.plot(epochs, [row.test_loss for row in controlled], label="Controlled test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.title(f"{dataset_name} loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row.test_accuracy for row in vanilla], label="Adam")
    plt.plot(epochs, [row.test_accuracy for row in controlled], label="Controlled Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset_name} test accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot([log.alpha for log in step_logs])
    plt.xlabel("Minibatch step")
    plt.ylabel("alpha")
    plt.title("Controlled Adam global step size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_controlled_alpha.png", dpi=160)
    plt.close()

    finite = [(i, log.rho) for i, log in enumerate(step_logs) if np.isfinite(log.rho)]
    if finite:
        plt.figure(figsize=(7, 4))
        plt.plot([i for i, _ in finite], [rho for _, rho in finite], label="rho")
        finite_ema = [
            (i, log.rho_ema)
            for i, log in enumerate(step_logs)
            if np.isfinite(log.rho_ema)
        ]
        if finite_ema:
            plt.plot(
                [i for i, _ in finite_ema],
                [rho for _, rho in finite_ema],
                label="rho EMA",
            )
        plt.xlabel("Minibatch step")
        plt.ylabel("rho")
        plt.title("Controlled Adam same-minibatch rho")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_controlled_rho.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="fashion_mnist",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path("outputs") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data, dataset_name = load_image_dataset_or_digits(
        args.data_dir,
        args.dataset,
        args.train_subset,
        args.test_subset,
        args.seed,
        args.download,
        args.fashion_folder,
    )
    train_loader_vanilla = make_loader(train_data, args.batch_size, True, args.seed)
    train_loader_controlled = make_loader(train_data, args.batch_size, True, args.seed)
    eval_train_loader = make_loader(train_data, args.batch_size, False, args.seed)
    eval_test_loader = make_loader(test_data, args.batch_size, False, args.seed)

    base_model = SmallMLP().to(device)
    vanilla_model = SmallMLP().to(device)
    controlled_model = SmallMLP().to(device)
    vanilla_model.load_state_dict(base_model.state_dict())
    controlled_model.load_state_dict(base_model.state_dict())

    criterion = nn.CrossEntropyLoss()
    vanilla_metrics = train_vanilla_adam(
        vanilla_model,
        train_loader_vanilla,
        eval_train_loader,
        eval_test_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
    )
    controlled_metrics, step_logs = train_controlled_adam(
        controlled_model,
        train_loader_controlled,
        eval_train_loader,
        eval_test_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
        args.controlled_kp,
        args.controlled_rho_star,
        args.controlled_rho_beta,
        args.controlled_alpha_min,
        args.controlled_alpha_max,
        args.controlled_min_alpha_factor,
        args.controlled_max_alpha_factor,
    )

    prefix = dataset_name
    save_epoch_metrics(
        output_dir / f"{prefix}_epoch_metrics.csv",
        vanilla_metrics,
        controlled_metrics,
    )
    save_step_logs(output_dir / f"{prefix}_controlled_step_diagnostics.csv", step_logs)
    plot_metrics(output_dir, dataset_name, vanilla_metrics, controlled_metrics, step_logs)

    print(f"Dataset: {dataset_name}")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(
        "Vanilla Adam final test accuracy: "
        f"{vanilla_metrics[-1].test_accuracy:.4f}"
    )
    print(
        "Controlled Adam final test accuracy: "
        f"{controlled_metrics[-1].test_accuracy:.4f}"
    )
    print(
        "Controlled Adam accepted rate: "
        f"{controlled_metrics[-1].accepted_rate:.4f}"
    )


if __name__ == "__main__":
    main()
