"""Compare vanilla Muon and controlled Muon on image classification tasks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from controlled_muon.torch_optimizers import ControlledMuonStep, MuonConfig, TorchControlledMuon


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SmallMLP(nn.Module):
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


class SmallCIFARCNN(nn.Module):
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
        return self.classifier(self.features(x))


def _reshape_param_for_muon(param: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(param.shape)
    if param.ndim == 0:
        raise ValueError("Muon parameters must have at least one dimension.")
    if param.ndim == 1:
        matrix = param.detach().clone().reshape(-1, 1)
    elif param.ndim == 2:
        matrix = param.detach().clone()
    else:
        matrix = param.detach().clone().reshape(param.shape[0], -1)
    return matrix, shape


def _direction_for_param(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    config: MuonConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_momentum = config.momentum * momentum_buffer + grad
    if config.nesterov:
        matrix_to_orthogonalize = config.momentum * new_momentum + grad
    else:
        matrix_to_orthogonalize = new_momentum
    matrix, shape = _reshape_param_for_muon(matrix_to_orthogonalize)
    from controlled_muon.orthogonalization import orthogonalize

    ortho_update = orthogonalize(matrix.cpu().numpy(), method=config.orthogonalizer, ns_steps=config.ns_steps)
    direction = -config.update_scale * torch.from_numpy(ortho_update).to(device=grad.device, dtype=grad.dtype).reshape(shape)
    return direction, new_momentum


def make_model(model_name: str, dataset_name: str) -> nn.Module:
    if model_name == "auto":
        model_name = "cnn" if dataset_name == "cifar10" else "mlp"
    if model_name == "mlp":
        if dataset_name == "cifar10":
            raise ValueError("The MLP model expects 28x28 grayscale inputs, not CIFAR-10.")
        return SmallMLP()
    if model_name == "cnn":
        if dataset_name != "cifar10":
            raise ValueError("The CNN model is currently configured for CIFAR-10 RGB inputs.")
        return SmallCIFARCNN()
    raise ValueError(f"Unknown model: {model_name}")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    mean_alpha: float | None = None
    mean_rho: float | None = None
    accepted_rate: float | None = None


@dataclass
class OptimizerRun:
    name: str
    metrics: list[EpochMetrics]
    step_logs: list[ControlledMuonStep] | None = None


@dataclass
class DatasetBundle:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def deterministic_indices(length: int, size: int, seed: int) -> list[int] | None:
    if size <= 0 or size >= length:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:size].tolist()


def subset_from_indices(dataset, indices: list[int] | None):
    if indices is None:
        return dataset
    return Subset(dataset, indices)


def deterministic_subset(dataset, size: int, seed: int):
    indices = deterministic_indices(len(dataset), size, seed)
    return subset_from_indices(dataset, indices)


def load_idx_image_dataset(folder: Path, train: bool) -> TensorDataset:
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


def load_image_dataset_bundle(
    data_dir: Path,
    dataset: str,
    train_subset: int,
    test_subset: int,
    seed: int,
    allow_download: bool,
    fashion_folder: Path | None,
) -> DatasetBundle:
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
            train = datasets.CIFAR10(root=data_dir, train=True, download=allow_download, transform=train_transform)
            eval_train = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)
            test = datasets.CIFAR10(root=data_dir, train=False, download=allow_download, transform=eval_transform)
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
                train_transform="RandomCrop(32,padding=4) -> RandomHorizontalFlip() -> ToTensor() -> Normalize",
                eval_transform="ToTensor() -> Normalize",
                train_size_full=len(train),
                test_size_full=len(test),
                train_size_used=len(train_data),
                test_size_used=len(test_data),
            )

        transform = transforms.ToTensor()
        dataset_cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[dataset]
        train = dataset_cls(root=data_dir, train=True, download=allow_download, transform=transform)
        test = dataset_cls(root=data_dir, train=False, download=allow_download, transform=transform)
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
                "Could not load CIFAR-10. Re-run with --download or place the torchvision CIFAR-10 files under --data-dir."
            ) from exc
        try:
            from sklearn.datasets import load_digits
        except Exception as sklearn_exc:
            raise RuntimeError(f"Could not load {dataset} and sklearn digits fallback is unavailable.") from sklearn_exc
        print(f"{dataset} unavailable ({exc}). Falling back to sklearn digits.")
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
        images = torch.nn.functional.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
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


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
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


def train_vanilla_muon(
    model: nn.Module,
    train_loader: DataLoader,
    eval_train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    config: MuonConfig,
) -> list[EpochMetrics]:
    metrics = []
    config = config
    momentum_buffers = [torch.zeros_like(param) for param in model.parameters() if param.requires_grad]
    for epoch in range(1, epochs + 1):
        set_seed(seed + epoch)
        model.train()
        mb_index = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            loss = criterion(model(x), y)
            loss.backward()
            with torch.no_grad():
                muon_params = [param for param in model.parameters() if param.requires_grad]
                for idx, param in enumerate(muon_params):
                    if param.grad is None:
                        continue
                    direction, new_momentum = _direction_for_param(param.grad.detach(), momentum_buffers[idx], config)
                    momentum_buffers[idx] = new_momentum
                    param.add_(direction, alpha=lr)
                    mb_index += 1
        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        metrics.append(EpochMetrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy))
    return metrics


def train_controlled_muon(
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
) -> tuple[list[EpochMetrics], list[ControlledMuonStep]]:
    optimizer = TorchControlledMuon(
        model.parameters(),
        alpha0=alpha0,
        config=MuonConfig(),
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
    for epoch in range(1, epochs + 1):
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


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_metrics(output_dir: Path, dataset_name: str, runs: list[OptimizerRun]) -> None:
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

    runs_with_steps = [run for run in runs if run.step_logs]
    if runs_with_steps:
        plt.figure(figsize=(7, 4))
        for run in runs_with_steps:
            assert run.step_logs is not None
            plt.plot([log.alpha for log in run.step_logs], label=run.name)
        plt.xlabel("Minibatch step")
        plt.ylabel("alpha")
        plt.title("Muon global step size")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_controlled_alpha.png", dpi=160)
        plt.close()


def save_epoch_metrics(path: Path, runs: list[OptimizerRun]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["optimizer", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "mean_alpha", "mean_rho", "accepted_rate"])
        for run in runs:
            for row in run.metrics:
                writer.writerow([
                    run.name,
                    row.epoch,
                    row.train_loss,
                    row.train_accuracy,
                    row.test_loss,
                    row.test_accuracy,
                    "" if row.mean_alpha is None else row.mean_alpha,
                    "" if row.mean_rho is None else row.mean_rho,
                    "" if row.accepted_rate is None else row.accepted_rate,
                ])


def save_step_logs(path: Path, step_logs: list[ControlledMuonStep], optimizer_name: str | None = None) -> None:
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
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def optimizer_variant_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    variants = [
        {
            "name": "vanilla_muon",
            "type": "fixed-step Muon implemented in this runner",
            "learning_rate": args.lr,
            "direction": "Muon-like matrix orthogonalization",
        }
    ]
    controlled_common = {
        "direction": "Muon orthogonalized direction with same-minibatch actual/predicted controller",
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
                    "name": "fixed_muon_direction",
                    "type": "TorchControlledMuon direction with fixed alpha",
                    "alpha": args.lr,
                    "reject_bad_steps": False,
                },
                {**controlled_common, "name": "controlled_raw_rho", "use_rho_ema": False, "trust_region_expand": False, "reject_bad_steps": True},
                {**controlled_common, "name": "controlled_ema", "use_rho_ema": True, "trust_region_expand": False, "reject_bad_steps": True},
                {**controlled_common, "name": "controlled_ema_trust", "use_rho_ema": True, "reject_bad_steps": True},
            ]
        )
    else:
        variants.append({**controlled_common, "name": "controlled_muon", "use_rho_ema": True, "reject_bad_steps": True})
    return variants


def save_run_metadata(output_dir: Path, args: argparse.Namespace, dataset_bundle: DatasetBundle, model: nn.Module, dataset_name: str) -> None:
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
        },
        "optimizers": optimizer_variant_specs(args),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "run_metadata.txt").write_text("Benchmark metadata\n==================\n\n" + json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], default="fashion_mnist")
    parser.add_argument("--model", choices=["auto", "mlp", "cnn"], default="auto")
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
    parser.add_argument("--controlled-trust-region-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controlled-trust-rho-threshold", type=float, default=0.9)
    parser.add_argument("--controlled-trust-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--controlled-trust-expand-factor", type=float, default=1.5)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fashion-folder", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
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
) -> OptimizerRun:
    model = make_model(args.model, dataset_name).to(device)
    model.load_state_dict(base_state)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed)
    metrics, step_logs = train_controlled_muon(
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
    )
    return OptimizerRun(name, metrics, step_logs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path("outputs") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_image_dataset_bundle(args.data_dir, args.dataset, args.train_subset, args.test_subset, args.seed, args.download, args.fashion_folder)
    train_data = datasets.train_data
    eval_train_data = datasets.eval_train_data
    test_data = datasets.test_data
    dataset_name = datasets.name
    eval_train_loader = make_loader(eval_train_data, args.batch_size, False, args.seed)
    eval_test_loader = make_loader(test_data, args.batch_size, False, args.seed)

    base_model = make_model(args.model, dataset_name).to(device)
    base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    save_run_metadata(output_dir, args, datasets, base_model, dataset_name)

    criterion = nn.CrossEntropyLoss()
    runs: list[OptimizerRun] = []

    vanilla_model = make_model(args.model, dataset_name).to(device)
    vanilla_model.load_state_dict(base_state)
    vanilla_metrics = train_vanilla_muon(
        vanilla_model,
        make_loader(train_data, args.batch_size, True, args.seed),
        eval_train_loader,
        eval_test_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
        args.seed,
        MuonConfig(),
    )
    runs.append(OptimizerRun("vanilla_muon", vanilla_metrics))

    if args.ablation:
        runs.append(
            run_controlled_variant(
                "fixed_muon_direction",
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
            )
        )
        runs.append(run_controlled_variant("controlled_raw_rho", base_state, train_data, eval_train_loader, eval_test_loader, criterion, device, args, dataset_name, alpha0=args.lr, kp=args.controlled_kp, rho_beta=0.0, min_alpha_factor=args.controlled_min_alpha_factor, max_alpha_factor=args.controlled_max_alpha_factor, trust_region_expand=False, use_rho_ema=False, reject_bad_steps=True))
        runs.append(run_controlled_variant("controlled_ema", base_state, train_data, eval_train_loader, eval_test_loader, criterion, device, args, dataset_name, alpha0=args.lr, kp=args.controlled_kp, rho_beta=args.controlled_rho_beta, min_alpha_factor=args.controlled_min_alpha_factor, max_alpha_factor=args.controlled_max_alpha_factor, trust_region_expand=False, use_rho_ema=True, reject_bad_steps=True))
        runs.append(run_controlled_variant("controlled_ema_trust", base_state, train_data, eval_train_loader, eval_test_loader, criterion, device, args, dataset_name, alpha0=args.lr, kp=args.controlled_kp, rho_beta=args.controlled_rho_beta, min_alpha_factor=args.controlled_min_alpha_factor, max_alpha_factor=args.controlled_max_alpha_factor, trust_region_expand=args.controlled_trust_region_expand, use_rho_ema=True, reject_bad_steps=True))
    else:
        runs.append(run_controlled_variant("controlled_muon", base_state, train_data, eval_train_loader, eval_test_loader, criterion, device, args, dataset_name, alpha0=args.lr, kp=args.controlled_kp, rho_beta=args.controlled_rho_beta, min_alpha_factor=args.controlled_min_alpha_factor, max_alpha_factor=args.controlled_max_alpha_factor, trust_region_expand=args.controlled_trust_region_expand, use_rho_ema=True, reject_bad_steps=True))

    prefix = dataset_name
    save_epoch_metrics(output_dir / f"{prefix}_epoch_metrics.csv", runs)
    for run in runs:
        if run.step_logs is None:
            continue
        diagnostics_name = f"{prefix}_{slugify(run.name)}_step_diagnostics.csv"
        save_step_logs(output_dir / diagnostics_name, run.step_logs, optimizer_name=run.name)
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
