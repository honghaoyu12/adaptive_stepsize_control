"""Run multi-seed Fashion-MNIST smoke benchmarks for PI and vanilla optimizers.

The runner mirrors the controlled Adam/Muon image demos at a smaller scope:
local Fashion-MNIST IDX gzip files, deterministic subsets, same initial model
per seed, and CSV summaries across seeds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pi_adam_optimizer"))
sys.path.insert(0, str(ROOT / "pi_muon_optimizer"))

from pi_adam import PIAdam  # noqa: E402
from pi_muon import PIMuon, default_muon_param_groups, newton_schulz_orthogonalize  # noqa: E402


@dataclass
class EpochMetrics:
    seed: int
    optimizer: str
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float
    optimizer_steps: int
    mean_alpha: float
    mean_rho: float
    accepted_rate: float
    fallback_rate: float
    mean_backtracks: float


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_idx_image_dataset(folder: Path, train: bool) -> TensorDataset:
    image_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
    label_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"
    image_path = folder / image_name
    label_path = folder / label_name
    with gzip.open(image_path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"Unexpected image header in {image_path}")
        image_data = handle.read()
    with gzip.open(label_path, "rb") as handle:
        magic, label_count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label header in {label_path}")
        label_data = handle.read()
    if count != label_count:
        raise ValueError("Image and label counts do not match.")

    images = torch.frombuffer(bytearray(image_data), dtype=torch.uint8)
    images = images.reshape(count, 1, rows, cols).float() / 255.0
    labels = torch.frombuffer(bytearray(label_data), dtype=torch.uint8).long()
    return TensorDataset(images, labels)


def deterministic_subset(dataset, size: int, seed: int):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
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
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += y.size(0)
    return total_loss / total, total_correct / total


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def as_muon_matrix(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(tensor.shape)
    if tensor.ndim == 2:
        return tensor, shape
    raise ValueError("Official-style Muon supports only 2D tensors.")


@torch.no_grad()
def vanilla_muon_direction(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_momentum = momentum_buffer.lerp(grad, 1.0 - momentum)
    update_raw = grad.lerp(new_momentum, momentum) if nesterov else new_momentum
    matrix, shape = as_muon_matrix(update_raw)
    ortho = newton_schulz_orthogonalize(matrix, steps=ns_steps)
    rows, cols = matrix.shape
    ortho = ortho * math.sqrt(max(1.0, rows / max(cols, 1)))
    return -ortho.reshape(shape), new_momentum


@torch.no_grad()
def vanilla_adamw_direction(
    grad: torch.Tensor,
    state: dict[str, torch.Tensor | int],
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.Tensor:
    beta1, beta2 = betas
    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)

    state["step"] = int(state["step"]) + 1
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    assert isinstance(exp_avg, torch.Tensor)
    assert isinstance(exp_avg_sq, torch.Tensor)

    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    step = int(state["step"])
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
    update = exp_avg.div(bias_correction1).div(denom)
    return -update


def make_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    common = dict(
        alpha0=args.alpha0,
        rho_star=args.rho_star,
        kp=args.kp,
        ki=args.ki,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        reject_bad_steps=args.reject_bad_steps,
        max_backtracks=args.max_backtracks,
        backtrack_shrink=args.backtrack_shrink,
        use_rho_ema=args.use_rho_ema,
        trust_region_expand=args.trust_region_expand,
        trust_region_rho_threshold=args.trust_region_rho_threshold,
        trust_region_alpha_threshold=args.trust_region_alpha_threshold,
        trust_region_expand_factor=args.trust_region_expand_factor,
        weight_decay=args.weight_decay,
    )
    if name == "vanilla_adam":
        if args.weight_decay != 0.0:
            return torch.optim.AdamW(model.parameters(), lr=args.alpha0, weight_decay=args.weight_decay)
        return torch.optim.Adam(model.parameters(), lr=args.alpha0)
    if name == "pi_adam":
        return PIAdam(
            model.parameters(),
            **common,
            rho_smoothing=args.rho_beta,
            multiplicative_clip=(args.min_alpha_factor, args.max_alpha_factor),
        )
    if name == "pi_muon":
        return PIMuon(
            default_muon_param_groups(model.named_parameters()),
            **common,
            beta_rho=args.rho_beta,
            multiplier_min=args.min_alpha_factor,
            multiplier_max=args.max_alpha_factor,
            ns_steps=args.muon_ns_steps,
        )
    raise ValueError(f"Unknown PI optimizer {name}")


def train_vanilla_muon_step(
    model: nn.Module,
    muon_params: set[nn.Parameter],
    momentum_buffers: dict[nn.Parameter, torch.Tensor],
    adamw_states: dict[nn.Parameter, dict[str, torch.Tensor | int]],
    lr: float,
    args: argparse.Namespace,
) -> None:
    params = [param for param in model.parameters() if param.requires_grad]
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if param in muon_params and param.ndim == 2:
                direction, new_momentum = vanilla_muon_direction(
                    grad,
                    momentum_buffers[param],
                    momentum=args.muon_momentum,
                    nesterov=args.muon_nesterov,
                    ns_steps=args.muon_ns_steps,
                )
                momentum_buffers[param] = new_momentum
            else:
                direction = vanilla_adamw_direction(grad, adamw_states[param])
            if args.weight_decay != 0.0:
                param.mul_(1.0 - lr * args.weight_decay)
            param.add_(direction, alpha=lr)


def train_one(
    seed: int,
    optimizer_name: str,
    base_state: dict[str, torch.Tensor],
    train_data,
    eval_train_loader: DataLoader,
    eval_test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[EpochMetrics]:
    model = SmallMLP().to(device)
    model.load_state_dict(base_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = None if optimizer_name == "vanilla_muon" else make_optimizer(optimizer_name, model, args)
    vanilla_muon_params: set[nn.Parameter] | None = None
    vanilla_muon_momentum_buffers: dict[nn.Parameter, torch.Tensor] | None = None
    vanilla_muon_adamw_states: dict[nn.Parameter, dict[str, torch.Tensor | int]] | None = None
    if optimizer_name == "vanilla_muon":
        groups = default_muon_param_groups(model.named_parameters())
        vanilla_muon_params = {
            param
            for group in groups
            if bool(group.get("use_muon", True))
            for param in group["params"]
        }
        vanilla_muon_momentum_buffers = {
            param: torch.zeros_like(param)
            for param in vanilla_muon_params
        }
        vanilla_muon_adamw_states = {
            param: {}
            for param in model.parameters()
            if param.requires_grad and param not in vanilla_muon_params
        }
    train_loader = make_loader(train_data, args.batch_size, True, seed)

    metrics: list[EpochMetrics] = []
    start = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        set_seed(seed + epoch)
        model.train()
        epoch_alpha: list[float] = []
        epoch_rho: list[float] = []
        epoch_accepted: list[float] = []
        epoch_fallback: list[float] = []
        epoch_backtracks: list[float] = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if optimizer_name == "vanilla_adam":
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_alpha.append(args.alpha0)
                epoch_rho.append(float("nan"))
                epoch_accepted.append(1.0)
                epoch_fallback.append(0.0)
                epoch_backtracks.append(0.0)
            elif optimizer_name == "vanilla_muon":
                assert vanilla_muon_params is not None
                assert vanilla_muon_momentum_buffers is not None
                assert vanilla_muon_adamw_states is not None
                for param in model.parameters():
                    param.grad = None
                loss = criterion(model(x), y)
                loss.backward()
                train_vanilla_muon_step(
                    model,
                    vanilla_muon_params,
                    vanilla_muon_momentum_buffers,
                    vanilla_muon_adamw_states,
                    args.alpha0,
                    args,
                )
                epoch_alpha.append(args.alpha0)
                epoch_rho.append(float("nan"))
                epoch_accepted.append(1.0)
                epoch_fallback.append(0.0)
                epoch_backtracks.append(0.0)
            else:
                assert optimizer is not None

                def closure(backward: bool = True):
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x), y)
                    if backward:
                        loss.backward()
                    return loss

                result = optimizer.step(closure)
                stats = result if optimizer_name == "pi_adam" else optimizer.last_stats
                assert stats is not None
                epoch_alpha.append(float(stats.alpha_next if stats.alpha_next is not None else stats.alpha))
                epoch_rho.append(float("nan") if stats.rho is None else float(stats.rho))
                epoch_accepted.append(float(stats.accepted))
                epoch_fallback.append(float(stats.used_fallback_direction))
                epoch_backtracks.append(float(stats.backtracks))
            optimizer_steps += 1

        train_loss, train_accuracy = evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, eval_test_loader, criterion, device)
        row = EpochMetrics(
            seed=seed,
            optimizer=optimizer_name,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            elapsed_seconds=time.perf_counter() - start,
            optimizer_steps=optimizer_steps,
            mean_alpha=float(np.mean(epoch_alpha)),
            mean_rho=finite_mean(epoch_rho),
            accepted_rate=float(np.mean(epoch_accepted)),
            fallback_rate=float(np.mean(epoch_fallback)),
            mean_backtracks=float(np.mean(epoch_backtracks)),
        )
        metrics.append(row)
        if args.print_every and (epoch == 1 or epoch % args.print_every == 0):
            print(
                f"[seed={seed} {optimizer_name}] epoch={epoch} "
                f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
                f"mean_alpha={row.mean_alpha:.3e} mean_rho={row.mean_rho:.3f} "
                f"accepted={row.accepted_rate:.3f}",
                flush=True,
            )
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    optimizers = sorted({str(row["optimizer"]) for row in seed_rows})
    summary = []
    for optimizer in optimizers:
        rows = [row for row in seed_rows if row["optimizer"] == optimizer]
        final_values = [float(row["final_test_accuracy"]) for row in rows]
        best_values = [float(row["best_test_accuracy"]) for row in rows]
        summary.append(
            {
                "optimizer": optimizer,
                "n_seeds": len(rows),
                "final_test_accuracy_mean": statistics.fmean(final_values),
                "final_test_accuracy_std": statistics.stdev(final_values) if len(final_values) > 1 else 0.0,
                "best_test_accuracy_mean": statistics.fmean(best_values),
                "best_test_accuracy_std": statistics.stdev(best_values) if len(best_values) > 1 else 0.0,
                "final_test_loss_mean": statistics.fmean(float(row["final_test_loss"]) for row in rows),
                "mean_alpha_final_epoch": statistics.fmean(float(row["final_mean_alpha"]) for row in rows),
                "mean_rho_final_epoch": statistics.fmean(float(row["final_mean_rho"]) for row in rows),
                "accepted_rate_final_epoch": statistics.fmean(float(row["final_accepted_rate"]) for row in rows),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(seed) for seed in args.seeds]

    train_full = load_idx_image_dataset(args.fashion_folder, train=True)
    test_full = load_idx_image_dataset(args.fashion_folder, train=False)

    all_epoch_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(seed)
        train_data = deterministic_subset(train_full, args.train_subset, seed)
        test_data = deterministic_subset(test_full, args.test_subset, seed + 1)
        eval_train_loader = make_loader(train_data, args.batch_size, False, seed)
        eval_test_loader = make_loader(test_data, args.batch_size, False, seed)
        base_model = SmallMLP().to(device)
        base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}

        for optimizer_name in args.optimizers:
            rows = train_one(
                seed,
                optimizer_name,
                base_state,
                train_data,
                eval_train_loader,
                eval_test_loader,
                args,
                device,
            )
            all_epoch_rows.extend(row.__dict__ for row in rows)
            final = rows[-1]
            best = max(rows, key=lambda row: row.test_accuracy)
            seed_summary_rows.append(
                {
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "final_test_accuracy": final.test_accuracy,
                    "final_test_loss": final.test_loss,
                    "best_test_accuracy": best.test_accuracy,
                    "best_epoch": best.epoch,
                    "final_train_accuracy": final.train_accuracy,
                    "final_mean_alpha": final.mean_alpha,
                    "final_mean_rho": final.mean_rho,
                    "final_accepted_rate": final.accepted_rate,
                    "final_fallback_rate": final.fallback_rate,
                    "final_mean_backtracks": final.mean_backtracks,
                    "elapsed_seconds": final.elapsed_seconds,
                }
            )

    epoch_fields = list(EpochMetrics.__dataclass_fields__.keys())
    seed_fields = [
        "seed",
        "optimizer",
        "final_test_accuracy",
        "final_test_loss",
        "best_test_accuracy",
        "best_epoch",
        "final_train_accuracy",
        "final_mean_alpha",
        "final_mean_rho",
        "final_accepted_rate",
        "final_fallback_rate",
        "final_mean_backtracks",
        "elapsed_seconds",
    ]
    final_summary = summarize(seed_summary_rows)
    final_fields = list(final_summary[0].keys())

    write_csv(output_dir / "epoch_metrics.csv", all_epoch_rows, epoch_fields)
    write_csv(output_dir / "seed_summary.csv", seed_summary_rows, seed_fields)
    write_csv(output_dir / "final_summary.csv", final_summary, final_fields)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"Outputs written to: {output_dir.resolve()}")
    for row in final_summary:
        print(
            f"{row['optimizer']}: final_acc={row['final_test_accuracy_mean']:.4f} "
            f"+/- {row['final_test_accuracy_std']:.4f}, "
            f"best_acc={row['best_test_accuracy_mean']:.4f} +/- {row['best_test_accuracy_std']:.4f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fashion-folder", type=Path, default=ROOT / "fashion")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pi_fashion_mnist_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["vanilla_adam", "vanilla_muon", "pi_adam", "pi_muon"],
        default=["vanilla_adam", "vanilla_muon", "pi_adam", "pi_muon"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=4096)
    parser.add_argument("--test-subset", type=int, default=1024)
    parser.add_argument("--alpha0", type=float, default=1e-3)
    parser.add_argument("--rho-star", type=float, default=0.7)
    parser.add_argument("--kp", type=float, default=0.05)
    parser.add_argument("--ki", type=float, default=0.001)
    parser.add_argument("--rho-beta", type=float, default=0.9)
    parser.add_argument("--alpha-min", type=float, default=1e-5)
    parser.add_argument("--alpha-max", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-alpha-factor", type=float, default=0.8)
    parser.add_argument("--max-alpha-factor", type=float, default=1.05)
    parser.add_argument("--reject-bad-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rho-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-backtracks", type=int, default=3)
    parser.add_argument("--backtrack-shrink", type=float, default=0.5)
    parser.add_argument("--trust-region-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-region-rho-threshold", type=float, default=0.9)
    parser.add_argument("--trust-region-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--trust-region-expand-factor", type=float, default=1.5)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
