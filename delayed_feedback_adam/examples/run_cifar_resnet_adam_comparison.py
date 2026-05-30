"""Compare same-step controlled Adam and delayed-feedback Adam on CIFAR ResNet.

This staged runner reuses the CIFAR-10 data/model utilities from
``controlled_adam_project/examples/run_mnist_demo.py`` and adds delayed Adam
variants from this subproject.  It is intended for CPU-friendly ResNet subset
benchmarks before any larger CIFAR-10 claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_ADAM_SRC = ROOT / "controlled_adam_project" / "src"
CONTROLLED_ADAM_EXAMPLES = ROOT / "controlled_adam_project" / "examples"
for path in (CONTROLLED_ADAM_SRC, CONTROLLED_ADAM_EXAMPLES, ROOT / "delayed_feedback_adam"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import run_mnist_demo as image_base
from delayed_feedback_adam import DelayedFeedbackAdam


VARIANTS = (
    "vanilla_adam",
    "controlled_raw_rho",
    "controlled_ema",
    "controlled_ema_trust",
    "delayed_current",
    "delayed_raw",
    "delayed_ema",
    "delayed_safe",
    "delayed_ema_rho60",
    "delayed_ema_rho70",
    "delayed_ema_fast",
    "delayed_ema_floor90",
    "delayed_raw_rho60",
)


@dataclass(frozen=True)
class DelayedVariantConfig:
    """Neural-scale delayed Adam settings."""

    name: str
    description: str
    alpha_bounds: tuple[float, float]
    rho_star: float
    kp: float
    rho_beta: float
    multiplier_bounds: tuple[float, float]
    rho_clip: tuple[float, float] = (-1.0, 2.0)
    alpha_init: float = 1.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class DelayedStepLog:
    """Step diagnostics compatible with the existing plotting helper."""

    loss: float
    alpha: float
    alpha_multiplier: float
    rho: float
    rho_ema: float
    predicted_decrease: float
    actual_decrease: float
    multiplier: float
    controller_applied: bool
    trust_region_expanded: bool = False


def delayed_variant_configs() -> dict[str, DelayedVariantConfig]:
    """Return delayed Adam candidates for CIFAR ResNet."""

    return {
        "delayed_current": DelayedVariantConfig(
            name="delayed_current",
            description=(
                "README-style delayed controller with a neural-safe cap: "
                "rho*=0.8, kp=0.05, rho_beta=0.95, alpha in [0.5, 2.0]."
            ),
            alpha_bounds=(0.5, 2.0),
            rho_star=0.8,
            kp=0.05,
            rho_beta=0.95,
            multiplier_bounds=(0.8, 1.25),
        ),
        "delayed_raw": DelayedVariantConfig(
            name="delayed_raw",
            description=(
                "Raw delayed rho with the same gentle factor clip used by the "
                "balanced controlled-Adam ResNet runs."
            ),
            alpha_bounds=(0.75, 1.5),
            rho_star=0.8,
            kp=0.02,
            rho_beta=0.0,
            multiplier_bounds=(0.98, 1.015),
        ),
        "delayed_ema": DelayedVariantConfig(
            name="delayed_ema",
            description=(
                "EMA-smoothed delayed rho for noisy minibatches; same alpha range "
                "as the balanced controlled-Adam setting."
            ),
            alpha_bounds=(0.75, 1.5),
            rho_star=0.8,
            kp=0.02,
            rho_beta=0.95,
            multiplier_bounds=(0.98, 1.015),
        ),
        "delayed_safe": DelayedVariantConfig(
            name="delayed_safe",
            description=(
                "Conservative delayed EMA controller with tighter alpha cap and "
                "slower per-step changes."
            ),
            alpha_bounds=(0.8, 1.25),
            rho_star=0.85,
            kp=0.01,
            rho_beta=0.95,
            multiplier_bounds=(0.995, 1.005),
        ),
        "delayed_ema_rho60": DelayedVariantConfig(
            name="delayed_ema_rho60",
            description=(
                "Delayed EMA with lower rho target. This tests whether the "
                "one-step delayed minibatch rho signal is naturally lower than "
                "same-step rho and should not drive alpha to the floor."
            ),
            alpha_bounds=(0.75, 1.5),
            rho_star=0.60,
            kp=0.02,
            rho_beta=0.95,
            multiplier_bounds=(0.98, 1.015),
        ),
        "delayed_ema_rho70": DelayedVariantConfig(
            name="delayed_ema_rho70",
            description="Delayed EMA with intermediate rho target 0.70.",
            alpha_bounds=(0.75, 1.5),
            rho_star=0.70,
            kp=0.02,
            rho_beta=0.95,
            multiplier_bounds=(0.98, 1.015),
        ),
        "delayed_ema_fast": DelayedVariantConfig(
            name="delayed_ema_fast",
            description=(
                "More responsive delayed EMA: lower rho target, lower smoothing, "
                "and slightly wider per-step multiplier clip."
            ),
            alpha_bounds=(0.75, 1.5),
            rho_star=0.60,
            kp=0.03,
            rho_beta=0.90,
            multiplier_bounds=(0.97, 1.02),
        ),
        "delayed_ema_floor90": DelayedVariantConfig(
            name="delayed_ema_floor90",
            description=(
                "Delayed EMA with a higher alpha floor so noisy delayed feedback "
                "cannot shrink below 0.9x the base learning rate."
            ),
            alpha_bounds=(0.90, 1.5),
            rho_star=0.70,
            kp=0.02,
            rho_beta=0.95,
            multiplier_bounds=(0.98, 1.015),
        ),
        "delayed_raw_rho60": DelayedVariantConfig(
            name="delayed_raw_rho60",
            description=(
                "Raw delayed rho with lower target. This tests whether smoothing "
                "is helping or merely delaying recovery."
            ),
            alpha_bounds=(0.75, 1.5),
            rho_star=0.60,
            kp=0.02,
            rho_beta=0.0,
            multiplier_bounds=(0.98, 1.015),
        ),
    }


def selected_variants(args: argparse.Namespace) -> list[str]:
    """Return optimizer variants for this run."""

    if args.variants:
        return list(dict.fromkeys(args.variants))
    if args.smoke:
        return ["vanilla_adam", "controlled_raw_rho", "delayed_raw", "delayed_ema"]
    if args.delayed_screen:
        return [
            "vanilla_adam",
            "delayed_raw",
            "delayed_ema",
            "delayed_safe",
            "delayed_ema_rho60",
            "delayed_ema_rho70",
            "delayed_ema_fast",
            "delayed_ema_floor90",
            "delayed_raw_rho60",
        ]
    return [
        "vanilla_adam",
        "controlled_raw_rho",
        "controlled_ema_trust",
        "delayed_raw",
        "delayed_ema",
        "delayed_safe",
    ]


def make_checkpoint(args: argparse.Namespace, dataset_name: str) -> image_base.CheckpointConfig:
    """Return checkpoint/progress configuration."""

    return image_base.CheckpointConfig(
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        save_every=args.checkpoint_every,
        print_every=args.print_every,
    )


def train_delayed_adam(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    config: DelayedVariantConfig,
    checkpoint: image_base.CheckpointConfig | None,
) -> tuple[list[image_base.EpochMetrics], list[DelayedStepLog]]:
    """Train with DelayedFeedbackAdam and collect epoch/step diagnostics."""

    optimizer = DelayedFeedbackAdam(
        model.parameters(),
        lr=args.lr,
        alpha_init=config.alpha_init,
        alpha_bounds=config.alpha_bounds,
        rho_star=config.rho_star,
        kp=config.kp,
        ki=config.ki,
        kd=config.kd,
        rho_beta=config.rho_beta,
        rho_clip=config.rho_clip,
        multiplier_bounds=config.multiplier_bounds,
        decoupled_weight_decay=args.delayed_decoupled_weight_decay,
        weight_decay=args.weight_decay,
    )

    metrics: list[image_base.EpochMetrics] = []
    step_logs: list[DelayedStepLog] = []
    run_start_time = time.perf_counter()
    optimizer_steps = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        image_base.set_seed(args.seed + epoch)
        model.train()
        epoch_logs: list[DelayedStepLog] = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step(loss=loss.item())
            diagnostics = optimizer.get_diagnostics()
            alpha_multiplier = float(diagnostics["alpha"])
            step_log = DelayedStepLog(
                loss=float(loss.item()),
                alpha=args.lr * alpha_multiplier,
                alpha_multiplier=alpha_multiplier,
                rho=nan_float(diagnostics["last_rho_raw"]),
                rho_ema=nan_float(diagnostics["rho_bar"]),
                predicted_decrease=nan_float(diagnostics["last_predicted_decrease"]),
                actual_decrease=nan_float(diagnostics["last_actual_decrease"]),
                multiplier=nan_float(diagnostics["last_multiplier"]),
                controller_applied=bool(diagnostics["last_controller_applied"]),
            )
            step_logs.append(step_log)
            epoch_logs.append(step_log)
            optimizer_steps += 1

        train_loss, train_accuracy = image_base.evaluate(
            model,
            eval_train_loader,
            criterion,
            device,
        )
        test_loss, test_accuracy = image_base.evaluate(model, test_loader, criterion, device)
        finite_rhos = [log.rho_ema for log in epoch_logs if np.isfinite(log.rho_ema)]
        elapsed = time.perf_counter() - start_time
        metrics.append(
            image_base.EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                elapsed_seconds=time.perf_counter() - run_start_time,
                optimizer_steps=optimizer_steps,
                mean_alpha=float(np.mean([log.alpha for log in epoch_logs])),
                mean_rho=float(np.mean(finite_rhos)) if finite_rhos else float("nan"),
                accepted_rate=1.0,
            )
        )
        image_base.maybe_print_epoch(config.name, metrics[-1], elapsed, checkpoint)
        image_base.maybe_save_checkpoint(
            config.name,
            model,
            epoch,
            metrics,
            checkpoint,
            optimizer_state=optimizer.state_dict(),
        )

    return metrics, step_logs


def nan_float(value: Any) -> float:
    """Convert optional diagnostic scalar to float or NaN."""

    if value is None:
        return float("nan")
    return float(value)


def save_delayed_step_logs(path: Path, logs: list[DelayedStepLog], optimizer_name: str) -> None:
    """Save delayed Adam step diagnostics."""

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "optimizer",
                "step",
                "loss",
                "effective_lr",
                "alpha_multiplier",
                "rho_raw",
                "rho_bar",
                "predicted_decrease",
                "actual_decrease",
                "multiplier",
                "controller_applied",
            ]
        )
        for step, log in enumerate(logs):
            writer.writerow(
                [
                    optimizer_name,
                    step,
                    log.loss,
                    log.alpha,
                    log.alpha_multiplier,
                    log.rho,
                    log.rho_ema,
                    log.predicted_decrease,
                    log.actual_decrease,
                    log.multiplier,
                    int(log.controller_applied),
                ]
            )


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_bundle: image_base.DatasetBundle,
    model: nn.Module,
    variants: list[str],
) -> None:
    """Write run metadata."""

    delayed_configs = delayed_variant_configs()
    metadata = {
        "script": Path(__file__).name,
        "torch_version": torch.__version__,
        "device": args.device,
        "seed": args.seed,
        "dataset": {
            "name": dataset_bundle.name,
            "data_dir": str(args.data_dir),
            "train_size_full": dataset_bundle.train_size_full,
            "test_size_full": dataset_bundle.test_size_full,
            "train_size_used": dataset_bundle.train_size_used,
            "test_size_used": dataset_bundle.test_size_used,
            "train_transform": dataset_bundle.train_transform,
            "eval_transform": dataset_bundle.eval_transform,
        },
        "model": {
            "class": model.__class__.__name__,
            "trainable_parameters": image_base.count_parameters(model),
            "architecture": str(model),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "variants": variants,
        },
        "controlled_adam": {
            "alpha_min": args.controlled_alpha_min,
            "alpha_max": args.controlled_alpha_max,
            "rho_star": args.controlled_rho_star,
            "rho_beta": args.controlled_rho_beta,
            "kp": args.controlled_kp,
            "min_alpha_factor": args.controlled_min_alpha_factor,
            "max_alpha_factor": args.controlled_max_alpha_factor,
            "trust_region_rho_threshold": args.controlled_trust_rho_threshold,
            "trust_region_alpha_threshold": args.controlled_trust_alpha_threshold,
            "trust_region_expand_factor": args.controlled_trust_expand_factor,
            "trust_region_max_factor": args.controlled_trust_max_factor,
            "trust_region_patience": args.controlled_trust_patience,
            "max_backtracks": args.controlled_max_backtracks,
            "backtrack_shrink": args.controlled_backtrack_shrink,
            "rho_min": args.controlled_rho_min,
            "absolute_predicted_floor": args.controlled_absolute_predicted_floor,
            "relative_predicted_floor": args.controlled_relative_predicted_floor,
            "rho_clip_min": args.controlled_rho_clip_min,
            "rho_clip_max": args.controlled_rho_clip_max,
        },
        "delayed_adam": {
            name: delayed_configs[name].__dict__
            for name in variants
            if name in delayed_configs
        },
    }
    with (output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / "run_metadata.txt").open("w") as handle:
        handle.write(json.dumps(metadata, indent=2))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "controlled_adam_project" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cifar10_resnet_adam_delayed_smoke")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--delayed-screen",
        action="store_true",
        help="Run vanilla plus focused delayed Adam parameter candidates.",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument("--controlled-kp", type=float, default=0.02)
    parser.add_argument("--controlled-rho-star", type=float, default=0.80)
    parser.add_argument("--controlled-rho-beta", type=float, default=0.90)
    parser.add_argument("--controlled-alpha-min", type=float, default=1e-3)
    parser.add_argument("--controlled-alpha-max", type=float, default=1.5e-3)
    parser.add_argument("--controlled-min-alpha-factor", type=float, default=0.98)
    parser.add_argument("--controlled-max-alpha-factor", type=float, default=1.015)
    parser.add_argument("--controlled-trust-rho-threshold", type=float, default=0.90)
    parser.add_argument("--controlled-trust-alpha-threshold", type=float, default=1.05e-3)
    parser.add_argument("--controlled-trust-expand-factor", type=float, default=1.10)
    parser.add_argument("--controlled-trust-max-factor", type=float, default=1.10)
    parser.add_argument("--controlled-trust-patience", type=int, default=2)
    parser.add_argument("--controlled-max-backtracks", type=int, default=1)
    parser.add_argument("--controlled-backtrack-shrink", type=float, default=0.5)
    parser.add_argument("--controlled-rho-min", type=float, default=0.0)
    parser.add_argument("--controlled-absolute-predicted-floor", type=float, default=1e-12)
    parser.add_argument("--controlled-relative-predicted-floor", type=float, default=1e-8)
    parser.add_argument("--controlled-rho-clip-min", type=float, default=-1.0)
    parser.add_argument("--controlled-rho-clip-max", type=float, default=3.0)
    parser.add_argument(
        "--delayed-decoupled-weight-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison."""

    args = parse_args()
    image_base.set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = selected_variants(args)

    datasets = image_base.load_image_dataset_bundle(
        args.data_dir,
        "cifar10",
        args.train_subset,
        args.test_subset,
        args.seed,
        args.download,
        None,
    )
    train_data = datasets.train_data
    eval_train_loader = image_base.make_loader(datasets.eval_train_data, args.batch_size, False, args.seed)
    eval_test_loader = image_base.make_loader(datasets.test_data, args.batch_size, False, args.seed)
    checkpoint = make_checkpoint(args, datasets.name)

    base_model = image_base.make_model("resnet_cifar", datasets.name).to(device)
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base_model.state_dict().items()
    }
    criterion = nn.CrossEntropyLoss()
    runs: list[image_base.OptimizerRun] = []
    delayed_configs = delayed_variant_configs()
    write_metadata(args.output_dir, args, datasets, base_model, variants)

    for variant in variants:
        model = image_base.make_model("resnet_cifar", datasets.name).to(device)
        model.load_state_dict(base_state)
        train_loader = image_base.make_loader(train_data, args.batch_size, True, args.seed)

        if variant == "vanilla_adam":
            metrics = image_base.train_vanilla_adam(
                model,
                train_loader,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args.epochs,
                args.lr,
                args.seed,
                run_name=variant,
                checkpoint=checkpoint,
            )
            runs.append(image_base.OptimizerRun(variant, metrics))
        elif variant.startswith("controlled_"):
            use_rho_ema = variant != "controlled_raw_rho"
            trust_region_expand = variant == "controlled_ema_trust"
            metrics, step_logs = image_base.train_controlled_adam(
                model,
                train_loader,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args.epochs,
                args.lr,
                args.controlled_kp,
                args.controlled_rho_star,
                args.controlled_rho_beta if use_rho_ema else 0.0,
                args.controlled_alpha_min,
                args.controlled_alpha_max,
                args.controlled_min_alpha_factor,
                args.controlled_max_alpha_factor,
                trust_region_expand,
                args.controlled_trust_rho_threshold,
                args.controlled_trust_alpha_threshold,
                args.controlled_trust_expand_factor,
                args.seed,
                use_rho_ema=use_rho_ema,
                reject_bad_steps=True,
                run_name=variant,
                checkpoint=checkpoint,
                max_backtracks=args.controlled_max_backtracks,
                backtrack_shrink=args.controlled_backtrack_shrink,
                rho_min=args.controlled_rho_min,
                absolute_predicted_floor=args.controlled_absolute_predicted_floor,
                relative_predicted_floor=args.controlled_relative_predicted_floor,
                rho_clip_min=args.controlled_rho_clip_min,
                rho_clip_max=args.controlled_rho_clip_max,
                trust_region_max_factor=args.controlled_trust_max_factor,
                trust_region_patience=args.controlled_trust_patience,
            )
            runs.append(image_base.OptimizerRun(variant, metrics, step_logs))
            image_base.save_step_logs(
                args.output_dir / f"cifar10_{image_base.slugify(variant)}_step_diagnostics.csv",
                step_logs,
                optimizer_name=variant,
            )
        else:
            config = delayed_configs[variant]
            metrics, step_logs = train_delayed_adam(
                model,
                train_loader,
                eval_train_loader,
                eval_test_loader,
                criterion,
                device,
                args,
                config,
                checkpoint,
            )
            runs.append(image_base.OptimizerRun(variant, metrics, step_logs))  # type: ignore[arg-type]
            save_delayed_step_logs(
                args.output_dir / f"cifar10_{image_base.slugify(variant)}_step_diagnostics.csv",
                step_logs,
                variant,
            )

    image_base.save_epoch_metrics(args.output_dir / "cifar10_epoch_metrics.csv", runs)
    image_base.plot_metrics(args.output_dir, "cifar10", runs)

    print(f"Outputs written to: {args.output_dir.resolve()}")
    for run in runs:
        final = run.metrics[-1]
        print(f"{run.name} final test accuracy: {final.test_accuracy:.4f}")
        if final.mean_alpha is not None:
            print(f"{run.name} final mean effective lr: {final.mean_alpha:.3e}")
        if final.mean_rho is not None:
            print(f"{run.name} final mean rho signal: {final.mean_rho:.3f}")


if __name__ == "__main__":
    main()
