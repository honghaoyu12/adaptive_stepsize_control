"""Plotting and CSV utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controlled_muon.optimizers import OptimizationHistory


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_objective_values(
    vanilla: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "objective_value.png"

    plt.figure(figsize=(7, 4))
    plt.semilogy(vanilla.fs, label="Vanilla Muon")
    plt.semilogy(controlled.fs, label="Controlled Muon")
    plt.xlabel("Iteration")
    plt.ylabel("f(W)")
    plt.title("Objective value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_distance_to_target(
    vanilla: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "distance_to_target.png"

    plt.figure(figsize=(7, 4))
    plt.semilogy(vanilla.distances, label="Vanilla Muon")
    plt.semilogy(controlled.distances, label="Controlled Muon")
    plt.xlabel("Iteration")
    plt.ylabel(r"$||W - T||_F$")
    plt.title("Distance to target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_controlled_step_size(
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "controlled_alpha.png"

    plt.figure(figsize=(7, 4))
    plt.plot(controlled.step_sizes, label=r"$\alpha_t$")
    plt.xlabel("Iteration")
    plt.ylabel("Global multiplier")
    plt.title("Controlled Muon global step size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_rho_ratio(
    controlled: OptimizationHistory,
    rho_star: float,
    output_dir: str | Path,
) -> Path:
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "rho_ratio.png"

    finite_rhos = np.where(np.isfinite(controlled.rhos), controlled.rhos, np.nan)

    plt.figure(figsize=(7, 4))
    plt.plot(finite_rhos, label=r"$\rho_t$")
    plt.axhline(rho_star, linestyle="--", label=r"target $\rho^\star$")
    plt.xlabel("Iteration")
    plt.ylabel("Actual / predicted decrease")
    plt.title("Controller feedback ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_accepted_steps(
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    if controlled.accepted is None:
        raise ValueError("controlled.accepted is required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "accepted_steps.png"

    plt.figure(figsize=(7, 3))
    plt.plot(controlled.accepted.astype(int), marker="o", markersize=3, label="Accepted")
    plt.xlabel("Iteration")
    plt.ylabel("Accepted step")
    plt.yticks([0, 1])
    plt.title("Controlled Muon accept/reject decisions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_controlled_diagnostics(
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / "controlled_diagnostics.csv"

    accepted = controlled.accepted.astype(int) if controlled.accepted is not None else np.ones_like(controlled.fs)

    data = np.column_stack(
        [
            np.arange(len(controlled.fs)),
            controlled.fs,
            controlled.distances,
            controlled.step_sizes,
            controlled.rhos,
            controlled.predicted_decreases,
            controlled.actual_decreases,
            controlled.directional_derivatives,
            accepted,
        ]
    )

    header = (
        "iteration,f,distance_to_target,alpha,rho,predicted_decrease,"
        "actual_decrease,directional_derivative,accepted"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path
