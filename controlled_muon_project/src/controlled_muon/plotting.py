"""Plotting and CSV helpers for controlled Muon demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controlled_muon.objectives import Objective
from controlled_muon.optimizers import OptimizationHistory


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_objective(
    objective_name: str,
    vanilla: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_objective.png"

    plt.figure(figsize=(7, 4))
    use_log_scale = np.all(vanilla.fs > 0) and np.all(controlled.fs > 0)
    plot_fn = plt.semilogy if use_log_scale else plt.plot
    plot_fn(vanilla.fs, color="tab:orange", linewidth=2.0, label="Vanilla Muon")
    plot_fn(controlled.fs, color="tab:red", linewidth=2.0, label="Controlled Muon")
    plt.xlabel("Iteration")
    plt.ylabel("f(W)")
    plt.title(f"Objective value: {objective_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_alpha(
    objective_name: str,
    vanilla: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_alpha.png"

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(vanilla.step_sizes)), vanilla.step_sizes, color="tab:orange", label="Vanilla Muon alpha")
    plt.plot(
        np.arange(len(controlled.step_sizes)),
        controlled.step_sizes,
        color="tab:red",
        label="Controlled Muon alpha",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Global step size")
    plt.title(f"Global step size: {objective_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_rho(
    objective_name: str,
    controlled: OptimizationHistory,
    rho_star: float,
    output_dir: str | Path,
) -> Path:
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_rho.png"
    plt.figure(figsize=(7, 4))
    finite_rhos = np.where(np.isfinite(controlled.rhos), controlled.rhos, np.nan)
    plt.plot(finite_rhos, color="tab:red", linewidth=1.8, label=r"$\rho_t$")
    plt.axhline(rho_star, color="black", linewidth=1.2, linestyle="--", label=r"target $\rho^\star$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\rho_t$")
    plt.title(f"Actual / predicted decrease: {objective_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_trajectory(
    objective: Objective,
    vanilla: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    if vanilla.Ws.shape[1] != 2 or controlled.Ws.shape[1] != 2:
        raise ValueError("Trajectory plot only supports 2D states.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective.name}_trajectory.png"

    minima = getattr(objective, "global_minima", np.empty((0, 2)))
    all_xs = np.vstack([vanilla.Ws, controlled.Ws, minima])
    xy_min = all_xs.min(axis=0)
    xy_max = all_xs.max(axis=0)
    xy_span = np.maximum(xy_max - xy_min, 1e-8)
    padding = np.maximum(0.3, 0.08 * xy_span)
    xmin, ymin = xy_min - padding
    xmax, ymax = xy_max + padding

    xx = np.linspace(xmin, xmax, 320)
    yy = np.linspace(ymin, ymax, 320)
    X, Y = np.meshgrid(xx, yy)
    Z = np.array([objective.value(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(6, 6))
    contour = plt.contourf(X, Y, Z, levels=45, cmap="viridis", alpha=0.84)
    plt.contour(X, Y, Z, levels=14, colors="white", linewidths=0.45, alpha=0.58)
    plt.colorbar(contour, label="f(x, y)")
    plt.plot(vanilla.Ws[:, 0], vanilla.Ws[:, 1], marker="o", markersize=2.7, linewidth=1.8, color="tab:orange", label="Vanilla Muon")
    plt.plot(controlled.Ws[:, 0], controlled.Ws[:, 1], marker="o", markersize=2.7, linewidth=1.8, color="tab:red", label="Controlled Muon")
    if len(minima) > 0:
        plt.scatter(minima[:, 0], minima[:, 1], marker="*", s=125, color="white", edgecolor="black", label="Known minimum", zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectory on objective landscape: {objective.name}")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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
    objective_name: str,
    history: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    if history.rhos is None or history.accepted is None:
        raise ValueError("Controlled Muon diagnostics are required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_controlled_muon_diagnostics.csv"
    data = np.column_stack(
        [
            np.arange(len(history.step_sizes)),
            history.fs,
            history.distances,
            history.step_sizes,
            history.rhos,
            history.predicted_decreases,
            history.actual_decreases,
            history.directional_derivatives,
            history.accepted.astype(int),
        ]
    )
    header = (
        "iteration,f,distance_to_target,alpha,rho,predicted_decrease,"
        "actual_decrease,directional_derivative,accepted"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path
