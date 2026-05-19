"""Plotting and CSV helpers for optimizer comparison demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controlled_adam.objectives import Objective
from controlled_adam.optimizers import OptimizationHistory


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return an output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_objective(
    objective_name: str,
    adam: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    """Plot objective values for vanilla Adam and controlled Adam."""
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_objective.png"

    plt.figure(figsize=(7, 4))
    use_log_scale = np.all(adam.fs > 0) and np.all(controlled.fs > 0)
    plot_fn = plt.semilogy if use_log_scale else plt.plot
    plot_fn(adam.fs, color="tab:orange", linewidth=2.0, label="Vanilla Adam")
    plot_fn(controlled.fs, color="tab:red", linewidth=2.0, label="Controlled Adam")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(f"Objective value: {objective_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def plot_alpha(
    objective_name: str,
    adam: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    """Plot global learning-rate multipliers."""
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_alpha.png"

    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(len(adam.alphas)),
        adam.alphas,
        color="tab:orange",
        linewidth=2.0,
        label="Vanilla Adam alpha",
    )
    plt.plot(
        np.arange(len(controlled.alphas)),
        controlled.alphas,
        color="tab:red",
        linewidth=2.0,
        label="Controlled Adam alpha",
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
    """Plot actual-over-predicted decrease ratio for controlled Adam."""
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_rho.png"

    plt.figure(figsize=(7, 4))
    plt.plot(controlled.rhos, color="tab:red", linewidth=1.8, label=r"$\rho_t$")
    plt.axhline(
        rho_star,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=r"target $\rho^\star$",
    )
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
    adam: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    """Plot trajectories in 2D on a filled objective landscape."""
    if adam.xs.shape[1] != 2 or controlled.xs.shape[1] != 2:
        raise ValueError("Trajectory plot only supports 2D states.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective.name}_trajectory.png"

    minima = getattr(objective, "global_minima", np.empty((0, 2)))
    all_xs = np.vstack([adam.xs, controlled.xs, minima])
    xy_min = all_xs.min(axis=0)
    xy_max = all_xs.max(axis=0)
    xy_span = np.maximum(xy_max - xy_min, 1e-8)
    padding = np.maximum(0.3, 0.08 * xy_span)
    xmin, ymin = xy_min - padding
    xmax, ymax = xy_max + padding

    xx = np.linspace(xmin, xmax, 320)
    yy = np.linspace(ymin, ymax, 320)
    X, Y = np.meshgrid(xx, yy)
    Z = np.array(
        [objective.value(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]
    )
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(6, 6))
    contour = plt.contourf(X, Y, Z, levels=45, cmap="viridis", alpha=0.84)
    plt.contour(X, Y, Z, levels=14, colors="white", linewidths=0.45, alpha=0.58)
    plt.colorbar(contour, label="f(x, y)")
    plt.plot(
        adam.xs[:, 0],
        adam.xs[:, 1],
        marker="o",
        markersize=2.7,
        linewidth=1.8,
        color="tab:orange",
        label="Vanilla Adam",
    )
    plt.plot(
        controlled.xs[:, 0],
        controlled.xs[:, 1],
        marker="o",
        markersize=2.7,
        linewidth=1.8,
        color="tab:red",
        label="Controlled Adam",
    )
    if len(minima) > 0:
        plt.scatter(
            minima[:, 0],
            minima[:, 1],
            marker="*",
            s=125,
            color="white",
            edgecolor="black",
            label="Known minimum",
            zorder=5,
        )
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


def save_controlled_diagnostics(
    objective_name: str,
    history: OptimizationHistory,
    output_dir: str | Path,
) -> Path:
    """Save controlled Adam diagnostics as a CSV file."""
    if history.rhos is None or history.accepted is None:
        raise ValueError("Controlled Adam diagnostics are required.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{objective_name}_controlled_adam_diagnostics.csv"

    data = np.column_stack(
        [
            np.arange(len(history.alphas)),
            history.fs[1:],
            history.alphas,
            history.grad_norms,
            history.rhos,
            history.predicted_decreases,
            history.actual_decreases,
            history.accepted.astype(int),
            history.descent_scores,
        ]
    )
    header = (
        "iteration,f,alpha,grad_norm,rho,predicted_decrease,"
        "actual_decrease,accepted,descent_score"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")

    return path
