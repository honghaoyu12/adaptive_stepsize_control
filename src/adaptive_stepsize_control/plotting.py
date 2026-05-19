"""Plotting utilities for the adaptive step-size demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_stepsize_control.objectives import ObjectiveFunction
from adaptive_stepsize_control.optimizers import OptimizationHistory


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_objective_values(
    fixed: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
    filename: str = "objective_value.png",
    stochastic: OptimizationHistory | None = None,
) -> Path:
    """Plot objective values for the optimizer runs."""
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(7, 4))
    plt.semilogy(fixed.fs, label="Fixed-step GD")
    if stochastic is not None:
        plt.semilogy(stochastic.fs, label="SGD")
    plt.semilogy(controlled.fs, label="Controlled GD")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Objective value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def plot_step_sizes(
    fixed: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
    filename: str = "adaptive_step_size.png",
    stochastic: OptimizationHistory | None = None,
) -> Path:
    """Plot learning rates for the optimizer runs."""
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(fixed.etas)), fixed.etas, label="Fixed-step GD")
    if stochastic is not None:
        plt.plot(np.arange(len(stochastic.etas)), stochastic.etas, label="SGD")
    plt.plot(np.arange(len(controlled.etas)), controlled.etas, label="Controlled GD")
    plt.xlabel("Iteration")
    plt.ylabel("eta")
    plt.title("Step size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def plot_adaptive_step_size(
    controlled: OptimizationHistory,
    output_dir: str | Path,
    filename: str = "adaptive_step_size.png",
) -> Path:
    """Plot the adaptive learning rate."""
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(7, 4))
    plt.plot(controlled.etas, label="Controlled GD")
    plt.xlabel("Iteration")
    plt.ylabel("eta")
    plt.title("Adaptive step size")
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
    filename: str = "rho_ratio.png",
) -> Path:
    """Plot actual-over-predicted decrease ratio."""
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required for this plot.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(7, 4))
    plt.plot(controlled.rhos, label=r"$\rho_t$")
    plt.axhline(rho_star, linestyle="--", label=r"target $\rho^\star$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\rho_t$")
    plt.title("Actual decrease / predicted decrease")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def plot_trajectory(
    objective: ObjectiveFunction,
    fixed: OptimizationHistory,
    controlled: OptimizationHistory,
    output_dir: str | Path,
    filename: str = "trajectory.png",
    stochastic: OptimizationHistory | None = None,
) -> Path:
    """Plot the 2D optimization trajectories on objective contours."""
    if fixed.xs.shape[1] != 2 or controlled.xs.shape[1] != 2:
        raise ValueError("Trajectory plot only supports 2D states.")
    if stochastic is not None and stochastic.xs.shape[1] != 2:
        raise ValueError("Trajectory plot only supports 2D states.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    minima = getattr(objective, "global_minima", np.empty((0, 2)))
    histories = [fixed.xs, controlled.xs, minima]
    if stochastic is not None:
        histories.append(stochastic.xs)
    all_xs = np.vstack(histories)
    x_min, y_min = all_xs.min(axis=0)
    x_max, y_max = all_xs.max(axis=0)
    x_pad = max(0.2, 0.08 * (x_max - x_min))
    y_pad = max(0.2, 0.08 * (y_max - y_min))

    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 300)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 300)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.array([objective.value(point) for point in points]).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    contour = plt.contourf(xx, yy, zz, levels=40, cmap="viridis", alpha=0.82)
    plt.contour(xx, yy, zz, levels=12, colors="white", linewidths=0.45, alpha=0.55)
    plt.colorbar(contour, label="f(x, y)")
    plt.plot(
        fixed.xs[:, 0],
        fixed.xs[:, 1],
        marker="o",
        markersize=3,
        linewidth=1.8,
        color="tab:orange",
        label="Fixed-step GD",
    )
    if stochastic is not None:
        plt.plot(
            stochastic.xs[:, 0],
            stochastic.xs[:, 1],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            color="tab:blue",
            label="SGD",
        )
    plt.plot(
        controlled.xs[:, 0],
        controlled.xs[:, 1],
        marker="o",
        markersize=3,
        linewidth=1.8,
        color="tab:red",
        label="Controlled GD",
    )
    if len(minima) > 0:
        plt.scatter(
            minima[:, 0],
            minima[:, 1],
            marker="*",
            s=120,
            color="white",
            edgecolor="black",
            label="Known minimum",
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Optimization trajectory on objective landscape")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(y_grid[0], y_grid[-1])
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def save_summary_csv(
    controlled: OptimizationHistory,
    output_dir: str | Path,
    filename: str = "controlled_diagnostics.csv",
) -> Path:
    """Save diagnostics from controlled gradient descent as CSV."""
    if controlled.rhos is None:
        raise ValueError("controlled.rhos is required for the summary CSV.")

    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename

    accepted = (
        controlled.accepted.astype(int)
        if controlled.accepted is not None
        else np.ones_like(controlled.fs)
    )

    iterations = np.arange(len(controlled.etas))
    post_step_fs = controlled.fs[1:]

    data = np.column_stack(
        [
            iterations,
            post_step_fs,
            controlled.etas,
            controlled.rhos,
            controlled.predicted_decreases,
            controlled.actual_decreases,
            controlled.prediction_errors,
            accepted,
        ]
    )

    header = (
        "iteration,f,eta,rho,predicted_decrease,actual_decrease,"
        "prediction_error,accepted"
    )

    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path
