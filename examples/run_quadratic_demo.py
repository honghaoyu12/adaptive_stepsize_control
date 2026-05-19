"""Run the adaptive step-size control demo on a 2D quadratic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_stepsize_control.objectives import QuadraticObjective
from adaptive_stepsize_control.optimizers import (
    controlled_gradient_descent,
    fixed_gradient_descent,
    stochastic_gradient_descent,
)
from adaptive_stepsize_control.plotting import (
    plot_objective_values,
    plot_rho_ratio,
    plot_step_sizes,
    plot_trajectory,
    save_summary_csv,
)


def main() -> None:
    objective = QuadraticObjective.anisotropic_2d()

    x0 = np.array([2.0, 2.0])
    steps = 80
    rho_star = 0.8

    fixed = fixed_gradient_descent(
        objective=objective,
        x0=x0,
        eta=0.035,
        steps=steps,
    )

    stochastic = stochastic_gradient_descent(
        objective=objective,
        x0=x0,
        eta=0.025,
        steps=steps,
        gradient_noise_scale=0.08,
        seed=0,
    )

    controlled = controlled_gradient_descent(
        objective=objective,
        x0=x0,
        eta0=0.08,
        steps=steps,
        kp=0.7,
        rho_star=rho_star,
        eta_min=1e-8,
        eta_max=0.2,
        reject_bad_steps=True,
    )

    output_dir = Path(__file__).resolve().parents[1] / "outputs"

    paths = [
        plot_objective_values(fixed, controlled, output_dir, stochastic=stochastic),
        plot_step_sizes(fixed, controlled, output_dir, stochastic=stochastic),
        plot_rho_ratio(controlled, rho_star, output_dir),
        plot_trajectory(objective, fixed, controlled, output_dir, stochastic=stochastic),
        save_summary_csv(controlled, output_dir),
    ]

    print("Demo complete. Files written:")
    for path in paths:
        print(f"- {path}")

    print()
    print("Fixed-step GD")
    print(f"  final x: {fixed.xs[-1]}")
    print(f"  final f: {fixed.fs[-1]:.6e}")

    print()
    print("SGD")
    print(f"  final x: {stochastic.xs[-1]}")
    print(f"  final f: {stochastic.fs[-1]:.6e}")

    print()
    print("Controlled GD")
    print(f"  final x: {controlled.xs[-1]}")
    print(f"  final f: {controlled.fs[-1]:.6e}")
    print(f"  final eta: {controlled.etas[-1]:.6e}")
    print(f"  final rho: {controlled.rhos[-1]:.6e}")


if __name__ == "__main__":
    main()
