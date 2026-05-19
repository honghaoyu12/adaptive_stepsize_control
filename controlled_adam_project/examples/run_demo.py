"""Run vanilla Adam and outer-loop controlled Adam on toy objectives."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from controlled_adam.objectives import (
    Ackley,
    AnisotropicQuadratic,
    Beale,
    Easom,
    GoldsteinPrice,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
    SixHumpCamel,
)
from controlled_adam.optimizers import controlled_adam, vanilla_adam
from controlled_adam.plotting import (
    plot_alpha,
    plot_objective,
    plot_rho,
    plot_trajectory,
    save_controlled_diagnostics,
)


def run_one_objective(
    objective,
    x0: np.ndarray,
    steps: int,
    vanilla_alpha: float,
    controlled_alpha0: float,
    kp: float,
    rho_star: float,
    alpha_max: float,
    output_dir: Path,
) -> None:
    """Run both optimizers on one objective and save plots."""
    adam = vanilla_adam(
        objective=objective,
        x0=x0,
        alpha=vanilla_alpha,
        steps=steps,
    )

    controlled = controlled_adam(
        objective=objective,
        x0=x0,
        alpha0=controlled_alpha0,
        steps=steps,
        kp=kp,
        rho_star=rho_star,
        rho_min=0.0,
        alpha_min=1e-8,
        alpha_max=alpha_max,
        reject_bad_steps=True,
    )

    paths = [
        plot_objective(objective.name, adam, controlled, output_dir),
        plot_alpha(objective.name, adam, controlled, output_dir),
        plot_rho(objective.name, controlled, rho_star, output_dir),
        plot_trajectory(objective, adam, controlled, output_dir),
        save_controlled_diagnostics(objective.name, controlled, output_dir),
    ]

    print(f"\n{objective.name}")
    print("-" * len(objective.name))
    print(f"Vanilla Adam final x:      {adam.xs[-1]}")
    print(f"Vanilla Adam final f:      {adam.fs[-1]:.6e}")
    print(f"Controlled Adam final x:   {controlled.xs[-1]}")
    print(f"Controlled Adam final f:   {controlled.fs[-1]:.6e}")
    print(f"Controlled Adam final alpha: {controlled.alphas[-1]:.6e}")
    print(f"Controlled Adam accepted steps: {int(controlled.accepted.sum())}/{len(controlled.accepted)}")
    print("Files written:")
    for path in paths:
        print(f"  - {path}")


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"

    cases = [
        {
            "objective": AnisotropicQuadratic(),
            "x0": np.array([2.0, 2.0]),
            "steps": 250,
            "vanilla_alpha": 0.003,
            "controlled_alpha0": 0.003,
            "kp": 0.1,
            "rho_star": 0.8,
            "alpha_max": 0.5,
        },
        {
            "objective": Rosenbrock(),
            "x0": np.array([-1.5, 1.5]),
            "steps": 3000,
            "vanilla_alpha": 0.003,
            "controlled_alpha0": 0.003,
            "kp": 0.05,
            "rho_star": 0.5,
            "alpha_max": 0.05,
        },
        {
            "objective": Himmelblau(),
            "x0": np.array([-3.5, 0.5]),
            "steps": 700,
            "vanilla_alpha": 0.01,
            "controlled_alpha0": 0.01,
            "kp": 0.08,
            "rho_star": 0.7,
            "alpha_max": 0.08,
        },
        {
            "objective": Rastrigin(),
            "x0": np.array([3.3, 2.8]),
            "steps": 900,
            "vanilla_alpha": 0.004,
            "controlled_alpha0": 0.004,
            "kp": 0.04,
            "rho_star": 0.5,
            "alpha_max": 0.04,
        },
        {
            "objective": Beale(),
            "x0": np.array([1.0, 1.0]),
            "steps": 1500,
            "vanilla_alpha": 0.003,
            "controlled_alpha0": 0.003,
            "kp": 0.06,
            "rho_star": 0.6,
            "alpha_max": 0.08,
        },
        {
            "objective": Ackley(),
            "x0": np.array([2.5, 2.0]),
            "steps": 1200,
            "vanilla_alpha": 0.01,
            "controlled_alpha0": 0.01,
            "kp": 0.05,
            "rho_star": 0.5,
            "alpha_max": 0.08,
        },
        {
            "objective": SixHumpCamel(),
            "x0": np.array([1.2, -1.0]),
            "steps": 800,
            "vanilla_alpha": 0.01,
            "controlled_alpha0": 0.01,
            "kp": 0.06,
            "rho_star": 0.6,
            "alpha_max": 0.08,
        },
        {
            "objective": GoldsteinPrice(),
            "x0": np.array([0.5, -0.5]),
            "steps": 1200,
            "vanilla_alpha": 0.003,
            "controlled_alpha0": 0.003,
            "kp": 0.04,
            "rho_star": 0.5,
            "alpha_max": 0.04,
        },
        {
            "objective": Easom(),
            "x0": np.array([2.6, 3.6]),
            "steps": 1000,
            "vanilla_alpha": 0.01,
            "controlled_alpha0": 0.01,
            "kp": 0.05,
            "rho_star": 0.5,
            "alpha_max": 0.08,
        },
    ]

    for case in cases:
        run_one_objective(output_dir=output_dir, **case)


if __name__ == "__main__":
    main()
