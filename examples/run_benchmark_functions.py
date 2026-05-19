"""Compare fixed and controlled gradient descent on several 2D landscapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive_stepsize_control.objectives import (
    BealeObjective,
    HimmelblauObjective,
    ObjectiveFunction,
    RastriginObjective,
    RosenbrockObjective,
)
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


@dataclass(frozen=True)
class BenchmarkCase:
    """Configuration for one optimizer comparison."""

    objective: ObjectiveFunction
    x0: np.ndarray
    steps: int
    fixed_eta: float
    stochastic_eta: float
    stochastic_noise_scale: float
    controlled_eta0: float
    controlled_eta_max: float
    kp: float = 0.4
    rho_star: float = 0.8
    seed: int = 0


def run_case(case: BenchmarkCase, output_dir: Path) -> None:
    """Run both optimizers and save diagnostics for one objective."""
    objective = case.objective
    prefix = objective.name

    fixed = fixed_gradient_descent(
        objective=objective,
        x0=case.x0,
        eta=case.fixed_eta,
        steps=case.steps,
    )

    stochastic = stochastic_gradient_descent(
        objective=objective,
        x0=case.x0,
        eta=case.stochastic_eta,
        steps=case.steps,
        gradient_noise_scale=case.stochastic_noise_scale,
        seed=case.seed,
    )

    controlled = controlled_gradient_descent(
        objective=objective,
        x0=case.x0,
        eta0=case.controlled_eta0,
        steps=case.steps,
        kp=case.kp,
        rho_star=case.rho_star,
        eta_min=1e-8,
        eta_max=case.controlled_eta_max,
        reject_bad_steps=True,
    )

    paths = [
        plot_objective_values(
            fixed,
            controlled,
            output_dir,
            filename=f"{prefix}_objective_value.png",
            stochastic=stochastic,
        ),
        plot_step_sizes(
            fixed,
            controlled,
            output_dir,
            filename=f"{prefix}_adaptive_step_size.png",
            stochastic=stochastic,
        ),
        plot_rho_ratio(
            controlled,
            case.rho_star,
            output_dir,
            filename=f"{prefix}_rho_ratio.png",
        ),
        plot_trajectory(
            objective,
            fixed,
            controlled,
            output_dir,
            filename=f"{prefix}_trajectory.png",
            stochastic=stochastic,
        ),
        save_summary_csv(
            controlled,
            output_dir,
            filename=f"{prefix}_controlled_diagnostics.csv",
        ),
    ]

    print(f"{objective.name}")
    print(f"  start x: {case.x0}")
    print(f"  start f: {objective.value(case.x0):.6e}")
    print(f"  fixed final x: {fixed.xs[-1]}")
    print(f"  fixed final f: {fixed.fs[-1]:.6e}")
    print(f"  sgd final x: {stochastic.xs[-1]}")
    print(f"  sgd final f: {stochastic.fs[-1]:.6e}")
    print(f"  controlled final x: {controlled.xs[-1]}")
    print(f"  controlled final f: {controlled.fs[-1]:.6e}")
    print(f"  controlled final eta: {controlled.etas[-1]:.6e}")
    print(f"  rejected controlled steps: {np.count_nonzero(~controlled.accepted)}")
    print("  files written:")
    for path in paths:
        print(f"  - {path}")
    print()


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "benchmarks"

    cases = [
        BenchmarkCase(
            objective=RosenbrockObjective(),
            x0=np.array([-1.2, 1.0]),
            steps=3000,
            fixed_eta=1e-3,
            stochastic_eta=8e-4,
            stochastic_noise_scale=0.08,
            controlled_eta0=2e-3,
            controlled_eta_max=0.05,
        ),
        BenchmarkCase(
            objective=HimmelblauObjective(),
            x0=np.array([-3.5, 0.5]),
            steps=400,
            fixed_eta=5e-3,
            stochastic_eta=4e-3,
            stochastic_noise_scale=0.06,
            controlled_eta0=1e-2,
            controlled_eta_max=0.05,
        ),
        BenchmarkCase(
            objective=RastriginObjective(),
            x0=np.array([3.3, 2.8]),
            steps=500,
            fixed_eta=5e-4,
            stochastic_eta=4e-4,
            stochastic_noise_scale=0.04,
            controlled_eta0=2e-3,
            controlled_eta_max=0.02,
            kp=0.25,
        ),
        BenchmarkCase(
            objective=BealeObjective(),
            x0=np.array([1.0, 1.0]),
            steps=1000,
            fixed_eta=1e-3,
            stochastic_eta=8e-4,
            stochastic_noise_scale=0.08,
            controlled_eta0=5e-3,
            controlled_eta_max=0.05,
        ),
    ]

    for case in cases:
        run_case(case, output_dir)


if __name__ == "__main__":
    main()
