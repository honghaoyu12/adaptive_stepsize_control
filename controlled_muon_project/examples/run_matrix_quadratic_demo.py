"""Compare vanilla Muon and controlled Muon on a matrix quadratic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from controlled_muon.objectives import MatrixQuadraticObjective
from controlled_muon.optimizers import MuonConfig, controlled_muon, vanilla_muon
from controlled_muon.plotting import (
    plot_accepted_steps,
    plot_controlled_step_size,
    plot_distance_to_target,
    plot_objective_values,
    plot_rho_ratio,
    save_controlled_diagnostics,
)


def main() -> None:
    objective = MatrixQuadraticObjective.random_anisotropic(
        shape=(12, 8),
        seed=11,
        curvature_min=0.2,
        curvature_max=50.0,
    )

    W0 = np.zeros_like(objective.target)
    steps = 160
    rho_star = 0.7

    # Use the Newton-Schulz polar iteration to keep the demo Muon-like.
    config = MuonConfig(
        momentum=0.90,
        nesterov=True,
        orthogonalizer="newton_schulz",
        ns_steps=8,
        update_scale=1.0,
    )

    vanilla = vanilla_muon(
        objective=objective,
        W0=W0,
        eta=0.035,
        steps=steps,
        config=config,
    )

    controlled = controlled_muon(
        objective=objective,
        W0=W0,
        alpha0=0.08,
        steps=steps,
        config=config,
        kp=0.45,
        rho_star=rho_star,
        rho_min=0.0,
        alpha_min=1e-6,
        alpha_max=0.3,
        descent_fail_shrink=0.25,
        rollback_state_on_reject=True,
    )

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    paths = [
        plot_objective_values(vanilla, controlled, output_dir),
        plot_distance_to_target(vanilla, controlled, output_dir),
        plot_controlled_step_size(controlled, output_dir),
        plot_rho_ratio(controlled, rho_star, output_dir),
        plot_accepted_steps(controlled, output_dir),
        save_controlled_diagnostics(controlled, output_dir),
    ]

    print("Demo complete. Files written:")
    for path in paths:
        print(f"- {path}")

    print()
    print("Vanilla Muon")
    print(f"  final f: {vanilla.fs[-1]:.6e}")
    print(f"  final distance to target: {vanilla.distances[-1]:.6e}")

    print()
    print("Controlled Muon")
    print(f"  final f: {controlled.fs[-1]:.6e}")
    print(f"  final distance to target: {controlled.distances[-1]:.6e}")
    print(f"  final alpha: {controlled.step_sizes[-1]:.6e}")
    print(f"  accepted steps: {controlled.accepted.sum()} / {len(controlled.accepted)}")


if __name__ == "__main__":
    main()
