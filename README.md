# Adaptive Step-Size Control for Gradient Descent

This project demonstrates a simple feedback-control interpretation of gradient descent.

Instead of using a fixed learning rate, we adapt the learning rate by comparing the **actual decrease** in the objective with the **first-order predicted decrease** from Taylor expansion.

The demo uses a two-dimensional quadratic objective,

```math
f(x, y) = \frac{1}{2}(50x^2 + y^2),
```

which has strong curvature in the `x` direction and weaker curvature in the `y` direction. This makes it a useful toy problem for seeing why adaptive step sizes can help.

---

## Core idea

For gradient descent,

```math
x_{t+1} = x_t - \eta_t \nabla f(x_t),
```

let

```math
g_t = \nabla f(x_t).
```

A first-order Taylor prediction gives

```math
\hat f_{t+1} = f_t - \eta_t \lVert g_t \rVert^2.
```

The predicted decrease is

```math
\Delta \hat f_t = \eta_t \lVert g_t \rVert^2,
```

and the actual decrease is

```math
\Delta f_t = f_t - f_{t+1}.
```

The controller uses the ratio

```math
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t}.
```

If `rho_t` is close to 1, the Taylor model predicted the step well.

If `rho_t` is small or negative, the step was too aggressive.

If `rho_t` is larger than the target, the step may be increased.

The proportional controller updates the learning rate as

```math
\eta_{t+1} = \eta_t \exp\left(K_p(\rho_t - \rho^\star)\right).
```

This keeps the learning rate positive while increasing or decreasing it smoothly.

---

## Workspace structure

```text
adaptive_stepsize_control/
├── README.md
├── PROJECT_HANDOFF.md
├── CONVERSATION_LOG.md
├── OPTIMIZER_VARIANTS_BENCHMARK_REPORT.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── adaptive_stepsize_control/
│       ├── __init__.py
│       ├── objectives.py
│       ├── optimizers.py
│       └── plotting.py
├── examples/
│   ├── run_quadratic_demo.py
│   └── run_benchmark_functions.py
├── tests/
│   └── test_quadratic_demo.py
├── controlled_adam_project/
├── controlled_muon_project/
├── pi_adam_optimizer/
└── pi_muon_optimizer/
```

The root project is the original gradient-descent demo. The controlled
subprojects apply the same actual-versus-predicted decrease controller to
stronger optimizer directions, while the PI folders package the integral
controller variants as standalone PyTorch optimizers:

- `controlled_adam_project/`: Adam chooses the direction; the controller chooses
  the global multiplier.
- `controlled_muon_project/`: Muon-style orthogonalization chooses the
  matrix-shaped direction; the controller chooses the global multiplier.
- `pi_adam_optimizer/` and `pi_muon_optimizer/`: standalone PI-controller
  versions of the Adam and Muon controllers, packaged as PyTorch optimizers.
  PI Muon follows official `torch.optim.Muon` neural-network scope: Muon for
  2D hidden matrix parameters and AdamW-style fallback for the rest.

For a detailed transfer note for another machine or coding agent, read
`PROJECT_HANDOFF.md`.

Project memory is split across the main documents below:

- `CONVERSATION_LOG.md` preserves the nuanced discussion history.
- `DEVELOPMENT_LOG.md` records the chronological engineering and benchmark
  timeline.
- `PROJECT_HANDOFF.md` summarizes the current state and next steps for another
  machine or coding agent.
- `OPTIMIZER_VARIANTS_BENCHMARK_REPORT.md` explains the five neural optimizer
  variants and compares their benchmark performance.
- `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md` explains the self-contained 2D
  function benchmark suite for manager-facing optimizer behavior reports.
- `OPTIMIZER_IMPLEMENTATION_AUDIT.md` records the latest AdamW/Muon
  implementation audit and the remaining intentional differences from PyTorch.

---

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -e .
```

Alternatively, install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Run the demo

```bash
python examples/run_quadratic_demo.py
```

This will run both:

1. fixed-step gradient descent;
2. controlled gradient descent with adaptive learning rate.

It will save plots into the `outputs/` directory:

```text
outputs/objective_value.png
outputs/adaptive_step_size.png
outputs/rho_ratio.png
outputs/trajectory.png
```

Current output cleanup note: older experimental results were archived into
timestamped backup folders so new runs can write to clean top-level output
directories:

```text
outputs/backup_20260526_182414/
controlled_adam_project/outputs/backup_20260526_182414/
controlled_muon_project/outputs/backup_20260526_182414/
```

---

## Run tests

```bash
pytest
```

The tests check that the objective and gradient are correct, and that the controlled optimizer reduces the objective on the demo problem.

---

## Notes

This method is closely related to ideas from:

- adaptive step-size control;
- backtracking line search;
- trust-region methods;
- feedback control of numerical algorithms.

Unlike AdaGrad, RMSProp, or Adam, this method does not adapt the learning rate from gradient-history statistics. Instead, it adapts the learning rate from the mismatch between the predicted and actual decrease in the objective.

A natural hybrid would combine this global controller with a diagonal adaptive preconditioner such as AdaGrad:

```math
x_{t+1} = x_t - \eta_t D_t \nabla f(x_t),
```

where `D_t` controls coordinate-wise scaling and `eta_t` is controlled using the actual-versus-predicted decrease ratio.

---

## Related subproject: controlled Adam

The workspace also contains `controlled_adam_project/`, which applies the same
actual-versus-predicted decrease idea as an outer-loop controller around Adam.
Adam supplies the preconditioned direction, while the controller adapts the
global step multiplier.

The subproject includes 2D function benchmarks and PyTorch neural-network
comparisons on MNIST, Fashion-MNIST, and CIFAR-10. Run it from
`controlled_adam_project/`:

```bash
PYTHONPATH=src python examples/run_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py
PYTHONPATH=src python examples/run_mnist_demo.py --dataset fashion_mnist --download --ablation
```

For manager-facing deterministic function comparisons, use
`FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`. It describes the nine-function,
five-start suite and regenerates a standalone report at
`controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md`.
It also documents the newer longer manager reports, the 15-start aggregate
and 60-start aggregate function reports, and the focused Rastrigin basin benchmark at
`controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/`.

The image benchmark runner supports long-run visibility with
`--print-every N` and `--checkpoint-every N`.

Recent CIFAR-10 Adam tuning runs on a 20k/5k subset showed a useful pattern:

- conservative alpha caps around `1.25e-3` were stable but a bit too cautious;
- a balanced cap around `1.5e-3` improved the raw-rho controller enough to
  beat fixed Adam-direction on peak test accuracy;
- a more open `2e-3` cap did not improve further and made the later epochs less
  efficient.

The best recent result came from the balanced raw-rho setting:

- `controlled_raw_rho`: best test accuracy `0.8232`
- `fixed_adam_direction`: best test accuracy `0.8220`
- `vanilla_adam`: best test accuracy `0.8158`

The archived output folders that capture this tuning sequence are now under
`controlled_adam_project/outputs/backup_20260526_182414/`:

- `controlled_adam_project/outputs/backup_20260526_182414/cifar10_20k_5k_20epochs_adam_conservative_rhostar82/`
- `controlled_adam_project/outputs/backup_20260526_182414/cifar10_20k_5k_20epochs_adam_tuned_balanced_alpha15_rhostar80_beta90/`
- `controlled_adam_project/outputs/backup_20260526_182414/cifar10_20k_5k_20epochs_adam_tuned_open_alpha20_rhostar78_beta90/`
- `controlled_adam_project/outputs/backup_20260526_182414/cifar10_20k_5k_20epochs_adam_tuned_balanced_alpha15_rhostar78_beta90/`
- `controlled_adam_project/outputs/backup_20260526_182414/cifar10_20k_5k_20epochs_adam_tuned_balanced_alpha15_rhostar80_beta85/`

For minibatch training, the control ratio evaluates the trial loss on the
**same minibatch** used to compute the gradient:

```math
\rho_t
=
\frac{
f_{B_t}(\theta_t) - f_{B_t}(\theta_t + \alpha_t p_t)
}{
-\alpha_t \nabla f_{B_t}(\theta_t)^\top p_t
}.
```

Using a different minibatch for the after-step loss would mix real optimization
progress with random minibatch variation, so it would not be a reliable control
signal.

This same-minibatch trial evaluation adds one extra forward loss evaluation per
controlled optimizer step, but not an extra backward pass. Since backpropagation
is usually the dominant cost in neural-network training, the overhead is
expected to be moderate rather than a full doubling of training time. The fair
comparison is still accuracy/loss versus wall-clock time, because the extra
forward pass, data pipeline, model architecture, and hardware utilization can
change the practical cost.

If MNIST is unavailable locally and download is disabled, the script falls back
to `sklearn.datasets.load_digits` for offline smoke testing.

## Related subproject: controlled Muon

The workspace also contains `controlled_muon_project/`, which mirrors the Adam
subproject but uses Muon-style orthogonalized update directions. It supports the
same 2D function benchmark suite and the same MNIST, Fashion-MNIST, and CIFAR-10
image benchmark interface.

Run it from `controlled_muon_project/`:

```bash
PYTHONPATH=src python examples/run_matrix_quadratic_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py
PYTHONPATH=src python examples/run_mnist_demo.py --dataset fashion_mnist --download --ablation
```

The current PyTorch Muon implementation is intentionally educational and uses
CPU/NumPy orthogonalization, so CIFAR-10 runs are slower than the Adam runs.
Use `--print-every N` for live progress on longer Muon experiments.

The Muon function report is generated at
`controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md`.
