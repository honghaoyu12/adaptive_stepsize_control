# Conversation Log

Last updated: 2026-05-19

This log summarizes the current state of the whole `adaptive_stepsize_control`
workspace so future sessions can resume without rediscovering the project.

## Workspace Overview

The workspace contains two related Python projects:

1. Root project: `adaptive_stepsize_control`
   - Demonstrates fixed-step gradient descent, noisy stochastic gradient
     descent, and feedback-controlled gradient descent.
   - The controller adjusts a global learning rate by comparing actual
     objective decrease with first-order predicted decrease.

2. Subproject: `controlled_adam_project`
   - Compares vanilla Adam with outer-loop controlled Adam.
   - Adam supplies the preconditioned direction.
   - The outer controller chooses the scalar global multiplier `alpha` using an
     actual-over-predicted decrease ratio.

Both projects use a `src/` layout, so commands should be run with
`PYTHONPATH=src` unless the package is installed editable with `pip install -e .`.

## Root Project: Adaptive Step-Size Control

Important files:

- `README.md`
- `examples/run_quadratic_demo.py`
- `examples/run_benchmark_functions.py`
- `src/adaptive_stepsize_control/objectives.py`
- `src/adaptive_stepsize_control/optimizers.py`
- `src/adaptive_stepsize_control/plotting.py`
- `tests/test_quadratic_demo.py`
- `outputs/`

Implemented optimizers:

- `fixed_gradient_descent`
- `stochastic_gradient_descent`
- `controlled_gradient_descent`

Implemented root-project objectives:

- `QuadraticObjective`
- `RosenbrockObjective`
- `HimmelblauObjective`
- `RastriginObjective`
- `BealeObjective`

Important plotting updates:

- Objective plots compare fixed GD, SGD, and controlled GD.
- Step-size plots include fixed GD, SGD, and controlled GD.
- Trajectory plots use filled objective contours, contour lines, colorbars, and
  known-minimum markers.
- Optimizer histories include the initial point first, then post-step states.
  This fixed an earlier issue where plotted trajectories appeared not to start
  from the same initial point.

Useful commands:

```bash
PYTHONPATH=src pytest -q
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_quadratic_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_benchmark_functions.py
```

Most recent known test result:

```text
4 passed
```

Root output folders:

- `outputs/`
- `outputs/benchmarks/`

## Subproject: Controlled Adam

Path:

```text
controlled_adam_project/
```

Important files:

- `controlled_adam_project/README.md`
- `controlled_adam_project/examples/run_demo.py`
- `controlled_adam_project/examples/run_mnist_demo.py`
- `controlled_adam_project/src/controlled_adam/objectives.py`
- `controlled_adam_project/src/controlled_adam/optimizers.py`
- `controlled_adam_project/src/controlled_adam/torch_optimizers.py`
- `controlled_adam_project/src/controlled_adam/plotting.py`
- `controlled_adam_project/tests/test_optimizers.py`
- `controlled_adam_project/tests/test_torch_optimizers.py`
- `controlled_adam_project/outputs/`

Implemented optimizers:

- `vanilla_adam`
- `controlled_adam`
- `TorchControlledAdam` for PyTorch minibatch experiments

Controlled Adam idea:

- Adam computes the direction
  `p = -m_hat / (sqrt(v_hat) + eps)`.
- The controller tries `x_trial = x + alpha * p`.
- It computes:
  - predicted decrease: `-alpha * grad.T @ p`
  - actual decrease: `f(x) - f(x_trial)`
  - ratio: `rho = actual / predicted`
- It updates:
  `alpha_next = alpha * exp(kp * (rho - rho_star))`
- It can reject bad steps, backtrack, shrink on non-descent directions, and
  clip `alpha`.
- `TorchControlledAdam` now supports EMA-smoothed rho control and clipped
  multiplicative alpha updates for minibatch neural-network experiments.

Implemented controlled-Adam objectives:

- `AnisotropicQuadratic`
- `Rosenbrock`
- `Himmelblau`
- `Rastrigin`
- `Beale`
- `Ackley`
- `SixHumpCamel`
- `GoldsteinPrice`
- `Easom`

Important controlled-Adam plotting updates:

- Objective plots compare vanilla Adam and controlled Adam.
- Alpha plots compare fixed Adam alpha and controlled Adam alpha.
- Rho plots show the controlled Adam actual-over-predicted ratio.
- Trajectory plots use filled objective landscapes, contour overlays, colorbars,
  and known-minimum markers.
- Histories now include the initial point before post-step states.
- Objective plots automatically use linear scale if objective values are
  nonpositive, because functions like Six-Hump Camel and Easom can be negative.

Useful commands:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py --download
```

Most recent known test result:

```text
7 passed
```

Controlled-Adam demo currently generates plots and CSV diagnostics for:

- `quadratic`
- `rosenbrock`
- `himmelblau`
- `rastrigin`
- `beale`
- `ackley`
- `six_hump_camel`
- `goldstein_price`
- `easom`

## MNIST Experiment

Implemented in `controlled_adam_project/examples/run_mnist_demo.py`.

The experiment compares vanilla Adam with same-minibatch controlled Adam on a
small PyTorch MLP. It writes epoch metrics, controlled-step diagnostics, loss
plots, accuracy plots, alpha plots, and rho plots under `outputs/mnist/` by
default.

Use:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py --download
```

For offline smoke testing without cached MNIST, omit `--download`; the script
falls back to `sklearn.datasets.load_digits`.

Fashion-MNIST local-folder workflow:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 3 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --output-dir outputs/fashion_mnist
```

The root-level `fashion/` folder was validated as Fashion-MNIST IDX gzip files:
60,000 train images/labels and 10,000 test images/labels, 28x28.

Earlier 20-epoch Fashion-MNIST benchmark with raw per-minibatch rho:

```text
Dataset: fashion_mnist
Vanilla Adam final test accuracy: 0.8369
Controlled Adam final test accuracy: 0.7773
Controlled Adam accepted rate: 0.9766 overall
```

Most recent 20-epoch Fashion-MNIST benchmark with EMA-smoothed rho and clipped
alpha updates:

```text
Dataset: fashion_mnist
Vanilla Adam final test accuracy: 0.8369
Controlled Adam final test accuracy: 0.7988
Controlled Adam accepted rate: 0.9672 overall
Controlled Adam final mean alpha: 2.87e-05
```

The EMA controller improved controlled Adam over the raw-rho controller
(`0.7988` vs `0.7773` final test accuracy), but vanilla Adam still wins on this
specific Fashion-MNIST subset/settings.

Important rho/alpha diagnosis:

- When global step size `alpha` becomes tiny, `rho` naturally tends toward 1
  because the objective is locally linear at sufficiently small scales.
- Taylor expansion gives roughly:

```text
rho ~= 1 - [0.5 * alpha * p^T H p] / [-g^T p]
```

- Therefore `rho ~= 1` after alpha collapse is not strong evidence of healthy
  optimization. It can mean the step is so small that the first-order model is
  trivially accurate.
- This is a failure mode:

```text
alpha collapses -> rho approaches 1 -> controller thinks the step quality is good
```

- Future variants should include a progress signal, alpha recovery rule,
  higher useful alpha floor, or treat tiny predicted/actual decreases as
  uninformative.

Fashion-MNIST outputs:

- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_controlled_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_controlled_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ema/fashion_mnist_controlled_rho.png`

Note: `outputs/fashion_mnist/` also contains older `sklearn_digits_*` files from
an earlier fallback run. The `fashion_mnist_*` files are the real Fashion-MNIST
benchmark.

Most recent smoke run:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --epochs 1 \
  --train-subset 256 \
  --test-subset 128 \
  --batch-size 64 \
  --output-dir outputs/mnist_smoke
```

Result:

```text
Dataset: sklearn_digits
Vanilla Adam final test accuracy: 0.5547
Controlled Adam final test accuracy: 0.5547
Controlled Adam accepted rate: 1.0000
```

Smoke outputs:

- `controlled_adam_project/outputs/mnist_smoke/mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/mnist_smoke/mnist_controlled_step_diagnostics.csv`
- `controlled_adam_project/outputs/mnist_smoke/mnist_loss.png`
- `controlled_adam_project/outputs/mnist_smoke/mnist_accuracy.png`
- `controlled_adam_project/outputs/mnist_smoke/mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/mnist_smoke/mnist_controlled_rho.png`

Important agreed caveat:

- The controlled Adam trial loss must be evaluated on the **same minibatch**
  used to compute the gradient.
- For minibatch `B_t`, use:

```text
rho_t =
    [f_Bt(theta_t) - f_Bt(theta_t + alpha_t p_t)]
    / [-alpha_t grad f_Bt(theta_t)^T p_t]
```

- Do not evaluate `loss_after` on a different minibatch. That would confuse
  minibatch sampling noise with true optimization progress.

Implemented minibatch step:

1. Draw `B_t`.
2. Compute `loss_before` and gradients on `B_t`.
3. Use Adam moments to form direction `p_t`.
4. Compute predicted decrease.
5. Take trial step.
6. Recompute `loss_after` on the same `B_t`.
7. Accept/reject, update `alpha`, and log diagnostics.

Fair-comparison requirements:

- same model architecture;
- same initial weights;
- same train/test split;
- same minibatch order and random seed;
- log train loss, train accuracy, test accuracy, alpha, rho, and accepted-step
  rate.

Practical note:

- Full MNIST may require network access if it is not cached. An offline fallback
  is `sklearn.datasets.load_digits`, but that is not full MNIST.

## Current Observations

Root project:

- Controlled GD tends to outperform fixed GD and noisy SGD on simple
  ill-conditioned or curved-valley examples.
- Rastrigin shows a useful limitation: all local gradient-based methods can get
  trapped in a local well.

Controlled Adam subproject:

- Controlled Adam strongly improves over vanilla Adam on the quadratic,
  Rosenbrock, Beale, and Goldstein-Price settings currently in the demo.
- Vanilla Adam currently beats controlled Adam on the chosen Himmelblau setting.
- Both Adam variants land in the same local basin on Rastrigin and Ackley with
  the current starting points.
- Six-Hump Camel and Easom include negative objective values, so their objective
  curves are plotted on a linear scale.

## Notes For Future Work

Potential next steps:

- Tune per-objective hyperparameters for controlled Adam, especially
  Himmelblau, Ackley, and Six-Hump Camel.
- Add side-by-side summary tables of final objective values and accepted-step
  rates.
- Add optional start-point sweeps to show basin sensitivity.
- Run the MNIST experiment on full MNIST with download/cache available and
  compare multiple seeds.
- Tune controlled Adam hyperparameters for the MNIST experiment.
- Consider sharing plotting style helpers between the root project and
  `controlled_adam_project` if the projects are later merged.
