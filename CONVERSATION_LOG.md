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

Trust-region recovery update:

- Implemented in `TorchControlledAdam`.
- If an accepted same-minibatch trial step has no backtracking, high smoothed
  `rho`, and tiny `alpha`, the next-alpha multiplier is forced to be at least
  `trust_region_expand_factor`.
- Default settings:

```text
trust_region_expand = True
trust_region_rho_threshold = 0.9
trust_region_alpha_threshold = 1e-4
trust_region_expand_factor = 1.5
```

- New diagnostics in controlled step CSV:
  `alpha_next`, `alpha_update_factor`, and `trust_region_expanded`.
- Tests passed after implementation: `PYTHONPATH=src pytest -q` in
  `controlled_adam_project` gives `9 passed`.

20-epoch Fashion-MNIST benchmark after trust-region recovery:

```text
Dataset: fashion_mnist
Vanilla Adam final test accuracy: 0.8369
Controlled Adam final test accuracy: 0.8115
Controlled Adam final epoch accepted rate: 0.9375
Controlled Adam overall accepted rate: 0.9500
Trust-region expansions: 44 / 640 minibatch steps
Controlled Adam final mean alpha: 1.20e-04
Controlled Adam final next alpha: 1.22e-04
```

This improved controlled Adam over the EMA-only run (`0.8115` vs `0.7988`
final test accuracy) and avoided collapse to the old tiny alpha scale, but
vanilla Adam still wins under this subset/settings (`0.8369`).

Adam-direction ablation mode:

- Added `--ablation` to `controlled_adam_project/examples/run_mnist_demo.py`.
- It trains these variants from the same initialization and minibatch order:
  `vanilla_adam`, `fixed_adam_direction`, `controlled_raw_rho`,
  `controlled_ema`, and `controlled_ema_trust`.
- Purpose: separate the effect of the Adam direction from raw rho control,
  EMA smoothing, and trust-region recovery.
- Smoke test command:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 1 \
  --train-subset 512 \
  --test-subset 256 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_ablation_smoke
```

- Smoke test passed and wrote shared epoch metrics plus one step-diagnostics
  CSV per Adam-direction variant.

20-epoch Fashion-MNIST ablation result:

```text
vanilla_adam          test_acc=0.8369  train_acc=0.8997
fixed_adam_direction test_acc=0.8291  train_acc=0.8967  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.7988  train_acc=0.8193  mean_alpha=2.86e-05
controlled_ema       test_acc=0.7988  train_acc=0.8176  mean_alpha=2.87e-05
controlled_ema_trust test_acc=0.8115  train_acc=0.8391  mean_alpha=1.20e-04
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9453 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9656 alpha_final=3.60e-05
controlled_ema       accepted=0.9672 alpha_final=3.62e-05
controlled_ema_trust accepted=0.9500 alpha_final=1.22e-04 trust_expansions=44/640
```

Interpretation:

- The Adam-style direction with fixed scalar alpha nearly matches PyTorch Adam,
  so the direction implementation is not the main issue.
- The raw and EMA rho controllers shrink alpha too far, leading to undertraining.
- Trust recovery improves the controlled optimizer by keeping alpha larger, but
  it is still much smaller than the useful fixed `1e-3` scale.
- Next likely improvement: make alpha recovery/progress checks more aggressive,
  or separate step acceptance from alpha adaptation so noisy rho does not drive
  the global multiplier too low.

40-epoch Fashion-MNIST ablation result:

```text
vanilla_adam          test_acc=0.8350  train_acc=0.9473
fixed_adam_direction test_acc=0.8389  train_acc=0.9443  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.8027  train_acc=0.8240  mean_alpha=1.19e-05
controlled_ema       test_acc=0.8018  train_acc=0.8232  mean_alpha=1.19e-05
controlled_ema_trust test_acc=0.8252  train_acc=0.8613  mean_alpha=1.27e-04
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9469 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9609 alpha_final=1.23e-05
controlled_ema       accepted=0.9633 alpha_final=1.23e-05
controlled_ema_trust accepted=0.9484 alpha_final=1.48e-04 trust_expansions=109/1280
```

Interpretation after 40 epochs:

- `fixed_adam_direction` slightly beats PyTorch Adam on this deterministic
  subset/run, so the Adam-direction implementation remains healthy.
- Raw rho and EMA controllers undertrain badly because alpha collapses toward
  the lower scale over time.
- Trust-region recovery is clearly better than raw/EMA control, but it still
  stays far below the useful fixed `1e-3` scale and therefore has lower train
  accuracy.
- The next controller improvement should probably target the alpha adaptation
  rule directly, not the direction.

40-epoch Fashion-MNIST ablation with candidate tuned controller settings:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 40 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 1e-4 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 2.0 \
  --controlled-max-alpha-factor 1.2 \
  --output-dir outputs/fashion_mnist_40epochs_ablation_tuned1
```

Result:

```text
vanilla_adam          test_acc=0.8350  train_acc=0.9473
fixed_adam_direction test_acc=0.8389  train_acc=0.9443  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.8242  train_acc=0.8574  mean_alpha=1.19e-04
controlled_ema       test_acc=0.8281  train_acc=0.8574  mean_alpha=1.20e-04
controlled_ema_trust test_acc=0.8467  train_acc=0.9019  mean_alpha=1.52e-04
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9469 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9508 alpha_final=1.43e-04
controlled_ema       accepted=0.9477 alpha_final=1.46e-04
controlled_ema_trust accepted=0.9328 alpha_final=2.04e-04 alpha_max_seen=1.92e-03 trust_expansions=88/1280
```

Interpretation:

- The candidate settings substantially improved every controlled variant by
  preventing the global alpha from collapsing into the `1e-5` range.
- `controlled_ema_trust` is now best on this deterministic subset/run:
  `0.8467` test accuracy vs `0.8350` vanilla Adam and `0.8389`
  fixed Adam-direction.
- It also has the best final test loss (`0.5007`) among the variants in this
  run.
- This supports the hypothesis that the previous weakness was mostly
  controller conservatism, not the Adam direction or same-minibatch diagnostic
  machinery.

40-epoch Fashion-MNIST ablation with aggressive controller settings:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 40 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 2e-4 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.25 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive
```

Result:

```text
vanilla_adam          test_acc=0.8350  train_acc=0.9473
fixed_adam_direction test_acc=0.8389  train_acc=0.9443  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.8389  train_acc=0.8862  mean_alpha=2.16e-04
controlled_ema       test_acc=0.8389  train_acc=0.8855  mean_alpha=2.20e-04
controlled_ema_trust test_acc=0.8438  train_acc=0.9048  mean_alpha=2.30e-04
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9469 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9305 alpha_final=2.22e-04 alpha_max_seen=2.03e-03
controlled_ema       accepted=0.9344 alpha_final=2.18e-04 alpha_max_seen=2.16e-03
controlled_ema_trust accepted=0.9375 alpha_final=2.23e-04 alpha_max_seen=2.16e-03 trust_expansions=23/1280
```

Interpretation:

- Aggressive tuning remains strong and beats vanilla Adam, but it is slightly
  below the first tuned candidate for `controlled_ema_trust`
  (`0.8438` vs `0.8467`).
- Lowering `rho_star` to `0.5` helped raw/EMA control substantially, suggesting
  the old target ratio was too conservative for stochastic neural-net training.
- The higher alpha floor and larger growth cap keep all controlled variants in
  a healthier alpha range.

100-epoch Fashion-MNIST ablation with tuned1 controller settings:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 100 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 1e-4 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 2.0 \
  --controlled-max-alpha-factor 1.2 \
  --output-dir outputs/fashion_mnist_100epochs_ablation_tuned1
```

Final epoch result:

```text
vanilla_adam          test_acc=0.8350  train_acc=0.9934
fixed_adam_direction test_acc=0.8359  train_acc=0.9946  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.8418  train_acc=0.9016  mean_alpha=1.04e-04
controlled_ema       test_acc=0.8398  train_acc=0.9001  mean_alpha=1.06e-04
controlled_ema_trust test_acc=0.8369  train_acc=0.9500  mean_alpha=2.78e-04
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=24  best_test_acc=0.8398
fixed_adam_direction epoch=89  best_test_acc=0.8447
controlled_raw_rho   epoch=94  best_test_acc=0.8457
controlled_ema       epoch=93  best_test_acc=0.8418
controlled_ema_trust epoch=45  best_test_acc=0.8506
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9287 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9403 alpha_final=1.18e-04
controlled_ema       accepted=0.9406 alpha_final=1.15e-04
controlled_ema_trust accepted=0.9225 alpha_final=1.02e-04 alpha_max_seen=1.92e-03 trust_expansions=191/3200
```

Interpretation:

- Tuned trust has the best peak accuracy (`0.8506` at epoch 45), but it gives
  back some performance by epoch 100.
- Raw rho has the best final epoch among controlled variants in this long run,
  though with much lower train accuracy than Adam/fixed-direction.
- Vanilla Adam and fixed-direction heavily fit the small training subset by
  epoch 100, while the controlled variants behave more conservatively.
- For longer runs, the next useful addition is probably scheduling or model
  selection: save/best-epoch reporting, early stopping, or a late-training alpha
  decay/recovery policy.

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
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_controlled_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_trust/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_20epochs_ablation/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive/fashion_mnist_controlled_rho.png`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_loss.png`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_accuracy.png`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_controlled_alpha.png`
- `controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1/fashion_mnist_controlled_rho.png`

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
