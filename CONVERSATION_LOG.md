# Conversation Log

Last updated: 2026-05-24

Current handoff note: see `PROJECT_HANDOFF.md` first. For a concise
chronological engineering timeline, see `DEVELOPMENT_LOG.md`. This file remains
important as the nuanced conversation memory: discussion, interpretation,
questions, and why decisions evolved.

For manager-facing deterministic function optimization results, see
`FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md` and regenerate the local report with
`controlled_adam_project/examples/run_function_benchmark_report.py`.

This log summarizes the current state of the whole `adaptive_stepsize_control`
workspace so future sessions can resume without rediscovering the project.

## Workspace Overview

The workspace contains three related Python projects:

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
   - Now includes a self-contained deterministic function benchmark report
     runner for manager updates.

3. Subproject: `controlled_muon_project`
   - Muon version of the controlled Adam project.
   - Muon-style orthogonalization supplies matrix-shaped directions.
   - Supports the same function benchmark suite plus MNIST, Fashion-MNIST, and
     CIFAR-10 image benchmarks.

Recent Adam CIFAR-10 tuning context:

- Conservative run (`alpha_max=1.25e-3`) was stable but slightly too cautious.
- Balanced run (`alpha_max=1.5e-3`) gave the best controlled result so far,
  with `controlled_raw_rho` reaching `0.8232` best test accuracy on a 20k/5k
  CIFAR-10 subset for 20 epochs.
- A follow-up balanced run with `rho_star=0.78` kept raw-rho strong but did not
  improve beyond the best `0.8232` peak; it finished around `0.8214` final and
  best test accuracy.
- A faster-EMA run with `rho_beta=0.85` made EMA ramp sooner, but it did not
  improve final performance: raw-rho again peaked at `0.8232`, EMA peaked at
  `0.8164`, and EMA-trust peaked at `0.8114`.
- More open run (`alpha_max=2e-3`) did not improve beyond that and made late
  epochs less efficient.
- Across these runs, raw-rho control outperformed the EMA variants on this
  CIFAR setup; the trust hook did not materially change the result.
- We then implemented `--model lenet_cifar` and ran the first architecture
  transfer test. LeNet was fast, but the controller did not help:
  vanilla/fixed Adam finished at `0.5868`, while controlled raw-rho, EMA, and
  EMA-trust finished at `0.5656`, `0.5620`, and `0.5722`.
- This led to a runtime discussion: CIFAR ResNet is likely much slower on the
  current CPU setup, so it should start with a 3-epoch 5k/1k smoke run, then a
  10-epoch 10k/2k medium run before any full 20-epoch ablation.
- Given the runtime concern, Fashion-MNIST CNN is probably the cheaper next
  architecture-variation test.
- We followed that advice and ran a Fashion-MNIST CNN benchmark. Raw-rho
  slightly beat fixed Adam-direction on final accuracy (`0.8870` vs `0.8866`),
  while EMA and EMA-trust were close behind at `0.8844`. The run was fast
  enough to be practical, around `4.5-5.6s` per epoch per variant.
- This makes Fashion-MNIST CNN a better low-cost architecture-transfer signal
  than LeNet, while CIFAR ResNet still looks like a staged experiment rather
  than a default next run.
- The five-seed Fashion-MNIST CNN follow-up showed that the single-seed raw-rho
  edge did not hold: vanilla/fixed Adam averaged about `0.8945/0.8946` final
  test accuracy, while controlled raw-rho averaged `0.8889`. Controlled variants
  had about `1.22x-1.24x` relative wall-clock time.
- We discussed better Fashion-MNIST CNN parameters. The current setting appears
  too conservative because controlled variants ended near `alpha ~= 6.8e-4`,
  below the Adam-scale `1e-3` baseline. The recommended next candidate is

- Candidate A was run on three Fashion-MNIST CNN seeds. It improved controlled
  variants by keeping alpha near `8.9e-4`, but still did not clearly beat fixed
  Adam-direction: three-seed final accuracies were vanilla `0.8946`, fixed
  `0.8960`, raw-rho `0.8944`, EMA `0.8945`, and EMA-trust `0.8943`.
  `alpha_min=9e-4`, `alpha_max=1.5e-3`, `rho_star=0.75`, `rho_beta=0.90`,
  `kp=0.02`, with factor clip `[0.98, 1.015]`.

Recent deterministic function optimization report:

- The user wanted a short but detailed non-deep-learning report for a manager
  update focused on how the controlled optimizers behave on functions.
- We added `controlled_adam_project/examples/run_function_benchmark_report.py`.
  It is intentionally self-contained and simple: nine existing 2D functions,
  five fixed starts per function, and four optimizers.
- The compared optimizers are vanilla Adam, controlled raw-rho Adam,
  controlled EMA-rho Adam, and controlled EMA+trust Adam.
- The generated local report is
  `controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md`.
  Outputs are gitignored, so the tracked reference is
  `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`.
- We also generated a shorter manager-facing Adam report at
  `controlled_adam_project/outputs/function_report_manager_trimmed/` using only
  Quadratic, Beale, and Goldstein-Price. This is the better folder for a concise
  update focused on controlled Adam's strengths.
- Important result snapshot: controlled raw-rho ties or wins success rate on
  all nine functions, EMA-rho ties or wins on seven, and vanilla Adam still
  ties or wins several median residual comparisons when its fixed learning rate
  happens to be well matched.
- After the user asked why the trust-region method was absent from the
  benchmark, we added `controlled_ema_trust` to the deterministic Adam report.
  In the current trimmed three-function run, `median_trust_expansions` is `0.0`,
  so EMA+trust overlaps EMA-rho there; it is present for consistency, not
  because the trust expansion visibly activates on those plots.
- The honest manager story is: the controller improves robustness and helps
  especially on curved or scale-sensitive landscapes such as Rosenbrock, Beale,
  and Goldstein-Price; it is not a global optimizer and does not solve local
  basin traps on multimodal functions such as Rastrigin, Ackley, and Six-hump
  camel.
- The user then asked for the Muon variants too. We added
  `controlled_muon_project/examples/run_function_benchmark_report.py`, which
  compares `vanilla_muon`, `controlled_raw_rho`, `controlled_ema`, and
  `controlled_ema_trust` on the same nine functions.
- The Muon function report is generated at
  `controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md`.
- We also generated a shorter Chinese manager-facing Muon report at
  `controlled_muon_project/outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md`.
- The earlier function-report-only `fixed_muon_direction` diagnostic was
  removed because it duplicated `vanilla_muon` in the local 2D runner: both use
  fixed alpha and no rho controller.
- For 2D vector objectives, the Muon report uses a vector analogue: the
  momentum vector is treated as a column matrix and orthogonalized with the
  same utility used elsewhere in the Muon subproject.
- Both Adam and Muon function reports now generate Chinese companion reports and
  standalone `*_surface_3d.png` plots with formulas printed inside the figures.
- The Muon result is more conservative than Adam: vanilla Muon is very
  competitive on this suite, tying or winning success rate on all nine
  objectives. Controlled Muon improves some cases, especially Quadratic
  residuals, but does not dominate overall. This is useful because it shows the
  controller is not automatically better for every base direction.

- Candidate C was then run on the same three Fashion-MNIST CNN seeds with a
  near-fixed Adam setting. It kept alpha close to `9.3e-4` and nudged
  controlled_raw_rho slightly ahead of the other controlled variants, but it
  still did not beat fixed Adam-direction on mean final accuracy. The
  three-seed means were vanilla `0.8946`, fixed `0.8960`, raw-rho `0.8951`,
  EMA `0.8937`, and EMA-trust `0.8937`.

- Candidate B was run next with a faster-recovery setting
  (`alpha_min=8e-4`, `alpha_max=1.5e-3`, `rho_star=0.70`, `kp=0.03`). It did
  not improve the picture: mean final accuracies were vanilla `0.8946`, fixed
  `0.8960`, raw-rho `0.8921`, EMA `0.8930`, and EMA-trust `0.8930`.
- Added `--model resnet_cifar`, a compact CIFAR-friendly ResNet with
  approximately 175k trainable parameters, and ran the staged 3-epoch 5k/1k
  CIFAR-10 smoke test. The run completed cleanly. Final test accuracies were
  vanilla `0.3610`, fixed `0.3730`, raw-rho `0.4100`, EMA `0.3800`, and
  EMA-trust `0.3800`; all controlled variants accepted every step and kept
  alpha near `1e-3`.
- The next staged ResNet benchmark was expanded to 20 epochs on 10k train /
  2k test, still with the balanced Adam-scale controller setting and full
  ablation. It completed successfully with per-epoch checkpoints and progress
  prints. Final / best test accuracy was: vanilla Adam `0.6915 / 0.6915`,
  fixed Adam-direction `0.6875 / 0.6975`, raw-rho `0.7395 / 0.7395`, EMA
  `0.7135 / 0.7135`, and EMA-trust `0.7135 / 0.7135`.
- In this ResNet run, raw-rho was clearly best on the single seed. All
  controlled/fixed variants accepted every step. Raw-rho, EMA, and EMA-trust
  reached the `alpha_max=1.5e-3` cap by around the middle of training. EMA and
  EMA-trust were numerically identical, so the trust-region expansion again
  did not create a distinct trajectory under these conservative bounds.
- Later, when the user asked why the three-seed trust summary matched EMA, we
  checked the step diagnostics and found the exact cause: `controlled_ema_trust`
  had `0/1580` trust expansions for each balanced ResNet seed (`123`, `456`,
  `789`). The run used `alpha_min=1e-3`, but the trust trigger was
  `trust_region_alpha_threshold=1e-4`, which is below the active alpha floor.
  Therefore the trust branch was enabled but unreachable; in this benchmark,
  EMA+trust should be treated as the same effective algorithm as EMA.
- We concluded that a meaningful Adam-scale trust follow-up should keep
  `alpha_min` near `1e-3` to avoid collapse, but raise
  `trust_region_alpha_threshold` near the floor, such as `1e-3` or `1.05e-3`,
  with a gentle `trust_region_expand_factor` such as `1.1` or `1.2`.
- Two follow-up ResNet parameter tests were run. Candidate 1 raised
  `alpha_max` to `1.75e-3` with a more cautious controller
  (`rho_star=0.82`, `kp=0.015`), but it underperformed: raw-rho finished
  `0.7120`, EMA/EMA-trust finished `0.6695`, and the higher cap did not help.
- The stronger-fixed-LR control used `lr=1.5e-3` with the original
  `alpha_max=1.5e-3`. Vanilla Adam improved to `0.7065`, fixed Adam-direction
  reached best `0.7040`, and EMA/EMA-trust reached `0.7250`, but raw-rho only
  reached `0.6985`. This suggests the original raw-rho `0.7395` result was not
  merely because `1.5e-3` is a better fixed learning rate. Ramping from
  `1e-3` to `1.5e-3` appears different from starting at `1.5e-3`.
- We then ran the original balanced ResNet setting on two more seeds (`456`
  and `789`) and combined them with the existing seed `123` run. Three-seed
  final accuracy means were: vanilla `0.6887`, fixed `0.6938`, raw-rho
  `0.7150`, EMA `0.7083`, and EMA-trust `0.7083`. Three-seed best accuracy
  means were: vanilla `0.6998`, fixed `0.7118`, raw-rho `0.7235`, EMA
  `0.7190`, and EMA-trust `0.7190`. This supports a modest controlled-optimizer
  advantage on this setup, but also shows seed `123` raw-rho was unusually
  strong.


All projects use a `src/` layout, so commands should be run with
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

CIFAR-10 CNN benchmark:

- Added CIFAR-10 support to `controlled_adam_project/examples/run_mnist_demo.py`.
- Added `SmallCIFARCNN`, a simple 3-layer convolutional classifier for 32x32 RGB
  images.
- Added `--model {auto,mlp,cnn}`. `auto` uses the MLP for MNIST/Fashion-MNIST
  and the CNN for CIFAR-10.
- CIFAR-10 download through torchvision initially produced a truncated archive.
  A resumable `curl` repair fixed the archive checksum, then torchvision
  extracted it successfully under `controlled_adam_project/data/`.
- Added a root `.gitignore` rule for local CIFAR tarballs.

Smoke checks:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --download \
  --epochs 1 \
  --train-subset 256 \
  --test-subset 128 \
  --batch-size 64 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/cifar10_cnn_ablation_smoke
```

and a no-download load smoke after extraction:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 1 \
  --train-subset 64 \
  --test-subset 32 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir outputs/cifar10_cnn_load_smoke
```

20-epoch CIFAR-10 CNN ablation on a 5000/1000 deterministic subset with tuned1
controller settings:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 20 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 1e-4 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 2.0 \
  --controlled-max-alpha-factor 1.2 \
  --output-dir outputs/cifar10_cnn_20epochs_ablation_tuned1
```

Final epoch result:

```text
vanilla_adam          test_acc=0.397  train_acc=0.4092
fixed_adam_direction test_acc=0.402  train_acc=0.4182  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.302  train_acc=0.3056  mean_alpha=1.01e-04
controlled_ema       test_acc=0.305  train_acc=0.3060  mean_alpha=1.01e-04
controlled_ema_trust test_acc=0.311  train_acc=0.3216  mean_alpha=1.00e-04
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=19  best_test_acc=0.406
fixed_adam_direction epoch=20  best_test_acc=0.402
controlled_raw_rho   epoch=20  best_test_acc=0.302
controlled_ema       epoch=20  best_test_acc=0.305
controlled_ema_trust epoch=16  best_test_acc=0.314
```

Diagnostics:

```text
fixed_adam_direction accepted=0.8938 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.8588 alpha_final=1.01e-04
controlled_ema       accepted=0.8638 alpha_final=1.01e-04
controlled_ema_trust accepted=0.8300 alpha_final=1.01e-04 alpha_max_seen=2.33e-03 trust_expansions=21/800
```

Interpretation:

- The CIFAR-10 CNN path works and writes the same epoch/diagnostic plots.
- On this harder RGB benchmark, the Fashion-MNIST tuned controller is too
  conservative. Controlled variants remain around the alpha floor and lag
  vanilla/fixed Adam-direction by about 9 percentage points.
- Next likely CIFAR improvement: tune controller settings separately for CNNs,
  probably with a larger useful alpha floor or a progress-aware recovery rule.

20-epoch CIFAR-10 CNN ablation with first CIFAR-specific tuning:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 20 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-trust-alpha-threshold 2e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.25 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1
```

Final epoch result:

```text
vanilla_adam          test_acc=0.397  train_acc=0.4092
fixed_adam_direction test_acc=0.402  train_acc=0.4182  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.337  train_acc=0.3484  mean_alpha=4.76e-04
controlled_ema       test_acc=0.352  train_acc=0.3494  mean_alpha=4.81e-04
controlled_ema_trust test_acc=0.354  train_acc=0.3506  mean_alpha=4.52e-04
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=19  best_test_acc=0.406
fixed_adam_direction epoch=20  best_test_acc=0.402
controlled_raw_rho   epoch=19  best_test_acc=0.353
controlled_ema       epoch=19  best_test_acc=0.361
controlled_ema_trust epoch=19  best_test_acc=0.363
```

Diagnostics:

```text
fixed_adam_direction accepted=0.8938 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.8600 alpha_final=5.14e-04 alpha_max_seen=1.62e-03
controlled_ema       accepted=0.8600 alpha_final=5.06e-04 alpha_max_seen=1.62e-03
controlled_ema_trust accepted=0.8512 alpha_final=5.01e-04 alpha_max_seen=3.17e-03 trust_expansions=1/800
```

Interpretation:

- Raising the alpha floor from `1e-4` to `5e-4` improved the controlled CNN
  variants substantially, but they still lag fixed Adam-direction.
- Trust expansion barely fires on this setting (`1/800`), so the main effect is
  the higher alpha floor and lower `rho_star`.
- CIFAR/CNN likely needs either an even higher floor closer to `1e-3`, looser
  acceptance, or a progress-aware rule rather than relying on high-rho trust
  expansion.

Stronger CIFAR-10 CNN setup:

- Replaced the tiny CIFAR CNN with a stronger batch-normalized CNN:
  two Conv-BN-ReLU layers per block, three max-pool blocks, and a small MLP
  classifier head.
- BatchNorm uses `track_running_stats=False` so same-minibatch trial loss
  evaluations do not mutate running statistics.
- CIFAR-10 training now uses normalization plus random crop and horizontal flip.
- Train/test metrics use deterministic normalized evaluation transforms on the
  same subset indices, so reported train accuracy is not polluted by random
  augmentation.
- The script resets the seed at the start of each epoch for each optimizer
  variant, keeping augmentation randomness aligned across variants.

Smoke checks after the stronger-CNN refactor:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 1 \
  --train-subset 128 \
  --test-subset 64 \
  --batch-size 64 \
  --lr 1e-3 \
  --output-dir outputs/cifar10_stronger_cnn_smoke

MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 1 \
  --train-subset 128 \
  --test-subset 64 \
  --batch-size 64 \
  --lr 1e-3 \
  --output-dir outputs/fashion_mnist_after_cifar_refactor_smoke
```

10-epoch CIFAR-10 stronger-CNN ablation on the 5000/1000 deterministic subset:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 10 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-trust-alpha-threshold 2e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.25 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1
```

Final epoch result:

```text
vanilla_adam          test_acc=0.584  train_acc=0.6528
fixed_adam_direction test_acc=0.596  train_acc=0.6456  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.515  train_acc=0.5426  mean_alpha=2.39e-03
controlled_ema       test_acc=0.589  train_acc=0.6590  mean_alpha=6.60e-04
controlled_ema_trust test_acc=0.589  train_acc=0.6590  mean_alpha=6.60e-04
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=9   best_test_acc=0.611
fixed_adam_direction epoch=9   best_test_acc=0.601
controlled_raw_rho   epoch=8   best_test_acc=0.553
controlled_ema       epoch=10  best_test_acc=0.589
controlled_ema_trust epoch=10  best_test_acc=0.589
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9925 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9900 alpha_final=3.07e-03 alpha_max_seen=3.08e-03
controlled_ema       accepted=0.9925 alpha_final=9.34e-04 alpha_max_seen=2.30e-03
controlled_ema_trust accepted=0.9925 alpha_final=9.34e-04 alpha_max_seen=2.30e-03 trust_expansions=0/400
```

Interpretation:

- The vanilla CIFAR baseline is now much more reasonable for a 5000-image
  subset and 10 epochs (`0.584` final, `0.611` best).
- EMA-controlled Adam is close to vanilla/fixed-direction and keeps alpha near
  the useful `1e-3` scale.
- Raw rho overshoots alpha to about `3e-3` and performs worse.
- Trust recovery did not fire in this run because EMA control already kept
  alpha in a healthy range.
- A longer run or full-dataset run is now meaningful, but it will take
  noticeably longer with the stronger CNN.

40-epoch CIFAR-10 stronger-CNN ablation on the 5000/1000 deterministic subset:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-trust-alpha-threshold 2e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.25 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1
```

Final epoch result:

```text
vanilla_adam          test_acc=0.718  train_acc=0.8804
fixed_adam_direction test_acc=0.717  train_acc=0.8886  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.640  train_acc=0.6954  mean_alpha=4.99e-02
controlled_ema       test_acc=0.601  train_acc=0.6364  mean_alpha=5.00e-02
controlled_ema_trust test_acc=0.601  train_acc=0.6364  mean_alpha=5.00e-02
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=34  best_test_acc=0.720
fixed_adam_direction epoch=40  best_test_acc=0.717
controlled_raw_rho   epoch=23  best_test_acc=0.692
controlled_ema       epoch=23  best_test_acc=0.672
controlled_ema_trust epoch=23  best_test_acc=0.672
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9956 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9956 alpha_final=4.92e-02 alpha_max_seen=5.00e-02
controlled_ema       accepted=0.9969 alpha_final=5.00e-02 alpha_max_seen=5.00e-02
controlled_ema_trust accepted=0.9969 alpha_final=5.00e-02 alpha_max_seen=5.00e-02 trust_expansions=0/1600
```

Interpretation:

- Vanilla/fixed Adam are now credible on the subset, reaching about `0.72`
  test accuracy.
- Controlled variants peak around epoch 23, then degrade because alpha keeps
  growing until it hits `alpha_max=0.05`, far above the useful fixed `1e-3`
  scale.
- Trust recovery is irrelevant in this long CIFAR run because it never fires;
  the problem is now excessive alpha growth, not collapse.
- Next likely CIFAR setting: cap `controlled-alpha-max` around `2e-3` to
  `5e-3`, or add a schedule/progress-aware rule that stops expansion when
  generalization or actual progress stalls.

40-epoch CIFAR-10 stronger-CNN ablation with capped alpha:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-alpha-max 3e-3 \
  --controlled-trust-alpha-threshold 2e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.1 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3
```

Final epoch result:

```text
vanilla_adam          test_acc=0.718  train_acc=0.8804
fixed_adam_direction test_acc=0.717  train_acc=0.8886  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.683  train_acc=0.8296  mean_alpha=2.98e-03
controlled_ema       test_acc=0.715  train_acc=0.8754  mean_alpha=3.00e-03
controlled_ema_trust test_acc=0.684  train_acc=0.8522  mean_alpha=3.00e-03
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=34  best_test_acc=0.720
fixed_adam_direction epoch=40  best_test_acc=0.717
controlled_raw_rho   epoch=38  best_test_acc=0.739
controlled_ema       epoch=40  best_test_acc=0.715
controlled_ema_trust epoch=33  best_test_acc=0.720
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9956 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9938 alpha_final=3.00e-03 alpha_max_seen=3.00e-03
controlled_ema       accepted=0.9950 alpha_final=3.00e-03 alpha_max_seen=3.00e-03
controlled_ema_trust accepted=0.9944 alpha_final=3.00e-03 alpha_max_seen=3.00e-03 trust_expansions=2/1600
```

Interpretation:

- Capping alpha at `3e-3` largely fixed the previous long-run failure where
  alpha grew to `5e-2`.
- Controlled EMA now essentially matches vanilla/fixed Adam at the final epoch
  (`0.715` vs `0.718/0.717`).
- Raw rho reaches the best peak score (`0.739` at epoch 38) but gives some
  performance back by epoch 40.
- Trust expansion again has little impact on this CIFAR setting; the main
  control variables are alpha cap, rho target, and EMA/raw rho behavior.
- Next useful experiment: add best-epoch reporting/early stopping and test a
  slightly tighter cap, e.g. `controlled-alpha-max=2e-3`, because every
  controlled variant still ends at the cap.

40-epoch CIFAR-10 stronger-CNN ablation with tighter alpha cap and metadata:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-alpha-max 1.5e-3 \
  --controlled-max-alpha-factor 1.05 \
  --controlled-rho-star 0.6 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 1.5 \
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

The runner now writes `run_metadata.json` and `run_metadata.txt` before
training, including:

- Python/PyTorch/device/seed;
- dataset full sizes and subset sizes;
- train/eval transforms;
- model class, full architecture, and trainable parameter count;
- optimizer variants and all controller hyperparameters.

Run metadata summary:

```text
dataset: CIFAR-10
train_size_full/test_size_full: 50000/10000
train_size_used/test_size_used: 5000/1000
model: SmallCIFARCNN
trainable_parameters: 815018
train_transform: RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor, Normalize
eval_transform: ToTensor, Normalize
```

Final epoch result:

```text
vanilla_adam          test_acc=0.718  train_acc=0.8804
fixed_adam_direction test_acc=0.717  train_acc=0.8886  mean_alpha=1.00e-03
controlled_raw_rho   test_acc=0.711  train_acc=0.8704  mean_alpha=1.06e-03
controlled_ema       test_acc=0.704  train_acc=0.8550  mean_alpha=1.50e-03
controlled_ema_trust test_acc=0.731  train_acc=0.8936  mean_alpha=9.49e-04
```

Best epoch by test accuracy:

```text
vanilla_adam          epoch=34  best_test_acc=0.720
fixed_adam_direction epoch=40  best_test_acc=0.717
controlled_raw_rho   epoch=38  best_test_acc=0.735
controlled_ema       epoch=33  best_test_acc=0.709
controlled_ema_trust epoch=40  best_test_acc=0.731
```

Diagnostics:

```text
fixed_adam_direction accepted=0.9956 alpha_final=1.00e-03
controlled_raw_rho   accepted=0.9931 alpha_final=1.29e-03 alpha_max_seen=1.50e-03
controlled_ema       accepted=0.9944 alpha_final=1.50e-03 alpha_max_seen=1.50e-03
controlled_ema_trust accepted=0.9938 alpha_final=1.23e-03 alpha_max_seen=1.50e-03 trust_expansions=1/1600
```

Interpretation:

- This is the best CIFAR-10 stronger-CNN final result so far.
- `controlled_ema_trust` beats vanilla/fixed Adam at the final epoch
  (`0.731` vs `0.718/0.717`), while raw rho has the highest peak (`0.735`).
- The tighter `1.5e-3` cap keeps controlled alpha near the useful Adam scale
  instead of letting it drift to `3e-3` or `5e-2`.
- Trust expansion still barely fires, so the improvement mostly comes from
  alpha range and slower growth, not from frequent trust recovery.
- Best-epoch reporting and incremental progress logging are now high-priority
  quality-of-life improvements for long CNN runs.

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
- `controlled_adam_project/outputs/cifar10_cnn_ablation_smoke/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_load_smoke/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned1/cifar10_controlled_rho.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_cnn_20epochs_ablation_tuned_cifar1/cifar10_controlled_rho.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_smoke/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/fashion_mnist_after_cifar_refactor_smoke/fashion_mnist_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1/cifar10_controlled_rho.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1/cifar10_controlled_rho.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3/cifar10_controlled_rho.png`
- `controlled_adam_project/outputs/cifar10_metadata_smoke/run_metadata.json`
- `controlled_adam_project/outputs/cifar10_metadata_smoke/run_metadata.txt`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/run_metadata.json`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/run_metadata.txt`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_epoch_metrics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_fixed_adam_direction_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_controlled_raw_rho_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_controlled_ema_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_controlled_ema_trust_step_diagnostics.csv`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_loss.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_accuracy.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_controlled_alpha.png`
- `controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3/cifar10_controlled_rho.png`

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

---

## 2026-05-20 Larger CIFAR-10 Adam Run With Progress Logging

User asked for a larger CIFAR-10 test with more data and specifically wanted
checkpoints plus periodic printed results because previous CIFAR/Muon runs were
long and felt opaque.

Implemented in:

```text
controlled_adam_project/examples/run_mnist_demo.py
```

Added:

```text
--print-every N
--checkpoint-every N
```

Behavior:

- `--print-every` prints compact per-epoch metrics for each optimizer variant.
- `--checkpoint-every` writes per-epoch checkpoints under
  `<output_dir>/checkpoints/`.
- Checkpoints include model state, optimizer/controller state where available,
  epoch number, dataset name, run name, and accumulated metrics.

Verification before the long run:

```text
controlled_adam_project tests: 9 passed
tiny CIFAR progress/checkpoint smoke run completed successfully
```

Larger CIFAR-10 command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 20000 \
  --test-subset 5000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-alpha-max 1.5e-3 \
  --controlled-max-alpha-factor 1.05 \
  --controlled-rho-star 0.6 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 1.5 \
  --checkpoint-every 1 \
  --print-every 1 \
  --output-dir outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Output:

```text
controlled_adam_project/outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Final test accuracy:

```text
vanilla_adam          0.8288
fixed_adam_direction 0.8280
controlled_raw_rho   0.8232
controlled_ema       0.8278
controlled_ema_trust 0.8278
```

Best test accuracy:

```text
vanilla_adam          0.8344 at epoch 38
fixed_adam_direction 0.8402 at epoch 39
controlled_raw_rho   0.8292 at epoch 38
controlled_ema       0.8324 at epoch 37
controlled_ema_trust 0.8324 at epoch 37
```

Important diagnostics:

```text
controlled_raw_rho:   final mean_alpha 1.5e-3, final mean_rho 0.8148, accepted 100%
controlled_ema:       final mean_alpha 1.5e-3, final mean_rho 0.8166, accepted 100%
controlled_ema_trust: final mean_alpha 1.5e-3, final mean_rho 0.8166, accepted 100%
```

Interpretation discussed:

- The larger 20k/5k subset made CIFAR accuracy much more plausible than the
  earlier 5k/1k runs. Vanilla Adam reached about 83% test accuracy rather than
  the earlier low-70s.
- The previous concern that vanilla Adam was "unreasonably low" was partly a
  small-data and noisy-evaluation issue.
- Fixed Adam-direction had the best peak result on this run: `0.8402`.
- Controlled variants did not collapse. They accepted all steps and quickly
  saturated the `1.5e-3` alpha cap.
- Same-minibatch rho judged the larger steps as good, but validation accuracy
  did not improve over fixed Adam-direction. This is an important caveat:
  same-batch actual-vs-predicted progress can be locally valid while still not
  optimizing generalization best.
- `controlled_ema` and `controlled_ema_trust` were identical here, so this
  setting did not meaningfully test trust-region behavior.
- Next Adam CIFAR tuning ideas:
  - try a lower alpha cap or slower schedule;
  - add automatic best-epoch summaries;
  - consider validation-aware best-checkpoint reporting;
  - eventually run full CIFAR only with progress logging/checkpointing enabled.

## Muon Multi-Seed Timing And Overhead Discussion

The user asked for a 5-seed, 20-epoch Fashion-MNIST benchmark for the Muon
variants. We ran the diagnostic on a 1024/512 subset with seeds 123, 456, 789,
2024, and 2025:

```text
controlled_muon_project/outputs/fashion_mnist_muon_multiseed_20epoch_5seeds_1k
```

Aggregate result:

```text
vanilla_muon          final 0.6051 +/- 0.0181  best 0.6051 +/- 0.0181
fixed_muon_direction final 0.6051 +/- 0.0181  best 0.6051 +/- 0.0181
controlled_raw_rho   final 0.7289 +/- 0.0070  best 0.7289 +/- 0.0070
controlled_ema       final 0.7293 +/- 0.0067  best 0.7293 +/- 0.0067
controlled_ema_trust final 0.7293 +/- 0.0067  best 0.7293 +/- 0.0067
```

We then discussed whether the extra same-minibatch trial evaluation makes the
controlled optimizer too expensive. The answer was nuanced:

- The controlled variants add one extra forward loss evaluation after the trial
  step on the same minibatch.
- They do not add a second backward pass.
- Since backward is usually more expensive than forward in deep learning, the
  expected overhead is moderate rather than a full 2x.
- This can be manageable in large training if the controller reaches a target
  loss/accuracy in fewer steps.
- It is still not free, so serious claims should compare loss and accuracy
  versus wall-clock time, not only versus epochs or optimizer steps.

On the small five-seed CPU diagnostic, elapsed times were noisy and comparable
across vanilla and controlled Muon variants, so this run supports the idea that
the overhead is manageable but is not a precise timing benchmark.

## Function-Optimization Manager Reports And Rastrigin Basin Discussion

The user wanted a longer version of the trimmed manager function report. We
added a `--step-multiplier` CLI option to both Adam and Muon function report
runners, then generated 3x and 10x versions of the three-function trimmed
reports.

The 10x Adam result changed the interpretation: vanilla Adam can catch up on
some objectives when given many more iterations. That makes the honest manager
claim narrower and stronger: controlled Adam often improves early/local
progress and step-size robustness, but it is not guaranteed to dominate every
eventual residual after very long local optimization.

The user then asked about Rosenbrock, Himmelblau, and Rastrigin. We generated
six-function 10x reports including those functions. Himmelblau was a clear
controlled-Adam speed example: all Adam variants succeeded, but controlled Adam
reached the success criterion much earlier. Rosenbrock showed a speed-versus-
eventual-success tradeoff. Rastrigin showed the expected limitation: local
optimizers generally settle into local basins rather than reliably finding the
true global minimum.

To aggregate over more initial conditions, we added
`--random-starts-per-objective` and `--random-seed` to the function report
runners and generated:

```text
controlled_adam_project/outputs/function_report_manager_extended_10x_15starts
controlled_adam_project/outputs/function_report_manager_extended_10x_30starts
controlled_adam_project/outputs/function_report_manager_extended_10x_60starts
controlled_adam_project/outputs/function_report_manager_extended_60starts_default_steps
```

These broader-start reports gave the clearest version of the budget story. With
normal/default iteration budgets and 60 starts, controlled Adam looks strong as
a practical fixed-budget optimizer: Beale success is `33-37%` for controlled
variants versus `5%` for vanilla, Rosenbrock is `28-32%` versus `8%`,
Himmelblau is `97%` versus `92%`, and vanilla Adam has `0%` success on the
ill-conditioned Quadratic within 300 steps. With 10x iteration budgets, vanilla
Adam catches up on several functions, so the manager interpretation should be:
controlled Adam gives faster useful progress under a limited budget; it is not
a guarantee of best eventual performance after very long local optimization.

Himmelblau remains the cleanest manager example. Rosenbrock is a good
speed-versus-eventual-success example. Beale and Goldstein-Price are
budget-sensitive: they favor controlled Adam under the normal budget, but the
10x run is more nuanced.

Finally, the user asked whether changing Rastrigin initial points was a good
idea. We agreed that it is the right test, but should be framed as a
basin-of-attraction benchmark rather than as cherry-picked starts. We added:

```text
controlled_adam_project/examples/run_rastrigin_basin_benchmark.py
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts
```

The benchmark samples 30 starts per radius from `Uniform([-r, r]^2)` around the
true minimizer `(0, 0)`. All methods succeed reliably up to radius `0.5`;
success falls to about `57%` at radius `0.75`, about `23%` at radius `1.0`,
and `0%` at radius `4.0`. Controlled Adam usually reaches success faster inside
the correct basin, but it does not solve global basin selection. The important
nuance for future reporting is: controlled Adam improves local convergence
speed inside the right basin; Rastrigin from far-away starts needs multi-start
or global exploration.
