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
├── pi_muon_optimizer/
├── delayed_feedback_adam/
└── delayed_feedback_muon/
```

The root project is the original gradient-descent demo. The controlled
subprojects apply the same actual-versus-predicted decrease controller to
stronger optimizer directions, while the PI folders package the integral
controller variants as standalone PyTorch optimizers. The delayed-feedback
folders test a lower-overhead controller variant that uses the next naturally
computed training loss instead of an immediate trial-point reevaluation:

- `controlled_adam_project/`: Adam chooses the direction; the controller chooses
  the global multiplier. Its image benchmark checkpoints can now be
  post-processed into PCA training-trajectory plots.
- `controlled_muon_project/`: Muon-style orthogonalization chooses the
  matrix-shaped direction; the controller chooses the global multiplier.
- `pi_adam_optimizer/` and `pi_muon_optimizer/`: standalone PI-controller
  versions of the Adam and Muon controllers, packaged as PyTorch optimizers.
  PI Muon follows official `torch.optim.Muon` neural-network scope: Muon for
  2D hidden matrix parameters and AdamW-style fallback for the rest.
- `delayed_feedback_adam/` and `delayed_feedback_muon/`: standalone PyTorch
  optimizers that adapt the same global multiplier from delayed
  actual-versus-predicted feedback. They avoid the extra same-minibatch forward
  pass, but they cannot reject a bad step before it happens and their minibatch
  feedback is noisier.

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

Recent controlled-Adam function tuning shows that the current report defaults
are not the best available parameters for the deterministic suite. A refined
30-start sweep over raw-rho, EMA-rho, and EMA+trust variants is stored at:

```text
outputs/controlled_adam_refined_tuning_sweep_30runs/
```

The best tuned average success rates across nine functions were:

```text
raw-rho current 0.370 -> tuned 0.437
EMA-rho current 0.359 -> tuned 0.444
EMA+trust current 0.359 -> tuned 0.489
```

The main lesson is that the existing controlled settings often let alpha
collapse to `1e-8`. Raising `alpha_min` near `1e-5`, using a stronger gain, and
lowering the rho target improved raw-rho and EMA-rho. EMA+trust improved only
after the trust expansion gate was made reachable, with an alpha threshold near
`1e-2`; this confirms that earlier trust comparisons were mostly dormant rather
than evidence against the trust idea. Treat these as deterministic function
benchmark findings, not automatic neural-network defaults.

We also tested a simplified preset interface so the optimizer does not require
many independent knobs. The preset sweep lives at:

```text
outputs/controlled_adam_simplified_tuning_sweep_30runs/
```

It exposes only `family`, `response_preset`, and `alpha_preset`. The best
function-suite preset was `aggressive_high_floor`, which derives alpha bounds
from each objective's base `alpha0`:

```text
raw-rho current 0.370 -> simplified 0.470
EMA-rho current 0.359 -> simplified 0.485
EMA+trust current 0.359 -> simplified 0.504
```

This is a better usability direction than continuing to expose every raw
parameter. The next step is to validate conservative/balanced/aggressive
presets on a neural task before treating any preset as a default.

We also checked the fixed gradient-descent baseline in the tuned no-momentum
function report. The original run used the same `alpha0` as Adam, which is too
large as a raw-gradient learning rate on Goldstein-Price. The tuned report
runner now supports:

```bash
--gradient-descent-alpha-multiplier VALUE
```

This affects only `gradient_descent`; Adam and controlled Adam still use the
original `alpha0`. Full reruns live in:

```text
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p03/
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p05/
```

On Goldstein-Price, `0.03 * alpha0` improved fixed GD success from `0.033` to
`0.367` and reduced the median final residual from `2.64e20` to `27.0`.
Increasing to `0.05 * alpha0` was slightly better overall than `0.03 * alpha0`
but made Goldstein-Price unstable again, with median final residual `6.88e14`.

The image benchmark runner supports long-run visibility with
`--print-every N` and `--checkpoint-every N`.

Checkpointed Adam image runs can be visualized in the PCA trajectory plane used
by loss-landscape papers:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/plot_pca_training_trajectory.py \
  outputs/YOUR_RUN_WITH_CHECKPOINTS \
  --runs vanilla_adam controlled_raw_rho controlled_ema \
  --output-dir outputs/YOUR_RUN_WITH_CHECKPOINTS/pca_trajectory
```

The script projects saved checkpoints using trainable parameters by default and
writes `pca_trajectory_coordinates.csv`, `pca_explained_variance.csv`, and
`pca_training_trajectory.png`.

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

A newer apples-to-apples CIFAR-10 ResNet comparison was run from the
delayed-feedback Adam project so same-step controlled Adam and delayed-feedback
Adam could be compared in one script. It used a 10k/2k CIFAR-10 subset,
20 epochs, batch size 128, seed 123, and base learning rate `1e-3`.

Output:

```text
outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/
```

Key final metrics:

```text
optimizer                 test_acc  test_loss  train_loss
vanilla_adam              0.6915    0.9454     0.7499
controlled_raw_rho        0.7395    0.7715     0.5830
controlled_ema            0.7135    0.8488     0.6656
controlled_ema_trust      0.7135    0.8488     0.6656
delayed_raw               0.6825    0.9768     0.7820
delayed_ema               0.6960    0.9009     0.7334
delayed_safe              0.6985    0.9156     0.7194
delayed_ema_floor90       0.6950    0.8840     0.6783
```

Interpretation:

- Same-step controlled raw-rho was best on this single-seed subset run.
- Same-step EMA improved over vanilla, but less than raw-rho.
- EMA+trust exactly matched EMA because trust expansion fired zero times.
- Delayed Adam variants avoided the extra same-step forward pass, but their
  delayed rho signal was much lower than same-step rho. With same-step-style
  `rho_star` values, they were pushed to their configured alpha floors and only
  matched or slightly beat vanilla Adam.
- The delayed result is not a reason to abandon delayed feedback; it says the
  delayed rho scale needs delayed-specific calibration, especially lower
  `rho_star` and floor/cap settings derived from observed delayed `rho_bar`.

For controlled Adam, fixed-learning-rate Adam baselines are now considered a
required diagnostic, not a dismissal of the controller. If Adam with a larger
fixed learning rate performs well, the question becomes whether the controller
adds useful adaptive behavior beyond choosing a larger cap. Future ResNet
comparisons should include fixed Adam at several learning rates and simple
warmup schedules such as `1e-3 -> 1.5e-3`, then ask whether controlled Adam is
less sensitive to its upper bound than vanilla Adam is to its fixed learning
rate.

Follow-up controlled Adam diagnostics on the same 10k/2k CIFAR-10 ResNet
subset clarified the alpha-control behavior:

- The ResNet used in these runs is the project-specific `SmallCIFARResNet`,
  not torchvision ResNet-18. It has a `3x3` stride-1 CIFAR stem, three
  residual stages with widths `16/32/64`, two basic blocks per stage, adaptive
  average pooling, a `64 -> 10` linear head, and `175258` trainable
  parameters.
- A high-cap backtracking sweep showed that the sharp alpha drops in the
  high-LR plots came from hard backtracking, especially `backtrack_shrink=0.5`.
  Gentler shrink factors smoothed the plot but did not solve high-cap
  performance by themselves.
- A clean cap sweep with fixed `lr=1e-3` found that all tested caps eventually
  saturated under `rho_star=0.80`. Raw-rho was strongest around
  `alpha_max=1.5e-3` to `1.75e-3`; EMA's best single-seed endpoint in that
  sweep was `0.7355` at `alpha_max=2.25e-3`.
- Raising `rho_star` to `0.85` made alpha self-limiting and prevented cap
  saturation, but was too conservative for the 20-epoch run. Adding asymmetric
  downward gain via `--controlled-kp-down 0.08` delayed saturation but did not
  improve accuracy.

The detailed reports are under:

```text
outputs/cifar10_resnet_adam_backtracking_sweep_highlr2e3_cap2p5e3_seed123/
outputs/cifar10_resnet_adam_cap_sweep_lr1e3_seed123/
outputs/cifar10_resnet_adam_rhostar_asym_gain_seed123/
```

The current controlled Adam mechanics are written in paper-style pseudocode in
`controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md`.

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
