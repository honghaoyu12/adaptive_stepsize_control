# Project Handoff: Adaptive Step-Size Control

Last updated: 2026-05-26

This document is for a future coding agent taking over the workspace. It is intentionally comprehensive and operational: it explains what the project is, what has been implemented, how to run it, what benchmark results are already known, and what caveats matter.

Related project-memory documents:

- `CONVERSATION_LOG.md`: nuanced discussion history and interpretation.
- `DEVELOPMENT_LOG.md`: chronological engineering and benchmark timeline.
- `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`: self-contained deterministic 2D
  function benchmark suite and manager-report guide.
- `OPTIMIZER_VARIANTS_BENCHMARK_REPORT.md`: comprehensive five-variant optimizer comparison and benchmark interpretation.
- `PROJECT_HANDOFF.md`: current-state operating manual.

## High-Level Goal

The project studies outer-loop step-size controllers for optimization. The central idea is:

1. Let a base optimizer choose a search direction.
2. Take a trial step using a scalar global multiplier.
3. Compare actual objective decrease with first-order predicted decrease.
4. Update the global multiplier using that ratio.

For a direction `p_t` and scalar step `alpha_t`:

```math
\Delta \hat f_t = -\alpha_t \nabla f(\theta_t)^T p_t
```

```math
\Delta f_t = f(\theta_t) - f(\theta_t + \alpha_t p_t)
```

```math
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t}
```

The controller updates the global multiplier roughly as:

```math
\alpha_{t+1} = \alpha_t \exp(K_p(\rho_t - \rho^\star))
```

For minibatch neural-network training, the trial loss must be evaluated on the same minibatch used to compute the gradient:

```math
f_{B_t}(\theta_t)
\quad\text{and}\quad
f_{B_t}(\theta_t + \alpha_t p_t)
```

Using a different minibatch would confuse minibatch noise with true optimization progress.

## Repository Layout

The workspace contains five related project areas:

```text
adaptive_stepsize_control/
├── src/adaptive_stepsize_control/       # root gradient-descent controller demo
├── controlled_adam_project/             # Adam direction + outer controller
├── controlled_muon_project/             # Muon direction + outer controller
├── pi_adam_optimizer/                   # standalone PI-controlled Adam
├── pi_muon_optimizer/                   # standalone PI-controlled Muon
├── examples/                            # root project examples
├── tests/                               # root project tests
├── fashion/                             # local Fashion-MNIST IDX gzip files
├── mnist/                               # local MNIST files
├── CONVERSATION_LOG.md                  # nuanced discussion history
├── DEVELOPMENT_LOG.md                   # engineering timeline
└── PROJECT_HANDOFF.md                   # this file
```

The root, controlled Adam, and controlled Muon projects use a `src/` layout.
The PI optimizer folders are standalone module/demo folders. Either install the
`src/` projects editable with `pip install -e .`, or run with `PYTHONPATH=src`
where needed.

Generated outputs and local datasets are intentionally ignored by git.

Output cleanup note: existing experiment results were moved out of the active
top-level output locations on 2026-05-26. The current active output directories
contain only `.gitkeep` and the backup folder:

```text
outputs/backup_20260526_182414/
controlled_adam_project/outputs/backup_20260526_182414/
controlled_muon_project/outputs/backup_20260526_182414/
```

New experiments should write directly under `outputs/`,
`controlled_adam_project/outputs/`, or `controlled_muon_project/outputs/` with
fresh descriptive folder names. Treat the backup folders as archival.

## Current Git State

The latest project update was committed locally in:

```text
9c55036 Add PI optimizers and align Muon implementation
```

That commit added the PI Adam and PI Muon subprojects, aligned the neural Muon
implementations with official `torch.optim.Muon` behavior, updated the audit
and project-memory documents, and archived older experimental outputs. At the
time of this handoff, the only expected untracked files are generated artifacts
kept out of git by design:

```text
fashionmnist_20epoch_metrics.png
fashionmnist_20epoch_metrics.summary.csv
scheduled_iterate_muon_academic_report.pdf
```

Historical CIFAR-10 Adam tuning sweeps recorded in this handoff:

- balanced cap run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.80`
  - `rho_beta=0.90`
  - best controlled raw-rho test accuracy: `0.8232`
- open cap run:
  - `alpha_max=2e-3`
  - `rho_star=0.78`
  - `rho_beta=0.90`
  - best controlled raw-rho test accuracy: `0.8154`

- follow-up balanced run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.78`
  - `rho_beta=0.90`
  - final/best controlled raw-rho test accuracy: `0.8214`

- faster-EMA balanced run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.80`
  - `rho_beta=0.85`
  - raw-rho best test accuracy: `0.8232`
  - EMA best test accuracy: `0.8164`
  - EMA-trust best test accuracy: `0.8114`

Interpretation:

- the controller was too conservative at `1.25e-3`;
- `1.5e-3` appears to be a better operating point for raw-rho on this CIFAR
  subset;
- `2e-3` was too open and did not improve further;
- EMA variants were stable but behind raw-rho on these runs.
- lowering `rho_beta` to `0.85` made EMA more responsive but did not improve
  the final recommendation.

Architecture-transfer follow-up:

- Added `--model lenet_cifar` to the Adam CIFAR runner.
- Ran a 20-epoch 20k/5k CIFAR-10 LeNet-style ablation with the balanced
  controller settings.
- Final test accuracy:

```text
vanilla_adam          0.5868
fixed_adam_direction 0.5868
controlled_raw_rho   0.5656
controlled_ema       0.5620
controlled_ema_trust 0.5722
```

Interpretation:

- LeNet was fast, about `6-7s` per epoch per variant, but the controller
  underperformed the vanilla/fixed baselines.
- This weakens the hypothesis that the current controller setting transfers
  automatically across architectures.
- A CIFAR ResNet test should be staged carefully on this CPU machine: first a
  3-epoch 5k/1k smoke run, then a 10-epoch 10k/2k medium run, and only then a
  larger run if the timing and early signal look useful.
- A Fashion-MNIST CNN test is probably the cheaper next architecture variation.
- We then ran the Fashion-MNIST CNN benchmark. Raw-rho slightly beat fixed
  Adam-direction on final test accuracy (`0.8870` vs `0.8866`), and the run
  was fast enough to be practical on the current CPU (`4.5-5.6s` per epoch per
  variant).
- This makes Fashion-MNIST CNN a good low-cost architecture-transfer check.
- The five-seed Fashion-MNIST CNN follow-up showed that the single-seed raw-rho
  edge did not hold: vanilla/fixed Adam averaged about `0.8945/0.8946` final
  test accuracy, while controlled raw-rho averaged `0.8889`. Controlled variants
  had about `1.22x-1.24x` relative wall-clock time.
- Recommended next Fashion-MNIST CNN parameter candidates:
  - Candidate A: `alpha_min=9e-4`, `alpha_max=1.5e-3`, `rho_star=0.75`,
    `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`.
  - Candidate B: `alpha_min=8e-4`, `alpha_max=1.5e-3`, `rho_star=0.70`,
    `rho_beta=0.90`, `kp=0.03`, factor clip `[0.98, 1.02]`.
  - Candidate C: `alpha_min=9.5e-4`, `alpha_max=1.25e-3`, `rho_star=0.75`,
    `rho_beta=0.90`, `kp=0.01`, factor clip `[0.995, 1.005]`.
- Candidate A was run on three Fashion-MNIST CNN seeds. It improved controlled
  variants by keeping alpha near `8.9e-4`, but still did not clearly beat fixed
  Adam-direction: three-seed final accuracies were vanilla `0.8946`, fixed
  `0.8960`, raw-rho `0.8944`, EMA `0.8945`, and EMA-trust `0.8943`.

- Candidate C was then run on the same three Fashion-MNIST CNN seeds. It kept
  alpha close to `9.3e-4` and lifted raw-rho slightly above the Candidate A
  controlled results, but still did not beat fixed Adam-direction on mean
  final accuracy: vanilla `0.8946`, fixed `0.8960`, raw-rho `0.8951`, EMA
  `0.8937`, and EMA-trust `0.8937`.

- Candidate B was then run with a faster-recovery setting. It underperformed
  both Candidate A and Candidate C: mean final accuracies were vanilla
  `0.8946`, fixed `0.8960`, raw-rho `0.8921`, EMA `0.8930`, and EMA-trust
  `0.8930`. That suggests the more aggressive controller is too jumpy for this
  Fashion-MNIST CNN setup.

- Added `--model resnet_cifar`, implemented as `SmallCIFARResNet` with
  `175258` trainable parameters.
- Ran the staged CIFAR-10 ResNet smoke test: 5k/1k, 3 epochs, seed `123`,
  full ablation, balanced controller (`alpha_min=1e-3`, `alpha_max=1.5e-3`,
  `rho_star=0.80`, `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`).
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_smoke_5k_1k_3epoch_balanced/`.
- Final test accuracy: vanilla `0.3610`, fixed `0.3730`, raw-rho `0.4100`,
  EMA `0.3800`, EMA-trust `0.3800`.
- The smoke run completed cleanly: all controlled variants accepted every step
  and alpha stayed near `1e-3`.
- Then ran the staged 20-epoch CIFAR-10 ResNet benchmark on 10k train / 2k
  test with the same balanced settings, retaining per-epoch progress prints and
  checkpoints.
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced/`.
- Final / best test accuracy: vanilla Adam `0.6915 / 0.6915`, fixed
  Adam-direction `0.6875 / 0.6975`, raw-rho `0.7395 / 0.7395`, EMA
  `0.7135 / 0.7135`, EMA-trust `0.7135 / 0.7135`.
- Diagnostics: all fixed/controlled variants accepted every step. Raw-rho,
  EMA, and EMA-trust reached `alpha_max=1.5e-3`; final mean rho was about
  `0.88`. EMA and EMA-trust were identical, so trust-region expansion did not
  materially alter this run.
- Later diagnostic check: for the three balanced CIFAR ResNet seeds
  `123/456/789`, `controlled_ema_trust` recorded `0/1580` trust expansions in
  every run. This happened because the run used `alpha_min=1e-3` but
  `trust_region_alpha_threshold=1e-4`; the trust trigger was below the allowed
  alpha floor, so the branch could not activate. Treat these EMA-trust numbers
  as EMA-rho results with a dormant trust hook, not as evidence that the trust
  expansion rule was tested.
- To test trust-region expansion properly in this Adam-scale CIFAR regime, set
  `trust_region_alpha_threshold` near the alpha floor, such as `1e-3` or
  `1.05e-3`, and use a gentle `trust_region_expand_factor` such as `1.1` or
  `1.2`.
- Follow-up Candidate 1 higher-cap test (`alpha_max=1.75e-3`,
  `rho_star=0.82`, `kp=0.015`) was worse: raw-rho final/best `0.7120`,
  EMA/EMA-trust final `0.6695`, best `0.6915`.
- Follow-up stronger fixed-LR control (`lr=1.5e-3`, `alpha_max=1.5e-3`) made
  vanilla Adam stronger (`0.7065`) and EMA/EMA-trust reached `0.7250`, but
  fixed Adam-direction best was only `0.7040` and raw-rho final/best was
  `0.6985`. This suggests the original raw-rho `0.7395` result was not merely
  caused by using a larger fixed Adam learning rate; ramping from `1e-3` to
  the cap seems important.
- Multi-seed validation of the original balanced setting was then run for
  seeds `123`, `456`, and `789`. Three-seed final accuracy means were:
  vanilla `0.6887`, fixed `0.6938`, raw-rho `0.7150`, EMA `0.7083`, and
  EMA-trust `0.7083`. Best accuracy means were: vanilla `0.6998`, fixed
  `0.7118`, raw-rho `0.7235`, EMA `0.7190`, and EMA-trust `0.7190`.
- Interpretation: controlled variants still show a modest mean advantage on
  the ResNet subset benchmark, but the seed `123` raw-rho result was unusually
  strong. Treat this as encouraging evidence, not a settled claim.
- Next staged ResNet step: either run a reduced-variant 5-seed benchmark
  (`vanilla_adam`, `fixed_adam_direction`, `controlled_raw_rho`,
  `controlled_ema`) or move to a larger train/test subset for the same balanced
  setting.

There are also two user-provided/untracked comparison files at repo root:

- `fashionmnist_20epoch_metrics.summary.csv`
- `fashionmnist_20epoch_metrics.png`

Do not delete user-added files unless explicitly asked.

## PI Optimizer Subprojects

Paths:

```text
pi_adam_optimizer/
pi_muon_optimizer/
```

Purpose: standalone PyTorch optimizer versions of the controlled Adam and Muon
ideas with a PI controller instead of the older proportional-only controller.
The PI optimizers preserve the same-batch actual-vs-predicted loss signal and
add a leaky, clipped integral term:

```text
log(alpha_next) = log(alpha_used) + kp * (rho_bar - rho_star) + ki * integral
```

Current implementation state:

- `PIAdam` uses Adam's bias-corrected direction plus a global PI-controlled
  multiplier.
- `PIMuon` uses official-style Muon for 2D hidden matrix parameters and
  AdamW-style fallback directions for all other parameters.
- Both PI optimizers support optional rho EMA smoothing, bad-step rejection,
  bounded backtracking, non-descent fallback, trust-region expansion, and
  decoupled AdamW/Muon-style weight decay.
- The PI Fashion-MNIST runner also has corrected vanilla baselines:
  `vanilla_adam` uses Adam or AdamW depending on weight decay, and
  `vanilla_muon` uses official-style Muon with AdamW fallback.

Important files:

- `pi_adam_optimizer/pi_adam.py`
- `pi_adam_optimizer/README.md`
- `pi_adam_optimizer/PI_ADAM_DESIGN_AND_COMPARISON.md`
- `pi_muon_optimizer/pi_muon.py`
- `pi_muon_optimizer/README.md`
- `pi_muon_optimizer/PI_MUON_DESIGN_AND_COMPARISON.md`
- `examples/run_pi_fashion_mnist_multiseed.py`
- `examples/plot_pi_fashion_mnist_results.py`

Verification already run:

```bash
pytest -q pi_adam_optimizer/test_pi_adam.py pi_muon_optimizer/test_pi_muon.py
python examples/run_pi_fashion_mnist_multiseed.py --output-dir outputs/optimizer_audit_pi_smoke --seeds 101 --optimizers vanilla_adam vanilla_muon pi_adam pi_muon --epochs 1 --train-subset 512 --test-subset 256 --batch-size 128 --alpha0 1e-2 --weight-decay 0.01 --print-every 1
```

Known result:

```text
15 passed
```

## Root Project

Path: repo root.

Purpose: demonstrate fixed gradient descent, stochastic gradient descent, and controlled gradient descent on deterministic 2D functions.

Important files:

- `src/adaptive_stepsize_control/objectives.py`
- `src/adaptive_stepsize_control/optimizers.py`
- `src/adaptive_stepsize_control/plotting.py`
- `examples/run_quadratic_demo.py`
- `examples/run_benchmark_functions.py`
- `tests/test_quadratic_demo.py`

Implemented objectives include:

- quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale

Useful commands:

```bash
PYTHONPATH=src pytest -q
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_quadratic_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_benchmark_functions.py
```

Known behavior:

- Plots compare fixed GD, SGD, and controlled GD.
- Step-size plots include both fixed and adaptive curves.
- Trajectory plots overlay optimizer paths on objective landscapes.
- Optimizer histories include the initial point, so all methods start visibly from the same location.

## Controlled Adam Subproject

Path:

```text
controlled_adam_project/
```

Purpose: compare vanilla Adam with Adam-direction variants controlled by the actual-over-predicted decrease ratio.

Important files:

- `controlled_adam_project/src/controlled_adam/objectives.py`
- `controlled_adam_project/src/controlled_adam/optimizers.py`
- `controlled_adam_project/src/controlled_adam/torch_optimizers.py`
- `controlled_adam_project/src/controlled_adam/plotting.py`
- `controlled_adam_project/examples/run_demo.py`
- `controlled_adam_project/examples/run_function_benchmark_report.py`
- `controlled_adam_project/examples/run_mnist_demo.py`
- `controlled_adam_project/tests/test_optimizers.py`
- `controlled_adam_project/tests/test_torch_optimizers.py`
- `controlled_adam_project/README.md`

### Adam Optimizer Variants

Toy deterministic code:

- `vanilla_adam`
- `controlled_adam`

PyTorch minibatch code:

- `TorchControlledAdam`

The PyTorch runner supports:

- `vanilla_adam`
- `fixed_adam_direction`
- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

`fixed_adam_direction` is important because it isolates whether the Adam-direction implementation itself is healthy. It uses the same Adam-direction machinery but keeps alpha fixed.

### Controlled Adam Features

`TorchControlledAdam` supports:

- same-minibatch trial loss evaluation
- bad-step rejection
- backtracking
- non-descent shrink
- EMA-smoothed rho control
- clipped alpha update factors
- trust-region style alpha recovery
- diagnostics: `alpha_next`, `alpha_update_factor`, `trust_region_expanded`

Trust-region recovery rule:

- If a step is accepted without backtracking,
- and the smoothed rho is high,
- and alpha is tiny,
- then force a larger expansion factor.

This was added because alpha sometimes collapsed to very small values while rho became high, which means the local model was too conservative rather than the optimizer being done.

The "alpha is tiny" threshold must be chosen relative to the configured alpha
floor. If `trust_region_alpha_threshold < alpha_min`, then the trust branch is
effectively unreachable. This exact mismatch occurred in the balanced CIFAR
ResNet Adam runs: `alpha_min=1e-3` and `trust_region_alpha_threshold=1e-4`.

### Adam Objectives

The 2D deterministic benchmark suite includes:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

### Adam Commands

Run tests:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Run function benchmarks:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_demo.py
```

Run the self-contained deterministic function benchmark report:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

This report runner compares vanilla Adam, controlled raw-rho Adam, controlled
EMA-rho Adam, and controlled EMA+trust Adam on nine 2D objectives with five
fixed starts each. It writes CSVs, trajectory plots over landscapes, residual
curves, alpha curves, and a standalone Markdown report at
`controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md`.
The tracked top-level guide is `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`.

For the trimmed manager version:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

The trimmed Adam report currently shows controlled variants beating vanilla on
the chosen residual comparisons, but `controlled_ema_trust` mostly overlaps
`controlled_ema_rho` because the tiny-alpha trust expansion hook does not fire
on those three selected functions.

Recent function-report follow-ups:

- Added `--step-multiplier`, `--random-starts-per-objective`, and
  `--random-seed` to both Adam and Muon function report runners.
- Generated 10x three-function reports:
  - `controlled_adam_project/outputs/function_report_manager_trimmed_10x/`
  - `controlled_muon_project/outputs/function_report_manager_trimmed_10x/`
- Generated six-function 10x reports including Rosenbrock, Himmelblau, and
  Rastrigin:
  - `controlled_adam_project/outputs/function_report_manager_extended_10x/`
  - `controlled_muon_project/outputs/function_report_manager_extended_10x/`
- Generated a broader 15-start Adam aggregate:
  `controlled_adam_project/outputs/function_report_manager_extended_10x_15starts/`.
- Generated larger Adam aggregates:
  - `controlled_adam_project/outputs/function_report_manager_extended_10x_30starts/`
  - `controlled_adam_project/outputs/function_report_manager_extended_10x_60starts/`
  - `controlled_adam_project/outputs/function_report_manager_extended_60starts_default_steps/`

Interpretation for manager-facing function optimization:

- Controlled Adam is strongest as an early/local progress and step-size
  robustness story, not as a guarantee of best eventual residual after very
  long runs.
- The 60-start default-step run is the best manager-facing evidence for fixed
  practical budgets. Within the default steps, controlled variants outperform
  vanilla success rate on Beale, Goldstein-Price, Rosenbrock, Himmelblau, and
  Quadratic.
- Himmelblau is the cleanest extra example: all optimizers succeed, but
  controlled Adam reaches the success criterion much faster. In the 60-start
  default run, controlled variants reach success around `84-88` iterations,
  while vanilla takes about `291`.
- Rosenbrock shows controlled Adam can succeed faster while vanilla Adam can
  eventually catch up or exceed success with enough iterations. In the 60-start
  default run, controlled success is `28-32%` versus vanilla `8%`; in the
  60-start 10x run, vanilla reaches `100%` but takes about `8085` median
  successful iterations versus controlled variants around `3954-5060`.
- Beale and Goldstein-Price become more nuanced under 60-start aggregation:
  controlled variants are clearly better under the default budget, but vanilla
  catches up more under the 10x budget.
- Rastrigin is a limitation case: local step-size control does not solve global
  basin selection.

Focused Rastrigin basin benchmark:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_rastrigin_basin_benchmark.py \
  --output-dir outputs/rastrigin_basin_benchmark_30starts \
  --starts-per-radius 30 \
  --steps 12000
```

This benchmark samples starts from boxes around `(0, 0)` and writes:

```text
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/RASTRIGIN_BASIN_BENCHMARK_REPORT_ZH.md
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/aggregate_results.csv
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/rastrigin_success_rate_by_radius.png
```

Main result: all methods succeed reliably up to radius `0.5`; success falls to
about `57%` at radius `0.75`, `23%` at radius `1.0`, and `0%` at radius `4.0`.
Controlled Adam usually reaches success faster inside the correct basin, but it
does not expand the global basin enough to solve far-away Rastrigin starts.

Run Fashion-MNIST ablation using local IDX files:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 20 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_20epochs_ablation
```

Run CIFAR-10 with tuned cap:

```bash
cd controlled_adam_project
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
  --print-every 1 \
  --checkpoint-every 1 \
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

### Controlled Adam Neural Benchmark Setup

Datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10

Models:

- `SmallMLP` for MNIST/Fashion-MNIST:
  - Flatten
  - Linear `28*28 -> 128`
  - ReLU
  - Linear `128 -> 10`
- `SmallCIFARCNN` for CIFAR-10:
  - Conv-BN-ReLU blocks:
    - `3 -> 32 -> 32`
    - `32 -> 64 -> 64`
    - `64 -> 128 -> 128`
  - MaxPool after each block
  - Linear `128*4*4 -> 256 -> 10`
  - BatchNorm uses `track_running_stats=False` so same-minibatch trial loss does not mutate running statistics.
  - About `815,018` trainable parameters.

CIFAR transforms:

- train: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `ToTensor`, Normalize
- eval: `ToTensor`, Normalize
- train metrics use deterministic eval transform on the same subset indices

The runner writes:

- `run_metadata.json`
- `run_metadata.txt`
- epoch metrics CSV with train loss, train accuracy, test loss, test accuracy, cumulative wall-clock seconds, and cumulative optimizer steps
- step diagnostics CSVs
- test loss plot
- train loss plot
- train-vs-test loss plot
- accuracy plot
- loss/accuracy plots versus optimizer steps
- loss/accuracy plots versus wall-clock time
- controlled alpha plot
- controlled rho plot
- optional per-epoch checkpoints when `--checkpoint-every N` is set

The Adam image runner also supports `--print-every N` for live epoch summaries.

Checkpointed Adam image runs can be post-processed with:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/plot_pca_training_trajectory.py \
  outputs/YOUR_RUN_WITH_CHECKPOINTS \
  --runs vanilla_adam fixed_adam_direction controlled_raw_rho controlled_ema \
  --output-dir outputs/YOUR_RUN_WITH_CHECKPOINTS/pca_trajectory
```

This implements the training-trajectory PCA view used in loss-landscape
visualization work. It flattens trainable parameters from per-epoch checkpoints,
fits a 2D PCA plane to checkpoint displacements, and writes:

- `pca_trajectory_coordinates.csv`
- `pca_explained_variance.csv`
- `pca_training_trajectory.png`

By default, the origin is the final checkpoint of the first selected run. Use
`--reference-run` or `--reference-epoch` to choose a different origin, and use
`--no-center-for-pca` to fit PCA directly on final-relative displacements.

### Important Adam Results

Fashion-MNIST comparison to another project:

- Other project files at repo root:
  - `fashionmnist_20epoch_metrics.summary.csv`
  - `fashionmnist_20epoch_metrics.png`
- Other project Fashion-MNIST Adam:
  - train accuracy about `91.80%`
  - validation accuracy about `83.40%`
- Our comparable 20-epoch Fashion-MNIST Adam:
  - train accuracy about `89.97%`
  - test accuracy about `83.69%`
- Conclusion: our Adam baseline is comparable; the apparent 90% number was training accuracy, not validation/test.

Best controlled Adam CIFAR run so far:

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

Final test accuracy:

```text
vanilla_adam          0.718
fixed_adam_direction 0.717
controlled_raw_rho   0.711
controlled_ema       0.704
controlled_ema_trust 0.731
```

Best test accuracy:

```text
vanilla_adam          0.720 at epoch 34
fixed_adam_direction 0.717 at epoch 40
controlled_raw_rho   0.735 at epoch 38
controlled_ema       0.709 at epoch 33
controlled_ema_trust 0.731 at epoch 40
```

Interpretation:

- Tight alpha cap around Adam scale is important.
- `controlled_ema_trust` achieved the best final CIFAR result so far.
- `controlled_raw_rho` achieved the best peak accuracy.
- Trust expansion rarely fired in this CIFAR run; the benefit mostly came from keeping alpha bounded near a useful scale.

Larger Adam CIFAR-10 run:

Output:

```text
controlled_adam_project/outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Command:

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

Interpretation:

- This 20k/5k result is the most informative Adam CIFAR result so far.
- Larger-data CIFAR accuracy is much more plausible than the earlier 5000/1000 subset numbers.
- Fixed Adam-direction had the best peak test accuracy.
- Controlled variants all accepted 100% of steps and quickly saturated the `1.5e-3` alpha cap.
- EMA and EMA-trust were identical in this run; trust-region behavior was not meaningfully distinguished.

## Controlled Muon Subproject

Path:

```text
controlled_muon_project/
```

Purpose: Muon version of `controlled_adam_project`. Muon supplies an orthogonalized matrix-shaped direction; the outer controller adapts the scalar global multiplier.

Important files:

- `controlled_muon_project/src/controlled_muon/objectives.py`
- `controlled_muon_project/src/controlled_muon/optimizers.py`
- `controlled_muon_project/src/controlled_muon/orthogonalization.py`
- `controlled_muon_project/src/controlled_muon/torch_optimizers.py`
- `controlled_muon_project/src/controlled_muon/plotting.py`
- `controlled_muon_project/examples/run_matrix_quadratic_demo.py`
- `controlled_muon_project/examples/run_mnist_demo.py`
- `controlled_muon_project/tests/test_objectives.py`
- `controlled_muon_project/tests/test_optimizers.py`
- `controlled_muon_project/README.md`

### Muon Core

There are two Muon implementations:

1. NumPy toy optimizer in `optimizers.py`
   - Used for deterministic matrix quadratic and 2D function demos.
   - Original matrix quadratic class remains for backward compatibility.

2. PyTorch minibatch optimizer in `torch_optimizers.py`
   - Used for Fashion-MNIST/CIFAR benchmarks.
   - Implements Muon-style direction with outer-loop controller.

Muon direction:

```text
B_t = lerp(B_{t-1}, G_t, 1 - momentum)
U_t = lerp(G_t, B_t, momentum)     # if Nesterov
O_t = NewtonSchulz_quintic(U_t)
P_t = -shape_scale * update_scale * O_t
```

Orthogonalization methods:

- exact SVD polar factor
- Muon-style quintic Newton-Schulz iteration with PyTorch's default
  coefficients `(3.4445, -4.7750, 2.0315)` and 5 default steps

PyTorch tensor handling:

- the neural-network Muon path follows `torch.optim.Muon` scope
- only 2D hidden matrix parameters use Muon by default
- vectors, scalars, norms, biases, embeddings, heads, and convolution kernels
  use AdamW-style fallback updates
- nonzero weight decay is decoupled, as in AdamW/Muon

This is educational and CPU-heavy, not a production Muon implementation. The
CIFAR runs are slower than Adam because each controlled variant performs
orthogonalization and same-minibatch trial loss evaluations. The deterministic
2D function runner still uses a vector analogue of Muon and should not be read
as a full neural-network `torch.optim.Muon` replacement.

Important benchmark note: older neural Muon tables in this handoff predate the
official-style parameter grouping fix unless an output folder explicitly says
`official_muon`. Treat those old all-parameter neural Muon results as archival
context. Future neural Muon comparisons should use only the corrected
official-style path: Muon for 2D hidden matrix parameters and AdamW-style
fallback for the rest.

### Muon Optimizer Variants

The image runner supports:

- `vanilla_muon`
- `fixed_muon_direction`
- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

`vanilla_muon` and `fixed_muon_direction` should be almost identical in behavior when alpha is fixed. They are both useful sanity checks.

### Muon Objectives

The Muon subproject now supports the same 2D benchmark objectives as Adam:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

It also retains:

- `MatrixQuadraticObjective`

### Muon Commands

Run tests:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Known latest test result:

```text
12 passed
```

Run function/objective demo:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_matrix_quadratic_demo.py
```

Run deterministic 2D function benchmark report:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

This report compares `vanilla_muon`, `controlled_raw_rho`, `controlled_ema`,
and `controlled_ema_trust` on the same nine 2D objectives used by the Adam
report. It writes
`controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md`.
The shorter manager-facing version is
`controlled_muon_project/outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md`.
Both Muon function reports include Chinese companion reports and standalone
`*_surface_3d.png` plots with the objective formula printed inside each figure.
The earlier `fixed_muon_direction` diagnostic was removed from the function
report because it duplicated `vanilla_muon` in this local 2D vector runner:
both used fixed alpha and no rho controller.
For these 2D vector objectives, Muon is implemented as a vector analogue by
treating the momentum vector as a column matrix before orthogonalization. Do
not overclaim it as a full matrix Muon benchmark.

Run Fashion-MNIST Muon ablation:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 20 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_muon_20epoch_ablation
```

Run CIFAR-10 Muon 10-epoch ablation:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --data-dir ../controlled_adam_project/data \
  --epochs 10 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/cifar10_muon_10epoch_ablation
```

Run CIFAR-10 Muon 40-epoch ablation:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --data-dir ../controlled_adam_project/data \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/cifar10_muon_40epoch_ablation
```

Warning: the historical 40-epoch Muon CIFAR run was slow and originally silent
until completion. The current runner now supports `--print-every` per-epoch
progress output, but long Muon jobs are still CPU-heavy because
orthogonalization uses the educational NumPy/CPU path.

### Muon Benchmark Results

Fashion-MNIST, 20 epochs, 4096 train / 1024 test:

Output:

```text
controlled_muon_project/outputs/fashion_mnist_muon_20epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.7432  best 0.7432 at epoch 20
fixed_muon_direction final 0.7432  best 0.7432 at epoch 20
controlled_raw_rho   final 0.8320  best 0.8418 at epoch 14
controlled_ema       final 0.8301  best 0.8389 at epoch 18
controlled_ema_trust final 0.8301  best 0.8389 at epoch 18
```

Interpretation:

- Controlled Muon is clearly better than fixed/vanilla Muon on this subset.
- Raw rho had the best peak and final accuracy in this run.
- EMA and EMA-trust were identical because trust expansion did not materially change behavior.

CIFAR-10, 10 epochs, 5000 train / 1000 test:

Output:

```text
controlled_muon_project/outputs/cifar10_muon_10epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.578  best 0.578 at epoch 10
fixed_muon_direction final 0.582  best 0.582 at epoch 10
controlled_raw_rho   final 0.665  best 0.665 at epoch 10
controlled_ema       final 0.644  best 0.661 at epoch 9
controlled_ema_trust final 0.644  best 0.661 at epoch 9
```

CIFAR-10, 40 epochs, 5000 train / 1000 test:

Output:

```text
controlled_muon_project/outputs/cifar10_muon_40epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.699  best 0.709 at epoch 31
fixed_muon_direction final 0.694  best 0.701 at epoch 38
controlled_raw_rho   final 0.725  best 0.731 at epoch 27
controlled_ema       final 0.725  best 0.725 at epoch 40
controlled_ema_trust final 0.725  best 0.725 at epoch 40
```

Final train accuracy:

```text
vanilla_muon          0.8842
fixed_muon_direction 0.8824
controlled_raw_rho   0.8732
controlled_ema       0.8988
controlled_ema_trust 0.8988
```

Step diagnostics:

```text
controlled_raw_rho:   1600 steps, alpha_final ~0.04965, alpha_max 0.05, accepted 100%
controlled_ema:       1600 steps, alpha_final ~0.04507, alpha_max 0.05, accepted 100%
controlled_ema_trust: 1600 steps, alpha_final ~0.04507, alpha_max 0.05, accepted 100%
fixed_muon_direction: 1600 steps, alpha fixed 0.001, accepted 100%
```

Interpretation:

- Controlled Muon beats vanilla/fixed Muon on CIFAR-10 subset runs.
- Raw rho achieved the best peak accuracy.
- EMA and EMA-trust achieved the best final tie.
- Trust-region expansion did not fire in the 40-epoch CIFAR Muon run.
- Alpha for controlled Muon grew close to the default cap `0.05`.

## Dataset Notes

Local Fashion-MNIST:

```text
fashion/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

These were validated as 60,000 train images/labels and 10,000 test images/labels.

Local CIFAR-10:

```text
controlled_adam_project/data/cifar-10-batches-py/
```

The Muon project currently reuses that path via:

```bash
--data-dir ../controlled_adam_project/data
```

On another machine, either copy this folder or run the Adam/Muon runner with `--download` if network access works.

## Known Caveats And Risks

1. Same-minibatch trial loss is mandatory.
   Do not change controlled Adam/Muon to evaluate `f(theta_{t+1})` on a fresh minibatch.
   The controlled variants intentionally add one extra forward loss evaluation
   on that same minibatch, but not an extra backward pass. Treat overhead claims
   through loss/accuracy versus wall-clock time, because the practical cost
   depends on the model, hardware, and data pipeline.

2. CIFAR Muon is slow.
   The current PyTorch Muon implementation orthogonalizes tensors through
   NumPy/CPU. It is acceptable for research demos but not efficient. Use
   `--print-every` for live progress, and add checkpointing or incremental CSV
   flushing before running more long Muon jobs.

3. Muon dependency metadata was recently updated.
   `controlled_muon_project/pyproject.toml` and `requirements.txt` now include `torch`, `torchvision`, and `scikit-learn`.

4. Generated outputs are ignored by git.
   If a future machine needs benchmark outputs, copy them explicitly or rerun commands.

5. There may be unrelated local/user files.
   Do not delete root `fashionmnist_20epoch_metrics.*` files or dataset folders unless the user explicitly asks.

6. Current benchmark subset results are deterministic but subset-limited.
   Accuracy numbers are for 4096/1024 Fashion-MNIST and 5000/1000 CIFAR-10 unless stated otherwise. The larger Adam CIFAR run uses 20000/5000. These are not full-dataset claims.

7. The Adam and Muon projects have similar but not identical control behavior.
   Adam needs a tight alpha cap around Adam scale for CIFAR; Muon naturally grew to much larger alpha values and still improved over fixed Muon in the subset runs.

## Recommended Next Steps For The Next Agent

1. Consider extending the PCA trajectory post-processor with optional loss
   contour evaluation on the same PCA plane. The current implementation plots
   trajectories only; contour evaluation would require rebuilding the dataset
   and running many forward passes.

2. Add checkpointing and incremental epoch-metrics flushing to the Muon
   `run_mnist_demo.py`. Per-epoch progress printing already exists through
   `--print-every`, but long jobs would still benefit from partial CSV output
   and resumable checkpoints.

3. Add automatic best-epoch summary CSV/JSON so users do not need ad hoc parsing.

4. Consider a faster torch-native Muon implementation:
   - avoid NumPy round-trips where possible
   - use batched or GPU-compatible orthogonalization
   - treat bias/BatchNorm parameters separately if needed

5. Tune Muon controller settings separately from Adam.
   The default `alpha_max=0.05` worked surprisingly well on Muon CIFAR subset runs, but should be stress-tested.

6. Run full-dataset benchmarks only after checkpointing or incremental metrics
   writing is added.

7. Consider early stopping or best-checkpoint reporting.
   In long runs, peak accuracy can occur before the final epoch.

## Quick Verification Commands

Root project:

```bash
PYTHONPATH=src pytest -q
```

Controlled Adam:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Controlled Muon:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Latest known Muon test result:

```text
12 passed
```

## Important Philosophy For Future Changes

The purpose is not simply to beat Adam or Muon in every setting. The purpose is to understand whether an objective-feedback controller can safely and usefully adapt the global multiplier on top of a strong optimizer direction.

When adding experiments, preserve these comparisons:

- base optimizer
- fixed direction with fixed alpha
- raw rho controller
- EMA rho controller
- EMA plus trust/recovery controller

This separation is what lets the user distinguish direction quality from controller quality.
