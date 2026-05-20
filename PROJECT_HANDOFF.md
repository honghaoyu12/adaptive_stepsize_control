# Project Handoff: Adaptive Step-Size Control

Last updated: 2026-05-20

This document is for a future coding agent taking over the workspace. It is intentionally comprehensive and operational: it explains what the project is, what has been implemented, how to run it, what benchmark results are already known, and what caveats matter.

Related project-memory documents:

- `CONVERSATION_LOG.md`: nuanced discussion history and interpretation.
- `DEVELOPMENT_LOG.md`: chronological engineering and benchmark timeline.
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

The workspace contains three related projects:

```text
adaptive_stepsize_control/
├── src/adaptive_stepsize_control/       # root gradient-descent controller demo
├── controlled_adam_project/             # Adam direction + outer controller
├── controlled_muon_project/             # Muon direction + outer controller
├── examples/                            # root project examples
├── tests/                               # root project tests
├── fashion/                             # local Fashion-MNIST IDX gzip files
├── mnist/                               # local MNIST files
├── CONVERSATION_LOG.md                  # nuanced discussion history
├── DEVELOPMENT_LOG.md                   # engineering timeline
└── PROJECT_HANDOFF.md                   # this file
```

All projects use a `src/` layout. Either install editable with `pip install -e .`, or run with `PYTHONPATH=src`.

Generated outputs and local datasets are intentionally ignored by git.

## Current Git State

The Muon benchmark and handoff work was committed and pushed in:

```text
20d1850 Add controlled Muon benchmarks and project handoff
```

After that commit, additional documentation updates were made to:

- `README.md`
- `CONVERSATION_LOG.md`
- `PROJECT_HANDOFF.md`
- `DEVELOPMENT_LOG.md`
- `controlled_muon_project/README.md`

After the documentation commit, the Adam CIFAR runner was updated with
per-epoch progress printing and checkpointing, and a larger 20k/5k CIFAR-10
ablation was run. Those changes may be uncommitted depending on when this
handoff is read.

There are also two user-provided/untracked comparison files at repo root:

- `fashionmnist_20epoch_metrics.summary.csv`
- `fashionmnist_20epoch_metrics.png`

Do not delete user-added files unless explicitly asked.

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
- epoch metrics CSV with train loss, train accuracy, test loss, and test accuracy
- step diagnostics CSVs
- test loss plot
- train loss plot
- train-vs-test loss plot
- accuracy plot
- controlled alpha plot
- controlled rho plot
- optional per-epoch checkpoints when `--checkpoint-every N` is set

The Adam image runner also supports `--print-every N` for live epoch summaries.

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
M_t = momentum * M_{t-1} + G_t
B_t = momentum * M_t + G_t     # if Nesterov
O_t = orthogonalize(B_t)
P_t = -update_scale * O_t
```

Orthogonalization methods:

- exact SVD polar factor
- Newton-Schulz polar iteration

PyTorch tensor handling:

- 1D parameters are reshaped to `(-1, 1)`
- 2D parameters are kept 2D
- conv kernels and higher-rank tensors are flattened to `(out_channels, -1)`
- orthogonalized matrices are reshaped back to original tensor shape

This is educational and CPU-heavy, not a production Muon implementation. The CIFAR runs are slower than Adam because each controlled variant performs orthogonalization and same-minibatch trial loss evaluations.

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
6 passed
```

Run function/objective demo:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_matrix_quadratic_demo.py
```

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

Warning: the 40-epoch Muon CIFAR run is slow and currently silent until the end. It took long enough that it was mistaken for being stuck, but it did eventually finish and write metrics/plots. A next agent should add per-epoch progress logging and incremental CSV flushing.

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

2. CIFAR Muon is slow.
   The current PyTorch Muon implementation orthogonalizes tensors through NumPy/CPU. It is acceptable for research demos but not efficient. Add progress logging before running more long Muon jobs.

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

1. Add per-epoch progress printing and incremental CSV flushing to the Muon `run_mnist_demo.py`. Adam now has per-epoch printing and checkpointing, but still does not flush epoch metrics incrementally during a run.

2. Add automatic best-epoch summary CSV/JSON so users do not need ad hoc parsing.

3. Commit the current handoff and Muon updates once reviewed.

4. Consider a faster torch-native Muon implementation:
   - avoid NumPy round-trips where possible
   - use batched or GPU-compatible orthogonalization
   - treat bias/BatchNorm parameters separately if needed

5. Tune Muon controller settings separately from Adam.
   The default `alpha_max=0.05` worked surprisingly well on Muon CIFAR subset runs, but should be stress-tested.

6. Run full-dataset benchmarks only after progress logging is added.

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
6 passed
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
