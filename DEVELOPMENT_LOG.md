# Development Log

Last updated: 2026-05-20

This is the chronological engineering log for `adaptive_stepsize_control`.

Use the project documents this way:

- `CONVERSATION_LOG.md`: nuanced discussion history, reasoning, interpretation, and collaboration memory.
- `DEVELOPMENT_LOG.md`: concise technical timeline of what changed, what commands were run, and what results were observed.
- `PROJECT_HANDOFF.md`: current operating manual for another machine or coding agent.

## 1. Root Adaptive Step-Size Project

Initial project:

- Implemented a feedback-controlled gradient descent demo.
- Compared fixed-step gradient descent with controlled gradient descent.
- Used actual-versus-predicted decrease ratio:

```math
\rho_t = \frac{f(x_t) - f(x_{t+1})}{\eta_t \|\nabla f(x_t)\|^2}
```

Implemented root modules:

- `src/adaptive_stepsize_control/objectives.py`
- `src/adaptive_stepsize_control/optimizers.py`
- `src/adaptive_stepsize_control/plotting.py`
- `examples/run_quadratic_demo.py`
- `tests/test_quadratic_demo.py`

Added and refined:

- stochastic gradient descent comparison
- additional benchmark functions
- objective landscape trajectory plots
- fixed step-size curves in step-size plots
- consistent initial-state recording so all trajectories visibly start from the same initial position

Root benchmark objectives:

- quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale

Verification:

```bash
PYTHONPATH=src pytest -q tests
```

Latest known result:

```text
4 passed
```

## 2. Controlled Adam Subproject

Created `controlled_adam_project` to apply the same controller idea around Adam.

Core concept:

- Adam supplies the preconditioned direction:

```math
p_t = -\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

- The controller chooses scalar `alpha_t`.
- Trial step:

```math
\theta_{t+1}^{trial} = \theta_t + \alpha_t p_t
```

- Predicted decrease:

```math
-\alpha_t \nabla f(\theta_t)^T p_t
```

Implemented:

- `controlled_adam_project/src/controlled_adam/objectives.py`
- `controlled_adam_project/src/controlled_adam/optimizers.py`
- `controlled_adam_project/src/controlled_adam/plotting.py`
- `controlled_adam_project/examples/run_demo.py`
- tests for deterministic objectives and optimizer behavior

Added 2D objectives:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

Plotting improvements:

- objective values
- alpha curves for fixed Adam and controlled Adam
- rho ratio curves
- trajectory over filled objective landscape with contour lines and known minima

## 3. PyTorch Controlled Adam

Implemented `TorchControlledAdam` in:

```text
controlled_adam_project/src/controlled_adam/torch_optimizers.py
```

Important design decision:

- For minibatch training, evaluate `loss_after` on the same minibatch used to compute `loss_before` and gradients.
- This avoids confusing minibatch variation with optimizer progress.

Implemented features:

- same-minibatch trial loss closure
- bad-step rejection
- limited backtracking
- non-descent shrink
- alpha bounds
- EMA-smoothed rho
- clipped multiplicative alpha updates
- trust-region style recovery
- diagnostics for `alpha_next`, `alpha_update_factor`, and `trust_region_expanded`

PyTorch benchmark runner:

```text
controlled_adam_project/examples/run_mnist_demo.py
```

Supported datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10

Supported models:

- `SmallMLP` for MNIST/Fashion-MNIST
- `SmallCIFARCNN` for CIFAR-10

Supported variants:

- `vanilla_adam`
- `fixed_adam_direction`
- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

Added metadata output:

- `run_metadata.json`
- `run_metadata.txt`

These include dataset, transforms, model architecture, parameter count, optimizer settings, seed, device, and training configuration.

The neural benchmark epoch metrics CSV records both training and test loss.
The plotting code now writes:

- `*_loss.png` for test/validation loss
- `*_train_loss.png` for training loss
- `*_train_test_loss.png` for overlaid train/test loss

Verification:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Latest known result:

```text
9 passed
```

## 4. Fashion-MNIST Adam Experiments

Validated local Fashion-MNIST files under:

```text
fashion/
```

Files:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Confirmed local IDX format and dataset sizes:

- 60,000 training samples
- 10,000 test samples
- 28x28 grayscale images

Compared against user-supplied artifacts:

- `fashionmnist_20epoch_metrics.summary.csv`
- `fashionmnist_20epoch_metrics.png`

Observation:

- The other project's apparent 90% number was training accuracy, not validation/test accuracy.
- Our Adam baseline was comparable on test accuracy.

Key Adam findings:

- Fixed Adam direction nearly matched PyTorch Adam, so the direction implementation was healthy.
- Raw rho and EMA control often shrank alpha too far and undertrained.
- Trust-region style recovery improved controlled Adam by avoiding tiny-alpha collapse.
- Tuned trust settings produced strong Fashion-MNIST subset results.

Notable Fashion-MNIST tuned result:

```text
controlled_ema_trust: test_acc 0.8467
vanilla_adam:         test_acc 0.8350
fixed_adam_direction:test_acc 0.8389
```

Interpretation:

- Controller conservatism was the main weakness, not the Adam direction.
- Preventing alpha collapse improved controlled variants substantially.

## 5. CIFAR-10 Adam Experiments

Added CIFAR-10 support to the Adam benchmark runner.

Dataset handling:

- CIFAR-10 was downloaded/repaired locally under:

```text
controlled_adam_project/data/cifar-10-batches-py/
```

Initial CIFAR CNN was too small and produced weak results.

Replaced it with `SmallCIFARCNN`:

- Conv-BN-ReLU blocks:
  - `3 -> 32 -> 32`
  - `32 -> 64 -> 64`
  - `64 -> 128 -> 128`
- MaxPool after each block
- classifier `128*4*4 -> 256 -> 10`
- BatchNorm uses `track_running_stats=False`
- about `815,018` trainable parameters

CIFAR transforms:

- training: random crop, horizontal flip, tensor conversion, normalization
- evaluation: deterministic tensor conversion and normalization

Best Adam CIFAR result so far:

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

- Controlled Adam needed an alpha cap near Adam's useful scale.
- Too-large alpha caps let controlled variants grow too aggressive.
- `controlled_ema_trust` had the best final test accuracy in the strongest CIFAR run.
- `controlled_raw_rho` had the best peak accuracy.

Larger CIFAR-10 run with progress/checkpoints:

Implemented per-epoch progress printing and checkpointing in
`controlled_adam_project/examples/run_mnist_demo.py`.

New flags:

```text
--print-every N
--checkpoint-every N
```

Smoke validation:

```text
controlled_adam_project: 9 tests passed
tiny CIFAR checkpoint/progress smoke run completed successfully
```

Larger run command:

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

Diagnostics:

```text
controlled_raw_rho:   final mean_alpha 1.5e-3, accepted 100%
controlled_ema:       final mean_alpha 1.5e-3, accepted 100%
controlled_ema_trust: final mean_alpha 1.5e-3, accepted 100%
```

Interpretation:

- Larger-data CIFAR performance is much more plausible than the earlier 5000/1000 subset runs.
- Fixed Adam-direction had the best peak test accuracy on this 20k/5k run.
- The controlled variants quickly saturated the `1.5e-3` alpha cap and accepted all steps.
- EMA and EMA-trust were identical in this setting, so trust-region expansion did not produce distinct behavior.
- The next Adam CIFAR question is whether a lower cap, schedule, or validation-aware best-checkpoint reporting improves generalization.

## 6. Git Repository Setup

Initialized and pushed the repository to:

```text
https://github.com/honghaoyu12/adaptive_stepsize_control.git
```

Recent pushed commit before the documentation updates:

```text
20d1850 Add controlled Muon benchmarks and project handoff
```

The root `.gitignore` excludes:

- generated output folders
- local datasets
- local virtual environments
- cache files
- CIFAR tarballs

Two user-supplied Fashion-MNIST comparison artifacts remain untracked by design:

- `fashionmnist_20epoch_metrics.png`
- `fashionmnist_20epoch_metrics.summary.csv`

## 7. Controlled Muon Subproject

Added `controlled_muon_project`, the Muon version of the Adam subproject.

Initial Muon project contained:

- matrix quadratic objective
- NumPy Muon optimizer
- orthogonalization utilities
- matrix quadratic demo
- basic tests

Then expanded it to match the Adam subproject's current benchmark surface.

Implemented or updated:

- `controlled_muon_project/src/controlled_muon/objectives.py`
- `controlled_muon_project/src/controlled_muon/optimizers.py`
- `controlled_muon_project/src/controlled_muon/orthogonalization.py`
- `controlled_muon_project/src/controlled_muon/torch_optimizers.py`
- `controlled_muon_project/src/controlled_muon/plotting.py`
- `controlled_muon_project/examples/run_matrix_quadratic_demo.py`
- `controlled_muon_project/examples/run_mnist_demo.py`
- `controlled_muon_project/tests/test_optimizers.py`
- `controlled_muon_project/README.md`
- `controlled_muon_project/pyproject.toml`
- `controlled_muon_project/requirements.txt`

Kept backward compatibility:

- Restored and retained `MatrixQuadraticObjective`.
- Existing matrix-quadratic tests still pass.

Added 2D objective parity with Adam:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

Added PyTorch Muon optimizer:

```text
controlled_muon_project/src/controlled_muon/torch_optimizers.py
```

Tensor handling:

- 1D tensors reshape to `(-1, 1)`
- 2D tensors are orthogonalized directly
- convolution kernels and higher-rank tensors reshape to `(out_channels, -1)`
- orthogonalized updates reshape back to original tensor shape

Important caveat:

- The current implementation uses NumPy/CPU orthogonalization.
- It is useful for research demos but slow for CIFAR.

Verification:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Latest known result:

```text
6 passed
```

## 8. Fashion-MNIST Muon Benchmark

Command:

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

Result:

```text
vanilla_muon          final 0.7432  best 0.7432 at epoch 20
fixed_muon_direction final 0.7432  best 0.7432 at epoch 20
controlled_raw_rho   final 0.8320  best 0.8418 at epoch 14
controlled_ema       final 0.8301  best 0.8389 at epoch 18
controlled_ema_trust final 0.8301  best 0.8389 at epoch 18
```

Interpretation:

- Controlled Muon was clearly better than fixed/vanilla Muon on this Fashion-MNIST subset.
- Raw rho had the best final and peak accuracy in this run.
- EMA and EMA-trust were identical in this run.

## 9. CIFAR-10 Muon Benchmarks

10-epoch command:

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

10-epoch result:

```text
vanilla_muon          final 0.578  best 0.578 at epoch 10
fixed_muon_direction final 0.582  best 0.582 at epoch 10
controlled_raw_rho   final 0.665  best 0.665 at epoch 10
controlled_ema       final 0.644  best 0.661 at epoch 9
controlled_ema_trust final 0.644  best 0.661 at epoch 9
```

40-epoch command:

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

40-epoch result:

```text
vanilla_muon          final 0.699  best 0.709 at epoch 31
fixed_muon_direction final 0.694  best 0.701 at epoch 38
controlled_raw_rho   final 0.725  best 0.731 at epoch 27
controlled_ema       final 0.725  best 0.725 at epoch 40
controlled_ema_trust final 0.725  best 0.725 at epoch 40
```

Controller diagnostics:

```text
controlled_raw_rho:   alpha_final ~0.04965, alpha_max 0.05, accepted 100%
controlled_ema:       alpha_final ~0.04507, alpha_max 0.05, accepted 100%
controlled_ema_trust: alpha_final ~0.04507, alpha_max 0.05, accepted 100%
fixed_muon_direction: alpha fixed 0.001, accepted 100%
```

Interpretation:

- Controlled Muon beat vanilla/fixed Muon on CIFAR-10 subset runs.
- Raw rho had the best peak accuracy.
- EMA and EMA-trust tied for best final accuracy.
- Trust-region expansion did not fire in the 40-epoch CIFAR Muon run.
- The 40-epoch run was slow and silent until completion, which motivated a recommended next step: add progress logging and incremental metrics flushing.

## 10. Documentation Work

Added:

- `PROJECT_HANDOFF.md`
- this `DEVELOPMENT_LOG.md`

Updated:

- root `README.md`
- `CONVERSATION_LOG.md`
- `controlled_muon_project/README.md`
- `controlled_adam_project/README.md`

Document roles:

- `CONVERSATION_LOG.md`: nuanced conversation memory and discussion context.
- `DEVELOPMENT_LOG.md`: engineering chronology and benchmark timeline.
- `PROJECT_HANDOFF.md`: current state, commands, caveats, and next steps for another agent.

## 11. Recommended Next Engineering Steps

1. Add per-epoch progress logging to the Muon image benchmark runner. Adam now has `--print-every` and `--checkpoint-every`.

2. Flush epoch metrics incrementally so long runs produce visible progress and partial results.

3. Add an automatic best-epoch summary file.

4. Consider a faster torch-native Muon implementation that avoids NumPy round trips.

5. Tune Muon controller parameters separately from Adam.

6. Run larger/full-dataset experiments only after progress logging is in place.

7. Keep preserving the ablation structure:
   - base optimizer
   - fixed direction
   - raw rho control
   - EMA rho control
   - EMA plus trust/recovery control
