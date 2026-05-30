# Development Log

Last updated: 2026-05-30

This is the chronological engineering log for `adaptive_stepsize_control`.

Use the project documents this way:

- `CONVERSATION_LOG.md`: nuanced discussion history, reasoning, interpretation, and collaboration memory.
- `DEVELOPMENT_LOG.md`: concise technical timeline of what changed, what commands were run, and what results were observed.
- `PROJECT_HANDOFF.md`: current operating manual for another machine or coding agent.
- `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`: self-contained deterministic
  2D function benchmark suite and manager-report guide.

## 0. Output Cleanup

On 2026-05-26, existing experimental results were moved into timestamped
backup folders so new runs start from uncluttered output directories:

```text
outputs/backup_20260526_182414/
controlled_adam_project/outputs/backup_20260526_182414/
controlled_muon_project/outputs/backup_20260526_182414/
```

The root and Muon top-level `outputs` directories contain `.gitkeep` plus the
backup folder. The Adam output directory may also contain ignored PCA
smoke/validation outputs from the trajectory-visualization checks; those are
generated artifacts and can be regenerated.

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

The neural benchmark epoch metrics CSV records training loss, test loss,
cumulative wall-clock seconds, and cumulative optimizer steps.
The plotting code now writes:

- `*_loss.png` for test/validation loss
- `*_train_loss.png` for training loss
- `*_train_test_loss.png` for overlaid train/test loss
- `*_loss_vs_steps.png` and `*_accuracy_vs_steps.png`
- `*_loss_vs_time.png` and `*_accuracy_vs_time.png`

Verification:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Latest known result:

```text
9 passed
```

Recent CIFAR-10 Adam tuning sequence:

1. Conservative cap run on 20k/5k CIFAR-10 for 20 epochs:
   - `alpha_max=1.25e-3`
   - `rho_star=0.82`
   - `rho_beta=0.95`
   - Result: stable, but a bit too cautious.
   - Best controlled result: `controlled_raw_rho` at `0.8164`
   - Fixed Adam-direction baseline: `0.8220`

2. Balanced cap run:
   - `alpha_max=1.5e-3`
   - `rho_star=0.80`
   - `rho_beta=0.90`
   - Result: best raw-rho peak improved to `0.8232`, slightly above fixed Adam-direction.

3. Open cap run:
   - `alpha_max=2e-3`
   - `rho_star=0.78`
   - `rho_beta=0.90`
   - Result: did not improve further; later epochs became less efficient.

4. Follow-up balanced run:
   - `alpha_max=1.5e-3`
   - `rho_star=0.78`
   - `rho_beta=0.90`
   - Result: `controlled_raw_rho` reached `0.8214` final test accuracy and
     `0.8214` best test accuracy at epoch 20.

5. Faster-EMA balanced run:
   - `alpha_max=1.5e-3`
   - `rho_star=0.80`
   - `rho_beta=0.85`
   - Result: raw-rho reproduced the earlier `0.8232` best test accuracy, but
     final raw-rho was `0.8138`; EMA reached `0.8110` final / `0.8164` best,
     and EMA-trust reached `0.8094` final / `0.8114` best.

Conclusion:

- Raw-rho control benefited most from opening the cap modestly.
- EMA variants remained stable but lagged raw-rho on this CIFAR setting.
- The trust-region expansion hook did not materially change the tuned CIFAR runs.
- The best current 20-epoch 20k/5k CIFAR setting remains the balanced
  `alpha_max=1.5e-3`, `rho_star=0.80`, `rho_beta=0.90` raw-rho run by peak
  test accuracy.

Architecture-transfer follow-up:

- Added `LeNetCIFAR` model support in `controlled_adam_project/examples/run_mnist_demo.py`
  via `--model lenet_cifar`.
- Ran 20-epoch CIFAR-10, 20k/5k, LeNet-style ablation with the balanced Adam
  controller setting.
- Final test accuracy:

```text
vanilla_adam          0.5868
fixed_adam_direction 0.5868
controlled_raw_rho   0.5656
controlled_ema       0.5620
controlled_ema_trust 0.5722
```

Interpretation:

- LeNet was much faster than `SmallCIFARCNN`, about `6-7s` per epoch per
  variant.
- The controller mostly stayed near the alpha floor and did not improve this
  weaker architecture.
- The current balanced controller setting does not appear architecture-agnostic.
- A CIFAR ResNet run should start with a small smoke test because a full
  20-epoch, 5-variant ablation may take many hours on the current CPU setup.
- A Fashion-MNIST CNN is likely the cheaper next architecture-variation test.

Fashion-MNIST CNN benchmark:

- Added `FashionCNN` model support in `controlled_adam_project/examples/run_mnist_demo.py`
  via `--model fashion_cnn`.
- A single 20-epoch, 20k/5k Fashion-MNIST CNN run gave a tiny raw-rho edge:

```text
vanilla_adam          0.8858
fixed_adam_direction 0.8866
controlled_raw_rho   0.8870
controlled_ema       0.8844
controlled_ema_trust 0.8844
```

- A five-seed follow-up showed that the single-seed edge did not hold:

```text
vanilla_adam          0.8945 +/- 0.0061 final, 0.8962 +/- 0.0041 best
fixed_adam_direction 0.8946 +/- 0.0062 final, 0.8962 +/- 0.0045 best
controlled_raw_rho   0.8889 +/- 0.0071 final, 0.8918 +/- 0.0047 best
controlled_ema       0.8872 +/- 0.0064 final, 0.8912 +/- 0.0038 best
controlled_ema_trust 0.8884 +/- 0.0064 final, 0.8909 +/- 0.0034 best
```

Interpretation:

- Fashion-MNIST CNN is a cheap and useful architecture-transfer testbed.
- The current CIFAR-derived balanced controller setting is too conservative
  for this model: controlled variants ended near `alpha ~= 6.8e-4`, below the
  Adam-scale `1e-3` baseline.
- Controlled variants also had `1.22x-1.24x` relative wall-clock time, so the
  underperformance is not offset by speed.

Recommended Fashion-MNIST CNN parameter candidates:

```text
Candidate A:
alpha_min=9e-4, alpha_max=1.5e-3, rho_star=0.75, rho_beta=0.90,
kp=0.02, min_alpha_factor=0.98, max_alpha_factor=1.015

Candidate B:
alpha_min=8e-4, alpha_max=1.5e-3, rho_star=0.70, rho_beta=0.90,
kp=0.03, min_alpha_factor=0.98, max_alpha_factor=1.02

Candidate C:
alpha_min=9.5e-4, alpha_max=1.25e-3, rho_star=0.75, rho_beta=0.90,
kp=0.01, min_alpha_factor=0.995, max_alpha_factor=1.005
```

Candidate A is the recommended next run because it directly tests whether
preventing alpha from drifting below Adam scale fixes the Fashion-MNIST CNN
underperformance.

Candidate A result:

- Ran Candidate A on seeds `123`, `456`, and `789` for Fashion-MNIST CNN,
  20k/5k, 20 epochs.
- Three-seed final test accuracy:

```text
vanilla_adam          0.8946 +/- 0.0084
fixed_adam_direction 0.8960 +/- 0.0084
controlled_raw_rho   0.8944 +/- 0.0074
controlled_ema       0.8945 +/- 0.0086
controlled_ema_trust 0.8943 +/- 0.0082
```

- Three-seed best test accuracy:

```text
vanilla_adam          0.8965 +/- 0.0057
fixed_adam_direction 0.8977 +/- 0.0056
controlled_raw_rho   0.8964 +/- 0.0068
controlled_ema       0.8965 +/- 0.0069
controlled_ema_trust 0.8963 +/- 0.0065
```

Interpretation:

- Candidate A improved controlled variants compared with the prior balanced
  setting on the same three seeds.
- Final controlled alpha rose from about `6.8e-4` to about `8.9e-4`.
- The improvement nearly closed the gap to vanilla/fixed Adam, but did not
  clearly beat fixed Adam-direction.

Candidate C result:

- Ran Candidate C on seeds `123`, `456`, and `789` for Fashion-MNIST CNN,
  20k/5k, 20 epochs.
- Three-seed final test accuracy:

```text
vanilla_adam          0.8946 +/- 0.0084
fixed_adam_direction 0.8960 +/- 0.0084
controlled_raw_rho   0.8951 +/- 0.0089
controlled_ema       0.8937 +/- 0.0092
controlled_ema_trust 0.8937 +/- 0.0092
```

- Three-seed best test accuracy:

```text
vanilla_adam          0.8965 +/- 0.0057
fixed_adam_direction 0.8977 +/- 0.0056
controlled_raw_rho   0.8978 +/- 0.0062
controlled_ema       0.8974 +/- 0.0063
controlled_ema_trust 0.8974 +/- 0.0063
```

Interpretation:

- Candidate C kept alpha near `9.3e-4`, so it behaved like a near-fixed Adam
  controller.
- Raw-rho improved a little relative to Candidate A, but the EMA-based
  variants slipped slightly below it.
- The controller still did not clearly beat fixed Adam-direction on mean final
  accuracy, so the evidence still favors caution over more aggressive tuning.

Candidate B result:

- Ran Candidate B on seeds `123`, `456`, and `789` for Fashion-MNIST CNN,
  20k/5k, 20 epochs.
- Three-seed final test accuracy:

```text
vanilla_adam          0.8946 +/- 0.0084
fixed_adam_direction 0.8960 +/- 0.0084
controlled_raw_rho   0.8921 +/- 0.0064
controlled_ema       0.8930 +/- 0.0078
controlled_ema_trust 0.8930 +/- 0.0078
```

- Three-seed best test accuracy:

```text
vanilla_adam          0.8965 +/- 0.0057
fixed_adam_direction 0.8977 +/- 0.0056
controlled_raw_rho   0.8941 +/- 0.0052
controlled_ema       0.8949 +/- 0.0063
controlled_ema_trust 0.8949 +/- 0.0063
```

Interpretation:

- Candidate B was the most aggressive Fashion-MNIST CNN candidate, but it was
  worse than Candidate A and Candidate C.
- It pushed the controller too low on alpha too quickly, especially for
  raw-rho.
- The best next move is probably to stop tuning this CNN family and move to the
  staged CIFAR ResNet smoke test.

CIFAR ResNet smoke test:

- Added `BasicBlock` and `SmallCIFARResNet` support in
  `controlled_adam_project/examples/run_mnist_demo.py` via
  `--model resnet_cifar`.
- Architecture: CIFAR-style stem (`3x3`, stride `1`, no initial max pool),
  three residual stages with widths `16`, `32`, and `64`, two basic blocks per
  stage, adaptive average pooling, and a linear 10-class head.
- Trainable parameters: `175258`.
- Ran the staged smoke test on CIFAR-10: 5k train / 1k test, 3 epochs,
  batch size `128`, seed `123`, `lr=1e-3`, full ablation, balanced controller
  settings (`alpha_min=1e-3`, `alpha_max=1.5e-3`, `rho_star=0.80`,
  `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`).
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_smoke_5k_1k_3epoch_balanced/`.
- Final test accuracy:

```text
vanilla_adam          0.3610
fixed_adam_direction 0.3730
controlled_raw_rho   0.4100
controlled_ema       0.3800
controlled_ema_trust 0.3800
```

Interpretation:

- The ResNet runner and CIFAR pipeline work.
- Controlled variants accepted every step and kept alpha near `1e-3`.
- Raw-rho showed the strongest early signal on this smoke run, so the next
  staged test should be 10k/2k for 10 epochs with the same balanced settings.

CIFAR ResNet 20-epoch staged benchmark:

- Ran a larger single-seed CIFAR-10 ResNet benchmark after the smoke test:
  10k train / 2k test, 20 epochs, batch size `128`, seed `123`, `lr=1e-3`,
  full ablation, same balanced controller settings
  (`alpha_min=1e-3`, `alpha_max=1.5e-3`, `rho_star=0.80`,
  `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`).
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced/`.
- The run used `--checkpoint-every 1` and `--print-every 1`, so it produced
  one checkpoint per epoch per variant plus progress logs.
- Final / best test accuracy:

```text
vanilla_adam          final 0.6915  best 0.6915 at epoch 20
fixed_adam_direction final 0.6875  best 0.6975 at epoch 19
controlled_raw_rho   final 0.7395  best 0.7395 at epoch 20
controlled_ema       final 0.7135  best 0.7135 at epoch 20
controlled_ema_trust final 0.7135  best 0.7135 at epoch 20
```

- Final diagnostics:

```text
fixed_adam_direction mean_alpha 0.001000  mean_rho 0.8853  accepted 1.0000
controlled_raw_rho   mean_alpha 0.001500  mean_rho 0.8821  accepted 1.0000
controlled_ema       mean_alpha 0.001500  mean_rho 0.8768  accepted 1.0000
controlled_ema_trust mean_alpha 0.001500  mean_rho 0.8768  accepted 1.0000
```

Interpretation:

- Raw-rho gave the strongest result so far on the staged ResNet architecture,
  beating vanilla Adam by about `4.8` percentage points in final test accuracy
  on this subset/seed.
- EMA and EMA-trust beat vanilla Adam but underperformed raw-rho.
- EMA and EMA-trust were identical, so the trust-region hook was dormant or
  saturated by the tight `alpha_max=1.5e-3` cap.
- Every controlled/fixed trial step was accepted. The alpha gate did not act as
  a rejection-heavy line search here; it mainly allowed a controlled ramp from
  Adam scale to the cap.
- This is still a single-seed, subset-limited result and should be followed by
  a multi-seed ResNet test or a cap/schedule ablation before making strong
  claims.

CIFAR ResNet Candidate 1 higher-cap test:

- Tested a slightly higher cap on the same 20-epoch 10k/2k ResNet setup:
  `alpha_min=1e-3`, `alpha_max=1.75e-3`, `rho_star=0.82`,
  `rho_beta=0.90`, `kp=0.015`, factor clip `[0.98, 1.012]`.
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_candidate1_higher_cap/`.
- Final / best test accuracy:

```text
vanilla_adam          final 0.6915  best 0.6915 at epoch 20
fixed_adam_direction final 0.6875  best 0.6975 at epoch 19
controlled_raw_rho   final 0.7120  best 0.7120 at epoch 20
controlled_ema       final 0.6695  best 0.6915 at epoch 17
controlled_ema_trust final 0.6695  best 0.6915 at epoch 17
```

Interpretation:

- Candidate 1 was worse than the balanced `alpha_max=1.5e-3` setting.
- The higher cap did not improve generalization. Raw-rho still beat vanilla
  Adam but lost the strong `0.7395` result from the balanced run.
- EMA and EMA-trust saturated at `1.75e-3` and performed worse.

CIFAR ResNet stronger fixed-LR control:

- Tested whether the original raw-rho gain was simply because Adam wanted a
  larger fixed learning rate. Used the same 20-epoch 10k/2k ResNet setup with
  `lr=1.5e-3`, `alpha_min=1e-3`, `alpha_max=1.5e-3`,
  `rho_star=0.80`, `rho_beta=0.90`, `kp=0.02`, factor clip
  `[0.98, 1.015]`.
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_lr15e4_control/`.
- Final / best test accuracy:

```text
vanilla_adam          final 0.7065  best 0.7065 at epoch 20
fixed_adam_direction final 0.6905  best 0.7040 at epoch 18
controlled_raw_rho   final 0.6985  best 0.6985 at epoch 20
controlled_ema       final 0.7250  best 0.7250 at epoch 20
controlled_ema_trust final 0.7250  best 0.7250 at epoch 20
```

Interpretation:

- Vanilla Adam at `1.5e-3` is a stronger baseline than vanilla Adam at
  `1e-3`, but it still did not match the original raw-rho `0.7395` result.
- Fixed Adam-direction at `1.5e-3` also did not explain the original raw-rho
  gain; its best was `0.7040`.
- Starting the controlled run at/near the cap was worse for raw-rho than
  starting at `1e-3` and ramping upward.
- EMA/EMA-trust did best in this control run at `0.7250`, but still trailed
  the original raw-rho balanced run.
- This supports the hypothesis that the ramp schedule / controller trajectory
  matters, not only the final `1.5e-3` scale.

CIFAR ResNet balanced multi-seed validation:

- Ran the original balanced ResNet setting for seeds `456` and `789`, and
  combined those results with the existing seed `123` run.
- Shared setup: CIFAR-10, `resnet_cifar`, 10k train / 2k test, 20 epochs,
  batch size `128`, `lr=1e-3`, full ablation, `alpha_min=1e-3`,
  `alpha_max=1.5e-3`, `rho_star=0.80`, `rho_beta=0.90`, `kp=0.02`, factor
  clip `[0.98, 1.015]`.
- Output directories:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced/`,
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced_seed_456/`,
  and
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced_seed_789/`.
- Per-seed final / best test accuracy:

```text
seed 123:
vanilla_adam          final 0.6915  best 0.6915
fixed_adam_direction final 0.6875  best 0.6975
controlled_raw_rho   final 0.7395  best 0.7395
controlled_ema       final 0.7135  best 0.7135
controlled_ema_trust final 0.7135  best 0.7135

seed 456:
vanilla_adam          final 0.6730  best 0.7065
fixed_adam_direction final 0.6830  best 0.7270
controlled_raw_rho   final 0.6950  best 0.7205
controlled_ema       final 0.6925  best 0.7245
controlled_ema_trust final 0.6925  best 0.7245

seed 789:
vanilla_adam          final 0.7015  best 0.7015
fixed_adam_direction final 0.7110  best 0.7110
controlled_raw_rho   final 0.7105  best 0.7105
controlled_ema       final 0.7190  best 0.7190
controlled_ema_trust final 0.7190  best 0.7190
```

- Three-seed aggregate:

```text
vanilla_adam          final 0.6887 +/- 0.0145  best 0.6998 +/- 0.0076
fixed_adam_direction final 0.6938 +/- 0.0150  best 0.7118 +/- 0.0148
controlled_raw_rho   final 0.7150 +/- 0.0226  best 0.7235 +/- 0.0147
controlled_ema       final 0.7083 +/- 0.0140  best 0.7190 +/- 0.0055
controlled_ema_trust final 0.7083 +/- 0.0140  best 0.7190 +/- 0.0055
```

Interpretation:

- The controlled variants retain a mean final-accuracy advantage over vanilla
  Adam and fixed Adam-direction on this three-seed ResNet subset benchmark.
- Raw-rho has the best mean final and best accuracy, but also the largest
  final-accuracy standard deviation because seed `123` was much stronger than
  seeds `456` and `789`.
- EMA and EMA-trust are more stable across seeds but slightly behind raw-rho on
  the mean.
- EMA-trust again matches EMA exactly, so the trust expansion path remains
  inactive under these capped Adam-scale settings. We later checked the
  minibatch step diagnostics directly: `controlled_ema_trust` had `0/1580`
  trust expansions for each of seeds `123`, `456`, and `789`.
- The inactivity was not a mystery in the optimizer logic. The balanced ResNet
  config used `alpha_min=1e-3` and `alpha_max=1.5e-3`, while the trust-region
  trigger still used `trust_region_alpha_threshold=1e-4`. Since the active alpha
  floor was already ten times larger than the trust threshold, the condition
  `alpha_used <= trust_region_alpha_threshold` could not be satisfied in normal
  accepted steps. In this benchmark, `controlled_ema_trust` should therefore be
  interpreted as an EMA-rho run with the trust hook enabled but dormant.
- A proper Adam-scale trust-region follow-up should keep the anti-collapse floor
  but move the trust threshold near it, for example
  `trust_region_alpha_threshold=1e-3` or `1.05e-3`, and use a gentle
  `trust_region_expand_factor` such as `1.1` or `1.2`.
- All fixed/controlled variants accepted essentially every step. This remains
  an alpha-governor regime rather than a rejection-heavy line search.

CIFAR ResNet controlled-vs-delayed Adam comparison:

- Added combined runner:

```text
delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py
```

- Purpose: compare same-step controlled Adam variants and delayed-feedback
  Adam variants in the same CIFAR-10 ResNet benchmark. This run explicitly
  included both raw-rho and EMA variants.
- Command:

```bash
MPLCONFIGDIR=/private/tmp python3 delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py \
  --epochs 20 \
  --train-subset 10000 \
  --test-subset 2000 \
  --batch-size 128 \
  --output-dir outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema \
  --print-every 1 \
  --checkpoint-every 5 \
  --variants vanilla_adam controlled_raw_rho controlled_ema controlled_ema_trust delayed_raw delayed_ema delayed_safe delayed_ema_floor90
```

- Output:

```text
outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/
```

- The output folder contains `REPORT.md`, epoch metrics, step diagnostics,
  plots, and checkpoints every 5 epochs.
- Final epoch metrics:

```text
optimizer                 test_acc  test_loss  train_loss  train_acc
vanilla_adam              0.6915    0.9454     0.7499      0.7398
controlled_raw_rho        0.7395    0.7715     0.5830      0.7935
controlled_ema            0.7135    0.8488     0.6656      0.7662
controlled_ema_trust      0.7135    0.8488     0.6656      0.7662
delayed_raw               0.6825    0.9768     0.7820      0.7290
delayed_ema               0.6960    0.9009     0.7334      0.7383
delayed_safe              0.6985    0.9156     0.7194      0.7452
delayed_ema_floor90       0.6950    0.8840     0.6783      0.7630
```

- Best test accuracy:

```text
vanilla_adam              0.6915 at epoch 20
controlled_raw_rho        0.7395 at epoch 20
controlled_ema            0.7135 at epoch 20
controlled_ema_trust      0.7135 at epoch 20
delayed_raw               0.6975 at epoch 19
delayed_ema               0.6960 at epoch 20
delayed_safe              0.6985 at epoch 20
delayed_ema_floor90       0.6950 at epoch 20
```

- Diagnostics:
  - same-step controlled variants: `1580/1580` accepted, zero backtracks;
  - `controlled_ema_trust`: zero trust expansions, exactly matches
    `controlled_ema`;
  - same-step controlled raw-rho and EMA both reached the `1.5e-3` max
    effective learning rate;
  - delayed variants applied the controller on `1579/1580` steps;
  - delayed variants stayed at or near their alpha floors:
    `7.5e-4`, `8.0e-4`, or `9.0e-4`;
  - final same-step rho was about `0.88`, while final delayed rho was about
    `0.13-0.18`.

Interpretation:

- Same-step controlled raw-rho remains the strongest result on this single-seed
  10k/2k ResNet subset.
- Same-step EMA is better than vanilla but weaker than raw-rho.
- EMA+trust was not a distinct algorithm in this run because trust expansion
  never fired.
- Delayed Adam is much cheaper per step, but the tested delayed parameters are
  not well calibrated. Reusing same-step-style targets such as
  `rho_star=0.7-0.8` makes the delayed controller read minibatch training as
  low-quality and shrink alpha to the floor.
- The next delayed Adam experiment should tune around the observed delayed
  rho scale, likely `rho_star` near `0.15-0.30`, with alpha floors/caps around
  Adam scale.

Fixed-LR interpretation note:

- The user correctly pointed out that if higher fixed-LR Adam works, one could
  also raise the controlled optimizer's upper bound. The reason to run fixed-LR
  baselines is not to dismiss controlled Adam; it is to separate three
  explanations:
  1. Adam simply wanted a larger fixed learning rate;
  2. the controlled run is acting like a useful warmup/ramp schedule;
  3. the rho controller is genuinely adapting step sizes in a way fixed LR or
     a simple schedule does not.
- A stronger controlled optimizer should be less sensitive to the exact upper
  bound than vanilla Adam is to fixed LR, and it should avoid destructive steps
  when given a broad cap.
- Future ResNet comparisons should include vanilla Adam at multiple LRs
  (`1e-3`, `1.25e-3`, `1.5e-3`, maybe `2e-3`) and a simple warmup/ramp from
  `1e-3` to `1.5e-3`.

Controlled Adam ResNet backtracking/cap/target diagnostics:

- Clarified the exact ResNet architecture used by the CIFAR runner:
  `SmallCIFARResNet`, not torchvision ResNet-18. It has a `3x3` stride-1 CIFAR
  stem, three residual stages of width `16`, `32`, and `64`, two basic blocks
  per stage, adaptive average pooling, and a `64 -> 10` head. Each block uses
  two `3x3` convolution/batchnorm layers and an identity or projection skip.
  Trainable parameters: `175258`.
- Added a detailed paper-style algorithm note:
  `controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md`.
- Exposed backtracking knobs in the CIFAR comparison runner:
  `--controlled-max-backtracks`, `--controlled-backtrack-shrink`, and
  `--controlled-rho-min`.
- High-LR/high-cap backtracking sweep output:
  `outputs/cifar10_resnet_adam_backtracking_sweep_highlr2e3_cap2p5e3_seed123/`.
  The large alpha cliffs in the high-cap plot were caused by backtracking.
  `backtrack_shrink=0.5` creates a visible half-step drop. Gentler shrink
  values (`0.7` or `0.8`) smooth alpha but do not outperform the lower-cap
  balanced setting. Best new setting in that sweep was `backtrack_shrink=0.7`,
  `rho_min=0.1`: raw-rho final `0.7030`, EMA final `0.7090`.
- Clean alpha-cap sweep output:
  `outputs/cifar10_resnet_adam_cap_sweep_lr1e3_seed123/`. This fixed
  `lr=1e-3`, `alpha_min=1e-3`, `rho_star=0.80`, `kp=0.02`, and swept
  `alpha_max`.

```text
alpha_max   raw-rho final   EMA final
1.50e-3     0.7395          0.7135
1.75e-3     0.7395          0.7240
2.00e-3     0.7290          0.7245
2.25e-3     0.7155          0.7355
```

- In the clean cap sweep there were zero backtracking events, yet alpha reached
  every tested cap under `rho_star=0.80`. Interpretation: cap saturation is not
  automatically bad, but the current same-step rho target is permissive enough
  that it does not find an interior alpha optimum on this task.
- A temporary separate decrease-gain experiment was tried and then removed
  from the active implementation because it added tuning complexity without
  improving the 20-epoch CIFAR result. The current controller exposes only the
  single proportional gain `kp`.
- Raised-target / separate-decrease-gain test output:
  `outputs/cifar10_resnet_adam_rhostar_asym_gain_seed123/`.

```text
setting                         raw-rho final   EMA final   alpha behavior
rho_star=0.80 symmetric         0.7155          0.7355      reaches cap
rho_star=0.85 symmetric         0.7115          0.6980      avoids cap
rho_star=0.80, extra down gain  0.6950          0.7330      mostly reaches cap
```

- Raising `rho_star` worked mechanically and made alpha self-limiting, but
  `0.85` was too conservative for this 20-epoch setup. The extra decrease gain
  delayed saturation but did not improve accuracy because rho remained above
  target often enough that alpha eventually climbed back.
- Recommended next compact controller test:
  `rho_star=0.825`, `kp=0.02`, `alpha_max=2.25e-3`, variants
  `controlled_raw_rho` and `controlled_ema`.

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

Latest local commits after the PI optimizer, documentation refresh, and PCA
trajectory work:

```text
12a12df Add PCA trajectory visualization for Adam checkpoints
87af501 Refresh project documentation state
9c55036 Add PI optimizers and align Muon implementation
```

The root `.gitignore` excludes:

- generated output folders
- local datasets
- local virtual environments
- cache files
- CIFAR tarballs

User-supplied comparison/report artifacts remain untracked by design:

- `fashionmnist_20epoch_metrics.png`
- `fashionmnist_20epoch_metrics.summary.csv`
- `scheduled_iterate_muon_academic_report.pdf`

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

- neural-network Muon paths now follow `torch.optim.Muon` scope
- 2D hidden matrix parameters are orthogonalized directly
- vectors, scalars, norms, biases, embeddings, heads, and convolution kernels
  use AdamW-style fallback updates
- Muon momentum uses the PyTorch `lerp` convention, with the quintic
  Newton-Schulz coefficients `(3.4445, -4.7750, 2.0315)` and 5 default steps

Important caveat:

- The current implementation uses NumPy/CPU orthogonalization.
- It is useful for research demos but slow for CIFAR.
- Historical Muon benchmark tables before the official-style grouping fix used
  a broader local Muon variant and should be treated as archival.

Verification:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Latest known result:

```text
12 passed
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

Additional multi-seed diagnostic:

```bash
cd controlled_muon_project
for seed in 123 456 789 2024 2025; do
  MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
    --dataset fashion_mnist \
    --fashion-folder ../fashion \
    --epochs 20 \
    --train-subset 1024 \
    --test-subset 512 \
    --batch-size 128 \
    --lr 1e-3 \
    --seed "$seed" \
    --ablation \
    --print-every 10 \
    --output-dir "outputs/fashion_mnist_muon_multiseed_20epoch_5seeds_1k/seed_${seed}"
done
```

Aggregate result:

```text
vanilla_muon          final 0.6051 +/- 0.0181  best 0.6051 +/- 0.0181
fixed_muon_direction final 0.6051 +/- 0.0181  best 0.6051 +/- 0.0181
controlled_raw_rho   final 0.7289 +/- 0.0070  best 0.7289 +/- 0.0070
controlled_ema       final 0.7293 +/- 0.0067  best 0.7293 +/- 0.0067
controlled_ema_trust final 0.7293 +/- 0.0067  best 0.7293 +/- 0.0067
```

Timing interpretation:

- Controlled variants add one same-minibatch forward loss evaluation per optimizer step.
- They do not add an extra backward pass, so the theoretical overhead is usually moderate rather than a full 2x.
- On this small five-seed CPU run, elapsed times were noisy but comparable to vanilla Muon.
- Larger claims should compare loss and accuracy versus wall-clock time, not only epochs or optimizer steps.

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

## 11. Deterministic Function Optimization Manager Suite

Added a self-contained function optimization benchmark/report runner:

```text
controlled_adam_project/examples/run_function_benchmark_report.py
```

Added the tracked guide:

```text
FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md
```

Purpose:

- provide a simple, deterministic, non-deep-learning benchmark suite for
  manager updates;
- compare optimizer behavior on the existing 2D functions;
- generate plots that show trajectories on objective landscapes, objective
  residual curves, and global alpha schedules;
- keep the suite reproducible from code because generated outputs are ignored
  by git.

Command run:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

Generated local outputs:

```text
controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md
controlled_adam_project/outputs/function_report_multistart/per_start_results.csv
controlled_adam_project/outputs/function_report_multistart/aggregate_results.csv
controlled_adam_project/outputs/function_report_multistart/benchmark_config.csv
controlled_adam_project/outputs/function_report_multistart/*_trajectory_comparison.png
controlled_adam_project/outputs/function_report_multistart/*_objective_curves.png
controlled_adam_project/outputs/function_report_multistart/*_alpha_curves.png
```

Benchmark design:

- objectives: Quadratic, Rosenbrock, Himmelblau, Rastrigin, Beale, Ackley,
  Six-hump camel, Goldstein-Price, and Easom;
- starts: five fixed starts per objective;
- optimizers: vanilla Adam, controlled Adam with raw rho, controlled Adam with
  EMA-smoothed rho, and controlled Adam with EMA-smoothed rho plus the
  trust-region tiny-alpha expansion hook;
- success: residual above known global minimum or distance to known global
  minimizer;
- residual is used instead of raw objective value because some objectives have
  negative global minima.

Current result snapshot:

```text
Highest success-rate winner counts:
vanilla Adam          5
controlled raw-rho   9
controlled EMA-rho   7
controlled EMA+trust  7

Lowest median final residual winner counts:
vanilla Adam          5
controlled raw-rho   3
controlled EMA-rho   4
controlled EMA+trust  3

Lowest median best residual winner counts:
vanilla Adam          5
controlled raw-rho   3
controlled EMA-rho   4
controlled EMA+trust  3
```

Ties are counted for every tied optimizer, so row totals can exceed nine.

Interpretation:

- controlled raw-rho ties or wins success rate on every objective in this
  suite;
- controlled variants improve success rate over vanilla Adam on Quadratic,
  Rosenbrock, Himmelblau, and Beale;
- controlled variants improve median best residual over vanilla Adam on
  Quadratic, Rosenbrock, Beale, and Goldstein-Price;
- the trust-region Adam variant is included for consistency with neural-network
  benchmarks, but on these deterministic 2D functions it usually overlaps
  EMA-rho because the high-rho/tiny-alpha expansion condition rarely triggers;
- vanilla Adam still ties or wins some residual metrics when the fixed learning
  rate is well matched;
- Rastrigin, Ackley, and Six-hump camel are useful limitation cases because
  local step-size control does not solve global basin selection.

Trimmed manager report:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

This three-objective report keeps Quadratic, Beale, and Goldstein-Price. It is
intended for a shorter manager update where the goal is to show controlled Adam
clearly without displaying every limitation case in the first figure. Winner
counts in the current trimmed run are:

```text
Highest success-rate winner counts:
vanilla Adam          1
controlled raw-rho   3
controlled EMA-rho   2
controlled EMA+trust  2

Lowest median final residual winner counts:
vanilla Adam          0
controlled raw-rho   1
controlled EMA-rho   2
controlled EMA+trust  2

Lowest median best residual winner counts:
vanilla Adam          0
controlled raw-rho   1
controlled EMA-rho   2
controlled EMA+trust  2
```

The trimmed report's `median_trust_expansions` values are `0.0`, so EMA+trust
should be described as included for completeness rather than as a separately
activated trust-region effect in this specific manager slice.

Verification:

```bash
python -m py_compile controlled_adam_project/examples/run_function_benchmark_report.py
```

Muon extension:

- Added `controlled_muon_project/examples/run_function_benchmark_report.py`.
- The Muon function report compares `vanilla_muon`, `controlled_raw_rho`,
  `controlled_ema`, and `controlled_ema_trust` on the same nine 2D functions
  and five starts per function.
- For 2D vector functions, the runner uses a vector analogue of Muon by treating
  the momentum vector as a column matrix and passing it through the same
  orthogonalization utility used by the Muon subproject.
- Removed the earlier `fixed_muon_direction` diagnostic from the function
  report because it duplicated `vanilla_muon` in this local 2D runner: both used
  a fixed alpha and no rho controller.
- Generated local report:
  `controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md`.
- Generated trimmed manager report:
  `controlled_muon_project/outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md`.
- Both Adam and Muon function reports now generate Chinese companion reports
  and standalone `*_surface_3d.png` objective-landscape plots with formulas
  printed inside the figures.

Muon current result snapshot:

```text
Highest success-rate winner counts:
vanilla Muon           9
controlled raw-rho    7
controlled EMA-rho    7
controlled EMA+trust  7

Lowest median final residual winner counts:
vanilla Muon           4
controlled raw-rho    3
controlled EMA-rho    3
controlled EMA+trust  4

Lowest median best residual winner counts:
vanilla Muon           6
controlled raw-rho    3
controlled EMA-rho    1
controlled EMA+trust  2
```

Muon interpretation:

- The 2D Muon story differs from Adam: vanilla Muon is very competitive on this
  normalized vector-direction suite.
- Controlled Muon improves some cases, especially Quadratic residuals, but it
  does not dominate overall.
- Present this as an honest diagnostic: alpha control is not automatically
  better for every base direction.
- Because this is a vector analogue, it should not be overclaimed as a full
  matrix Muon benchmark.

Muon verification:

```bash
python -m py_compile controlled_muon_project/examples/run_function_benchmark_report.py
cd controlled_muon_project && PYTHONPATH=src pytest -q
```

Result:

```text
12 passed
```

## 12. Function Manager Report Follow-Ups

Added reproducibility controls to the Adam and Muon deterministic function
report runners:

```text
--step-multiplier
--random-starts-per-objective
--random-seed
```

Generated longer manager-facing function reports:

```text
controlled_adam_project/outputs/function_report_manager_trimmed_10x
controlled_adam_project/outputs/function_report_manager_extended_10x
controlled_adam_project/outputs/function_report_manager_extended_10x_15starts
controlled_adam_project/outputs/function_report_manager_extended_10x_30starts
controlled_adam_project/outputs/function_report_manager_extended_10x_60starts
controlled_adam_project/outputs/function_report_manager_extended_60starts_default_steps
controlled_muon_project/outputs/function_report_manager_trimmed_10x
controlled_muon_project/outputs/function_report_manager_extended_10x
```

Main interpretation:

- Longer Adam runs show that vanilla Adam can catch up on some objectives with
  enough iterations, so the best controlled-Adam claim is faster local progress
  and better step-size robustness, not universal final dominance.
- The 60-start default-step run is the strongest current manager-facing
  function result for a fixed practical iteration budget. Controlled variants
  improve success rate over vanilla Adam on Beale, Goldstein-Price,
  Rosenbrock, Himmelblau, and Quadratic within the default step counts.
- In that 60-start default run, Beale success is `33-37%` for controlled
  variants versus `5%` for vanilla; Rosenbrock is `28-32%` versus `8%`;
  Himmelblau is `97%` versus `92%`; and vanilla Adam gets `0%` success on the
  ill-conditioned Quadratic within 300 steps.
- Himmelblau is a clean controlled-Adam manager example: all optimizers
  succeed, but controlled Adam reaches the success criterion much faster.
- Rosenbrock shows a speed-versus-eventual-success tradeoff.
- The 60-start 10x run is the corresponding caveat. Vanilla Adam catches up or
  exceeds success on several functions with enough time, but controlled
  variants still tend to reach successful runs faster. On Rosenbrock, vanilla
  reaches `100%` success but takes about `8085` median successful iterations,
  while controlled variants take about `3954-5060`.
- Beale and Goldstein-Price are budget-sensitive: controlled variants are much
  better under the default budget, but vanilla catches up more under 10x.
- Muon 10x reports did not materially improve the Muon story; vanilla Muon
  remains very competitive in the 2D vector analogue.

Added a focused Rastrigin basin benchmark:

```text
controlled_adam_project/examples/run_rastrigin_basin_benchmark.py
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts
```

This benchmark samples 30 starts per radius from boxes centered on `(0, 0)` and
plots success rate versus initialization radius. Result: all optimizers solve
Rastrigin reliably when initialized inside radius `0.5`, success drops sharply
outside that basin, and none reach the true global minimum at radius `4.0`.
Controlled Adam reaches successful runs faster than vanilla Adam inside the
correct basin, but it does not solve global basin selection.

## 13. PI Optimizer Subprojects And Implementation Audit

Added standalone PI-controller optimizer folders:

```text
pi_adam_optimizer/
pi_muon_optimizer/
```

Implementation state:

- `PIAdam` wraps Adam's bias-corrected direction with a global PI-controlled
  multiplier.
- `PIMuon` applies official-style Muon only to 2D hidden matrix parameters and
  uses AdamW-style fallback directions for non-2D or excluded parameters.
- Both PI optimizers support optional EMA smoothing, bad-step rejection,
  bounded backtracking, non-descent fallback, trust-region expansion, and
  decoupled weight decay.
- The PI Fashion-MNIST runner now compares against corrected vanilla baselines:
  Adam or AdamW for the Adam family, and official-style Muon with AdamW
  fallback for the Muon family.

Updated or added documentation:

- `OPTIMIZER_IMPLEMENTATION_AUDIT.md`
- `pi_adam_optimizer/README.md`
- `pi_adam_optimizer/PI_ADAM_DESIGN_AND_COMPARISON.md`
- `pi_muon_optimizer/README.md`
- `pi_muon_optimizer/PI_MUON_DESIGN_AND_COMPARISON.md`

Verification:

```bash
python -m py_compile pi_adam_optimizer/pi_adam.py pi_muon_optimizer/pi_muon.py examples/run_pi_fashion_mnist_multiseed.py controlled_muon_project/src/controlled_muon/orthogonalization.py controlled_muon_project/src/controlled_muon/optimizers.py controlled_muon_project/src/controlled_muon/torch_optimizers.py controlled_muon_project/examples/run_mnist_demo.py controlled_muon_project/examples/run_function_benchmark_report.py controlled_muon_project/examples/run_matrix_quadratic_demo.py
pytest -q pi_adam_optimizer/test_pi_adam.py pi_muon_optimizer/test_pi_muon.py
PYTHONPATH=controlled_muon_project/src pytest -q controlled_muon_project/tests
PYTHONPATH=controlled_adam_project/src pytest -q controlled_adam_project/tests
pytest -q tests
```

Focused results:

```text
PI optimizer tests: 15 passed
controlled_muon_project tests: 12 passed
controlled_adam_project tests: 9 passed
root tests: 4 passed
```

Archived previous generated experiment outputs into:

```text
outputs/backup_20260526_182414/
controlled_adam_project/outputs/backup_20260526_182414/
controlled_muon_project/outputs/backup_20260526_182414/
```

The local commits containing the PI optimizer, documentation refresh, and PCA
trajectory work are:

```text
12a12df Add PCA trajectory visualization for Adam checkpoints
87af501 Refresh project documentation state
9c55036 Add PI optimizers and align Muon implementation
```

## 14. PCA Training-Trajectory Visualization

Added a checkpoint post-processor for high-dimensional Adam training paths:

```text
controlled_adam_project/examples/plot_pca_training_trajectory.py
```

Purpose:

- Reproduce the PCA trajectory visualization idea from neural loss-landscape
  papers.
- Use checkpoints produced by `controlled_adam_project/examples/run_mnist_demo.py`
  with `--checkpoint-every N`.
- Flatten trainable parameters by default, excluding BatchNorm running buffers.
- Fit a 2D PCA plane to checkpoint displacements.
- Overlay one or more optimizer trajectories in that shared plane.

Main outputs:

```text
pca_trajectory_coordinates.csv
pca_explained_variance.csv
pca_training_trajectory.png
```

Typical command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/plot_pca_training_trajectory.py \
  outputs/YOUR_RUN_WITH_CHECKPOINTS \
  --runs vanilla_adam fixed_adam_direction controlled_raw_rho controlled_ema \
  --output-dir outputs/YOUR_RUN_WITH_CHECKPOINTS/pca_trajectory
```

Implementation details:

- The script accepts either a run directory containing `checkpoints/` or the
  checkpoint directory itself.
- It infers dataset/model from `run_metadata.json` when available, or accepts
  explicit `--dataset` and `--model`.
- The default reference origin is the final checkpoint of the first selected
  run. `--reference-run` and `--reference-epoch` override this.
- Standard PCA centering is used by default while coordinates remain
  final-relative. `--no-center-for-pca` fits directly on final-relative
  displacements.
- Misspelled `--runs` values now fail clearly and list available runs.

Verification:

```text
python -m py_compile examples/plot_pca_training_trajectory.py
single-run smoke: 20 checkpoints, 175258 parameters
multi-run smoke: 80 checkpoints, 206922 parameters
checkpoint-directory input plus --no-center-for-pca: passed
bad --runs name: clear ValueError with available run names
```

The README in `controlled_adam_project/` and the root README now include the
usage command.

## 15. Delayed-Feedback Optimizer Subprojects

Added two standalone optimizer experiments:

```text
delayed_feedback_adam/
delayed_feedback_muon/
```

Purpose:

- Test a lower-overhead variant of the actual-versus-predicted decrease
  controller.
- Instead of evaluating `f(theta + alpha p)` immediately after a trial step,
  store the previous loss and predicted decrease, then use the next naturally
  computed loss as delayed feedback.
- Avoid the extra same-minibatch forward pass required by same-step controlled
  Adam/Muon and PI optimizers.

Tradeoff:

- The delayed controller cannot reject a bad step before it happens.
- Feedback is one step late.
- In minibatch training, consecutive losses usually come from different
  minibatches, so the delayed rho estimate mixes optimization progress with
  sampling noise.

Implementation state:

- `DelayedFeedbackAdam` wraps Adam/AdamW-style directions with delayed P/PI/PID
  alpha control.
- `DelayedFeedbackMuon` wraps 2D Muon directions with auxiliary AdamW fallback
  for non-2D parameters.
- The delayed Muon code was aligned with local PyTorch `2.10.0`
  `torch.optim.Muon` behavior:
  - `buf.lerp_(grad, 1 - momentum)` momentum update;
  - `grad.lerp(buf, momentum)` Nesterov update;
  - automatic Muon selection is 2D-only;
  - `adjust_lr_fn=None` and `"original"` both use original Muon shape scaling;
  - decoupled weight decay uses base learning rate, not shape-adjusted Muon LR.
- The delayed Adam/AdamW path was checked against local `torch.optim.AdamW` for
  the supported simple non-AMSGrad path.

Verification:

```text
cd delayed_feedback_adam && PYTHONPATH=. python -m pytest -q tests
4 passed

cd delayed_feedback_muon && PYTHONPATH=. python -m pytest -q tests
6 passed
```

Important note:

- Running these tests from the monorepo root without installing the standalone
  packages can hit import shadowing because the outer directory has the same
  name as the inner package. Run from inside each subproject with
  `PYTHONPATH=.` or install the subproject editable.

## 16. Controlled Adam Function Tuning Sweeps

After comparing Adam variants, SGD, and SGD with momentum on the deterministic
function suite, we tuned the three controlled Adam families directly:

```text
controlled_adam_project/examples/run_controlled_adam_parameter_sweep.py
controlled_adam_project/examples/run_controlled_adam_tuning_sweep.py
controlled_adam_project/examples/run_controlled_adam_refined_tuning_sweep.py
controlled_adam_project/examples/run_controlled_adam_simplified_tuning_sweep.py
```

The first diagnostic sweep checked why `controlled_ema_trust` did not improve
over `controlled_ema_rho`. It showed the original trust gate was too
conservative: the current setting used `rho>=0.90`, `alpha<=1e-4`, expand
`x1.5`, and the trust expansion count was usually zero. Wider gates, especially
near `alpha<=1e-2`, made trust expansion active and improved success.

The broad sweep output is:

```text
outputs/controlled_adam_tuning_sweep_30runs/
```

It tested 50 variants on 9 objectives with 30 starts each. The best broad
results were:

```text
raw current 0.370 -> best broad 0.426
EMA current 0.359 -> best broad 0.419
trust current 0.359 -> best broad 0.441
```

The refined sweep output is:

```text
outputs/controlled_adam_refined_tuning_sweep_30runs/
```

Command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_controlled_adam_refined_tuning_sweep.py \
  --output-dir /Users/honghaoyu/adaptive_stepsize_control/outputs/controlled_adam_refined_tuning_sweep_30runs \
  --random-starts-per-objective 25 \
  --random-seed 20260527
```

Validation:

```text
per_start_results.csv: 33,210 rows
aggregate_results.csv: 1,107 aggregate rows
variant_config.csv: 123 variants
all objective/variant groups: 30 starts
```

Refined results:

```text
raw-rho current 0.370 -> tuned 0.437
EMA-rho current 0.359 -> tuned 0.444
EMA+trust current 0.359 -> tuned 0.489
```

Best tuned variants:

```text
raw-rho:    rho_star - 0.2, kp x2, alpha_min=1e-5
EMA-rho:    rho_star - 0.2, kp x2, alpha_min=1e-5
EMA+trust:  rho_beta=0.95, rho>=0.70, alpha<=1e-2, expand x2, alpha_min=1e-5
```

Mean log10 median-best residuals also improved:

```text
raw-rho:    -1.838 -> -2.636
EMA-rho:    -2.025 -> -2.587
EMA+trust:  -2.023 -> -3.018
```

Interpretation:

- Raw-rho and EMA-rho were often over-shrinking to the old `alpha_min=1e-8`.
  A higher floor, stronger proportional gain, and lower rho target improved the
  deterministic function suite.
- EMA+trust did not help before because the trust expansion condition rarely
  fired. The tuned trust settings record median trust expansions around `12`
  across objective rows, and then trust becomes the best controlled family.
- Ackley and Rastrigin remain basin-selection limitation cases. Tuning improves
  local step-size behavior but does not make the method a global optimizer.
- These are deterministic function-suite parameters, not neural defaults.
  Neural Adam-scale trust tests must match `trust_region_alpha_threshold` to the
  active alpha range, for example near the `alpha_min` used in that run.

Simplified preset sweep:

Purpose:

- Reduce the apparent optimizer tuning burden.
- Test a low-dimensional interface instead of exposing every raw
  hyperparameter.
- Expose only:
  - `family`: raw-rho, EMA-rho, or EMA+trust;
  - `response_preset`: conservative, balanced, or aggressive;
  - `alpha_preset`: low_floor, mid_floor, wide_cap, or high_floor.

Command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_controlled_adam_simplified_tuning_sweep.py \
  --output-dir /Users/honghaoyu/adaptive_stepsize_control/outputs/controlled_adam_simplified_tuning_sweep_30runs \
  --random-starts-per-objective 25 \
  --random-seed 20260527
```

Validation:

```text
per_start_results.csv: 10,530 rows
aggregate_results.csv: 351 aggregate rows
variant_config.csv: 39 variants
variant_expanded_by_objective.csv: 351 objective-expanded settings
all objective/variant groups: 30 starts
```

Preset definitions:

```text
response=conservative: kp x1,   rho_star +0.0, rho_beta 0.95, trust rho>=0.80, expand x1.5
response=balanced:     kp x1.5, rho_star -0.1, rho_beta 0.90, trust rho>=0.70, expand x2
response=aggressive:   kp x2,   rho_star -0.2, rho_beta 0.90, trust rho>=0.60, expand x3

alpha=low_floor:  alpha_min=0.001*alpha0, alpha_max=25*alpha0, trust_threshold=1*alpha0
alpha=mid_floor:  alpha_min=0.003*alpha0, alpha_max=25*alpha0, trust_threshold=2*alpha0
alpha=wide_cap:   alpha_min=0.003*alpha0, alpha_max=50*alpha0, trust_threshold=3*alpha0
alpha=high_floor: alpha_min=0.01*alpha0,  alpha_max=50*alpha0, trust_threshold=3*alpha0
```

Result:

```text
raw-rho current 0.370 -> simplified 0.470
EMA-rho current 0.359 -> simplified 0.485
EMA+trust current 0.359 -> simplified 0.504
```

The winning preset for all three families was `aggressive_high_floor`.
For EMA+trust it recorded median trust expansions around `16` across objective
rows. This simplified preset recovered and slightly exceeded the earlier
refined-grid result while using far fewer user-facing choices.

Interpretation:

- The simplified interface is the better usability direction.
- The gain is a coupled preset effect: higher alpha floor, wider alpha cap,
  lower rho target, stronger gain, and reachable trust expansion.
- The result does not prove `aggressive_high_floor` is a neural-network default.
  The next validation should map these relative alpha settings to a neural base
  learning rate and test conservative/balanced/aggressive presets on a small
  Fashion-MNIST or CIFAR run.

Tuned no-momentum function benchmark:

- Added:

```text
controlled_adam_project/examples/run_function_benchmark_tuned_simplified_report.py
```

- Purpose: rerun the full deterministic function benchmark using the
  simplified `aggressive_high_floor` controlled Adam parameters, while omitting
  SGD with momentum by request.
- Command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_function_benchmark_tuned_simplified_report.py \
  --output-dir /Users/honghaoyu/adaptive_stepsize_control/outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum \
  --random-starts-per-objective 25 \
  --random-seed 20260527
```

- Validation:

```text
per_start_results.csv: 1,350 rows = 9 objectives x 5 optimizers x 30 starts
aggregate_results.csv: 45 aggregate rows
optimizer set: gradient_descent, vanilla_adam, controlled_raw_rho,
               controlled_ema_rho, controlled_ema_trust
```

- Headline averages:

```text
gradient_descent        success 0.278  mean log10 best residual -2.354
vanilla_adam            success 0.307  mean log10 best residual -2.215
tuned raw-rho           success 0.470  mean log10 best residual -3.384
tuned EMA-rho           success 0.485  mean log10 best residual -3.246
tuned EMA+trust         success 0.504  mean log10 best residual -3.499
```

- Winner counts across nine objectives:

```text
highest success rate:        GD 4, Adam 5, raw 7, EMA 7, EMA+trust 9
lowest median final residual GD 5, Adam 5, raw 1, EMA 3, EMA+trust 7
lowest median best residual  GD 5, Adam 5, raw 1, EMA 3, EMA+trust 7
```

- Interpretation: tuned EMA+trust is the strongest controlled variant in this
  no-momentum function report. Trust expansion is now active, with median trust
  expansions around `16` across objective rows. The tuned controlled variants
  improve over vanilla Adam most clearly on Beale, Quadratic, and Rosenbrock;
  Ackley and Rastrigin remain basin-selection limitation cases.

SGD/GD baseline learning-rate follow-up:

- We investigated why fixed gradient descent behaved strangely on
  Goldstein-Price in the tuned no-momentum report. The cause was not plotting:
  Goldstein-Price has very large raw gradients, and `alpha0 = 0.003` produces
  first steps that can leave the meaningful basin immediately. In the original
  report, `best_iteration = 0` for `29/30` Goldstein-Price GD starts.
- Updated
  `controlled_adam_project/examples/run_function_benchmark_tuned_simplified_report.py`
  with a GD-only CLI option:

```bash
--gradient-descent-alpha-multiplier VALUE
```

- This changes only the fixed `gradient_descent` baseline:

```text
gradient_descent alpha = VALUE * alpha0
vanilla_adam and controlled Adam alpha0 = unchanged
```

- Ran full 30-start reports for:

```text
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p03/
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p05/
```

- Validation: both runs have `1,350` per-start rows and `45` aggregate rows.
  Non-GD aggregate metrics are identical to the original tuned no-momentum run;
  only `gradient_descent` changes.
- GD baseline comparison:

```text
GD alpha       avg success  mean log10 best  Goldstein success  Goldstein best  Goldstein final
1.0 * alpha0   0.278        -2.354           0.033              1512.07         2.64e20
0.03 * alpha0  0.167        -0.255           0.367              27.0            27.0
0.05 * alpha0  0.174        -0.784           0.167              1072.09         6.88e14
```

- Interpretation: `0.03 * alpha0` is the most Goldstein-stable GD baseline we
  tested. `0.05 * alpha0` is slightly better overall than `0.03 * alpha0`, but
  already reintroduces Goldstein instability. The original `1.0 * alpha0` is
  faster on some functions where GD does not explode, so fixed GD needs
  function-specific tuning or line search if it is meant to be a strong
  baseline.

## 17. Controlled Adam Non-Descent Gradient Fallback

Changed the controlled Adam non-descent branch after discussing whether the
old behavior wasted minibatches. Previously, if Adam's momentum direction was
not a descent direction for the current minibatch, the optimizer shrank
`alpha` and skipped the parameter update while still updating Adam moments.

The new behavior is less conservative:

```text
if -<g, d_adam> <= 0 and ||g|| > 0 and ||d_adam|| > 0:
    d = -g * ||d_adam|| / (||g|| + eps)
else:
    d = d_adam
```

The same same-minibatch trial evaluation, backtracking, bad-step rejection,
clipped-rho control, and single-gain alpha update then run on the effective
direction. The old shrink-and-skip path remains only for degenerate
zero-gradient or zero-direction cases.

Files changed:

- `controlled_adam_project/src/controlled_adam/optimizers.py`
- `controlled_adam_project/src/controlled_adam/torch_optimizers.py`
- `controlled_adam_project/examples/run_function_benchmark_report.py`
- `controlled_adam_project/examples/run_mnist_demo.py`
- `controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md`

Diagnostics added:

- deterministic histories now include `gradient_fallback_used`;
- PyTorch minibatch step logs now include `used_gradient_fallback`;
- MNIST/CIFAR step CSVs include `used_gradient_fallback`.

Validation:

```text
cd controlled_adam_project
PYTHONPATH=src pytest -q tests
11 passed
```

## 18. Controlled Adam v5.1 Controller Stabilization

Implemented the v5.1 subset of `controlled_adam_production_fix_plan_v5.md`,
with one deliberate change: the non-descent fallback remains the norm-matched
negative-gradient fallback rather than raw `-g`.

Mechanics changed:

- Rho now uses a floored denominator:

```text
predicted_safe = max(predicted_raw,
                     absolute_predicted_floor,
                     ratio_eps,
                     relative_predicted_floor * |loss_before|)
```

- Acceptance uses measured rho, while EMA/controller updates use clipped rho:

```text
rho_measured = actual / predicted_safe
rho_clipped = clip(rho_measured, rho_clip_min, rho_clip_max)
```

- A fully rejected trial sequence cannot increase the next alpha.
- Default backtracking depth is now `max_backtracks=1`, but it remains
  configurable.
- Trust expansion now requires `trust_region_patience` consecutive accepted,
  non-backtracked, high-rho steps and is capped by `trust_region_max_factor`.

Defaults added:

```text
absolute_predicted_floor = 1e-12
relative_predicted_floor = 1e-8
rho_clip_min = -1.0
rho_clip_max = 3.0
trust_region_patience = 2
trust_region_max_factor = 1.5
max_backtracks = 1
```

Diagnostics added:

- `rho_clipped`
- `predicted_decrease_safe`
- `predicted_was_floored`
- `rho_was_clipped`
- `direction_type`
- `trust_good_count`

Updated code paths:

- `TorchControlledAdam`
- deterministic `controlled_adam`
- local EMA/trust helper in the function benchmark report
- MNIST/CIFAR step diagnostics CSVs
- deterministic diagnostics CSVs

Validation:

```text
cd controlled_adam_project
PYTHONPATH=src pytest -q tests
17 passed
```

AdamW mode from the v5 plan was intentionally not implemented in this pass; it
should be a separate semantic patch.

## 19. Controlled Adam Single-Gain Documentation Cleanup

Removed the temporary separate downward-gain parameter from the active
documentation and clarified the simplified controller interface:

- active controlled Adam exposes one proportional gain, `kp`;
- historical `kp_down` / separate downward-gain experiments remain useful as
  evidence, but should not be treated as active API;
- backtracking is current-step retry logic, while alpha control is a next-step
  multiplier update;
- an accepted step can still decrease the next alpha when the same-minibatch
  loss decreased but the ratio was below `rho_star`;
- rho clipping is for controller robustness, factor clipping limits the
  per-step alpha-change speed, and alpha clipping enforces the configured
  operating range.

Updated `controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md` to use compact
PyCharm-readable display equations and to keep code names in prose rather than
as English-like math variables.

## 20. Recommended Next Engineering Steps

1. Tune delayed-feedback Adam using delayed-specific rho targets. The latest
   ResNet comparison showed delayed `rho_bar` around `0.13-0.24`, far below the
   same-step target range. Start with `rho_star=0.15-0.30`, alpha floors around
   `0.8-1.1x` base LR, and caps around `1.25-1.75x` base LR.

2. Add fixed-LR and simple warmup baselines for controlled Adam ResNet
   comparisons. Include vanilla Adam at `1e-3`, `1.25e-3`, `1.5e-3`, and
   possibly `2e-3`, plus a simple `1e-3 -> 1.5e-3` warmup/ramp. This is the
   control that separates a useful rho controller from a larger LR cap or a
   basic schedule.

3. Validate the simplified controlled-Adam preset interface on a neural task
   before changing any default neural settings. The deterministic sweep suggests
   `aggressive_high_floor` is strong on 2D functions, but neural runs operate
   at Adam-scale alpha ranges such as `1e-3`; alpha bounds and trust thresholds
   must be derived from the neural base learning rate and checked on
   Fashion-MNIST or CIFAR.

4. Add checkpointing and incremental epoch-metrics flushing to the Muon image
   benchmark runner. It now has `--print-every`, but long jobs still need
   partial CSV output and resumable checkpoints.

5. Add an automatic best-epoch summary file.

6. Consider a faster torch-native Muon implementation that avoids NumPy round trips.

7. Tune Muon controller parameters separately from Adam.

8. Run larger/full-dataset experiments only after checkpointing or incremental
   metrics writing is in place.

9. Keep preserving the ablation structure:
   - base optimizer
   - fixed direction
   - raw rho control
   - EMA rho control
   - EMA plus trust/recovery control
   Also preserve fixed-LR and simple-schedule baselines so controller wins can
   be interpreted rather than accidentally conflated with ordinary LR tuning.
