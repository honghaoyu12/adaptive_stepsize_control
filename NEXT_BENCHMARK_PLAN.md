# Next Benchmark Plan

Last updated: 2026-05-26

Note: for manager-facing deterministic function optimization results, use the
new tracked guide `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md` and the report
runner `controlled_adam_project/examples/run_function_benchmark_report.py`.
This plan remains focused on neural-network and architecture-transfer
benchmarks.

Implementation note: future Muon neural-network benchmarks should use only the
official-style Muon paths. The older all-parameter local Muon baseline was
incorrect for neural networks and should be ignored. Official-style Muon means
Muon on 2D hidden matrix parameters and AdamW-style fallback for vectors,
scalars, norms, biases, embeddings, heads, and convolution kernels.

Output note: previous results were archived under
`outputs/backup_20260526_182414/`,
`controlled_adam_project/outputs/backup_20260526_182414/`, and
`controlled_muon_project/outputs/backup_20260526_182414/`. Put new benchmark
runs in fresh top-level output folders outside those backups.

This document lays out the next benchmark direction for the adaptive
same-minibatch controller work. The goal is to test whether the controlled
optimizer behavior transfers beyond the current CIFAR-10 `SmallCIFARCNN`
setup.

## Motivation

Recent CIFAR-10 Adam experiments showed that the controller is sensitive to
Adam-scale alpha bounds. The best current 20-epoch 20k/5k CIFAR-10 result came
from the balanced raw-rho setting:

```text
alpha_max = 1.5e-3
rho_star = 0.80
rho_beta = 0.90
kp = 0.02
min_alpha_factor = 0.98
max_alpha_factor = 1.015
```

This setting gave `controlled_raw_rho` a best test accuracy of `0.8232`, just
above the fixed Adam-direction baseline at `0.8220`. More conservative settings
were stable but too cautious, while a more open `2e-3` cap did not improve the
result.

However, this evidence is still narrow. It comes mostly from one dataset,
one CNN architecture, and one subset size. The next step should test whether
the method is robust across architectures and datasets.

## Core Questions

1. Does the controller help only the current `SmallCIFARCNN`, or does it
   transfer to other architectures?
2. Does the same Adam-scale controller setting work for easier and harder image
   datasets?
3. Does the extra same-minibatch forward pass remain manageable when the model
   architecture changes?
4. Are raw-rho, EMA, and EMA-trust variants consistently different, or are the
   current differences architecture-specific?
5. Do accepted-rate, rho, and alpha diagnostics explain performance changes
   across settings?

## Recommended Test Order

### 1. CIFAR-10 With A LeNet-Style CNN

Purpose:

- Establish a simple architecture sanity check.
- Compare against the classic small CNN the user referenced earlier.
- Determine whether our current `SmallCIFARCNN` results depend on BatchNorm,
  larger width, or deeper convolutional blocks.

Suggested architecture:

```text
Input: 3 x 32 x 32
Conv2d(3 -> 6, kernel_size=5)
ReLU
MaxPool2d(2, 2)
Conv2d(6 -> 16, kernel_size=5)
ReLU
MaxPool2d(2, 2)
Flatten
Linear(16 * 5 * 5 -> 120)
ReLU
Linear(120 -> 84)
ReLU
Linear(84 -> 10)
```

Recommended run:

```text
dataset = CIFAR-10
train_subset = 20000
test_subset = 5000
epochs = 20
batch_size = 128
lr = 1e-3
optimizer variants = vanilla_adam, fixed_adam_direction, controlled_raw_rho,
                     controlled_ema, controlled_ema_trust
controller = current balanced Adam-scale setting
```

Expected value:

- If the controller still helps, the effect is not tied to the current larger
  CNN.
- If the controller fails, we learn that architecture scale/BatchNorm strongly
  affects the controller signal.

### 2. CIFAR-10 With A Small ResNet

Purpose:

- Test a stronger and more realistic CIFAR architecture.
- Check whether the same controller works when the baseline is better tuned and
  the loss landscape differs from the current CNN.

Suggested architecture:

- CIFAR-style ResNet-18 or a smaller ResNet variant.
- Use CIFAR-compatible first convolution, e.g. `3x3`, stride `1`, no initial
  ImageNet-style max pool.

Recommended run:

```text
dataset = CIFAR-10
train_subset = 20000
test_subset = 5000
epochs = 20 initially, then 40 if promising
batch_size = 128
lr = 1e-3 for Adam baseline
controller = current balanced Adam-scale setting
```

Expected value:

- This is the most important next architecture test.
- If controlled raw-rho remains competitive on ResNet, the method becomes much
  more credible.

### 3. Fashion-MNIST With A CNN

Purpose:

- Add an easier image task that is still convolutional.
- Separate image-model behavior from the existing Fashion-MNIST MLP results.
- Run faster than CIFAR-10 while still testing neural-network dynamics.

Suggested architecture:

- Small grayscale CNN with two convolution/pooling blocks and one or two linear
  layers.

Recommended run:

```text
dataset = Fashion-MNIST
train_subset = 10000 or 20000
test_subset = 5000
epochs = 20
batch_size = 128
lr = 1e-3
controller = current balanced Adam-scale setting, plus existing Fashion-MNIST
             tuned trust-region settings if needed
```

Expected value:

- If the controller improves CNN Fashion-MNIST but not CIFAR-10, the bottleneck
  may be task difficulty/generalization rather than optimizer mechanics.

### 4. CIFAR-100

Purpose:

- Use the same image size and preprocessing family as CIFAR-10, but a harder
  100-class target.
- Test whether controller behavior changes when generalization is harder and
  gradients are noisier.

Recommended run:

```text
dataset = CIFAR-100
train_subset = 20000
test_subset = 5000
epochs = 20
batch_size = 128
architecture = SmallCIFARCNN or CIFAR ResNet
controller = current balanced Adam-scale setting
```

Expected value:

- Useful after CIFAR-10 ResNet support exists.
- It may expose whether same-minibatch rho tracks training progress but not
  validation performance.

### 5. SVHN

Purpose:

- Test a different RGB image distribution with the same input size.
- Check whether behavior is CIFAR-specific.

Recommended run:

```text
dataset = SVHN
train_subset = 20000
test_subset = 5000
epochs = 20
batch_size = 128
architecture = SmallCIFARCNN or CIFAR ResNet
controller = current balanced Adam-scale setting
```

Expected value:

- Good distribution-shift check after CIFAR-10 and Fashion-MNIST CNN are in
  place.

## Architecture Priority

1. LeNet-style CNN
2. CIFAR ResNet-18 or small ResNet
3. Fashion-MNIST CNN
4. Optional MLP-Mixer or tiny transformer later

The LeNet-style CNN is fastest to implement and gives an immediate sanity
check. The CIFAR ResNet is the most important credibility test.

## Controller Settings To Carry Forward

Primary Adam setting:

```text
alpha_min = 7e-4
alpha_max = 1.5e-3
kp = 0.02
rho_star = 0.80
rho_beta = 0.90
min_alpha_factor = 0.98
max_alpha_factor = 1.015
trust_rho_threshold = 0.86
trust_alpha_threshold = 1e-3
trust_expand_factor = 1.02
```

Secondary setting if EMA looks too slow:

```text
alpha_min = 7e-4
alpha_max = 1.5e-3
kp = 0.02
rho_star = 0.80
rho_beta = 0.85
min_alpha_factor = 0.98
max_alpha_factor = 1.015
```

Current evidence says the secondary faster-EMA setting makes EMA ramp earlier
but does not improve the current CIFAR result. Keep it as a diagnostic, not the
main recommendation.

## Metrics To Report For Every Run

Every benchmark should report:

- final train loss and train accuracy
- final test loss and test accuracy
- best test accuracy and best epoch
- loss and accuracy versus epoch
- loss and accuracy versus optimizer step
- loss and accuracy versus wall-clock time
- final alpha and alpha trajectory
- rho trajectory
- accepted rate
- number of backtracked steps
- trust-region expansion count
- elapsed wall-clock time and relative time versus vanilla Adam
- full model architecture and parameter count
- dataset subset size, transforms, seed, batch size, and learning rate
- for checkpointed Adam image runs, optional PCA training-trajectory plots via
  `controlled_adam_project/examples/plot_pca_training_trajectory.py`

## First Concrete Implementation Task

Add model selection support to `controlled_adam_project/examples/run_mnist_demo.py`:

```text
--model auto
--model mlp
--model cnn
--model lenet_cifar
--model resnet_cifar
--model fashion_cnn
```

Then run the first LeNet-style CIFAR-10 benchmark:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --model lenet_cifar \
  --epochs 20 \
  --train-subset 20000 \
  --test-subset 5000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 7e-4 \
  --controlled-alpha-max 1.5e-3 \
  --controlled-kp 0.02 \
  --controlled-rho-star 0.80 \
  --controlled-rho-beta 0.90 \
  --controlled-min-alpha-factor 0.98 \
  --controlled-max-alpha-factor 1.015 \
  --controlled-trust-rho-threshold 0.86 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 1.02 \
  --print-every 1 \
  --checkpoint-every 0 \
  --output-dir outputs/cifar10_20k_5k_20epochs_lenet_adam_balanced
```


## First Test Result

We ran the first LeNet-style CIFAR-10 benchmark with the balanced Adam-scale
setting. The outcome was informative:

- vanilla Adam and fixed Adam-direction both finished at `0.5868` final test accuracy;
- `controlled_raw_rho` finished at `0.5656`;
- `controlled_ema` finished at `0.5620`;
- `controlled_ema_trust` finished at `0.5722`.

The run was much faster than `SmallCIFARCNN`: roughly `6-7s` per epoch per
variant rather than about `50-65s`. But the controller mostly stayed near the
alpha floor and did not help this weaker architecture. This suggests the
controller is not automatically architecture-agnostic and that the current
balanced setting is better matched to the larger BatchNorm CNN than to LeNet.

## Runtime Guidance

The LeNet experiment is cheap enough for full ablations on the current CPU
setup. A CIFAR ResNet is likely much longer than the current `SmallCIFARCNN`,
which already takes about `50-65s` per epoch per variant. A full 20-epoch,
5-variant ResNet ablation could take many hours.

Recommended staging before any large ResNet run:

1. Smoke run: 3 epochs, 5k train / 1k test, full ablation. Completed.
2. Medium run: 10 epochs, 10k train / 2k test, preferably only
   `vanilla_adam`, `fixed_adam_direction`, and `controlled_raw_rho`.
3. Large run: only if the first two stages show reasonable runtime and signal.

For the current machine, a Fashion-MNIST CNN is likely a better next test than
CIFAR ResNet because it adds architecture variation without the same runtime
risk.

## CIFAR ResNet Smoke Result

Added `--model resnet_cifar`, implemented as `SmallCIFARResNet`:

```text
Input: 3 x 32 x 32
Conv2d(3 -> 16, kernel_size=3, stride=1, padding=1)
BatchNorm2d
ReLU
2 residual blocks at width 16
2 residual blocks at width 32, first block stride 2
2 residual blocks at width 64, first block stride 2
AdaptiveAvgPool2d(1, 1)
Linear(64 -> 10)
```

The model has `175258` trainable parameters.

Smoke run:

```text
dataset = CIFAR-10
model = resnet_cifar
train_subset = 5000
test_subset = 1000
epochs = 3
batch_size = 128
seed = 123
lr = 1e-3
controller = balanced Adam-scale setting
output = controlled_adam_project/outputs/cifar10_resnet_smoke_5k_1k_3epoch_balanced/
```

Controller settings:

```text
alpha_min = 1e-3
alpha_max = 1.5e-3
rho_star = 0.80
rho_beta = 0.90
kp = 0.02
min_alpha_factor = 0.98
max_alpha_factor = 1.015
```

Final test accuracy:

```text
vanilla_adam          0.3610
fixed_adam_direction 0.3730
controlled_raw_rho   0.4100
controlled_ema       0.3800
controlled_ema_trust 0.3800
```

Diagnostics:

- all controlled variants accepted every step;
- final mean alpha stayed near `1e-3`;
- raw-rho showed the strongest early signal;
- the run completed successfully, so the staged ResNet path is viable.

## CIFAR ResNet 20-Epoch Staged Result

After the smoke run, we ran the same compact ResNet on a larger CIFAR-10
subset for 20 epochs:

```text
dataset = CIFAR-10
model = resnet_cifar
train_subset = 10000
test_subset = 2000
epochs = 20
batch_size = 128
seed = 123
lr = 1e-3
controller = balanced Adam-scale setting
output = controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced/
```

Final / best test accuracy:

```text
vanilla_adam          final 0.6915  best 0.6915 at epoch 20
fixed_adam_direction final 0.6875  best 0.6975 at epoch 19
controlled_raw_rho   final 0.7395  best 0.7395 at epoch 20
controlled_ema       final 0.7135  best 0.7135 at epoch 20
controlled_ema_trust final 0.7135  best 0.7135 at epoch 20
```

Final diagnostics:

```text
fixed_adam_direction mean_alpha 0.001000  mean_rho 0.8853  accepted 1.0000
controlled_raw_rho   mean_alpha 0.001500  mean_rho 0.8821  accepted 1.0000
controlled_ema       mean_alpha 0.001500  mean_rho 0.8768  accepted 1.0000
controlled_ema_trust mean_alpha 0.001500  mean_rho 0.8768  accepted 1.0000
```

Interpretation:

- This is the strongest Adam-controller architecture-transfer signal so far:
  raw-rho beat vanilla Adam and fixed Adam-direction by a visible margin on the
  10k/2k ResNet run.
- The controller overhead was manageable enough for this staged CPU benchmark,
  but controlled variants took longer than vanilla because they include the
  same-minibatch trial forward pass and diagnostics.
- All controlled variants accepted every step, so the useful behavior here was
  mainly alpha adaptation rather than rejection/backtracking.
- EMA smoothing reduced the final gain relative to raw-rho. EMA-trust was
  identical to EMA, meaning the trust expansion path was not active under this
  cap.
- Later diagnostic check: `controlled_ema_trust` recorded `0/1580` trust
  expansions for each balanced ResNet seed (`123`, `456`, and `789`). The
  specific reason was parameter mismatch: `alpha_min=1e-3` but
  `trust_region_alpha_threshold=1e-4`, so the trust trigger was below the
  allowed alpha floor.

Recommended next ResNet tests:

1. Multi-seed 20-epoch 10k/2k ResNet run if runtime is acceptable.
2. Cap ablation around `alpha_max = 1.25e-3`, `1.5e-3`, and `1.75e-3` for
   raw-rho and EMA only.
3. A reduced-variant longer run, such as vanilla Adam, fixed Adam-direction,
   and raw-rho for 40 epochs, to see whether the raw-rho advantage persists.

## CIFAR ResNet Follow-Up Parameter Tests

Candidate 1 higher-cap test:

```text
alpha_min = 1e-3
alpha_max = 1.75e-3
rho_star = 0.82
rho_beta = 0.90
kp = 0.015
factor clip = [0.98, 1.012]
```

Result on the same 20-epoch 10k/2k ResNet setup:

```text
controlled_raw_rho   final 0.7120  best 0.7120
controlled_ema       final 0.6695  best 0.6915
controlled_ema_trust final 0.6695  best 0.6915
```

Conclusion: the higher cap was worse than the balanced `alpha_max=1.5e-3`
setting.

Stronger fixed-LR control:

```text
lr = 1.5e-3
alpha_min = 1e-3
alpha_max = 1.5e-3
rho_star = 0.80
rho_beta = 0.90
kp = 0.02
factor clip = [0.98, 1.015]
```

Result:

```text
vanilla_adam          final 0.7065  best 0.7065
fixed_adam_direction final 0.6905  best 0.7040
controlled_raw_rho   final 0.6985  best 0.6985
controlled_ema       final 0.7250  best 0.7250
controlled_ema_trust final 0.7250  best 0.7250
```

Conclusion: a larger fixed Adam learning rate improves vanilla Adam, but it
does not explain the original raw-rho `0.7395` result. Starting at the cap is
not equivalent to ramping there from `1e-3`. The next most informative test is
therefore multi-seed validation of the original balanced `lr=1e-3`,
`alpha_max=1.5e-3` schedule.

## CIFAR ResNet Balanced Multi-Seed Result

The original balanced setting was evaluated on seeds `123`, `456`, and `789`.

Three-seed aggregate:

```text
vanilla_adam          final 0.6887 +/- 0.0145  best 0.6998 +/- 0.0076
fixed_adam_direction final 0.6938 +/- 0.0150  best 0.7118 +/- 0.0148
controlled_raw_rho   final 0.7150 +/- 0.0226  best 0.7235 +/- 0.0147
controlled_ema       final 0.7083 +/- 0.0140  best 0.7190 +/- 0.0055
controlled_ema_trust final 0.7083 +/- 0.0140  best 0.7190 +/- 0.0055
```

Conclusion:

- Controlled variants retain a modest mean advantage over vanilla and fixed
  Adam-direction.
- Raw-rho has the best mean final and best accuracy, but its seed-to-seed
  variance is larger.
- EMA and EMA-trust are slightly lower on mean accuracy but more stable.
- EMA-trust was exactly identical to EMA because trust expansion did not fire:
  `0/1580` expansions for each of the three seeds. This should be interpreted
  as a dormant trust hook, not as a real test of the trust-region mechanism.
- The original seed `123` raw-rho result was unusually strong, so future claims
  should use multi-seed aggregates rather than single-run peaks.
- Next useful trust-specific test: keep the Adam-scale floor but move
  `trust_region_alpha_threshold` near that floor, for example `1e-3` to
  `1.05e-3`, and use a gentle expansion factor such as `1.1` or `1.2`.

## First Fashion-MNIST CNN Result

We ran the proposed Fashion-MNIST CNN benchmark with the balanced Adam-scale
setting. This is the best signal so far for architecture transfer beyond the
current CIFAR CNN, because it is fast enough to iterate on but still exercises a
convolutional network.

Final test accuracy:

```text
vanilla_adam          0.8858
fixed_adam_direction 0.8866
controlled_raw_rho   0.8870
controlled_ema       0.8844
controlled_ema_trust 0.8844
```

Interpretation:

- The run was very fast, about `4.5-5.6s` per epoch per variant.
- The controller did not collapse to the alpha floor the way it did on LeNet.
- Raw-rho slightly beat the fixed baseline on final accuracy, but the margin is
  tiny.
- EMA and EMA-trust were close but did not improve on raw-rho.
- This is a more encouraging architecture-transfer signal than LeNet, but not a
  decisive win.

## Fashion-MNIST CNN Multi-Seed Result

We then ran a 5-seed Fashion-MNIST CNN benchmark on the same 20k/5k split size
using seeds `123`, `456`, `789`, `2024`, and `2025`. This tested whether the
single-seed raw-rho edge survived seed variation.

Aggregate final test accuracy:

```text
vanilla_adam          0.8945 +/- 0.0061
fixed_adam_direction 0.8946 +/- 0.0062
controlled_raw_rho   0.8889 +/- 0.0071
controlled_ema       0.8872 +/- 0.0064
controlled_ema_trust 0.8884 +/- 0.0064
```

Aggregate best test accuracy:

```text
vanilla_adam          0.8962 +/- 0.0041
fixed_adam_direction 0.8962 +/- 0.0045
controlled_raw_rho   0.8918 +/- 0.0047
controlled_ema       0.8912 +/- 0.0038
controlled_ema_trust 0.8909 +/- 0.0034
```

Timing:

```text
vanilla_adam          1.00x
fixed_adam_direction 1.19x
controlled_raw_rho   1.22x
controlled_ema       1.24x
controlled_ema_trust 1.24x
```

Interpretation:

- The single-seed raw-rho edge did not survive multi-seed testing.
- Vanilla Adam and fixed Adam-direction were effectively tied and better than
  the controlled variants.
- Controlled variants kept alpha near `6.8e-4`, below the initial Adam-scale
  `1e-3`, suggesting this balanced CIFAR-derived setting is too conservative
  for Fashion-MNIST CNN.
- The overhead was moderate but not free, around `1.22x-1.24x` for controlled
  variants.
- Fashion-MNIST CNN remains useful as a cheap testbed, but it does not support
  the current controller setting as a win.

## Next Fashion-MNIST CNN Parameter Candidates

The multi-seed Fashion-MNIST CNN result suggests that the CIFAR-derived
balanced setting is too conservative for this model. The controlled variants
ended near `alpha ~= 6.8e-4`, below the Adam-scale `1e-3` baseline. The next
parameter sweep should therefore test whether keeping alpha closer to the Adam
scale fixes the underperformance.

Candidate A: Adam-scale floor.

```text
alpha_min = 9e-4
alpha_max = 1.5e-3
rho_star = 0.75
rho_beta = 0.90
kp = 0.02
min_alpha_factor = 0.98
max_alpha_factor = 1.015
```

Purpose:

- Cleanest hypothesis test.
- Prevents the controller from training mostly below the Adam baseline scale.
- Recommended first, ideally on three seeds before repeating five seeds.

Candidate B: faster recovery.

```text
alpha_min = 8e-4
alpha_max = 1.5e-3
rho_star = 0.70
rho_beta = 0.90
kp = 0.03
min_alpha_factor = 0.98
max_alpha_factor = 1.02
```

Purpose:

- Lets alpha recover upward faster.
- Still keeps the same `1.5e-3` cap that worked best on CIFAR.
- More aggressive than Candidate A, so use after A.

Candidate C: near-fixed Adam with mild control.

```text
alpha_min = 9.5e-4
alpha_max = 1.25e-3
rho_star = 0.75
rho_beta = 0.90
kp = 0.01
min_alpha_factor = 0.995
max_alpha_factor = 1.005
```

Purpose:

- Conservative "do not hurt Adam" test.
- Checks whether the controller can provide diagnostics and mild correction
  without drifting away from the baseline Adam scale.

## Candidate C Fashion-MNIST CNN Result

We ran Candidate C on three seeds (`123`, `456`, `789`) for Fashion-MNIST CNN,
20k/5k, 20 epochs. Candidate C was the near-fixed Adam controller.

Candidate C settings:

```text
alpha_min = 9.5e-4
alpha_max = 1.25e-3
rho_star = 0.75
rho_beta = 0.90
kp = 0.01
min_alpha_factor = 0.995
max_alpha_factor = 1.005
```

Three-seed aggregate final test accuracy:

```text
vanilla_adam          0.8946 +/- 0.0084
fixed_adam_direction 0.8960 +/- 0.0084
controlled_raw_rho   0.8951 +/- 0.0089
controlled_ema       0.8937 +/- 0.0092
controlled_ema_trust 0.8937 +/- 0.0092
```

Three-seed aggregate best test accuracy:

```text
vanilla_adam          0.8965 +/- 0.0057
fixed_adam_direction 0.8977 +/- 0.0056
controlled_raw_rho   0.8978 +/- 0.0062
controlled_ema       0.8974 +/- 0.0063
controlled_ema_trust 0.8974 +/- 0.0063
```

Compared with Candidate A on the same three seeds, Candidate C nudged raw-rho
slightly higher and kept alpha even closer to the Adam baseline, but the EMA
variants slipped a little. It remained below fixed Adam-direction on mean final
accuracy.

Interpretation:

- Candidate C confirms that simply pinning alpha near Adam scale is not enough
  to beat fixed Adam-direction on Fashion-MNIST CNN.
- Raw-rho appears to be the most resilient controlled variant in this regime.
- The next Fashion-MNIST CNN test should probably move to a different dataset
  or architecture rather than tuning this CNN endlessly.

## Candidate B Fashion-MNIST CNN Result

We also ran Candidate B on the same three seeds. This was the more aggressive
recovery setting.

Candidate B settings:

```text
alpha_min = 8e-4
alpha_max = 1.5e-3
rho_star = 0.70
rho_beta = 0.90
kp = 0.03
min_alpha_factor = 0.98
max_alpha_factor = 1.02
```

Three-seed aggregate final test accuracy:

```text
vanilla_adam          0.8946 +/- 0.0084
fixed_adam_direction 0.8960 +/- 0.0084
controlled_raw_rho   0.8921 +/- 0.0064
controlled_ema       0.8930 +/- 0.0078
controlled_ema_trust 0.8930 +/- 0.0078
```

Three-seed aggregate best test accuracy:

```text
vanilla_adam          0.8965 +/- 0.0057
fixed_adam_direction 0.8977 +/- 0.0056
controlled_raw_rho   0.8941 +/- 0.0052
controlled_ema       0.8949 +/- 0.0063
controlled_ema_trust 0.8949 +/- 0.0063
```

Interpretation:

- Candidate B was too aggressive.
- It pushed raw-rho below both Candidate A and Candidate C.
- This is a strong hint that the next step should move away from Fashion-MNIST
  CNN tuning and toward the staged CIFAR ResNet smoke test.

## Decision Criteria

The controller looks promising if it:

- beats or matches fixed Adam-direction on best test accuracy;
- stays within modest wall-clock overhead;
- does not rely on frequent backtracking;
- has stable alpha/rho diagnostics;
- transfers across at least two architectures.

The controller looks architecture-specific if:

- it only helps the current `SmallCIFARCNN`;
- it consistently loses on LeNet and ResNet;
- alpha saturates at the cap but validation does not improve;
- EMA and trust variants repeatedly collapse to the same trajectory without
  distinct behavior.

However, identical EMA and EMA-trust trajectories only count as evidence
against the trust-region idea if the trust branch was reachable. If
`trust_region_alpha_threshold` is below `alpha_min`, the branch is unreachable by
construction and the run only tests EMA-rho.

## Current Recommendation

Do not keep tuning only the current `SmallCIFARCNN` CIFAR-10 setup. The first
LeNet architecture test did not favor the controller, and the Fashion-MNIST CNN
candidate sweep now shows a clear pattern: Candidate A was the best of the
three, Candidate C was close behind, and Candidate B was worse. The CIFAR
ResNet smoke run completed cleanly and showed a positive raw-rho early signal.
The next useful experiment is the staged 10-epoch 10k/2k ResNet run with the
same balanced settings.
