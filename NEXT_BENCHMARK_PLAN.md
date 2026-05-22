# Next Benchmark Plan

Last updated: 2026-05-22

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

1. Smoke run: 3 epochs, 5k train / 1k test, full ablation.
2. Medium run: 10 epochs, 10k train / 2k test, preferably only
   `vanilla_adam`, `fixed_adam_direction`, and `controlled_raw_rho`.
3. Large run: only if the first two stages show reasonable runtime and signal.

For the current machine, a Fashion-MNIST CNN is likely a better next test than
CIFAR ResNet because it adds architecture variation without the same runtime
risk.

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

## Current Recommendation

Do not keep tuning only the current `SmallCIFARCNN` CIFAR-10 setup. The first
LeNet architecture test did not favor the controller, and the first
Fashion-MNIST CNN multi-seed run showed that the CIFAR-derived setting was too
conservative. The next cheapest useful experiment is Candidate A on
Fashion-MNIST CNN, probably for three seeds first. CIFAR ResNet should still be
staged as a smoke run before any large ablation.
