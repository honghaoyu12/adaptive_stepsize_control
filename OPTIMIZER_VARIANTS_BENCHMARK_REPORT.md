# Optimizer Variant Benchmark Report

Last updated: 2026-05-21

Repository root:

```text
/Users/honghaoyu/adaptive_stepsize_control
```

This document summarizes the five optimizer variants implemented in the Adam
and Muon neural-network benchmark runners, explains what each variant is meant
to isolate, and records the main empirical results from the Fashion-MNIST and
CIFAR-10 benchmarks run so far.

It also includes a short section on the deterministic 2D function experiments.
Those function experiments predate the full five-variant neural ablation and
therefore compare only vanilla Adam against controlled Adam, but they are still
useful for understanding the controller behavior in clean deterministic
settings.

This report is intentionally self-contained. It includes the relevant
implementation paths, benchmark commands, dataset assumptions, model
architectures, metric definitions, and interpretation. It does not require
reading the README, handoff, development log, or conversation log.

## 0. Self-Contained Project Map

### 0.1 Relevant Repository Layout

```text
adaptive_stepsize_control/
├── OPTIMIZER_VARIANTS_BENCHMARK_REPORT.md
├── controlled_adam_project/
│   ├── examples/
│   │   ├── run_demo.py                 # deterministic 2D Adam demos
│   │   └── run_mnist_demo.py           # Adam neural benchmark runner
│   ├── src/controlled_adam/
│   │   ├── objectives.py               # deterministic 2D objectives
│   │   ├── optimizers.py               # NumPy vanilla/controlled Adam
│   │   ├── plotting.py                 # deterministic objective plots
│   │   └── torch_optimizers.py         # PyTorch controlled Adam
│   ├── data/
│   │   ├── MNIST/
│   │   ├── FashionMNIST/
│   │   └── cifar-10-batches-py/
│   └── outputs/                        # Adam benchmark outputs
├── controlled_muon_project/
│   ├── examples/
│   │   ├── run_matrix_quadratic_demo.py
│   │   └── run_mnist_demo.py           # Muon neural benchmark runner
│   ├── src/controlled_muon/
│   │   ├── objectives.py
│   │   ├── optimizers.py
│   │   ├── orthogonalization.py
│   │   ├── plotting.py
│   │   └── torch_optimizers.py         # PyTorch controlled Muon
│   └── outputs/                        # Muon benchmark outputs
└── fashion/                            # local flat Fashion-MNIST IDX gzip files
```

The root project also contains the original controlled gradient-descent demo,
but the five-variant optimizer comparison discussed here lives in the Adam and
Muon subprojects.

### 0.2 Implementation Files

Adam implementation:

```text
controlled_adam_project/src/controlled_adam/torch_optimizers.py
controlled_adam_project/examples/run_mnist_demo.py
```

Muon implementation:

```text
controlled_muon_project/src/controlled_muon/torch_optimizers.py
controlled_muon_project/src/controlled_muon/orthogonalization.py
controlled_muon_project/examples/run_mnist_demo.py
```

Deterministic 2D Adam implementation:

```text
controlled_adam_project/src/controlled_adam/optimizers.py
controlled_adam_project/examples/run_demo.py
```

### 0.3 Output File Conventions

Each neural benchmark output directory usually contains:

```text
<dataset>_epoch_metrics.csv
<dataset>_loss.png
<dataset>_train_loss.png
<dataset>_train_test_loss.png
<dataset>_accuracy.png
<dataset>_loss_vs_steps.png
<dataset>_accuracy_vs_steps.png
<dataset>_loss_vs_time.png
<dataset>_accuracy_vs_time.png
<dataset>_controlled_alpha.png
<dataset>_controlled_rho.png
<dataset>_<optimizer>_step_diagnostics.csv
run_metadata.json
run_metadata.txt
```

Some older output directories predate `run_metadata.json`,
`elapsed_seconds`, `optimizer_steps`, and the `*_vs_time.png` /
`*_vs_steps.png` plots. Their accuracy/loss data are still valid, but exact
wall-clock comparisons require rerunning with the current scripts.

### 0.4 How To Reproduce The Main Benchmarks

All commands below assume the repository root is:

```text
/Users/honghaoyu/adaptive_stepsize_control
```

Use `MPLCONFIGDIR=/private/tmp` on this machine to avoid Matplotlib cache
permission issues.

Fashion-MNIST Adam, tuned 40-epoch ablation:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 40 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_40epochs_ablation_tuned1
```

CIFAR-10 Adam, 5k/1k subset, tight alpha cap:

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
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

CIFAR-10 Adam, larger 20k/5k subset with checkpoints and live progress:

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

Fashion-MNIST Muon, 20-epoch ablation:

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

CIFAR-10 Muon, 40-epoch ablation:

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

Deterministic 2D Adam function benchmark:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_demo.py
```

### 0.5 How To Verify The Code

Adam tests:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Muon tests:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Recent verification while editing this report:

```text
controlled_adam_project: 9 passed
controlled_muon_project: 6 passed
```

### 0.6 Glossary

`alpha`

: The scalar global multiplier applied to the base optimizer direction.

`p_t`

: The direction proposed by Adam moments or Muon orthogonalization.

`rho`

: Actual decrease divided by predicted decrease on the same objective or
  minibatch.

`rho_star`

: Target rho value used by the proportional controller.

`rho_ema`

: Exponential moving average of rho used to smooth stochastic minibatch noise.

`accepted_rate`

: Fraction of minibatch trial steps accepted during an epoch.

`fixed direction`

: The controlled optimizer direction path with `alpha` fixed at `lr`. This is
  a baseline for separating direction implementation effects from adaptive
  alpha effects.

`trust-region expansion`

: The recovery rule that expands alpha more aggressively when rho is good and
  alpha is near a tiny threshold.

`best test accuracy`

: Highest test accuracy reached at any epoch in a run.

`final test accuracy`

: Test accuracy at the last epoch.

## 1. Core Control Idea

The controller is an outer-loop scalar step-size controller. A base optimizer
chooses a search direction `p_t`. The controller chooses a scalar multiplier
`alpha_t`, takes a trial step, and compares the actual decrease with the
first-order predicted decrease.

For a deterministic objective:

```text
theta_trial = theta_t + alpha_t p_t
predicted_decrease = -alpha_t * grad f(theta_t)^T p_t
actual_decrease    = f(theta_t) - f(theta_trial)
rho_t              = actual_decrease / predicted_decrease
```

For minibatch training, the same idea is applied on the current minibatch
`B_t`:

```text
rho_t =
    [f_Bt(theta_t) - f_Bt(theta_t + alpha_t p_t)]
    / [-alpha_t * grad f_Bt(theta_t)^T p_t]
```

The same minibatch must be used before and after the trial step. If the after
step loss used a different minibatch, random minibatch variation would be mixed
with true optimization progress, and `rho_t` would stop being a reliable
control signal.

The proportional controller updates the global multiplier roughly as:

```text
alpha_{t+1} = alpha_t * exp(kp * (rho_control - rho_star))
```

where `rho_control` is either raw `rho_t` or an exponential moving average of
`rho_t`, depending on the variant.

## 2. The Five Neural Optimizer Variants

The neural benchmark runners implement the same five-way ablation for Adam and
Muon directions.

### 2.1 Vanilla Base Optimizer

Names:

- Adam runner: `vanilla_adam`
- Muon runner: `vanilla_muon`

Purpose:

- Establish the ordinary baseline for the chosen direction family.
- Does not perform same-minibatch trial loss measurement.
- Does not adapt a separate outer-loop global multiplier.

Adam behavior:

- Uses `torch.optim.Adam` with fixed learning rate `lr`.

Muon behavior:

- Uses the project’s educational Muon-style update implementation with fixed
  learning rate `lr`.
- The implementation orthogonalizes parameter update directions. It is useful
  for research experiments but CPU-heavy, especially for CIFAR-10.

### 2.2 Fixed Direction Through the Controller Path

Names:

- Adam runner: `fixed_adam_direction`
- Muon runner: `fixed_muon_direction`

Purpose:

- Isolate the cost and behavior of using the controlled optimizer direction
  code path without allowing alpha adaptation.
- Uses the same direction machinery and same-minibatch measurement as the
  controlled variants, but keeps `alpha` fixed.
- This is a key control baseline: if it matches vanilla, the direction
  implementation is faithful; if it differs, the difference is in the direction
  implementation or measurement path, not adaptive control.

Configuration:

```text
alpha = lr
kp = 0
min_alpha_factor = 1
max_alpha_factor = 1
reject_bad_steps = False
trust_region_expand = False
```

### 2.3 Controlled Raw Rho

Name:

- `controlled_raw_rho`

Purpose:

- Test the most direct controller: update `alpha` using the current minibatch’s
  raw actual-over-predicted ratio.

Behavior:

```text
rho_control = rho_t
alpha_{t+1} = alpha_t * exp(kp * (rho_t - rho_star))
```

Advantages:

- Very responsive.
- Can adapt rapidly when the local predicted decrease is too conservative.

Risks:

- Sensitive to one-minibatch noise.
- Can grow too aggressively if same-minibatch progress is good but validation
  generalization is not improved.

### 2.4 Controlled EMA Rho

Name:

- `controlled_ema`

Purpose:

- Smooth the control signal so one minibatch does not dominate alpha updates.

Behavior:

```text
rho_ema_t = beta * rho_ema_{t-1} + (1 - beta) * rho_t
rho_control = rho_ema_t
```

Advantages:

- More stable than raw rho.
- Better starting point for stochastic training because it separates persistent
  optimization progress from one-batch accidents.

Risks:

- Can still saturate the alpha cap if the smoothed same-minibatch ratio remains
  high.
- Can respond too slowly if the optimization regime changes.

### 2.5 Controlled EMA Rho With Trust-Region Expansion

Name:

- `controlled_ema_trust`

Purpose:

- Add a recovery/expansion rule inspired by classical trust-region methods.
- If rho is good but alpha is very small, the controller should not remain
  trapped at a tiny global step size.

Behavior:

- Uses EMA-smoothed rho.
- If:
  - the step was accepted without backtracking,
  - the smoothed rho is above a threshold,
  - and the current alpha is below a small threshold,
  then force a larger expansion factor for the next alpha.

Interpretation:

- This is not a full trust-region method. It is a trust-region-style recovery
  mechanism grafted onto the scalar alpha controller.
- It was helpful in some Fashion-MNIST Adam runs where alpha otherwise became
  too small.
- It had little or no effect in several CIFAR and Muon runs because alpha was
  already large or already at the cap.

## 3. Benchmark Setup

### 3.1 Datasets

The neural benchmarks used:

- Fashion-MNIST, usually `4096` train / `1024` test subset.
- CIFAR-10, initially `5000` train / `1000` test subset.
- Larger CIFAR-10 Adam run with `20000` train / `5000` test subset.

These are subset benchmarks unless explicitly stated otherwise. They are useful
for optimizer development, but they are not full-dataset claims.

### 3.2 Models

Fashion-MNIST uses `SmallMLP`:

```text
Flatten
Linear(28*28 -> 128)
ReLU
Linear(128 -> 10)
```

Approximate trainable parameters:

```text
101,770
```

CIFAR-10 uses `SmallCIFARCNN`:

```text
Conv-BN-ReLU block: 3 -> 32 -> 32
MaxPool
Conv-BN-ReLU block: 32 -> 64 -> 64
MaxPool
Conv-BN-ReLU block: 64 -> 128 -> 128
MaxPool
Flatten
Linear(128*4*4 -> 256)
ReLU
Linear(256 -> 10)
```

Approximate trainable parameters:

```text
815,018
```

For CIFAR-10, BatchNorm uses `track_running_stats=False`. This is important
because controlled variants repeatedly evaluate same-minibatch trial losses;
trial evaluations should not mutate BatchNorm running statistics.

### 3.3 CIFAR-10 Transforms

Training transform:

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(CIFAR10_MEAN, CIFAR10_STD)
```

Evaluation transform:

```text
ToTensor()
Normalize(CIFAR10_MEAN, CIFAR10_STD)
```

Train metrics use a deterministic evaluation transform on the same train subset
indices, rather than train-time random augmentation.

### 3.4 Metrics Recorded

The epoch metrics CSV records:

```text
optimizer
epoch
train_loss
train_accuracy
test_loss
test_accuracy
elapsed_seconds
optimizer_steps
mean_alpha
mean_rho
accepted_rate
```

Older benchmark CSVs may not contain `elapsed_seconds` and `optimizer_steps`
because wall-clock/step-axis tracking was added later.

The runners now plot:

- loss vs epoch
- training loss vs epoch
- train/test loss vs epoch
- accuracy vs epoch
- loss and accuracy vs optimizer steps
- loss and accuracy vs wall-clock time
- controlled alpha diagnostics
- controlled rho diagnostics

## 4. High-Level Findings

### 4.1 Adam Direction

The controlled Adam story is mixed and highly sensitive to alpha bounds.

Main observations:

- On deterministic 2D objectives, controlled Adam often improves dramatically
  over fixed-alpha vanilla Adam on curved or ill-conditioned objectives.
- On Fashion-MNIST, trust-region recovery can help controlled Adam avoid tiny
  alpha collapse and can beat vanilla/fixed Adam on the best 40-epoch tuned run.
- On longer Fashion-MNIST, controlled variants often reduce overfitting, but
  the best final test accuracy and best peak accuracy are not always the same.
- On CIFAR-10, alpha caps are crucial.
- A too-large alpha cap, such as `5e-2`, lets controlled Adam grow far beyond
  the useful Adam scale and hurts accuracy.
- A tight cap around `1e-3` to `1.5e-3` works much better.
- On the larger 20k/5k CIFAR-10 run, fixed Adam-direction had the best peak
  accuracy; controlled variants saturated the `1.5e-3` cap and did not beat
  fixed Adam-direction.

### 4.2 Muon Direction

Controlled Muon has been more consistently positive on the subset benchmarks
run so far.

Main observations:

- Vanilla/fixed Muon with `lr=1e-3` is much weaker than controlled Muon on the
  Fashion-MNIST and CIFAR-10 subset runs.
- Controlled Muon naturally grew alpha much larger than the fixed `1e-3`.
- On Fashion-MNIST, controlled Muon reached roughly `0.84` best test accuracy,
  while vanilla/fixed Muon stayed around `0.743`.
- On CIFAR-10 40 epochs, controlled Muon reached `0.731` best test accuracy,
  while vanilla/fixed Muon peaked around `0.709/0.701`.
- EMA and EMA-trust were identical in the Muon runs where trust expansion did
  not fire.

### 4.3 Same-Minibatch Rho Is Not A Generalization Metric

A recurring lesson is that same-minibatch rho is an optimization progress
signal, not a validation/generalization signal.

The larger CIFAR-10 Adam run is the clearest example:

- all controlled variants accepted 100% of steps,
- all controlled variants had healthy rho values,
- controlled variants quickly saturated the `1.5e-3` alpha cap,
- but fixed Adam-direction had the best peak validation accuracy.

So the controller can correctly detect same-minibatch improvement while still
choosing steps that do not improve held-out performance as much as a more
conservative fixed scale.

## 5. Deterministic 2D Function Results

These experiments compare vanilla Adam and controlled Adam, not all five
neural ablation variants. They are still useful because there is no minibatch
noise and the actual-over-predicted ratio has a clean interpretation.

| Objective | Steps | Vanilla Adam Final f | Controlled Adam Final f | Winner | Controlled Final Alpha | Accepted Steps |
|---|---:|---:|---:|---|---:|---:|
| quadratic | 250 | `44.02` | `6.11949e-14` | controlled | `1e-08` | `175/250` |
| rosenbrock | 3000 | `3.58707` | `0.000108333` | controlled | `0.05` | `2990/3000` |
| himmelblau | 700 | `7.20932e-20` | `0.00323636` | vanilla | `1e-08` | `683/700` |
| rastrigin | 900 | `17.9092` | `17.9092` | effectively tied | `1e-08` | `884/900` |
| beale | 1500 | `0.0197465` | `4.19317e-08` | controlled | `1e-08` | `1498/1500` |
| ackley | 1200 | `6.55965` | `6.55965` | effectively tied | `1e-08` | `1155/1200` |
| six_hump_camel | 800 | `-0.215464` | `-0.215464` | effectively tied | `1e-08` | `800/800` |
| goldstein_price | 1200 | `3` | `3` | effectively tied | `1e-08` | `444/1200` |
| easom | 1000 | `-1` | `-0.999999` | effectively tied | `1e-08` | `1000/1000` |

Interpretation:

- Controlled Adam is very strong on the anisotropic quadratic, Rosenbrock, and
  Beale objectives.
- On multimodal functions such as Rastrigin and Ackley, both methods can land
  in the same local basin.
- The controlled alpha often decays to `alpha_min` near convergence or after
  rejected steps. This is acceptable in deterministic convergence plots but
  motivated later neural experiments with EMA smoothing and trust-region-style
  recovery.

## 6. Adam Neural Benchmark Results

### 6.1 Fashion-MNIST, 20 Epoch Ablation

Output:

```text
controlled_adam_project/outputs/fashion_mnist_20epochs_ablation
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8369` | `0.8369` | 19 | `0.8997` | `0.3017 / 0.5196` | | | |
| `fixed_adam_direction` | `0.8291` | `0.8369` | 13 | `0.8967` | `0.3083 / 0.5258` | `0.001` | `0.684` | `0.906` |
| `controlled_raw_rho` | `0.7988` | `0.7998` | 15 | `0.8193` | `0.5357 / 0.6234` | `2.86e-05` | `0.990` | `1.000` |
| `controlled_ema` | `0.7988` | `0.7988` | 15 | `0.8176` | `0.5410 / 0.6269` | `2.87e-05` | `0.991` | `1.000` |
| `controlled_ema_trust` | `0.8115` | `0.8164` | 18 | `0.8391` | `0.4743 / 0.5771` | `0.000120` | `0.961` | `0.938` |

Interpretation:

- Early raw-rho and EMA controllers collapsed to a tiny alpha and underfit.
- Trust-region recovery helped relative to raw/EMA control, but vanilla Adam
  remained best in this particular 20-epoch run.

### 6.2 Fashion-MNIST, 40 Epoch Untuned Ablation

Output:

```text
controlled_adam_project/outputs/fashion_mnist_40epochs_ablation
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8350` | `0.8398` | 24 | `0.9473` | `0.1779 / 0.5467` | | | |
| `fixed_adam_direction` | `0.8389` | `0.8428` | 24 | `0.9443` | `0.1834 / 0.5396` | `0.001` | `0.656` | `0.938` |
| `controlled_raw_rho` | `0.8027` | `0.8027` | 40 | `0.8240` | `0.5181 / 0.6101` | `1.19e-05` | `0.996` | `0.969` |
| `controlled_ema` | `0.8018` | `0.8027` | 38 | `0.8232` | `0.5205 / 0.6113` | `1.19e-05` | `0.996` | `0.969` |
| `controlled_ema_trust` | `0.8252` | `0.8311` | 36 | `0.8613` | `0.4114 / 0.5394` | `0.000127` | `0.945` | `0.938` |

Interpretation:

- Fixed Adam-direction slightly beat vanilla in final and best accuracy.
- Raw/EMA still underfit due to tiny alpha.
- EMA-trust recovered meaningful progress but did not yet beat fixed Adam.

### 6.3 Fashion-MNIST, 40 Epoch Tuned Trust Run

Output:

```text
controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned1
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8350` | `0.8398` | 24 | `0.9473` | `0.1779 / 0.5467` | | | |
| `fixed_adam_direction` | `0.8389` | `0.8428` | 24 | `0.9443` | `0.1834 / 0.5396` | `0.001` | `0.656` | `0.938` |
| `controlled_raw_rho` | `0.8242` | `0.8271` | 36 | `0.8574` | `0.4176 / 0.5421` | `0.000119` | `0.938` | `0.969` |
| `controlled_ema` | `0.8281` | `0.8291` | 36 | `0.8574` | `0.4210 / 0.5446` | `0.000120` | `0.946` | `0.969` |
| `controlled_ema_trust` | `0.8467` | `0.8467` | 40 | `0.9019` | `0.2938 / 0.5007` | `0.000152` | `0.861` | `0.875` |

Diagnostics:

```text
controlled_ema_trust: 1280 steps, accepted 93.28%, alpha_max_seen 0.001915, trust_expansions 88
controlled_ema:       1280 steps, accepted 94.77%, alpha_final 0.000144, trust_expansions 0
controlled_raw_rho:   1280 steps, accepted 95.08%, alpha_final 0.000141, trust_expansions 0
```

Interpretation:

- This is the clearest Adam Fashion-MNIST win for the trust-region variant.
- EMA-trust achieved the best final and best peak test accuracy among the five
  variants.
- It also had a healthier training accuracy than raw/EMA without overfitting as
  strongly as vanilla/fixed Adam.

### 6.4 Fashion-MNIST, 40 Epoch Aggressive Tuning

Output:

```text
controlled_adam_project/outputs/fashion_mnist_40epochs_ablation_tuned2_aggressive
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8350` | `0.8398` | 24 | `0.9473` | `0.1779 / 0.5467` | | | |
| `fixed_adam_direction` | `0.8389` | `0.8428` | 24 | `0.9443` | `0.1834 / 0.5396` | `0.001` | `0.656` | `0.938` |
| `controlled_raw_rho` | `0.8389` | `0.8389` | 38 | `0.8862` | `0.3404 / 0.5129` | `0.000216` | `0.854` | `0.906` |
| `controlled_ema` | `0.8389` | `0.8389` | 40 | `0.8855` | `0.3415 / 0.5135` | `0.000220` | `0.852` | `0.906` |
| `controlled_ema_trust` | `0.8438` | `0.8438` | 40 | `0.9048` | `0.2879 / 0.5012` | `0.000230` | `0.828` | `0.969` |

Interpretation:

- More aggressive control improved raw/EMA substantially relative to the
  untuned run.
- EMA-trust remained the best controlled Adam variant, but the tuned1 setting
  was slightly better (`0.8467` vs `0.8438` final test accuracy).

### 6.5 Fashion-MNIST, 100 Epoch Tuned Run

Output:

```text
controlled_adam_project/outputs/fashion_mnist_100epochs_ablation_tuned1
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8350` | `0.8398` | 24 | `0.9934` | `0.0418 / 0.7246` | | | |
| `fixed_adam_direction` | `0.8359` | `0.8447` | 89 | `0.9946` | `0.0414 / 0.7044` | `0.001` | `-0.947` | `0.844` |
| `controlled_raw_rho` | `0.8418` | `0.8457` | 94 | `0.9016` | `0.3107 / 0.5063` | `0.000104` | `0.896` | `0.906` |
| `controlled_ema` | `0.8398` | `0.8418` | 93 | `0.9001` | `0.3121 / 0.5077` | `0.000106` | `0.902` | `0.906` |
| `controlled_ema_trust` | `0.8369` | `0.8506` | 45 | `0.9500` | `0.1822 / 0.5286` | `0.000278` | `0.802` | `0.844` |

Diagnostics:

```text
controlled_ema_trust: 3200 steps, accepted 92.25%, trust_expansions 191, alpha_max_seen 0.001915
controlled_raw_rho:   3200 steps, accepted 94.03%, final test 0.8418
controlled_ema:       3200 steps, accepted 94.06%, final test 0.8398
```

Interpretation:

- EMA-trust had the best peak accuracy (`0.8506` at epoch 45), but its final
  epoch was lower.
- Raw-rho had the best final test accuracy among controlled variants at epoch
  100.
- Vanilla/fixed Adam heavily overfit the train subset by epoch 100, with train
  accuracy around `0.99` and worse test loss.
- Controlled variants, especially raw/EMA, trained more conservatively and had
  lower train accuracy but better final test loss.

## 7. Adam CIFAR-10 Benchmark Results

### 7.1 CIFAR-10, 10 Epoch Stronger CNN

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.5840` | `0.6110` | 9 | `0.6528` | `0.9689 / 1.1312` | | | |
| `fixed_adam_direction` | `0.5960` | `0.6010` | 9 | `0.6456` | `0.9778 / 1.1339` | `0.001` | `0.815` | `1.000` |
| `controlled_raw_rho` | `0.5150` | `0.5530` | 8 | `0.5426` | `1.2668 / 1.4222` | `0.00239` | `0.731` | `1.000` |
| `controlled_ema` | `0.5890` | `0.5890` | 10 | `0.6590` | `0.9533 / 1.1312` | `0.000660` | `0.847` | `1.000` |
| `controlled_ema_trust` | `0.5890` | `0.5890` | 10 | `0.6590` | `0.9533 / 1.1312` | `0.000660` | `0.847` | `1.000` |

Interpretation:

- The stronger CNN made CIFAR behavior much more plausible than the earlier
  tiny CNN.
- Raw-rho was too aggressive/unstable in early CIFAR tuning.
- EMA and EMA-trust matched each other because trust expansion did not produce
  distinct behavior here.

### 7.2 CIFAR-10, 40 Epoch With Too-Large Alpha Cap

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_tuned_cifar1
```

Alpha cap:

```text
controlled-alpha-max = 5e-2
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.7180` | `0.7200` | 34 | `0.8804` | `0.3306 / 0.9306` | | | |
| `fixed_adam_direction` | `0.7170` | `0.7170` | 40 | `0.8886` | `0.3158 / 0.9006` | `0.001` | `0.819` | `1.000` |
| `controlled_raw_rho` | `0.6400` | `0.6920` | 23 | `0.6954` | `0.8854 / 1.0459` | `0.0499` | `0.597` | `1.000` |
| `controlled_ema` | `0.6010` | `0.6720` | 23 | `0.6364` | `1.0467 / 1.1636` | `0.0500` | `0.600` | `1.000` |
| `controlled_ema_trust` | `0.6010` | `0.6720` | 23 | `0.6364` | `1.0467 / 1.1636` | `0.0500` | `0.600` | `1.000` |

Interpretation:

- This run showed that a large cap is dangerous for Adam-direction CIFAR.
- Controlled variants grew to `0.05`, far above the useful Adam-scale learning
  rate, and performance degraded.
- Trust expansion was irrelevant because alpha was already too large.

### 7.3 CIFAR-10, 40 Epoch With Alpha Cap `3e-3`

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_3e3
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.7180` | `0.7200` | 34 | `0.8804` | `0.3306 / 0.9306` | | | |
| `fixed_adam_direction` | `0.7170` | `0.7170` | 40 | `0.8886` | `0.3158 / 0.9006` | `0.001` | `0.819` | `1.000` |
| `controlled_raw_rho` | `0.6830` | `0.7390` | 38 | `0.8296` | `0.4803 / 0.9649` | `0.00298` | `0.743` | `1.000` |
| `controlled_ema` | `0.7150` | `0.7150` | 40 | `0.8754` | `0.3642 / 0.8810` | `0.00300` | `0.775` | `1.000` |
| `controlled_ema_trust` | `0.6840` | `0.7200` | 33 | `0.8522` | `0.4214 / 1.0292` | `0.00300` | `0.769` | `1.000` |

Interpretation:

- Reducing the cap from `0.05` to `0.003` largely fixed the catastrophic
  behavior.
- Raw-rho achieved the best peak (`0.739`) but gave back performance by the
  final epoch.
- EMA reached a final result near vanilla/fixed.
- EMA-trust did not help in this setting.

### 7.4 CIFAR-10, 40 Epoch With Alpha Cap `1.5e-3`

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.7180` | `0.7200` | 34 | `0.8804` | `0.3306 / 0.9306` | | | |
| `fixed_adam_direction` | `0.7170` | `0.7170` | 40 | `0.8886` | `0.3158 / 0.9006` | `0.001` | `0.819` | `1.000` |
| `controlled_raw_rho` | `0.7110` | `0.7350` | 38 | `0.8704` | `0.3641 / 0.9283` | `0.00106` | `0.794` | `1.000` |
| `controlled_ema` | `0.7040` | `0.7090` | 33 | `0.8550` | `0.4007 / 0.9474` | `0.00150` | `0.777` | `1.000` |
| `controlled_ema_trust` | `0.7310` | `0.7310` | 40 | `0.8936` | `0.3017 / 0.8696` | `0.000949` | `0.857` | `1.000` |

Diagnostics:

```text
controlled_ema_trust: 1600 steps, accepted 99.38%, trust_expansions 1, alpha_max_seen 0.0015
controlled_raw_rho:   1600 steps, accepted 99.31%, best test 0.735 at epoch 38
fixed_adam_direction: 1600 steps, accepted 99.56%, alpha fixed 0.001
```

Interpretation:

- This was the best final CIFAR result on the 5k/1k Adam subset.
- EMA-trust beat vanilla/fixed Adam at the final epoch.
- Raw-rho still had a strong peak, but final accuracy was lower than EMA-trust.
- The result supported the conclusion that Adam CIFAR needs a tight alpha cap
  near the Adam learning-rate scale.

### 7.5 Larger CIFAR-10, 20k/5k, 40 Epoch

Output:

```text
controlled_adam_project/outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Dataset size:

```text
20000 train / 5000 test
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_adam` | `0.8288` | `0.8344` | 38 | `0.9361` | `0.1830 / 0.5951` | | | |
| `fixed_adam_direction` | `0.8280` | `0.8402` | 39 | `0.9359` | `0.1856 / 0.6189` | `0.001` | `0.831` | `1.000` |
| `controlled_raw_rho` | `0.8232` | `0.8292` | 38 | `0.9264` | `0.2048 / 0.6444` | `0.00150` | `0.815` | `1.000` |
| `controlled_ema` | `0.8278` | `0.8324` | 37 | `0.9314` | `0.1965 / 0.6401` | `0.00150` | `0.817` | `1.000` |
| `controlled_ema_trust` | `0.8278` | `0.8324` | 37 | `0.9314` | `0.1965 / 0.6401` | `0.00150` | `0.817` | `1.000` |

Diagnostics:

```text
fixed_adam_direction: alpha fixed 0.001, accepted 100%, best test 0.8402
controlled_raw_rho:   alpha saturated at 0.0015, accepted 100%
controlled_ema:       alpha saturated at 0.0015, accepted 100%
controlled_ema_trust: identical to controlled_ema, trust_expansions 0
```

Interpretation:

- The larger subset made CIFAR performance much more plausible.
- The earlier low-70s CIFAR results were partly a small-data/subset issue.
- Fixed Adam-direction had the best peak accuracy.
- Controlled variants did not collapse and did not reject steps, but saturating
  the `1.5e-3` cap did not improve validation accuracy over fixed direction.
- This is the clearest evidence that same-minibatch rho can be locally healthy
  without producing the best generalization.

### 7.6 CIFAR-10, 20k/5k, 20 Epoch Adam Tuning Follow-Ups

After the 40-epoch run, we ran several shorter 20-epoch sweeps on the same
20k/5k CIFAR-10 subset to isolate Adam-scale controller parameters.

| Setting | Key Parameters | Best Controlled Variant | Final Test Acc | Best Test Acc | Interpretation |
|---|---|---|---:|---:|---|
| Conservative | `alpha_max=1.25e-3`, `rho_star=0.82`, `rho_beta=0.95` | `controlled_raw_rho` | `0.8124` | `0.8164` | Stable, but too cautious. |
| Balanced | `alpha_max=1.5e-3`, `rho_star=0.80`, `rho_beta=0.90` | `controlled_raw_rho` | `0.8138` | `0.8232` | Best current peak result. |
| Open | `alpha_max=2e-3`, `rho_star=0.78`, `rho_beta=0.90` | `controlled_ema` / `controlled_ema_trust` | `0.8190` | `0.8190` | Stable, but not better; cap likely too open. |
| Lower target | `alpha_max=1.5e-3`, `rho_star=0.78`, `rho_beta=0.90` | `controlled_raw_rho` | `0.8214` | `0.8214` | Strong final result, but below the balanced peak. |
| Faster EMA | `alpha_max=1.5e-3`, `rho_star=0.80`, `rho_beta=0.85` | `controlled_raw_rho` | `0.8138` | `0.8232` | EMA ramped earlier but did not improve final/best recommendation. |

Baseline references for these 20-epoch runs:

```text
vanilla_adam:          final 0.7990, best 0.8158
fixed_adam_direction: final 0.8144, best 0.8220
```

The practical conclusion is that the best Adam-scale controller setting so far
is still the balanced raw-rho run with `alpha_max=1.5e-3`, `rho_star=0.80`,
and `rho_beta=0.90`. Lowering `rho_star` to `0.78` improved the final raw-rho
epoch but did not exceed the best peak. Lowering `rho_beta` to `0.85` made EMA
more responsive but did not close the gap to raw-rho.

## 8. Muon Neural Benchmark Results

### 8.1 Fashion-MNIST, 20 Epoch Muon Ablation

Output:

```text
controlled_muon_project/outputs/fashion_mnist_muon_20epoch_ablation
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.7432` | `0.7432` | 20 | `0.7668` | `0.8561 / 0.9012` | | | |
| `fixed_muon_direction` | `0.7432` | `0.7432` | 20 | `0.7668` | `0.8561 / 0.9012` | `0.001` | `0.996` | `1.000` |
| `controlled_raw_rho` | `0.8320` | `0.8418` | 14 | `1.0000` | `0.0107 / 0.8031` | `0.0136` | `0.689` | `1.000` |
| `controlled_ema` | `0.8301` | `0.8389` | 18 | `0.9949` | `0.0273 / 0.7537` | `0.0145` | `0.703` | `1.000` |
| `controlled_ema_trust` | `0.8301` | `0.8389` | 18 | `0.9949` | `0.0273 / 0.7537` | `0.0145` | `0.703` | `1.000` |

Diagnostics:

```text
controlled_raw_rho:   640 steps, alpha_max_seen 0.02494, accepted 100%
controlled_ema:       640 steps, alpha_max_seen 0.02605, accepted 100%
controlled_ema_trust: 640 steps, trust_expansions 0, accepted 100%
```

Interpretation:

- Controlled Muon was clearly better than vanilla/fixed Muon on this subset.
- The fixed `1e-3` Muon scale was too conservative.
- Controlled variants increased alpha into the `1e-2` range and reached much
  better accuracy.
- EMA and EMA-trust matched because trust expansion did not fire.

### 8.2 CIFAR-10, 10 Epoch Muon Ablation

Output:

```text
controlled_muon_project/outputs/cifar10_muon_10epoch_ablation
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.5780` | `0.5780` | 10 | `0.6558` | `0.9974 / 1.1609` | | | |
| `fixed_muon_direction` | `0.5820` | `0.5820` | 10 | `0.6556` | `0.9972 / 1.1509` | `0.001` | `0.963` | `1.000` |
| `controlled_raw_rho` | `0.6650` | `0.6650` | 10 | `0.7330` | `0.7520 / 0.9901` | `0.0221` | `0.793` | `1.000` |
| `controlled_ema` | `0.6440` | `0.6610` | 9 | `0.7170` | `0.7929 / 1.0157` | `0.0219` | `0.779` | `1.000` |
| `controlled_ema_trust` | `0.6440` | `0.6610` | 9 | `0.7170` | `0.7929 / 1.0157` | `0.0219` | `0.779` | `1.000` |

Interpretation:

- Controlled Muon improved strongly over vanilla/fixed Muon even after 10
  epochs.
- Raw-rho had the best final and best peak result in this short run.

### 8.3 CIFAR-10, 40 Epoch Muon Ablation

Output:

```text
controlled_muon_project/outputs/cifar10_muon_40epoch_ablation
```

| Optimizer | Final Test Acc | Best Test Acc | Best Epoch | Final Train Acc | Final Train/Test Loss | Final Alpha | Final Rho | Final Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.6990` | `0.7090` | 31 | `0.8842` | `0.3298 / 0.9643` | | | |
| `fixed_muon_direction` | `0.6940` | `0.7010` | 38 | `0.8824` | `0.3342 / 0.9613` | `0.001` | `0.909` | `1.000` |
| `controlled_raw_rho` | `0.7250` | `0.7310` | 27 | `0.8732` | `0.3901 / 1.1870` | `0.0498` | `0.701` | `1.000` |
| `controlled_ema` | `0.7250` | `0.7250` | 40 | `0.8988` | `0.3039 / 1.1170` | `0.0458` | `0.678` | `1.000` |
| `controlled_ema_trust` | `0.7250` | `0.7250` | 40 | `0.8988` | `0.3039 / 1.1170` | `0.0458` | `0.678` | `1.000` |

Diagnostics:

```text
controlled_raw_rho:   1600 steps, alpha_final 0.04965, alpha_max 0.05, accepted 100%
controlled_ema:       1600 steps, alpha_final 0.04507, alpha_max 0.05, accepted 100%
controlled_ema_trust: 1600 steps, alpha_final 0.04507, trust_expansions 0, accepted 100%
```

Interpretation:

- Controlled Muon beat vanilla/fixed Muon on both final and peak accuracy.
- Raw-rho had the best peak accuracy (`0.731` at epoch 27).
- EMA and EMA-trust tied for final accuracy (`0.725`).
- Trust expansion did not fire.
- The Muon implementation is slower than Adam because it performs CPU/NumPy
  orthogonalization and, for controlled variants, same-minibatch trial loss
  evaluations.

## 9. Cross-Benchmark Pattern Summary

### 9.1 Best Variant By Benchmark

| Benchmark | Best Final Test Accuracy | Best Peak Test Accuracy | Main Winner |
|---|---:|---:|---|
| Adam Fashion 20 epoch | `0.8369` vanilla | `0.8369` vanilla/fixed | vanilla/fixed |
| Adam Fashion 40 epoch untuned | `0.8389` fixed | `0.8428` fixed | fixed direction |
| Adam Fashion 40 epoch tuned1 | `0.8467` EMA-trust | `0.8467` EMA-trust | controlled EMA-trust |
| Adam Fashion 40 epoch aggressive | `0.8438` EMA-trust | `0.8438` EMA-trust | controlled EMA-trust |
| Adam Fashion 100 epoch tuned1 | `0.8418` raw-rho | `0.8506` EMA-trust | raw final / EMA-trust peak |
| Adam CIFAR 5k/1k cap `1.5e-3` | `0.7310` EMA-trust | `0.7350` raw-rho | EMA-trust final / raw peak |
| Adam CIFAR 20k/5k | `0.8288` vanilla | `0.8402` fixed | vanilla final / fixed peak |
| Muon Fashion 20 epoch | `0.8320` raw-rho | `0.8418` raw-rho | controlled raw-rho |
| Muon CIFAR 10 epoch | `0.6650` raw-rho | `0.6650` raw-rho | controlled raw-rho |
| Muon CIFAR 40 epoch | `0.7250` raw/EMA/EMA-trust | `0.7310` raw-rho | controlled variants |

### 9.2 Adam-Specific Lessons

1. Alpha cap matters more than almost anything else on CIFAR.

   The `5e-2` cap was too large for Adam-direction CIFAR. The controller saw
   acceptable same-minibatch progress and grew alpha all the way to the cap,
   but validation accuracy suffered.

2. Trust-region recovery helps when alpha collapses.

   Fashion-MNIST raw/EMA variants often shrank to tiny alpha values around
   `1e-5`. EMA-trust could recover enough step size to become competitive or
   best.

3. Fixed Adam-direction is a strong baseline.

   On the larger CIFAR run, fixed Adam-direction achieved the best peak
   accuracy. This means controlled variants should be judged against fixed
   direction, not only against vanilla Adam.

4. Same-minibatch rho is necessary but not sufficient.

   A good rho means the local minibatch step did what the first-order model
   predicted. It does not guarantee better validation accuracy.

### 9.3 Muon-Specific Lessons

1. Controlled Muon has been consistently helpful on the subset runs.

   The controller discovered much larger useful alpha values than the fixed
   `1e-3` baseline.

2. Raw-rho often had the best peak.

   On Muon Fashion-MNIST and CIFAR-10, raw-rho reached the best peak accuracy.

3. EMA and EMA-trust usually matched.

   Trust expansion did not fire in the main Muon runs. This means EMA-trust was
   effectively just EMA in those experiments.

4. Muon needs separate tuning from Adam.

   Adam CIFAR needed a tight cap near `1e-3`; Muon CIFAR benefited from alphas
   near `0.05`. These are very different regimes.

## 10. Current Caveats

1. Most results are single-seed subset benchmarks.

   We should not overclaim. The strongest next test is multi-seed evaluation,
   especially on CIFAR-10.

2. Historical CSVs do not all include wall-clock and optimizer-step fields.

   The runners now record those fields, but older long runs need to be rerun
   for exact accuracy/loss versus wall-clock comparisons.

3. The controlled variants have extra compute cost.

   Controlled Adam/Muon evaluate same-minibatch trial losses. For wall-clock
   fairness, future reports should emphasize accuracy-vs-time as well as
   accuracy-vs-epoch.

   The extra measurement is one additional forward loss evaluation after the
   trial step on the same minibatch, not an additional backward pass. Since
   backward is usually more expensive than forward, this kind of overhead can be
   manageable in large neural training, but it is not free. The right empirical
   question is whether the controller reaches a target loss or accuracy sooner
   in wall-clock time.

4. Same-minibatch trial loss is mandatory.

   This is not optional. Using a different minibatch for the after-step loss
   would corrupt the controller signal.

5. Muon implementation is educational, not optimized.

   It uses CPU/NumPy orthogonalization and is much slower than a production
   Muon implementation would be.

6. Best-final and best-peak tell different stories.

   For example, Adam Fashion-MNIST 100 epoch tuned1 had EMA-trust best peak but
   raw-rho best final. Future benchmark summaries should report both.

## 11. Recommended Next Experiments

### 11.1 Multi-Seed CIFAR-10 20k/5k

Run the larger CIFAR setup over at least 5 seeds.

Report:

- final test accuracy mean/std
- best test accuracy mean/std
- train/test loss
- accuracy vs optimizer steps
- accuracy vs wall-clock time

This is the cleanest next step before full CIFAR-10.

### 11.2 Full CIFAR-10 With Fewer Variants

Start with:

- `vanilla_adam`
- `fixed_adam_direction`
- best controlled Adam setting
- best controlled Muon setting

Use progress logging and checkpoints.

### 11.3 Learning-Rate Schedule Baselines

Compare controlled variants to stronger hand-tuned baselines:

- Adam fixed LR
- AdamW fixed LR
- AdamW cosine decay
- possibly SGD+momentum/cosine for CIFAR

The controller should eventually be compared against serious baselines, not
only fixed `lr=1e-3`.

### 11.4 Noisy-Label Stress Test

Same-minibatch rho may over-trust local minibatch improvement. Add label noise:

- CIFAR-10 with 10% noisy labels
- CIFAR-10 with 20% noisy labels

Then compare train loss, test loss, and test accuracy versus time.

### 11.5 Best-Checkpoint Reporting

Add automatic summary files:

```text
best_epoch_summary.csv
best_epoch_summary.json
```

Each should include:

- best test accuracy and epoch
- final test accuracy
- final train/test loss
- best test loss and epoch
- final alpha/rho/acceptance
- total wall-clock time

## 12. Bottom Line

The five-variant ablation has been valuable because it separates several
questions that would otherwise be tangled together:

1. Is the base optimizer/direction already strong?
2. Does the controlled direction path match the vanilla baseline?
3. Does raw same-minibatch rho help?
4. Does EMA smoothing help?
5. Does trust-region-style recovery help when alpha becomes too small?

The current evidence says:

- Controlled Adam is promising but sensitive. It can win on tuned Fashion-MNIST
  and smaller CIFAR settings, but on the larger CIFAR run the fixed
  Adam-direction baseline had the best peak accuracy.
- Controlled Muon is more consistently positive in the subset benchmarks,
  likely because the fixed Muon learning rate was too conservative and the
  controller found a much larger useful alpha.
- Trust-region recovery is useful for Adam when alpha collapse is the failure
  mode, but irrelevant when alpha is already at the cap or when trust expansion
  does not fire.
- Future claims should be based on multi-seed, wall-clock-aware benchmarks.
