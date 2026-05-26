# Controlled Optimizer Performance And Overhead

This note summarizes the observed performance of the controlled optimizer
variants, with special attention to wall-clock time and the overhead from the
extra same-minibatch forward pass.

Current-status note: the Muon neural results in this overhead note are
historical and predate the official-style Muon parameter-scope cleanup. Treat
them as evidence about controller overhead and alpha adaptation, not as current
vanilla-Muon quality comparisons. Future Muon neural comparisons should use the
corrected official-style Muon path.

The main conclusion is:

```text
The controlled optimizers are not intrinsically cheaper per step.
They can still be more wall-clock efficient when the extra forward-pass
overhead is modest and the adaptive alpha produces much more progress per step.
```

## What Extra Work Does The Controller Add?

A vanilla optimizer step roughly does:

```text
forward pass
backward pass
optimizer direction/update
```

A controlled optimizer step roughly does:

```text
forward pass
backward pass
optimizer direction
trial parameter update
extra forward pass on the same minibatch
accept/reject and alpha-control logic
```

The important point is that the controller adds an extra **forward** loss
evaluation, not an extra backward pass. Backward propagation is usually more
expensive than a forward pass, and in the current Muon implementation the
direction computation itself also has nontrivial cost because it performs
CPU/NumPy orthogonalization.

Therefore the expected overhead is not `2x`. A rough mental model is:

```text
if backward ~= 2 * forward,
vanilla cost ~= forward + backward = 3F
controlled cost ~= forward + backward + extra forward = 4F
overhead ~= 4F / 3F = 1.33x
```

The actual overhead depends on the model, device, batch size, data pipeline,
and optimizer direction cost.

## Why The Controlled Curves Can Look Faster

The recent wall-clock plots show controlled optimizers reaching lower loss or
higher accuracy at similar elapsed times. This is possible because wall-clock
efficiency is:

```text
progress per step / time per step
```

The controller may increase time per step, but it can increase progress per
step even more.

In the Muon benchmarks, the fixed Muon learning rate was:

```text
alpha = 1e-3
```

The controlled Muon variants started at the same value but grew alpha to roughly:

```text
Fashion-MNIST 20 epoch: alpha ~= 1.0e-2
CIFAR-10 20 epoch:     alpha ~= 1.5e-2 to 1.6e-2
```

That is a `10x` to `16x` larger global multiplier than the fixed baseline.
The controller paid for an extra forward pass, but it also took much more
effective steps.

## Backtracking Was Almost Never Used

The expensive worst case would be repeated trial evaluations:

```text
try alpha
try 0.5 alpha
try 0.25 alpha
try 0.125 alpha
```

In the completed multi-seed Muon runs, that did not happen. The controlled
variants almost always accepted the first trial step.

Summary of step diagnostics:

| Benchmark | Variant | Accept Rate | Fraction Backtracked | Mean Backtracks | Trust Expansion |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 20e 3k/1k | `controlled_raw_rho` | `1.000` | `0.0000` | `0.0000` | `0.0000` |
| CIFAR-10 20e 3k/1k | `controlled_ema` | `1.000` | `0.0000` | `0.0000` | `0.0000` |
| CIFAR-10 20e 3k/1k | `controlled_ema_trust` | `1.000` | `0.0000` | `0.0000` | `0.0000` |
| Fashion-MNIST 20e 1k/512 | `controlled_raw_rho` | `1.000` | `0.0000` | `0.0000` | `0.0000` |
| Fashion-MNIST 20e 1k/512 | `controlled_ema` | `1.000` | `0.0000` | `0.0000` | `0.0000` |
| Fashion-MNIST 40e 1k/512 | `controlled_raw_rho` | `0.999` | `0.0010` | `0.0010` | `0.0000` |
| Fashion-MNIST 40e 1k/512 | `controlled_ema` | `0.999` | `0.0031` | `0.0031` | `0.0000` |

This explains a lot of the wall-clock behavior. The controller usually paid
for only one extra forward pass per step. It did not usually pay for multiple
backtracked forward passes.

## Completed Muon Multi-Seed Results

### CIFAR-10, 20 Epochs, 3 Seeds, 3000/1000 Subset

Output folder:

```text
controlled_muon_project/outputs/cifar10_muon_multiseed_20epoch_3seeds_3k_1k
```

| Optimizer | Final Test Acc | Best Test Acc | Mean Time | Relative Time | Final Alpha |
|---|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.6090 +/- 0.0026` | `0.6110 +/- 0.0026` | `287.6s` | `1.00x` |  |
| `fixed_muon_direction` | `0.6080 +/- 0.0020` | `0.6080 +/- 0.0020` | `313.2s` | `1.09x` | `1.000e-03` |
| `controlled_raw_rho` | `0.6677 +/- 0.0139` | `0.6800 +/- 0.0108` | `293.3s` | `1.02x` | `1.549e-02` |
| `controlled_ema` | `0.6670 +/- 0.0030` | `0.6760 +/- 0.0053` | `287.9s` | `1.00x` | `1.630e-02` |
| `controlled_ema_trust` | `0.6670 +/- 0.0030` | `0.6760 +/- 0.0053` | `290.4s` | `1.01x` | `1.630e-02` |

Interpretation:

- Controlled Muon reached much higher accuracy at almost the same recorded
  wall-clock time.
- The controlled variants used `480` optimizer steps, same as vanilla/fixed.
- The difference came from larger effective alpha, not more optimizer steps.
- Backtracking did not occur in this run.
- EMA and EMA-trust were identical because trust expansion did not fire.

### Fashion-MNIST, 20 Epochs, 5 Seeds, 1024/512 Subset

Output folder:

```text
controlled_muon_project/outputs/fashion_mnist_muon_multiseed_20epoch_5seeds_1k
```

| Optimizer | Final Test Acc | Best Test Acc | Mean Time | Relative Time | Final Alpha |
|---|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.6051 +/- 0.0181` | `0.6051 +/- 0.0181` | `4.2s` | `1.00x` |  |
| `fixed_muon_direction` | `0.6051 +/- 0.0181` | `0.6051 +/- 0.0181` | `4.3s` | `1.04x` | `1.000e-03` |
| `controlled_raw_rho` | `0.7289 +/- 0.0070` | `0.7289 +/- 0.0070` | `4.1s` | `0.99x` | `1.009e-02` |
| `controlled_ema` | `0.7293 +/- 0.0067` | `0.7293 +/- 0.0067` | `4.4s` | `1.05x` | `1.016e-02` |
| `controlled_ema_trust` | `0.7293 +/- 0.0067` | `0.7293 +/- 0.0067` | `4.0s` | `0.96x` | `1.016e-02` |

Interpretation:

- Controlled Muon substantially outperformed vanilla/fixed Muon.
- The wall-clock times are very small and noisy, so this should not be treated
  as a precise timing benchmark.
- It still supports the qualitative point that the extra forward pass was not
  a dominant cost in this setting.
- No backtracking occurred.

### Fashion-MNIST, 40 Epochs, 3 Seeds, 1024/512 Subset

Output folder:

```text
controlled_muon_project/outputs/fashion_mnist_muon_multiseed_40epoch_1k
```

| Optimizer | Final Test Acc | Best Test Acc | Mean Time | Relative Time | Final Alpha |
|---|---:|---:|---:|---:|---:|
| `vanilla_muon` | `0.6738 +/- 0.0085` | `0.6745 +/- 0.0096` | `7.1s` | `1.00x` |  |
| `fixed_muon_direction` | `0.6738 +/- 0.0085` | `0.6745 +/- 0.0096` | `8.0s` | `1.13x` | `1.000e-03` |
| `controlled_raw_rho` | `0.8040 +/- 0.0152` | `0.8118 +/- 0.0063` | `10.1s` | `1.43x` | `1.286e-02` |
| `controlled_ema` | `0.8034 +/- 0.0168` | `0.8125 +/- 0.0085` | `11.0s` | `1.56x` | `9.705e-03` |
| `controlled_ema_trust` | `0.8034 +/- 0.0168` | `0.8125 +/- 0.0085` | `8.9s` | `1.26x` | `9.705e-03` |

Interpretation:

- Here the controlled variants were visibly slower in wall-clock time, but
  still much more accurate.
- Backtracking was still almost nonexistent: roughly `0.1%` to `0.3%` of steps.
- The observed overhead is consistent with the extra forward-pass cost plus CPU
  timing noise.

## What The Diagnostics Say

### Acceptance Rate

The controlled variants typically had:

```text
accepted_rate ~= 1.000
```

This means nearly every trial step reduced same-minibatch loss. The acceptance
gate was still useful as a safety brake, but it almost never had to reject a
step in these runs.

### Backtracking

Backtracking was effectively zero in the main completed runs:

```text
CIFAR-10 20e:       0.0000 fraction backtracked
Fashion-MNIST 20e:  0.0000 fraction backtracked
Fashion-MNIST 40e:  about 0.001 to 0.003 fraction backtracked
```

Therefore the practical overhead was usually:

```text
one extra forward pass
```

not:

```text
many repeated trial forward passes
```

### Alpha Growth

Fixed Muon used:

```text
alpha = 1e-3
```

Controlled Muon learned much larger alphas:

```text
Fashion-MNIST 20e:  alpha ~= 1.0e-2
CIFAR-10 20e:       alpha ~= 1.5e-2 to 1.6e-2
Fashion-MNIST 40e:  alpha ~= 0.9e-2 to 1.3e-2
```

This is the main reason the controlled methods made more progress.

### Trust Expansion

The trust-region expansion did not fire in these completed runs:

```text
trust_region_expanded fraction = 0.0000
```

The current condition is strict:

```text
backtracks == 0
rho_control >= 0.9
alpha_used <= 1e-4
```

Since the neural runs start at `alpha_0 = 1e-3`, the threshold
`alpha_used <= 1e-4` usually prevents trust expansion. This explains why
`controlled_ema` and `controlled_ema_trust` often have identical metrics.

The balanced CIFAR-10 ResNet Adam summary makes this especially clear. That run
used `alpha_min = 1e-3` and `alpha_max = 1.5e-3`, but the trust trigger was left
at `trust_region_alpha_threshold = 1e-4`. Because the trigger was below the hard
floor, the trust branch could not activate; the recorded diagnostics show
`0/1580` trust expansions for each of seeds `123`, `456`, and `789`.

This is an important reporting caveat: in that benchmark, EMA+trust should not
be presented as a distinct trust-region result. It is better described as
EMA-rho with the trust hook enabled but dormant. A meaningful Adam-scale trust
test should set `trust_region_alpha_threshold` near the active alpha floor, for
example `1e-3` to `1.05e-3`, and use a modest expansion factor such as `1.1` or
`1.2`.

## Why Similar Wall-Clock Time Is Plausible

The controlled optimizer can appear almost as fast as vanilla in wall-clock
plots because several effects combine:

1. The extra measurement is a forward pass, not backward pass.
2. Backtracking almost never happened.
3. Muon direction computation is already expensive, so the added forward pass
   is only part of total step cost.
4. The controller increased alpha by roughly an order of magnitude.
5. The fixed baseline was probably too conservative for Muon in these settings.
6. The benchmarks are small CPU runs, so elapsed-time measurements include
   noise from evaluation, plotting, scheduling, and Python overhead.

The fair interpretation is:

```text
The controller was more optimization-efficient per unit wall-clock time in
these runs.
```

The unfair interpretation would be:

```text
The controlled optimizer step is cheaper than vanilla.
```

That is not what we should claim.

## Caveats

1. These are subset benchmarks, not full-dataset production claims.

2. Timing is not a dedicated microbenchmark. The recorded elapsed time includes
   training and evaluation in the runner.

3. Small CPU runs can have noisy wall-clock measurements. For example, some
   Fashion-MNIST 20-epoch controlled variants appeared slightly faster than
   vanilla, which should be interpreted as timing noise plus similar cost, not
   a true per-step speed advantage.

4. The controlled variants often reached much higher train accuracy. On larger
   runs we should watch validation/test generalization and not optimize only
   same-minibatch progress.

5. If alpha becomes more aggressive, backtracking may occur more often. Then
   the overhead could increase.

## Recommended Next Timing Experiments

To make a stronger wall-clock claim, run a dedicated timing benchmark:

1. Disable plotting during timing.
2. Use warmup epochs before measuring.
3. Record time per optimizer step, time per epoch, and time to target accuracy.
4. Report the number of trial forward passes per accepted step.
5. Report fraction of steps with backtracking.
6. Compare against a tuned fixed-alpha Muon baseline, not only `1e-3`.
7. Repeat on GPU if available, because forward/backward cost ratios can differ
   from CPU.

The most important metric is:

```text
time to reach a target loss or accuracy
```

not just final accuracy at a fixed epoch.
