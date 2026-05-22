# Controlled Optimizer Viability Analysis

**Date:** 2026-05-21  
**Topic:** Viability of the controlled optimizer idea after reviewing the performance and overhead report.

---

## Executive Summary

The new overhead report makes the controlled optimizer idea **more viable than before**, but in a narrower and more precise way.

The strongest updated conclusion is:

\[
\boxed{
\text{The controller is viable when adaptive alpha improves progress per step enough to pay for the extra forward pass.}
}
\]

The report weakens the objection that the extra same-minibatch forward pass automatically makes the method uncompetitive. In the completed Muon multi-seed subset runs, controlled variants often achieved substantially higher accuracy at similar or moderately higher wall-clock time. The reason is not that controlled optimizer steps are cheaper. They are not. The reason is that the controller discovered a much larger useful global step multiplier.

However, the current evidence does **not** yet prove that the controlled optimizer is generally better than well-tuned fixed-learning-rate baselines.

The central remaining question is:

\[
\boxed{
\text{Does controlled Muon beat tuned fixed-alpha Muon, or only a conservative }10^{-3}\text{ baseline?}
}
\]

Until that is answered, the best claim is:

\[
\boxed{
\text{The optimizer is promising as an adaptive learning-rate-scale controller.}
}
\]

not yet:

\[
\boxed{
\text{The optimizer is proven to be a generally superior optimizer wrapper.}
}
\]

---

# 1. What Extra Work Does The Controller Add?

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

The controlled optimizer adds an extra **forward** loss evaluation, not an extra backward pass.

If:

\[
\text{backward} \approx 2 \times \text{forward},
\]

then vanilla cost is roughly:

\[
F+B \approx F+2F=3F,
\]

and controlled cost is roughly:

\[
F+B+F \approx F+2F+F=4F.
\]

So the expected overhead is:

\[
\frac{4F}{3F}=1.33\times.
\]

This is meaningful, but not catastrophic. The key question is whether controlled updates produce enough extra progress per step to compensate.

The new report shows that, at least in the completed Muon subset experiments, this overhead did not automatically destroy wall-clock competitiveness.

---

# 2. Why The New Report Improves The Viability Case

Previously, the biggest practical worry was:

\[
\text{extra forward pass} \Rightarrow \text{too much overhead}.
\]

The new report shows this is not automatically true.

In Muon runs, fixed Muon used:

\[
\alpha = 10^{-3}.
\]

Controlled Muon grew alpha to approximately:

\[
\alpha \approx 10^{-2}
\]

on Fashion-MNIST and approximately:

\[
\alpha \approx 1.5\times 10^{-2} \text{ to } 1.6\times 10^{-2}
\]

on CIFAR-10 subset runs.

So the mechanism is:

\[
\boxed{
\text{the controller discovered a much better effective step scale.}
}
\]

This makes the method more promising because the extra forward-pass overhead can be outweighed by improved optimization progress.

The controlled optimizer is therefore not intrinsically faster per step. Rather:

\[
\boxed{
\text{progress per step increased enough to compensate for extra cost per step.}
}
\]

That is a much more defensible and interesting claim.

---

# 3. Strongest Positive Evidence

## 3.1 CIFAR-10 Muon Multi-Seed Result

The strongest result is the CIFAR-10 Muon multi-seed experiment:

```text
CIFAR-10, 20 epochs, 3 seeds, 3000/1000 subset
```

| Optimizer | Final Test Accuracy | Best Test Accuracy | Relative Time | Final Alpha |
|---|---:|---:|---:|---:|
| `vanilla_muon` | `0.6090 ± 0.0026` | `0.6110 ± 0.0026` | `1.00x` | — |
| `fixed_muon_direction` | `0.6080 ± 0.0020` | `0.6080 ± 0.0020` | `1.09x` | `1.000e-03` |
| `controlled_raw_rho` | `0.6677 ± 0.0139` | `0.6800 ± 0.0108` | `1.02x` | `1.549e-02` |
| `controlled_ema` | `0.6670 ± 0.0030` | `0.6760 ± 0.0053` | `1.00x` | `1.630e-02` |
| `controlled_ema_trust` | `0.6670 ± 0.0030` | `0.6760 ± 0.0053` | `1.01x` | `1.630e-02` |

This shows:

\[
\boxed{
\text{large accuracy gain at roughly similar recorded wall-clock time.}
}
\]

This is the most encouraging result so far.

It suggests that, at least for Muon on these subset tasks, the controller is not merely improving accuracy per epoch. It may also improve practical wall-clock efficiency.

---

## 3.2 Fashion-MNIST Muon Multi-Seed Result

For Fashion-MNIST, 20 epochs, 5 seeds, 1024/512 subset:

| Optimizer | Final Test Accuracy | Relative Time | Final Alpha |
|---|---:|---:|---:|
| `vanilla_muon` | `0.6051 ± 0.0181` | `1.00x` | — |
| `fixed_muon_direction` | `0.6051 ± 0.0181` | `1.04x` | `1.000e-03` |
| `controlled_raw_rho` | `0.7289 ± 0.0070` | `0.99x` | `1.009e-02` |
| `controlled_ema` | `0.7293 ± 0.0067` | `1.05x` | `1.016e-02` |
| `controlled_ema_trust` | `0.7293 ± 0.0067` | `0.96x` | `1.016e-02` |

The timing numbers are small and noisy, so they should not be overinterpreted. But the qualitative pattern is encouraging:

\[
\boxed{
\text{controlled Muon found a much larger useful alpha and substantially improved accuracy.}
}
\]

---

## 3.3 Fashion-MNIST 40-Epoch Muon Result

For Fashion-MNIST, 40 epochs, 3 seeds, 1024/512 subset:

| Optimizer | Final Test Accuracy | Best Test Accuracy | Relative Time | Final Alpha |
|---|---:|---:|---:|---:|
| `vanilla_muon` | `0.6738 ± 0.0085` | `0.6745 ± 0.0096` | `1.00x` | — |
| `fixed_muon_direction` | `0.6738 ± 0.0085` | `0.6745 ± 0.0096` | `1.13x` | `1.000e-03` |
| `controlled_raw_rho` | `0.8040 ± 0.0152` | `0.8118 ± 0.0063` | `1.43x` | `1.286e-02` |
| `controlled_ema` | `0.8034 ± 0.0168` | `0.8125 ± 0.0085` | `1.56x` | `9.705e-03` |
| `controlled_ema_trust` | `0.8034 ± 0.0168` | `0.8125 ± 0.0085` | `1.26x` | `9.705e-03` |

This run is more nuanced.

The controlled variants are clearly slower in wall-clock time, but they also achieve much higher accuracy. This suggests that the controller can still be wall-clock worthwhile, but the advantage is no longer “free.”

The right metric here should be:

```text
time to target accuracy
```

not only final accuracy at a fixed epoch.

For example:

```text
time to reach 0.70 accuracy
time to reach 0.75 accuracy
time to reach 0.80 accuracy
```

This would directly measure whether the controlled optimizer is practically faster.

---

# 4. Backtracking Was Almost Never Used

The expensive worst case would be repeated trial evaluations:

```text
try alpha
try 0.5 alpha
try 0.25 alpha
try 0.125 alpha
...
```

But the completed multi-seed Muon runs showed almost no backtracking.

| Benchmark | Variant | Accept Rate | Fraction Backtracked | Mean Backtracks |
|---|---:|---:|---:|---:|
| CIFAR-10 20e 3k/1k | `controlled_raw_rho` | `1.000` | `0.0000` | `0.0000` |
| CIFAR-10 20e 3k/1k | `controlled_ema` | `1.000` | `0.0000` | `0.0000` |
| CIFAR-10 20e 3k/1k | `controlled_ema_trust` | `1.000` | `0.0000` | `0.0000` |
| Fashion-MNIST 20e 1k/512 | `controlled_raw_rho` | `1.000` | `0.0000` | `0.0000` |
| Fashion-MNIST 20e 1k/512 | `controlled_ema` | `1.000` | `0.0000` | `0.0000` |
| Fashion-MNIST 40e 1k/512 | `controlled_raw_rho` | `0.999` | `0.0010` | `0.0010` |
| Fashion-MNIST 40e 1k/512 | `controlled_ema` | `0.999` | `0.0031` | `0.0031` |

Therefore, practical overhead was usually:

\[
\boxed{
\text{one extra forward pass per step}
}
\]

not:

\[
\boxed{
\text{many repeated trial forward passes per step}.
}
\]

This makes the method much more practical than a naive worst-case analysis suggests.

It also suggests that the accept/reject machinery is mostly acting as a safety brake, not as the main driver of performance.

The main driver is alpha adaptation.

---

# 5. What The Results Actually Prove

The current evidence supports the following claims.

## Claim 1: The controller can discover a much better global step scale

\[
\boxed{
\text{The controller can discover a much better global step scale than a conservative fixed baseline.}
}
\]

This is strongly supported by the Muon experiments.

The fixed Muon baseline used:

\[
\alpha = 10^{-3}.
\]

Controlled Muon discovered useful alphas around:

\[
10^{-2}
\]

to

\[
1.6\times10^{-2}.
\]

That is roughly a \(10\times\) to \(16\times\) increase.

---

## Claim 2: The extra forward pass is not automatically fatal

\[
\boxed{
\text{The extra forward pass is not automatically fatal.}
}
\]

This is supported by the wall-clock results, especially for controlled Muon.

The controller pays more per step, but it may make enough extra progress per step to compensate.

---

## Claim 3: Backtracking overhead is not a major issue in these runs

\[
\boxed{
\text{Backtracking overhead is not a major issue in the completed runs.}
}
\]

This is strongly supported by the diagnostics.

Nearly all controlled steps were accepted immediately.

---

## Claim 4: Controlled Muon can improve wall-clock efficiency on subset tasks

\[
\boxed{
\text{The controller improves wall-clock efficiency in some subset experiments.}
}
\]

This is supported by the CIFAR-10 20e 3k/1k result, where controlled variants achieved much higher accuracy at roughly the same recorded time.

However, this should be verified with dedicated timing experiments.

---

# 6. What The Results Do Not Yet Prove

The current evidence does **not** yet prove:

\[
\boxed{
\text{Controlled Muon beats the best fixed-alpha Muon.}
}
\]

This is the biggest missing piece.

The fixed Muon baseline used:

\[
\alpha=10^{-3}.
\]

Controlled Muon grew to:

\[
\alpha\approx 10^{-2}\text{ to }1.6\times10^{-2}.
\]

So a skeptical interpretation is:

```text
controlled Muon won because the fixed baseline learning rate was too small.
```

That may still be useful because automatic LR-scale discovery is valuable, but it is not the same as proving a better optimizer.

The crucial next experiment is:

\[
\boxed{
\text{fixed-alpha Muon sweep versus controlled Muon.}
}
\]

For example:

\[
\alpha\in
\{
10^{-3},
3\times10^{-3},
10^{-2},
2\times10^{-2},
5\times10^{-2}
\}.
\]

If controlled Muon beats the best fixed-alpha Muon, the idea becomes substantially stronger.

If tuned fixed-alpha Muon matches controlled Muon, the contribution is still meaningful but different:

\[
\boxed{
\text{automatic learning-rate-scale discovery}
}
\]

rather than:

\[
\boxed{
\text{intrinsically better optimizer dynamics.}
}
\]

---

# 7. Updated Viability Assessment

## 7.1 Deterministic Optimization

Viability: **high**.

The controller is mathematically coherent and works well on several deterministic objectives.

For deterministic optimization, the extra evaluation cost is often acceptable, and actual-vs-predicted decrease has a clean interpretation.

---

## 7.2 Small Neural Networks

Viability: **promising**.

The controlled Muon results are encouraging, especially because gains appear in multi-seed subset runs.

The method is not just producing a one-off lucky result.

---

## 7.3 Adam-Style Wrappers

Viability: **mixed**.

Controlled Adam can help in some Fashion-MNIST and smaller CIFAR settings, but can also saturate alpha caps without improving larger CIFAR validation performance.

The lesson is:

\[
\boxed{
\text{same-minibatch }\rho\text{ is an optimization signal, not a generalization signal.}
}
\]

Adam-style wrappers likely need:

- tighter alpha caps;
- base LR schedules;
- validation/control-batch diagnostics;
- update-to-weight constraints.

---

## 7.4 Muon Wrappers

Viability: **stronger than Adam so far**.

Controlled Muon is currently the best evidence for the idea.

The controller appears especially useful when the fixed Muon scale is too conservative.

However, this interpretation must be tested against tuned fixed-alpha Muon baselines.

---

## 7.5 Language Models

Viability: **possible, but only with modifications**.

For large language models, per-step extra forward passes may be too expensive.

More viable versions are:

1. periodic \(\rho\)-measurement;
2. cheap gradient/update-signal controller with occasional \(\rho\)-calibration;
3. bounded multiplier over a standard base learning-rate schedule.

The per-step controller is probably not the right LM version.

The better LM version is:

\[
\boxed{
\text{slow, bounded, periodic learning-rate-scale calibration.}
}
\]

---

# 8. Implications For Language Models

For language models, the current per-step same-minibatch forward controller is probably too expensive unless the gain is very large.

A better design is periodic control.

Measure \(\rho_t\) every:

\[
k=10,20,50,100
\]

steps.

Approximate overhead under the \(3F\) vanilla-step model is:

\[
\text{overhead}
\approx
\frac{1}{3k}.
\]

So:

\[
k=10
\Rightarrow
\text{about }3.3\%\text{ overhead},
\]

\[
k=20
\Rightarrow
\text{about }1.7\%\text{ overhead},
\]

\[
k=50
\Rightarrow
\text{about }0.7\%\text{ overhead}.
\]

This makes the method much more plausible for large models.

A language-model-friendly version should use:

\[
\eta_t^{\mathrm{eff}}
=
\eta_t^{\mathrm{schedule}}\alpha_t,
\]

with conservative bounds such as:

\[
\alpha_t\in[0.8,1.25].
\]

The controller should act as a slow calibration mechanism, not an aggressive per-step optimizer.

---

# 9. Suggested Future Optimizer Design

The most practical next version is not a fully aggressive per-step controller.

A better design is:

\[
\eta_t^{\mathrm{eff}}
=
\eta_t^{\mathrm{base}}\alpha_t,
\]

with:

\[
\alpha_t\in[\alpha_{\min},\alpha_{\max}].
\]

Use a normal base optimizer and schedule:

```text
AdamW / Muon / hybrid optimizer
warmup + cosine or other base schedule
```

Then use the controller as a bounded multiplier.

For most steps:

\[
\theta_{t+1}
=
\theta_t+\eta_t^{\mathrm{base}}\alpha_t p_t.
\]

Every \(k\) steps:

1. compute same-minibatch \(\rho_t\);
2. update smoothed \(\bar\rho_t\);
3. adjust \(\alpha_t\) slowly.

For example:

\[
\bar\rho_t
=
\beta\bar\rho_{t-k}+(1-\beta)\rho_t,
\]

\[
\alpha_{t+1}
=
\operatorname{clip}
\left(
\alpha_t
\exp[K_P(\bar\rho_t-\rho^\star)],
\alpha_{\min},
\alpha_{\max}
\right).
\]

Also clip the per-update multiplier:

\[
\frac{\alpha_{t+1}}{\alpha_t}
\in[m_{\min},m_{\max}].
\]

For large models, conservative values might be:

\[
\alpha_t\in[0.8,1.25],
\]

\[
\frac{\alpha_{t+1}}{\alpha_t}
\in[0.98,1.02].
\]

---

# 10. Most Important Next Experiments

## 10.1 Tuned Fixed-Alpha Muon Sweep

Run fixed Muon with:

\[
\alpha\in
\{
10^{-3},
3\times10^{-3},
10^{-2},
2\times10^{-2},
5\times10^{-2}
\}.
\]

Compare against controlled Muon.

This answers:

\[
\boxed{
\text{adaptive control versus simply choosing a better fixed alpha.}
}
\]

This is the single most important next experiment.

---

## 10.2 Time-To-Target Accuracy

Report metrics such as:

```text
time to reach 65% CIFAR accuracy
time to reach 70% CIFAR accuracy
time to reach a target validation loss
```

This is more informative than final accuracy at a fixed epoch.

For example, if controlled Muon reaches 0.65 accuracy in 150 seconds and tuned fixed Muon reaches it in 180 seconds, that is strong evidence for wall-clock efficiency.

---

## 10.3 Dedicated Timing Benchmark

Disable plotting and reduce evaluation overhead.

Measure:

```text
train step time
trial forward time
optimizer direction time
data loading time
evaluation time
```

The goal is to separate optimizer overhead from logging, plotting, evaluation, and Python noise.

The current wall-clock numbers are useful but not definitive.

---

## 10.4 GPU Timing

CPU subset timing is useful for development, but GPU behavior may differ substantially.

A GPU timing benchmark is necessary before making strong claims about practical competitiveness.

On GPU, the relative costs of forward, backward, data loading, and Muon orthogonalization may be quite different.

---

## 10.5 Periodic Controller Experiment

Run controlled Muon/Adam with \(\rho\) measured every:

\[
1,5,10,20,50
\]

steps.

If \(k=10\) or \(k=20\) preserves most of the gain, the method becomes much more viable for large models.

This experiment is especially important for language-model viability.

---

# 11. Main Risks That Remain

## Risk 1: Tuned fixed baselines may close the gap

Controlled Muon may be mainly discovering that \(10^{-3}\) is too conservative.

This is still useful, but it weakens the claim that the controller is intrinsically better.

---

## Risk 2: Same-minibatch \(\rho\) may not predict validation performance

Earlier Adam CIFAR results showed that healthy same-minibatch \(\rho\) can coexist with worse validation performance.

The controller should eventually include validation/control-batch diagnostics or update-scale constraints.

Possible additions:

\[
\text{control-batch }\rho,
\]

\[
\frac{\|\Delta\theta\|}{\|\theta\|+\epsilon},
\]

\[
\text{gradient alignment},
\]

\[
\text{generalization-gap signal}.
\]

---

## Risk 3: CPU timing may not transfer to GPU/TPU

The overhead profile can change significantly on accelerator hardware.

The current CPU runs are useful for debugging and preliminary evidence, but they cannot settle large-scale viability.

---

## Risk 4: Muon implementation is educational, not optimized

The current Muon implementation uses CPU/NumPy orthogonalization.

A production Muon implementation may change the relative cost profile.

This could either help or hurt the controlled optimizer’s relative overhead.

---

## Risk 5: Full-dataset behavior is unknown

Most current results are subset benchmarks.

Full CIFAR-10, larger models, and eventually language-model-style experiments are needed.

---

# 12. Updated Bottom Line

The new overhead report makes the optimizer idea **more promising**.

The strongest updated conclusion is:

\[
\boxed{
\text{The controller can be wall-clock competitive when adaptive alpha greatly improves progress per step.}
}
\]

The most important practical finding is:

\[
\boxed{
\text{backtracking almost never happened, so overhead was usually only one extra forward pass.}
}
\]

The most important scientific caveat is:

\[
\boxed{
\text{the controlled Muon gains must be compared against tuned fixed-alpha Muon.}
}
\]

The most promising near-term direction is:

\[
\boxed{
\text{bounded periodic alpha control over a base optimizer and schedule.}
}
\]

For small and medium-scale neural networks, the idea is already worth pursuing seriously.

For language models, the per-step version is probably too expensive, but the periodic or hybrid controller version remains plausible.