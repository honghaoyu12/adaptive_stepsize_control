# Method Note: Delayed-Feedback Adam

This document explains the optimizer in this repository and how it differs from the earlier optimizer that directly used `f(x_{t+1})`.

---

## 1. Motivation

The earlier controller used the actual-vs-predicted decrease ratio

```math
\rho_t
=
\frac{f(x_t)-f(x_{t+1})}{\Delta\hat f_t}.
```

For a step

```math
x_{t+1}=x_t+\alpha_t p_t,
```

the first-order predicted decrease is

```math
\Delta\hat f_t=-\alpha_t\nabla f(x_t)^\top p_t.
```

This is attractive because it directly measures whether the proposed step size was appropriate.

However, in neural-network training, evaluating `f(x_{t+1})` immediately after the step often requires an extra forward pass. If one wants to use the same minibatch, the overhead is unavoidable.

The delayed version avoids this by using the next normally-computed loss value.

---

## 2. Same-step controller

The same-step controller does this:

1. Compute current loss and gradient at `x_t`.
2. Compute an optimizer direction `p_t`.
3. Form a trial step:

```math
x_{t+1}^{\mathrm{trial}}=x_t+\alpha_t p_t.
```

4. Evaluate the objective at the trial point:

```math
f(x_{t+1}^{\mathrm{trial}}).
```

5. Compute

```math
\rho_t
=
\frac{f(x_t)-f(x_{t+1}^{\mathrm{trial}})}{-\alpha_t g_t^\top p_t}.
```

6. Update the next learning-rate multiplier.

This is close to line-search and trust-region logic. It can also reject bad trial steps.

The downside is overhead.

---

## 3. Delayed controller

The delayed controller takes the step without immediately evaluating the trial point.

At step `t-1`, it stores:

```math
f_{t-1},
\qquad
\Delta\hat f_{t-1}=-\alpha_{t-1}g_{t-1}^\top p_{t-1}.
```

At step `t`, the normal training loop computes `f_t`. The controller then estimates the previous step's quality:

```math
\tilde\rho_{t-1}
=
\frac{f_{t-1}-f_t}{\Delta\hat f_{t-1}+\epsilon}.
```

The controller error is

```math
e_{t-1}=\bar\rho_{t-1}-\rho^\star.
```

The P controller updates

```math
\log\alpha_t
=
\log\alpha_{t-1}+K_Pe_{t-1}.
```

Equivalently,

```math
\alpha_t
=
\alpha_{t-1}\exp(K_Pe_{t-1}).
```

---

## 4. Why the delayed version is lower-overhead

In a standard training loop, the current loss is already computed before `loss.backward()`.

Therefore, the delayed controller can use that loss to evaluate the previous step.

It avoids:

- an additional forward pass;
- a second evaluation of the same minibatch;
- closure-based trial-step reevaluation.

The only extra costs are:

- storing previous loss and previous predicted decrease;
- computing the predicted decrease for the current step;
- smoothing and clipping scalar diagnostics.

---

## 5. What is lost compared with the old version?

The delayed version loses immediate step-quality feedback.

The same-step controller observes

```math
f(x_t)-f(x_{t+1}).
```

The delayed controller observes it one step later.

In deterministic optimization, that is still accurate, just delayed.

In minibatch training, it becomes noisy because the loss at step `t-1` and the loss at step `t` may come from different minibatches.

So the delayed version is no longer a precise line search. It is better interpreted as a zero-extra-forward-pass adaptive learning-rate regulator.

---

## 6. Why Adam is the inner optimizer

Adam gives a direction

```math
p_t=-\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

The controller does not replace Adam. It scales Adam's global step size.

The full update is

```math
x_{t+1}=x_t+\eta\alpha_t p_t.
```

Adam controls the preconditioned direction. The delayed feedback controller controls the global multiplier `alpha_t`.

With decoupled weight decay enabled, the inner direction/update follows the
supported local `torch.optim.AdamW` path. This was checked against PyTorch
`2.10.0`; the intentional difference from PyTorch is the outer delayed
feedback multiplier, not the AdamW moment or decoupled decay mechanics.

---

## 7. Controller variants

The optimizer supports P, PI, and PID forms.

P:

```math
\log\alpha_t=\log\alpha_{t-1}+K_Pe_{t-1}.
```

PI:

```math
I_t=\lambda_I I_{t-1}+e_{t-1},
```

```math
\log\alpha_t=\log\alpha_{t-1}+K_Pe_{t-1}+K_II_t.
```

PID:

```math
D_t=\lambda_DD_{t-1}+(1-\lambda_D)(e_{t-1}-e_{t-2}),
```

```math
\log\alpha_t=\log\alpha_{t-1}+K_Pe_{t-1}+K_II_t+K_DD_t.
```

For neural networks, use P first. Add PI only after the P version is stable. Be cautious with D because minibatch noise makes derivative signals unstable.

---

## 8. Failure modes

Potential problems:

1. **Noisy minibatch feedback**: consecutive minibatches may have different difficulty.
2. **Delayed response**: the controller reacts one step late.
3. **Small predicted decrease**: the ratio can become unstable if the denominator is tiny.
4. **Non-descent Adam direction**: momentum can occasionally point uphill relative to the current gradient.
5. **Overreaction**: without clipping, one noisy loss can change alpha too much.
6. **Integral windup**: PI/PID can accumulate large integral errors if alpha is saturated.

The implementation addresses these with smoothing, clipping, denominator floors, optional gradient fallback, multiplier bounds, and integral bounds.

---

## 9. Recommended first experiment

Use delayed P control only:

```python
optimizer = DelayedFeedbackAdam(
    model.parameters(),
    lr=1e-3,
    alpha_init=1.0,
    rho_star=0.8,
    kp=0.05,
    ki=0.0,
    kd=0.0,
    rho_beta=0.95,
    rho_clip=(-1.0, 2.0),
    multiplier_bounds=(0.8, 1.25),
    alpha_bounds=(0.1, 10.0),
)
```

If alpha collapses, reduce `kp`, lower `rho_star`, or increase smoothing.

If alpha saturates high, raise `rho_star`, reduce `kp`, or lower `alpha_bounds[1]`.

If alpha oscillates, reduce `kp`, increase `rho_beta`, or tighten `multiplier_bounds`.

## 10. Neural calibration from CIFAR ResNet

A later CIFAR-10 ResNet comparison tested delayed Adam variants beside the
same-step controlled Adam variants:

```text
outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/
```

Setup: 10k train / 2k test CIFAR-10 subset, 20 epochs, batch size 128, seed 123,
base learning rate `1e-3`.

Final results:

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

The key lesson is not just the accuracy ranking. The delayed rho signal lived
on a different scale from the same-step rho signal. Same-step controlled Adam
finished with mean rho around `0.88` and expanded to the `1.5e-3` cap. Delayed
variants finished with mean rho around `0.13-0.18` and mostly sat on their
alpha floors.

Therefore, for shuffled minibatch neural training, delayed feedback should be
calibrated separately from same-step control. Reusing `rho_star=0.7-0.8` can
make the delayed controller shrink too aggressively. A reasonable next neural
sweep is:

```text
rho_star: 0.15, 0.20, 0.25, 0.30
alpha floor: 0.8x, 0.9x, 1.0x, 1.1x base LR
alpha cap: 1.25x, 1.5x, 1.75x base LR
rho_beta: 0.90 or 0.95
kp: 0.01 or 0.02
```

Treat delayed Adam as a low-overhead regulator with its own feedback scale, not
as a drop-in no-extra-forward equivalent of same-step controlled Adam.
