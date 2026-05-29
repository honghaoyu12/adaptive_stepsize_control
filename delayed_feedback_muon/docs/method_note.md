# Method Note: Delayed Feedback Muon

This note explains the optimizer implemented in this project and how it differs from both vanilla Muon and the earlier same-step controller that evaluates `f(x_{t+1})` immediately.

---

## 1. Vanilla Muon inner update

Muon is an optimizer for matrix-like neural network parameters. Its basic idea is:

1. build a momentum update;
2. orthogonalize that update using Newton-Schulz iterations;
3. step along the orthogonalized update.

For a weight matrix `W`, gradient `G_t`, and momentum coefficient `mu`, Muon forms

```math
B_t = \operatorname{lerp}(B_{t-1}, G_t, 1-\mu).
```

With Nesterov momentum,

```math
\widetilde B_t = \operatorname{lerp}(G_t, B_t, \mu).
```

Then it approximately computes

```math
O_t \approx \operatorname{Ortho}(\widetilde B_t).
```

The Muon direction is

```math
p_t = -O_t.
```

In this implementation, 2D tensors are treated as matrix-like and use Muon by
default. Non-2D tensors use auxiliary AdamW. This matches the local
`torch.optim.Muon` implementation's 2D-parameter scope for the automatic path.

The current code was checked against PyTorch `2.10.0`: Muon momentum/Nesterov,
shape learning-rate adjustment, decoupled weight decay, and the AdamW fallback
match the supported local `torch.optim.Muon`/`torch.optim.AdamW` behavior. The
delayed controller wrapped around those directions remains the experimental
part.

---

## 2. Our outer controller

Our outer controller is independent of the specific inner optimizer. It only needs:

```math
p_t = \text{inner optimizer direction}
```

and the current gradient

```math
g_t = \nabla f(x_t).
```

The update is

```math
x_{t+1}=x_t+\eta\alpha_t p_t,
```

where `eta` is the base learning rate and `alpha_t` is the adaptive global multiplier.

The first-order predicted decrease is

```math
\Delta \hat f_t = -\eta\alpha_t g_t^\top p_t.
```

For Muon, `p_t` is the orthogonalized momentum direction. For auxiliary AdamW, `p_t` is the AdamW direction.

---

## 3. Old same-step controller

The earlier version computed

```math
x_{t+1}^{\mathrm{trial}} = x_t+\eta\alpha_t p_t
```

and immediately evaluated

```math
f(x_{t+1}^{\mathrm{trial}}).
```

Then it formed

```math
\rho_t
=
\frac{f(x_t)-f(x_{t+1}^{\mathrm{trial}})}{-\eta\alpha_t g_t^\top p_t}.
```

This gives accurate same-step feedback, and in deterministic optimization it can be used for step rejection. The cost is that in neural-network training it usually requires an extra forward pass.

---

## 4. New delayed controller

The delayed controller stores the previous loss and previous predicted decrease.

At step `t-1`, store:

```math
f_{t-1},
\qquad
\Delta \hat f_{t-1}.
```

At step `t`, the training loop naturally computes `f_t`. Then the controller estimates the previous step's quality:

```math
\tilde\rho_{t-1}
=
\frac{f_{t-1}-f_t}{\Delta \hat f_{t-1}+\epsilon}.
```

Then it updates `alpha_t` before the current step.

For P control:

```math
\alpha_t
=
\alpha_{t-1}\exp[K_P(\bar\rho_{t-1}-\rho^\star)].
```

For optional PI/PID control:

```math
\log\alpha_t
=
\log\alpha_{t-1}
+K_Pe_{t-1}
+K_II_t
+K_DD_t.
```

where

```math
e_{t-1}=\bar\rho_{t-1}-\rho^\star.
```

---

## 5. Why this is lower overhead

The delayed version does not evaluate the trial point immediately. It uses the next loss value computed by the normal training loop.

Therefore it adds no extra forward pass and no extra backward pass.

The overhead relative to vanilla Muon is mainly:

- storing a few scalar controller values;
- computing the predicted-decrease diagnostic;
- smoothing and clipping scalar feedback values.

The Newton-Schulz work is not caused by the controller; it is part of Muon itself.

---

## 6. What we lose

The delayed version is not a true line search.

Because it applies the step first and evaluates later, it cannot reject a bad step before it happens.

In minibatch training, the loss difference may compare different batches:

```math
f_{B_{t-1}}(\theta_{t-1}) - f_{B_t}(\theta_t).
```

So the signal is noisy. This is why the optimizer includes smoothing and clipping.

---

## 7. Practical interpretation

The same-step version is best interpreted as:

```text
line-search/trust-region-like step-quality control
```

The delayed version is best interpreted as:

```text
low-overhead delayed learning-rate regulation
```

For large neural networks, this is often the more practical tradeoff.
