# PI-Controlled Adam: Design and Comparison Notes

This document describes the PI-controlled Adam optimizer in this folder and
explains how it differs from the controlled Adam implementation in
`controlled_adam_project`.

The short version is:

- Adam still chooses the search direction.
- The PI controller chooses a global step-size multiplier.
- The control signal is the same-batch actual-vs-predicted decrease ratio.
- Compared with `controlled_adam_project`, this implementation adds an
  integral term and packages the method as a PyTorch `Optimizer`.

## Files

```text
pi_adam_optimizer/
├── pi_adam.py
├── demo_toy_regression.py
├── README.md
├── requirements.txt
└── PI_ADAM_DESIGN_AND_COMPARISON.md
```

The main implementation is `pi_adam.py`. The demo is intentionally small and
full-batch so the controller signal is deterministic and easy to inspect.

## Core Idea

At iteration `t`, the current parameters are `x_t`, the current gradient is
`g_t`, and Adam proposes a direction

```text
p_t = - m_hat_t / (sqrt(v_hat_t) + adam_eps)
```

where `m_hat_t` and `v_hat_t` are the usual Adam bias-corrected first and
second moment estimates.

The PI wrapper does not replace Adam's direction logic. Instead, it introduces
a scalar global multiplier `alpha_t`:

```text
x_trial = x_t + alpha_t * p_t
```

The first-order predicted decrease under the current gradient is

```text
predicted_decrease = -alpha_t * dot(g_t, p_t)
```

The actual decrease is measured by evaluating the same objective again after
the trial update:

```text
actual_decrease = f(x_t) - f(x_trial)
```

The controller signal is

```text
rho_t = actual_decrease / (predicted_decrease + predicted_decrease_eps)
```

When `rho_t` is close to 1, the local linear model predicted the step well.
When `rho_t` is between 0 and 1, the step helped but less than predicted. When
`rho_t` is negative, the same-batch loss increased.

## Same-Batch Closure Contract

For neural-network training, `f` is usually a minibatch loss. The optimizer must
evaluate both `f(theta_t)` and `f(theta_trial)` on the exact same batch. If the
two evaluations use different minibatches, `rho_t` mixes optimizer quality with
batch-to-batch noise.

The expected closure shape is:

```python
def closure(backward: bool = True):
    if backward:
        optimizer.zero_grad(set_to_none=True)
    output = model(x_batch)
    loss = criterion(output, y_batch)
    if backward:
        loss.backward()
    return loss
```

Inside `PIAdam.step(closure)`:

1. `closure(backward=True)` computes the original loss and gradients.
2. Adam state is updated and the direction is built.
3. The trial step is applied in-place.
4. `closure(backward=False)` evaluates the same-batch loss after the trial
   step.
5. `rho_t` is computed.
6. The PI controller updates `alpha`.
7. If rejection is enabled and the step is rejected, parameters are restored.

The extra cost is one additional forward pass per optimizer step.

## PI Controller

The implementation controls `log(alpha)` rather than `alpha` directly. This
guarantees positivity and makes updates multiplicative.

First, the raw ratio is optionally smoothed:

```text
rho_bar_t = rho_smoothing * rho_bar_{t-1}
            + (1 - rho_smoothing) * rho_t
```

On the first valid step, `rho_bar_t` is initialized to `rho_t`.

The control error is

```text
error_t = rho_bar_t - rho_star
```

The integral state is leaky:

```text
I_t = integral_decay * I_{t-1} + error_t
```

It is then clipped to `integral_clip` to reduce windup. The log-step update is

```text
delta_log_alpha = kp * error_t + ki * I_t
log_alpha_next <- log(alpha_used) + delta_log_alpha
```

`delta_log_alpha` can be clipped by `multiplicative_clip`, and `alpha` itself
is clipped to `[alpha_min, alpha_max]`.

## Direction Construction

The Adam direction is built from the current gradients:

1. Update first moment:

   ```text
   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
   ```

2. Update second moment:

   ```text
   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
   ```

3. Bias-correct:

   ```text
   m_hat_t = m_t / (1 - beta1^t)
   v_hat_t = v_t / (1 - beta2^t)
   ```

4. Form direction:

   ```text
   p_t = -m_hat_t / (sqrt(v_hat_t) + adam_eps)
   ```

If `weight_decay` is nonzero, the implementation follows PyTorch AdamW-style
decoupled decay during the parameter update:

```text
x_trial = (1 - alpha_t * weight_decay) * x_t + alpha_t * p_t
```

For clean controller experiments, `weight_decay=0.0` is easiest to interpret
because then `rho_t` measures the same loss used by the closure.

## Non-Descent Directions

The trust ratio is meaningful only when

```text
-dot(g_t, p_t) > 0
```

Adam momentum can occasionally produce a non-descent direction under the
current minibatch gradient. If `fallback_to_gradient=True`, `PIAdam` replaces
the Adam direction for that step with steepest descent:

```text
p_t = -g_t
```

If the fallback is disabled and the predicted decrease is not positive, the
optimizer does not take a trial step. It shrinks `alpha` by
`non_descent_shrink` and reports the step as skipped.

## Rejection, Backtracking, and Trust-Region Expansion

By default:

```python
reject_bad_steps=False
```

This is deliberate. In stochastic training, a single minibatch `rho_t` can be
noisy. Hard rejection may fight the natural noise of the training process.

If `reject_bad_steps=True`, the optimizer tries the current `alpha` first and
then retries smaller trial multipliers:

```text
trial_alpha = alpha * backtrack_shrink^k
```

for `k = 0, ..., max_backtracks`. The first trial with `rho_t > rho_min` is
accepted. If all trials fail, the original parameters are restored and the
step reports `accepted=False`.

When a trial is accepted, the PI controller updates the next multiplier from
the accepted `alpha_used`, not from the original pre-backtracking alpha. This
matches the old P-controller helpers' line-search behavior.

The implementation also includes the optional trust-region expansion used by
the old PyTorch controlled Adam helper. If the first trial succeeds, the
smoothed controller signal is at least `trust_region_rho_threshold`, and
`alpha_used` is no larger than `trust_region_alpha_threshold`, then the
controller enforces at least `trust_region_expand_factor` growth for the next
alpha, subject to `alpha_max`.

Important implementation detail: optimizer internal Adam moments are not rolled
back when a trial parameter step is rejected. That means rejection restores the
model weights, but the optimizer state still includes the latest gradients.
This is usually acceptable for the intended stochastic setting because
rejection is off by default. For deterministic line-search-style experiments,
this should be kept in mind.

## Diagnostics

Each call to `step` returns a `PIAdamDiagnostics` object with:

- `loss_before`: same-batch loss before the trial step.
- `loss_after`: same-batch loss after the trial step, if evaluated.
- `actual_decrease`: `loss_before - loss_after`.
- `predicted_decrease`: first-order predicted decrease.
- `rho`: raw actual-over-predicted ratio.
- `rho_bar`: smoothed ratio used by the controller.
- `error`: `rho_bar - rho_star`.
- `integral`: current leaky integral state.
- `alpha`: updated global multiplier after controller update.
- `log_alpha`: current log multiplier.
- `delta_log_alpha`: controller's latest log update.
- `accepted`: whether the trial parameter step remains applied.
- `used_fallback_direction`: whether `-g` replaced Adam's direction.
- `backtracks`: number of backtracking reductions used.
- `alpha_next`: next global multiplier after the controller update.
- `alpha_update_factor`: `alpha_next / alpha_used`.
- `trust_region_expanded`: whether the expansion rule fired.
- `skipped_reason`: reason no normal trial step occurred, if any.

These fields are useful for plotting loss, alpha, rho, and integral state over
time. The demo writes those plots to `outputs/`.

## Comparison With `controlled_adam_project`

There are two controlled Adam implementations in `controlled_adam_project`:

- A deterministic NumPy optimizer in
  `controlled_adam_project/src/controlled_adam/optimizers.py`.
- A PyTorch helper class in
  `controlled_adam_project/src/controlled_adam/torch_optimizers.py`.

The PI Adam implementation differs in several important ways.

### Controller Law

`controlled_adam_project` uses a proportional update:

```text
alpha_next = alpha_used * exp(kp * (rho_control - rho_star))
```

The PyTorch helper can smooth `rho` with an EMA, but it does not integrate
past error.

`PIAdam` uses:

```text
log_alpha_next = log_alpha
                 + kp * (rho_bar - rho_star)
                 + ki * integral
```

The integral term lets persistent under- or over-shooting accumulate and push
the multiplier until the long-run measured ratio is closer to target.

### API Shape

`TorchControlledAdam` is a custom helper. The training loop computes gradients,
then calls:

```python
step = optimizer.step(loss_before, reevaluate_loss)
```

`PIAdam` subclasses `torch.optim.Optimizer` and owns the closure workflow:

```python
diag = optimizer.step(closure)
```

This makes the PI version feel more like a standard PyTorch optimizer, although
it still requires a closure because of the same-batch reevaluation.

### Step Acceptance and Backtracking

`TorchControlledAdam` supports multiple trial alphas through backtracking. It
can reject poor steps and retry smaller trial steps before giving up.

`PIAdam` now mirrors this behavior: when `reject_bad_steps=True`, it tries the
current alpha and then backtracks by `backtrack_shrink` up to
`max_backtracks`. The difference is the next alpha is chosen by a PI controller
instead of the old proportional-only controller.

### Trust-Region Expansion

`TorchControlledAdam` has explicit trust-region expansion logic: if the
controller signal is very good and alpha is small, it can force a larger alpha
growth factor.

`PIAdam` has the same optional expansion branch. It is controlled by
`trust_region_expand`, `trust_region_rho_threshold`,
`trust_region_alpha_threshold`, and `trust_region_expand_factor`.

### Handling Non-Descent Directions

`TorchControlledAdam` shrinks alpha and stays put when the Adam direction is
not descent-like.

`PIAdam` can instead fall back to the steepest-descent direction. This means
more steps may still make progress even when Adam's momentum is temporarily
misaligned with the current gradient.

### Scope

`controlled_adam_project` is broader. It includes deterministic objective
benchmarks, PyTorch image demos, tests, and multiple controller variants.

`pi_adam_optimizer` is narrower: it is a focused, standalone PI-controlled
Adam wrapper plus one compact demo.

## Hyperparameter Guidance

The most important parameters are:

- `alpha0`: initial global multiplier.
- `rho_star`: target actual-over-predicted ratio.
- `kp`: proportional gain.
- `ki`: integral gain.
- `rho_smoothing`: EMA coefficient for noisy minibatches.
- `use_rho_ema`: whether the controller uses smoothed or raw rho.
- `integral_decay`: leak factor for the integral state.
- `integral_clip`: anti-windup bounds.
- `multiplicative_clip`: per-step alpha change limits.
- `alpha_min`, `alpha_max`: hard multiplier bounds.
- `reject_bad_steps`, `rho_min`: optional step rejection.
- `max_backtracks`, `backtrack_shrink`: optional line-search retries.
- `trust_region_expand`: optional small-alpha expansion rule.

For deterministic or full-batch training, a reasonable starting point is:

```python
PIAdam(
    model.parameters(),
    alpha0=1e-3,
    rho_star=0.8,
    kp=0.05,
    ki=0.001,
    integral_decay=0.95,
    rho_smoothing=0.9,
    multiplicative_clip=(0.8, 1.25),
)
```

For noisy minibatch training, start more conservatively:

```python
kp = 0.01
ki = 0.0
rho_smoothing = 0.99
multiplicative_clip = (0.9, 1.1)
```

After the proportional controller behaves sensibly, add a small `ki`.

## Practical Interpretation

PIAdam asks a simple question after every proposed Adam step:

```text
Did the same-batch loss decrease about as much as the first-order model said?
```

If the answer is consistently yes or better, `alpha` grows. If the answer is
consistently worse than the target, `alpha` shrinks. The integral term gives
the optimizer memory of persistent bias in this ratio, while clipping and
smoothing keep the feedback loop from reacting too violently to noisy
minibatches.
