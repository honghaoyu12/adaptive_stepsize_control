# Controlled Adam Algorithm

This note describes the controlled Adam optimizer implemented in
`src/controlled_adam/torch_optimizers.py` as `TorchControlledAdam`.

The method keeps Adam's adaptive direction, but replaces Adam's fixed scalar
learning rate with an outer controller. The controller compares the actual
loss decrease from a trial step with the first-order decrease predicted by the
current minibatch gradient. This ratio is used to decide whether the global
step multiplier `alpha` should increase, decrease, or stay near its current
value.

## Problem Setting

Let the model parameters be `theta`. At iteration `t`, the training loop samples
a minibatch `B_t` and defines the minibatch loss

```text
L_t(theta) = loss(theta; B_t).
```

The training loop computes

```text
ell_t = L_t(theta_t)
g_t = grad L_t(theta_t).
```

Then the optimizer is called with `ell_t`, the already-computed gradients, and a
closure that can re-evaluate `L_t(theta)` on the same minibatch without another
backward pass. The same-minibatch re-evaluation is important: the actual and
predicted decreases must refer to the same local objective.

## State Variables

Controlled Adam maintains the usual Adam moment state:

```text
m_t: first-moment accumulator
v_t: second-moment accumulator
t:   step counter
```

It also maintains controller state:

```text
alpha_t:       current global step multiplier
rho_bar_t:     optional exponential moving average of rho
alpha_min:     lower bound for alpha
alpha_max:     upper bound for alpha
rho_star:      target actual/predicted decrease ratio
rho_min:       minimum rho required to accept a trial step
kp:            proportional gain used when rho_bar_t >= rho_star
kp_down:       proportional gain used when rho_bar_t < rho_star
rho_beta:      EMA coefficient for rho smoothing
eta_bt:        backtracking shrink factor, called backtrack_shrink in code
J:             maximum number of backtracking retries
```

If `kp_down` is not provided, the implementation sets `kp_down = kp`, recovering
the symmetric controller.

## Core Quantities

Adam computes a direction, but controlled Adam interprets the scalar step length
separately.

For each parameter tensor, after updating Adam's moments,

```text
m_t = beta1 m_{t-1} + (1 - beta1) g_t
v_t = beta2 v_{t-1} + (1 - beta2) g_t^2

m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)

d_t = - m_hat_t / (sqrt(v_hat_t) + eps).
```

The trial update is

```text
theta_trial = theta_t + alpha_trial d_t.
```

The predicted decrease is the first-order decrease predicted by the current
minibatch gradient along the Adam direction:

```text
s_t = - <g_t, d_t>
predicted_decrease = alpha_trial s_t.
```

Here `s_t` is called `descent_score` in the code. If `s_t > 0`, then the Adam
direction is a descent direction according to the current minibatch gradient. If
`s_t <= 0`, the method does not try a parameter update.

After applying a trial step and re-evaluating the same minibatch loss,

```text
actual_decrease = L_t(theta_t) - L_t(theta_trial).
```

The agreement ratio is

```text
rho_t = actual_decrease / (predicted_decrease + ratio_eps).
```

Interpretation:

- `rho_t ~= 1`: the local linear prediction was accurate.
- `0 < rho_t < 1`: the step helped, but less than predicted.
- `rho_t <= 0`: the trial step did not decrease the minibatch loss.
- `rho_t > 1`: the step improved more than predicted.

## Algorithm 1: Controlled Adam With Same-Minibatch Trial Evaluation

```text
Inputs:
    initial parameters theta_0
    initial alpha alpha_0
    Adam parameters beta1, beta2, eps
    controller parameters rho_star, rho_min, kp, kp_down
    alpha bounds alpha_min, alpha_max
    alpha factor bounds gamma_min, gamma_max
    backtracking parameters J, eta_bt
    smoothing parameters use_rho_ema, rho_beta
    non_descent_shrink
    optional trust-region expansion parameters

Initialize:
    m_0 = 0
    v_0 = 0
    rho_bar = undefined
    alpha = clip(alpha_0, alpha_min, alpha_max)

For t = 1, 2, ...:

    1. Sample minibatch B_t.

    2. Compute minibatch loss and gradients:

           ell_before = L_t(theta)
           g = grad L_t(theta)

    3. Update Adam moments:

           m = beta1 * m + (1 - beta1) * g
           v = beta2 * v + (1 - beta2) * g^2

           m_hat = m / (1 - beta1^t)
           v_hat = v / (1 - beta2^t)

    4. Form the Adam direction:

           d = -m_hat / (sqrt(v_hat) + eps)

    5. Compute the predicted decrease per unit alpha:

           descent_score = - <g, d>

       If descent_score <= 0:

           alpha_next = clip(alpha * non_descent_shrink,
                             alpha_min,
                             alpha_max)

           theta is not changed.
           Adam moments remain updated.
           alpha = alpha_next.
           Continue to the next minibatch.

    6. Save the current parameters:

           theta_base = theta

    7. Backtracking trial loop:

       For j = 0, 1, ..., J:

           alpha_trial = alpha * eta_bt^j

           theta_trial = theta_base + alpha_trial * d
           theta = theta_trial

           ell_after = L_t(theta)        evaluated on the same minibatch

           predicted = alpha_trial * descent_score
           actual = ell_before - ell_after
           rho = actual / (predicted + ratio_eps)

           If reject_bad_steps is false, or rho > rho_min:

               Accept the trial step:
                   theta_{t+1} = theta_trial
                   alpha_used = alpha_trial

               Go to Step 8.

       If no trial step is accepted:

           Restore parameters:
               theta = theta_base

           Keep the Adam moments already updated from this minibatch.
           Use the last attempted rho and alpha_trial for diagnostics and for
           the controller update.

    8. Compute the rho signal used by the controller:

       If use_rho_ema is false:

           rho_control = rho

       Else if rho_bar is undefined:

           rho_bar = rho
           rho_control = rho_bar

       Else:

           rho_bar = rho_beta * rho_bar + (1 - rho_beta) * rho
           rho_control = rho_bar

    9. Compute the proportional alpha update:

           error = rho_control - rho_star

           If error >= 0:
               gain = kp
           Else:
               gain = kp_down

           raw_factor = exp(gain * error)
           factor = clip(raw_factor, gamma_min, gamma_max)

       In code:

           gamma_min = min_alpha_factor
           gamma_max = max_alpha_factor

   10. Optional trust-region expansion:

       If trust_region_expand is enabled, and

           j == 0
           rho_control >= trust_region_rho_threshold
           alpha_used <= trust_region_alpha_threshold

       then:

           factor = max(factor, trust_region_expand_factor)

       This expansion is applied after the ordinary factor clipping. The final
       alpha is still clipped by alpha_min and alpha_max.

   11. Update alpha:

           alpha_next = clip(alpha_used * factor,
                             alpha_min,
                             alpha_max)

           alpha = alpha_next

   12. Return diagnostics for this minibatch:

           loss_before
           loss_after
           alpha_used
           rho
           predicted_decrease
           actual_decrease
           accepted
           descent_score
           number of backtracks
           rho_control
           alpha_next
           alpha_next / alpha_used
           whether trust-region expansion occurred
```

## Algorithm 2: Variant Definitions

The project uses the same base algorithm with different choices of the rho
signal and trust expansion.

```text
Controlled Adam, raw-rho:
    use_rho_ema = false
    rho_control = rho_t
    trust_region_expand = false

Controlled Adam, EMA-rho:
    use_rho_ema = true
    rho_control = rho_bar_t
    trust_region_expand = false

Controlled Adam, EMA + trust:
    use_rho_ema = true
    rho_control = rho_bar_t
    trust_region_expand = true
```

In the CIFAR runner, `controlled_raw_rho` uses `rho_beta=0.0` because no EMA is
used. `controlled_ema` and `controlled_ema_trust` use the configured
`rho_beta`.

## Important Implementation Details

### The controller changes a global multiplier, not Adam's moments

The Adam direction is computed exactly from Adam's bias-corrected first and
second moments. The controller does not alter `m_t`, `v_t`, `beta1`, `beta2`, or
`eps`. It only changes the scalar multiplier `alpha`.

### The trial loss is evaluated on the same minibatch

The ratio `rho_t` is meaningful only because both `ell_before` and `ell_after`
are evaluated on the same minibatch `B_t`. If `ell_after` used a different
minibatch, the ratio would mix optimization progress with minibatch sampling
noise.

### Backtracking changes the accepted step length

If the first trial is poor, the method tries

```text
alpha, alpha * eta_bt, alpha * eta_bt^2, ...
```

up to `J` backtracking retries. The accepted step uses the first trial whose
`rho > rho_min`. This is what causes sharp alpha drops in the alpha-vs-step
plots when `eta_bt` is small, for example `eta_bt = 0.5`.

### The next alpha is based on the accepted alpha

After accepting a backtracked step, the next alpha is computed from
`alpha_used`, not from the original pre-backtracking alpha:

```text
alpha_next = clip(alpha_used * factor, alpha_min, alpha_max).
```

So backtracking has two effects:

1. The current parameter update is smaller.
2. The next controller state starts from the smaller accepted alpha.

### Rejected steps still update Adam moments

The implementation updates Adam's moment state before trial evaluation. If all
trial steps fail, parameters are restored to `theta_base`, but the Adam moments
remain updated with the current minibatch gradient. This is intentional in the
current implementation and matches the fact that the optimizer has consumed the
gradient information from this minibatch.

### Alpha can decrease in two ways

There are two distinct decrease mechanisms.

First, backtracking can reduce the step used on the current minibatch:

```text
alpha_used = alpha * eta_bt^j, with j > 0.
```

Second, the proportional controller can reduce the next alpha whenever
`rho_control < rho_star`:

```text
alpha_next = alpha_used * exp(kp_down * (rho_control - rho_star)).
```

Because `rho_control - rho_star` is negative in this case, the exponential
factor is below one. If `kp_down > kp`, decreases are stronger than increases
of the same absolute rho error.

### Alpha saturation means the rho target is permissive for that run

If alpha repeatedly reaches `alpha_max`, the controller is saying that recent
`rho_control` values are high enough relative to `rho_star` that it wants a
larger step. This does not prove that `alpha_max` is globally best for
validation accuracy. It only means that the local actual-vs-predicted decrease
test is satisfied often enough to push alpha upward.

## Compact Mathematical Summary

The controlled Adam direction is

```text
d_t = - m_hat_t / (sqrt(v_hat_t) + eps).
```

The descent score is

```text
s_t = - <g_t, d_t>.
```

For a trial multiplier `a`, the trial point is

```text
theta_trial(a) = theta_t + a d_t.
```

The agreement ratio is

```text
rho_t(a) =
    [L_t(theta_t) - L_t(theta_trial(a))]
    /
    [a s_t + ratio_eps].
```

The accepted multiplier is the first backtracking candidate

```text
a_t = alpha_t * eta_bt^j
```

such that

```text
rho_t(a_t) > rho_min,
```

unless bad-step rejection is disabled.

The control signal is either

```text
rho_control_t = rho_t
```

or

```text
rho_control_t = rho_beta rho_control_{t-1}
                + (1 - rho_beta) rho_t.
```

The next multiplier is

```text
e_t = rho_control_t - rho_star
k_t = kp       if e_t >= 0
      kp_down  otherwise

gamma_t = clip(exp(k_t e_t), min_alpha_factor, max_alpha_factor)

alpha_{t+1} = clip(a_t gamma_t, alpha_min, alpha_max).
```

With trust expansion enabled, `gamma_t` may be increased to at least
`trust_region_expand_factor` when the accepted step had no backtracking, the
rho signal is high, and alpha is still very small.
