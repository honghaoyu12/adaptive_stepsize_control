# Raw-Rho Controlled Optimizer Algorithm

This note explains the raw-rho controller and its two main extensions used in
the Adam and Muon subprojects:

- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

The key point is that each training step has three separate stages:

1. Choose a direction using a base optimizer such as Adam or Muon.
2. Decide whether a candidate step should be accepted.
3. Update the next global step size using the raw measured ratio `rho_t`.

The backtracking loop is **not** part of the `s_t` calculation. The quantity
`s_t` is computed once before backtracking starts. It is a gate that checks
whether the proposed direction is a descent direction.

## Notation

- `theta_t`: current parameters.
- `B_t`: current minibatch.
- `f_B(theta)`: loss evaluated on minibatch `B`.
- `L_before = f_{B_t}(theta_t)`: loss before the trial step.
- `g_t = grad f_{B_t}(theta_t)`: minibatch gradient.
- `p_t`: direction proposed by the base optimizer.
- `alpha_t`: current global step size.
- `s_t = - <g_t, p_t>`: predicted decrease per unit step size.
- `rho_t`: actual decrease divided by predicted decrease.
- `rho_star`: target value for `rho_t`.
- `rho_min`: minimum acceptable `rho_t` for accepting a step.
- `M`: maximum number of backtracking reductions.
- `gamma`: backtracking shrink factor.
- `K_p`: proportional controller gain.
- `c_min, c_max`: lower and upper bounds on the multiplicative alpha update.
- `alpha_min, alpha_max`: lower and upper bounds on alpha itself.

In the current neural benchmark defaults:

```text
alpha_0 = 1e-3
rho_star = 0.7
rho_min = 0.0
K_p = 0.05
alpha_min = 1e-5
alpha_max = 5e-2
c_min = 0.8
c_max = 1.05
M = 3
gamma = 0.5
non_descent_shrink = 0.5
```

Because `M = 3` and `gamma = 0.5`, the algorithm may try four step sizes:

```text
alpha_t
0.5 * alpha_t
0.25 * alpha_t
0.125 * alpha_t
```

That is the original step plus three backtracked steps.

## Base Direction

The raw-rho controller does not replace Adam or Muon. Adam or Muon still chooses
the direction `p_t`.

For Adam:

```text
p_t = - m_hat_t / (sqrt(v_hat_t) + eps_adam)
```

For Muon:

```text
p_t = - Orthogonalize(momentum-adjusted gradient)
```

The controller only chooses the scalar multiplier `alpha_t`.

The candidate update has the form:

```text
theta_trial = theta_t + alpha_trial * p_t
```

## Why Compute `s_t`?

Before trying a step, the algorithm checks whether the proposed direction is
locally predicted to decrease the loss.

Using first-order Taylor expansion:

```text
f_B(theta_t + alpha * p_t)
  approx f_B(theta_t) + alpha * <g_t, p_t>
```

Therefore the predicted decrease is:

```text
predicted decrease = - alpha * <g_t, p_t>
```

We define:

```text
s_t = - <g_t, p_t>
```

So:

```text
predicted decrease = alpha * s_t
```

If `s_t > 0`, the direction is predicted to reduce the minibatch loss.

If `s_t <= 0`, the direction is not a descent direction. In that case the
algorithm does not run backtracking, because even an infinitesimally small step
is not predicted to help.

Instead it rejects the update immediately:

```text
theta_{t+1} = theta_t
alpha_{t+1} = clip(alpha_t * non_descent_shrink, alpha_min, alpha_max)
```

With the default `non_descent_shrink = 0.5`, this halves alpha after a
non-descent direction.

## Backtracking Is Separate

Backtracking starts only if:

```text
s_t > 0
```

At that point the direction is reasonable, but the current `alpha_t` may still
be too large. Backtracking tries smaller and smaller values of alpha until the
same-minibatch loss actually decreases.

For each attempt `j = 0, 1, ..., M`:

```text
alpha_trial = alpha_t * gamma^j
theta_trial = theta_t + alpha_trial * p_t
L_after = f_{B_t}(theta_trial)
```

The loss `L_after` must be evaluated on the same minibatch `B_t`. Otherwise the
ratio would mix optimizer progress with minibatch sampling noise.

Then compute:

```text
predicted = alpha_trial * s_t
actual = L_before - L_after
rho_t = actual / (predicted + epsilon)
```

Here:

- `predicted` is what the first-order model expected.
- `actual` is what really happened on the same minibatch.
- `rho_t` compares actual progress to predicted progress.

The step is accepted when:

```text
rho_t > rho_min
```

With the default `rho_min = 0`, this means:

```text
actual > 0
L_after < L_before
```

So the step is accepted if it actually decreases the same-minibatch loss.

If the condition fails, the algorithm tries a smaller alpha. With the defaults:

```text
try alpha_t
if rejected, try 0.5 * alpha_t
if rejected, try 0.25 * alpha_t
if rejected, try 0.125 * alpha_t
if rejected, reject the whole step
```

If all attempts fail, the parameters are restored:

```text
theta_{t+1} = theta_t
```

## Raw-Rho Alpha Update

After the step attempt, the raw-rho variant uses the immediate measured ratio:

```text
rho_control = rho_t
```

There is no EMA smoothing:

```text
rho_control != beta * rho_control_old + (1 - beta) * rho_t
```

There is also no trust-region expansion in the raw-rho variant.

The next alpha is updated by:

```text
factor_raw = exp(K_p * (rho_t - rho_star))
factor = clip(factor_raw, c_min, c_max)
alpha_{t+1} = clip(alpha_used * factor, alpha_min, alpha_max)
```

With the current defaults:

```text
factor_raw = exp(0.05 * (rho_t - 0.7))
factor = clip(factor_raw, 0.8, 1.05)
alpha_{t+1} = clip(alpha_used * factor, 1e-5, 5e-2)
```

Interpretation:

- If `rho_t > rho_star`, the step worked better than the target, so alpha tends
  to increase.
- If `rho_t = rho_star`, alpha stays almost unchanged.
- If `0 < rho_t < rho_star`, the step decreased the loss but less than desired,
  so alpha tends to shrink.
- If `rho_t <= 0`, the step is rejected or backtracked because the loss did not
  decrease.

## EMA-Rho Variant

The EMA-rho variant uses the same direction calculation, same `s_t` descent
gate, same backtracking loop, and same step acceptance rule.

The only difference is the control signal used to update alpha.

Raw-rho uses:

```text
rho_control = rho_t
```

EMA-rho instead keeps an exponential moving average:

```text
if rho_ema is uninitialized then
    rho_ema = rho_t
else
    rho_ema = beta * rho_ema + (1 - beta) * rho_t
end if

rho_control = rho_ema
```

In the current defaults:

```text
beta = rho_beta = 0.9
```

Then alpha is updated using `rho_control`:

```text
factor_raw = exp(K_p * (rho_control - rho_star))
factor = clip(factor_raw, c_min, c_max)
alpha_{t+1} = clip(alpha_used * factor, alpha_min, alpha_max)
```

So with defaults:

```text
rho_ema = 0.9 * rho_ema + 0.1 * rho_t
factor_raw = exp(0.05 * (rho_ema - 0.7))
factor = clip(factor_raw, 0.8, 1.05)
alpha_{t+1} = clip(alpha_used * factor, 1e-5, 5e-2)
```

The motivation is to avoid letting one noisy minibatch measurement dominate the
global step-size update. The step acceptance test still uses the current trial
step's measured `rho_t`; EMA only smooths the signal used for alpha adaptation.

## EMA Plus Trust-Region Variant

The EMA-trust variant starts from the EMA-rho variant:

```text
rho_control = rho_ema
```

It then computes the usual multiplicative alpha update:

```text
factor_raw = exp(K_p * (rho_control - rho_star))
factor = clip(factor_raw, c_min, c_max)
```

After that, it may force a larger expansion if all trust-region conditions are
met.

In the implementation, trust expansion fires only when:

```text
trust_region_expand is true
and backtracks == 0
and rho_control >= trust_region_rho_threshold
and alpha_used <= trust_region_alpha_threshold
```

With the current defaults:

```text
trust_region_expand = true for controlled_ema_trust
trust_region_rho_threshold = 0.9
trust_region_alpha_threshold = 1e-4
trust_region_expand_factor = 1.5
```

If those conditions hold:

```text
factor = max(factor, trust_region_expand_factor)
```

Then alpha is updated normally:

```text
alpha_{t+1} = clip(alpha_used * factor, alpha_min, alpha_max)
```

The idea is borrowed from classical trust-region methods:

- if the local model is very trustworthy, meaning `rho_control` is high;
- and the trial step was accepted without backtracking;
- and the current radius/step size is still tiny;

then the controller should expand more aggressively instead of growing alpha
only by the usual small capped factor.

In many of our neural benchmarks this trust expansion did not actually change
behavior, because `alpha_used` was already above the tiny threshold
`1e-4`. That is why `controlled_ema` and `controlled_ema_trust` often produced
identical curves.

The threshold must also be compatible with the hard alpha floor. In the
balanced CIFAR-10 ResNet Adam benchmark, the controller used
`alpha_min = 1e-3` and `alpha_max = 1.5e-3`, while the trust threshold remained
`trust_region_alpha_threshold = 1e-4`. Because `1e-4 < alpha_min`, the condition
`alpha_used <= trust_region_alpha_threshold` was effectively unreachable. A
diagnostic check confirmed `0/1580` trust expansions for each of seeds `123`,
`456`, and `789`, so `controlled_ema_trust` was algorithmically identical to
`controlled_ema` in that benchmark.

For an Adam-scale experiment where the alpha floor is intentionally kept near
the vanilla learning rate, the trust threshold should be moved near that floor:

```text
alpha_min = 1e-3
trust_region_alpha_threshold = 1e-3 or 1.05e-3
trust_region_expand_factor = 1.1 or 1.2
```

This makes the rule mean "if alpha is near the active floor and rho is good,
expand gently." It is different from the older collapse-recovery setting where
`alpha_min` could be much smaller and `1e-4` was a reachable tiny-alpha region.

## Full Pseudocode

```text
Algorithm: Raw-Rho Controlled Optimizer

Inputs:
  theta_0
  alpha_0
  base optimizer direction rule Direction(...)
  rho_star, rho_min
  K_p
  alpha_min, alpha_max
  c_min, c_max
  M
  gamma
  non_descent_shrink
  epsilon

for t = 0, 1, 2, ... do

  Sample minibatch B_t

  L_before = f_{B_t}(theta_t)
  g_t = grad f_{B_t}(theta_t)

  p_t = Direction(g_t, optimizer_state)

  s_t = - <g_t, p_t>

  if s_t <= 0 then
      theta_{t+1} = theta_t
      alpha_{t+1} = clip(alpha_t * non_descent_shrink,
                         alpha_min, alpha_max)
      continue
  end if

  theta_original = theta_t
  accepted = false

  for j = 0, 1, ..., M do

      alpha_trial = alpha_t * gamma^j
      theta_trial = theta_original + alpha_trial * p_t

      L_after = f_{B_t}(theta_trial)

      predicted = alpha_trial * s_t
      actual = L_before - L_after
      rho_t = actual / (predicted + epsilon)

      if rho_t > rho_min then
          accepted = true
          alpha_used = alpha_trial
          break
      end if

  end for

  if accepted then
      theta_{t+1} = theta_trial
  else
      theta_{t+1} = theta_original
      alpha_used = alpha_trial
  end if

  factor_raw = exp(K_p * (rho_t - rho_star))
  factor = clip(factor_raw, c_min, c_max)
  alpha_{t+1} = clip(alpha_used * factor, alpha_min, alpha_max)

end for
```

## Full Pseudocode With EMA And Trust Options

This version shows all three variants in one algorithm by changing two boolean
flags:

```text
use_rho_ema:
  false for controlled_raw_rho
  true  for controlled_ema and controlled_ema_trust

trust_region_expand:
  false for controlled_raw_rho and controlled_ema
  true  for controlled_ema_trust
```

```text
Algorithm: Controlled Optimizer With Optional EMA And Trust Expansion

Inputs:
  theta_0
  alpha_0
  base optimizer direction rule Direction(...)
  rho_star, rho_min
  K_p
  alpha_min, alpha_max
  c_min, c_max
  M
  gamma
  non_descent_shrink
  epsilon
  use_rho_ema
  beta
  trust_region_expand
  trust_region_rho_threshold
  trust_region_alpha_threshold
  trust_region_expand_factor

Initialize:
  rho_ema = undefined

for t = 0, 1, 2, ... do

  Sample minibatch B_t

  L_before = f_{B_t}(theta_t)
  g_t = grad f_{B_t}(theta_t)

  p_t = Direction(g_t, optimizer_state)

  s_t = - <g_t, p_t>

  if s_t <= 0 then
      theta_{t+1} = theta_t
      alpha_{t+1} = clip(alpha_t * non_descent_shrink,
                         alpha_min, alpha_max)
      continue
  end if

  theta_original = theta_t
  accepted = false

  for j = 0, 1, ..., M do

      alpha_trial = alpha_t * gamma^j
      theta_trial = theta_original + alpha_trial * p_t

      L_after = f_{B_t}(theta_trial)

      predicted = alpha_trial * s_t
      actual = L_before - L_after
      rho_t = actual / (predicted + epsilon)

      if rho_t > rho_min then
          accepted = true
          alpha_used = alpha_trial
          backtracks = j
          break
      end if

  end for

  if accepted then
      theta_{t+1} = theta_trial
  else
      theta_{t+1} = theta_original
      alpha_used = alpha_trial
      backtracks = M
  end if

  if use_rho_ema then
      if rho_ema is undefined then
          rho_ema = rho_t
      else
          rho_ema = beta * rho_ema + (1 - beta) * rho_t
      end if
      rho_control = rho_ema
  else
      rho_control = rho_t
  end if

  factor_raw = exp(K_p * (rho_control - rho_star))
  factor = clip(factor_raw, c_min, c_max)

  if trust_region_expand
     and backtracks == 0
     and rho_control >= trust_region_rho_threshold
     and alpha_used <= trust_region_alpha_threshold then

      factor = max(factor, trust_region_expand_factor)

  end if

  alpha_{t+1} = clip(alpha_used * factor, alpha_min, alpha_max)

end for
```

## Short Mental Model

The algorithm is easiest to remember as:

```text
1. Adam/Muon chooses the direction.
2. s_t checks whether the direction is descent at all.
3. Backtracking checks whether the chosen alpha is too large.
4. rho_t measures how well the step matched the first-order prediction.
5. Raw-rho uses rho_t directly to choose the next alpha.
6. EMA-rho smooths rho_t before choosing the next alpha.
7. EMA-trust can force faster alpha expansion when rho is good, no
   backtracking was needed, and alpha is still tiny.
```
