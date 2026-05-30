# Controlled Adam Algorithm

This document describes the simplified raw same-minibatch controlled Adam core.

The key idea is:

```text
Adam chooses the direction.
The controller chooses the scalar step multiplier alpha.
```

The controller evaluates a trial step on the same minibatch used to compute the
gradient. It compares the actual same-minibatch loss decrease with the decrease
predicted by the current gradient.

## Flow-To-Algorithm Map

| Flow item | Section |
| --- | --- |
| start minibatch | 1 |
| compute `loss_before` on this minibatch | 1 |
| compute gradient on this minibatch | 1 |
| update Adam moments using this gradient | 2 |
| build Adam direction from the updated moments | 2 |
| check whether Adam direction is downhill | 3 |
| replace non-downhill Adam direction with scaled negative gradient | 3 |
| if no usable downhill direction exists, shrink alpha and skip movement | 4 |
| save current theta as `theta_base` | 5 |
| try trial steps using alpha, then smaller backtracked alphas | 5 |
| temporarily set theta to the trial point | 5 |
| compute `loss_after` on the same minibatch | 5 |
| compute predicted decrease | 6 |
| compute actual decrease | 6 |
| compute rho from actual decrease and safe predicted decrease | 6 |
| accept a trial step if it reduced loss enough | 7 |
| reject a trial alpha and try the next smaller alpha | 7 |
| if all trials are rejected, restore theta | 8 |
| compute `rho_control` from clipped rho | 9 |
| compare `rho_control` with `rho_star` | 9 |
| propose increasing or decreasing next alpha | 9 |
| clip the proposed alpha change | 9 |
| after full rejection, prevent alpha from increasing | 9 |
| set `alpha = alpha_next` | 9 |
| record diagnostics and move to next minibatch | 10 |

## Notation

The math uses compact symbols. The matching code names are shown explicitly.

- $\theta_t$: parameters at the start of minibatch step $t$
- $\theta_b$: saved base parameters, code `theta_base`
- $\theta^{(j)}$: trial parameters for backtracking attempt $j$
- $B_t$: current minibatch
- $L_t(\theta)$: loss on minibatch $B_t$
- $\ell_b = L_t(\theta_t)$: code `loss_before`
- $\ell_j = L_t(\theta^{(j)})$: code `loss_after`
- $g_t = \nabla_\theta L_t(\theta_t)$: minibatch gradient
- $d_{\mathrm{adam}}$: Adam direction, code `d_adam`
- $d_t$: direction used for the trial step
- $s_t$: predicted decrease per unit alpha, code `descent_score`
- $\alpha_t$: current alpha
- $\alpha_{\mathrm{trial}}^{(j)}$: trial alpha, code `alpha_trial`
- $\alpha_{\mathrm{used}}$: accepted or final rejected alpha, code `alpha_used`
- $\Delta_{\mathrm{raw}}^{(j)}$: code `predicted_raw`
- $\Delta_{\mathrm{floor}}$: code `predicted_floor`
- $\Delta_{\mathrm{safe}}^{(j)}$: code `predicted_safe`
- $\Delta_{\mathrm{act}}^{(j)}$: code `actual`
- $\rho_{\mathrm{measured}}^{(j)}$: code `rho_measured`
- $\rho_{\mathrm{clipped}}^{(j)}$: code `rho_clipped`
- $\rho_{\mathrm{control}}$: code `rho_control`
- $\rho_\star$: target rho, code `rho_star`
- $k_p$: proportional gain, code `kp`
- $\gamma$: alpha update factor, code `factor`
- $q_{\mathrm{nd}}$: code `non_descent_shrink`
- $f_{\mathrm{abs}}$: code `absolute_predicted_floor`
- $\epsilon_\rho$: code `ratio_eps`
- $f_{\mathrm{rel}}$: code `relative_predicted_floor`
- $c_{\min}, c_{\max}$: code `rho_clip_min`, `rho_clip_max`
- $\gamma_{\min}, \gamma_{\max}$: code `min_alpha_factor`, `max_alpha_factor`

For clipping, write

$$
\mathrm{clip}(x,l,u)=\min(\max(x,l),u).
$$

## 1. Start Minibatch, Compute Loss And Gradient

This corresponds to:

```text
start minibatch
compute loss_before on this minibatch
compute gradient on this minibatch
```

For the current minibatch $B_t$,

$$
\ell_b = L_t(\theta_t), \qquad g_t = \nabla_\theta L_t(\theta_t).
$$

The same minibatch $B_t$ will be reused for every trial loss evaluation in this
optimizer step.

## 2. Update Adam Moments And Build Adam Direction

This corresponds to:

```text
update Adam moments using this gradient
build Adam direction from the updated moments
```

Adam updates

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.
$$

Bias correction gives

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

Adam's proposed direction is

$$
d_{\mathrm{adam}} = -\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

The controller does not change Adam's moment formulas. It only controls the
scalar alpha.

## 3. Check Whether The Direction Is Downhill

This corresponds to:

```text
check whether Adam direction is downhill for this minibatch

if Adam direction is not downhill:
    replace it with a scaled negative-gradient direction
```

Start with

$$
d_t = d_{\mathrm{adam}}.
$$

Compute

$$
s_t = -g_t^\top d_t.
$$

Here $s_t$ is `descent_score` in the code. If $s_t>0$, then $d_t$ is downhill
for this minibatch.

If $s_t \le 0$, replace Adam's direction with the scaled negative-gradient
direction:

$$
d_t = -g_t \frac{\|d_{\mathrm{adam}}\|}{\|g_t\|+\varepsilon}.
$$

When $\|g_t\|>0$ and $\|d_{\mathrm{adam}}\|>0$, this replacement is downhill for
the current minibatch.

## 4. Degenerate No-Direction Case

This corresponds to:

```text
if no usable downhill direction exists:
    shrink alpha
    do not change theta
    move to next minibatch
```

This happens only if neither Adam's direction nor the scaled negative-gradient
fallback gives a usable downhill direction.

Then the optimizer leaves $\theta_t$ unchanged and shrinks alpha:

$$
\alpha_{t+1} = \mathrm{clip}(\alpha_t q_{\mathrm{nd}}, \alpha_{\min}, \alpha_{\max}).
$$

Then the optimizer moves to the next minibatch.

## 5. Save Base Parameters And Try Trial Steps

This corresponds to:

```text
save current theta as theta_base

try trial steps using alpha, then smaller backtracked alphas

for each trial alpha:
    temporarily set theta = theta_base + trial_alpha * direction
    compute loss_after on the same minibatch
```

Save

$$
\theta_b = \theta_t.
$$

Backtracking tries

$$
\alpha_{\mathrm{trial}}^{(j)} = \alpha_t \eta_{\mathrm{bt}}^j, \qquad j=0,1,\dots,J.
$$

For trial $j$, temporarily set

$$
\theta^{(j)} = \theta_b + \alpha_{\mathrm{trial}}^{(j)} d_t.
$$

Then evaluate the same minibatch:

$$
\ell_j = L_t(\theta^{(j)}).
$$

## 6. Compute Predicted Decrease, Actual Decrease, And Rho

This corresponds to:

```text
compute predicted decrease from the gradient
compute actual decrease from loss_before - loss_after
compute rho = actual decrease / safe predicted decrease
```

The predicted decrease per unit alpha is $s_t$. Therefore the raw predicted
decrease for trial $j$ is

$$
\Delta_{\mathrm{raw}}^{(j)} = \alpha_{\mathrm{trial}}^{(j)}s_t.
$$

The prediction floor is

$$
\Delta_{\mathrm{floor}} = \max(f_{\mathrm{abs}}, \epsilon_\rho, f_{\mathrm{rel}}\cdot|\ell_b|).
$$

The safe predicted decrease is

$$
\Delta_{\mathrm{safe}}^{(j)} = \max(\Delta_{\mathrm{raw}}^{(j)}, \Delta_{\mathrm{floor}}).
$$

The actual same-minibatch decrease is

$$
\Delta_{\mathrm{act}}^{(j)} = \ell_b-\ell_j.
$$

Written directly with the loss function:

$$
\Delta_{\mathrm{act}}^{(j)} = L_t(\theta_b)-L_t(\theta^{(j)}).
$$

The measured rho is

$$
\rho_{\mathrm{measured}}^{(j)} =
\frac{L_t(\theta_b)-L_t(\theta^{(j)})}{\Delta_{\mathrm{safe}}^{(j)}}.
$$

Substituting the safe predicted decrease:

$$
\rho_{\mathrm{measured}}^{(j)} =
\frac{L_t(\theta_b)-L_t(\theta^{(j)})}
{\max(\alpha_{\mathrm{trial}}^{(j)}s_t, \Delta_{\mathrm{floor}})}.
$$

The clipped rho is

$$
\rho_{\mathrm{clipped}}^{(j)} =
\mathrm{clip}(\rho_{\mathrm{measured}}^{(j)}, c_{\min}, c_{\max}).
$$

The important split is:

```text
rho_measured is used for accept/reject.
rho_clipped is used for the next-alpha controller.
```

## 7. Accept Or Reject Each Trial

This corresponds to:

```text
if trial step reduced loss enough:
    keep theta at this trial point
    mark accepted = true
    remember alpha_used = trial_alpha
    stop trying smaller alphas

otherwise:
    reject this trial alpha
    try the next smaller alpha
```

The normal acceptance rule is

$$
\rho_{\mathrm{measured}}^{(j)} > \rho_{\min}.
$$

With the usual setting $\rho_{\min}=0$, this is equivalent to

$$
L_t(\theta_b)-L_t(\theta^{(j)}) > 0.
$$

So by default, the trial is accepted if it reduced the same-minibatch loss.

If trial $j$ is accepted:

$$
\theta_{t+1} = \theta^{(j)},
\qquad \alpha_{\mathrm{used}} = \alpha_{\mathrm{trial}}^{(j)}.
$$

Then the backtracking loop stops.

If trial $j$ is rejected, the optimizer tries the next smaller
$\alpha_{\mathrm{trial}}$.

## 8. If All Trial Alphas Are Rejected

This corresponds to:

```text
if all trial alphas were rejected:
    restore theta back to theta_base
    mark accepted = false
    remember alpha_used = the last attempted alpha
```

If no trial is accepted:

$$
\theta_{t+1} = \theta_b.
$$

The final attempted alpha is remembered:

$$
\alpha_{\mathrm{used}} = \alpha_{\mathrm{trial}}^{(J)}.
$$

The minibatch still updated Adam's moment buffers, but it produced no parameter
movement.

## 9. Compute The Next Alpha

This corresponds to:

```text
compute rho_control from the clipped rho of the accepted or final rejected trial

compare rho_control with rho_star

if rho_control is above rho_star:
    propose increasing next alpha

if rho_control is below rho_star:
    propose decreasing next alpha

clip the proposed alpha change so it is not too abrupt

if all trials were rejected:
    prevent alpha from increasing

set alpha = alpha_next
```

Set

$$
\rho_{\mathrm{control}} = \rho_{\mathrm{clipped}}.
$$

Compute the error:

$$
e_t = \rho_{\mathrm{control}} - \rho_\star.
$$

Compute the raw multiplicative factor:

$$
\gamma_{\mathrm{raw}} = \exp(k_p e_t).
$$

Clip the factor:

$$
\gamma = \mathrm{clip}(\gamma_{\mathrm{raw}}, \gamma_{\min}, \gamma_{\max}).
$$

If all trials were rejected:

$$
\gamma \leftarrow \min(\gamma,1).
$$

Then

$$
\alpha_{t+1} = \mathrm{clip}(\alpha_{\mathrm{used}}\gamma, \alpha_{\min}, \alpha_{\max}).
$$

Interpretation:

```text
rho_control > rho_star  -> alpha tends to increase
rho_control = rho_star  -> alpha tends to stay similar
rho_control < rho_star  -> alpha tends to decrease
all trials rejected     -> alpha cannot increase
```

## 10. Record Diagnostics And Move To Next Minibatch

This corresponds to:

```text
record diagnostics
move to next minibatch
```

The important diagnostics are:

```text
loss_before
loss_after
alpha_used
alpha_next
predicted_raw
predicted_safe
actual
rho_measured
rho_clipped
accepted
number of backtracks
whether the gradient fallback was used
```

## Why Rho, Gamma, And Alpha Are Clipped

There are several separate stabilizers around Step 9. They do not serve the same
purpose.

First, the denominator in $\rho_{\mathrm{measured}}$ is floored:

$$
\Delta_{\mathrm{safe}}^{(j)} = \max(\Delta_{\mathrm{raw}}^{(j)}, \Delta_{\mathrm{floor}}).
$$

This prevents a tiny predicted decrease from creating a huge ratio. Without
this floor, a numerically tiny prediction could make the controller believe a
minibatch step was extremely good or extremely bad even when both the predicted
and actual changes were too small to be informative.

Second, rho is clipped before it controls the next alpha:

$$
\rho_{\mathrm{control}} = \mathrm{clip}(\rho_{\mathrm{measured}}, c_{\min}, c_{\max}).
$$

The unclipped $\rho_{\mathrm{measured}}$ is still used to decide whether the
current trial step is accepted. The clipped value is used only for the next
alpha update. This keeps a single noisy minibatch from causing an extreme
controller reaction.

Third, the multiplicative factor is clipped:

$$
\gamma = \mathrm{clip}(\exp(k_p(\rho_{\mathrm{control}}-\rho_\star)), \gamma_{\min}, \gamma_{\max}).
$$

This limits how fast alpha can change from one minibatch to the next. It makes
the controller smoother even when the rho signal is noisy.

Finally, alpha itself is clipped:

$$
\alpha_{t+1} = \mathrm{clip}(\alpha_{\mathrm{used}}\gamma, \alpha_{\min}, \alpha_{\max}).
$$

This enforces the allowed operating range. Factor clipping controls the speed
of change; alpha clipping controls the absolute range.

## Why Alpha Can Decrease Without Backtracking

Backtracking asks:

```text
Should we keep this current trial step?
```

Alpha control asks:

```text
How large should alpha be on the next minibatch?
```

Therefore a step can be accepted while still causing the next alpha to decrease.

For example, suppose $\rho_{\min}=0$, $\rho_\star=0.7$, and $k_p=0.05$.
If a trial has

$$
\rho_{\mathrm{measured}}=0.3,
$$

then it is accepted because

$$
0.3 > \rho_{\min}=0.
$$

But it is below target:

$$
0.3 < \rho_\star=0.7.
$$

So the next alpha factor is

$$
\exp(0.05(0.3-0.7)) = \exp(-0.02) \approx 0.98.
$$

The current step was useful enough to keep, but weaker than the controller
target, so the next alpha is slightly smaller.
