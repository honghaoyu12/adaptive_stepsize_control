# Controlled Adam Fix Plan v5 — Audited Current-Framework Specification

This document corrects the previous fix plan revisions for `TorchControlledAdam` so that it is consistent with the current optimizer framework.

The key convention is:

```text
alpha_t is the effective scalar step size / learning rate.
```

There is no separate external learning-rate variable or multiplier in this version.

The optimizer step always has the wrapper form:

```text
theta_trial = theta_t + alpha_t * p_t
```

where `p_t` is the direction proposed by the inner optimizer or by the fallback rule.

This version keeps the deliberately experimental scope. It does not address full FSDP/DDP correctness, AMP unscale details, stochastic closure determinism, or schedule-aware multiplier design. Those are intentionally deferred.

---

## 0. Selected Fixes in This Pass

The fixes included in this pass are:

1. Use fallback-to-gradient when the Adam direction is non-descent.
2. Stabilize the rho ratio with a predicted-decrease floor and rho clipping.
3. Make trust expansion bounded and patience-based.
4. Fix backtracking to `J = 1`.
5. Add current-framework AdamW handling with `alpha_t` as the only step-size scalar.

The fixes intentionally not included in this pass are:

1. Adam moment rollback/dampening after rejected steps.
2. Full closure determinism for dropout, random augmentation, and BatchNorm.
3. FSDP/DDP/distributed scalar synchronization.
4. AMP/GradScaler-specific implementation details.
5. Schedule-aware separation into an external schedule and a separate multiplier.

---

## 1. Current Framework

At iteration `t`, the training loop samples a minibatch `B_t` and defines the same-minibatch loss:

```text
L_t(theta) = loss(theta; B_t).
```

The training loop computes:

```text
ell_before = L_t(theta_t)
g_t        = grad L_t(theta_t)
```

The optimizer computes a proposed direction `p_t`, evaluates a same-minibatch trial point,

```text
theta_trial = theta_t + alpha_t * p_t,
```

and compares actual decrease with first-order predicted decrease.

The actual decrease is:

```text
actual = L_t(theta_t) - L_t(theta_trial).
```

The predicted decrease is:

```text
predicted = -alpha_t * <g_t, p_t>.
```

The agreement ratio is:

```text
rho_t = actual / (predicted + ratio_eps).
```

This ratio is meaningful only when the before and after losses are evaluated on the same minibatch `B_t`.

---

## 2. Adam Direction

The optimizer maintains standard Adam moments:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

with bias correction:

```text
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
```

The Adam direction is:

```text
d_adam = -m_hat_t / (sqrt(v_hat_t) + eps)
```

The Adam descent score is:

```text
s_adam = -<g_t, d_adam>.
```

If:

```text
s_adam > descent_score_min
```

then the Adam direction is locally descent-aligned with the current minibatch gradient.

---

## 3. Fix 1: Fallback to Gradient for Non-Descent Adam Directions

### Rule

If the Adam direction is descent-aligned:

```text
s_adam > descent_score_min
```

then use Adam's direction:

```text
d_control = d_adam
s_control = s_adam
direction_type = "adam"
```

If the Adam direction is not descent-aligned:

```text
s_adam <= descent_score_min
```

then fall back to the negative gradient direction:

```text
grad_norm_sq = <g_t, g_t>
```

If:

```text
grad_norm_sq <= gradient_norm_floor
```

then skip the update:

```text
theta_next = theta_t
alpha_next = alpha_t
skip rho update
accepted = false
direction_type = "zero_gradient_skip"
```

Otherwise use:

```text
d_control = -g_t
s_control = grad_norm_sq
direction_type = "gradient_fallback"
```

### Why this fix is needed

A non-descent Adam direction is primarily a direction-quality failure, not a scalar step-size failure. Shrinking `alpha` alone does not fix the bad direction. The fallback direction:

```text
d_control = -g_t
```

guarantees:

```text
s_control = -<g_t, d_control> = <g_t, g_t> > 0
```

as long as the gradient is nonzero.

### Diagnostics

Record:

```text
direction_type
used_gradient_fallback
s_adam
s_control
grad_norm_sq
fallback_rate
zero_gradient_skip_rate
```

### Scale caution

The raw gradient direction and the Adam direction may have different norms. This specification keeps the fallback simple and relies on the same trial/backtracking/rho mechanism to reject an unsafe fallback step. If gradient-fallback rejection becomes frequent, add a future optional shrink factor:

```text
alpha_trial = gradient_fallback_alpha_factor * alpha_t
```

with `gradient_fallback_alpha_factor <= 1`. This is not required in the current pass, but it should be monitored.

For consistency, use:

```text
descent_score_min = 0.0
```

or a very small nonnegative tolerance. If `descent_score_min` is set larger than zero, then the zero-gradient skip threshold should be chosen so that a fallback gradient direction with `grad_norm_sq > gradient_norm_floor` is also considered acceptable.

---

## 4. Fix 2: Stabilize the Rho Ratio

For a trial multiplier `a`, define:

```text
theta_trial(a) = theta_t + a * p_control
```

where `p_control` is the final direction used by the wrapper. For plain Adam, `p_control = d_control`. For AdamW, `p_control` is defined in Section 7.

The predicted decrease is:

```text
predicted_raw(a) = -a * <g_t, p_control>
```

Equivalently, if `s_control = -<g_t, p_control>`:

```text
predicted_raw(a) = a * s_control.
```

Define a prediction floor:

```text
predicted_floor = max(
    absolute_predicted_floor,
    relative_predicted_floor * abs(ell_before)
)
```

Then:

```text
predicted_safe = max(predicted_raw, predicted_floor)
```

The actual decrease is:

```text
actual(a) = ell_before - L_t(theta_trial(a))
```

The measured ratio, using the safe denominator, is:

```text
rho_measured = actual(a) / predicted_safe
```

The clipped ratio is:

```text
rho_clipped = clip(rho_measured, rho_clip_min, rho_clip_max)
```

Use:

```text
rho_accept = rho_measured
rho_control_input = rho_clipped
```

### Acceptance uses measured rho

```text
accept if reject_bad_steps == false or rho_accept > rho_min
```

### EMA uses clipped rho

```text
if use_rho_ema:
    if rho_bar is undefined:
        rho_bar = rho_clipped
    else:
        rho_bar = rho_beta * rho_bar + (1 - rho_beta) * rho_clipped
    rho_control = rho_bar
else:
    rho_control = rho_clipped
```

### Recommended defaults

```text
absolute_predicted_floor = 1e-12
relative_predicted_floor = 1e-8
rho_clip_min = -1.0
rho_clip_max = 3.0
```

### Why this fix is needed

The measured rho value is the correct quantity for deciding whether the trial step genuinely passed the local decrease test. However, extreme rho values can corrupt the controller and EMA state. Therefore:

```text
measured rho controls acceptance;
clipped rho controls alpha adaptation.
```

---

## 5. Fix 3: Trust Expansion With Hard Bounds and Patience

The ordinary proportional controller factor is:

```text
error = rho_control - rho_star

gain = kp      if error >= 0
       kp_down otherwise

raw_factor = exp(gain * error)
ordinary_factor = clip(raw_factor, gamma_min, gamma_max)

# If all trial candidates were rejected, do not allow the controller
# to increase alpha based on a noisy or threshold-failed signal.
if not accepted:
    ordinary_factor = min(ordinary_factor, 1.0)
```

Update consecutive trust evidence:

```text
if accepted and accepted_j == 0 and rho_control >= trust_region_rho_threshold:
    trust_good_count += 1
else:
    trust_good_count = 0
```

Trust expansion conditions:

```text
trust_region_expand_enabled == true
accepted == true
accepted_j == 0
rho_control >= trust_region_rho_threshold
alpha_used <= trust_region_alpha_threshold
trust_good_count >= trust_region_patience
```

If these conditions hold:

```text
factor = max(ordinary_factor, trust_region_expand_factor)
factor = min(factor, trust_region_max_factor)
trust_expanded = true
```

Otherwise:

```text
factor = ordinary_factor
trust_expanded = false
```

Finally apply a hard global factor bound. If trust expansion is disabled, the ordinary upper bound remains `gamma_max`. If trust expansion is enabled, the hard upper bound may extend to `trust_region_max_factor`:

```text
if trust_region_expand_enabled:
    gamma_hard_max = max(gamma_max, trust_region_max_factor)
else:
    gamma_hard_max = gamma_max

factor = clip(factor, gamma_min, gamma_hard_max)
```

Recommended defaults:

```text
gamma_max = 1.05
trust_region_expand_factor = 1.5
trust_region_max_factor = 1.5
trust_region_patience = 2
```

### Why this fix is needed

Trust expansion is meant to recover from alpha collapse. It should not become an uncontrolled jump mechanism. Requiring consecutive good evidence and applying a hard trust bound makes it a recovery rule rather than a one-batch overreaction.

---

## 6. Fix 4: Limit Backtracking With J = 1

Set:

```text
J = 1
```

This is a hard design choice for this experimental implementation.

The only candidates are:

```text
a_0 = alpha_t
a_1 = alpha_t * eta_bt
```

Recommended:

```text
eta_bt = 0.5
```

### Candidate loop

For:

```text
j in {0, 1}
```

compute:

```text
alpha_trial = alpha_t * eta_bt^j
theta_trial = theta_t + alpha_trial * p_control
ell_after = closure(theta_trial)
```

Then compute:

```text
predicted_raw
predicted_safe
actual
rho_measured
rho_clipped
```

Accept if:

```text
reject_bad_steps == false or rho_measured > rho_min
```

If accepted:

```text
accepted = true
alpha_used = alpha_trial
theta_next = theta_trial
accepted_j = j
break
```

If neither candidate is accepted:

```text
accepted = false
alpha_used = alpha_t * eta_bt
theta_next = theta_t
accepted_j = None
```

The controller may still update alpha downward using the last attempted rho, but no parameter update is applied. In this case, the controller should not increase alpha on the next update; enforce:

```text
if not accepted:
    factor = min(factor, 1.0)
```

This prevents a failed trial sequence from paradoxically increasing the next step size.

### Why this fix is needed

`J = 1` gives one emergency shrink without turning the optimizer into an expensive stochastic line-search method. The controller should adapt future alpha; backtracking should only be a guardrail.

---

## 7. Fix 5: Current-Framework AdamW Handling

This section replaces the previous, inconsistent AdamW notation.

In this framework, the scalar `alpha_t` is the effective learning rate. Therefore AdamW-style decoupled decay must be expressed using `alpha_t` if it is included in the actual parameter update.

### 7.1 Plain Adam direction

The Adam loss direction is:

```text
d_control = d_adam
```

or, if fallback is active:

```text
d_control = -g_t
```

### 7.2 AdamW full controlled direction

For AdamW mode, define the full proposed direction as:

```text
p_control = d_control - weight_decay * theta_t
```

Then the trial step is:

```text
theta_trial = theta_t + alpha_trial * p_control
```

Equivalently:

```text
theta_trial = theta_t + alpha_trial * d_control - alpha_trial * weight_decay * theta_t
```

The accepted update is:

```text
theta_{t+1} = theta_t + alpha_used * p_control
```

or equivalently:

```text
theta_{t+1} = (1 - alpha_used * weight_decay) * theta_t + alpha_used * d_control.
```

This keeps the wrapper abstraction intact:

```text
inner optimizer proposes p_control
alpha_t scales p_control
```

### 7.3 Rho for AdamW

For AdamW mode, the predicted decrease must use the full direction `p_control`:

```text
s_control = -<g_t, p_control>
predicted_raw = alpha_trial * s_control
```

The actual decrease is measured using the full AdamW trial point:

```text
actual = L_t(theta_t) - L_t(theta_t + alpha_trial * p_control)
```

Then:

```text
rho_measured = actual / predicted_safe
```

### 7.4 Why this is the correct current-framework fix

The previous document incorrectly mixed two frameworks:

```text
alpha_t as learning rate
```

and a different future design where `alpha_t` would be only a multiplier over an external schedule.

In the current framework, the correct AdamW-style update is:

```text
theta_next = theta_t + alpha_t * (d_control - weight_decay * theta_t).
```

So there is no separate external learning-rate variable anywhere in the current algorithm.

### 7.5 Rejection behavior

If the trial step is rejected, then:

```text
theta_next = theta_t
```

No partial AdamW decay is applied on rejected steps.

Reason:

A rejected step should mean no parameter update. Applying decay alone would make `accepted = false` misleading and would break the clean wrapper semantics.

### 7.6 Diagnostics

Record:

```text
adamw_mode
weight_decay
weight_decay_direction_norm
p_control_norm
d_control_norm
decay_to_control_direction_ratio
```

---

## 8. Revised Full Algorithm

```text
Inputs:
    theta_t
    ell_before = L_t(theta_t)
    g_t = grad L_t(theta_t)
    same-minibatch closure for no-backward loss evaluation

State:
    Adam moments m, v
    alpha_t
    rho_bar
    trust_good_count

Fixed:
    J = 1
```

### Step 1: Update Adam moments

```text
m = beta1 * m + (1 - beta1) * g_t
v = beta2 * v + (1 - beta2) * g_t^2

m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
```

### Step 2: Compute Adam direction

```text
d_adam = -m_hat / (sqrt(v_hat) + eps)
s_adam = -dot(g_t, d_adam)
```

### Step 3: Choose loss direction

```text
if s_adam > descent_score_min:
    d_control = d_adam
    direction_type = "adam"
else:
    grad_norm_sq = dot(g_t, g_t)

    if grad_norm_sq <= gradient_norm_floor:
        theta_next = theta_t
        alpha_next = alpha_t
        skip rho update
        return diagnostics

    d_control = -g_t
    direction_type = "gradient_fallback"
```

### Step 4: Build full wrapper direction

If plain Adam mode:

```text
p_control = d_control
```

If AdamW mode:

```text
p_control = d_control - weight_decay * theta_t
```

Compute:

```text
s_control = -dot(g_t, p_control)
```

If AdamW decay makes the full direction non-descent:

```text
s_control <= descent_score_min
```

then fall back to the pure loss direction:

```text
p_control = d_control
s_control = -dot(g_t, p_control)
adamw_decay_suppressed_for_step = true
```

If this is still non-descent, fall back to gradient as in Step 3.

This extra check is necessary because even if `d_control` is descent-aligned, adding the decoupled decay component may make the full proposed direction less descent-aligned for the minibatch data loss.

### Step 5: Trial loop with J = 1

```text
accepted = false
last_rho_measured = None
last_rho_clipped = None
last_alpha_trial = None

for j in {0, 1}:
    alpha_trial = alpha_t * eta_bt^j
    theta_trial = theta_t + alpha_trial * p_control
    ell_after = closure(theta_trial)

    predicted_raw = alpha_trial * s_control
    predicted_floor = max(absolute_predicted_floor,
                          relative_predicted_floor * abs(ell_before))
    predicted_safe = max(predicted_raw, predicted_floor)

    actual = ell_before - ell_after
    rho_measured = actual / predicted_safe
    rho_clipped = clip(rho_measured, rho_clip_min, rho_clip_max)

    last_rho_measured = rho_measured
    last_rho_clipped = rho_clipped
    last_alpha_trial = alpha_trial

    if reject_bad_steps == false or rho_measured > rho_min:
        accepted = true
        alpha_used = alpha_trial
        theta_next = theta_trial
        accepted_j = j
        break

if not accepted:
    alpha_used = last_alpha_trial
    theta_next = theta_t
    accepted_j = None
```

### Step 6: Update rho signal

Use clipped rho for controller state:

```text
rho_for_control = last_rho_clipped

if use_rho_ema:
    if rho_bar is undefined:
        rho_bar = rho_for_control
    else:
        rho_bar = rho_beta * rho_bar + (1 - rho_beta) * rho_for_control
    rho_control = rho_bar
else:
    rho_control = rho_for_control
```

### Step 7: Controller factor

```text
error = rho_control - rho_star

gain = kp if error >= 0 else kp_down

raw_factor = exp(gain * error)
ordinary_factor = clip(raw_factor, gamma_min, gamma_max)

# If all trial candidates were rejected, do not allow the controller
# to increase alpha based on a noisy or threshold-failed signal.
if not accepted:
    ordinary_factor = min(ordinary_factor, 1.0)
```

### Step 8: Trust expansion

```text
if accepted and accepted_j == 0 and rho_control >= trust_region_rho_threshold:
    trust_good_count += 1
else:
    trust_good_count = 0

factor = ordinary_factor
trust_expanded = false

if (trust_region_expand_enabled
    and accepted
    and accepted_j == 0
    and rho_control >= trust_region_rho_threshold
    and alpha_used <= trust_region_alpha_threshold
    and trust_good_count >= trust_region_patience):

    factor = max(factor, trust_region_expand_factor)
    factor = min(factor, trust_region_max_factor)
    trust_expanded = true

if trust_region_expand_enabled:
    gamma_hard_max = max(gamma_max, trust_region_max_factor)
else:
    gamma_hard_max = gamma_max

factor = clip(factor, gamma_min, gamma_hard_max)
```

### Step 9: Update alpha

```text
alpha_next = clip(alpha_used * factor, alpha_min, alpha_max)
alpha_t = alpha_next
```

Adam moments remain updated in this experimental version, even if the parameter step is rejected.

---

## 9. Remaining Known Limitations

The following issues remain intentionally unresolved in this fix pass.

### 9.1 Adam moments still update on rejected steps

Current behavior remains:

```text
parameters may be restored, but Adam moments remain updated.
```

This is acceptable for now because the project is experimental, but it should be monitored.

Add diagnostics:

```text
rejection_rate
consecutive_rejections
moment_update_on_reject_count
```

If rejection becomes frequent, a future version should add:

```text
state_on_reject = keep | rollback | dampen
```

### 9.2 Closure determinism is ignored for now

Dropout, random augmentation, BatchNorm, and RNG replay are not handled in this pass.

This is acceptable for controlled small experiments, but the same-minibatch rho signal is cleanest only if before/after losses evaluate the same stochastic objective.

### 9.3 AMP and distributed training are ignored for now

No FSDP/DDP/AMP guarantees are included in this pass.

That is acceptable because the intended scope is experimental single-process or simple training, not large distributed training.

### 9.4 Alpha is the effective scalar step size

This document intentionally uses:

```text
theta_trial = theta + alpha * p_control
```

There is no separate base learning rate.

A future schedule-aware version may introduce a separate schedule and a dimensionless multiplier, but that is outside the current implementation and is not part of this specification.

---

## 10. Updated Testing Plan

### Test 1: Adam direction is descent

Construct a case where:

```text
-dot(g, d_adam) > descent_score_min
```

Verify:

```text
direction_type == "adam"
d_control == d_adam
```

### Test 2: gradient fallback triggers

Construct a case where:

```text
-dot(g, d_adam) <= descent_score_min
```

Verify:

```text
direction_type == "gradient_fallback"
d_control == -g
```

### Test 3: zero-gradient skip

Construct a case where:

```text
dot(g, g) <= gradient_norm_floor
```

Verify:

```text
no parameter update
alpha unchanged
rho update skipped
```

### Test 4: AdamW full direction uses alpha only

In AdamW mode, verify:

```text
p_control = d_control - weight_decay * theta
```

and:

```text
theta_trial = theta + alpha * p_control
```

There should be no separate external learning-rate variable in this implementation.

### Test 5: AdamW rho uses full p_control

Verify that predicted decrease uses:

```text
-alpha * dot(g, p_control)
```

not:

```text
-alpha * dot(g, d_control)
```

### Test 6: AdamW rejected step does not decay parameters

If all candidates are rejected, verify:

```text
theta_next == theta_t
```

not:

```text
theta_next == theta_t - alpha * weight_decay * theta_t
```

### Test 7: acceptance uses raw rho

Construct a case where:

```text
rho_measured != rho_clipped
```

Verify:

```text
acceptance decision uses rho_measured
EMA/controller uses rho_clipped
```

### Test 8: predicted-decrease floor

Construct a case where:

```text
predicted_raw < predicted_floor
```

Verify:

```text
predicted_safe == predicted_floor
predicted_was_floored == true
```

### Test 9: rho clipping before EMA

Construct a case where:

```text
rho_measured > rho_clip_max
```

Verify:

```text
rho_clipped == rho_clip_max
rho_bar uses rho_clipped
```

### Test 10: trust patience resets

Verify:

```text
trust_good_count resets to 0 when rho_control falls below threshold
```

and trust expansion only fires after:

```text
trust_good_count >= trust_region_patience
```

### Test 11: trust expansion hard bound

Verify:

```text
factor <= trust_region_max_factor
```

when trust expansion fires.

### Test 12: J fixed to 1

Pass a larger configured value, such as:

```text
J = 5
```

Verify:

```text
optimizer.J == 1
```

and only two candidates are evaluated.

### Test 13: no alpha increase after full rejection

Construct a case where both `j = 0` and `j = 1` candidates are rejected. Verify:

```text
accepted == false
factor <= 1.0
alpha_next <= alpha_t * eta_bt
```

This prevents a failed trial sequence from increasing the next step size.

---

## 11. Consistency Checklist

This document should satisfy:

```text
No separate external learning-rate scalar is used in the current algorithm.
alpha is always the effective scalar step size.
Adam mode: p_control = d_control.
AdamW mode: p_control = d_control - weight_decay * theta.
Trial point always equals theta + alpha * p_control.
Predicted decrease always uses -alpha * <g, p_control>.
Acceptance uses measured rho.
EMA/controller uses clipped rho.
Backtracking uses J = 1.
Rejected steps do not change parameters.
If all trial candidates are rejected, alpha is not increased.
Adam moments remain updated on rejection, intentionally.
```

---

## 12. Evaluation After These Fixes

After these fixes, the optimizer is more internally consistent than the previous version.

### What improves

1. Non-descent Adam directions no longer waste the minibatch; fallback-to-gradient gives a safe descent direction.
2. The rho ratio cannot explode as easily from tiny predicted decreases because the denominator is floored.
3. EMA cannot be polluted by extreme rho outliers because clipping happens before EMA.
4. Trust expansion can recover from alpha collapse but cannot bypass hard bounds.
5. Backtracking overhead is bounded by `J = 1`.
6. AdamW now follows the current framework: `alpha` scales the full proposed direction, including decoupled weight decay.
7. There is no inconsistent mixture of `alpha` and a separate external learning-rate variable.

### What remains risky

1. Rejected steps still update Adam moments.
2. The closure may not be deterministic if dropout, random augmentation, or BatchNorm are active.
3. The method still requires same-minibatch trial forward evaluations.
4. AMP and distributed training are not addressed.
5. The method is not yet schedule-aware.

For the current experimental phase, these omissions are acceptable as long as benchmark claims remain appropriately scoped.

### Overall verdict

With these fixes, the optimizer becomes a stronger experimental controlled-Adam / controlled-AdamW implementation.

It should be described as:

```text
an experimental optimizer wrapper for controlled adaptive step-size learning
```

not yet as:

```text
a production-ready AdamW replacement.
```

The most important next empirical test is whether these fixes reduce:

1. alpha collapse on Fashion-MNIST-style runs;
2. alpha overgrowth or cap saturation on CIFAR-style runs;
3. non-descent Adam-direction failures.
