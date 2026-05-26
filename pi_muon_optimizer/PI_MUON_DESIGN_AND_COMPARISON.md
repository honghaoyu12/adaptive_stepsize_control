# PI-Controlled Muon: Design and Comparison Notes

This document describes the PI-controlled Muon optimizer in this folder and
explains how it differs from the controlled Muon implementation in
`controlled_muon_project`.

The short version is:

- Muon chooses directions for 2D hidden matrix parameters.
- AdamW-style logic handles non-matrix or excluded parameters.
- A PI controller chooses one global step-size multiplier.
- The control signal is the same-batch actual-vs-predicted decrease ratio.
- Compared with `controlled_muon_project`, this implementation is more like a
  standalone PyTorch optimizer for neural networks and less like a toy
  line-search benchmark.

## Files

```text
pi_muon_optimizer/
├── pi_muon.py
├── demo_toy_mlp.py
├── README.md
├── requirements.txt
└── PI_MUON_DESIGN_AND_COMPARISON.md
```

The main implementation is `pi_muon.py`. The demo trains a small MLP on a
synthetic regression problem and prints controller diagnostics.

## Core Idea

At iteration `t`, the optimizer builds a proposed direction `p_t` for all
trainable parameters. The proposed direction is a mixture of:

- Muon-style directions for 2D hidden matrix parameters, matching
  `torch.optim.Muon`'s parameter scope.
- AdamW-style directions for other parameters.

The PI controller then applies one global scalar multiplier:

```text
theta_trial = theta_t + alpha_t * p_t
```

The predicted decrease is the first-order decrease under the current gradient:

```text
predicted_decrease = -alpha_t * dot(g_t, p_t)
```

The actual decrease is measured by evaluating the same minibatch after the
trial step:

```text
actual_decrease = f_B(theta_t) - f_B(theta_trial)
```

The controller signal is:

```text
rho_t = actual_decrease / (predicted_decrease + predicted_eps)
```

This ratio asks whether Muon's proposed geometry, at the current global scale,
actually delivered the loss reduction predicted by the local linear model.

## Same-Batch Closure Contract

The optimizer must evaluate before and after losses on the same minibatch.
Otherwise, `rho_t` is contaminated by minibatch difficulty variation.

The required closure shape is:

```python
def closure(backward: bool = True):
    optimizer.zero_grad(set_to_none=True)
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    if backward:
        loss.backward()
    return loss
```

Inside `PIMuon.step(closure)`:

1. `closure(backward=True)` computes the original loss and gradients.
2. Muon and AdamW fallback directions are built.
3. If the full direction is not descent-like, an SGD fallback may be used.
4. The trial step is applied in-place.
5. `closure(backward=False)` evaluates the same-batch trial loss.
6. `rho_t` is computed.
7. The PI controller updates `alpha`.
8. If rejection is enabled and the step is rejected, the parameter delta is
   reversed.

The additional cost is one extra forward pass per optimizer step.

## Parameter Grouping

Muon is usually intended for hidden matrix weights, not every parameter in a
network. This implementation includes a helper:

```python
param_groups = default_muon_param_groups(model.named_parameters())
optimizer = PIMuon(param_groups)
```

By default, parameters are assigned as follows:

- Use Muon for trainable 2D parameters whose names do not match excluded
  keywords.
- Use AdamW-style fallback for scalar/vector parameters, convolution kernels,
  biases, norms, embeddings, heads, and output layers.

The default exclusion keywords are:

```text
embed, embedding, lm_head, head, norm, bias
```

This is conservative. It reflects a common practice: apply Muon to hidden
matrix weights where orthogonalized updates make sense, and use AdamW-style
updates for parameters where Muon geometry is less appropriate.

Custom groups are also supported:

```python
optimizer = PIMuon([
    {"params": hidden_matrix_params, "use_muon": True},
    {"params": other_params, "use_muon": False},
])
```

## Muon Direction

For a 2D matrix parameter, the optimizer first updates an EMA-style momentum
buffer in the same convention as `torch.optim.Muon`:

```text
buffer <- lerp(buffer, grad, 1 - muon_momentum)
```

If Nesterov mode is enabled, the raw update sent to orthogonalization is:

```text
update_raw = lerp(grad, buffer, muon_momentum)
```

Otherwise:

```text
update_raw = buffer
```

Only 2D tensors are sent through Muon by the default grouping. Other tensors use
the AdamW fallback.

The matrix is passed through a Newton-Schulz-style quintic iteration that
approximately pushes singular values toward 1 while preserving the singular
directions. By default it uses PyTorch Muon's coefficients
`(3.4445, -4.7750, 2.0315)` and 5 Newton-Schulz steps. The implementation
normalizes the matrix before iteration and transposes tall matrices during the
iteration for the usual Muon-style shape.

After orthogonalization, optional shape scaling is applied:

```text
ortho <- ortho * sqrt(max(1, rows / cols))
```

The final parameter direction is:

```text
p_t = -unflatten(ortho)
```

## AdamW Fallback Direction

For parameters not using Muon, the optimizer builds an AdamW-style direction:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

After bias correction:

```text
adam_update = m_hat_t / (sqrt(v_hat_t) + adam_eps)
p_t = -adam_update
```

If group-level `weight_decay` is nonzero, the parameter update follows PyTorch
AdamW-style decoupled decay:

```text
theta_trial = (1 - alpha_t * weight_decay) * theta_t + alpha_t * p_t
```

For clean controller experiments, `weight_decay=0.0` is easiest to interpret
because the measured `rho_t` then corresponds directly to the closure loss.

## PI Controller

The controller operates in log-space:

```text
log_alpha <- log_alpha + log_multiplier
alpha = exp(log_alpha)
```

First, the raw ratio can be clipped by `rho_clip`:

```text
rho_t <- clip(rho_t, -rho_clip, rho_clip)
```

Then an EMA measurement is maintained:

```text
rho_bar_t = beta_rho * rho_bar_{t-1}
            + (1 - beta_rho) * rho_t
```

The error is:

```text
error_t = rho_bar_t - rho_star
```

The leaky integral state is:

```text
I_t = integral_decay * I_{t-1} + error_t
```

with optional clipping to `[-integral_clip, integral_clip]`.

The PI update is:

```text
log_multiplier = kp * error_t + ki * I_t
```

`log_multiplier` is clipped to the range implied by
`[multiplier_min, multiplier_max]`, then `log_alpha` is clipped to
`[log(alpha_min), log(alpha_max)]`.

## Non-Descent Directions

The predicted decrease is meaningful only if:

```text
-dot(g_t, p_t) > 0
```

Muon's orthogonalized direction can occasionally be non-descent-like under the
current minibatch gradient. If `fallback_to_sgd_if_not_descent=True`, the
optimizer replaces the whole proposed direction with an SGD direction:

```text
p_t = -g_t
```

This fallback is applied only for the current step. The diagnostic field
`used_fallback_direction` records whether it happened.

## Rejection, Backtracking, and Trust-Region Expansion

By default:

```python
reject_bad_steps=False
```

This is deliberate for stochastic neural-network training. A single minibatch
ratio can be noisy, and hard rejection can add instability.

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
the old PyTorch controlled Muon helper. If the first trial succeeds, the
smoothed controller signal is at least `trust_region_rho_threshold`, and
`alpha_used` is no larger than `trust_region_alpha_threshold`, then the
controller enforces at least `trust_region_expand_factor` growth for the next
alpha, subject to `alpha_max`.

Important implementation detail: internal optimizer states are not rolled back.
Momentum buffers and Adam moments have already been updated. This matches the
research-wrapper nature of the code and is safest with the default
`reject_bad_steps=False`.

## Diagnostics

After each step, `optimizer.last_stats` contains a `PIControlStats` object with:

- `loss_before`: same-batch loss before the trial step.
- `loss_after`: same-batch loss after the trial step.
- `actual_decrease`: `loss_before - loss_after`.
- `predicted_decrease`: first-order predicted decrease.
- `rho`: raw actual-over-predicted ratio.
- `rho_bar`: smoothed ratio used by the controller.
- `error`: `rho_bar - rho_star`.
- `integral`: current leaky integral state.
- `alpha`: updated global multiplier after PI control.
- `log_alpha`: current log multiplier.
- `log_multiplier`: latest clipped log update.
- `accepted`: whether the trial parameter step remains applied.
- `used_fallback_direction`: whether SGD fallback was used.
- `grad_dot_direction`: raw `dot(g_t, p_t)` before multiplying by `-alpha`.
- `backtracks`: number of backtracking reductions used.
- `alpha_next`: next global multiplier after the controller update.
- `alpha_update_factor`: `alpha_next / alpha_used`.
- `trust_region_expanded`: whether the expansion rule fired.
- `skipped_reason`: reason no normal trial step occurred, if any.

These diagnostics are the main way to understand whether the controller is
shrinking, growing, saturating, or reacting to noisy ratio estimates.

## Comparison With `controlled_muon_project`

There are two controlled Muon implementations in `controlled_muon_project`:

- A deterministic NumPy optimizer in
  `controlled_muon_project/src/controlled_muon/optimizers.py`.
- A PyTorch helper class in
  `controlled_muon_project/src/controlled_muon/torch_optimizers.py`.

The PI Muon implementation differs in several important ways.

### Controller Law

`controlled_muon_project` uses a proportional update:

```text
alpha_next = alpha_used * exp(kp * (rho_control - rho_star))
```

The PyTorch helper can smooth `rho`, but there is no accumulated integral
state.

`PIMuon` uses:

```text
log_alpha_next = log_alpha
                 + kp * (rho_bar - rho_star)
                 + ki * integral
```

The integral term lets persistent controller error accumulate. If the measured
ratio sits below the target for many steps, `alpha` keeps being pushed down. If
the measured ratio stays above target, `alpha` keeps being pushed up until the
feedback loop balances or hits clipping bounds.

### API Shape

`TorchControlledMuon` is a custom helper. The training loop computes gradients,
then calls:

```python
step = optimizer.step(loss_before, reevaluate_loss)
```

`PIMuon` subclasses `torch.optim.Optimizer` and owns the closure workflow:

```python
loss_after = optimizer.step(closure)
stats = optimizer.last_stats
```

This makes `PIMuon` easier to use as a standalone optimizer, while still
requiring a closure for same-batch reevaluation.

### Parameter Treatment

The current neural-network paths in both `TorchControlledMuon` and `PIMuon`
follow the official `torch.optim.Muon` scope:

- Muon only for 2D hidden matrix parameters by default.
- AdamW-style fallback for vectors, scalars, convolution kernels, biases,
  norms, embeddings, heads, and user-designated non-Muon groups.

The deterministic 2D function benchmark in `controlled_muon_project` remains a
Muon-style vector/matrix diagnostic rather than a neural-network optimizer
replacement.

### Orthogonalization Implementation

`controlled_muon_project` uses the shared NumPy orthogonalization utility in
`controlled_muon.orthogonalization`.

`PIMuon` implements Newton-Schulz orthogonalization directly in PyTorch. This
keeps tensors on their device and makes the optimizer less dependent on a NumPy
round-trip.

### Step Acceptance and Backtracking

`TorchControlledMuon` supports backtracking. If a trial step is rejected, it
can retry smaller `alpha` values before giving up.

`PIMuon` now mirrors this behavior: when `reject_bad_steps=True`, it tries the
current alpha and then backtracks by `backtrack_shrink` up to
`max_backtracks`. The difference is the next alpha is chosen by a PI controller
instead of the old proportional-only controller.

### Trust-Region Expansion

`TorchControlledMuon` can force a trust-region expansion when the measured
ratio is high and `alpha` is small.

`PIMuon` has the same optional expansion branch. It is controlled by
`trust_region_expand`, `trust_region_rho_threshold`,
`trust_region_alpha_threshold`, and `trust_region_expand_factor`.

### Non-Descent Handling

`TorchControlledMuon` shrinks alpha and stays put if the Muon direction is not
descent-like.

`PIMuon` can replace the direction with SGD for that step. This gives the
optimizer a way to continue making progress when orthogonalization and
momentum produce a direction that is temporarily misaligned with the current
gradient.

### Scope

`controlled_muon_project` is broader. It contains deterministic matrix
objectives, NumPy optimizers, PyTorch benchmark helpers, demos, and tests.

`pi_muon_optimizer` is narrower and more focused: it is a standalone
PI-controlled Muon/AdamW optimizer wrapper plus a compact toy MLP demo.

## Hyperparameter Guidance

The most important controller parameters are:

- `alpha0`: initial global multiplier.
- `rho_star`: target actual-over-predicted ratio.
- `kp`: proportional gain.
- `ki`: integral gain.
- `beta_rho`: EMA smoothing for noisy `rho_t`.
- `use_rho_ema`: whether the controller uses smoothed or raw rho.
- `integral_decay`: leak factor for the integral state.
- `integral_clip`: anti-windup bound.
- `multiplier_min`, `multiplier_max`: per-step alpha change limits.
- `alpha_min`, `alpha_max`: hard alpha bounds.
- `rho_clip`: raw ratio clipping before control.
- `reject_bad_steps`, `rho_min`: optional step rejection.
- `max_backtracks`, `backtrack_shrink`: optional line-search retries.
- `trust_region_expand`: optional small-alpha expansion rule.

Important direction parameters are:

- `muon_momentum`: Muon momentum coefficient.
- `muon_nesterov`: whether to use Nesterov-style lookahead.
- `ns_steps`: number of Newton-Schulz iterations.
- `muon_shape_scale`: whether to apply rectangular shape scaling.
- `adam_betas`, `adam_eps`: AdamW fallback behavior.
- `weight_decay`: PyTorch-style decoupled parameter decay.

A reasonable first run is:

```python
optimizer = PIMuon(
    param_groups,
    alpha0=3e-3,
    rho_star=0.8,
    kp=0.05,
    ki=0.001,
    beta_rho=0.9,
    integral_decay=0.95,
    multiplier_min=0.85,
    multiplier_max=1.15,
    alpha_min=1e-5,
    alpha_max=5e-2,
)
```

For noisier minibatch training:

```python
kp = 0.01
ki = 0.0
beta_rho = 0.99
multiplier_min = 0.95
multiplier_max = 1.05
```

After the proportional-only controller behaves sensibly, add a small integral
gain.

## Practical Interpretation

Muon changes the geometry of matrix updates by orthogonalizing them. That
geometry can be useful, but it also makes raw update magnitude less directly
connected to gradient magnitude. `PIMuon` separates the two jobs:

```text
Muon/AdamW choose the direction.
The PI controller chooses the global scale.
```

The controller asks after every step:

```text
Given this proposed direction and scale, did the same-batch loss decrease as
much as the first-order model predicted?
```

The result is an optimizer that keeps Muon's directional idea while using
feedback to adapt the global multiplier over training.
