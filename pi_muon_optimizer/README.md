# PI-Controlled Muon Optimizer Wrapper

This project implements an experimental PyTorch optimizer that combines:

1. **Muon-style matrix updates** for hidden matrix parameters;
2. **AdamW-style fallback updates** for non-matrix parameters;
3. an outer-loop **PI controller** that adapts a global step multiplier using the actual-vs-predicted decrease ratio.

The implementation is intended for research and experimentation, not production-scale training.

---

## Background

Muon, short for **MomentUm Orthogonalized by Newton-Schulz**, is designed for 2D neural-network hidden-layer parameters. Its core idea is to take SGD-momentum-like updates and post-process matrix-valued updates with Newton-Schulz approximate orthogonalization. Scalar/vector parameters, and often embeddings/output heads, are commonly optimized with AdamW instead.

This implementation follows the current `torch.optim.Muon` direction
conventions for neural-network paths: 2D hidden matrix parameters use Muon,
other parameters use AdamW-style fallback updates, the momentum update uses
PyTorch's `lerp` convention, the Newton-Schulz defaults are the quintic
coefficients `(3.4445, -4.7750, 2.0315)` with 5 steps, and the original
rectangular shape scaling is enabled by default.

This project wraps that Muon-style direction with our controller:

\[
\theta_{t+1}^{\mathrm{trial}} = \theta_t + \alpha_t p_t,
\]

where \(p_t\) is proposed by Muon/AdamW and \(\alpha_t\) is a global multiplier adapted by PI control.

---

## Step-quality signal

Let

\[
g_t = \nabla f(\theta_t).
\]

For a proposed direction \(p_t\), the first-order predicted decrease is

\[
\Delta \hat f_t = -\alpha_t g_t^\top p_t.
\]

The actual same-minibatch decrease is

\[
\Delta f_t = f_{B_t}(\theta_t) - f_{B_t}(\theta_t + \alpha_t p_t).
\]

The controller uses

\[
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t + \epsilon}.
\]

Interpretation:

- \(\rho_t \approx 1\): actual decrease matched the first-order prediction;
- \(0 < \rho_t < 1\): step helped, but less than predicted;
- \(\rho_t < 0\): same-minibatch loss increased;
- \(\rho_t > 1\): step was better than predicted.

---

## PI controller

We smooth the measurement:

\[
\bar\rho_t = \beta_\rho \bar\rho_{t-1} + (1-\beta_\rho)\rho_t.
\]

Define the controller error:

\[
e_t = \bar\rho_t - \rho^\star.
\]

The integral state is leaky:

\[
I_t = \lambda_I I_{t-1} + e_t.
\]

The global multiplier is updated in log-space:

\[
\log \alpha_{t+1}
=
\log \alpha_t
+
K_P e_t
+
K_I I_t.
\]

Equivalently:

\[
\alpha_{t+1}
=
\alpha_t\exp(K_P e_t + K_I I_t).
\]

The implementation clips both the multiplicative change and the absolute \(\alpha_t\) range.

---

## Why PI around Muon?

Muon orthogonalizes matrix updates, which changes update geometry and discards much of the raw magnitude information. That makes a global step multiplier important. Our outer PI controller asks:

> Given Muon's proposed direction, did the loss decrease as much as the local linear model predicted?

So the split is:

\[
\text{Muon chooses the geometry/direction;}
\]

\[
\text{the PI controller chooses the global step length.}
\]

---

## Files

```text
pi_muon_optimizer/
├── pi_muon.py
├── demo_toy_mlp.py
├── PI_MUON_DESIGN_AND_COMPARISON.md
├── README.md
└── requirements.txt
```

For a more detailed explanation of the algorithm, implementation choices,
parameter grouping, and differences from `controlled_muon_project`, see
`PI_MUON_DESIGN_AND_COMPARISON.md`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the demo

```bash
python demo_toy_mlp.py
```

The demo trains a tiny MLP on synthetic regression data and prints diagnostics:

- current loss;
- global multiplier \(\alpha_t\);
- raw \(\rho_t\);
- smoothed \(\bar\rho_t\);
- controller error;
- integral state;
- whether a fallback direction was used.

---

## Closure pattern

The optimizer requires a closure with the signature:

```python
def closure(backward: bool = True):
    optimizer.zero_grad(set_to_none=True)
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    if backward:
        loss.backward()
    return loss
```

This is necessary because the controller needs:

1. a loss and gradient at \(\theta_t\);
2. a forward-only loss at \(\theta_t + \alpha_t p_t\) on the **same minibatch**.

The extra cost is one additional forward pass per optimization step.

---

## Parameter grouping

Use:

```python
from pi_muon import PIMuon, default_muon_param_groups

param_groups = default_muon_param_groups(model.named_parameters())
optimizer = PIMuon(param_groups)
```

The helper follows `torch.optim.Muon`'s parameter scope: Muon is applied only
to 2D hidden matrix weights. Scalar/vector parameters, convolution kernels,
biases, norms, embeddings, and heads use AdamW-style fallback updates.

For custom control, pass your own parameter groups:

```python
optimizer = PIMuon([
    {"params": hidden_matrix_params, "use_muon": True},
    {"params": other_params, "use_muon": False},
])
```

---

## Practical defaults

For a first run:

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

For noisier training, reduce `kp` and `ki`, increase `beta_rho`, and tighten `multiplier_min` / `multiplier_max`.

---

## Important caveats

### 1. This is not production Muon

The implementation is intentionally readable. It is not distributed, fused, or optimized for large-scale training.

### 2. Same-minibatch loss matters

Do not compute \(f(\theta_t)\) on one batch and \(f(\theta_{t+1})\) on another batch. That would mix optimization progress with minibatch difficulty variation.

### 3. Hard rejection is off by default

For stochastic neural-network training, rejecting steps based on noisy minibatch measurements can be harmful. The default is:

```python
reject_bad_steps=False
```

For deterministic experiments, you may try:

```python
reject_bad_steps=True
rho_min=0.0
```

With rejection enabled, the optimizer uses bounded backtracking before giving
up on a step.

### 4. Weight decay and \(\rho_t\)

Nonzero `weight_decay` follows PyTorch AdamW/Muon-style decoupled parameter
decay:

```text
theta <- (1 - alpha * weight_decay) * theta
theta <- theta + alpha * p
```

The decay is not folded into Adam moments or Muon momentum. If your closure
loss does not include the corresponding regularization term, then \(\rho_t\)
measures same-batch data loss, not the full regularized objective. For clean
experiments, start with `weight_decay=0.0`.

### 5. Muon follows PyTorch's 2D-parameter scope

The default helper follows `torch.optim.Muon`: Muon is applied only to 2D
hidden matrix weights. Other parameters, including vectors, biases, norms,
embeddings, heads, and convolution kernels, use AdamW-style directions.

---

## Minimal usage

```python
model = MyModel()
param_groups = default_muon_param_groups(model.named_parameters())
optimizer = PIMuon(param_groups)

for xb, yb in loader:
    def closure(backward: bool = True):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        if backward:
            loss.backward()
        return loss

    optimizer.step(closure)
    print(optimizer.last_stats)
```
