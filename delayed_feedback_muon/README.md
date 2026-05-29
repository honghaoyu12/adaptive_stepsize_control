# Delayed Feedback Muon

This project implements a **Muon-style optimizer** with an outer delayed feedback controller for the global learning-rate multiplier.

The main class is:

```python
from delayed_feedback_muon import DelayedFeedbackMuon
```

It is designed to test the low-overhead idea we discussed:

> Instead of evaluating `f(x_{t+1})` immediately after a trial step, use the loss naturally computed at the next training iteration to evaluate how the previous step performed.

This avoids the extra forward pass required by a same-step actual-vs-predicted decrease controller.

---

## 1. Inner optimizer: Muon + auxiliary AdamW

Muon is commonly described as **MomentUm Orthogonalized by Newton-Schulz**. For matrix-like hidden-layer parameters, it:

1. forms a momentum buffer;
2. optionally applies a Nesterov-style momentum correction;
3. approximately orthogonalizes the update matrix using Newton-Schulz iterations;
4. applies the orthogonalized update.

For a gradient matrix `G_t`, Muon forms a momentum update:

```math
B_t = \operatorname{lerp}(B_{t-1}, G_t, 1-\mu).
```

With Nesterov enabled, the matrix to orthogonalize is

```math
\widetilde B_t = \operatorname{lerp}(G_t, B_t, \mu).
```

Then Newton-Schulz produces an approximately semi-orthogonal matrix:

```math
O_t \approx \operatorname{Ortho}(\widetilde B_t).
```

The Muon direction is

```math
p_t^{\mathrm{Muon}} = -O_t.
```

This implementation uses Muon automatically for 2D tensors. Non-2D parameters,
such as biases and normalization gains, are updated with auxiliary AdamW.

That split follows the usual practical advice for Muon-style training: use Muon for hidden weight matrices and use AdamW-style updates for parameters that are not suitable for matrix orthogonalization.

The Muon and AdamW inner directions were checked against the local
`torch.optim.Muon` and `torch.optim.AdamW` implementations in PyTorch `2.10.0`.
The delayed controller is the experimental part; the supported inner optimizer
paths follow PyTorch conventions for momentum, Nesterov, shape learning-rate
adjustment, and decoupled weight decay.

---

## 2. Outer controller: delayed actual-vs-predicted decrease

The optimizer applies an outer multiplier `alpha_t`. For any inner direction `p_t`, the update is

```math
x_{t+1}=x_t+\eta\alpha_t p_t.
```

The first-order predicted decrease is

```math
\Delta \hat f_t = -\eta\alpha_t g_t^\top p_t.
```

The old same-step controller would immediately evaluate

```math
f(x_t+\eta\alpha_t p_t)
```

and compute

```math
\rho_t=
\frac{f(x_t)-f(x_t+\eta\alpha_t p_t)}{-\eta\alpha_t g_t^\top p_t}.
```

That is accurate but usually requires an extra forward pass.

The delayed controller waits until the next iteration. At step `t-1`, it stores

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

Then it updates the multiplier:

```math
\alpha_t
=
\alpha_{t-1}\exp\left[K_P(\bar\rho_{t-1}-\rho^\star)\right].
```

The implementation also supports optional PI/PID terms.

---

## 3. Why this has lower overhead

A normal PyTorch training loop already computes the current loss and gradient:

```python
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step(loss=loss.item())
```

The delayed controller uses the current `loss` as the measurement needed to evaluate the previous step.

So the delayed version adds:

- no extra forward pass;
- no extra backward pass;
- only a few scalar controller variables;
- one extra inner-product diagnostic per parameter tensor;
- Newton-Schulz work for Muon itself, which is intrinsic to Muon rather than caused by the delayed controller.

---

## 4. Difference from the old `f(x_{t+1})` optimizer

| Feature | Same-step controller | Delayed controller in this project |
|---|---:|---:|
| Uses `f(x_{t+1})` immediately? | Yes | No |
| Extra forward pass? | Usually yes | No |
| Can reject a bad step before applying it? | Yes | No |
| Feedback delay | None | One step |
| Feedback quality in deterministic optimization | Accurate | Accurate but delayed |
| Feedback quality in minibatch training | Accurate if same batch is reused | Noisy if consecutive minibatches differ |
| Best interpretation | Line-search/trust-region-like | Delayed adaptive learning-rate regulator |

The old method is more faithful to line search or trust-region logic because it evaluates the trial point immediately.

The delayed method is cheaper but weaker: it reacts after the step has already happened.

---

## 5. Difference from DelayedFeedbackAdam

`DelayedFeedbackAdam` used Adam as the inner optimizer:

```math
p_t^{\mathrm{Adam}}
= -\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

`DelayedFeedbackMuon` instead uses Muon for 2D matrix tensors:

```math
p_t^{\mathrm{Muon}}
= -\operatorname{Ortho}(\widetilde B_t).
```

So the outer controller is conceptually the same, but the inner direction is different.

| Component | DelayedFeedbackAdam | DelayedFeedbackMuon |
|---|---|---|
| Matrix weights | Adam direction | Muon orthogonalized momentum direction |
| Biases / 1D parameters | Adam | auxiliary AdamW |
| Step-size controller | delayed `rho` controller | delayed `rho` controller |
| Main added cost beyond standard optimizer | small | Newton-Schulz orthogonalization from Muon |

---

## 6. Important caveat for minibatch training

If step `t-1` used minibatch `B_{t-1}` and step `t` uses minibatch `B_t`, then the delayed loss difference is

```math
f_{B_{t-1}}(\theta_{t-1}) - f_{B_t}(\theta_t).
```

This is not the exact same-minibatch decrease. It mixes optimization progress with minibatch sampling noise.

Therefore the optimizer uses:

1. clipping of raw `rho`;
2. exponential smoothing of `rho`;
3. clipping of the per-step learning-rate multiplier;
4. bounds on the global multiplier `alpha`;
5. optional fallback to the negative gradient if Muon or AdamW proposes a non-descent direction.

---

## 7. Installation

From the project root:

```bash
pip install -e .
```

---

## 8. Minimal usage

```python
import torch
from delayed_feedback_muon import DelayedFeedbackMuon

model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
criterion = torch.nn.MSELoss()

optimizer = DelayedFeedbackMuon(
    model.parameters(),
    lr=2e-2,
    weight_decay=0.01,
    alpha_init=1.0,
    rho_star=0.8,
    kp=0.05,
    rho_beta=0.95,
)

for inputs, targets in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()

    # Important: pass the current loss value.
    optimizer.step(loss=loss.item())

    diagnostics = optimizer.get_diagnostics()
    print(diagnostics["alpha"], diagnostics["rho_bar"])
```

---

## 9. Explicit parameter groups

You can explicitly separate Muon and auxiliary AdamW parameters:

```python
muon_params = []
adamw_params = []
for name, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in name and "head" not in name:
        muon_params.append(p)
    else:
        adamw_params.append(p)

optimizer = DelayedFeedbackMuon([
    {"params": muon_params, "use_muon": True, "lr": 2e-2},
    {"params": adamw_params, "use_muon": False, "lr": 1e-3},
])
```

---

## 10. Recommended initial settings

For deterministic toy problems:

```python
rho_beta=0.0
rho_star=0.5 or 0.8
kp=0.05 to 0.5
ki=0.0
kd=0.0
```

For neural-network minibatch training:

```python
rho_beta=0.95
rho_star=0.5 to 0.8
kp=0.01 to 0.1
ki=0.0
kd=0.0
rho_clip=(-1.0, 2.0)
multiplier_bounds=(0.8, 1.25)
alpha_bounds=(0.1, 10.0)
```

Start with delayed P control first. Only add PI after delayed P is stable. Be cautious with PID because the derivative term can amplify minibatch noise.

---

## 11. References

- PyTorch Muon documentation: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- Keller Jordan's Muon write-up: https://kellerjordan.github.io/posts/muon/
- Keller Jordan's Muon repository: https://github.com/KellerJordan/Muon
