# Delayed Feedback Adam

This project implements an Adam-style optimizer with an outer feedback controller for the global learning-rate multiplier.

The main class is:

```python
from delayed_feedback_adam import DelayedFeedbackAdam
```

It is designed to test the low-overhead idea we discussed:

> Instead of evaluating `f(x_{t+1})` immediately after a trial step, use the loss naturally computed at the next iteration to evaluate how the previous step performed.

This avoids the extra forward pass required by a same-step actual-vs-predicted decrease controller.

---

## 1. Inner optimizer: Adam

Adam maintains first- and second-moment estimates of the gradient:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t \odot g_t.
```

After bias correction,

```math
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
```

The Adam direction is

```math
p_t = -\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

With `decoupled_weight_decay=True`, the implementation follows the supported
local `torch.optim.AdamW` update path. This was checked against PyTorch `2.10.0`;
the delayed controller is the experimental part, while the AdamW direction and
decoupled weight-decay behavior match PyTorch for the simple non-AMSGrad path.

The parameter update is

```math
x_{t+1} = x_t + \eta\alpha_t p_t,
```

where `eta` is the base learning rate and `alpha_t` is the controller's global multiplier.

---

## 2. Outer controller: delayed actual-vs-predicted decrease

For a general inner direction `p_t`, the predicted first-order decrease is

```math
\Delta \hat f_t = -\eta\alpha_t g_t^\top p_t.
```

The current-step controller we discussed earlier would immediately evaluate

```math
f(x_t+\eta\alpha_t p_t)
```

and compute

```math
\rho_t = \frac{f(x_t)-f(x_t+\eta\alpha_t p_t)}{-\eta\alpha_t g_t^\top p_t}.
```

That is accurate, but in neural-network training it usually requires an extra forward pass.

The delayed controller instead waits until the next iteration.

At step `t-1`, it stores:

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

In a normal PyTorch training loop, you already compute the current loss and gradient:

```python
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step(loss=loss.item())
```

The delayed controller uses the current `loss` as the measurement needed to evaluate the previous step.

So the delayed version adds:

- no extra forward pass;
- no extra backward pass;
- only a few scalar variables;
- one additional inner product between gradients and Adam directions.

The same-step version is more accurate, but normally requires an extra forward pass to evaluate the trial point.

---

## 4. Key difference from the old same-step controller

| Feature | Same-step controller | Delayed controller in this project |
|---|---:|---:|
| Uses `f(x_{t+1})` immediately? | Yes | No |
| Extra forward pass? | Usually yes | No |
| Can reject a bad step before applying it? | Yes | No |
| Feedback delay | None | One step |
| Feedback quality in deterministic optimization | Accurate | Accurate but delayed |
| Feedback quality in minibatch training | Accurate if same batch is reused | Noisy if consecutive minibatches differ |
| Best interpretation | Line-search/trust-region-like | Delayed adaptive learning-rate regulator |

---

## 5. Important caveat for minibatch training

If step `t-1` used minibatch `B_{t-1}` and step `t` uses minibatch `B_t`, then the delayed loss difference is

```math
f_{B_{t-1}}(\theta_{t-1}) - f_{B_t}(\theta_t).
```

This is not the exact same-minibatch decrease. It mixes optimization progress with minibatch sampling noise.

Therefore the optimizer uses:

1. clipping of raw `rho`;
2. exponential smoothing of `rho`;
3. clipping of the per-step learning-rate multiplier;
4. bounds on the global multiplier `alpha`.

---

## 6. Installation

From the project root:

```bash
pip install -e .
```

---

## 7. Minimal usage

```python
import torch
from delayed_feedback_adam import DelayedFeedbackAdam

model = torch.nn.Linear(10, 1)
criterion = torch.nn.MSELoss()
optimizer = DelayedFeedbackAdam(
    model.parameters(),
    lr=1e-3,
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

## 8. Recommended initial settings

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
rho_star=0.5 to 0.8  # initial guess only; see CIFAR note below
kp=0.01 to 0.1
ki=0.0
kd=0.0
rho_clip=(-1.0, 2.0)
multiplier_bounds=(0.8, 1.25)
alpha_bounds=(0.1, 10.0)
```

Start with delayed P control first. Only add PI after the delayed P version is stable.

### CIFAR ResNet calibration note

A 20-epoch CIFAR-10 ResNet comparison was run from the monorepo root:

```text
outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/
```

It compared vanilla Adam, same-step controlled Adam, and delayed-feedback Adam
variants on a 10k/2k CIFAR-10 subset. The delayed variants avoided the extra
same-step forward pass, but their delayed rho signal was much lower than the
same-step rho signal:

```text
same-step controlled final mean rho: about 0.88
delayed final mean rho: about 0.13 to 0.18
```

With same-step-style targets such as `rho_star=0.7` or `0.8`, the delayed
controller drove alpha to its configured floor. The best delayed final accuracy
in that run was `delayed_safe` at `0.6985`, only slightly above vanilla Adam
at `0.6915` and below same-step controlled Adam.

For ordinary shuffled minibatch training, do not assume same-step `rho_star`
values transfer directly to delayed feedback. A better next sweep should test
delayed-specific targets around `rho_star=0.15` to `0.30`, with alpha floors
near `0.8x` to `1.1x` the base learning rate and caps near `1.25x` to `1.75x`.

---

## 9. Files

```text
delayed_feedback_adam/
├── README.md
├── pyproject.toml
├── requirements.txt
├── delayed_feedback_adam/
│   ├── __init__.py
│   └── optimizer.py
├── docs/
│   └── method_note.md
├── examples/
│   ├── train_toy_regression.py
│   └── compare_with_torch_adam.py
└── tests/
    └── test_optimizer_smoke.py
```
