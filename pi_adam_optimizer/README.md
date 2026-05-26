# PI-Controlled Adam Wrapper

This project implements a **PI outer-loop controller wrapped around Adam**.

Adam supplies a search direction. The PI controller adjusts a global step-size
multiplier using the ratio between actual and predicted loss decrease.

## Core formulation

Let the current parameters be \(x_t\), the gradient be

\[
g_t = \nabla f(x_t),
\]

and let Adam propose a direction

\[
p_t = -\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon_{\mathrm{Adam}}}.
\]

The wrapper takes a trial step

\[
x_{t+1}^{\mathrm{trial}} = x_t + \alpha_t p_t.
\]

The first-order predicted decrease is

\[
\Delta \hat f_t = -\alpha_t g_t^\top p_t.
\]

The actual decrease is

\[
\Delta f_t = f(x_t) - f(x_{t+1}^{\mathrm{trial}}).
\]

The controller signal is

\[
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t + \epsilon}.
\]

We smooth it:

\[
\bar\rho_t = \beta_\rho \bar\rho_{t-1} + (1-\beta_\rho)\rho_t.
\]

Define the PI error:

\[
e_t = \bar\rho_t - \rho^\star.
\]

The leaky integral state is

\[
I_t = \lambda_I I_{t-1} + e_t.
\]

With anti-windup clipping:

\[
I_t \leftarrow \operatorname{clip}(I_t, I_{\min}, I_{\max}).
\]

Finally, update the global step-size multiplier in log-space:

\[
\log \alpha_{t+1}
=
\log \alpha_t + K_P e_t + K_I I_t.
\]

Equivalently,

\[
\alpha_{t+1}
=
\alpha_t \exp(K_P e_t + K_I I_t).
\]

The implementation also clips \(\alpha_t\) and optionally clips the single-step
multiplicative change.

## Why log-space?

Updating \(\log\alpha_t\) guarantees

\[
\alpha_t > 0.
\]

It also makes the controller naturally multiplicative, which is usually more
appropriate for learning rates than additive updates.

## Why same-batch evaluation matters

For neural networks, we usually do not know the true population objective.
Instead, for a minibatch \(B_t\), we estimate

\[
f_{B_t}(\theta_t)
\]

and

\[
f_{B_t}(\theta_{t+1}^{\mathrm{trial}}).
\]

These two losses must be evaluated on the **same minibatch**. Otherwise
\(\rho_t\) mixes optimization progress with minibatch noise.

The optimizer therefore expects a PyTorch closure that can be called twice:

```python
def closure(backward: bool = True):
    if backward:
        optimizer.zero_grad(set_to_none=True)
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    if backward:
        loss.backward()
    return loss
```

Inside `optimizer.step(closure)`, the closure is called once with
`backward=True` and once with `backward=False`.

## Files

```text
pi_adam_optimizer/
├── pi_adam.py
├── demo_toy_regression.py
├── PI_ADAM_DESIGN_AND_COMPARISON.md
├── requirements.txt
└── README.md
```

For a more detailed explanation of the algorithm, implementation choices, and
differences from `controlled_adam_project`, see
`PI_ADAM_DESIGN_AND_COMPARISON.md`.

## Run the demo

```bash
pip install -r requirements.txt
python demo_toy_regression.py
```

The demo writes plots to `outputs/`.

## Practical defaults

For deterministic or full-batch problems:

```python
optimizer = PIAdam(
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

For noisy minibatch training, start even more conservatively:

```python
kp=0.01
ki=0.0       # begin with P only
rho_smoothing=0.99
multiplicative_clip=(0.9, 1.1)
```

Then add a small integral term only after the P controller is stable.

## Important caveats

1. This costs one extra forward pass per optimizer step, because it evaluates
   the same-batch loss after the trial update.
2. If Adam's momentum direction is not a descent direction, the implementation
   can fall back to the steepest-descent direction.
3. Hard rejection is disabled by default because minibatch \(\rho_t\) is noisy.
   For deterministic optimization, `reject_bad_steps=True` enables bounded
   backtracking before a step is rejected.
4. The integral term can wind up, so the implementation clips the integral state.
5. Nonzero `weight_decay` follows PyTorch AdamW-style decoupled parameter
   decay. It is not folded into Adam moments or the predicted-decrease
   direction.
