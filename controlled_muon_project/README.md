# Controlled Muon: Outer-Loop Step-Size Control for Muon

This project compares:

1. **Vanilla Muon** with a fixed global learning rate.
2. **Controlled Muon**, where Muon proposes the matrix-valued update direction and an outer feedback controller adapts the global step size.

The point is to separate two roles:

```text
Muon:              chooses update geometry / direction
Outer controller:  chooses global step length
```

This is the same idea we discussed for Adam, but now applied to a matrix optimizer.

---

## Mathematical formulation

For a matrix parameter

```math
W_t \in \mathbb{R}^{m \times n},
```

let

```math
G_t = \nabla_W f(W_t).
```

Muon first forms a momentum buffer

```math
M_t = \mu M_{t-1} + G_t.
```

With Nesterov-style momentum, the matrix sent into the orthogonalization step is

```math
B_t = \mu M_t + G_t.
```

Without Nesterov,

```math
B_t = M_t.
```

Muon then approximately orthogonalizes the update matrix:

```math
O_t = \operatorname{Ortho}(B_t),
```

where `Ortho` is implemented either by an exact SVD polar factor or by Newton-Schulz iterations.

The Muon proposed descent direction is

```math
P_t = -O_t.
```

Vanilla Muon applies

```math
W_{t+1} = W_t + \eta P_t.
```

Controlled Muon applies

```math
W_{t+1}^{\mathrm{trial}} = W_t + \alpha_t P_t,
```

where `alpha_t` is adapted by an outer-loop feedback controller.

The first-order predicted decrease is

```math
\Delta \hat f_t = -\alpha_t \langle G_t, P_t \rangle_F,
```

where

```math
\langle A, B \rangle_F = \operatorname{tr}(A^\top B).
```

The actual decrease is

```math
\Delta f_t = f(W_t) - f(W_t + \alpha_t P_t).
```

The actual-over-predicted ratio is

```math
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t}.
```

The controller updates

```math
\alpha_{t+1}
=
\operatorname{clip}\left(
\alpha_t \exp\left[K_p(\rho_t - \rho^\star)\right],
\alpha_{\min},
\alpha_{\max}
\right).
```

If the proposed Muon direction is not descent-like, meaning

```math
-\langle G_t, P_t \rangle_F \leq 0,
```

the controlled version rejects the step and shrinks `alpha_t`.

---

## Toy objective

The demo uses an anisotropic matrix quadratic:

```math
f(W) = \frac{1}{2}\sum_{i,j} C_{ij}(W_{ij} - T_{ij})^2,
```

where `C` is a positive curvature matrix and `T` is a target matrix.

The gradient is

```math
\nabla_W f(W) = C \odot (W - T).
```

This is deliberately simple, deterministic, and matrix-shaped, so it is easy to inspect whether the controller behaves sensibly.

---

## Project structure

```text
controlled_muon_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── controlled_muon/
│       ├── __init__.py
│       ├── objectives.py
│       ├── optimizers.py
│       ├── orthogonalization.py
│       └── plotting.py
├── examples/
│   └── run_matrix_quadratic_demo.py
├── tests/
│   ├── test_objectives.py
│   └── test_optimizers.py
└── outputs/
    └── .gitkeep
```

---

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -e .[dev]
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Run the demo

```bash
python examples/run_matrix_quadratic_demo.py
```

This writes plots and diagnostics into `outputs/`:

```text
outputs/objective_value.png
outputs/distance_to_target.png
outputs/controlled_alpha.png
outputs/rho_ratio.png
outputs/accepted_steps.png
outputs/controlled_diagnostics.csv
```

---

## Run tests

```bash
pytest
```

---

## Important caveats

This is a compact research prototype, not a production deep-learning optimizer.

For stochastic neural-network training, the actual decrease should be computed on the same mini-batch used to compute the gradient:

```math
f_{B_t}(W_t)
\quad\text{and}\quad
f_{B_t}(W_t + \alpha_t P_t).
```

Otherwise the ratio `rho_t` will be contaminated by mini-batch noise.

For practical Muon training, many parameter groups are not matrix-shaped and are usually handled by AdamW or a similar auxiliary optimizer. A full implementation would build one combined proposed update direction across all parameter groups, then apply the same actual-over-predicted decrease controller to the global multiplier.
