# Adaptive Step-Size Control for Gradient Descent

This project demonstrates a simple feedback-control interpretation of gradient descent.

Instead of using a fixed learning rate, we adapt the learning rate by comparing the **actual decrease** in the objective with the **first-order predicted decrease** from Taylor expansion.

The demo uses a two-dimensional quadratic objective,

```math
f(x, y) = \frac{1}{2}(50x^2 + y^2),
```

which has strong curvature in the `x` direction and weaker curvature in the `y` direction. This makes it a useful toy problem for seeing why adaptive step sizes can help.

---

## Core idea

For gradient descent,

```math
x_{t+1} = x_t - \eta_t \nabla f(x_t),
```

let

```math
g_t = \nabla f(x_t).
```

A first-order Taylor prediction gives

```math
\hat f_{t+1} = f_t - \eta_t \lVert g_t \rVert^2.
```

The predicted decrease is

```math
\Delta \hat f_t = \eta_t \lVert g_t \rVert^2,
```

and the actual decrease is

```math
\Delta f_t = f_t - f_{t+1}.
```

The controller uses the ratio

```math
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t}.
```

If `rho_t` is close to 1, the Taylor model predicted the step well.

If `rho_t` is small or negative, the step was too aggressive.

If `rho_t` is larger than the target, the step may be increased.

The proportional controller updates the learning rate as

```math
\eta_{t+1} = \eta_t \exp\left(K_p(\rho_t - \rho^\star)\right).
```

This keeps the learning rate positive while increasing or decreasing it smoothly.

---

## Project structure

```text
adaptive_stepsize_control/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── adaptive_stepsize_control/
│       ├── __init__.py
│       ├── objectives.py
│       ├── optimizers.py
│       └── plotting.py
├── examples/
│   └── run_quadratic_demo.py
├── tests/
│   └── test_quadratic_demo.py
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

pip install -e .
```

Alternatively, install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Run the demo

```bash
python examples/run_quadratic_demo.py
```

This will run both:

1. fixed-step gradient descent;
2. controlled gradient descent with adaptive learning rate.

It will save plots into the `outputs/` directory:

```text
outputs/objective_value.png
outputs/adaptive_step_size.png
outputs/rho_ratio.png
outputs/trajectory.png
```

---

## Run tests

```bash
pytest
```

The tests check that the objective and gradient are correct, and that the controlled optimizer reduces the objective on the demo problem.

---

## Notes

This method is closely related to ideas from:

- adaptive step-size control;
- backtracking line search;
- trust-region methods;
- feedback control of numerical algorithms.

Unlike AdaGrad, RMSProp, or Adam, this method does not adapt the learning rate from gradient-history statistics. Instead, it adapts the learning rate from the mismatch between the predicted and actual decrease in the objective.

A natural hybrid would combine this global controller with a diagonal adaptive preconditioner such as AdaGrad:

```math
x_{t+1} = x_t - \eta_t D_t \nabla f(x_t),
```

where `D_t` controls coordinate-wise scaling and `eta_t` is controlled using the actual-versus-predicted decrease ratio.

---

## Related subproject: controlled Adam

The workspace also contains `controlled_adam_project/`, which applies the same
actual-versus-predicted decrease idea as an outer-loop controller around Adam.
Adam supplies the preconditioned direction, while the controller adapts the
global step multiplier.

The subproject includes a PyTorch neural-network comparison on MNIST:
vanilla Adam versus controlled Adam. Run it from `controlled_adam_project/`:

```bash
python examples/run_mnist_demo.py --download
```

For minibatch training, the control ratio evaluates the trial loss on the
**same minibatch** used to compute the gradient:

```math
\rho_t
=
\frac{
f_{B_t}(\theta_t) - f_{B_t}(\theta_t + \alpha_t p_t)
}{
-\alpha_t \nabla f_{B_t}(\theta_t)^\top p_t
}.
```

Using a different minibatch for the after-step loss would mix real optimization
progress with random minibatch variation, so it would not be a reliable control
signal.

If MNIST is unavailable locally and download is disabled, the script falls back
to `sklearn.datasets.load_digits` for offline smoke testing.
