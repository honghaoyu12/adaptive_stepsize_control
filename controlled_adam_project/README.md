# Outer-Loop Controlled Adam

This project compares **vanilla Adam** with an **outer-loop controlled Adam** optimizer on simple deterministic objective functions.

The key idea is that Adam chooses a preconditioned search direction, while an outer feedback controller chooses the global step length by comparing the **actual objective decrease** with the **first-order predicted decrease**.

---

## 1. Vanilla Adam

Adam maintains exponential moving averages of gradients and squared gradients:

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
\hat v_t = \frac{v_t}{1-\beta_2^t},
```

vanilla Adam updates

```math
x_{t+1}
=
x_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

---

## 2. Outer-loop controlled Adam

Define Adam's proposed direction as

```math
p_t
=
-
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

Instead of using a fixed global learning rate, we write

```math
x_{t+1}^{\text{trial}}
=
x_t + \alpha_t p_t.
```

Here Adam/RMSProp-style internal adaptation chooses the direction `p_t`, while the outer-loop controller chooses the scalar global multiplier `alpha_t`.

The first-order Taylor predicted decrease is

```math
\Delta \hat f_t
=
-\alpha_t g_t^\top p_t.
```

The actual decrease is

```math
\Delta f_t
=
f(x_t)-f(x_t+\alpha_t p_t).
```

The actual-over-predicted decrease ratio is

```math
\rho_t
=
\frac{\Delta f_t}{\Delta \hat f_t}
=
\frac{f(x_t)-f(x_t+\alpha_t p_t)}{-\alpha_t g_t^\top p_t}.
```

Then the outer-loop proportional controller updates

```math
\alpha_{t+1}
=
\alpha_t \exp\left[K_p(\rho_t-\rho^\star)\right].
```

For minibatch neural-network experiments, `TorchControlledAdam` can smooth the
control signal with an exponential moving average,

```math
\bar\rho_t
=
\beta_\rho \bar\rho_{t-1}
+
(1-\beta_\rho)\rho_t,
```

and update `alpha_t` from `bar rho_t` instead of the raw per-minibatch `rho_t`.
It also clips the multiplicative alpha update factor. This reduces the chance
that one bad minibatch permanently collapses the global step size.

The PyTorch controller also includes an optional trust-region style recovery
rule. If a same-minibatch trial step is accepted without backtracking, the
smoothed ratio is high, and `alpha_t` is already tiny, the controller forces a
larger expansion factor for the next step. This treats "high rho at tiny alpha"
as evidence that the current radius may be too conservative rather than as a
reason to keep crawling upward slowly.

Interpretation:

- If `rho_t > rho_star`, the step performed well relative to the local model, so increase the global step size.
- If `rho_t < rho_star`, the step underperformed, so decrease the global step size.
- If `rho_t < rho_min`, the implementation can reject the step and keep the previous point.

---

## 3. Why this is not just Adam

Adam adapts using gradient statistics:

```math
m_t, v_t.
```

The outer-loop controller adapts using objective feedback:

```math
\rho_t
=
\frac{\text{actual decrease}}{\text{predicted decrease}}.
```

So the hybrid is best viewed as:

```text
Adam:       choose a preconditioned direction.
Controller: choose the global step length along that direction.
```

---

## 4. Included objectives

The project includes several deterministic 2D objectives:

### Anisotropic quadratic

```math
f(x,y) = \frac{1}{2}(50x^2+y^2).
```

This is convex but ill-conditioned, so it is useful for studying step-size sensitivity.

### Rosenbrock function

```math
f(x,y) = (a-x)^2 + b(y-x^2)^2.
```

By default, `a=1` and `b=100`. This is nonconvex and has a narrow curved valley.

### Himmelblau function

```math
f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2.
```

This is nonconvex and has four equivalent minima.

### Rastrigin function

```math
f(x,y) = 20 + x^2 + y^2 - 10\cos(2\pi x) - 10\cos(2\pi y).
```

This has many local wells, so it is useful for seeing when local methods get trapped.

### Beale function

```math
f(x,y) =
(1.5-x+xy)^2
+(2.25-x+xy^2)^2
+(2.625-x+xy^3)^2.
```

This has a curved valley and a sharp global minimum at `(3, 0.5)`.

### Ackley function

Ackley has a broad basin with oscillatory ripples, making it a useful test for
local methods on mildly deceptive multimodal landscapes.

### Six-hump camel function

Six-hump camel has multiple local minima and two global minima, so it is useful
for comparing which basin different methods enter from the same starting point.

### Goldstein-Price function

Goldstein-Price has steep nonlinear coupling and a large dynamic range, making
it a stress test for global step-size control.

### Easom function

Easom has a very sharp isolated global minimum at `(pi, pi)`, which makes it a
good precision and targeting test.

---

## 5. Project structure

```text
controlled_adam_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── controlled_adam/
│       ├── __init__.py
│       ├── objectives.py
│       ├── optimizers.py
│       ├── torch_optimizers.py
│       └── plotting.py
├── examples/
│   ├── run_demo.py
│   └── run_mnist_demo.py
├── tests/
│   └── test_optimizers.py
└── outputs/
    └── .gitkeep
```

---

## 6. Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## 7. Run the demo

```bash
python examples/run_demo.py
```

This produces plots in `outputs/` for every objective:

```text
outputs/quadratic_objective.png
outputs/quadratic_alpha.png
outputs/quadratic_rho.png
outputs/quadratic_trajectory.png
outputs/rosenbrock_objective.png
outputs/rosenbrock_alpha.png
outputs/rosenbrock_rho.png
outputs/rosenbrock_trajectory.png
outputs/himmelblau_objective.png
outputs/himmelblau_alpha.png
outputs/himmelblau_rho.png
outputs/himmelblau_trajectory.png
outputs/rastrigin_objective.png
outputs/rastrigin_alpha.png
outputs/rastrigin_rho.png
outputs/rastrigin_trajectory.png
outputs/beale_objective.png
outputs/beale_alpha.png
outputs/beale_rho.png
outputs/beale_trajectory.png
outputs/ackley_objective.png
outputs/ackley_alpha.png
outputs/ackley_rho.png
outputs/ackley_trajectory.png
outputs/six_hump_camel_objective.png
outputs/six_hump_camel_alpha.png
outputs/six_hump_camel_rho.png
outputs/six_hump_camel_trajectory.png
outputs/goldstein_price_objective.png
outputs/goldstein_price_alpha.png
outputs/goldstein_price_rho.png
outputs/goldstein_price_trajectory.png
outputs/easom_objective.png
outputs/easom_alpha.png
outputs/easom_rho.png
outputs/easom_trajectory.png
```

It also saves diagnostic CSV files:

```text
outputs/quadratic_controlled_adam_diagnostics.csv
outputs/rosenbrock_controlled_adam_diagnostics.csv
outputs/himmelblau_controlled_adam_diagnostics.csv
outputs/rastrigin_controlled_adam_diagnostics.csv
outputs/beale_controlled_adam_diagnostics.csv
outputs/ackley_controlled_adam_diagnostics.csv
outputs/six_hump_camel_controlled_adam_diagnostics.csv
outputs/goldstein_price_controlled_adam_diagnostics.csv
outputs/easom_controlled_adam_diagnostics.csv
```

---

## 8. Run the MNIST experiment

The project includes a PyTorch experiment comparing vanilla Adam with
same-minibatch controlled Adam on a small MLP classifier.

```bash
python examples/run_mnist_demo.py --download
```

By default, the script uses a small deterministic subset for quick experiments.
Useful options include:

```bash
python examples/run_mnist_demo.py \
  --epochs 3 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --download
```

If MNIST is already cached, `--download` is not required. If MNIST is not
available and downloading is disabled or fails, the script falls back to
`sklearn.datasets.load_digits` for offline smoke testing. That fallback is not
full MNIST, but it exercises the same optimizer code path.

The same script can benchmark Fashion-MNIST. If the four IDX gzip files are in
a flat folder, pass that folder directly:

```bash
python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 3 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --output-dir outputs/fashion_mnist
```

The MNIST experiment writes:

```text
outputs/mnist/mnist_epoch_metrics.csv
outputs/mnist/mnist_controlled_step_diagnostics.csv
outputs/mnist/mnist_loss.png
outputs/mnist/mnist_accuracy.png
outputs/mnist/mnist_controlled_alpha.png
outputs/mnist/mnist_controlled_rho.png
```

The controlled optimizer implementation lives in
`src/controlled_adam/torch_optimizers.py` as `TorchControlledAdam`.

The script also supports CIFAR-10 with a batch-normalized CNN, CIFAR
normalization, and train-time random crop/flip augmentation:

```bash
python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --download \
  --model auto \
  --epochs 10 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-trust-alpha-threshold 2e-3 \
  --controlled-trust-expand-factor 2.5 \
  --controlled-max-alpha-factor 1.25 \
  --controlled-rho-star 0.5 \
  --output-dir outputs/cifar10_stronger_cnn_10epochs_ablation_tuned_cifar1
```

`--model auto` uses the MLP for MNIST/Fashion-MNIST and the CNN for CIFAR-10.
If CIFAR-10 is already extracted under `data/`, `--download` is not required.
For CIFAR-10, training uses random augmentation while train/test metrics use
deterministic normalized evaluation transforms on the same subset indices.

Recent CIFAR-10 observations:

- The stronger CNN reaches a much more reasonable subset baseline than the
  earlier tiny CNN: about `0.58-0.61` test accuracy after 10 epochs and about
  `0.72` after 40 epochs with vanilla/fixed Adam on the 5000/1000 subset.
- In the 40-epoch controlled run with `controlled-alpha-max=5e-2`, controlled
  alpha grew all the way to the cap, far above the useful fixed `1e-3` scale,
  and accuracy degraded after peaking around epoch 23.
- For the next CIFAR tuning pass, prefer a smaller cap such as
  `--controlled-alpha-max 3e-3` and possibly a slower growth cap such as
  `--controlled-max-alpha-factor 1.1`.
- The `3e-3` cap largely fixes the 40-epoch degradation: controlled EMA reaches
  about `0.715` final test accuracy on the 5000/1000 subset, close to
  vanilla/fixed Adam at about `0.718/0.717`. Raw rho peaks higher
  (`0.739` around epoch 38) but gives some performance back by the final epoch.
- Since controlled variants still end at the alpha cap, a useful next check is
  a tighter cap such as `--controlled-alpha-max 2e-3` plus best-epoch reporting
  or early stopping.
- A tighter `--controlled-alpha-max 1.5e-3` with
  `--controlled-max-alpha-factor 1.05` and `--controlled-rho-star 0.6`
  produced the best CIFAR final result so far on the 5000/1000 subset:
  `controlled_ema_trust` reached `0.731` final test accuracy, compared with
  `0.718` for vanilla Adam and `0.717` for fixed Adam-direction.
- Each benchmark output directory now includes `run_metadata.json` and
  `run_metadata.txt` with the dataset, transforms, model architecture,
  trainable parameter count, optimizer variants, and controller hyperparameters.

To run an ablation that isolates which part of the controller matters, add
`--ablation`:

```bash
python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 20 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_20epochs_ablation
```

This trains all variants from the same initialization and minibatch order:

- `vanilla_adam`: PyTorch Adam baseline.
- `fixed_adam_direction`: the same Adam-style direction with fixed scalar
  alpha and same-minibatch diagnostics, but no alpha control.
- `controlled_raw_rho`: raw per-minibatch rho controller.
- `controlled_ema`: EMA-smoothed rho controller.
- `controlled_ema_trust`: EMA-smoothed rho plus trust-region recovery.

The ablation writes one shared epoch CSV and one step-diagnostics CSV for each
Adam-direction variant.

---

## 9. Run tests

```bash
pytest
```

---

## 10. Controlled Adam on minibatches

The key stochastic-training caveat is that the controlled ratio is computed on
the **same minibatch** before and after a trial step. For minibatch `B_t`, the
implemented ratio is

```math
\rho_t
=
\frac{
f_{B_t}(\theta_t) - f_{B_t}(\theta_t + \alpha_t p_t)
}{
-\alpha_t \nabla f_{B_t}(\theta_t)^\top p_t
}.
```

Both losses in the numerator use the same `B_t`. This avoids confusing random
minibatch-to-minibatch variation with genuine progress caused by the proposed
parameter update.

The controlled Adam minibatch step is:

1. Draw minibatch `B_t`.
2. Compute `loss_before = f_Bt(theta_t)`.
3. Backpropagate to obtain `grad f_Bt(theta_t)`.
4. Use Adam's moment estimates to construct the direction `p_t`.
5. Compute the first-order predicted decrease
   `predicted = -alpha_t * grad^T p_t`.
6. Take a trial step to `theta_trial = theta_t + alpha_t p_t`.
7. Recompute `loss_after = f_Bt(theta_trial)` on the **same minibatch**.
8. Compute `actual = loss_before - loss_after` and
   `rho = actual / predicted`.
9. Update the optional `rho` EMA.
10. Accept or reject the trial step, then update `alpha_t` with a clipped
    multiplicative factor.
11. If enabled, apply the trust-region recovery rule when an accepted,
    non-backtracked step has high smoothed `rho` and tiny `alpha_t`.

This requires one extra forward pass per minibatch, plus parameter rollback when
a step is rejected. For a fair comparison with vanilla Adam, the MNIST
experiment uses:

- the same model architecture;
- the same initial weights;
- the same train/test split;
- the same minibatch order and random seed;
- logged train loss, train accuracy, test accuracy, alpha, rho, and accepted
  step rate.

The default neural-network controller settings are:

```text
rho_beta = 0.9
kp = 0.05
rho_star = 0.7
alpha_min = 1e-5
alpha_max = 5e-2
min_alpha_factor = 0.8
max_alpha_factor = 1.05
trust_region_expand = True
trust_region_rho_threshold = 0.9
trust_region_alpha_threshold = 1e-4
trust_region_expand_factor = 1.5
```

### Interpreting rho when alpha is tiny

If the global step size `alpha_t` becomes very small, `rho_t` often drifts
toward 1. This does not necessarily mean optimization is healthy. For a smooth
loss,

```math
f(\theta + \alpha p)
\approx
f(\theta) + \alpha g^\top p
+ \frac{1}{2}\alpha^2 p^\top H p.
```

Therefore

```math
\rho
\approx
1
-
\frac{
\frac{1}{2}\alpha p^\top H p
}{
-g^\top p
}.
```

As `alpha -> 0`, the second-order error disappears and `rho -> 1` even if the
actual loss decrease is too small to matter. In other words, `rho` measures
first-order model agreement, not progress magnitude.

This creates a possible failure mode:

```text
alpha collapses -> rho approaches 1 -> controller sees "good" steps
```

while learning may have stalled. Future controller variants should consider
progress magnitude, alpha recovery rules, or treating tiny predicted/actual
decreases as an uninformative ratio.

The current PyTorch implementation uses a first recovery rule for this case:
when `rho_ema >= trust_region_rho_threshold`, `alpha_t <=
trust_region_alpha_threshold`, and the accepted trial step did not require
backtracking, the next-alpha multiplier is at least
`trust_region_expand_factor`. This mirrors classical trust-region logic:
expand the radius only when the local model is reliable and the step appears to
be limited by the current radius.

---

## 11. Practical notes

For deterministic toy functions, it is meaningful to evaluate the before/after
objective exactly. For stochastic minibatch training, same-minibatch evaluation
is required for a meaningful local progress ratio.

The implementation includes safeguards:

- reject steps with `rho_t <= rho_min`;
- shrink `alpha_t` if Adam's momentum direction is not descent-like;
- clip `alpha_t` between `alpha_min` and `alpha_max`.
- optionally update `alpha_t` from an EMA-smoothed rho signal and clip the
  multiplicative alpha update factor.
- optionally force trust-region expansion after high-quality tiny accepted
  steps, and log `alpha_next`, `alpha_update_factor`, and
  `trust_region_expanded` in minibatch diagnostics.
