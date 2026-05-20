# Controlled Muon

This project mirrors the current controlled Adam project, but swaps in Muon-style orthogonalized update directions.

It supports:

1. The 2D function benchmark suite.
2. MNIST, Fashion-MNIST, and CIFAR-10 image benchmarks.
3. Vanilla vs controlled optimizer comparisons.
4. Ablation variants with raw rho, EMA rho, and trust-region style recovery.

## Core idea

Muon chooses the update geometry through orthogonalization. An outer controller chooses the global step size by comparing actual and predicted decrease:

```math
\rho_t = \frac{f(W_t) - f(W_t + \alpha_t P_t)}{-\alpha_t \langle G_t, P_t \rangle_F}
```

The controller adapts `alpha_t` from `rho_t`, optionally smoothing it with an exponential moving average and clipping the multiplicative update factor.

For stochastic neural-network training, the extra trial evaluation must use the same minibatch as the gradient. Otherwise the ratio is dominated by minibatch noise instead of true progress.

## Included objectives

The 2D benchmark suite now matches the Adam project:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

## Image benchmarks

The main runner supports:

- `mnist`
- `fashion_mnist`
- `cifar10`

The same runner supports the same style of controlled variants as the Adam project.

## Run the function demo

```bash
python examples/run_matrix_quadratic_demo.py
```

## Run the image benchmark

```bash
python examples/run_mnist_demo.py --dataset fashion_mnist --ablation
```

For CIFAR-10:

```bash
python examples/run_mnist_demo.py --dataset cifar10 --model auto --ablation
```

## Outputs

The runner writes:

- epoch metrics CSV
- step diagnostics CSV
- loss plot
- accuracy plot
- controlled alpha plot
- run metadata JSON/TXT

## Tests

```bash
pytest
```
