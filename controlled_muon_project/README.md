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
python examples/run_mnist_demo.py --dataset fashion_mnist --download --ablation
```

For CIFAR-10:

```bash
python examples/run_mnist_demo.py --dataset cifar10 --model auto --download --ablation
```

For local Fashion-MNIST IDX files in the parent workspace:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset fashion_mnist \
  --fashion-folder ../fashion \
  --epochs 20 \
  --train-subset 4096 \
  --test-subset 1024 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/fashion_mnist_muon_20epoch_ablation
```

For local CIFAR-10 reused from the Adam subproject:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --data-dir ../controlled_adam_project/data \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/cifar10_muon_40epoch_ablation
```

## Recent benchmark results

Fashion-MNIST, 20 epochs, 4096 train / 1024 test:

| Optimizer | Final test acc | Best test acc | Best epoch |
|---|---:|---:|---:|
| `vanilla_muon` | `0.7432` | `0.7432` | 20 |
| `fixed_muon_direction` | `0.7432` | `0.7432` | 20 |
| `controlled_raw_rho` | `0.8320` | `0.8418` | 14 |
| `controlled_ema` | `0.8301` | `0.8389` | 18 |
| `controlled_ema_trust` | `0.8301` | `0.8389` | 18 |

CIFAR-10, 40 epochs, 5000 train / 1000 test:

| Optimizer | Final test acc | Best test acc | Best epoch |
|---|---:|---:|---:|
| `vanilla_muon` | `0.699` | `0.709` | 31 |
| `fixed_muon_direction` | `0.694` | `0.701` | 38 |
| `controlled_raw_rho` | `0.725` | `0.731` | 27 |
| `controlled_ema` | `0.725` | `0.725` | 40 |
| `controlled_ema_trust` | `0.725` | `0.725` | 40 |

On these subset runs, the controlled Muon variants beat the fixed and vanilla
Muon baselines. The CIFAR-10 run is slow because the current implementation
does CPU/NumPy orthogonalization and evaluates same-minibatch trial losses for
controlled variants.

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
