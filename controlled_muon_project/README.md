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

For neural-network benchmarks, the PyTorch Muon path follows
`torch.optim.Muon`'s scope: Muon is used only for 2D hidden matrix weights.
Non-2D parameters, convolution kernels, biases, batch/norm parameters,
embeddings, and heads use AdamW-style fallback updates. Nonzero
`--weight-decay` is decoupled in the AdamW/Muon sense. The Muon direction uses
PyTorch's `lerp` momentum convention, quintic Newton-Schulz coefficients
`(3.4445, -4.7750, 2.0315)`, 5 default Newton-Schulz steps, and original
rectangular shape scaling.

## Run the function demo

```bash
python examples/run_matrix_quadratic_demo.py
```

For a self-contained deterministic 2D function optimization report, run:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

This report compares `vanilla_muon`, `controlled_raw_rho`, `controlled_ema`,
and `controlled_ema_trust` on the same nine 2D functions used by the Adam
report. It writes:

```text
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md
outputs/function_report_multistart/per_start_results.csv
outputs/function_report_multistart/aggregate_results.csv
outputs/function_report_multistart/benchmark_config.csv
outputs/function_report_multistart/*_surface_3d.png
outputs/function_report_multistart/*_trajectory_comparison.png
outputs/function_report_multistart/*_objective_curves.png
outputs/function_report_multistart/*_alpha_curves.png
```

For the shorter manager-facing report:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

The trimmed report writes Chinese and English Markdown files in
`outputs/function_report_manager_trimmed/`. The older function-report-only
`fixed_muon_direction` diagnostic was removed because it duplicated
`vanilla_muon` in the local 2D runner: both used fixed alpha and no rho
controller.

For 2D vector functions, the report uses a vector analogue of Muon: the
momentum vector is treated as a column matrix and passed through the same
orthogonalization utility used elsewhere in this subproject. The top-level
`FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md` explains how to interpret the Adam
and Muon reports together.

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

The historical tables below predate the official-style Muon parameter-scope
cleanup and should be treated as archival. For current PI-vs-vanilla
Fashion-MNIST comparisons from the latest experiment round, use the archived
`../outputs/backup_20260526_182414/pi_official_muon_*` folders in the parent
workspace, or rerun the PI benchmark into a fresh top-level output folder.

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

Fashion-MNIST, 20 epochs, five seeds, 1024 train / 512 test:

| Optimizer | Final test acc mean +/- std | Best test acc mean +/- std |
|---|---:|---:|
| `vanilla_muon` | `0.6051 +/- 0.0181` | `0.6051 +/- 0.0181` |
| `fixed_muon_direction` | `0.6051 +/- 0.0181` | `0.6051 +/- 0.0181` |
| `controlled_raw_rho` | `0.7289 +/- 0.0070` | `0.7289 +/- 0.0070` |
| `controlled_ema` | `0.7293 +/- 0.0067` | `0.7293 +/- 0.0067` |
| `controlled_ema_trust` | `0.7293 +/- 0.0067` | `0.7293 +/- 0.0067` |

This five-seed diagnostic run is saved under
`outputs/fashion_mnist_muon_multiseed_20epoch_5seeds_1k/`. It also records
elapsed seconds. On this small CPU run, the controlled variants had comparable
wall-clock time to vanilla Muon, though the timing is noisy. Conceptually, the
controller adds one extra same-minibatch forward pass but not an extra backward
pass, so the overhead should usually be much less than doubling the training
cost. Larger claims should use loss/accuracy versus wall-clock time.

## Outputs

The runner writes:

- epoch metrics CSV with train/test metrics, cumulative wall-clock seconds, and cumulative optimizer steps
- step diagnostics CSV
- loss plots by epoch, optimizer steps, and wall-clock time
- accuracy plot
- accuracy plots by optimizer steps and wall-clock time
- controlled alpha plot
- run metadata JSON/TXT

## Tests

```bash
pytest
```
