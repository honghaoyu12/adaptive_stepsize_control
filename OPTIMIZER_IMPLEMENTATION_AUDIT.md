# Optimizer Implementation Audit

This note records the current consistency checks for the PI Adam, PI Muon, and
controlled Muon implementations.

## Official References Checked

- Local PyTorch 2.10 source:
  - `torch/optim/adamw.py`
  - `torch/optim/adam.py`
  - `torch/optim/_muon.py`
- Upstream PyTorch source:
  - `torch.optim.AdamW`
  - `torch.optim.Muon`

## AdamW Consistency

PyTorch `AdamW` is `Adam` with `decoupled_weight_decay=True`.

The relevant update is:

```text
theta <- (1 - lr * weight_decay) * theta
m     <- beta1 * m + (1 - beta1) * grad
v     <- beta2 * v + (1 - beta2) * grad^2
theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
```

Our AdamW-style paths now match this structure:

- `pi_adam_optimizer/pi_adam.py`
- `pi_muon_optimizer/pi_muon.py` AdamW fallback
- `examples/run_pi_fashion_mnist_multiseed.py` vanilla Adam baseline when
  `--weight-decay` is nonzero
- `examples/run_pi_fashion_mnist_multiseed.py` vanilla Muon fallback
- `controlled_muon_project/src/controlled_muon/torch_optimizers.py`
- `controlled_muon_project/examples/run_mnist_demo.py`

Important controller detail: decoupled decay is applied to the parameter trial
update, but it is not folded into the Adam direction used for
`predicted_decrease = -alpha * dot(g, p)`. This keeps the controller signal tied
to the closure loss and mirrors AdamW's separation of moments from decay.

## Muon Consistency

PyTorch `torch.optim.Muon` only supports 2D parameters. Users are expected to
optimize non-2D tensors with AdamW or another optimizer.

The relevant update is:

```text
buffer <- lerp(buffer, grad, 1 - momentum)
update <- lerp(grad, buffer, momentum)      # if nesterov
update <- NewtonSchulz(update)
lr     <- lr * sqrt(max(1, rows / cols))    # original adjust_lr_fn
theta  <- (1 - lr * weight_decay) * theta
theta  <- theta - lr * update
```

Our neural-network Muon paths now follow this:

- Muon is applied only to 2D hidden matrix parameters by default.
- Non-2D parameters use AdamW-style fallback.
- Momentum and Nesterov use the PyTorch `lerp` convention.
- Newton-Schulz coefficients are `(3.4445, -4.7750, 2.0315)`.
- Default Newton-Schulz steps are `5`.
- Original rectangular shape scaling is used.
- Complex and sparse gradients are rejected.

## Intentional Differences

- `PIMuon` keeps Newton-Schulz in float32 by default for CPU/demo stability.
  PyTorch converts the update to `bfloat16` inside Muon. `PIMuon` exposes
  `ns_use_bfloat16` for experiments.
- The PI optimizers evaluate a same-minibatch trial loss to adapt a global
  multiplier. Official AdamW and Muon are fixed-schedule optimizers and do not
  do this trial evaluation.
- The deterministic 2D function Muon benchmark in `controlled_muon_project`
  remains a Muon-style vector/matrix diagnostic. It is not intended to be a
  drop-in neural-network `torch.optim.Muon` replacement.

## Verification

Passed focused checks:

```bash
python -m py_compile pi_adam_optimizer/pi_adam.py pi_muon_optimizer/pi_muon.py examples/run_pi_fashion_mnist_multiseed.py controlled_muon_project/src/controlled_muon/orthogonalization.py controlled_muon_project/src/controlled_muon/optimizers.py controlled_muon_project/src/controlled_muon/torch_optimizers.py controlled_muon_project/examples/run_mnist_demo.py controlled_muon_project/examples/run_function_benchmark_report.py controlled_muon_project/examples/run_matrix_quadratic_demo.py
pytest -q pi_adam_optimizer/test_pi_adam.py pi_muon_optimizer/test_pi_muon.py
PYTHONPATH=controlled_muon_project/src pytest -q controlled_muon_project/tests
PYTHONPATH=controlled_adam_project/src pytest -q controlled_adam_project/tests
pytest -q tests
```

The tests include one-step comparisons against `torch.optim.AdamW` for PI Adam
and PI Muon AdamW fallback behavior, plus tests for official-style Muon
parameter grouping, momentum updates, quintic Newton-Schulz defaults, and
rectangular shape scaling.

Passed smoke runs:

```bash
python examples/run_pi_fashion_mnist_multiseed.py --output-dir outputs/optimizer_audit_pi_smoke --seeds 101 --optimizers vanilla_adam vanilla_muon pi_adam pi_muon --epochs 1 --train-subset 512 --test-subset 256 --batch-size 128 --alpha0 1e-2 --weight-decay 0.01 --print-every 1
MPLCONFIGDIR=/private/tmp PYTHONPATH=controlled_muon_project/src python controlled_muon_project/examples/run_mnist_demo.py --dataset fashion_mnist --fashion-folder fashion --output-dir outputs/optimizer_audit_controlled_muon_smoke --epochs 1 --train-subset 512 --test-subset 256 --batch-size 128 --lr 1e-2 --weight-decay 0.01 --print-every 1
```

The Fashion-MNIST PI runner keeps `vanilla_adam` as true `torch.optim.Adam`
when `--weight-decay=0`. When nonzero decay is requested, it switches that
baseline to `torch.optim.AdamW` so the fixed Adam-family baseline uses the same
decoupled decay convention as `PIAdam`.
