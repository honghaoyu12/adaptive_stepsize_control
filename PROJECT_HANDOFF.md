# Project Handoff: Adaptive Step-Size Control

Last updated: 2026-05-29

This document is for a future coding agent taking over the workspace. It is intentionally comprehensive and operational: it explains what the project is, what has been implemented, how to run it, what benchmark results are already known, and what caveats matter.

Related project-memory documents:

- `CONVERSATION_LOG.md`: nuanced discussion history and interpretation.
- `DEVELOPMENT_LOG.md`: chronological engineering and benchmark timeline.
- `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`: self-contained deterministic 2D
  function benchmark suite and manager-report guide.
- `OPTIMIZER_VARIANTS_BENCHMARK_REPORT.md`: comprehensive five-variant optimizer comparison and benchmark interpretation.
- `PROJECT_HANDOFF.md`: current-state operating manual.

Recommended reading order for a new coding agent:

1. Read this file first for current state, known caveats, commands, and next steps.
2. Read the root `README.md` for the public-facing project overview.
3. Read the subproject README for the code area being changed.
4. Use `DEVELOPMENT_LOG.md` when you need chronology, exact benchmark context,
   or previously run commands.
5. Use `CONVERSATION_LOG.md` only when the rationale behind a decision is still
   unclear after reading the handoff and development log.

Before making changes, run `git status --short` and preserve untracked generated
or user-provided artifacts. At this handoff, the expected untracked files are
listed in the Current Git State section below.

Supporting documents and how to treat them:

- `RAW_RHO_CONTROLLED_OPTIMIZER_ALGORITHM.md`: algorithm note for raw-rho,
  EMA-rho, and EMA+trust controller mechanics.
- `controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md`: paper-style
  description of the currently implemented minibatch controlled Adam algorithm,
  including same-minibatch trial evaluation, backtracking, rho EMA,
  trust-region expansion, and optional asymmetric `kp_down`.
- `NEXT_BENCHMARK_PLAN.md`: current neural-network benchmark planning note.
- `OPTIMIZER_IMPLEMENTATION_AUDIT.md`: implementation audit for AdamW/Muon
  alignment and intentional differences.
- `CONTROLLED_OPTIMIZER_PERFORMANCE_AND_OVERHEAD.md`: overhead and wall-clock
  interpretation note. Muon neural results inside it are historical unless they
  explicitly mention the official-style Muon grouping fix.
- `controlled_optimizer_viability_analysis.md`: viability analysis built from
  earlier overhead evidence. Treat Muon neural numbers there as historical for
  current baseline-quality claims.
- `pi_adam_optimizer/PI_ADAM_DESIGN_AND_COMPARISON.md` and
  `pi_muon_optimizer/PI_MUON_DESIGN_AND_COMPARISON.md`: standalone PI optimizer
  design notes.
- `delayed_feedback_adam/docs/method_note.md` and
  `delayed_feedback_muon/docs/method_note.md`: low-overhead delayed-feedback
  controller notes. These explain the tradeoff between avoiding the extra
  same-minibatch forward pass and accepting one-step-delayed, noisier feedback.

## High-Level Goal

The project studies outer-loop step-size controllers for optimization. The central idea is:

1. Let a base optimizer choose a search direction.
2. Take a trial step using a scalar global multiplier.
3. Compare actual objective decrease with first-order predicted decrease.
4. Update the global multiplier using that ratio.

For a direction `p_t` and scalar step `alpha_t`:

```math
\Delta \hat f_t = -\alpha_t \nabla f(\theta_t)^T p_t
```

```math
\Delta f_t = f(\theta_t) - f(\theta_t + \alpha_t p_t)
```

```math
\rho_t = \frac{\Delta f_t}{\Delta \hat f_t}
```

The controller updates the global multiplier roughly as:

```math
\alpha_{t+1} = \alpha_t \exp(K_p(\rho_t - \rho^\star))
```

For minibatch neural-network training, the trial loss must be evaluated on the same minibatch used to compute the gradient:

```math
f_{B_t}(\theta_t)
\quad\text{and}\quad
f_{B_t}(\theta_t + \alpha_t p_t)
```

Using a different minibatch would confuse minibatch noise with true optimization progress.

## Repository Layout

The workspace contains seven related project areas:

```text
adaptive_stepsize_control/
├── src/adaptive_stepsize_control/       # root gradient-descent controller demo
├── controlled_adam_project/             # Adam direction + outer controller
├── controlled_muon_project/             # Muon direction + outer controller
├── pi_adam_optimizer/                   # standalone PI-controlled Adam
├── pi_muon_optimizer/                   # standalone PI-controlled Muon
├── delayed_feedback_adam/               # delayed-feedback Adam wrapper
├── delayed_feedback_muon/               # delayed-feedback Muon/AdamW wrapper
├── examples/                            # root project examples
├── tests/                               # root project tests
├── fashion/                             # local Fashion-MNIST IDX gzip files
├── mnist/                               # local MNIST files
├── CONVERSATION_LOG.md                  # nuanced discussion history
├── DEVELOPMENT_LOG.md                   # engineering timeline
└── PROJECT_HANDOFF.md                   # this file
```

The root, controlled Adam, and controlled Muon projects use a `src/` layout.
The PI optimizer and delayed-feedback folders are standalone module/demo
folders. Either install the `src/` projects editable with `pip install -e .`,
or run with `PYTHONPATH=src` where needed. For standalone folders, run tests
from inside the folder with `PYTHONPATH=.`, because running from the monorepo
root can shadow the inner package with the outer directory name.

Generated outputs and local datasets are intentionally ignored by git.

Output cleanup note: existing experiment results were moved out of the active
top-level output locations on 2026-05-26. The archived experiment folders are:

```text
outputs/backup_20260526_182414/
controlled_adam_project/outputs/backup_20260526_182414/
controlled_muon_project/outputs/backup_20260526_182414/
```

New experiments should write directly under `outputs/`,
`controlled_adam_project/outputs/`, or `controlled_muon_project/outputs/` with
fresh descriptive folder names. Treat the backup folders as archival. Some
local smoke/validation outputs, such as PCA trajectory checks, may also appear
under ignored output directories and can be regenerated.

## Moving This Workspace To A More Powerful Computer

Important: a plain `git clone` is not enough unless the current dirty worktree
is committed first. As of 2026-05-29, several important code/documentation
changes and whole subprojects are still uncommitted or untracked. If the user
wants the stronger machine to continue exactly from this state, either commit
the work first or copy the entire working tree, including untracked files,
ignored outputs, and local datasets.

Minimum transfer checklist:

```text
tracked git files
uncommitted modified files
untracked subprojects: delayed_feedback_adam/, delayed_feedback_muon/
untracked controlled Adam sweep/report scripts
root outputs/ with recent CIFAR and function benchmark reports
controlled_adam_project/data/ with CIFAR-10 and MNIST/Fashion-MNIST caches
fashion/ and mnist/ local IDX gzip folders, if present
controlled_adam_project/outputs/ and controlled_muon_project/outputs/, if historical plots/checkpoints matter
root PDF/PNG/CSV artifacts the user may care about
```

Suggested transfer style from the old machine:

```bash
rsync -a \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  /Users/honghaoyu/adaptive_stepsize_control/ \
  USER@NEW_HOST:/path/to/adaptive_stepsize_control/
```

If using git instead of `rsync`, first commit or otherwise package the current
dirty worktree. Then separately copy ignored data/output directories, because
they are intentionally not tracked by git.

Recommended Python setup on the new machine:

```bash
cd /path/to/adaptive_stepsize_control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
python -m pip install -r controlled_adam_project/requirements.txt
python -m pip install -r controlled_muon_project/requirements.txt
python -m pip install -r delayed_feedback_adam/requirements.txt
python -m pip install -r delayed_feedback_muon/requirements.txt
python -m pip install -r pi_adam_optimizer/requirements.txt
python -m pip install -r pi_muon_optimizer/requirements.txt
```

On a CUDA machine, install the appropriate PyTorch/torchvision wheels for that
machine first, then install the remaining requirements. Do not assume the Mac
CPU environment here matches the stronger machine.

Many historical commands used `MPLCONFIGDIR=/private/tmp` because this machine
is macOS. On Linux, use any writable temp directory instead, for example
`MPLCONFIGDIR=/tmp`.

Useful post-transfer checks:

```bash
git status --short
find outputs -maxdepth 1 -type d | sort
test -d controlled_adam_project/data/cifar-10-batches-py && echo "CIFAR present"
```

If CIFAR-10 is missing, most CIFAR runners can recreate it with `--download`.
For offline use, copy `controlled_adam_project/data/cifar-10-batches-py/`.

Recent ignored output folders that matter for the latest Adam discussion:

```text
outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/
outputs/cifar10_resnet_adam_backtracking_sweep_highlr2e3_cap2p5e3_seed123/
outputs/cifar10_resnet_adam_cap_sweep_lr1e3_seed123/
outputs/cifar10_resnet_adam_rhostar_asym_gain_seed123/
outputs/controlled_adam_simplified_tuning_sweep_30runs/
```

Each of those folders contains its own report and CSV/plot artifacts. Copy them
if the new agent should analyze existing results instead of rerunning.

## Current Git State

The latest project update was committed locally in:

```text
12a12df Add PCA trajectory visualization for Adam checkpoints
```

Recent commits before it include:

```text
87af501 Refresh project documentation state
9c55036 Add PI optimizers and align Muon implementation
```

Together these commits added the PI Adam and PI Muon subprojects, aligned the
neural Muon implementations with official `torch.optim.Muon` behavior, updated
the audit and project-memory documents, archived older experimental outputs,
and added the Adam checkpoint PCA trajectory post-processor.

At the time of this handoff, the workspace is not clean. Preserve these local
changes and generated artifacts unless the user explicitly asks to remove or
commit them. Important uncommitted changes include:

```text
README.md
PROJECT_HANDOFF.md
DEVELOPMENT_LOG.md
CONVERSATION_LOG.md
controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md
controlled_adam_project/README.md
controlled_adam_project/src/controlled_adam/torch_optimizers.py
controlled_adam_project/src/controlled_adam/optimizers.py
controlled_adam_project/examples/run_mnist_demo.py
delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py
delayed_feedback_adam/
delayed_feedback_muon/
controlled_adam_project/examples/run_controlled_adam_*_sweep.py
controlled_adam_project/examples/run_function_benchmark_tuned_simplified_report.py
fashionmnist_20epoch_metrics.png
fashionmnist_20epoch_metrics.summary.csv
scheduled_iterate_muon_academic_report.pdf
```

There are also unrelated or earlier modified files in the current worktree,
including `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`,
`controlled_adam_project/examples/run_function_benchmark_report.py`, and
`controlled_muon_project/examples/run_function_benchmark_report.py`. Do not
revert them casually; treat them as user/local work unless the user explicitly
asks for cleanup.

The root `outputs/` directory also contains ignored benchmark outputs,
including the latest controlled-vs-delayed Adam CIFAR-10 ResNet run described
below. These outputs are useful for analysis but are not tracked by git.

Historical CIFAR-10 Adam tuning sweeps recorded in this handoff:

- balanced cap run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.80`
  - `rho_beta=0.90`
  - best controlled raw-rho test accuracy: `0.8232`
- open cap run:
  - `alpha_max=2e-3`
  - `rho_star=0.78`
  - `rho_beta=0.90`
  - best controlled raw-rho test accuracy: `0.8154`

- follow-up balanced run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.78`
  - `rho_beta=0.90`
  - final/best controlled raw-rho test accuracy: `0.8214`

- faster-EMA balanced run:
  - `alpha_max=1.5e-3`
  - `rho_star=0.80`
  - `rho_beta=0.85`
  - raw-rho best test accuracy: `0.8232`
  - EMA best test accuracy: `0.8164`
  - EMA-trust best test accuracy: `0.8114`

Interpretation:

- the controller was too conservative at `1.25e-3`;
- `1.5e-3` appears to be a better operating point for raw-rho on this CIFAR
  subset;
- `2e-3` was too open and did not improve further;
- EMA variants were stable but behind raw-rho on these runs.
- lowering `rho_beta` to `0.85` made EMA more responsive but did not improve
  the final recommendation.

Architecture-transfer follow-up:

- Added `--model lenet_cifar` to the Adam CIFAR runner.
- Ran a 20-epoch 20k/5k CIFAR-10 LeNet-style ablation with the balanced
  controller settings.
- Final test accuracy:

```text
vanilla_adam          0.5868
fixed_adam_direction 0.5868
controlled_raw_rho   0.5656
controlled_ema       0.5620
controlled_ema_trust 0.5722
```

Interpretation:

- LeNet was fast, about `6-7s` per epoch per variant, but the controller
  underperformed the vanilla/fixed baselines.
- This weakens the hypothesis that the current controller setting transfers
  automatically across architectures.
- A CIFAR ResNet test should be staged carefully on this CPU machine: first a
  3-epoch 5k/1k smoke run, then a 10-epoch 10k/2k medium run, and only then a
  larger run if the timing and early signal look useful.
- A Fashion-MNIST CNN test is probably the cheaper next architecture variation.
- We then ran the Fashion-MNIST CNN benchmark. Raw-rho slightly beat fixed
  Adam-direction on final test accuracy (`0.8870` vs `0.8866`), and the run
  was fast enough to be practical on the current CPU (`4.5-5.6s` per epoch per
  variant).
- This makes Fashion-MNIST CNN a good low-cost architecture-transfer check.
- The five-seed Fashion-MNIST CNN follow-up showed that the single-seed raw-rho
  edge did not hold: vanilla/fixed Adam averaged about `0.8945/0.8946` final
  test accuracy, while controlled raw-rho averaged `0.8889`. Controlled variants
  had about `1.22x-1.24x` relative wall-clock time.
- Recommended next Fashion-MNIST CNN parameter candidates:
  - Candidate A: `alpha_min=9e-4`, `alpha_max=1.5e-3`, `rho_star=0.75`,
    `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`.
  - Candidate B: `alpha_min=8e-4`, `alpha_max=1.5e-3`, `rho_star=0.70`,
    `rho_beta=0.90`, `kp=0.03`, factor clip `[0.98, 1.02]`.
  - Candidate C: `alpha_min=9.5e-4`, `alpha_max=1.25e-3`, `rho_star=0.75`,
    `rho_beta=0.90`, `kp=0.01`, factor clip `[0.995, 1.005]`.
- Candidate A was run on three Fashion-MNIST CNN seeds. It improved controlled
  variants by keeping alpha near `8.9e-4`, but still did not clearly beat fixed
  Adam-direction: three-seed final accuracies were vanilla `0.8946`, fixed
  `0.8960`, raw-rho `0.8944`, EMA `0.8945`, and EMA-trust `0.8943`.

- Candidate C was then run on the same three Fashion-MNIST CNN seeds. It kept
  alpha close to `9.3e-4` and lifted raw-rho slightly above the Candidate A
  controlled results, but still did not beat fixed Adam-direction on mean
  final accuracy: vanilla `0.8946`, fixed `0.8960`, raw-rho `0.8951`, EMA
  `0.8937`, and EMA-trust `0.8937`.

- Candidate B was then run with a faster-recovery setting. It underperformed
  both Candidate A and Candidate C: mean final accuracies were vanilla
  `0.8946`, fixed `0.8960`, raw-rho `0.8921`, EMA `0.8930`, and EMA-trust
  `0.8930`. That suggests the more aggressive controller is too jumpy for this
  Fashion-MNIST CNN setup.

- Added `--model resnet_cifar`, implemented as `SmallCIFARResNet` with
  `175258` trainable parameters.
- Ran the staged CIFAR-10 ResNet smoke test: 5k/1k, 3 epochs, seed `123`,
  full ablation, balanced controller (`alpha_min=1e-3`, `alpha_max=1.5e-3`,
  `rho_star=0.80`, `rho_beta=0.90`, `kp=0.02`, factor clip `[0.98, 1.015]`).
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_smoke_5k_1k_3epoch_balanced/`.
- Final test accuracy: vanilla `0.3610`, fixed `0.3730`, raw-rho `0.4100`,
  EMA `0.3800`, EMA-trust `0.3800`.
- The smoke run completed cleanly: all controlled variants accepted every step
  and alpha stayed near `1e-3`.
- Then ran the staged 20-epoch CIFAR-10 ResNet benchmark on 10k train / 2k
  test with the same balanced settings, retaining per-epoch progress prints and
  checkpoints.
- Output directory:
  `controlled_adam_project/outputs/cifar10_resnet_10k_2k_20epoch_balanced/`.
- Final / best test accuracy: vanilla Adam `0.6915 / 0.6915`, fixed
  Adam-direction `0.6875 / 0.6975`, raw-rho `0.7395 / 0.7395`, EMA
  `0.7135 / 0.7135`, EMA-trust `0.7135 / 0.7135`.
- Diagnostics: all fixed/controlled variants accepted every step. Raw-rho,
  EMA, and EMA-trust reached `alpha_max=1.5e-3`; final mean rho was about
  `0.88`. EMA and EMA-trust were identical, so trust-region expansion did not
  materially alter this run.
- Later diagnostic check: for the three balanced CIFAR ResNet seeds
  `123/456/789`, `controlled_ema_trust` recorded `0/1580` trust expansions in
  every run. This happened because the run used `alpha_min=1e-3` but
  `trust_region_alpha_threshold=1e-4`; the trust trigger was below the allowed
  alpha floor, so the branch could not activate. Treat these EMA-trust numbers
  as EMA-rho results with a dormant trust hook, not as evidence that the trust
  expansion rule was tested.
- To test trust-region expansion properly in this Adam-scale CIFAR regime, set
  `trust_region_alpha_threshold` near the alpha floor, such as `1e-3` or
  `1.05e-3`, and use a gentle `trust_region_expand_factor` such as `1.1` or
  `1.2`.
- Follow-up Candidate 1 higher-cap test (`alpha_max=1.75e-3`,
  `rho_star=0.82`, `kp=0.015`) was worse: raw-rho final/best `0.7120`,
  EMA/EMA-trust final `0.6695`, best `0.6915`.
- Follow-up stronger fixed-LR control (`lr=1.5e-3`, `alpha_max=1.5e-3`) made
  vanilla Adam stronger (`0.7065`) and EMA/EMA-trust reached `0.7250`, but
  fixed Adam-direction best was only `0.7040` and raw-rho final/best was
  `0.6985`. This suggests the original raw-rho `0.7395` result was not merely
  caused by using a larger fixed Adam learning rate; ramping from `1e-3` to
  the cap seems important.
- Multi-seed validation of the original balanced setting was then run for
  seeds `123`, `456`, and `789`. Three-seed final accuracy means were:
  vanilla `0.6887`, fixed `0.6938`, raw-rho `0.7150`, EMA `0.7083`, and
  EMA-trust `0.7083`. Best accuracy means were: vanilla `0.6998`, fixed
  `0.7118`, raw-rho `0.7235`, EMA `0.7190`, and EMA-trust `0.7190`.
- Interpretation: controlled variants still show a modest mean advantage on
  the ResNet subset benchmark, but the seed `123` raw-rho result was unusually
  strong. Treat this as encouraging evidence, not a settled claim.
- Next staged ResNet step: either run a reduced-variant 5-seed benchmark
  (`vanilla_adam`, `fixed_adam_direction`, `controlled_raw_rho`,
  `controlled_ema`) or move to a larger train/test subset for the same balanced
  setting.

Controlled-vs-delayed Adam ResNet follow-up:

- Added and used
  `delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py`.
- Purpose: compare same-step controlled Adam and delayed-feedback Adam in one
  CIFAR-10 ResNet runner, with both raw-rho and EMA-style variants present.
- Run setup: CIFAR-10 `resnet_cifar`, 10k train / 2k test, 20 epochs, batch
  size `128`, seed `123`, base LR `1e-3`, checkpoints every 5 epochs.
- Output directory:
  `outputs/cifar10_resnet_adam_delayed_10k_2k_20epoch_seed123_raw_ema/`.
- The output directory contains `REPORT.md`, `cifar10_epoch_metrics.csv`,
  step diagnostics for each controlled/delayed variant, accuracy/loss plots,
  alpha/rho plots, and checkpoints.
- Final metrics:

```text
optimizer                 test_acc  test_loss  train_loss  final_mean_lr  final_mean_rho
vanilla_adam              0.6915    0.9454     0.7499      n/a            n/a
controlled_raw_rho        0.7395    0.7715     0.5830      1.500e-3      0.882
controlled_ema            0.7135    0.8488     0.6656      1.500e-3      0.877
controlled_ema_trust      0.7135    0.8488     0.6656      1.500e-3      0.877
delayed_raw               0.6825    0.9768     0.7820      7.528e-4      0.177
delayed_ema               0.6960    0.9009     0.7334      7.500e-4      0.151
delayed_safe              0.6985    0.9156     0.7194      8.000e-4      0.170
delayed_ema_floor90       0.6950    0.8840     0.6783      9.000e-4      0.133
```

- Best delayed final accuracy was `delayed_safe` at `0.6985`, only slightly
  above vanilla Adam and below both same-step controlled Adam variants.
- `delayed_raw` had best delayed peak accuracy `0.6975` at epoch 19, but lower
  final accuracy `0.6825`.
- Same-step controlled variants accepted every step and did no backtracking.
  Raw-rho and EMA both reached `alpha_max=1.5e-3`.
- `controlled_ema_trust` had zero trust expansions, so it exactly matched
  `controlled_ema`.
- Delayed variants applied their controller on `1579/1580` steps; the first
  step has no previous feedback. All delayed variants were floor-bound for most
  of training.
- Important interpretation: delayed rho is not on the same numerical scale as
  same-step rho in ordinary minibatch CIFAR. Same-step final mean rho was about
  `0.88`, while delayed final mean rho was about `0.13-0.18`. Reusing
  same-step targets such as `rho_star=0.7-0.8` makes the delayed controller
  think steps are poor and shrink alpha. Future delayed tuning should target
  the observed delayed rho scale, for example testing lower `rho_star` values
  around `0.15-0.30`, plus floor/cap sweeps around Adam scale.
- Important fixed-LR discussion: if higher fixed-LR Adam works, that does not
  make controlled Adam bad. It identifies a confound. We need fixed Adam at
  multiple learning rates and simple warmup schedules to separate:
  1. "Adam just wanted a larger LR";
  2. "the controlled ramp behaves like a useful schedule";
  3. "the rho controller is genuinely selecting useful step sizes."
  A strong controller should be less sensitive to the exact upper bound than
  vanilla Adam is to its fixed LR, and should avoid destructive use of an overly
  broad bound.

Latest same-step controlled Adam ResNet controller diagnostics:

- The exact "ResNet" in these benchmarks is `SmallCIFARResNet`, not
  torchvision ResNet-18. It is defined in
  `controlled_adam_project/examples/run_mnist_demo.py`: `3x3` stride-1 CIFAR
  stem, stages of width `16`, `32`, and `64`, two `BasicBlock`s per stage,
  adaptive average pooling, and a `64 -> 10` linear head. Each block is
  `Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm` plus identity or
  projection skip. Parameter count: `175258`.
- The full current controlled Adam algorithm is now documented in
  `controlled_adam_project/CONTROLLED_ADAM_ALGORITHM.md`. That note matches
  the PyTorch implementation, including the detail that if every trial step is
  rejected, parameters are restored but Adam's moment state has still consumed
  the minibatch gradient.
- High-LR/high-cap backtracking sweep:
  `outputs/cifar10_resnet_adam_backtracking_sweep_highlr2e3_cap2p5e3_seed123/`.
  Setup: 10k/2k CIFAR-10, 20 epochs, seed `123`, `lr=2e-3`,
  `alpha_min=1e-3`, `alpha_max=2.5e-3`, `rho_star=0.80`,
  `rho_beta=0.90`, `kp=0.02`. The large alpha cliffs in the old plot were
  caused by backtracking, especially `backtrack_shrink=0.5`. Gentler
  backtracking smoothed the curve but did not beat the lower-cap balanced
  setting. Best new backtracking compromise was `backtrack_shrink=0.7`,
  `rho_min=0.1`: raw-rho final `0.7030`, EMA final `0.7090`.
- Clean cap sweep:
  `outputs/cifar10_resnet_adam_cap_sweep_lr1e3_seed123/`. Setup: same
  10k/2k ResNet benchmark, `lr=1e-3`, `alpha_min=1e-3`, `rho_star=0.80`,
  `kp=0.02`, cap-only changes. Final test accuracy:

```text
alpha_max   controlled_raw_rho   controlled_ema
1.50e-3     0.7395               0.7135
1.75e-3     0.7395               0.7240
2.00e-3     0.7290               0.7245
2.25e-3     0.7155               0.7355
```

  Both controlled variants eventually reached every tested cap. Raw-rho was
  best around `1.5e-3` to `1.75e-3`; EMA liked more cap on this single seed.
  This means cap saturation is not intrinsically bad, but the current
  same-step rho target does not discover an interior alpha optimum by itself.
- Added optional asymmetric gain support to controlled Adam:
  `TorchControlledAdam(..., kp_down=...)` and
  `--controlled-kp-down` in the CIFAR comparison CLI. If omitted,
  `kp_down=kp`, so old behavior is unchanged. The controller now uses `kp` for
  `rho_control >= rho_star` and `kp_down` for `rho_control < rho_star`.
- Raised-target / asymmetric-gain test:
  `outputs/cifar10_resnet_adam_rhostar_asym_gain_seed123/`. With
  `alpha_max=2.25e-3`, `rho_star=0.85` prevented cap saturation but was too
  conservative: raw-rho final `0.7115`, EMA final `0.6980`. With
  `rho_star=0.80`, `kp_down=0.08`, saturation was delayed but not removed:
  raw-rho final `0.6950`, EMA final `0.7330`. The most sensible next setting
  is the middle ground `rho_star=0.825`, `kp_down=0.04` or `0.06`,
  `alpha_max=2.25e-3`.

There are also user-provided/untracked comparison/report artifacts at repo root:

- `fashionmnist_20epoch_metrics.summary.csv`
- `fashionmnist_20epoch_metrics.png`
- `scheduled_iterate_muon_academic_report.pdf`

Do not delete user-added files unless explicitly asked.

## PI Optimizer Subprojects

Paths:

```text
pi_adam_optimizer/
pi_muon_optimizer/
```

Purpose: standalone PyTorch optimizer versions of the controlled Adam and Muon
ideas with a PI controller instead of the older proportional-only controller.
The PI optimizers preserve the same-batch actual-vs-predicted loss signal and
add a leaky, clipped integral term:

```text
log(alpha_next) = log(alpha_used) + kp * (rho_bar - rho_star) + ki * integral
```

Current implementation state:

- `PIAdam` uses Adam's bias-corrected direction plus a global PI-controlled
  multiplier.
- `PIMuon` uses official-style Muon for 2D hidden matrix parameters and
  AdamW-style fallback directions for all other parameters.
- Both PI optimizers support optional rho EMA smoothing, bad-step rejection,
  bounded backtracking, non-descent fallback, trust-region expansion, and
  decoupled AdamW/Muon-style weight decay.
- The PI Fashion-MNIST runner also has corrected vanilla baselines:
  `vanilla_adam` uses Adam or AdamW depending on weight decay, and
  `vanilla_muon` uses official-style Muon with AdamW fallback.

Important files:

- `pi_adam_optimizer/pi_adam.py`
- `pi_adam_optimizer/README.md`
- `pi_adam_optimizer/PI_ADAM_DESIGN_AND_COMPARISON.md`
- `pi_muon_optimizer/pi_muon.py`
- `pi_muon_optimizer/README.md`
- `pi_muon_optimizer/PI_MUON_DESIGN_AND_COMPARISON.md`
- `examples/run_pi_fashion_mnist_multiseed.py`
- `examples/plot_pi_fashion_mnist_results.py`

Verification already run:

```bash
pytest -q pi_adam_optimizer/test_pi_adam.py pi_muon_optimizer/test_pi_muon.py
python examples/run_pi_fashion_mnist_multiseed.py --output-dir outputs/optimizer_audit_pi_smoke --seeds 101 --optimizers vanilla_adam vanilla_muon pi_adam pi_muon --epochs 1 --train-subset 512 --test-subset 256 --batch-size 128 --alpha0 1e-2 --weight-decay 0.01 --print-every 1
```

Known result:

```text
15 passed
```

## Delayed-Feedback Optimizer Subprojects

Paths:

```text
delayed_feedback_adam/
delayed_feedback_muon/
```

Purpose: standalone PyTorch optimizer experiments for the lower-overhead
controller idea. Instead of evaluating `f(theta + alpha p)` immediately after
a trial step, these optimizers use the next loss value already computed by the
training loop to estimate the previous step's actual-versus-predicted decrease:

```text
rho_{t-1} = (loss_{t-1} - loss_t) / predicted_decrease_{t-1}
```

This avoids the extra same-minibatch forward pass used by the controlled Adam,
controlled Muon, and PI optimizers. The tradeoff is that the feedback is delayed
by one step, cannot reject a bad step before it happens, and is noisier under
ordinary minibatch shuffling because consecutive losses usually come from
different minibatches.

Current implementation state:

- `DelayedFeedbackAdam` wraps an Adam/AdamW-style direction with delayed P/PI/PID
  alpha control.
- `DelayedFeedbackMuon` wraps official-style 2D Muon directions with auxiliary
  AdamW fallback for non-2D parameters.
- On 2026-05-27, the delayed implementations were checked against the local
  PyTorch `2.10.0` source. AdamW behavior matched the supported simple path.
  Muon was aligned to `torch.optim.Muon` conventions:
  - momentum buffer update uses `buf.lerp_(grad, 1 - momentum)`;
  - Nesterov update uses `grad.lerp(buf, momentum)`;
  - automatic Muon selection is 2D-only;
  - `adjust_lr_fn=None` and `"original"` both use PyTorch's original shape
    scaling;
  - decoupled weight decay uses the base learning rate, not the shape-adjusted
    Muon learning rate.

Important files:

- `delayed_feedback_adam/delayed_feedback_adam/optimizer.py`
- `delayed_feedback_adam/README.md`
- `delayed_feedback_adam/docs/method_note.md`
- `delayed_feedback_muon/delayed_feedback_muon/optimizer.py`
- `delayed_feedback_muon/README.md`
- `delayed_feedback_muon/docs/method_note.md`

Verification already run:

```bash
cd delayed_feedback_adam
PYTHONPATH=. python -m pytest -q tests
# 4 passed

cd ../delayed_feedback_muon
PYTHONPATH=. python -m pytest -q tests
# 6 passed
```

## Root Project

Path: repo root.

Purpose: demonstrate fixed gradient descent, stochastic gradient descent, and controlled gradient descent on deterministic 2D functions.

Important files:

- `src/adaptive_stepsize_control/objectives.py`
- `src/adaptive_stepsize_control/optimizers.py`
- `src/adaptive_stepsize_control/plotting.py`
- `examples/run_quadratic_demo.py`
- `examples/run_benchmark_functions.py`
- `tests/test_quadratic_demo.py`

Implemented objectives include:

- quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale

Useful commands:

```bash
PYTHONPATH=src pytest -q
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_quadratic_demo.py
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_benchmark_functions.py
```

Known behavior:

- Plots compare fixed GD, SGD, and controlled GD.
- Step-size plots include both fixed and adaptive curves.
- Trajectory plots overlay optimizer paths on objective landscapes.
- Optimizer histories include the initial point, so all methods start visibly from the same location.

## Controlled Adam Subproject

Path:

```text
controlled_adam_project/
```

Purpose: compare vanilla Adam with Adam-direction variants controlled by the actual-over-predicted decrease ratio.

Important files:

- `controlled_adam_project/src/controlled_adam/objectives.py`
- `controlled_adam_project/src/controlled_adam/optimizers.py`
- `controlled_adam_project/src/controlled_adam/torch_optimizers.py`
- `controlled_adam_project/src/controlled_adam/plotting.py`
- `controlled_adam_project/examples/run_demo.py`
- `controlled_adam_project/examples/run_function_benchmark_report.py`
- `controlled_adam_project/examples/run_mnist_demo.py`
- `controlled_adam_project/tests/test_optimizers.py`
- `controlled_adam_project/tests/test_torch_optimizers.py`
- `controlled_adam_project/README.md`

### Adam Optimizer Variants

Toy deterministic code:

- `vanilla_adam`
- `controlled_adam`

PyTorch minibatch code:

- `TorchControlledAdam`

The PyTorch runner supports:

- `vanilla_adam`
- `fixed_adam_direction`
- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

`fixed_adam_direction` is important because it isolates whether the Adam-direction implementation itself is healthy. It uses the same Adam-direction machinery but keeps alpha fixed.

### Controlled Adam Features

`TorchControlledAdam` supports:

- same-minibatch trial loss evaluation
- bad-step rejection
- backtracking
- non-descent shrink
- EMA-smoothed rho control
- clipped alpha update factors
- trust-region style alpha recovery
- diagnostics: `alpha_next`, `alpha_update_factor`, `trust_region_expanded`

Trust-region recovery rule:

- If a step is accepted without backtracking,
- and the smoothed rho is high,
- and alpha is tiny,
- then force a larger expansion factor.

This was added because alpha sometimes collapsed to very small values while rho became high, which means the local model was too conservative rather than the optimizer being done.

The "alpha is tiny" threshold must be chosen relative to the configured alpha
floor. If `trust_region_alpha_threshold < alpha_min`, then the trust branch is
effectively unreachable. This exact mismatch occurred in the balanced CIFAR
ResNet Adam runs: `alpha_min=1e-3` and `trust_region_alpha_threshold=1e-4`.

### Adam Objectives

The 2D deterministic benchmark suite includes:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

### Adam Commands

Run tests:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Run function benchmarks:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_demo.py
```

Run the self-contained deterministic function benchmark report:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

This report runner compares vanilla Adam, controlled raw-rho Adam, controlled
EMA-rho Adam, and controlled EMA+trust Adam on nine 2D objectives with five
fixed starts each. It writes CSVs, trajectory plots over landscapes, residual
curves, alpha curves, and a standalone Markdown report at
`controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md`.
The tracked top-level guide is `FUNCTION_OPTIMIZATION_BENCHMARK_SUITE.md`.

For the trimmed manager version:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

The trimmed Adam report currently shows controlled variants beating vanilla on
the chosen residual comparisons, but `controlled_ema_trust` mostly overlaps
`controlled_ema_rho` because the tiny-alpha trust expansion hook does not fire
on those three selected functions.

Recent function-report follow-ups:

- Added `--step-multiplier`, `--random-starts-per-objective`, and
  `--random-seed` to both Adam and Muon function report runners.
- Generated 10x three-function reports:
  - `controlled_adam_project/outputs/function_report_manager_trimmed_10x/`
  - `controlled_muon_project/outputs/function_report_manager_trimmed_10x/`
- Generated six-function 10x reports including Rosenbrock, Himmelblau, and
  Rastrigin:
  - `controlled_adam_project/outputs/function_report_manager_extended_10x/`
  - `controlled_muon_project/outputs/function_report_manager_extended_10x/`
- Generated a broader 15-start Adam aggregate:
  `controlled_adam_project/outputs/function_report_manager_extended_10x_15starts/`.
- Generated larger Adam aggregates:
  - `controlled_adam_project/outputs/function_report_manager_extended_10x_30starts/`
  - `controlled_adam_project/outputs/function_report_manager_extended_10x_60starts/`
  - `controlled_adam_project/outputs/function_report_manager_extended_60starts_default_steps/`

Controlled Adam tuning sweeps:

- Added tuning scripts:
  - `controlled_adam_project/examples/run_controlled_adam_parameter_sweep.py`
  - `controlled_adam_project/examples/run_controlled_adam_tuning_sweep.py`
  - `controlled_adam_project/examples/run_controlled_adam_refined_tuning_sweep.py`
  - `controlled_adam_project/examples/run_controlled_adam_simplified_tuning_sweep.py`
- Generated broad and refined 30-start outputs under the root output directory:
  - `outputs/controlled_adam_parameter_sweep_30runs/`
  - `outputs/controlled_adam_tuning_sweep_30runs/`
  - `outputs/controlled_adam_refined_tuning_sweep_30runs/`
  - `outputs/controlled_adam_simplified_tuning_sweep_30runs/`
- The refined sweep command was:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_controlled_adam_refined_tuning_sweep.py \
  --output-dir /Users/honghaoyu/adaptive_stepsize_control/outputs/controlled_adam_refined_tuning_sweep_30runs \
  --random-starts-per-objective 25 \
  --random-seed 20260527
```

- Validation: `per_start_results.csv` has `33,210` rows, exactly
  `9 objectives x 123 variants x 30 starts`, and every objective/variant group
  has 30 starts.
- Refined result summary:

```text
raw-rho current 0.370 -> tuned 0.437
EMA-rho current 0.359 -> tuned 0.444
EMA+trust current 0.359 -> tuned 0.489
```

- Best raw-rho setting: `rho_star - 0.2`, `kp x2`, `alpha_min=1e-5`.
- Best EMA-rho setting: `rho_star - 0.2`, `kp x2`, `alpha_min=1e-5`.
- Best EMA+trust setting by success/residual tie-break:
  `rho_beta=0.95`, `rho>=0.70`, `alpha<=1e-2`, expand `x2`,
  `alpha_min=1e-5`.
- Interpretation: all three controlled families can improve on the
  deterministic suite. Raw/EMA mainly benefit from avoiding collapse to
  `alpha_min=1e-8`; tuned trust improves most because its expansion gate
  finally fires, with median trust expansions around `12` across objective
  rows. This is function-suite evidence, not a reason to change neural defaults
  without an Adam-scale validation run.

Simplified preset sweep:

- Motivation: reduce the apparent optimizer tuning burden. Instead of exposing
  many raw knobs, test a preset interface:
  - `family`: raw-rho, EMA-rho, or EMA+trust;
  - `response_preset`: conservative, balanced, or aggressive;
  - `alpha_preset`: low_floor, mid_floor, wide_cap, or high_floor.
- Command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_controlled_adam_simplified_tuning_sweep.py \
  --output-dir /Users/honghaoyu/adaptive_stepsize_control/outputs/controlled_adam_simplified_tuning_sweep_30runs \
  --random-starts-per-objective 25 \
  --random-seed 20260527
```

- Validation: `per_start_results.csv` has `10,530` rows, exactly
  `9 objectives x 39 variants x 30 starts`, and every objective/variant group
  has 30 starts.
- Output summary:
  `outputs/controlled_adam_simplified_tuning_sweep_30runs/CONTROLLED_ADAM_SIMPLIFIED_TUNING_SWEEP.md`.
- Best simplified result:

```text
raw-rho current 0.370 -> simplified 0.470
EMA-rho current 0.359 -> simplified 0.485
EMA+trust current 0.359 -> simplified 0.504
```

- The common winning preset was `aggressive_high_floor`:

```text
kp_multiplier = 2
rho_star_delta = -0.2
rho_beta = 0.90
trust_region_rho_threshold = 0.60
trust_region_expand_factor = 3
alpha_min = 0.01 * alpha0
alpha_max = 50 * alpha0
trust_region_alpha_threshold = 3 * alpha0
```

- Interpretation: the simplified preset recovered and slightly exceeded the
  earlier refined-grid results on deterministic functions, while reducing the
  user-facing interface to three choices. The improvement is a coupled preset
  effect: higher floor, wider cap, lower rho target, stronger gain, and
  reachable trust expansion. Validate on neural tasks before making it a
  default.

Tuned no-momentum function benchmark:

- Added `controlled_adam_project/examples/run_function_benchmark_tuned_simplified_report.py`.
- This reruns the deterministic function benchmark with the
  `aggressive_high_floor` controlled parameters and omits SGD with momentum.
- Output:
  `outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum/`.
- Validation: `per_start_results.csv` has `1,350` rows, exactly
  `9 objectives x 5 optimizers x 30 starts`. The optimizer set is
  `gradient_descent`, `vanilla_adam`, `controlled_raw_rho`,
  `controlled_ema_rho`, and `controlled_ema_trust`.
- Headline averages:

```text
gradient_descent        success 0.278  mean log10 best residual -2.354
vanilla_adam            success 0.307  mean log10 best residual -2.215
tuned raw-rho           success 0.470  mean log10 best residual -3.384
tuned EMA-rho           success 0.485  mean log10 best residual -3.246
tuned EMA+trust         success 0.504  mean log10 best residual -3.499
```

- Interpretation: tuned EMA+trust is strongest in this no-momentum function
  report, and trust expansion is active with median trust expansions around
  `16` across objective rows. The tuned controlled variants most clearly beat
  vanilla Adam on Beale, Quadratic, and Rosenbrock. Ackley and Rastrigin remain
  basin-selection limitation cases.

### GD Baseline Learning-Rate Sensitivity

- Goldstein-Price exposed that the original fixed GD baseline was using an
  Adam-scale learning rate as a raw-gradient learning rate. With
  `alpha = alpha0 = 0.003`, the first Goldstein-Price GD step is often enormous,
  so `best_iteration = 0` for `29/30` starts in the original tuned report.
- `controlled_adam_project/examples/run_function_benchmark_tuned_simplified_report.py`
  now accepts:

```bash
--gradient-descent-alpha-multiplier VALUE
```

- The multiplier applies only to `gradient_descent`; vanilla Adam and all
  controlled Adam variants still use the original `alpha0`.
- New full reports:

```text
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p03/
outputs/function_benchmark_30runs_controlled_adam_tuned_no_momentum_gd_lr0p05/
```

- Both have `1,350` per-start rows and `45` aggregate rows. Non-GD aggregate
  metrics are unchanged from the original tuned no-momentum report.
- GD comparison:

```text
GD alpha       avg success  mean log10 best  Goldstein success  Goldstein best  Goldstein final
1.0 * alpha0   0.278        -2.354           0.033              1512.07         2.64e20
0.03 * alpha0  0.167        -0.255           0.367              27.0            27.0
0.05 * alpha0  0.174        -0.784           0.167              1072.09         6.88e14
```

- Practical interpretation: use `0.03 * alpha0` when you want a
  Goldstein-stable fixed-GD reference. Use `0.05 * alpha0` only as a larger-LR
  stress point. The original `1.0 * alpha0` remains a useful reminder that one
  fixed raw-gradient LR can look good on some functions and fail badly on
  others.

Interpretation for manager-facing function optimization:

- Controlled Adam is strongest as an early/local progress and step-size
  robustness story, not as a guarantee of best eventual residual after very
  long runs.
- The 60-start default-step run is the best manager-facing evidence for fixed
  practical budgets. Within the default steps, controlled variants outperform
  vanilla success rate on Beale, Goldstein-Price, Rosenbrock, Himmelblau, and
  Quadratic.
- Himmelblau is the cleanest extra example: all optimizers succeed, but
  controlled Adam reaches the success criterion much faster. In the 60-start
  default run, controlled variants reach success around `84-88` iterations,
  while vanilla takes about `291`.
- Rosenbrock shows controlled Adam can succeed faster while vanilla Adam can
  eventually catch up or exceed success with enough iterations. In the 60-start
  default run, controlled success is `28-32%` versus vanilla `8%`; in the
  60-start 10x run, vanilla reaches `100%` but takes about `8085` median
  successful iterations versus controlled variants around `3954-5060`.
- Beale and Goldstein-Price become more nuanced under 60-start aggregation:
  controlled variants are clearly better under the default budget, but vanilla
  catches up more under the 10x budget.
- Rastrigin is a limitation case: local step-size control does not solve global
  basin selection.

Focused Rastrigin basin benchmark:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_rastrigin_basin_benchmark.py \
  --output-dir outputs/rastrigin_basin_benchmark_30starts \
  --starts-per-radius 30 \
  --steps 12000
```

This benchmark samples starts from boxes around `(0, 0)` and writes:

```text
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/RASTRIGIN_BASIN_BENCHMARK_REPORT_ZH.md
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/aggregate_results.csv
controlled_adam_project/outputs/rastrigin_basin_benchmark_30starts/rastrigin_success_rate_by_radius.png
```

Main result: all methods succeed reliably up to radius `0.5`; success falls to
about `57%` at radius `0.75`, `23%` at radius `1.0`, and `0%` at radius `4.0`.
Controlled Adam usually reaches success faster inside the correct basin, but it
does not expand the global basin enough to solve far-away Rastrigin starts.

Run Fashion-MNIST ablation using local IDX files:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
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

Run CIFAR-10 with tuned cap:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-alpha-max 1.5e-3 \
  --controlled-max-alpha-factor 1.05 \
  --controlled-rho-star 0.6 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 1.5 \
  --print-every 1 \
  --checkpoint-every 1 \
  --output-dir outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

### Controlled Adam Neural Benchmark Setup

Datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10

Models:

- `SmallMLP` for MNIST/Fashion-MNIST:
  - Flatten
  - Linear `28*28 -> 128`
  - ReLU
  - Linear `128 -> 10`
- `SmallCIFARCNN` for CIFAR-10:
  - Conv-BN-ReLU blocks:
    - `3 -> 32 -> 32`
    - `32 -> 64 -> 64`
    - `64 -> 128 -> 128`
  - MaxPool after each block
  - Linear `128*4*4 -> 256 -> 10`
  - BatchNorm uses `track_running_stats=False` so same-minibatch trial loss does not mutate running statistics.
  - About `815,018` trainable parameters.

CIFAR transforms:

- train: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `ToTensor`, Normalize
- eval: `ToTensor`, Normalize
- train metrics use deterministic eval transform on the same subset indices

The runner writes:

- `run_metadata.json`
- `run_metadata.txt`
- epoch metrics CSV with train loss, train accuracy, test loss, test accuracy, cumulative wall-clock seconds, and cumulative optimizer steps
- step diagnostics CSVs
- test loss plot
- train loss plot
- train-vs-test loss plot
- accuracy plot
- loss/accuracy plots versus optimizer steps
- loss/accuracy plots versus wall-clock time
- controlled alpha plot
- controlled rho plot
- optional per-epoch checkpoints when `--checkpoint-every N` is set

The Adam image runner also supports `--print-every N` for live epoch summaries.

Checkpointed Adam image runs can be post-processed with:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/plot_pca_training_trajectory.py \
  outputs/YOUR_RUN_WITH_CHECKPOINTS \
  --runs vanilla_adam fixed_adam_direction controlled_raw_rho controlled_ema \
  --output-dir outputs/YOUR_RUN_WITH_CHECKPOINTS/pca_trajectory
```

This implements the training-trajectory PCA view used in loss-landscape
visualization work. It flattens trainable parameters from per-epoch checkpoints,
fits a 2D PCA plane to checkpoint displacements, and writes:

- `pca_trajectory_coordinates.csv`
- `pca_explained_variance.csv`
- `pca_training_trajectory.png`

By default, the origin is the final checkpoint of the first selected run. Use
`--reference-run` or `--reference-epoch` to choose a different origin, and use
`--no-center-for-pca` to fit PCA directly on final-relative displacements.

### Important Adam Results

Fashion-MNIST comparison to another project:

- Other project files at repo root:
  - `fashionmnist_20epoch_metrics.summary.csv`
  - `fashionmnist_20epoch_metrics.png`
- Other project Fashion-MNIST Adam:
  - train accuracy about `91.80%`
  - validation accuracy about `83.40%`
- Our comparable 20-epoch Fashion-MNIST Adam:
  - train accuracy about `89.97%`
  - test accuracy about `83.69%`
- Conclusion: our Adam baseline is comparable; the apparent 90% number was training accuracy, not validation/test.

Best controlled Adam CIFAR run so far:

Output:

```text
controlled_adam_project/outputs/cifar10_stronger_cnn_40epochs_ablation_alpha_cap_1p5e3
```

Final test accuracy:

```text
vanilla_adam          0.718
fixed_adam_direction 0.717
controlled_raw_rho   0.711
controlled_ema       0.704
controlled_ema_trust 0.731
```

Best test accuracy:

```text
vanilla_adam          0.720 at epoch 34
fixed_adam_direction 0.717 at epoch 40
controlled_raw_rho   0.735 at epoch 38
controlled_ema       0.709 at epoch 33
controlled_ema_trust 0.731 at epoch 40
```

Interpretation:

- Tight alpha cap around Adam scale is important.
- `controlled_ema_trust` achieved the best final CIFAR result so far.
- `controlled_raw_rho` achieved the best peak accuracy.
- Trust expansion rarely fired in this CIFAR run; the benefit mostly came from keeping alpha bounded near a useful scale.

Larger Adam CIFAR-10 run:

Output:

```text
controlled_adam_project/outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Command:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --epochs 40 \
  --train-subset 20000 \
  --test-subset 5000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --controlled-alpha-min 5e-4 \
  --controlled-alpha-max 1.5e-3 \
  --controlled-max-alpha-factor 1.05 \
  --controlled-rho-star 0.6 \
  --controlled-trust-alpha-threshold 1e-3 \
  --controlled-trust-expand-factor 1.5 \
  --checkpoint-every 1 \
  --print-every 1 \
  --output-dir outputs/cifar10_20k_5k_40epochs_ablation_progress
```

Final test accuracy:

```text
vanilla_adam          0.8288
fixed_adam_direction 0.8280
controlled_raw_rho   0.8232
controlled_ema       0.8278
controlled_ema_trust 0.8278
```

Best test accuracy:

```text
vanilla_adam          0.8344 at epoch 38
fixed_adam_direction 0.8402 at epoch 39
controlled_raw_rho   0.8292 at epoch 38
controlled_ema       0.8324 at epoch 37
controlled_ema_trust 0.8324 at epoch 37
```

Interpretation:

- This 20k/5k result is the most informative Adam CIFAR result so far.
- Larger-data CIFAR accuracy is much more plausible than the earlier 5000/1000 subset numbers.
- Fixed Adam-direction had the best peak test accuracy.
- Controlled variants all accepted 100% of steps and quickly saturated the `1.5e-3` alpha cap.
- EMA and EMA-trust were identical in this run; trust-region behavior was not meaningfully distinguished.

## Controlled Muon Subproject

Path:

```text
controlled_muon_project/
```

Purpose: Muon version of `controlled_adam_project`. Muon supplies an orthogonalized matrix-shaped direction; the outer controller adapts the scalar global multiplier.

Important files:

- `controlled_muon_project/src/controlled_muon/objectives.py`
- `controlled_muon_project/src/controlled_muon/optimizers.py`
- `controlled_muon_project/src/controlled_muon/orthogonalization.py`
- `controlled_muon_project/src/controlled_muon/torch_optimizers.py`
- `controlled_muon_project/src/controlled_muon/plotting.py`
- `controlled_muon_project/examples/run_matrix_quadratic_demo.py`
- `controlled_muon_project/examples/run_mnist_demo.py`
- `controlled_muon_project/tests/test_objectives.py`
- `controlled_muon_project/tests/test_optimizers.py`
- `controlled_muon_project/README.md`

### Muon Core

There are two Muon implementations:

1. NumPy toy optimizer in `optimizers.py`
   - Used for deterministic matrix quadratic and 2D function demos.
   - Original matrix quadratic class remains for backward compatibility.

2. PyTorch minibatch optimizer in `torch_optimizers.py`
   - Used for Fashion-MNIST/CIFAR benchmarks.
   - Implements Muon-style direction with outer-loop controller.

Muon direction:

```text
B_t = lerp(B_{t-1}, G_t, 1 - momentum)
U_t = lerp(G_t, B_t, momentum)     # if Nesterov
O_t = NewtonSchulz_quintic(U_t)
P_t = -shape_scale * update_scale * O_t
```

Orthogonalization methods:

- exact SVD polar factor
- Muon-style quintic Newton-Schulz iteration with PyTorch's default
  coefficients `(3.4445, -4.7750, 2.0315)` and 5 default steps

PyTorch tensor handling:

- the neural-network Muon path follows `torch.optim.Muon` scope
- only 2D hidden matrix parameters use Muon by default
- vectors, scalars, norms, biases, embeddings, heads, and convolution kernels
  use AdamW-style fallback updates
- nonzero weight decay is decoupled, as in AdamW/Muon

This is educational and CPU-heavy, not a production Muon implementation. The
CIFAR runs are slower than Adam because each controlled variant performs
orthogonalization and same-minibatch trial loss evaluations. The deterministic
2D function runner still uses a vector analogue of Muon and should not be read
as a full neural-network `torch.optim.Muon` replacement.

Important benchmark note: older neural Muon tables in this handoff predate the
official-style parameter grouping fix unless an output folder explicitly says
`official_muon`. Treat those old all-parameter neural Muon results as archival
context. Future neural Muon comparisons should use only the corrected
official-style path: Muon for 2D hidden matrix parameters and AdamW-style
fallback for the rest.

### Muon Optimizer Variants

The image runner supports:

- `vanilla_muon`
- `fixed_muon_direction`
- `controlled_raw_rho`
- `controlled_ema`
- `controlled_ema_trust`

`vanilla_muon` and `fixed_muon_direction` should be almost identical in behavior when alpha is fixed. They are both useful sanity checks.

### Muon Objectives

The Muon subproject now supports the same 2D benchmark objectives as Adam:

- Anisotropic quadratic
- Rosenbrock
- Himmelblau
- Rastrigin
- Beale
- Ackley
- Six-hump camel
- Goldstein-Price
- Easom

It also retains:

- `MatrixQuadraticObjective`

### Muon Commands

Run tests:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Known latest test result:

```text
12 passed
```

Run function/objective demo:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_matrix_quadratic_demo.py
```

Run deterministic 2D function benchmark report:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

This report compares `vanilla_muon`, `controlled_raw_rho`, `controlled_ema`,
and `controlled_ema_trust` on the same nine 2D objectives used by the Adam
report. It writes
`controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md`.
The shorter manager-facing version is
`controlled_muon_project/outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md`.
Both Muon function reports include Chinese companion reports and standalone
`*_surface_3d.png` plots with the objective formula printed inside each figure.
The earlier `fixed_muon_direction` diagnostic was removed from the function
report because it duplicated `vanilla_muon` in this local 2D vector runner:
both used fixed alpha and no rho controller.
For these 2D vector objectives, Muon is implemented as a vector analogue by
treating the momentum vector as a column matrix before orthogonalization. Do
not overclaim it as a full matrix Muon benchmark.

Run Fashion-MNIST Muon ablation:

```bash
cd controlled_muon_project
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

Run CIFAR-10 Muon 10-epoch ablation:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_mnist_demo.py \
  --dataset cifar10 \
  --data-dir ../controlled_adam_project/data \
  --epochs 10 \
  --train-subset 5000 \
  --test-subset 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --ablation \
  --output-dir outputs/cifar10_muon_10epoch_ablation
```

Run CIFAR-10 Muon 40-epoch ablation:

```bash
cd controlled_muon_project
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

Warning: the historical 40-epoch Muon CIFAR run was slow and originally silent
until completion. The current runner now supports `--print-every` per-epoch
progress output, but long Muon jobs are still CPU-heavy because
orthogonalization uses the educational NumPy/CPU path.

### Muon Benchmark Results

Fashion-MNIST, 20 epochs, 4096 train / 1024 test:

Output:

```text
controlled_muon_project/outputs/fashion_mnist_muon_20epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.7432  best 0.7432 at epoch 20
fixed_muon_direction final 0.7432  best 0.7432 at epoch 20
controlled_raw_rho   final 0.8320  best 0.8418 at epoch 14
controlled_ema       final 0.8301  best 0.8389 at epoch 18
controlled_ema_trust final 0.8301  best 0.8389 at epoch 18
```

Interpretation:

- Controlled Muon is clearly better than fixed/vanilla Muon on this subset.
- Raw rho had the best peak and final accuracy in this run.
- EMA and EMA-trust were identical because trust expansion did not materially change behavior.

CIFAR-10, 10 epochs, 5000 train / 1000 test:

Output:

```text
controlled_muon_project/outputs/cifar10_muon_10epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.578  best 0.578 at epoch 10
fixed_muon_direction final 0.582  best 0.582 at epoch 10
controlled_raw_rho   final 0.665  best 0.665 at epoch 10
controlled_ema       final 0.644  best 0.661 at epoch 9
controlled_ema_trust final 0.644  best 0.661 at epoch 9
```

CIFAR-10, 40 epochs, 5000 train / 1000 test:

Output:

```text
controlled_muon_project/outputs/cifar10_muon_40epoch_ablation
```

Final and best test accuracy:

```text
vanilla_muon          final 0.699  best 0.709 at epoch 31
fixed_muon_direction final 0.694  best 0.701 at epoch 38
controlled_raw_rho   final 0.725  best 0.731 at epoch 27
controlled_ema       final 0.725  best 0.725 at epoch 40
controlled_ema_trust final 0.725  best 0.725 at epoch 40
```

Final train accuracy:

```text
vanilla_muon          0.8842
fixed_muon_direction 0.8824
controlled_raw_rho   0.8732
controlled_ema       0.8988
controlled_ema_trust 0.8988
```

Step diagnostics:

```text
controlled_raw_rho:   1600 steps, alpha_final ~0.04965, alpha_max 0.05, accepted 100%
controlled_ema:       1600 steps, alpha_final ~0.04507, alpha_max 0.05, accepted 100%
controlled_ema_trust: 1600 steps, alpha_final ~0.04507, alpha_max 0.05, accepted 100%
fixed_muon_direction: 1600 steps, alpha fixed 0.001, accepted 100%
```

Interpretation:

- Controlled Muon beats vanilla/fixed Muon on CIFAR-10 subset runs.
- Raw rho achieved the best peak accuracy.
- EMA and EMA-trust achieved the best final tie.
- Trust-region expansion did not fire in the 40-epoch CIFAR Muon run.
- Alpha for controlled Muon grew close to the default cap `0.05`.

## Dataset Notes

Local Fashion-MNIST:

```text
fashion/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

These were validated as 60,000 train images/labels and 10,000 test images/labels.

Local CIFAR-10:

```text
controlled_adam_project/data/cifar-10-batches-py/
```

The Muon project currently reuses that path via:

```bash
--data-dir ../controlled_adam_project/data
```

On another machine, either copy this folder or run the Adam/Muon runner with `--download` if network access works.

## Known Caveats And Risks

1. Same-minibatch trial loss is mandatory.
   Do not change controlled Adam/Muon to evaluate `f(theta_{t+1})` on a fresh minibatch.
   The controlled variants intentionally add one extra forward loss evaluation
   on that same minibatch, but not an extra backward pass. Treat overhead claims
   through loss/accuracy versus wall-clock time, because the practical cost
   depends on the model, hardware, and data pipeline.

2. CIFAR Muon is slow.
   The current PyTorch Muon implementation orthogonalizes tensors through
   NumPy/CPU. It is acceptable for research demos but not efficient. Use
   `--print-every` for live progress, and add checkpointing or incremental CSV
   flushing before running more long Muon jobs.

3. Muon dependency metadata was recently updated.
   `controlled_muon_project/pyproject.toml` and `requirements.txt` now include `torch`, `torchvision`, and `scikit-learn`.

4. Generated outputs are ignored by git.
   If a future machine needs benchmark outputs, copy them explicitly or rerun commands.

5. There may be unrelated local/user files.
   Do not delete root `fashionmnist_20epoch_metrics.*`,
   `scheduled_iterate_muon_academic_report.pdf`, or dataset folders unless the
   user explicitly asks.

6. Current benchmark subset results are deterministic but subset-limited.
   Accuracy numbers are for 4096/1024 Fashion-MNIST, 5000/1000 CIFAR-10,
   10000/2000 CIFAR-10 ResNet, or 20000/5000 larger CIFAR-10 runs as stated
   in each section. These are not full-dataset claims unless explicitly
   labeled as full-dataset runs.

7. The Adam and Muon projects have similar but not identical control behavior.
   Adam needs a tight alpha cap around Adam scale for CIFAR; Muon naturally grew to much larger alpha values and still improved over fixed Muon in the subset runs.

8. Delayed-feedback optimizers are not same-step controllers.
   They avoid the extra forward pass by using the next naturally computed loss
   as feedback for the previous step. This is lower overhead, but it is not a
   line-search/trust-region replacement, cannot undo the previous step, and
   mixes minibatch noise into the rho estimate unless batches are repeated or
   full-batch data is used.

## Recommended Next Steps For The Next Agent

1. Run the next compact same-step controlled Adam ResNet controller test:
   `rho_star=0.825`, `kp=0.02`, `kp_down=0.04` and/or `0.06`,
   `alpha_max=2.25e-3`, variants `controlled_raw_rho` and `controlled_ema`,
   on the same 10k/2k, 20-epoch, seed-123 CIFAR ResNet setup. This tests the
   middle ground between permissive `rho_star=0.80` cap saturation and
   over-conservative `rho_star=0.85` floor-hugging.

   Example command for `kp_down=0.04`:

```bash
MPLCONFIGDIR=/tmp python3 delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py \
  --epochs 20 \
  --train-subset 10000 \
  --test-subset 2000 \
  --batch-size 128 \
  --lr 1e-3 \
  --controlled-alpha-min 1e-3 \
  --controlled-alpha-max 2.25e-3 \
  --controlled-rho-star 0.825 \
  --controlled-rho-beta 0.90 \
  --controlled-kp 0.02 \
  --controlled-kp-down 0.04 \
  --controlled-min-alpha-factor 0.98 \
  --controlled-max-alpha-factor 1.015 \
  --output-dir outputs/cifar10_resnet_adam_rhostar0p825_kpdown0p04_cap2p25e3_lr1e3_10k_2k_20epoch_seed123 \
  --print-every 1 \
  --checkpoint-every 5 \
  --variants controlled_raw_rho controlled_ema
```

2. Tune delayed-feedback Adam on CIFAR/Fashion-MNIST using delayed-specific rho
   targets. The latest ResNet run showed delayed `rho_bar` near `0.13-0.24`,
   far below the same-step target range. Start by sweeping `rho_star` around
   `0.15-0.30`, alpha floors near `0.8-1.1x` base LR, and caps near
   `1.25-1.75x` base LR. Compare wall-clock, alpha traces, and rho diagnostics.

3. Add stronger fixed-LR and warmup baselines for controlled Adam. Include
   vanilla Adam at `1e-3`, `1.25e-3`, `1.5e-3`, and possibly `2e-3`, plus a
   simple warmup from `1e-3` to `1.5e-3`. This tests whether controlled Adam is
   doing more than discovering a higher cap.

4. Validate the simplified controlled-Adam preset interface on a neural task
   before changing any default neural settings. The deterministic sweep suggests
   the `aggressive_high_floor` preset is strong on 2D functions, but neural runs
   operate at Adam-scale alpha ranges such as `1e-3`; alpha bounds and trust
   thresholds must be derived from the neural base learning rate and checked on
   Fashion-MNIST or CIFAR.

5. Consider extending the PCA trajectory post-processor with optional loss
   contour evaluation on the same PCA plane. The current implementation plots
   trajectories only; contour evaluation would require rebuilding the dataset
   and running many forward passes.

6. Add checkpointing and incremental epoch-metrics flushing to the Muon
   `run_mnist_demo.py`. Per-epoch progress printing already exists through
   `--print-every`, but long jobs would still benefit from partial CSV output
   and resumable checkpoints.

7. Add automatic best-epoch summary CSV/JSON so users do not need ad hoc parsing.

8. Consider a faster torch-native Muon implementation:
   - avoid NumPy round-trips where possible
   - use batched or GPU-compatible orthogonalization
   - treat bias/BatchNorm parameters separately if needed

9. Tune Muon controller settings separately from Adam.
   The default `alpha_max=0.05` worked surprisingly well on Muon CIFAR subset runs, but should be stress-tested.

10. Run full-dataset benchmarks only after checkpointing or incremental metrics
   writing is added.

11. Consider early stopping or best-checkpoint reporting.
   In long runs, peak accuracy can occur before the final epoch.

## Quick Verification Commands

After moving to a new machine and activating the Python environment, first
confirm the important packages and data path:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY

test -d controlled_adam_project/data/cifar-10-batches-py && echo "CIFAR present"
```

Root project:

```bash
PYTHONPATH=src pytest -q
```

Controlled Adam:

```bash
cd controlled_adam_project
PYTHONPATH=src pytest -q
```

Controlled Muon:

```bash
cd controlled_muon_project
PYTHONPATH=src pytest -q
```

Latest known Muon test result:

```text
12 passed
```

Delayed-feedback standalone optimizers:

```bash
cd delayed_feedback_adam
PYTHONPATH=. python -m pytest -q tests

cd ../delayed_feedback_muon
PYTHONPATH=. python -m pytest -q tests
```

Latest known delayed-feedback test results:

```text
delayed_feedback_adam: 4 passed
delayed_feedback_muon: 6 passed
```

PI optimizer smoke tests:

```bash
cd pi_adam_optimizer
python test_pi_adam.py

cd ../pi_muon_optimizer
python test_pi_muon.py
```

Controlled Adam syntax check after the recent `kp_down` addition:

```bash
python3 -m py_compile \
  controlled_adam_project/src/controlled_adam/torch_optimizers.py \
  controlled_adam_project/src/controlled_adam/optimizers.py \
  controlled_adam_project/examples/run_mnist_demo.py \
  delayed_feedback_adam/examples/run_cifar_resnet_adam_comparison.py
```

## Important Philosophy For Future Changes

The purpose is not simply to beat Adam or Muon in every setting. The purpose is to understand whether an objective-feedback controller can safely and usefully adapt the global multiplier on top of a strong optimizer direction.

When adding experiments, preserve these comparisons:

- base optimizer
- fixed direction with fixed alpha
- raw rho controller
- EMA rho controller
- EMA plus trust/recovery controller

This separation is what lets the user distinguish direction quality from controller quality.

Also preserve fixed-LR and simple-schedule baselines. They are not arguments
against controlled optimization; they are the controls needed to determine
whether an observed win comes from adaptive rho feedback, a larger effective
learning rate, or a warmup-like trajectory to the cap.
