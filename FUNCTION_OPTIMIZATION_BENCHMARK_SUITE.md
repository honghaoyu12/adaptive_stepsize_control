# Function Optimization Benchmark Suite

Last updated: 2026-05-25

This document describes the simple deterministic benchmark suites for explaining
how the controlled optimizers behave on hand-written 2D functions. It is meant
for project communication and manager updates, not for neural-network results.

The Adam runnable implementation is:

```text
controlled_adam_project/examples/run_function_benchmark_report.py
```

The Adam generated report is:

```text
controlled_adam_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md
```

The Muon runnable implementation is:

```text
controlled_muon_project/examples/run_function_benchmark_report.py
```

The Muon generated report is:

```text
controlled_muon_project/outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md
```

The output directory is ignored by git, so rerun the command below on a new
machine to regenerate the report, CSV files, and plots.

## Purpose

The suite answers a focused question:

```text
Given the same starting points and Adam direction, does the outer-loop
actual-vs-predicted decrease controller make optimization more robust on
standard 2D objective functions?
```

It intentionally excludes MNIST, Fashion-MNIST, CIFAR-10, and all deep-learning
benchmarks. This keeps the story short, visual, deterministic, and easy to
explain.

## Adam Optimizers Compared

The Adam suite compares four optimizers:

| Optimizer | Meaning |
|---|---|
| Vanilla Adam | Adam with a fixed global learning rate. |
| Controlled Adam, raw rho | Adam direction plus a global multiplier updated from the current actual/predicted decrease ratio. |
| Controlled Adam, EMA rho | Same controller, but the rho signal is smoothed before changing the global multiplier. |
| Controlled Adam, EMA + trust | EMA-rho controller plus the trust-region style tiny-alpha expansion hook used in the neural-network benchmarks. |

The controller uses:

```text
rho_t = actual decrease / first-order predicted decrease
alpha_{t+1} = clip(alpha_t * exp(kp * (rho_signal - rho_star)))
```

For these deterministic functions, the before-step and after-step objective
values are exact. There is no minibatch noise.

## Muon Optimizers Compared

The Muon function report compares four variants:

| Optimizer | Meaning |
|---|---|
| Vanilla Muon | Muon-style normalized direction with a fixed global learning rate. |
| Controlled Muon, raw rho | Muon direction with alpha updated directly from the current rho. |
| Controlled Muon, EMA rho | Muon direction with alpha updated from an EMA-smoothed rho. |
| Controlled Muon, EMA + trust | EMA-rho controller plus the trust-region style tiny-alpha expansion hook. |

For 2D vector functions, the Muon report uses a vector analogue of Muon: the
momentum vector is treated as a column matrix and passed through the same
orthogonalization utility used elsewhere in the Muon subproject. This makes the
function report useful for direction-control diagnostics, but it is not a full
matrix-valued Muon stress test.

The older `fixed_muon_direction` diagnostic was removed from the function
report because, in this local 2D runner, it duplicated `vanilla_muon`: both used
the same Muon-style direction with a fixed alpha and no rho controller.

## Function Suite

The benchmark uses nine 2D objective functions already implemented in
`controlled_adam_project`:

| Function | What it demonstrates |
|---|---|
| Quadratic | Smooth convex problem with strong curvature imbalance. |
| Rosenbrock | Narrow curved valley where fixed step sizes are hard to tune. |
| Himmelblau | Multiple equivalent minima and basin behavior. |
| Rastrigin | Many local wells; useful limitation case for local methods. |
| Beale | Curved valley with a sharp global minimum. |
| Ackley | Broad basin with oscillatory ripples. |
| Six-hump camel | Multiple local basins and two global minima. |
| Goldstein-Price | Steep nonlinear coupling and large dynamic range. |
| Easom | Very sharp isolated optimum near `(pi, pi)`. |

Each function uses five fixed starting points. The starts are hand-picked and
deterministic, so every optimizer sees the same problem instances.

## How To Run

From the Adam subproject directory:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

The command writes:

```text
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT.md
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_BENCHMARK_REPORT_ZH.md
outputs/function_report_multistart/per_start_results.csv
outputs/function_report_multistart/aggregate_results.csv
outputs/function_report_multistart/benchmark_config.csv
outputs/function_report_multistart/*_trajectory_comparison.png
outputs/function_report_multistart/*_objective_curves.png
outputs/function_report_multistart/*_alpha_curves.png
outputs/function_report_multistart/success_rate_by_objective.png
outputs/function_report_multistart/median_best_residual_by_objective.png
```

For the shorter manager-facing report with only three representative
objectives, run:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

The current trimmed report is the best starting point for a concise manager
update because it avoids overwhelming the story with all nine functions.
It also writes `FUNCTION_OPTIMIZATION_BENCHMARK_REPORT_ZH.md`, which is the
Chinese version intended for the manager-facing readout.

For longer or broader manager runs, the same runner supports:

```text
--step-multiplier K
--random-starts-per-objective N
--random-seed SEED
```

`--step-multiplier` scales each objective's built-in iteration budget.
`--random-starts-per-objective` appends deterministic random starts sampled
from that objective's plotting domain, while preserving the original fixed
starts. This is the preferred way to aggregate over more initial conditions
without hand-picking favorable starts.

From the Muon subproject directory:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_multistart
```

The Muon command writes the same style of files, with the standalone report:

```text
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md
outputs/function_report_multistart/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md
```

For the shorter manager-facing Muon report with the same three representative
objectives, run:

```bash
cd controlled_muon_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed \
  --objectives quadratic beale goldstein_price
```

This writes:

```text
outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT.md
outputs/function_report_manager_trimmed/FUNCTION_OPTIMIZATION_MUON_BENCHMARK_REPORT_ZH.md
```

The Muon function runner supports the same `--step-multiplier`,
`--random-starts-per-objective`, and `--random-seed` options as the Adam runner.

## Metrics

The main metrics are:

- success rate across the five starts;
- median final residual;
- median best residual;
- median distance to the nearest known global minimizer;
- iterations to first success, when success happens;
- accepted-step rate and final alpha diagnostics for controlled runs.

Residual means:

```text
residual = f(x) - f_min
```

This is better than raw objective value because some functions have negative
global minima, especially Easom and Six-hump camel.

Success is counted when either:

```text
residual <= objective_tolerance
```

or:

```text
distance_to_known_global_minimizer <= distance_tolerance
```

## Current Adam Result Snapshot

The latest regenerated run used nine objectives and five starts per objective.
Winner counts across objectives were:

| Criterion | Vanilla Adam | Controlled raw-rho | Controlled EMA-rho | Controlled EMA + trust |
|---|---:|---:|---:|---:|
| Highest success rate | 5 | 9 | 7 | 7 |
| Lowest median final residual | 5 | 3 | 4 | 3 |
| Lowest median best residual | 5 | 3 | 4 | 3 |

Ties are counted for every tied optimizer, so row totals can exceed nine.

The important interpretation is nuanced:

- Controlled raw-rho ties or wins success rate on every objective in this
  suite.
- Controlled variants improve success rate over vanilla Adam on Quadratic,
  Rosenbrock, Himmelblau, and Beale.
- Controlled variants improve median best residual over vanilla Adam on
  Quadratic, Rosenbrock, Beale, and Goldstein-Price.
- Vanilla Adam still ties or wins several residual comparisons when its fixed
  learning rate is already well matched.
- Rastrigin, Ackley, and Six-hump camel are useful limitation cases: local
  step-size control does not solve global basin selection by itself.
- The EMA+trust variant is included for consistency with the neural-network
  benchmarks. On the current function suite it mostly overlaps EMA-rho because
  the tiny-alpha/high-rho trust expansion hook rarely triggers.

The current three-function trimmed Adam report, using Quadratic, Beale, and
Goldstein-Price, gives a cleaner manager story:

| Criterion | Vanilla Adam | Controlled raw-rho | Controlled EMA-rho | Controlled EMA + trust |
|---|---:|---:|---:|---:|
| Highest success rate | 1 | 3 | 2 | 2 |
| Lowest median final residual | 0 | 1 | 2 | 2 |
| Lowest median best residual | 0 | 1 | 2 | 2 |

In that trimmed run, `median_trust_expansions` is `0.0` for the selected
objectives, so EMA+trust should be described as included for completeness rather
than as an independently demonstrated improvement on those plots.

## Longer And Aggregated Manager Runs

After the initial trimmed report, we generated longer deterministic manager
runs to test whether the apparent controlled-Adam advantage was only an
early-iteration effect.

Three-function 10x report:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_trimmed_10x \
  --objectives quadratic beale goldstein_price \
  --step-multiplier 10
```

Extended six-function 10x report:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_extended_10x \
  --objectives quadratic beale goldstein_price rosenbrock himmelblau rastrigin \
  --step-multiplier 10
```

Extended six-function 10x report with 15 starts per objective:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_extended_10x_15starts \
  --objectives quadratic beale goldstein_price rosenbrock himmelblau rastrigin \
  --step-multiplier 10 \
  --random-starts-per-objective 10 \
  --random-seed 20260525
```

Larger 60-start aggregate with the normal/default iteration budgets:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_extended_60starts_default_steps \
  --objectives quadratic beale goldstein_price rosenbrock himmelblau rastrigin \
  --random-starts-per-objective 55 \
  --random-seed 20260525
```

Larger 60-start aggregate with 10x iteration budgets:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python examples/run_function_benchmark_report.py \
  --output-dir outputs/function_report_manager_extended_10x_60starts \
  --objectives quadratic beale goldstein_price rosenbrock himmelblau rastrigin \
  --step-multiplier 10 \
  --random-starts-per-objective 55 \
  --random-seed 20260525
```

An intermediate 30-start 10x aggregate also exists at:

```text
controlled_adam_project/outputs/function_report_manager_extended_10x_30starts/
```

Key interpretation:

- The three-function 10x report shows that vanilla Adam can catch up on some
  objectives when given many more iterations. The strongest claim is therefore
  not "controlled Adam always wins eventually"; it is "controlled Adam often
  reaches good local progress faster and is more robust to step-size scale."
- In the extended 10x report, Himmelblau is a clean controlled-Adam speed win:
  all Adam variants reach 100% success, but controlled Adam reaches the success
  criterion in about `82-83` iterations, while vanilla Adam takes about `299`.
- Rosenbrock shows a speed-versus-final-success tradeoff. Controlled Adam
  reaches success faster on successful runs, while vanilla Adam eventually gets
  higher success with enough iterations.
- The default-step 60-start aggregate is probably the best manager-facing
  evidence for "useful progress under a fixed practical budget": controlled
  variants beat vanilla success rate on Beale, Goldstein-Price, Rosenbrock,
  Himmelblau, and Quadratic within the default iteration budgets.
- In the default-step 60-start aggregate, Beale success is `33-37%` for
  controlled variants versus `5%` for vanilla Adam; Rosenbrock success is
  `28-32%` versus `8%`; Himmelblau success is `97%` versus `92%`; and vanilla
  Adam has `0%` success on the ill-conditioned Quadratic within 300 steps.
- The 60-start 10x aggregate shows the caveat: with much more time, vanilla
  Adam catches up or exceeds success on some objectives, but controlled
  variants still usually reach successful runs faster. For example, on
  Rosenbrock controlled variants reach success around `3954-5060` median
  successful iterations while vanilla takes about `8085`.
- Himmelblau remains the cleanest speed example at both budgets: in the
  60-start default run, controlled variants reach success around `84-88`
  iterations while vanilla takes about `291`; in the 60-start 10x run,
  controlled variants take about `84-91` and vanilla about `299`.
- Rastrigin remains a limitation case. More starts and more steps do not turn a
  local optimizer into a global optimizer.

The corresponding Muon 10x reports were also generated:

```text
controlled_muon_project/outputs/function_report_manager_trimmed_10x/
controlled_muon_project/outputs/function_report_manager_extended_10x/
```

They did not materially improve the Muon story. Vanilla Muon remains very
competitive on the 2D vector analogue, so the manager-facing function story
should stay centered on Adam unless the goal is explicitly to discuss Muon
limitations.

## Rastrigin Basin Benchmark

Rastrigin deserves a separate focused experiment because the global minimum is
known at `(0, 0)`, but the landscape has many local wells. Longer local
optimization from far away usually converges to a nearby local basin rather than
the true global basin.

The focused benchmark is implemented in:

```text
controlled_adam_project/examples/run_rastrigin_basin_benchmark.py
```

Run it with:

```bash
cd controlled_adam_project
MPLCONFIGDIR=/private/tmp PYTHONPATH=src:examples python examples/run_rastrigin_basin_benchmark.py \
  --output-dir outputs/rastrigin_basin_benchmark_30starts \
  --starts-per-radius 30 \
  --steps 12000
```

This samples 30 starts per radius from:

```text
x0 ~ Uniform([-r, r]^2)
r in {0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4}
```

The generated files are:

```text
outputs/rastrigin_basin_benchmark_30starts/RASTRIGIN_BASIN_BENCHMARK_REPORT.md
outputs/rastrigin_basin_benchmark_30starts/RASTRIGIN_BASIN_BENCHMARK_REPORT_ZH.md
outputs/rastrigin_basin_benchmark_30starts/aggregate_results.csv
outputs/rastrigin_basin_benchmark_30starts/per_run_results.csv
outputs/rastrigin_basin_benchmark_30starts/rastrigin_success_rate_by_radius.png
outputs/rastrigin_basin_benchmark_30starts/rastrigin_median_best_by_radius.png
outputs/rastrigin_basin_benchmark_30starts/rastrigin_iterations_to_success_by_radius.png
```

Main result:

| Radius | Success pattern |
|---:|---|
| `0.05` to `0.5` | All methods reach 100% success. |
| `0.75` | Success falls to about 57% for all methods. |
| `1.0` | Success falls to about 23% for all methods. |
| `1.5` to `3.0` | Only rare starts reach the true global minimum. |
| `4.0` | No optimizer reaches the true global minimum. |

Inside the correct basin, controlled Adam usually reaches success faster than
vanilla Adam. For example, median successful iterations were about `44` vs `73`
at radius `0.5`, and about `53` vs `96` at radius `1.0`.

The honest conclusion is:

```text
Controlled Adam improves local convergence speed inside the correct Rastrigin
basin, but it does not solve global basin selection. Far-away Rastrigin starts
need multi-start or global exploration.
```

## Current Muon Result Snapshot

The latest regenerated Muon run used the same nine objectives and five starts
per objective. Winner counts were:

| Criterion | Vanilla Muon | Controlled raw-rho | Controlled EMA-rho | Controlled EMA + trust |
|---|---:|---:|---:|---:|
| Highest success rate | 9 | 7 | 7 | 7 |
| Lowest median final residual | 4 | 3 | 3 | 4 |
| Lowest median best residual | 6 | 3 | 1 | 2 |

Ties are counted for every tied optimizer, so row totals can exceed nine.

The Muon interpretation is different from Adam:

- Vanilla Muon is very strong on this 2D vector suite.
- Controlled Muon improves some residual cases, such as Quadratic, but does not
  dominate overall.
- This is an important honest comparison: the controller is useful only when
  its alpha adaptation improves the base direction's fixed-scale behavior.
- Because the 2D direction is normalized, this report should be presented as a
  Muon-style diagnostic rather than a definitive claim about matrix Muon.

## Recommended Manager Figures

For a short Adam update, start with:

```text
success_rate_by_objective.png
median_best_residual_by_objective.png
rosenbrock_trajectory_comparison.png
rosenbrock_objective_curves.png
beale_objective_curves.png
goldstein_price_trajectory_comparison.png
rastrigin_trajectory_comparison.png
```

Suggested story:

1. Use the success-rate chart to show robustness across starts.
2. Use Rosenbrock, Beale, and Goldstein-Price to show why global step-size
   control helps on curved, steep, or scale-sensitive landscapes.
3. Use Rastrigin or Ackley to be honest about limitations: a local optimizer can
   still settle in a local basin.
4. Use alpha curves to show that the controller is changing only the scalar
   global multiplier on top of the same Adam direction.
5. If Rastrigin is shown, prefer the dedicated basin plot
   `rastrigin_success_rate_by_radius.png` rather than implying that any local
   optimizer solves the full multi-modal problem from arbitrary starts.

For Muon, use the same plot names under
`controlled_muon_project/outputs/function_report_multistart/`, but frame the
story differently: fixed Muon is already highly competitive on these 2D vector
objectives, while controlled Muon is a diagnostic for when alpha adaptation
helps or hurts that normalized direction.

## Scope And Caveats

This suite is intentionally small and deterministic. It demonstrates optimizer
behavior clearly, but it is not a complete global-optimization benchmark.

Important caveats:

- The starting points are representative, not exhaustive.
- Hyperparameters are held fixed per objective; heavily retuning vanilla Adam
  could change individual outcomes.
- The controller adapts step quality along a local direction; it does not add a
  mechanism for escaping all local minima.
- The generated report is reproducible from code, but generated outputs are not
  tracked by git.
