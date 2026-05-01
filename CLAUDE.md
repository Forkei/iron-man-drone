# Iron Man Drone — Claude Code Context

## Critical rules (never violate)

- **CTBR action space** — collective thrust + body rates. Never direct motor commands.
- **Rotation matrix** in observations, never quaternion (~63% perf drop with quaternion, per SimpleFlight ablation).
- **Previous action u_{t-1} NOT in actor observation** — causes non-stationarity.
- **Time step k ONLY in critic**, not actor — causes OOD failures on long flights if in actor.
- **Separate actor/critic nn.Module instances with separate optimizers** — sharing weights or optimizers breaks the asymmetric advantage.
- **Entropy coefficient << reward** — at init, entropy must be < 10% of total reward. Sanity check before first training.
- **No fine-tuning new capabilities onto converged policies** — bake via domain randomization from epoch 0. This rule killed the previous attempt.
- **No training without a hypothesis doc** — `notes/M1_hypothesis.md` is the gating artifact for M1.

## Repository layout

```
simpleflight/SimpleFlight/   Cloned from https://github.com/thu-uav/SimpleFlight
experiments/m1_baseline/     M1 training run configs + results
notes/                       Hypothesis docs (written BEFORE training)
scripts/                     Setup and run scripts
```

## Stack (M1)

- **SimpleFlight repo**: `https://github.com/thu-uav/SimpleFlight` (thu-uav, NOT zhuchao0903)
- **Isaac Sim**: 2022.2.0 (required — no PyTorch-only path)
- **Python**: 3.7 (required by Isaac Sim 2022.x)
- **Conda env**: `sim`
- **torchrl**: pinned commit e39e701 (submodule)
- **tensordict**: pinned commit 5e6205c (submodule)
- **Platform**: WSL2 Ubuntu 22.04/24.04 (Linux only — Isaac Sim does not run on Windows)

## M1 training workflow

1. `bash scripts/setup_env.sh` — install Isaac Sim + SimpleFlight
2. `bash scripts/eval_baseline.sh` — validate published checkpoints vs paper Table III
3. Read `notes/M1_hypothesis.md` — gating artifact
4. `bash scripts/train_m1.sh` — train
5. `bash scripts/eval_m1.sh --checkpoint PATH` — evaluate
6. Write `experiments/m1_baseline/M1_results.md`
7. `git tag m1-baseline`

## M1 success criteria

- figure_eight_normal MED < 0.056 m (2× paper's 0.028 m)
- All benchmark trajectories complete without crash
- Smooth reward curve, no entropy collapse

## Do NOT do in M1

No hardware, no cameras, no obstacles, no language interface, no fault tolerance.
Resist scope creep. M1 is a clean reproduction of one published recipe.
