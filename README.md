# Iron Man Drone

Progressive RL drone project. Five milestones from SimpleFlight reproduction to GRaD-Nav++ language-commanded flight.

## Milestones

| # | Goal | Status |
|---|---|---|
| M1 | Reproduce SimpleFlight (Chen et al., RAL 2025) in Omnidrones | 🔄 In progress |
| M2 | Fault tolerance via MAVEN-style DR | ⏳ Pending M1 |
| M3 | Visual obstacle avoidance | ⏳ Pending M2 |
| M4 | 3DGS-SLAM mapping + landmark return | ⏳ Pending M3 |
| M5 | GRaD-Nav++ replication — language commands | ⏳ Pending M4 |

## M1 Quick Start

See `scripts/setup_env.sh` for full WSL2 environment setup.

```bash
# 1. Clone Omnidrones
git clone https://github.com/btx0424/OmniDrones simpleflight/OmniDrones
cd simpleflight/OmniDrones && pip install -e .

# 2. Validate baseline checkpoints
python scripts/eval_baseline.py --traj figure_eight_normal

# 3. Train
python scripts/train_m1.py --config experiments/m1_baseline/config.yaml
```

## Repo layout

```
simpleflight/       Omnidrones source + SimpleFlight assets
experiments/        Training runs — each run is a timestamped subdirectory
  m1_baseline/      First M1 run
    config.yaml     Hyperparameters (frozen before run)
    checkpoints/    .pt files (gitignored)
    logs/           TensorBoard (gitignored)
    M1_results.md   Written after run completes
notes/              Hypothesis docs — written BEFORE training runs
  M1_hypothesis.md  M1 training hypothesis (gating artifact)
scripts/            Reproducible run/eval scripts
  setup_env.sh      WSL2 environment setup
  train_m1.py       M1 training entry point
  eval_baseline.py  Evaluate SimpleFlight published checkpoints
```

## Non-negotiable constraints

- CTBR action space (body rates + collective thrust)
- Rotation matrix in obs, never quaternion
- Separate actor/critic networks + separate optimizers
- Previous action NOT in actor observation
- Time vector ONLY in critic
- Entropy coefficient << reward (< 10% of reward at init)
- No fine-tuning new capabilities onto converged policies

## Papers

- **SimpleFlight**: Chen et al., RAL 2025, arXiv:2412.11764
- **GRaD-Nav++**: Chen et al., Stanford, RAL 2025, arXiv:2506.14009
- **MAVEN**: arXiv:2603.10714
