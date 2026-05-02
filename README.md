# Iron Man Drone

Progressive RL drone project. Five milestones from SimpleFlight reproduction to GRaD-Nav++ language-commanded flight. Built on MuJoCo MJX + JAX.

## Milestones

| # | Goal | Status |
|---|---|---|
| M1 | SimpleFlight recipe on MuJoCo MJX | 🔄 In progress |
| M2 | Fault tolerance (MAVEN-style DR) | ⏳ Pending M1 |
| M3 | Visual obstacle avoidance | ⏳ Pending M2 |
| M4 | 3DGS-SLAM mapping + landmark return | ⏳ Pending M3 |
| M5 | GRaD-Nav++ — language commands | ⏳ Pending M4 |

## M1 Quick Start (WSL2)

```bash
# 1. Setup environment (one-time)
bash scripts/setup_env.sh
conda activate drone

# 2. Verify GPU
python -c "import jax; print(jax.devices())"

# 3. Read the gate doc
cat notes/M1_hypothesis.md

# 4. Train
python scripts/train_m1.py

# 5. Evaluate
python scripts/eval_m1.py --checkpoint experiments/m1_baseline/RUN/checkpoints/final
```

## Repo layout

```
src/iron_man_drone/
  envs/
    crazyflie.xml          MuJoCo MJCF (dynamics only, forces applied programmatically)
    quadrotor_env.py       MJX env: parallel envs via jax.vmap
    trajectories.py        Figure-eight, pentagram, polynomial, zigzag
  control/
    ctbr_controller.py     CTBR → motors (rate PD + allocation mixer + first-order lag)
  policy/
    networks.py            Flax Actor (45-dim) + Critic (46-dim), 3×256 ELU+LN
    ppo.py                 PPO: jax.lax.scan rollout, separate actor/critic TrainStates
  utils/
    domain_randomization.py  k_f ± 30% per episode
experiments/m1_baseline/
  config.yaml              Frozen hyperparameters (SimpleFlight Table VI)
notes/
  M1_hypothesis.md         Gating artifact — must exist before training
scripts/
  setup_env.sh             WSL2 environment setup
  train_m1.py              Training entry point (gates on hypothesis doc)
  eval_m1.py               Evaluation vs paper Table III targets
```

## Non-negotiable constraints

- CTBR action space (body rates + collective thrust)
- Rotation matrix in obs, never quaternion
- Separate actor/critic Flax modules + separate Optax optimizers
- Previous action NOT in actor observation
- Time step k ONLY in critic obs
- Entropy << reward at init (< 10%)
- No fine-tuning new capabilities onto converged policies

## Papers

- **SimpleFlight**: Chen et al., RAL 2025, arXiv:2412.11764
- **GRaD-Nav++**: Chen et al., Stanford, RAL 2025, arXiv:2506.14009
- **MAVEN**: arXiv:2603.10714
