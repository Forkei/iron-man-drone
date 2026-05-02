# Iron Man Drone — Claude Code Context

## Critical rules (never violate)

- **CTBR action space** — collective thrust + body rates. Never direct motor commands.
- **Rotation matrix** (9-dim) in observations, never quaternion — ~63% perf drop with quaternion.
- **Previous action u_{t-1} NOT in actor observation** — causes non-stationarity.
- **Time step k ONLY in critic**, not actor — causes OOD failures on long flights.
- **Separate actor/critic Flax modules with separate Optax optimizers** — sharing breaks the asymmetric advantage.
- **Entropy coefficient << reward** — at init, entropy must be < 10% of total reward. Sanity check before first training.
- **No fine-tuning new capabilities onto converged policies** — bake via DR from epoch 0. This rule killed the previous attempt.
- **No training without a hypothesis doc** — `notes/M1_hypothesis.md` is the gating artifact.
- **Do not scope creep into M2+ features in M1** — fault tolerance, cameras, obstacles are M2+.

## Repository layout (v2 MJX stack)

```
src/iron_man_drone/
  envs/
    crazyflie.xml          MuJoCo MJCF model (dynamics only, no actuators)
    quadrotor_env.py       MJX env: step, reset, obs, reward, termination
    trajectories.py        Figure-eight, pentagram, polynomial, zigzag generators
  control/
    ctbr_controller.py     CTBR → motor speeds (rate PD + mixer + motor lag)
  policy/
    networks.py            Flax Actor + Critic (ELU, LayerNorm, 256 hidden)
    ppo.py                 PPO trainer (PureJaxRL-style, jax.lax.scan rollout)
  utils/
    domain_randomization.py  k_f ± 30% per episode
experiments/m1_baseline/
  config.yaml              Frozen hyperparameters
notes/
  M1_hypothesis.md         Gating artifact — written before training
scripts/
  setup_env.sh             WSL2 setup (JAX + MuJoCo MJX, no Isaac Sim)
  train_m1.py              Training entry point
  eval_m1.py               Evaluation vs paper Table III
```

## Stack (M1, supersedes v1)

- **Simulator**: MuJoCo MJX (JAX-native, GPU-parallel via vmap)
- **Python**: 3.10+
- **JAX**: with CUDA 12 (`pip install "jax[cuda12]"`)
- **MuJoCo**: >= 3.1.6 (MJX bundled)
- **RL framework**: Flax + Optax + Distrax (PureJaxRL-style)
- **Platform**: WSL2 Ubuntu 22.04/24.04 (JAX works fine, no Isaac Sim baggage)
- **Conda env**: `drone`

## Observation dimensions

- **Actor obs**: 45-dim = [e^W (30), v (3), R (9), ω (3)]
- **Critic obs**: 46-dim = [actor_obs (45), k (1)]
- e^W = relative positions to next 10 ref points in world frame (50ms spacing)
- v = linear velocity (world frame) from qvel[:3]
- R = rotation matrix body→world, flattened, from xmat[drone_body_id]
- ω = body rates from qvel[3:6]
- k = normalized episode step in [0,1] (CRITIC ONLY)

## Dynamics constants (Crazyflie 2.1)

- mass: 0.0321 kg
- inertia: diag(1.4e-5, 1.4e-5, 2.17e-5) kg·m²
- kf: 2.350347298350041e-08 N/(rad/s)²
- km: 7.24e-10 N·m/(rad/s)²
- motor_tau: 0.025 s
- arm_length: 0.046 m (→ d = 0.0325 m offset in X-config)
- max_rotor_speed: 2315 rad/s

## M1 workflow

1. `bash scripts/setup_env.sh` — install JAX + MuJoCo MJX
2. `conda activate drone`
3. Verify: `python -c "import jax; print(jax.devices())"`
4. Read `notes/M1_hypothesis.md` — gating artifact
5. `python scripts/train_m1.py`
6. `python scripts/eval_m1.py --checkpoint PATH`
7. Write `experiments/m1_baseline/M1_results.md`
8. `git tag m1-baseline`

## M1 success criteria

- MJX sim > 50k steps/sec at 1024 envs on 4070
- figure_eight_normal MED < 0.056 m
- All benchmark trajectories complete without crash
- Stable reward/loss curves, entropy not collapsed
