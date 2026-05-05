"""
Stationary hover test to discriminate policy bias from reference/code bug.
Runs epoch_006000 policy on a stationary target at (0, 0, 1m).
If drone hovers with ~5cm x-offset: bias is in the policy.
If drone hovers cleanly near origin: bias is in figure-eight reference computation.

Usage:
  python scripts/hover_test.py --checkpoint ~/ckpt_ep6000_diag
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hover_steps", type=int, default=500)
    args = parser.parse_args()

    # Load checkpoint
    import orbax.checkpoint as ocp
    import yaml
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    cfg_path = REPO_ROOT / "experiments/m1_baseline/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )
    key = jax.random.PRNGKey(0)
    _, _, actor_state_init, critic_state_init = create_train_states(key, ppo_cfg)

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_path = str(Path(args.checkpoint).expanduser().resolve())
    restored = checkpointer.restore(ckpt_path, item={"actor": actor_state_init, "critic": critic_state_init})
    actor_state = restored["actor"]
    print(f"Checkpoint loaded.")

    from iron_man_drone.envs.quadrotor_env import (
        VecEnv, DT, EPISODE_STEPS, _build_obs, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
    )
    from iron_man_drone.envs.trajectories import (
        Trajectory, MAX_SEGS, TRAJ_POLY,
        eval_trajectory_position, get_reference_window,
    )

    class EvalCfg:
        num_envs = 1

    env = VecEnv(EvalCfg())
    drone_id = env.mj_model.body("drone").id

    # Build a stationary trajectory: all waypoints at (0, 0, 1), zero velocity
    # We use TRAJ_POLY with all waypoints identical so eval gives constant (0,0,1)
    hover_traj = Trajectory(
        waypoints=jnp.tile(jnp.array([0.0, 0.0, 1.0]), (MAX_SEGS + 1, 1)),
        cum_times=jnp.concatenate([jnp.zeros(1), jnp.full((MAX_SEGS,), jnp.inf)]),
        traj_type=jnp.array(TRAJ_POLY, dtype=jnp.int32),
        total_time=jnp.array(EPISODE_STEPS * DT, dtype=jnp.float32),
    )

    # Verify trajectory evaluates to (0,0,1) everywhere
    test_pos = np.array(eval_trajectory_position(hover_traj, jnp.float32(5.0)))
    assert np.allclose(test_pos, [0.0, 0.0, 1.0], atol=1e-4), f"Hover traj not stationary: {test_pos}"
    print(f"Hover trajectory verified: eval at t=5.0s -> {test_pos} (should be [0,0,1])")

    eval_reset = jax.jit(env._reset_fn)
    eval_step  = jax.jit(env._step_fn)

    # Run from a clean reset, inject hover trajectory
    key, rk = jax.random.split(key)
    state, a_obs, _ = eval_reset(rk, jnp.ones(()))
    state = state._replace(traj=hover_traj)
    a_obs, _ = _build_obs(state.mjx_data, hover_traj, state.step, drone_id)

    positions = []
    for si in range(args.hover_steps):
        mean, _ = actor_state.apply_fn(actor_state.params, a_obs[None])
        action = mean[0]
        state, a_obs, _, _, done = eval_step(state, action, jnp.ones(()))
        state = state._replace(traj=hover_traj)
        positions.append(np.array(state.mjx_data.xpos[drone_id]))
        if bool(done):
            print(f"  Terminated at step {si}")
            break

    positions = np.array(positions)   # (T, 3)
    T = len(positions)

    # Analyse last 200 steps (after any transient)
    steady = positions[min(200, T//2):]
    mean_pos = steady.mean(axis=0)
    std_pos  = steady.std(axis=0)

    print()
    print("=" * 60)
    print("  HOVER TEST RESULTS")
    print("=" * 60)
    print(f"  Steps run: {T}  (steady-state: last {len(steady)})")
    print(f"  Mean position:  x={mean_pos[0]:+.4f}  y={mean_pos[1]:+.4f}  z={mean_pos[2]:+.4f} m")
    print(f"  Std  position:  x={std_pos[0]:.4f}   y={std_pos[1]:.4f}   z={std_pos[2]:.4f} m")
    print()

    xy_offset = np.linalg.norm(mean_pos[:2])
    z_error   = abs(mean_pos[2] - 1.0)
    print(f"  XY offset from origin:  {xy_offset:.4f} m  (ref: {[0.0,0.0]})")
    print(f"  Z error from 1m:        {z_error:.4f} m")
    print()

    fig8_bias = np.array([-0.0464, -0.0180])  # from trajectory diagnosis

    if xy_offset > 0.02:
        print(f"  >> POLICY BIAS confirmed: drone hovers {xy_offset:.3f}m off origin")
        proj = np.dot(mean_pos[:2], fig8_bias) / np.linalg.norm(fig8_bias)
        print(f"  >> Projection onto figure-eight offset direction: {proj:.4f}m")
        if abs(proj) > 0.02:
            print(f"  >> Same direction as fig-8 offset — pure policy bias (not reference frame bug)")
        else:
            print(f"  >> Different direction from fig-8 offset — partial code bug possible")
    else:
        print(f"  >> CLEAN HOVER: bias is in figure-eight reference/frame, NOT in policy")
        print(f"  >> Check eval_trajectory_position for figure-eight or _build_obs frame issue")

    print("=" * 60)


if __name__ == "__main__":
    main()
