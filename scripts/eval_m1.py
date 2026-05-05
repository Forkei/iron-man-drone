"""
Evaluate a trained M1 policy on all benchmark trajectories.
Computes MED (Mean Euclidean Distance, x-y) and compares to paper Table III.

Usage:
  python scripts/eval_m1.py --checkpoint experiments/m1_baseline/RUN/checkpoints/final
  python scripts/eval_m1.py --checkpoint PATH --num_episodes 10
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).parent.parent

PAPER_MED = {
    "figure_eight_slow":   0.020,
    "figure_eight_normal": 0.028,
    "figure_eight_fast":   0.050,
    "pentagram_slow":      0.030,
    "pentagram_fast":      0.060,
    "random_polynomial":   0.030,
    "random_zigzag":       0.050,
}
THRESHOLD = {k: 2 * v for k, v in PAPER_MED.items()}


def _inject_traj(state, eval_traj):
    """Replace the trajectory in a single-env batched state."""
    from iron_man_drone.envs.quadrotor_env import EnvState
    batched_traj = jax.tree_util.tree_map(lambda x: x[None], eval_traj)
    return EnvState(
        mjx_data=state.mjx_data,
        rotor_speeds=state.rotor_speeds,
        prev_action=state.prev_action,
        traj=batched_traj,
        step=state.step,
        done=state.done,
    )


def evaluate_trajectory(actor_state, env, eval_traj, traj_name, num_episodes=5, key=None):
    """Run num_episodes episodes on eval_traj. Returns mean MED (x-y)."""
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS
    from iron_man_drone.envs.trajectories import get_reference_pos

    mj_model = env.mj_model
    drone_body_id = mj_model.body("drone").id

    all_meds = []
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)

        state, a_obs, c_obs = env.batch_reset(
            jax.random.split(reset_key, 1), jnp.ones(1)
        )
        # Override env's random trajectory with the eval trajectory
        state = _inject_traj(state, eval_traj)
        # Rebuild obs with the injected trajectory.
        # state.mjx_data is batched (1 env); _build_obs expects unbatched — squeeze first.
        from iron_man_drone.envs.quadrotor_env import _build_obs, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
        unbatched_mjx = jax.tree_util.tree_map(lambda x: x[0], state.mjx_data)
        a_obs_new, _ = _build_obs(
            unbatched_mjx, eval_traj, jnp.zeros((), dtype=jnp.int32), drone_body_id
        )
        a_obs = a_obs_new[None]  # add batch dim

        positions = []
        ref_positions = []

        for step_i in range(EPISODE_STEPS):
            mean, _ = actor_state.apply_fn(actor_state.params, a_obs)
            action = mean  # deterministic at eval

            state, a_obs, c_obs, reward, done = env.batch_step(
                state, action, jnp.ones(1)
            )
            # Re-inject trajectory so env step doesn't drift on auto-reset
            state = _inject_traj(state, eval_traj)

            pos = np.array(state.mjx_data.xpos[:, drone_body_id, :])  # (1, 3)
            ref = np.array(get_reference_pos(eval_traj, jnp.int32(step_i)))  # (3,)

            positions.append(pos[0])
            ref_positions.append(ref)

            if done[0]:
                print(f"    Episode {ep} terminated at step {step_i}")
                break

        positions    = np.array(positions)
        ref_positions = np.array(ref_positions)
        xy_err = np.linalg.norm(positions[:, :2] - ref_positions[:, :2], axis=1)
        med = xy_err.mean()
        all_meds.append(med)
        print(f"    Ep {ep}: MED={med:.4f}m")

    return np.mean(all_meds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    args.checkpoint = str(Path(args.checkpoint).resolve())

    print("=== M1 Policy Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")

    import yaml
    import orbax.checkpoint as ocp
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    # Load config to get network dims
    config_path = Path(args.checkpoint).parent.parent / "config_frozen.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "experiments/m1_baseline/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        actor_obs_dim=cfg["observation"]["actor_dim"],
        critic_obs_dim=cfg["observation"]["critic_dim"],
        action_dim=cfg["action"]["dim"],
        hidden_dim=cfg["network"]["hidden_dim"],
        num_layers=cfg["network"]["num_layers"],
    )

    init_key = jax.random.PRNGKey(0)
    _actor, _critic, actor_state, _critic_state = create_train_states(init_key, ppo_cfg)

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(args.checkpoint)
    actor_state = actor_state.replace(params=ckpt["actor"]["params"])
    print("Checkpoint loaded.")

    from iron_man_drone.envs.quadrotor_env import VecEnv, DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory,
        make_pentagram_trajectory,
        sample_polynomial_trajectory,
        sample_zigzag_trajectory,
    )

    class EvalCfg:
        num_envs = 1

    env = VecEnv(EvalCfg())
    key = jax.random.PRNGKey(0)
    ls  = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS

    eval_trajs = {
        "figure_eight_slow":   make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="slow"),
        "figure_eight_normal": make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="normal"),
        "figure_eight_fast":   make_figure_eight_trajectory(DT, EPISODE_STEPS, ls, speed="fast"),
        "pentagram_slow":      make_pentagram_trajectory(DT, EPISODE_STEPS, ls, speed="slow"),
        "pentagram_fast":      make_pentagram_trajectory(DT, EPISODE_STEPS, ls, speed="fast"),
    }

    results = {}
    print()
    print(f"{'Trajectory':<25} {'Our MED':>9} {'Paper':>8} {'Ratio':>7} {'Pass?':>7}")
    print("-" * 62)

    for name, traj in eval_trajs.items():
        key, k = jax.random.split(key)
        med    = evaluate_trajectory(actor_state, env, traj, name, args.num_episodes, k)
        paper  = PAPER_MED[name]
        ratio  = med / paper
        passed = med < THRESHOLD[name]
        results[name] = {"med": med, "paper": paper, "ratio": ratio, "pass": passed}
        print(f"  {name:<23} {med:9.4f} {paper:8.4f} {ratio:7.2f} {'PASS' if passed else 'FAIL':>7}")

    all_pass = all(r["pass"] for r in results.values())
    print()
    print(f"M1 RESULT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if not all_pass:
        print(f"  Failed: {', '.join(k for k, v in results.items() if not v['pass'])}")
        print("  See notes/M1_hypothesis.md failure mode section.")

    run_dir = Path(args.checkpoint).parent.parent
    output = args.output or str(run_dir / "M1_eval_results.md")
    with open(output, "w") as f:
        f.write("# M1 Evaluation Results\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write("| Trajectory | Our MED (m) | Paper MED (m) | Ratio | Pass? |\n")
        f.write("|---|---|---|---|---|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['med']:.4f} | {r['paper']:.4f} | {r['ratio']:.2f} | "
                    f"{'✓' if r['pass'] else '✗'} |\n")
        f.write(f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**\n")
    print(f"\nResults written to: {output}")


if __name__ == "__main__":
    main()
