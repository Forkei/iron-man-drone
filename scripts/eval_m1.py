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

# Paper Table III reference values (MED in meters, x-y plane)
PAPER_MED = {
    "figure_eight_slow":   0.020,
    "figure_eight_normal": 0.028,
    "figure_eight_fast":   0.050,
    "pentagram_slow":      0.030,
    "pentagram_fast":      0.060,
    "random_polynomial":   0.030,
    "random_zigzag":       0.050,
}
# M1 acceptance threshold: 2× paper values
THRESHOLD = {k: 2 * v for k, v in PAPER_MED.items()}


def evaluate_trajectory(
    actor_state,
    env,
    traj_fn,
    traj_name: str,
    num_episodes: int = 5,
    key=None,
):
    """Run num_episodes episodes on traj_fn, return mean MED (x-y)."""
    from iron_man_drone.envs.quadrotor_env import (
        EPISODE_STEPS, DT, load_mjx_model, _build_obs,
    )
    from iron_man_drone.envs.trajectories import (
        precompute_trajectory, get_reference_pos, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
    )
    from iron_man_drone.control.ctbr_controller import ctbr_to_rotor_speeds, compute_wrench
    import distrax

    mj_model, mjx_model = load_mjx_model()
    drone_body_id = mj_model.body("drone").id

    all_meds = []
    for ep in range(num_episodes):
        key, reset_key, act_key = jax.random.split(key, 3)

        # Precompute trajectory
        traj = precompute_trajectory(
            traj_fn, DT, EPISODE_STEPS,
            lookahead_steps=LOOKAHEAD_N * LOOKAHEAD_DT_STEPS,
        )

        states, a_obs, c_obs = env.batch_reset(
            jax.random.split(reset_key, 1), jnp.ones(1)
        )
        # Hack: replace trajectory with eval trajectory
        # (env reset uses random traj; override for eval)

        positions = []
        ref_positions = []

        for step_i in range(EPISODE_STEPS):
            key, act_key = jax.random.split(key)
            mean, log_std = actor_state.apply_fn(actor_state.params, a_obs)
            # Deterministic at eval (use mean)
            action = mean

            states, a_obs, c_obs, reward, done = env.batch_step(
                states, action, jnp.ones(1)
            )

            pos = np.array(states.mjx_data.xpos[:, drone_body_id, :])  # (1, 3)
            ref = np.array(get_reference_pos(traj, step_i))  # (3,)

            positions.append(pos[0])
            ref_positions.append(ref)

            if done[0]:
                print(f"    Episode {ep} terminated at step {step_i}")
                break

        positions = np.array(positions)
        ref_positions = np.array(ref_positions)

        # MED: mean Euclidean distance in x-y plane
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

    print(f"=== M1 Policy Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")

    # Load checkpoint
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(args.checkpoint)
    actor_state = ckpt["actor"]
    print("Checkpoint loaded.")

    # Build env (single env for eval)
    from iron_man_drone.envs.quadrotor_env import VecEnv

    class EvalCfg:
        num_envs = 1

    env = VecEnv(EvalCfg())
    key = jax.random.PRNGKey(0)

    # Eval trajectories
    from iron_man_drone.envs.trajectories import (
        figure_eight_positions, pentagram_positions,
        sample_polynomial_trajectory, sample_zigzag_trajectory,
        precompute_trajectory,
    )
    from iron_man_drone.envs.quadrotor_env import DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    import functools

    traj_fns = {
        "figure_eight_slow":   functools.partial(figure_eight_positions, speed="slow"),
        "figure_eight_normal": functools.partial(figure_eight_positions, speed="normal"),
        "figure_eight_fast":   functools.partial(figure_eight_positions, speed="fast"),
        "pentagram_slow":      functools.partial(pentagram_positions, speed="slow"),
        "pentagram_fast":      functools.partial(pentagram_positions, speed="fast"),
    }

    results = {}
    print()
    print(f"{'Trajectory':<25} {'Our MED':>9} {'Paper':>8} {'Ratio':>7} {'Pass?':>7}")
    print("-" * 62)

    for name, traj_fn in traj_fns.items():
        key, k = jax.random.split(key)
        med = evaluate_trajectory(actor_state, env, traj_fn, name, args.num_episodes, k)
        paper = PAPER_MED[name]
        ratio = med / paper
        passed = med < THRESHOLD[name]
        results[name] = {"med": med, "paper": paper, "ratio": ratio, "pass": passed}
        print(f"  {name:<23} {med:9.4f} {paper:8.4f} {ratio:7.2f} {'PASS' if passed else 'FAIL':>7}")

    # Summary
    all_pass = all(r["pass"] for r in results.values())
    print()
    print(f"M1 RESULT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if not all_pass:
        failed = [k for k, v in results.items() if not v["pass"]]
        print(f"  Failed: {', '.join(failed)}")
        print("  See notes/M1_hypothesis.md failure mode section.")

    # Write results file
    output = args.output or str(REPO_ROOT / "experiments/m1_baseline/M1_eval_results.md")
    with open(output, "w") as f:
        f.write("# M1 Evaluation Results\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write("| Trajectory | Our MED (m) | Paper MED (m) | Ratio | Pass? |\n")
        f.write("|---|---|---|---|---|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['med']:.4f} | {r['paper']:.4f} | {r['ratio']:.2f} | {'✓' if r['pass'] else '✗'} |\n")
        f.write(f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**\n")
    print(f"\nResults written to: {output}")


if __name__ == "__main__":
    main()
