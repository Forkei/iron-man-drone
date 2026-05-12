"""
Unified evaluation suite for M1 and M2 policies.

Single source of truth for the T/4-corrected eval methodology.
eval_m1_full.py and eval_m2_full.py are thin wrappers around this module.

Backend: JAX lax.scan — full 1000-step episode compiled as one XLA kernel.
Termination: crash-only (height, tilt, bbox). Step timeout excluded so T/4
             offset does not shorten effective episode length.

T/4 phase offset (SimpleFlight methodology):
  Figure-eight is evaluated starting at traj_t0 = T/4, placing the reference
  at (0,0,1) = drone spawn. state.step is initialized to offset_steps so
  actor lookahead and error measurement are aligned. Pentagram, polynomial,
  and zigzag use t=0 (no offset).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

EVAL_SEEDS: list[int] = [42, 99, 7]

# T/4 step offsets for each figure-eight speed, at DT=0.01 s.
# offset = round(T / 4 / DT)
F8_OFFSETS: dict[str, int] = {"slow": 375, "normal": 138, "fast": 88}
F8_PERIODS: dict[str, float] = {"slow": 15.0, "normal": 5.5, "fast": 3.5}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TrajConfig(NamedTuple):
    """One trajectory in the eval suite."""
    name: str
    traj: object          # iron_man_drone.envs.trajectories.Trajectory
    offset_steps: int     # 0 for t=0 trajectories; F8_OFFSETS[speed] for figure-eight


class EvalCondition(NamedTuple):
    """One physical condition under which each trajectory is evaluated."""
    name: str
    priv_state: jnp.ndarray   # (8,): [η1,η2,η3,η4, mass_scale, Fx, Fy, Fz]


@dataclass
class EpisodeResult:
    med: float        # mean XY error over active steps
    n_steps: int      # number of active (non-crashed) steps
    crashed: bool


@dataclass
class ConditionResult:
    mean_med: float
    per_seed: list[EpisodeResult]     # one per seed in EVAL_SEEDS

    @property
    def meds(self) -> list[float]:
        return [r.med for r in self.per_seed]

    @property
    def any_crashed(self) -> bool:
        return any(r.crashed for r in self.per_seed)


# SuiteResult: traj_name → condition_name → ConditionResult
SuiteResult = dict[str, dict[str, ConditionResult]]


# ---------------------------------------------------------------------------
# Trajectory suite factories
# ---------------------------------------------------------------------------

def make_m1_traj_suite() -> list[TrajConfig]:
    """Seven-trajectory suite for M1 eval (no fault conditions)."""
    return _build_traj_suite()


def make_m2_traj_suite() -> list[TrajConfig]:
    """Seven-trajectory suite for M2 eval (same trajectories, two conditions)."""
    return _build_traj_suite()


def _build_traj_suite() -> list[TrajConfig]:
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    from iron_man_drone.envs.trajectories import (
        make_figure_eight_trajectory, make_pentagram_trajectory,
        sample_polynomial_trajectory, sample_zigzag_trajectory,
    )

    lookahead = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS

    configs: list[TrajConfig] = []
    for speed, off in F8_OFFSETS.items():
        total = EPISODE_STEPS + off + lookahead + 5
        traj = make_figure_eight_trajectory(DT, total, lookahead, speed=speed)
        configs.append(TrajConfig(f"figure_eight_{speed}", traj, off))

    for speed in ("slow", "fast"):
        traj = make_pentagram_trajectory(DT, EPISODE_STEPS, lookahead, speed=speed)
        configs.append(TrajConfig(f"pentagram_{speed}", traj, 0))

    poly = sample_polynomial_trajectory(jax.random.PRNGKey(42), DT, EPISODE_STEPS, lookahead)
    configs.append(TrajConfig("polynomial", poly, 0))

    zigzag = sample_zigzag_trajectory(jax.random.PRNGKey(42), DT, EPISODE_STEPS, lookahead)
    configs.append(TrajConfig("zigzag", zigzag, 0))

    return configs


def nominal_condition() -> EvalCondition:
    """Nominal physical condition: all rotors at 1.0, no mass variation."""
    return EvalCondition(
        name="nominal",
        priv_state=jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)]),
    )


def fault_condition(rotor_eta: float = 0.70, rotor_idx: int = 0) -> EvalCondition:
    """Single-rotor fault condition."""
    etas = jnp.ones(4).at[rotor_idx].set(rotor_eta)
    return EvalCondition(
        name=f"fault_eta{int(rotor_eta * 100)}",
        priv_state=jnp.concatenate([etas, jnp.ones(1), jnp.zeros(3)]),
    )


# ---------------------------------------------------------------------------
# Core eval engine
# ---------------------------------------------------------------------------

def run_eval_suite(
    actor_apply_fn: Callable,
    actor_params: object,
    env: object,
    traj_configs: list[TrajConfig],
    conditions: list[EvalCondition],
    obs_dim: int,
    seeds: list[int] = EVAL_SEEDS,
    verbose: bool = True,
    render_depth: bool = False,
    figures_dir: Path | None = None,
) -> SuiteResult:
    """
    Run all (trajectory, condition, seed) combinations via lax.scan.

    Args:
        actor_apply_fn: Flax apply fn: (params, obs) -> (mean, log_std).
                        obs shape: (1, obs_dim).
        actor_params:   Pytree of actor parameters.
        env:            VecEnv(num_envs=1, fault_prob=0.0, ...).
        traj_configs:   Output of make_m1_traj_suite() or make_m2_traj_suite().
        conditions:     List of EvalCondition (nominal, fault, etc.).
        obs_dim:        42 for M1 (priv_state dropped); 50 for M2 (priv_state kept).
        seeds:          RNG seeds for reset; default [42, 99, 7].
        verbose:        Print per-seed results.

    Returns:
        SuiteResult: dict[traj_name][condition_name] = ConditionResult.

    Note on obs_dim:
        _build_obs always returns 50-dim obs (M2). For M1 (obs_dim=42), this
        function truncates to the first 42 dims [e_W(30), v(3), R(9)].
        This is correct because the first 42 dims are identical in M1 and M2.

    render_depth / figures_dir:
        When render_depth=True, env must be a DepthVecEnv with a render_single()
        method. After the first episode reset of each trajectory, one depth frame is
        rendered and saved as PNG to figures_dir (default: notes/figures/).
        This is a strict no-op when render_depth=False — the lax.scan loop and all
        eval logic are unchanged; no depth-related code executes.
    """
    from iron_man_drone.envs.quadrotor_env import (
        EPISODE_STEPS, _build_obs, MIN_HEIGHT, MAX_HEIGHT_ABOVE_REF, MAX_TILT_RAD,
    )
    from iron_man_drone.envs.trajectories import get_reference_pos

    drone_id = env.mj_model.body("drone").id
    cos_max_tilt = float(jnp.cos(MAX_TILT_RAD))

    def _precompute_refs(traj, offset_steps: int) -> jnp.ndarray:
        steps = jnp.arange(EPISODE_STEPS, dtype=jnp.int32) + offset_steps
        return jnp.array(jax.vmap(lambda s: get_reference_pos(traj, s)[:2])(steps))

    @functools.partial(jax.jit, static_argnums=())
    def _eval_episode(
        actor_params,
        reset_key: jnp.ndarray,
        priv_state: jnp.ndarray,
        eval_traj,
        ref_xy: jnp.ndarray,
        offset_steps: jnp.ndarray,
    ):
        state, _, _ = env._reset_fn(reset_key)
        state = state._replace(
            traj=eval_traj,
            priv_state=priv_state,
            rotor_efficiency=priv_state[:4],
            mass_scale=priv_state[4],
            kf_multiplier=jnp.ones(()),
            step=jnp.int32(offset_steps),
        )
        full_obs, _ = _build_obs(
            state.mjx_data, eval_traj, state.step, drone_id, priv_state
        )
        a_obs = full_obs[:obs_dim]

        _cos_tilt = jnp.array(cos_max_tilt)

        def scan_step(carry, ref_xy_t):
            state, a_obs, already_done = carry
            mean, _ = actor_apply_fn(actor_params, a_obs[None])
            action = mean[0]
            # Use crash-only done — step timeout fires early with T/4 offset.
            new_state, new_full_obs, _, _, _ = env._step_fn(state, action)
            new_state = new_state._replace(traj=eval_traj)
            new_a_obs = new_full_obs[:obs_dim]

            pos = new_state.mjx_data.xpos[drone_id]
            body_z_z = new_state.mjx_data.xmat[drone_id].reshape(-1)[8]
            horiz_dist = jnp.linalg.norm(pos[:2] - ref_xy_t)
            crash = (
                (pos[2] < MIN_HEIGHT)
                | (horiz_dist > MAX_HEIGHT_ABOVE_REF)
                | (jnp.abs(pos[2] - 1.0) > MAX_HEIGHT_ABOVE_REF)
                | (body_z_z < _cos_tilt)
            )
            error = jnp.linalg.norm(pos[:2] - ref_xy_t)
            active = ~already_done
            return (new_state, new_a_obs, already_done | crash), (
                jnp.where(active, error, 0.0),
                active,
            )

        _, (errors, active_mask) = jax.lax.scan(
            scan_step, (state, a_obs, jnp.bool_(False)), ref_xy
        )
        n = jnp.sum(active_mask)
        med = jnp.where(n > 0, jnp.sum(errors) / n.astype(jnp.float32), jnp.nan)
        return med, n

    # JIT warmup with a dummy figure_eight_normal trajectory
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    from iron_man_drone.envs.quadrotor_env import DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS
    _dummy_traj = make_figure_eight_trajectory(DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal")
    _dummy_ref  = _precompute_refs(_dummy_traj, 0)
    _dummy_priv = nominal_condition().priv_state
    if verbose:
        print("Warming up JIT (~1 min)...")
    _eval_episode(actor_params, jax.random.PRNGKey(0), _dummy_priv, _dummy_traj, _dummy_ref, jnp.int32(0))
    if verbose:
        print("JIT warmed up.\n")

    # Depth rendering setup (no-op when render_depth=False)
    if render_depth:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _fig_dir = figures_dir or (REPO_ROOT / "notes" / "figures")
        _fig_dir.mkdir(parents=True, exist_ok=True)

    results: SuiteResult = {}

    for tc in traj_configs:
        results[tc.name] = {}
        ref_xy = _precompute_refs(tc.traj, tc.offset_steps)
        offset_tag = f"[T/4 offset={tc.offset_steps}]" if tc.offset_steps > 0 else "[t=0]"
        if verbose:
            print(f"── {tc.name} {offset_tag}")

        # Render one depth frame from the initial state of this trajectory (first seed).
        # The lax.scan loop is unchanged — this render happens at the Python level before
        # the scan runs. render_depth=False → this block does not execute.
        if render_depth and hasattr(env, "render_single"):
            _init_state, _, _ = env._reset_fn(jax.random.PRNGKey(seeds[0]))
            _init_state = _init_state._replace(
                traj=tc.traj,
                step=jnp.int32(tc.offset_steps),
            )
            _depth_frame = env.render_single(_init_state)   # (64, 64) float32
            _png_path = _fig_dir / f"depth_{tc.name}.png"
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(_depth_frame, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{tc.name}  (0=near, 1={env.DEPTH_MAX_M if hasattr(env, 'DEPTH_MAX_M') else 5.0:.0f}m)")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(_png_path, dpi=100)
            plt.close(fig)
            if verbose:
                print(f"  [depth] saved {_png_path.name}")

        for cond in conditions:
            if verbose:
                print(f"  {cond.name}:")
            seed_results: list[EpisodeResult] = []
            for seed in seeds:
                _med, _n = _eval_episode(
                    actor_params,
                    jax.random.PRNGKey(seed),
                    cond.priv_state,
                    tc.traj,
                    ref_xy,
                    jnp.int32(tc.offset_steps),
                )
                med = float(_med)
                n   = int(_n)
                crashed = n < EPISODE_STEPS
                seed_results.append(EpisodeResult(med=med, n_steps=n, crashed=crashed))
                if verbose:
                    print(f"      seed {seed:3d}: {med:.4f} m  ({n} steps)")

            meds = [r.med for r in seed_results]
            mean_med = float(np.mean(meds))
            results[tc.name][cond.name] = ConditionResult(
                mean_med=mean_med,
                per_seed=seed_results,
            )
            if verbose:
                print(f"    → {cond.name}: {mean_med:.4f} m  [{min(meds):.4f}, {max(meds):.4f}]")

        if verbose:
            print()

    return results


def print_suite_summary(results: SuiteResult, title: str = "Eval Summary") -> None:
    """Print a formatted summary table from SuiteResult."""
    cond_names = list(next(iter(results.values())).keys())
    header = f"  {'Trajectory':<24}" + "".join(f"{c:>12}" for c in cond_names)
    print("=" * max(72, len(header) + 4))
    print(f"  {title}")
    print("=" * max(72, len(header) + 4))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for traj_name, cond_results in results.items():
        row = f"  {traj_name:<24}"
        for cond_name in cond_names:
            cr = cond_results[cond_name]
            crash_mark = "*" if cr.any_crashed else " "
            row += f"{cr.mean_med:>10.4f}m{crash_mark}"
        print(row)
    print("=" * max(72, len(header) + 4))
    print("  * = at least one seed crashed")


# ---------------------------------------------------------------------------
# Checkpoint loader (shared by wrappers)
# ---------------------------------------------------------------------------

def load_actor_from_checkpoint(
    checkpoint_path: str | Path,
    obs_dim: int,
    action_dim: int = 4,
    hidden_dim: int = 256,
    num_layers: int = 3,
) -> tuple[object, object]:
    """
    Load actor from an Orbax checkpoint directory.

    Reads config_frozen.yaml from the run directory (checkpoint_path/../../config_frozen.yaml).
    Falls back to the provided obs_dim if config is missing.

    Returns:
        (actor_state, actor_apply_fn) where actor_apply_fn(params, obs) -> (mean, log_std).
    """
    import yaml
    import orbax.checkpoint as ocp
    from iron_man_drone.policy.ppo import PPOConfig, create_train_states

    ckpt_path = Path(checkpoint_path).resolve()
    config_path = ckpt_path.parent.parent / "config_frozen.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        actor_dim  = cfg["observation"]["actor_dim"]
        critic_dim = cfg["observation"]["critic_dim"]
        hidden     = cfg["network"]["hidden_dim"]
        layers     = cfg["network"]["num_layers"]
    else:
        actor_dim  = obs_dim
        critic_dim = obs_dim + 1
        hidden     = hidden_dim
        layers     = num_layers

    ppo_cfg = PPOConfig(
        actor_obs_dim=actor_dim,
        critic_obs_dim=critic_dim,
        action_dim=action_dim,
        hidden_dim=hidden,
        num_layers=layers,
    )
    _, _, actor_state, _ = create_train_states(jax.random.PRNGKey(0), ppo_cfg)

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(str(ckpt_path))
    actor_state = actor_state.replace(params=restored["actor"]["params"])

    @jax.jit
    def actor_apply_fn(params, obs):
        return actor_state.apply_fn(params, obs)

    # Warm up actor JIT
    dummy = jnp.zeros((1, actor_dim))
    actor_apply_fn(actor_state.params, dummy)

    return actor_state, actor_apply_fn


# ---------------------------------------------------------------------------
# Smoke test (CI / regression guard)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # eval_suite.py → evaluation/ → iron_man_drone/ → src/ → repo root
_M1_DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments/m1_3_polynomial_fix"
    / "m1_3_polynomial_fix_1777900285/checkpoints/epoch_013000"
)


def smoke_test(
    checkpoint_path: str | Path | None = None,
    expected_med: float = 0.040,
    tol: float = 0.003,
    verbose: bool = True,
) -> bool:
    """
    Regression guard: run M1.3 figure_eight_normal (nominal, 3 seeds) and assert
    mean MED is within tol of expected_med.

    Uses the T/4-corrected methodology. Default checkpoint is M1.3 epoch_013000.

    Note on expected_med: the published M1.3 number is 0.037 m (from eval_m1_full.py,
    CPU mujoco). The lax.scan GPU MJX eval gives ~0.040 m for the same policy — a
    systematic ~0.003 m offset from the physics backend difference. expected_med=0.040
    is the correct GPU-eval baseline for this smoke test.

    Returns True if the assertion passes, raises AssertionError otherwise.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else _M1_DEFAULT_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"Smoke test checkpoint not found: {ckpt}")

    if verbose:
        print(f"Smoke test: {ckpt.name}  expected_med={expected_med:.3f} ± {tol:.3f}")

    # M1 actor: 42-dim obs. VecEnv with nominal DR.
    from iron_man_drone.envs.quadrotor_env import EPISODE_STEPS, DT, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS

    class _Cfg:
        num_envs = 1

    from iron_man_drone.envs.quadrotor_env import VecEnv
    env = VecEnv(_Cfg(), fault_prob=0.0, eta_min=0.5, mass_lo=1.0, mass_hi=1.0)

    actor_state, actor_apply_fn = load_actor_from_checkpoint(str(ckpt), obs_dim=42)

    lookahead = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
    offset = F8_OFFSETS["normal"]
    from iron_man_drone.envs.trajectories import make_figure_eight_trajectory
    traj = make_figure_eight_trajectory(DT, EPISODE_STEPS + offset + lookahead + 5, lookahead, speed="normal")
    tc = TrajConfig("figure_eight_normal", traj, offset)
    cond = nominal_condition()

    results = run_eval_suite(
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_state.params,
        env=env,
        traj_configs=[tc],
        conditions=[cond],
        obs_dim=42,
        verbose=verbose,
    )

    mean_med = results["figure_eight_normal"]["nominal"].mean_med
    lo, hi = expected_med - tol, expected_med + tol
    ok = lo <= mean_med <= hi
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"Smoke test: MED={mean_med:.4f} m  expected=[{lo:.3f}, {hi:.3f}]  → {status}")
    assert ok, (
        f"Smoke test FAILED: figure_eight_normal nominal MED={mean_med:.4f} m, "
        f"expected {expected_med:.3f} ± {tol:.3f}"
    )
    return True
