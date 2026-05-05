"""
MuJoCo MJX quadrotor environment — M2 (RMA fault-tolerant variant).

Parallel envs via jax.vmap over state with a shared mjx_model.
All functions are pure JAX — jit-compilable, vmappable.

M2 changes from M1.3:
  - EnvState carries all DR params (kf_multiplier, rotor_efficiency,
    mass_scale, priv_state). Sampled at reset, constant within episode.
  - Per-rotor efficiency applied to thrust in step().
  - Mass variation applied as an extra external force in step().
  - Privileged state vector priv_state (8-dim) appended to observations.
  - Reset takes only key (no kf_multiplier arg) — all DR sampled internally.
  - Step takes only (state, action) — DR read from state.

Observation (M2):
  Actor  (50-dim): [e^W (30), v (3), R (9), priv_state (8)]
  Critic (51-dim): [e^W (30), v (3), R (9), priv_state (8), k (1)]

  priv_state = [η1, η2, η3, η4, mass_scale, Fx, Fy, Fz]
    η_i ∈ [0.5,1.0] for the degraded rotor, 1.0 otherwise
    mass_scale ∈ [0.8, 1.2]
    Fx, Fy, Fz = 0 in Phase 1 (wind excluded from Phase 1 training)

  In Phase 1 (this commit): priv_state is passed directly to actor as the
  z-slot (no encoder yet). When μ is added in the next commit, the actor
  will receive z = μ(priv_state) instead, but actor_obs_dim stays 50.
"""

from __future__ import annotations
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from ..control.ctbr_controller import (
    ctbr_to_rotor_speeds,
    compute_wrench,
    HOVER_THRUST,
    MASS,
    GRAVITY,
    KF,
)
from .trajectories import (
    Trajectory,
    get_reference_pos,
    get_reference_window,
    sample_polynomial_trajectory,
    sample_zigzag_trajectory,
)
from ..utils.domain_randomization import sample_m2_dr_params


# ── Constants ─────────────────────────────────────────────────────────────────

SIM_FREQ = 100
DT = 1.0 / SIM_FREQ
EPISODE_STEPS = 1000
LOOKAHEAD_N = 10
LOOKAHEAD_DT_STEPS = 5

MAX_HEIGHT_ABOVE_REF = 5.0
MIN_HEIGHT = 0.05
MAX_TILT_RAD = jnp.pi / 3.0

PRIV_STATE_DIM = 8   # [η1,η2,η3,η4, mass_scale, Fx,Fy,Fz]
ACTOR_OBS_DIM = 50   # 30 + 3 + 9 + 8
CRITIC_OBS_DIM = 51  # 50 + 1


# ── State type ────────────────────────────────────────────────────────────────

class EnvState(NamedTuple):
    mjx_data: mjx.Data
    rotor_speeds: jnp.ndarray      # (4,)
    prev_action: jnp.ndarray       # (4,)
    traj: Trajectory
    step: jnp.ndarray              # scalar int
    done: jnp.ndarray              # scalar bool
    # M2: per-episode DR params, constant within episode
    kf_multiplier: jnp.ndarray     # scalar
    rotor_efficiency: jnp.ndarray  # (4,)
    mass_scale: jnp.ndarray        # scalar
    priv_state: jnp.ndarray        # (8,)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_mjx_model(xml_path: str | None = None) -> tuple[mujoco.MjModel, mjx.Model]:
    if xml_path is None:
        xml_path = os.path.join(os.path.dirname(__file__), "crazyflie.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = DT
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


# ── Observation builder ───────────────────────────────────────────────────────

def _build_obs(
    mjx_data: mjx.Data,
    traj: Trajectory,
    step: jnp.ndarray,
    drone_body_id: int,
    priv_state: jnp.ndarray,   # (8,) privileged state vector
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build actor obs (50-dim) and critic obs (51-dim).

    M2:
      actor  = [e^W (30), v (3), R (9), priv_state (8)]
      critic = [e^W (30), v (3), R (9), priv_state (8), k (1)]

    The 8-dim priv_state slot is:
      - Phase 1 (this commit): the raw privileged state e_t (ground truth)
      - Phase 1 (next commit): z = μ(e_t) from the privileged encoder
      - Phase 2 deployment: ẑ = ϕ(history) from the adaptation encoder
    The actor_obs_dim=50 stays fixed across all phases.
    """
    pos = mjx_data.xpos[drone_body_id]
    R = mjx_data.xmat[drone_body_id].reshape(-1)   # (9,)
    v = mjx_data.qvel[:3]

    ref_window = get_reference_window(
        traj, step,
        lookahead_n=LOOKAHEAD_N,
        lookahead_steps_per_point=LOOKAHEAD_DT_STEPS,
    )
    e_W = (ref_window - pos[None, :]).reshape(-1)   # (30,)

    k = step.astype(jnp.float32) / EPISODE_STEPS

    actor_obs = jnp.concatenate([e_W, v, R, priv_state])           # (50,)
    critic_obs = jnp.concatenate([actor_obs, jnp.array([k])])       # (51,)

    return actor_obs, critic_obs


# ── Reward ────────────────────────────────────────────────────────────────────

def _compute_reward(
    pos: jnp.ndarray,
    ref_pos: jnp.ndarray,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    lambda_smooth: float = 0.4,
) -> tuple[jnp.ndarray, dict]:
    dist_sq = jnp.sum((pos - ref_pos) ** 2)
    r_task = jnp.exp(-dist_sq)
    action_diff_sq = jnp.sum((action - prev_action) ** 2)
    r_smooth = jnp.exp(-action_diff_sq)
    r_total = r_task + lambda_smooth * r_smooth
    return r_total, {"r_task": r_task, "r_smooth": r_smooth}


# ── Termination ───────────────────────────────────────────────────────────────

def _check_done(
    mjx_data: mjx.Data,
    ref_pos: jnp.ndarray,
    drone_body_id: int,
    step: jnp.ndarray,
) -> jnp.ndarray:
    pos = mjx_data.xpos[drone_body_id]
    height_crash = pos[2] < MIN_HEIGHT
    horiz_dist = jnp.linalg.norm(pos[:2] - ref_pos[:2])
    vert_dist = jnp.abs(pos[2] - ref_pos[2])
    bbox_exit = (horiz_dist > MAX_HEIGHT_ABOVE_REF) | (vert_dist > MAX_HEIGHT_ABOVE_REF)
    R_flat = mjx_data.xmat[drone_body_id].reshape(-1)
    body_z_world = jnp.array([R_flat[2], R_flat[5], R_flat[8]])
    cos_tilt = body_z_world[2]
    tilt_crash = cos_tilt < jnp.cos(MAX_TILT_RAD)
    timeout = step >= EPISODE_STEPS
    return height_crash | bbox_exit | tilt_crash | timeout


# ── Reset ─────────────────────────────────────────────────────────────────────

def make_reset_fn(
    mjx_model: mjx.Model,
    mj_model: mujoco.MjModel,
    cfg,
    fault_prob: float = 0.7,
    eta_min: float = 0.5,
    mass_lo: float = 0.8,
    mass_hi: float = 1.2,
):
    """
    Args:
        fault_prob: probability of a rotor fault episode (0.0 = nominal only)
        eta_min/mass_lo/mass_hi: DR ranges; set fault_prob=0 and mass_lo=mass_hi=1
            for nominal-only mode used in validation gate 4.
    """
    drone_body_id = mj_model.body("drone").id

    def reset(key: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        k_dr, k1, k2, k3, k4 = jax.random.split(key, 5)

        # Sample all DR params for this episode
        dr = sample_m2_dr_params(
            k_dr,
            fault_prob=fault_prob,
            eta_min=eta_min,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
        )

        # Fresh MJX data
        mjx_data = mjx.make_data(mjx_model)

        # Initial position: small random offset around (0, 0, 1)
        init_pos = jax.random.uniform(k1, (3,), minval=-0.1, maxval=0.1)
        init_pos = init_pos.at[2].set(
            1.0 + jax.random.uniform(k2, (), minval=-0.05, maxval=0.05)
        )
        qpos = jnp.concatenate([init_pos, jnp.array([1.0, 0.0, 0.0, 0.0])])
        qvel = jnp.zeros(6)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        # Hover rotor speeds adjusted for mass_scale.
        # kf_multiplier and rotor_efficiency deviations cause small initial drift
        # that the policy corrects within a few steps — acceptable.
        hover_omega = jnp.sqrt(dr.mass_scale * MASS * GRAVITY / (4.0 * KF)) * jnp.ones(4)

        # Sample trajectory (50/50 polynomial vs zigzag)
        use_zigzag = jax.random.uniform(k3) > 0.5
        poly_traj = sample_polynomial_trajectory(
            k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
        )
        zigzag_traj = sample_zigzag_trajectory(
            k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS
        )
        traj = jax.tree_util.tree_map(
            lambda p, z: jnp.where(use_zigzag, z, p),
            poly_traj, zigzag_traj,
        )

        state = EnvState(
            mjx_data=mjx_data,
            rotor_speeds=hover_omega,
            prev_action=jnp.zeros(4),
            traj=traj,
            step=jnp.zeros((), dtype=jnp.int32),
            done=jnp.zeros((), dtype=jnp.bool_),
            kf_multiplier=dr.kf_multiplier,
            rotor_efficiency=dr.rotor_efficiency,
            mass_scale=dr.mass_scale,
            priv_state=dr.priv_state,
        )

        actor_obs, critic_obs = _build_obs(
            mjx_data, traj, jnp.zeros((), dtype=jnp.int32), drone_body_id, dr.priv_state
        )
        return state, actor_obs, critic_obs

    return reset


# ── Step ──────────────────────────────────────────────────────────────────────

def make_step_fn(
    mjx_model: mjx.Model,
    mj_model: mujoco.MjModel,
    cfg,
):
    drone_body_id = mj_model.body("drone").id
    n_bodies = mj_model.nbody

    def step(
        state: EnvState,
        action: jnp.ndarray,    # (4,) CTBR
    ) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # 1. CTBR → new rotor speeds
        omega_current = state.mjx_data.qvel[3:6]
        new_rotor_speeds = ctbr_to_rotor_speeds(
            action, state.rotor_speeds, omega_current, DT
        )

        # 2. Apply per-rotor efficiency + global kf_multiplier.
        #    thrust_i = KF * (Ω_i * sqrt(kf_mult * η_i))² = KF * kf_mult * η_i * Ω_i²
        effective_speeds = new_rotor_speeds * jnp.sqrt(
            state.kf_multiplier * state.rotor_efficiency
        )

        # 3. Compute wrench in body frame
        force_body, torque_body = compute_wrench(effective_speeds)

        # 4. Rotate to world frame
        R = state.mjx_data.xmat[drone_body_id].reshape(3, 3)
        force_world = R @ force_body
        torque_world = R @ torque_body

        # 5. Apply forces via xfrc_applied.
        #    Mass variation: extra downward force = (mass_scale - 1) * m * g.
        #    This simulates a heavier drone without modifying the shared MJX model.
        extra_weight = (state.mass_scale - 1.0) * MASS * GRAVITY
        force_world_total = force_world + jnp.array([0.0, 0.0, -extra_weight])

        xfrc = jnp.zeros((n_bodies, 6))
        xfrc = xfrc.at[drone_body_id, :3].set(force_world_total)
        xfrc = xfrc.at[drone_body_id, 3:].set(torque_world)
        mjx_data = state.mjx_data.replace(xfrc_applied=xfrc)

        # 6. Step physics
        mjx_data = mjx.step(mjx_model, mjx_data)
        new_step = state.step + 1

        # 7. Reward
        pos = mjx_data.xpos[drone_body_id]
        ref_pos = get_reference_pos(state.traj, new_step)
        reward, _ = _compute_reward(pos, ref_pos, action, state.prev_action)

        # 8. Termination
        done = _check_done(mjx_data, ref_pos, drone_body_id, new_step)

        # 9. New state — DR fields carry through unchanged (constant per episode)
        new_state = EnvState(
            mjx_data=mjx_data,
            rotor_speeds=new_rotor_speeds,
            prev_action=action,
            traj=state.traj,
            step=new_step,
            done=done,
            kf_multiplier=state.kf_multiplier,
            rotor_efficiency=state.rotor_efficiency,
            mass_scale=state.mass_scale,
            priv_state=state.priv_state,
        )

        # 10. Observations
        actor_obs, critic_obs = _build_obs(
            mjx_data, state.traj, new_step, drone_body_id, state.priv_state
        )

        return new_state, actor_obs, critic_obs, reward, done

    return step


# ── Vectorized environment factory ────────────────────────────────────────────

class VecEnv:
    """
    Batched environment using jax.vmap.

    M2 interface change: reset(keys) and step(states, actions) — no kf_multipliers arg.
    All DR params are sampled inside reset() and carried in EnvState.
    """

    def __init__(self, cfg, fault_prob=0.7, eta_min=0.5, mass_lo=0.8, mass_hi=1.2):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.mj_model, self.mjx_model = load_mjx_model()
        self._reset_fn = make_reset_fn(
            self.mjx_model, self.mj_model, cfg,
            fault_prob=fault_prob, eta_min=eta_min, mass_lo=mass_lo, mass_hi=mass_hi,
        )
        self._step_fn = make_step_fn(self.mjx_model, self.mj_model, cfg)

        self.reset = jax.jit(jax.vmap(self._reset_fn))
        self.step = jax.jit(jax.vmap(self._step_fn))

    def batch_reset(self, keys: jnp.ndarray):
        """keys: (N,) PRNGKeys"""
        return self.reset(keys)

    def batch_step(self, states: EnvState, actions: jnp.ndarray):
        """actions: (N, 4)"""
        return self.step(states, actions)
