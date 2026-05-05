"""
MuJoCo MJX quadrotor environment — SimpleFlight recipe (M1).

Parallel envs via jax.vmap over (state, action) with a shared mjx_model.
All functions are pure JAX — jit-compilable, vmappable.

Observation (matching SimpleFlight paper Section III-B exactly):
  Actor  (42-dim): [e^W (30), v (3), R (9)]
  Critic (43-dim): [e^W (30), v (3), R (9), k (1)]
  where:
    e^W  = relative positions to next 10 reference points in world frame
    v    = linear velocity (world frame)
    R    = rotation matrix, flattened (body → world)
    k    = normalized timestep (0 → 1 over episode)  [CRITIC ONLY]

Body rates ω are NOT in the actor obs — paper Section III-B defines
o_t = [e^W_t, v_t, R_t] ∈ ℝ^42 with no ω term.

Action (4-dim, Gaussian policy):
  [ω_x^d, ω_y^d, ω_z^d, c]  (raw, unbounded — squashed in CTBR controller)
"""

from __future__ import annotations
import os
from functools import partial
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


# ── Constants ─────────────────────────────────────────────────────────────────

SIM_FREQ = 100           # Hz
DT = 1.0 / SIM_FREQ     # s
EPISODE_STEPS = 1000     # 10 seconds at 100 Hz
LOOKAHEAD_N = 10         # reference window size
LOOKAHEAD_DT_STEPS = 5  # 5 steps = 50 ms per lookahead point

# Termination bounds
MAX_HEIGHT_ABOVE_REF = 5.0   # m — bounding box radius
MIN_HEIGHT = 0.05             # m — ground crash threshold
MAX_TILT_RAD = jnp.pi / 3.0  # 60° — max pitch/roll before episode ends

ACTOR_OBS_DIM = 42   # 30 + 3 + 9  (paper Section III-B: no body rates)
CRITIC_OBS_DIM = 43  # 42 + 1


# ── State type ────────────────────────────────────────────────────────────────

class EnvState(NamedTuple):
    mjx_data: mjx.Data           # MuJoCo simulation state
    rotor_speeds: jnp.ndarray    # (4,) [rad/s]
    prev_action: jnp.ndarray     # (4,) previous CTBR action (for smoothness reward)
    traj: Trajectory             # reference trajectory for this episode
    step: jnp.ndarray            # scalar int — current episode step
    done: jnp.ndarray            # scalar bool


# ── Model loading (once, outside JAX) ─────────────────────────────────────────

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build actor obs (42-dim) and critic obs (43-dim).
    Matches SimpleFlight paper Section III-B exactly:
      actor  = [e^W (30), v (3), R (9)]
      critic = [e^W (30), v (3), R (9), k (1)]
    """
    # Position in world frame
    pos = mjx_data.xpos[drone_body_id]           # (3,)

    # Rotation matrix body → world, flattened to (9,) for obs
    # MuJoCo >= 3.2 returns (3,3); older versions return (9,) — reshape handles both
    R = mjx_data.xmat[drone_body_id].reshape(-1)  # (9,)

    # Linear velocity (world frame) — qvel[:3] for free joint
    v = mjx_data.qvel[:3]                        # (3,)

    # Relative positions to next 10 reference points in world frame
    ref_window = get_reference_window(           # (10, 3)
        traj, step,
        lookahead_n=LOOKAHEAD_N,
        lookahead_steps_per_point=LOOKAHEAD_DT_STEPS,
    )
    e_W = (ref_window - pos[None, :]).reshape(-1)  # (30,)

    # Normalized timestep (for critic ONLY — causes OOD failures if in actor)
    k = step.astype(jnp.float32) / EPISODE_STEPS  # scalar in [0, 1]

    actor_obs = jnp.concatenate([e_W, v, R])                    # (42,)
    critic_obs = jnp.concatenate([actor_obs, jnp.array([k])])   # (43,)

    return actor_obs, critic_obs


# ── Reward ────────────────────────────────────────────────────────────────────

def _compute_reward(
    pos: jnp.ndarray,              # (3,) current position
    ref_pos: jnp.ndarray,          # (3,) reference position
    action: jnp.ndarray,           # (4,) current CTBR action
    prev_action: jnp.ndarray,      # (4,) previous CTBR action
    kf_multiplier: float = 1.0,
    lambda_smooth: float = 0.4,
) -> tuple[jnp.ndarray, dict]:
    """
    r_total = r_task + λ * r_smooth
    r_task  = exp(-||pos - ref||²) ∈ [0, 1]   (soft negative L2)
    r_smooth = exp(-||u_t - u_{t-1}||²) ∈ [0, 1]
    """
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
    """Episode ends if: height crash, bounding box exit, or excessive tilt."""
    pos = mjx_data.xpos[drone_body_id]

    # Height crash
    height_crash = pos[2] < MIN_HEIGHT

    # Bounding box relative to reference position
    horiz_dist = jnp.linalg.norm(pos[:2] - ref_pos[:2])
    vert_dist = jnp.abs(pos[2] - ref_pos[2])
    bbox_exit = (horiz_dist > MAX_HEIGHT_ABOVE_REF) | (vert_dist > MAX_HEIGHT_ABOVE_REF)

    # Tilt check via z-axis of rotation matrix
    # Body z-axis in world frame = column 2 of the rotation matrix.
    # reshape(-1) handles both (3,3) [MuJoCo >= 3.2] and (9,) [older].
    R_flat = mjx_data.xmat[drone_body_id].reshape(-1)
    body_z_world = jnp.array([R_flat[2], R_flat[5], R_flat[8]])
    cos_tilt = body_z_world[2]  # dot with world z = [0,0,1]
    tilt_crash = cos_tilt < jnp.cos(MAX_TILT_RAD)

    # Episode timeout
    timeout = step >= EPISODE_STEPS

    return height_crash | bbox_exit | tilt_crash | timeout


# ── Reset ─────────────────────────────────────────────────────────────────────

def make_reset_fn(
    mjx_model: mjx.Model,
    mj_model: mujoco.MjModel,
    cfg,
):
    drone_body_id = mj_model.body("drone").id

    def reset(key: jnp.ndarray, kf_multiplier: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Fresh MJX data
        mjx_data = mjx.make_data(mjx_model)

        # Randomize initial position slightly above origin
        init_pos = jax.random.uniform(k1, (3,), minval=-0.1, maxval=0.1)
        init_pos = init_pos.at[2].set(1.0 + jax.random.uniform(k2, (), minval=-0.05, maxval=0.05))

        # Set initial state: position, quaternion (upright), zero velocity
        qpos = jnp.concatenate([init_pos, jnp.array([1.0, 0.0, 0.0, 0.0])])  # w,x,y,z
        qvel = jnp.zeros(6)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        # Rotor speeds at hover
        hover_omega_sq = (MASS * GRAVITY * kf_multiplier) / (4.0 * KF * kf_multiplier)
        hover_omega = jnp.sqrt(jnp.clip(hover_omega_sq, 0.0, None)) * jnp.ones(4)

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
            poly_traj,
            zigzag_traj,
        )

        state = EnvState(
            mjx_data=mjx_data,
            rotor_speeds=hover_omega,
            prev_action=jnp.zeros(4),
            traj=traj,
            step=jnp.zeros((), dtype=jnp.int32),
            done=jnp.zeros((), dtype=jnp.bool_),
        )

        actor_obs, critic_obs = _build_obs(mjx_data, traj, jnp.zeros((), dtype=jnp.int32), drone_body_id)
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
        action: jnp.ndarray,       # (4,) CTBR action from policy
        kf_multiplier: jnp.ndarray,
    ) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One environment step.
        Returns: (new_state, actor_obs, critic_obs, reward, done)
        """
        # 1. CTBR → new rotor speeds (includes rate controller + motor dynamics)
        omega_current = state.mjx_data.qvel[3:6]
        new_rotor_speeds = ctbr_to_rotor_speeds(
            action, state.rotor_speeds, omega_current, DT
        )
        # Apply DR: scale rotor thrust by kf_multiplier
        effective_speeds = new_rotor_speeds * jnp.sqrt(kf_multiplier)

        # 2. Compute wrench in body frame
        force_body, torque_body = compute_wrench(effective_speeds)

        # 3. Rotate force and torque from body to world frame
        # xmat[body_id] stores the rotation matrix as 9 values (row-major R_bw)
        R = state.mjx_data.xmat[drone_body_id].reshape(3, 3)  # body → world
        force_world = R @ force_body
        torque_world = R @ torque_body

        # 4. Apply external wrench to drone body via xfrc_applied
        # xfrc_applied shape: (nbody, 6) = [force(3), torque(3)] in world frame
        xfrc = jnp.zeros((n_bodies, 6))
        xfrc = xfrc.at[drone_body_id, :3].set(force_world)
        xfrc = xfrc.at[drone_body_id, 3:].set(torque_world)
        mjx_data = state.mjx_data.replace(xfrc_applied=xfrc)

        # 5. Step MJX physics
        mjx_data = mjx.step(mjx_model, mjx_data)

        new_step = state.step + 1

        # 6. Reward
        pos = mjx_data.xpos[drone_body_id]
        ref_pos = get_reference_pos(state.traj, new_step)
        reward, _ = _compute_reward(pos, ref_pos, action, state.prev_action)

        # 7. Termination
        done = _check_done(mjx_data, ref_pos, drone_body_id, new_step)

        # 8. New state
        new_state = EnvState(
            mjx_data=mjx_data,
            rotor_speeds=new_rotor_speeds,
            prev_action=action,
            traj=state.traj,
            step=new_step,
            done=done,
        )

        # 9. Observations
        actor_obs, critic_obs = _build_obs(mjx_data, state.traj, new_step, drone_body_id)

        return new_state, actor_obs, critic_obs, reward, done

    return step


# ── Vectorized environment factory ────────────────────────────────────────────

class VecEnv:
    """
    Batched environment using jax.vmap.
    All methods accept/return batched arrays (leading num_envs dimension).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.mj_model, self.mjx_model = load_mjx_model()
        self._reset_fn = make_reset_fn(self.mjx_model, self.mj_model, cfg)
        self._step_fn = make_step_fn(self.mjx_model, self.mj_model, cfg)

        # JIT-compiled vectorized versions
        self.reset = jax.jit(jax.vmap(self._reset_fn))
        self.step = jax.jit(jax.vmap(self._step_fn))

    def batch_reset(self, keys: jnp.ndarray, kf_multipliers: jnp.ndarray):
        """keys: (N,) PRNGKeys, kf_multipliers: (N,)"""
        return self.reset(keys, kf_multipliers)

    def batch_step(
        self,
        states: EnvState,
        actions: jnp.ndarray,    # (N, 4)
        kf_multipliers: jnp.ndarray,  # (N,)
    ):
        return self.step(states, actions, kf_multipliers)
