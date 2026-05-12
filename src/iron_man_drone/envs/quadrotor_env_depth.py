"""
DepthVecEnv — MJX physics + MJWarp depth rendering (M2.5).

Physics, controller, reward, termination and policy interface are identical to
quadrotor_env.py (M2).  Additions vs M2:
  - crazyflie_depth.xml: 16 mocap obstacle slots + forward-facing depth camera.
  - sample_obstacle_configs called in batch_reset at the Python level (numpy, not JAX).
  - DepthEnvState carries obstacle fields and the last rendered depth frame.
  - batch_render(): transfers MJX state → MJWarp, renders, returns (N, 64, 64) float32.

Depth tensor is computed but NOT consumed by actor or critic in M2.5.
Obstacle privileged state is stored in EnvState but NOT wired to policy in M2.5.
Both are available for M3 to wire in without changing the env interface.
"""

from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import NamedTuple

from .quadrotor_env import (
    DT,
    EPISODE_STEPS,
    LOOKAHEAD_N,
    LOOKAHEAD_DT_STEPS,
    _build_obs,
    _compute_reward,
    _check_done,
    make_step_fn,
    load_mjx_model,
    EnvState,
)
from ..control.ctbr_controller import MASS, GRAVITY, KF
from .trajectories import sample_polynomial_trajectory, sample_zigzag_trajectory
from ..utils.domain_randomization import sample_m2_dr_params
from ..utils.obstacle_randomization import (
    N_OBSTACLE_SLOTS,
    sample_obstacle_configs,
)

DEPTH_XML      = os.path.join(os.path.dirname(__file__), "crazyflie_depth.xml")
DEPTH_CAM_NAME = "depth_cam"
DEPTH_RES      = 64
DEPTH_MAX_M    = 5.0


# ── State type ─────────────────────────────────────────────────────────────────

class DepthEnvState(NamedTuple):
    # M2 fields — identical layout and semantics to EnvState in quadrotor_env.py
    mjx_data:         mjx.Data
    rotor_speeds:     jnp.ndarray   # (4,)
    prev_action:      jnp.ndarray   # (4,)
    traj:             object        # Trajectory NamedTuple
    step:             jnp.ndarray   # scalar int32
    done:             jnp.ndarray   # scalar bool
    kf_multiplier:    jnp.ndarray   # scalar
    rotor_efficiency: jnp.ndarray   # (4,)
    mass_scale:       jnp.ndarray   # scalar
    priv_state:       jnp.ndarray   # (8,)
    # M2.5 additions — carried every step, not consumed by actor/critic in M2.5
    obstacle_positions:    jnp.ndarray  # (N_OBSTACLE_SLOTS=16, 3) — centers; inactive at [100,100,100]
    obstacle_half_extents: jnp.ndarray  # (N_OBSTACLE_SLOTS=16, 3) — half-extents; inactive zeros
    n_obstacles:           jnp.ndarray  # scalar int32 — active obstacle count for M3 masking
    depth:                 jnp.ndarray  # (DEPTH_RES, DEPTH_RES) float32 in [0,1]; zeros until batch_render


# ── State conversion helpers ───────────────────────────────────────────────────

def _to_m2_state(s: DepthEnvState) -> EnvState:
    """Extract the M2-compatible base state (drops obstacle/depth fields)."""
    return EnvState(
        mjx_data=s.mjx_data,
        rotor_speeds=s.rotor_speeds,
        prev_action=s.prev_action,
        traj=s.traj,
        step=s.step,
        done=s.done,
        kf_multiplier=s.kf_multiplier,
        rotor_efficiency=s.rotor_efficiency,
        mass_scale=s.mass_scale,
        priv_state=s.priv_state,
    )


def _from_m2_state(new_m2: EnvState, old_depth: DepthEnvState) -> DepthEnvState:
    """Reconstruct DepthEnvState from a new M2 state, carrying obstacle fields through."""
    return DepthEnvState(
        mjx_data=new_m2.mjx_data,
        rotor_speeds=new_m2.rotor_speeds,
        prev_action=new_m2.prev_action,
        traj=new_m2.traj,
        step=new_m2.step,
        done=new_m2.done,
        kf_multiplier=new_m2.kf_multiplier,
        rotor_efficiency=new_m2.rotor_efficiency,
        mass_scale=new_m2.mass_scale,
        priv_state=new_m2.priv_state,
        obstacle_positions=old_depth.obstacle_positions,
        obstacle_half_extents=old_depth.obstacle_half_extents,
        n_obstacles=old_depth.n_obstacles,
        depth=jnp.zeros((DEPTH_RES, DEPTH_RES), dtype=jnp.float32),
    )


# ── Reset ─────────────────────────────────────────────────────────────────────

def make_depth_reset_fn(
    mjx_model: mjx.Model,
    mj_model:  mujoco.MjModel,
    cfg,
    n_obstacles: int,
    fault_prob: float = 0.7,
    eta_min:    float = 0.5,
    mass_lo:    float = 0.8,
    mass_hi:    float = 1.2,
):
    """
    Returns reset(key, obstacle_positions, obstacle_half_extents) -> (DepthEnvState, actor_obs, critic_obs).

    obstacle_positions    (N_OBSTACLE_SLOTS, 3) JAX float32 — sampled outside JIT by DepthVecEnv.batch_reset.
    obstacle_half_extents (N_OBSTACLE_SLOTS, 3) JAX float32.
    Both are vmapped over the env batch axis, so each env gets its own layout.
    """
    drone_body_id = mj_model.body("drone").id
    _n_obs_arr = jnp.array(n_obstacles, dtype=jnp.int32)

    def reset(
        key: jnp.ndarray,
        obstacle_positions:    jnp.ndarray,  # (N_OBSTACLE_SLOTS, 3)
        obstacle_half_extents: jnp.ndarray,  # (N_OBSTACLE_SLOTS, 3)
    ):
        k_dr, k1, k2, k3, k4 = jax.random.split(key, 5)

        dr = sample_m2_dr_params(
            k_dr, fault_prob=fault_prob, eta_min=eta_min, mass_lo=mass_lo, mass_hi=mass_hi,
        )

        mjx_data = mjx.make_data(mjx_model)

        init_pos = jax.random.uniform(k1, (3,), minval=-0.1, maxval=0.1)
        init_pos = init_pos.at[2].set(
            1.0 + jax.random.uniform(k2, (), minval=-0.05, maxval=0.05)
        )
        qpos = jnp.concatenate([init_pos, jnp.array([1.0, 0.0, 0.0, 0.0])])
        qvel = jnp.zeros(6)

        # Set obstacle mocap positions so MJX kinematics place them correctly.
        # Inactive slots are already at [100,100,100] from sample_obstacle_configs.
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, mocap_pos=obstacle_positions)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        hover_omega = jnp.sqrt(dr.mass_scale * MASS * GRAVITY / (4.0 * KF)) * jnp.ones(4)

        use_zigzag = jax.random.uniform(k3) > 0.5
        poly_traj   = sample_polynomial_trajectory(k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS)
        zigzag_traj = sample_zigzag_trajectory(    k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS)
        traj = jax.tree_util.tree_map(
            lambda p, z: jnp.where(use_zigzag, z, p), poly_traj, zigzag_traj
        )

        state = DepthEnvState(
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
            obstacle_positions=obstacle_positions,
            obstacle_half_extents=obstacle_half_extents,
            n_obstacles=_n_obs_arr,
            depth=jnp.zeros((DEPTH_RES, DEPTH_RES), dtype=jnp.float32),
        )

        actor_obs, critic_obs = _build_obs(
            mjx_data, traj, jnp.zeros((), dtype=jnp.int32), drone_body_id, dr.priv_state
        )
        return state, actor_obs, critic_obs

    return reset


# ── Step ──────────────────────────────────────────────────────────────────────

def make_depth_step_fn(
    mjx_model: mjx.Model,
    mj_model:  mujoco.MjModel,
    cfg,
):
    """
    Wraps M2's make_step_fn — reuses all physics/reward/termination logic exactly.
    Carries obstacle_positions, obstacle_half_extents, n_obstacles through each step.
    depth is zeroed each step; populated externally by DepthVecEnv.batch_render.
    """
    _inner_step = make_step_fn(mjx_model, mj_model, cfg)

    def step(
        state:  DepthEnvState,
        action: jnp.ndarray,    # (4,) CTBR
    ):
        new_m2, actor_obs, critic_obs, reward, done = _inner_step(
            _to_m2_state(state), action
        )
        new_state = _from_m2_state(new_m2, state)
        return new_state, actor_obs, critic_obs, reward, done

    return step


# ── Vectorized depth environment ───────────────────────────────────────────────

class DepthVecEnv:
    """
    Batched depth environment: MJX physics + MJWarp depth rendering.

    Interface mirrors VecEnv (quadrotor_env.py) with two additions:
      - batch_reset samples obstacle configs in Python before calling JIT'd reset.
      - batch_render() transfers MJX state → MJWarp and returns depth frames.

    MJWarp is initialized at construction time (GPU required at import of this env).
    """

    def __init__(
        self,
        cfg,
        n_obstacles: int   = 0,
        fault_prob:  float = 0.7,
        eta_min:     float = 0.5,
        mass_lo:     float = 0.8,
        mass_hi:     float = 1.2,
    ):
        self.cfg         = cfg
        self.num_envs    = cfg.num_envs
        self.n_obstacles = n_obstacles

        self.mj_model, self.mjx_model = load_mjx_model(DEPTH_XML)

        self.cam_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAM_NAME
        )

        _reset_fn = make_depth_reset_fn(
            self.mjx_model, self.mj_model, cfg, n_obstacles,
            fault_prob=fault_prob, eta_min=eta_min, mass_lo=mass_lo, mass_hi=mass_hi,
        )
        _step_fn = make_depth_step_fn(self.mjx_model, self.mj_model, cfg)

        # Raw (non-vmapped) functions — needed by eval_suite.run_eval_suite
        self._raw_reset_fn = _reset_fn   # (key, centers, half_extents) → (DepthEnvState, ao, co)
        self._step_fn      = _step_fn    # (DepthEnvState, action) → (DepthEnvState, ao, co, r, d)

        # vmap: reset takes (key, obs_pos, obs_he) — all batched over axis 0
        self._reset_jit = jax.jit(jax.vmap(_reset_fn, in_axes=(0, 0, 0)))
        self._step_jit  = jax.jit(jax.vmap(_step_fn))

        # JAX-traceable step/reset for collect_rollout (ppo.py lax.scan compatibility).
        # reset uses no obstacles (parked at 100) so it is a pure JAX function.
        # batch_reset does Python-level obstacle sampling and is used at episode start.
        _c_park  = jnp.full((N_OBSTACLE_SLOTS, 3), 100.0, dtype=jnp.float32)
        _he_zero = jnp.zeros((N_OBSTACLE_SLOTS, 3), dtype=jnp.float32)

        def _reset_no_obs(key):
            return _reset_fn(key, _c_park, _he_zero)

        self.step  = self._step_jit
        self.reset = jax.jit(jax.vmap(_reset_no_obs))

        self._init_mjwarp()

    # ── MJWarp initialization ─────────────────────────────────────────────────

    def _init_mjwarp(self):
        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, mj_data)

        self._mjw_model = mjw.put_model(self.mj_model)
        self._mjw_data  = mjw.put_data(
            self.mj_model, mj_data, nworld=self.num_envs, njmax=200
        )
        self._rc = mjw.create_render_context(
            self.mj_model,
            nworld=self.num_envs,
            cam_res=(DEPTH_RES, DEPTH_RES),
            render_depth=True,
            render_rgb=False,
            use_shadows=False,
        )
        self._depth_buf = wp.zeros(
            (self.num_envs, DEPTH_RES, DEPTH_RES), dtype=wp.float32, device="cuda:0"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def batch_reset(self, keys: jnp.ndarray):
        """
        Sample per-env obstacle layouts (numpy, outside JIT), then vmap reset.

        keys: (N, 2) uint32 PRNGKey batch.
        Returns (DepthEnvState, actor_obs, critic_obs).
        """
        N = int(keys.shape[0])
        seed = int(jax.random.randint(keys[0], shape=(), minval=0, maxval=2**30))
        rng  = np.random.default_rng(seed=seed)

        all_centers = np.empty((N, N_OBSTACLE_SLOTS, 3), dtype=np.float32)
        all_he      = np.empty((N, N_OBSTACLE_SLOTS, 3), dtype=np.float32)
        for i in range(N):
            all_centers[i], all_he[i] = sample_obstacle_configs(rng, self.n_obstacles)

        obs_pos = jnp.array(all_centers)   # (N, 16, 3)
        obs_he  = jnp.array(all_he)        # (N, 16, 3)

        return self._reset_jit(keys, obs_pos, obs_he)

    def batch_step(
        self,
        states:  DepthEnvState,
        actions: jnp.ndarray,   # (N, 4)
    ):
        """Step physics. Returns (DepthEnvState, actor_obs, critic_obs, reward, done)."""
        return self._step_jit(states, actions)

    def batch_render(self, states: DepthEnvState) -> jnp.ndarray:
        """
        Render depth frames for all envs.

        State transfer (Option A): MJX qpos + obstacle_positions → MJWarp, then
        mjw.forward computes kinematics, mjw.render produces depth images.
        Obstacle positions are written from EnvState directly (bypassing MJX mocap
        propagation), so render positions exactly match the episode layout.

        Returns: (N, DEPTH_RES, DEPTH_RES) float32 in [0, 1]
                 0 = nearest (0 m), 1 = farthest (DEPTH_MAX_M = 5 m).
        """
        import warp as wp
        import mujoco_warp as mjw

        # ── State transfer: MJX → MJWarp ─────────────────────────────────────

        # Drone DoF: qpos (N, 7) float32 — position (3) + quaternion (4)
        qpos_np = np.array(states.mjx_data.qpos)           # (N, 7)
        self._mjw_data.qpos.assign(qpos_np)

        # Obstacle positions via mocap_pos: (N, 16) vec3f
        # wp.from_numpy interprets (N, 16, 3) float32 as (N, 16) of vec3f
        mocap_np = np.array(states.obstacle_positions)      # (N, 16, 3)
        self._mjw_data.mocap_pos.assign(
            wp.from_numpy(mocap_np, dtype=wp.vec3f, device="cuda:0")
        )

        # ── Forward kinematics: body positions → geom positions ──────────────
        mjw.forward(self._mjw_model, self._mjw_data)

        # ── Render ────────────────────────────────────────────────────────────
        mjw.render(self._mjw_model, self._mjw_data, self._rc)
        mjw.get_depth(self._rc, self.cam_id, DEPTH_MAX_M, self._depth_buf)
        wp.synchronize()

        # Copy to JAX via numpy (avoids stale-buffer aliasing on the next render call)
        depth_np = self._depth_buf.numpy()                  # (N, 64, 64) float32 CPU
        return jnp.array(depth_np)                          # (N, 64, 64) float32 GPU

    # ── eval_suite compatibility ───────────────────────────────────────────────

    @property
    def _reset_fn(self):
        """
        Single-key reset with all obstacles parked — compatible with run_eval_suite.

        eval_suite uses n_obstacles=0 (obstacles parked at [100,100,100]) so that
        SC-1/SC-2 test dynamics parity between crazyflie_depth.xml and crazyflie.xml
        without obstacle interference.
        """
        _fn = self._raw_reset_fn
        _c  = jnp.full((N_OBSTACLE_SLOTS, 3), 100.0, dtype=jnp.float32)
        _he = jnp.zeros((N_OBSTACLE_SLOTS, 3), dtype=jnp.float32)

        def _reset(key: jnp.ndarray):
            return _fn(key, _c, _he)

        return _reset

    def render_single(self, state: DepthEnvState) -> np.ndarray:
        """
        Render depth for a single (unbatched) DepthEnvState.
        Adds/removes the batch dimension around batch_render.
        Returns (DEPTH_RES, DEPTH_RES) float32 numpy array in [0, 1].
        """
        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        depth = self.batch_render(batched)   # (1, 64, 64)
        return np.array(depth[0])            # (64, 64)
