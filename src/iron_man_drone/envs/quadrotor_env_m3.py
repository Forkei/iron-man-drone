"""
M3 environment — visual obstacle avoidance with joint fault-encoder training.

Extends DepthVecEnv (MJX physics + MJWarp depth rendering) with:
  - 16-bin min-pooled depth observation for actor
  - History buffer in state for joint encoder training
  - Obstacle proximity reward + crash termination
  - Multi-mode procedural scene generator (via scene_generator.py)
  - Asymmetric critic: gets K=5 nearest obstacle distances (ground truth)

Interface change from M2.5:
  batch_step returns (M3EnvState, base_obs_42, priv_state_8, reward, done)
  NOT full actor/critic obs — those are assembled in the training loop:
    actor_obs  (66): [base_obs(42), z_hat(8), depth_bins(16)]
    critic_obs (72): [actor_obs(66), k(1), obs_dists(5)]
  This allows the encoder to be applied outside the JIT-compiled step.

M3 spec note on obs dims: z slot is 8-dim (encoder output), not 32-dim.
M3_spec.md diagram was conceptual; this file is authoritative.
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
    DT, EPISODE_STEPS, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS,
    _compute_reward, _check_done, make_step_fn, load_mjx_model,
)
from .quadrotor_env_depth import (
    DEPTH_XML, DEPTH_CAM_NAME, DEPTH_RES, DEPTH_MAX_M,
    DepthEnvState, _to_m2_state,
)
from ..control.ctbr_controller import MASS, GRAVITY, KF
from .trajectories import (
    sample_polynomial_trajectory, sample_zigzag_trajectory,
    get_reference_window, get_reference_pos, DT as TRAJ_DT,
)
from ..utils.domain_randomization import sample_m2_dr_params
from ..utils.obstacle_randomization import N_OBSTACLE_SLOTS
from ..utils.scene_generator import (
    sample_mode, sample_scene, TRAINING_MODES, TRAINING_WEIGHTS,
)
from ..policy.encoder import (
    H, OBS_DIM as ENC_OBS_DIM, ACTION_DIM,
    build_history_window, denormalize_e_hat,
)

# ── Observation / reward constants ─────────────────────────────────────────────
OBS_BASE_DIM   = 42    # [e^W(30), v(3), R(9)] — encoder input, no z
ENCODER_DIM    = 8     # encoder output (matches priv_state dim)
DEPTH_N_BINS   = 16    # min-pooled horizontal depth sectors
ACTOR_OBS_DIM  = OBS_BASE_DIM + ENCODER_DIM + DEPTH_N_BINS  # 66
K_NEAREST      = 5     # nearest-obstacle distances in critic
CRITIC_OBS_DIM = ACTOR_OBS_DIM + 1 + K_NEAREST              # 72

def _min_dist_jax(
    drone_pos: jnp.ndarray,
    centers: jnp.ndarray,
    half_extents: jnp.ndarray,
    n_obstacles: jnp.ndarray,
) -> jnp.ndarray:
    """
    L∞ surface distance from drone_pos to nearest active obstacle.
    JAX-compatible with dynamic n_obstacles inside vmap/JIT.
    Returns jnp.inf when no obstacles are active.
    """
    diff   = jnp.abs(drone_pos - centers) - half_extents   # (N_SLOTS, 3)
    dists  = jnp.max(jnp.maximum(diff, 0.0), axis=1)       # (N_SLOTS,)
    active = jnp.arange(N_OBSTACLE_SLOTS) < n_obstacles
    return jnp.min(jnp.where(active, dists, jnp.inf))


D_SAFE   = 0.5    # m — proximity penalty onset
D_CRASH  = 0.15   # m — collision termination threshold

W_TRACK    = 2.0
W_SMOOTH   = 0.1
W_SURVIVE  = 0.01
W_OBSTACLE = 0.5
W_CRASH    = 10.0

# Max speed for M3 trajectory generator (spec: 3 m/s training cap)
M3_MAX_VEL = 3.0    # m/s at waypoints
M3_MAX_ACC = 6.0    # m/s²


# ── State type ─────────────────────────────────────────────────────────────────

class M3EnvState(NamedTuple):
    # Physics (M2 fields)
    mjx_data:         mjx.Data
    rotor_speeds:     jnp.ndarray   # (4,)
    prev_action:      jnp.ndarray   # (4,)
    traj:             object
    step:             jnp.ndarray   # scalar int32
    done:             jnp.ndarray   # scalar bool
    kf_multiplier:    jnp.ndarray
    rotor_efficiency: jnp.ndarray   # (4,)
    mass_scale:       jnp.ndarray
    priv_state:       jnp.ndarray   # (8,) ground-truth fault state
    # Obstacle fields (M2.5)
    obstacle_positions:    jnp.ndarray  # (N_OBSTACLE_SLOTS, 3)
    obstacle_half_extents: jnp.ndarray  # (N_OBSTACLE_SLOTS, 3)
    n_obstacles:           jnp.ndarray  # scalar int32
    depth:                 jnp.ndarray  # (64, 64) float32 in [0,1]; zeroed each step
    # Encoder history (M3) — ring buffers, oldest first
    obs_base_buf: jnp.ndarray   # (H=50, 42) — base obs history for encoder
    action_buf:   jnp.ndarray   # (H=50, 4)  — action history for encoder


# ── Depth binning ──────────────────────────────────────────────────────────────

def compute_depth_bins(depth: jnp.ndarray) -> jnp.ndarray:
    """
    (64, 64) float32 depth image in [0,1] → (DEPTH_N_BINS=16,) min-pooled bins.

    Divides horizontal axis into 16 equal sectors (4 px wide each at 64-px width).
    Takes the minimum depth value over all rows in each sector.
    Output in [0,1]: 1.0 = no obstacle within far-plane range.

    Follows Loquercio et al.: min-pool captures nearest obstacle per sector.
    """
    # depth: (H_px, W_px) = (64, 64)
    sector_w = DEPTH_RES // DEPTH_N_BINS   # = 4 pixels per bin
    # Reshape to (H_px, N_BINS, sector_w) then min over H and sector_w
    d = depth.reshape(DEPTH_RES, DEPTH_N_BINS, sector_w)   # (64, 16, 4)
    return jnp.min(d, axis=(0, 2))                          # (16,)


def compute_depth_bins_batch(depth_batch: jnp.ndarray) -> jnp.ndarray:
    """(N, 64, 64) → (N, 16)."""
    return jax.vmap(compute_depth_bins)(depth_batch)


# ── Obstacle distances for critic ─────────────────────────────────────────────

def compute_k_nearest_dists(
    drone_pos: jnp.ndarray,            # (3,)
    obstacle_positions: jnp.ndarray,   # (N_OBSTACLE_SLOTS, 3)
    obstacle_half_extents: jnp.ndarray,# (N_OBSTACLE_SLOTS, 3)
    n_obstacles: jnp.ndarray,          # scalar int32 (static when possible)
) -> jnp.ndarray:
    """
    K=5 nearest obstacle L∞ surface distances, sorted ascending.
    Returns (K,) with inf-padding for inactive slots.
    """
    pos = jnp.asarray(drone_pos)
    c   = obstacle_positions    # (N, 3)
    he  = obstacle_half_extents # (N, 3)

    diff  = jnp.abs(pos - c) - he                  # (N, 3)
    dists = jnp.max(jnp.maximum(diff, 0.0), axis=1) # (N,) L∞ SDF per slot

    # Mask inactive slots to a large finite value.
    # Using inf here would corrupt LayerNorm in the critic (inf * weight → ±inf,
    # then LayerNorm mean(+inf, -inf) = NaN → NaN critic values → NaN advantages
    # → NaN actor gradient after the very first PPO update).
    # 10.0 m is far beyond any obstacle interaction range (D_SAFE=0.5, D_CRASH=0.15).
    slot_active = jnp.arange(N_OBSTACLE_SLOTS) < n_obstacles
    dists = jnp.where(slot_active, dists, 10.0)

    # Top-K nearest (sorted ascending)
    return jnp.sort(dists)[:K_NEAREST]   # (K,)


# ── Observation construction ───────────────────────────────────────────────────

def build_m3_base_obs(
    mjx_data: mjx.Data,
    traj,
    step: jnp.ndarray,
    drone_body_id: int,
) -> jnp.ndarray:
    """
    42-dim base obs: [e^W(30), v(3), R(9)].
    Does NOT include z (encoder latent) or depth bins.
    This is both the encoder input and the first 42 dims of actor obs.
    """
    pos   = mjx_data.xpos[drone_body_id]           # (3,)
    R_mat = mjx_data.xmat[drone_body_id].reshape(-1) # (9,)
    v     = mjx_data.qvel[:3]                       # (3,)

    ref = get_reference_window(traj, step, LOOKAHEAD_N, LOOKAHEAD_DT_STEPS)  # (10, 3)
    e_W = (ref - pos[None, :]).reshape(-1)          # (30,)

    return jnp.concatenate([e_W, v, R_mat])         # (42,)


def build_full_obs(
    base_obs: jnp.ndarray,    # (42,)
    z_hat: jnp.ndarray,       # (8,)  encoder output (or priv_state in ground-truth mode)
    depth_bins: jnp.ndarray,  # (16,)
    k: jnp.ndarray,           # scalar, step / EPISODE_STEPS
    obs_dists: jnp.ndarray,   # (K_NEAREST,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble actor (66-dim) and critic (72-dim) observations.
    Called from the training loop after encoder + depth bins are computed.
    """
    actor_obs  = jnp.concatenate([base_obs, z_hat, depth_bins])          # (66,)
    critic_obs = jnp.concatenate([actor_obs, jnp.array([k]), obs_dists]) # (72,)
    return actor_obs, critic_obs


# ── Reward function ────────────────────────────────────────────────────────────

def _compute_m3_reward(
    pos: jnp.ndarray,
    ref_pos: jnp.ndarray,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    d_min: jnp.ndarray,       # scalar, min surface distance to nearest obstacle
    crashed: jnp.ndarray,     # bool scalar
) -> tuple[jnp.ndarray, dict]:
    """
    M3 reward: tracking + smoothness + survive + obstacle proximity + crash terminal.
    Weights per spec §5.5.
    """
    dist_sq  = jnp.sum((pos - ref_pos) ** 2)
    r_track  = jnp.exp(-dist_sq)

    action_diff_sq = jnp.sum((action - prev_action) ** 2)
    r_smooth = jnp.exp(-action_diff_sq)

    # Obstacle proximity: linear penalty onset at D_SAFE, reaches -1 at surface
    in_zone  = d_min < D_SAFE
    r_obs    = jnp.where(in_zone, -(D_SAFE - d_min) / D_SAFE, 0.0)

    r_crash  = jnp.where(crashed, -W_CRASH, 0.0)

    r_total = (
        W_TRACK   * r_track
        + W_SMOOTH  * r_smooth
        + W_SURVIVE
        + W_OBSTACLE * r_obs
        + r_crash
    )
    return r_total, {
        "r_track": r_track, "r_smooth": r_smooth,
        "r_obs": r_obs, "r_crash": r_crash,
    }


# ── Reset ──────────────────────────────────────────────────────────────────────

def make_m3_reset_fn(
    mjx_model: mjx.Model,
    mj_model:  mujoco.MjModel,
    cfg,
    fault_prob: float = 0.7,
    eta_min:    float = 0.5,
    mass_lo:    float = 0.8,
    mass_hi:    float = 1.2,
    override_traj=None,
):
    """
    Returns reset(key, obstacle_positions, obstacle_half_extents, n_obstacles)
    → (M3EnvState, base_obs_42, priv_state_8).

    obstacle_* arrays are sampled outside JIT by M3VecEnv.batch_reset.
    n_obstacles passed as a JAX scalar (vmapped per env).
    """
    drone_body_id = mj_model.body("drone").id

    def reset(
        key: jnp.ndarray,
        obstacle_positions:    jnp.ndarray,  # (N_OBSTACLE_SLOTS, 3)
        obstacle_half_extents: jnp.ndarray,  # (N_OBSTACLE_SLOTS, 3)
        n_obstacles:           jnp.ndarray,  # scalar int32
    ):
        k_dr, k1, k2, k3, k4 = jax.random.split(key, 5)

        dr = sample_m2_dr_params(k_dr, fault_prob=fault_prob, eta_min=eta_min,
                                  mass_lo=mass_lo, mass_hi=mass_hi)

        mjx_data = mjx.make_data(mjx_model)
        init_pos = jax.random.uniform(k1, (3,), minval=-0.1, maxval=0.1)
        init_pos = init_pos.at[2].set(1.0 + jax.random.uniform(k2, (), minval=-0.05, maxval=0.05))
        qpos = jnp.concatenate([init_pos, jnp.array([1.0, 0.0, 0.0, 0.0])])
        mjx_data = mjx_data.replace(
            qpos=qpos,
            qvel=jnp.zeros(6),
            mocap_pos=obstacle_positions,
        )
        mjx_data = mjx.forward(mjx_model, mjx_data)

        hover_omega = jnp.sqrt(dr.mass_scale * MASS * GRAVITY / (4.0 * KF)) * jnp.ones(4)

        if override_traj is not None:
            traj = override_traj
        else:
            use_zigzag = jax.random.uniform(k3) > 0.5
            poly_traj = sample_polynomial_trajectory(
                k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS,
                max_vel=M3_MAX_VEL, max_acc=M3_MAX_ACC,
            )
            zigzag_traj = sample_zigzag_trajectory(k4, DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS)
            traj = jax.tree_util.tree_map(
                lambda p, z: jnp.where(use_zigzag, z, p), poly_traj, zigzag_traj
            )

        state = M3EnvState(
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
            n_obstacles=n_obstacles,
            depth=jnp.zeros((DEPTH_RES, DEPTH_RES), dtype=jnp.float32),
            obs_base_buf=jnp.zeros((H, ENC_OBS_DIM), dtype=jnp.float32),
            action_buf=jnp.zeros((H, ACTION_DIM), dtype=jnp.float32),
        )

        base_obs = build_m3_base_obs(mjx_data, traj, jnp.zeros((), dtype=jnp.int32), drone_body_id)
        return state, base_obs, dr.priv_state

    return reset


# ── Step ───────────────────────────────────────────────────────────────────────

def make_m3_step_fn(
    mjx_model: mjx.Model,
    mj_model:  mujoco.MjModel,
    cfg,
):
    """
    Returns step(state, action) → (M3EnvState, base_obs_42, priv_state_8, reward, done).

    Extends M2 physics with:
    - Obstacle proximity reward + crash termination
    - History buffer roll for encoder
    """
    _inner_step = make_step_fn(mjx_model, mj_model, cfg)
    drone_body_id = mj_model.body("drone").id

    def step(state: M3EnvState, action: jnp.ndarray):
        # ── M2 physics step (tracking reward + M2 termination) ───────────────
        m2_state_in = _to_m2_state(state)
        new_m2, _, _, _m2_reward, m2_done = _inner_step(m2_state_in, action)

        new_mjx = new_m2.mjx_data

        # ── Obstacle proximity + collision ────────────────────────────────────
        drone_pos = new_mjx.xpos[drone_body_id]
        d_min = _min_dist_jax(
            drone_pos,
            state.obstacle_positions,
            state.obstacle_half_extents,
            state.n_obstacles,
        )
        crashed = d_min < D_CRASH

        # ── M3 reward (replaces M2 reward) ────────────────────────────────────
        ref_pos = get_reference_pos(state.traj, new_m2.step)
        reward, reward_info = _compute_m3_reward(
            drone_pos, ref_pos, action, state.prev_action, d_min, crashed
        )

        done = m2_done | crashed

        # ── Base obs ─────────────────────────────────────────────────────────
        base_obs = build_m3_base_obs(new_mjx, state.traj, new_m2.step, drone_body_id)

        # ── Roll history buffers ──────────────────────────────────────────────
        # obs_base_buf: drop oldest (index 0), append current base_obs at end
        new_obs_base_buf = jnp.roll(state.obs_base_buf, -1, axis=0).at[-1].set(base_obs)
        # action_buf: drop oldest, append action that produced this new state
        new_action_buf = jnp.roll(state.action_buf, -1, axis=0).at[-1].set(action)

        new_state = M3EnvState(
            mjx_data=new_mjx,
            rotor_speeds=new_m2.rotor_speeds,
            prev_action=action,
            traj=state.traj,
            step=new_m2.step,
            done=done,
            kf_multiplier=state.kf_multiplier,
            rotor_efficiency=state.rotor_efficiency,
            mass_scale=state.mass_scale,
            priv_state=state.priv_state,
            obstacle_positions=state.obstacle_positions,
            obstacle_half_extents=state.obstacle_half_extents,
            n_obstacles=state.n_obstacles,
            depth=jnp.zeros((DEPTH_RES, DEPTH_RES), dtype=jnp.float32),
            obs_base_buf=new_obs_base_buf,
            action_buf=new_action_buf,
        )

        return new_state, base_obs, state.priv_state, reward, done

    return step


# ── Vectorized M3 environment ──────────────────────────────────────────────────

class M3VecEnv:
    """
    Batched M3 environment.

    Interface:
      batch_reset(keys, modes, density_mult) → (M3EnvState, base_obs, priv_state)
      batch_step(states, actions)            → (M3EnvState, base_obs, priv_state, reward, done)
      batch_render(states)                   → (N, 64, 64) depth float32 in [0,1]
      compute_k_nearest_batch(states)        → (N, K_NEAREST) distances for critic

    Training loop assembles full actor/critic obs by calling:
      compute_depth_bins_batch(depth)
      build_full_obs(base_obs, z_hat, depth_bins, k, obs_dists)
    """

    def __init__(
        self,
        cfg,
        fault_prob: float = 0.7,
        eta_min:    float = 0.5,
        mass_lo:    float = 0.8,
        mass_hi:    float = 1.2,
        override_traj=None,
    ):
        self.cfg      = cfg
        self.num_envs = cfg.num_envs

        self.mj_model, self.mjx_model = load_mjx_model(DEPTH_XML)
        self.drone_body_id = self.mj_model.body("drone").id
        self.cam_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAM_NAME
        )

        _reset_fn = make_m3_reset_fn(
            self.mjx_model, self.mj_model, cfg,
            fault_prob=fault_prob, eta_min=eta_min, mass_lo=mass_lo, mass_hi=mass_hi,
            override_traj=override_traj,
        )
        _step_fn = make_m3_step_fn(self.mjx_model, self.mj_model, cfg)

        self._raw_reset_fn = _reset_fn
        self._raw_step_fn  = _step_fn

        # vmap: all inputs batched over axis 0
        self._reset_jit = jax.jit(jax.vmap(_reset_fn, in_axes=(0, 0, 0, 0)))
        self._step_jit  = jax.jit(jax.vmap(_step_fn))

        # JIT'd helpers
        self._k_nearest_jit      = jax.jit(jax.vmap(self._compute_k_nearest_single))
        self._depth_bins_batch_jit = jax.jit(compute_depth_bins_batch)

        self._init_mjwarp()

    def _compute_k_nearest_single(self, state: M3EnvState) -> jnp.ndarray:
        drone_pos = state.mjx_data.xpos[self.drone_body_id]
        return compute_k_nearest_dists(
            drone_pos,
            state.obstacle_positions,
            state.obstacle_half_extents,
            state.n_obstacles,
        )

    # ── MJWarp initialization ─────────────────────────────────────────────────

    def _init_mjwarp(self):
        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, mj_data)

        self._mjw_model = mjw.put_model(self.mj_model)
        # njmax=0: no constraint buffers — render path only needs kinematics
        self._mjw_data  = mjw.put_data(
            self.mj_model, mj_data, nworld=self.num_envs, njmax=0
        )
        ncam = self.mj_model.ncam
        self._rc = mjw.create_render_context(
            self.mj_model,
            nworld=self.num_envs,
            cam_res=(DEPTH_RES, DEPTH_RES),
            render_depth=[True] * ncam,
            render_rgb=[False] * ncam,
            use_shadows=False,
        )
        # depth_buf removed: mujoco_warp 3.5.0 stores depth in rc.depth_data directly

    # ── Public interface ──────────────────────────────────────────────────────

    def batch_reset(
        self,
        keys: jnp.ndarray,
        modes: list[str] | None = None,
        density_mult: float = 1.0,
    ) -> tuple[M3EnvState, jnp.ndarray, jnp.ndarray]:
        """
        Sample per-env scene layouts (numpy, outside JIT), then vmap reset.
        Returns (M3EnvState, base_obs (N,42), priv_state (N,8)).
        modes: list of N mode strings; if None, sample from TRAINING_MODES.
        """
        N   = int(keys.shape[0])
        seed = int(jax.random.randint(keys[0], shape=(), minval=0, maxval=2**30))
        rng  = np.random.default_rng(seed=seed)

        all_centers  = np.empty((N, N_OBSTACLE_SLOTS, 3), dtype=np.float32)
        all_he       = np.empty((N, N_OBSTACLE_SLOTS, 3), dtype=np.float32)
        all_n_obs    = np.empty((N,), dtype=np.int32)

        for i in range(N):
            mode = modes[i] if modes is not None else sample_mode(rng)
            centers, he = sample_scene(rng, mode, density_mult=density_mult)
            all_centers[i] = centers
            all_he[i]      = he
            # n_obstacles = count of active slots (center != 100)
            all_n_obs[i]   = int(np.sum(centers[:, 0] < 50.0))

        obs_pos  = jnp.array(all_centers)
        obs_he   = jnp.array(all_he)
        n_obs_jx = jnp.array(all_n_obs, dtype=jnp.int32)

        return self._reset_jit(keys, obs_pos, obs_he, n_obs_jx)

    def batch_step(
        self,
        states:  M3EnvState,
        actions: jnp.ndarray,   # (N, 4)
    ) -> tuple[M3EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """(M3EnvState, base_obs (N,42), priv_state (N,8), reward (N,), done (N,))."""
        return self._step_jit(states, actions)

    def batch_render(self, states: M3EnvState) -> jnp.ndarray:
        """MJX state → MJWarp render → (N, 64, 64) float32 depth in [0,1]."""
        import warp as wp
        import mujoco_warp as mjw

        qpos_np  = np.array(states.mjx_data.qpos)
        self._mjw_data.qpos.assign(qpos_np)

        mocap_np = np.array(states.obstacle_positions)
        self._mjw_data.mocap_pos.assign(
            wp.from_numpy(mocap_np, dtype=wp.vec3f, device="cuda:0")
        )

        # kinematics only: updates body/camera poses from qpos without constraint solving
        mjw.kinematics(self._mjw_model, self._mjw_data)
        mjw.camlight(self._mjw_model, self._mjw_data)
        mjw.render(self._mjw_model, self._mjw_data, self._rc)
        wp.synchronize()

        # mujoco_warp 3.5.0: depth in rc.depth_data (nworld, H*W), 0.0=no hit
        depth_raw = self._rc.depth_data.numpy()    # (N, H*W) meters
        depth_norm = np.where(
            depth_raw == 0.0, 1.0,
            np.clip(depth_raw / DEPTH_MAX_M, 0.0, 1.0)
        )
        depth_np = depth_norm.reshape(self.num_envs, DEPTH_RES, DEPTH_RES)
        return jnp.array(depth_np)   # (N, 64, 64)

    def compute_k_nearest_batch(self, states: M3EnvState) -> jnp.ndarray:
        """(N, K_NEAREST) nearest-obstacle distances. Infinity for inactive slots."""
        return self._k_nearest_jit(states)

    def compute_depth_bins(self, depth_batch: jnp.ndarray) -> jnp.ndarray:
        """(N, 64, 64) → (N, 16)."""
        return self._depth_bins_batch_jit(depth_batch)
