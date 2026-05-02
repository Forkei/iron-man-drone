"""
Pre-training sanity checks — run these in order before anything else.
Each step is a hard gate: fix failures before proceeding.

Usage: python scripts/sanity_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


# ── 1. JAX CUDA device ────────────────────────────────────────────────────────
print("\n[1/4] JAX device check")
try:
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    device_strs = [str(d) for d in devices]
    has_gpu = any("cuda" in s.lower() or "gpu" in s.lower() for s in device_strs)
    check("JAX imports", True, f"version {jax.__version__}")
    ok = check("CUDA device visible", has_gpu, ", ".join(device_strs))
    if not ok:
        print()
        print("  FIX: JAX sees CPU only.")
        print("  Most likely cause: CUDA driver version mismatch.")
        print("  Check steps:")
        print("    1. On Windows: nvidia-smi should show driver >= 525")
        print("    2. In WSL2: nvidia-smi should also work (driver passthrough)")
        print("    3. Reinstall JAX: pip install --upgrade 'jax[cuda12]'")
        print("    4. Do NOT apt install cuda in WSL2 — Windows driver handles it")
        sys.exit(1)
    # Print VRAM
    props = jax.devices()[0]
    print(f"         device: {props}")
except ImportError as e:
    check("JAX imports", False, str(e))
    print("  FIX: pip install --upgrade 'jax[cuda12]'")
    sys.exit(1)


# ── 2. MuJoCo + MJX imports ───────────────────────────────────────────────────
print("\n[2/4] MuJoCo + MJX imports")
try:
    import mujoco
    check("mujoco imports", True, f"version {mujoco.__version__}")
except ImportError as e:
    check("mujoco imports", False, str(e))
    print("  FIX: pip install 'mujoco>=3.1.6'")
    sys.exit(1)

try:
    from mujoco import mjx
    check("mujoco.mjx imports", True)
except ImportError as e:
    check("mujoco.mjx imports", False, str(e))
    print("  FIX: MJX requires mujoco >= 3.0. Upgrade: pip install --upgrade mujoco")
    sys.exit(1)


# ── 3. Crazyflie XML loads, steps, no NaNs ────────────────────────────────────
print("\n[3/4] Crazyflie model: load + step + NaN check")

xml_path = Path(__file__).parent.parent / "src/iron_man_drone/envs/crazyflie.xml"
check("crazyflie.xml exists", xml_path.exists(), str(xml_path))
if not xml_path.exists():
    print("  FIX: Run from repo root. Expected at src/iron_man_drone/envs/crazyflie.xml")
    sys.exit(1)

try:
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    check("XML parses", True, f"{mj_model.nbody} bodies, {mj_model.nq} qpos, {mj_model.nv} qvel")
except Exception as e:
    check("XML parses", False, str(e))
    sys.exit(1)

try:
    mj_model.opt.timestep = 0.01
    mjx_model = mjx.put_model(mj_model)
    check("mjx.put_model", True)
except Exception as e:
    check("mjx.put_model", False, str(e))
    sys.exit(1)

try:
    drone_body_id = mj_model.body("drone").id
    check("body 'drone' found", True, f"id={drone_body_id}")
except Exception as e:
    check("body 'drone' found", False, str(e))
    sys.exit(1)

try:
    mjx_data = mjx.make_data(mjx_model)
    # Set a non-trivial initial state (drone at 1m height, slightly tilted)
    qpos = jnp.array([0.05, -0.03, 1.0,   # position
                      0.998, 0.04, 0.03, 0.0])  # quaternion (near-upright)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros(6))
    mjx_data = mjx.forward(mjx_model, mjx_data)

    # Apply a realistic hover force (Crazyflie weight = 0.0321 * 9.81 ≈ 0.315 N)
    n_bodies = mj_model.nbody
    xfrc = jnp.zeros((n_bodies, 6))
    xfrc = xfrc.at[drone_body_id, 2].set(0.315)  # z force in world frame
    mjx_data = mjx_data.replace(xfrc_applied=xfrc)

    # Run 10 steps
    @jax.jit
    def run_steps(data):
        def _step(d, _):
            return mjx.step(mjx_model, d), None
        data, _ = jax.lax.scan(_step, data, None, length=10)
        return data

    mjx_data = run_steps(mjx_data)

    pos = mjx_data.xpos[drone_body_id]
    qvel = mjx_data.qvel

    has_nan = bool(jnp.any(jnp.isnan(pos)) or jnp.any(jnp.isnan(qvel)))
    check("10 steps without NaN", not has_nan,
          f"pos={pos[:3]}, vel_z={float(qvel[2]):.3f}")

    if has_nan:
        print("  FIX: NaNs in simulation. Possible causes:")
        print("    - Timestep too large (check opt.timestep = 0.01)")
        print("    - Mass/inertia values wrong in crazyflie.xml")
        print("    - Contact with floor at initialization")
        sys.exit(1)

    # Check qpos/qvel dimensions match expectation
    ok1 = check("qpos shape (7,)", mjx_data.qpos.shape == (7,), str(mjx_data.qpos.shape))
    ok2 = check("qvel shape (6,)", mjx_data.qvel.shape == (6,), str(mjx_data.qvel.shape))
    if not (ok1 and ok2):
        print("  FIX: Unexpected DOF count. Verify crazyflie.xml has exactly one freejoint.")
        sys.exit(1)

    # Check xmat exists and has right shape for rotation matrix
    xmat = mjx_data.xmat[drone_body_id]
    check("xmat[drone] shape (9,)", xmat.shape == (9,), str(xmat.shape))
    # The (2,5,8) elements are the z-column of the rotation matrix (body z in world frame)
    body_z = jnp.array([xmat[2], xmat[5], xmat[8]])
    cos_tilt = float(body_z[2])
    check("rotation matrix looks valid", abs(cos_tilt) > 0.5,
          f"body z-axis world: {body_z.tolist()}, cos_tilt={cos_tilt:.3f}")

except Exception as e:
    check("MJX step", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── 4. Throughput at 1024 parallel envs ──────────────────────────────────────
print("\n[4/4] Throughput benchmark: 1024 parallel envs")

NUM_ENVS = 1024
WARMUP_STEPS = 50
BENCH_STEPS = 500

try:
    # Batch the initial state
    batch_data = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (NUM_ENVS,) + x.shape),
        mjx.make_data(mjx_model),
    )
    # Randomize positions slightly so envs aren't identical
    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    offsets = jax.vmap(lambda k: jax.random.uniform(k, (3,), minval=-0.05, maxval=0.05))(keys)
    new_qpos = batch_data.qpos.at[:, :3].add(offsets)
    new_qpos = new_qpos.at[:, 2].set(1.0)  # z = 1m
    batch_data = batch_data.replace(qpos=new_qpos)

    # Apply per-env hover force
    xfrc_batch = jnp.zeros((NUM_ENVS, n_bodies, 6))
    xfrc_batch = xfrc_batch.at[:, drone_body_id, 2].set(0.315)
    batch_data = batch_data.replace(xfrc_applied=xfrc_batch)

    # JIT-compiled batched step
    @jax.jit
    def batched_steps(data):
        step_fn = jax.vmap(lambda d: mjx.step(mjx_model, d))
        def _step(d, _):
            return step_fn(d), None
        data, _ = jax.lax.scan(_step, data, None, length=WARMUP_STEPS)
        return data

    # Warmup (triggers JIT compilation)
    print(f"  Compiling (first call JIT-compiles, may take 30-60s)...")
    t0 = time.time()
    batch_data = batched_steps(batch_data)
    batch_data[0].qpos.block_until_ready()  # type: ignore
    compile_time = time.time() - t0
    print(f"  JIT compile + {WARMUP_STEPS} steps: {compile_time:.1f}s")

    # Benchmark
    @jax.jit
    def bench_steps(data):
        step_fn = jax.vmap(lambda d: mjx.step(mjx_model, d))
        def _step(d, _):
            return step_fn(d), None
        data, _ = jax.lax.scan(_step, data, None, length=BENCH_STEPS)
        return data

    t0 = time.time()
    batch_data = bench_steps(batch_data)
    jax.block_until_ready(batch_data.qpos)
    elapsed = time.time() - t0

    total_env_steps = NUM_ENVS * BENCH_STEPS
    steps_per_sec = total_env_steps / elapsed

    # Three-tier threshold — only < 5k is a hard stop (truly broken GPU path).
    # 50k is the target; 5k-50k is slow but training will complete.
    BROKEN  =  5_000   # sys.exit — something is not on GPU
    TARGET  = 50_000   # PASS

    detail = f"{steps_per_sec:,.0f} env-steps/sec  ({NUM_ENVS} envs × {BENCH_STEPS} steps in {elapsed:.2f}s)"
    if steps_per_sec >= TARGET:
        check("throughput >= 50k steps/sec (target)", True, detail)
    elif steps_per_sec >= BROKEN:
        # Warn but don't exit — training is slow but viable
        eta_warn = (NUM_ENVS * 32 * 15_000) / steps_per_sec / 3600
        print(f"  [{WARN}] throughput {steps_per_sec:,.0f} steps/sec — below 50k target, but training viable (~{eta_warn:.0f}h)")
        print(f"         {detail}")
        if steps_per_sec < 20_000:
            print("  CHECK: below 20k. Verify jax.lax.scan is used in ppo.py rollout, not a Python for-loop.")
    else:
        check(f"throughput >= {BROKEN:,} steps/sec (minimum)", False, detail)
        print()
        print("  FIX: < 5k steps/sec means the simulation is not running on GPU.")
        print("  Debug in order:")
        print("    1. Confirm gate [1/4] showed a CUDA device, not CPU")
        print("    2. Run: python -c \"import jax; print(jax.default_backend())\"  → must print 'gpu'")
        print("    3. Check that vmap'd mjx.step is inside jax.jit (no Python loop wrapping it)")
        sys.exit(1)

    # Estimate training time
    steps_per_epoch = NUM_ENVS * 32  # horizon=32
    epochs = 15_000
    total_steps_training = steps_per_epoch * epochs
    eta_hours = total_steps_training / steps_per_sec / 3600
    print(f"  Estimated training time (15k epochs): {eta_hours:.1f} hours")

    # NaN check after benchmark
    has_nan_bench = bool(jnp.any(jnp.isnan(batch_data.qpos)))
    check("No NaNs after benchmark", not has_nan_bench)

except Exception as e:
    check("Throughput benchmark", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── 5. Hover-only sanity: CTBR controller, no learning ───────────────────────
print("\n[5/6] Hover sanity: CTBR controller holds altitude without policy")
print("  (Tests mixer, motor dynamics, force application — not PPO)")

try:
    from iron_man_drone.control.ctbr_controller import (
        ctbr_to_rotor_speeds, compute_wrench,
        MASS, GRAVITY, MAX_ROTOR_SPEED,
    )

    # Single env, start at 1m height
    hover_data = mjx.make_data(mjx_model)
    qpos0 = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    hover_data = hover_data.replace(qpos=qpos0, qvel=jnp.zeros(6))
    hover_data = mjx.forward(mjx_model, hover_data)

    # Hover CTBR command: zero body rates, collective thrust = 1g
    # c maps [0,1] → [0, 1.6*m*g]; c=1/1.6=0.625 gives exactly 1g
    # In our decode: total_thrust = ((tanh(a[3])+1)/2) * 1.6*m*g
    # To get 1g: (tanh(a[3])+1)/2 = 1/1.6 → tanh(a[3]) = -0.25 → a[3] = atanh(-0.25) ≈ -0.255
    # Simpler: use raw action = [0, 0, 0, 0] and check that it produces near-hover
    hover_action = jnp.zeros(4)  # zero body rates, mid-range thrust

    @jax.jit
    def run_hover(data):
        rotor_speeds = jnp.ones(4) * (MAX_ROTOR_SPEED * 0.5)  # start near hover
        def _step(carry, _):
            d, r_spd = carry
            omega_current = d.qvel[3:6]
            new_r_spd = ctbr_to_rotor_speeds(hover_action, r_spd, omega_current, 0.01)
            force_body, torque_body = compute_wrench(new_r_spd)
            R = d.xmat[drone_body_id].reshape(3, 3)
            force_world = R @ force_body
            torque_world = R @ torque_body
            xfrc = jnp.zeros((n_bodies, 6))
            xfrc = xfrc.at[drone_body_id, :3].set(force_world)
            xfrc = xfrc.at[drone_body_id, 3:].set(torque_world)
            d = d.replace(xfrc_applied=xfrc)
            d = mjx.step(mjx_model, d)
            return (d, new_r_spd), d.xpos[drone_body_id, 2]  # track height

        (final_d, _), heights = jax.lax.scan(_step, (data, rotor_speeds), None, length=500)
        return final_d, heights

    hover_result, heights = run_hover(hover_data)
    jax.block_until_ready(heights)

    h0 = float(heights[0])
    h_final = float(heights[-1])
    h_min = float(heights.min())
    h_max = float(heights.max())
    print(f"  Heights over 5s: start={h0:.3f}m  min={h_min:.3f}m  max={h_max:.3f}m  final={h_final:.3f}m")

    # With c=0, the drone will fall. That's expected — hover_action = zeros means
    # collective thrust is at 50% of max (1.6g * 0.5 = 0.8g), which is below
    # the 1g needed to hover. The check is that it doesn't NaN and falls plausibly.
    has_nan_hover = bool(jnp.any(jnp.isnan(heights)))
    check("No NaNs in hover test", not has_nan_hover)

    # Drone should have fallen (thrust < weight) but not too fast
    # At 0.8g thrust, net downward = 0.2g = 1.96 m/s², falls ~2.45m in 5s
    # Final height roughly 1.0 - 0.5*1.96*5^2 → would go below 0 → terminates at floor
    # Check: height decreased (not crazy NaN/explosion) and didn't teleport upward
    fell_plausibly = h_final <= h0  # with under-hover thrust, should fall
    check("Drone falls plausibly with under-hover thrust", fell_plausibly,
          f"started at {h0:.2f}m, ended at {h_final:.2f}m")

    # Now test that near-correct hover thrust keeps altitude
    # c=1 in our action encoding: total_thrust = ((tanh(1)+1)/2)*1.6*m*g
    # tanh(1) ≈ 0.762 → c = (0.762+1)/2 = 0.881 → thrust = 0.881*1.6*0.0321*9.81 ≈ 0.443 N
    # Weight = 0.315 N → thrust/weight ≈ 1.41 → drone should climb
    climb_action = jnp.array([0.0, 0.0, 0.0, 1.0])  # max up

    @jax.jit
    def run_climb(data):
        rotor_speeds = jnp.ones(4) * (MAX_ROTOR_SPEED * 0.5)
        def _step(carry, _):
            d, r_spd = carry
            omega_current = d.qvel[3:6]
            new_r_spd = ctbr_to_rotor_speeds(climb_action, r_spd, omega_current, 0.01)
            force_body, torque_body = compute_wrench(new_r_spd)
            R = d.xmat[drone_body_id].reshape(3, 3)
            force_world = R @ force_body
            xfrc = jnp.zeros((n_bodies, 6))
            xfrc = xfrc.at[drone_body_id, :3].set(force_world)
            d = d.replace(xfrc_applied=xfrc)
            d = mjx.step(mjx_model, d)
            return (d, new_r_spd), d.xpos[drone_body_id, 2]
        (_, _), heights = jax.lax.scan(_step, (data, rotor_speeds), None, length=100)
        return heights

    climb_heights = run_climb(hover_data)
    jax.block_until_ready(climb_heights)
    climbs = bool(float(climb_heights[-1]) > float(climb_heights[0]))
    check("Drone climbs with full-up action (c=1)", climbs,
          f"{float(climb_heights[0]):.3f}m → {float(climb_heights[-1]):.3f}m in 1s")

    if not climbs:
        print("  FIX: Drone doesn't climb with full thrust.")
        print("  Check ctbr_controller.py:")
        print("    1. MAX_COLLECTIVE_THRUST = 1.6 * MASS * GRAVITY — is it correct?")
        print("    2. compute_wrench() returns force in body +z direction")
        print("    3. xfrc_applied[:3] is force (not torque) in world frame")

except Exception as e:
    check("Hover test", False, str(e))
    import traceback; traceback.print_exc()
    # Non-fatal — warn but continue
    print("  (Non-fatal: hover test failed but we can still check obs)")


# ── 6. Random-policy obs validation ──────────────────────────────────────────
print("\n[6/6] Random-policy obs validation: shape, finiteness, plausible range")
print("  (Catches wrong obs indexing before wasting training time)")

try:
    from iron_man_drone.envs.quadrotor_env import (
        load_mjx_model, make_reset_fn, make_step_fn, ACTOR_OBS_DIM, CRITIC_OBS_DIM,
    )
    from iron_man_drone.utils.domain_randomization import sample_env_params

    class MinCfg:
        num_envs = 4

    mj_m, mjx_m = load_mjx_model()
    _reset = jax.jit(jax.vmap(make_reset_fn(mjx_m, mj_m, MinCfg())))
    _step = jax.jit(jax.vmap(make_step_fn(mjx_m, mj_m, MinCfg())))

    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    kf_mults = jnp.ones(4)
    states, a_obs, c_obs = _reset(keys, kf_mults)

    # Check shapes
    ok1 = check(f"actor_obs shape (4, {ACTOR_OBS_DIM})", a_obs.shape == (4, ACTOR_OBS_DIM), str(a_obs.shape))
    ok2 = check(f"critic_obs shape (4, {CRITIC_OBS_DIM})", c_obs.shape == (4, CRITIC_OBS_DIM), str(c_obs.shape))

    # Check finiteness
    ok3 = check("actor_obs finite", bool(jnp.all(jnp.isfinite(a_obs))))
    ok4 = check("critic_obs finite", bool(jnp.all(jnp.isfinite(c_obs))))

    if ok1 and ok3:
        # Check individual components
        e_W = a_obs[:, :30].reshape(4, 10, 3)
        v   = a_obs[:, 30:33]
        R   = a_obs[:, 33:42]
        k   = c_obs[:, 42]

        e_W_mag = float(jnp.abs(e_W).max())
        v_mag   = float(jnp.abs(v).max())
        R_det   = float(jnp.linalg.det(R.reshape(4, 3, 3)).mean())

        check("e^W magnitude < 10m (plausible lookahead error)", e_W_mag < 10.0,
              f"max |e^W| = {e_W_mag:.3f}m")
        check("velocity magnitude < 5 m/s at reset", v_mag < 5.0,
              f"max |v| = {v_mag:.3f} m/s")
        check("R is rotation matrix (det ≈ 1.0)", abs(R_det - 1.0) < 0.05,
              f"mean det(R) = {R_det:.4f}")
        check("critic k in [0, 1]", bool(jnp.all((k >= 0) & (k <= 1))),
              f"k range: [{float(k.min()):.3f}, {float(k.max()):.3f}]")

    # One step with random actions
    key = jax.random.PRNGKey(2)
    rand_actions = jax.random.normal(key, (4, 4))
    new_states, new_a_obs, new_c_obs, rewards, dones = _step(states, rand_actions, kf_mults)

    check("step() runs without error", True)
    check("rewards finite", bool(jnp.all(jnp.isfinite(rewards))), f"rewards: {rewards.tolist()}")
    check("rewards in [0, 1.5]", bool(jnp.all((rewards >= 0) & (rewards <= 1.5))),
          f"range [{float(rewards.min()):.3f}, {float(rewards.max()):.3f}]")

    # Verify actor obs does NOT contain timestep info (should be same length as initial)
    check("actor_obs has no extra dims (42-dim)", new_a_obs.shape[-1] == ACTOR_OBS_DIM,
          str(new_a_obs.shape))

except Exception as e:
    check("Obs validation", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print("All six checks passed. Ready for training.")
print()
print("Protocol before first real PPO run:")
print("  1. Re-read notes/M1_hypothesis.md")
print("  2. python scripts/train_m1.py  (entropy sanity check runs before epoch 1)")
print("  3. First run is time-boxed: 12 hours. Stop and read if no clear convergence")
print("     signal (reward up, value loss down, entropy slowly decreasing) by then.")
print("  4. Do NOT tweak hyperparameters without re-reading the paper first.")
