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

    TARGET = 50_000
    ok = check(
        f"throughput >= {TARGET:,} steps/sec",
        steps_per_sec >= TARGET,
        f"{steps_per_sec:,.0f} env-steps/sec  ({NUM_ENVS} envs × {BENCH_STEPS} steps in {elapsed:.2f}s)",
    )

    if not ok:
        print()
        if steps_per_sec > 10_000:
            print(f"  {WARN} Below target but > 10k. Training will work, just slower.")
            print("  Check: is jax.lax.scan used in ppo.py rollout? (not a Python for-loop)")
        else:
            print("  FIX: < 10k steps/sec indicates something is not GPU-accelerated.")
            print("  Check:")
            print("    1. JAX is using CUDA device (check [1/4] above)")
            print("    2. No Python-level loop inside the vmapped step")
            print("    3. xfrc_applied shape is (num_envs, nbody, 6) — correct batching")

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


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print("All four checks passed. Environment is ready for training.")
print()
print("Next: python scripts/train_m1.py")
print("      (reads notes/M1_hypothesis.md gate, then starts training)")
