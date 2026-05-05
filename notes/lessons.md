# Lessons Learned

---

## L1 — Resume-after-pause breaks near-converged PPO policies

**Observed (M1.3 run 1777826991):** Training was paused at epoch 2500, then resumed by restoring actor/critic params from checkpoint while reinitializing the Adam optimizer (momentum and variance reset to zero). Post-resume, eval MED immediately jumped from 0.085m to 0.215m at epoch 3000, then to 0.999m by epoch 5000 and never recovered across 11,000 additional epochs of training.

**Root cause:** Adam momentum reset perturbed a near-converged policy. With fresh Adam (mu=0, nu=0), the first update is approximately `lr * sign(g)` — a constant step in each parameter direction regardless of gradient magnitude. For a converged policy with small residual gradients, this applies a disproportionately large update. The perturbation pushed the policy into a crash-at-initialization failure mode: the drone crashes at step 8 of every figure-eight episode. Once in this state, PPO receives gradient signal only from the first 8 steps (near the initial observation, large tracking error); it has no signal from later trajectory states and cannot escape the failure attractor. 11,000 training epochs generated no recovery because the feedback loop is broken: crash early → short rollout → no signal from later states → policy reinforces early-crash behavior.

**Why training reward stayed plausible (1.31–1.37):** Training uses 4096 envs with random trajectories. Some randomly-initialized episodes start near the drone's initial position and succeed. The mean reward is dominated by those episodes. Training reward is a poor proxy for held-out eval performance when the training distribution is broad.

**Fix:** Do not pause-and-resume mid-run. Run uninterrupted for the full training duration. If resume is unavoidable:
1. Save full optimizer state (Adam mu, nu, step count) — not just params
2. Orbax PyTreeCheckpointer cannot round-trip optax namedtuples; use a custom serialization or `orbax.checkpoint.args.StandardSave` with registered types
3. Even with correct optimizer state, resuming a near-converged policy is risky; prefer restarting from scratch if the run is < 30% complete

---

## L2 — Training reward is a poor proxy for held-out eval MED

**Observed:** Post-resume policies showed training rewards of 1.31–1.37 (identical to healthy pre-resume policies) while held-out figure-eight MED was 0.97m (drone crashing). Pre-resume M1 baseline also showed reward plateau at 1.33 while MED was 0.105m — a significant gap.

**Reason:** Training trajectories (50% C2-continuous polynomial, 50% zigzag) are random and broad. A policy that only succeeds on easy instances (short distance from initial position to first waypoint) can maintain high mean training reward. The figure-eight eval is a specific, fixed trajectory with a defined initial offset that requires genuine trajectory-following capability.

**Rule:** Eval MED on held-out trajectories (figure_eight_normal, pentagram_slow, etc.) is the primary success signal. Training reward is supporting context only — useful for detecting catastrophic collapse but not for measuring generalization.

---

## L3 — Eval XLA cache eviction causes fps drop

**Observed (M1.2):** Training throughput dropped from ~65k fps to ~41k fps at epoch 1000 (first eval). The pre-warm compiled only a single eval step, not the full eval scan.

**Fix for M1.4+:** Pre-warm the full 1000-step eval loop, not just a single step. The eval uses a Python for-loop (not `lax.scan`) so no additional JIT concern, but each `jit(env._step_fn)` call is cached after first compile. The drop is from the full env step kernel being evicted when the eval's larger batch shape compiles.

---

## L4 — Polynomial generator was κ=0 from day 1 (M1/M1.1/M1.2)

**Observed:** Original polynomial generator applied a quintic scalar `h(τ) = 10τ³-15τ⁴+6τ⁵` to straight-line direction vectors. Result: paths with κ=0 everywhere, velocity=0 at every waypoint. Training distribution had 0% coverage of figure-eight apex curvature (κ=4.789 m⁻¹).

**Fix (M1.3):** C2-continuous piecewise quintic polynomial, solved via closed-form 6-BC system per segment, with random nonzero interior velocities/accelerations. Coverage jumped to 100% for figure-eight apex curvature.

**Result:** MED dropped from 0.105m plateau to 0.085m in 2000 epochs on the first M1.3 run. Pre-resume trajectory (0.105 → 0.099 → 0.085) still improving at epoch 2000 — retrain expected to show further improvement.
