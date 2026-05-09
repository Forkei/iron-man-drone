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

## L5 — Encoder startup instability: zero-padded history produces garbage ê_t for the first 50 steps

**Observed (M2 Phase 2 closed-loop eval):** On figure_eight_fast + fault (η=0.70), 2/3 seeds crashed within the first 500 steps. On figure_eight_normal + fault, no crashes — the slower trajectory gave the policy enough slack to absorb bad ê_t estimates during the zero-padding window.

**Root cause:** The causal encoder input at episode start is H=50 pairs of (obs_base, prev_action), all zeros. The encoder was never trained to handle zero-padded inputs at inference — training examples are sampled at t ∈ [49, 999] with a padded prefix, but the network still sees 49/50 real pairs at t=49. At deployment, all 50 pairs are zeros at t=0, and the encoder has to work through the padding window before it sees enough real history to identify the fault.

**Why it didn't matter on figure_eight_normal:** The normal figure-eight is slow enough that the policy can recover from a few bad ê_t steps before tracking error compounds. Fast trajectories with simultaneous fault leave no margin.

**Why this matters for M3:** Visual obstacle avoidance is exactly the regime where startup instability is dangerous — the policy may be asked to enter a cluttered environment from a standing start, where bad ê_t in the first 50 steps could cause early collision.

**Fix options (choose one before M3, one variable at a time):**
1. **Warmup buffer with replay:** Before the episode proper, roll out H steps in a safe open-air region to fill the ring buffer with real history. Simple but requires environmental assumptions.
2. **Train on true zero-padded startup:** Sample training examples from t ∈ [0, 999] (not just t ≥ 49). The encoder sees real zero-padded prefixes during training and learns to handle them.
3. **Recurrent encoder (LSTM/GRU):** Replace flat MLP with a recurrent architecture that processes (obs, action) pairs sequentially. Handles variable-length history naturally; startup state is the hidden state init (zeros = nominal assumption). More complex to train but eliminates the padding artifact entirely.

**Lowest-risk fix:** option 2 (train on true zero-padded prefixes) — it's a one-line change to the `sample_batch` function (`t_idx = rng.integers(0, 1000, ...)` instead of `0, 951`).

---

## L6 — Verify numbers before reacting to them

**Observed (three times in this project):**
- M2 "2.5× regression" on figure_eight_normal: apparent degradation from 0.037 m → 0.105 m was entirely a methodology artifact. The M2 inline eval used t=0 (inflating by ~0.044 m + DR penalty); M1.3 used T/4-corrected. Correct comparison: 0.037 m → 0.044 m (1.2× — a real but small gap).
- Encoder MSE prediction of 0.35: the user wrote "user prediction: normalized MSE ≈ 0.35" but intended 0.035 (normalized) or 0.35 raw. The encoder actually achieved 0.016, well within spec.
- Polynomial crashes in Phase 1 eval: 2/3 seeds crashed. First interpretation was "Phase 1 policy is unstable on polynomial." Investigation showed it was a distribution mismatch at the fixed eval seeds, not a general instability.

**Pattern:** In each case, a number appeared that was dramatically different from expectations. The correct first move — verify the number — was not always taken before anxiety set in.

**Rule:** When a number is dramatically off expected (>2× in either direction), stop and verify: (1) Is the methodology consistent with what the reference number used? (2) Is the number reading the right column/field? (3) Does the number make physical sense? Only after verifying do you update your model of the world.

---

## L7 — Privileged-state prediction from observable history is more tractable than intuition suggests

**Observed (M2 Phase 2):** A flat 2300→256→128→8 MLP trained on 20k episodes achieved val MSE 0.016 on normalized 8-dim privileged state — well below the spec gate of 0.020 and far better than the user's initial prediction of ~0.35. Training took less than 1 minute on a 4070.

**What the encoder learned:** η₁–η₄ (rotor efficiencies) are the easiest channels (MSE ~0.011): a degraded rotor forces a compensating roll/pitch moment that is directly visible in the rotation matrix R and velocity v over a 0.5 s window. m_scale (mass variation ±20%) is harder (MSE ~0.076): mass scales all forces uniformly and is harder to decouple from k_f variation. Wind channels are trivially near-zero (Phase 1 used wind=0).

**Implication:** When designing future systems, don't assume you need privileged access to physical state. The dynamics signature of perturbations is genuinely legible from a short window of observable (R, v, obs) history. Before adding a sensor or a privileged channel, ask whether a history encoder can learn it from what's already observable. The RMA pattern (train privileged first, then distill) is a reliable way to quantify this.

---

## L4 — Polynomial generator was κ=0 from day 1 (M1/M1.1/M1.2)

**Observed:** Original polynomial generator applied a quintic scalar `h(τ) = 10τ³-15τ⁴+6τ⁵` to straight-line direction vectors. Result: paths with κ=0 everywhere, velocity=0 at every waypoint. Training distribution had 0% coverage of figure-eight apex curvature (κ=4.789 m⁻¹).

**Fix (M1.3):** C2-continuous piecewise quintic polynomial, solved via closed-form 6-BC system per segment, with random nonzero interior velocities/accelerations. Coverage jumped to 100% for figure-eight apex curvature.

**Result:** MED dropped from 0.105m plateau to 0.085m in 2000 epochs on the first M1.3 run. Pre-resume trajectory (0.105 → 0.099 → 0.085) still improving at epoch 2000 — retrain expected to show further improvement.
