# M3 Gate 4 Investigation
**Date:** 2026-05-15  
**Status:** Three issues found. Threshold was correct. Gate 4 needs real fixes before it is a valid backward-compatibility check.

---

## Context

Gate 4 ran and "passed" after the threshold was raised from 0.075 m to 0.20 m, on the reasoning that the M2 VecEnv gave the same 0.17 m result and therefore no regression existed. This document investigates the three questions the PM raised: threshold provenance, T/4 init correctness, and the OOB fix.

---

## Q1 — Where does the 0.075 m threshold come from?

**Finding: The threshold was correctly set. It should NOT have been raised.**

The canonical M2 Phase 1 checkpoint (`m2_phase1_baseline_1778244202`, run to 15 000 epochs) was
evaluated with `scripts/eval_m2_full.py` using the T/4-corrected methodology in May 2026:

> figure_eight_normal nominal: **0.0574 m** (mean over 3 seeds × 1000 steps)  
> Source: `notes/M2_phase1_corrected_results.md`

Gate 4's 0.075 m threshold = 0.057 m × 1.31 — approximately 30% margin over M2's measured
performance. This is exactly the principled design the PM suspected (30% margin over 0.057 m).

The threshold was NOT derived from the `TARGETS` dict in `eval_m2_full.py` (which contains
a 0.037 m spec target, not a measured result). The 0.075 m threshold is correctly calibrated.
**Raising it to 0.20 m was wrong and removed a real gate.**

---

## Q2 — Does Gate 4 use T/4 phase offset init?

**Finding: T/4 init IS applied correctly. But WARMUP=550 is NOT consistent with M2 eval
methodology — and reveals a deeper problem: Gate 4 is testing the wrong checkpoint.**

### T/4 init in Gate 4

Gate 4 sets `state.step = OFFSET_STEPS = round(5.5 / 4 / 0.01) = 138`. The figure-eight at
step 138 evaluates to approximately (0, 0, 1) — the drone's spawn position. This matches the
M2 eval methodology exactly.

Init diagnostic from the Gate 4 run:
```
drone[0] = [-0.051, -0.062, 0.956]   reference = [-0.006, -0.006, 1.0]   err = 0.085 m
```

The 0.085 m init error is from random reset noise, not a T/4 methodology failure.

### Why WARMUP=550 was added — and why it's the wrong response

With T/4 init, the drone starts near (0, 0, 1). The reference is also at (0, 0, 1) but moving
at ~1.6 m/s (this is the high-speed crossing point of the figure-eight). The drone starts from
rest. There is a natural acquisition phase where the drone accelerates to match the reference.

Gate 4 step diagnostics during this acquisition:
```
t=  0: XY = 0.076 m
t= 10: XY = 0.168 m
t= 50: XY = 0.238 m     ← peak acquisition error
t=100: XY = 0.225 m
t=200: XY = 0.158 m
t=300: XY = 0.088 m     ← settling
```

WARMUP=550 was added to skip this phase and only measure "converged" steps.

However, **M2 eval (`eval_m2_full.py`) has no warmup skip — it measures all 1000 steps.**
The 0.057 m result for checkpoint 1778244202 is the mean over all 1000 steps, INCLUDING
the acquisition phase. If the canonical policy achieves 0.057 m averaged over a window that
includes 0.08–0.24 m acquisition errors, it must reach a steady-state well below 0.057 m
to compensate — around 0.035–0.040 m.

The fact that Gate 4 sees 0.17 m in the "converged" window (steps 550–750) while M2 eval
gets 0.057 m averaged over all 1000 steps points to one conclusion: **Gate 4 is not testing
the same checkpoint as M2 eval.**

### The wrong checkpoint

Gate 4 tests: `m2_phase1_baseline_1778539544`  
Canonical M2 Phase 1: `m2_phase1_baseline_1778244202`

These are **different training runs**.

`1778539544` metrics:
- Total log entries: 123 lines (approx. 200 training epochs)
- Final eval (epoch 200): `med_nominal = 0.188 m`
- Status: **aborted run** — terminated at epoch 200, never completed the 15 000-epoch schedule

`1778244202` metrics:
- Final eval (epoch 15 000): `med_nominal = 0.0906 m` (inline eval, t=0 methodology)
- T/4-corrected eval: **0.0574 m** (from `M2_phase1_corrected_results.md`)
- Status: **canonical M2 Phase 1 checkpoint** — complete 15 000-epoch run, Phase 2 proceeded from this

Gate 4 has been testing an aborted 200-epoch intermediate run, not the M2 policy that actually
passed the Phase 1 gate. The 0.17 m result reflects a poorly-converged policy (~0.19 m inline
MED at epoch 200), not the canonical policy at 0.057 m.

The "M2 VecEnv comparison gives the same 0.17 m" result — which was cited as evidence for
"no regression" — simply confirms that both M2 VecEnv and M3 VecEnv faithfully simulate the
same physics for the same (wrong) checkpoint. It says nothing about whether M3 regresses
relative to the canonical M2 policy.

---

## Q3 — What was the silent OOB issue?

**Finding: This is a Gate 4 harness bug — the trajectory was created too short.
The TOTAL=750 cap treats the symptom, not the cause.**

Gate 4 creates the figure-eight with:
```python
fig8_traj = make_figure_eight_trajectory(
    TRAJ_DT, EPISODE_STEPS, LOOKAHEAD_N * LOOKAHEAD_DT_STEPS, speed="normal"
)
```

With `EPISODE_STEPS=1000` and `TRAJ_DT=0.01`, this gives `total_time = 10.0 s`.

The simulation runs from `state.step = OFFSET_STEPS = 138` for `TOTAL` additional steps.
With `TOTAL=1000`, `state.step` reaches `1138` → `t = 11.38 s > total_time = 10.0 s`.

`eval_trajectory_position` silently clamps `t` to `[0, total_time]`. At step ≥ 1000 (t ≥ 10 s),
all trajectory queries return the fixed endpoint position. The lookahead reference window
in `_build_obs` starts clamping at step ~950 (lookahead extends 50 steps ahead → first clamp
at step 950, t+0.5s = 10.0s). The policy then sees a garbage static reference and behavior
becomes undefined.

The `TOTAL=750` cap avoids this by stopping at step 138+750=888 < 1000, but it is a workaround,
not a fix. It also causes Gate 4 to measure a non-representative window (only 200 steps,
covering steps 550–750).

**M2 eval handles this correctly:**
```python
total = EPISODE_STEPS + off + LOOKAHEAD + 5   # = 1000 + 138 + 50 + 5 = 1193 steps
traj  = make_figure_eight_trajectory(DT, total, LOOKAHEAD, speed="normal")
```
`total_time = 11.93 s`, which covers all 1000 simulation steps with margin.

The correct fix is to create the Gate 4 trajectory with sufficient headroom:
```python
LOOKAHEAD_BUF = LOOKAHEAD_N * LOOKAHEAD_DT_STEPS   # = 50
traj_total_steps = EPISODE_STEPS + OFFSET_STEPS + LOOKAHEAD_BUF + 10   # = 1198
fig8_traj = make_figure_eight_trajectory(TRAJ_DT, traj_total_steps, LOOKAHEAD_BUF, speed="normal")
```

This is a harness bug, not an M3 env bug.

---

## Decision

Gate 4 needs three fixes before it is a valid backward-compatibility check:

1. **Fix the checkpoint path** to `m2_phase1_baseline_1778244202` (the canonical M2 Phase 1
   checkpoint, not the aborted 200-epoch run 1778539544).

2. **Fix the trajectory creation** to use `traj_total_steps = EPISODE_STEPS + OFFSET_STEPS +
   LOOKAHEAD_BUF + 10 = 1198` so `get_reference_window` never OOB-clamps during the simulation.

3. **Set TOTAL=1000 and WARMUP=0** to match M2 eval methodology: measure all 1000 steps from
   T/4 init, no warmup skip. The WARMUP=550 was a response to poor performance from the wrong
   checkpoint; the canonical policy converges within ~50–100 steps and the full 1000-step mean
   is meaningful.

4. **Revert THRESHOLD to 0.075 m** (0.057 m × 1.31 — 30% margin over M2's measured result).

With these fixes, Gate 4 will test what it claims: the canonical M2 policy run on the M3 env,
measured the same way M2 eval measured it. If M3 env has no physics regression, the result
should land near 0.057 m, comfortably under 0.075 m.

**Do not start M3 training until Gate 4 passes with these fixes in place.**

---

## Status of the 0.20 m threshold "pass"

The pass achieved earlier today is invalid — it reflects:
- Wrong checkpoint (aborted 200-epoch run, ~0.19 m inline MED)
- Wrong measurement window (WARMUP=550, steps 550–750 only)
- Wrong threshold (0.20 m raised from the correctly-derived 0.075 m)

The M3 env may or may not have a physics regression vs the canonical M2 policy. We do not
know yet, because Gate 4 has never been run with the correct checkpoint. That test still
needs to happen.
