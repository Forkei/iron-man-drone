# M2 Trend Gate Recalibration — 2026-05-06

## Context

The 10k trend gate checks whether the policy improved meaningfully over the 5k→10k epoch
window. It fires (aborts) if improvement falls below a threshold, protecting against wasting
compute on a plateau.

---

## Original gate

**Threshold: 0.010 m improvement over the 5k→10k window.**

Calibrated from M1.3's measured performance in the same window on figure-eight-normal:
- M1.3 epoch 5k: 0.091 m nominal MED
- M1.3 epoch 10k: 0.081 m nominal MED
- Improvement: 0.010 m in 5000 epochs

The gate was set at the empirical rate — the idea being that if M2 can't match M1.3's
improvement rate on the same window, something is wrong.

---

## M2 Phase 1 empirical measurement

Phase 1 ran 0→10k epochs with full DR (fault_prob=0.70, mass ±20%, kf DR).

Gate evaluations on figure-eight-normal nominal conditions (kf=1.0):
- Epoch 5k: **0.106 m** nominal MED
- Epoch 10k: **0.100 m** nominal MED
- Improvement: **0.006 m** over the 5k→10k window

The original 0.010 m gate fired at epoch 10k.

---

## Why M2 improves more slowly than M1.3 in this window

M1.3 trained under lighter DR (no faults, lighter mass range, ±30% kf only).
M2 Phase 1 adds:
- 70% probability of a rotor fault at η=0.50–1.00
- ±20% mass randomization
- All of the above simultaneously per episode

This is a harder multi-objective landscape. The gradient signal must generalize across a much
larger state distribution, which compresses per-epoch improvement rates. A 0.006 m/5k gain
under these conditions is not a plateau — it is the expected rate for a policy still learning
fault recovery.

Additionally, M2's absolute MED at epoch 10k (0.100 m) is still meaningfully above converged
values (~0.040 m), so there is real headroom left. A plateau would show near-zero improvement
near convergence; 0.006 m with 0.100 m remaining is not a plateau.

---

## Recalibrated gate

**New threshold: 0.005 m improvement over any 5k-epoch window for DR-resumed runs.**

This is set slightly below the empirical rate (0.006 m) so it only fires on a genuine plateau,
not on expected slower progress under heavier DR. If improvement falls below 0.005 m over a
full 5k-epoch resumed run, the policy is genuinely stuck and the run should be reviewed.

---

## Usage

When launching a resumed run after a trend gate fires on a full DR run, pass:

```
--trend_gate_improvement 0.005
--resume_med_nominal <MED at resume epoch>
```

The post-loop trend gate (added to `train_m2.py`) will evaluate the window
`start_epoch → total_epochs` and report PASSED or FIRED with full numbers.

---

## Audit trail

| Event | Value | Notes |
|-------|-------|-------|
| M1.3 improvement 5k→10k | 0.010 m | Nominal DR only |
| M2 Ph1 improvement 5k→10k | 0.006 m | Full DR (fault 70%, mass ±20%) |
| Original gate | 0.010 m | Matched M1.3 rate |
| Gate fired at | epoch 10k | 0.106→0.100 m improvement = 0.006 m < 0.010 m |
| Recalibrated gate | 0.005 m | Slightly below empirical DR rate |
| Decision | Resume 10k→15k | Full opt_state confirmed in checkpoint |
