# M1.3 Eval Methodology — SimpleFlight Comparison

**Date:** 2026-05-05  
**Purpose:** Determine whether the M1.3 eval matches SimpleFlight's paper methodology before claiming M1 passes.

---

## Check 1 — SimpleFlight Initialization Protocol

**Source:** `omni_drones/envs/single/track.py` (thu-uav/SimpleFlight GitHub repo)

```python
self.origin = torch.tensor([0., 0., 1.], device=self.device)

# drone init
pos = torch.zeros(len(env_ids), 3, device=self.device)
pos = pos + self.origin  # init: (0, 0, 1)

# trajectory phase offset for figure_eight_normal
elif self.eval_traj == 'normal':
    self.ref = Lemniscate(T=5.5, origin=self.origin, device=self.device)
    self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
```

**Lemniscate position function:**
```python
def pos(self, t):
    sin_t = torch.sin(2 * torch.pi * t / self.T)
    cos_t = torch.cos(2 * torch.pi * t / self.T)
    x = torch.stack([cos_t, sin_t * cos_t, torch.zeros_like(t)], dim=-1)
    return (x + self.origin).to(self.device)
```

**Key finding:** SimpleFlight sets `traj_t0 = T/4 = 5.5/4 = 1.375s` for `figure_eight_normal`. At `t = T/4`:
- `sin(2π·(T/4)/T) = sin(π/2) = 1`
- `cos(2π·(T/4)/T) = cos(π/2) = 0`
- Trajectory position = `(0, 0, 0) + origin = (0, 0, 1)`

**The drone and the trajectory both start at (0, 0, 1). Initial position error = 0.**

The paper never states this explicitly — it is code-only behavior.

---

## Check 2 — SimpleFlight MED Aggregation

**Source:** Paper (arXiv 2412.11764), Section IV-A-3:

> "We assess tracking performance using the Mean Euclidean Distance (MED) between the quadrotor's actual and target positions in the x- and y-axes, averaged over the entire trajectory. For the figure-eight trajectory, MED is averaged over ten repetitions per trial across three trials."

**Source:** `omni_drones/envs/single/track.py`:
```python
distance = torch.norm(self.rpos[:, [0]], dim=-1)   # Euclidean distance to target
self.stats["tracking_error"].add_(-distance)        # accumulates sum of NEGATIVE distances
# at episode end:
self.stats["tracking_error"].div_(ep_len)           # divide by episode length = mean
```

**Source:** `scripts/eval.py`:
```python
info = {
    "eval/stats." + k: torch.nanmean(v.float()).item()
    for k, v in traj_stats.items()
}
```

**Key findings:**
- MED = **arithmetic mean** (not median, not RMSE) of per-step Euclidean distance
- Computed over **entire trajectory** — no steady-state exclusion window
- Averaged across multiple trial runs (3 trials × 10 reps for figure-eight)
- The paper says "x- and y-axes" but the code uses full 3D norm. Z error is small in practice since the lemniscate is horizontal.

---

## Check 3 — Acquisition Error Profile (first 200 steps)

**Plot:** `notes/m1_3_acq_error.png`

**With current eval (drone at (0,0,1), trajectory starts at (1,0,1), t=0):**

| Metric | Value |
|---|---|
| Initial XY error (step 0) | **1.000 m** |
| XY error at step 50 | 0.460 m |
| XY error at step 100 | 0.032 m |
| First step below 0.05m | **step 92, t=0.92s** |
| Mean over steps 0–200 | 0.255 m (acquisition-inflated) |
| Mean over steps 100–200 (steady-state) | **0.017 m** |

Error decays **smoothly and monotonically** from 1.0m to ~0.025m within 100 steps (~1s). The policy is doing the best physically possible with a 1m impossible initial offset. No instability, no divergence.

**T/4 phase geometry check (trajectory start, not a separate rollout):**
- Reference position at t=T/4 (step 138): `(-0.006, -0.006, 1.000)` — 0.0081m from drone spawn `(0, 0, 1)`
- Confirms: with T/4 offset, initial XY error ≈ 0 by construction

Note: a "T/4-shifted mean over 200 steps" cannot be computed from this rollout — it would require a separate rollout where the actor observations are built against the shifted reference. The 0.0081m initial offset is the relevant datum.

---

## Root Cause of Our Eval Discrepancy

Our implementation uses `t=0` as the trajectory start. At `t=0`:
- `x(0) = cos(0) = 1`
- `y(0) = sin(0)/2 = 0`
- Reference position = **(1, 0, 1)**

Drone spawns at **(0, 0, 1)**.
Initial XY offset = **1.0m**.

SimpleFlight uses `t=T/4` as the trajectory start. At `t=T/4 = 1.375s`:
- `x(T/4) = cos(π/2) = 0`
- `y(T/4) = sin(π)/2 = 0`
- Reference position = **(0, 0, 1)**

Drone spawns at **(0, 0, 1)**.
Initial XY offset = **0.0m**.

The same Lemniscate parametrization, but a T/4 phase shift eliminates the initial offset. SimpleFlight's eval by design has no acquisition phase. Ours has a 1m cold-start that inflates the mean by ~0.044m.

---

## Verdict

**Our 0.066–0.069m training eval is not comparable to the paper's methodology.** It is an eval design artifact: our trajectory starts at t=0 which puts the reference 1m from the drone's spawn point. SimpleFlight avoids this entirely by using a T/4 phase offset so the trajectory starts where the drone is.

**The correct fix is methodologically justified:**
Change `_run_med_eval` to start the trajectory at `traj_t0 = T/4`, matching SimpleFlight's code exactly. This is **not** "making a failing run pass by changing the metric" — it is correcting our eval to match the paper's actual methodology, which is documented in the SimpleFlight codebase.

| | Our current eval | SimpleFlight eval | Verdict |
|---|---|---|---|
| Drone init position | (0, 0, 1) | (0, 0, 1) | Same |
| Trajectory start | t=0 → (1, 0, 1) | t=T/4 → (0, 0, 1) | **Different — this is the bug** |
| Initial XY error | 1.0m | 0.0m | **Different** |
| Aggregation | mean over full episode | mean over full episode | Same |
| Steady-state exclusion | none | none | Same |

**Before the fix:**
- Our mean: ~0.069m (acquisition-inflated)
- Paper: 0.028m
- Apparent gap: 2.5×

**After the fix (predicted):**
- Our mean: ~0.025–0.033m (no acquisition artifact)
- Paper: 0.028m
- Gap: likely within noise / trial-averaging variance

---

## Required Change to `_run_med_eval`

Apply T/4 phase offset in `train_m1.py`:

```python
# Current (wrong): trajectory starts at t=0, reference at (1,0,1), drone at (0,0,1)
# This produces a 1m cold-start that inflates the mean by ~0.044m

# Fixed: apply T/4 phase offset matching SimpleFlight's traj_t0 = T/4
# At t=T/4, lemniscate evaluates to (0,0,1) = drone spawn position → initial error = 0
EVAL_TRAJ_T0_STEPS = int(round((5.5 / 4.0) / DT))  # 137 steps for DT=0.01

# In the rollout loop, offset reference lookups by EVAL_TRAJ_T0_STEPS:
ref_pos = get_reference_pos(_eval_traj, jnp.int32(si + EVAL_TRAJ_T0_STEPS))
# (and correspondingly for get_reference_window in obs construction)
```

MED aggregation (mean over full episode) is already correct — no other changes needed.

**Justification:** Direct code evidence from thu-uav/SimpleFlight `track.py`: `traj_t0 = T/4`. The paper does not state this; the code is the authority.
