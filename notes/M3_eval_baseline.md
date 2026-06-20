# M3 Eval Baseline — epoch 5899 (38% trained)

The "before" snapshot to compare the finished 500M run against.
Source: `scripts/eval_m3.py --n 128 --only random,fault70,clear` (greedy, 1000 steps).
Checkpoint: `/home/forke/m3_checkpoints/m3_run1/epoch_005899` (193M env-steps).
Episodes stratified by **ideal-path clearance** (min dist from the commanded
trajectory to any obstacle): unfair <0.15m, tight 0.15–0.35, moderate 0.35–0.5,
clear ≥0.5. "unfair" = commanded path inside the crash zone (perfect tracking
still crashes); the policy can only survive by deviating off the line.

## random (nominal, η=1.0), n=128

| Metric | Value |
|---|---|
| overall CF | 72.7% |
| **fair CF** (clearance ≥0.15m, n=91) | **93.4%** (gate ≥80% ✓) |
| MED on collision-free | 0.158 m (gate ≤0.10m ✗) |

By difficulty: unfair (n=37) CF **21.6%** · tight (n=20) CF 80.0% · moderate (n=19) CF 94.7% · clear (n=52) CF 98.1%
(crash rate 0% on moderate+clear — never fumbles a freebie.)

## fault70 (η=0.70 single rotor), n=128

| Metric | Value |
|---|---|
| overall CF | 37.5% |
| fair CF (n=82) | 53.7% (gate ≥50% ~✓) |
| **OOB rate** | **39.8%** — loses attitude control under fault |
| MED on collision-free | 0.173 m |

By difficulty: unfair (n=46) CF **8.7%** · tight (n=24) CF 41.7% · moderate (n=8) CF 50.0% · clear (n=50) CF 60.0%

## clear (no obstacles), n=128

CF 94.5% · MED 0.180 m (tracking-only; harder poly/zigzag traj than M2's figure-eight, not directly comparable to M2's 0.057m)

## The numbers that should improve at 500M

| Signal | Now (38%) | Why it should improve |
|---|---|---|
| Tracking MED | ~0.16 m | M2 needed full schedule to sharpen; reward near-saturated, watch for plateau |
| Fault OOB | 39.8% | encoder predicted to settle ~200M; we're at 193M |
| Unfair save-rate (deviate-to-survive) | 21.6% nom / 8.7% fault | the skill is emerging (confirmed on video); more exposure should make it reliable |

**Key watch:** prove the unfair save-rate goes UP (deviate-to-survive got more
reliable), not just that tracking sharpened. If save-rate stalls while tracking
improves, the policy learned precision at the cost of survival instinct — then
consider the M3.1 idea of an explicit actor-side danger signal.
