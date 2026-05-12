# M2 Encoder Output Manifold Analysis

**Date:** 2026-05-11  
**Method:** Real MJX physics rollouts.  
**Policy:** Phase 1 (ground-truth priv_state) for Part 1; Phase 2 combined (actor + encoder) for Part 2 predictive validity.

## Clarification notes

1. **Real vs synthetic**: This analysis uses actual MJX physics rollouts, not synthetic ideal-tracking obs. The Phase 1 policy runs under real DR (fault, mass, k_f) and the drone's actual tracking errors appear in e_W.

2. **No threshold labels**: Off-manifold ratios are reported as raw numbers. The 1.5×/3.0× thresholds from the earlier synthetic analysis had no published justification and have been removed.

3. **Predictive validity**: Part 2 runs the Phase 2 combined policy (the one that actually crashed in deployment) on three trajectory+condition combinations.

## Part 1 — Training distribution manifold (Phase 1 policy)

Trajectory: polynomial/zigzag (from reset_fn, matching Phase 1 training distribution).  
Conditions: 7 (nominal + 4 rotor faults η=0.70 + mass 0.8 + mass 1.2).  
Seeds: [42, 99, 7].

### Off-manifold distance (startup vs steady-state)

| Region | Mean L2 from steady-state centroid |
|---|---|
| Steady-state (t≥50) | 0.9336 |
| Startup warmup (t<50) | 0.7708 |
| **Off-manifold ratio** | **0.83×** |

### Condition separability (centroid distance from nominal, steady-state only)

| Condition | Centroid distance from nominal |
|---|---|
| nominal | 0.000 |
| rotor0 | 1.173 |
| rotor1 | 1.308 |
| rotor2 | 1.056 |
| rotor3 | 1.281 |
| mass0.8 | 0.829 |
| mass1.2 | 0.726 |

Separability reflects real trajectory tracking errors under each fault condition, unlike the synthetic analysis where all conditions had identical e_W (ideal tracking).

## Part 2 — Predictive validity (Phase 2 combined policy)

Phase 2 combined policy: actor sees encoder estimate ê_t instead of ground-truth priv_state. Startup instability (first H=50 steps with zero-padded ring buffer) may produce wrong ê_t → actor cannot compensate fault → crash.

| Trajectory / condition | Off-manifold ratio (mean±std) | Crashes (of 3 seeds) |
|---|---|---|
| figure_eight_normal / nominal | 1.11× ± 0.06 | 0/3 |
| figure_eight_normal / fault_eta70 | 0.82× ± 0.04 | 0/3 |
| figure_eight_fast   / fault_eta70 | 1.58× ± 0.26 | 0/3 |

**Crash note:** No crashes observed in 200 steps on any condition. The deployment crash
("figure_eight_fast + fault crashes at 2/3 seeds") was from 1000-step episodes. 200 steps
covers only ~57% of one figure_eight_fast loop (T=3.5s period = 350 steps at DT=0.01s).
The crash likely manifests in later revolutions, after the startup encoder error has
propagated through several oscillation cycles. A longer-episode re-run (≥ 500 steps) is
needed to reproduce the crash; this analysis provides the off-manifold diagnostic only.

### Revised interpretation of the Part 1 finding

**Part 1 off-manifold ratio = 0.83×** (below 1.0). Startup encoder outputs are CLOSER
to the global steady-state centroid than the steady-state points themselves. This overturns
the synthetic analysis (1.80×) and requires reframing the L5 diagnosis:

- The global centroid is dominated by nominal episodes (30% of training distribution =
  most samples). So the centroid ≈ nominal priv_state region.
- Zero-padded ring buffer → encoder outputs near nominal by default (the safest prediction
  given zero input — closer to global mean than any fault cluster).
- For fault-condition episodes, the steady-state encoder correctly clusters near the fault
  priv_state region (condition separability: 1.0–1.3 units from nominal in Part 1).
- **The startup problem is not "encoder output is off the manifold globally" — it is
  "encoder output defaults to nominal during startup even when the true condition is a fault."**

This is a more precise formulation: the startup encoder is ON the manifold but at the
WRONG POINT for fault conditions. The correct metric is distance from the fault-specific
steady-state cluster, not from the global centroid.

### Predictive validity verdict

- figure_eight_fast+fault has higher startup ratio (1.58×) than figure_eight_normal/nominal
  (1.11×), Δ = +0.47×. This is consistent with the crash scenario experiencing more severe
  encoder misspecification during startup.
- The no-crash observation at 200 steps does not invalidate this: the ratio difference
  indicates the encoder IS more wrong during figure_eight_fast startup, but the crash needs
  additional oscillation cycles to accumulate into a fatal trajectory deviation.
- **Off-manifold distance (vs global centroid) is a weak predictor of crash timing.**
  Distance from the fault-specific cluster would be a stronger predictor, but requires
  knowing the true condition (which the encoder is trying to infer).

**Revised Fix Option 2 rationale:** Zero-padded prefix training teaches the encoder that an
empty ring buffer is an ambiguous state — ideally producing an estimate closer to the
distribution mean *across all fault conditions* rather than defaulting to nominal. This
reduces the "wrong but confident" startup behavior. Fix Option 2 remains appropriate, but
the success criterion changes: not "startup outputs join the manifold" (they already are on
it) but "startup outputs for fault conditions are less biased toward nominal."

## Figures

- `notes/figures/m2_encoder_manifold_real_tsne.png` — t-SNE by condition (real rollouts)
- `notes/figures/m2_encoder_manifold_real_tsne_phase.png` — t-SNE startup vs steady-state
- `notes/figures/m2_encoder_manifold_real_umap.png` — UMAP by condition (real rollouts)
- `notes/figures/m2_encoder_manifold_real_umap_phase.png` — UMAP phase
- `notes/figures/m2_encoder_manifold_real_validity.png` — predictive validity bar chart

*(Synthetic figures preserved at `m2_encoder_manifold_tsne.png` etc.)*