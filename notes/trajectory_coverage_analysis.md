# Trajectory Coverage Analysis

**Date**: 2026-05-03  
**N**: 1000 polynomial + 1000 zigzag + 3 figure-eight variants  
**Resolution**: 10ms (100Hz), EPISODE_STEPS=1000 (10s)  

## Summary answer

**Figure-eight apex is IN-DISTRIBUTION for the training mix.**

Figure-eight (normal) apex: κ = 4.789 m⁻¹, ω = 3.622 rad/s.  
**100.0%** of training trajectories exceed this curvature. If apex overshoot persists after entropy fix, the cause is not distribution coverage.

---

## Figure-eight reference stats

| Variant | Period T | κ_max (m⁻¹) | ω_max (rad/s) | v_max (m/s) |
|---|---|---|---|---|
| slow | 15.0s | 4.790 | 1.328 | 0.592 |
| normal | 5.5s | 4.789 | 3.622 | 1.615 |
| fast | 3.5s | 4.786 | 5.689 | 2.538 |

---

## Polynomial training trajectories (N=1000)

Spatial bounds: ±1.5m XY. Seg duration: 1.5–4.0s. C2-continuous quintic polynomial (nonzero vel/acc at interior waypoints).

**Max curvature per trajectory κ_max (m⁻¹):**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 36.909 | 84.872 | 293.076 | 859.116 | 2127.400 | 2934.803 | 5090.858 |

**Max angular velocity per trajectory ω_max (rad/s):**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 7.810 | 12.081 | 20.961 | 38.695 | 63.634 | 79.759 | 121.624 |

Coverage of fig-8 normal apex: **100.0%** exceed κ=4.789 m⁻¹; **99.6%** exceed ω=3.622 rad/s

---

## Zigzag training trajectories (N=1000)

Spatial bounds: ±1.0m XY. Seg duration: 1.0–1.5s. Linear segments (theoretically infinite curvature at waypoints — sampled at 10ms resolution).

**Max curvature per trajectory κ_max (m⁻¹):**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 587.191 | 926.647 | 1898.279 | 4657.569 | 12187.243 | 22946.382 | 62708.363 |

**Max angular velocity per trajectory ω_max (rad/s):**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 160.484 | 213.440 | 312.620 | 494.167 | 792.038 | 1151.478 | 2260.487 |

Coverage of fig-8 normal apex: **100.0%** exceed κ=4.789 m⁻¹; **100.0%** exceed ω=3.622 rad/s

---

## OOD verdict and proposed M1.3 fixes

Mixed training (50/50 poly/zigzag): **100.0%** of trajectories exceed fig-8 normal apex curvature, **99.8%** exceed apex angular velocity.

**VERDICT: IN-DISTRIBUTION**

### Implication for M1.3

Apex curvature IS in the training distribution. If apex overshoot persists after M1.2 (entropy fix), investigate:
1. **Policy capacity**: 256-hidden 3-layer may not represent sharp-turn dynamics
2. **Reward shaping**: exp(-d²) gives weak gradient far from reference; consider exp(-d) or negative L2
3. **Observation horizon**: 10×50ms = 500ms lookahead; figure-eight apex turn window is ~200ms — policy may not see it coming early enough
