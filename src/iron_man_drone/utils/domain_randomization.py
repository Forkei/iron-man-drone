"""
Domain randomization.

M1: Only thrust coefficient k_f is randomized (±30%).
M2: Adds per-rotor efficiency (MAVEN convention: one rotor, η ∈ [0.5,1.0],
    70% of episodes), mass scale (±20%), privileged state vector.

DR params are now sampled inside reset() (per-episode), not in the training loop.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


class DRParams(NamedTuple):
    """All per-episode randomized parameters. Constant within an episode."""
    kf_multiplier: jnp.ndarray     # scalar,  U(0.7, 1.3) — global thrust coeff
    rotor_efficiency: jnp.ndarray  # (4,),   per-rotor efficiency ∈ [0.5, 1.0]
    mass_scale: jnp.ndarray        # scalar,  U(mass_lo, mass_hi)
    priv_state: jnp.ndarray        # (8,),   [η1,η2,η3,η4, mass_scale, Fx,Fy,Fz]
                                   #          wind slots Fx/Fy/Fz = 0 in Phase 1


def sample_kf_multiplier(key: jnp.ndarray) -> jnp.ndarray:
    """M1-compat: sample global thrust coefficient multiplier U(0.7, 1.3)."""
    return jax.random.uniform(key, shape=(), minval=0.7, maxval=1.3)


def sample_m2_dr_params(
    key: jnp.ndarray,
    fault_prob: float = 0.7,
    eta_min: float = 0.5,
    mass_lo: float = 0.8,
    mass_hi: float = 1.2,
) -> DRParams:
    """
    Sample all M2 DR parameters for one episode.

    Rotor fault convention (MAVEN): one randomly selected rotor is degraded
    with probability fault_prob. All four are degraded independently would
    risk total thrust < weight (can't hover). Single-rotor keeps
    T/W ≥ (3 + eta_min)/4 × T/W_nominal ≈ 1.57 — always hoverable.

    Args:
        fault_prob: probability any given episode has a rotor fault (default 0.7)
        eta_min:    lower bound for the degraded rotor efficiency (default 0.5)
        mass_lo/hi: mass multiplier range (default [0.8, 1.2])
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Global thrust coefficient
    kf_mult = jax.random.uniform(k1, shape=(), minval=0.7, maxval=1.3)

    # Per-rotor efficiency — single-rotor MAVEN convention
    is_fault = jax.random.uniform(k2) < fault_prob
    fault_rotor = jax.random.randint(k3, shape=(), minval=0, maxval=4)
    # maxval=1.0-1e-4 so faulted rotors are always clearly below nominal.
    # U(eta_min, 1.0) would occasionally return values ≥ 0.999 that are
    # indistinguishable from nominal and dilute the fault training signal.
    fault_eta = jax.random.uniform(k4, shape=(), minval=eta_min, maxval=1.0 - 1e-4)

    # One-hot mask for the degraded rotor
    fault_mask = jax.nn.one_hot(fault_rotor, 4)  # (4,): 1.0 at fault rotor
    fault_eff = (1.0 - fault_mask) + fault_mask * fault_eta  # one rotor degraded
    rotor_efficiency = jnp.where(is_fault, fault_eff, jnp.ones(4))

    # Mass scale
    mass_scale = jax.random.uniform(k5, shape=(), minval=mass_lo, maxval=mass_hi)

    # Privileged state: [η1,η2,η3,η4, mass_scale, Fx,Fy,Fz]
    # Wind slots are 0 in Phase 1 (wind added to Phase 1 training in a later iteration
    # only if OOD eval shows wind robustness is needed).
    priv_state = jnp.concatenate([
        rotor_efficiency,              # (4,)
        jnp.array([mass_scale]),       # (1,)
        jnp.zeros(3),                  # (3,) wind — zero for Phase 1
    ])

    return DRParams(
        kf_multiplier=kf_mult,
        rotor_efficiency=rotor_efficiency,
        mass_scale=mass_scale,
        priv_state=priv_state,
    )


def nominal_m2_dr_params() -> DRParams:
    """Return fully nominal DR params (all η=1, mass=1, no fault). For validation."""
    return DRParams(
        kf_multiplier=jnp.ones(()),
        rotor_efficiency=jnp.ones(4),
        mass_scale=jnp.ones(()),
        priv_state=jnp.concatenate([jnp.ones(4), jnp.ones(1), jnp.zeros(3)]),
    )
