"""
Domain randomization for M1.

M1: Only thrust coefficient k_f is randomized (±30%).
M2 will add: per-rotor efficiency, mass variation, wind.

Per the SimpleFlight paper: randomize ONLY sensitive, uncalibrated parameters.
Randomizing well-calibrated parameters (mass, inertia) hurts performance.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


def sample_kf_multiplier(key: jnp.ndarray) -> jnp.ndarray:
    """
    Sample thrust coefficient multiplier: U(0.7, 1.3).
    Applied as k_f_actual = k_f_nominal * multiplier.
    """
    return jax.random.uniform(key, shape=(), minval=0.7, maxval=1.3)


def sample_env_params(key: jnp.ndarray) -> dict:
    """
    Sample all randomized parameters for one episode.
    Returns a dict to be passed into the environment.
    """
    k1, k2 = jax.random.split(key, 2)
    return {
        "kf_multiplier": sample_kf_multiplier(k1),
        # M2+ will add:
        # "rotor_efficiency": jax.random.uniform(k2, (4,), 0.3, 1.0),
        # "mass_multiplier": ...,
        # "wind_force": ...,
    }
