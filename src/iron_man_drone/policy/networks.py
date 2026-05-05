"""
SimpleFlight actor-critic networks in Flax.

Architecture: 3-layer MLP, hidden=256, ELU + LayerNorm, asymmetric.
  Actor:  obs (42-dim) → Gaussian over CTBR (4-dim)
  Critic: obs (43-dim) → scalar value

Reference: Chen et al., RAL 2025, Section III-B.
"""

from __future__ import annotations
from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


def _mlp_layers(hidden_dim: int, num_layers: int) -> list:
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Dense(hidden_dim))
        layers.append(nn.elu)
        layers.append(nn.LayerNorm())
    return layers


class Actor(nn.Module):
    """
    Input:  actor_obs (42-dim): [e^W (30), v (3), R (9)]
    Output: mean and log_std for 4-dim CTBR Gaussian

    Matches SimpleFlight paper Section III-B exactly.
    CRITICAL invariants:
    - Does NOT receive body rates ω (not in paper obs)
    - Does NOT receive previous action u_{t-1}
    - Does NOT receive timestep k (critic only)
    - Uses rotation matrix R (9-dim), never quaternion
    """
    hidden_dim: int = 256
    num_layers: int = 3
    action_dim: int = 4
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.elu(x)
            x = nn.LayerNorm()(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.log_std_init),
            (self.action_dim,),
        )
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_dist(self, obs: jnp.ndarray):
        import distrax
        mean, log_std = self(obs)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))


class Critic(nn.Module):
    """
    Input:  critic_obs (43-dim): actor_obs (42) + timestep k (1)
    Output: scalar state value

    Asymmetric — receives privileged timestep k that actor does NOT see.
    Putting k in actor causes OOD failures on long-horizon flights.
    """
    hidden_dim: int = 256
    num_layers: int = 3

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.elu(x)
            x = nn.LayerNorm()(x)
        return nn.Dense(1)(x).squeeze(-1)


def init_networks(
    key: jnp.ndarray,
    actor_obs_dim: int = 42,
    critic_obs_dim: int = 43,
    hidden_dim: int = 256,
    num_layers: int = 3,
):
    """Initialize both networks and return (actor, critic, actor_params, critic_params)."""
    actor = Actor(hidden_dim=hidden_dim, num_layers=num_layers)
    critic = Critic(hidden_dim=hidden_dim, num_layers=num_layers)

    k1, k2 = jax.random.split(key)
    dummy_actor_obs = jnp.zeros((1, actor_obs_dim))
    dummy_critic_obs = jnp.zeros((1, critic_obs_dim))

    actor_params = actor.init(k1, dummy_actor_obs)
    critic_params = critic.init(k2, dummy_critic_obs)

    return actor, critic, actor_params, critic_params
