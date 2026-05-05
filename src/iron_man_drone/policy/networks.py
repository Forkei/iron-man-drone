"""
Actor-critic networks in Flax — M2 (RMA fault-tolerant variant).

Architecture: 3-layer MLP, hidden=256, ELU + LayerNorm, asymmetric.
  Actor:  obs (50-dim) → Gaussian over CTBR (4-dim)
  Critic: obs (51-dim) → scalar value

M2 obs layout:
  Actor  (50): [e^W(30), v(3), R(9), z(8)]     z = priv_state in Phase 1
  Critic (51): [e^W(30), v(3), R(9), e_t(8), k(1)]

Reference: Chen et al. SimpleFlight RAL 2025; Kumar et al. RMA arXiv 2107.04034.
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
    Input:  actor_obs (50-dim): [e^W (30), v (3), R (9), z (8)]
    Output: mean and log_std for 4-dim CTBR Gaussian

    M2: z is the 8-dim privileged latent. In Phase 1 it is e_t passed directly.
    In Phase 1 with encoder (next commit): z = μ(e_t). At deployment: z = ϕ(history).
    CRITICAL invariants (unchanged from M1):
    - Does NOT receive body rates ω
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
    Input:  critic_obs (51-dim): [e^W(30), v(3), R(9), e_t(8), k(1)]
    Output: scalar state value

    Asymmetric: critic receives raw privileged state e_t (ground truth physical
    params) for better value estimation. Actor gets z = μ(e_t) (learned latent).
    k (timestep) stays in critic only — OOD failure if put in actor.
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
    actor_obs_dim: int = 50,
    critic_obs_dim: int = 51,
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
