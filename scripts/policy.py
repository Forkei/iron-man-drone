"""
SimpleFlight actor-critic networks.
Architecture: 3-layer MLP, 256 hidden, ELU + LayerNorm, asymmetric actor-critic.
Reference: Chen et al., RAL 2025, Section III-B.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim)])
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Input:  [e^W (30), v (3), R (9)] = 42-dim
    Output: Gaussian distribution over CTBR [omega_x, omega_y, omega_z, c]

    IMPORTANT constraints (from SimpleFlight ablation):
    - Rotation matrix, NOT quaternion
    - Does NOT receive previous action u_{t-1}
    - Does NOT receive time step k (critic only)
    """

    def __init__(
        self,
        obs_dim: int = 42,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = _build_mlp(obs_dim, hidden_dim, action_dim * 2, num_layers)
        nn.init.constant_(self.net[-1].bias[action_dim:], log_std_init)

    def forward(self, obs: torch.Tensor) -> Normal:
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def get_action_and_logprob(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, dist.entropy().sum(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Input:  [e^W (30), v (3), R (9), k (1)] = 43-dim  (privileged: timestep k)
    Output: scalar state value

    Asymmetric: receives time step k which actor does NOT see.
    Putting k in actor causes OOD failures on long-horizon flights.
    """

    def __init__(
        self,
        obs_dim: int = 43,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dim, 1, num_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class SimpleFightPolicy(nn.Module):
    """Container for actor + critic with separate parameter groups."""

    def __init__(
        self,
        actor_obs_dim: int = 42,
        critic_obs_dim: int = 43,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.actor = Actor(actor_obs_dim, action_dim, hidden_dim, num_layers)
        self.critic = Critic(critic_obs_dim, hidden_dim, num_layers)

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()
