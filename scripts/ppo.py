"""
PPO trainer for SimpleFlight.
Hyperparameters from Chen et al., RAL 2025, Table VI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4      # intentionally lower than actor_lr
    critic_updates: int = 16     # critic gradient steps per PPO update
    entropy_coeff: float = 1e-3  # must stay << reward magnitude
    max_grad_norm: float = 0.5
    horizon: int = 32
    minibatch_size: int = 256
    num_envs: int = 128


@dataclass
class RolloutBuffer:
    """Stores one horizon of experience across all parallel envs."""
    actor_obs: list = field(default_factory=list)
    critic_obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self):
        self.actor_obs.clear()
        self.critic_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(self, actor_obs, critic_obs, action, log_prob, reward, value, done):
        self.actor_obs.append(actor_obs)
        self.critic_obs.append(critic_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def to_tensors(self, device: torch.device):
        return {
            "actor_obs":  torch.stack(self.actor_obs).to(device),
            "critic_obs": torch.stack(self.critic_obs).to(device),
            "actions":    torch.stack(self.actions).to(device),
            "log_probs":  torch.stack(self.log_probs).to(device),
            "rewards":    torch.stack(self.rewards).to(device),
            "values":     torch.stack(self.values).to(device),
            "dones":      torch.stack(self.dones).to(device),
        }


class PPOTrainer:
    def __init__(self, policy, cfg: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device

        # CRITICAL: separate optimizers for actor and critic
        self.actor_opt = torch.optim.Adam(policy.actor_parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(policy.critic_parameters(), lr=cfg.critic_lr)

        self.buffer = RolloutBuffer()

    def compute_gae(
        self,
        rewards: torch.Tensor,   # (T, N)
        values: torch.Tensor,    # (T, N)
        dones: torch.Tensor,     # (T, N)
        last_value: torch.Tensor,  # (N,)
    ):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_actor_obs: torch.Tensor, last_critic_obs: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            last_value = self.policy.critic(last_critic_obs)

        data = self.buffer.to_tensors(self.device)
        advantages, returns = self.compute_gae(
            data["rewards"], data["values"], data["dones"], last_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T, N = data["actor_obs"].shape[:2]
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in data.items()}
        flat["advantages"] = advantages.reshape(T * N)
        flat["returns"] = returns.reshape(T * N)

        # --- Actor update (one pass) ---
        actor_loss_total = 0.0
        entropy_total = 0.0
        idx = torch.randperm(T * N, device=self.device)

        for start in range(0, T * N, self.cfg.minibatch_size):
            mb_idx = idx[start : start + self.cfg.minibatch_size]
            new_log_prob, entropy = self.policy.actor.evaluate_actions(
                flat["actor_obs"][mb_idx], flat["actions"][mb_idx]
            )
            ratio = (new_log_prob - flat["log_probs"][mb_idx]).exp()
            adv = flat["advantages"][mb_idx]
            pg_loss = -torch.min(
                ratio * adv,
                ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv,
            ).mean()
            actor_loss = pg_loss - self.cfg.entropy_coeff * entropy.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()

            actor_loss_total += actor_loss.item()
            entropy_total += entropy.mean().item()

        # --- Critic update (more iterations than actor) ---
        critic_loss_total = 0.0
        for _ in range(self.cfg.critic_updates):
            idx = torch.randperm(T * N, device=self.device)
            for start in range(0, T * N, self.cfg.minibatch_size):
                mb_idx = idx[start : start + self.cfg.minibatch_size]
                value_pred = self.policy.critic(flat["critic_obs"][mb_idx])
                critic_loss = F.mse_loss(value_pred, flat["returns"][mb_idx])

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()

                critic_loss_total += critic_loss.item()

        self.buffer.clear()

        n_actor_batches = max(1, (T * N) // self.cfg.minibatch_size)
        n_critic_batches = max(1, self.cfg.critic_updates * (T * N) // self.cfg.minibatch_size)
        return {
            "actor_loss": actor_loss_total / n_actor_batches,
            "critic_loss": critic_loss_total / n_critic_batches,
            "entropy": entropy_total / n_actor_batches,
        }
