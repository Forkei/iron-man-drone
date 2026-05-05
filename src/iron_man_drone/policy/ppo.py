"""
PPO trainer — PureJaxRL style, adapted for SimpleFlight asymmetric actor-critic.

Key design decisions (from SimpleFlight + M1 spec):
  - Separate actor and critic TrainStates with separate optimizers
  - Critic LR (1e-4) < Actor LR (3e-4) — critical, do not share
  - 16 critic updates per actor update
  - Entropy coefficient 1e-3 — must stay << reward magnitude
  - GAE with γ=0.99, λ=0.95
  - jax.lax.scan for rollout loop (fully JIT-compiled, no Python overhead)

Reference: SimpleFlight Table VI + PureJaxRL (Lu et al., 2022).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import distrax

from .networks import Actor, Critic


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Hyperparameters from SimpleFlight Table VI
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4          # separate, lower than actor — critical
    critic_updates: int = 16         # critic steps per actor update
    entropy_coeff: float = 1e-3      # must be << reward magnitude
    max_grad_norm: float = 0.5
    value_loss_coeff: float = 0.5
    num_minibatches: int = 8
    ppo_epochs: int = 5              # passes over each rollout buffer
    horizon: int = 32                # steps per env per rollout
    num_envs: int = 1024
    actor_obs_dim: int = 50   # M2: [e^W(30), v(3), R(9), z(8)]
    critic_obs_dim: int = 51  # M2: actor_base(42) + e_t(8) + k(1)
    action_dim: int = 4
    hidden_dim: int = 256
    num_layers: int = 3


# ── Rollout transition ────────────────────────────────────────────────────────

class Transition(NamedTuple):
    actor_obs: jnp.ndarray    # (num_envs, actor_obs_dim)
    critic_obs: jnp.ndarray   # (num_envs, critic_obs_dim)
    action: jnp.ndarray       # (num_envs, action_dim)
    log_prob: jnp.ndarray     # (num_envs,)
    reward: jnp.ndarray       # (num_envs,)
    value: jnp.ndarray        # (num_envs,)
    done: jnp.ndarray         # (num_envs,) bool


# ── Train states (separate for actor and critic) ──────────────────────────────

def create_train_states(key: jnp.ndarray, cfg: PPOConfig):
    actor = Actor(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    critic = Critic(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)

    k1, k2 = jax.random.split(key)
    dummy_actor_obs = jnp.zeros((1, cfg.actor_obs_dim))
    dummy_critic_obs = jnp.zeros((1, cfg.critic_obs_dim))

    actor_params = actor.init(k1, dummy_actor_obs)
    critic_params = critic.init(k2, dummy_critic_obs)

    actor_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.actor_lr),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.critic_lr),
    )

    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_tx)

    return actor, critic, actor_state, critic_state


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(
    transitions: Transition,
    last_value: jnp.ndarray,   # (num_envs,)
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GAE advantages and returns.
    transitions.reward/value/done have shape (horizon, num_envs).
    Returns (advantages, returns), each (horizon, num_envs).
    """
    def _scan_gae(carry, transition):
        last_gae, last_value = carry
        done, value, reward = transition.done, transition.value, transition.reward
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * last_value * not_done - value
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        return (last_gae, value), last_gae

    # Scan in reverse over time
    _, advantages = jax.lax.scan(
        _scan_gae,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    returns = advantages + transitions.value
    return advantages, returns


# ── PPO update (one epoch over minibatches) ───────────────────────────────────

def _actor_loss_fn(actor_params, actor_apply, transitions, advantages, cfg):
    mean, log_std = actor_apply(actor_params, transitions.actor_obs)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    new_log_prob = dist.log_prob(transitions.action)
    entropy = dist.entropy()

    ratio = jnp.exp(new_log_prob - transitions.log_prob)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -ratio * adv
    pg_loss2 = -jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    entropy_loss = -cfg.entropy_coeff * entropy.mean()

    return pg_loss + entropy_loss, (pg_loss, entropy.mean())


def _critic_loss_fn(critic_params, critic_apply, transitions, returns):
    values = critic_apply(critic_params, transitions.critic_obs)
    return jnp.mean((values - returns) ** 2), values.mean()


def ppo_update(
    actor_state: TrainState,
    critic_state: TrainState,
    transitions: Transition,   # (horizon, num_envs, ...)
    last_value: jnp.ndarray,
    key: jnp.ndarray,
    cfg: PPOConfig,
) -> tuple[TrainState, TrainState, dict]:
    """One full PPO update (multiple epochs, minibatches)."""
    # Compute GAE
    advantages, returns = compute_gae(transitions, last_value, cfg.gamma, cfg.gae_lambda)

    # Flatten (horizon, num_envs) → (horizon * num_envs,)
    T, N = transitions.reward.shape
    flat = jax.tree_util.tree_map(lambda x: x.reshape(T * N, *x.shape[2:]), transitions)
    flat_advantages = advantages.reshape(T * N)
    flat_returns = returns.reshape(T * N)

    def _epoch(carry, _):
        actor_s, critic_s, key = carry
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, T * N)
        mb_size = T * N // cfg.num_minibatches

        def _minibatch(carry, mb_idx):
            actor_s, critic_s = carry
            # mb_idx is a traced int inside lax.scan — must use dynamic slice
            idx = jax.lax.dynamic_slice_in_dim(perm, mb_idx * mb_size, mb_size)
            mb_trans = jax.tree_util.tree_map(lambda x: x[idx], flat)
            mb_adv = flat_advantages[idx]
            mb_ret = flat_returns[idx]

            # Actor update
            (a_loss, (pg, ent)), a_grad = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_s.params, actor_s.apply_fn, mb_trans, mb_adv, cfg)
            actor_s = actor_s.apply_gradients(grads=a_grad)

            # Critic update (cfg.critic_updates times)
            def _critic_step(c_s, _):
                (c_loss, c_val), c_grad = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(c_s.params, c_s.apply_fn, mb_trans, mb_ret)
                return c_s.apply_gradients(grads=c_grad), c_loss

            critic_s, c_losses = jax.lax.scan(
                _critic_step, critic_s, None, length=cfg.critic_updates
            )

            return (actor_s, critic_s), {
                "actor_loss": a_loss,
                "pg_loss": pg,
                "entropy": ent,
                "critic_loss": c_losses.mean(),
            }

        (actor_s, critic_s), mb_metrics = jax.lax.scan(
            _minibatch, (actor_s, critic_s), jnp.arange(cfg.num_minibatches)
        )
        return (actor_s, critic_s, key), jax.tree_util.tree_map(jnp.mean, mb_metrics)

    (actor_state, critic_state, _), epoch_metrics = jax.lax.scan(
        _epoch, (actor_state, critic_state, key), None, length=cfg.ppo_epochs
    )
    metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
    return actor_state, critic_state, metrics


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(
    actor_state: TrainState,
    critic_state: TrainState,
    env_states,               # batched EnvState
    actor_obs: jnp.ndarray,   # (num_envs, actor_obs_dim)
    critic_obs: jnp.ndarray,  # (num_envs, critic_obs_dim)
    env_step_fn,              # vmapped env step: (states, actions) -> ...
    env_reset_fn,             # vmapped env reset: (keys,) -> ...
    key: jnp.ndarray,
    cfg: PPOConfig,
):
    """
    Collect cfg.horizon steps of experience using jax.lax.scan.

    M2: no kf_multipliers arg — all DR params live in EnvState and are
    re-sampled automatically when reset() is called on episode end.

    Returns (transitions, final_states, final_actor_obs, final_critic_obs).
    """
    def _env_step(carry, _):
        states, a_obs, c_obs, key = carry

        key, act_key, reset_key = jax.random.split(key, 3)
        mean, log_std = actor_state.apply_fn(actor_state.params, a_obs)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        actions = dist.sample(seed=act_key)
        log_probs = dist.log_prob(actions)

        values = critic_state.apply_fn(critic_state.params, c_obs)

        # Step — DR read from state (no external arg)
        new_states, new_a_obs, new_c_obs, rewards, dones = env_step_fn(
            states, actions
        )

        # Auto-reset: fresh DR params sampled from reset_key per env
        reset_keys = jax.random.split(reset_key, cfg.num_envs)
        reset_states, reset_a_obs, reset_c_obs = env_reset_fn(reset_keys)

        def _where_done(n, r):
            mask = dones.reshape((dones.shape[0],) + (1,) * (n.ndim - 1))
            return jnp.where(mask, r, n)

        new_states = jax.tree_util.tree_map(_where_done, new_states, reset_states)
        new_a_obs = jnp.where(dones[:, None], reset_a_obs, new_a_obs)
        new_c_obs = jnp.where(dones[:, None], reset_c_obs, new_c_obs)

        transition = Transition(
            actor_obs=a_obs,
            critic_obs=c_obs,
            action=actions,
            log_prob=log_probs,
            reward=rewards,
            value=values,
            done=dones,
        )
        return (new_states, new_a_obs, new_c_obs, key), transition

    (final_states, final_a_obs, final_c_obs, _), transitions = jax.lax.scan(
        _env_step,
        (env_states, actor_obs, critic_obs, key),
        None,
        length=cfg.horizon,
    )
    return transitions, final_states, final_a_obs, final_c_obs
