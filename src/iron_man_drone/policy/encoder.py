"""
Phase 2 causal adaptation encoder.

Maps a 0.5-second history of (obs_base, action) pairs to a normalized prediction
of the privileged state e_t = [η₁, η₂, η₃, η₄, m_scale, Fx, Fy, Fz].

Architecture: flat MLP 2300 → 256 → 128 → 8 → tanh
  - Input:  H × (obs_base_dim + action_dim) = 50 × 46 = 2300 dims (flattened)
  - Hidden: ELU + LayerNorm
  - Output: 8-dim ê_t in [-1, 1] (normalized; denormalize before passing to actor)

Normalization (raw → normalized):
  η₁–η₄    : (x − 0.75) / 0.25   physical [0.50, 1.00]  → normalized [-1.0, 1.0]
  m_scale   : (x − 1.00) / 0.20   physical [0.80, 1.20]  → normalized [-1.0, 1.0]
  Fx/Fy/Fz  : (x − 0.00) / 1.00   always 0.0 in Phase 1  → normalized  0.0

Reference: RMA §IV-B (Kumar et al., RSS 2021, arXiv:2107.04034)
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn
import numpy as np

H          = 50    # history window length (steps at 100 Hz = 0.5 s)
OBS_DIM    = 42   # observable base obs: [e_W(30), v(3), R(9)]
ACTION_DIM = 4    # CTBR action dim
WINDOW_DIM = H * (OBS_DIM + ACTION_DIM)  # 50 × 46 = 2300
E_T_DIM    = 8    # privileged state dim

# Normalization constants: normalize(x) = (x - BIAS) / SCALE
# Clipped to [-1, 1] by the encoder's tanh output.
E_T_BIAS  = np.array([0.75, 0.75, 0.75, 0.75, 1.00, 0.0, 0.0, 0.0], dtype=np.float32)
E_T_SCALE = np.array([0.25, 0.25, 0.25, 0.25, 0.20, 1.0, 1.0, 1.0], dtype=np.float32)


def normalize_e_t(e_t: np.ndarray) -> np.ndarray:
    """Normalize raw privileged state to [-1, 1] for encoder targets."""
    return (e_t - E_T_BIAS) / E_T_SCALE


def denormalize_e_hat(e_hat_norm: jnp.ndarray) -> jnp.ndarray:
    """Denormalize encoder output back to physical units for actor input."""
    bias  = jnp.array(E_T_BIAS)
    scale = jnp.array(E_T_SCALE)
    return e_hat_norm * scale + bias


class AdaptationEncoder(nn.Module):
    """
    Causal adaptation encoder ϕ.

    Call:
        encoder = AdaptationEncoder()
        e_hat_norm = encoder.apply(params, window)
        # window: (batch, WINDOW_DIM) or (WINDOW_DIM,)
        # e_hat_norm: (batch, 8) or (8,) in [-1, 1]
        # e_hat_raw = denormalize_e_hat(e_hat_norm)  → physical units for actor
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(E_T_DIM)(x)
        return jnp.tanh(x)


def build_history_window(
    obs_base_buf: jnp.ndarray,   # (H, OBS_DIM) — oldest first
    action_buf: jnp.ndarray,     # (H, ACTION_DIM) — oldest first; action_buf[i] = action taken AFTER obs_base_buf[i]
) -> jnp.ndarray:
    """
    Flatten (obs_base, prev_action) pairs into a 2300-dim encoder input.

    Pair k = (obs_base_buf[k], action_buf[k-1]) for k=1..H-1; pair 0 = (obs_base_buf[0], 0).
    Causal: action[k-1] was taken after observing obs[k-1], before obs[k] was seen.

    In the ring buffer at step t, obs_base_buf[H-1] = current obs_base[t]
    and action_buf[H-2] = prev action (action taken at step t-1).
    """
    # Shift actions: pair[k] uses action from position k-1 (prev action)
    prev_actions = jnp.concatenate([
        jnp.zeros((1, ACTION_DIM)),  # no action before episode start
        action_buf[:-1],             # (H-1, ACTION_DIM)
    ], axis=0)  # (H, ACTION_DIM)
    pairs = jnp.concatenate([obs_base_buf, prev_actions], axis=-1)  # (H, 46)
    return pairs.reshape(-1)  # (2300,)
