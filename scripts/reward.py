"""
SimpleFlight reward functions.
Reference: Chen et al., RAL 2025, Section III-C.

r_total = r_task + lambda_smooth * r_smooth

Explicitly NOT used (per ablation study):
- action magnitude penalty (||u_t||² alone)
- jerk/snap penalties
- action clipping
- low-pass filter on policy output
"""

import torch


def reward_task(pos: torch.Tensor, ref_pos: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Trajectory tracking reward: negative L2 distance, normalized to [0, 1].
    pos, ref_pos: (..., 3) world-frame positions
    """
    dist = (pos - ref_pos).norm(dim=-1)
    return torch.exp(-scale * dist)


def reward_smooth(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """
    Action smoothness reward: exp(-||u_t - u_{t-1}||²), range [0, 1].
    This is the key smoothness formulation from SimpleFlight.
    NOT ||u_t||² alone — that form penalizes magnitude rather than rate-of-change.
    """
    diff_sq = (action - prev_action).pow(2).sum(dim=-1)
    return torch.exp(-diff_sq)


def compute_reward(
    pos: torch.Tensor,
    ref_pos: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    lambda_smooth: float = 0.4,
    task_scale: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    r_task = reward_task(pos, ref_pos, scale=task_scale)
    r_smooth = reward_smooth(action, prev_action)
    r_total = r_task + lambda_smooth * r_smooth

    return r_total, {"r_task": r_task.mean().item(), "r_smooth": r_smooth.mean().item()}
