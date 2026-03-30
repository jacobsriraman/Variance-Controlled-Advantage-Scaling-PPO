from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


RewardTransform = Literal["identity", "scale", "zscore", "minmax", "tanh"]


@dataclass
class RewardTransformConfig:
    mode: RewardTransform = "identity"
    scale: float = 1.0
    target_std: float = 1.0
    tanh_gain: float = 1.0
    minmax_low: float = -1.0
    minmax_high: float = 1.0
    eps: float = 1e-8


def transform_rewards(
    rewards: np.ndarray,
    cfg: RewardTransformConfig,
) -> Tuple[np.ndarray, dict]:
    """
    Transform a 1D reward array collected over one rollout.

    This is intentionally explicit so the experiment can measure how reward
    distribution changes affect PPO optimization.

    Returns
    -------
    transformed_rewards : np.ndarray
    stats : dict
        Mean/variance summary of raw and transformed rewards.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    raw_mean = float(rewards.mean()) if rewards.size else 0.0
    raw_var = float(rewards.var()) if rewards.size else 0.0

    if cfg.mode == "identity":
        out = rewards.copy()

    elif cfg.mode == "scale":
        out = cfg.scale * rewards

    elif cfg.mode == "zscore":
        mu = rewards.mean()
        std = rewards.std()
        out = (rewards - mu) / (std + cfg.eps)
        out = cfg.target_std * out

    elif cfg.mode == "minmax":
        rmin = rewards.min()
        rmax = rewards.max()
        denom = (rmax - rmin) + cfg.eps
        unit = (rewards - rmin) / denom
        out = cfg.minmax_low + unit * (cfg.minmax_high - cfg.minmax_low)

    elif cfg.mode == "tanh":
        mu = rewards.mean()
        std = rewards.std()
        standardized = (rewards - mu) / (std + cfg.eps)
        out = cfg.target_std * np.tanh(cfg.tanh_gain * standardized)

    else:
        raise ValueError(f"Unknown reward transform mode: {cfg.mode}")

    transformed_mean = float(out.mean()) if out.size else 0.0
    transformed_var = float(out.var()) if out.size else 0.0

    stats = {
        "raw_reward_mean": raw_mean,
        "raw_reward_var": raw_var,
        "transformed_reward_mean": transformed_mean,
        "transformed_reward_var": transformed_var,
    }
    return out.astype(np.float32), stats

