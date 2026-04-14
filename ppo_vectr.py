from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from reward_transforms import RewardTransformConfig, transform_rewards


def mlp(sizes: List[int], activation=nn.Tanh, output_activation=nn.Identity) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    use_vectr: bool = False
    vectr_target_std: float = 1.0

    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64

    target_kl: float = 0.02

    reward_transform: RewardTransformConfig = field(
        default_factory=RewardTransformConfig
    )

    hidden_sizes: tuple = (64, 64)
    device: str = "cpu"

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_space, hidden_sizes=(64, 64)):
        super().__init__()
        self.is_discrete = hasattr(action_space, "n")
        self.is_continuous = not self.is_discrete

        if self.is_discrete:
            self.actor = mlp([obs_dim, *hidden_sizes, action_space.n])
        else:
            act_dim = int(np.prod(action_space.shape))
            self.actor_mean = mlp([obs_dim, *hidden_sizes, act_dim])
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = mlp([obs_dim, *hidden_sizes, 1])

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def _continuous_dist(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist, mean

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action, logprob, entropy, value
        """
        if self.is_discrete:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.value(obs)
            return action, logprob, entropy, value

        dist, mean = self._continuous_dist(obs)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        logprob = dist.log_prob(pre_tanh).sum(-1)
        logprob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return action, logprob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logprob, entropy, value, approx_action
        """
        if self.is_discrete:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            logprob = dist.log_prob(action.long())
            entropy = dist.entropy()
            value = self.value(obs)
            return logprob, entropy, value, action

        dist, mean = self._continuous_dist(obs)
        clipped_action = action.clamp(-0.999999, 0.999999)
        pre_tanh = torch.atanh(clipped_action)
        logprob = dist.log_prob(pre_tanh).sum(-1)
        logprob -= torch.log(1 - clipped_action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return logprob, entropy, value, action


class PPOTrainer:
    def __init__(self, env, obs_dim: int, action_space, cfg: PPOConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model = ActorCritic(obs_dim, action_space, cfg.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)

        self.reset_env()

        self.global_step = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def reset_env(self):
        obs, info = self.env.reset()
        self.obs = np.asarray(obs, dtype=np.float32)

    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, np.ndarray]:
        obs_buf = []
        action_buf = []
        logprob_buf = []
        value_buf = []
        reward_buf = []
        done_buf = []
        entropy_buf = []

        episode_reward_sum = 0.0
        episode_len = 0

        for _ in range(self.cfg.rollout_steps):
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t, logprob_t, entropy_t, value_t = self.model.act(obs_t)

            if self.model.is_discrete:
                action = int(action_t.item())
            else:
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            obs_buf.append(self.obs.copy())
            action_buf.append(action_t.squeeze(0).cpu().numpy())
            logprob_buf.append(float(logprob_t.item()))
            value_buf.append(float(value_t.item()))
            reward_buf.append(float(reward))
            done_buf.append(done)
            entropy_buf.append(float(entropy_t.item()))

            episode_reward_sum += float(reward)
            episode_len += 1

            self.obs = np.asarray(next_obs, dtype=np.float32)
            self.global_step += 1

            if done:
                self.episode_returns.append(episode_reward_sum)
                self.episode_lengths.append(episode_len)
                episode_reward_sum = 0.0
                episode_len = 0
                next_obs, info = self.env.reset()
                self.obs = np.asarray(next_obs, dtype=np.float32)

        rewards = np.asarray(reward_buf, dtype=np.float32)
        transformed_rewards, reward_stats = transform_rewards(rewards, self.cfg.reward_transform)

        last_obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.model.value(last_obs_t).item())

        rollout = {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(action_buf),
            "logprobs": np.asarray(logprob_buf, dtype=np.float32),
            "values": np.asarray(value_buf, dtype=np.float32),
            "rewards_raw": rewards,
            "rewards": transformed_rewards,
            "dones": np.asarray(done_buf, dtype=np.float32),
            "entropies": np.asarray(entropy_buf, dtype=np.float32),
            "last_value": np.asarray([last_value], dtype=np.float32),
            **reward_stats,
        }
        return rollout

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def train_one_update(self) -> Dict[str, float]:
        rollout = self.collect_rollout()

        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        last_value = float(rollout["last_value"][0])
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # ===== VECTR Advantage Variance Scaling =====
        if self.cfg.use_vectr:
            target_std = self.cfg.vectr_target_std
            current_std = advantages.std() + 1e-8
            alpha = target_std / current_std
            advantages = alpha * advantages
        else:
            # Standard PPO advantage normalization
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
 

        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        if self.model.is_discrete:
            actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device)
        else:
            actions = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=self.device)

        old_logprobs = torch.as_tensor(rollout["logprobs"], dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        batch_size = obs.shape[0]
        inds = np.arange(batch_size)

        approx_kl_vals = []
        clipfrac_vals = []
        grad_norm_vals = []
        entropy_vals = []
        policy_loss_vals = []
        value_loss_vals = []

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(inds)

            for start in range(0, batch_size, self.cfg.minibatch_size):
                mb_inds = inds[start : start + self.cfg.minibatch_size]

                new_logprob, entropy, value, _ = self.model.evaluate_actions(obs[mb_inds], actions[mb_inds])

                logratio = new_logprob - old_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = advantages_t[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (returns_t[mb_inds] - value).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()

                approx_kl_vals.append(approx_kl)
                clipfrac_vals.append(clipfrac)
                grad_norm_vals.append(float(grad_norm))
                entropy_vals.append(float(entropy_loss.item()))
                policy_loss_vals.append(float(policy_loss.item()))
                value_loss_vals.append(float(value_loss.item()))

                if approx_kl > self.cfg.target_kl:
                    break

        metrics = {
            "raw_reward_mean": float(rollout["raw_reward_mean"]),
            "raw_reward_var": float(rollout["raw_reward_var"]),
            "transformed_reward_mean": float(rollout["transformed_reward_mean"]),
            "transformed_reward_var": float(rollout["transformed_reward_var"]),
            "advantage_mean": float(advantages.mean()),
            "advantage_var": float(advantages.var()),
            "approx_kl": float(np.mean(approx_kl_vals)) if approx_kl_vals else 0.0,
            "clipfrac": float(np.mean(clipfrac_vals)) if clipfrac_vals else 0.0,
            "grad_norm": float(np.mean(grad_norm_vals)) if grad_norm_vals else 0.0,
            "entropy": float(np.mean(entropy_vals)) if entropy_vals else 0.0,
            "policy_loss": float(np.mean(policy_loss_vals)) if policy_loss_vals else 0.0,
            "value_loss": float(np.mean(value_loss_vals)) if value_loss_vals else 0.0,
            "episode_return_mean": float(np.mean(self.episode_returns[-10:])) if self.episode_returns else 0.0,
            "episode_length_mean": float(np.mean(self.episode_lengths[-10:])) if self.episode_lengths else 0.0,
        }
        return metrics

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

