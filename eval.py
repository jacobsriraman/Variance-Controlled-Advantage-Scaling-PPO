from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics

from ppo_vectr import ActorCritic


def make_env(env_id: str):
    return RecordEpisodeStatistics(gym.make(env_id))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    env = make_env(args.env_id)
    obs, info = env.reset(seed=0)
    obs_dim = int(np.prod(env.observation_space.shape))
    model = ActorCritic(obs_dim, env.action_space).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            if model.is_discrete:
                logits = model.actor(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                mean = model.actor_mean(obs_t)
                action = torch.tanh(mean).squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
        returns.append(ep_ret)
        print(f"episode {ep}: return={ep_ret:.3f}")

    print(f"mean return: {np.mean(returns):.3f} +/- {np.std(returns):.3f}")
    env.close()


if __name__ == "__main__":
    main()

