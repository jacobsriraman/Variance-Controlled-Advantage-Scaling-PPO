from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import trange

from ppo_vectr import PPOConfig, PPOTrainer
from reward_transforms import RewardTransformConfig


def make_env(env_id: str, seed: int | None = None):
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-updates", type=int, default=1000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-vectr", action="store_true")
    p.add_argument("--vectr-target-std", type=float, default=1.0)


    p.add_argument(
        "--reward-transform",
        type=str,
        default="identity",
        choices=["identity", "scale", "zscore", "minmax", "tanh"],
    )
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--reward-target-std", type=float, default=1.0)
    p.add_argument("--reward-tanh-gain", type=float, default=1.0)

    p.add_argument("--outdir", type=str, default="runs/vectr")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.outdir) / args.env_id / f"{args.reward_transform}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_space = env.action_space

    reward_cfg = RewardTransformConfig(
        mode=args.reward_transform,
        scale=args.reward_scale,
        target_std=args.reward_target_std,
        tanh_gain=args.reward_tanh_gain,
    )

    cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        reward_transform=reward_cfg,
        device=args.device,
        use_vectr=args.use_vectr,
        vectr_target_std=args.vectr_target_std,
    )

    trainer = PPOTrainer(env, obs_dim, action_space, cfg)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    history = []
    for update in trange(args.total_updates, desc="updates"):
        metrics = trainer.train_one_update()
        metrics["update"] = update
        metrics["global_step"] = trainer.global_step
        history.append(metrics)

        if (update + 1) % 10 == 0:
            df = pd.DataFrame(history)
            df.to_csv(run_dir / "metrics.csv", index=False)
            trainer.save(str(run_dir / "policy.pt"))

    df = pd.DataFrame(history)
    df.to_csv(run_dir / "metrics.csv", index=False)
    trainer.save(str(run_dir / "policy.pt"))
    env.close()

    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()

