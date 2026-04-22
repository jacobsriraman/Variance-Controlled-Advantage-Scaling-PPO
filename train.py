from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordEpisodeStatistics
import sys
from tqdm import trange

from datetime import datetime

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


def build_method_name(args) -> str:
    """
    Build a method name that encodes every hyperparameter that distinguishes
    this run's reward configuration, so that no two differently-configured
    runs ever resolve to the same folder.
    """
    if args.use_vectr:
        return f"vectr_std{args.vectr_target_std}"

    transform = args.reward_transform

    if transform == "identity":
        return "identity"
    elif transform == "scale":
        return f"scale_{args.reward_scale}"
    elif transform == "zscore":
        # zscore has no extra hyperparameters currently, but include the
        # target_std in case it is ever wired up — keeps folders distinct.
        return f"zscore_std{args.reward_target_std}"
    elif transform == "minmax":
        return "minmax"
    elif transform == "tanh":
        return f"tanh_gain{args.reward_tanh_gain}"
    else:
        # Fallback: should never be reached given argparse choices, but
        # if a new transform is added later this still produces a unique name.
        return transform

def evaluate_policy(model, env_id, device, n_episodes=5, seed=0):
    env = gym.make(env_id)
    returns = []

    model.eval()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0

        while not done:
            action = model.act(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)

    env.close()
    return np.mean(returns), np.std(returns)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = build_method_name(args)

    job_id = os.environ.get("SLURM_JOB_ID", "nojob")

    run_dir = (
        Path(args.outdir)
        / args.env_id
        / method_name
        / f"seed{args.seed}_job{job_id}_{timestamp}"
    )

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
    #print([attr for attr in dir(trainer.model) if not attr.startswith("_")])

    config_dict = vars(args)
    config_dict["method_name"] = method_name
    config_dict["timestamp"] = timestamp

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # trange = tqdm progress bar ----- that's what we see in .out file:
    history = []
    for update in trange(args.total_updates, desc="updates", file=sys.stdout):
        # This does one full PPO iteration: collect trajectories, compute advantages (GAE), apply scaling (VERY important for your project), update policy + value network
        metrics = trainer.train_one_update()
        # Storing metrics
        metrics["update"] = update
        metrics["global_step"] = trainer.global_step
        history.append(metrics)

        if (update + 1) % 10 == 0:
            
            eval_mean, eval_std = evaluate_policy(
            trainer.model,        
            args.env_id,
            trainer.device,
            n_episodes=5,
            seed=args.seed + 1000
            )
            
            metrics["eval_return_mean"] = eval_mean
            metrics["eval_return_std"] = eval_std
            
            print(
                f"[u {metrics['update'] + 1}] "
                f"ret={metrics['episode_return_mean']:.2f} "
                f"adv_var={metrics['advantage_var']:.3f} "
                f"kl={metrics['approx_kl']:.4f} "
                f"gn={metrics['grad_norm']:.2f}",
                f"gn={metrics['eval_return_mean']:.2f}",
                f"gn={metrics['eval_return_std']:.2f}",
                flush=True
                )

            
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
