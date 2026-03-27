"""
RL fine-tuning with PPO (Proximal Policy Optimization) for Terraria.

Loads a BC-pretrained policy from imitation_learning.py and fine-tunes it
with PPO using live gameplay. Reward = wood gained − time penalty.

Uses Stable-Baselines3's PPO with NatureCNN, GAE, and clipped surrogate
objective — state-of-the-art on-policy RL for visual control.

Usage:
    python rl_finetune.py --pretrained policy_bc.zip                 # fine-tune
    python rl_finetune.py --pretrained policy_bc.zip --steps 200000  # longer run
    python rl_finetune.py --eval --model policy_rl.zip               # evaluate
    python rl_finetune.py --scratch --steps 500000                   # train from scratch

Tensorboard:
    tensorboard --logdir tb_logs/
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from terraria_env import TerrariaEnv


# ── PPO hyperparameters (tuned for visual control) ─────────────────────────

PPO_DEFAULTS = dict(
    learning_rate=3e-4,
    n_steps=256,          # rollout length before each update
    batch_size=64,        # mini-batch size for PPO epochs
    n_epochs=4,           # reuse each rollout 4 times
    gamma=0.99,           # discount factor
    gae_lambda=0.95,      # GAE λ for advantage estimation
    clip_range=0.2,       # PPO clipping ε
    ent_coef=0.5,        # initial entropy bonus (annealed by callback)
    vf_coef=0.5,          # value function loss weight
    max_grad_norm=0.5,    # gradient clipping
    verbose=1,
    tensorboard_log="tb_logs",
)


# ── Custom callback: log wood count ────────────────────────────────────────


class WoodCountCallback(BaseCallback):
    """Log the wood count from info dict to tensorboard."""

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            wood = info.get("wood_count")
            if wood is not None:
                self.logger.record("env/wood_count", wood)
        return True


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal ent_coef from start_val to end_val over training."""

    def __init__(self, start_val=0.5, end_val=0.001):
        super().__init__()
        self.start_val = start_val
        self.end_val = end_val

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = self.start_val + (self.end_val - self.start_val) * progress
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True


# ── Environment factory ────────────────────────────────────────────────────


def make_env(fps=10, send_actions=True, max_steps=2048):
    """Create a vectorized, monitored Terraria environment."""

    def _init():
        env = TerrariaEnv(fps=fps, send_actions=send_actions, max_steps=max_steps)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_init])
    venv = VecTransposeImage(venv)  # HWC → CHW for CnnPolicy
    return venv


# ── Training ───────────────────────────────────────────────────────────────


def train(args):
    venv = make_env(fps=10, send_actions=True, max_steps=args.max_episode_steps)

    if args.pretrained:
        print(f"Loading pretrained policy: {args.pretrained}")
        model = PPO.load(args.pretrained, env=venv, device="auto", **PPO_DEFAULTS)
        # Reset learning rate schedule (BC training may have used a different one)
        model.learning_rate = PPO_DEFAULTS["learning_rate"]
    else:
        print("Training from scratch (no pretrained weights)")
        model = PPO("CnnPolicy", venv, device="auto", **PPO_DEFAULTS)

    os.makedirs("checkpoints", exist_ok=True)

    callbacks = [
        WoodCountCallback(),
        EntropyAnnealCallback(start_val=0.05, end_val=0.001),
        CheckpointCallback(
            save_freq=5000,
            save_path="checkpoints/",
            name_prefix="terraria_ppo",
        ),
    ]

    print(f"\nPPO fine-tuning for {args.steps:,} timesteps")
    print(f"  γ={PPO_DEFAULTS['gamma']}  λ={PPO_DEFAULTS['gae_lambda']}  "
          f"ε={PPO_DEFAULTS['clip_range']}  lr={PPO_DEFAULTS['learning_rate']}")
    print(f"  n_steps={PPO_DEFAULTS['n_steps']}  batch={PPO_DEFAULTS['batch_size']}  "
          f"epochs={PPO_DEFAULTS['n_epochs']}")
    print(f"  Tensorboard: tensorboard --logdir tb_logs/")
    print(f"{'─' * 60}\n")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(args.model)
    print(f"\nSaved final model → {args.model}")

    venv.close()


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate(args):
    import time

    env = TerrariaEnv(fps=10, send_actions=True, max_steps=args.max_episode_steps)
    model = PPO.load(args.model, device="auto")

    n_episodes = args.eval_episodes
    episode_rewards = []

    print(f"Evaluating for {n_episodes} episode(s) (Ctrl+C to stop early)...\n")

    try:
        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated

                wood = info.get("wood_count", "?")
                acts = action[:4]
                keys = ""
                if acts[0] > 0.5:
                    keys += "A"
                if acts[1] > 0.5:
                    keys += "D"
                if acts[2] > 0.5:
                    keys += "J"
                if acts[3] > 0.5:
                    keys += "S"
                if action[6] > 0.5:
                    keys += "2"
                if action[7] > 0.5:
                    keys += "5"

                if step % 10 == 0:
                    print(
                        f"  ep={ep} step={step:4d}  wood={wood}  "
                        f"R={total_reward:.1f}  [{keys:6s}]  "
                        f"mouse=({action[4]:.2f},{action[5]:.2f})"
                    )

            episode_rewards.append(total_reward)
            print(f"  ── Episode {ep} done: {step} steps, reward={total_reward:.1f} ──\n")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    if episode_rewards:
        print(f"\nResults over {len(episode_rewards)} episode(s):")
        print(f"  Mean reward: {np.mean(episode_rewards):.1f}")
        print(f"  Std reward:  {np.std(episode_rewards):.1f}")
        print(f"  Min/Max:     {np.min(episode_rewards):.1f} / {np.max(episode_rewards):.1f}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Terraria PPO fine-tuning")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pretrained", type=str, help="Path to BC-pretrained model (.zip)")
    mode.add_argument("--scratch", action="store_true", help="Train from scratch (no pretraining)")
    mode.add_argument("--eval", action="store_true", help="Evaluate a trained model")

    ap.add_argument("--model", default="policy_rl.zip", help="Output model path (default: policy_rl.zip)")
    ap.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    ap.add_argument("--max_episode_steps", type=int, default=2048, help="Max steps per episode")
    ap.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes to evaluate")
    args = ap.parse_args()

    if args.eval:
        evaluate(args)
    elif args.pretrained or args.scratch:
        train(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

