"""
Imitation learning (Behavioral Cloning + DAgger) for Terraria.

Uses the `imitation` library (HumanCompatibleAI) with Stable-Baselines3.
Trains a CnnPolicy from demos recorded by record_demo.py.

Usage:
    python imitation_learning.py --train                         # BC from demos/
    python imitation_learning.py --train --demos demos --epochs 50
    python imitation_learning.py --dagger --model policy_bc.zip  # DAgger refinement
    python imitation_learning.py --eval --model policy_bc.zip    # live inference
"""

import argparse
import glob
import os
import sys

import gymnasium
import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from terraria_env import TerrariaEnv, IMG_W, IMG_H, ACTION_DIM


# ── Load demos → imitation Transitions ─────────────────────────────────────


def load_demos(demo_dir):
    """Load .npz demo files into an imitation-compatible Transitions object."""
    files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz demo files in '{demo_dir}/'")

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []

    for f in files:
        data = np.load(f)
        frames = data["frames"]  # (N, H, W, 3) uint8
        actions = data["actions"]  # (N, 6) float32

        for i in range(len(frames) - 1):
            all_obs.append(frames[i])
            all_acts.append(actions[i])
            all_next_obs.append(frames[i + 1])
            all_dones.append(False)

        # Terminal transition for last frame
        if len(frames) > 1:
            all_obs.append(frames[-1])
            all_acts.append(actions[-1])
            all_next_obs.append(frames[-1])
            all_dones.append(True)

    # Transpose HWC → CHW to match CnnPolicy's observation space
    obs_arr = np.array(all_obs).transpose(0, 3, 1, 2)       # (N, 3, H, W)
    next_obs_arr = np.array(all_next_obs).transpose(0, 3, 1, 2)

    transitions = Transitions(
        obs=obs_arr,
        acts=np.array(all_acts),
        next_obs=next_obs_arr,
        dones=np.array(all_dones),
        infos=np.array([{}] * len(all_obs)),
    )
    print(f"Loaded {len(transitions)} transitions from {len(files)} demo(s)")
    return transitions


# ── Environment factory ────────────────────────────────────────────────────


def make_env(send_actions=False):
    def _init():
        return TerrariaEnv(fps=10, send_actions=send_actions)

    return _init


# ── Behavioral Cloning ─────────────────────────────────────────────────────


def train_bc(args):
    transitions = load_demos(args.demos)

    # Create a vec env so SB3 can infer observation/action spaces
    venv = DummyVecEnv([make_env(send_actions=False)])

    # Build a CnnPolicy via a temporary PPO (cleanest way to get a properly
    # configured SB3 policy with NatureCNN for image observations)
    tmp_model = PPO("CnnPolicy", venv, device="auto", verbose=0)
    policy = tmp_model.policy

    rng = np.random.default_rng(42)

    bc_trainer = BC(
        observation_space=policy.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=rng,
        batch_size=args.batch_size,
        device="auto",
    )

    print(f"\nTraining BC for {args.epochs} epochs  (batch={args.batch_size})")
    print(f"{'-' * 50}")
    bc_trainer.train(n_epochs=args.epochs, log_interval=5)

    # Save as a full PPO model so rl_finetune.py can load it directly
    tmp_model.policy = bc_trainer.policy
    tmp_model.save(args.model)
    print(f"\nSaved policy -> {args.model}")

    venv.close()


# ── DAgger ─────────────────────────────────────────────────────────────────


def train_dagger(args):
    """Interactive DAgger with F9 human override (Option C).

    Agent plays by default. Hold F9 to take over control and provide
    corrections.  When F9 is released the agent resumes.

    Each round:
      1. Agent plays using current policy (actions sent to game).
      2. When the agent does something wrong, hold F9 — the agent's inputs
         are suppressed and YOUR physical keys/mouse control the game.
      3. Expert label = your inputs while F9 is held, agent's action otherwise.
      4. New transitions are aggregated with existing demos.
      5. Policy is retrained via BC on the combined dataset.

    Controls:
      Ctrl — hold to take over (suppress agent, record your inputs)
      F10  — end the current round early
    """
    import ctypes
    import ctypes.wintypes as w
    import time

    from utility.screenshot_terraria import get_client_region

    VK_LCONTROL = 0xA2
    VK_F10 = 0x79
    VK_A = 0x41
    VK_D = 0x44
    VK_SPACE = 0x20
    VK_LBUTTON = 0x01
    VK_2 = 0x32
    VK_5 = 0x35
    GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
    GetCursorPos = ctypes.windll.user32.GetCursorPos

    def key_held(vk):
        return bool(GetAsyncKeyState(vk) & 0x8000)

    def get_mouse():
        pt = w.POINT()
        GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def sample_human_action(region):
        mx, my = get_mouse()
        mx_n = float(np.clip((mx - region["left"]) / max(region["width"], 1), 0, 1))
        my_n = float(np.clip((my - region["top"]) / max(region["height"], 1), 0, 1))
        return np.array([
            float(key_held(VK_A)),
            float(key_held(VK_D)),
            float(key_held(VK_SPACE)),
            float(key_held(VK_LBUTTON)),
            mx_n, my_n,
            float(key_held(VK_2)),
            float(key_held(VK_5)),
        ], dtype=np.float32)

    # Load pretrained model
    env = TerrariaEnv(fps=10, send_actions=True)
    venv = DummyVecEnv([make_env(send_actions=True)])
    model = PPO.load(args.model, env=venv, device="auto")

    # Load existing demos
    demo_dir = args.demos
    existing_transitions = load_demos(demo_dir)
    all_obs = list(existing_transitions.obs)
    all_acts = list(existing_transitions.acts)
    all_next_obs = list(existing_transitions.next_obs)
    all_dones = list(existing_transitions.dones)

    print(f"DAgger: {args.dagger_rounds} rounds, {args.dagger_steps} steps each")
    print(f"Starting with {len(all_obs)} existing transitions")
    print("Agent plays by default. Hold Ctrl to take over and correct.")
    print("F10 ends a round early.\n")

    for round_num in range(1, args.dagger_rounds + 1):
        print(f"-- Round {round_num}/{args.dagger_rounds} --")
        print("  Agent playing - hold Ctrl to override...")

        obs, _ = env.reset()
        round_obs, round_expert_acts, round_next_obs, round_dones = [], [], [], []
        human_frames = 0

        for step in range(args.dagger_steps):
            if key_held(VK_F10):
                print(f"  F10 pressed, ending round at step {step}")
                break

            human_override = key_held(VK_LCONTROL)

            # Agent predicts action
            obs_chw = obs.transpose(2, 0, 1)  # HWC -> CHW for model
            obs_t = np.expand_dims(obs_chw, 0)
            action, _ = model.predict(obs_t, deterministic=False)
            action = action.squeeze()  # (1,6) -> (6,)

            if human_override:
                # Human controls the game — suppress agent inputs
                env.send_actions = False
                env._release_all()  # release any keys the agent was holding
                next_obs, _, terminated, truncated, _ = env.step(action)

                # Record human's physical inputs as expert label
                region = get_client_region(env.hwnd)
                expert_action = sample_human_action(region)
                human_frames += 1
            else:
                # Agent controls the game — agent's action is the label
                env.send_actions = True
                next_obs, _, terminated, truncated, _ = env.step(action)
                expert_action = action.copy()

            # Store transition (CHW obs, expert action)
            round_obs.append(obs_chw)
            next_obs_chw = next_obs.transpose(2, 0, 1)
            round_expert_acts.append(expert_action)
            round_next_obs.append(next_obs_chw)
            round_dones.append(terminated or truncated)

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        # Ensure agent has control back and keys released
        env.send_actions = True
        env._release_all()

        n_new = len(round_obs)
        print(f"  Collected {n_new} transitions ({human_frames} human-corrected)")

        if n_new == 0:
            continue

        # Aggregate
        all_obs.extend(round_obs)
        all_acts.extend(round_expert_acts)
        all_next_obs.extend(round_next_obs)
        all_dones.extend(round_dones)

        # Retrain BC on combined dataset
        transitions = Transitions(
            obs=np.array(all_obs),
            acts=np.array(all_acts),
            next_obs=np.array(all_next_obs),
            dones=np.array(all_dones),
            infos=np.array([{}] * len(all_obs)),
        )

        rng = np.random.default_rng(42)
        bc_trainer = BC(
            observation_space=model.policy.observation_space,
            action_space=venv.action_space,
            demonstrations=transitions,
            policy=model.policy,
            rng=rng,
            batch_size=args.batch_size,
            device="auto",
        )

        print(f"  Retraining BC on {len(transitions)} transitions ({args.epochs} epochs)...")
        bc_trainer.train(n_epochs=args.epochs, log_interval=10)
        model.policy = bc_trainer.policy
        print(f"  Round {round_num} done.\n")

    env.close()
    venv.close()

    model.save(args.model)
    print(f"Saved DAgger-refined policy -> {args.model}")


# ── Live Evaluation ────────────────────────────────────────────────────────


def evaluate(args):
    import time

    env = TerrariaEnv(fps=10, send_actions=True)
    model = PPO.load(args.model, device="auto")

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    print("Running policy live (Ctrl+C to stop)...\n")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

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
            print(
                f"  step={step:4d}  wood={wood}  R={reward:+.2f}  "
                f"total={total_reward:.1f}  keys=[{keys:6s}]  "
                f"mouse=({action[4]:.2f},{action[5]:.2f})"
            )

            if terminated or truncated:
                print(f"  -- Episode done (total reward: {total_reward:.1f}) --")
                obs, _ = env.reset()
                total_reward = 0.0
                step = 0
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Terraria Behavioral Cloning / DAgger")
    ap.add_argument("--train", action="store_true", help="Train BC from demos")
    ap.add_argument("--dagger", action="store_true", help="DAgger refinement (interactive)")
    ap.add_argument("--eval", action="store_true", help="Run policy live in Terraria")
    ap.add_argument("--demos", default="demos", help="Demo folder (default: demos/)")
    ap.add_argument("--model", default="policy_bc.zip", help="Model path")
    ap.add_argument("--epochs", type=int, default=30, help="BC training epochs")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--dagger_rounds", type=int, default=5, help="DAgger rounds")
    ap.add_argument("--dagger_steps", type=int, default=500, help="Steps per DAgger round")
    args = ap.parse_args()

    if args.train:
        train_bc(args)
    elif args.dagger:
        train_dagger(args)
    elif args.eval:
        evaluate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()


# python imitation_learning.py --train --demos demos --epochs 50
# python imitation_learning.py --eval --model policy_bc.zip
# python imitation_learning.py --dagger --model policy_bc.zip --demos demos