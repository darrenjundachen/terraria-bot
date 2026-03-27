# Terraria AI Agent

A machine learning pipeline for teaching an AI agent to play Terraria using **imitation learning** (behavioral cloning + DAgger) and **reinforcement learning** (PPO fine-tuning). The agent learns to collect wood, manage tools, and navigate the game world from human demonstrations.

## Pipeline Overview

```
Human Gameplay ──► record_demo.py ──► demos/*.npz
                                          │
                                          ▼
                          imitation_learning.py --train
                          (Behavioral Cloning)
                                          │
                                          ▼
                                    policy_bc.zip
                                     │        │
                                     ▼        ▼
                    rl_finetune.py        imitation_learning.py --dagger
                    (PPO Fine-Tuning)     (Interactive Refinement)
                            │
                            ▼
                      policy_rl.zip ──► Evaluation
```

## Project Structure

| File | Description |
|------|-------------|
| `terraria_env.py` | Gymnasium environment wrapping live Terraria gameplay |
| `record_demo.py` | Record human gameplay demonstrations |
| `imitation_learning.py` | Behavioral Cloning and DAgger training |
| `rl_finetune.py` | PPO reinforcement learning fine-tuning |
| `demos/` | Recorded gameplay episodes (`.npz` files) |
| `utility/` | Helper modules for screen capture, game state reading, and OCR |

## Requirements

- **OS:** Windows (uses Win32 API for window management and input)
- **Game:** Terraria (must be running)
- **Python packages:** `gymnasium`, `stable-baselines3`, `imitation`, `opencv-python`, `mss`, `numpy`, `torch`, `pydirectinput`

## Usage

### 1. Record Demonstrations

```bash
python record_demo.py
```

- **F9** — Start recording an episode
- **F10** — Stop and save the current episode
- **Ctrl+C** — Quit

Saves episodes to `demos/` as `.npz` files containing frame/action pairs at 10 FPS.

### 2. Train with Behavioral Cloning

```bash
python imitation_learning.py --train --demos demos --epochs 50
```

Trains a CNN policy on the recorded demonstrations using behavioral cloning. Outputs `policy_bc.zip`.

### 3. Refine with DAgger (Optional)

```bash
python imitation_learning.py --dagger --model policy_bc.zip
```

Interactive training loop where the agent plays and you can hold **Ctrl** to override with human input. Corrections are incorporated into the training dataset.

### 4. Fine-Tune with PPO

```bash
python rl_finetune.py --pretrained policy_bc.zip --steps 200000
```

Fine-tunes the BC policy with PPO on live gameplay. Logs metrics to TensorBoard (`tb_logs/`).

To train from scratch without a pretrained model:

```bash
python rl_finetune.py --scratch --steps 500000
```

### 5. Evaluate

```bash
python imitation_learning.py --eval --model policy_bc.zip
python rl_finetune.py --eval --model policy_rl.zip --eval_episodes 5
```

## Environment Details

### Observation Space

160×120 RGB images captured from the Terraria game window.

### Action Space

8-dimensional continuous vector in [0, 1]:

| Index | Action |
|-------|--------|
| 0 | Move left (A) |
| 1 | Move right (D) |
| 2 | Jump (Space) |
| 3 | Swing/attack (Left click) |
| 4–5 | Mouse X, Y (normalized) |
| 6 | Switch to axe (2) |
| 7 | Switch to torch (5) |

### Reward Function

- **+wood_gained** — wood collected each step (0–50)
- **-0.01** — time penalty per step
- **-1.0 × (1 - brightness/10)** — darkness penalty (encourages lighting)

## Utility Modules

| Module | Description |
|--------|-------------|
| `utility/screenshot_terraria.py` | Find Terraria window, capture screenshots via Win32 API |
| `utility/game_reader.py` | Read game state (wood stack count, brightness) from screenshots |
| `utility/read_stack_count.py` | OCR pipeline using template matching to read inventory numbers |
| `utility/extract_templates.py` | Interactive tool to create digit templates from a screenshot |
| `utility/crop_region.py` | Crop a region from a screenshot |
| `utility/test_helpers.py` | Integration tests for all utility modules |
| `utility/templates/` | Digit template images (0–9) for OCR |

### Running Utility Tests

```bash
python -m utility.test_helpers
```

Requires Terraria to be running. Verifies window detection, screenshot capture, template loading, stack count reading, and game state extraction.

## Demo Data Format

Each `.npz` file in `demos/` contains:

```python
{
    "frames": np.array(shape=(N, 160, 120, 3), dtype=uint8),   # RGB screenshots
    "actions": np.array(shape=(N, 8), dtype=float32)            # action vectors
}
```

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Rollout steps | 256 |
| Batch size | 64 |
| PPO epochs | 4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.5 → 0.001 (annealed) |
| Value function coef | 0.5 |
| Max grad norm | 0.5 |
