"""
Gymnasium environment wrapping live Terraria.

Observation: 160x120 RGB screenshot (uint8, HWC)
Action:      8-dim continuous [0,1] — [left, right, jump, swing, mouse_x, mouse_y, axe, torch]
Reward:      Change in wood stack count − time penalty

Used by imitation_learning.py and rl_finetune.py.
"""

import time

import cv2
import gymnasium
import gymnasium.spaces
import mss
import numpy as np

from utility.screenshot_terraria import find_terraria_window, restore_window, get_client_region
from utility.game_reader import GameReader

try:
    import pydirectinput

    pydirectinput.PAUSE = 0  # no built-in delay between calls
except ImportError:
    pydirectinput = None

IMG_W, IMG_H = 160, 120
ACTION_DIM = 8


class TerrariaEnv(gymnasium.Env):
    """Live Terraria as a Gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, fps=10, render_mode=None, send_actions=True, max_steps=2048):
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self.fps = fps
        self.render_mode = render_mode
        self.send_actions = send_actions and pydirectinput is not None
        self.max_steps = max_steps

        self.hwnd = None
        self.sct = None
        self.reader = None
        self.prev_wood = None
        self._step_count = 0
        self._prev_keys = {"a": False, "d": False, "space": False, "2": False, "5": False}
        self._prev_click = False

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._lazy_init()

        if self.send_actions:
            self._release_all()
        self._step_count = 0

        full = self._grab_full()
        obs = cv2.resize(full, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        state = self.reader.read(full)
        self.prev_wood = state["stack_count"]
        return obs, {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)

        if self.send_actions:
            self._execute_action(action)

        time.sleep(1.0 / self.fps)

        full = self._grab_full()
        obs = cv2.resize(full, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        state = self.reader.read(full)
        wood = state["stack_count"]

        # Reward: wood gained − small time penalty
        reward = 0.0
        if wood is not None and self.prev_wood is not None:
            delta = wood - self.prev_wood
            if 0 <= delta <= 50:  # ignore large jumps (likely misreads)
                reward = float(delta)

        # Time penalty to encourage efficiency
        reward -= 0.01

        # Penalty for darkness (to encourage using torches/bright areas)
        brightness = state["brightness"]
        if brightness is not None and brightness < 10:
            reward -= 1 * (1.0 - brightness / 10.0)

        if wood is not None:
            self.prev_wood = wood

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        terminated = False

        return obs, reward, terminated, truncated, {"wood_count": wood, "brightness": state["brightness"]}

    def render(self):
        if self.render_mode == "rgb_array":
            full = self._grab_full()
            return cv2.resize(full, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        return None

    def close(self):
        if self.send_actions:
            self._release_all()
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    # ── Internals ──────────────────────────────────────────────────────

    def _lazy_init(self):
        # Recreate mss each time — it uses thread-local storage and breaks
        # if called from a different thread than it was created in.
        if self.sct is not None:
            self.sct.close()
        self.sct = mss.mss()
        if self.reader is None:
            self.reader = GameReader()
        if self.hwnd is None:
            match = find_terraria_window()
            if not match:
                raise RuntimeError("Terraria window not found")
            self.hwnd, title = match
            restore_window(self.hwnd)

    def _grab_full(self):
        """Capture full-res game frame (BGR, uint8)."""
        region = get_client_region(self.hwnd)
        return np.array(self.sct.grab(region))[:, :, :3]

    def _execute_action(self, action):
        left, right, jump, swing, mx, my, axe, torch = action

        self._set_key("a", left > 0.5)
        self._set_key("d", right > 0.5)
        self._set_key("space", jump > 0.5)
        self._set_key("2", axe > 0.5)
        self._set_key("5", torch > 0.5)

        # Mouse position (absolute screen coords)
        region = get_client_region(self.hwnd)
        target_x = int(region["left"] + mx * region["width"])
        target_y = int(region["top"] + my * region["height"])
        pydirectinput.moveTo(target_x, target_y)

        # Mouse click
        clicking = swing > 0.5
        if clicking and not self._prev_click:
            pydirectinput.mouseDown()
        elif not clicking and self._prev_click:
            pydirectinput.mouseUp()
        self._prev_click = clicking

    def _set_key(self, key, pressed):
        if pressed and not self._prev_keys[key]:
            pydirectinput.keyDown(key)
        elif not pressed and self._prev_keys[key]:
            pydirectinput.keyUp(key)
        self._prev_keys[key] = pressed

    def _release_all(self):
        for key in list(self._prev_keys):
            if self._prev_keys[key]:
                pydirectinput.keyUp(key)
            self._prev_keys[key] = False
        if self._prev_click:
            pydirectinput.mouseUp()
            self._prev_click = False
