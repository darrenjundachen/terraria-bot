"""
Record human demos of Terraria gameplay.

Captures (screenshot, action) pairs at a fixed rate while the human plays.
Saves episodes as compressed .npz files for imitation_learning.py.

Actions recorded (8-dim float32):
    [0] move left  (A key)        0/1
    [1] move right (D key)        0/1
    [2] jump       (Space)        0/1
    [3] swing axe  (Left click)   0/1
    [4] mouse X    normalised     [0, 1]
    [5] mouse Y    normalised     [0, 1]
    [6] switch axe (2 key)        0/1
    [7] switch torch (5 key)      0/1

Usage:
    python record_demo.py                  # defaults: 10 FPS, demos/
    python record_demo.py --fps 15 --out my_demos

Controls:
    F9  — Start recording an episode
    F10 — Stop and save the current episode
    Ctrl+C — Quit
"""

import argparse
import ctypes
import ctypes.wintypes as w
import os
import sys
import time

import cv2
import mss
import numpy as np

# Reuse window helpers
from utility.screenshot_terraria import find_terraria_window, restore_window, get_client_region

# ── Win32 input polling ────────────────────────────────────────────────────
user32 = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState
GetCursorPos = user32.GetCursorPos

VK_A = 0x41
VK_D = 0x44
VK_SPACE = 0x20
VK_LBUTTON = 0x01
VK_2 = 0x32
VK_5 = 0x35
VK_F9 = 0x78
VK_F10 = 0x79

IMG_W, IMG_H = 160, 120


def key_held(vk):
    return bool(GetAsyncKeyState(vk) & 0x8000)


def key_edge(vk, state):
    """Return True on the rising edge (was up, now down)."""
    cur = key_held(vk)
    edge = cur and not state.get(vk, False)
    state[vk] = cur
    return edge


def get_mouse():
    pt = w.POINT()
    GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def capture_frame(sct, region):
    raw = np.array(sct.grab(region))[:, :, :3]  # BGRA → BGR
    return cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def sample_action(region):
    mx, my = get_mouse()
    mx_n = float(np.clip((mx - region["left"]) / max(region["width"], 1), 0, 1))
    my_n = float(np.clip((my - region["top"]) / max(region["height"], 1), 0, 1))
    return np.array(
        [
            float(key_held(VK_A)),
            float(key_held(VK_D)),
            float(key_held(VK_SPACE)),
            float(key_held(VK_LBUTTON)),
            mx_n,
            my_n,
            float(key_held(VK_2)),
            float(key_held(VK_5)),
        ],
        dtype=np.float32,
    )


def record_episode(sct, hwnd, fps):
    """Record frames + actions until F10 is pressed."""
    frames, actions = [], []
    ks = {}
    dt = 1.0 / fps

    print("  [RECORDING] Press F10 to stop.")

    while True:
        t0 = time.perf_counter()

        if key_edge(VK_F10, ks):
            break

        region = get_client_region(hwnd)
        if region["width"] <= 0 or region["height"] <= 0:
            time.sleep(dt)
            continue

        frames.append(capture_frame(sct, region))
        actions.append(sample_action(region))

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return np.array(frames, dtype=np.uint8), np.array(actions, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Record Terraria gameplay demos")
    ap.add_argument("--fps", type=int, default=10, help="Capture rate (default: 10)")
    ap.add_argument("--out", default="demos", help="Output folder (default: demos/)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    match = find_terraria_window()
    if not match:
        print("Error: Terraria window not found.")
        return
    hwnd, title = match
    print(f"Found: {title}")
    restore_window(hwnd)

    print(f"FPS: {args.fps}   Output: {args.out}/")
    print("F9 = start recording   F10 = stop + save   Ctrl+C = quit\n")

    ks = {}
    ep = len([f for f in os.listdir(args.out) if f.endswith(".npz")])

    with mss.mss() as sct:
        while True:
            if key_edge(VK_F9, ks):
                ep += 1
                print(f"Episode {ep}:")
                frames, actions = record_episode(sct, hwnd, args.fps)

                if len(frames) == 0:
                    print("  Empty episode, skipped.")
                    continue

                path = os.path.join(args.out, f"episode_{ep:04d}.npz")
                np.savez_compressed(path, frames=frames, actions=actions)
                n = len(frames)
                mb = frames.nbytes / 1e6
                print(f"  Saved {path} ({n} frames, {mb:.1f} MB uncompressed)")
                print("  F9 for next episode, Ctrl+C to quit.")

            time.sleep(0.05)


if __name__ == "__main__":
    main()
