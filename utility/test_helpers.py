"""
Test all utility helpers against the live Terraria window.

Runs through:
  1. screenshot_terraria  — find window, capture screenshot
  2. read_stack_count     — template loading, digit recognition
  3. game_reader          — unified brightness + stack-count endpoint

Usage:
    python -m utility.test_helpers
"""

import os
import sys
import time

import cv2
import numpy as np

# Allow running both as `python -m utility.test_helpers` and directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utility.screenshot_terraria import (
    find_terraria_window,
    restore_window,
    get_client_region,
    capture_window,
)
from utility.read_stack_count import (
    load_templates,
    preprocess,
    match_digits,
    detections_to_number,
)
from utility.game_reader import GameReader, TEMPLATE_DIR

import mss


def _header(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def test_find_window():
    _header("screenshot_terraria — find_terraria_window")
    match = find_terraria_window()
    if not match:
        print("FAIL: Terraria window not found. Is the game running?")
        return None
    hwnd, title = match
    print(f"OK   hwnd={hwnd}  title={title!r}")
    return hwnd


def test_capture(hwnd):
    _header("screenshot_terraria — capture_window")
    restore_window(hwnd)

    region = get_client_region(hwnd)
    print(f"OK   client region: {region}")

    out = os.path.join(_HERE, "_test_screenshot.png")
    ok = capture_window(hwnd, out)
    if ok and os.path.exists(out):
        print(f"OK   saved test screenshot -> {out}")
    else:
        print("FAIL: capture_window returned False")
    return out


def test_load_templates():
    _header("read_stack_count — load_templates")
    templates = load_templates(TEMPLATE_DIR)
    if not templates:
        print(f"FAIL: No templates in {TEMPLATE_DIR}")
        return None
    print(f"OK   loaded digits: {sorted(templates.keys())}")
    return templates


def test_read_stack_count(image_path, templates):
    _header("read_stack_count — preprocess / match_digits / detections_to_number")
    img = cv2.imread(image_path)
    if img is None:
        print(f"FAIL: could not load {image_path}")
        return

    # Use the same crop region as game_reader
    from utility.game_reader import STACK_TOP, STACK_BOTTOM, STACK_LEFT, STACK_RIGHT

    crop = img[STACK_TOP:STACK_BOTTOM, STACK_LEFT:STACK_RIGHT]
    gray = preprocess(crop)
    dets = match_digits(gray, templates)
    count = detections_to_number(dets)
    print(f"OK   detections: {[(d[0], d[4]) for d in dets]}")
    print(f"OK   stack count = {count}")


def test_game_reader(hwnd):
    _header("game_reader — GameReader.read")
    reader = GameReader()

    with mss.mss() as sct:
        region = get_client_region(hwnd)
        frame = np.array(sct.grab(region))[:, :, :3]

    state = reader.read(frame)
    print(f"OK   stack_count = {state['stack_count']}")
    print(f"OK   brightness  = {state['brightness']:.1f}%")


def main():
    print("Utility helper tests")
    print("=" * 50)

    hwnd = test_find_window()
    if hwnd is None:
        sys.exit(1)

    screenshot_path = test_capture(hwnd)

    templates = test_load_templates()
    if templates is None:
        sys.exit(1)

    test_read_stack_count(screenshot_path, templates)
    test_game_reader(hwnd)

    # Clean up test screenshot
    if os.path.exists(screenshot_path):
        os.remove(screenshot_path)
        print(f"\nCleaned up {screenshot_path}")

    print("\n" + "=" * 50)
    print("  All tests passed")
    print("=" * 50)


if __name__ == "__main__":
    main()
