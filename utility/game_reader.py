"""
Unified game state reader — reads stack count and center brightness
from a Terraria frame in a single call.

Usage from terraria_env or other code:
    from utility.game_reader import GameReader

    reader = GameReader()
    state = reader.read(full_frame)
    # state == {"stack_count": 42, "brightness": 73.5}
"""

import os

import cv2
import numpy as np

from .read_stack_count import load_templates, preprocess, match_digits, detections_to_number

# Stack-count crop coordinates (relative to client area)
STACK_LEFT, STACK_TOP = 420, 43
STACK_RIGHT, STACK_BOTTOM = 456, 61

# Center brightness sample (pixels)
CENTER_SIZE = 64

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


class GameReader:
    """Reads stack count and brightness from a full-resolution game frame."""

    def __init__(self, template_dir=TEMPLATE_DIR):
        self.templates = load_templates(template_dir)
        if not self.templates:
            raise FileNotFoundError(
                f"No digit templates in '{template_dir}/'. Run extract_templates.py first."
            )

    def read(self, full_frame):
        """
        Return ``{"stack_count": int|None, "brightness": float}`` from a
        BGR full-resolution game frame.

        *brightness* is 0–100 (percentage of white at the frame centre).
        """
        return {
            "stack_count": self._read_stack_count(full_frame),
            "brightness": self._read_brightness(full_frame),
        }

    # ------------------------------------------------------------------

    def _read_stack_count(self, frame):
        crop = frame[STACK_TOP:STACK_BOTTOM, STACK_LEFT:STACK_RIGHT]
        if crop.size == 0:
            return None
        gray = preprocess(crop)
        dets = match_digits(gray, self.templates)
        return detections_to_number(dets)

    @staticmethod
    def _read_brightness(frame):
        h, w = frame.shape[:2]
        half = CENTER_SIZE // 2
        cy, cx = h // 2, w // 2
        center = frame[max(cy - half, 0):min(cy + half, h),
                       max(cx - half, 0):min(cx + half, w)]
        return float(np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY))) / 255.0 * 100.0
