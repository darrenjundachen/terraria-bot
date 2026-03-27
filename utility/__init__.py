"""
utility — helpers for reading live Terraria game state.

Public API:
    GameReader          Unified brightness + stack-count reader
    find_terraria_window, restore_window, get_client_region   Window management
    load_templates, preprocess, match_digits, detections_to_number   Low-level OCR
"""

from .game_reader import GameReader
from .screenshot_terraria import find_terraria_window, restore_window, get_client_region
from .read_stack_count import load_templates, preprocess, match_digits, detections_to_number
