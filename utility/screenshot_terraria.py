"""
Find the Terraria window and capture a screenshot using mss.

Usage:
    python screenshot_terraria.py [output_path]

Defaults to saving as 'terraria_screenshot.png' in the current directory.
"""

import ctypes
import ctypes.wintypes as w
import sys
import time

import mss
import mss.tools


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("showCmd", ctypes.c_uint),
        ("ptMinPosition", w.POINT),
        ("ptMaxPosition", w.POINT),
        ("rcNormalPosition", w.RECT),
    ]


WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, w.HWND, w.LPARAM)

user32 = ctypes.windll.user32
EnumWindows = user32.EnumWindows
GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
GetWindowRect = user32.GetWindowRect
GetClientRect = user32.GetClientRect
ShowWindow = user32.ShowWindow
SetForegroundWindow = user32.SetForegroundWindow
IsIconic = user32.IsIconic
GetWindowPlacement = user32.GetWindowPlacement
ClientToScreen = user32.ClientToScreen

SW_RESTORE = 9


def find_terraria_window():
    """Find the Terraria game window by title. Returns (hwnd, title) or None."""
    result = []

    exclude = ("visual studio", "file explorer", "explorer")

    def callback(hwnd, _lparam):
        length = GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            lower = title.lower()
            if "terraria" in lower and not any(ex in lower for ex in exclude):
                result.append((hwnd, title))
        return True

    EnumWindows(WNDENUMPROC(callback), 0)
    return result[0] if result else None


def restore_window(hwnd):
    """Restore the window if minimized and bring it to the foreground."""
    if IsIconic(hwnd):
        ShowWindow(hwnd, SW_RESTORE)
        time.sleep(1)
    SetForegroundWindow(hwnd)
    time.sleep(0.5)


def get_client_region(hwnd):
    """Get the client area screen coordinates (excludes title bar and borders)."""
    client_rect = w.RECT()
    GetClientRect(hwnd, ctypes.byref(client_rect))

    top_left = w.POINT(0, 0)
    ClientToScreen(hwnd, ctypes.byref(top_left))

    return {
        "left": top_left.x,
        "top": top_left.y,
        "width": client_rect.right,
        "height": client_rect.bottom,
    }


def capture_window(hwnd, output_path):
    """Capture a screenshot of the window's client area."""
    region = get_client_region(hwnd)

    if region["width"] <= 0 or region["height"] <= 0:
        print(f"Error: invalid client area size ({region['width']}x{region['height']})")
        return False

    with mss.mss() as sct:
        img = sct.grab(region)
        mss.tools.to_png(img.rgb, img.size, output=output_path)

    print(f"Saved {output_path} ({region['width']}x{region['height']})")
    return True


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "terraria_screenshot.png"

    match = find_terraria_window()
    if not match:
        print("Error: Terraria window not found. Is the game running?")
        sys.exit(1)

    hwnd, title = match
    print(f"Found: {title}")

    restore_window(hwnd)

    if not capture_window(hwnd, output_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
