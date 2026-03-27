"""
Step 1: Extract digit templates from a Terraria screenshot.

Usage:
  1. Take a Terraria screenshot that has visible stack counts (ideally slots
     showing numbers 0-9, or a single slot with "1234567890").
  2. Manually crop each digit from the screenshot and save as:
       templates/0.png, templates/1.png, ... templates/9.png

  OR use this helper which lets you click-crop digits interactively.

Run:
  python extract_templates.py <screenshot.png>
"""

import cv2
import numpy as np
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_templates.py <screenshot.png>")
        print()
        print("This opens the image and lets you draw a rectangle around each")
        print("digit (0-9). Press the digit key to label it, then repeat.")
        print("Press 'q' to quit and save all collected templates.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load '{sys.argv[1]}'")
        sys.exit(1)

    os.makedirs("templates", exist_ok=True)

    clone = img.copy()
    roi_start = None
    roi_end = None
    drawing = False

    def mouse_cb(event, x, y, flags, param):
        nonlocal roi_start, roi_end, drawing, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_start = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            clone = img.copy()
            cv2.rectangle(clone, roi_start, (x, y), (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_end = (x, y)
            drawing = False
            cv2.rectangle(clone, roi_start, roi_end, (0, 255, 0), 2)

    cv2.namedWindow("Select digits", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select digits", mouse_cb)

    print("Draw a rectangle around a digit, then press the digit key (0-9).")
    print("Press 'q' when done.")

    saved = set()
    while True:
        cv2.imshow("Select digits", clone)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        if ord('0') <= key <= ord('9') and roi_start and roi_end:
            digit = chr(key)
            x1 = min(roi_start[0], roi_end[0])
            y1 = min(roi_start[1], roi_end[1])
            x2 = max(roi_start[0], roi_end[0])
            y2 = max(roi_start[1], roi_end[1])

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print("Empty selection, try again.")
                continue

            # Save as raw grayscale (no thresholding — reader handles that)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            path = f"templates/{digit}.png"
            cv2.imwrite(path, gray)
            saved.add(digit)
            print(f"Saved template for '{digit}' -> {path}  ({gray.shape[1]}x{gray.shape[0]})")

            roi_start = roi_end = None

    cv2.destroyAllWindows()
    print(f"\nDone. Saved templates for digits: {sorted(saved)}")
    missing = set("0123456789") - saved
    if missing:
        print(f"Missing digits: {sorted(missing)}")
        print("You can re-run the script later to add them.")


if __name__ == "__main__":
    main()
