"""
Crop a region from a screenshot.

Usage:
    python crop_region.py <screenshot.png> [output.png]
"""

import sys
import cv2

LEFT, TOP = 418, 43
RIGHT, BOTTOM = 454, 61

def main():
    if len(sys.argv) < 2:
        print("Usage: python crop_region.py <screenshot.png> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cropped.png"

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not load '{input_path}'")
        sys.exit(1)

    crop = img[TOP:BOTTOM, LEFT:RIGHT]
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path} ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    main()
