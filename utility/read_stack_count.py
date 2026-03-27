"""
Step 2: Read stack counts from Terraria inventory slots using template matching.

Usage:
  python read_stack_count.py <slot_image.png>

Requires digit templates in the templates/ folder (see extract_templates.py).
"""

import cv2
import numpy as np
import os
import sys
import glob


def load_templates(template_dir="templates"):
    """Load digit templates 0-9 from disk."""
    templates = {}
    for digit in range(10):
        path = os.path.join(template_dir, f"{digit}.png")
        if os.path.exists(path):
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates[digit] = tmpl
    return templates


def preprocess(region):
    """Convert to grayscale."""
    if len(region.shape) == 3:
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return region


def match_digits(binary_region, templates, threshold=0.6):
    """
    Run cv2.matchTemplate for each digit template.
    Returns a list of (x_position, digit) sorted left-to-right.
    """
    detections = []

    for digit, tmpl in templates.items():
        th, tw = tmpl.shape[:2]

        # Skip if template is larger than the region
        if th > binary_region.shape[0] or tw > binary_region.shape[1]:
            continue

        result = cv2.matchTemplate(binary_region, tmpl, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for y, x in zip(*locations):
            detections.append((x, y, tw, th, digit, result[y, x]))

    # Non-maximum suppression: group overlapping detections
    detections = _nms(detections)

    # Sort left-to-right by x position
    detections.sort(key=lambda d: d[0])
    return detections


def _nms(detections, overlap_thresh=0.5):
    """Simple non-maximum suppression to remove duplicate detections."""
    if not detections:
        return []

    # Sort by confidence descending — the best match wins in overlap regions
    detections.sort(key=lambda d: d[5], reverse=True)
    keep = []

    for det in detections:
        x, y, w, h, digit, conf = det
        suppressed = False
        for kept in keep:
            kx, ky, kw, kh, _, _ = kept
            # Check overlap
            ix1 = max(x, kx)
            iy1 = max(y, ky)
            ix2 = min(x + w, kx + kw)
            iy2 = min(y + h, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = w * h
            if area > 0 and inter / area > overlap_thresh:
                suppressed = True
                break
            # Suppress if x-centers are too close (within half the wider template)
            cx = x + w / 2
            kcx = kx + kw / 2
            min_dist = max(w, kw) / 2
            if abs(cx - kcx) < min_dist:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep


def detections_to_number(detections):
    """Convert sorted detections into an integer."""
    if not detections:
        return None
    digits_str = "".join(str(d[4]) for d in detections)
    return int(digits_str)


def read_stack_count(slot_image_path, template_dir="templates", threshold=0.6):
    """Full pipeline: load image -> crop -> preprocess -> match -> return number."""
    templates = load_templates(template_dir)
    if not templates:
        raise FileNotFoundError(
            f"No digit templates found in '{template_dir}/'. "
            "Run extract_templates.py first."
        )

    img = cv2.imread(slot_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {slot_image_path}")

    binary = preprocess(img)
    detections = match_digits(binary, templates, threshold)
    return detections_to_number(detections)


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_stack_count.py <slot_image.png> [threshold]")
        print("       python read_stack_count.py slots/*.png")
        sys.exit(1)

    threshold = 0.6
    paths = []
    for arg in sys.argv[1:]:
        try:
            threshold = float(arg)
        except ValueError:
            paths.extend(glob.glob(arg))

    if not paths:
        print("No image files found.")
        sys.exit(1)

    templates = load_templates()
    if not templates:
        print("Error: No templates found in templates/ folder.")
        print("Run extract_templates.py first to create digit templates.")
        sys.exit(1)

    print(f"Loaded templates for digits: {sorted(templates.keys())}")
    print(f"Match threshold: {threshold}\n")

    for path in paths:
        try:
            count = read_stack_count(path, threshold=threshold)
            print(f"  {path} -> {count}")
        except Exception as e:
            print(f"  {path} -> ERROR: {e}")


if __name__ == "__main__":
    main()
