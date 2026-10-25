"""Shrink the table test images in tests/fixtures/table/ without changing detection.

Blacks out everything more than MARGIN_PX outside detection.table_polygon's bounding
box (detection never looks past 4 ball radii, ~48 px) and re-saves at maximum PNG
compression. Pixels on and around the table are untouched, so detections are
identical. Run after capturing new stills with the 's' key:

    python3 -m scripts.compact_table_fixtures
"""
from pathlib import Path

import cv2
import numpy as np

from src.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "table"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
MARGIN_PX = 64


def main():
    poly = np.array(load_config(str(CONFIG_PATH))["detection"]["table_polygon"])
    total_before = total_after = 0
    for path in sorted(TABLE_DIR.glob("*.png")):
        frame = cv2.imread(str(path))
        h, w = frame.shape[:2]
        x0, y0 = np.maximum(poly.min(axis=0) - MARGIN_PX, 0)
        x1, y1 = np.minimum(poly.max(axis=0) + MARGIN_PX, [w, h])
        compact = np.zeros_like(frame)
        compact[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        before = path.stat().st_size
        cv2.imwrite(str(path), compact, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        after = path.stat().st_size
        total_before += before
        total_after += after
        print(f"{path.name:28s} {before / 1e6:5.2f} MB -> {after / 1e6:5.2f} MB")
    print(f"total: {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
