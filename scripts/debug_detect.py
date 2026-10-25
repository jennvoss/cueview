"""Diagnostic: run detection on outputs/latest.jpg with varying radii + HSV masks.

Writes annotated outputs to outputs/debug/:
  - mask_<label>.jpg          binary HSV mask per configured range
  - circles_r<N>.jpg          input frame with Hough circles drawn, per radius
Prints how many circles each (radius, mask) combination finds.
"""
from pathlib import Path

import cv2
import numpy as np

from src.config_loader import load_config
from src.detection import hsv_range_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = PROJECT_ROOT / "outputs" / "latest.jpg"
DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

RADII_TO_TRY = [15, 18, 20, 22, 25, 28]


def main():
    if not FRAME_PATH.exists():
        raise SystemExit(f"No frame at {FRAME_PATH}. Run capture first to populate it.")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise SystemExit(f"Could not read {FRAME_PATH}")
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    cfg = load_config(str(CONFIG_PATH))
    det_cfg = cfg["detection"]
    hsv_ranges = det_cfg["hsv_ranges"]
    hough_dp = det_cfg.get("hough_dp", 1.2)
    hough_min_dist = det_cfg.get("hough_min_dist_px", 20)
    hough_param1 = det_cfg.get("hough_param1", 50)
    hough_param2 = det_cfg.get("hough_param2", 30)
    diameter_tolerance = det_cfg.get("diameter_tolerance", 0.20)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = {}
    for label, rng in hsv_ranges.items():
        mask = hsv_range_mask(hsv, rng)
        masks[label] = mask
        out = DEBUG_DIR / f"mask_{label}.jpg"
        cv2.imwrite(str(out), mask)
        nz = int(np.count_nonzero(mask))
        pct = 100.0 * nz / (h * w)
        print(f"  mask[{label}]: {nz} non-zero px ({pct:.2f}% of frame) -> {out.name}")

    print()
    print("Hough sweep (dp=%.2f, minDist=%d, p1=%d, p2=%d, tol=%.2f):"
          % (hough_dp, hough_min_dist, hough_param1, hough_param2, diameter_tolerance))

    for r_expected in RADII_TO_TRY:
        min_r = int(r_expected * (1.0 - diameter_tolerance))
        max_r = int(r_expected * (1.0 + diameter_tolerance))
        annotated = frame.copy()
        per_label_counts = {}

        for label, mask in masks.items():
            circles = cv2.HoughCircles(
                mask, cv2.HOUGH_GRADIENT,
                dp=hough_dp, minDist=hough_min_dist,
                param1=hough_param1, param2=hough_param2,
                minRadius=min_r, maxRadius=max_r,
            )
            count = 0
            if circles is not None:
                circles = circles[0, :]
                count = len(circles)
                color = (0, 255, 0) if label == "cue_ball" else (0, 255, 255)
                for x, y, rr in circles:
                    cv2.circle(annotated, (int(x), int(y)), int(rr), color, 2)
                    cv2.circle(annotated, (int(x), int(y)), 2, (0, 0, 255), -1)
            per_label_counts[label] = count

        label_text = f"r={r_expected} [{min_r}-{max_r}] " + " ".join(
            f"{lbl}={c}" for lbl, c in per_label_counts.items())
        cv2.putText(annotated, label_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        out = DEBUG_DIR / f"circles_r{r_expected:02d}.jpg"
        cv2.imwrite(str(out), annotated)
        total = sum(per_label_counts.values())
        print(f"  r={r_expected:3d} [{min_r}-{max_r} px]  total={total:3d}  "
              f"{per_label_counts}  -> {out.name}")


if __name__ == "__main__":
    main()
