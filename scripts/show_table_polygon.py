"""Overlay the configured table_polygon on outputs/latest.jpg.

Writes outputs/debug/polygon_overlay.jpg with the polygon drawn in orange
plus every Hough detection (green = kept, red = filtered out).
Use this to tune detection.table_polygon in config.yaml.
"""
from pathlib import Path

import cv2
import numpy as np

from src.config_loader import load_config
from src.detection import detect_balls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = PROJECT_ROOT / "outputs" / "latest.jpg"
DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def main():
    if not FRAME_PATH.exists():
        raise SystemExit(f"No frame at {FRAME_PATH}. Run capture first.")

    bgr = cv2.imread(str(FRAME_PATH))
    if bgr is None:
        raise SystemExit(f"Could not read {FRAME_PATH}")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(CONFIG_PATH))
    det_cfg = cfg["detection"]
    r = det_cfg.get("expected_radius_px", 12)

    polygon = det_cfg.get("table_polygon")
    if polygon is None:
        print("WARNING: no detection.table_polygon configured")
        polygon_np = None
    else:
        polygon_np = np.array(polygon, dtype=np.int32)

    # Run detection WITHOUT polygon filter, to show what would be kept vs dropped.
    det_cfg_no_poly = dict(det_cfg)
    det_cfg_no_poly.pop("table_polygon", None)
    all_dets = detect_balls(bgr, det_cfg_no_poly, r)

    kept, dropped = [], []
    for d in all_dets:
        if polygon_np is None:
            kept.append(d)
            continue
        inside = cv2.pointPolygonTest(polygon_np, d.center_px, False) >= 0
        (kept if inside else dropped).append(d)

    out = bgr.copy()
    if polygon_np is not None:
        cv2.polylines(out, [polygon_np], isClosed=True, color=(0, 165, 255), thickness=2)
    for d in kept:
        color = (0, 255, 0) if d.is_cue_ball else (255, 255, 0)
        cv2.circle(out, (int(d.center_px[0]), int(d.center_px[1])),
                   int(d.radius_px), color, 2)
    for d in dropped:
        cv2.circle(out, (int(d.center_px[0]), int(d.center_px[1])),
                   int(d.radius_px), (0, 0, 255), 2)

    out_path = DEBUG_DIR / "polygon_overlay.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")
    print(f"  kept    (green/yellow): {len(kept)}")
    print(f"  dropped (red, outside polygon): {len(dropped)}")
    if polygon_np is not None:
        print(f"  polygon: {polygon}")


if __name__ == "__main__":
    main()
