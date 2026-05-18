"""Find the cue ball in outputs/latest.jpg and print its HSV stats.

Heuristic: pick the brightest, least-colorful 30x30 patch on the table.
Then report H/S/V min/median/p10/p90/max so we can pick a robust HSV range.
"""
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = PROJECT_ROOT / "outputs" / "latest.jpg"


def main():
    bgr = cv2.imread(str(FRAME_PATH))
    if bgr is None:
        raise SystemExit(f"Could not read {FRAME_PATH}")
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Exclude the HUD overlay strip (top ~60 px) and the very edge of the frame.
    search_mask = np.zeros((h, w), dtype=bool)
    search_mask[60:h - 30, 60:w - 60] = True

    # Find brightest, least-saturated pixels (cue-ball-like) within the table area.
    V = hsv[..., 2].astype(np.int32)
    S = hsv[..., 1].astype(np.int32)
    # Cue ball is bright but not pure 255 (HUD text is exactly 255). Penalize 255 to avoid text.
    score = np.where(V >= 254, -1000, V - 2 * S)
    score = np.where(search_mask, score, -1000)
    flat = score.flatten()
    top_idx = np.argpartition(flat, -500)[-500:]
    ys, xs = np.unravel_index(top_idx, score.shape)

    # Cluster the candidate pixels into a tight box around the densest region.
    cy, cx = int(np.median(ys)), int(np.median(xs))
    half = 18
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    patch = hsv[y0:y1, x0:x1]
    print(f"Frame {w}x{h}.  Sampling {patch.shape[1]}x{patch.shape[0]} patch at center ({cx}, {cy}).")

    # Mask: only the patch pixels that are actually white-ish (S<80, V>120) — drop edges.
    s_patch = patch[..., 1]
    v_patch = patch[..., 2]
    mask = (s_patch < 80) & (v_patch > 120)
    if mask.sum() < 50:
        print("  Few white-ish pixels in patch — falling back to full patch.")
        mask = np.ones_like(s_patch, dtype=bool)
    h_vals = patch[..., 0][mask]
    s_vals = patch[..., 1][mask]
    v_vals = patch[..., 2][mask]
    print(f"  Sampled {mask.sum()} cue-ball-ish pixels.")
    for name, vals in (("H", h_vals), ("S", s_vals), ("V", v_vals)):
        print(f"  {name}: min={vals.min()} p10={int(np.percentile(vals,10))} "
              f"median={int(np.median(vals))} p90={int(np.percentile(vals,90))} max={vals.max()}")

    # Suggest a robust HSV range with margin.
    h_lo = int(max(0, np.percentile(h_vals, 5) - 5))
    h_hi = int(min(180, np.percentile(h_vals, 95) + 5))
    s_lo = 0
    s_hi = int(min(255, np.percentile(s_vals, 95) + 20))
    v_lo = int(max(0, np.percentile(v_vals, 5) - 20))
    v_hi = 255
    print()
    print(f"Suggested cue_ball HSV range: [[{h_lo},{s_lo},{v_lo}], [{h_hi},{s_hi},{v_hi}]]")

    # Save a sanity-check image showing the patch.
    debug_dir = PROJECT_ROOT / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sanity = bgr.copy()
    cv2.rectangle(sanity, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(sanity, "sampled patch", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out = debug_dir / "sample_patch.jpg"
    cv2.imwrite(str(out), sanity)
    print(f"Saved sanity check: {out}")


if __name__ == "__main__":
    main()
