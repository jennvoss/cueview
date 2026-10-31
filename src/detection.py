import cv2
import numpy as np
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional, Union
from src.constants import (
    BALL_DIAMETER_IN, ON_TABLE_MIN_FRACTION, ISOLATION_RING_MAX_FRACTION,
    MIN_COLOR_COVERAGE, MIN_BLACK_COVERAGE, MIN_OWN_COLOR,
    BALL_NUMBERS_BY_COLOR, STRIPE_MIN_WHITE,
    CLOTH_REFERENCE_BGR, CLOTH_GAIN_LIMITS, MAROON_MIN_HIGHLIGHT, BLUE_PURPLE_HUE_SPLIT,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    center_px: tuple[float, float]
    radius_px: float
    is_cue_ball: bool
    color_label: str
    ball: Optional[Union[int, str]] = None  # "cue", 1-15, or None if unknown


def table_mask(shape: tuple, table_polygon) -> np.ndarray:
    """Return a uint8 mask (255 = playing surface) for the given frame shape."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(table_polygon, dtype=np.int32)], 255)
    return mask


@lru_cache(maxsize=256)
def _disc_template(radius_q: int, inner_q: int):
    """Boolean disc (or ring) centered in a (2R+1)² box; radii are in quarter pixels."""
    radius, inner = radius_q / 4.0, inner_q / 4.0
    half = max(1, int(np.ceil(radius)))
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
    d2 = xs ** 2 + ys ** 2
    template = (d2 <= radius * radius) & (d2 > inner * inner)
    return half, template, int(template.sum())


def fraction_on_table(mask: np.ndarray, center: tuple, radius: float, inner_radius: float = 0.0) -> float:
    """Fraction of the disc's pixels (or the ring's, with inner_radius) set in mask; off-frame counts as unset."""
    half, template, total = _disc_template(int(round(radius * 4)), int(round(inner_radius * 4)))
    if total == 0:
        return 0.0
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = mask.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half + 1, w)
    y0, y1 = max(cy - half, 0), min(cy + half + 1, h)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    window = template[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)]
    return np.count_nonzero(window & (mask[y0:y1, x0:x1] > 0)) / total


def hsv_range_mask(hsv: np.ndarray, hsv_range) -> np.ndarray:
    """Mask for one [low, high] HSV range, or the union of a list of them (e.g. red wrapping past hue 179)."""
    ranges = np.array(hsv_range)
    if ranges.ndim == 2:
        ranges = ranges[np.newaxis]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, low.astype(np.uint8), high.astype(np.uint8))
    return mask


@lru_cache(maxsize=8)
def _cloth_sample_index(shape: tuple, polygon: tuple):
    """Pixel indices (every 3rd row/column) well inside the table, used to measure the cloth."""
    inside = cv2.erode(table_mask(shape, [list(p) for p in polygon]), None, iterations=20)
    ys, xs = np.nonzero(inside[::3, ::3])
    return ys * 3, xs * 3


def correct_to_cloth(frame: np.ndarray, table_polygon) -> np.ndarray:
    """Scale each color channel so the cloth's median color matches CLOTH_REFERENCE_BGR,
    undoing changes in room brightness and light color before the HSV ranges are applied."""
    ys, xs = _cloth_sample_index(frame.shape[:2], tuple(tuple(p) for p in table_polygon))
    if len(ys) == 0:
        return frame
    cloth = np.median(frame[ys, xs], axis=0)
    gains = np.clip(np.array(CLOTH_REFERENCE_BGR) / np.maximum(cloth, 1.0), *CLOTH_GAIN_LIMITS)
    levels = np.arange(256, dtype=np.float32)
    lut = np.stack([np.clip(levels * g, 0, 255) for g in gains], axis=1).astype(np.uint8)
    return cv2.LUT(frame, lut.reshape(256, 1, 3))


def highlight_above_surroundings(value: np.ndarray, center: tuple, radius: float) -> float:
    """Brightest spot in the disc (97th percentile of V) minus the median V of a ring of
    cloth around it (1.5-2.2 radii). Real balls show the overhead lamp; cloth doesn't."""
    def values(outer, inner=0.0):
        half, template, _ = _disc_template(int(round(outer * 4)), int(round(inner * 4)))
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = value.shape[:2]
        x0, x1 = max(cx - half, 0), min(cx + half + 1, w)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, h)
        window = template[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)]
        return value[y0:y1, x0:x1][window]
    disc, ring = values(radius), values(2.2 * radius, 1.5 * radius)
    if disc.size == 0 or ring.size == 0:
        return 0.0
    return float(np.percentile(disc, 97)) - float(np.median(ring))


def compute_expected_radius_px(calibration: dict) -> float:
    """
    Use homography inverse to convert BALL_DIAMETER_IN to pixel radius.
    Returns float.
    """
    homography = calibration["homography"]
    homography_inv = np.linalg.inv(homography)

    center_in = np.array([44.0, 22.0], dtype=np.float32)
    edge_in = np.array([44.0 + BALL_DIAMETER_IN / 2, 22.0], dtype=np.float32)

    def project_to_pixels(point_in):
        ones = np.ones((1, 1), dtype=np.float32)
        homogeneous = np.hstack([point_in.reshape(1, -1).astype(np.float32), ones])
        transformed = homography_inv @ homogeneous.T
        result = transformed[:2, :].T / transformed[2, :, np.newaxis]
        return result[0]

    center_px = project_to_pixels(center_in)
    edge_px = project_to_pixels(edge_in)

    radius_px = float(np.linalg.norm(edge_px - center_px))
    return radius_px


def detect_balls(
    frame: np.ndarray,
    config: dict,
    expected_radius_px: float,
) -> list[Detection]:
    """Detect balls; with a table_polygon, only the area around the table is processed."""
    table_polygon = config.get("table_polygon")
    if not table_polygon:
        return _detect_balls(frame, config, expected_radius_px)
    # Crop to the table plus a margin. The crop origin is a multiple of 6 px so Hough's
    # accumulator grid (dp=1.2 -> 1.2 px cells) lines up exactly as on the full frame.
    margin = int(np.ceil(4 * expected_radius_px))
    poly = np.array(table_polygon, dtype=np.int32)
    h, w = frame.shape[:2]
    x0 = max(int(poly[:, 0].min()) - margin, 0) // 6 * 6
    y0 = max(int(poly[:, 1].min()) - margin, 0) // 6 * 6
    x1 = min(int(poly[:, 0].max()) + margin, w)
    y1 = min(int(poly[:, 1].max()) + margin, h)
    cropped = dict(config, table_polygon=(poly - [x0, y0]).tolist())
    region = correct_to_cloth(frame[y0:y1, x0:x1], cropped["table_polygon"])
    detections = _detect_balls(region, cropped, expected_radius_px)
    return [replace(d, center_px=(d.center_px[0] + x0, d.center_px[1] + y0)) for d in detections]


def _detect_balls(
    frame: np.ndarray,
    config: dict,
    expected_radius_px: float,
) -> list[Detection]:
    """
    1. Convert frame to HSV.
    2. For each HSV range in config['detection']['hsv_ranges'], create a mask.
    3. Run HoughCircles on each masked channel.
    4. Filter circles outside expected_radius_px ± diameter_tolerance.
    5. Mark the circle matching cue_ball HSV range as is_cue_ball=True.
    6. Return list of Detection objects. Empty list if no balls found — never raise.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = []
    hsv_ranges = config.get("hsv_ranges", {})
    diameter_tolerance = config.get("diameter_tolerance", 0.20)
    hough_dp = config.get("hough_dp", 1.2)
    hough_min_dist = config.get("hough_min_dist_px", 20)
    hough_param1 = config.get("hough_param1", 50)
    hough_param2 = config.get("hough_param2", 30)
    table_polygon = config.get("table_polygon")

    min_radius = expected_radius_px * (1.0 - diameter_tolerance)
    max_radius = expected_radius_px * (1.0 + diameter_tolerance)

    # (x, y, r, label) -> label. The same circle can turn up in several masks (a ball
    # is also a hole in a range that matches the cloth), so each label keeps its own copy.
    detected_circles = {}
    masks_by_label = {}

    def hough(mask):
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=hough_dp,
            minDist=hough_min_dist,
            param1=hough_param1,
            param2=hough_param2,
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        return [] if circles is None else [(float(x), float(y), float(r)) for x, y, r in circles[0, :]]

    # Ignore everything well outside the table (rails, room) so Hough has fewer edges
    # and candidates; the margin keeps balls at the cushions and pocket mouths whole.
    near_table = None
    if table_polygon:
        near_table = cv2.dilate(table_mask(frame.shape, table_polygon), None,
                                iterations=int(np.ceil(3 * expected_radius_px)))

    for color_label, hsv_range in hsv_ranges.items():
        mask = hsv_range_mask(hsv, hsv_range)
        if near_table is not None:
            mask &= near_table
        masks_by_label[color_label] = mask
        for circle_key in hough(mask):
            detected_circles[circle_key + (color_label,)] = color_label

    # A stripe is only part of a circle in its color mask and part in the white one;
    # their union is a full disc. Label those circles by their strongest color.
    if masks_by_label:
        any_ball = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for m in masks_by_label.values():
            any_ball |= m
        found = {key[:3] for key in detected_circles}
        for circle_key in hough(any_ball):
            if circle_key in found:
                continue
            center = (circle_key[0], circle_key[1])
            own = {label: fraction_on_table(m, center, circle_key[2])
                   for label, m in masks_by_label.items() if label != "cue_ball"}
            best = max(own, key=own.get, default=None)
            if best is not None and own[best] >= MIN_OWN_COLOR:
                detected_circles[circle_key + (best,)] = best
            elif "cue_ball" in masks_by_label:
                detected_circles[circle_key + ("cue_ball",)] = "cue_ball"

    if table_polygon:
        on_table = table_mask(frame.shape, table_polygon)
        detected_circles = {
            key: label for key, label in detected_circles.items()
            if fraction_on_table(on_table, (key[0], key[1]), key[2]) > ON_TABLE_MIN_FRACTION
        }

    detected_circles = {
        key: label for key, label in detected_circles.items()
        if fraction_on_table(masks_by_label[label], (key[0], key[1]),
                             1.8 * expected_radius_px, 1.3 * expected_radius_px) <= ISOLATION_RING_MAX_FRACTION
    }

    # Overlapping ranges find the same ball more than once: keep the circle whose
    # own color mask covers it best and drop any other circle centered on that ball.
    # Stripes are part white, so a colored circle's fill counts white pixels too.
    white_mask = masks_by_label.get("cue_ball")

    fill_masks = {
        label: m if label == "cue_ball" or white_mask is None else m | white_mask
        for label, m in masks_by_label.items()
    }

    def fill_mask(label):
        return fill_masks[label]

    coverage = {
        key: fraction_on_table(fill_mask(label), (key[0], key[1]), key[2])
        for key, label in detected_circles.items()
    }
    merged = {}
    for key in sorted(detected_circles, key=lambda k: coverage[k], reverse=True):
        label = detected_circles[key]
        if coverage[key] < (MIN_BLACK_COVERAGE if label == "black" else MIN_COLOR_COVERAGE):
            continue
        if label != "cue_ball" and fraction_on_table(masks_by_label[label], (key[0], key[1]), key[2]) < MIN_OWN_COLOR:
            continue
        if label == "maroon" and highlight_above_surroundings(
                hsv[..., 2], (key[0], key[1]), expected_radius_px) < MAROON_MIN_HIGHLIGHT:
            continue
        if all(np.hypot(key[0] - k[0], key[1] - k[1]) > expected_radius_px for k in merged):
            merged[key] = detected_circles[key]
    detected_circles = merged

    # The cue ball is the whitest candidate with the least of any other color. A
    # stripe lying white side up usually still shows its band near the edge, so
    # color in the rim (inside the ball) counts against it too.
    colored_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for label, m in masks_by_label.items():
        if label != "cue_ball":
            colored_mask |= m

    def cue_score(key):
        center = (key[0], key[1])
        others = [fraction_on_table(m, center, key[2])
                  for label, m in masks_by_label.items() if label != "cue_ball"]
        rim = fraction_on_table(colored_mask, center, 1.1 * expected_radius_px, 0.65 * expected_radius_px)
        return coverage[key] - max(others, default=0.0) - rim

    cue_keys = [key for key, label in detected_circles.items() if label == "cue_ball"]
    best_cue_key = max(cue_keys, key=cue_score) if cue_keys else None

    def resolve_blue_purple(center, radius, color_label):
        """Blue and purple ranges overlap; pick by the median hue of those pixels instead."""
        if color_label not in ("blue", "purple") or "blue" not in masks_by_label or "purple" not in masks_by_label:
            return color_label
        half, template, _ = _disc_template(int(round(radius * 4)), 0)
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = hsv.shape[:2]
        x0, x1 = max(cx - half, 0), min(cx + half + 1, w)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, h)
        window = template[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)]
        either = (masks_by_label["blue"] | masks_by_label["purple"])[y0:y1, x0:x1] > 0
        hues = hsv[y0:y1, x0:x1, 0][window & either]
        if hues.size == 0:
            return color_label
        return "blue" if np.median(hues) < BLUE_PURPLE_HUE_SPLIT else "purple"

    def ball_number(center, radius, color_label, is_cue):
        if is_cue:
            return "cue"
        white = (fraction_on_table(white_mask, center, expected_radius_px, 0.6 * expected_radius_px)
                 if white_mask is not None else 0.0)
        if color_label == "cue_ball":
            # A stripe lying white side up: name it by whatever color shows.
            colors = {label: fraction_on_table(m, center, radius)
                      for label, m in masks_by_label.items() if label in BALL_NUMBERS_BY_COLOR and label != "black"}
            color_label = max(colors, key=colors.get, default=None)
            if color_label is None or colors[color_label] == 0:
                return None
            return BALL_NUMBERS_BY_COLOR[resolve_blue_purple(center, radius, color_label)][1]
        numbers = BALL_NUMBERS_BY_COLOR.get(color_label)
        if numbers is None:
            return None
        return numbers[1] if white >= STRIPE_MIN_WHITE else numbers[0]

    for (x, y, r, _), color_label in detected_circles.items():
        is_cue = ((x, y, r, color_label) == best_cue_key)
        color_label = resolve_blue_purple((x, y), r, color_label)
        detection = Detection(
            center_px=(x, y),
            radius_px=r,
            is_cue_ball=is_cue,
            color_label=color_label,
            ball=ball_number((x, y), r, color_label, is_cue),
        )
        detections.append(detection)

    return detections
