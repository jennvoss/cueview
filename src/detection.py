import cv2
import numpy as np
import logging
from dataclasses import dataclass
from src.constants import BALL_DIAMETER_IN

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    center_px: tuple[float, float]
    radius_px: float
    is_cue_ball: bool
    color_label: str


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

    min_radius = expected_radius_px * (1.0 - diameter_tolerance)
    max_radius = expected_radius_px * (1.0 + diameter_tolerance)

    detected_circles = {}

    for color_label, hsv_range in hsv_ranges.items():
        lower = np.array(hsv_range[0], dtype=np.uint8)
        upper = np.array(hsv_range[1], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

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

        if circles is not None:
            circles = circles[0, :]
            for x, y, r in circles:
                circle_key = (float(x), float(y), float(r))
                if circle_key not in detected_circles:
                    detected_circles[circle_key] = color_label

    for (x, y, r), color_label in detected_circles.items():
        is_cue = (color_label == "cue_ball")
        detection = Detection(
            center_px=(x, y),
            radius_px=r,
            is_cue_ball=is_cue,
            color_label=color_label,
        )
        detections.append(detection)

    return detections
