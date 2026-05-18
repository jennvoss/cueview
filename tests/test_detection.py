import pytest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection import detect_balls, compute_expected_radius_px, Detection
from src.calibration import load_calibration


@pytest.fixture
def fixture_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_config():
    return {
        "hsv_ranges": {
            "cue_ball": [[0, 0, 200], [180, 30, 255]],
            "solid_yellow": [[20, 100, 100], [30, 255, 255]],
            "solid_blue": [[100, 100, 100], [120, 255, 255]],
            "solid_green": [[40, 100, 100], [80, 255, 255]],
            "solid_red": [[160, 100, 100], [179, 255, 255]],
            "solid_purple": [[120, 50, 100], [160, 255, 255]],
            "solid_cyan": [[80, 100, 100], [100, 255, 255]],
            "solid_grey": [[0, 0, 100], [180, 50, 200]],
            "stripe_blue": [[100, 50, 100], [120, 200, 255]],
            "stripe_red": [[160, 50, 100], [179, 200, 255]],
            "stripe_green": [[40, 50, 100], [80, 200, 255]],
            "stripe_yellow": [[20, 50, 100], [30, 200, 255]],
            "stripe_purple": [[120, 0, 100], [160, 200, 255]],
            "stripe_orange": [[5, 100, 100], [20, 255, 255]],
            "stripe_magenta": [[140, 50, 100], [160, 255, 255]],
        },
        "hough_dp": 1.2,
        "hough_min_dist_px": 20,
        "hough_param1": 50,
        "hough_param2": 20,
        "diameter_tolerance": 0.20,
    }


@pytest.fixture
def calibration():
    fixture_path = Path(__file__).parent / "fixtures" / "calibration_sample.npz"
    return load_calibration(str(fixture_path))


def test_detect_single_white_ball(fixture_dir, test_config, calibration):
    """UT-DET-01: detect_balls(fixture_single_white_ball) returns 1 Detection with is_cue_ball=True"""
    image_path = fixture_dir / "fixture_single_white_ball.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    assert len(detections) == 1
    assert detections[0].is_cue_ball is True


def test_detect_16_balls(fixture_dir, test_config, calibration):
    """UT-DET-02: detect_balls(fixture_16_balls) returns ≥14 Detections; exactly 1 has is_cue_ball=True"""
    image_path = fixture_dir / "fixture_16_balls.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    assert len(detections) >= 14
    cue_count = sum(1 for d in detections if d.is_cue_ball)
    assert cue_count == 1


def test_detect_noise_coin(fixture_dir, test_config, calibration):
    """UT-DET-03: detect_balls(fixture_noise) does not return the coin (r=8 px, below size filter)"""
    image_path = fixture_dir / "fixture_noise.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    for det in detections:
        assert det.radius_px >= expected_radius * 0.8


def test_detect_noise_large_region(fixture_dir, test_config, calibration):
    """UT-DET-04: detect_balls(fixture_noise) does not return the large white region (r=60 px, above size filter)"""
    image_path = fixture_dir / "fixture_noise.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    for det in detections:
        assert det.radius_px <= expected_radius * 1.2


def test_detect_empty_table(fixture_dir, test_config, calibration):
    """UT-DET-05: detect_balls(fixture_empty_table) returns empty list without raising"""
    image_path = fixture_dir / "fixture_empty_table.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    assert len(detections) == 0


def test_detect_near_rail(fixture_dir, test_config, calibration):
    """UT-DET-06: detect_balls(fixture_near_rail) returns ≥3 of the 4 near-rail balls"""
    image_path = fixture_dir / "fixture_near_rail.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    assert len(detections) >= 3
