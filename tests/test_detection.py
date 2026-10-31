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
    """UT-DET-02: detect_balls(fixture_16_balls) finds ≥11 distinct balls, none twice; exactly 1 has is_cue_ball=True.
    Only 11 of the fixture's 16 pastel colors fall inside the test config's ranges. Earlier
    counts were inflated by duplicates and by "balls" found as holes in the green mask,
    which also matches this fixture's green cloth."""
    image_path = fixture_dir / "fixture_16_balls.png"
    frame = cv2.imread(str(image_path))

    expected_radius = compute_expected_radius_px(calibration)

    detections = detect_balls(frame, test_config, expected_radius)

    centers = [d.center_px for d in detections]
    for i, a in enumerate(centers):
        for b in centers[i + 1:]:
            assert np.hypot(a[0] - b[0], a[1] - b[1]) > expected_radius / 2, f"ball detected twice near {a}"
    assert len(detections) >= 11
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


class TestOnTableRule:
    """A ball counts as on the table only if most of its disc is inside table_polygon."""

    POLYGON = [[100, 100], [540, 100], [540, 380], [100, 380]]
    RADIUS = 20.0

    def _frame_with_ball(self, center):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, center, int(self.RADIUS), (255, 255, 255), -1)
        return frame

    def test_fraction_along_straight_edge(self):
        from src.detection import table_mask, fraction_on_table
        mask = table_mask((480, 640), self.POLYGON)
        r = self.RADIUS
        mostly_in = fraction_on_table(mask, (320.0, 100 + 0.25 * r), r)
        mostly_out = fraction_on_table(mask, (320.0, 100 - 0.25 * r), r)
        assert 0.55 < mostly_in < 0.75
        assert 0.25 < mostly_out < 0.45
        assert fraction_on_table(mask, (320.0, 240.0), r) == 1.0
        assert fraction_on_table(mask, (10.0, 10.0), r) == 0.0

    def test_ball_mostly_inside_edge_is_kept(self, test_config):
        cfg = dict(test_config, table_polygon=self.POLYGON)
        frame = self._frame_with_ball((320, int(100 + 0.25 * self.RADIUS)))
        detections = detect_balls(frame, cfg, self.RADIUS)
        assert len(detections) == 1

    def test_ball_mostly_outside_edge_is_dropped(self, test_config):
        cfg = dict(test_config, table_polygon=self.POLYGON)
        frame = self._frame_with_ball((320, int(100 - 0.25 * self.RADIUS)))
        assert detect_balls(frame, cfg, self.RADIUS) == []

    def test_ball_fully_inside_at_corner_is_kept(self, test_config):
        """Like the cue ball resting next to a corner pocket but on the cloth."""
        cfg = dict(test_config, table_polygon=self.POLYGON)
        frame = self._frame_with_ball((100 + int(self.RADIUS) + 1, 100 + int(self.RADIUS) + 1))
        assert len(detect_balls(frame, cfg, self.RADIUS)) == 1

    def test_center_inside_but_mostly_over_corner_is_dropped(self, test_config):
        """Center is inside the polygon, but ~3/4 of the ball hangs over the corner."""
        from src.detection import table_mask, fraction_on_table
        cfg = dict(test_config, table_polygon=self.POLYGON)
        center = (103, 103)
        assert cv2.pointPolygonTest(np.array(self.POLYGON, dtype=np.int32), center, False) > 0
        assert fraction_on_table(table_mask((480, 640), self.POLYGON), center, self.RADIUS) < 0.5
        assert detect_balls(self._frame_with_ball(center), cfg, self.RADIUS) == []


class TestLightingChanges:
    """Frames are color-corrected to the cloth, so a dimmer or warmer room still works."""

    TABLE = Path(__file__).parent / "fixtures" / "table"

    @pytest.fixture
    def real_config(self):
        from src.config_loader import load_config
        return load_config("config/config.yaml")["detection"]

    @staticmethod
    def _relight(img, gains_bgr):
        return np.clip(img.astype(np.float32) * np.array(gains_bgr, dtype=np.float32), 0, 255).astype(np.uint8)

    def test_correction_restores_cloth_color(self, real_config):
        from src.constants import CLOTH_REFERENCE_BGR
        from src.detection import correct_to_cloth, table_mask
        img = self._relight(cv2.imread(str(self.TABLE / "03_solids.png")), (0.6, 0.65, 0.7))
        fixed = correct_to_cloth(img, real_config["table_polygon"])
        inside = cv2.erode(table_mask(img.shape, real_config["table_polygon"]), None, iterations=20) > 0
        assert np.allclose(np.median(fixed[inside], axis=0), CLOTH_REFERENCE_BGR, atol=2)

    @pytest.mark.parametrize("gains", [(0.6, 0.65, 0.7), (0.8, 0.95, 1.1)], ids=["dim", "warm"])
    def test_detection_survives_lighting_change(self, real_config, gains):
        """Like turning the room lights down, or switching to a warmer bulb."""
        img = cv2.imread(str(self.TABLE / "03_solids.png"))
        expected = {(d.ball, round(d.center_px[0] / 6), round(d.center_px[1] / 6))
                    for d in detect_balls(img, real_config, 12.0)}
        relit = {(d.ball, round(d.center_px[0] / 6), round(d.center_px[1] / 6))
                 for d in detect_balls(self._relight(img, gains), real_config, 12.0)}
        assert {b for b, _, _ in relit} == {b for b, _, _ in expected}

    def test_highlight_separates_ball_from_cloth(self):
        from src.detection import highlight_above_surroundings
        value = np.full((100, 100), 80, dtype=np.uint8)
        cv2.circle(value, (50, 50), 12, 70, -1)    # dark ball, same tone as cloth
        assert highlight_above_surroundings(value, (50, 50), 12.0) < 10
        cv2.circle(value, (46, 46), 3, 220, -1)    # the lamp's reflection on a real ball
        assert highlight_above_surroundings(value, (50, 50), 12.0) > 100
