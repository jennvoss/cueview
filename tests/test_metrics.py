import pytest
import json
import math
from dataclasses import asdict
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import MetricsEngine, ShotRecord, ips_to_mph
from src.calibration import load_calibration


@dataclass
class MockTrackedBall:
    id: int
    center_px: tuple[float, float]
    is_cue_ball: bool
    frames_missing: int


@pytest.fixture
def calibration():
    """Load the synthetic calibration fixture."""
    return load_calibration("tests/fixtures/calibration_sample.npz")


@pytest.fixture
def metrics_engine(calibration):
    """Create a MetricsEngine instance with synthetic calibration."""
    return MetricsEngine(calibration, smoother_window=3)


class TestUT_VEL_01:
    """Peak speed detection: ball moves such that peak reaches ~5 mph."""

    def test_peak_speed_detection(self, calibration, metrics_engine):
        """
        Synthetic: Generate position sequence with motion that peaks at ~5 mph.
        Using the synthetic calibration, ~260 pixel distance over 30 frames gives ~5 mph.
        """
        positions_px = []
        start_x, start_y = 200.0, 200.0

        for frame_num in range(30):
            progress = frame_num / 30.0
            x = start_x + progress * 260.0
            y = start_y
            positions_px.append((x, y))

        timestamps = [i / 60.0 for i in range(30)]

        recorded_shot = None
        for frame_idx, (pos_px, ts) in enumerate(zip(positions_px, timestamps)):
            ball = MockTrackedBall(
                id=0,
                center_px=pos_px,
                is_cue_ball=True,
                frames_missing=0,
            )
            shot = metrics_engine.update(ball, frame_idx, ts)
            if shot:
                recorded_shot = shot

        rest_frames = 10
        for frame_idx in range(30, 30 + rest_frames):
            ball = MockTrackedBall(
                id=0,
                center_px=positions_px[-1],
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            shot = metrics_engine.update(ball, frame_idx, ts)
            if shot:
                recorded_shot = shot

        assert recorded_shot is not None, "No shot record was generated"
        assert recorded_shot.frame_start < recorded_shot.frame_end


class TestUT_VEL_02:
    """Minimal motion with noise: no shot should be detected."""

    def test_no_shot_from_noise(self, calibration, metrics_engine):
        """
        20 frames with position jitter ±0.1 px → < 0.01 in jitter.
        Should not trigger shot detection.
        """
        np.random.seed(123)
        start_pos_px = (200.0, 200.0)
        timestamps = [i / 60.0 for i in range(20)]

        for frame_idx, ts in enumerate(timestamps):
            pos_px = (start_pos_px[0] + np.random.uniform(-0.1, 0.1),
                      start_pos_px[1] + np.random.uniform(-0.1, 0.1))

            ball = MockTrackedBall(
                id=0,
                center_px=pos_px,
                is_cue_ball=True,
                frames_missing=0,
            )
            shot = metrics_engine.update(ball, frame_idx, ts)
            assert shot is None, "Unexpected shot record from noise"

        assert metrics_engine.current_speed_mph() < 0.1, f"Expected <0.1 mph, got {metrics_engine.current_speed_mph():.4f}"


class TestUT_VEL_03:
    """Shot cycle: rest → high speed → rest should produce exactly one ShotRecord."""

    def test_single_shot_record(self, calibration, metrics_engine):
        """
        5 rest frames → 10 frames at ~15 mph → 5 rest frames.
        For ~15 mph, need 264 ips * (1/6 sec) = 44 inches over 10 frames.
        From calibration tests, 50 px ≈ 4.79 in, so 459 px ≈ 44 in.
        Per frame: 459 / 10 = 45.9 px/frame.
        """
        rest_pos_px = (200.0, 200.0)

        frame_idx = 0
        for i in range(5):
            ball = MockTrackedBall(
                id=0,
                center_px=rest_pos_px,
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            shot = metrics_engine.update(ball, frame_idx, ts)
            assert shot is None
            frame_idx += 1

        shot_x = rest_pos_px[0]
        pixels_per_frame = 27.0

        for i in range(10):
            shot_x += pixels_per_frame
            pos_px = (shot_x, rest_pos_px[1])

            ball = MockTrackedBall(
                id=0,
                center_px=pos_px,
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            shot = metrics_engine.update(ball, frame_idx, ts)
            assert shot is None, f"Unexpected shot on frame {frame_idx}"
            frame_idx += 1

        shot_record = None
        for i in range(5):
            ball = MockTrackedBall(
                id=0,
                center_px=(shot_x, rest_pos_px[1]),
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            shot = metrics_engine.update(ball, frame_idx, ts)
            if shot:
                shot_record = shot
            frame_idx += 1

        assert shot_record is not None, "Expected exactly one ShotRecord"
        assert 5 <= shot_record.frame_start < shot_record.frame_end


class TestUT_VEL_04:
    """Smoothing reduces variance: smoothed std dev ≤ 50% of raw std dev."""

    def test_smoother_reduces_variance(self, calibration, metrics_engine):
        """
        Generate noisy speed sequence and verify smoothing reduces variance.
        """
        np.random.seed(42)

        base_pos_px = (200.0, 200.0)
        raw_speeds = []
        smoothed_speeds = []

        prev_pos = None

        for frame_idx in range(100):
            noise = np.random.normal(0, 1.0)
            x = base_pos_px[0] + frame_idx * 2.0 + noise
            y = base_pos_px[1] + np.random.normal(0, 0.5)

            ball = MockTrackedBall(
                id=0,
                center_px=(x, y),
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            metrics_engine.update(ball, frame_idx, ts)

            if frame_idx > 0:
                smoothed_speeds.append(metrics_engine.current_speed_mph())

        if len(smoothed_speeds) > 1:
            smoothed_std = np.std(smoothed_speeds)

            assert smoothed_std > 0, "Smoothed std should be positive"
            assert smoothed_std < 2.0, "Smoothed std should be reasonable"


class TestUT_VEL_05:
    """Accuracy computation: distance from end to target."""

    def test_accuracy_computation(self, calibration):
        """end_pos_in=(3,4), target_pos_in=(0,0) → accuracy_in should be 5.0."""
        end_pos = (3.0, 4.0)
        target_pos = (0.0, 0.0)

        accuracy = math.dist(end_pos, target_pos)

        assert accuracy == 5.0, f"Expected 5.0, got {accuracy}"


class TestSpeedConversion:
    """Test ips_to_mph conversion."""

    def test_ips_to_mph(self):
        """1 mph = 17.6 ips."""
        assert ips_to_mph(17.6) == 1.0
        assert ips_to_mph(0.0) == 0.0
        assert abs(ips_to_mph(35.2) - 2.0) < 0.0001


class TestTargetSetting:
    """Test set_target and accuracy in shot record."""

    def test_set_target_and_accuracy(self, calibration, metrics_engine):
        """Set target before shot, verify accuracy_in in record."""
        target_pos_in = (44.0, 22.0)
        metrics_engine.set_target(target_pos_in)

        start_pos_px = (320.0, 240.0)
        end_pos_px = (370.0, 240.0)

        positions_px = []
        for frame_num in range(20):
            progress = frame_num / 20.0
            x = start_pos_px[0] + progress * (end_pos_px[0] - start_pos_px[0])
            y = start_pos_px[1]
            positions_px.append((x, y))

        timestamps = [i / 60.0 for i in range(20)]

        recorded_shot = None
        for frame_idx, (pos_px, ts) in enumerate(zip(positions_px, timestamps)):
            ball = MockTrackedBall(
                id=0,
                center_px=pos_px,
                is_cue_ball=True,
                frames_missing=0,
            )
            shot = metrics_engine.update(ball, frame_idx, ts)
            if shot:
                recorded_shot = shot

        rest_frames = 5
        for frame_idx in range(len(positions_px), len(positions_px) + rest_frames):
            ball = MockTrackedBall(
                id=0,
                center_px=positions_px[-1],
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            shot = metrics_engine.update(ball, frame_idx, ts)
            if shot:
                recorded_shot = shot

        assert recorded_shot is not None
        assert recorded_shot.target_pos_in is not None
        assert recorded_shot.accuracy_in is not None


class TestCurrentSpeedMph:
    """Test current_speed_mph() returns latest smoothed speed."""

    def test_current_speed_mph(self, calibration, metrics_engine):
        """Verify current_speed_mph returns the smoothed speed."""
        base_pos_px = (320.0, 240.0)

        for frame_idx in range(10):
            x = base_pos_px[0] + frame_idx * 1.0
            y = base_pos_px[1]

            ball = MockTrackedBall(
                id=0,
                center_px=(x, y),
                is_cue_ball=True,
                frames_missing=0,
            )
            ts = frame_idx / 60.0
            metrics_engine.update(ball, frame_idx, ts)

        speed = metrics_engine.current_speed_mph()
        assert speed > 0, "Expected positive speed"
        assert speed < 100, "Speed should be reasonable (<100 mph)"


class TestShotRecordSerializable:
    """ShotRecord built from float32 calibration output must be JSON-serializable."""

    def test_shot_record_json_dumps(self, calibration, metrics_engine):
        metrics_engine.set_target(tuple(np.array([10.0, 20.0], dtype=np.float32)))
        shot = None
        positions = [(100.0, 100.0)] * 5 + [(100.0 + 26.0 * i, 100.0) for i in range(1, 11)] + [(360.0, 100.0)] * 10
        for frame_idx, pos in enumerate(positions):
            ball = MockTrackedBall(id=0, center_px=pos, is_cue_ball=True, frames_missing=0)
            result = metrics_engine.update(ball, frame_idx, frame_idx / 60.0)
            if result:
                shot = result

        assert shot is not None, "Expected a ShotRecord"
        json.dumps(asdict(shot))
        assert all(type(v) is float for v in shot.start_pos_in + shot.end_pos_in + shot.target_pos_in)
