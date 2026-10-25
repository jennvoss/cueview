import pytest
import json
import math
from dataclasses import asdict
import numpy as np
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import MetricsEngine, ShotRecord
from src.calibration import load_calibration, pixel_to_inches
from src.constants import REST_WINDOW_S

BALL_RADIUS_PX = 12.0


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
    return MetricsEngine(calibration, BALL_RADIUS_PX)


def feed(engine, positions, fps=60.0, start_frame=0):
    """Feed px positions to the engine; return list of (frame_idx, ShotRecord)."""
    shots = []
    for i, pos in enumerate(positions):
        frame_idx = start_frame + i
        ball = MockTrackedBall(id=0, center_px=pos, is_cue_ball=True, frames_missing=0)
        shot = engine.update(ball, frame_idx, frame_idx / fps)
        if shot:
            shots.append((frame_idx, shot))
    return shots


def still(pos, n):
    return [pos] * n


def roll(start, end, n):
    return [(start[0] + (end[0] - start[0]) * (i + 1) / n,
             start[1] + (end[1] - start[1]) * (i + 1) / n) for i in range(n)]


class TestNoPhantomShots:
    """A still ball with real-camera-sized noise must never start a shot."""

    @pytest.mark.parametrize("fps", [15.0, 60.0])
    def test_noisy_still_ball(self, metrics_engine, fps):
        # Live camera measured ~0.6 px std, 3.6 px max range; use ±2.5 px uniform
        rng = np.random.default_rng(0)
        n = int(fps * 30)
        positions = [(300 + rng.uniform(-2.5, 2.5), 200 + rng.uniform(-2.5, 2.5)) for _ in range(n)]
        for i, pos in enumerate(positions):
            ball = MockTrackedBall(id=0, center_px=pos, is_cue_ball=True, frames_missing=0)
            assert metrics_engine.update(ball, i, i / fps) is None
            assert metrics_engine.is_in_shot is False

    def test_missing_ball_returns_none(self, metrics_engine):
        assert metrics_engine.update(None, 0, 0.0) is None


class TestShotCycle:
    """Rest → roll → rest should produce exactly one ShotRecord."""

    @pytest.mark.parametrize("fps", [15.0, 60.0])
    def test_single_shot(self, calibration, metrics_engine, fps):
        rest_a, rest_b = (200.0, 200.0), (500.0, 200.0)
        n_rest = int(fps * 1.0)
        positions = still(rest_a, n_rest) + roll(rest_a, rest_b, int(fps * 0.5)) + still(rest_b, n_rest)
        shots = feed(metrics_engine, positions, fps)

        assert len(shots) == 1
        end_frame, shot = shots[0]
        stop_frame = n_rest + int(fps * 0.5) - 1
        # Shot ends once the ball has been still for REST_WINDOW_S, and not much later
        assert stop_frame + fps * REST_WINDOW_S - 1 <= end_frame <= stop_frame + fps * REST_WINDOW_S + 2
        assert n_rest <= shot.frame_start < stop_frame

        expected_start = pixel_to_inches(np.array([rest_a]), calibration)[0]
        expected_end = pixel_to_inches(np.array([rest_b]), calibration)[0]
        assert math.dist(shot.start_pos_in, expected_start) < 0.01
        assert math.dist(shot.end_pos_in, expected_end) < 0.01
        expected_angle = math.degrees(math.atan2(expected_end[1] - expected_start[1],
                                                 expected_end[0] - expected_start[0]))
        assert abs(shot.angle_deg - expected_angle) < 0.5

    def test_slow_roll_does_not_end_shot(self, metrics_engine):
        """Ball rolling at 30 px/s (2.5 radii/s) for 3 s stays in one shot."""
        fps = 60.0
        start = (200.0, 200.0)
        end = (200.0 + 30.0 * 3, 200.0)
        shots = feed(metrics_engine, still(start, 60), fps)
        shots += feed(metrics_engine, roll(start, end, 180), fps, start_frame=60)
        assert shots == []
        assert metrics_engine.is_in_shot is True
        shots += feed(metrics_engine, still(end, 60), fps, start_frame=240)
        assert len(shots) == 1

    def test_two_sequential_shots(self, metrics_engine):
        a, b, c = (200.0, 200.0), (500.0, 200.0), (500.0, 400.0)
        positions = still(a, 60) + roll(a, b, 20) + still(b, 60) + roll(b, c, 20) + still(c, 60)
        shots = feed(metrics_engine, positions)
        assert len(shots) == 2
        assert abs(shots[0][1].angle_deg - shots[1][1].angle_deg) > 45.0

    def test_small_nudge_is_not_a_shot(self, metrics_engine):
        """Moving less than one ball radius (e.g. bumping the table) is ignored."""
        positions = still((200.0, 200.0), 60) + roll((200.0, 200.0), (208.0, 200.0), 10) + still((208.0, 200.0), 60)
        assert feed(metrics_engine, positions) == []
        assert metrics_engine.is_in_shot is False


class TestTargetSetting:
    """Test set_target and accuracy in shot record."""

    def test_set_target_and_accuracy(self, calibration, metrics_engine):
        target_pos_in = (44.0, 22.0)
        metrics_engine.set_target(target_pos_in)

        end_px = (370.0, 240.0)
        positions = still((320.0, 240.0), 40) + roll((320.0, 240.0), end_px, 20) + still(end_px, 40)
        shots = feed(metrics_engine, positions)

        assert len(shots) == 1
        shot = shots[0][1]
        end_in = pixel_to_inches(np.array([end_px]), calibration)[0]
        assert shot.target_pos_in == target_pos_in
        assert abs(shot.accuracy_in - math.dist(end_in, target_pos_in)) < 0.01
        assert metrics_engine.target_pos_in is None


class TestShotRecordSerializable:
    """ShotRecord built from float32 calibration output must be JSON-serializable."""

    def test_shot_record_json_dumps(self, calibration, metrics_engine):
        metrics_engine.set_target(tuple(np.array([10.0, 20.0], dtype=np.float32)))
        positions = still((100.0, 100.0), 40) + roll((100.0, 100.0), (360.0, 100.0), 10) + still((360.0, 100.0), 40)
        shots = feed(metrics_engine, positions)

        assert len(shots) == 1
        shot = shots[0][1]
        json.dumps(asdict(shot))
        assert all(type(v) is float for v in shot.start_pos_in + shot.end_pos_in + shot.target_pos_in)
