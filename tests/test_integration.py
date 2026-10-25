import pytest
import sys
import os
import json
import tempfile
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calibration import load_calibration, undistort_frame, pixel_to_inches
from src.detection import detect_balls, Detection, compute_expected_radius_px
from src.tracking import BallTracker, TrackedBall
from src.metrics import MetricsEngine, ShotRecord
from src.output import SessionLogger, OverlayRenderer
from src.config_loader import load_config
from src.constants import BALL_DIAMETER_IN


def _sanitize_shot_record(shot: ShotRecord) -> ShotRecord:
    def to_python_float(val):
        if isinstance(val, np.ndarray):
            return float(val)
        elif hasattr(val, 'item'):
            return val.item()
        return val

    def to_python_tuple(t):
        if t is None:
            return None
        return tuple(to_python_float(x) for x in t)

    return ShotRecord(
        timestamp_iso=shot.timestamp_iso,
        frame_start=int(shot.frame_start),
        frame_end=int(shot.frame_end),
        angle_deg=to_python_float(shot.angle_deg),
        start_pos_in=to_python_tuple(shot.start_pos_in),
        end_pos_in=to_python_tuple(shot.end_pos_in),
        accuracy_in=to_python_float(shot.accuracy_in) if shot.accuracy_in is not None else None,
        target_pos_in=to_python_tuple(shot.target_pos_in),
    )


@dataclass
class MockCameraCapture:
    frames: list
    frame_index: int

    def read_frame(self):
        if self.frame_index >= len(self.frames):
            return (False, None)
        frame = self.frames[self.frame_index]
        self.frame_index += 1
        return (True, frame)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def create_synthetic_frame(width=640, height=480):
    green_felt = np.array([40, 120, 60], dtype=np.uint8)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = green_felt
    return frame


def draw_ball_on_frame(frame, x, y, radius=20, is_cue_ball=True):
    if is_cue_ball:
        color = (255, 255, 255)
    else:
        color = (0, 255, 255)
    cv2.circle(frame, (int(x), int(y)), int(radius), color, -1)


@pytest.fixture
def config():
    cfg = load_config("config/config.yaml")
    cfg['detection']['hough_param2'] = 20
    cfg['detection']['hsv_ranges']['cue_ball'] = [[0, 0, 200], [180, 30, 255]]
    cfg['detection'].pop('expected_radius_px', None)
    return cfg


@pytest.fixture
def calibration():
    return load_calibration("tests/fixtures/calibration_sample.npz")


@pytest.fixture
def expected_radius_px(calibration):
    return compute_expected_radius_px(calibration)


class TestIT_SYN_01:
    def test_stationary_cue_ball_no_shot(self, config, calibration, expected_radius_px):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_path = f.name

        try:
            frames = []
            for i in range(60):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, 320, 240, expected_radius_px, is_cue_ball=True)
                frames.append(frame)

            mock_cap = MockCameraCapture(frames, 0)

            tracker = BallTracker(method=config['tracking']['method'])
            engine = MetricsEngine(calibration, expected_radius_px)
            logger = SessionLogger(temp_log_path, config['output']['flush_interval_s'])

            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]
                undistorted = undistort_frame(frame, calibration)
                detections = detect_balls(undistorted, config['detection'], expected_radius_px)
                tracked = tracker.update(detections)
                cb = tracker.cue_ball()

                shot = engine.update(cb, frame_idx, frame_idx / 60.0)
                if shot:
                    logger.log_shot(shot)

            logger.close()

            with open(temp_log_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 0, f"Expected 0 log lines, got {len(lines)}"

        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)


class TestIT_SYN_02:
    def test_single_shot(self, config, calibration, expected_radius_px):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_path = f.name

        try:
            frames = []
            positions = []

            start_x, start_y = 200.0, 240.0

            for i in range(5):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, start_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((start_x, start_y))

            pixels_per_frame = 12.0
            current_x = start_x
            for i in range(20):
                current_x += pixels_per_frame
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            for i in range(40):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            mock_cap = MockCameraCapture(frames, 0)

            tracker = BallTracker(method=config['tracking']['method'])
            engine = MetricsEngine(calibration, expected_radius_px)
            logger = SessionLogger(temp_log_path, config['output']['flush_interval_s'])

            shot_records = []
            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]
                undistorted = undistort_frame(frame, calibration)
                detections = detect_balls(undistorted, config['detection'], expected_radius_px)
                tracked = tracker.update(detections)
                cb = tracker.cue_ball()

                shot = engine.update(cb, frame_idx, frame_idx / 60.0)
                if shot:
                    shot = _sanitize_shot_record(shot)
                    logger.log_shot(shot)
                    shot_records.append(shot)

            logger.close()

            assert len(shot_records) == 1, f"Expected 1 ShotRecord, got {len(shot_records)}"
            assert shot_records[0].frame_start < shot_records[0].frame_end

            with open(temp_log_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}"
            data = json.loads(lines[0])
            assert 'peak_speed_mph' not in data

        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)


class TestIT_SYN_03:
    def test_two_sequential_shots(self, config, calibration, expected_radius_px):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_path = f.name

        try:
            frames = []
            positions = []

            start_x, start_y = 200.0, 240.0

            for i in range(5):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, start_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((start_x, start_y))

            pixels_per_frame = 12.0
            current_x = start_x
            for i in range(15):
                current_x += pixels_per_frame
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            for i in range(30):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            first_shot_end_x = current_x

            for i in range(10):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            pixels_per_frame = 12.0
            for i in range(15):
                current_x += pixels_per_frame
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            for i in range(40):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            mock_cap = MockCameraCapture(frames, 0)

            tracker = BallTracker(method=config['tracking']['method'])
            engine = MetricsEngine(calibration, expected_radius_px)
            logger = SessionLogger(temp_log_path, config['output']['flush_interval_s'])

            shot_records = []
            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]
                undistorted = undistort_frame(frame, calibration)
                detections = detect_balls(undistorted, config['detection'], expected_radius_px)
                tracked = tracker.update(detections)
                cb = tracker.cue_ball()

                shot = engine.update(cb, frame_idx, frame_idx / 60.0)
                if shot:
                    shot = _sanitize_shot_record(shot)
                    logger.log_shot(shot)
                    shot_records.append(shot)

            logger.close()

            assert len(shot_records) == 2, f"Expected 2 ShotRecords, got {len(shot_records)}"

            with open(temp_log_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"

            for line in lines:
                data = json.loads(line)
                assert 'frame_start' in data
                assert 'frame_end' in data

        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)


class TestIT_SYN_04:
    def test_shot_with_target_set(self, config, calibration, expected_radius_px):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_path = f.name

        try:
            frames = []
            positions = []

            start_x, start_y = 200.0, 240.0

            for i in range(5):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, start_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((start_x, start_y))

            pixels_per_frame = 20.0
            current_x = start_x
            for i in range(20):
                current_x += pixels_per_frame
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            for i in range(40):
                frame = create_synthetic_frame()
                draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
                frames.append(frame)
                positions.append((current_x, start_y))

            mock_cap = MockCameraCapture(frames, 0)

            tracker = BallTracker(method=config['tracking']['method'])
            engine = MetricsEngine(calibration, expected_radius_px)
            logger = SessionLogger(temp_log_path, config['output']['flush_interval_s'])

            target_pos_in = (50.0, 25.0)
            engine.set_target(target_pos_in)

            shot_records = []
            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]
                undistorted = undistort_frame(frame, calibration)
                detections = detect_balls(undistorted, config['detection'], expected_radius_px)
                tracked = tracker.update(detections)
                cb = tracker.cue_ball()

                shot = engine.update(cb, frame_idx, frame_idx / 60.0)
                if shot:
                    shot = _sanitize_shot_record(shot)
                    logger.log_shot(shot)
                    shot_records.append(shot)

            logger.close()

            assert len(shot_records) == 1, f"Expected 1 ShotRecord, got {len(shot_records)}"
            assert shot_records[0].target_pos_in is not None, "Expected target_pos_in to be set"
            assert shot_records[0].accuracy_in is not None, "Expected accuracy_in to be not None"

        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)


class TestIT_SYN_05:
    def test_overlay_toggle(self, config, calibration, expected_radius_px):
        frames = []
        positions = []

        start_x, start_y = 200.0, 240.0

        for i in range(5):
            frame = create_synthetic_frame()
            draw_ball_on_frame(frame, start_x, start_y, expected_radius_px, is_cue_ball=True)
            frames.append(frame)
            positions.append((start_x, start_y))

        pixels_per_frame = 20.0
        current_x = start_x
        for i in range(40):
            current_x += pixels_per_frame
            frame = create_synthetic_frame()
            draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
            frames.append(frame)
            positions.append((current_x, start_y))

        for i in range(5):
            frame = create_synthetic_frame()
            draw_ball_on_frame(frame, current_x, start_y, expected_radius_px, is_cue_ball=True)
            frames.append(frame)
            positions.append((current_x, start_y))

        mock_cap = MockCameraCapture(frames, 0)

        tracker = BallTracker(method=config['tracking']['method'])
        engine = MetricsEngine(calibration, expected_radius_px)
        renderer = OverlayRenderer(enabled=True)

        trail = deque(maxlen=30)

        exception_raised = False
        try:
            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]
                undistorted = undistort_frame(frame, calibration)
                detections = detect_balls(undistorted, config['detection'], expected_radius_px)
                tracked = tracker.update(detections)
                cb = tracker.cue_ball()

                if cb:
                    trail.append(cb.center_px)

                shot = engine.update(cb, frame_idx, frame_idx / 60.0)

                if frame_idx == 30:
                    renderer.toggle()

                annotated = renderer.render(
                    undistorted.copy(),
                    tracked,
                    list(trail),
                )

                assert annotated is not None, f"Rendered frame is None at frame {frame_idx}"
                assert isinstance(annotated, np.ndarray), f"Rendered frame is not ndarray at frame {frame_idx}"
                assert annotated.shape == undistorted.shape, \
                    f"Rendered frame shape mismatch at frame {frame_idx}: {annotated.shape} vs {undistorted.shape}"
        except Exception as e:
            exception_raised = True
            raise

        assert not exception_raised, "Exception raised during overlay toggle test"
