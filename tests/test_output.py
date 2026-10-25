import pytest
import json
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.output import SessionLogger, OverlayRenderer, ShotRecord, REQUIRED_LOG_FIELDS
import numpy as np


@pytest.fixture
def temp_log_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_shot_record():
    return ShotRecord(
        timestamp_iso="2026-05-11T12:00:00Z",
        frame_start=0,
        frame_end=50,
        angle_deg=45.0,
        start_pos_in=(44.0, 22.0),
        end_pos_in=(50.0, 25.0),
        accuracy_in=2.5,
        target_pos_in=(52.0, 28.0),
    )


class TestSessionLogger:
    def test_log_shot_writes_json_line(self, temp_log_file, sample_shot_record):
        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.close()

        with open(temp_log_file, 'r') as f:
            line = f.readline()
        
        assert line, "Log file should contain at least one line"
        data = json.loads(line)
        assert data['timestamp_iso'] == sample_shot_record.timestamp_iso
        assert 'peak_speed_mph' not in data

    def test_log_shot_three_times(self, temp_log_file, sample_shot_record):
        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.log_shot(sample_shot_record)
        logger.log_shot(sample_shot_record)
        logger.close()

        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3, "Should have exactly 3 lines"
        for line in lines:
            data = json.loads(line)
            assert isinstance(data, dict)

    def test_all_required_fields_present(self, temp_log_file, sample_shot_record):
        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.close()

        with open(temp_log_file, 'r') as f:
            line = f.readline()
        
        data = json.loads(line)
        for field in REQUIRED_LOG_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_append_mode(self, temp_log_file, sample_shot_record):
        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.close()

        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.log_shot(sample_shot_record)
        logger.log_shot(sample_shot_record)
        logger.close()

        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4, "Should have 4 total lines after reopening"
        for line in lines:
            data = json.loads(line)
            assert all(field in data for field in REQUIRED_LOG_FIELDS)

    def test_flush(self, temp_log_file, sample_shot_record):
        logger = SessionLogger(temp_log_file)
        logger.log_shot(sample_shot_record)
        logger.flush()

        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1, "Should have 1 line after flush"
        logger.close()


class TestOverlayRenderer:
    def test_render_disabled_returns_unmodified_frame(self):
        renderer = OverlayRenderer(enabled=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = renderer.render(frame, [], [])
        assert np.array_equal(result, frame)

    def test_render_enabled_returns_modified_frame(self):
        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = renderer.render(frame, [], [])
        assert result is not None
        assert result.shape == frame.shape

    def test_render_with_tracked_balls(self):
        from dataclasses import dataclass
        
        @dataclass
        class MockBall:
            center_px: tuple[float, float]
            is_cue_ball: bool
            radius_px: float = 10.0

        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ball = MockBall(center_px=(320, 240), is_cue_ball=True, radius_px=15.0)
        result = renderer.render(frame, [ball], [])
        assert result is not None
        assert result.shape == frame.shape

    def test_render_with_trail(self):
        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail = [(100, 100), (110, 110), (120, 120)]
        result = renderer.render(frame, [], trail)
        assert result is not None
        assert result.shape == frame.shape

    def test_smoothing_reduces_jitter(self):
        renderer = OverlayRenderer(enabled=True)
        rng = np.random.default_rng(0)
        raw = [(300 + rng.uniform(-3, 3), 200 + rng.uniform(-3, 3)) for _ in range(60)]
        drawn = [renderer._smooth_center(7, c) for c in raw]
        raw_std = np.std([c[0] for c in raw[10:]])
        drawn_std = np.std([c[0] for c in drawn[10:]])
        assert drawn_std <= 0.5 * raw_std

    def test_smoothing_snaps_on_large_move(self):
        renderer = OverlayRenderer(enabled=True)
        renderer._smooth_center(7, (100.0, 100.0))
        assert renderer._smooth_center(7, (200.0, 100.0)) == (200.0, 100.0)

    def test_smoothing_forgets_dropped_tracks(self):
        from dataclasses import dataclass

        @dataclass
        class MockBall:
            id: int
            center_px: tuple
            is_cue_ball: bool = False

        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.render(frame, [MockBall(1, (100, 100)), MockBall(2, (300, 300))], [])
        renderer.render(frame, [MockBall(2, (300, 300))], [])
        assert set(renderer._smoothed_centers) == {2}

    def test_toggle(self):
        renderer = OverlayRenderer(enabled=True)
        assert renderer.enabled is True
        renderer.toggle()
        assert renderer.enabled is False
        renderer.toggle()
        assert renderer.enabled is True
