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
        peak_speed_mph=10.5,
        mean_speed_mph=8.2,
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
        assert data['peak_speed_mph'] == sample_shot_record.peak_speed_mph

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
        result = renderer.render(frame, [], 0.0, None, [])
        assert np.array_equal(result, frame)

    def test_render_enabled_returns_modified_frame(self):
        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = renderer.render(frame, [], 5.0, None, [])
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
        result = renderer.render(frame, [ball], 5.0, None, [])
        assert result is not None
        assert result.shape == frame.shape

    def test_render_with_trail(self):
        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail = [(100, 100), (110, 110), (120, 120)]
        result = renderer.render(frame, [], 5.0, None, trail)
        assert result is not None
        assert result.shape == frame.shape

    def test_render_with_last_shot(self):
        renderer = OverlayRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        shot = ShotRecord(
            timestamp_iso="2026-05-11T12:00:00Z",
            frame_start=0,
            frame_end=50,
            peak_speed_mph=10.5,
            mean_speed_mph=8.2,
            angle_deg=45.0,
            start_pos_in=(44.0, 22.0),
            end_pos_in=(50.0, 25.0),
            accuracy_in=2.5,
            target_pos_in=(52.0, 28.0),
        )
        result = renderer.render(frame, [], 5.0, shot, [])
        assert result is not None
        assert result.shape == frame.shape

    def test_toggle(self):
        renderer = OverlayRenderer(enabled=True)
        assert renderer.enabled is True
        renderer.toggle()
        assert renderer.enabled is False
        renderer.toggle()
        assert renderer.enabled is True
