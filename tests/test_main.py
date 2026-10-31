import argparse
import logging

import cv2
import numpy as np
import pytest

import src.main as main
from src.capture import CameraCapture, CameraNotFoundError, StillImageSource

CFG = {"camera": {"width": 128, "height": 72, "fps": 15}}
LOG = logging.getLogger("test_main")


def _write_image(path):
    cv2.imwrite(str(path), np.zeros((72, 128, 3), dtype=np.uint8))
    return path


def _args(image=None):
    return argparse.Namespace(image=image)


@pytest.fixture
def no_camera(monkeypatch):
    def raise_not_found(cfg):
        raise CameraNotFoundError("No Logitech camera detected.")
    monkeypatch.setattr(main, "find_camera_index", raise_not_found)


def test_find_still_image_prefers_raw_frame(tmp_path):
    assert main.find_still_image(tmp_path) is None
    _write_image(tmp_path / "latest.jpg")
    assert main.find_still_image(tmp_path).name == "latest.jpg"
    _write_image(tmp_path / main.RAW_FRAME_NAME)
    assert main.find_still_image(tmp_path).name == main.RAW_FRAME_NAME


def test_no_camera_falls_back_to_raw_frame(tmp_path, no_camera):
    _write_image(tmp_path / main.RAW_FRAME_NAME)
    source, info = main.open_frame_source(_args(), CFG, tmp_path, LOG)
    assert isinstance(source, StillImageSource)
    assert info["kind"] == "image"
    assert info["annotated"] is False
    assert info["path"].endswith(main.RAW_FRAME_NAME)


def test_no_camera_falls_back_to_annotated_frame_with_flag(tmp_path, no_camera):
    _write_image(tmp_path / "latest.jpg")
    _, info = main.open_frame_source(_args(), CFG, tmp_path, LOG)
    assert info["annotated"] is True


def test_no_camera_and_no_saved_frame_exits(tmp_path, no_camera):
    with pytest.raises(SystemExit):
        main.open_frame_source(_args(), CFG, tmp_path, LOG)


def test_image_flag_skips_camera(tmp_path, monkeypatch):
    def fail(cfg):
        raise AssertionError("camera should not be opened with --image")
    monkeypatch.setattr(main, "find_camera_index", fail)
    path = _write_image(tmp_path / "table.png")
    source, info = main.open_frame_source(_args(str(path)), CFG, tmp_path, LOG)
    assert isinstance(source, StillImageSource)
    assert info["path"] == str(path)


def test_camera_used_when_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "find_camera_index", lambda cfg: 0)
    monkeypatch.setattr(CameraCapture, "open", lambda self: None)
    source, info = main.open_frame_source(_args(), CFG, tmp_path, LOG)
    assert isinstance(source, CameraCapture)
    assert info == {"kind": "camera"}


def test_status_writer_saves_raw_frame_only_when_enabled(tmp_path):
    from src.metrics import MetricsEngine
    engine = MetricsEngine(main.identity_calibration(), 12.0)
    frame = np.full((72, 128, 3), 50, dtype=np.uint8)
    for enabled in (False, True):
        writer = main.StatusWriter(tmp_path, 1, False, main.identity_calibration(), 15,
                                   save_raw_frames=enabled)
        writer.write(0, [], [], engine, None, None, raw_frame=frame)
        assert (tmp_path / main.RAW_FRAME_NAME).exists() is enabled


def test_image_run_shows_unmarked_image_on_dashboard(tmp_path, monkeypatch):
    """Running with --image writes the image itself (no detection markers) to latest.jpg."""
    import json
    import sys
    from pathlib import Path

    src = Path(__file__).parent / "fixtures" / "table" / "12_midgame.png"
    keys = iter([-1] * 11 + [ord("q")])  # 12 frames, then quit
    monkeypatch.setattr(main.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *a: next(keys))
    monkeypatch.setattr(main, "STILL_IMAGE_FPS", 1000.0)
    monkeypatch.setattr(sys, "argv", ["src.main", "--image", str(src),
                                      "--web-data-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    main.main()

    shown = cv2.imread(str(tmp_path / "latest.jpg")).astype(int)
    original = cv2.imread(str(src)).astype(int)
    assert np.abs(shown - original).mean() < 3  # same picture, only JPEG noise
    overlay_green = (shown[..., 1] > 200) & (shown[..., 0] < 60) & (shown[..., 2] < 60)
    assert overlay_green.sum() == 0  # no detection rings drawn on it

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["source"]["kind"] == "image"
    assert status["score"]["solids_sunk"] == []  # 12 frames is well under SINK_AFTER_S
