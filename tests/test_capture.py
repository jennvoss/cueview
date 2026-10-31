import pytest
from unittest.mock import patch, MagicMock
import json
from src.capture import (
    list_video_devices,
    find_logitech_camera,
    find_camera_index,
    CameraCapture,
    CameraNotFoundError,
    StillImageSource,
)


class TestListVideoDevices:
    def test_ut_cap_01_list_video_devices_with_logitech(self):
        """UT-CAP-01: list_video_devices returns Logitech device correctly."""
        mock_output = json.dumps({
            "SPCameraDataType": [
                {
                    "_items": [
                        {
                            "_name": "Logi C920 HD Pro Webcam",
                            "usb_vendor_id": "046d",
                        },
                        {
                            "_name": "FaceTime HD Camera",
                            "usb_vendor_id": "0000",
                        },
                    ]
                }
            ]
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
            )

            devices = list_video_devices()

            assert len(devices) == 2
            assert devices[0]["is_logitech"] is True
            assert devices[1]["is_logitech"] is False


class TestFindLogitechCamera:
    def test_ut_cap_02_find_logitech_camera_returns_index(self):
        """UT-CAP-02: find_logitech_camera returns the correct index."""
        mock_devices = [
            {"index": 0, "name": "FaceTime", "vendor_id": "0000", "is_logitech": False},
            {"index": 1, "name": "Logi C920", "vendor_id": "046d", "is_logitech": True},
        ]

        with patch("src.capture.list_video_devices", return_value=mock_devices):
            result = find_logitech_camera()
            assert result == 1

    def test_ut_cap_03_find_logitech_camera_no_logitech_raises(self):
        """UT-CAP-03: find_logitech_camera raises when only FaceTime present."""
        mock_devices = [
            {"index": 0, "name": "FaceTime HD Camera", "vendor_id": "0000", "is_logitech": False},
        ]

        with patch("src.capture.list_video_devices", return_value=mock_devices):
            with pytest.raises(CameraNotFoundError) as exc_info:
                find_logitech_camera()
            assert "FaceTime HD Camera" in str(exc_info.value)

    def test_ut_cap_04_find_logitech_camera_empty_list_raises(self):
        """UT-CAP-04: find_logitech_camera raises when device list is empty."""
        with patch("src.capture.list_video_devices", return_value=[]):
            with pytest.raises(CameraNotFoundError) as exc_info:
                find_logitech_camera()
            assert "No devices found" in str(exc_info.value)


class TestFindCameraIndex:
    def test_ut_cap_05_find_camera_index_require_logitech_false(self):
        """UT-CAP-05: find_camera_index with require_logitech=False returns config index."""
        cfg = {
            "camera": {
                "require_logitech": False,
                "index": 2,
            }
        }

        with patch("src.capture.find_logitech_camera") as mock_find:
            result = find_camera_index(cfg)
            assert result == 2
            mock_find.assert_not_called()


class TestCameraCapture:
    def test_ut_cap_06_open_camera_not_found_raises(self):
        """UT-CAP-06: CameraCapture.open raises CameraNotFoundError when isOpened is False."""
        with patch("cv2.VideoCapture") as mock_cv2_cap:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2_cap.return_value = mock_cap

            cap = CameraCapture(0, 1920, 1080, 60)

            with pytest.raises(CameraNotFoundError):
                cap.open()

    def test_ut_cap_07_actual_fps_returns_fps(self):
        """UT-CAP-07: CameraCapture.actual_fps returns fps from device."""
        with patch("cv2.VideoCapture") as mock_cv2_cap:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 60.0
            mock_cv2_cap.return_value = mock_cap

            cap = CameraCapture(0, 1920, 1080, 60)
            cap.open()

            fps = cap.actual_fps()
            assert fps == 60.0

    def test_ut_cap_08_context_manager_calls_release(self):
        """UT-CAP-08: CameraCapture.__exit__ calls release."""
        with patch("cv2.VideoCapture") as mock_cv2_cap:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2_cap.return_value = mock_cap

            with CameraCapture(0, 1920, 1080, 60) as cap:
                pass

            mock_cap.release.assert_called_once()


class TestStillImageSource:
    @pytest.fixture
    def image_path(self, tmp_path):
        import cv2
        import numpy as np
        path = tmp_path / "still.png"
        img = np.zeros((72, 128, 3), dtype=np.uint8)
        img[10:20, 10:20] = (0, 0, 255)
        cv2.imwrite(str(path), img)
        return path

    def test_serves_same_image_every_frame(self, image_path):
        with StillImageSource(image_path, fps=1000) as source:
            ok1, f1 = source.read_frame()
            ok2, f2 = source.read_frame()
        assert ok1 and ok2
        assert f1.shape == (72, 128, 3)
        assert (f1 == f2).all()
        f1[:] = 0  # callers get a copy, not the stored image
        assert f2[15, 15, 2] == 255

    def test_paces_frames_to_fps(self, image_path):
        import time
        with StillImageSource(image_path, fps=20) as source:
            start = time.monotonic()
            for _ in range(5):
                source.read_frame()
            elapsed = time.monotonic() - start
        assert elapsed >= 4 / 20 - 0.01  # first frame is immediate

    def test_reports_fps_and_capture_time(self, image_path):
        import os
        os.utime(image_path, (1761400000, 1761400000))
        with StillImageSource(image_path, fps=5) as source:
            assert source.actual_fps() == 5

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            StillImageSource(tmp_path / "missing.png").open()
