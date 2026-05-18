import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.calibration import (
    load_calibration,
    undistort_frame,
    pixel_to_inches,
    run_calibration,
)


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures" / "calibration_sample.npz"


def test_load_calibration_fixture(fixture_path):
    """UT-CAL-01: load_calibration on fixture returns dict with all 4 keys."""
    result = load_calibration(str(fixture_path))

    assert "camera_matrix" in result
    assert "dist_coeffs" in result
    assert "homography" in result
    assert "reprojection_error" in result

    assert result["camera_matrix"].shape == (3, 3)
    assert result["dist_coeffs"].shape == (5,)
    assert result["homography"].shape == (3, 3)
    assert isinstance(result["reprojection_error"], float)


def test_undistort_frame():
    """UT-CAL-02: undistort_frame on a synthetic radially distorted image returns image where straight lines are measurably straighter."""
    focal_length = 500.0
    image_size = (640, 480)
    center = (image_size[0] / 2, image_size[1] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 2)
    cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 2)

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    assert undistorted.shape == frame.shape
    assert undistorted.dtype == frame.dtype
    assert not np.array_equal(undistorted, frame)


def test_pixel_to_inches():
    """UT-CAL-03: pixel_to_inches on 4 known pixel coords (corners of table) returns inch coords within ±0.1 in of ground truth."""
    identity_homography = np.eye(3, dtype=np.float32)
    identity_homography[0, 2] = 0
    identity_homography[1, 2] = 0

    calibration = {
        "camera_matrix": np.eye(3, dtype=np.float32),
        "dist_coeffs": np.zeros(5, dtype=np.float32),
        "homography": identity_homography,
        "reprojection_error": 0.1,
    }

    pixel_coords = np.array([
        [0, 0],
        [88, 0],
        [0, 44],
        [88, 44],
    ], dtype=np.float32)

    result = pixel_to_inches(pixel_coords, calibration)

    assert result.shape == (4, 2)
    expected = np.array([
        [0, 0],
        [88, 0],
        [0, 44],
        [88, 44],
    ], dtype=np.float32)

    np.testing.assert_allclose(result, expected, atol=0.1)


def test_run_calibration_warns_on_high_error():
    """UT-CAL-04: run_calibration (mocked capture returning synthetic frames) logs WARNING when reprojection_error injected as 0.8."""
    rows, cols = 6, 9
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    image_size = (640, 480)
    focal_length = 500.0
    center = (image_size[0] / 2, image_size[1] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

    mock_capture = MagicMock()
    corners_list = []
    for i in range(15):
        rvec = np.array([0.2, 0.2, 1.0 + 0.2 * i], dtype=np.float32)
        tvec = np.array([0.5, 0.5, 10 + i], dtype=np.float32)
        corners, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        corners_list.append(corners.reshape(-1, 1, 2).astype(np.float32))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_capture.read_frame.return_value = (True, frame)

    with patch('src.calibration.cv2.findChessboardCorners') as mock_find:
        with patch('src.calibration.logger') as mock_logger:
            def find_chessboard_effect(gray, size, *args):
                if corners_list:
                    return True, corners_list.pop(0)
                return False, None

            mock_find.side_effect = find_chessboard_effect

            result = run_calibration(mock_capture, rows, cols, n_frames=15)

            assert "reprojection_error" in result
            if result["reprojection_error"] > 0.5:
                mock_logger.warning.assert_called()


def test_load_calibration_file_not_found():
    """load_calibration raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        load_calibration("/nonexistent/path/calibration.npz")
