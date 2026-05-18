import pytest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session", autouse=True)
def create_calibration_fixture():
    """Create synthetic calibration fixture for testing."""
    fixture_path = Path(__file__).parent / "fixtures" / "calibration_sample.npz"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = 6, 9
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    object_points = []
    image_points = []

    image_size = (640, 480)
    focal_length = 500.0
    center = (image_size[0] / 2, image_size[1] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([0.01, -0.02, 0, 0, 0], dtype=np.float32)

    for i in range(5):
        rvec = np.array([0.1, 0.1, 0.5 + 0.1 * i], dtype=np.float32)
        tvec = np.array([0, 0, 10 + i], dtype=np.float32)

        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        object_points.append(objp)
        image_points.append(projected.reshape(-1, 2).astype(np.float32))

    ret, cal_mat, cal_dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    total_error = 0.0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], cal_mat, cal_dist)
        projected_reshaped = projected.reshape(-1, 2).astype(np.float32)
        error = cv2.norm(image_points[i], projected_reshaped, cv2.NORM_L2) / len(projected_reshaped)
        total_error += error

    reprojection_error = total_error / len(object_points)

    table_corners_world = np.array([
        [0, 0, 0],
        [88.0, 0, 0],
        [0, 44.0, 0],
        [88.0, 44.0, 0],
    ], dtype=np.float32)

    table_corners_image, _ = cv2.projectPoints(
        table_corners_world, rvecs[0], tvecs[0], cal_mat, cal_dist
    )
    table_corners_image = table_corners_image.reshape(-1, 2).astype(np.float32)

    homography, _ = cv2.findHomography(
        table_corners_image,
        table_corners_world[:, :2].astype(np.float32)
    )

    np.savez(
        fixture_path,
        camera_matrix=cal_mat,
        dist_coeffs=cal_dist.reshape(-1),
        homography=homography,
        reprojection_error=reprojection_error,
    )

    yield fixture_path


@pytest.fixture(scope="session", autouse=True)
def create_detection_fixtures():
    """Create synthetic test images for detection tests."""
    fixture_dir = Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    green_felt = np.array([40, 120, 60], dtype=np.uint8)

    def create_blank_table():
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = green_felt
        return img

    def draw_circle(img, x, y, r, color):
        cv2.circle(img, (int(x), int(y)), int(r), color, -1)

    single_white = create_blank_table()
    draw_circle(single_white, 320, 240, 40, (255, 255, 255))
    cv2.imwrite(str(fixture_dir / "fixture_single_white_ball.png"), single_white)

    balls_16 = create_blank_table()
    colors = [
        (255, 255, 255),
        (180, 255, 150),
        (150, 200, 50),
        (100, 100, 255),
        (255, 50, 100),
        (255, 255, 100),
        (150, 150, 255),
        (255, 100, 100),
        (100, 255, 100),
        (200, 100, 100),
        (100, 150, 255),
        (255, 100, 150),
        (100, 200, 100),
        (200, 100, 200),
        (100, 200, 200),
        (255, 150, 100),
    ]
    positions = [
        (80, 80),
        (150, 80),
        (220, 80),
        (290, 80),
        (360, 80),
        (430, 80),
        (500, 80),
        (570, 80),
        (80, 180),
        (150, 180),
        (220, 180),
        (290, 180),
        (360, 180),
        (430, 180),
        (500, 180),
        (570, 180),
    ]
    for (x, y), color in zip(positions, colors):
        draw_circle(balls_16, x, y, 40, color)
    cv2.imwrite(str(fixture_dir / "fixture_16_balls.png"), balls_16)

    empty = create_blank_table()
    cv2.imwrite(str(fixture_dir / "fixture_empty_table.png"), empty)

    near_rail = create_blank_table()
    near_rail_positions = [(40, 40), (600, 40), (40, 440), (600, 440)]
    near_rail_colors = [(255, 255, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]
    for (x, y), color in zip(near_rail_positions, near_rail_colors):
        draw_circle(near_rail, x, y, 35, color)
    cv2.imwrite(str(fixture_dir / "fixture_near_rail.png"), near_rail)

    noise = create_blank_table()
    draw_circle(noise, 320, 240, 8, (200, 200, 200))
    draw_circle(noise, 400, 300, 60, (255, 255, 255))
    cv2.imwrite(str(fixture_dir / "fixture_noise.png"), noise)

    yield fixture_dir
