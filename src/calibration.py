import cv2
import numpy as np
import logging
from src.constants import TABLE_WIDTH_IN, TABLE_HEIGHT_IN, BALL_DIAMETER_IN

logger = logging.getLogger(__name__)


def run_calibration(
    capture,
    rows: int,
    cols: int,
    n_frames: int = 15,
    output_path: str = "config/calibration.npz",
) -> dict:
    """
    Capture checkerboard frames interactively, compute camera matrix + distortion
    coefficients, compute homography from table corners, save to output_path.
    Returns: {
        "camera_matrix": np.ndarray,      # 3x3
        "dist_coeffs": np.ndarray,        # 1x5
        "homography": np.ndarray,         # 3x3  pixel -> real-world inches
        "reprojection_error": float,
    }
    Logs WARNING if reprojection_error > 0.5.
    Raises ValueError if fewer than 10 valid checkerboard frames found.
    """
    object_points = []
    image_points = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    frames_found = 0
    frame_count = 0

    logger.info(f"Starting calibration capture: need {n_frames} valid checkerboard frames (rows={rows}, cols={cols})")

    while frames_found < n_frames:
        ret, frame = capture.read_frame()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_find, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret_find:
            try:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except cv2.error:
                corners_refined = corners
            object_points.append(objp)
            image_points.append(corners_refined)
            frames_found += 1
            logger.info(f"Found checkerboard {frames_found}/{n_frames}")

    if frames_found < 10:
        raise ValueError(f"Only found {frames_found} valid checkerboard frames; need at least 10")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    total_error = 0.0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error

    reprojection_error = total_error / len(object_points)

    if reprojection_error > 0.5:
        logger.warning(f"High reprojection error: {reprojection_error:.4f}")

    table_corners_world = np.array([
        [0, 0, 0],
        [TABLE_WIDTH_IN, 0, 0],
        [0, TABLE_HEIGHT_IN, 0],
        [TABLE_WIDTH_IN, TABLE_HEIGHT_IN, 0],
    ], dtype=np.float32)

    table_corners_image, _ = cv2.projectPoints(
        table_corners_world, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
    )
    table_corners_image = table_corners_image.reshape(-1, 2).astype(np.float32)

    homography, _ = cv2.findHomography(
        table_corners_image,
        table_corners_world[:, :2].astype(np.float32)
    )

    result = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.reshape(-1),
        "homography": homography,
        "reprojection_error": float(reprojection_error),
    }

    np.savez(output_path, **result)
    logger.info(f"Calibration saved to {output_path}")

    return result


def load_calibration(path: str) -> dict:
    """Load .npz file. Return same dict shape as run_calibration(). Raise FileNotFoundError if missing."""
    try:
        data = np.load(path)
        return {
            "camera_matrix": data["camera_matrix"],
            "dist_coeffs": data["dist_coeffs"],
            "homography": data["homography"],
            "reprojection_error": float(data["reprojection_error"]),
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration file not found: {path}")


def undistort_frame(frame: np.ndarray, calibration: dict) -> np.ndarray:
    """Apply camera_matrix + dist_coeffs. Return corrected frame (same shape)."""
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def pixel_to_inches(
    pixel_coords: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    """Apply homography. Return (N, 2) array of real-world inch coordinates."""
    homography = calibration["homography"]
    ones = np.ones((pixel_coords.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([pixel_coords.astype(np.float32), ones])

    transformed = homography @ homogeneous.T
    result = transformed[:2, :].T / transformed[2, :, np.newaxis]

    return result.astype(np.float32)
