import cv2
import subprocess
import re
import logging
import json
import numpy as np
from src.constants import FPS_TARGET, FPS_MINIMUM

logger = logging.getLogger(__name__)

LOGITECH_USB_VENDOR_ID = "046d"


class CameraNotFoundError(RuntimeError):
    """Raised when no Logitech USB camera can be found or opened."""
    pass


def list_video_devices() -> list[dict]:
    """
    Enumerate all video capture devices visible to the OS.

    On macOS, shells out to system_profiler to get camera info.
    Returns list of dicts with keys: index, name, vendor_id, is_logitech.
    """
    devices = []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("system_profiler unavailable; returning empty device list")
        return []

    if result.returncode != 0:
        logger.warning("system_profiler failed; returning empty device list")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse system_profiler JSON output")
        return []

    cameras_list = data.get("SPCameraDataType", [])
    if not cameras_list:
        return []

    if isinstance(cameras_list[0], dict) and "_items" in cameras_list[0]:
        camera_entries = cameras_list[0].get("_items", [])
    else:
        camera_entries = cameras_list

    for idx, camera in enumerate(camera_entries):
        name = camera.get("_name", f"Camera {idx}")
        vendor_id = (camera.get("usb_vendor_id") or "").lower()

        if not vendor_id:
            model_id = camera.get("spcamera_model-id", "")
            m = re.search(r"VendorID_(\d+)", model_id)
            if m:
                try:
                    vendor_id = f"{int(m.group(1)):04x}"
                except ValueError:
                    pass

        if not vendor_id:
            unique_id = camera.get("spcamera_unique-id", "")
            m = re.search(r"([0-9a-fA-F]{4})([0-9a-fA-F]{4})$", unique_id)
            if m:
                vendor_id = m.group(1).lower()

        if not vendor_id:
            lname = name.lower()
            if any(tag in lname for tag in ("logitech", "logi ", "c920", "c922", "c925", "c930", "brio")):
                vendor_id = LOGITECH_USB_VENDOR_ID

        is_logitech = (vendor_id == LOGITECH_USB_VENDOR_ID)

        device = {
            "index": idx,
            "name": name,
            "vendor_id": vendor_id,
            "is_logitech": is_logitech,
        }
        devices.append(device)

    return devices


def find_logitech_camera() -> int:
    """
    Return the OpenCV VideoCapture index of the first Logitech USB camera found.

    Raises CameraNotFoundError if no Logitech device is found.
    """
    devices = list_video_devices()
    logitech_devices = [d for d in devices if d["is_logitech"]]

    if logitech_devices:
        device = logitech_devices[0]
        logger.info(f"Logitech camera found: {device['name']} at index {device['index']}")
        return device["index"]

    device_names = ", ".join([d["name"] for d in devices]) if devices else "No devices found"
    raise CameraNotFoundError(
        f"No Logitech camera detected. Ensure the camera is plugged in via USB and "
        f"recognised by macOS (check System Settings > Privacy & Security > Camera). "
        f"Connected devices: {device_names}"
    )


def find_camera_index(cfg: dict) -> int:
    """
    Resolve the camera index to use, respecting config.

    If cfg['camera']['require_logitech'] is True, find a Logitech camera.
    Otherwise, use the index from config.
    """
    if cfg['camera']['require_logitech']:
        return find_logitech_camera()
    else:
        return cfg['camera']['index']


class CameraCapture:
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> None:
        """
        Open cv2.VideoCapture at camera_index.
        Raise CameraNotFoundError if cap.isOpened() is False.
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise CameraNotFoundError(
                f"Could not open camera at index {self.camera_index}. Is the Logitech camera "
                f"connected and not in use by another application?"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """Return (success, bgr_frame). Mirror of cv2.VideoCapture.read()."""
        return self.cap.read()

    def actual_fps(self) -> float:
        """Return the fps reported by the device after open()."""
        actual = self.cap.get(cv2.CAP_PROP_FPS)

        if actual < FPS_MINIMUM:
            logger.warning(f"Camera FPS {actual} is below minimum {FPS_MINIMUM}")
        elif actual < FPS_TARGET:
            logger.info(f"Camera FPS {actual} is below target {FPS_TARGET}")

        return actual

    def release(self) -> None:
        """Release VideoCapture handle."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
