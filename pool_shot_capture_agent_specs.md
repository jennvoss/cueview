# Pool Shot Capture System — Agent Task Specifications
<!-- FORMAT: Each TASK block is a self-contained unit of work for an AI coding agent.
     Fields are machine-readable. Agents MUST NOT start a task until all DEPENDS_ON tasks are marked DONE.
     Complete tasks in dependency order. Run VERIFY commands before marking any task DONE. -->

---

## PROJECT CONTEXT

```yaml
project: pool-shot-capture
language: Python 3.11+
platform: macOS 13+ (Apple Silicon and Intel)
entry_point: src/main.py
config_file: config/config.yaml
test_runner: pytest
session_log: outputs/session.jsonl
```

### Repository layout (create exactly this structure)
```
pool-shot-capture/
├── config/
│   └── config.yaml
├── src/
│   ├── main.py
│   ├── capture.py
│   ├── calibration.py
│   ├── detection.py
│   ├── tracking.py
│   ├── metrics.py
│   └── output.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/           # synthetic images + calibration files
│   ├── test_calibration.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_metrics.py
│   └── test_output.py
├── outputs/                # gitignored; session logs written here
├── requirements.txt
└── README.md
```

### Shared constants (import from `src/constants.py` — create in TASK-01)
```python
TABLE_WIDTH_IN  = 88.0   # standard 8-ft table, playing surface inches
TABLE_HEIGHT_IN = 44.0
BALL_DIAMETER_IN = 2.25
FPS_TARGET       = 60
FPS_MINIMUM      = 30
REST_THRESHOLD_IPS = 0.5   # inches/sec below which ball is considered stationary
SHOT_START_MPH   = 1.0     # velocity spike threshold to open a shot event
SMOOTHER_WINDOW  = 3       # default moving-average window (frames)
MAX_REACQUIRE_FRAMES = 5   # frames tracker may lose cue ball before resetting
```

---

## TASK-01 · Project scaffold & config

```yaml
id: TASK-01
title: Project scaffold, constants, and config loader
depends_on: []
priority: MUST
output_files:
  - requirements.txt
  - config/config.yaml
  - src/constants.py
  - src/config_loader.py
```

### requirements.txt — pin to these exact versions
```
opencv-python>=4.9,<5.0
numpy>=1.26,<2.0
scipy>=1.12,<2.0
filterpy>=1.4,<2.0
pyyaml>=6.0,<7.0
pytest>=8.0,<9.0
```

### config/config.yaml — create with these keys and defaults
```yaml
camera:
  # Device selection — agent must implement find_logitech_camera() in capture.py.
  # index is ignored when require_logitech is true; the system enumerates USB devices
  # and selects the first Logitech camera found. Set require_logitech to false only
  # for unit tests that use a mock capture source.
  index: 0                  # fallback only; overridden by find_logitech_camera()
  require_logitech: true    # raise CameraNotFoundError if no Logitech device present
  width: 1920
  height: 1080
  fps: 60

calibration:
  file: config/calibration.npz
  checkerboard_rows: 6      # inner corners
  checkerboard_cols: 9
  skip: false               # override with --no-calibration flag

detection:
  hsv_ranges:
    cue_ball:   [[0,0,200],   [180,30,255]]   # white
    solid_yellow: [[20,100,100],[30,255,255]]
    # extend for all 15 object ball colours as needed
  hough_dp: 1.2
  hough_min_dist_px: 20
  hough_param1: 50
  hough_param2: 30
  diameter_tolerance: 0.20   # ±20% of expected ball diameter in pixels

tracking:
  method: kalman             # kalman | centroid
  max_reacquire_frames: 5

metrics:
  smoother_window: 3         # frames

output:
  log_file: outputs/session.jsonl
  flush_interval_s: 30
  overlay: true
  log_level: INFO            # DEBUG | INFO | WARNING
```

### src/constants.py
Define all constants listed in SHARED CONSTANTS above. No logic — constants only.

### src/config_loader.py
```python
# Public API this module must expose:

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config. Raise FileNotFoundError if missing."""

def override_config(cfg: dict, overrides: dict) -> dict:
    """Deep-merge CLI overrides into cfg. Return merged dict. Do not mutate inputs."""
```

### VERIFY
```bash
python -c "from src.config_loader import load_config; c = load_config(); assert c['camera']['fps'] == 60"
```

---

## TASK-02 · Camera capture module

```yaml
id: TASK-02
title: Camera capture and frame provider — Logitech USB camera required
depends_on: [TASK-01]
priority: MUST
output_files:
  - src/capture.py
  - tests/test_capture.py
extra_dependencies:
  - pyobjc-framework-CoreMediaIO  # macOS camera enumeration (add to requirements.txt)
```

### Add to requirements.txt
```
pyobjc-framework-CoreMediaIO>=10.0
```

### src/capture.py — required public API

```python
import cv2
import subprocess
import re
import logging
import numpy as np
from src.constants import FPS_TARGET, FPS_MINIMUM

logger = logging.getLogger(__name__)

LOGITECH_USB_VENDOR_ID = "046d"   # Logitech's registered USB vendor ID (lowercase hex)

class CameraNotFoundError(RuntimeError):
    """Raised when no Logitech USB camera can be found or opened."""
    pass


def list_video_devices() -> list[dict]:
    """
    Enumerate all video capture devices visible to the OS.

    On macOS, shells out to:
        system_profiler SPCameraDataType -json
    Parses the JSON output and returns a list of dicts, each with keys:
        {
            "index": int,          # OpenCV VideoCapture index (0-based, in discovery order)
            "name": str,           # Human-readable device name, e.g. "Logi C920 HD Pro Webcam"
            "vendor_id": str,      # USB vendor ID hex string, lowercase, e.g. "046d"
            "is_logitech": bool,   # True iff vendor_id == LOGITECH_USB_VENDOR_ID
        }

    Implementation notes:
    - Run: subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True)
    - Parse the JSON. Each camera entry appears under the key "Camera" or "_items".
    - Extract vendor ID from the "usb_vendor_id" field if present, else from device name heuristic.
    - Assign OpenCV index by iterating cv2.VideoCapture(i) for i in range(10) and matching
      device names to the system_profiler output. Stop at the first index that fails isOpened().
    - If system_profiler is unavailable (e.g. in CI), return an empty list — do not raise.
    - Never raise from this function; log WARNING on any parse failure and return what was found.
    """

def find_logitech_camera() -> int:
    """
    Return the OpenCV VideoCapture index of the first Logitech USB camera found.

    Algorithm:
    1. Call list_video_devices().
    2. Filter to devices where is_logitech=True.
    3. If none found: raise CameraNotFoundError with message:
       "No Logitech camera detected. Ensure the camera is plugged in via USB and
        recognised by macOS (check System Settings > Privacy & Security > Camera).
        Connected devices: <comma-separated device names from list_video_devices()>"
    4. If one or more found: return the index of the first one and log INFO:
       "Logitech camera found: <name> at index <index>"

    This function must never return the index of the built-in FaceTime/Continuity camera.
    If the only camera present is a built-in Mac camera, raise CameraNotFoundError.
    """

def find_camera_index(cfg: dict) -> int:
    """
    Resolve the camera index to use, respecting config.

    If cfg['camera']['require_logitech'] is True (default):
        return find_logitech_camera()   # raises CameraNotFoundError if not found
    Else:
        return cfg['camera']['index']   # use raw index from config (for tests/mocks)
    """


class CameraCapture:
    def __init__(self, camera_index: int, width: int, height: int, fps: int): ...

    def open(self) -> None:
        """
        Open cv2.VideoCapture at camera_index.
        Set CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS.
        Raise CameraNotFoundError (NOT generic RuntimeError) if cap.isOpened() is False,
        with message: "Could not open camera at index <index>. Is the Logitech camera
        connected and not in use by another application?"
        """

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """Return (success, bgr_frame). Mirror of cv2.VideoCapture.read()."""

    def actual_fps(self) -> float:
        """Return the fps reported by the device after open()."""

    def release(self) -> None:
        """Release VideoCapture handle."""

    def __enter__(self): ...
    def __exit__(self, *_): ...
```

### Behaviour rules
- `find_logitech_camera()` MUST raise `CameraNotFoundError` — not a generic exception — when no Logitech device is found. Agents and callers must be able to `except CameraNotFoundError` specifically.
- The error message MUST list all currently connected camera names so the user can diagnose the problem without re-running a separate diagnostic.
- `open()` MUST raise `CameraNotFoundError` (not `RuntimeError`) for consistency with `find_logitech_camera()`.
- If `actual_fps()` < `FPS_MINIMUM` (30), log WARNING and continue — do not raise.
- If `actual_fps()` < `FPS_TARGET` (60), log INFO noting degraded fps.
- Must be usable as a context manager (`with CameraCapture(...) as cap:`).

### How find_logitech_camera() is called from main.py
```python
# In main() before constructing CameraCapture:
try:
    camera_index = find_camera_index(cfg)
except CameraNotFoundError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

with CameraCapture(camera_index, ...) as cap:
    ...
```

### tests/test_capture.py — must cover

| Test ID | Setup | Assertion |
|---------|-------|-----------|
| UT-CAP-01 | `list_video_devices()` with mocked `subprocess.run` returning a fixture JSON with one Logitech device (vendor_id="046d") and one Apple FaceTime camera. | Returns list of 2 dicts; only the Logitech entry has `is_logitech=True` |
| UT-CAP-02 | `find_logitech_camera()` with mocked `list_video_devices()` returning one Logitech device at index 1. | Returns `1` |
| UT-CAP-03 | `find_logitech_camera()` with mocked `list_video_devices()` returning only a built-in FaceTime camera. | Raises `CameraNotFoundError`; error message contains the FaceTime camera name |
| UT-CAP-04 | `find_logitech_camera()` with mocked `list_video_devices()` returning empty list. | Raises `CameraNotFoundError`; error message says "No devices found" |
| UT-CAP-05 | `find_camera_index(cfg)` with `require_logitech=False` and `index=2`. | Returns `2` without calling `find_logitech_camera()` |
| UT-CAP-06 | `CameraCapture.open()` with mocked `cv2.VideoCapture` where `isOpened()` returns `False`. | Raises `CameraNotFoundError` (not `RuntimeError`) |
| UT-CAP-07 | `CameraCapture.actual_fps()` after open. | Returns value from `cap.get(cv2.CAP_PROP_FPS)` |
| UT-CAP-08 | `CameraCapture.__exit__()`. | Calls `release()` |

### VERIFY
```bash
pytest tests/test_capture.py -v
# All 8 tests must pass with no real camera attached.
```

---

## TASK-03 · Calibration module

```yaml
id: TASK-03
title: Lens calibration and pixel-to-inch homography
depends_on: [TASK-01]
priority: MUST
output_files:
  - src/calibration.py
  - tests/test_calibration.py
  - tests/fixtures/calibration_sample.npz   # synthetic fixture — see below
```

### src/calibration.py — required public API
```python
def run_calibration(
    capture: "CameraCapture",
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

def load_calibration(path: str) -> dict:
    """Load .npz file. Return same dict shape as run_calibration(). Raise FileNotFoundError if missing."""

def undistort_frame(frame: np.ndarray, calibration: dict) -> np.ndarray:
    """Apply camera_matrix + dist_coeffs. Return corrected frame (same shape)."""

def pixel_to_inches(
    pixel_coords: np.ndarray,   # shape (N, 2) float32
    calibration: dict,
) -> np.ndarray:
    """Apply homography. Return (N, 2) array of real-world inch coordinates."""
```

### Fixture: tests/fixtures/calibration_sample.npz
Generate synthetically in `tests/conftest.py` using `cv2.calibrateCamera` on a set of perfect planar points. Store as an `.npz` with keys: `camera_matrix`, `dist_coeffs`, `homography`, `reprojection_error`.

### tests/test_calibration.py — must cover
| Test ID | Assertion |
|---------|-----------|
| UT-CAL-01 | `load_calibration` on fixture returns dict with all 4 keys; values match stored arrays within `np.allclose` |
| UT-CAL-02 | `undistort_frame` on a synthetic radially distorted image returns image where straight lines are measurably straighter |
| UT-CAL-03 | `pixel_to_inches` on 4 known pixel coords (corners of table) returns inch coords within ±0.1 in of ground truth |
| UT-CAL-04 | `run_calibration` (mocked capture returning synthetic frames) logs WARNING when reprojection_error injected as 0.8 |

### VERIFY
```bash
pytest tests/test_calibration.py -v
# All 4 tests must pass.
```

---

## TASK-04 · Ball detection module

```yaml
id: TASK-04
title: HSV + Hough circle ball detector
depends_on: [TASK-01, TASK-03]
priority: MUST
output_files:
  - src/detection.py
  - tests/test_detection.py
  - tests/fixtures/  # synthetic test images — see below
```

### src/detection.py — required public API
```python
@dataclass
class Detection:
    center_px: tuple[float, float]   # (x, y) in pixel space
    radius_px: float
    is_cue_ball: bool
    color_label: str                  # e.g. "cue_ball", "solid_yellow", "unknown"

def detect_balls(
    frame: np.ndarray,               # undistorted BGR frame
    config: dict,                    # detection section of config
    expected_radius_px: float,       # computed from calibration + BALL_DIAMETER_IN
) -> list[Detection]:
    """
    1. Convert frame to HSV.
    2. For each HSV range in config['detection']['hsv_ranges'], create a mask.
    3. Run HoughCircles on each masked channel.
    4. Filter circles outside expected_radius_px ± diameter_tolerance.
    5. Mark the circle matching cue_ball HSV range as is_cue_ball=True.
    6. Return list of Detection objects. Empty list if no balls found — never raise.
    """

def compute_expected_radius_px(calibration: dict) -> float:
    """
    Use homography inverse to convert BALL_DIAMETER_IN to pixel radius.
    Returns float.
    """
```

### Synthetic test images (generate in conftest.py)
Create 640×480 BGR images with:
- `fixture_single_white_ball.png` — one white circle on green felt, centre (320, 240), radius 20 px
- `fixture_16_balls.png` — 16 circles, one white + 15 various colours, no overlap, all >30 px from edge
- `fixture_empty_table.png` — green felt, no circles
- `fixture_near_rail.png` — 4 white/colour circles within 15 px of image edge
- `fixture_noise.png` — green felt with a coin (r=8 px) and a large white region (r=60 px)

### tests/test_detection.py — must cover
| Test ID | Assertion |
|---------|-----------|
| UT-DET-01 | `detect_balls(fixture_single_white_ball)` returns 1 Detection with `is_cue_ball=True` |
| UT-DET-02 | `detect_balls(fixture_16_balls)` returns ≥14 Detections; exactly 1 has `is_cue_ball=True` |
| UT-DET-03 | `detect_balls(fixture_noise)` does not return the coin (r=8 px, below size filter) |
| UT-DET-04 | `detect_balls(fixture_noise)` does not return the large white region (r=60 px, above size filter) |
| UT-DET-05 | `detect_balls(fixture_empty_table)` returns empty list without raising |
| UT-DET-06 | `detect_balls(fixture_near_rail)` returns ≥3 of the 4 near-rail balls |

### VERIFY
```bash
pytest tests/test_detection.py -v
# All 6 tests must pass.
```

---

## TASK-05 · Ball tracking module

```yaml
id: TASK-05
title: Cue ball tracker (Kalman + centroid fallback)
depends_on: [TASK-01, TASK-04]
priority: MUST
output_files:
  - src/tracking.py
  - tests/test_tracking.py
```

### src/tracking.py — required public API
```python
@dataclass
class TrackedBall:
    id: int
    center_px: tuple[float, float]
    is_cue_ball: bool
    frames_missing: int              # 0 if detected this frame

class BallTracker:
    def __init__(self, method: str = "kalman", max_reacquire: int = MAX_REACQUIRE_FRAMES): ...

    def update(self, detections: list[Detection]) -> list[TrackedBall]:
        """
        Match detections to existing tracked balls by nearest-centroid.
        For cue ball specifically:
          - Re-use existing cue ball ID if distance < 2 × expected_radius_px.
          - Increment frames_missing if cue ball not detected this frame.
          - Purge cue ball track if frames_missing > max_reacquire.
        Returns updated list of TrackedBall. Never raises.
        """

    def cue_ball(self) -> TrackedBall | None:
        """Return current cue ball TrackedBall, or None if not tracked."""

    def reset(self) -> None:
        """Clear all tracks."""
```

### Kalman filter spec (when method="kalman")
- State vector: `[x, y, vx, vy]` (position + velocity in pixel space)
- Use `filterpy.kalman.KalmanFilter` with:
  - `dim_x=4`, `dim_z=2`
  - `F` = constant-velocity transition matrix
  - `H` = observation matrix extracting `[x, y]`
  - `R` = `np.eye(2) * 5.0` (measurement noise)
  - `Q` = `np.eye(4) * 0.1` (process noise)
- When no detection available for cue ball: call `kf.predict()` only (no update step).

### tests/test_tracking.py — must cover
| Test ID | Assertion |
|---------|-----------|
| UT-TRK-01 | Feed 10 frames of linearly moving cue ball (5 px/frame). Same ID throughout; final position within 3 px of expected. |
| UT-TRK-02 | Feed 5 normal frames, then 4 frames with NO cue ball detection, then resume. Cue ball regains same ID on frame 10; `frames_missing` resets to 0. |
| UT-TRK-03 | Feed 100 frames with ±3 px Gaussian noise on cue ball position. No ID change (ID stable throughout). |

### VERIFY
```bash
pytest tests/test_tracking.py -v
```

---

## TASK-06 · Velocity and metrics module

```yaml
id: TASK-06
title: Velocity computation, shot detection, and accuracy scoring
depends_on: [TASK-01, TASK-03, TASK-05]
priority: MUST
output_files:
  - src/metrics.py
  - tests/test_metrics.py
```

### src/metrics.py — required public API
```python
@dataclass
class ShotRecord:
    timestamp_iso: str               # ISO 8601 UTC
    frame_start: int
    frame_end: int
    peak_speed_mph: float
    mean_speed_mph: float
    angle_deg: float                 # trajectory angle at strike, 0=right, CCW positive
    start_pos_in: tuple[float, float]
    end_pos_in: tuple[float, float]
    accuracy_in: float | None        # None if no target set
    target_pos_in: tuple[float, float] | None

class MetricsEngine:
    def __init__(self, calibration: dict, smoother_window: int = SMOOTHER_WINDOW): ...

    def update(
        self,
        tracked_ball: TrackedBall | None,
        frame_index: int,
        timestamp: float,            # time.perf_counter() at frame capture
    ) -> ShotRecord | None:
        """
        Called once per frame with the current cue ball state.
        - Convert pixel position to inches via pixel_to_inches().
        - Compute instantaneous speed (inches/sec → mph).
        - Apply moving-average smoother over last smoother_window speeds.
        - If smoothed speed < REST_THRESHOLD_IPS: mark ball as at rest.
        - Shot START: transition from rest → smoothed speed > SHOT_START_MPH.
        - Shot END: transition from moving → at rest, OR cue ball enters pocket area.
        - On shot END: build and return ShotRecord. Otherwise return None.
        """

    def set_target(self, target_pos_in: tuple[float, float]) -> None:
        """Set intended target for next shot. Cleared after shot END."""

    def current_speed_mph(self) -> float:
        """Return latest smoothed speed in mph."""
```

### Speed conversion
```python
# 1 mph = 17.6 inches/sec
def ips_to_mph(ips: float) -> float:
    return ips / 17.6
```

### Accuracy computation
```python
# Euclidean distance in inches between end_pos_in and target_pos_in
import math
accuracy_in = math.dist(end_pos_in, target_pos_in)  # None if target not set
```

### tests/test_metrics.py — must cover
| Test ID | Input | Assertion |
|---------|-------|-----------|
| UT-VEL-01 | Synthetic: ball moves 10 in over 10 frames at 60 fps → 1 in/frame → 6 in/s → 0.34 mph. Adjust frame count so peak reaches ~5 mph. | `peak_speed_mph` within ±0.5 of expected |
| UT-VEL-02 | 20 frames with position jitter ±0.3 px (converts to <0.01 in). | `current_speed_mph()` returns 0.0; no ShotRecord emitted |
| UT-VEL-03 | 5 rest frames → 10 frames at 15 mph → 5 rest frames. | Exactly one ShotRecord; `peak_speed_mph` between 13 and 17 |
| UT-VEL-04 | Compare raw vs smoothed speed on noisy sequence. | Smoothed std dev ≤ 50% of raw std dev |
| UT-VEL-05 | `end_pos_in=(3,4)`, `target_pos_in=(0,0)`. | `accuracy_in == 5.0` |

### VERIFY
```bash
pytest tests/test_metrics.py -v
```

---

## TASK-07 · Output module

```yaml
id: TASK-07
title: Overlay renderer and session log writer
depends_on: [TASK-01, TASK-06]
priority: MUST
output_files:
  - src/output.py
  - tests/test_output.py
```

### src/output.py — required public API
```python
REQUIRED_LOG_FIELDS = {
    "timestamp_iso", "frame_start", "frame_end",
    "peak_speed_mph", "mean_speed_mph", "angle_deg",
    "start_pos_in", "end_pos_in", "accuracy_in", "target_pos_in",
}

class SessionLogger:
    def __init__(self, log_path: str, flush_interval_s: float = 30.0): ...

    def log_shot(self, record: ShotRecord) -> None:
        """Append JSON line to log_path. All REQUIRED_LOG_FIELDS must be present."""

    def flush(self) -> None:
        """Force flush file buffer to disk."""

    def close(self) -> None:
        """Flush and close file handle."""


class OverlayRenderer:
    def __init__(self, enabled: bool = True): ...

    def render(
        self,
        frame: np.ndarray,                        # BGR, modified in-place
        tracked_balls: list[TrackedBall],
        speed_mph: float,
        last_shot: ShotRecord | None,
        trail: list[tuple[float, float]],          # last N cue ball pixel positions
    ) -> np.ndarray:
        """
        Draw on frame (in-place):
          - Green circle around each detected ball; yellow for cue ball
          - Cue ball trajectory trail (fading dots or line)
          - Top-left HUD: current speed (mph), last shot peak speed (mph)
        Return the annotated frame.
        If self.enabled is False, return frame unmodified.
        """

    def toggle(self) -> None:
        """Flip self.enabled."""
```

### tests/test_output.py — must cover
| Test ID | Assertion |
|---------|-----------|
| UT-OUT-01 | `log_shot` called 3× writes exactly 3 lines to the log file; each line is valid JSON |
| UT-OUT-02 | Each parsed JSON record contains all keys in `REQUIRED_LOG_FIELDS` |
| UT-OUT-03 | Close logger, reopen file in append mode, log one more shot → 4 lines total, no corruption |

### VERIFY
```bash
pytest tests/test_output.py -v
```

---

## TASK-08 · Main entry point

```yaml
id: TASK-08
title: Main loop, CLI, and graceful shutdown
depends_on: [TASK-02, TASK-03, TASK-04, TASK-05, TASK-06, TASK-07]
priority: MUST
output_files:
  - src/main.py
```

### CLI interface (use argparse)
```
usage: python src/main.py [OPTIONS]

Options:
  --camera-index INT     Camera device index (default: from config)
  --config PATH          Path to config YAML (default: config/config.yaml)
  --no-calibration       Skip calibration load; use identity matrices
  --record               Record raw session video to outputs/session_<timestamp>.mp4
  --output-dir PATH      Override output directory for logs and recordings
  --log-level LEVEL      DEBUG | INFO | WARNING (default: from config)
```

### Main loop pseudocode (implement exactly this logic)
```python
def main():
    cfg = load_config(args.config)
    cfg = override_config(cfg, cli_overrides)
    calibration = load_calibration(cfg) if not args.no_calibration else identity_calibration()
    expected_radius_px = compute_expected_radius_px(calibration)

    tracker  = BallTracker(method=cfg['tracking']['method'])
    engine   = MetricsEngine(calibration, cfg['metrics']['smoother_window'])
    logger   = SessionLogger(cfg['output']['log_file'], cfg['output']['flush_interval_s'])
    renderer = OverlayRenderer(enabled=cfg['output']['overlay'])
    trail    = deque(maxlen=30)   # last 30 cue ball pixel positions

    with CameraCapture(...) as cap:
        frame_index = 0
        while True:
            ok, frame = cap.read_frame()
            if not ok:
                break

            undistorted = undistort_frame(frame, calibration)
            detections  = detect_balls(undistorted, cfg['detection'], expected_radius_px)
            tracked     = tracker.update(detections)
            cb          = tracker.cue_ball()

            if cb:
                trail.append(cb.center_px)

            shot = engine.update(cb, frame_index, time.perf_counter())
            if shot:
                logger.log_shot(shot)
                print_shot_summary(shot)   # stdout

            annotated = renderer.render(undistorted, tracked, engine.current_speed_mph(), shot, list(trail))
            cv2.imshow("Pool Shot Capture", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                renderer.toggle()
            elif key == ord('t'):
                # user sets target at cue ball current position (simplification)
                if cb:
                    pos_in = pixel_to_inches(np.array([cb.center_px]), calibration)[0]
                    engine.set_target(tuple(pos_in))

            frame_index += 1

    logger.close()
    cv2.destroyAllWindows()
```

### Keyboard controls
| Key | Action |
|-----|--------|
| `q` | Quit cleanly; flush log |
| `o` | Toggle overlay on/off |
| `t` | Set shot target at current cue ball position |

### VERIFY
```bash
# Dry-run with --no-calibration and a synthetic video source (mock cap.read_frame)
python -c "
import sys; sys.argv = ['main', '--no-calibration', '--log-level', 'DEBUG']
from unittest.mock import patch, MagicMock
import numpy as np
fake_frame = np.zeros((480,640,3), dtype=np.uint8)
# patch CameraCapture.read_frame to return 3 frames then stop
# assert no exception raised
print('Dry-run OK')
"
```

---

## TASK-09 · Integration test suite

```yaml
id: TASK-09
title: Integration tests (camera-free synthetic end-to-end)
depends_on: [TASK-08]
priority: MUST
output_files:
  - tests/test_integration.py
```

### tests/test_integration.py — must cover

These tests wire all modules together using a synthetic video source (no real camera).

| Test ID | Synthetic input | Assertion |
|---------|----------------|-----------|
| IT-SYN-01 | 60 frames: cue ball stationary. | No ShotRecord emitted; log file has 0 lines |
| IT-SYN-02 | 5 rest + 20 moving (10 mph) + 5 rest frames. | Exactly 1 ShotRecord; `peak_speed_mph` 8–12; 1 log line written |
| IT-SYN-03 | Two sequential shots with 10-frame rest gap between them. | Exactly 2 ShotRecords; 2 log lines; IDs are sequential |
| IT-SYN-04 | Shot with target set before shot start. | ShotRecord has non-None `accuracy_in` |
| IT-SYN-05 | Overlay toggle mid-sequence (key='o' injected at frame 30). | No exception; frames before and after are both valid BGR arrays |

### VERIFY
```bash
pytest tests/test_integration.py -v
```

---

## TASK-10 · Performance benchmark script

```yaml
id: TASK-10
title: Benchmark harness for latency, CPU, and memory
depends_on: [TASK-08]
priority: SHOULD
output_files:
  - tests/benchmark.py
```

### tests/benchmark.py — behaviour
- Runs the full pipeline on 1000 synthetic frames (no camera).
- Measures and prints:
  - p50 / p95 / p99 frame latency (ms)
  - Peak RSS memory (MB)
- Fails with non-zero exit code if p95 latency > 100 ms on Apple M1 or later.

```bash
# Run:
python tests/benchmark.py
# Expected stdout example:
# Latency p50=12ms  p95=34ms  p99=58ms
# Memory peak=210MB
# PASS
```

---

## TASK-11 · README

```yaml
id: TASK-11
title: README with setup and run instructions
depends_on: [TASK-08]
priority: MUST
output_files:
  - README.md
```

### README.md must include exactly these sections

1. **Requirements** — Python 3.11+, macOS 13+, Logitech USB camera (required)
2. **Installation**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Camera calibration** — step-by-step: print checkerboard, run calibration routine, confirm reprojection error
4. **Running**
   ```bash
   python src/main.py
   python src/main.py --no-calibration   # skip for first smoke test
   ```
5. **Keyboard controls** — table matching TASK-08 keyboard controls
6. **Session log format** — show an example JSON line with all fields
7. **Troubleshooting: camera not found** — include the following content verbatim:
   - What `CameraNotFoundError` means and when it fires
   - Step 1: confirm the Logitech camera is physically connected via USB
   - Step 2: open System Settings → Privacy & Security → Camera and ensure the terminal app has permission
   - Step 3: run `system_profiler SPCameraDataType` in Terminal and confirm the Logitech device appears
   - Note that `--no-calibration` alone does not bypass camera detection; use `require_logitech: false` in config.yaml only for development/testing with a non-Logitech camera
8. **Running tests**
   ```bash
   pytest tests/ -v
   python tests/benchmark.py
   ```
8. **Config reference** — one-line description of every key in config.yaml

---

## COMPLETION CHECKLIST

An agent MUST verify each item below before declaring the project done:

```
[ ] pip install -r requirements.txt completes with no errors in a fresh venv
[ ] pytest tests/ -v → 0 failures, 0 errors (all 8 UT-CAP tests pass without a real camera)
[ ] python tests/benchmark.py → PASS
[ ] python src/main.py --no-calibration launches without exception (mock camera acceptable)
[ ] With no USB camera attached: python src/main.py exits with code 1 and prints a CameraNotFoundError message listing connected devices
[ ] With Logitech camera attached: python src/main.py --no-calibration opens the Logitech device (not the built-in Mac camera)
[ ] outputs/session.jsonl created and contains valid JSON lines after a synthetic run
[ ] All REQUIRED_LOG_FIELDS present in every log line
[ ] README.md contains all 8 required sections
[ ] README.md documents the CameraNotFoundError and how to resolve it (USB connection, macOS camera permissions)
```
