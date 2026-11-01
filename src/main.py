import argparse
import json
import os
import sys
import time
import logging
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
import cv2
from collections import deque
from pathlib import Path
from dotenv import load_dotenv

from src.config_loader import load_config, override_config
from src.capture import CameraCapture, StillImageSource, find_camera_index, CameraNotFoundError
from src.calibration import load_calibration, undistort_frame, pixel_to_inches
from src.detection import detect_balls, compute_expected_radius_px, player_over_table
from src.tracking import BallTracker
from src.metrics import MetricsEngine
from src.output import SessionLogger, OverlayRenderer
from src.scorekeeper import ScoreKeeper
from src.drills import DrillTracker, DRILLS, read_selection


def identity_calibration() -> dict:
    """Return identity matrices for calibration."""
    return {
        "camera_matrix": np.eye(3, dtype=np.float32),
        "dist_coeffs": np.zeros(5, dtype=np.float32),
        "homography": np.eye(3, dtype=np.float32),
        "reprojection_error": 0.0,
    }


def print_shot_summary(shot) -> None:
    """Print a shot summary to stdout."""
    print(f"Shot recorded: frames {shot.frame_start}-{shot.frame_end}, "
          f"angle={shot.angle_deg:.1f}°")


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
STILLS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "table"


def save_still(frame) -> Path:
    """Save the raw (pre-overlay) frame as a PNG test fixture."""
    STILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = STILLS_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
    cv2.imwrite(str(path), frame)
    return path


# Live table_polygon tuning, enabled with TUNE_TABLE_POLYGON=true in .env.
TUNE_POLYGON_ENV = "TUNE_TABLE_POLYGON"
APP_WINDOW = "CueView"
# Trackbars live on the video window: on macOS a trackbar-only window has no
# size, and a separate window gets buried under the video window.
POLYGON_WINDOW = APP_WINDOW
CORNER_NAMES = ("TL", "TR", "BR", "BL")


class PolygonTuner:
    """Trackbars for each table_polygon corner, seeded from config.yaml."""

    def __init__(self, polygon, width: int, height: int):
        cv2.namedWindow(POLYGON_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow(POLYGON_WINDOW, 0, 0)
        for name, (x, y) in zip(CORNER_NAMES, polygon):
            cv2.createTrackbar(f"{name} x", POLYGON_WINDOW, int(x), width, lambda _v: None)
            cv2.createTrackbar(f"{name} y", POLYGON_WINDOW, int(y), height, lambda _v: None)

    def read(self) -> list:
        return [
            [cv2.getTrackbarPos(f"{name} x", POLYGON_WINDOW),
             cv2.getTrackbarPos(f"{name} y", POLYGON_WINDOW)]
            for name in CORNER_NAMES
        ]

    @staticmethod
    def print_yaml(polygon) -> None:
        print("  table_polygon:")
        for x, y in polygon:
            print(f"    - [{x}, {y}]")


def polygon_tuning_enabled() -> bool:
    value = os.environ.get(TUNE_POLYGON_ENV, "false").strip().lower()
    if value not in ("true", "false"):
        logging.getLogger(__name__).warning(
            f"{TUNE_POLYGON_ENV}={value!r} is not true/false; treating as false")
    return value == "true"


RAW_FRAME_NAME = "latest_raw.png"
RAW_FRAME_INTERVAL_S = 5.0
STILL_IMAGE_FPS = 5.0


def find_still_image(data_dir: Path):
    """Most recent saved frame to score when no camera is connected: the raw frame if
    there is one, else the annotated latest.jpg (its drawn markers can skew detection)."""
    for name in (RAW_FRAME_NAME, "latest.jpg"):
        path = Path(data_dir) / name
        if path.exists():
            return path
    return None


def open_frame_source(args, cfg: dict, data_dir: Path, logger):
    """Open the camera, or a still image with --image or when no camera is connected.
    Returns (opened source, source info for status.json)."""
    if args.image is None:
        try:
            cap = CameraCapture(
                find_camera_index(cfg),
                cfg["camera"]["width"],
                cfg["camera"]["height"],
                cfg["camera"]["fps"],
            )
            cap.open()
            return cap, {"kind": "camera"}
        except CameraNotFoundError as e:
            still = find_still_image(data_dir)
            if still is None:
                print(f"ERROR: {e}\nNo saved frame in {data_dir} to use instead; "
                      f"pass --image PATH to score a still image.", file=sys.stderr)
                sys.exit(1)
            logger.warning(f"No camera ({e}). Scoring the last saved frame instead: {still}")
    else:
        still = Path(args.image)

    source = StillImageSource(still, fps=STILL_IMAGE_FPS)
    try:
        source.open()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    annotated = still.name == "latest.jpg"
    if annotated:
        logger.warning("This frame has detection markers drawn on it, so the score may be off. "
                       f"Run once with the camera connected to save an unmarked {RAW_FRAME_NAME}.")
    return source, {"kind": "image", "path": str(still), "annotated": annotated}


class StatusWriter:
    def __init__(self, data_dir: Path, status_interval: int, write_frames: bool,
                 calibration: dict, target_fps: float, source: dict = None,
                 save_raw_frames: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.data_dir / "status.json"
        self.status_tmp = self.data_dir / "status.json.tmp"
        self.frame_path = self.data_dir / "latest.jpg"
        # Unannotated frame, saved every RAW_FRAME_INTERVAL_S while the camera runs, so
        # the app can score from it later when no camera is connected.
        self.raw_frame_path = self.data_dir / RAW_FRAME_NAME
        self.save_raw_frames = save_raw_frames
        self.last_raw_save = None
        self.source = source or {"kind": "camera"}
        self.status_interval = status_interval
        self.write_frames = write_frames
        self.calibration = calibration
        self.target_fps = float(target_fps)
        self.frame_times = deque(maxlen=30)
        self.logger = logging.getLogger(__name__ + ".StatusWriter")

    def should_write(self, frame_index: int) -> bool:
        return self.status_interval > 0 and frame_index % self.status_interval == 0

    def tick(self) -> None:
        self.frame_times.append(time.perf_counter())

    def _measured_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / span

    def write(self, frame_index: int, tracked: list, detections: list,
              engine: MetricsEngine, last_shot, annotated_frame, score: dict = None,
              drill: dict = None,
              raw_frame=None) -> None:
        try:
            tracked_balls = []
            for tb in tracked:
                center_in = None
                try:
                    pos_in = pixel_to_inches(np.array([tb.center_px]), self.calibration)[0]
                    center_in = [float(pos_in[0]), float(pos_in[1])]
                except Exception:
                    center_in = None
                tracked_balls.append({
                    "id": int(tb.id),
                    "center_px": [float(tb.center_px[0]), float(tb.center_px[1])],
                    "center_in": center_in,
                    "is_cue_ball": bool(tb.is_cue_ball),
                    "frames_missing": int(tb.frames_missing),
                })

            target_pos_in = None
            if engine.target_pos_in is not None:
                target_pos_in = [float(engine.target_pos_in[0]), float(engine.target_pos_in[1])]

            last_shot_dict = None
            if last_shot is not None:
                last_shot_dict = asdict(last_shot)
                for k in ("start_pos_in", "end_pos_in", "target_pos_in"):
                    if last_shot_dict.get(k) is not None:
                        last_shot_dict[k] = list(last_shot_dict[k])

            # Add detection information
            detections_info = []
            for detection in detections:
                detections_info.append({
                    "center_px": [float(detection.center_px[0]), float(detection.center_px[1])],
                    "radius_px": float(detection.radius_px),
                    "is_cue_ball": bool(detection.is_cue_ball),
                    "color_label": detection.color_label,
                    "ball": detection.ball,
                })

            status = {
                "status": "running",
                "frame_index": int(frame_index),
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "target_fps": self.target_fps,
                "measured_fps": round(self._measured_fps(), 2),
                "detections_this_frame": int(len(detections)),
                "tracked_balls": tracked_balls,
                "is_in_shot": bool(engine.is_in_shot),
                "target_pos_in": target_pos_in,
                "last_shot": last_shot_dict,
                "detections": detections_info,
                "score": score,
                "source": self.source,
                "drill": drill,
            }
            with open(self.status_tmp, "w") as f:
                json.dump(status, f)
            os.replace(self.status_tmp, self.status_path)
        except Exception as e:
            self.logger.warning(f"Failed to write status.json: {e}")

        if self.write_frames and annotated_frame is not None:
            try:
                cv2.imwrite(str(self.frame_path), annotated_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 75])
            except Exception as e:
                self.logger.warning(f"Failed to write latest.jpg: {e}")

        now = time.monotonic()
        if (self.save_raw_frames and raw_frame is not None
                and (self.last_raw_save is None or now - self.last_raw_save >= RAW_FRAME_INTERVAL_S)):
            try:
                tmp = self.raw_frame_path.with_suffix(".tmp.png")
                cv2.imwrite(str(tmp), raw_frame)
                os.replace(tmp, self.raw_frame_path)
                self.last_raw_save = now
            except Exception as e:
                self.logger.warning(f"Failed to write {RAW_FRAME_NAME}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CueView — camera-based 8-ball scorekeeper"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera device index (default: from config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip calibration load; use identity matrices"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record raw session video to outputs/session_<timestamp>.mp4"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for logs and recordings"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING"],
        help="DEBUG | INFO | WARNING (default: from config)"
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=5,
        help="Write outputs/status.json every N frames (0 disables; default: 5)"
    )
    parser.add_argument(
        "--no-frame-dump",
        action="store_true",
        help="Skip writing outputs/latest.jpg (status JSON still written)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Score a still image instead of the camera (default: camera, falling back "
             "to the last saved frame when no camera is connected)"
    )
    parser.add_argument(
        "--web-data-dir",
        type=str,
        default=None,
        help="Override directory for status.json and latest.jpg (default: dir of session log)"
    )
    return parser.parse_args()


def main():
    load_dotenv(ENV_FILE)
    args = parse_args()

    cfg = load_config(args.config)

    cli_overrides = {}
    if args.camera_index is not None:
        cli_overrides["camera"] = {"index": args.camera_index}
    if args.output_dir is not None:
        cli_overrides["output"] = {"log_file": str(Path(args.output_dir) / "session.jsonl")}
    if args.log_level is not None:
        cli_overrides["output"] = {**(cli_overrides.get("output", {})), "log_level": args.log_level}

    cfg = override_config(cfg, cli_overrides)

    log_level = cfg.get("output", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    skip_calibration = args.no_calibration or cfg["calibration"].get("skip", False)
    if skip_calibration:
        calibration = identity_calibration()
        reason = "--no-calibration" if args.no_calibration else "calibration.skip=true"
        logger.info(f"Using identity calibration ({reason})")
    else:
        try:
            calibration = load_calibration(cfg["calibration"]["file"])
            logger.info(f"Loaded calibration from {cfg['calibration']['file']}")
        except FileNotFoundError as e:
            logger.error(f"{e}")
            sys.exit(1)

    radius_override = cfg["detection"].get("expected_radius_px")
    if radius_override is not None:
        expected_radius_px = float(radius_override)
        logger.info(f"Using detection.expected_radius_px override: {expected_radius_px} px")
    else:
        expected_radius_px = compute_expected_radius_px(calibration)

    tracker = BallTracker(
        method=cfg["tracking"]["method"],
        max_reacquire=cfg["tracking"].get("max_reacquire_frames", 5)
    )
    engine = MetricsEngine(calibration, expected_radius_px)
    scorekeeper = ScoreKeeper()
    drill = DrillTracker()
    logger_output = SessionLogger(
        cfg["output"]["log_file"],
        cfg["output"]["flush_interval_s"]
    )
    # table_polygon is only drawn while tuning; PolygonTuner sets it each frame
    renderer = OverlayRenderer(enabled=cfg["output"]["overlay"])

    polygon_tuner = None
    if polygon_tuning_enabled():
        default_polygon = [[0, 0], [cfg["camera"]["width"], 0],
                           [cfg["camera"]["width"], cfg["camera"]["height"]],
                           [0, cfg["camera"]["height"]]]
        polygon_tuner = PolygonTuner(
            cfg["detection"].get("table_polygon") or default_polygon,
            cfg["camera"]["width"],
            cfg["camera"]["height"],
        )
        logger.info(f"{TUNE_POLYGON_ENV} on: table_polygon sliders enabled, press 'p' to print")

    if args.web_data_dir is not None:
        web_data_dir = Path(args.web_data_dir)
    else:
        web_data_dir = Path(cfg["output"]["log_file"]).parent

    cap, source_info = open_frame_source(args, cfg, web_data_dir, logger)
    try:
        actual_fps = cap.actual_fps()
        if source_info["kind"] == "camera":
            logger.info(f"Camera opened at {actual_fps} fps")
        else:
            logger.info(f"Scoring still image {source_info['path']} (not live)")

        status_writer = StatusWriter(
            data_dir=web_data_dir,
            status_interval=args.status_interval,
            # Don't redraw markers onto latest.jpg when it is itself the still being scored.
            write_frames=not args.no_frame_dump and not source_info.get("annotated", False),
            calibration=calibration,
            target_fps=actual_fps,
            source=source_info,
            save_raw_frames=source_info["kind"] == "camera",
        )

        frame_index = 0
        last_shot = None

        while True:
            ok, frame = cap.read_frame()
            if not ok:
                logger.info("End of video stream")
                break

            if polygon_tuner:
                live_polygon = polygon_tuner.read()
                cfg["detection"]["table_polygon"] = live_polygon
                renderer.table_polygon = np.array(live_polygon, dtype=np.int32)

            undistorted = undistort_frame(frame, calibration)
            detections = detect_balls(undistorted, cfg["detection"], expected_radius_px)
            tracked = tracker.update(detections)
            cb = tracker.cue_ball()
            player = player_over_table(undistorted, cfg["detection"].get("table_polygon"), detections, expected_radius_px)
            scorekeeper.update_detections(detections, time.perf_counter(), player_present=player)
            drill.update(detections, time.perf_counter(), cfg["detection"].get("table_polygon"))

            shot = engine.update(cb, frame_index, time.perf_counter())
            if shot:
                logger_output.log_shot(shot)
                print_shot_summary(shot)
                last_shot = shot

            annotated = renderer.render(undistorted, tracked)
            cv2.imshow(APP_WINDOW, annotated)

            status_writer.tick()
            if status_writer.should_write(frame_index):
                # Switch drills when one is chosen on the Drills page.
                chosen = read_selection(web_data_dir)
                if chosen and chosen != drill.drill.id:
                    drill = DrillTracker(DRILLS[chosen])
                    logger.info(f"Drill: {drill.drill.name}")
                # The dashboard shows the plain frame; detection markers stay in the
                # capture window only.
                status_writer.write(
                    frame_index, tracked, detections, engine, last_shot, undistorted,
                    score=scorekeeper.snapshot(), drill=drill.snapshot(), raw_frame=undistorted,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received")
                break
            elif key == ord('s'):
                logger.info(f"Saved still: {save_still(undistorted)}")
            elif key == ord('p') and polygon_tuner:
                PolygonTuner.print_yaml(live_polygon)
            elif key == ord('o'):
                renderer.toggle()
                logger.info(f"Overlay toggled: {renderer.enabled}")
            elif key == ord('t'):
                if cb:
                    pos_in = pixel_to_inches(np.array([cb.center_px]), calibration)[0]
                    engine.set_target(tuple(pos_in))
                    logger.info(f"Target set at {pos_in}")

            frame_index += 1
    finally:
        cap.release()

    logger_output.close()
    cv2.destroyAllWindows()
    logger.info("Session closed cleanly")


if __name__ == "__main__":
    main()
