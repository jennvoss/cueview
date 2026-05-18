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

from src.config_loader import load_config, override_config
from src.capture import CameraCapture, find_camera_index, CameraNotFoundError
from src.calibration import load_calibration, undistort_frame, pixel_to_inches
from src.detection import detect_balls, compute_expected_radius_px
from src.tracking import BallTracker
from src.metrics import MetricsEngine
from src.output import SessionLogger, OverlayRenderer


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
    print(f"Shot recorded: peak={shot.peak_speed_mph:.2f} mph, "
          f"mean={shot.mean_speed_mph:.2f} mph, angle={shot.angle_deg:.1f}°")


class StatusWriter:
    def __init__(self, data_dir: Path, status_interval: int, write_frames: bool,
                 calibration: dict, target_fps: float):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.data_dir / "status.json"
        self.status_tmp = self.data_dir / "status.json.tmp"
        self.frame_path = self.data_dir / "latest.jpg"
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
              engine: MetricsEngine, last_shot, annotated_frame) -> None:
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

            status = {
                "status": "running",
                "frame_index": int(frame_index),
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "target_fps": self.target_fps,
                "measured_fps": round(self._measured_fps(), 2),
                "detections_this_frame": int(len(detections)),
                "tracked_balls": tracked_balls,
                "current_speed_mph": float(engine.current_smoothed_speed_mph),
                "peak_speed_mph": float(engine.peak_speed_mph),
                "is_in_shot": bool(engine.is_in_shot),
                "target_pos_in": target_pos_in,
                "last_shot": last_shot_dict,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pool Shot Capture — real-time ball tracking and metrics"
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
        "--web-data-dir",
        type=str,
        default=None,
        help="Override directory for status.json and latest.jpg (default: dir of session log)"
    )
    return parser.parse_args()


def main():
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
    engine = MetricsEngine(calibration, cfg["metrics"]["smoother_window"])
    logger_output = SessionLogger(
        cfg["output"]["log_file"],
        cfg["output"]["flush_interval_s"]
    )
    renderer = OverlayRenderer(
        enabled=cfg["output"]["overlay"],
        table_polygon=cfg["detection"].get("table_polygon"),
    )
    trail = deque(maxlen=30)

    try:
        camera_index = find_camera_index(cfg)
    except CameraNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    with CameraCapture(
        camera_index,
        cfg["camera"]["width"],
        cfg["camera"]["height"],
        cfg["camera"]["fps"]
    ) as cap:
        actual_fps = cap.actual_fps()
        logger.info(f"Camera opened at {actual_fps} fps")

        if args.web_data_dir is not None:
            web_data_dir = Path(args.web_data_dir)
        else:
            web_data_dir = Path(cfg["output"]["log_file"]).parent
        status_writer = StatusWriter(
            data_dir=web_data_dir,
            status_interval=args.status_interval,
            write_frames=not args.no_frame_dump,
            calibration=calibration,
            target_fps=actual_fps,
        )

        frame_index = 0
        last_shot = None

        while True:
            ok, frame = cap.read_frame()
            if not ok:
                logger.info("End of video stream")
                break

            undistorted = undistort_frame(frame, calibration)
            detections = detect_balls(undistorted, cfg["detection"], expected_radius_px)
            tracked = tracker.update(detections)
            cb = tracker.cue_ball()

            if cb:
                trail.append(cb.center_px)

            shot = engine.update(cb, frame_index, time.perf_counter())
            if shot:
                logger_output.log_shot(shot)
                print_shot_summary(shot)
                last_shot = shot

            annotated = renderer.render(
                undistorted,
                tracked,
                engine.current_speed_mph(),
                last_shot,
                list(trail)
            )
            cv2.imshow("Pool Shot Capture", annotated)

            status_writer.tick()
            if status_writer.should_write(frame_index):
                status_writer.write(
                    frame_index, tracked, detections, engine, last_shot, annotated
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received")
                break
            elif key == ord('o'):
                renderer.toggle()
                logger.info(f"Overlay toggled: {renderer.enabled}")
            elif key == ord('t'):
                if cb:
                    pos_in = pixel_to_inches(np.array([cb.center_px]), calibration)[0]
                    engine.set_target(tuple(pos_in))
                    logger.info(f"Target set at {pos_in}")

            frame_index += 1

    logger_output.close()
    cv2.destroyAllWindows()
    logger.info("Session closed cleanly")


if __name__ == "__main__":
    main()
