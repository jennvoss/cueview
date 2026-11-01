import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

REQUIRED_LOG_FIELDS = {
    "timestamp_iso", "frame_start", "frame_end",
    "angle_deg",
    "start_pos_in", "end_pos_in", "accuracy_in", "target_pos_in",
}


@dataclass
class ShotRecord:
    timestamp_iso: str
    frame_start: int
    frame_end: int
    angle_deg: float
    start_pos_in: Tuple[float, float]
    end_pos_in: Tuple[float, float]
    accuracy_in: Optional[float]
    target_pos_in: Optional[Tuple[float, float]]


class SessionLogger:
    def __init__(self, log_path: str, flush_interval_s: float = 30.0):
        self.log_path = log_path
        self.flush_interval_s = flush_interval_s
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.file = open(log_path, "a")

    def log_shot(self, record: ShotRecord) -> None:
        record_dict = asdict(record)
        record_dict["start_pos_in"] = list(record.start_pos_in)
        record_dict["end_pos_in"] = list(record.end_pos_in)
        if record.target_pos_in:
            record_dict["target_pos_in"] = list(record.target_pos_in)

        missing_fields = REQUIRED_LOG_FIELDS - set(record_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        json_line = json.dumps(record_dict)
        self.file.write(json_line + "\n")

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.flush()
        self.file.close()


# Drawn ball circles are smoothed per track id so Hough center noise doesn't
# make them shake; a move larger than SNAP_DISTANCE_PX (a real roll) jumps
# straight to the new position instead of lagging behind it.
SMOOTHING_ALPHA = 0.25
SNAP_DISTANCE_PX = 20.0


class OverlayRenderer:
    def __init__(self, enabled: bool = True, table_polygon: Optional[List] = None):
        self.enabled = enabled
        self.table_polygon = None
        if table_polygon:
            self.table_polygon = np.array(table_polygon, dtype=np.int32)
        self._smoothed_centers = {}

    def _smooth_center(self, track_id, center: Tuple[float, float]) -> Tuple[float, float]:
        if track_id is None:
            return center
        prev = self._smoothed_centers.get(track_id)
        if prev is None or np.hypot(center[0] - prev[0], center[1] - prev[1]) > SNAP_DISTANCE_PX:
            smoothed = (float(center[0]), float(center[1]))
        else:
            smoothed = (prev[0] + SMOOTHING_ALPHA * (center[0] - prev[0]),
                        prev[1] + SMOOTHING_ALPHA * (center[1] - prev[1]))
        self._smoothed_centers[track_id] = smoothed
        return smoothed

    def render(
        self,
        frame: np.ndarray,
        tracked_balls: List,
    ) -> np.ndarray:
        if not self.enabled:
            return frame

        frame = frame.copy() if frame is not None else frame

        if frame is None or frame.size == 0:
            return frame

        if self.table_polygon is not None:
            cv2.polylines(frame, [self.table_polygon], isClosed=True,
                          color=(0, 165, 255), thickness=2)

        live_ids = set()
        for ball in tracked_balls:
            track_id = getattr(ball, "id", None)
            live_ids.add(track_id)
            center = tuple(int(round(v)) for v in self._smooth_center(track_id, ball.center_px))
            radius = int(max(1, ball.radius_px if hasattr(ball, 'radius_px') else 10))

            color = (0, 255, 255) if ball.is_cue_ball else (0, 255, 0)
            cv2.circle(frame, center, radius, color, 2)

        for track_id in list(self._smoothed_centers):
            if track_id not in live_ids:
                del self._smoothed_centers[track_id]

        return frame

    def toggle(self) -> None:
        self.enabled = not self.enabled
