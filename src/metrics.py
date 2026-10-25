import math
import logging
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from src.constants import REST_WINDOW_S, REST_MAX_SPREAD_RADII, SHOT_START_RADII
from src.calibration import pixel_to_inches

logger = logging.getLogger(__name__)


@dataclass
class ShotRecord:
    timestamp_iso: str
    frame_start: int
    frame_end: int
    angle_deg: float
    start_pos_in: tuple
    end_pos_in: tuple
    accuracy_in: Optional[float]
    target_pos_in: Optional[tuple]


def _to_float_tuple(pos) -> Optional[tuple]:
    # pixel_to_inches yields numpy float32, which json can't serialize
    if pos is None:
        return None
    return tuple(float(v) for v in pos)


class MetricsEngine:
    """
    Detects shots from how far the cue ball moves, measured in ball radii (px),
    so it works without calibration and at any frame rate.

      - At rest: every position in the last REST_WINDOW_S stays within
        REST_MAX_SPREAD_RADII of the window's centroid.
      - Shot start: ball moves more than SHOT_START_RADII from where it last rested.
      - Shot end: ball is at rest again.
    """

    def __init__(self, calibration: dict, ball_radius_px: float):
        self.calibration = calibration
        self.ball_radius_px = float(ball_radius_px)

        self.window = deque()  # (timestamp, center_px) covering the last REST_WINDOW_S
        self.rest_anchor_px = None

        self.is_in_shot = False
        self.shot_start_frame = None
        self.shot_start_pos_in = None
        self.shot_start_timestamp = None

        self.target_pos_in = None

    def update(
        self,
        tracked_ball,
        frame_index: int,
        timestamp: float,
    ):
        if tracked_ball is None:
            return None

        center_px = (float(tracked_ball.center_px[0]), float(tracked_ball.center_px[1]))
        self._push(timestamp, center_px)
        at_rest, rest_center_px = self._rest_state(timestamp)

        if not self.is_in_shot:
            if self.rest_anchor_px is None or at_rest:
                self.rest_anchor_px = rest_center_px if at_rest else center_px
            if math.dist(center_px, self.rest_anchor_px) > SHOT_START_RADII * self.ball_radius_px:
                self.is_in_shot = True
                self.shot_start_frame = frame_index
                self.shot_start_pos_in = self._to_inches(self.rest_anchor_px)
                self.shot_start_timestamp = timestamp
            return None

        if not at_rest:
            return None

        shot_record = self._build_shot_record(frame_index, timestamp, self._to_inches(center_px))
        self.is_in_shot = False
        self.rest_anchor_px = rest_center_px
        self.target_pos_in = None
        return shot_record

    def _push(self, timestamp: float, center_px: tuple) -> None:
        self.window.append((timestamp, center_px))
        # Keep one sample at or before the cutoff so we know the window is fully covered
        cutoff = timestamp - REST_WINDOW_S
        while len(self.window) > 1 and self.window[1][0] <= cutoff:
            self.window.popleft()

    def _rest_state(self, timestamp: float):
        points = np.array([p for _, p in self.window])
        centroid = (float(points[:, 0].mean()), float(points[:, 1].mean()))
        covered = timestamp - self.window[0][0] >= REST_WINDOW_S - 1e-9
        spread = float(np.max(np.hypot(points[:, 0] - centroid[0], points[:, 1] - centroid[1])))
        at_rest = covered and spread <= REST_MAX_SPREAD_RADII * self.ball_radius_px
        return at_rest, centroid

    def _to_inches(self, center_px: tuple) -> tuple:
        return tuple(pixel_to_inches(np.array([center_px], dtype=np.float32), self.calibration)[0])

    def _build_shot_record(self, frame_end: int, timestamp: float, end_pos_in: tuple) -> ShotRecord:
        dx = end_pos_in[0] - self.shot_start_pos_in[0]
        dy = end_pos_in[1] - self.shot_start_pos_in[1]
        angle_deg = math.degrees(math.atan2(dy, dx))

        accuracy_in = None
        if self.target_pos_in is not None:
            accuracy_in = float(math.dist(end_pos_in, self.target_pos_in))

        timestamp_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        return ShotRecord(
            timestamp_iso=timestamp_iso,
            frame_start=self.shot_start_frame,
            frame_end=frame_end,
            angle_deg=float(angle_deg),
            start_pos_in=_to_float_tuple(self.shot_start_pos_in),
            end_pos_in=_to_float_tuple(end_pos_in),
            accuracy_in=accuracy_in,
            target_pos_in=_to_float_tuple(self.target_pos_in),
        )

    def set_target(self, target_pos_in: tuple) -> None:
        self.target_pos_in = target_pos_in
