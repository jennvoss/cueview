import math
import logging
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from src.constants import REST_THRESHOLD_IPS, SHOT_START_MPH, SMOOTHER_WINDOW
from src.calibration import pixel_to_inches

logger = logging.getLogger(__name__)


@dataclass
class ShotRecord:
    timestamp_iso: str
    frame_start: int
    frame_end: int
    peak_speed_mph: float
    mean_speed_mph: float
    angle_deg: float
    start_pos_in: tuple
    end_pos_in: tuple
    accuracy_in: Optional[float]
    target_pos_in: Optional[tuple]


def ips_to_mph(ips: float) -> float:
    return ips / 17.6


class MetricsEngine:
    def __init__(self, calibration: dict, smoother_window: int = SMOOTHER_WINDOW):
        self.calibration = calibration
        self.smoother_window = smoother_window

        self.speeds_history = deque(maxlen=smoother_window)
        self.positions_history = deque(maxlen=100)
        self.timestamps_history = deque(maxlen=100)

        self.is_in_shot = False
        self.shot_start_frame = None
        self.shot_start_pos_in = None
        self.shot_start_timestamp = None
        self.peak_speed_mph = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0

        self.target_pos_in = None
        self.current_smoothed_speed_mph = 0.0

    def update(
        self,
        tracked_ball,
        frame_index: int,
        timestamp: float,
    ):
        if tracked_ball is None:
            return None

        center_px = np.array([tracked_ball.center_px], dtype=np.float32)
        center_in = pixel_to_inches(center_px, self.calibration)[0]

        self.positions_history.append(center_in)
        self.timestamps_history.append(timestamp)

        instantaneous_speed_ips = self._compute_instantaneous_speed()
        instantaneous_speed_mph = ips_to_mph(instantaneous_speed_ips)

        self.speeds_history.append(instantaneous_speed_mph)
        self.current_smoothed_speed_mph = sum(self.speeds_history) / len(self.speeds_history)

        is_at_rest = self.current_smoothed_speed_mph < ips_to_mph(REST_THRESHOLD_IPS)

        shot_record = None

        if not self.is_in_shot and not is_at_rest and self.current_smoothed_speed_mph > SHOT_START_MPH:
            self.is_in_shot = True
            self.shot_start_frame = frame_index
            self.shot_start_pos_in = tuple(center_in)
            self.shot_start_timestamp = timestamp
            self.peak_speed_mph = self.current_smoothed_speed_mph
            self.speed_sum = self.current_smoothed_speed_mph
            self.speed_count = 1
        elif self.is_in_shot:
            self.peak_speed_mph = max(self.peak_speed_mph, self.current_smoothed_speed_mph)
            self.speed_sum += self.current_smoothed_speed_mph
            self.speed_count += 1

            if is_at_rest:
                shot_record = self._build_shot_record(
                    frame_index,
                    timestamp,
                    center_in,
                )
                self.is_in_shot = False
                self.target_pos_in = None

        return shot_record

    def _compute_instantaneous_speed(self) -> float:
        if len(self.positions_history) < 2:
            return 0.0

        prev_pos = self.positions_history[-2]
        curr_pos = self.positions_history[-1]

        distance_in = math.dist(prev_pos, curr_pos)

        prev_time = self.timestamps_history[-2]
        curr_time = self.timestamps_history[-1]
        time_delta = curr_time - prev_time

        if time_delta <= 0:
            return 0.0

        speed_ips = distance_in / time_delta
        return speed_ips

    def _build_shot_record(self, frame_end: int, timestamp: float, end_pos_in: tuple) -> ShotRecord:
        mean_speed_mph = self.speed_sum / self.speed_count if self.speed_count > 0 else 0.0

        angle_deg = self._compute_angle()

        accuracy_in = None
        if self.target_pos_in is not None:
            accuracy_in = math.dist(end_pos_in, self.target_pos_in)

        timestamp_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        return ShotRecord(
            timestamp_iso=timestamp_iso,
            frame_start=self.shot_start_frame,
            frame_end=frame_end,
            peak_speed_mph=self.peak_speed_mph,
            mean_speed_mph=mean_speed_mph,
            angle_deg=angle_deg,
            start_pos_in=self.shot_start_pos_in,
            end_pos_in=tuple(end_pos_in),
            accuracy_in=accuracy_in,
            target_pos_in=self.target_pos_in,
        )

    def _compute_angle(self) -> float:
        if len(self.positions_history) < 2:
            return 0.0

        start_pos = self.positions_history[0]
        end_pos = self.positions_history[-1]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def set_target(self, target_pos_in: tuple) -> None:
        self.target_pos_in = target_pos_in

    def current_speed_mph(self) -> float:
        return self.current_smoothed_speed_mph
