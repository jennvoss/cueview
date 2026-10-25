from dataclasses import dataclass
from typing import Optional
import numpy as np
from filterpy.kalman import KalmanFilter

from src.constants import MAX_REACQUIRE_FRAMES
from src.detection import Detection


@dataclass
class TrackedBall:
    id: int
    center_px: tuple[float, float]
    is_cue_ball: bool
    frames_missing: int


class BallTracker:
    def __init__(self, method: str = "kalman", max_reacquire: int = MAX_REACQUIRE_FRAMES):
        self.method = method
        self.max_reacquire = max_reacquire
        self.next_id = 0
        self.tracks = {}
        self.cue_ball_id = None
        self.cue_ball_kf = None

    def update(self, detections: list[Detection]) -> list[TrackedBall]:
        """
        Match detections to existing tracked balls by nearest-centroid.
        For cue ball specifically:
          - Re-use existing cue ball ID if distance < 2 × expected_radius_px.
          - Increment frames_missing if cue ball not detected this frame.
          - Purge cue ball track if frames_missing > max_reacquire.
        Returns updated list of TrackedBall. Never raises.
        """
        # Find cue ball detection
        cue_ball_detection = None
        other_detections = []

        for det in detections:
            if det.is_cue_ball:
                cue_ball_detection = det
            else:
                other_detections.append(det)

        # Handle cue ball tracking with Kalman filter
        if self.method == "kalman":
            if cue_ball_detection is not None:
                if self.cue_ball_id is None:
                    # Initialize new cue ball track
                    self.cue_ball_id = self.next_id
                    self.next_id += 1
                    self.cue_ball_kf = self._create_kalman_filter()
                    x, y = cue_ball_detection.center_px
                    self.cue_ball_kf.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
                else:
                    # Update existing cue ball track
                    x, y = cue_ball_detection.center_px
                    self.cue_ball_kf.predict()
                    self.cue_ball_kf.update(np.array([[x], [y]], dtype=np.float32))

                # Report the filtered estimate, not the raw (noisy) detection
                state = self.cue_ball_kf.x
                self.tracks[self.cue_ball_id] = {
                    'is_cue_ball': True,
                    'frames_missing': 0,
                    'last_pos': (float(state[0, 0]), float(state[1, 0])),
                }
            else:
                # No cue ball detection this frame
                if self.cue_ball_id is not None:
                    if self.cue_ball_kf is not None:
                        self.cue_ball_kf.predict()
                        predicted_state = self.cue_ball_kf.x
                        predicted_pos = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
                    else:
                        predicted_pos = self.tracks[self.cue_ball_id].get('last_pos', (0, 0))

                    self.tracks[self.cue_ball_id]['frames_missing'] += 1
                    self.tracks[self.cue_ball_id]['last_pos'] = predicted_pos

                    if self.tracks[self.cue_ball_id]['frames_missing'] > self.max_reacquire:
                        del self.tracks[self.cue_ball_id]
                        self.cue_ball_id = None
                        self.cue_ball_kf = None
        else:
            # Centroid method
            if cue_ball_detection is not None:
                if self.cue_ball_id is None:
                    self.cue_ball_id = self.next_id
                    self.next_id += 1

                self.tracks[self.cue_ball_id] = {
                    'is_cue_ball': True,
                    'frames_missing': 0,
                    'last_pos': cue_ball_detection.center_px,
                }
            else:
                if self.cue_ball_id is not None:
                    self.tracks[self.cue_ball_id]['frames_missing'] += 1

                    if self.tracks[self.cue_ball_id]['frames_missing'] > self.max_reacquire:
                        del self.tracks[self.cue_ball_id]
                        self.cue_ball_id = None

        # Match other detections to existing tracks
        unmatched_detections = list(range(len(other_detections)))
        unmatched_track_ids = list(self.tracks.keys())

        if self.cue_ball_id in unmatched_track_ids:
            unmatched_track_ids.remove(self.cue_ball_id)

        # Greedy nearest-centroid matching
        for tid in list(unmatched_track_ids):
            best_dist = float('inf')
            best_det_idx = None

            track_pos = self.tracks[tid]['last_pos']

            for det_idx in unmatched_detections:
                det_pos = other_detections[det_idx].center_px
                dist = ((track_pos[0] - det_pos[0]) ** 2 + (track_pos[1] - det_pos[1]) ** 2) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            if best_det_idx is not None and best_dist < 50:  # reasonable threshold
                det = other_detections[best_det_idx]
                self.tracks[tid]['last_pos'] = det.center_px
                self.tracks[tid]['frames_missing'] = 0
                unmatched_detections.remove(best_det_idx)
                unmatched_track_ids.remove(tid)
            else:
                self.tracks[tid]['frames_missing'] += 1
                if self.tracks[tid]['frames_missing'] > 10:
                    del self.tracks[tid]
                    unmatched_track_ids.remove(tid)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = other_detections[det_idx]
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                'is_cue_ball': False,
                'frames_missing': 0,
                'last_pos': det.center_px,
            }

        # Build result list
        result = []
        for track_id, track_data in self.tracks.items():
            result.append(TrackedBall(
                id=track_id,
                center_px=track_data['last_pos'],
                is_cue_ball=track_data['is_cue_ball'],
                frames_missing=track_data['frames_missing'],
            ))

        return result

    def cue_ball(self) -> Optional[TrackedBall]:
        """Return current cue ball TrackedBall, or None if not tracked."""
        if self.cue_ball_id is not None and self.cue_ball_id in self.tracks:
            track_data = self.tracks[self.cue_ball_id]
            return TrackedBall(
                id=self.cue_ball_id,
                center_px=track_data['last_pos'],
                is_cue_ball=True,
                frames_missing=track_data['frames_missing'],
            )
        return None

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks = {}
        self.cue_ball_id = None
        self.cue_ball_kf = None

    def _create_kalman_filter(self) -> KalmanFilter:
        """Create and configure a Kalman filter for cue ball tracking."""
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, vx, vy]
        kf.x = np.array([[0], [0], [0], [0]], dtype=np.float32)

        # Constant-velocity transition matrix
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Measurement noise
        kf.R = np.eye(2, dtype=np.float32) * 5.0

        # Process noise
        kf.Q = np.eye(4, dtype=np.float32) * 1.0

        # State covariance
        kf.P = np.eye(4, dtype=np.float32)

        return kf
