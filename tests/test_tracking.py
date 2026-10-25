import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracking import BallTracker, TrackedBall
from src.detection import Detection
from src.constants import MAX_REACQUIRE_FRAMES


class TestBallTracker:

    def test_linear_motion_single_cue_ball(self):
        """UT-TRK-01: Feed 10 frames of linearly moving cue ball (5 px/frame).
        Same ID throughout; final position within 3 px of expected."""
        tracker = BallTracker(method="kalman")

        # Start at (100, 100), move 5 px right each frame
        start_x, start_y = 100.0, 100.0
        velocity_x = 5.0

        tracked_balls = None
        for frame_idx in range(10):
            x = start_x + velocity_x * frame_idx
            y = start_y

            detection = Detection(
                center_px=(x, y),
                radius_px=20.0,
                is_cue_ball=True,
                color_label="cue_ball",
            )

            tracked_balls = tracker.update([detection])

        assert len(tracked_balls) == 1
        ball = tracked_balls[0]

        assert ball.is_cue_ball is True
        assert ball.frames_missing == 0

        expected_x = start_x + velocity_x * 9
        expected_y = start_y

        # Position should be within 3 px of expected
        distance = ((ball.center_px[0] - expected_x) ** 2 + (ball.center_px[1] - expected_y) ** 2) ** 0.5
        assert distance <= 3.0, f"Distance {distance} exceeds 3 px; pos={ball.center_px}, expected=({expected_x}, {expected_y})"

    def test_missing_and_reacquire(self):
        """UT-TRK-02: Feed 5 normal frames, then 4 frames with NO cue ball detection,
        then resume. Cue ball regains same ID on frame 10; frames_missing resets to 0."""
        tracker = BallTracker(method="kalman", max_reacquire=MAX_REACQUIRE_FRAMES)

        # Frames 0-4: cue ball present
        cue_ball_id = None
        for frame_idx in range(5):
            detection = Detection(
                center_px=(100.0 + frame_idx, 100.0),
                radius_px=20.0,
                is_cue_ball=True,
                color_label="cue_ball",
            )
            tracked_balls = tracker.update([detection])
            if cue_ball_id is None:
                cue_ball_id = tracked_balls[0].id

        # Frames 5-8: no cue ball detection
        for frame_idx in range(4):
            tracked_balls = tracker.update([])
            if len(tracked_balls) > 0:
                assert tracked_balls[0].id == cue_ball_id
                assert tracked_balls[0].frames_missing > 0

        # Frame 9: cue ball reappears
        detection = Detection(
            center_px=(109.0, 100.0),
            radius_px=20.0,
            is_cue_ball=True,
            color_label="cue_ball",
        )
        tracked_balls = tracker.update([detection])

        assert len(tracked_balls) == 1
        ball = tracked_balls[0]
        assert ball.id == cue_ball_id, "Cue ball should retain same ID after reacquisition"
        assert ball.frames_missing == 0, "frames_missing should reset to 0 on detection"

    def test_noisy_motion(self):
        """UT-TRK-03: Feed 100 frames with ±3 px Gaussian noise on cue ball position.
        No ID change (ID stable throughout)."""
        tracker = BallTracker(method="kalman")

        np.random.seed(42)
        start_x, start_y = 200.0, 200.0
        velocity_x = 2.0

        cue_ball_id = None
        for frame_idx in range(100):
            # Add Gaussian noise
            noise_x = np.random.normal(0, 3.0)
            noise_y = np.random.normal(0, 3.0)

            x = start_x + velocity_x * frame_idx + noise_x
            y = start_y + noise_y

            detection = Detection(
                center_px=(x, y),
                radius_px=20.0,
                is_cue_ball=True,
                color_label="cue_ball",
            )

            tracked_balls = tracker.update([detection])

            if cue_ball_id is None:
                cue_ball_id = tracked_balls[0].id
            else:
                assert tracked_balls[0].id == cue_ball_id, "Cue ball ID should remain stable"

            assert tracked_balls[0].is_cue_ball is True
            assert tracked_balls[0].frames_missing == 0

    def test_reports_filtered_position(self):
        """Stationary cue ball with ±3 px noise: reported position is the Kalman
        estimate, so it jitters less than the raw detections."""
        tracker = BallTracker(method="kalman")
        rng = np.random.default_rng(0)

        raw_x, reported_x = [], []
        for _ in range(200):
            x = 300.0 + rng.normal(0, 3.0)
            tracker.update([Detection(center_px=(x, 200.0), radius_px=20.0,
                                      is_cue_ball=True, color_label="cue_ball")])
            raw_x.append(x)
            reported_x.append(tracker.cue_ball().center_px[0])

        raw_jitter = np.mean(np.abs(np.diff(raw_x[20:])))
        reported_jitter = np.mean(np.abs(np.diff(reported_x[20:])))
        assert reported_jitter <= 0.7 * raw_jitter

    def test_reset(self):
        """Test that reset() clears all tracks."""
        tracker = BallTracker(method="kalman")

        detection = Detection(
            center_px=(100.0, 100.0),
            radius_px=20.0,
            is_cue_ball=True,
            color_label="cue_ball",
        )

        tracked_balls = tracker.update([detection])
        assert len(tracked_balls) == 1

        tracker.reset()

        # After reset, next detection should get a new ID
        tracked_balls_2 = tracker.update([detection])
        assert len(tracked_balls_2) == 1
        # ID should be different if tracker properly reset
        # (new ID would be incremented from the reset state)

    def test_cue_ball_method(self):
        """Test the cue_ball() method returns the current cue ball or None."""
        tracker = BallTracker(method="kalman")

        # Initially, no cue ball
        assert tracker.cue_ball() is None

        # Add a cue ball detection
        detection = Detection(
            center_px=(100.0, 100.0),
            radius_px=20.0,
            is_cue_ball=True,
            color_label="cue_ball",
        )

        tracker.update([detection])

        # Now we should have a cue ball
        cue_ball = tracker.cue_ball()
        assert cue_ball is not None
        assert cue_ball.is_cue_ball is True
        assert cue_ball.frames_missing == 0
