import math

import pytest

from src.config_loader import load_config
from src.drills import DrillTracker, STOP_SHOT, TableGeometry

POLYGON = load_config("config/config.yaml")["detection"]["table_polygon"]
GEO = TableGeometry(POLYGON)
FPS = 10.0
CUE_MARK = GEO.point(STOP_SHOT.cue_start)
OBJECT_MARK = GEO.point(STOP_SHOT.object_start)
POCKET = GEO.point(STOP_SHOT.pocket)
IN = GEO.px_per_in


class Det:
    def __init__(self, x, y, cue=False):
        self.center_px, self.is_cue_ball = (x, y), cue


class Sim:
    """Feeds the tracker frame by frame and keeps the clock."""

    def __init__(self):
        self.tracker = DrillTracker(STOP_SHOT)
        self.t = 0.0

    def frames(self, seconds, cue=None, obj=None):
        for _ in range(int(seconds * FPS)):
            self.step(cue, obj)

    def step(self, cue=None, obj=None):
        dets = ([Det(*cue, cue=True)] if cue else []) + ([Det(*obj)] if obj else [])
        self.tracker.update(dets, self.t, POLYGON)
        self.t += 1 / FPS

    def roll(self, seconds, cue_path=None, obj_path=None):
        """Move balls linearly along (start, end) paths; None end = ball disappears at the end."""
        n = int(seconds * FPS)
        for i in range(1, n + 1):
            f = i / n
            pos = lambda path: None if path is None or path[1] is None and f == 1 else (
                path[0] if path[1] is None else (path[0][0] + (path[1][0] - path[0][0]) * f,
                                                path[0][1] + (path[1][1] - path[0][1]) * f))
            self.step(pos(cue_path), pos(obj_path))


def setup_ready():
    sim = Sim()
    sim.frames(0.5)
    assert sim.tracker.state == "setup"
    sim.frames(1.5, cue=CUE_MARK, obj=OBJECT_MARK)
    assert sim.tracker.state == "ready"
    return sim


def offset(point, up_in=0.0, right_in=0.0):
    """Move along the shot line (up the table toward the pocket) and to the shooter's right."""
    return (point[0] + right_in * IN, point[1] - up_in * IN)


def test_waits_for_balls_on_their_marks():
    sim = Sim()
    sim.frames(2, cue=offset(CUE_MARK, right_in=8), obj=OBJECT_MARK)
    snap = sim.tracker.snapshot()
    assert snap["state"] == "setup"
    assert snap["placed"] == {"cue": False, "object": True}
    sim.frames(0.5, cue=CUE_MARK, obj=OBJECT_MARK)
    assert sim.tracker.state == "setup"  # must sit still for DRILL_SETTLE_S first
    sim.frames(1.0, cue=CUE_MARK, obj=OBJECT_MARK)
    assert sim.tracker.state == "ready"


def test_snapshot_has_marks_and_instructions():
    snap = setup_ready().tracker.snapshot()
    assert snap["name"] == "Stop shot" and snap["speed"] == 3 and snap["steps"]
    for key in ("cue_start", "object_start", "pocket", "cue_target"):
        assert len(snap["marks"][key]) == 2
    assert snap["marks"]["ball_radius_px"] > 5


def test_perfect_stop_shot():
    sim = setup_ready()
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, POCKET))
    sim.frames(0.3, cue=OBJECT_MARK)  # object ball dropped
    sim.frames(2.0, cue=OBJECT_MARK)
    result = sim.tracker.snapshot()["result"]
    assert sim.tracker.state == "result"
    assert result["rating"] == "perfect"
    assert result["pocketed"] and result["pocket"] == "top side"
    assert result["cue_error_in"] < 1
    assert sim.tracker.snapshot()["stats"] == {"attempts": 1, "made": 1, "perfect": 1}


def test_follow_through_is_explained():
    sim = setup_ready()
    end = offset(OBJECT_MARK, up_in=8)
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, end), obj_path=(OBJECT_MARK, POCKET))
    sim.frames(2.5, cue=end)
    result = sim.tracker.snapshot()["result"]
    assert result["rating"] == "pocketed"
    assert any("rolled 8 in past" in n and "lower" in n for n in result["notes"])


def test_missed_right_of_pocket():
    sim = setup_ready()
    miss = offset(OBJECT_MARK, up_in=8, right_in=6)
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, miss))
    sim.frames(2.5, cue=OBJECT_MARK, obj=miss)
    result = sim.tracker.snapshot()["result"]
    assert result["rating"] == "missed" and not result["pocketed"]
    assert any("missed right" in n and "aim left" in n for n in result["notes"])


def test_too_soft():
    sim = setup_ready()
    short = offset(OBJECT_MARK, up_in=4)
    sim.roll(0.5, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.5, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, short))
    sim.frames(2.5, cue=OBJECT_MARK, obj=short)
    result = sim.tracker.snapshot()["result"]
    assert result["rating"] == "missed"
    assert any("short of the pocket" in n and "harder" in n for n in result["notes"])


def test_scratch():
    sim = setup_ready()
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, POCKET), obj_path=(OBJECT_MARK, POCKET))
    sim.frames(2.5)
    result = sim.tracker.snapshot()["result"]
    assert result["rating"] == "scratch"
    assert any("Scratch" in n for n in result["notes"])


def test_moving_balls_by_hand_returns_to_setup():
    sim = setup_ready()
    sim.frames(0.5)  # both balls picked up
    assert sim.tracker.state == "setup"


def test_new_attempt_after_result():
    sim = setup_ready()
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, POCKET))
    sim.frames(2.5, cue=OBJECT_MARK)
    assert sim.tracker.state == "result"
    sim.frames(1.5, cue=CUE_MARK, obj=OBJECT_MARK)  # set up again
    snap = sim.tracker.snapshot()
    assert snap["state"] == "ready"
    assert snap["result"] is not None  # last result stays visible
    assert snap["stats"]["attempts"] == 1


def test_geometry_maps_corners_and_scale():
    assert GEO.point((0, 0)) == pytest.approx(tuple(map(float, POLYGON[0])))
    assert GEO.point((8, 4)) == pytest.approx(tuple(map(float, POLYGON[2])))
    assert 9 < GEO.px_per_in < 13


def test_moving_only_the_cue_ball_is_not_an_attempt():
    """Picking the cue ball up to re-spot it leaves the object ball untouched: no attempt."""
    sim = setup_ready()
    sim.frames(3.0, obj=OBJECT_MARK)
    snap = sim.tracker.snapshot()
    assert snap["state"] == "setup"
    assert snap["result"] is None and snap["stats"]["attempts"] == 0


def test_pocket_not_confused_by_other_detections():
    """After the object ball drops in the top side pocket, a mislabelled cue ball / hand near
    the bottom of the table must not be taken for it (it was reported as the bottom side pocket)."""
    sim = setup_ready()
    bottom = offset(CUE_MARK, up_in=-8, right_in=-4)  # near the shooter, by the bottom side pocket
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, offset(OBJECT_MARK, up_in=2, right_in=4)), obj_path=(OBJECT_MARK, POCKET))
    for _ in range(int(2.5 * FPS)):
        sim.step(cue=offset(OBJECT_MARK, up_in=2, right_in=4), obj=bottom)  # spurious non-cue detection
    result = sim.tracker.snapshot()["result"]
    assert result["pocketed"] is True and result["pocket"] == "top side"
    assert not any("instead of" in n for n in result["notes"])


def test_cue_ball_seen_twice_is_not_the_object_ball():
    """A second, unflagged detection on top of the cue ball is ignored as the object ball."""
    sim = setup_ready()
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.4, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, POCKET))
    for _ in range(int(2.5 * FPS)):
        sim.step(cue=OBJECT_MARK, obj=offset(OBJECT_MARK, right_in=0.5))  # the cue ball, mislabelled
    result = sim.tracker.snapshot()["result"]
    assert result["pocketed"] is True and result["pocket"] == "top side"


def test_object_ball_lost_mid_roll_uses_its_heading():
    """Blurred out of sight halfway to the pocket: credited to the pocket it was heading for."""
    sim = setup_ready()
    halfway = offset(OBJECT_MARK, up_in=5)
    sim.roll(0.4, cue_path=(CUE_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, OBJECT_MARK))
    sim.roll(0.3, cue_path=(OBJECT_MARK, OBJECT_MARK), obj_path=(OBJECT_MARK, halfway))
    sim.frames(2.5, cue=OBJECT_MARK)
    result = sim.tracker.snapshot()["result"]
    assert result["pocketed"] is True and result["pocket"] == "top side"
