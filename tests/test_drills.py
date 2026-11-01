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
    assert any("missed right" in n and "left side" in n and "Aim a touch to the right" in n for n in result["notes"])


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


# -- other drills: the same simulator, driven by each drill's own marks -------------------

from src.drills import DRAW_SHOT  # noqa: E402


class DrillSim(Sim):
    def __init__(self, drill):
        self.tracker = DrillTracker(drill)
        self.t = 0.0
        self.cue_mark, self.object_mark = GEO.point(drill.cue_start), GEO.point(drill.object_start)
        self.pocket, self.target = GEO.point(drill.pocket), GEO.point(drill.cue_target)


def play(drill, cue_end, object_end=None, pocketed=True):
    """Set up, shoot: object ball to the pocket (or object_end), cue ball to cue_end (None = scratch)."""
    sim = DrillSim(drill)
    sim.frames(1.5, cue=sim.cue_mark, obj=sim.object_mark)
    assert sim.tracker.state == "ready"
    contact = sim.object_mark
    sim.roll(0.4, cue_path=(sim.cue_mark, contact), obj_path=(sim.object_mark, sim.object_mark))
    obj_to = sim.pocket if pocketed else object_end
    sim.roll(0.5, cue_path=(contact, cue_end if cue_end else sim.pocket), obj_path=(sim.object_mark, obj_to))
    sim.frames(2.5, cue=cue_end, obj=None if pocketed else object_end)
    return sim.tracker.snapshot()["result"]


def toward_shooter(point, inches):
    """Move down the table (toward the bottom rail, where the shooter stands)."""
    return (point[0], point[1] + inches * IN)


def test_draw_shot_perfect():
    result = play(DRAW_SHOT, GEO.point(DRAW_SHOT.cue_target))
    assert result["rating"] == "perfect" and result["pocketed"]
    assert "Lovely draw" in result["notes"][0]


def test_draw_shot_not_enough_draw():
    result = play(DRAW_SHOT, toward_shooter(GEO.point(DRAW_SHOT.object_start), 3))  # came back only 3 in
    assert result["rating"] == "pocketed"
    assert any("not enough draw" in n and "Hit lower" in n for n in result["notes"])


def test_draw_shot_too_much_draw():
    result = play(DRAW_SHOT, toward_shooter(GEO.point(DRAW_SHOT.cue_target), 8))
    assert any("drew back 8 in too far" in n for n in result["notes"])


from src.drills import FOLLOW_SHOT  # noqa: E402


def along_line(drill, point, inches):
    """Move along the drill's shot line (cue mark -> pocket); negative = back toward the shooter."""
    a, b = GEO.point(drill.cue_start), GEO.point(drill.pocket)
    n = math.dist(a, b)
    return (point[0] + (b[0] - a[0]) / n * inches * IN, point[1] + (b[1] - a[1]) / n * inches * IN)


def test_follow_shot_marks_are_in_line():
    cue, obj, pocket = (GEO.point(p) for p in (FOLLOW_SHOT.cue_start, FOLLOW_SHOT.object_start, FOLLOW_SHOT.pocket))
    cross = (obj[0] - cue[0]) * (pocket[1] - cue[1]) - (obj[1] - cue[1]) * (pocket[0] - cue[0])
    assert abs(cross) / math.dist(cue, pocket) < 3  # within 3 px of the line


def test_follow_shot_perfect():
    result = play(FOLLOW_SHOT, GEO.point(FOLLOW_SHOT.cue_target))
    assert result["rating"] == "perfect" and result["pocket"] == "top-right corner"


def test_follow_shot_not_enough_follow():
    result = play(FOLLOW_SHOT, along_line(FOLLOW_SHOT, GEO.point(FOLLOW_SHOT.object_start), 1))
    assert any("not enough follow" in n and "higher" in n for n in result["notes"])


def test_follow_shot_too_much_follow():
    result = play(FOLLOW_SHOT, along_line(FOLLOW_SHOT, GEO.point(FOLLOW_SHOT.cue_target), 5))
    assert any("followed 5 in too far" in n for n in result["notes"])


def test_follow_shot_scratch():
    result = play(FOLLOW_SHOT, None)
    assert result["rating"] == "scratch"
    assert any("followed it all the way in" in n or "It followed all the way in" in n for n in result["notes"])


from src.drills import CUT_SHOT, DRILL_ORDER, aim_point  # noqa: E402


def test_cut_shot_ghost_ball_is_behind_object_ball():
    ghost, obj, pocket = aim_point(CUT_SHOT, GEO), GEO.point(CUT_SHOT.object_start), GEO.point(CUT_SHOT.pocket)
    assert math.dist(ghost, obj) == pytest.approx(2.25 * IN, rel=0.01)       # one ball width
    assert math.dist(ghost, pocket) > math.dist(obj, pocket)                  # on the far side


def test_cut_shot_perfect():
    result = play(CUT_SHOT, GEO.point(CUT_SHOT.cue_target))
    assert result["rating"] == "perfect" and result["pocket"] == "top side"


def test_cut_shot_cue_ball_short_and_long():
    target, ghost = GEO.point(CUT_SHOT.cue_target), aim_point(CUT_SHOT, GEO)
    n = math.dist(ghost, target)
    toward = lambda d: (target[0] + (target[0] - ghost[0]) / n * d * IN, target[1] + (target[1] - ghost[1]) / n * d * IN)
    short = play(CUT_SHOT, toward(-10))
    assert any("short of its target: hit a little firmer" in n for n in short["notes"])
    long = play(CUT_SHOT, toward(10))
    assert any("past its target: hit a little softer" in n for n in long["notes"])


def test_cut_shot_missed_pocket_mentions_ghost_ball():
    obj = GEO.point(CUT_SHOT.object_start)
    result = play(CUT_SHOT, GEO.point(CUT_SHOT.cue_target), object_end=(obj[0] - 6 * IN, obj[1] - 9 * IN), pocketed=False)
    assert result["rating"] == "missed"
    assert any("missed left" in n and "ghost-ball" in n for n in result["notes"])


@pytest.mark.parametrize("drill", DRILL_ORDER, ids=lambda d: d.id)
def test_every_drill_marks_on_table_and_distinct(drill):
    snap = DrillTracker(drill)
    snap.update([], 0.0, POLYGON)
    marks = snap.snapshot()["marks"]
    for key in ("cue_start", "object_start", "pocket", "cue_target", "aim"):
        x, y = marks[key]
        assert 150 < x < 1180 and 60 < y < 590
    assert math.dist(marks["cue_start"], marks["object_start"]) > 4 * IN  # balls not on top of each other


def test_follow_shot_object_ball_hidden_behind_cue_ball():
    """The cue ball follows right behind the object ball, so it's never seen moving on its own:
    credit the drill's pocket rather than guessing the nearest one."""
    sim = DrillSim(FOLLOW_SHOT)
    sim.frames(1.5, cue=sim.cue_mark, obj=sim.object_mark)
    sim.roll(0.4, cue_path=(sim.cue_mark, sim.object_mark), obj_path=(sim.object_mark, sim.object_mark))
    end = along_line(FOLLOW_SHOT, sim.target, 2)
    sim.roll(0.5, cue_path=(sim.object_mark, end), obj_path=(sim.object_mark, sim.object_mark))  # hidden by the cue ball
    sim.frames(2.5, cue=end)
    result = sim.tracker.snapshot()["result"]
    assert result["pocketed"] and result["pocket"] == "top-right corner"
