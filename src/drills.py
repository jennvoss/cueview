"""Practice drills: where to place the balls, how to hit the shot, and how it went.

Positions are given on the diamond grid used throughout the project: columns 0-8 along
the long rails (0 = left corner pockets, 4 = side pockets, 8 = right corner pockets) and
rows 0-4 along the short rails (0 = top rail, 2 = center line, 4 = bottom rail), as seen
in the camera image. They are mapped onto the frame through detection.table_polygon.

A DrillTracker is fed every frame's detections and moves through:
  setup   -> waiting for the balls to be placed on their marks
  ready   -> balls placed and still; waiting for the shot
  shot    -> the cue ball has left its mark; waiting for everything to stop
  result  -> analysis of the last attempt (stays until the balls are set up again)
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from src.constants import (
    TABLE_WIDTH_IN, TABLE_HEIGHT_IN, BALL_DIAMETER_IN,
    DRILL_PLACE_TOLERANCE_IN, DRILL_SETTLE_S, DRILL_REST_S, DRILL_SHOT_TIMEOUT_S,
    DRILL_MOVE_THRESHOLD_IN, DRILL_PERFECT_IN, DRILL_GOOD_IN,
    DRILL_MAX_BALL_SPEED_IN_S, DRILL_MAX_TRACK_GAP_S, DRILL_POCKET_NEAR_IN,
)


@dataclass(frozen=True)
class Drill:
    id: str
    name: str
    summary: str
    cue_start: tuple            # (col, row)
    object_start: tuple         # (col, row)
    pocket: tuple               # (col, row) of the target pocket
    cue_target: tuple           # (col, row) where the cue ball should stop
    tip: tuple                  # cue-tip contact point on the cue ball: (x, y) in ball
                                # radii from center, +x = right, +y = up (shooter's view)
    tip_text: str
    speed: int                  # 1 (very soft) to 5 (very hard)
    speed_text: str
    steps: tuple = field(default_factory=tuple)


STOP_SHOT = Drill(
    id="stop-shot",
    name="Stop shot",
    summary="Pocket the object ball in the top side pocket and stop the cue ball dead, "
            "right where the object ball was.",
    cue_start=(4, 3),
    object_start=(4, 1),
    pocket=(4, 0),
    cue_target=(4, 1),
    tip=(0.0, -0.25),
    tip_text="Half a cue tip below center, with no side spin.",
    speed=3,
    speed_text="Medium: firm enough that the object ball would roll about one table "
               "length if it missed the pocket.",
    steps=(
        "Place the object ball on its mark, one diamond out from the top side pocket.",
        "Place the cue ball on its mark, two diamonds behind it in a straight line to the pocket.",
        "Stand at the bottom rail and aim straight through the middle of the object ball.",
        "Hit the cue ball half a tip below center at medium speed and keep your stroke straight.",
    ),
)

DRILLS = {drill.id: drill for drill in (STOP_SHOT,)}


class TableGeometry:
    """Maps the diamond grid onto frame pixels using the table polygon (TL, TR, BR, BL)."""

    def __init__(self, table_polygon):
        self.tl, self.tr, self.br, self.bl = [tuple(map(float, p)) for p in table_polygon]
        width_px = (math.dist(self.tl, self.tr) + math.dist(self.bl, self.br)) / 2
        height_px = (math.dist(self.tl, self.bl) + math.dist(self.tr, self.br)) / 2
        self.px_per_in = (width_px / TABLE_WIDTH_IN + height_px / TABLE_HEIGHT_IN) / 2

    def point(self, grid) -> tuple:
        u, v = grid[0] / 8.0, grid[1] / 4.0
        top = (self.tl[0] + (self.tr[0] - self.tl[0]) * u, self.tl[1] + (self.tr[1] - self.tl[1]) * u)
        bottom = (self.bl[0] + (self.br[0] - self.bl[0]) * u, self.bl[1] + (self.br[1] - self.bl[1]) * u)
        return (top[0] + (bottom[0] - top[0]) * v, top[1] + (bottom[1] - top[1]) * v)

    def pockets(self) -> dict:
        names = {(0, 0): "top-left corner", (4, 0): "top side", (8, 0): "top-right corner",
                 (0, 4): "bottom-left corner", (4, 4): "bottom side", (8, 4): "bottom-right corner"}
        return {name: self.point(grid) for grid, name in names.items()}

    def inches(self, px: float) -> float:
        return px / self.px_per_in


class DrillTracker:
    def __init__(self, drill: Drill = STOP_SHOT):
        self.drill = drill
        self.state = "setup"
        self.geometry: Optional[TableGeometry] = None
        self.placed_since: Optional[float] = None
        self.shot_start: Optional[float] = None
        self.last_motion: Optional[float] = None
        self.cue_pos: Optional[tuple] = None
        self.object_pos: Optional[tuple] = None
        self.object_last_seen: Optional[tuple] = None
        self.object_seen_at: Optional[float] = None
        self.cue_seen_at: Optional[float] = None
        self.result: Optional[dict] = None
        self.cue_placed = False
        self.object_placed = False
        self.attempts = 0
        self.made = 0
        self.perfect = 0

    # -- frame updates ---------------------------------------------------------------

    def update(self, detections, timestamp: float, table_polygon) -> None:
        if not table_polygon:
            return
        self.geometry = TableGeometry(table_polygon)
        cue = next((d for d in detections if d.is_cue_ball), None)
        others = [d for d in detections if not d.is_cue_ball]
        if self.state in ("setup", "result"):
            self._watch_setup(cue, others, timestamp)
        elif self.state == "ready":
            self._watch_ready(cue, others, timestamp)
        elif self.state == "shot":
            self._watch_shot(cue, others, timestamp)

    def _marks(self):
        g = self.geometry
        return g.point(self.drill.cue_start), g.point(self.drill.object_start)

    def _on_mark(self, ball, mark) -> bool:
        return ball is not None and self.geometry.inches(math.dist(ball.center_px, mark)) <= DRILL_PLACE_TOLERANCE_IN

    def _placement(self, cue, others):
        cue_mark, object_mark = self._marks()
        object_ball = min(others, key=lambda d: math.dist(d.center_px, object_mark), default=None)
        return self._on_mark(cue, cue_mark), self._on_mark(object_ball, object_mark), object_ball

    def _watch_setup(self, cue, others, timestamp):
        cue_ok, object_ok, object_ball = self._placement(cue, others)
        self.cue_placed, self.object_placed = cue_ok, object_ok
        if not (cue_ok and object_ok):
            self.placed_since = None
            return
        if self.placed_since is None:
            self.placed_since = timestamp
        if timestamp - self.placed_since >= DRILL_SETTLE_S:
            self.state = "ready"
            self.cue_pos, self.object_pos = cue.center_px, object_ball.center_px

    def _watch_ready(self, cue, others, timestamp):
        cue_ok, object_ok, object_ball = self._placement(cue, others)
        self.cue_placed, self.object_placed = cue_ok, object_ok
        moved = cue is None or self.geometry.inches(math.dist(cue.center_px, self.cue_pos)) > DRILL_MOVE_THRESHOLD_IN
        if moved and object_ok:
            # Cue ball left its mark with the object ball still waiting: the shot is on.
            self.state = "shot"
            self.shot_start = self.last_motion = timestamp
            self.object_last_seen, self.object_seen_at = self.object_pos, timestamp
            self.cue_seen_at = timestamp if cue is not None else None
            self._track(cue, others, timestamp)
        elif not object_ok:
            # Balls were picked up or moved by hand: back to setup.
            self.state = "setup"
            self.placed_since = None

    def _watch_shot(self, cue, others, timestamp):
        self._track(cue, others, timestamp)
        resting = timestamp - self.last_motion >= DRILL_REST_S
        if resting or timestamp - self.shot_start >= DRILL_SHOT_TIMEOUT_S:
            self._finish(timestamp)

    def _track(self, cue, others, timestamp):
        move_px = DRILL_MOVE_THRESHOLD_IN * self.geometry.px_per_in / 2
        if cue is not None:
            if self.cue_pos is None or math.dist(cue.center_px, self.cue_pos) > move_px:
                self.last_motion = timestamp
            self.cue_pos, self.cue_seen_at = cue.center_px, timestamp
        obj = self._object_candidate(cue, others, timestamp)
        if obj is not None:
            if math.dist(obj.center_px, self.object_last_seen) > move_px:
                self.last_motion = timestamp
            self.object_last_seen, self.object_seen_at = obj.center_px, timestamp

    def _object_candidate(self, cue, others, timestamp):
        """The detection that can plausibly be the object ball: close enough to where it was
        last seen for the time that has passed, and not the cue ball seen a second time."""
        g = self.geometry
        gap = min(timestamp - self.object_seen_at, DRILL_MAX_TRACK_GAP_S)
        reach_px = (DRILL_MAX_BALL_SPEED_IN_S * gap + BALL_DIAMETER_IN) * g.px_per_in
        cue_px = cue.center_px if cue is not None else None
        candidates = [d for d in others
                      if math.dist(d.center_px, self.object_last_seen) <= reach_px
                      and (cue_px is None or g.inches(math.dist(d.center_px, cue_px)) > BALL_DIAMETER_IN)]
        return min(candidates, key=lambda d: math.dist(d.center_px, self.object_last_seen), default=None)

    # -- analysis --------------------------------------------------------------------

    def _finish(self, timestamp):
        object_visible = timestamp - self.object_seen_at < DRILL_REST_S
        if object_visible and self.geometry.inches(math.dist(self.object_last_seen, self.object_pos)) < DRILL_MOVE_THRESHOLD_IN:
            # The object ball never moved: the cue ball was picked up or nudged, not shot.
            self.state = "setup"
            self.placed_since = None
            return
        self.result = analyse(self.drill, self.geometry,
                              object_start=self.object_pos,
                              object_end=self.object_last_seen,
                              object_visible=object_visible,
                              cue_end=self.cue_pos,
                              cue_visible=self.cue_seen_at is not None and timestamp - self.cue_seen_at < DRILL_REST_S)
        self.attempts += 1
        self.made += self.result["pocketed"]
        self.perfect += self.result["rating"] == "perfect"
        self.state = "result"
        self.placed_since = None
        self.cue_placed = self.object_placed = False

    def snapshot(self) -> dict:
        drill = self.drill
        out = {
            "id": drill.id, "name": drill.name, "summary": drill.summary, "steps": list(drill.steps),
            "tip": list(drill.tip), "tip_text": drill.tip_text,
            "speed": drill.speed, "speed_text": drill.speed_text,
            "state": self.state, "result": self.result,
            "placed": {"cue": self.cue_placed, "object": self.object_placed},
            "stats": {"attempts": self.attempts, "made": self.made, "perfect": self.perfect},
        }
        if self.geometry is not None:
            g = self.geometry
            r = BALL_DIAMETER_IN / 2 * g.px_per_in
            out["marks"] = {
                "cue_start": _px(g.point(drill.cue_start)), "object_start": _px(g.point(drill.object_start)),
                "pocket": _px(g.point(drill.pocket)), "cue_target": _px(g.point(drill.cue_target)),
                "ball_radius_px": round(r, 1),
            }
        return out


def _px(point) -> list:
    return [round(point[0], 1), round(point[1], 1)]


def analyse(drill: Drill, geometry: TableGeometry, object_start, object_end, object_visible: bool,
            cue_end, cue_visible: bool) -> dict:
    """Compare where the balls ended up with where the drill wanted them, and say why."""
    g = geometry
    pocket = g.point(drill.pocket)
    target = g.point(drill.cue_target)
    # Shot direction: from the cue ball's mark through the object ball to the pocket.
    ax, ay = _unit(g.point(drill.cue_start), pocket)
    notes = []

    pocketed = not object_visible
    pocket_name = None
    if pocketed:
        pocket_name, nearest = min(g.pockets().items(), key=lambda kv: math.dist(kv[1], object_end))
        if g.inches(math.dist(nearest, object_end)) > DRILL_POCKET_NEAR_IN and object_end != object_start:
            # Lost mid-roll (e.g. motion blur): use the pocket it was heading toward.
            heading = _unit(object_start, object_end)
            pocket_name = max(g.pockets().items(),
                              key=lambda kv: _dot(_unit(object_end, kv[1]), heading))[0]
        wanted = min(g.pockets().items(), key=lambda kv: math.dist(kv[1], pocket))[0]
        if pocket_name != wanted:
            notes.append(f"The object ball went into the {pocket_name} pocket instead of the {wanted} pocket.")
            pocketed = False
    else:
        travel = g.inches(_dot(_sub(object_end, object_start), (ax, ay)))
        remaining = g.inches(_dot(_sub(pocket, object_end), (ax, ay)))
        side = g.inches(_cross((ax, ay), _sub(object_end, object_start)))
        if abs(side) > 2 * BALL_DIAMETER_IN or travel > g.inches(math.dist(object_start, pocket)):
            notes.append(f"The object ball missed {'right' if side > 0 else 'left'} of the pocket: "
                         f"aim {'left' if side > 0 else 'right'} of where you aimed, straight through its center.")
        elif remaining > BALL_DIAMETER_IN:
            notes.append(f"The object ball stopped {remaining:.0f} in short of the pocket: hit a little harder.")
        else:
            notes.append("The object ball rattled in the jaws: aim straight through its center and hit a touch firmer.")

    cue_error_in = None
    if not cue_visible:
        rating = "scratch"
        notes.append("Scratch: the cue ball went into a pocket. For a stop shot hit lower on the cue ball "
                     "so it stops instead of following the object ball.")
    else:
        offset = _sub(cue_end, target)
        along = g.inches(_dot(offset, (ax, ay)))
        across = g.inches(_cross((ax, ay), offset))
        cue_error_in = math.hypot(along, across)
        if along > BALL_DIAMETER_IN / 2:
            notes.append(f"The cue ball rolled {along:.0f} in past the object ball's spot: it still had "
                         "topspin at contact. Hit a little lower on the cue ball, or a little firmer.")
        elif along < -BALL_DIAMETER_IN / 2:
            notes.append(f"The cue ball came back {-along:.0f} in: that's draw. Hit a little closer to center.")
        if abs(across) > BALL_DIAMETER_IN / 2:
            notes.append(f"The cue ball drifted {abs(across):.0f} in to the {'right' if across > 0 else 'left'}: "
                         f"the hit wasn't full, or the tip was {'right' if across > 0 else 'left'} of center. "
                         "Keep the tip on the vertical center line.")
        if not pocketed:
            rating = "missed"
        elif cue_error_in <= DRILL_PERFECT_IN:
            rating = "perfect"
        elif cue_error_in <= DRILL_GOOD_IN:
            rating = "good"
        else:
            rating = "pocketed"
    if rating == "perfect":
        notes.insert(0, "Textbook stop shot: the cue ball stopped right where the object ball was.")
    headline = {
        "perfect": "Perfect!",
        "good": "Good shot: pocketed, cue ball close",
        "pocketed": "Pocketed, but the cue ball wandered",
        "missed": "Missed the pocket",
        "scratch": "Scratch",
    }[rating]
    return {
        "rating": rating, "headline": headline, "pocketed": pocketed, "pocket": pocket_name,
        "cue_end": _px(cue_end) if cue_visible else None,
        "object_end": _px(object_end) if object_visible else None,
        "cue_error_in": None if cue_error_in is None else round(cue_error_in, 1),
        "notes": notes,
    }


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _cross(axis, v):
    """Signed sideways component of v relative to axis; positive = the shooter's right.
    Image y points down, so right of a heading (ax, ay) is (-ay, ax): shooting up the
    screen, (0, -1), the shooter's right is screen right, (1, 0)."""
    return v[0] * -axis[1] + v[1] * axis[0]


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n
