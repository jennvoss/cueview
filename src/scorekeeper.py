"""8-ball scorekeeper: which balls are on the table and which have been sunk.

Fed the balls detected in each frame. Single frames are unreliable (motion blur, a
player's arm, an occasional misread), so a ball only counts as sunk after it hasn't been
seen for SINK_AFTER_S, and only counts as back on the table (e.g. after a re-rack) once it
has been seen continuously for RETURN_AFTER_S. With update_detections, each spot on the
table is named by majority vote over its recent readings, so a ball that is now and then
misread as another number doesn't keep that other ball on the table.
"""
import math
from collections import Counter, deque
from typing import Iterable, Optional, Union

from src.constants import (
    SINK_AFTER_S, RETURN_AFTER_S, NAME_VOTE_WINDOW_S, NAME_VOTE_RADIUS_PX,
    WARMUP_S, WARMUP_MIN_SEEN_FRACTION,
)

SOLIDS = list(range(1, 8))
STRIPES = list(range(9, 16))
ALL_BALLS = ["cue"] + SOLIDS + [8] + STRIPES

Ball = Union[int, str]


class ScoreKeeper:
    def __init__(self, sink_after_s: float = SINK_AFTER_S, return_after_s: float = RETURN_AFTER_S,
                 warmup_s: float = WARMUP_S):
        self.sink_after_s = sink_after_s
        self.return_after_s = return_after_s
        # No score until the first warmup_s of frames show which balls are on the table;
        # assuming they all are would flash every ball onto the dashboard at startup.
        self.warmup_s = warmup_s
        self.ready = False
        self.warmup_start = None
        self.warmup_frames = 0
        self.warmup_seen = Counter()
        self.on_table = {ball: True for ball in ALL_BALLS}
        self.last_seen: dict = {}
        self.seen_since: dict = {}
        self.spots: list = []  # [{"pos": (x, y), "votes": deque[(timestamp, ball)]}]

    def update_detections(self, detections: Iterable, timestamp: float) -> None:
        """Feed a frame's detections (anything with .ball and .center_px). Each is matched to
        the nearest recent spot within NAME_VOTE_RADIUS_PX and named by that spot's majority."""
        named = []
        used = set()
        for det in detections:
            if det.ball is None:
                continue
            pos = (float(det.center_px[0]), float(det.center_px[1]))
            spot = min((s for i, s in enumerate(self.spots) if i not in used
                        and math.dist(s["pos"], pos) <= NAME_VOTE_RADIUS_PX),
                       key=lambda s: math.dist(s["pos"], pos), default=None)
            if spot is None:
                spot = {"pos": pos, "votes": deque()}
                self.spots.append(spot)
            used.add(next(i for i, s in enumerate(self.spots) if s is spot))
            spot["pos"] = pos
            spot["votes"].append((timestamp, det.ball))
            named.append(spot)
        for spot in self.spots:
            while spot["votes"] and timestamp - spot["votes"][0][0] > NAME_VOTE_WINDOW_S:
                spot["votes"].popleft()
        self.spots = [s for s in self.spots if s["votes"]]
        self.update([self._majority(spot) for spot in named], timestamp)

    @staticmethod
    def _majority(spot) -> Ball:
        counts = Counter(ball for _, ball in spot["votes"])
        top = max(counts.values())
        # On a tie, the most recent of the tied names wins.
        return next(ball for _, ball in reversed(spot["votes"]) if counts[ball] == top)

    def update(self, balls_seen: Iterable[Optional[Ball]], timestamp: float) -> None:
        seen = {ball for ball in balls_seen if ball in self.on_table}
        if not self.ready:
            self._warm_up(seen, timestamp)
            return
        for ball in ALL_BALLS:
            if ball in seen:
                self.last_seen[ball] = timestamp
                self.seen_since.setdefault(ball, timestamp)
                if not self.on_table[ball] and timestamp - self.seen_since[ball] >= self.return_after_s:
                    self.on_table[ball] = True
            else:
                self.seen_since.pop(ball, None)
                last = self.last_seen.setdefault(ball, timestamp)
                if self.on_table[ball] and timestamp - last >= self.sink_after_s:
                    self.on_table[ball] = False

    def _warm_up(self, seen: set, timestamp: float) -> None:
        if self.warmup_start is None:
            self.warmup_start = timestamp
        self.warmup_frames += 1
        self.warmup_seen.update(seen)
        for ball in seen:
            self.last_seen[ball] = timestamp
        if timestamp - self.warmup_start < self.warmup_s:
            return
        for ball in ALL_BALLS:
            self.on_table[ball] = self.warmup_seen[ball] / self.warmup_frames >= WARMUP_MIN_SEEN_FRACTION
            if not self.on_table[ball]:
                self.last_seen.pop(ball, None)
        self.ready = True

    def snapshot(self) -> dict:
        if not self.ready:
            return {"game": "8-ball", "ready": False, "on_table": [], "solids_sunk": [],
                    "stripes_sunk": [], "eight_sunk": False, "cue_sunk": False}
        return {
            "game": "8-ball",
            "ready": True,
            "on_table": [ball for ball in ALL_BALLS if self.on_table[ball]],
            "solids_sunk": [ball for ball in SOLIDS if not self.on_table[ball]],
            "stripes_sunk": [ball for ball in STRIPES if not self.on_table[ball]],
            "eight_sunk": not self.on_table[8],
            "cue_sunk": not self.on_table["cue"],
        }
