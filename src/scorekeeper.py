"""8-ball scorekeeper: which balls are on the table and which have been sunk.

Fed the set of ball numbers detected in each frame. Single frames are unreliable (motion
blur, a player's arm, an occasional misread), so a ball only counts as sunk after it
hasn't been seen for SINK_AFTER_S, and only counts as back on the table (e.g. after a
re-rack) once it has been seen continuously for RETURN_AFTER_S.
"""
from typing import Iterable, Optional, Union

from src.constants import SINK_AFTER_S, RETURN_AFTER_S

SOLIDS = list(range(1, 8))
STRIPES = list(range(9, 16))
ALL_BALLS = ["cue"] + SOLIDS + [8] + STRIPES

Ball = Union[int, str]


class ScoreKeeper:
    def __init__(self, sink_after_s: float = SINK_AFTER_S, return_after_s: float = RETURN_AFTER_S):
        self.sink_after_s = sink_after_s
        self.return_after_s = return_after_s
        # Until a ball has been missing for sink_after_s it is assumed on the table.
        self.on_table = {ball: True for ball in ALL_BALLS}
        self.last_seen: dict = {}
        self.seen_since: dict = {}

    def update(self, balls_seen: Iterable[Optional[Ball]], timestamp: float) -> None:
        seen = {ball for ball in balls_seen if ball in self.on_table}
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

    def snapshot(self) -> dict:
        return {
            "game": "8-ball",
            "on_table": [ball for ball in ALL_BALLS if self.on_table[ball]],
            "solids_sunk": [ball for ball in SOLIDS if not self.on_table[ball]],
            "stripes_sunk": [ball for ball in STRIPES if not self.on_table[ball]],
            "eight_sunk": not self.on_table[8],
            "cue_sunk": not self.on_table["cue"],
        }
