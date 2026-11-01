from src.scorekeeper import ScoreKeeper, ALL_BALLS, SOLIDS, STRIPES

FPS = 15.0


def feed(keeper, seen, seconds, start):
    """Report the same set of balls every frame for `seconds`; return the end time."""
    t = start
    for _ in range(int(seconds * FPS)):
        keeper.update(seen, t)
        t += 1 / FPS
    return t


def test_starts_with_every_ball_on_table():
    snap = ScoreKeeper().snapshot()
    assert snap["on_table"] == ALL_BALLS
    assert snap["solids_sunk"] == [] and snap["stripes_sunk"] == []
    assert snap["eight_sunk"] is False and snap["cue_sunk"] is False


def test_ball_sunk_after_missing_long_enough():
    keeper = ScoreKeeper(sink_after_s=3.0)
    everything = set(ALL_BALLS)
    t = feed(keeper, everything, 1.0, 0.0)
    t = feed(keeper, everything - {3, 12}, 2.0, t)
    assert keeper.snapshot()["solids_sunk"] == []  # missing only 2 s so far
    feed(keeper, everything - {3, 12}, 1.5, t)
    snap = keeper.snapshot()
    assert snap["solids_sunk"] == [3]
    assert snap["stripes_sunk"] == [12]
    assert 3 not in snap["on_table"] and 12 not in snap["on_table"]


def test_brief_dropouts_do_not_sink_a_ball():
    keeper = ScoreKeeper(sink_after_s=3.0)
    t = 0.0
    for _ in range(20):  # 7 seen every other half second, never gone 3 s
        t = feed(keeper, set(ALL_BALLS), 0.5, t)
        t = feed(keeper, set(ALL_BALLS) - {7}, 0.5, t)
    assert 7 in keeper.snapshot()["on_table"]


def test_ball_returns_after_seen_continuously():
    keeper = ScoreKeeper(sink_after_s=3.0, return_after_s=1.0)
    t = feed(keeper, set(ALL_BALLS) - {8}, 4.0, 0.0)
    assert keeper.snapshot()["eight_sunk"] is True
    t = feed(keeper, set(ALL_BALLS), 0.5, t)
    assert keeper.snapshot()["eight_sunk"] is True  # seen only 0.5 s
    feed(keeper, set(ALL_BALLS), 1.0, t)
    assert keeper.snapshot()["eight_sunk"] is False


def test_unknown_balls_ignored():
    keeper = ScoreKeeper()
    keeper.update([None, 99, "cue"], 0.0)
    assert keeper.snapshot()["on_table"] == ALL_BALLS


def test_full_game_split():
    keeper = ScoreKeeper(sink_after_s=1.0)
    feed(keeper, {"cue", 8}, 2.0, 0.0)
    snap = keeper.snapshot()
    assert snap["solids_sunk"] == SOLIDS
    assert snap["stripes_sunk"] == STRIPES
    assert snap["on_table"] == ["cue", 8]


class Det:
    def __init__(self, ball, x, y):
        self.ball, self.center_px = ball, (x, y)


def feed_detections(keeper, frames, start, fps=FPS):
    t = start
    for dets in frames:
        keeper.update_detections(dets, t)
        t += 1 / fps
    return t


def test_misread_ball_does_not_keep_sunk_ball_on_table():
    """The 12 sits still and is read as the 8 about 1 frame in 4; the real 8 is in a pocket."""
    keeper = ScoreKeeper(sink_after_s=3.0)
    frames = [[Det("cue", 877, 479), Det(8 if i % 4 == 0 else 12, 366, 520)] for i in range(int(6 * FPS))]
    feed_detections(keeper, frames, 0.0)
    snap = keeper.snapshot()
    assert 12 in snap["on_table"]
    assert 8 not in snap["on_table"]
    assert snap["eight_sunk"] is True


def test_rarely_detected_ball_keeps_its_name():
    """A ball found only every third frame (like a hard-to-see maroon 7) still counts as on the table."""
    keeper = ScoreKeeper(sink_after_s=3.0)
    frames = [[Det("cue", 500, 300)] + ([Det(7, 353, 486)] if i % 3 == 0 else []) for i in range(int(6 * FPS))]
    feed_detections(keeper, frames, 0.0)
    assert 7 in keeper.snapshot()["on_table"]


def test_separate_balls_keep_separate_names():
    """Touching balls (two radii apart) are different spots, so neither outvotes the other."""
    keeper = ScoreKeeper(sink_after_s=3.0)
    frames = [[Det("cue", 100, 100), Det(3, 500, 300), Det(11, 524, 300)] for _ in range(int(5 * FPS))]
    feed_detections(keeper, frames, 0.0)
    on_table = keeper.snapshot()["on_table"]
    assert 3 in on_table and 11 in on_table


def test_ball_rolling_to_new_spot_keeps_its_name():
    """A ball moving a few pixels per frame stays one spot; a far jump starts a new spot."""
    keeper = ScoreKeeper(sink_after_s=3.0)
    frames = [[Det("cue", 100, 100), Det(5, 300 + 4 * i, 300)] for i in range(int(4 * FPS))]
    feed_detections(keeper, frames, 0.0)
    assert 5 in keeper.snapshot()["on_table"]
    assert len(keeper.spots) == 2
