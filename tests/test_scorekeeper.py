from src.scorekeeper import ScoreKeeper, ALL_BALLS, SOLIDS, STRIPES

FPS = 15.0


def feed(keeper, seen, seconds, start):
    """Report the same set of balls every frame for `seconds`; return the end time."""
    t = start
    for _ in range(int(seconds * FPS)):
        keeper.update(seen, t)
        t += 1 / FPS
    return t


def test_no_score_until_warmed_up():
    """At startup nothing is reported, rather than every ball flashing onto the table."""
    keeper = ScoreKeeper(warmup_s=1.5)
    snap = keeper.snapshot()
    assert snap["ready"] is False
    assert snap["on_table"] == [] and snap["solids_sunk"] == [] and snap["stripes_sunk"] == []
    feed(keeper, {"cue", 8}, 1.0, 0.0)
    assert keeper.snapshot()["ready"] is False


def test_first_score_matches_the_table():
    """Balls missing during warm-up are sunk in the very first score, with no 3 s wait."""
    keeper = ScoreKeeper(warmup_s=1.5, sink_after_s=3.0)
    feed(keeper, {"cue", 1, 8, 12}, 1.6, 0.0)
    snap = keeper.snapshot()
    assert snap["ready"] is True
    assert snap["on_table"] == ["cue", 1, 8, 12]
    assert snap["solids_sunk"] == [2, 3, 4, 5, 6, 7]
    assert 12 not in snap["stripes_sunk"] and len(snap["stripes_sunk"]) == 6


def test_warmup_ignores_occasional_misread():
    """A ball named in only a few warm-up frames (a misread) starts sunk."""
    keeper = ScoreKeeper(warmup_s=1.5)
    t = 0.0
    for i in range(int(1.6 * FPS)):
        keeper.update(["cue", 12] + ([8] if i % 6 == 0 else []), t)
        t += 1 / FPS
    assert keeper.snapshot()["eight_sunk"] is True


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
    keeper = ScoreKeeper(warmup_s=0.5)
    feed(keeper, [None, 99, "cue"], 1.0, 0.0)
    assert keeper.snapshot()["on_table"] == ["cue"]


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


def test_moving_phantoms_do_not_keep_a_sunk_ball():
    """A cue stick sweeping across the table, read as the (sunk) 15 at a new spot each frame."""
    keeper = ScoreKeeper(sink_after_s=3.0)
    frames = []
    for i in range(int(6 * FPS)):
        frames.append([Det("cue", 328, 130), Det(8, 928, 130), Det(15, 200 + 15 * i, 300 + 2 * i)])
    feed_detections(keeper, frames, 0.0)
    snap = keeper.snapshot()
    assert 15 in snap["stripes_sunk"]
    assert 8 in snap["on_table"]


def test_unknown_stripes_keep_recently_seen_stripes():
    """10 and 12 roll to rest white side up (unnameable) while the 15, long gone, stays sunk."""
    keeper = ScoreKeeper(sink_after_s=3.0, warmup_s=0.5)
    t = feed(keeper, {"cue", 8, 10, 12, 15}, 1.0, 0.0)
    t = feed(keeper, {"cue", 8, 10, 12}, 4.0, t)          # 15 pocketed
    assert 15 in keeper.snapshot()["stripes_sunk"]
    t = 0.0
    for _ in range(int(6 * FPS)):                          # 10 and 12 now unnameable
        keeper.update(["cue", 8, "stripe", "stripe"], 5.0 + t)
        t += 1 / FPS
    snap = keeper.snapshot()
    assert 10 in snap["on_table"] and 12 in snap["on_table"]
    assert 15 in snap["stripes_sunk"]


def test_unknown_stripe_brings_back_the_only_missing_stripe():
    """With every other stripe accounted for, an unnameable stripe on the table is the 15."""
    keeper = ScoreKeeper(sink_after_s=3.0, warmup_s=0.5)
    others = {"cue", 8, 9, 10, 11, 12, 13, 14}
    t = feed(keeper, others | {15}, 1.0, 0.0)
    t = feed(keeper, others, 4.0, t)
    assert 15 in keeper.snapshot()["stripes_sunk"]
    feed(keeper, list(others) + ["stripe"], 1.5, t)
    assert 15 in keeper.snapshot()["on_table"]


def test_player_over_table_pauses_scoring():
    """While the player leans over: stick phantoms named 15 don't count, the 3 hidden under
    their arm isn't sunk, and the 15 they pocketed is sunk once play resumes."""
    keeper = ScoreKeeper(sink_after_s=3.0, warmup_s=0.5)
    table = [Det("cue", 100, 100), Det(3, 300, 300), Det(8, 500, 300), Det(15, 700, 300)]
    t = feed_detections(keeper, [table] * int(2 * FPS), 0.0)
    for i in range(int(5 * FPS)):                     # leaning over for 5 s: 3 hidden, stick read as 15s
        keeper.update_detections([Det("cue", 100, 100), Det(8, 500, 300), Det(15, 600 + 3 * i, 500)], t,
                                 player_present=True)
        t += 1 / FPS
    assert keeper.snapshot()["paused"] is True
    assert 3 in keeper.snapshot()["on_table"]
    after = [Det("cue", 100, 100), Det(3, 300, 300), Det(8, 500, 300)]   # 15 went in
    t = feed_detections(keeper, [after] * int(1.0 * FPS), t)
    snap = keeper.snapshot()
    assert snap["paused"] is False and 3 in snap["on_table"] and 15 in snap["on_table"]  # not 3 s yet
    feed_detections(keeper, [after] * int(3.0 * FPS), t)
    snap = keeper.snapshot()
    assert 15 in snap["stripes_sunk"] and 3 in snap["on_table"]


def test_start_with_stripes_white_side_up():
    """Capture starts with three stripes white side up (unnameable) and four unaccounted for:
    the count is restored (six stripes on, one sunk), though which numbers can't be known."""
    keeper = ScoreKeeper(warmup_s=1.5, return_after_s=1.0)
    frame = ["cue", 7, 11, 13, 14, "stripe", "stripe", "stripe"]
    t = feed(keeper, frame, 1.6, 0.0)
    assert keeper.snapshot()["stripes_sunk"] == [9, 10, 12, 15]  # right after warm-up nothing is known
    feed(keeper, frame, 1.5, t)
    snap = keeper.snapshot()
    assert len([b for b in snap["on_table"] if isinstance(b, int) and b > 8]) == 6
    assert len(snap["stripes_sunk"]) == 1
    assert {11, 13, 14} <= set(snap["on_table"])
