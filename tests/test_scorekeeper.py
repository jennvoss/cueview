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
