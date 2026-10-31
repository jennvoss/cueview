TABLE_WIDTH_IN = 88.0
TABLE_HEIGHT_IN = 44.0
BALL_DIAMETER_IN = 2.25
FPS_TARGET = 60
FPS_MINIMUM = 30
# Shot detection, in multiples of the expected ball radius (px)
REST_WINDOW_S = 0.5           # ball must stay put this long to count as at rest
REST_MAX_SPREAD_RADII = 0.5   # max distance from the window centroid while at rest
SHOT_START_RADII = 1.0        # distance from the rest position that starts a shot
MAX_REACQUIRE_FRAMES = 5
# 8-ball numbering by hsv_ranges label: (solid, stripe). Black is only the 8.
BALL_NUMBERS_BY_COLOR = {
    "yellow": (1, 9), "blue": (2, 10), "red": (3, 11), "purple": (4, 12),
    "orange": (5, 13), "green": (6, 14), "maroon": (7, 15), "black": (8, 8),
}
# A colored ball is a stripe when at least this fraction of it is white; solids show
# only their number circle (measured max 0.14), stripes 0.13 and up.
STRIPE_MIN_WHITE = 0.15
# Scorekeeper: a ball counts as sunk once it hasn't been seen for SINK_AFTER_S, and
# as back on the table once it has been seen continuously for RETURN_AFTER_S.
SINK_AFTER_S = 3.0
RETURN_AFTER_S = 1.0
# A ball is on the table only if more than this fraction of its disc lies inside
# detection.table_polygon; otherwise it is in a pocket and ignored.
ON_TABLE_MIN_FRACTION = 0.5
# A detected circle is dropped when its own color also fills more than this fraction
# of a ring just outside it (1.3–1.8 ball radii): that's a pocket hole or the cloth,
# not a ball sitting on the cloth.
ISOLATION_RING_MAX_FRACTION = 0.4
# A detected circle must be at least this full of its own color; Hough circles on
# shadow smears at pocket mouths are mostly empty.
MIN_COLOR_COVERAGE = 0.3
# Shadows at pocket mouths are black too, so the "black" label needs a fuller circle:
# measured shadow false positives fill at most 0.39, real 8 balls at least 0.75.
MIN_BLACK_COVERAGE = 0.6
# A colored (non-cue) circle must also be at least this full of its own color, or
# it's a white ball seen through a loose range rather than that color.
MIN_OWN_COLOR = 0.15
