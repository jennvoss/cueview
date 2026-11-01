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
# A colored ball is a stripe when at least this fraction of an outer ring (0.6-1.0 ball
# radii) is white: a stripe's white caps sit at the edge, while a solid's number circle
# and the lamp's highlight sit toward the middle. Measured on the test images plus live
# frames: solids at most 0.10, stripes mostly 0.12 and up (106/107 right).
STRIPE_MIN_WHITE = 0.10
# Scorekeeper: a ball counts as sunk once it hasn't been seen for SINK_AFTER_S, and
# as back on the table once it has been seen continuously for RETURN_AFTER_S.
SINK_AFTER_S = 3.0
RETURN_AFTER_S = 1.0
# A ball in one spot is named by majority vote over its last NAME_VOTE_WINDOW_S of
# readings, so an occasional misread (a band-up 12 read as the 8) doesn't keep a sunk ball
# on the table. Detections within NAME_VOTE_RADIUS_PX of each other are the same spot.
NAME_VOTE_WINDOW_S = 2.0
# At startup the scorekeeper watches for WARMUP_S before reporting a score (the dashboard
# shows "reading the table" meanwhile); a ball seen in at least WARMUP_MIN_SEEN_FRACTION of
# those frames starts on the table, the rest start sunk.
WARMUP_S = 1.5
WARMUP_MIN_SEEN_FRACTION = 0.3
NAME_VOTE_RADIUS_PX = 12.0
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
# Shadows at pocket mouths are black too, so the "black" label needs a fuller circle, of
# black pixels only (white doesn't count, unlike stripes): measured shadow false positives
# fill at most 0.39, a dark-banded 12 about 0.25, real 8 balls at least 0.75.
MIN_BLACK_COVERAGE = 0.6
# A colored (non-cue) circle must also be at least this full of its own color, or
# it's a white ball seen through a loose range rather than that color.
MIN_OWN_COLOR = 0.15
# Lighting changes (brightness and color temperature) are undone per frame by scaling
# each BGR channel so the cloth's median color matches the cloth in
# tests/fixtures/table/, the lighting hsv_ranges were tuned under.
CLOTH_REFERENCE_BGR = (29.0, 27.0, 86.0)
CLOTH_GAIN_LIMITS = (0.5, 2.5)
# Maroon is nearly the cloth's color, so a maroon circle must also show the overhead
# lamp's highlight: brightest spot minus the surrounding cloth, in V levels. Measured:
# real 7/15 balls 46-157, cloth that matched the maroon range at most 10.
MAROON_MIN_HIGHLIGHT = 25
# The blue and purple hsv_ranges overlap, so a ball found by either is named by the
# median hue of its blue/purple pixels. Measured medians: blue 2/10 112-114, purple 4/12 117-120.
BLUE_PURPLE_HUE_SPLIT = 116
# Extra circle search on how far each pixel is from the local cloth color, for balls
# nearly the cloth's color (the maroon 7/15). It needs a stricter Hough threshold than the
# color masks: at 25 it finds ~46 candidates per frame and every live 7; at 10 it finds ~960.
OFF_CLOTH_GAIN = 3.0
OFF_CLOTH_HOUGH_PARAM2 = 25
