"""Detection against real camera stills in tests/fixtures/table/.

Each <name>.png has a <name>.json ground truth:
  {"image": "<name>.png", "balls": [{"ball": "cue"|1..15, "x": px, "y": px,
                                     "on_table": bool, "occluded": bool (optional)}]}
on_table is what a person sees (ball in a pocket = false), not derived from table_polygon.
Occluded balls are on the table but hidden (e.g. under an arm), so they aren't required.

Checks that fail today are listed in KNOWN_FAILURES and run as xfail, so fixing
detection turns them into XPASS; remove the entry once it passes.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config_loader import load_config
from src.detection import detect_balls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = Path(__file__).parent / "fixtures" / "table"
CASES = sorted(p.stem for p in TABLE_DIR.glob("*.json"))

# (check, image) -> reason it fails today
_STRIPE_11_TWICE = "red-striped 11 also picked up as a second, offset maroon circle"
KNOWN_FAILURES = {
    ("found", "08_rails"): "maroon-striped 15 against the right cushion not detected",
    ("found", "09b_near_every_pocket"): "red 3 in the shadow of the top-right pocket is darker than the red range allows",
    ("no_false_positives", "12_midgame"): _STRIPE_11_TWICE,
    ("no_false_positives", "13_cue_stick_b"): "player over the table: watch, arm and cue stick detected as balls; " + _STRIPE_11_TWICE,
    ("cue", "05_stripes_white_up"): "blue-striped 10 lying white side up chosen as the cue ball",
    ("identity", "07_rack"): "yellow-striped 9 in the rack shows too little band to name, so it's an unknown stripe",
    ("identity", "05_stripes_white_up"): "stripes lying white side up: too little color showing to name them",
    ("identity", "06_solid_stripe_pairs"): "stripes lying white side up left unnamed or misnamed",
    ("identity", "13_cue_stick_b"): "9 lying white side up left unnamed",
}


def _params(check):
    params = []
    for name in CASES:
        reason = KNOWN_FAILURES.get((check, name))
        marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
        params.append(pytest.param(name, id=name, marks=marks))
    return params


@pytest.fixture(scope="module")
def detection_config():
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))["detection"]


@pytest.fixture(scope="module")
def radius_px(detection_config):
    return float(detection_config["expected_radius_px"])


_cache = {}


def _run(name, detection_config, radius_px):
    if name not in _cache:
        gt = json.loads((TABLE_DIR / f"{name}.json").read_text())
        frame = cv2.imread(str(TABLE_DIR / gt["image"]))
        assert frame is not None, f"could not read {gt['image']}"
        _cache[name] = (gt["balls"], detect_balls(frame, detection_config, radius_px))
    return _cache[name]


def _near(det, ball, radius_px):
    return np.hypot(det.center_px[0] - ball["x"], det.center_px[1] - ball["y"]) <= radius_px


def _match(balls, detections, radius_px):
    """Greedy nearest-first one-to-one matching; returns {ball index: detection index}."""
    pairs = sorted(
        (np.hypot(d.center_px[0] - b["x"], d.center_px[1] - b["y"]), bi, di)
        for bi, b in enumerate(balls) for di, d in enumerate(detections)
    )
    matched, used = {}, set()
    for dist, bi, di in pairs:
        if dist > radius_px or bi in matched or di in used:
            continue
        matched[bi] = di
        used.add(di)
    return matched


def _fmt(ball):
    return f"{ball['ball']}@({ball['x']:.0f},{ball['y']:.0f})"


@pytest.mark.parametrize("name", _params("found"))
def test_on_table_balls_found(name, detection_config, radius_px):
    balls, detections = _run(name, detection_config, radius_px)
    on_table = [b for b in balls if b["on_table"]]
    matched = _match(on_table, detections, radius_px)
    missed = [_fmt(b) for i, b in enumerate(on_table) if i not in matched and not b.get("occluded")]
    assert not missed, f"not detected: {', '.join(missed)}"


@pytest.mark.parametrize("name", _params("pocket_ignored"))
def test_pocketed_balls_ignored(name, detection_config, radius_px):
    balls, detections = _run(name, detection_config, radius_px)
    counted = [_fmt(b) for b in balls if not b["on_table"]
               if any(_near(d, b, radius_px) for d in detections)]
    assert not counted, f"in a pocket but detected: {', '.join(counted)}"


@pytest.mark.parametrize("name", _params("no_false_positives"))
def test_no_false_positives(name, detection_config, radius_px):
    balls, detections = _run(name, detection_config, radius_px)
    on_table = [b for b in balls if b["on_table"]]
    matched = set(_match(on_table, detections, radius_px).values())
    extra = []
    for i, d in enumerate(detections):
        if i in matched:
            continue
        desc = f"{d.color_label}@({d.center_px[0]:.0f},{d.center_px[1]:.0f})"
        twin = next((b for b in on_table if _near(d, b, radius_px)), None)
        extra.append(f"{desc} (second detection of {twin['ball']})" if twin else desc)
    assert not extra, f"detections with no ball there: {', '.join(extra)}"


@pytest.mark.parametrize("name", _params("cue"))
def test_cue_ball_identified(name, detection_config, radius_px):
    balls, detections = _run(name, detection_config, radius_px)
    cue = next((b for b in balls if b["ball"] == "cue" and b["on_table"]), None)
    flagged = [d for d in detections if d.is_cue_ball]
    if cue is None:
        assert not flagged, f"cue ball reported but none on the table: {flagged}"
        return
    assert len(flagged) == 1, f"expected 1 cue ball detection, got {len(flagged)}"
    assert _near(flagged[0], cue, radius_px), (
        f"cue ball reported at ({flagged[0].center_px[0]:.0f},{flagged[0].center_px[1]:.0f}), "
        f"actually at ({cue['x']:.0f},{cue['y']:.0f})")


@pytest.mark.parametrize("name", _params("identity"))
def test_ball_identity(name, detection_config, radius_px):
    balls, detections = _run(name, detection_config, radius_px)
    on_table = [b for b in balls if b["on_table"] and b["ball"] != "cue"]
    if not on_table:
        pytest.skip("no numbered balls on the table")
    # Only balls that were found are judged here; missing ones fail test_on_table_balls_found.
    matched = _match(on_table, detections, radius_px)
    wrong = [f"{_fmt(on_table[i])} reported as {detections[di].ball}"
             for i, di in matched.items() if detections[di].ball != on_table[i]["ball"]]
    assert not wrong, "; ".join(wrong)
