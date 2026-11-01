import json
import os
import time

import pytest

from web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_is_scoreboard(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "<title>CueView</title>" in html
    assert 'id="feed"' in html
    for group in ("balls-table", "balls-solids", "balls-stripes"):
        assert f'id="{group}"' in html
    assert 'href="/debug"' in html
    assert 'id="cue-chip"' in html  # cue ball shown in the header, not in the ball list


def test_debug_is_old_dashboard(client):
    res = client.get("/debug")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "<title>CueView — Debug</title>" in html
    assert "panel-detection" in html
    assert 'href="/"' in html


@pytest.mark.parametrize("path", ["/static/scoreboard.css", "/static/scoreboard.js", "/static/app.js"])
def test_static_assets_served(client, path):
    assert client.get(path).status_code == 200


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """Point the web app at a temporary outputs folder."""
    import web.app as webapp
    monkeypatch.setattr(webapp, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(webapp, "FRAME_PATH", tmp_path / "latest.jpg")
    return tmp_path


def _write_status(path, age_s, **fields):
    path.write_text(json.dumps({"status": "running", **fields}))
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_status_live_while_capture_running(client, outputs):
    _write_status(outputs / "status.json", age_s=0, score={"on_table": ["cue"]})
    (outputs / "latest.jpg").write_bytes(b"jpg")
    data = client.get("/api/status").get_json()
    assert data["live"] is True


def test_status_not_live_keeps_last_known_score(client, outputs):
    _write_status(outputs / "status.json", age_s=600, score={"on_table": ["cue"]})
    (outputs / "latest.jpg").write_bytes(b"jpg")
    data = client.get("/api/status").get_json()
    assert data["live"] is False
    assert data["status"] == "idle"
    assert data["score"] == {"on_table": ["cue"]}  # last known score is still returned


def test_status_without_capture_or_frame(client, outputs):
    data = client.get("/api/status").get_json()
    assert data["live"] is False


def test_last_frame_served_when_not_live(client, outputs):
    _write_status(outputs / "status.json", age_s=600)
    (outputs / "latest.jpg").write_bytes(b"\xff\xd8last-frame")
    res = client.get("/api/frame.jpg")
    assert res.status_code == 200
    assert res.data == b"\xff\xd8last-frame"


def test_status_from_still_image_is_not_live(client, outputs):
    _write_status(outputs / "status.json", age_s=0, score={"on_table": ["cue", 8]},
                  source={"kind": "image", "path": "outputs/latest_raw.png",
                          "annotated": False})
    (outputs / "latest.jpg").write_bytes(b"jpg")  # rewritten just now while scoring the still
    data = client.get("/api/status").get_json()
    assert data["status"] == "running"
    assert data["live"] is False
    assert data["score"] == {"on_table": ["cue", 8]}


def test_drills_page(client):
    res = client.get("/drills", follow_redirects=True)
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "<title>CueView — Drills</title>" in html
    for element in ('id="feed"', 'id="overlay"', 'id="steps"', 'id="tip-diagram"', 'id="speed-meter"', 'id="result"'):
        assert element in html
    assert 'href="/drills"' in client.get("/").get_data(as_text=True)


def test_drill_api_gives_instructions_and_marks(client):
    data = client.get("/api/drills/stop-shot").get_json()
    assert data["name"] == "Stop shot"
    assert data["state"] == "setup" and data["result"] is None
    assert data["steps"] and data["tip_text"] and 1 <= data["speed"] <= 5
    for key in ("cue_start", "object_start", "pocket", "cue_target"):
        x, y = data["marks"][key]
        assert 0 < x < 1280 and 0 < y < 720


def test_unknown_drill_404(client):
    assert client.get("/api/drills/no-such-drill").status_code == 404


@pytest.mark.parametrize("path", ["/", "/drills", "/debug"])
def test_every_page_has_the_same_nav(client, path):
    import re
    html = client.get(path, follow_redirects=True).get_data(as_text=True)  # /drills -> /drills/<id>
    nav = re.search(r'<nav class="nav-links".*?</nav>', html, re.S).group(0)
    links = re.findall(r'<a class="nav-link" href="([^"]+)"( aria-current="page")?>([^<]+)</a>', nav)
    assert [(href, label) for href, _, label in links] == [("/", "Scoreboard"), ("/drills", "Drills"), ("/debug", "Debug")]
    assert [href for href, current, _ in links if current] == [path]


def test_each_drill_has_a_page_with_navigation(client):
    from src.drills import DRILL_ORDER
    for i, drill in enumerate(DRILL_ORDER):
        html = client.get(f"/drills/{drill.id}").get_data(as_text=True)
        assert f'data-drill-id="{drill.id}"' in html
        for other in DRILL_ORDER:  # tabs to every drill
            assert f'href="/drills/{other.id}"' in html
        assert f"Drill {i + 1} of {len(DRILL_ORDER)}" in html
        assert ('class="drill-step prev"' in html) == (i > 0)
        assert ('class="drill-step next"' in html) == (i < len(DRILL_ORDER) - 1)
        data = client.get(f"/api/drills/{drill.id}").get_data(as_text=True)
        assert drill.name in data


def test_selecting_a_drill_tells_capture(client, tmp_path, monkeypatch):
    import web.app as webapp
    from src.drills import read_selection
    monkeypatch.setattr(webapp, "OUTPUTS_DIR", tmp_path)
    assert client.post("/api/drills/cut-shot/select").status_code == 200
    assert read_selection(tmp_path) == "cut-shot"
    # /drills opens the drill capture is running
    assert client.get("/drills").headers["Location"].endswith("/drills/cut-shot")
    assert client.post("/api/drills/nope/select").status_code == 404
