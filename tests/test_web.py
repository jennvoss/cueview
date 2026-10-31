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
    assert "Pool Scorekeeper" in html
    assert 'id="feed"' in html
    for group in ("balls-table", "balls-solids", "balls-stripes"):
        assert f'id="{group}"' in html
    assert 'href="/debug"' in html
    assert 'id="cue-chip"' in html  # cue ball shown in the header, not in the ball list


def test_debug_is_old_dashboard(client):
    res = client.get("/debug")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
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
