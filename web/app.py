import json
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, render_template, send_file, url_for

from src.config_loader import load_config
from src.drills import DRILLS, DRILL_ORDER, DrillTracker, read_selection, write_selection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
CALIBRATION_PATH = PROJECT_ROOT / "config" / "calibration.npz"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATUS_PATH = OUTPUTS_DIR / "status.json"
FRAME_PATH = OUTPUTS_DIR / "latest.jpg"
SESSION_LOG_PATH = OUTPUTS_DIR / "session.jsonl"

STATUS_STALE_SECONDS = 5.0

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/debug")
def debug():
    return render_template("debug.html")


@app.route("/drills")
def drills():
    """Open the drill last chosen (capture is running it), or the first one."""
    return redirect(url_for("drill_page", drill_id=read_selection(OUTPUTS_DIR) or DRILL_ORDER[0].id))


@app.route("/drills/<drill_id>")
def drill_page(drill_id):
    if drill_id not in DRILLS:
        return redirect(url_for("drills"))
    order = [{"id": d.id, "name": d.name} for d in DRILL_ORDER]
    index = next(i for i, d in enumerate(DRILL_ORDER) if d.id == drill_id)
    return render_template("drills.html", drills=order, drill_id=drill_id, index=index)


@app.route("/api/drills/<drill_id>/select", methods=["POST"])
def api_select_drill(drill_id):
    """Make this the drill capture tracks (capture picks it up within a few frames)."""
    if drill_id not in DRILLS:
        return jsonify({"error": f"unknown drill {drill_id}"}), 404
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_selection(OUTPUTS_DIR, drill_id)
    return jsonify({"drill": drill_id})


@app.route("/api/drills/<drill_id>")
def api_drill(drill_id):
    """A drill's instructions and marks on this table, available without capture running."""
    drill = DRILLS.get(drill_id)
    if drill is None:
        return jsonify({"error": f"unknown drill {drill_id}"}), 404
    tracker = DrillTracker(drill)
    try:
        polygon = load_config(str(CONFIG_PATH))["detection"].get("table_polygon")
    except FileNotFoundError:
        polygon = None
    if polygon:
        tracker.update([], 0.0, polygon)
    return jsonify(tracker.snapshot())


@app.route("/api/config")
def api_config():
    try:
        cfg = load_config(str(CONFIG_PATH))
    except FileNotFoundError:
        return jsonify({"error": "config.yaml not found"}), 404
    return jsonify(cfg)


@app.route("/api/calibration")
def api_calibration():
    if not CALIBRATION_PATH.exists():
        return jsonify({"error": "no calibration file"}), 404
    try:
        with np.load(str(CALIBRATION_PATH)) as data:
            payload = {key: data[key].tolist() for key in data.files}
    except Exception as e:
        return jsonify({"error": f"failed to load calibration: {e}"}), 500
    return jsonify(payload)


@app.route("/api/status")
def api_status():
    # "live" is true only while capture is writing fresh status; otherwise the
    # pages show the last saved frame and say it isn't live.
    if not STATUS_PATH.exists():
        return jsonify({"status": "idle", "message": "no live capture", "live": False})
    try:
        age = time.time() - STATUS_PATH.stat().st_mtime
        with open(STATUS_PATH) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({"status": "idle", "message": f"unreadable status: {e}", "live": False})
    if age > STATUS_STALE_SECONDS:
        payload["status"] = "idle"
        payload["stale_minutes"] = int(age // 60)
    source = payload.get("source") or {"kind": "camera"}
    payload["live"] = payload.get("status") == "running" and source.get("kind") != "image"
    return jsonify(payload)


@app.route("/api/shots")
def api_shots():
    if not SESSION_LOG_PATH.exists():
        return jsonify({"shots": []})
    shots = []
    try:
        with open(SESSION_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    shots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        return jsonify({"error": f"failed to read session log: {e}"}), 500
    shots.reverse()
    return jsonify({"shots": shots, "count": len(shots)})


@app.route("/api/frame.jpg")
def api_frame():
    if not FRAME_PATH.exists():
        return jsonify({"error": "no frame available"}), 404
    response = send_file(str(FRAME_PATH), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
