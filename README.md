# CueView

An automatic 8-ball scorekeeper. A camera above the pool table watches each shot,
works out which balls went into the pockets, and keeps score in a browser
dashboard. Under the hood, frames go through capture → undistort → detect →
track → score, and a debug dashboard shows what each stage is doing.

## Requirements

- **Python 3.11+**
- macOS 13+ (tested) — should also work on Linux with a UVC camera
- USB webcam — a Logitech model is recommended; any UVC-compatible camera works
- For calibration: a printed checkerboard pattern (default 9×6 inner corners)

## Install

```bash
git clone <this repo>            # if you haven't already
cd cueview
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Calibration (one-time)

The capture pipeline expects `config/calibration.npz` with the camera intrinsics
and a pixel→inches homography for the table. If you don't have a calibration
file you can either:

- Run a calibration session against a printed checkerboard placed on the table
  (see `src/calibration.py` — `run_calibration()`), which produces
  `config/calibration.npz`, or
- Skip calibration entirely with `--no-calibration` (uses identity matrices —
  positions will be reported in pixels rather than table inches).

## Run the capture pipeline

In one terminal:

```bash
python3 -m src.main
```

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--camera-index N` | from config | Pick a specific camera device index |
| `--config PATH` | `config/config.yaml` | Use a different config file |
| `--no-calibration` | off | Use identity matrices instead of `calibration.npz` |
| `--record` | off | Record raw session video to `outputs/session_<ts>.mp4` |
| `--output-dir PATH` | `outputs/` | Override directory for logs and recordings |
| `--log-level LEVEL` | from config | `DEBUG` / `INFO` / `WARNING` |
| `--status-interval N` | `5` | Write `outputs/status.json` every N frames (0 to disable) |
| `--no-frame-dump` | off | Skip writing `outputs/latest.jpg` (status JSON still written) |
| `--web-data-dir PATH` | (output dir) | Override directory for `status.json` and `latest.jpg` |
| `--image PATH` | (camera) | Score a still image instead of the camera |

### Running without the camera

If no camera is connected, capture doesn't exit. It scores the last saved frame
instead, and the scoreboard marks the feed **Not live**. While the camera is
running, capture saves an unmarked frame to `outputs/latest_raw.png` every 5 s
for this. If that file doesn't exist yet, it falls back to `outputs/latest.jpg`,
which has detection markers drawn on it, so the score may be off. To score any
image, pass it directly:

```bash
python3 -m src.main --image tests/fixtures/table/12_midgame.png
```

### Keyboard controls (with the capture window focused)

| Key | Action |
|---|---|
| `q` | Quit |
| `o` | Toggle overlay (HUD, ball annotations, trail) |
| `t` | Set the current cue-ball position as a target — the next shot will report accuracy |
| `s` | Save the raw frame (no overlay) as a PNG to `tests/fixtures/table/` |
| `p` | Print the current `table_polygon` as YAML (only with `TUNE_TABLE_POLYGON=true`) |

### Tuning the table polygon

Set `TUNE_TABLE_POLYGON=true` in `.env` to add x/y sliders for each
`detection.table_polygon` corner to the capture window. They start at the values
in `config.yaml`, and the outline and detection filter update live as you drag.
Press `p` to print the result, then paste it into `config.yaml`. Defaults to
`false`.

`.env` is not committed; create it from the template on a fresh checkout:

```bash
cp .env.example .env
```

## Run the browser dashboard

In a **second terminal** (with the venv activated):

```bash
python3 -m web.app
```

Then open:

- <http://localhost:5000> — **Scorekeeper** (8-ball): the camera feed on the left;
  on the right, the balls on the table, the solids sunk, and the stripes sunk.
  A ball counts as sunk once it hasn't been seen for 3 s, and as back on the
  table after 1 s in view (`SINK_AFTER_S` / `RETURN_AFTER_S` in `src/constants.py`).
- <http://localhost:5000/debug> — **Debug dashboard**, described below.

The debug dashboard works whether or not capture is currently running:

- **With capture running** — Capture, Detection, Tracking, and Metrics panels
  update live every ~500 ms, and the latest annotated frame is shown above the
  grid. The status pill switches to `RUNNING`.
- **Standalone** — Config and Calibration panels still render, and the Output
  panel shows all completed shots from `outputs/session.jsonl`.

Each panel displays the data produced by one stage of the pipeline:

| Panel | What it shows |
|---|---|
| **Capture** | Camera index, configured resolution, target / measured FPS, frame index |
| **Calibration** | Reprojection error, camera matrix, distortion coefficients, pixel→inch homography |
| **Detection** | Per-frame detection count + Hough/HSV configuration |
| **Tracking** | Currently-tracked balls (id, cue flag, pixel + inch position, frames missing) |
| **Metrics** | In-shot flag, target, last completed shot |
| **Output** | Total shot count, scrollable per-shot history table |

## Project layout

```
cueview/
├── config/
│   ├── config.yaml          # Tunable parameters for every stage
│   └── calibration.npz      # (created by calibration; gitignored)
├── src/
│   ├── main.py              # Entry point / capture loop + StatusWriter
│   ├── capture.py           # Camera enumeration + frame grabber
│   ├── calibration.py       # Intrinsics + homography
│   ├── detection.py         # HSV + Hough circle ball detector
│   ├── tracking.py          # Kalman cue-ball tracker + centroid matcher
│   ├── metrics.py           # Shot detection, angle, accuracy
│   ├── output.py            # Session log + overlay renderer
│   ├── config_loader.py     # YAML loader
│   └── constants.py         # Shared constants (table size, thresholds, ...)
├── web/
│   ├── app.py               # Flask app (port 5000)
│   ├── templates/index.html # Dashboard UI
│   └── static/{app.js, style.css}
├── tests/                   # pytest suites (unit + integration)
├── outputs/                 # gitignored — session.jsonl, status.json, latest.jpg
├── requirements.txt
└── README.md
```

## Running tests

```bash
pytest
```

All unit and integration suites should pass green from a clean checkout.

## Troubleshooting

- **`CameraNotFoundError`** — confirm the camera is plugged in and check that
  `camera.require_logitech: false` in `config/config.yaml` if you're using a
  non-Logitech webcam. Use `--camera-index N` to pick the device explicitly.
- **`config/calibration.npz` missing** — run a calibration session or pass
  `--no-calibration` to use identity matrices.
- **Dashboard shows IDLE while capture is running** — check that
  `outputs/status.json` is being written (`ls -la outputs/`) and confirm
  `--status-interval` isn't `0`.
- **Frame image doesn't refresh** — confirm `outputs/latest.jpg` exists and
  that you didn't pass `--no-frame-dump`.
