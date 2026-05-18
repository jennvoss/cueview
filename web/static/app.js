const STATUS_POLL_MS = 500;
const SHOTS_POLL_MS = 2000;
const FRAME_POLL_MS = 500;

const $ = (id) => document.getElementById(id);

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function fmtPos(p) {
  if (!p) return "—";
  return `(${fmt(p[0], 1)}, ${fmt(p[1], 1)})`;
}

function renderMatrix(el, m) {
  el.innerHTML = "";
  if (!m) return;
  for (const row of m) {
    const tr = document.createElement("tr");
    for (const v of row) {
      const td = document.createElement("td");
      td.textContent = fmt(v, 4);
      tr.appendChild(td);
    }
    el.appendChild(tr);
  }
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

async function loadConfigOnce() {
  const cfg = await fetchJSON("/api/config");
  if (!cfg) return;

  $("cap-index").textContent = cfg.camera?.index ?? "—";
  $("cap-res").textContent = `${cfg.camera?.width ?? "?"} × ${cfg.camera?.height ?? "?"}`;
  $("cap-fps-target").textContent = cfg.camera?.fps ?? "—";
  $("cap-logi").textContent = cfg.camera?.require_logitech ? "yes" : "no";

  $("det-dp").textContent = cfg.detection?.hough_dp ?? "—";
  $("det-mindist").textContent = cfg.detection?.hough_min_dist_px ?? "—";
  $("det-params").textContent =
    `${cfg.detection?.hough_param1 ?? "?"} / ${cfg.detection?.hough_param2 ?? "?"}`;
  $("det-tol").textContent = cfg.detection?.diameter_tolerance ?? "—";

  const hsv = cfg.detection?.hsv_ranges ?? {};
  const hsvTable = $("det-hsv");
  hsvTable.innerHTML = "<thead><tr><th>label</th><th>low</th><th>high</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const [label, range] of Object.entries(hsv)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${JSON.stringify(range.low ?? range[0])}</td><td>${JSON.stringify(range.high ?? range[1])}</td>`;
    tbody.appendChild(tr);
  }
  hsvTable.appendChild(tbody);

  $("trk-method").textContent = cfg.tracking?.method ?? "—";
  $("trk-reacquire").textContent = cfg.tracking?.max_reacquire_frames ?? "—";
  $("met-smoother").textContent = cfg.metrics?.smoother_window ?? "—";
  $("out-file").textContent = cfg.output?.log_file ?? "—";
}

async function loadCalibrationOnce() {
  const cal = await fetchJSON("/api/calibration");
  if (!cal) {
    $("cal-err").textContent = "no calibration file";
    return;
  }
  $("cal-err").textContent = fmt(cal.reprojection_error, 4);
  renderMatrix($("cal-camera-matrix"), cal.camera_matrix);
  renderMatrix($("cal-homography"), cal.homography);
  $("cal-dist").textContent = (cal.dist_coeffs || []).map((v) => fmt(v, 4)).join(", ");
}

function updateStatusPill(state, meta) {
  const pill = $("status-pill");
  pill.dataset.state = state;
  $("status-label").textContent = state.toUpperCase();
  $("status-meta").textContent = meta || "";
}

async function pollStatus() {
  const s = await fetchJSON("/api/status");
  if (!s) {
    updateStatusPill("idle", "no connection");
    return;
  }

  if (s.status === "running") {
    updateStatusPill("running", `frame ${s.frame_index}`);
  } else {
    const tail = s.stale_seconds ? `stale ${s.stale_seconds}s` : (s.message || "no live capture");
    updateStatusPill("idle", tail);
  }

  $("cap-fps-measured").textContent = fmt(s.measured_fps, 1);
  $("cap-frame").textContent = s.frame_index ?? "—";
  $("det-count").textContent = s.detections_this_frame ?? "—";

  const tracks = s.tracked_balls || [];
  $("trk-count").textContent = tracks.length;
  const tbody = $("trk-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const t of tracks) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.id}</td>
      <td>${t.is_cue_ball ? "✓" : ""}</td>
      <td>${fmtPos(t.center_px)}</td>
      <td>${fmtPos(t.center_in)}</td>
      <td>${t.frames_missing}</td>`;
    if (t.is_cue_ball) tr.classList.add("cue");
    tbody.appendChild(tr);
  }

  $("met-speed").textContent = fmt(s.current_speed_mph, 2);
  $("met-peak").textContent = fmt(s.peak_speed_mph, 2);
  $("met-inshot").textContent = s.is_in_shot ? "yes" : "no";
  $("met-inshot").className = s.is_in_shot ? "in-shot" : "";
  $("met-target").textContent = fmtPos(s.target_pos_in);

  const last = s.last_shot;
  if (last) {
    $("met-last-shot").innerHTML = `
      <div><strong>${fmt(last.peak_speed_mph, 2)}</strong> mph peak,
           <strong>${fmt(last.mean_speed_mph, 2)}</strong> mph mean,
           ${fmt(last.angle_deg, 1)}° angle</div>
      <div>start ${fmtPos(last.start_pos_in)} → end ${fmtPos(last.end_pos_in)}</div>
      <div>${last.timestamp_iso || ""}</div>
      ${last.accuracy_in != null ? `<div>accuracy: ${fmt(last.accuracy_in, 2)} in</div>` : ""}`;
  }
}

async function pollShots() {
  const data = await fetchJSON("/api/shots");
  if (!data) return;
  const shots = data.shots || [];
  $("out-count").textContent = shots.length;
  if (shots.length > 0) {
    const peaks = shots.map((s) => s.peak_speed_mph || 0);
    const mean = peaks.reduce((a, b) => a + b, 0) / peaks.length;
    $("out-mean").textContent = fmt(mean, 2) + " mph";
    $("out-max").textContent = fmt(Math.max(...peaks), 2) + " mph";
  } else {
    $("out-mean").textContent = "—";
    $("out-max").textContent = "—";
  }
  const tbody = $("out-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const s of shots) {
    const tr = document.createElement("tr");
    const t = (s.timestamp_iso || "").replace("T", " ").slice(0, 19);
    tr.innerHTML = `
      <td>${t}</td>
      <td>${fmt(s.peak_speed_mph, 2)}</td>
      <td>${fmt(s.mean_speed_mph, 2)}</td>
      <td>${fmt(s.angle_deg, 1)}</td>
      <td>${fmtPos(s.start_pos_in)}</td>
      <td>${fmtPos(s.end_pos_in)}</td>
      <td>${s.accuracy_in != null ? fmt(s.accuracy_in, 2) : "—"}</td>`;
    tbody.appendChild(tr);
  }
}

function refreshFrame() {
  const img = $("frame");
  const empty = $("frame-empty");
  const url = "/api/frame.jpg?t=" + Date.now();
  const probe = new Image();
  probe.onload = () => {
    img.src = url;
    img.style.display = "block";
    empty.style.display = "none";
  };
  probe.onerror = () => {
    img.style.display = "none";
    empty.style.display = "block";
  };
  probe.src = url;
}

(async function init() {
  await loadConfigOnce();
  await loadCalibrationOnce();
  pollStatus();
  pollShots();
  refreshFrame();
  setInterval(pollStatus, STATUS_POLL_MS);
  setInterval(pollShots, SHOTS_POLL_MS);
  setInterval(refreshFrame, FRAME_POLL_MS);
})();
