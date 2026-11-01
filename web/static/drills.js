const STATUS_POLL_MS = 500;
const FEED_POLL_MS = 300;
const DRILL_ID = document.body.dataset.drillId;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);

const STATE_TEXT = {
  setup: "Place the balls on the marks.",
  ready: "Ready: take your shot.",
  shot: "Watching the shot…",
  result: "Set the balls up again for another try.",
};
const RATING_TEXT = { perfect: "Perfect", good: "Good", pocketed: "Pocketed", missed: "Missed", scratch: "Scratch" };

let drill = null;   // static instructions and marks from /api/drills
let feedLoaded = false;

function svg(tag, attrs, text) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text) el.textContent = text;
  return el;
}

function renderInstructions(d) {
  $("drill-name").textContent = d.name;
  $("drill-summary").textContent = d.summary;
  $("steps").replaceChildren(...d.steps.map((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    return li;
  }));
  // Tip diagram: +x right, +y up in ball radii; the SVG ball has radius 50.
  $("tip-dot").setAttribute("cx", d.tip[0] * 50);
  $("tip-dot").setAttribute("cy", -d.tip[1] * 50);
  $("tip-text").textContent = d.tip_text;
  $("speed-meter").replaceChildren(...[1, 2, 3, 4, 5].map((n) => {
    const s = document.createElement("span");
    if (n <= d.speed) s.className = "on";
    return s;
  }));
  $("speed-text").textContent = `${d.speed} of 5. ${d.speed_text}`;
}

function renderOverlay(d, live) {
  const overlay = $("overlay");
  overlay.replaceChildren();
  const m = d.marks;
  if (!m) return;
  const r = m.ball_radius_px;
  const placed = (live && live.placed) || {};
  const result = live && live.state === "result" ? live.result : null;

  const aim = m.aim || m.object_start;
  const targetR = Math.max(m.target_radius_px || 0, r * 2.2);
  overlay.append(
    // Cue ball to the ghost ball, then the object ball to the pocket.
    svg("line", { class: "aim", x1: m.cue_start[0], y1: m.cue_start[1], x2: aim[0], y2: aim[1] }),
    svg("line", { class: "aim", x1: m.object_start[0], y1: m.object_start[1], x2: m.pocket[0], y2: m.pocket[1] }),
  );
  if (d.kind !== "stop") {
    // Where the cue ball should go after contact.
    overlay.append(svg("line", { class: "cue-path", x1: aim[0], y1: aim[1], x2: m.cue_target[0], y2: m.cue_target[1] }));
  }
  if (d.kind === "cut") {
    overlay.append(
      svg("circle", { class: "ghost", cx: aim[0], cy: aim[1], r }),
      svg("text", { class: "label", x: aim[0] - r * 1.6, y: aim[1] + r * 2.4, "text-anchor": "end" }, "Ghost ball: aim here"),
    );
  }
  overlay.append(
    svg("circle", { class: `mark object${placed.object ? " placed" : ""}`, cx: m.object_start[0], cy: m.object_start[1], r: r + 3 }),
    svg("circle", { class: `mark cue${placed.cue ? " placed" : ""}`, cx: m.cue_start[0], cy: m.cue_start[1], r: r + 3 }),
    svg("circle", { class: "target", cx: m.cue_target[0], cy: m.cue_target[1], r: targetR }),
    svg("text", { class: "label", x: m.object_start[0] + r * 2.6, y: m.object_start[1] + 6 }, "Object ball"),
    svg("text", { class: "label", x: m.cue_start[0] + r * 2.6, y: m.cue_start[1] + 6 }, "Cue ball"),
    svg("text", { class: "label", x: m.cue_target[0] - targetR - 6, y: m.cue_target[1] - targetR, "text-anchor": "end" },
        d.kind === "stop" ? "Cue ball stops here" : "Cue ball target"),
  );
  if (result) {
    if (result.cue_end) overlay.append(svg("circle", { class: "end cue", cx: result.cue_end[0], cy: result.cue_end[1], r: r * 0.55 }));
    if (result.object_end) overlay.append(svg("circle", { class: "end object", cx: result.object_end[0], cy: result.object_end[1], r: r * 0.55 }));
  }
}

function renderLive(live) {
  const state = live ? live.state : null;
  $("drill-state").dataset.state = state || "idle";
  $("state-line").textContent = state ? STATE_TEXT[state] : "Start capture to track your shots.";

  // Steps 1-2 tick off as each ball sits on its mark.
  const placed = (live && live.placed) || {};
  const items = $("steps").children;
  if (items.length >= 2) {
    const ready = state === "ready";
    items[0].classList.toggle("done", Boolean(placed.object || ready));
    items[1].classList.toggle("done", Boolean(placed.cue || ready));
  }

  const result = live && live.result;
  $("result").hidden = !result;
  if (result) {
    $("result").dataset.rating = result.rating;
    $("result-rating").textContent = RATING_TEXT[result.rating] || "";
    $("result-headline").textContent = result.headline;
    const notes = [...result.notes];
    if (result.cue_error_in != null) {
      notes.push(result.cue_error_in < 1 ? "Cue ball finished within an inch of the target."
                                         : `Cue ball finished ${Math.round(result.cue_error_in)} in from the target.`);
    }
    $("result-notes").replaceChildren(...notes.map((n) => {
      const li = document.createElement("li");
      li.textContent = n;
      return li;
    }));
  }

  const stats = (live && live.stats) || { attempts: 0, made: 0, perfect: 0 };
  $("stat-attempts").textContent = stats.attempts;
  $("stat-attempts-label").textContent = stats.attempts === 1 ? "shot" : "shots";
  $("stat-made").textContent = stats.made;
  $("stat-perfect").textContent = stats.perfect;
}

function setStatus(state) {
  $("status-pill").dataset.state = state;
  $("status-label").textContent = state.toUpperCase();
}

async function loadDrill() {
  // Tell capture to track this drill (it switches within a few frames).
  fetch(`/api/drills/${DRILL_ID}/select`, { method: "POST" }).catch(() => {});
  try {
    const res = await fetch(`/api/drills/${DRILL_ID}`);
    if (res.ok) {
      drill = await res.json();
      renderInstructions(drill);
      renderOverlay(drill, null);
    }
  } catch (e) { /* server not reachable */ }
}

async function pollStatus() {
  let s = null;
  try {
    const res = await fetch("/api/status");
    if (res.ok) s = await res.json();
  } catch (e) { /* server not reachable */ }
  const live = s && s.live && s.drill && s.drill.id === DRILL_ID ? s.drill : null;
  const switching = Boolean(s && s.live && s.drill && s.drill.id !== DRILL_ID);
  setStatus(s && s.status === "running" ? "running" : "idle");
  $("feed-badge").hidden = Boolean(s && s.live) || !feedLoaded;
  $("feed-badge").textContent = "Not live";
  renderLive(live);
  if (switching) $("state-line").textContent = "Switching capture to this drill…";
  // Marks from capture follow any live tuning of the table outline.
  if (drill) renderOverlay(live && live.marks ? { ...drill, marks: live.marks } : drill, live);
}

function refreshFeed() {
  const url = "/api/frame.jpg?t=" + Date.now();
  const probe = new Image();
  probe.onload = () => {
    $("feed").src = url;
    $("feed").style.display = "block";
    $("feed-empty").style.display = "none";
    feedLoaded = true;
  };
  probe.onerror = () => {
    if (!feedLoaded) $("feed-empty").style.display = "flex";
  };
  probe.src = url;
}

loadDrill().then(pollStatus);
refreshFeed();
setInterval(pollStatus, STATUS_POLL_MS);
setInterval(refreshFeed, FEED_POLL_MS);
