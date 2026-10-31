const STATUS_POLL_MS = 500;
const FEED_POLL_MS = 300;
const SOLIDS = [1, 2, 3, 4, 5, 6, 7];
const STRIPES = [9, 10, 11, 12, 13, 14, 15];

const $ = (id) => document.getElementById(id);

function ballEl(n) {
  const el = document.createElement("div");
  el.className = "ball";
  el.dataset.n = n;
  if (n === "cue") {
    el.classList.add("cue");
    el.title = "Cue ball";
  } else {
    if (n > 8) el.classList.add("stripe");
    el.title = `${n} ball`;
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = n;
    el.appendChild(num);
  }
  return el;
}

function renderLiveState(s) {
  const live = Boolean(s && s.live);
  const badge = $("feed-badge");
  badge.hidden = live || !feedLoaded;
  const kind = s && s.status === "running" && s.source && s.source.kind === "image" ? "still image" : "last frame";
  badge.textContent = `Not live · ${kind}`;

  const note = $("score-note");
  const still = s && s.status === "running" && s.source && s.source.kind === "image";
  note.hidden = live;
  if (still) {
    note.textContent = s.source.annotated
      ? "Score from a saved frame that has detection markers drawn on it, so it may be off."
      : "Score from the saved still image.";
  } else {
    note.textContent = s && s.score ? "Last known score (capture isn't running)."
                                     : "No score yet. It appears once capture is running.";
  }
}

function renderBalls(container, balls, emptyText) {
  container.replaceChildren(...balls.map(ballEl));
  if (balls.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
  }
}

function renderScore(score) {
  if (!score) {
    // No score saved yet: don't claim the table is empty.
    for (const id of ["balls-table", "balls-solids", "balls-stripes"]) renderBalls($(id), [], "–");
    for (const id of ["count-table", "count-solids", "count-stripes"]) $(id).textContent = "";
    $("cue-chip").hidden = true;
    $("notice").hidden = true;
    return;
  }
  // The cue ball isn't one of the balls to sink: it gets a marker in the section
  // header (replaced by the scratch notice when it's off the table), not a spot in the list.
  const onTable = (score.on_table || []).filter((b) => b !== "cue");
  renderBalls($("balls-table"), onTable, "No balls on the table");
  $("cue-chip").hidden = !(score.on_table || []).includes("cue");
  renderBalls($("balls-solids"), score.solids_sunk || [], "None yet");
  renderBalls($("balls-stripes"), score.stripes_sunk || [], "None yet");

  $("count-table").textContent = onTable.length;
  $("count-solids").textContent = `${(score.solids_sunk || []).length} / ${SOLIDS.length}`;
  $("count-stripes").textContent = `${(score.stripes_sunk || []).length} / ${STRIPES.length}`;

  const notes = [];
  if (score.eight_sunk) notes.push("8 ball sunk");
  if (score.cue_sunk) notes.push("Scratch: cue ball off the table");
  const notice = $("notice");
  notice.hidden = notes.length === 0;
  notice.textContent = notes.join(" · ");
}

function setStatus(state) {
  $("status-pill").dataset.state = state;
  $("status-label").textContent = state.toUpperCase();
}

async function pollStatus() {
  let s = null;
  try {
    const res = await fetch("/api/status");
    if (res.ok) s = await res.json();
  } catch (e) { /* server not reachable */ }
  lastStatus = s;
  if (!s) {
    setStatus("idle");
    renderLiveState(null);
    return;
  }
  setStatus(s.status === "running" ? "running" : "idle");
  renderLiveState(s);
  renderScore(s.score || null);
}

let feedLoaded = false;
let lastStatus = null;

function refreshFeed() {
  const url = "/api/frame.jpg?t=" + Date.now();
  const probe = new Image();
  probe.onload = () => {
    $("feed").src = url;
    $("feed").style.display = "block";
    $("feed-empty").style.display = "none";
    if (!feedLoaded) {
      feedLoaded = true;
      renderLiveState(lastStatus);
    }
  };
  probe.onerror = () => {
    if (!feedLoaded) $("feed-empty").style.display = "flex";
  };
  probe.src = url;
}

renderScore(null);
pollStatus();
refreshFeed();
setInterval(pollStatus, STATUS_POLL_MS);
setInterval(refreshFeed, FEED_POLL_MS);
