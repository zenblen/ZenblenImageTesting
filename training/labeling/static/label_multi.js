"use strict";
// Multi-mode segmentation labeler. Free-draw, multi-shape (multiple same-class
// polygons per image). The active MODE drives the whole theme; only the active
// mode's shapes are ever on the canvas (each mode is an independent pass whose
// work is saved to its own dataset). Points are kept in IMAGE (natural) pixels;
// the canvas backing store is the natural image size and CSS scales it to fit.

const MODES = {
  standard: { name: "MODE 1 · STANDARD (in-cup)", dataset: "→ standard_dataset", color: "#39c07a", seedEndpoint: null              },
  spill:    { name: "MODE 2 · SPILLED SMOOTHIE",   dataset: "→ spill_dataset",    color: "#e8833a", seedEndpoint: null              },
  logo:     { name: "MODE 3 · LOGO",               dataset: "→ logo_dataset",     color: "#a06cd5", seedEndpoint: null              },
  chunk:    { name: "MODE 4 · CHUNK (unblended)",  dataset: "→ chunk_dataset",    color: "#e0524a", seedEndpoint: "/api/chunk_seed" },
  unmixed:  { name: "MODE 5 · UNMIXED (streaks)",  dataset: "→ unmixed_dataset",  color: "#2ec5c5", seedEndpoint: null              },
  powder:   { name: "MODE 6 · POWDER (on table)",  dataset: "→ powder_dataset",   color: "#e0a33f", seedEndpoint: null              },
};
const KEY_TO_MODE = { "1": "standard", "2": "spill", "3": "logo", "4": "chunk", "5": "unmixed",
                      "6": "powder" };

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const doneEl = document.getElementById("done");
const frameEl = document.getElementById("frame");
const fileidEl = document.getElementById("fileid");

// ?mode=<m> picks the initial pipeline so a deep link opens the right pass.
// Falls back to spill (the primary hand pipeline) for a bare URL or an
// unknown value, so a typo cannot leave the UI in a mode the API rejects.
let mode = (() => {
  const m = new URLSearchParams(location.search).get("mode");
  return m && MODES[m] ? m : "spill";
})();
let fileId = null;
let shapes = [];           // [[[x,y],...], ...]  committed + active polygons
let activeIdx = 0;         // index into shapes currently being drawn/edited
let status = null;         // saved status for (fileId, mode): 'labeled'|'clean'|null
let dirty = false;         // any unsaved change this image/mode
let img = new Image();
let drag = null;           // {s, v} vertex being dragged
let undoStack = [];
const visited = [];
let histIndex = -1;

// ---- coordinate + hit helpers ----------------------------------------------
function cssScale() {
  const rect = canvas.getBoundingClientRect();
  return canvas.width ? rect.width / canvas.width : 1;
}
function toImageCoords(evt) {
  const rect = canvas.getBoundingClientRect();
  const s = rect.width / canvas.width;
  return [(evt.clientX - rect.left) / s, (evt.clientY - rect.top) / s];
}
function hitTol() { return 9 / cssScale(); }
function handleRadius() { return 6 / cssScale(); }

function hitVertex(pt) {
  const tol = hitTol();
  let best = null, bestD = tol;
  for (let s = 0; s < shapes.length; s++) {
    for (let v = 0; v < shapes[s].length; v++) {
      const d = Math.hypot(shapes[s][v][0] - pt[0], shapes[s][v][1] - pt[1]);
      if (d <= bestD) { bestD = d; best = { s, v }; }
    }
  }
  return best;
}
function segDist(pt, a, b) {
  const vx = b[0]-a[0], vy = b[1]-a[1];
  const wx = pt[0]-a[0], wy = pt[1]-a[1];
  const len2 = vx*vx + vy*vy || 1e-9;
  let t = Math.max(0, Math.min(1, (wx*vx + wy*vy) / len2));
  return Math.hypot(pt[0]-(a[0]+t*vx), pt[1]-(a[1]+t*vy));
}
function hitEdge(pt) {
  const tol = hitTol();
  let best = null, bestD = tol;
  for (let s = 0; s < shapes.length; s++) {
    const p = shapes[s];
    if (p.length < 2) continue;
    for (let i = 0; i < p.length; i++) {
      const d = segDist(pt, p[i], p[(i+1) % p.length]);
      if (d <= bestD) { bestD = d; best = { s, i }; }
    }
  }
  return best;
}
function clamp(x, hi) { return Math.max(0, Math.min(hi, Math.round(x))); }

// ---- rendering --------------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const color = MODES[mode].color;
  ctx.lineWidth = 2 / cssScale();
  for (let s = 0; s < shapes.length; s++) {
    const p = shapes[s];
    if (p.length) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(p[0][0], p[0][1]);
      for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
      if (p.length >= 3) ctx.closePath();
      // Spill/chunk: outline only — fill covers the region and makes it harder
      // to see/identify what you're labeling. Other modes keep translucent fill.
      if (mode !== "spill" && mode !== "chunk" && mode !== "unmixed") {
        ctx.fillStyle = hexA(color, s === activeIdx ? 0.22 : 0.12);
        ctx.fill();
      }
      ctx.stroke();
    }
    const r = handleRadius();
    for (let v = 0; v < p.length; v++) {
      ctx.beginPath();
      ctx.arc(p[v][0], p[v][1], r, 0, Math.PI * 2);
      ctx.fillStyle = (drag && drag.s === s && drag.v === v) ? "#ffd166"
                    : (s === activeIdx ? color : "#8b94a3");
      ctx.fill();
      ctx.lineWidth = 1 / cssScale();
      ctx.strokeStyle = "#fff";
      ctx.stroke();
    }
  }

  const active = shapes[activeIdx] || [];
  if (!active.length) {
    ctx.save();
    ctx.fillStyle = "#ffcc00";
    ctx.font = `${Math.round(22 / cssScale())}px system-ui, sans-serif`;
    ctx.fillText("click to place points", 12 / cssScale(), 40 / cssScale());
    ctx.restore();
  }
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}

// ---- edit ops ---------------------------------------------------------------
function snapshot() { undoStack.push(JSON.stringify(shapes)); }
function markDirty() { dirty = true; updateSidebar(); }

canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const hit = hitVertex(toImageCoords(e));
  if (hit) { snapshot(); shapes[hit.s].splice(hit.v, 1); markDirty(); draw(); }
});
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !fileId) return;
  const pt = toImageCoords(e);
  const hv = hitVertex(pt);
  if (hv) {
    snapshot(); drag = hv; activeIdx = hv.s;
    canvas.setPointerCapture(e.pointerId); draw(); return;
  }
  const he = hitEdge(pt);
  if (he) {
    snapshot();
    shapes[he.s].splice(he.i + 1, 0, [clamp(pt[0], canvas.width), clamp(pt[1], canvas.height)]);
    drag = { s: he.s, v: he.i + 1 }; activeIdx = he.s; markDirty();
    canvas.setPointerCapture(e.pointerId); draw(); return;
  }
  // Empty space: append a point to the active shape.
  snapshot();
  if (!shapes[activeIdx]) shapes[activeIdx] = [];
  shapes[activeIdx].push([clamp(pt[0], canvas.width), clamp(pt[1], canvas.height)]);
  markDirty();
  canvas.setPointerCapture(e.pointerId); draw();
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const pt = toImageCoords(e);
  shapes[drag.s][drag.v] = [clamp(pt[0], canvas.width), clamp(pt[1], canvas.height)];
  markDirty(); draw();
});
canvas.addEventListener("pointerup", () => { drag = null; draw(); });

// ---- keyboard ---------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (KEY_TO_MODE[k]) { switchMode(KEY_TO_MODE[k]); return; }
  if (k === "arrowleft")  { e.preventDefault(); goPrev(); return; }
  if (k === "arrowright") { e.preventDefault(); goNext(); return; }
  if (!fileId) return;
  switch (k) {
    case "enter": save(); break;
    case "k": markClean(); break;
    case "s": skip(); break;
    case "n": newShape(); break;
    case "x": clearActive(); break;
    case "d": deleteActiveShape(); break;
    case "z": undo(); break;
  }
});

function newShape() {
  const a = shapes[activeIdx];
  if (a && a.length < 3) return;              // nothing worth finishing
  snapshot();
  shapes.push([]);
  activeIdx = shapes.length - 1;
  updateSidebar(); draw();
}
function clearActive() {
  if (!shapes[activeIdx] || !shapes[activeIdx].length) return;
  snapshot(); shapes[activeIdx] = []; drag = null; markDirty(); draw();
}
// Remove the whole active polygon (click any of its vertices/edges first to
// make it active). Always leaves at least one blank shape to keep drawing.
function deleteActiveShape() {
  const a = shapes[activeIdx];
  if (!a || (shapes.length === 1 && !a.length)) return;   // nothing to delete
  snapshot();
  shapes.splice(activeIdx, 1);
  if (!shapes.length) shapes = [[]];
  activeIdx = Math.min(activeIdx, shapes.length - 1);
  drag = null; markDirty(); updateSidebar(); draw();
}
function undo() {
  if (!undoStack.length) return;
  shapes = JSON.parse(undoStack.pop());
  if (activeIdx >= shapes.length) activeIdx = Math.max(0, shapes.length - 1);
  if (!undoStack.length) dirty = false;
  updateSidebar(); draw();
}

// ---- validity + save --------------------------------------------------------
function validShapes() {
  return shapes.filter((p) => p.length >= 3).map((p) => ({ polygon: p }));
}
function save() {
  const v = validShapes();
  if (!v.length) { flashWarn("Draw at least one shape (≥3 points), or Mark Clean."); return; }
  post({ file_id: fileId, mode, status: "labeled", shapes: v }, () => { dirty = false; goNext(); });
}
function markClean() {
  // Guard: Clean means "no anomaly" — refuse to silently discard drawn work.
  const v = validShapes();
  if (v.length) {
    if (!confirm(`This image has ${v.length} ${mode} shape(s). Mark CLEAN and DELETE them?`)) return;
  }
  post({ file_id: fileId, mode, status: "clean", shapes: [] }, () => { dirty = false; goNext(); });
}
function skip() { goNext(); }   // no state written (spec: advance without saving)

function post(body, done) {
  fetch("/api/annotate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(() => { refreshProgress(); done && done(); })
    .catch((err) => flashWarn("Save failed: " + err.message));
}

// ---- mode switching ---------------------------------------------------------
function switchMode(m) {
  if (m === mode) return;
  if (dirty && validShapes().length &&
      !confirm(`Unsaved ${mode} shapes will be discarded when you switch. Continue?`)) return;
  mode = m;
  applyTheme();
  if (fileId != null) loadItem(fileId);   // keep the current image in the new mode
  else goNext();
  refreshProgress();
}
function applyTheme() {
  document.documentElement.dataset.mode = mode;
  document.getElementById("modename").textContent = MODES[mode].name;
  document.getElementById("dataset").textContent = MODES[mode].dataset;
  document.title = ({ standard: "🟢", spill: "🟠", logo: "🟣", chunk: "🔴",
                      unmixed: "🔵", powder: "🟡" }[mode] ?? "⚪") + " label · " + mode;
  document.querySelectorAll(".modebtn").forEach((b) =>
    b.classList.toggle("active", b.dataset.m === mode));
}

// ---- navigation -------------------------------------------------------------
function goNext() {
  if (histIndex >= 0 && histIndex < visited.length - 1) {
    histIndex++; loadItem(visited[histIndex]); return;
  }
  const after = fileId != null ? `&after=${fileId}` : "";
  fetch(`/api/next?mode=${mode}${after}`).then((r) => r.json()).then((d) => {
    if (d.done) { showDone(); return; }
    setItem(d);
    if (visited[visited.length - 1] !== d.file_id) visited.push(d.file_id);
    histIndex = visited.length - 1;
  });
}
function goPrev() {
  if (histIndex > 0) { histIndex--; loadItem(visited[histIndex]); return; }
  // At the left edge of this session's history: seek the previous image in the
  // WHOLE pool (any status), so you can always go back and re-edit an image even
  // if it wasn't touched this run. Prepend it so forward replay keeps working.
  if (fileId == null) return;
  fetch(`/api/seek?mode=${mode}&file_id=${fileId}&dir=prev`)
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => {
      if (d.edge) { flashWarn("Already at the first image."); return; }
      setItem(d);
      if (visited[0] !== d.file_id) visited.unshift(d.file_id);
      histIndex = 0;
    })
    .catch((err) => flashWarn(
      "Can't go back (" + err.message + "). Restart the server so /api/seek loads."));
}
function loadItem(id) {
  fetch(`/api/item/${id}?mode=${mode}`).then((r) => r.json()).then(setItem);
}

function setItem(d) {
  fileId = d.file_id;
  status = d.status || null;
  shapes = (d.shapes && d.shapes.length) ? d.shapes.map((p) => p.slice()) : [[]];
  activeIdx = shapes.length - 1;
  dirty = false; drag = null; undoStack = [];
  doneEl.style.display = "none"; frameEl.style.display = "";
  fileidEl.textContent = `#${d.file_id}` + (status ? ` (was: ${status})` : "");
  document.getElementById("priobadge").classList.toggle("on", !!d.priority);
  updateSidebar();

  img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    // Seed this mode from its candidate endpoint if we have no shapes yet
    // (chunk mode: classical/YOLO seeds from run_chunk_seed.py).
    if (MODES[mode].seedEndpoint && !status && shapes.every((p) => !p.length)) {
      fetch(`${MODES[mode].seedEndpoint}/${d.file_id}`).then((r) => r.json()).then((s) => {
        if (s.shapes && s.shapes.length) {
          shapes = s.shapes.map((p) => p.slice());
          activeIdx = shapes.length - 1;
          updateSidebar();
        } else if (s.polygon && s.polygon.length >= 3) {
          shapes = [s.polygon]; activeIdx = 0; updateSidebar();
        }
        draw();
      });
    } else {
      draw();
    }
  };
  img.src = `/image/${d.file_id}?t=${d.file_id}`;
}

function showDone() {
  fileId = null; frameEl.style.display = "none"; doneEl.style.display = "block";
  fileidEl.textContent = "";
  document.getElementById("priobadge").classList.remove("on");
  updateSidebar();
}

// ---- sidebar / progress -----------------------------------------------------
function flashWarn(msg) {
  const w = document.getElementById("warn");
  w.textContent = msg; w.style.display = "block";
  clearTimeout(flashWarn._t);
  flashWarn._t = setTimeout(() => { w.style.display = "none"; }, 3500);
}
function updateSidebar() {
  const n = validShapes().length;
  document.getElementById("layercount").innerHTML =
    fileId ? `${mode} shapes: <b>${n}</b>${dirty ? " ·  unsaved" : ""}` : "";
  const cleanBtn = document.getElementById("btnClean");
  // Clean is only meaningful with no shapes; keep it enabled but it will confirm.
  cleanBtn.classList.toggle("clean", n === 0);
  ["btnSave", "btnClean", "btnSkip", "btnNew", "btnDel"].forEach((id) =>
    (document.getElementById(id).disabled = !fileId));
}
function refreshProgress() {
  fetch("/api/progress").then((r) => r.json()).then((p) => {
    const rows = Object.entries(p.modes).map(([m, c]) => {
      const prio = c.priority_total
        ? ` · <span class="prio">priority ${c.priority_remaining}/${c.priority_total} left</span>`
        : "";
      return `<div><b>${m}</b>: ${c.decided}/${p.total} · ${c.labeled} labeled · ${c.clean} clean${prio}</div>`;
    });
    document.getElementById("counts").innerHTML = rows.join("");
  });
}

// ---- wire buttons + boot ----------------------------------------------------
document.getElementById("btnSave").onclick = save;
document.getElementById("btnClean").onclick = markClean;
document.getElementById("btnSkip").onclick = skip;
document.getElementById("btnNew").onclick = newShape;
document.getElementById("btnDel").onclick = deleteActiveShape;
document.querySelectorAll(".modebtn").forEach((b) =>
  (b.onclick = () => switchMode(b.dataset.m)));

applyTheme();
refreshProgress();
goNext();
