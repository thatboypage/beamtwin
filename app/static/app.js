// BeamTwin web page: talks to the API in api/main.py and draws the results.
// Plain JavaScript, no build step: the browser runs this file as it is.

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

// Light-to-dark blue ramp for probability (same as the notebook figures).
const RAMP = ["#f9fbfc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];

const state = {
  options: null,
  simulated: null,     // {location, severity, shapes, frequencies} from the last simulation
  result: null,        // last detection result
  zoomed: true,        // show the map around the estimate (true) or every candidate (false)
};

// Which part of the probability grid to show: index ranges [i0, i1] (location) and [j0, j1] (severity).
function mapView() {
  const r = state.result;
  const nLoc = r.locations.length, nSev = r.severities.length;
  if (!state.zoomed) return { i0: 0, i1: nLoc - 1, j0: 0, j1: nSev - 1 };
  const near = (values, centre, half) => {
    let lo = values.findIndex((v) => v >= centre - half);
    let hi = values.length - 1 - [...values].reverse().findIndex((v) => v <= centre + half);
    return [Math.max(lo, 0), Math.min(hi, values.length - 1)];
  };
  const [i0, i1] = near(r.locations, r.location, 0.15);     // ±15 % of the length
  const [j0, j1] = near(r.severities, r.severity, 0.10);    // ±10 % stiffness
  return { i0, i1, j0, j1 };
}

// ---------- Talking to the API ----------

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => `${d.loc.slice(1).join(" → ")}: ${d.msg}`).join("; ")
      : data.detail || `The server answered ${response.status}.`;
    throw new Error(detail);
  }
  return data;
}

function readBeam() {
  const material = state.options.materials[$("material").value];
  return {
    length: Number($("length").value),
    width: Number($("width").value) / 1000,
    thickness: Number($("thickness").value) / 1000,
    E: material.E,
    rho: material.rho,
  };
}

function beamIsValid(beam) {
  return [beam.length, beam.width, beam.thickness].every((v) => Number.isFinite(v) && v > 0);
}

// ---------- Form behaviour ----------

function showError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = !message;
}

function updateSliderLabels() {
  $("location-out").textContent = `${$("location").value} % of the length`;
  $("severity-out").textContent = Number($("severity").value) === 0
    ? "no crack (healthy beam)"
    : `${$("severity").value} % stiffness lost`;
}

function forgetSimulation() {
  state.simulated = null;
  $("use-shapes").checked = false;
  $("use-shapes").disabled = true;
}

let analyzeTimer;
function onBeamChanged() {
  // The old simulated measurement belongs to the old beam, so it no longer applies.
  if (state.simulated) {
    forgetSimulation();
    ["f1", "f2", "f3", "f4"].forEach((id) => ($(id).value = ""));
  }
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(refreshHealthy, 250);
  drawBeam();
}

async function refreshHealthy() {
  const beam = readBeam();
  if (!beamIsValid(beam)) return;
  try {
    const healthy = await api("/api/analyze", beam);
    healthy.frequencies_hz.forEach((f, i) => ($(`f${i + 1}`).placeholder = f.toFixed(2)));
    showError("");
  } catch (err) {
    showError(err.message);
  }
}

async function simulate() {
  const beam = readBeam();
  if (!beamIsValid(beam)) return showError("Enter a positive length, width and thickness first.");
  const button = $("simulate");
  button.disabled = true;
  try {
    const location = Number($("location").value) / 100;
    const severity = Number($("severity").value) / 100;
    const sim = await api("/api/simulate", { beam, location, severity, noise: $("noise").value });
    sim.frequencies_hz.forEach((f, i) => ($(`f${i + 1}`).value = f.toFixed(4)));
    state.simulated = { location, severity, shapes: sim.shapes, frequencies: sim.frequencies_hz.map((f) => f.toFixed(4)) };
    $("use-shapes").disabled = false;
    $("use-shapes").checked = true;
    showError("");
    $("beam-caption").textContent = severity > 0
      ? `A crack is hidden at ${Math.round(location * 100)} % of the length. Now find it.`
      : "No crack this time: a healthy beam. Will the model raise a false alarm?";
    drawBeam();
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
  }
}

function onFrequencyEdited() {
  // Typed-in frequencies no longer match the simulated mode shapes or the hidden crack.
  if (!state.simulated) return;
  const typed = ["f1", "f2", "f3", "f4"].map((id) => $(id).value);
  if (typed.some((v, i) => v !== state.simulated.frequencies[i])) {
    forgetSimulation();
    $("beam-caption").textContent = "Using your own frequencies.";
    drawBeam();
  }
}

async function detect(event) {
  event.preventDefault();
  const beam = readBeam();
  const frequencies = ["f1", "f2", "f3", "f4"].map((id) => Number($(id).value));
  if (!beamIsValid(beam)) return showError("Enter a positive length, width and thickness.");
  if (frequencies.some((f) => !Number.isFinite(f) || f <= 0) || ["f1", "f2", "f3", "f4"].some((id) => $(id).value === "")) {
    return showError("Enter all four frequencies, or simulate a measurement first.");
  }
  const body = { beam, frequencies, noise: $("noise").value };
  if ($("use-shapes").checked && state.simulated) body.shapes = state.simulated.shapes;

  const button = $("detect");
  button.disabled = true;
  button.textContent = "Searching 18,281 possible cracks…";
  try {
    state.result = await api("/api/detect", body);
    showError("");
    renderResult(state.result, frequencies);
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Find damage";
  }
}

// ---------- Results ----------

const pct = (x) => `${Math.round(x * 100)} %`;

function renderResult(r, measured) {
  const damaged = r.probability_damaged >= 0.5;
  const [lo, hi] = r.location_interval;
  const [slo, shi] = r.severity_interval;
  const source = r.used_mode_shapes ? "4 frequencies and 36 mode-shape readings" : "4 frequencies";
  const verdict = $("verdict");
  verdict.innerHTML = damaged
    ? `Damage likely at <strong>${pct(r.location)}</strong> of the length (90 % range ${pct(lo)}–${pct(hi)}),
       with about <strong>${pct(r.severity)}</strong> of its stiffness lost (${pct(slo)}–${pct(shi)}).`
    : `<strong>No clear damage found.</strong>`;
  const sub = document.createElement("span");
  sub.className = "sub";
  let text = `Probability that the beam is damaged: ${Math.round(r.probability_damaged * 100)} %. `
    + `Found from ${source} in ${r.seconds.toFixed(2)} s.`;
  if (!damaged) text += " Small damage near the free tip is hard to rule out, because the beam barely bends there.";
  if (state.simulated) {
    text += state.simulated.severity > 0
      ? ` The hidden crack was at ${pct(state.simulated.location)} with ${pct(state.simulated.severity)} stiffness lost.`
      : " The beam was actually healthy.";
  }
  sub.textContent = text;
  verdict.appendChild(sub);
  verdict.hidden = false;

  $("freq-body").innerHTML = r.healthy_frequencies_hz.map((h, i) => `
    <tr><td>${i + 1}</td><td>${h.toFixed(2)}</td><td>${measured[i].toFixed(2)}</td>
    <td>${r.frequency_drop_pct[i].toFixed(2)} %</td></tr>`).join("");

  const warnings = $("warnings");
  warnings.innerHTML = r.warnings.map((w) => `<li>${w}</li>`).join("");
  warnings.hidden = r.warnings.length === 0;

  state.zoomed = damaged;                 // zoom in when there is a clear estimate to look at
  $("zoom").hidden = !damaged;
  $("detail").hidden = false;
  $("beam-caption").textContent = damaged
    ? "Blue band: where the damage probably is (90 % range). Blue line: best estimate."
    : "No damage is marked, because the model is not confident there is any.";
  drawBeam();
  drawMap();
}

// ---------- Beam drawing ----------

function el(tag, attrs, parent, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function drawBeam() {
  const svg = $("beam-svg");
  [...svg.querySelectorAll(":not(title)")].forEach((n) => n.remove());
  // On narrow screens the drawing is scaled down a lot, so draw it with bigger text,
  // fewer scale marks and more room between the rows of labels.
  const compact = svg.getBoundingClientRect().width < 520;
  const size = compact ? { tick: 24, label: 26, height: 230 } : { tick: 12, label: 13, height: 170 };
  svg.setAttribute("viewBox", `0 0 640 ${size.height}`);
  const x0 = 60, x1 = 610, top = compact ? 84 : 72, h = 16;
  const X = (fraction) => x0 + fraction * (x1 - x0);
  const length = Number($("length").value) || 1;

  // Result band and best estimate (only when damage is likely)
  const r = state.result;
  const showResult = r && r.probability_damaged >= 0.5 && $("detail").hidden === false;
  if (showResult) {
    const [lo, hi] = r.location_interval;
    el("rect", { x: X(lo), y: top - 22, width: Math.max(X(hi) - X(lo), 3), height: h + 44, class: "band", rx: 2 }, svg);
  }

  // Wall and clamp
  el("rect", { x: x0 - 18, y: top - 34, width: 18, height: h + 68, class: "clamp" }, svg);
  for (let y = top - 30; y < top + h + 34; y += 10) {
    el("line", { x1: x0 - 32, y1: y + 10, x2: x0 - 18, y2: y, class: "hatch" }, svg);
  }
  // Beam
  el("rect", { x: x0, y: top, width: x1 - x0, height: h, class: "beam-body" }, svg);

  // Scale along the beam, in metres
  (compact ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).forEach((f) => {
    el("line", { x1: X(f), y1: top + h + 26, x2: X(f), y2: top + h + 32, class: "tick" }, svg);
    const anchor = compact ? (f === 0 ? "start" : f === 1 ? "end" : "middle") : "middle";
    el("text", { x: X(f), y: top + h + 32 + size.tick + 2, "text-anchor": anchor, class: "tick-label",
      style: `font-size:${size.tick}px` }, svg, `${(f * length).toFixed(2)} m`);
  });

  if (showResult) {
    el("line", { x1: X(r.location), y1: top - 22, x2: X(r.location), y2: top + h + 22, class: "detected" }, svg);
    const anchor = r.location > 0.85 ? "end" : r.location < 0.15 ? "start" : "middle";
    el("text", { x: X(r.location), y: top - 30, "text-anchor": anchor, class: "band-label",
      style: `font-size:${size.label}px` }, svg, `Detected: ${pct(r.location)}`);
  }

  // Hidden crack from the simulation
  const sim = state.simulated;
  if (sim && sim.severity > 0) {
    el("line", { x1: X(sim.location), y1: top - 4, x2: X(sim.location), y2: top + h + 4, class: "hidden-crack-line" }, svg);
    el("circle", { cx: X(sim.location), cy: top + h + 14, r: 5, class: "hidden-crack" }, svg);
    const anchor = sim.location > 0.85 ? "end" : sim.location < 0.15 ? "start" : "middle";
    el("text", { x: X(sim.location), y: top + h + 32 + 2 * size.tick + 14, "text-anchor": anchor,
      class: "marker-label", style: `font-size:${size.label}px` }, svg, `Hidden crack: ${pct(sim.location)}`);
  }
}

// ---------- Probability map ----------

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const RAMP_RGB = RAMP.map(hexToRgb);
function rampColor(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP_RGB.length - 1);
  const i = Math.min(Math.floor(x), RAMP_RGB.length - 2);
  const f = x - i;
  return RAMP_RGB[i].map((c, k) => Math.round(c + f * (RAMP_RGB[i + 1][k] - c)));
}

function drawMap() {
  const r = state.result;
  if (!r) return;
  const v = mapView();
  const nLoc = v.i1 - v.i0 + 1, nSev = v.j1 - v.j0 + 1;
  const canvas = $("map");
  canvas.width = nLoc;
  canvas.height = nSev;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(nLoc, nSev);
  for (let i = 0; i < nLoc; i++) {
    for (let j = 0; j < nSev; j++) {
      const [red, green, blue] = rampColor(r.posterior[v.i0 + i][v.j0 + j]);
      const p = ((nSev - 1 - j) * nLoc + i) * 4;          // high severity at the top
      image.data.set([red, green, blue, 255], p);
    }
  }
  ctx.putImageData(image, 0, 0);
  $("zoom").textContent = state.zoomed ? "Show all possible cracks" : "Zoom to the estimate";
  drawMapAxes();
}

function mapBox() {
  const wrap = canvas => canvas.parentElement.getBoundingClientRect();
  const canvas = $("map");
  const outer = wrap(canvas);
  const inner = canvas.getBoundingClientRect();
  return { left: inner.left - outer.left, top: inner.top - outer.top, width: inner.width, height: inner.height };
}

function drawMapAxes() {
  const r = state.result;
  const svg = $("map-axes");
  svg.innerHTML = "";
  const box = mapBox();
  const v = mapView();
  const locMin = r.locations[v.i0], locMax = r.locations[v.i1];
  const sevMin = r.severities[v.j0], sevMax = r.severities[v.j1];
  const X = (loc) => box.left + ((loc - locMin) / (locMax - locMin)) * box.width;
  const Y = (sev) => box.top + (1 - (sev - sevMin) / (sevMax - sevMin)) * box.height;
  const ticks = (lo, hi, step) => {
    const out = [];
    for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(t);
    return out;
  };

  ticks(locMin, locMax, state.zoomed ? 0.05 : 0.2).forEach((loc) => {
    el("text", { x: X(loc), y: box.top + box.height + 16, "text-anchor": "middle" }, svg, `${Math.round(loc * 100)} %`);
  });
  ticks(sevMin, sevMax, state.zoomed ? 0.05 : 0.1).forEach((sev) => {
    el("text", { x: box.left - 8, y: Y(sev) + 4, "text-anchor": "end" }, svg, `${Math.round(sev * 100)} %`);
  });
  el("text", { x: box.left + box.width / 2, y: box.top + box.height + 34, "text-anchor": "middle" }, svg,
    "Crack position (% of length from the clamp)");
  el("text", { x: 10, y: box.top + box.height / 2, "text-anchor": "middle",
    transform: `rotate(-90 10 ${box.top + box.height / 2})` }, svg, "Stiffness lost");

  const sim = state.simulated;
  const inView = (loc, sev) => loc >= locMin && loc <= locMax && sev >= sevMin && sev <= sevMax;
  if (sim && sim.severity > 0 && inView(sim.location, sim.severity)) {
    el("circle", { cx: X(sim.location), cy: Y(sim.severity), r: 8, fill: "none", stroke: "#1c2a36", "stroke-width": 2 }, svg);
  }
  const s = 6, cx = X(r.location), cy = Y(r.severity);
  el("path", { d: `M${cx - s} ${cy - s}L${cx + s} ${cy + s}M${cx - s} ${cy + s}L${cx + s} ${cy - s}`,
    stroke: "#1c2a36", "stroke-width": 2.5 }, svg);
}

function onMapHover(event) {
  const r = state.result;
  if (!r) return;
  const canvas = $("map");
  const rect = canvas.getBoundingClientRect();
  const fx = (event.clientX - rect.left) / rect.width;
  const fy = (event.clientY - rect.top) / rect.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return hideTip();
  const v = mapView();
  const i = v.i0 + Math.round(fx * (v.i1 - v.i0));
  const j = v.j0 + Math.round((1 - fy) * (v.j1 - v.j0));
  const tip = $("map-tip");
  const wrapRect = canvas.parentElement.getBoundingClientRect();
  tip.textContent = `Crack at ${(r.locations[i] * 100).toFixed(1)} %, ${(r.severities[j] * 100).toFixed(1)} % stiffness lost: `
    + `${Math.round(r.posterior[i][j] * 100)} % as likely as the best estimate`;
  tip.style.left = `${event.clientX - wrapRect.left}px`;
  tip.style.top = `${event.clientY - wrapRect.top}px`;
  tip.hidden = false;
}
function hideTip() { $("map-tip").hidden = true; }

// ---------- Start ----------

async function init() {
  try {
    state.options = await api("/api/options");
  } catch (err) {
    showError(`Could not reach the BeamTwin server: ${err.message}`);
    return;
  }
  const { materials, noise_levels: noise } = state.options;
  $("material").innerHTML = Object.entries(materials)
    .map(([key, m]) => `<option value="${key}">${m.label}</option>`).join("");
  $("noise").innerHTML = Object.entries(noise)
    .map(([key, n]) => `<option value="${key}"${key === "medium" ? " selected" : ""}>${n.label}</option>`).join("");

  ["material", "length", "width", "thickness"].forEach((id) => $(id).addEventListener("input", onBeamChanged));
  ["location", "severity"].forEach((id) => $(id).addEventListener("input", updateSliderLabels));
  ["f1", "f2", "f3", "f4"].forEach((id) => $(id).addEventListener("input", onFrequencyEdited));
  $("simulate").addEventListener("click", simulate);
  $("controls").addEventListener("submit", detect);
  $("map").addEventListener("mousemove", onMapHover);
  $("zoom").addEventListener("click", () => { state.zoomed = !state.zoomed; hideTip(); drawMap(); });
  $("map").addEventListener("mouseleave", hideTip);
  window.addEventListener("resize", () => { drawBeam(); if (state.result) drawMapAxes(); });

  updateSliderLabels();
  drawBeam();
  refreshHealthy();
}

init();
