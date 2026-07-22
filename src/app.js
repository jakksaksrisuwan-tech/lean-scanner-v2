// App entry point — auto-capture document scanner with a 4-phase UI.
//
// Phases of the live UI:
//   SEARCHING  nothing rendered (detector looking for a quad)
//   IDENTIFYING translucent blue segmentation grid + blue outline + circle
//   LOCKED     segmentation fades; solid blue circle at the centroid
//   FIRING     fire animation: circle splits into 4 wedges fanning out to
//              detected corners over 600 ms. Dewarp + CLAHE run during.
//   → SEARCHING (after fire cooldown resets lock-gate)
//
// Two sources feed the loop:
//   - CameraSource  (getUserMedia)
//   - SimulatorSource (offline synthetic doc)
//
// Use the "simulator" button to switch.

import { Pipeline } from "./pipeline.js";
import { loadConfig } from "./config.js";
import { SyntheticDocumentSource } from "./simulator.js";

const cfg = loadConfig();

// ── DOM ──────────────────────────────────────────────────────────────────
const video      = document.getElementById("cam");
const overlay    = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const stateEl    = document.getElementById("state");
const fpsEl      = document.getElementById("fps");
const qualityEl  = document.getElementById("quality");
const countEl    = document.getElementById("count");
const capturesEl = document.getElementById("captures");
const btnSim     = document.getElementById("btn-sim");
const btnCamera  = document.getElementById("btn-camera");
const btnFire    = document.getElementById("btn-fire");
const btnDump    = document.getElementById("btn-dump");
const btnSuper   = document.getElementById("btn-super");
const btnDownload = document.getElementById("btn-download");

// ── Service worker ───────────────────────────────────────────────────────
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js", { scope: "./" })
    .catch((e) => console.warn("sw registration failed:", e));
}

// ── Pipeline + captures ────────────────────────────────────────────────
const pipeline = new Pipeline(cfg);
const captures = [];   // {warped, w, h}

// ── Phase state ─────────────────────────────────────────────────────────
const PHASES = {
  SEARCHING:  { label: "searching",  cls: "searching" },
  IDENTIFYING:{ label: "identifying",cls: "identifying" },
  LOCKED:     { label: "locked",     cls: "locked" },
  FIRING:     { label: "firing",     cls: "firing" },
};

// All mutable state lives in `s` so the loop can be linear, readable.
const s = {
  phase: PHASES.SEARCHING,
  prevFrame: null,            // previous result, used to compute the phase
                              // (so we can detect rising / falling edges)
  smoothedQuad: null,        // last detected quad (lock-gate smoothed)
  rawQuad: null,              // raw detection (no smoothing)
  segmentation: null,        // 24x24 paper-likelihood mask
  centroid: null,             // {cx, cy}
  lastImageData: null,        // for force-fire
  lastSize: { w: 0, h: 0 },
  // Lock-gate state machine mirror
  lastGateState: null,
  // Phase transitions: only transition once per rising edge
  fireStartedAt: 0,
  // Loop control
  stopRequested: false,
  activeSource: null,
  // FPS
  lastFpsT: performance.now(),
  frameCount: 0,
};

// ── Phase advance ────────────────────────────────────────────────────────

function advancePhase(result) {
  const gate = result.gateState;
  if (result.capture) {
    // Lock-gate fired this frame — start (or continue) firing
    return PHASES.FIRING;
  }
  if (!result.rawQuad) {
    // No detection. Lock-gate is SEARCHING/UNSTABLE — back to searching.
    return PHASES.SEARCHING;
  }
  // Detection succeeded. Map gate state to UI phase:
  if (gate === "LOCKED") {
    // The next capture should fire. We won't transition to FIRE here —
    // we let the next step() with capture=null do that to avoid flicker.
    return PHASES.LOCKED;
  }
  if (gate === "ALMOST" || gate === "COOLDOWN") {
    return PHASES.LOCKED;
  }
  return PHASES.IDENTIFYING;
}

// ── Drawing primitives ───────────────────────────────────────────────────

function drawSegmentationMask(W, H, seg24) {
  if (!seg24) return;
  const G = 24;
  const cellW = W / G, cellH = H / G;
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const v = seg24[cy * G + cx];
      if (v < 0.1) continue;
      overlayCtx.fillStyle = `rgba(74, 161, 255, ${(v * 0.30).toFixed(3)})`;
      overlayCtx.fillRect(cx * cellW, cy * cellH, cellW + 1, cellH + 1);
    }
  }
}

function drawSolidCircle(cx, cy, r, fillColor, ringColor = "rgba(255,255,255,0.65)") {
  overlayCtx.fillStyle = fillColor;
  overlayCtx.beginPath();
  overlayCtx.arc(cx, cy, r, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.strokeStyle = ringColor;
  overlayCtx.lineWidth = 2;
  overlayCtx.stroke();
}

// Mean of 4 corners (8-quad array) — used for the UI centroid when
// the raw v2 detector's centroid jumps around frame-to-frame.
function quadCentroid(quad) {
  if (!quad) return null;
  let sx = 0, sy = 0;
  for (let i = 0; i < 4; i++) { sx += quad[i * 2]; sy += quad[i * 2 + 1]; }
  return { x: sx / 4, y: sy / 4 };
}

function drawQuadOutline(quad, color, lineWidth = 3, withDots = true) {
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = lineWidth;
  overlayCtx.beginPath();
  overlayCtx.moveTo(quad[0], quad[1]);
  overlayCtx.lineTo(quad[2], quad[3]);
  overlayCtx.lineTo(quad[4], quad[5]);
  overlayCtx.lineTo(quad[6], quad[7]);
  overlayCtx.closePath();
  overlayCtx.stroke();
  if (withDots) {
    overlayCtx.fillStyle = color;
    for (let i = 0; i < 4; i++) {
      overlayCtx.beginPath();
      overlayCtx.arc(quad[i * 2], quad[i * 2 + 1], 4, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  }
}

function drawWedges(centroid, quad, progress) {
  // progress: 0..1, tips at centroid → quad corners
  // Easing: cubic ease-in-out
  const t = progress;
  const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  const corners = [
    { x: quad[0], y: quad[1] },
    { x: quad[2], y: quad[3] },
    { x: quad[4], y: quad[5] },
    { x: quad[6], y: quad[7] },
  ];
  overlayCtx.fillStyle = "rgba(74, 161, 255, 0.85)";
  for (let i = 0; i < 4; i++) {
    const c = corners[i];
    const next = corners[(i + 1) % 4];
    const tipX = centroid.cx + (c.x - centroid.cx) * e;
    const tipY = centroid.cy + (c.y - centroid.cy) * e;
    const dx = c.x - centroid.cx, dy = c.y - centroid.cy;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const px = -dy / len, py = dx / len;
    const baseHalf = Math.max(10, len * 0.10 * (1 - e * 0.7));
    const nx = centroid.cx + (next.x - centroid.cx) * 0.25;
    const ny = centroid.cy + (next.y - centroid.cy) * 0.25;
    overlayCtx.beginPath();
    overlayCtx.moveTo(centroid.cx, centroid.cy);
    overlayCtx.lineTo(tipX + px * baseHalf, tipY + py * baseHalf);
    overlayCtx.lineTo(tipX - px * baseHalf, tipY - py * baseHalf);
    overlayCtx.lineTo(nx, ny);
    overlayCtx.closePath();
    overlayCtx.fill();
  }
}

// ── Render current phase ────────────────────────────────────────────────

function renderUI(result, frame) {
  const { w, h } = s.lastSize;
  // Camera mode: the native <video> element IS the live preview (60fps,
  // free) — the canvas only carries overlays, so clear to transparent.
  // Simulator mode: no video element, blit the synthetic frame.
  if (s.activeSource === "camera") {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  } else {
    overlayCtx.putImageData(frame.imageData, 0, 0);
  }

  if (s.phase === PHASES.IDENTIFYING) {
    drawSegmentationMask(w, h, s.segmentation);
    // Debug overlay: draw BOTH the smoothed Kalman quad (low alpha,
    // for "what the lock-gate sees") AND the raw detector quad (high
    // alpha, for "what the algorithm actually sees this frame").
    if (s.smoothedQuad) drawQuadOutline(s.smoothedQuad, "rgba(255, 200, 0, 0.4)", 2);
    if (s.rawQuad) {
      drawQuadOutline(s.rawQuad, "rgba(74, 161, 255, 0.85)", 3);
      const rc = quadCentroid(s.rawQuad);
      if (rc) drawSolidCircle(rc.x, rc.y, 30, "rgba(74,161,255,0.6)");
    }
  } else if (s.phase === PHASES.LOCKED) {
    const uiQuad = s.smoothedQuad || s.rawQuad;
    const uiCentroid = uiQuad ? quadCentroid(uiQuad) : s.centroid;
    if (uiCentroid) drawSolidCircle(uiCentroid.x, uiCentroid.y, 36, "#4ea1ff");
    if (uiQuad) drawQuadOutline(uiQuad, "rgba(74, 161, 255, 0.85)", 4);
  } else if (s.phase === PHASES.FIRING) {
    const elapsed = performance.now() - s.fireStartedAt;
    const t = Math.min(1, elapsed / 600);
    const uiQuad = s.smoothedQuad || s.rawQuad;
    const uiCentroid = uiQuad ? quadCentroid(uiQuad) : s.centroid;
    if (uiCentroid && uiQuad) drawWedges(uiCentroid, uiQuad, t);
  }
  // SEARCHING: nothing to render on top of the base frame

  stateEl.textContent = s.phase.label;
  stateEl.className = "state " + s.phase.cls;
  qualityEl.textContent = result.quality.toFixed(2);
  countEl.textContent = String(captures.length);
}

// ── Superscan ───────────────────────────────────────────────────────────
// Collect several full-res frames of the same doc, then fuse: text from
// the sharpest frame, illumination statistically cleaned (shadows and
// wrinkle shading move between frames; the paper doesn't).
const SUPER_N = 8, SUPER_MS = 4000;

function grabFullRGBQuad(quadSmall, smallW) {
  let img, vw, vh;
  if (s.activeSource === "camera" && video.readyState >= 2 && video.videoWidth > 0) {
    vw = video.videoWidth; vh = video.videoHeight;
    const c = document.createElement("canvas");
    c.width = vw; c.height = vh;
    c.getContext("2d").drawImage(video, 0, 0);
    img = c.getContext("2d").getImageData(0, 0, vw, vh);
  } else if (s.lastImageData) {
    img = s.lastImageData; vw = img.width; vh = img.height;
  } else return null;
  const f = vw / smallW;
  let quad = quadSmall.map((v) => v * f);
  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += quad[i * 2]; cy += quad[i * 2 + 1]; }
  cx /= 4; cy /= 4;
  quad = quad.map((v, i) => (i % 2 === 0 ? cx + (v - cx) * 1.03 : cy + (v - cy) * 1.03));
  const rgb = new Uint8ClampedArray(vw * vh * 3);
  for (let i = 0; i < vw * vh; i++) {
    rgb[i * 3] = img.data[i * 4];
    rgb[i * 3 + 1] = img.data[i * 4 + 1];
    rgb[i * 3 + 2] = img.data[i * 4 + 2];
  }
  return { rgb, w: vw, h: vh, quad };
}

async function finishSuperscan() {
  const frames = s.superscan.frames;
  s.superscan = null;
  if (!frames.length) { qualityEl.textContent = "superscan: no frames"; return; }
  qualityEl.textContent = `superscan: fusing ${frames.length}...`;
  await new Promise((r) => setTimeout(r, 30)); // let the label paint
  const { fuseScans } = await import("./dewarp/superscan.js");
  const cap = fuseScans(frames, { bw: document.getElementById("chk-bw").checked });
  if (!cap) { qualityEl.textContent = "superscan failed"; return; }
  captures.push(cap);
  addThumbnail(cap);
  countEl.textContent = String(captures.length);
  btnDownload.disabled = false;
  qualityEl.textContent = `superscan ✓ (${cap.frames} frames)`;
}

btnSuper.addEventListener("click", () => {
  if (s.superscan) return;
  s.superscan = { frames: [], until: performance.now() + SUPER_MS };
  qualityEl.textContent = "superscan 0/" + SUPER_N;
});

// Render loop: overlays repaint at display rate regardless of how slow
// detection is. Detection (~3fps on old phones) only updates s.lastResult;
// the live camera preview is the <video> element itself and never lags.
(function renderLoop() {
  if (s.lastResult && s.lastFrame) renderUI(s.lastResult, s.lastFrame);
  requestAnimationFrame(renderLoop);
})();

// ── Loop ────────────────────────────────────────────────────────────────

async function loop(getFrame) {
  while (!s.stopRequested) {
    try {
    const frame = await getFrame();
    if (!frame) break;

    s.lastImageData = frame.imageData;
    s.lastSize = { w: frame.width, h: frame.height };
    // Only resize the overlay canvas when its size actually changes.
    // Resizing a canvas wipes its pixel buffer AND resets context state.
    if (overlay.width !== frame.width || overlay.height !== frame.height) {
      overlay.width = frame.width;
      overlay.height = frame.height;
    }

    const result = pipeline.step(frame.imageData);
    s.smoothedQuad = result.smoothedQuad;
    s.rawQuad = result.rawQuad;
    s.segmentation = result.segmentation;
    s.centroid = result.centroid;
    const gateState = result.gateState;
    const q = result.quality;

    const nextPhase = advancePhase(result);
    if (nextPhase === PHASES.FIRING && s.phase !== PHASES.FIRING) {
      s.fireStartedAt = performance.now();
    }
    s.phase = nextPhase;
    s.lastGateState = result.gateState;

    // Capture fired — cut the page from the full-res camera frame when
    // possible; the small pipeline capture is the simulator fallback.
    if (result.capture) {
      const fireQuad = result.rawQuad || s.rawQuad;
      let cap = null;
      if (fireQuad) cap = await hiResCapture(fireQuad, frame.width).catch(() => null);
      if (!cap) cap = { warped: result.capture.warped, w: result.capture.w, h: result.capture.h };
      captures.push(cap);
      addThumbnail(cap);
      btnDownload.disabled = false;
      // ~700px page width is the floor for reliable OCR on receipt text.
      if (cap.w < 700) qualityEl.textContent = `low res ${cap.w}px — move closer`;
    }

    s.lastResult = result;
    s.lastFrame = frame;

    // Superscan collection: one full-res grab per detection while active
    if (s.superscan) {
      if (result.rawQuad) {
        const g = grabFullRGBQuad(result.rawQuad, frame.width);
        if (g) s.superscan.frames.push(g);
        qualityEl.textContent = `superscan ${s.superscan.frames.length}/${SUPER_N}`;
      }
      if (s.superscan.frames.length >= SUPER_N || performance.now() > s.superscan.until) {
        finishSuperscan();
      }
    }

    // FPS counter (1s window)
    s.frameCount++;
    const now = performance.now();
    if (now - s.lastFpsT >= 1000) {
      fpsEl.textContent = String(Math.round((s.frameCount * 1000) / (now - s.lastFpsT)));
      s.frameCount = 0;
      s.lastFpsT = now;
    }

    // Yield to the browser between frames
    await new Promise((r) => setTimeout(r, 0));
    } catch (e) {
      // Don't flip the state pill to "ERROR" — that flashes the UX
      // off. Log to console + window for out-of-band inspection, then
      // keep looping. The user can see state and queue progress
      // regardless of pipeline errors.
      console.error("loop error:", e && e.message, e && e.stack);
      window.__loopErr = (window.__loopErr || []);
      window.__loopErr.push({ t: performance.now(), msg: e && e.message, stk: e && e.stack });
      if (window.__loopErr.length > 5) window.__loopErr.shift();
      await new Promise((r) => setTimeout(r, 100));
    }
  }
}

// ── Thumbnail strip ──────────────────────────────────────────────────────

function addThumbnail(capture) {
  const img = new Image();
  img.src = warpedToDataURL(capture);
  img.addEventListener("click", () => openViewer(captures.indexOf(capture)));
  capturesEl.appendChild(img);
}

// ── Capture viewer (tap thumbnail: preview / navigate / save / delete) ──
const viewer = document.getElementById("viewer");
const viewerImg = document.getElementById("viewer-img");
const viewerCounter = document.getElementById("viewer-counter");
let viewerIdx = -1;

function openViewer(idx) {
  if (idx < 0 || idx >= captures.length) return;
  viewerIdx = idx;
  viewerImg.src = warpedToDataURL(captures[idx]);
  viewerCounter.textContent = `${idx + 1} / ${captures.length}`;
  document.getElementById("viewer-prev").disabled = idx === 0;
  document.getElementById("viewer-next").disabled = idx === captures.length - 1;
  viewer.hidden = false;
}
function closeViewer() { viewer.hidden = true; viewerIdx = -1; }

function captureToFile(c, i) {
  const b64 = warpedToDataURL(c).split(",")[1];
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let j = 0; j < bin.length; j++) u8[j] = bin.charCodeAt(j);
  return new File([u8], `page_${i + 1}.png`, { type: "image/png" });
}

document.getElementById("viewer-close").addEventListener("click", closeViewer);
viewer.addEventListener("click", (e) => { if (e.target === viewer) closeViewer(); });
document.getElementById("viewer-prev").addEventListener("click", () => openViewer(viewerIdx - 1));
document.getElementById("viewer-next").addEventListener("click", () => openViewer(viewerIdx + 1));
document.addEventListener("keydown", (e) => {
  if (viewer.hidden) return;
  if (e.key === "ArrowLeft") openViewer(viewerIdx - 1);
  else if (e.key === "ArrowRight") openViewer(viewerIdx + 1);
  else if (e.key === "Escape") closeViewer();
});
document.getElementById("viewer-save").addEventListener("click", async () => {
  if (viewerIdx < 0) return;
  const file = captureToFile(captures[viewerIdx], viewerIdx);
  const files = [file];
  if (navigator.canShare && navigator.canShare({ files })) {
    try { await navigator.share({ files }); return; } catch (e) { /* cancelled */ }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(file);
  a.download = file.name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
});
document.getElementById("viewer-del").addEventListener("click", () => {
  if (viewerIdx < 0) return;
  captures.splice(viewerIdx, 1);
  capturesEl.children[viewerIdx]?.remove();
  countEl.textContent = String(captures.length);
  btnDownload.disabled = captures.length === 0;
  if (captures.length === 0) closeViewer();
  else openViewer(Math.min(viewerIdx, captures.length - 1));
});

function warpedToDataURL({ warped, w, h }) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    img.data[i * 4]     = warped[i * 3];
    img.data[i * 4 + 1] = warped[i * 3 + 1];
    img.data[i * 4 + 2] = warped[i * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

// ── Sources ──────────────────────────────────────────────────────────────

// Camera: getUserMedia. If unavailable (e.g. headless chrome), falls
// through to simulator.
async function startCamera() {
  s.stopRequested = true;
  await new Promise((r) => setTimeout(r, 50));  // let old loop exit
  s.stopRequested = false;
  document.body.classList.remove("sim-mode");
  s.activeSource = "camera";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      // Ask for the highest resolution the camera will give: detection
      // runs on a 640px downscale, but the dewarped capture is cut from
      // the full-res frame — that's what OCR quality lives on.
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 2560 },
        height: { ideal: 2560 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    // Focus: iOS Safari exposes no focusMode/focusDistance, but zoom IS
    // exposed (iOS 17+). Blur usually means the phone is inside the
    // lens's min focus distance — standing back at 2x zoom fixes focus
    // without losing resolution. Double-tap the view to cycle 1x/2x/3x.
    const track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    try {
      if (caps.focusMode && caps.focusMode.includes("continuous")) {
        await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
      }
    } catch (e) { /* unsupported — fine */ }
    if (caps.zoom) {
      let zoomIdx = 0;
      const zooms = [1, 2, 3].filter((z) => z >= (caps.zoom.min || 1) && z <= (caps.zoom.max || 1));
      overlay.addEventListener("dblclick", async () => {
        if (!zooms.length) return;
        zoomIdx = (zoomIdx + 1) % zooms.length;
        try { await track.applyConstraints({ advanced: [{ zoom: zooms[zoomIdx] }] }); } catch (e) {}
        stateEl.textContent = `zoom ${zooms[zoomIdx]}x`;
      });
    }
  } catch (err) {
    console.warn("camera denied, falling back to simulator:", err.message);
    return startSimulator();
  }

  loop(cameraFrameSource);
}

async function cameraFrameSource() {
  if (video.readyState < 2) return null;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (vw === 0 || vh === 0) return null;
  // Detection frame: 640px long edge. The full-res frame is only grabbed
  // at capture time (grabFullFrame) — copying 2560px RGBA every tick
  // would dominate the frame budget.
  const scale = Math.min(1, 640 / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(video, 0, 0, w, h);
  return {
    imageData: c.getContext("2d").getImageData(0, 0, w, h),
    width: w, height: h,
  };
}

// Grab the camera at native resolution and dewarp `quadSmall` (coords in
// the small detection frame) from it. Returns a capture {warped,w,h} or
// null when the camera isn't the active source.
async function hiResCapture(quadSmall, smallW) {
  if (s.activeSource !== "camera" || video.readyState < 2) return null;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (vw === 0 || smallW >= vw) return null;
  const f = vw / smallW;
  let quad = quadSmall.map((v) => v * f);
  // Pad 3% outward from the centroid: a tight quad clips half-characters
  // at the paper edge (verified on SRD: padded beats tight 95-43 on OCR).
  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += quad[i * 2]; cy += quad[i * 2 + 1]; }
  cx /= 4; cy /= 4;
  quad = quad.map((v, i) => (i % 2 === 0 ? cx + (v - cx) * 1.03 : cy + (v - cy) * 1.03));
  const c = document.createElement("canvas");
  c.width = vw; c.height = vh;
  c.getContext("2d").drawImage(video, 0, 0);
  const img = c.getContext("2d").getImageData(0, 0, vw, vh);
  const { warpQuad } = await import("./dewarp/homography.js");
  const { claheSimple } = await import("./dewarp/contrast.js");
  const rgb = new Uint8ClampedArray(vh * vw * 3);
  for (let i = 0; i < vh * vw; i++) {
    rgb[i * 3] = img.data[i * 4];
    rgb[i * 3 + 1] = img.data[i * 4 + 1];
    rgb[i * 3 + 2] = img.data[i * 4 + 2];
  }
  const warped = warpQuad(rgb, vh, vw, 3, quad, cfg.dewarp.fixedAspect);
  const n = warped.height * warped.width;
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    gray[i] = Math.round((warped.data[i * 3] + warped.data[i * 3 + 1] + warped.data[i * 3 + 2]) / 3);
  }
  const eq = claheSimple(gray, warped.height, warped.width, cfg.dewarp.claheClip, cfg.dewarp.claheGrid);
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = eq[i]; out[i * 3 + 1] = eq[i]; out[i * 3 + 2] = eq[i];
  }
  return { warped: out, w: warped.width, h: warped.height };
}

// Simulator: synthetic document source. No camera permission required.
let simulator = null;
async function startSimulator() {
  s.stopRequested = true;
  await new Promise((r) => setTimeout(r, 50));
  s.stopRequested = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  document.body.classList.add("sim-mode");
  s.activeSource = "simulator";

  // Fresh simulator + a fresh pipeline so the lock-gate isn't pre-locked.
  try {
    simulator = new SyntheticDocumentSource({ width: 640, height: 480, seed: 42 });
  } catch (e) {
    stateEl.textContent = "ERR";
    stateEl.className = "state unstable";
    document.title = "ERR " + (e && e.message);
    console.error("SyntheticDocumentSource ctor error:", e && e.message, "\n", e && e.stack);
    return;
  }
  pipeline.frameIdx = 0;
  pipeline.captures = [];
  pipeline.gate.kf.reset();
  pipeline.gate._frameIdx = 0;
  pipeline.gate._prevSmoothed = null;
  pipeline.gate.state = "SEARCHING";

  loop(simulatorFrameSource);
}

async function simulatorFrameSource() {
  try {
    // Static mode: read(60) puts us at the simulated "hold" frame and
    // the doc stays put. Keep returning the same frame so the lock-gate
    // can satisfy its 8-stable-frame debounce.
    const { ok, imageData } = simulator.read(60);
    simulator.frameIdx = 60;  // pin so the natural advance inside read doesn't drift
    if (!ok || !imageData) return null;
    return {
      imageData,
      width: imageData.width,
      height: imageData.height,
    };
  } catch (e) {
    // Surface on screen (visible from outside) AND on console.
    stateEl.textContent = "ERR";
    stateEl.className = "state unstable";
    document.title = "ERR " + (e && e.message);
    console.error("simulatorFrameSource error:", e && e.message, "\n", e && e.stack);
    return null;
  }
}

// ── Buttons ──────────────────────────────────────────────────────────────

btnSim.addEventListener("click", () => startSimulator());

btnCamera.addEventListener("click", () => startCamera());
// Enable the camera button only on mobile (heuristic: touch capability
// + narrow viewport). Desktop goes through a prompt dialog flow.
if (typeof window !== "undefined" && (
    "ontouchstart" in window ||
    navigator.maxTouchPoints > 0 ||
    matchMedia("(pointer: coarse)").matches
)) {
  btnCamera.disabled = false;
}

btnDump.addEventListener("click", () => {
  if (!s.lastImageData) return;
  const W = s.lastSize.w, H = s.lastSize.h;
  const id = s.lastImageData;
  const rgb = new Uint8ClampedArray(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    rgb[i*3]   = id.data[i*4];
    rgb[i*3+1] = id.data[i*4+1];
    rgb[i*3+2] = id.data[i*4+2];
  }
  let bin = "";
  for (let i = 0; i < rgb.length; i++) bin += String.fromCharCode(rgb[i]);
  const b64 = btoa(bin);
  const prevLabel = qualityEl.textContent;
  btnDump.disabled = true;
  qualityEl.textContent = "uploading...";
  window.__dumpIdx = (window.__dumpIdx || 0) + 1;
  fetch("/api/dump", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      w: W, h: H, b64,
      rawQuad: s.rawQuad || null,
      smoothQuad: s.smoothedQuad || null,
      centroid: s.centroid || null,
      // confidence is recomputed per the pipeline step result; qualityEl
      // displays the latest quality number, which is what we want here.
      conf: parseFloat(qualityEl.textContent) || null,
      detMs: -1,
      dumpIdx: window.__dumpIdx,
    }),
  })
  .then(r => r.json())
  .then(_ => {
    qualityEl.textContent = "dumped ✓";
    setTimeout(() => { qualityEl.textContent = prevLabel; }, 800);
  })
  .catch(e => { qualityEl.textContent = "err " + e.message; })
  .finally(() => { btnDump.disabled = false; });
});

btnFire.addEventListener("click", async () => {
  if (!s.rawQuad || !s.lastImageData) return;
  // Prefer a full-res cut; fall back to the small detection frame.
  const hi = await hiResCapture(s.rawQuad, s.lastSize.w).catch(() => null);
  if (hi) {
    captures.push(hi);
    addThumbnail(hi);
    countEl.textContent = String(captures.length);
    btnDownload.disabled = false;
    return;
  }
  // Synthesize a capture on the current frame (no lock-gate needed)
  import("./dewarp/homography.js").then(({ warpQuad }) =>
    import("./dewarp/contrast.js").then(({ claheSimple }) => {
      const { lastImageData, lastSize } = s;
      const { width: W, height: H } = lastSize;
      const rgb = new Uint8ClampedArray(H * W * 3);
      for (let i = 0; i < H * W; i++) {
        rgb[i * 3]     = lastImageData.data[i * 4];
        rgb[i * 3 + 1] = lastImageData.data[i * 4 + 1];
        rgb[i * 3 + 2] = lastImageData.data[i * 4 + 2];
      }
      const warped = warpQuad(rgb, H, W, 3, s.rawQuad, cfg.dewarp.fixedAspect);
      const gray = new Uint8Array(warped.height * warped.width);
      for (let i = 0; i < warped.height * warped.width; i++) {
        gray[i] = Math.round((warped.data[i * 3] + warped.data[i * 3 + 1] + warped.data[i * 3 + 2]) / 3);
      }
      const eq = claheSimple(gray, warped.height, warped.width, cfg.dewarp.claheClip, cfg.dewarp.claheGrid);
      const out = new Uint8ClampedArray(warped.height * warped.width * 3);
      for (let i = 0; i < warped.height * warped.width; i++) {
        out[i * 3] = eq[i]; out[i * 3 + 1] = eq[i]; out[i * 3 + 2] = eq[i];
      }
      const cap = { warped: out, w: warped.width, h: warped.height };
      captures.push(cap);
      addThumbnail(cap);
      countEl.textContent = String(captures.length);
      btnDownload.disabled = false;
    })
  );
});

btnDownload.addEventListener("click", async () => {
  // iOS Safari ignores the download attribute on data: URLs and blocks
  // rapid multi-clicks — Web Share (saves to Photos/Files) is the path
  // that actually works on iPhone. Anchor download is the desktop path.
  const files = captures.map((c, i) => {
    const b64 = warpedToDataURL(c).split(",")[1];
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) u8[j] = bin.charCodeAt(j);
    return new File([u8], `page_${i + 1}.png`, { type: "image/png" });
  });
  if (navigator.canShare && navigator.canShare({ files })) {
    try { await navigator.share({ files }); return; } catch (e) { /* cancelled → fall through */ }
  }
  for (const f of files) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(f);
    a.download = f.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    await new Promise((r) => setTimeout(r, 150));
  }
});

// ── Boot ─────────────────────────────────────────────────────────────────

// Default to simulator on first load. Camera mode is opt-in via the
// `force fire` button.
//
// Why simulator-first: getUserMedia in headless browsers, embedded
// webviews, or any context the user doesn't trust, will hang on a
// permission prompt or reject silently. Simulator mode needs no
// permission, runs offline, and is the safe default.
startSimulator();
