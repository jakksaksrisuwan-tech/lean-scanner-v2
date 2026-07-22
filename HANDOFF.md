# Lean Scanner v2 — Handoff

## UPDATE 2026-07-20 (later session): v3 detector shipped, problem below is fixed

The v2 luma detector described in this document was replaced by
`src/detector/v3.js` (+ Python mirror `classical.py:_detect_v3`).
Key insight: paper is **desaturated** (R≈G≈B high), wood/table bgs are
**saturated** (blue channel low) — so `min(R,G,B)` separates paper from
bg where luma cannot. Algorithm: min-channel → Otsu → 5×5 open →
biggest 8-connected component → minAreaRect (hull + rotating calipers).
No tuned constants except sanity limits (area 2–90% of frame,
solidity ≥ 0.6). The v2 edge-touch filter was removed — a receipt
running past the frame edge is a valid live-scanner detection.

Results:

| Corpus | v2 | v3 |
|---|---|---|
| 29 phone frames (`captures-debug/`) | 11/29 | **29/29**, IoU 1.000 vs cv2 reference |
| SRD 200 receipts (`../receipts-dataset/`) | ~22% | **94%** (nulls: whole-frame receipts + heavily crumpled) |
| Synth (test_v3_detector.mjs) | pass | pass (IoU ≥ 0.98) |

Tests/benches — run `npm test` (python 43/43 + corpus + synth) or
`npm run test:all` (adds SRD bench, floor 82%):
- `tests/test_real_corpus.mjs` — every saved phone dump must detect,
  with golden-quad snapshots (`tests/fixtures/golden_quads.json`):
  corner drift >5% of long edge AND IoU<0.85 vs golden fails. After a
  verified improvement re-pin with `npm run golden:update`. Also runs
  35 receipt-free negatives (`tests/fixtures/negatives/`, ≤2 fires).
- `benches/srd_bench.mjs` — 200 real bills, detection floor.
- `benches/ocr_bench.py` — detect→dewarp→OCR (vision/tesseract/larngear),
  appends to artifacts/bench/ocr_results.jsonl.
An ink-density fallback (v3.js detectInk) handles light bgs (white
wall / steel table); gated to globally-desaturated scenes. JS-only —
python mirror lacks it. `pipeline.js` + `camera_dump.html` import v3.
Everything below this line describes the v2 era — kept for history.

---

## TL;DR

This is a port of `lean_scanner` (v1, JS PWA) onto the new `autocapture-draft` (python package) detector pipeline. The python math is solid and unit-tested. The JS detector runs on real phone frames **but the algorithm is fundamentally wrong for textured wood-table backgrounds** — the synth benchmark lied.

The single biggest issue: v2's "find biggest contour, fit 4-vertex polygon" strategy grabs a merged blob of receipt + wood-grain + lamp highlights, and `approxPolyDP` picks vertices on that merged outline. Result: most detected quads have at least one corner on the bg, not the receipt.

**A simple edge-touching filter (reject quads touching 2+ frame edges) was just added and removes the worst off-frame false positives**, but does not recover detections where the algorithm picks the wrong blob. The 4 null-detection cases (small receipts on dark wood) still return null.

If you're starting over, the most useful artifacts are:

| File | Status | Use it for |
|---|---|---|
| `autocapture/detector/classical.py:_detect_v2` | ✅ working (synth) | python port of the algorithm |
| `src/detector/v2.js:detectHighContrast` | ⚠️ working, leam-in only | JS port (mirror of `_detect_v2` + edge filter) |
| `src/tracker/{kalman,lock}.js` | ✅ working | 8D Kalman + lock-gate FSM |
| `tests/test_consistency.py` | ✅ 43/43 pass | python unit tests |
| `tests/replay_dumps.mjs` + `replay_new_batch.mjs` | ✅ working | offline replay of saved camera frames through the JS detector |
| `tests/render_dumps.py` (workspace) | ✅ working | render saved `.rgb.raw` files + their saved quads as a grid PNG for visual review |
| `benches/{shared_dataset,detector_bench}.py` | ✅ working | synth eval harness, port of v1's `dataset.js` |
| `captures-debug/dump_20260720_183*_480x640.json` | ✅ ground-truth user frames | real phone frames with detector output + rgb payload |
| `research/v2-detector-on-real-bgs.md` | ✅ done | the research deliverable that produced the edge-filter fix |

## Current state

### What works

- **Python detector on synth dark-wood**: 94% detection, 0.92 IoU at 0-30° rotation. Reproduces via `benches/detector_bench.py`.
- **JS detector on synth** (via `test_v2_detector.mjs`): same algorithm by construction — same numbers, ~5-50ms/frame in Node.
- **Pipeline glue**: pipeline.js, app.js, lock-gate, Kalman — all working. `processFrame(imageData)` and `Pipeline.step()` both call v2.
- **Real phone camera frame capture** via `src/camera_dump.html` — dumps RGB bytes + sidecar JSON to `captures-debug/` via `POST /api/dump` (or `/api/dump-rich` for layered uploads).
- **Cloudflared tunnel** at `https://aids-shake-flight-hockey.trycloudflare.com/` pointing at `localhost:8081` (python `server.py`). Server is the v1's `server.py` + a small `do_POST` handler for the dump endpoints.

### What's broken

#### 1. Wrong-region detection on real wood-bgs (root cause)

The algorithm: binarize (median + 30) → 5×5 open → 9×9 close → `findContours(RETR_EXTERNAL)` → for each contour, multi-ratio `approxPolyDP` ladder → pick first 4-vertex convex quad with valid aspect → pick the LARGEST contour's quad.

**The bug**: on phone frames where the bg is textured wood, the receipt + bg gradients merge into one giant connected blob. `findContours(RETR_EXTERNAL)` returns that as the largest contour. `approxPolyDP` finds a 4-vertex approximation of the merged shape — which has vertices wherever the merged outline happens to have angles.

Empirical evidence (29 saved phone dumps, 480×640):

- 11 detected with quad inside frame and at-or-near the receipt. **All are visually acceptable.** (These are the "good" cases.)
- 6 detected with quad vertices OUTSIDE the frame — including one frame with corners at `(786, -568)` (off-screen by hundreds of pixels). **These were the user-visible "off to the right" failures.**
- 12 frames return null. The receipt is visible in all 12 but the detector doesn't find it.

The 12 null + 6 wild cases are the **same root cause**: binarize threshold + close kernel merges receipt + bg into one blob, and `findContours` returns the wrong one or no 4-vertex fit.

#### 2. The synth benchmark misled us

The synth dark-wood dataset from `../lean_scanner/src/benchmark/dataset.js` doesn't include wood-grain texture. The receipt sits on a smooth luma gradient. So `bg_median + 30` cleanly isolates the receipt from bg, and the algorithm works.

Real phone frames have wood grain (texture + color variation + lamp highlights + table edge gradients). Synth → real generalization fails on this point.

#### 3. Centroid and corners drift when the user moves the camera

Confirmed in code: `renderUI` in `src/app.js` uses `smoothQuad` for drawing. The Kalman filter introduces lag, so when the user moves the phone, the smoothed quad trails the raw detection by ~50-100ms. Visually that's "the centroid is moving the wrong way". Will be more obvious at higher frame rates.

#### 4. Null detection for small receipts on dark wood

Of the 12 null frames in the most recent batch, most have a small receipt (≤25% of frame) sitting on a darker wood patch. The binarize threshold catches too few pixels (because the bg near the receipt is darker than the global median) — the receipt's contours drop below `minQuadAreaRatio` and the candidate is rejected.

#### 5. Python unit tests are the only validation

We don't have a JS-side test that runs on real phone frames. The `tests/replay_dumps.mjs` script was added late and exercises v2.js against saved `.rgb.raw` files but isn't wired into a test harness yet.

## Key code paths

### Detector

- `src/detector/v2.js` — main file. ~290 lines.
  - Stage 1: grayscale + downsample to `long_edge`
  - Stage 2: binarize via `cfg.threshold = bg_median + 30`
  - Stage 3: 5×5 morphological open + 9×9 morphological close
  - Stage 4: 8-connected components → take biggest, get its bbox
  - Stage 5: per-row/col sample of bbox edges (with 8px min-run to skip noise)
  - Stage 6: line-fit via least-squares to the 4 side samples; intersect for 4 corners
  - Stage 7 (NEW): edge-touching filter — reject quads touching 2+ frame edges
  - Returns `{quad, segmentation, cx, cy, confidence}` or null
- `autocapture/detector/classical.py:_detect_v2` — python implementation. Uses cv2.findContours + approxPolyDP ladder (different algorithm than the JS one but produces the same answer on most inputs).
- `src/pipeline.js` — wraps v2 + quality + lock-gate. Calls `processFrame` and `Pipeline.step`.
- `src/tracker/lock.js` — Kalman-smoothed quad stability gate. Fires after 8 stable frames within `cornerDistPx=20`.

### Dump pipeline (real data capture)

- `src/camera_dump.html` — minimal page with a "dump" button. Runs v2 on a live frame, POSTs `{rgb, rawQuad, smoothQuad, centroid, conf, detMs}` to `/api/dump`. Renders the same data into a `<pre>` on the page so the user can sanity-check.
- `server.py:do_POST` — accepts `/api/dump` (legacy single-RGB) or `/api/dump-rich` (multi-layer dict). Writes `.rgb.raw` + `.json` sidecar. Tolerates garbage base64 (tries `validate=False` first, falls back to padding-stuffed decode).
- `tests/replay_dumps.mjs` — replay saved `.rgb.raw` through v2.js, compare against saved `rawQuad`.
- `tests/render_dumps.py` — render a saved dump's image + saved quad + saved smoothed quad as a PNG grid. **Best tool for visual inspection.**

### Bg handling

- v1 bg profiles (in `../lean_scanner/src/benchmark/dataset.js`): `dark-wood`, `light-table`, `white`. Python port at `benches/shared_dataset.py`.
- v2 detector only works on `dark-wood`. On `light-table` / `white`, binarize fails and the detector returns null.

## Difficulties / known landmines

### 1. The binarize threshold

`bg_luma = median(4000 random pixels)` then `thr = bg_luma + 30`. Works on a uniform bg with consistent bg luma. Fails on textured bgs where the "median pixel" is randomly sampled — sometimes includes receipt edge, sometimes not, varying the threshold frame-to-frame.

**Alternative**: Otsu's method auto-picks the threshold, ignoring sample bias. Works on synth. Tested on phone frames: gives a similar mask but the close-kernel still merges receipt+bg.

### 2. The morphological close kernel

5×5 open kills tiny noise. 9×9 close fills holes inside the receipt body. But on phone frames the bg highlights in the bg ALSO have holes that the close fills, leading to a connected blob covering 40-60% of the frame. That merged blob is what `findContours` picks.

**Alternative**: skip close entirely. Or use opening (small open + LARGER open). Or process the bg separately: subtract a heavily-blurred version of the frame from the original to find "local highlight" pixels, mask those out, then binarize.

### 3. The line-fit

`fitLine` is least-squares. On a single tight cluster (e.g. all "top edge" samples within 1px of each other), the line direction is unstable. Slight noise → wildly different slope → intersect goes off-screen.

**Alternative**: RANSAC. Or use the bounding box of the side samples directly (median x of left samples = left line, etc.).

### 4. The binarize-vs-merge trade-off

If you lower the threshold (catch more bg → bigger blob → easier to find the receipt's full silhouette but merged with bg). If you raise it (catch less → cleaner receipt body but maybe missing edges). Can't tune for both.

**Real solution**: text-density-based detection. Find regions of HIGH gradient density (= text) rather than HIGH luma (= paper). The text is unique to the receipt, while bg gradients appear elsewhere too.

### 5. The "12 null" frames are not recoverable with current algorithm

The 12 null frames (small receipts on dark wood) have a fundamentally different signal: the receipt is the brightest thing in a small region, but the global median + 30 threshold catches it just barely. The first fit-luma < threshold check rejects it. No amount of tuning will save these — they need a different algorithm.

## Recommended next steps

### Path A: tune existing algorithm further (cheap)

If you want to keep the line-fit approach, the next cheap wins are:

1. **Adaptive binarization** (Gaussian + local threshold) instead of global. 10 lines of code in JS. Might fix the 12 null frames.
2. **Pre-mask lamp highlights**: a `cv2.inRange(hsv, ...)` for the warmest-brightest pixels → subtract those from the binarize mask. 30 lines.
3. **RANSAC line-fit**: replace `fitLine` with RANSAC. 50 lines.

### Path B: switch to text-density-based detection (medium cost)

Find regions where Sobel-gradient density is high (= text rows packed together). The receipt body has high density (text + receipt edge + many small high-gradient elements). Wood grain has high gradient magnitude but LOW density (sparse). Cable has very high gradient but ZERO density (1-pixel-wide line).

1. Sobel magnitude → threshold top 5%
2. Dilate with `(7×7)` 2x to merge nearby glyphs into one connected region
3. `findContours` on the dilated mask → biggest connected blob
4. Filter by `solidity > 0.85`, `aspect in [0.4, 2.5]`, `bbox_area > 5%fa`
5. `approxPolyDP` for the 4 corners

I've tested this on 3 frames (in `/tmp/v3_candidate.py` etc., kept around as throwaway). Worked on 1 frame, missed the others. Not a guaranteed win but it's a fundamentally different signal.

### Path C: pivot to a different algorithm entirely (high cost)

If neither A nor B gets above 70% real-world accuracy, the line-fit algorithm is the wrong approach. Next options:

- **CNN-based detector** (YOLO / RT-DETR for doc detection). Breaks the no-deps rule.
- **Edge-based**: Canny + Hough lines + intersect. Already implemented as fallback in `detect_opencv`. 8% detection on SRD — much worse than v2.
- **CLAHE + adaptive threshold**: a different preprocess. Untested.
- **Tesseract text-region detection**: find "regions of text" first, then the bbox of those. `tesseract.js` is too heavy.

### Path D: admit defeat on the leam-in and ship a working synth-only demo

The v2 detector is rock-solid on synth dark-wood + 0-30° rotation. We could ship that as a "scanner demo" without promising phone use, and let the user demo the dewarp pipeline that doesn't depend on detection quality.

This is the safest bet if the user wants to ship something working today.

## Open questions for next session

1. **Is the leam-in scope worth keeping?** Original scope: dark-wood bg + 0-30° rotation + bright paper. The 29 saved phone dumps contradict this (real wood is blue-tinted under fluorescent light, the algorithm's leam-in assumption is wrong). Either narrow further (e.g. white paper on pure-black bg only) or accept that the leam-in was overconfident.

2. **What's the user's tolerance for null-detections?** The 12 null frames in the current batch return null even though the receipt is visible. That's a UX problem — user sees nothing happening on the phone. Options: (a) show "DOC NOT FOUND" overlay, (b) keep current behavior (silent null), (c) add a manual fire button (already exists as `force fire`).

3. **Should the JS detector and Python detector be the same algorithm?** Currently they're different (Python uses `findContours`, JS uses row/col scan + line-fit). Both produce similar results on synth but diverge on real frames. **The JS one is what runs in production — the Python one is just for offline benchmarks.** Pick one and keep them in sync.

4. **Where should real-data tests live?** Currently we have ad-hoc `tests/replay_dumps.mjs` and visual `tests/render_dumps.py`. Neither is wired into a test harness. Should we have a `tests/test_real_corpus.mjs` that loads the 29 saved frames and asserts `quad_inside_frame` on each?

5. **How to handle the camera's color space?** Phone frames come in as RGBA. JS saves as RGB (drops alpha). My analysis scripts treat as BGR (because cv2). This means `node tests/replay_dumps.mjs` and Python's `_detect_v2` see different byte orders of the same frame. **Should standardize on RGB everywhere**, then convert at the very edge.

## File locations

- Workspace root: `/Users/jsaksrisuwan/workspace/lean_scanner_v2/`
- v1 reference: `/Users/jsaksrisuwan/workspace/lean_scanner/`
- Saved phone frames (ground truth): `/Users/jsaksrisuwan/workspace/lean_scanner_v2/captures-debug/`
- Python synth data: `/Users/jsaksrisuwan/workspace/lean_scanner_v2/benches/`
- SRD receipt images: `/Users/jsaksrisuwan/workspace/receipts-dataset/`
- Research deliverable: `/Users/jsaksrisuwan/workspace/lean_scanner_v2/research/v2-detector-on-real-bgs.md`
- Learnings log: `/Users/jsaksrisuwan/.learnings/LEARNINGS.md`
- Tunnel URL: `https://aids-shake-flight-hockey.trycloudflare.com/` (live now, but the cloudflared process needs restarting after every system restart — see v1 HANDOFF.md for the incantation)

## Verification status (last fresh run, 2026-07-20)

- python tests: **43/43 pass**
- python synth bench: 93.3% detection, IoU 0.92 (small N=30 sample, full bench at count=50 in `artifacts/bench/RESULTS.md`)
- JS replay on 29 saved phone frames: **11 detected-and-correct**, 6 detected-and-wild-filtered, 12 null (unchanged from baseline)
- Tunnel endpoints: GET / and POST /api/dump-rich both 200

## Closing note

The most useful thing the next person can do is NOT write more detection code. Instead:

1. Run `tests/render_dumps.py` on the saved phone frames and visually verify what's happening.
2. Look at the masks in `/tmp/v2_*_mask.png` — those show exactly what the algorithm sees as "foreground".
3. Spend 30 minutes with the SRD receipt images (`/Users/jsaksrisuwan/workspace/receipts-dataset/`) and a few different thresholds. Understand why 200 SRD frames get only 22% detection rate.
4. If you decide to switch algorithms, throw away the JS-side line-fit code entirely and start over. Don't try to patch it.
