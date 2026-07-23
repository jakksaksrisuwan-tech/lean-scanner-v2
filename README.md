# Lean Scanner v2

Auto-capture document scanner PWA — point the phone at a receipt, it
detects, locks, fires, dewarps, and saves. No shutter button, no server
round-trips, no ML models, no runtime dependencies: the whole pipeline
is hand-rolled JS running in the browser at ~10–20 fps on an iPhone.

A Python mirror of the detector + math primitives lives in
`autocapture/` for offline benchmarks and unit tests.

## Pipeline

```
camera (native <video>, 60fps preview)
    │  one frame per detection cycle, 640px long edge
    ▼
detector v3 (src/detector/v3.js) — jury of hypotheses
    │  generators propose candidate quads (run lazily, cheap first):
    │    whiteness  min(R,G,B) → Otsu (recursive on trimodal scenes)
    │               → blobs scored by boundary sharpness + separation
    │    ink        gradient density → dilate/erode → trimmed hull
    │               (never on saturated bgs — wood grain false-fires)
    │    flood      Gaussian-biased basin flood localizes, detector
    │               re-runs inside the padded ROI (stats become local)
    │    refine     detector re-runs in a crop around the leading quad
    │  corners: max-area hull quad + sides fitted to sharp step edges
    │           (soft light transitions never bend a side)
    │  judge: document ink = dense, paper-adjacent gradients inside the
    │         candidates' consensus region; a candidate must keep ≥95%
    │         of consensus ink; winner = fill × boundary sharpness
    ▼
quality scorer → lock gate (8-D Kalman + N-stable-frames FSM)
    │  drift tolerance & snap thresholds relative to quad size
    ▼
fire → hi-res capture (up to 2560px) → dewarp (homography, 3% pad)
    │  + CLAHE
    ▼
captures → download (Web Share on iOS) / superscan fusion
```

## Superscan

Multi-frame statistical cleanup (`src/dewarp/superscan.js`): collect ~8
aligned captures, then

- **text** ← single sharpest frame, untouched (no ghosting)
- **illumination** ← per-pixel 70th percentile across frames — moving
  shadows and glare are outliers, rejected
- **creases** move *with* the paper, so they survive fusion — removed by
  Sauvola-style local-contrast whitening (ink is high-contrast, crease
  shading is low-contrast)
- optional **B&W** hard binarization (checkbox) for the fax/archive look

Returns `warped` (whitened, for humans) and `ocrSource` (pre-whiten
fusion, for OCR — every cosmetic step measurably costs OCR accuracy:
0.32 single → 0.39 fused → 0.37 whitened → 0.30 binarized on the
2026-07-22 burst).

## Run

```bash
python3 server.py 8081          # serves the PWA + /api/dump endpoints
cloudflared tunnel --url http://localhost:8081   # phone access
```

Open `/` for the app (simulator runs by default — real SRD receipts on
synthetic wood; tap **camera** for live). `/src/camera_dump.html` is the
debug page: view detections live, tap **dump** to save the raw frame +
detector output to `captures-debug/` for offline replay.

## Tests & benchmarks

```bash
npm test              # python 43/43 + real-corpus regression + synth
npm run test:all      # + SRD 200-bill detection bench (floor 82%)
npm run golden:update # re-pin golden quads after a verified improvement
python3 benches/ocr_bench.py --engines vision,tesseract  # end-to-end OCR
```

The regression corpus is every phone frame ever dumped
(`captures-debug/`, 124 frames): each must detect, stay in-frame, and
match its pinned golden quad within 5% corner drift / 0.85 IoU. 35
receipt-free negative crops must NOT fire (≤2 allowed — two contain
genuine white objects). New dumps join the corpus automatically;
`--update` pins them AFTER visual review — pinning blind once endorsed
a regressed quad as truth. Known-hard scenes (white-on-white folds,
bedsheet, hand-held, glossy sofa) stay deliberately un-pinned as flags.

Detection: 124/124 phone corpus, 92.5% SRD, ~15 ms easy scenes /
~165 ms when the full jury convenes (Node). OCR
baselines (SRD dewarped crops, `artifacts/bench/ocr_results.jsonl`):
Apple Vision 8.1 price-tokens/receipt · larngear 7.3 · Tesseract 4.4.

## Layout

```
index.html, sw.js          PWA shell (bump CACHE version on JS changes!)
src/app.js                 UI, phases, capture, superscan collection
src/detector/v3.js         detector: generators + consensus judge
src/detector/primitives.js shared pixel ops (morphology, components, hull)
src/tracker/{kalman,lock}  smoothing + fire gate
src/dewarp/                homography, shading, contrast, superscan
src/simulator.js           real-receipt simulator (assets/sim/)
autocapture/               Python mirror (benches; lacks ink path)
tests/                     corpus regression, synth, python suite
benches/                   SRD detection + OCR benchmarks
captures-debug/            phone-frame corpus (ground truth)
HANDOFF.md                 session-by-session history
```

## Known limitations

- Planar homography only — curled/folded pages keep their geometric
  waviness (crease *shading* is cleaned by superscan; crease *shape*
  would need a mesh dewarp model). Baseline-flatten research is parked
  in research/ with measured trade-offs.
- Four known-hard scenes fail on both current and previous
  architectures: white-on-white with deep folds, patterned bedsheet,
  hand-held in cluttered rooms, part of the glossy-sofa corpus.
- Python detector mirror lacks the ink-density fallback path.
- iOS focus: no tap-to-focus API in WebKit — double-tap zoom (1x/2x/3x)
  works around min-focus-distance blur.
