# Flood-fill detector experiment (2026-07-23)

Idea (user's): treat documentness as a 3D landscape biased by a centered
Gaussian; the doc is a dip; flood it and read the boundary off the
waterline. Implemented: evidence = max(whiteness, ink-density) x Gaussian
prior, priority-queue flood from argmax, waterline chosen by max
rectangularity of the puddle (border-touching floods excluded).

## Results

Hard-scene localization (6-scene panel incl. both known-hard battery
frames): 6/6 visually correct — the ONLY approach so far that handles
battery-on-tile with off-center subject. One unified path, no scene gate.

Full corpus (77 golden frames): mean IoU vs golden 0.78, 53 frames below
the 0.85 bar. Corner precision is the killer: flood works at 160px and
approxPolyDP on the puddle is +-4-8px at full scale vs the production
path's sub-pixel side refinement.

Negatives: 35/35 FIRE. Flood has no "no document" concept — argmax
evidence always exists, rectangularity always picks some waterline.
Production's paperness gates are what keep an empty table quiet.

## Verdict

Not a replacement. Two viable roles:
1. RESCUE path: when production paths null or fail sanity, flood the
   scene — but needs an evidence-contrast gate (inside vs outside quad)
   before it can be trusted not to fire on empty tables.
2. HYBRID (most promising): flood as coarse localizer -> constrain the
   existing hull + least-squares refinement to the flood ROI at 320px.
   Combines flood's scene-robustness with production corner precision.

Prototype: research/flood_prototype.py (offline, cv2).

## Outcome (shipped as v5.0, 2026-07-23)

The hybrid landed as flood-RESCUE with confidence arbitration:
1. classic full-frame runs first; confident results (>=0.8) return verbatim
2. weak/null classic -> floodLocalize (bucket-queue, 160px, rectangularity
   waterline) -> padded ROI crop -> classic detector re-run INSIDE the crop
   in strict mode (sharpness floor for all blob sizes) -> higher confidence
   wins
3. rescue acceptance also requires the quad interior to the crop (the 25%
   ROI pad guarantees a true doc fits; clipped slivers = zipper bands etc.)

Rejected during calibration: ink-fraction gate (zipper teeth measure 0.32,
denser than receipts); depth-of-basin gate (positives/negatives overlap).
What worked: strict sharpness (kills gloss highlights) + interior-quad.

Final: 81 corpus frames green incl. both former known-hards, negatives
2/35, SRD 93.0%, ~15ms typical / ~50ms when rescue engages.

## Era ablation (2026-07-23, benches/ablation.mjs)

Golden-independent metrics: cover = fraction of the frame's ink inside
the quad (content completeness); fill = ink density inside the quad
(tightness). Both computed identically per frame, so era-vs-era
comparison is fair even where goldens were pinned by one era.

| scene         | cover base/cur | fill base/cur | verdict |
|---------------|----------------|---------------|---------|
| wood (x3)     | 0.36-0.47 / 0.38-0.49 | 0.22-0.26 / 0.23-0.27 | current, slight |
| wall-steel    | 0.87 / 0.80    | 0.26 / 0.32   | current tighter, nibbles content |
| charcoal      | 0.58 / 0.52    | 0.16 / 0.19   | current, noisy metric |
| tile-battery  | 0.99 / 0.96    | 0.37 / 0.52   | current, much tighter |
| gloss-battery | 1.00 / 0.88    | 0.22 / 0.29   | current tighter, nibbles |
| white-fold    | 0.95 / 0.59    | 0.12 / 0.15   | BASELINE by a mile (diagonal cuts) |
| bedsheet      | 0.64 / 0.57    | 0.09 / 0.13   | baseline, both weak |

Interpretation: the composed pipeline pays rent on 8/10 scenes (tighter
with minor content cost); on low-boundary-contrast scenes (white-fold,
bedsheet) the modern pipeline's cuts drop a THIRD of the content that
the baseline's loose blob quad kept.

## Recommended next step (not yet implemented)

Cover-constrained selection: keep multiple candidates through the
pipeline (simple loose blob quad + composed result) and choose by
maximize FILL subject to COVER >= 0.9 x best available cover. This is
the calibrated content-guard designed right: per-frame, relative,
measured — not a fixed ring threshold (the naive version collapsed SRD
to 1.5%). Requires retaining a pre-polish candidate in detectWhiteness.

## Jury branch completion (2026-07-23, branch jury-of-hypotheses)

Judge evolution, each step referee-driven:
1. cover x fill (ablation recipe) -> 3 scene regressions: fill is blind
   to boundary placement.
2. + sharpness seat (fill x f(edgeQuality)) -> fixed 2 frames.
3. cover polluted by bg texture (leather seams count as ink; correct
   tight quad scored cover 0.46, ineligible). Density filter: creases
   are locally dense, no help. Paper-adjacency filter: sheen is
   paperish, partial help.
4. CONSENSUS selection: document ink = ink inside majority of
   candidates; eligibility = keep >=95% of consensus ink; winner =
   fill x sharpness. Sidesteps map pollution entirely.
5. consider() gained absolute in-frame sanity (+-8px).

DISCOVERY: v6.0-main sofa goldens were CORRUPTED — later edits in the
trimodal session regressed the scene after the verification render,
and --update pinned the regression (diamond quads [264,240,643,22...]).
Blind pinning after intermediate edits is the failure mode; verify-
then-pin must be atomic. Stripped on this branch; main carries them
until merge.

Final referee: corpus 124/124 (98 pinned + 26 known-hard flags),
negatives 2/35, SRD 92.5%, synth OK. Perf: 289ms SRD mean — flood +
multi-generator runs on hard scenes; easy scenes exit after one
generator. Perf tuning is the remaining pre-merge work, plus the four
known-hard scenes (white-fold, bedsheet, hand-held, sofa) which now
fail equally on BOTH architectures.
