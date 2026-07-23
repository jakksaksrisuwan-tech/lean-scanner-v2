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
