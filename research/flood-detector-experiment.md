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
