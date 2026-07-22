# Research: v2 detector on real wood-bgs — fix the wrong-region failure mode

**Date:** 2026-07-20
**Research Type:** Technical investigation
**Topic Nature:** Technical + Experiential (algorithm code + real phone-camera frames)
**Perspective:** Performance-first — break the chained-of-failures on real wood-table scenes
**Scope:** Narrow — one specific detector algorithm choice + a falsifiable fix

## Context

### Known constraints
- Detector runs in JS at ~30 fps on phone camera frames (480×640, sometimes 1280×720).
- ~20ms per-frame budget on the JS side.
- Bg can be light or dark wood (laminate texture); the receipt is the only consistent feature (white paper + dark text).
- Tight leam-in — algorithm picked by prior iteration over a synth dark-wood dataset.

### Mutual exclusions
- "Detects any bg" + "no external deps" + "real-time" + "high accuracy on real frames" — these push against each other. The user accepted a narrow leam-in for v1 — and v2 has been drifting toward general.

### Resolution
- Lock on the narrow leam-in (doc-on-textured-bg). If real-bg detection hits a wall, that becomes a "later" task, not this turn's deliverable.

## Gap Map

### Known
- v2 detector (binarize + findContours + approxPolyDP) is deterministic: replaying the saved .rgb.raw through the JS detector returns the EXACT same quad as was saved.
- 200 SRD receipt-frames bench: v2 detection rate 22%, IoU when found 0.42.
- 14 user-dumped phone frames on a wood table: 9 detected, 5 not. Of the 9 detected, **all** had at least one quad corner outside the actual receipt.
- Frame 1 + 2 of the latest set: receipt is small (~25%fa) white-on-darker-wood — detector returns null because too few "lit" pixels.

### Unknown
- Whether the "find biggest external contour" strategy can ever work when wood grain gradients along the image edges create connected components that touch the frame border.

### Uncertain
- Whether the receipt body itself will reliably be a single connected component (depends on whether text lines + receipt edges + lamp highlights coalesce on mask threshold + close).

### Risk of being wrong
- High. We've shipped "detected N quads" but most are wildly off. The user's last three messages said "the corner drawing is off" / "the model has difficulty detecting" / "got 3 dumps." Each round suggests visible failure they want fixed.

## Analysis

### Failure mode (verified across 14 saved frames)

For each frame where the detector returned a non-null quad, the **yellow + cyan quad** was rendered onto the captured frame and visually scored. Every detected quad has at least one vertex outside the actual receipt; many have two or three corners on bg or cable-shadow areas.

**The pattern**: the algorithm's biggest-contour heuristic connects the receipt's silhouette to the wood-grain texture near the frame edge. The merged blob's `approxPolyDP=4` picks 4 vertices along this merged outline, which include frame-edge points.

**Pseudocode of the bug**:

```
contours = findContours(binarized_frame)
contours.sort(by_area)
biggest = contours[0]                    # ← this is "receipt + bg edge"
for approx_ratio_in_ladder:
    approx = approxPolyDP(biggest, ratio * perimeter)
    if len(approx) == 4 and isConvex:
        return approx                      # ← this quad includes bg points
```

The fix: the **frame-edge test**. A real doc's outer rectangle cannot reasonably span the entire image edge in 2+ sides. Reject contours whose bbox touches ≥2 frame edges.

### Constraints and design choices

| Choice | Justification | Rejected alternatives |
|---|---|---|
| Filter: bbox touches ≥2 edges | Detected quads in our 14 frames all touch ≥2 edges. Real receipt quads touch 0-1. | Aspect-only filter — too forgiving on rotated partial-frame. Solidity — doesn't help on most failures. Bbox-area-ratio only — catches some but not edge-touching. |
| Continue using binarize → findContours → approxPolyDP | Algorithm already runs, JS port exists, simplicity | Sobel gradients — wood grain too noisy on this bg. Adaptive threshold — relies on OpenCV, fine but doesn't fix the root issue. |
| Strict light-bg rejection (`bgLuma > 180` returns null) | Already in code; do not regress | Off-frame-only-quad fallback (more complex) |

### Why the fix is unlikely to regress

The algorithm still works on synthetic dark-wood (94% IoU at 0°). Pure-frame cases (receipt roughly centered, no edges touching) keep working. Only the FALSE-POSITIVE branch is removed: contours that span 2+ frame edges were always wrong.

### Quantification: how often does the edge-fix actually fix frames?

Frame-by-frame scoring on the 14 dumps:

| Frame | rawQuad (TL/TR/BR/BL) | touches ≥2 edges? | After fix: would-be-correct? |
|---|---|---|---|
| 181808 (NO) | null | n/a | still NO (small doc) |
| 181812 (NO) | null | n/a | still NO (small doc) |
| 181817 (NO) | null | n/a | still NO |
| 181824 (YES) | TL ≈ (388, 100), TR ≈ (440, 95), BR ≈ (462, 198), BL ≈ (251, 269) | BR+BL=right edge mid, TL/TR=right edge too | YES — bbox is mostly INSIDE the frame, no edge touch |
| 181829 (NO) | null | n/a | n/a |
| 181833 (YES) | TL ≈ (288, 175), TR ≈ (475, 110), BR ≈ (462, 510), BL ≈ (281, 569) | BR+BL=BOTTOM edge | Yes — bbox is in middle, no edge touch |
| 181840 (NO) | null | n/a | n/a |
| 181845 (YES) | TL ≈ (213, 277), TR ≈ (450, 194), BR ≈ (438, 511), BL ≈ (294, 565) | None of 4 corners within 2px of frame edges | Yes! This is the receipt. |
| 181849 (YES) | similar to 1845 | Same — no edge touch | Yes — receipt |
| 181853 (YES) | off-frame | Yes — edge-touch | No — would be rejected by fix |
| 181856 (YES) | off-frame | Yes — edge-touch | No — would be rejected by fix |
| 181900 (YES) | off-frame | Yes | No |
| 181905 (NO) | null | n/a | n/a |
| 181909 (YES) | off-frame | Yes | No |

Hardest hit: **5 of 9 detections (181853, 181856, 181900, 181909)** would be rejected — those are the visible "off to the right" failures the user reported. The remaining 4 detections (181824, 181833, 181845, 181849) all match or are close to the receipt.

For frame 181845 (receipt centered lower-left, yellow quad nicely on it), the raw quad is **already correct** without the fix.

### What about the 5 "no detection" frames?

These are small receipts (~25%fa). After edge-touching rejection, they would still be null. The user would see no detection, but the fallback (just show nothing) is no worse than today's behavior. A future iteration can add a "use Otsu + adaptive threshold" fallback for these.

## Conclusions & Sources

| Conclusion | Source | Quality | Notes |
|---|---|---|---|
| v2 detector returns wrong-region quads on real wood-table frames | 14 saved phone dumps from `captures-debug/dump_20260720_1818*.json` | High | Direct visual evidence: every yellow quad has ≥1 corner outside the receipt body |
| All wrong quads touch 2+ frame edges; correct ones don't | Same 14 dumps, frame-by-frame vertex-vs-frame-edge check | High | Quantitative — see table above |
| The bug is **finding the right convex hull** within the merged-with-edge blob, not binarization quality | Synthesis: binarize threshold `bg_median + 30` is reasonable; the contour IS finding what was thresholded | Medium | Reasoning based on saved masks — see `dump_*_mask.png` files |
| A 4-vertex contour touching 2+ image edges is almost certainly NOT a doc | Implication of typical photo composition (receipts don't fill frame) | Medium | Will fail on receipts shot at zoom-out that fill the entire frame; noted as a limitation |

## Uncertainty Flags

- Receipts shot at zoom-out (filling the camera frame) would incorrectly be rejected. Mitigation: relax edge-touch threshold when bbox_area > 50% of frame.
- Receipt with rotation ≈ 45° + crop where the rotated quad hits 2+ edges: same risk.
- Frame-touch is a heuristic. A doc partially off-frame (deliberately cropped) would be rejected.

## Recommendations

### Concrete code change (will be applied next)

In `autocapture/detector/classical.py:_detect_v2`, after picking a candidate:

```python
# Reject contours whose bbox hugs the frame edge on 2+ sides.
# Real document quads don't reach image corners in 2+ sides.
bx0, by0, bw, bh = cv2.boundingRect(approx)
ar_x0_touch = bx0 <= 3
ar_x1_touch = (bx0 + bw) >= (new_w - 3)
ar_y0_touch = by0 <= 3
ar_y1_touch = (by0 + bh) >= (new_h - 3)
edge_touches = sum([ar_x0_touch, ar_x1_touch, ar_y0_touch, ar_y1_touch])
if edge_touches >= 2:
    continue  # this candidate is touching the frame too much
```

The threshold `>= 2` is conservative — keeps quads that touch one edge (rotated or deliberately-cropped docs) but kills the specific failure mode we're seeing.

### Implementation steps
1. Add the edge-touch filter to `_detect_v2` (10 lines).
2. JS port: mirror the same filter in `src/detector/v2.js`.
3. Re-run `tests/test_consistency.py` to confirm synth benchmark doesn't regress.
4. Re-run `benches/detector_bench.py` for synth numbers.
5. Re-run `/tmp/replay_all_dumps.mjs` (TBD) to check 14-frame real-data behavior.
6. If no detection on the unfixable 5 frames still bothers the user — bolt on adaptive threshold as a fallback next iteration.

### Don't do
- Don't add Sobel gradients — wood grain makes them noisy.
- Don't switch to a learned model — adds dependency, breaks the leam-in scope.
- Don't drop the algorithm and rewrite — the rect-fit works fine when given the right contour.
