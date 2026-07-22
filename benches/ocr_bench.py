"""End-to-end OCR benchmark: detect -> dewarp (padded) -> OCR -> coherence.

Runs the SRD bills dataset through the Python mirror of the production
detector, dewarps with 3% quad padding, then OCRs the crops with one or
more engines and scores coherence. Use this to track whether detector /
dewarp changes help or hurt the thing that actually matters (OCR).

Engines:
  vision    - Apple Vision via scripts/vision_ocr.swift (macOS)
  tesseract - tesseract CLI, --psm 4
  larngear  - larngear_docling service at localhost:8000 (/convert)

Usage:
  python3 benches/ocr_bench.py --engines vision,tesseract [--limit 50]

Results append to artifacts/bench/ocr_results.jsonl (one line per run,
tagged with date + engines) so successive runs are comparable.
"""
import argparse, glob, json, os, re, subprocess, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autocapture.detector.classical import _detect_v3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "/Users/jsaksrisuwan/workspace/receipts-dataset"
WORK = os.path.join(ROOT, "artifacts", "bench", "ocr_crops")
RESULTS = os.path.join(ROOT, "artifacts", "bench", "ocr_results.jsonl")
PRICE = re.compile(r"\d+[.,]\d{2}\b")


class Cfg:
    long_edge = 640


def dewarp_all(limit=None):
    os.makedirs(WORK, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATASET, "*.jpg")))[:limit]
    crops = []
    for f in files:
        name = os.path.basename(f)[:-4]
        out = os.path.join(WORK, name + ".png")
        bgr = cv2.imread(f)
        q = _detect_v3(bgr, Cfg)
        if q is None:
            continue
        q = q.astype(np.float32)
        c = q.mean(0)
        q = c + (q - c) * 1.03  # same padding as src/app.js hiResCapture
        w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
        h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
        W, H = int(w), int(h)
        if W < 40 or H < 40:
            continue
        M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
        cv2.imwrite(out, cv2.warpPerspective(bgr, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
        crops.append((name, out))
    return len(files), crops


def score(text):
    return {"chars": len(text.replace("\n", "")), "prices": len(PRICE.findall(text))}


def ocr_vision(crops):
    reqs = "\n".join(
        json.dumps({"id": i, "cmd": "ocr", "path": p, "level": "accurate", "languages": ["en-US"]})
        for i, (_, p) in enumerate(crops)
    ) + "\n"
    r = subprocess.run(["swift", os.path.join(ROOT, "scripts", "vision_ocr.swift")],
                       input=reqs, capture_output=True, text=True, timeout=3600)
    out = {}
    for line in r.stdout.splitlines():
        try:
            j = json.loads(line)
            out[crops[j["id"]][0]] = j.get("text", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    return out


def ocr_tesseract(crops):
    out = {}
    for name, p in crops:
        r = subprocess.run(["tesseract", p, "stdout", "--psm", "4", "-l", "eng"],
                           capture_output=True, text=True, timeout=120)
        out[name] = r.stdout
    return out


def ocr_larngear(crops, url="http://localhost:8000"):
    import urllib.request
    out = {}
    for name, p in crops:
        body, boundary = _multipart(p)
        req = urllib.request.Request(
            url + "/convert?output=markdown", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                j = json.loads(resp.read())
            out[name] = j.get("markdown", "") if isinstance(j, dict) else str(j)
        except Exception as e:
            out[name] = ""
            print(f"  larngear fail {name}: {e}", file=sys.stderr)
    return out


def _multipart(path):
    boundary = "larngearbench"
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: image/png\r\n\r\n"
            ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


ENGINES = {"vision": ocr_vision, "tesseract": ocr_tesseract, "larngear": ocr_larngear}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="vision")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    total, crops = dewarp_all(args.limit)
    print(f"detected+dewarped {len(crops)}/{total}")

    run = {"ts": time.strftime("%Y-%m-%d %H:%M"), "detected": len(crops), "total": total, "engines": {}}
    for eng in args.engines.split(","):
        t0 = time.time()
        texts = ENGINES[eng](crops)
        scores = [score(t) for t in texts.values()]
        agg = {k: round(sum(s[k] for s in scores) / max(1, len(scores)), 2) for k in ("chars", "prices")}
        agg["secs"] = round(time.time() - t0, 1)
        run["engines"][eng] = agg
        print(f"{eng}: {agg}")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(run) + "\n")
    print("appended ->", RESULTS)


if __name__ == "__main__":
    main()
