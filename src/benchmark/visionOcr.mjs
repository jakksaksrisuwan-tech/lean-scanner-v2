// Apple Vision OCR client.
//
// Spawns the Swift helper at scripts/vision_ocr.swift exactly once and
// keeps it alive across frames. Communication is line-delimited JSON on
// stdin/stdout. Per-frame calls round-trip in well under 100ms once the
// Vision model is loaded.
//
// The Swift helper is the only piece of code that touches the Vision
// framework; everything else stays in pure JS so the benchmark runs the
// same way on any Mac with the Apple Vision SDK.

import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");
const SCRIPT = join(REPO_ROOT, "scripts", "vision_ocr.swift");

let proc = null;
let nextId = 1;
const pending = new Map();

function ensure() {
  if (proc) return;
  proc = spawn("swift", [SCRIPT], { stdio: ["pipe", "pipe", "pipe"] });
  let buf = "";
  proc.stdout.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        const waiter = pending.get(msg.id);
        if (!waiter) continue;
        pending.delete(msg.id);
        if (msg.error) waiter.reject(new Error(msg.error));
        else waiter.resolve(msg);
      } catch (e) {
        // Print and ignore: protocol error means the helper is wedged.
        console.error("vision_ocr: bad JSON line:", line);
      }
    }
  });
  proc.stderr.on("data", (chunk) => process.stderr.write("[vision_ocr] " + chunk));
  proc.on("exit", (code) => {
    for (const [, w] of pending) w.reject(new Error(`vision_ocr exited code=${code}`));
    pending.clear();
    proc = null;
  });
}

function call(payload) {
  ensure();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    proc.stdin.write(JSON.stringify({ id, ...payload }) + "\n");
  });
}

/**
 * OCR a single image file (already on disk as PNG / JPEG / etc.).
 * Returns the recognized text plus per-line confidence.
 *
 * @param {string} path
 * @param {object} [opts]
 * @param {"fast"|"accurate"} [opts.level="accurate"]
 * @param {string[]} [opts.languages]  e.g. ["en-US"]
 * @returns {Promise<{text: string, lines: Array<{text: string, confidence: number, bbox: number[]}>}>}
 */
export function ocrFile(path, opts = {}) {
  const { level = "accurate", languages = ["en-US"] } = opts;
  return call({ cmd: "ocr", path, level, languages });
}

/** Same API but operates on a Uint8Array PNG. Writes a temp file once. */
export async function ocrPng(pngBytes, opts = {}) {
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(join(tmpdir(), "ls-ocr-"));
  const path = join(dir, "frame.png");
  writeFileSync(path, pngBytes);
  try {
    return await ocrFile(path, opts);
  } finally {
    const { rmSync } = await import("node:fs");
    try { rmSync(dir, { recursive: true, force: true }); } catch {}
  }
}

/** Force shutdown (mostly for tests / clean exit). */
export function shutdown() {
  if (!proc) return;
  try { proc.stdin.end(); } catch {}
  try { proc.kill(); } catch {}
  proc = null;
}
