// Apple Vision OCR helper.
//
// Read JSON-line requests on stdin. Each request is one of:
//
//   {"id": 1, "cmd": "ocr", "path": "/abs/path.png", "level": "accurate", "languages": ["en-US"]}
//
// Respond with one JSON line per request, e.g.:
//
//   {"id": 1, "text": "TOTAL $5.99", "lines": [...]}
//
// The Vision request is created lazily on the first call and reused for
// the lifetime of the process so the per-frame latency stays low.

import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count >= 1 else { exit(2) }

let stdin = FileHandle.standardInput
let stdout = FileHandle.standardOutput

func emit(_ obj: [String: Any]) {
  if let data = try? JSONSerialization.data(withJSONObject: obj, options: []) {
    stdout.write(data)
    stdout.write(Data([0x0A]))
  }
}

var ocrRequest: VNRecognizeTextRequest?

func ensureRequest(level: VNRequestTextRecognitionLevel, languages: [String]) {
  if ocrRequest != nil { return }
  let req = VNRecognizeTextRequest()
  req.recognitionLevel = level
  req.usesLanguageCorrection = false
  if !languages.isEmpty {
    req.recognitionLanguages = languages
  }
  ocrRequest = req
}

func freshRequest(level: VNRequestTextRecognitionLevel, languages: [String]) -> VNRecognizeTextRequest {
  let req = VNRecognizeTextRequest()
  req.recognitionLevel = level
  req.usesLanguageCorrection = false
  if !languages.isEmpty {
    req.recognitionLanguages = languages
  }
  return req
}

func performOCR(path: String, level: VNRequestTextRecognitionLevel, languages: [String]) throws -> [String: Any] {
  let url = URL(fileURLWithPath: path)
  guard let image = NSImage(contentsOf: url),
        let tiff = image.tiffRepresentation,
        let bitmap = NSBitmapImageRep(data: tiff),
        let cg = bitmap.cgImage else {
    return ["text": "", "lines": [], "warning": "could not load image"]
  }
  // Use a fresh request per call. Reusing a single VNRecognizeTextRequest
  // across calls causes observations to leak between images, which breaks
  // the recognizer on the 2nd+ image.
  let req = freshRequest(level: level, languages: languages)
  let handler = VNImageRequestHandler(cgImage: cg, options: [:])
  try handler.perform([req])
  let observations = (req.results as? [VNRecognizedTextObservation]) ?? []
  var lines: [[String: Any]] = []
  var full: [String] = []
  for obs in observations {
    guard let top = obs.topCandidates(1).first else { continue }
    let text = top.string
    full.append(text)
    let bb = obs.boundingBox
    let conf = top.confidence
    lines.append([
      "text": text,
      "confidence": conf,
      "bbox": [bb.origin.x, bb.origin.y, bb.size.width, bb.size.height],
    ])
  }
  return ["text": full.joined(separator: "\n"), "lines": lines]
}

// Read stdin fully, split on newlines, process each request.
var buffer = Data()
while true {
  let chunk = stdin.availableData
  if chunk.isEmpty { break }
  buffer.append(chunk)
  while let nl = buffer.firstIndex(of: 0x0A) {
    let line = buffer.subdata(in: 0..<nl)
    buffer.removeSubrange(0..<(nl + 1))
    if line.isEmpty { continue }
    guard let raw = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else {
      emit(["error": "bad-json"])
      continue
    }
    let id = raw["id"] as? Int ?? 0
    let cmd = raw["cmd"] as? String ?? ""
    if cmd == "ocr" {
      let path = raw["path"] as? String ?? ""
      let levelStr = raw["level"] as? String ?? "accurate"
      let langs = raw["languages"] as? [String] ?? ["en-US"]
      let level: VNRequestTextRecognitionLevel = levelStr == "fast"
        ? .fast
        : .accurate
      do {
        let payload = try performOCR(path: path, level: level, languages: langs)
        var merged: [String: Any] = ["id": id]
        for (k, v) in payload { merged[k] = v }
        emit(merged)
      } catch {
        emit(["id": id, "error": "\(error)"])
      }
    } else if cmd == "ping" {
      emit(["id": id, "pong": true])
    } else {
      emit(["id": id, "error": "unknown-cmd"])
    }
  }
}
