"""Tiny static server for the lean-scanner PWA.

Why not `python -m http.server`? Two reasons:
1. The service worker (sw.js) requires `Service-Worker-Allowed` header.
2. getUserMedia needs a secure context, which means HTTPS on real hosts
   or localhost. This server prints the URL you should open.

Run:
    python3 server.py            # default port 8000
    python3 server.py 8080       # custom port
"""
from __future__ import annotations
import http.server
import socketserver
import sys
import os


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".webmanifest": "application/manifest+json",
        ".wasm": "application/wasm",
    }

    def end_headers(self) -> None:
        # Service worker scope
        if self.path.endswith("/sw.js") or self.path == "/sw.js":
            self.send_header("Service-Worker-Allowed", "/")
        # Avoid caching during dev
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        # /api/dump:    simple — raw RGB + metadata.
        # /api/dump-rich: named layers (rgb, mask, edges, contours_dbg, …).
        # Both write per-dump files to captures-debug/.
        is_rich = self.path == "/api/dump-rich"
        if self.path not in ("/api/dump", "/api/dump-rich"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            import json as _json, base64 as _b64, os as _os, time as _time
            payload = _json.loads(body)
            w = int(payload["w"]); h = int(payload["h"])
            out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                    "captures-debug")
            _os.makedirs(out_dir, exist_ok=True)
            ts = _time.strftime('%Y%m%d_%H%M%S')
            base = f"dump_{ts}_{w}x{h}"
            main_path = _os.path.join(out_dir, base)

            def _write_layer(name, b64, ext="rgb"):
                if not b64: return None
                # base64.b64decode in some Python builds raises on bad
                # padding (incl. validate=False). Tolerate it — we'd rather
                # store garbage than 500 the whole dump.
                try:
                    raw = _b64.b64decode(b64, validate=False)
                except Exception:
                    # Fall back to manual decode with padding-zeroing.
                    raw = _b64.b64decode(b64 + "=" * ((4 - len(b64) % 4) % 4))
                file_ext = ext if name != "rgb" else "raw"
                p = f"{main_path}.{name}.{file_ext}"
                with open(p, "wb") as f:
                    f.write(raw)
                return _os.path.basename(p)

            written = []
            if is_rich:
                # layers is a dict of {layer_name: base64_str}; required "rgb"
                layers = payload.get("layers") or {}
                for name, b64 in layers.items():
                    # Convention: extensions describe the encoding. The
                    # rgb layer gets `.raw` extension to avoid `.rgb.rgb`.
                    ext = "png" if name in ("mask", "edges", "heatmap", "contours_dbg",
                                            "mask_closed", "text_score") else "raw"
                    fn = _write_layer(name, b64, ext=ext)
                    if fn: written.append((name, fn))
            else:
                # legacy single-RGB payload
                try:
                    raw = _b64.b64decode(payload["b64"], validate=False)
                except Exception:
                    raw = _b64.b64decode(payload["b64"] + "=" * ((4 - len(payload["b64"]) % 4) % 4))
                p = main_path + ".rgb.raw"
                with open(p, "wb") as f:
                    f.write(raw)
                written.append(("rgb", _os.path.basename(p)))

            meta = {
                "w": w, "h": h,
                "time": _time.time(),
                "layers": [{ "name": n, "path": p } for n, p in written],
                "rawQuad": payload.get("rawQuad"),
                "smoothQuad": payload.get("smoothQuad"),
                "centroid": payload.get("centroid"),
                "conf": payload.get("conf"),
                "detMs": payload.get("detMs"),
                "dumpIdx": payload.get("dumpIdx"),
                "mode": "rich" if is_rich else "simple",
            }
            with open(main_path + ".json", "w") as f:
                _json.dump(meta, f, indent=2)
            resp = _json.dumps({"path": _os.path.basename(main_path + ".json"),
                                "layers": meta["layers"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            import traceback as _tb, sys as _sys
            _tb.print_exc(file=_sys.stderr)
            err = ('{"error": "' + str(e).replace('"', '\\"') + '"}').encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", port), Handler) as httpd:
        host = "localhost"  # getUserMedia needs a secure context
        print(f"lean-scanner serving at http://{host}:{port}/")
        print(f"open http://{host}:{port}/ in chrome/safari (must be localhost or https)")
        print("ctrl-c to quit")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())