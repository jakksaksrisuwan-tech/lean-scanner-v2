// Service worker for the lean-scanner PWA.
// Offline-first: cache the app shell + modules, network-fall back.
const CACHE = "lean-scanner-v5.0-flood";
const SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./src/app.js",
  "./src/config.js",
  "./src/pipeline.js",
  "./src/quality.js",
  "./src/simulator.js",
  "./src/math/ndarray.js",
  "./src/math/svd.js",
  "./src/detector/quad.js",
  "./src/detector/classical.js",
  "./src/detector/v2.js",
  "./src/detector/v3.js",
  "./src/detector/radial.js",
  "./src/dewarp/homography.js",
  "./src/dewarp/superscan.js",
  "./src/dewarp/shading.js",
  "./src/dewarp/contrast.js",
  "./src/tracker/kalman.js",
  "./src/tracker/lock.js",
  "./tests/test_consistency.html",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  e.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((resp) => {
        // Cache successful same-origin GETs opportunistically
        if (resp.ok && new URL(req.url).origin === self.location.origin) {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return resp;
      }).catch(() => caches.match("./index.html"));
    })
  );
});