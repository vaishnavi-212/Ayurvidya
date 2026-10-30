// Offline support: the app shell is cached; verse data is fetched fresh and the last copy is kept for offline use.
// (Browsers only run this on https:// or localhost - i.e. after Stage 6 deployment.)
const CACHE = "ayurvidya-v2";
const SHELL = ["./", "index.html", "style.css", "app.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png"];

self.addEventListener("install", e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;                 // chat questions always go to the server
  if (new URL(req.url).pathname.startsWith("/notes")) return;   // a student's own notes are never cached
  e.respondWith(fetch(req).then(res => {             // network first, keep a copy
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req)));
});