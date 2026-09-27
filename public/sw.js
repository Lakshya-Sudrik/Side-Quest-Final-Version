/**
 * SideQuest — Service Worker
 * Strategy:
 *   - Static assets (JS/CSS/fonts/images): Cache-first with network fallback
 *   - API requests (fetch to /api/*): Network-first with cache fallback
 *   - Offline fallback: serves cached shell or shows offline page
 */

const CACHE_NAME = 'sidequest-v1';
const OFFLINE_URL = '/';

// Assets to pre-cache on install (app shell)
const PRECACHE_ASSETS = [
  '/',
  '/src/main.tsx',
  '/src/index.css',
  '/manifest.json',
];

// ─────────────────────────────────────────────────────
// INSTALL: Pre-cache app shell
// ─────────────────────────────────────────────────────
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Pre-caching app shell');
      // Gracefully handle pre-caching (don't fail if a resource is unavailable)
      return Promise.allSettled(
        PRECACHE_ASSETS.map((url) =>
          cache.add(url).catch(() => console.warn(`[SW] Failed to cache: ${url}`))
        )
      );
    }).then(() => self.skipWaiting())
  );
});

// ─────────────────────────────────────────────────────
// ACTIVATE: Clean up old caches
// ─────────────────────────────────────────────────────
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => {
            console.log('[SW] Deleting old cache:', key);
            return caches.delete(key);
          })
      )
    ).then(() => self.clients.claim())
  );
});

// ─────────────────────────────────────────────────────
// FETCH: Smart caching strategy
// ─────────────────────────────────────────────────────
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests and browser extensions
  if (request.method !== 'GET') return;
  if (!url.protocol.startsWith('http')) return;

  // API calls → Network-first, fallback to cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }

  // Google Fonts & external CDN → Cache-first (stale-while-revalidate)
  if (
    url.hostname.includes('fonts.googleapis.com') ||
    url.hostname.includes('fonts.gstatic.com') ||
    url.hostname.includes('lh3.googleusercontent.com')
  ) {
    event.respondWith(cacheFirstStrategy(request));
    return;
  }

  // App shell & static assets → Cache-first
  event.respondWith(cacheFirstStrategy(request));
});

// ─────────────────────────────────────────────────────
// Cache-first: Try cache, fall back to network + update cache
// ─────────────────────────────────────────────────────
async function cacheFirstStrategy(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch {
    // Offline and not cached — return offline fallback
    const fallback = await caches.match(OFFLINE_URL);
    return fallback || new Response('<h1>SideQuest is offline</h1><p>Please reconnect to continue exploring hidden gems.</p>', {
      headers: { 'Content-Type': 'text/html' },
    });
  }
}

// ─────────────────────────────────────────────────────
// Network-first: Try network, fall back to cache
// ─────────────────────────────────────────────────────
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch {
    const cached = await caches.match(request);
    return cached || new Response(JSON.stringify({ error: 'Offline — data unavailable' }), {
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
