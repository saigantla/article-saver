// Service Worker for Article Saver - Offline Support
// Handles caching of app shell, CDN resources, and API responses

const CACHE_VERSION = 'v38';  // Add time displays to audio player
const CACHE_NAMES = {
  app: `app-shell-${CACHE_VERSION}`,
  cdn: `cdn-assets-${CACHE_VERSION}`,
  api: `api-cache-${CACHE_VERSION}`
};

// Resources to cache immediately on install
const APP_SHELL = [
  '/',
  '/index.html'
];

const CDN_RESOURCES = [
  'https://cdn.tailwindcss.com',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap',
  'https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300;1,400&display=swap'
];

// ============================================================================
// INSTALL EVENT
// ============================================================================

self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');

  event.waitUntil(
    (async () => {
      try {
        // Open app shell cache
        const appCache = await caches.open(CACHE_NAMES.app);
        console.log('[SW] Caching app shell:', APP_SHELL);
        await appCache.addAll(APP_SHELL);

        // Open CDN cache
        const cdnCache = await caches.open(CACHE_NAMES.cdn);
        console.log('[SW] Caching CDN resources:', CDN_RESOURCES);

        // Cache CDN resources with error handling (they might fail)
        await Promise.allSettled(
          CDN_RESOURCES.map(url =>
            cdnCache.add(url).catch(err =>
              console.warn('[SW] Failed to cache:', url, err)
            )
          )
        );

        console.log('[SW] Service worker installed successfully');

        // Skip waiting to activate immediately
        self.skipWaiting();
      } catch (error) {
        console.error('[SW] Install failed:', error);
      }
    })()
  );
});

// ============================================================================
// ACTIVATE EVENT
// ============================================================================

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');

  event.waitUntil(
    (async () => {
      try {
        // Get all cache names
        const cacheNames = await caches.keys();

        // Get list of current cache names
        const currentCaches = Object.values(CACHE_NAMES);

        // Delete old caches
        await Promise.all(
          cacheNames.map(cacheName => {
            if (!currentCaches.includes(cacheName)) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );

        console.log('[SW] Service worker activated');

        // Claim all clients immediately
        await self.clients.claim();
      } catch (error) {
        console.error('[SW] Activation failed:', error);
      }
    })()
  );
});

// ============================================================================
// FETCH EVENT
// ============================================================================

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle http/https requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // Determine caching strategy based on request
  event.respondWith(handleFetch(request));
});

// ============================================================================
// FETCH HANDLERS
// ============================================================================

async function handleFetch(request) {
  const url = new URL(request.url);

  // Strategy 1: App Shell - Cache-first
  if (isAppShell(url)) {
    return cacheFirst(request, CACHE_NAMES.app);
  }

  // Strategy 2: CDN Resources - Cache-first
  if (isCDN(url)) {
    return cacheFirst(request, CACHE_NAMES.cdn);
  }

  // Strategy 3: API Read Operations - Network-first with cache fallback
  if (isAPIRead(url)) {
    return networkFirst(request, CACHE_NAMES.api);
  }

  // Strategy 4: API Write Operations - Network-only (no caching)
  if (isAPIWrite(request)) {
    return networkOnly(request);
  }

  // Default: Network-first
  return networkFirst(request, CACHE_NAMES.api);
}

// ============================================================================
// CACHING STRATEGIES
// ============================================================================

// Cache-first: Try cache, fallback to network, update cache
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  if (cached) {
    console.log('[SW] Cache hit:', request.url);

    // Update cache in background (stale-while-revalidate)
    fetch(request)
      .then(response => {
        if (response && response.status === 200) {
          cache.put(request, response.clone());
        }
      })
      .catch(() => {}); // Ignore background update errors

    return cached;
  }

  console.log('[SW] Cache miss, fetching:', request.url);
  try {
    const response = await fetch(request);
    if (response && response.status === 200) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[SW] Fetch failed:', request.url, error);
    throw error;
  }
}

// Network-first: Try network, fallback to cache
async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);

  try {
    console.log('[SW] Fetching from network:', request.url);
    const response = await fetch(request);

    if (response && response.status === 200) {
      console.log('[SW] Caching response:', request.url);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    console.warn('[SW] Network failed, trying cache:', request.url);
    const cached = await cache.match(request);

    if (cached) {
      console.log('[SW] Cache hit (fallback):', request.url);
      return cached;
    }

    console.error('[SW] Cache miss, request failed:', request.url);
    throw error;
  }
}

// Network-only: Always fetch from network
async function networkOnly(request) {
  console.log('[SW] Network-only:', request.url);
  return fetch(request);
}

// ============================================================================
// URL CLASSIFICATION HELPERS
// ============================================================================

function isAppShell(url) {
  const pathname = url.pathname;
  return pathname === '/' || pathname === '/index.html';
}

function isCDN(url) {
  const cdnHosts = [
    'cdn.tailwindcss.com',
    'cdnjs.cloudflare.com',
    'fonts.googleapis.com',
    'fonts.gstatic.com'
  ];
  return cdnHosts.some(host => url.hostname.includes(host));
}

function isAPIRead(url) {
  const pathname = url.pathname;
  const method = url.method || 'GET';

  // GET requests to API endpoints
  return (
    pathname.startsWith('/articles') ||
    pathname.startsWith('/health')
  ) && method === 'GET';
}

function isAPIWrite(request) {
  const method = request.method;
  return ['POST', 'PUT', 'DELETE', 'PATCH'].includes(method);
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);

  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
      })
    );
  }
});

console.log('[SW] Service worker loaded');
