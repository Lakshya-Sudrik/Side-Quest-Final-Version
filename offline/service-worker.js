/*
 * SideQuest Service Worker
 * Enables offline functionality and caching
 */

const CACHE_NAME = 'sidequest-v1';
const STATIC_CACHE = 'sidequest-static-v1';
const DATA_CACHE = 'sidequest-data-v1';

// Static assets to cache
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png'
];

// Install event
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[ServiceWorker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== DATA_CACHE) {
            console.log('[ServiceWorker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - Network first, falling back to cache
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          // Clone the response before caching
          const responseClone = response.clone();
          
          caches.open(DATA_CACHE)
            .then((cache) => {
              cache.put(event.request, responseClone);
            });
          
          return response;
        })
        .catch(() => {
          // Return cached API response if available
          return caches.match(event.request)
            .then((cachedResponse) => {
              if (cachedResponse) {
                return cachedResponse;
              }
              
              // Return offline data for specific endpoints
              if (url.pathname.includes('/recommend')) {
                return new Response(JSON.stringify({
                  offline: true,
                  message: 'Offline mode - showing cached recommendations',
                  recommendations: getOfflineRecommendations(url.searchParams.get('city'))
                }), {
                  headers: { 'Content-Type': 'application/json' }
                });
              }
              
              return new Response(JSON.stringify({
                offline: true,
                message: 'No internet connection. Showing cached data.'
              }), {
                headers: { 'Content-Type': 'application/json' }
              });
            });
        })
    );
    return;
  }
  
  // Handle static assets - Cache first
  event.respondWith(
    caches.match(event.request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          // Return cached version, but fetch in background to update
          fetch(event.request)
            .then((response) => {
              caches.open(STATIC_CACHE)
                .then((cache) => {
                  cache.put(event.request, response);
                });
            })
            .catch(() => {});
          
          return cachedResponse;
        }
        
        return fetch(event.request)
          .then((response) => {
            const responseClone = response.clone();
            
            caches.open(STATIC_CACHE)
              .then((cache) => {
                cache.put(event.request, responseClone);
              });
            
            return response;
          });
      })
  );
});

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-reviews') {
    event.waitUntil(syncReviews());
  }
  
  if (event.tag === 'sync-checkins') {
    event.waitUntil(syncCheckins());
  }
});

// Sync reviews when back online
async function syncReviews() {
  const db = await openDB();
  const tx = db.transaction('pendingReviews', 'readonly');
  const store = tx.objectStore('pendingReviews');
  const reviews = await store.getAll();
  
  for (const review of reviews) {
    try {
      await fetch('/api/reviews', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(review)
      });
      
      // Remove from pending after successful sync
      const deleteTx = db.transaction('pendingReviews', 'readwrite');
      deleteTx.objectStore('pendingReviews').delete(review.id);
    } catch (error) {
      console.error('Failed to sync review:', error);
    }
  }
}

// Sync check-ins when back online
async function syncCheckins() {
  const db = await openDB();
  const tx = db.transaction('pendingCheckins', 'readonly');
  const store = tx.objectStore('pendingCheckins');
  const checkins = await store.getAll();
  
  for (const checkin of checkins) {
    try {
      await fetch('/api/checkins', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(checkin)
      });
      
      const deleteTx = db.transaction('pendingCheckins', 'readwrite');
      deleteTx.objectStore('pendingCheckins').delete(checkin.id);
    } catch (error) {
      console.error('Failed to sync checkin:', error);
    }
  }
}

// Open IndexedDB
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('SideQuestDB', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      
      if (!db.objectStoreNames.contains('cachedPlaces')) {
        db.createObjectStore('cachedPlaces', { keyPath: 'id' });
      }
      
      if (!db.objectStoreNames.contains('pendingReviews')) {
        db.createObjectStore('pendingReviews', { keyPath: 'id' });
      }
      
      if (!db.objectStoreNames.contains('pendingCheckins')) {
        db.createObjectStore('pendingCheckins', { keyPath: 'id' });
      }
      
      if (!db.objectStoreNames.contains('userPreferences')) {
        db.createObjectStore('userPreferences', { keyPath: 'key' });
      }
    };
  });
}

// Get offline recommendations (cached data)
function getOfflineRecommendations(city) {
  // This would normally come from IndexedDB
  // For now, return mock offline data
  const offlineData = {
    'Delhi': [
      { name: 'Chai Wala Corner', type: 'Hidden Gem', rating: 4.3 },
      { name: 'Old City Walk', type: 'Experience', rating: 4.5 }
    ],
    'Mumbai': [
      { name: 'Art Studio Café', type: 'Hidden Gem', rating: 4.1 },
      { name: 'Marine Drive Sunset', type: 'Experience', rating: 4.7 }
    ],
    'Jaipur': [
      { name: 'Café Padasanai', type: 'Hidden Gem', rating: 4.2 },
      { name: 'Blue City Tour', type: 'Experience', rating: 4.6 }
    ]
  };
  
  return offlineData[city] || offlineData['Delhi'];
}

// Push notification handler
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  
  const options = {
    body: data.body || 'New hidden gem near you!',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1,
      url: data.url || '/'
    },
    actions: [
      { action: 'explore', title: 'Explore', icon: '/icons/explore.png' },
      { action: 'dismiss', title: 'Dismiss', icon: '/icons/dismiss.png' }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'SideQuest', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url)
    );
  }
});
