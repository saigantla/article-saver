// IndexedDB Wrapper for Article Saver - Offline Support
// Provides simple promise-based API for storing articles locally

const DB_NAME = 'ArticleArchive';
const DB_VERSION = 1;

// Global database instance
let dbInstance = null;

// ============================================================================
// DATABASE INITIALIZATION
// ============================================================================

/**
 * Initialize IndexedDB - creates database and object stores
 * @returns {Promise<IDBDatabase>}
 */
function init() {
  return new Promise((resolve, reject) => {
    if (dbInstance) {
      resolve(dbInstance);
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      console.error('[DB] Failed to open database:', request.error);
      reject(request.error);
    };

    request.onsuccess = () => {
      dbInstance = request.result;
      console.log('[DB] Database opened successfully');
      resolve(dbInstance);
    };

    request.onupgradeneeded = (event) => {
      console.log('[DB] Upgrading database schema...');
      const db = event.target.result;

      // Create articles object store
      if (!db.objectStoreNames.contains('articles')) {
        const articlesStore = db.createObjectStore('articles', { keyPath: 'id' });

        // Create indexes for efficient querying
        articlesStore.createIndex('captured_at', 'captured_at', { unique: false });
        articlesStore.createIndex('source_domain', 'source_domain', { unique: false });

        console.log('[DB] Created articles object store');
      }

      // Create metadata object store
      if (!db.objectStoreNames.contains('metadata')) {
        db.createObjectStore('metadata', { keyPath: 'key' });
        console.log('[DB] Created metadata object store');
      }
    };
  });
}

// ============================================================================
// ARTICLE CRUD OPERATIONS
// ============================================================================

/**
 * Get all articles from IndexedDB
 * @returns {Promise<Array>} Array of article objects
 */
async function getAllArticles() {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readonly');
    const store = transaction.objectStore('articles');
    const request = store.getAll();

    request.onsuccess = () => {
      console.log(`[DB] Retrieved ${request.result.length} articles`);
      resolve(request.result);
    };

    request.onerror = () => {
      console.error('[DB] Failed to get articles:', request.error);
      reject(request.error);
    };
  });
}

/**
 * Get single article by ID
 * @param {number} id - Article ID
 * @returns {Promise<Object|null>} Article object or null
 */
async function getArticle(id) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readonly');
    const store = transaction.objectStore('articles');
    const request = store.get(id);

    request.onsuccess = () => {
      if (request.result) {
        console.log(`[DB] Retrieved article ${id}`);
      } else {
        console.log(`[DB] Article ${id} not found`);
      }
      resolve(request.result || null);
    };

    request.onerror = () => {
      console.error(`[DB] Failed to get article ${id}:`, request.error);
      reject(request.error);
    };
  });
}

/**
 * Save single article to IndexedDB
 * @param {Object} article - Article object with id, title, content, etc.
 * @returns {Promise<void>}
 */
async function saveArticle(article) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readwrite');
    const store = transaction.objectStore('articles');

    // Add cached_at timestamp
    const articleToSave = {
      ...article,
      cached_at: Date.now()
    };

    const request = store.put(articleToSave);

    request.onsuccess = () => {
      console.log(`[DB] Saved article ${article.id}`);
      resolve();
    };

    request.onerror = () => {
      console.error(`[DB] Failed to save article ${article.id}:`, request.error);
      reject(request.error);
    };
  });
}

/**
 * Save multiple articles to IndexedDB (batch operation)
 * @param {Array} articles - Array of article objects
 * @returns {Promise<void>}
 */
async function saveArticles(articles) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readwrite');
    const store = transaction.objectStore('articles');

    const timestamp = Date.now();

    // Add all articles to the transaction
    articles.forEach(article => {
      const articleToSave = {
        ...article,
        cached_at: timestamp
      };
      store.put(articleToSave);
    });

    transaction.oncomplete = () => {
      console.log(`[DB] Saved ${articles.length} articles`);
      resolve();
    };

    transaction.onerror = () => {
      console.error('[DB] Failed to save articles:', transaction.error);
      reject(transaction.error);
    };
  });
}

/**
 * Delete article from IndexedDB
 * @param {number} id - Article ID
 * @returns {Promise<void>}
 */
async function deleteArticle(id) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readwrite');
    const store = transaction.objectStore('articles');
    const request = store.delete(id);

    request.onsuccess = () => {
      console.log(`[DB] Deleted article ${id}`);
      resolve();
    };

    request.onerror = () => {
      console.error(`[DB] Failed to delete article ${id}:`, request.error);
      reject(request.error);
    };
  });
}

/**
 * Check if article exists in IndexedDB
 * @param {number} id - Article ID
 * @returns {Promise<boolean>}
 */
async function hasArticle(id) {
  const article = await getArticle(id);
  return article !== null;
}

/**
 * Clear all articles from IndexedDB
 * @returns {Promise<void>}
 */
async function clearAllArticles() {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readwrite');
    const store = transaction.objectStore('articles');
    const request = store.clear();

    request.onsuccess = () => {
      console.log('[DB] Cleared all articles');
      resolve();
    };

    request.onerror = () => {
      console.error('[DB] Failed to clear articles:', request.error);
      reject(request.error);
    };
  });
}

// ============================================================================
// METADATA OPERATIONS
// ============================================================================

/**
 * Get metadata value by key
 * @param {string} key - Metadata key
 * @returns {Promise<any>} Metadata value or null
 */
async function getMetadata(key) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['metadata'], 'readonly');
    const store = transaction.objectStore('metadata');
    const request = store.get(key);

    request.onsuccess = () => {
      if (request.result) {
        console.log(`[DB] Retrieved metadata: ${key}`);
        resolve(request.result.value);
      } else {
        console.log(`[DB] Metadata not found: ${key}`);
        resolve(null);
      }
    };

    request.onerror = () => {
      console.error(`[DB] Failed to get metadata ${key}:`, request.error);
      reject(request.error);
    };
  });
}

/**
 * Set metadata value
 * @param {string} key - Metadata key
 * @param {any} value - Metadata value
 * @returns {Promise<void>}
 */
async function setMetadata(key, value) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['metadata'], 'readwrite');
    const store = transaction.objectStore('metadata');
    const request = store.put({ key, value });

    request.onsuccess = () => {
      console.log(`[DB] Set metadata: ${key}`);
      resolve();
    };

    request.onerror = () => {
      console.error(`[DB] Failed to set metadata ${key}:`, request.error);
      reject(request.error);
    };
  });
}

/**
 * Delete metadata by key
 * @param {string} key - Metadata key
 * @returns {Promise<void>}
 */
async function deleteMetadata(key) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['metadata'], 'readwrite');
    const store = transaction.objectStore('metadata');
    const request = store.delete(key);

    request.onsuccess = () => {
      console.log(`[DB] Deleted metadata: ${key}`);
      resolve();
    };

    request.onerror = () => {
      console.error(`[DB] Failed to delete metadata ${key}:`, request.error);
      reject(request.error);
    };
  });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get database statistics
 * @returns {Promise<Object>} Stats object with article count, storage size, etc.
 */
async function getStats() {
  const db = await init();

  return new Promise(async (resolve, reject) => {
    try {
      const transaction = db.transaction(['articles'], 'readonly');
      const store = transaction.objectStore('articles');
      const countRequest = store.count();

      countRequest.onsuccess = async () => {
        const stats = {
          articleCount: countRequest.result,
          lastSync: await getMetadata('last_sync'),
          dbName: DB_NAME,
          dbVersion: DB_VERSION
        };

        console.log('[DB] Database stats:', stats);
        resolve(stats);
      };

      countRequest.onerror = () => {
        reject(countRequest.error);
      };
    } catch (error) {
      reject(error);
    }
  });
}

/**
 * Delete oldest articles to free up space
 * @param {number} count - Number of oldest articles to delete
 * @returns {Promise<number>} Number of articles deleted
 */
async function deleteOldestArticles(count) {
  const db = await init();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['articles'], 'readwrite');
    const store = transaction.objectStore('articles');
    const index = store.index('captured_at');
    const request = index.openCursor();

    let deleted = 0;

    request.onsuccess = (event) => {
      const cursor = event.target.result;

      if (cursor && deleted < count) {
        cursor.delete();
        deleted++;
        cursor.continue();
      } else {
        console.log(`[DB] Deleted ${deleted} oldest articles`);
        resolve(deleted);
      }
    };

    request.onerror = () => {
      console.error('[DB] Failed to delete oldest articles:', request.error);
      reject(request.error);
    };
  });
}

// ============================================================================
// EXPORT PUBLIC API
// ============================================================================

window.db = {
  // Core operations
  init,

  // Article CRUD
  getAllArticles,
  getArticle,
  saveArticle,
  saveArticles,
  deleteArticle,
  hasArticle,
  clearAllArticles,

  // Metadata
  getMetadata,
  setMetadata,
  deleteMetadata,

  // Utilities
  getStats,
  deleteOldestArticles
};

console.log('[DB] IndexedDB wrapper loaded');
