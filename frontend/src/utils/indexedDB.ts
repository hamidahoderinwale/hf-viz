/**
 * IndexedDB utility for client-side caching of embeddings and model data.
 * Enables offline access and faster subsequent loads.
 */

const DB_NAME = 'hf_viz_cache';
const DB_VERSION = 1;

interface CacheEntry<T> {
  key: string;
  data: T;
  timestamp: number;
  version: string;
}

const CACHE_VERSION = '1.0.0'; // Increment to invalidate old cache
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const MAX_CACHE_SIZE = 100; // Maximum number of entries per store

class IndexedDBCache {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  private async init(): Promise<void> {
    if (this.db) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create object stores
        if (!db.objectStoreNames.contains('embeddings')) {
          db.createObjectStore('embeddings', { keyPath: 'key' });
        }
        if (!db.objectStoreNames.contains('reduced_embeddings')) {
          db.createObjectStore('reduced_embeddings', { keyPath: 'key' });
        }
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'key' });
        }
        if (!db.objectStoreNames.contains('stats')) {
          db.createObjectStore('stats', { keyPath: 'key' });
        }
      };
    });

    return this.initPromise;
  }

  private async getStore(storeName: string): Promise<IDBObjectStore> {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');

    const transaction = this.db.transaction([storeName], 'readwrite');
    return transaction.objectStore(storeName);
  }

  async set<T>(storeName: string, key: string, data: T): Promise<void> {
    const store = await this.getStore(storeName);
    
    // Enforce cache size limit
    const count = await this.getCacheSize(storeName);
    if (count >= MAX_CACHE_SIZE) {
      // Delete oldest entries (simple FIFO - delete first entry)
      const transaction = store.transaction;
      const clearRequest = store.clear();
      await new Promise<void>((resolve, reject) => {
        clearRequest.onsuccess = () => resolve();
        clearRequest.onerror = () => reject(clearRequest.error);
      });
      // Re-get store after clear
      const newStore = await this.getStore(storeName);
      const entry: CacheEntry<T> = {
        key,
        data,
        timestamp: Date.now(),
        version: CACHE_VERSION,
      };
      return new Promise((resolve, reject) => {
        const request = newStore.put(entry);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    }
    
    const entry: CacheEntry<T> = {
      key,
      data,
      timestamp: Date.now(),
      version: CACHE_VERSION,
    };

    return new Promise((resolve, reject) => {
      const request = store.put(entry);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async get<T>(storeName: string, key: string): Promise<T | null> {
    const store = await this.getStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => {
        const entry = request.result as CacheEntry<T> | undefined;
        if (!entry) {
          resolve(null);
          return;
        }

        // Check version
        if (entry.version !== CACHE_VERSION) {
          // Version mismatch, delete old entry
          store.delete(key);
          resolve(null);
          return;
        }

        // Check TTL
        const age = Date.now() - entry.timestamp;
        if (age > CACHE_TTL_MS) {
          // Cache expired, delete old entry
          store.delete(key);
          resolve(null);
          return;
        }

        resolve(entry.data);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async has(storeName: string, key: string): Promise<boolean> {
    const store = await this.getStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.getKey(key);
      request.onsuccess = () => resolve(request.result !== undefined);
      request.onerror = () => reject(request.error);
    });
  }

  async delete(storeName: string, key: string): Promise<void> {
    const store = await this.getStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async clear(storeName: string): Promise<void> {
    const store = await this.getStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getCacheSize(storeName: string): Promise<number> {
    const store = await this.getStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.count();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Helper methods for specific data types
  async cacheModels(key: string, models: any[]): Promise<void> {
    return this.set('models', key, models);
  }

  async getCachedModels(key: string): Promise<any[] | null> {
    return this.get<any[]>('models', key);
  }

  async cacheStats(key: string, stats: any): Promise<void> {
    return this.set('stats', key, stats);
  }

  async getCachedStats(key: string): Promise<any | null> {
    return this.get<any>('stats', key);
  }

  // Generate cache key from filter parameters
  static generateCacheKey(params: {
    minDownloads: number;
    minLikes: number;
    searchQuery?: string;
    projectionMethod: string;
  }): string {
    return JSON.stringify({
      minDownloads: params.minDownloads,
      minLikes: params.minLikes,
      searchQuery: params.searchQuery || '',
      projectionMethod: params.projectionMethod,
    });
  }
}

// Export singleton instance
export const cache = new IndexedDBCache();
export { IndexedDBCache };
export default cache;

