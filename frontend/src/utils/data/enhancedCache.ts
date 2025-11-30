/**
 * Enhanced IndexedDB cache with TTL and automatic cleanup.
 * Uses idb library for better reliability and performance.
 */
import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface CacheEntry {
  key: string;
  data: any;
  timestamp: number;
  ttl: number;
  size?: number;
}

interface CacheDB extends DBSchema {
  cache: {
    key: string;
    value: CacheEntry;
    indexes: { 'by-timestamp': number };
  };
}

class EnhancedIndexedDBCache {
  private db: IDBPDatabase<CacheDB> | null = null;
  private dbName = 'hfviz-cache-v2';
  private readonly storeName: 'cache' = 'cache';
  private maxSize = 100 * 1024 * 1024; // 100MB
  private defaultTTL = 5 * 60 * 1000; // 5 minutes

  async init(): Promise<void> {
    try {
      this.db = await openDB<CacheDB>(this.dbName, 2, {
        upgrade(db, oldVersion) {
          if (!db.objectStoreNames.contains('cache')) {
            const store = db.createObjectStore('cache', { keyPath: 'key' });
            store.createIndex('by-timestamp', 'timestamp');
          }
        },
      });

      // Cleanup on init
      await this.cleanup();
    } catch (err) {
      console.error('Failed to initialize IndexedDB:', err);
    }
  }

  async get<T = any>(key: string): Promise<T | null> {
    if (!this.db) await this.init();
    if (!this.db) return null;

    try {
      const entry = await this.db.get(this.storeName, key);
      
      if (!entry) return null;

      // Check if expired
      if (Date.now() - entry.timestamp > entry.ttl) {
        await this.delete(key);
        return null;
      }

      return entry.data as T;
    } catch (err) {
      console.error('IndexedDB get error:', err);
      return null;
    }
  }

  async set(key: string, data: any, ttl: number = this.defaultTTL): Promise<boolean> {
    if (!this.db) await this.init();
    if (!this.db) return false;

    try {
      // Estimate size
      const size = new Blob([JSON.stringify(data)]).size;

      // Check if we need to make space
      await this.ensureSpace(size);

      const entry: CacheEntry = {
        key,
        data,
        timestamp: Date.now(),
        ttl,
        size,
      };

      await this.db.put(this.storeName, entry);
      return true;
    } catch (err) {
      console.error('IndexedDB set error:', err);
      return false;
    }
  }

  async delete(key: string): Promise<boolean> {
    if (!this.db) await this.init();
    if (!this.db) return false;

    try {
      await this.db.delete(this.storeName, key);
      return true;
    } catch (err) {
      console.error('IndexedDB delete error:', err);
      return false;
    }
  }

  async clear(): Promise<boolean> {
    if (!this.db) await this.init();
    if (!this.db) return false;

    try {
      await this.db.clear(this.storeName);
      return true;
    } catch (err) {
      console.error('IndexedDB clear error:', err);
      return false;
    }
  }

  async cleanup(): Promise<void> {
    if (!this.db) return;

    try {
      const tx = this.db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const now = Date.now();

      let cursor = await store.openCursor();
      while (cursor) {
        const entry = cursor.value;
        if (now - entry.timestamp > entry.ttl) {
          await cursor.delete();
        }
        cursor = await cursor.continue();
      }

      await tx.done;
    } catch (err) {
      console.error('IndexedDB cleanup error:', err);
    }
  }

  async ensureSpace(requiredSize: number): Promise<void> {
    if (!this.db) return;

    try {
      // Get all entries sorted by timestamp
      const tx = this.db.transaction(this.storeName, 'readwrite');
      const index = tx.objectStore(this.storeName).index('by-timestamp');
      const entries: CacheEntry[] = [];

      let cursor = await index.openCursor();
      while (cursor) {
        entries.push(cursor.value);
        cursor = await cursor.continue();
      }

      await tx.done;

      // Calculate total size
      let totalSize = entries.reduce((sum, e) => sum + (e.size || 0), 0);

      // Remove oldest entries if needed
      if (totalSize + requiredSize > this.maxSize) {
        const sorted = entries.sort((a, b) => a.timestamp - b.timestamp);
        
        for (const entry of sorted) {
          if (totalSize + requiredSize <= this.maxSize) break;
          await this.delete(entry.key);
          totalSize -= (entry.size || 0);
        }
      }
    } catch (err) {
      console.error('IndexedDB ensureSpace error:', err);
    }
  }

  async stats(): Promise<{ count: number; totalSize: number }> {
    if (!this.db) await this.init();
    if (!this.db) return { count: 0, totalSize: 0 };

    try {
      const tx = this.db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      
      let count = 0;
      let totalSize = 0;
      
      let cursor = await store.openCursor();
      while (cursor) {
        count++;
        totalSize += cursor.value.size || 0;
        cursor = await cursor.continue();
      }

      await tx.done;
      return { count, totalSize };
    } catch (err) {
      console.error('IndexedDB stats error:', err);
      return { count: 0, totalSize: 0 };
    }
  }
}

// Export singleton
export const enhancedCache = new EnhancedIndexedDBCache();

// Auto-cleanup every 5 minutes
if (typeof window !== 'undefined') {
  setInterval(() => {
    enhancedCache.cleanup();
  }, 5 * 60 * 1000);
}

export default enhancedCache;


