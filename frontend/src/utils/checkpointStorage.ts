import type { Checkpoint } from "../types/Checkpoint";

class CheckpointStorage {
  private dbName = "ConversationCheckpoints";
  private dbVersion = 1;
  private storeName = "checkpoints";
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        // create object store
        // if it doesn't exist
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, {
            keyPath: "threadId",
          });
          // create index on checkpoint
          // timestamp for further sorting
          // and filtering
          store.createIndex("timestamp", "timestamp", { unique: false });
        }
      };
    });
  }

  private async ensureDB(): Promise<IDBDatabase> {
    if (!this.db) {
      await this.init();
    }
    return this.db!;
  }

  createCheckpoint(
    threadId: string,
    userId: string,
    title: string,
    checkpointData: any,
  ): Checkpoint {
    return {
      threadId,
      userId,
      title,
      data: checkpointData,
      timestamp: Date.now(),
    };
  }

  async storeCheckpoint(checkpoint: Checkpoint): Promise<void> {
    const db = await this.ensureDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      // put overwrites as only
      // a single checkpoint is
      // stored per conversation
      const request = store.put(checkpoint);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async getCheckpoint(threadId: string): Promise<Checkpoint | null> {
    const db = await this.ensureDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.get(threadId);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        resolve(request.result || null);
      };
    });
  }

  async getAllCheckpoints(): Promise<Checkpoint[]> {
    const db = await this.ensureDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        resolve(request.result || []);
      };
    });
  }

  async getAllCheckpointsSorted(
    filter?: (checkpoint: Checkpoint) => boolean,
  ): Promise<Checkpoint[]> {
    const db = await this.ensureDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const index = store.index("timestamp");
      const checkpoints: Checkpoint[] = [];
      // the indexed timestamp is
      // used to properly traverse
      // the checkpoints from latest
      const request = index.openCursor(null, "prev");

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor) {
          const value = cursor.value as Checkpoint;
          if (!filter || filter(value)) {
            checkpoints.push(value);
          }
          cursor.continue();
        } else {
          resolve(checkpoints);
        }
      };
    });
  }

  async deleteCheckpoint(threadId: string): Promise<void> {
    const db = await this.ensureDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(threadId);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }
}

// export singleton instance
export const checkpointStorage = new CheckpointStorage();
