/**
 * Worker Thread Pool Manager
 * Enables non-blocking parallel PDF processing
 */

import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { Worker } from 'worker_threads';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Represents a task to be processed by a worker
 */
export interface WorkerTask<T = any> {
  operation: 'extract' | 'search' | 'render' | 'analyze';
  documentPath: string;
  params: Record<string, any>;
}

/**
 * Result returned from a worker
 */
export interface WorkerResult<T = any> {
  success: boolean;
  data?: T;
  error?: Error | string;
  duration: number;
}

interface QueuedTask {
  task: WorkerTask<any>;
  resolve: (value: WorkerResult<any>) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
}

/**
 * Thread pool for parallel PDF processing.
 *
 * Workers are spawned **lazily** on the first {@link runTask} call — not
 * in the constructor. Merely importing the library (or using the
 * synchronous native APIs such as `extractText*`, `classifyPage`,
 * `prefetchModels`, which never touch the pool) spawns zero
 * `worker_threads`. Spawned workers are `unref()`'d so an idle/working
 * pool never keeps the event loop alive and process teardown
 * terminating them is not an abnormal exit (#521 — fixes spurious
 * "Worker N exited with code 1" on any short-lived consumer).
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private queue: QueuedTask[] = [];
  private activeCount = 0;
  private started = false;
  private terminated = false;
  private readonly defaultTimeout = 30000; // 30 seconds

  /**
   * Configure the worker pool. Does NOT spawn workers — they are
   * created lazily on first use (see {@link runTask}).
   * @param poolSize - Number of worker threads to create on first use
   */
  constructor(private poolSize: number = 4) {
    this.validatePoolSize();
  }

  private validatePoolSize(): void {
    if (this.poolSize < 1 || this.poolSize > 32) {
      throw new Error(`Pool size must be between 1 and 32, got ${this.poolSize}`);
    }
  }

  /** Spawn the worker threads on first real use (idempotent). */
  private ensureStarted(): void {
    if (this.started || this.terminated) return;
    this.started = true;
    try {
      for (let i = 0; i < this.poolSize; i++) {
        const worker = new Worker(path.join(__dirname, 'worker.js'));

        worker.on('error', (error: unknown) => {
          console.error(`Worker ${i} error:`, error);
          this.handleWorkerError(error instanceof Error ? error : new Error(String(error)));
        });

        worker.on('exit', (code) => {
          // Suppressed during intentional teardown (`terminated` is set
          // synchronously before workers are stopped); only a genuine
          // mid-run crash is warned about.
          if (code !== 0 && !this.terminated) {
            console.warn(`Worker ${i} exited with code ${code}`);
          }
        });

        // Do not keep the process alive just because a pooled worker
        // is idle; a normal process exit terminating an unref'd worker
        // is expected, not an error (#521).
        worker.unref();

        this.workers.push(worker);
      }
    } catch (error) {
      // Best-effort terminate any workers spawned before the failure:
      // `cleanup()` only drops references, so without this a partial
      // init would leak live (even if unref'd) worker threads.
      for (const worker of this.workers) {
        try {
          void worker.terminate();
        } catch {
          /* already gone / best-effort */
        }
      }
      this.cleanup();
      this.started = false;
      throw new Error(
        `Failed to initialize worker pool: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  /**
   * Run a task in the worker pool
   * @param task - The task to run
   * @param timeout - Optional timeout in milliseconds
   * @returns Promise that resolves with the result
   */
  public async runTask<T = any>(
    task: WorkerTask<T>,
    timeout: number = this.defaultTimeout
  ): Promise<WorkerResult<T>> {
    if (this.terminated) {
      throw new Error('Worker pool has been terminated');
    }

    if (timeout < 1000 || timeout > 300000) {
      throw new Error('Timeout must be between 1 and 300 seconds');
    }

    // Lazy spawn on first real task — keeps import + synchronous-native
    // call paths free of worker_threads entirely (#521).
    this.ensureStarted();

    return new Promise<WorkerResult<T>>((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        this.queue = this.queue.filter((q) => q.task !== task);
        reject(
          new Error(
            `Worker task timeout after ${timeout}ms: ${task.operation} on ${task.documentPath}`
          )
        );
      }, timeout);

      this.queue.push({
        task,
        resolve,
        reject,
        timeout: timeoutHandle,
      });

      this.processQueue();
    });
  }

  private processQueue(): void {
    if (this.queue.length === 0 || this.activeCount >= this.poolSize) {
      return;
    }

    const queuedTask = this.queue.shift();
    if (!queuedTask) return;

    const { task, resolve, reject, timeout } = queuedTask;

    // Find an available worker
    const workerIndex = this.activeCount % this.poolSize;
    const worker = this.workers[workerIndex];

    if (!worker) {
      reject(new Error('No available worker'));
      clearTimeout(timeout);
      return;
    }

    this.activeCount++;

    const messageHandler = (result: WorkerResult<any>) => {
      clearTimeout(timeout);
      resolve(result as WorkerResult<any>);
      this.activeCount--;
      worker.off('message', messageHandler);
      worker.off('error', errorHandler);
      this.processQueue();
    };

    const errorHandler = (error: Error) => {
      clearTimeout(timeout);
      reject(error);
      this.activeCount--;
      worker.off('message', messageHandler);
      worker.off('error', errorHandler);
      this.processQueue();
    };

    worker.on('message', messageHandler);
    worker.once('error', errorHandler);

    try {
      worker.postMessage(task);
    } catch (error) {
      clearTimeout(timeout);
      reject(error instanceof Error ? error : new Error(String(error)));
      this.activeCount--;
      worker.off('message', messageHandler);
      worker.off('error', errorHandler);
      this.processQueue();
    }
  }

  private handleWorkerError(error: Error): void {
    if (this.queue.length > 0) {
      const queuedTask = this.queue.shift();
      if (queuedTask) {
        clearTimeout(queuedTask.timeout);
        queuedTask.reject(error);
        this.activeCount--;
        this.processQueue();
      }
    }
  }

  /**
   * Terminate all workers
   * @returns Promise that resolves when all workers are terminated
   */
  public async terminate(): Promise<void> {
    // Set synchronously and first: the per-worker 'exit' handler keys
    // its warn off `!terminated`, so flipping this before stopping the
    // workers suppresses the spurious teardown warning (#521).
    this.terminated = true;

    // Reject all queued tasks
    while (this.queue.length > 0) {
      const queuedTask = this.queue.shift();
      if (queuedTask) {
        clearTimeout(queuedTask.timeout);
        queuedTask.reject(new Error('Worker pool terminated'));
      }
    }

    // Terminate all workers
    await Promise.all(
      this.workers.map((worker) =>
        worker.terminate().catch((error) => console.warn('Error terminating worker:', error))
      )
    );

    this.cleanup();
  }

  private cleanup(): void {
    this.workers = [];
    this.queue = [];
    this.activeCount = 0;
  }

  /**
   * Synchronously mark the pool terminated. Intended solely for the
   * process `'exit'` hook, which cannot run the async {@link terminate}.
   * Flipping this flag is what silences the per-worker teardown 'exit'
   * handler — exposed as a method so shutdown code never has to reach
   * into private state via an unsafe cast.
   */
  public markTerminatedForExit(): void {
    this.terminated = true;
  }

  /**
   * Get current pool statistics
   */
  public getStats(): {
    poolSize: number;
    activeWorkers: number;
    queuedTasks: number;
    terminated: boolean;
  } {
    return {
      poolSize: this.poolSize,
      activeWorkers: this.activeCount,
      queuedTasks: this.queue.length,
      terminated: this.terminated,
    };
  }
}

/**
 * Global worker pool instance (singleton).
 * Auto-configured based on CPU count. Construction is cheap — no
 * `worker_threads` are spawned until the pool is actually used (#521).
 */
const hardwareConcurrency = Math.max(1, os.cpus().length);

export const workerPool = new WorkerPool(Math.min(hardwareConcurrency, 8));

/**
 * Graceful shutdown — without hijacking the host's signal semantics.
 *
 * `process.on('exit')` runs synchronous code only, so it cannot await
 * `terminate()`. The async graceful terminate runs on `beforeExit`
 * (normal event-loop drain — the path that produced the spurious
 * "Worker N exited with code 1" for short-lived consumers such as
 * `prefetchModels()`); on the final hard `exit` we only flip the
 * synchronous `terminated` flag so any in-flight worker 'exit' events
 * stay silent.
 *
 * Deliberately NO `SIGINT`/`SIGTERM` listeners: registering them in a
 * library overrides Node's default "terminate on Ctrl-C / TERM" for
 * every consumer that merely imports this package — a breaking
 * operational change. Pooled workers are `unref()`'d (see
 * {@link WorkerPool}), so an abrupt signal teardown already exits
 * cleanly without us intercepting the signal.
 */
let shuttingDown = false;
function gracefulShutdown(): void {
  if (shuttingDown) return;
  shuttingDown = true;
  void workerPool.terminate().catch(() => {
    /* best-effort */
  });
}
process.once('beforeExit', gracefulShutdown);
process.on('exit', () => {
  // Sync only: ensure `terminated` is set so worker teardown is silent.
  workerPool.markTerminatedForExit();
});
