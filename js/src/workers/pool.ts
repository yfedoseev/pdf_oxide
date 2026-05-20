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
  // Per-worker busy flag, parallel to `workers`. The old scheduler
  // picked `activeCount % poolSize` which is a *global* counter and
  // does not identify which worker actually freed up — when tasks
  // finish out of order it could post a new task to a still-busy
  // worker (which then receives two messages and crosses results)
  // while an idle worker sat unused. `workerBusy[i] === false` is the
  // single source of truth for "worker i can accept a new task".
  // (#523 Copilot review.)
  private workerBusy: boolean[] = [];
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
        this.workerBusy.push(false);
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
    if (this.queue.length === 0) {
      return;
    }

    // Find the lowest-index idle worker. Scanning is O(poolSize) which
    // is bounded to 32 by `validatePoolSize`. Returning early when no
    // worker is free leaves the task on the queue; the next handler
    // completion will call back into `processQueue` and pick it up.
    const workerIndex = this.workerBusy.findIndex((b) => !b);
    if (workerIndex === -1) {
      return;
    }
    const worker = this.workers[workerIndex];
    if (!worker) {
      // Should be impossible — `workers` and `workerBusy` are kept in
      // lockstep — but fail safe rather than dereferencing undefined.
      return;
    }

    const queuedTask = this.queue.shift();
    if (!queuedTask) return;

    const { task, resolve, reject, timeout } = queuedTask;

    this.workerBusy[workerIndex] = true;
    this.activeCount++;

    // `once` for the message handler — Node's `worker.on('message')`
    // fires for every postMessage from that worker, so registering an
    // `on` listener and `off`-ing it inside the callback still leaves
    // a brief window where a second task posted to the same worker
    // would deliver its result to the wrong listener. `once` removes
    // the listener as soon as it fires, eliminating that window even
    // if a future caller (or refactor) ever overlaps tasks on the
    // same worker. The error handler is already `once`.
    const messageHandler = (result: WorkerResult<any>) => {
      clearTimeout(timeout);
      worker.off('error', errorHandler);
      this.workerBusy[workerIndex] = false;
      this.activeCount--;
      resolve(result as WorkerResult<any>);
      this.processQueue();
    };

    const errorHandler = (error: Error) => {
      clearTimeout(timeout);
      worker.off('message', messageHandler);
      this.workerBusy[workerIndex] = false;
      this.activeCount--;
      reject(error);
      this.processQueue();
    };

    worker.once('message', messageHandler);
    worker.once('error', errorHandler);

    try {
      worker.postMessage(task);
    } catch (error) {
      clearTimeout(timeout);
      worker.off('message', messageHandler);
      worker.off('error', errorHandler);
      this.workerBusy[workerIndex] = false;
      this.activeCount--;
      reject(error instanceof Error ? error : new Error(String(error)));
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
    this.workerBusy = [];
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
