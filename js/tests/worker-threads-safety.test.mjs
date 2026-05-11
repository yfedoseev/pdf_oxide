// Tests that concurrent worker_threads each opening their own PdfDocument handle
// (from the same serialised bytes) do not crash or corrupt each other's state.
//
// Each worker re-opens its own document instance; this exercises the per-document
// Mutex path (src/ffi.rs exclusive-ownership contract) under parallel load without
// requiring cross-thread handle sharing, which Node worker_threads do not support.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';

const __filename = fileURLToPath(import.meta.url);

// ── Worker body ───────────────────────────────────────────────────────────────
// When loaded as a worker, run N extract-text calls on its own handle.
if (!isMainThread) {
  const { handle, iterations } = workerData;
  let mod;
  try {
    mod = await import('../lib/index.js');
  } catch {
    parentPort.postMessage({ ok: false, error: 'library not built' });
    process.exit(0);
  }

  // Re-open from the bytes that were serialised into workerData so each
  // worker gets its own handle copy (sharing handles across workers is valid
  // usage but re-opening is simpler and still exercises the mutex path when
  // multiple workers call methods simultaneously on their own handles).
  const doc = mod.PdfDocument.openFromBuffer(Buffer.from(handle));
  let errors = 0;
  for (let i = 0; i < iterations; i++) {
    try {
      const text = doc.extractText(0);
      if (typeof text !== 'string') errors++;
    } catch {
      errors++;
    }
  }
  doc.close();
  parentPort.postMessage({ ok: errors === 0, errors });
  process.exit(0);
}

// ── Main thread helpers ───────────────────────────────────────────────────────

let Pdf, PdfDocument;
const libAvailable = await (async () => {
  try {
    ({ Pdf, PdfDocument } = await import('../lib/index.js'));
    return true;
  } catch {
    return false;
  }
})();

const skip = !libAvailable;

function makeDocBytes() {
  const bytes = Pdf.fromMarkdown('# Thread Safety\n\nConcurrency test.').saveToBytes();
  return bytes; // Buffer
}

function spawnWorker(handle, iterations) {
  return new Promise((resolve, reject) => {
    const w = new Worker(__filename, {
      workerData: { handle: Array.from(handle), iterations },
    });
    w.on('message', resolve);
    w.on('error', reject);
    w.on('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
    });
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test('concurrent workers each open and query the same PDF bytes without errors', {
  skip,
}, async () => {
  const docBytes = makeDocBytes();
  const WORKERS = 8;
  const ITERATIONS = 20;

  const results = await Promise.all(
    Array.from({ length: WORKERS }, () => spawnWorker(docBytes, ITERATIONS))
  );

  for (const r of results) {
    assert.ok(r.ok, `Worker reported ${r.errors} errors`);
  }
});

test('single-threaded repeated calls still work after mutex changes', { skip }, () => {
  const doc = PdfDocument.openFromBuffer(makeDocBytes());
  try {
    for (let i = 0; i < 50; i++) {
      const text = doc.extractText(0);
      assert.equal(typeof text, 'string');
    }
    const count = doc.pageCount();
    assert.ok(count > 0);
  } finally {
    doc.close();
  }
});

test('close then re-open does not crash', { skip }, () => {
  const bytes = makeDocBytes();
  const doc = PdfDocument.openFromBuffer(bytes);
  doc.close();
  // Re-open from the same bytes
  const doc2 = PdfDocument.openFromBuffer(bytes);
  assert.ok(doc2.pageCount() > 0);
  doc2.close();
});
