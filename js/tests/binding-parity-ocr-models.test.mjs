// Cross-binding API-parity tests for the Node binding: OCR model
// provisioning. Network-free: only the air-gapped manifest is
// asserted (no downloads). Skips gracefully when the lib isn't built.

import assert from 'node:assert/strict';
import { test } from 'node:test';

let mod;
try {
  mod = await import('../lib/index.js');
} catch {
  /* library not built — tests skip */
}
const skip = !mod?.Pdf;

// ── #519 OCR model provisioning ──────────────────────────────────────────────

test('#519 provisioning trio is exported and idiomatic', { skip }, () => {
  assert.equal(typeof mod.prefetchModels, 'function');
  assert.equal(typeof mod.modelManifest, 'function');
  assert.equal(typeof mod.prefetchAvailable, 'function');

  // Network-free: the manifest is static and always safe to call.
  const manifest = mod.modelManifest();
  assert.equal(typeof manifest, 'string');
  assert.ok(manifest.includes('det.onnx'), 'manifest must list the shared detector det.onnx');
  assert.ok(manifest.includes('english'), 'manifest must list the english recognition model');

  // prefetchAvailable() is a pure feature probe (no I/O).
  assert.equal(typeof mod.prefetchAvailable(), 'boolean');
});
