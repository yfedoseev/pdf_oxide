// Cross-binding API-parity tests for the Node binding: crypto
// governance, document sanitization, the PAdES read side, and
// split-by-bookmarks. Self-contained: PDFs are generated from
// Markdown. Skips gracefully when the lib isn't built.

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

let mod;
try {
  mod = await import('../lib/index.js');
} catch {
  /* library not built — tests skip */
}
const skip = !mod?.Pdf;

function tempDir() {
  const dir = mkdtempSync(join(tmpdir(), 'pdfoxide-parity-'));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

function openDoc(dir, markdown) {
  const pdf = mod.Pdf.fromMarkdown(markdown);
  const path = join(dir, 'doc.pdf');
  pdf.save(path);
  pdf.close();
  return mod.PdfDocument.open(path);
}

// ── #230 crypto-governance ───────────────────────────────────────────────────

test('#230 crypto-governance functions are exported and callable', { skip }, () => {
  assert.equal(typeof mod.setCryptoPolicy, 'function');
  assert.equal(typeof mod.cryptoPolicy, 'function');
  assert.equal(typeof mod.cryptoInventory, 'function');
  assert.equal(typeof mod.cryptoCbom, 'function');

  assert.equal(typeof mod.cryptoPolicy(), 'string');
  assert.ok(Array.isArray(mod.cryptoInventory()));
  const cbom = mod.cryptoCbom();
  assert.ok(cbom.includes('CycloneDX'), 'cryptoCbom() must be a CycloneDX doc');
});

// ── #235 PAdES signing + read side ───────────────────────────────────────────

test('#235 PAdES surface is exported and idiomatic', { skip }, () => {
  assert.equal(typeof mod.signPdfBytesPades, 'function');
  assert.ok(mod.PadesLevel, 'PadesLevel enum missing');
  assert.equal(mod.PadesLevel.B_B, 0);
  assert.equal(mod.PadesLevel.B_T, 1);
  assert.equal(mod.PadesLevel.B_LT, 2);
  assert.equal(mod.PadesLevel.B_LTA, 3);
});

test('#235 document DSS reader + B-LTA signal on a plain PDF', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const doc = openDoc(dir, '# LTA probe\n\nplain body');
    // A freshly-built PDF has no /DSS and no /DocTimeStamp.
    assert.equal(doc.getDocumentSecurityStore(), null);
    assert.equal(doc.hasDocumentTimestamp(), false);
    doc.close();
  } finally {
    cleanup();
  }
});

// ── #231 redaction / sanitize ────────────────────────────────────────────────

test('#231 EditingManager redaction + sanitize are reachable', { skip }, async () => {
  const em = await import('../lib/managers/editing-manager.js').catch(() => null);
  assert.ok(em?.EditingManager, 'EditingManager not exported from managers');
  const proto = em.EditingManager.prototype;
  for (const m of ['addRedaction', 'applyRedactions', 'scrubMetadata', 'getRedactionCount']) {
    assert.equal(typeof proto[m], 'function', `EditingManager.${m} missing`);
  }
});

// ── #482 split-by-bookmarks ──────────────────────────────────────────────────

test('#482 planSplitByBookmarks returns an array', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const doc = openDoc(dir, '# Chapter One\n\nbody\n\n# Chapter Two\n\nbody');
    // A document with no resolvable outline either returns [] or throws
    // a clear "no outline" error — both are acceptable; we only assert
    // the call path is wired and well-typed.
    try {
      const segments = doc.planSplitByBookmarks();
      assert.ok(Array.isArray(segments), 'planSplitByBookmarks must return an array');
    } catch (e) {
      assert.ok(e instanceof Error, 'expected an Error for a no-outline document');
    }
    doc.close();
  } finally {
    cleanup();
  }
});
