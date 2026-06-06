// Core functional test-parity suite (WASM / Node) — mirrors the shared
// cross-language spec (docs/releases/plans/v0.3.61/core-test-parity-spec.md)
// with the idiomatic wasm-bindgen API. Every binding asserts the same
// behaviors.
//
// The WASM package (pdf_oxide.js + .wasm) is produced by wasm-pack in CI and
// is gitignored. If it has not been built, the whole suite self-skips rather
// than failing — matching the graceful-degradation contract used elsewhere.
import assert from 'node:assert';
import { describe, it, before } from 'node:test';

let WasmPdfDocument;
let WasmPdf;
let available = false;

before(async () => {
  try {
    const mod = await import('./pdf_oxide.js');
    WasmPdfDocument = mod.WasmPdfDocument;
    WasmPdf = mod.WasmPdf;
    available = typeof WasmPdfDocument === 'function' && typeof WasmPdf === 'function';
  } catch {
    available = false;
  }
});

function makeBytes() {
  const pdf = WasmPdf.fromText(
    'Core parity across all bindings.\nSecond line of text.',
    'Core Parity',
    'pdf_oxide',
  );
  return pdf.toBytes();
}

function open() {
  return new WasmPdfDocument(makeBytes());
}

describe('core parity (WASM)', () => {
  it('create pdf from text → %PDF', (t) => {
    if (!available) return t.skip('wasm package not built');
    const bytes = makeBytes();
    assert.ok(bytes.length > 4);
    assert.strictEqual(Buffer.from(bytes.subarray(0, 5)).toString('latin1'), '%PDF-');
  });

  it('open + page count == 1', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      assert.strictEqual(doc.pageCount(), 1);
    } finally {
      doc.free();
    }
  });

  it('extract text returns a string', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      assert.strictEqual(typeof doc.extractText(0), 'string');
    } finally {
      doc.free();
    }
  });

  it('convert markdown / html / plain return strings', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      assert.strictEqual(typeof doc.toMarkdown(0), 'string');
      assert.strictEqual(typeof doc.toHtml(0), 'string');
      assert.strictEqual(typeof doc.toPlainText(0), 'string');
    } finally {
      doc.free();
    }
  });

  it('search returns results without throwing', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      const res = doc.search('parity', true);
      assert.ok(res !== undefined && res !== null);
    } finally {
      doc.free();
    }
  });

  it('structured extraction works', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      assert.strictEqual(typeof doc.extractStructured(0), 'string');
    } finally {
      doc.free();
    }
  });

  it('exposes the PDF version', (t) => {
    if (!available) return t.skip('wasm package not built');
    const doc = open();
    try {
      const v = doc.version();
      assert.ok(v[0] >= 1);
    } finally {
      doc.free();
    }
  });
});
