// Core functional test-parity suite (Node) — mirrors the shared cross-language
// spec (docs/releases/plans/v0.3.61/core-test-parity-spec.md) with the idiomatic
// Node API.
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { Pdf, PdfDocument } from '../lib/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = join(here, '..', '..', 'tests', 'fixtures', 'simple.pdf');

const open = () => PdfDocument.open(fixture);

describe('core parity (Node)', () => {
  it('open + page count == 1', () => {
    const doc = open();
    try {
      assert.strictEqual(doc.getPageCount(), 1);
    } finally {
      doc.close();
    }
  });

  it('extract text returns a string', () => {
    const doc = open();
    try {
      assert.strictEqual(typeof doc.extractText(0), 'string');
    } finally {
      doc.close();
    }
  });

  it('convert markdown / html / plain return strings', () => {
    const doc = open();
    try {
      assert.strictEqual(typeof doc.toMarkdown(0), 'string');
      assert.strictEqual(typeof doc.toHtml(0), 'string');
      assert.strictEqual(typeof doc.toPlainText(0), 'string');
    } finally {
      doc.close();
    }
  });

  it('search returns results without throwing', () => {
    const doc = open();
    try {
      // searchAll is the idiomatic doc-level search in the Node binding
      // (mirrors Go's SearchAll / C#'s SearchAll in the parity spec).
      const res = doc.searchAll('the');
      assert.ok(res !== undefined && res !== null);
    } finally {
      doc.close();
    }
  });

  it('create pdf from text → %PDF', () => {
    const bytes = Pdf.fromText('Core parity across all bindings.').saveToBytes();
    assert.ok(bytes.length > 0);
    assert.strictEqual(bytes.subarray(0, 4).toString('latin1'), '%PDF');
  });

  it('open from buffer (in-memory bytes)', () => {
    const buf = readFileSync(fixture);
    const doc = PdfDocument.openFromBuffer(buf);
    try {
      assert.strictEqual(doc.getPageCount(), 1);
    } finally {
      doc.close();
    }
  });

  it('opening a missing path throws', () => {
    assert.throws(() => PdfDocument.open('/no/such/file/does/not/exist.pdf'));
  });
});
