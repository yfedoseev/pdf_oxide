// Regression test for issue #648 — the documented Node.js quickstart must work,
// and the common misuse (`new PdfDocument(path)`) must fail with an actionable
// error instead of a cryptic native `invalid arguments` TypeError.
import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { PdfDocument } from '../lib/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = join(here, '..', '..', 'tests', 'fixtures', 'simple.pdf');

describe('README quickstart (#648)', () => {
  it('exposes the PdfDocument.open factory', () => {
    assert.strictEqual(typeof PdfDocument.open, 'function');
  });

  it('the documented happy path works: PdfDocument.open(path).extractText(0)', () => {
    const doc = PdfDocument.open(fixture);
    try {
      const text = doc.extractText(0);
      assert.strictEqual(typeof text, 'string');
    } finally {
      doc.close();
    }
  });

  it('new PdfDocument(path) throws an actionable error pointing at .open()', () => {
    assert.throws(
      () => new PdfDocument('report.pdf'),
      /Use PdfDocument\.open\(/,
      'constructor must reject a path string with a message that names PdfDocument.open'
    );
  });
});
