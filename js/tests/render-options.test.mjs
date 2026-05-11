// Tests for renderPageWithOptions + estimateRenderTime TS wrappers
// over the pdf_render_page_with_options N-API export.

import assert from 'node:assert/strict';
import { test } from 'node:test';

let Pdf, PdfDocument;
try {
  ({ Pdf, PdfDocument } = await import('../lib/index.js'));
} catch {
  // library not built — all tests will be skipped
}

const skip = !Pdf;

function makeDoc() {
  const bytes = Pdf.fromMarkdown('# Render\n\nBody.').saveToBytes();
  return PdfDocument.openFromBuffer(Buffer.from(bytes));
}

function isPng(b) {
  return b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
}

function isJpeg(b) {
  return b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
}

test('renderPageWithOptions defaults produce PNG bytes', { skip }, () => {
  const doc = makeDoc();
  const bytes = doc.renderPageWithOptions(0);
  assert.ok(isPng(bytes), 'default format should be PNG');
  assert.ok(bytes.length > 128);
});

test('renderPageWithOptions with format=jpeg emits JPEG', { skip }, () => {
  const doc = makeDoc();
  const bytes = doc.renderPageWithOptions(0, { format: 'jpeg' });
  assert.ok(isJpeg(bytes));
});

test('renderPageWithOptions higher DPI → more bytes', { skip }, () => {
  const doc = makeDoc();
  const small = doc.renderPageWithOptions(0, { dpi: 72 });
  const large = doc.renderPageWithOptions(0, { dpi: 300 });
  assert.ok(isPng(small) && isPng(large));
  assert.ok(large.length > small.length);
});

test('renderPageWithOptions transparentBackground still PNG', { skip }, () => {
  const doc = makeDoc();
  const bytes = doc.renderPageWithOptions(0, { transparentBackground: true });
  assert.ok(isPng(bytes));
});

test('renderPageWithOptions RGB background accepted', { skip }, () => {
  const doc = makeDoc();
  const bytes = doc.renderPageWithOptions(0, { background: [0.2, 0.2, 0.2, 1] });
  assert.ok(isPng(bytes));
});

test('renderPageWithOptions renderAnnotations=false accepted', { skip }, () => {
  const doc = makeDoc();
  const bytes = doc.renderPageWithOptions(0, { renderAnnotations: false });
  assert.ok(isPng(bytes));
});

test('renderPageWithOptions rejects invalid dpi', { skip }, () => {
  const doc = makeDoc();
  assert.throws(() => doc.renderPageWithOptions(0, { dpi: 0 }), /dpi/);
});

test('renderPageWithOptions rejects invalid jpegQuality', { skip }, () => {
  const doc = makeDoc();
  assert.throws(
    () => doc.renderPageWithOptions(0, { format: 'jpeg', jpegQuality: 0 }),
    /jpegQuality/
  );
});

test('estimateRenderTime returns a non-negative number', { skip }, () => {
  const doc = makeDoc();
  const ms = doc.estimateRenderTime(0, 150);
  assert.equal(typeof ms, 'number');
  assert.ok(ms >= 0);
});

// Raw RGBA pixel buffer tests (issue #446)

test('renderToPixmap returns premultiplied RGBA buffer', { skip }, () => {
  const doc = makeDoc();
  const px = doc.renderToPixmap(0, 72);
  assert.ok(px.width > 0, 'width > 0');
  assert.ok(px.height > 0, 'height > 0');
  assert.strictEqual(px.data.length, px.width * px.height * 4, 'data length = w*h*4');
});

test('renderToPixmap data is not PNG-encoded', { skip }, () => {
  const doc = makeDoc();
  const px = doc.renderToPixmap(0, 72);
  // Full 8-byte PNG signature: avoids false positives from pixel data that
  // happen to share the first byte (0x89) with the PNG magic.
  const pngSig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.ok(!px.data.slice(0, 8).equals(pngSig), 'raw RGBA must not start with PNG magic bytes');
});

test('renderToPixmap dimensions match renderPageWithOptions at same DPI', { skip }, () => {
  const doc = makeDoc();
  const pngBytes = doc.renderPageWithOptions(0, { dpi: 72 });
  const px = doc.renderToPixmap(0, 72);
  // PNG IHDR: width at bytes 16-19, height at 20-23 (big-endian)
  const pngW = pngBytes.readUInt32BE(16);
  const pngH = pngBytes.readUInt32BE(20);
  assert.strictEqual(px.width, pngW, 'width must match PNG IHDR');
  assert.strictEqual(px.height, pngH, 'height must match PNG IHDR');
});

test('renderToPixmap rejects invalid dpi', { skip }, () => {
  const doc = makeDoc();
  assert.throws(() => doc.renderToPixmap(0, 0), /dpi/);
});

// Async render variants (issue #481)

test('renderPageWithOptionsAsync returns PNG bytes', { skip }, async () => {
  const doc = makeDoc();
  const bytes = await doc.renderPageWithOptionsAsync(0);
  assert.ok(isPng(bytes), 'async default format should be PNG');
});

test('renderPageFitAsync returns PNG bytes', { skip }, async () => {
  const doc = makeDoc();
  const bytes = await doc.renderPageFitAsync(0, 400, 600);
  assert.ok(isPng(bytes), 'renderPageFitAsync should produce PNG');
});

test('renderToPixmapAsync — same doc concurrent renders produce valid results', {
  skip,
}, async () => {
  const doc = makeDoc();
  const results = await Promise.all(
    Array.from({ length: 4 }, () => doc.renderToPixmapAsync(0, 72))
  );
  for (const px of results) {
    assert.ok(px.width > 0 && px.height > 0, 'valid dimensions');
    assert.strictEqual(px.data.length, px.width * px.height * 4, 'correct data length');
  }
});
