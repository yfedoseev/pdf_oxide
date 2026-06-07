// TDD tests for Pdf.fromHtmlCss / Pdf.fromHtmlCssWithFonts
// These functions exist in binding.cc (pdfFromHtmlCss / pdfFromHtmlCssWithFonts)
// but were never wrapped in the TypeScript layer. This file defines the expected
// API before the implementation is added.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const __dir = dirname(fileURLToPath(import.meta.url));

// Minimal font for embedding — use DejaVuSans from fixtures if available,
// otherwise skip font-cascade tests.
async function loadFont() {
  const candidates = [
    // Git-tracked font — available on every OS runner (ubuntu/macOS/windows).
    join(__dir, '../../tests/fixtures/fonts/DejaVuSans.ttf'),
    join(__dir, '../../tools/benchmark-harness/fixtures/fonts/DejaVuSans.ttf'),
    join(__dir, '../fixtures/DejaVuSans.ttf'),
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
  ];
  for (const p of candidates) {
    try {
      return await readFile(p);
    } catch {
      // try next
    }
  }
  return null;
}

let Pdf;
try {
  ({ Pdf } = await import('../lib/index.js'));
} catch {
  // library not built yet — all tests below will be skipped
}

function isPdf(buf) {
  return (
    buf &&
    buf.length > 4 &&
    buf[0] === 0x25 &&
    buf[1] === 0x50 &&
    buf[2] === 0x44 &&
    buf[3] === 0x46
  );
}

test('Pdf.fromHtmlCss is exported', { skip: !Pdf }, () => {
  assert.strictEqual(typeof Pdf.fromHtmlCss, 'function');
});

test('Pdf.fromHtmlCssWithFonts is exported', { skip: !Pdf }, () => {
  assert.strictEqual(typeof Pdf.fromHtmlCssWithFonts, 'function');
});

test('fromHtmlCss produces a valid PDF', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file to test HTML+CSS → PDF');

  const html = '<html><body><h1>Hello CSS</h1><p>World</p></body></html>';
  const css = 'body { font-size: 14px; } h1 { color: black; }';

  const pdf = Pdf.fromHtmlCss(html, css, font);
  const bytes = pdf.saveToBytes();
  assert.ok(isPdf(bytes), 'output should start with %PDF-');
  assert.ok(bytes.length > 200, 'output should be a non-trivial PDF');
  pdf.close();
});

test('fromHtmlCss returns a Pdf with a positive byte count', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');

  const pdf = Pdf.fromHtmlCss('<p>test</p>', 'p { font-size: 12px; }', font);
  const bytes = pdf.saveToBytes();
  assert.ok(bytes.length > 0);
  pdf.close();
});

test('fromHtmlCssWithFonts produces a valid PDF', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');

  const html = '<p>Multi-font</p>';
  const css = 'p { font-family: Body; }';
  const pdf = Pdf.fromHtmlCssWithFonts(html, css, ['Body'], [font]);
  const bytes = pdf.saveToBytes();
  assert.ok(isPdf(bytes));
  pdf.close();
});

test('fromHtmlCss throws on null html', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font);
  assert.throws(() => Pdf.fromHtmlCss(null, '', font));
});

test('fromHtmlCss throws on null font bytes', { skip: !Pdf }, () => {
  assert.throws(() => Pdf.fromHtmlCss('<p>hi</p>', '', null));
});

test('fromHtmlCssWithFonts throws when families/fonts arrays length mismatch', {
  skip: !Pdf,
}, async () => {
  const font = await loadFont();
  assert.ok(font);
  assert.throws(() => Pdf.fromHtmlCssWithFonts('<p>hi</p>', '', ['A', 'B'], [font]));
});

// ── CSS property correctness ───────────────────────────────────────────────
// Each test generates two PDFs that differ only in one CSS property and
// asserts the byte output is different — proving the property is applied.

test('CSS font-size changes output bytes', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');
  const html = '<p>text</p>';
  const small = Pdf.fromHtmlCss(html, 'p { font-size: 12px; }', font);
  const large = Pdf.fromHtmlCss(html, 'p { font-size: 48px; }', font);
  const a = small.saveToBytes();
  const b = large.saveToBytes();
  small.close();
  large.close();
  assert.notDeepStrictEqual(a, b, 'CSS font-size had no effect on output');
});

test('CSS color changes output bytes', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');
  const html = '<p>text</p>';
  const black = Pdf.fromHtmlCss(html, 'p { color: black; }', font);
  const red = Pdf.fromHtmlCss(html, 'p { color: red; }', font);
  const a = black.saveToBytes();
  const b = red.saveToBytes();
  black.close();
  red.close();
  assert.notDeepStrictEqual(a, b, 'CSS color had no effect on output');
});

test('CSS background-color changes output bytes', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');
  const html = '<p>text</p>';
  const none = Pdf.fromHtmlCss(html, '', font);
  const yellow = Pdf.fromHtmlCss(html, 'body { background-color: yellow; }', font);
  const a = none.saveToBytes();
  const b = yellow.saveToBytes();
  none.close();
  yellow.close();
  assert.notDeepStrictEqual(a, b, 'CSS background-color had no effect on output');
});

test('CSS text-decoration underline changes output bytes', { skip: !Pdf }, async () => {
  const font = await loadFont();
  assert.ok(font, 'need a font file');
  const html = '<p>text</p>';
  const none = Pdf.fromHtmlCss(html, '', font);
  const underline = Pdf.fromHtmlCss(html, 'p { text-decoration: underline; }', font);
  const a = none.saveToBytes();
  const b = underline.saveToBytes();
  none.close();
  underline.close();
  assert.notDeepStrictEqual(a, b, 'CSS text-decoration had no effect on output');
});
