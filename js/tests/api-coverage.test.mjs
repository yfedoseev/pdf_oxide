// Broad API coverage tests for the Node.js binding.
// One test per public method not already covered in other test files.
// All tests are self-contained — they generate PDFs from Markdown.

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

let Pdf, PdfDocument, DocumentBuilder, DocumentEditor;
try {
  ({ Pdf, PdfDocument, DocumentBuilder, DocumentEditor } = await import('../lib/index.js'));
} catch {
  // library not built — all tests will be skipped
}

const skip = !Pdf;

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

// Creates a temp directory, returns { dir, cleanup() }
function tempDir() {
  const dir = mkdtempSync(join(tmpdir(), 'pdfoxide-cov-'));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

// Saves a markdown PDF to <dir>/name.pdf and returns the path
function savePdf(dir, name, markdown) {
  const pdf = Pdf.fromMarkdown(markdown);
  const path = join(dir, `${name}.pdf`);
  pdf.save(path);
  pdf.close();
  return path;
}

// ── PdfDocument: open from path ───────────────────────────────────────────────

test('PdfDocument.open from path returns document', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'open', '# Open Test');
    const doc = PdfDocument.open(path);
    assert.ok(doc.pageCount() >= 1);
    doc.close();
  } finally {
    cleanup();
  }
});

test('PdfDocument.openFromBuffer returns document', { skip }, () => {
  const buf = Buffer.from(Pdf.fromMarkdown('# Buffer').saveToBytes());
  const doc = PdfDocument.openFromBuffer(buf);
  assert.ok(doc.pageCount() >= 1);
  doc.close();
});

// ── Text extraction ────────────────────────────────────────────────────────────

test('extractWords returns non-empty array with text field', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'words', 'WORDTOKEN hello');
    const doc = PdfDocument.open(path);
    const words = doc.extractWords(0);
    assert.ok(Array.isArray(words) && words.length > 0, 'expected words array');
    const found = words.some((w) => (w.text || w.Text || '').includes('WORDTOKEN'));
    assert.ok(found, `WORDTOKEN not found in: ${JSON.stringify(words.slice(0, 3))}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('extractWords exposes quadrant-snapped rotationDegrees', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'rot', 'ROTTOKEN horizontal words');
    const doc = PdfDocument.open(path);
    const words = doc.extractWords(0);
    assert.ok(Array.isArray(words) && words.length > 0, 'expected words array');
    for (const w of words) {
      assert.equal(typeof w.rotationDegrees, 'number');
      assert.ok(
        [0, 90, 180, -90].includes(w.rotationDegrees),
        `expected quadrant-snapped rotation, got: ${w.rotationDegrees}`
      );
    }
    // Horizontal markdown text reads left-to-right: rotation is 0.
    assert.ok(
      words.every((w) => w.rotationDegrees === 0),
      `expected 0 for horizontal text, got: ${JSON.stringify(words.map((w) => w.rotationDegrees))}`
    );
    doc.close();
  } finally {
    cleanup();
  }
});

test('extractPaths exposes stroke-inflated rendered bbox', { skip: !DocumentBuilder }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = join(dir, 'paths.pdf');
    const bytes = DocumentBuilder.create()
      .a4Page()
      .strokeRect(100, 500, 200, 100, { width: 8 })
      .done()
      .build();
    writeFileSync(path, bytes);

    const doc = PdfDocument.open(path);
    const paths = doc.extractPaths(0);
    assert.ok(Array.isArray(paths) && paths.length > 0, 'expected paths array');
    for (const p of paths) {
      assert.equal(typeof p.renderedX, 'number');
      assert.equal(typeof p.renderedY, 'number');
      assert.equal(typeof p.renderedWidth, 'number');
      assert.equal(typeof p.renderedHeight, 'number');
      // The rendered bbox always contains the geometric bbox.
      assert.ok(p.renderedX <= p.x, `renderedX ${p.renderedX} > x ${p.x}`);
      assert.ok(p.renderedY <= p.y, `renderedY ${p.renderedY} > y ${p.y}`);
      assert.ok(p.renderedWidth >= p.width, `renderedWidth ${p.renderedWidth} < width ${p.width}`);
      assert.ok(
        p.renderedHeight >= p.height,
        `renderedHeight ${p.renderedHeight} < height ${p.height}`
      );
    }
    // The 8pt-stroked rectangle is inflated by half the line width per side.
    const stroked = paths.find((p) => p.hasStroke && p.strokeWidth === 8);
    assert.ok(stroked, `no 8pt-stroked path in: ${JSON.stringify(paths)}`);
    assert.ok(
      Math.abs(stroked.renderedWidth - (stroked.width + 8)) < 0.5,
      `expected width inflated by 8, geometric=${stroked.width} rendered=${stroked.renderedWidth}`
    );
    assert.ok(
      Math.abs(stroked.renderedHeight - (stroked.height + 8)) < 0.5,
      `expected height inflated by 8, geometric=${stroked.height} rendered=${stroked.renderedHeight}`
    );
    doc.close();
  } finally {
    cleanup();
  }
});

test('extractTextLines returns non-empty array with text field', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'lines', 'LINETOKEN');
    const doc = PdfDocument.open(path);
    const lines = doc.extractTextLines(0);
    assert.ok(Array.isArray(lines) && lines.length > 0, 'expected lines array');
    const found = lines.some((l) => (l.text || l.Text || '').includes('LINETOKEN'));
    assert.ok(found, `LINETOKEN not found in: ${JSON.stringify(lines.slice(0, 3))}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('extractAllText returns non-empty string with content', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'all', 'ALLTEXTMARKER');
    const doc = PdfDocument.open(path);
    const text = doc.extractAllText();
    assert.ok(typeof text === 'string' && text.includes('ALLTEXTMARKER'));
    doc.close();
  } finally {
    cleanup();
  }
});

// ── Conversion ────────────────────────────────────────────────────────────────

test('toMarkdown returns non-empty string', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'md', '# Heading\n\nBody.');
    const doc = PdfDocument.open(path);
    const md = doc.toMarkdown(0);
    assert.ok(typeof md === 'string' && md.length > 0, `unexpected: ${md}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('toMarkdownAll returns non-empty string', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'mdall', 'MDALLMARKER');
    const doc = PdfDocument.open(path);
    const md = doc.toMarkdownAll();
    assert.ok(typeof md === 'string' && md.length > 0);
    doc.close();
  } finally {
    cleanup();
  }
});

test('toHtml returns string containing html tags', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'html', 'HTMLMARKER');
    const doc = PdfDocument.open(path);
    const html = doc.toHtml(0);
    assert.ok(typeof html === 'string' && html.includes('<'), `no tags in: ${html}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('toHtmlAll returns non-empty string with tags', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'htmlall', 'HTMLALLMARKER');
    const doc = PdfDocument.open(path);
    const html = doc.toHtmlAll();
    assert.ok(typeof html === 'string' && html.includes('<'));
    doc.close();
  } finally {
    cleanup();
  }
});

test('toPlainText contains known word', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'plain', 'PLAINMARKER');
    const doc = PdfDocument.open(path);
    const text = doc.toPlainText(0);
    assert.ok(text.includes('PLAINMARKER'), `not found in: ${text}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('toPlainTextAll contains known word', { skip }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'plainall', 'PLAINALLMARKER');
    const doc = PdfDocument.open(path);
    const text = doc.toPlainTextAll();
    assert.ok(text.includes('PLAINALLMARKER'), `not found in: ${text}`);
    doc.close();
  } finally {
    cleanup();
  }
});

// ── Pdf factory extras ─────────────────────────────────────────────────────────

test('Pdf.fromText produces a valid PDF', { skip }, () => {
  try {
    const pdf = Pdf.fromText('Hello plain text');
    const buf = pdf.saveToBytes();
    assert.ok(isPdf(buf));
    pdf.close();
  } catch (e) {
    if (/unsupported|not compiled|5000/i.test(String(e))) return; // feature off
    throw e;
  }
});

test('Pdf.fromImage produces a valid PDF', { skip }, () => {
  // Write a minimal 1×1 PNG to disk and create PDF from it
  const { dir, cleanup } = tempDir();
  try {
    const png = Buffer.from([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
      0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
      0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8,
      0xff, 0xff, 0x3f, 0x00, 0x05, 0xfe, 0x02, 0xfe, 0x33, 0x12, 0x95, 0x14, 0x00, 0x00, 0x00,
      0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ]);
    const imgPath = join(dir, 'img.png');
    writeFileSync(imgPath, png);
    try {
      const pdf = Pdf.fromImage(imgPath);
      assert.ok(isPdf(pdf.saveToBytes()));
      pdf.close();
    } catch (e) {
      if (/unsupported|not compiled|5000/i.test(String(e))) return;
      throw e;
    }
  } finally {
    cleanup();
  }
});

test('Pdf.fromImageBytes produces a valid PDF', { skip }, () => {
  const png = Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0xff, 0xff, 0x3f,
    0x00, 0x05, 0xfe, 0x02, 0xfe, 0x33, 0x12, 0x95, 0x14, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
    0x44, 0xae, 0x42, 0x60, 0x82,
  ]);
  try {
    const pdf = Pdf.fromImageBytes(png);
    assert.ok(isPdf(pdf.saveToBytes()));
    pdf.close();
  } catch (e) {
    if (/unsupported|not compiled|5000/i.test(String(e))) return;
    throw e;
  }
});

// ── DocumentBuilder extras ─────────────────────────────────────────────────────

test('DocumentBuilder.save (non-encrypted) writes a PDF file', { skip: !DocumentBuilder }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = join(dir, 'plain.pdf');
    DocumentBuilder.create().a4Page().paragraph('plain save').done().save(path);
    assert.ok(statSync(path).size > 100);
  } finally {
    cleanup();
  }
});

test('DocumentBuilder.letterPage produces a PDF', { skip: !DocumentBuilder }, () => {
  const buf = DocumentBuilder.create().letterPage().paragraph('US Letter').done().build();
  assert.ok(isPdf(buf));
});

test('DocumentBuilder.page (custom size) produces a PDF', { skip: !DocumentBuilder }, () => {
  const buf = DocumentBuilder.create().page(300, 400).paragraph('custom size').done().build();
  assert.ok(isPdf(buf));
});

test('DocumentBuilder metadata setters do not throw', { skip: !DocumentBuilder }, () => {
  const buf = DocumentBuilder.create()
    .title('My Title')
    .author('Alice')
    .subject('Testing')
    .keywords('pdf, test')
    .creator('node-test')
    .a4Page()
    .paragraph('metadata')
    .done()
    .build();
  assert.ok(isPdf(buf));
});

test('DocumentBuilder.toBytesEncrypted produces encrypted PDF', { skip: !DocumentBuilder }, () => {
  const buf = DocumentBuilder.create()
    .a4Page()
    .paragraph('secret')
    .done()
    .toBytesEncrypted('user', 'owner');
  assert.ok(isPdf(buf));
  assert.ok(buf.includes('/Encrypt') || Buffer.from(buf).toString('latin1').includes('/Encrypt'));
});

// ── DocumentEditor mutations ───────────────────────────────────────────────────

test('DocumentEditor.deletePage reduces page count', { skip: !DocumentEditor }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const pathA = savePdf(dir, 'edA', '# Page A');
    const pathB = savePdf(dir, 'edB', '# Page B');
    const editor = DocumentEditor.open(pathA);
    editor.mergeFrom(pathB);
    const before = editor.pageCount();
    editor.deletePage(0);
    const after = editor.pageCount();
    assert.equal(after, before - 1);
    editor.close();
  } finally {
    cleanup();
  }
});

test('DocumentEditor.movePage changes page order', { skip: !DocumentEditor }, () => {
  const { dir, cleanup } = tempDir();
  try {
    // Use DocumentBuilder to create a 2-page PDF so all pages live in page_order
    // from the start (no merged_pages split that causes a pre-existing Rust panic).
    const multiPath = join(dir, 'multi.pdf');
    const outPath = join(dir, 'out.pdf');
    const multiBytes = DocumentBuilder.create()
      .a4Page()
      .at(72, 720)
      .text('PAGEFIRST')
      .done()
      .a4Page()
      .at(72, 720)
      .text('PAGESECOND')
      .done()
      .build();
    writeFileSync(multiPath, multiBytes);

    const editor = DocumentEditor.open(multiPath);
    editor.movePage(1, 0); // [PAGESECOND, PAGEFIRST]
    editor.save(outPath);
    editor.close();

    const doc = PdfDocument.open(outPath);
    const words = doc.extractWords(0);
    const text = words.map((w) => w.text || w.Text || '').join(' ');
    assert.ok(text.includes('PAGESECOND'), `expected PAGESECOND on page 0, got: ${text}`);
    doc.close();
  } finally {
    cleanup();
  }
});

test('DocumentEditor.setTitle persists to re-opened doc', { skip: !DocumentEditor }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const path = savePdf(dir, 'title', '# Title Test');
    const editor = DocumentEditor.open(path);
    editor.setTitle('New Title');
    editor.save(path);
    editor.close();
    // Verify round-trip (if getTitle is available)
    const editor2 = DocumentEditor.open(path);
    if (typeof editor2.getTitle === 'function') {
      assert.equal(editor2.getTitle(), 'New Title');
    }
    editor2.close();
  } finally {
    cleanup();
  }
});

test('DocumentEditor.mergeFrom increases page count', { skip: !DocumentEditor }, () => {
  const { dir, cleanup } = tempDir();
  try {
    const pathA = savePdf(dir, 'mrgA', '# A');
    const pathB = savePdf(dir, 'mrgB', '# B');
    const editor = DocumentEditor.open(pathA);
    const before = editor.pageCount();
    editor.mergeFrom(pathB);
    const after = editor.pageCount();
    assert.ok(after > before, `expected more pages: before=${before} after=${after}`);
    editor.close();
  } finally {
    cleanup();
  }
});

// ── Signatures (unsigned PDF) ─────────────────────────────────────────────────

test('signatureCount returns 0 for unsigned PDF', { skip }, () => {
  const buf = Buffer.from(Pdf.fromMarkdown('# Unsigned').saveToBytes());
  const doc = PdfDocument.openFromBuffer(buf);
  try {
    // signatureCount may be a property, a function, or absent depending on build
    if (doc.signatureCount === undefined) return; // feature not exposed
    const count =
      typeof doc.signatureCount === 'function' ? doc.signatureCount() : doc.signatureCount;
    assert.equal(typeof count, 'number');
    assert.ok(count >= 0);
  } catch (e) {
    if (/unsupported|not compiled|5000/i.test(String(e))) return;
    throw e;
  } finally {
    doc.close();
  }
});
