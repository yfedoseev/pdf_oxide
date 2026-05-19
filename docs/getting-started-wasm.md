# Getting Started with PDFOxide (WebAssembly)

PDFOxide compiles to WebAssembly for use in browsers and Node.js. The same Rust core that powers the Python and Rust APIs runs directly in JavaScript/TypeScript with near-native performance.

## Installation

### From npm (recommended)

```bash
npm install pdf-oxide-wasm
```

```javascript
// Works everywhere — the package ships three builds and routes each
// consumer to the right one through `package.json` conditional exports:
//   Node.js           → nodejs/  (CommonJS; loads the .wasm via fs)
//   Bundlers          → bundler/ (ESM; Vite / webpack / Rollup / esbuild
//                                  resolve the .wasm import natively)
//   Browsers / Deno   → web/     (ESM; loads the .wasm via fetch())
//   Cloudflare Workers → web/
const { WasmPdfDocument } = require("pdf-oxide-wasm");
// or ESM:
// import { WasmPdfDocument } from "pdf-oxide-wasm";
```

If your bundler's condition resolution is unusual (or you want to bypass
it), you can import a specific build directly via a subpath:

```javascript
import { WasmPdfDocument } from "pdf-oxide-wasm/bundler";  // for bundlers
import { WasmPdfDocument } from "pdf-oxide-wasm/web";       // for browsers
import { WasmPdfDocument } from "pdf-oxide-wasm/nodejs";    // for Node
```

### Building from Source

#### Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target
- `wasm-bindgen-cli` (must match the `wasm-bindgen` version in `Cargo.lock`)

```bash
# Install the WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen CLI (check Cargo.lock for the exact version)
cargo install wasm-bindgen-cli --version 0.2.118 --locked
```

### Build all three targets (what the release workflow does)

```bash
cargo build --lib --target wasm32-unknown-unknown --features wasm --release

for target in bundler nodejs web; do
  wasm-bindgen --target "$target" --out-dir "pkg/$target/" \
    target/wasm32-unknown-unknown/release/pdf_oxide.wasm
done
```

This produces `pkg/bundler/`, `pkg/nodejs/`, `pkg/web/`, each containing
`pdf_oxide.js`, `pdf_oxide.d.ts`, `pdf_oxide_bg.wasm`, and
`pdf_oxide_bg.wasm.d.ts`. The bundler target additionally emits
`pdf_oxide_bg.js` (the glue is split out so bundlers can import the
`.wasm` directly).

### Size-Optimized Build

For smaller WASM binaries, use the `release-small` profile:

```bash
cargo build --lib --target wasm32-unknown-unknown --features wasm \
  --profile release-small
```

## Quick Start

### Node.js (ESM)

```javascript
import { readFileSync } from "fs";
import { WasmPdfDocument, WasmPdf } from "pdf-oxide-wasm";
// Or if you built locally: from "./pkg/nodejs/pdf_oxide.js"

// Open a PDF file
const bytes = new Uint8Array(readFileSync("document.pdf"));
const doc = new WasmPdfDocument(bytes);

// Basic info
console.log(`Pages: ${doc.pageCount()}`);
console.log(`Version: ${doc.version()}`);

// Extract text
const text = doc.extractText(0);
console.log(text);

// Clean up
doc.free();
```

### Browser (vanilla, no bundler)

```html
<script type="module">
// Use the /web build explicitly when there's no bundler to pick the
// `"browser"` export condition for you.
import init, { WasmPdfDocument, WasmPdf } from "./pkg/web/pdf_oxide.js";

await init();

// Load PDF from fetch
const response = await fetch("document.pdf");
const bytes = new Uint8Array(await response.arrayBuffer());
const doc = new WasmPdfDocument(bytes);

console.log(`Pages: ${doc.pageCount()}`);
console.log(doc.extractText(0));
doc.free();
</script>
```

### Browser with a bundler (Vite, webpack, Rollup, esbuild)

```javascript
// No init() call needed — the bundler resolves the `.wasm` via the
// `import * as wasm from "./pdf_oxide_bg.wasm"` statement inside the
// package. For Vite, use `vite-plugin-wasm`.
import { WasmPdfDocument } from "pdf-oxide-wasm";

const bytes = new Uint8Array(await file.arrayBuffer());
const doc = new WasmPdfDocument(bytes);
console.log(doc.extractText(0));
doc.free();
```

### Browser with File Input

```html
<input type="file" id="pdfInput" accept=".pdf" />
<pre id="output"></pre>

<script type="module">
import init, { WasmPdfDocument } from "./pkg/web/pdf_oxide.js";
await init();

document.getElementById("pdfInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const bytes = new Uint8Array(await file.arrayBuffer());
  const doc = new WasmPdfDocument(bytes);

  let result = `Pages: ${doc.pageCount()}\n\n`;
  for (let i = 0; i < doc.pageCount(); i++) {
    result += `--- Page ${i + 1} ---\n`;
    result += doc.extractText(i) + "\n\n";
  }

  document.getElementById("output").textContent = result;
  doc.free();
});
</script>
```

## Creating PDFs

Create new PDFs from Markdown, HTML, or plain text using `WasmPdf`:

```javascript
import { WasmPdf, WasmPdfDocument } from "./pkg/nodejs/pdf_oxide.js";

// From Markdown
const pdf = WasmPdf.fromMarkdown("# Hello World\n\nThis is a PDF.", "My Title", "Author");
const bytes = pdf.toBytes(); // Uint8Array
console.log(`PDF size: ${pdf.size} bytes`);

// From HTML
const invoice = WasmPdf.fromHtml(
  "<h1>Invoice</h1><p>Thank you for your purchase.</p>",
  "Invoice #123"
);

// From plain text
const notes = WasmPdf.fromText("Meeting notes\n\nAction items:\n- Review PR\n- Update docs");

// Save to file (Node.js)
import { writeFileSync } from "fs";
writeFileSync("output.pdf", pdf.toBytes());

// Download in browser
const blob = new Blob([pdf.toBytes()], { type: "application/pdf" });
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "output.pdf";
a.click();
```

## Text Extraction

### Single Page

```javascript
const doc = new WasmPdfDocument(bytes);
const text = doc.extractText(0); // page 0
```

### All Pages

```javascript
const allText = doc.extractAllText(); // pages separated by form feed
```

### Convert to Markdown

```javascript
// Single page
const markdown = doc.toMarkdown(0);

// With options
const md = doc.toMarkdown(0, true, true); // detect_headings, include_images

// All pages
const allMarkdown = doc.toMarkdownAll();
```

### Convert to HTML

```javascript
const html = doc.toHtml(0);

// With layout preservation
const layoutHtml = doc.toHtml(0, true, true); // preserve_layout, detect_headings

// All pages
const allHtml = doc.toHtmlAll();
```

### Convert to Plain Text

```javascript
const plain = doc.toPlainText(0);
const allPlain = doc.toPlainTextAll();
```

## Structured Extraction

Get character-level and span-level data with positions and font metadata:

```javascript
// 1. Scoped extraction (v0.3.14)
// Area: [x, y, width, height]
const headerRegion = doc.within(0, [0, 700, 612, 92]);
const headerText = headerRegion.extractText();

// 2. Character-level data
const chars = doc.extractChars(0);
// Returns: [{ char, bbox: {x, y, width, height}, font_name, font_size, font_weight, is_italic, color: {r, g, b} }, ...]

for (const c of chars) {
  console.log(`'${c.char}' at (${c.bbox.x}, ${c.bbox.y}) font=${c.font_name}`);
}

// 3. Word-level extraction (v0.3.14)
const words = doc.extractWords(0);
for (const w of words) {
  console.log(`Word: ${w.text} at ${w.bbox.x},${w.bbox.y}`);
}

// 4. Line-level extraction (v0.3.14)
const lines = doc.extractTextLines(0);
for (const line of lines) {
  console.log(`Line: ${line.text}`);
}

// 5. Span-level data
const spans = doc.extractSpans(0);
// Returns: [{ text, bbox, font_name, font_size, font_weight, is_italic, color }, ...]

for (const span of spans) {
  console.log(`"${span.text}" size=${span.font_size}`);
}
```

## Working with Form Fields

Extract form field data and export filled values:

```javascript
const doc = new WasmPdfDocument(bytes);

// Get all form fields
const fields = doc.getFormFields();
// Returns: [{ name, field_type, value, flags }, ...]

for (const f of fields) {
  console.log(`${f.name} (${f.field_type}) = ${f.value}`);
}

// Export form data as FDF or XFDF
const fdfBytes = doc.exportFormData();       // FDF format (default)
const xfdfBytes = doc.exportFormData("xfdf"); // XFDF format
```

### Form Fields in Text Extraction

Filled form field values appear inline in `toMarkdown` and `toHtml`:

```javascript
// Include form field values (default)
const md = doc.toMarkdown(0, true, true, true); // ..., include_form_fields=true
const html = doc.toHtml(0, true, true, true);

// Exclude form field values
const mdClean = doc.toMarkdown(0, true, true, false); // include_form_fields=false
```

## Text Search

Search across all pages or within a specific page:

```javascript
// Search all pages
const results = doc.search("hello", true); // case_insensitive=true
// Returns: [{ page, text, bbox, start_index, end_index, span_boxes }, ...]

for (const r of results) {
  console.log(`Found "${r.text}" on page ${r.page}`);
}

// Search single page
const pageResults = doc.searchPage(0, "hello", true, true); // case_insensitive, literal

// Regex search
const regexResults = doc.search("\\d{4}-\\d{2}-\\d{2}"); // find dates

// Whole word match
const wordResults = doc.search("test", false, true, true); // literal, whole_word
```

## Image Metadata

```javascript
// Get image metadata (does NOT return raw bytes)
const images = doc.extractImages(0);
// Returns: [{ width, height, color_space, bits_per_component, bbox }, ...]

for (const img of images) {
  console.log(`Image: ${img.width}x${img.height} ${img.color_space}`);
}
```

## Editing PDFs

### Metadata

```javascript
const doc = new WasmPdfDocument(bytes);

doc.setTitle("Updated Title");
doc.setAuthor("Jane Doe");
doc.setSubject("Quarterly Report");
doc.setKeywords("finance, Q4, 2025");

const edited = doc.saveToBytes(); // Uint8Array with changes applied
```

### Page Rotation

```javascript
// Get current rotation
const rotation = doc.pageRotation(0); // 0, 90, 180, or 270

// Set absolute rotation
doc.setPageRotation(0, 90);

// Add to current rotation
doc.rotatePage(0, 90); // if was 90, now 180

// Rotate all pages
doc.rotateAllPages(180);
```

### Page Dimensions

```javascript
// Get MediaBox [llx, lly, urx, ury]
const mediaBox = doc.pageMediaBox(0);
console.log(`Page size: ${mediaBox[2]}x${mediaBox[3]} points`);

// Set MediaBox
doc.setPageMediaBox(0, 0, 0, 612, 792); // US Letter

// Get CropBox (may be null if not set)
const cropBox = doc.pageCropBox(0);

// Set CropBox
doc.setPageCropBox(0, 50, 50, 562, 742);

// Crop margins from all pages (points)
doc.cropMargins(36, 36, 36, 36); // 0.5 inch margins
```

### Erase / Whiteout

```javascript
// Erase a single region
doc.eraseRegion(0, 100, 700, 300, 720); // llx, lly, urx, ury

// Erase multiple regions at once
const rects = new Float32Array([
  100, 700, 300, 720,  // region 1
  100, 650, 300, 670,  // region 2
]);
doc.eraseRegions(0, rects);

// Clear pending erase operations
doc.clearEraseRegions(0);
```

### Annotations

```javascript
// Flatten annotations into page content (makes them permanent)
doc.flattenPageAnnotations(0);

// Flatten all pages
doc.flattenAllAnnotations();
```

### Redaction

```javascript
// Apply redactions on a page (permanently removes content)
doc.applyPageRedactions(0);

// Apply redactions on all pages
doc.applyAllRedactions();
```

### Image Manipulation

```javascript
// List images on a page
const images = doc.pageImages(0);
// Returns: [{ name, bounds: [x, y, width, height], matrix: [a, b, c, d, e, f] }, ...]

// Reposition an image
doc.repositionImage(0, images[0].name, 100, 500);

// Resize an image
doc.resizeImage(0, images[0].name, 200, 150);

// Set full bounds
doc.setImageBounds(0, images[0].name, 100, 500, 200, 150);
```

## Saving

```javascript
// Save with edits
const output = doc.saveToBytes(); // Uint8Array

// Save with encryption (AES-256)
const encrypted = doc.saveEncryptedToBytes(
  "user-password",
  "owner-password",  // optional, defaults to user password
  true,   // allow_print
  true,   // allow_copy
  false,  // allow_modify
  true    // allow_annotate
);
```

## Encrypted PDFs

```javascript
const doc = new WasmPdfDocument(encryptedBytes);

// Authenticate before accessing content
const success = doc.authenticate("password");
if (success) {
  const text = doc.extractText(0);
  console.log(text);
}
```

## Document Info

```javascript
const doc = new WasmPdfDocument(bytes);

const [major, minor] = doc.version();
console.log(`PDF ${major}.${minor}`);
console.log(`Pages: ${doc.pageCount()}`);
console.log(`Tagged PDF: ${doc.hasStructureTree()}`);
```

## Memory Management

WASM objects hold Rust memory that must be freed explicitly:

```javascript
const doc = new WasmPdfDocument(bytes);
try {
  // ... work with doc
} finally {
  doc.free();
}

// Or with the using declaration (TC39 Explicit Resource Management):
using doc = new WasmPdfDocument(bytes);
// automatically freed when doc goes out of scope
```

## TypeScript

Type definitions are generated alongside the JS bindings. Import directly:

```typescript
import { WasmPdfDocument, WasmPdf } from "./pkg/nodejs/pdf_oxide.js";

const doc: WasmPdfDocument = new WasmPdfDocument(bytes);
const text: string = doc.extractText(0);
const markdown: string = doc.toMarkdown(0);
const pdf: WasmPdf = WasmPdf.fromMarkdown("# Hello");
const size: number = pdf.size;
```

## Error Handling

All methods that can fail throw JavaScript `Error` objects:

```javascript
try {
  const doc = new WasmPdfDocument(new Uint8Array([0, 1, 2])); // invalid PDF
} catch (e) {
  console.error(`Failed to open: ${e.message}`);
}

try {
  doc.extractText(999); // invalid page index
} catch (e) {
  console.error(`Extraction failed: ${e.message}`);
}
```

## API Reference

### WasmPdf (PDF Creation)

| Method | Returns | Description |
|--------|---------|-------------|
| `WasmPdf.fromMarkdown(content, title?, author?)` | `WasmPdf` | Create PDF from Markdown |
| `WasmPdf.fromHtml(content, title?, author?)` | `WasmPdf` | Create PDF from HTML |
| `WasmPdf.fromText(content, title?, author?)` | `WasmPdf` | Create PDF from plain text |
| `.toBytes()` | `Uint8Array` | Get PDF as bytes |
| `.size` | `number` | PDF size in bytes (readonly) |

### WasmPdfDocument (Read, Extract, Edit)

**Read-Only:**

| Method | Returns | Description |
|--------|---------|-------------|
| `new WasmPdfDocument(data)` | `WasmPdfDocument` | Load PDF from `Uint8Array` |
| `.pageCount()` | `number` | Number of pages |
| `.version()` | `Uint8Array` | PDF version as `[major, minor]` |
| `.authenticate(password)` | `boolean` | Decrypt an encrypted PDF |
| `.hasStructureTree()` | `boolean` | Check if Tagged PDF |

**Text Extraction:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.extractText(page)` | `string` | Plain text from one page |
| `.extractAllText()` | `string` | Plain text from all pages |
| `.extractChars(page)` | `Array` | Character-level data with positions |
| `.extractSpans(page)` | `Array` | Span-level data with font info |

**Format Conversion:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.toMarkdown(page, headings?, images?)` | `string` | Convert page to Markdown |
| `.toMarkdownAll(headings?, images?)` | `string` | Convert all pages to Markdown |
| `.toHtml(page, layout?, headings?)` | `string` | Convert page to HTML |
| `.toHtmlAll(layout?, headings?)` | `string` | Convert all pages to HTML |
| `.toPlainText(page)` | `string` | Convert page to plain text |
| `.toPlainTextAll()` | `string` | Convert all pages to plain text |

**Search:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.search(pattern, case?, literal?, word?, max?)` | `Array` | Search all pages |
| `.searchPage(page, pattern, case?, literal?, word?, max?)` | `Array` | Search one page |

**Image Info:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.extractImages(page)` | `Array` | Image metadata (no raw bytes) |
| `.pageImages(page)` | `Array` | Image names and bounds |

**Document Structure:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.getOutline()` | `Array\|null` | Document bookmarks / table of contents |
| `.getAnnotations(page)` | `Array` | Annotation metadata (type, rect, contents, etc.) |
| `.extractPaths(page)` | `Array` | Vector paths (lines, curves, shapes) |

**Form Fields:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.getFormFields()` | `Array` | All form fields with name, type, value, flags |
| `.exportFormData(format?)` | `Uint8Array` | Export form data as FDF (default) or XFDF |

**Editing:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.setTitle(title)` | `void` | Set document title |
| `.setAuthor(author)` | `void` | Set document author |
| `.setSubject(subject)` | `void` | Set document subject |
| `.setKeywords(keywords)` | `void` | Set document keywords |
| `.setPageRotation(page, degrees)` | `void` | Set page rotation |
| `.rotatePage(page, degrees)` | `void` | Add to page rotation |
| `.rotateAllPages(degrees)` | `void` | Rotate all pages |
| `.setPageMediaBox(page, llx, lly, urx, ury)` | `void` | Set MediaBox |
| `.setPageCropBox(page, llx, lly, urx, ury)` | `void` | Set CropBox |
| `.cropMargins(left, right, top, bottom)` | `void` | Crop all page margins |
| `.eraseRegion(page, llx, lly, urx, ury)` | `void` | Whiteout a region |
| `.eraseRegions(page, rects)` | `void` | Whiteout multiple regions |
| `.clearEraseRegions(page)` | `void` | Clear pending erases |
| `.flattenPageAnnotations(page)` | `void` | Flatten annotations on page |
| `.flattenAllAnnotations()` | `void` | Flatten all annotations |
| `.applyPageRedactions(page)` | `void` | Apply redactions on page |
| `.applyAllRedactions()` | `void` | Apply all redactions |
| `.repositionImage(page, name, x, y)` | `void` | Move image |
| `.resizeImage(page, name, w, h)` | `void` | Resize image |
| `.setImageBounds(page, name, x, y, w, h)` | `void` | Set image bounds |

**Save:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.saveToBytes()` | `Uint8Array` | Save edited PDF |
| `.saveEncryptedToBytes(pass, owner?, print?, copy?, modify?, annotate?)` | `Uint8Array` | Save with AES-256 encryption |
| `.free()` | `void` | Release WASM memory |

## Feature Availability

Some features require native dependencies and are **not available** in WebAssembly builds:

| Feature | WASM | Notes |
|---------|------|-------|
| Text extraction | Yes | Full support |
| PDF creation | Yes | Markdown, HTML, text, images |
| PDF editing | Yes | Full support |
| Encryption | Yes | AES-256 |
| OCR | Default build: **No**. `wasm-ocr` build: **Yes** (experimental) | Pure-Rust [`tract`](https://github.com/sonos/tract) backend — no native lib, no `onnxruntime-web` JS bridge. Output-equivalent to native `ort`. See *OCR (wasm-ocr build)* below. |
| Digital signatures | **No** | Requires native crypto libraries |
| Page rendering | **No** | Requires tiny-skia (native only) |

### OCR (`wasm-ocr` build)

The **default** `pdf-oxide-wasm` package has no OCR. The opt-in
`wasm-ocr` build runs OCR entirely in-WASM via pure-Rust `tract`, with
host-supplied model bytes (no filesystem):

```sh
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  wasm-pack build --target web -- --no-default-features --features wasm-ocr
```

```js
import init, { WasmOcrEngine, WasmPdfDocument, modelManifest } from "pdf-oxide";
await init();

// modelManifest() lists the detector + per-language recognizer/dict
// URLs. Fetch them once, cache them with the Cache API / IndexedDB,
// then hand the bytes in:
const ocr = new WasmOcrEngine(detBytes, recBytes, dictString);
const doc = new WasmPdfDocument(pdfBytes);

// Auto-route per page (classify first, OCR only what needs it):
function extractPage(p) {
  const kind = doc.classifyPage(p);                       // 'TextLayer'|'Scanned'|...
  return (kind === 'TextLayer' || kind === 'Empty')
    ? doc.extractText(p)
    : doc.extractTextOcr(p, ocr);
}
```

OCR inference is CPU-bound and synchronous — run it in a **Web Worker**
so it doesn't block the UI thread. Full recipe (fetch + Cache API,
size budget, Web Worker pattern): the **WebAssembly** section of the
[OCR Guide](OCR_GUIDE.md#webassembly).

For OCR in the **native** bindings (Rust / Python / Node / Go / C#),
see [OCR Guide](OCR_GUIDE.md).

## Next Steps

- [TypeScript Definitions](../examples/wasm_node/pdf_oxide.d.ts) — Full type signatures
- [Node.js Example](../examples/wasm_node/extract_text.mjs) — Working demo script
- [API Reference](https://docs.rs/pdf_oxide) — Full Rust API documentation
- [GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues) — Report bugs or request features
