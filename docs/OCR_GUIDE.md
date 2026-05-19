# OCR Guide — Extracting Text from Scanned PDFs

PDFOxide can extract text from scanned PDFs using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models via ONNX Runtime. This guide covers model selection, configuration, and best practices.

## How It Works

The OCR pipeline has three stages:

1. **Detection** (DBNet++): Finds text regions (bounding boxes) in the page image
2. **Recognition** (SVTR): Reads text from each cropped region
3. **Postprocessing**: Sorts results in reading order and joins into text

PDFOxide automatically detects whether a page is scanned or has native text, so you can use the same API for both.

## Quick Start

```bash
# 1. Download recommended models (~12.5 MB total)
./scripts/setup_ocr_models.sh

# 2. Run (Rust)
cargo run --features ocr --example ocr_scanned_pdf -- \
    --pdf scanned.pdf \
    --det .models/det.onnx \
    --rec .models/rec.onnx \
    --dict .models/en_dict.txt
```

## OCR Support by Binding

OCR *recognition* needs the native `ocr` feature compiled in **plus** an
ONNX Runtime shared library and provisioned models at runtime.
**Auto mode works in every binding regardless**: when OCR is unavailable
it degrades gracefully to native text with a typed
`ocr_requested_but_unavailable` reason — never a crash or silent empty.

| Binding | OCR recognition | How |
|---|---|---|
| Rust | yes | build with `--features ocr` |
| Python | yes | the published wheel ships `ocr`; supply ONNX Runtime + models |
| Node.js / TypeScript | yes (v0.3.52+) | the published prebuilt ships `ocr`; `npm i onnxruntime-node` + models |
| Go (cgo + purego) | yes (v0.3.52+) | the published native lib ships `ocr`; supply ONNX Runtime + models |
| C# / .NET | yes (v0.3.52+) | the published native lib ships `ocr`; supply ONNX Runtime + models |
| WASM (browser/Deno/edge) — default `pdf-oxide-wasm` | no | ships without the OCR backend |
| WASM — `wasm-ocr` build | yes (experimental, #524) | pure-Rust `tract` backend, no native lib / no JS bridge; host supplies model bytes (see *WebAssembly* below). **Output-equivalent to the native `ort` path** — verified both at the inference-engine level (max abs diff ≤ 3e-6 on the real det/rec graphs) and end-to-end (byte-identical recognized text on a shared fixture). "Experimental" refers to cross-target (browser/Deno/edge) hardening, not OCR quality |

Before v0.3.52 only Rust and the Python wheel shipped with `ocr`; Node/Go/C#
required a source build. As of v0.3.52 their prebuilts include it (#520).

## Model Selection

PDFOxide supports PaddleOCR v3, v4, and v5 models. Detection and recognition models can be mixed across versions.

### Recommended: V4 Detection + V5 Recognition

| Component | Model | Size | Source |
|-----------|-------|------|--------|
| Detection | ch_PP-OCRv4_det | 4.7 MB | [deepghs/paddleocr](https://huggingface.co/deepghs/paddleocr) |
| Recognition | en_PP-OCRv5_mobile_rec | 7.8 MB | [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx) |
| Dictionary | PP-OCRv5 English | 4 KB | Same as recognition |

This combination delivers the best English accuracy because:

- **V4 detection** reliably segments text lines with minimal false positives. It uses a MaxSide resize strategy that downscales images to 960px, which is well-matched to its training data.
- **V5 recognition** has the highest character-level accuracy for English text. It processes cropped text regions independently of the detection model's resize strategy.

### All Tested Combinations

| Combination | Detection | Recognition | English Accuracy | Total Size | Config |
|---|---|---|---|---|---|
| **V4 det + V5 rec** | ch_PP-OCRv4_det | en_PP-OCRv5_mobile_rec | Best | ~12.5 MB | `OcrConfig::default()` |
| V4 det + V4 rec | ch_PP-OCRv4_det | en_PP-OCRv4_rec | Good | ~12.4 MB | `OcrConfig::default()` |
| V5 det + V5 rec | PP-OCRv5_server_det | en_PP-OCRv5_mobile_rec | Good (different error profile) | ~96 MB | `OcrConfig::v5()` |
| V3 det + V3 rec | en_PP-OCRv3_det | en_PP-OCRv3_rec | Fair | ~11 MB | `OcrConfig::default()` |

### When to Use Full V5

The full V5 stack (V5 detection + V5 recognition) uses a much larger detection model (88 MB vs 4.7 MB). Use it when:

- You need to detect text in complex layouts (mixed orientations, curved text)
- The V4 detector misses text regions in your documents
- You have sufficient memory and don't mind slower inference

For standard English documents (reports, articles, invoices), the V4+V5 combination is both faster and more accurate.

## Detection Resize Strategies

The detection model needs to resize input images before inference. PDFOxide supports two strategies:

### MaxSide (V3/V4 default)

Scales the image **down** so the longest side fits within a limit (default: 960px). This is fast and works well with V3/V4 models that were trained on smaller inputs.

```
Original: 2480×3508 (300 DPI A4 scan)
After MaxSide(960): 679×960
```

### MinSide (V5)

Scales the image **up** so the shortest side is at least a minimum (default: 64px), but caps the longest side at a limit (default: 4000px). This preserves high resolution, which V5 server models need for accurate detection.

```
Original: 2480×3508 (300 DPI A4 scan)
After MinSide(64, 4000): 2480×3508 (unchanged — already above minimum)

Original: 30×20 (tiny image)
After MinSide(64, 4000): 96×64 (scaled up)
```

**Important:** Using MaxSide with V5 detection models (or MinSide with V3/V4 models) will produce poor results. Always match the strategy to the model version.

## Configuration Reference

> **Recommended entrypoint — auto mode.** Each binding exposes an
> `extract_text_auto` / `extractTextAuto` / `ExtractTextAuto` that
> classifies each page (native text / scanned / hybrid) and routes
> accordingly: native extraction when text is present, OCR when the
> page is image-only, and merged output for hybrid pages. When OCR is
> unavailable (no models, no ORT, or the binding lacks the `ocr`
> feature) it **degrades gracefully** to native text with the typed
> reason `ocr_requested_but_unavailable` — never a crash. The manual
> `OcrEngine` usage below is for advanced cases where you want
> control over the config or want to OCR a single image directly.

### Rust

**Auto mode (recommended):**

```rust
use pdf_oxide::PdfDocument;

let doc = PdfDocument::open("scanned-or-mixed.pdf")?;
// Per-page: native text if present, OCR if scanned, hybrid merge
// otherwise. Falls back to native text if OCR isn't built/available.
let text = doc.extract_text_auto(0)?;
```

**Manual `OcrEngine` (advanced — direct control over models/config):**

```rust
use pdf_oxide::ocr::{OcrConfig, OcrEngine, DetResizeStrategy};

// Default config (V3/V4 detection + any recognition)
let config = OcrConfig::default();

// V5 config (V5 detection + any recognition)
let config = OcrConfig::v5();

// Custom config
let config = OcrConfig::builder()
    .det_threshold(0.3)       // Detection confidence (0.0-1.0, default: 0.3)
    .box_threshold(0.6)       // Box filter threshold (0.0-1.0, default: 0.6)
    .rec_threshold(0.5)       // Recognition confidence (0.0-1.0, default: 0.5)
    .num_threads(4)           // ONNX Runtime threads (default: 4)
    .max_candidates(1000)     // Max text box candidates (default: 1000)
    .unclip_ratio(1.5)        // Box expansion ratio (default: 1.5)
    .rec_target_height(48)    // Recognition input height (default: 48)
    .det_resize_strategy(DetResizeStrategy::MaxSide { max_side: 960 })
    .build();

let engine = OcrEngine::new("det.onnx", "rec.onnx", "dict.txt", config)?;
```

### Python

```bash
# Recommended: Install with OCR support
pip install pdf_oxide[ocr]
```

**Auto mode (recommended):**

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("scanned-or-mixed.pdf")
# Per-page native/OCR/hybrid routing. Gracefully falls back to native
# text if OCR isn't installed or models aren't present.
text = doc.extract_text_auto(0)
```

**Manual `OcrEngine` (advanced):**

```python
from pdf_oxide import OcrConfig, OcrEngine

# Default config (V3/V4 detection)
config = OcrConfig()

# V5 config
config = OcrConfig(use_v5=True)

# Custom config
config = OcrConfig(
    det_threshold=0.3,
    box_threshold=0.6,
    rec_threshold=0.5,
    num_threads=4,
    max_candidates=1000,
    use_v5=False,  # True for V5 detection models
)

engine = OcrEngine(
    det_model_path=".models/det.onnx",
    rec_model_path=".models/rec.onnx",
    dict_path=".models/en_dict.txt",
    config=config,  # Optional, defaults to OcrConfig()
)
```

### Node.js / TypeScript

The published `pdf-oxide` prebuilt ships with OCR as of v0.3.52. Supply
ONNX Runtime via npm and provision models from JS — no Python, no shell
scripts:

```bash
npm install pdf-oxide onnxruntime-node
```

```js
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
// ONNX Runtime shared lib, straight from the npm package
// (adjust the path per OS/arch):
process.env.ORT_DYLIB_PATH = require.resolve(
  'onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1');
process.env.PDF_OXIDE_MODEL_DIR = '/path/to/models';

const px = await import('pdf-oxide');
px.prefetchModels(['english']);          // one-off: downloads det/rec/dict
const doc = px.PdfDocument.open('scan.pdf');
console.log(doc.extractTextAuto(0));     // native + OCR'd image text
```

Run `prefetchModels()` as a one-off provisioning step, or place
`det.onnx` / `rec.onnx` / `en_dict.txt` in `PDF_OXIDE_MODEL_DIR`
yourself.

### Go

The published Go native library ships with `ocr` as of v0.3.52. Supply
ONNX Runtime + models via environment variables; everything else is
identical to the Node/Python path.

```bash
# install onnxruntime however you prefer (apt / brew / tarball / etc.)
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export PDF_OXIDE_MODEL_DIR=/path/to/models
```

**Auto mode (recommended):**

```go
import po "github.com/yfedoseev/pdf_oxide/go"

// One-off provisioning (mirrors Node's prefetchModels):
_, _ = po.PrefetchModels("english")

doc, err := po.Open("scanned-or-mixed.pdf")
if err != nil { panic(err) }
defer doc.Close()
text, err := doc.ExtractTextAuto(0)   // native / OCR / hybrid auto
```

**Manual `OcrEngine` (advanced):**

```go
eng, err := po.NewOcrEngine(
    "/path/to/models/det.onnx",
    "/path/to/models/rec.onnx",
    "/path/to/models/en_dict.txt",
)
if err != nil { panic(err) }
defer eng.Close()
text, err := doc.ExtractTextWithOcr(0, eng)
```

`doc.ClassifyPage(0)` exposes the page-type classification for routing
decisions, and `po.PrefetchModels("english", "chinese")` (variadic)
downloads the manifest entries into `PDF_OXIDE_MODEL_DIR`.

### C# / .NET

The published `PdfOxide` NuGet package ships with `ocr` as of v0.3.52.
Same shape as Go: supply an ONNX Runtime shared library and models.

```bash
# Linux / macOS:  /usr/lib/libonnxruntime.so  /  /usr/local/lib/libonnxruntime.dylib
# Windows:        onnxruntime.dll on PATH
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export PDF_OXIDE_MODEL_DIR=/path/to/models
```

**Auto mode (recommended):**

```csharp
using PdfOxide.Core;

// One-off provisioning (downloads det/rec/dict into PDF_OXIDE_MODEL_DIR):
OcrEngine.PrefetchModels("english");

using var doc = PdfDocument.Open("scanned-or-mixed.pdf");
string text = doc.ExtractTextAuto(0);          // native / OCR / hybrid auto
```

**Manual `OcrEngine` (advanced):**

```csharp
using var eng = OcrEngine.Load(
    "/path/to/models/det.onnx",
    "/path/to/models/rec.onnx",
    "/path/to/models/en_dict.txt");
string text = eng.ExtractText(doc, 0);
```

`doc.ClassifyPage(0)` returns the page-type classification string;
`OcrEngine.PageNeedsOcr(doc, 0)` is the shortcut needs-OCR check.

## Page Type Detection

PDFOxide automatically classifies pages before extraction:

| Page Type | Description | Action |
|-----------|-------------|--------|
| **NativeText** | Has substantial embedded text | Uses standard text extraction |
| **ScannedPage** | Large image, no/minimal text | Full OCR |
| **HybridPage** | Some native text + large images | Uses whichever source produces more text |

```rust
use pdf_oxide::ocr::{detect_page_type, PageType};

match detect_page_type(&mut doc, 0)? {
    PageType::NativeText => println!("Native text"),
    PageType::ScannedPage => println!("Needs OCR"),
    PageType::HybridPage => println!("Mixed content"),
}
```

## Dictionary Setup

PaddleOCR dictionaries are text files with one character per line. The model's output classes map to dictionary entries by index.

**Critical:** The dictionary must include a space character as the last line. PaddleOCR models output space as the final class (e.g., index 96 for V3/V4 with 97 classes, index 437 for V5 with 438 classes). If space is missing, words will run together.

```bash
# Download dictionary
curl -L https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt -o dict.txt

# Add space as last line (required!)
echo " " >> dict.txt
```

The `setup_ocr_models.sh` script handles this automatically.

## ONNX Runtime Setup

The OCR feature requires ONNX Runtime v1.23+ at runtime.

### Option 1: System Install

```bash
# Ubuntu/Debian
apt install libonnxruntime-dev

# Or download from GitHub releases
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-1.23.0.tgz
tar xzf onnxruntime-linux-x64-1.23.0.tgz
```

### Option 2: Environment Variables

```bash
export ORT_LIB_LOCATION=/path/to/onnxruntime/lib
export ORT_PREFER_DYNAMIC_LINK=1

# Then build
cargo build --features ocr
```

### macOS

```bash
brew install onnxruntime
export ORT_LIB_LOCATION=$(brew --prefix onnxruntime)/lib
```

### WebAssembly

The **default** `pdf-oxide-wasm` package ships **without** OCR — its
`WasmOcrEngine` / `extractTextOcr` throw an error directing you to the
`wasm-ocr` build. (The native `ort` OCR backend links a native ONNX
Runtime shared library and does not target `wasm32`.) Auto mode still
works there, falling back to native text with a typed reason.

The **`wasm-ocr` build** (issue #524, *experimental*) runs OCR entirely
in-WASM via a pure-Rust [`tract`](https://github.com/sonos/tract)
backend — no native library, no `onnxruntime-web` JS bridge. Build it
with the `wasm_js` getrandom backend flag:

```sh
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  wasm-pack build --target web -- --no-default-features --features wasm-ocr
```

Model **delivery is host-side** (the browser has no filesystem and the
models are tens of MB). Fetch the detector + recognizer ONNX and the
char dictionary — `modelManifest()` returns the URLs — cache them with
the Cache API (or IndexedDB), then hand the bytes in:

```js
import init, { WasmOcrEngine, WasmPdfDocument, modelManifest } from "pdf-oxide";
await init();

// One-time: fetch + cache the (large) models. modelManifest() lists
// the detector + per-language recognizer/dict URLs.
const cache = await caches.open("pdf-oxide-ocr-v1");
async function cached(url) {
  let r = await cache.match(url);
  if (!r) { await cache.add(url); r = await cache.match(url); }
  return new Uint8Array(await r.arrayBuffer());
}
const m = JSON.parse(modelManifest());
const det  = await cached(m.detector.url);
const en   = m.languages.find(l => l.language === "english");
const rec  = await cached(en.rec_url);
const dict = new TextDecoder().decode(await cached(en.dict_url));

// Build the engine ONCE — extractTextOcr borrows it (it's not
// consumed), so the same handle is reusable across pages and across
// documents.
const ocr = new WasmOcrEngine(det, rec, dict);
const doc = new WasmPdfDocument(pdfBytes);
for (let p = 0; p < doc.pageCount(); p++) {
  const text = doc.extractTextOcr(p, ocr);
  // ... use text
}
// Or, for a raw scan image:  JSON.parse(ocr.ocrImage(pngBytes))
```

**Auto-routing per page** (classify, then OCR only when needed):

```js
function extractPage(doc, pageIndex, ocrEngine) {
  // 'TextLayer' | 'Scanned' | 'ImageText' | 'Mixed' | 'Empty'
  const kind = doc.classifyPage(pageIndex);
  if (kind === 'Scanned' || kind === 'ImageText' || kind === 'Mixed') {
    return doc.extractTextOcr(pageIndex, ocrEngine);        // run OCR
  }
  return doc.extractText(pageIndex);                        // native path
}
```

This mirrors the native `extract_text_auto` flow: native extraction
where text exists, OCR where it doesn't, no OCR cost on text-layer
pages.

OCR inference is CPU-bound and **synchronous** — run it in a **Web
Worker** so it doesn't block the UI thread; model fetch/caching is
async on the host as shown.

The tract backend is **output-equivalent to the native `ort` path**:
verified at the inference-engine level (identical outputs on the real
PaddleOCR det/rec graphs, max abs diff ≤ 3e-6) and end-to-end
(byte-identical recognized text on a shared fixture). The
`ort_vs_tract_*` equivalence tests in `src/ocr/backend.rs` pin this.

**Footprint.** A `--release` `wasm-ocr` build is ~23 MB raw →
~20.6 MB after `wasm-opt -Oz` → **~7 MB gzipped** over the wire (build
the release `.wasm` with the same `RUSTFLAGS` as above, then
`wasm-opt -Oz --enable-bulk-memory ...`). The PaddleOCR models
(det ≈ 4.7 MB + rec ≈ 7.8 MB) are **not** in the `.wasm` — the host
fetches them once and caches them (Cache API / IndexedDB), so they
cost nothing on repeat loads.

wasm OCR is still labelled *experimental* because cross-target
(browser / Deno / edge) integration testing is pending (#524 / #7) —
**not** because of recognition quality (matches native exactly) or
size (shippable, measured above).

## Troubleshooting

### Garbled output (e.g., `0I0f0m0j0p...`)

The dictionary file may be incorrect or the space character is missing. Re-download the dictionary and ensure space is the last line:
```bash
echo " " >> .models/en_dict.txt
```

### Words run together (no spaces)

Same cause — the space character is missing from the end of the dictionary file.

### V5 detection produces worse results than V4

Make sure you're using `OcrConfig::v5()` (Rust) or `OcrConfig(use_v5=True)` (Python). V5 models need the MinSide resize strategy to preserve image resolution. Using the default MaxSide strategy will downscale the image to 960px, which is too small for the V5 detector.

### Build error: `no method named tls_config`

This is a known bug in `ort-sys` 2.0.0-rc.11 when using the `download-binaries` feature. Install ONNX Runtime manually and set `ORT_LIB_LOCATION` instead.

### Python segfault (exit code 139)

Ensure you're using the latest version with the infinite recursion fix. Older versions had a bug where `extract_text()` → `needs_ocr()` → `detect_page_type()` → `extract_text()` caused a stack overflow.

## Examples

### Rust

```bash
cargo run --features ocr --example ocr_scanned_pdf -- \
    --pdf scanned.pdf \
    --det .models/det.onnx \
    --rec .models/rec.onnx \
    --dict .models/en_dict.txt

# With V5 detection
cargo run --features ocr --example ocr_scanned_pdf -- \
    --pdf scanned.pdf \
    --det .models/v5/det.onnx \
    --rec .models/v5/rec.onnx \
    --dict .models/v5/en_dict.txt \
    --v5
```

### Python

```bash
python examples/ocr_example.py scanned.pdf \
    --det .models/det.onnx \
    --rec .models/rec.onnx \
    --dict .models/en_dict.txt

# With V5 detection
python examples/ocr_example.py scanned.pdf \
    --det .models/v5/det.onnx \
    --rec .models/v5/rec.onnx \
    --dict .models/v5/en_dict.txt \
    --v5
```
