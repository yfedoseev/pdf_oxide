# Architecture Overview

System design and implementation details for pdf_oxide.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Module Organization](#module-organization)
- [Algorithm Details](#algorithm-details)
- [Performance Design](#performance-design)
- [Extension Points](#extension-points)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API Layer                          │
│  (PdfDocument, Exporters, Configuration)                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────────────────┐
│                     Processing Pipeline                          │
│                                                                  │
│  ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐ │
│  │  Lexer   │──▶│ Parser  │──▶│  Stream  │──▶│   Layout    │ │
│  │          │   │         │   │ Decoder  │   │  Analysis   │ │
│  └──────────┘   └─────────┘   └──────────┘   └─────────────┘ │
│                                                       │          │
│  ┌──────────┐   ┌─────────┐   ┌──────────┐          │         │
│  │   Text   │◀──│  Font   │◀──│  Object  │◀─────────┘         │
│  │Extractor │   │ Parser  │   │  Resolver│                     │
│  └──────────┘   └─────────┘   └──────────┘                     │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────┐   ┌─────────┐   ┌──────────┐                    │
│  │ Markdown │   │  HTML   │   │   Text   │                    │
│  │ Exporter │   │Exporter │   │ Exporter │                    │
│  └──────────┘   └─────────┘   └──────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────────────────┐
│                     Storage Layer                                │
│  (Memory-mapped files, Object cache, Font cache)               │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Lexer (`src/lexer.rs`)

**Purpose:** Tokenize PDF byte stream into lexical tokens.

**Key Features:**
- Zero-copy tokenization using byte slices
- Handles all PDF token types (numbers, strings, names, operators)
- Efficient whitespace and comment skipping
- Position tracking for error reporting

**Performance:**
- ~1-2ms for typical PDFs
- O(n) complexity where n = file size
- Minimal allocations (uses `&[u8]` slices)

**Example:**
```rust
pub struct Lexer<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Lexer<'a> {
    pub fn next_token(&mut self) -> Result<Token<'a>> {
        // Zero-copy token extraction
    }
}
```

### 2. Parser (`src/parser.rs`)

**Purpose:** Parse PDF objects from token stream.

**Key Features:**
- Recursive descent parser
- Cycle detection for circular references
- Recursion depth limiting (max 100 levels)
- Proper error context propagation

**Object Types:**
- Null, Boolean, Integer, Real
- String, Name
- Array, Dictionary
- Stream, Reference

**Example:**
```rust
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    objects: HashMap<ObjectId, Object>,
    recursion_depth: u32,
}

impl<'a> Parser<'a> {
    pub fn parse_object(&mut self) -> Result<Object> {
        // Recursive object parsing with cycle detection
    }
}
```

### 3. Stream Decoder (`src/stream_decoder.rs`)

**Purpose:** Decompress PDF streams using various filters.

**Supported Filters:**
- **FlateDecode**: Zlib/Deflate (most common)
- **LZWDecode**: LZW compression
- **ASCII85Decode**: ASCII85 encoding
- **ASCIIHexDecode**: Hex encoding
- **RunLengthDecode**: Run-length encoding
- **DCTDecode**: JPEG (passthrough)
- **CCITTFaxDecode**: CCITT Group 3/4

**Design:**
```rust
pub trait StreamFilter {
    fn decode(&self, data: &[u8], params: &DecodeParms) -> Result<Vec<u8>>;
}

pub struct StreamDecoder {
    filters: Vec<Box<dyn StreamFilter>>,
}
```

**Performance:**
- Streaming decompression where possible
- Decompression bomb protection (10× ratio limit)
- Reusable buffers to minimize allocations

### 4. Layout Analysis (`src/layout/`)

**Purpose:** Understand document spatial structure.

**Submodules:**

#### 4.1 DBSCAN Clustering (`layout/dbscan.rs`)

**Algorithm:** Density-Based Spatial Clustering of Applications with Noise

**Parameters:**
- Epsilon (ε): 1.5× median character height
- MinPts: 2 for chars→words, 3 for words→lines

**Data Structure:** R*-tree for O(log n) neighborhood queries

**Performance:** O(n log n) instead of naive O(n²)

```rust
pub struct DbscanClusterer {
    spatial_index: RTree<Character>,
    epsilon: f32,
    min_points: usize,
}

impl DbscanClusterer {
    pub fn cluster(&self, chars: &[Character]) -> Vec<Cluster> {
        // DBSCAN with spatial indexing
    }
}
```

#### 4.2 XY-Cut (`layout/xy_cut.rs`)

**Algorithm:** Recursive projection-based page segmentation

**Steps:**
1. Project content onto X-axis (horizontal)
2. Find gaps in projection (valleys)
3. Split at largest gap
4. Recurse on sub-regions
5. Repeat for Y-axis

**Configuration:**
- Split threshold: 0.05 × page dimension (5%)
- Max recursion: 10 levels
- Gaussian smoothing: σ=2.0

```rust
pub struct XyCutSegmenter {
    threshold: f32,
    max_depth: u32,
}

impl XyCutSegmenter {
    pub fn segment(&self, page: &Page) -> Vec<Region> {
        self.segment_recursive(page, 0)
    }

    fn segment_recursive(&self, region: &Region, depth: u32) -> Vec<Region> {
        // Recursive XY-Cut with projection analysis
    }
}
```

#### 4.3 Font Clustering (`layout/font_clustering.rs`)

**Purpose:** Group text by font characteristics for formatting detection.

**Features:**
- Size tolerance: ±1pt
- Family exact matching
- Outlier rejection: fonts used in <2% of characters

**Use Cases:**
- Bold/italic detection
- Heading identification
- Style consistency checking

### 5. Text Extraction (`src/text.rs`)

**Purpose:** Extract text with proper spacing and formatting.

**Key Algorithms:**
- **Word spacing detection**: 0.25× character width threshold
- **Line spacing detection**: 1.2× character height threshold
- **Bold detection**: Font weight analysis
- **Reading order**: Top-to-bottom, left-to-right (with column detection)

**Quality:**
- 100% word spacing accuracy (tested on 103 PDFs)
- 100% whitespace preservation
- 37% more bold sections detected vs. reference

```rust
pub struct TextExtractor {
    config: TextConfig,
    font_cache: HashMap<String, Font>,
}

impl TextExtractor {
    pub fn extract(&mut self, page: &Page) -> Result<String> {
        // Character clustering → word formation → line formation → text
    }
}
```

### 6. Font Parser (`src/font.rs`)

**Purpose:** Parse PDF fonts and handle character encoding.

**Features:**
- **Standard fonts**: Built-in Type1 fonts (Times, Helvetica, etc.)
- **ToUnicode CMap**: Unicode mapping for custom encodings
- **Encoding tables**: PDFDocEncoding, MacRomanEncoding, WinAnsiEncoding
- **Font metrics**: Character widths, ascent, descent

**Caching:**
- Parsed fonts cached per document
- Reduces re-parsing overhead for multi-page documents
- ~25% speedup on typical documents

### 7. Image Extraction (`src/image.rs`)

**Purpose:** Extract embedded images from PDFs.

**Supported Formats:**
- JPEG (DCTDecode)
- PNG (FlateDecode + predictor)
- TIFF (CCITTFaxDecode)
- Raw RGB/CMYK

**Features:**
- Color space conversion (CMYK → RGB, etc.)
- Predictor functions (PNG filters)
- Decompression
- Metadata extraction (dimensions, DPI)

### 8. Exporters (`src/converters/`)

**Purpose:** Convert extracted content to various formats.

#### 8.1 Markdown Exporter

**Quality:** 99.8/100 (tested on 103 PDFs)

**Features:**
- Bold text detection and formatting
- Heading detection (font size heuristics)
- List detection
- Link preservation
- Proper whitespace normalization

#### 8.2 HTML Exporter

**Quality:** 94.0/100

**Features:**
- Semantic HTML5
- URL/email linkification
- Bold/italic tags
- Proper entity encoding
- CSS-friendly structure

#### 8.3 Plain Text Exporter

**Quality:** 100.0/100 (whitespace)

**Features:**
- Whitespace normalization
- Word spacing preservation
- Line break consistency

## Data Flow

### Opening a Document

```
1. User calls PdfDocument::open("file.pdf")
   ↓
2. Memory-map file (or load into buffer)
   ↓
3. Parse header, locate cross-reference table
   ↓
4. Build object index (lazy loading)
   ↓
5. Parse catalog and page tree
   ↓
6. Return PdfDocument handle
```

### Extracting Text from a Page

```
1. User calls doc.extract_text(page_num)
   ↓
2. Resolve page object from page tree
   ↓
3. Get content stream(s)
   ↓
4. Decode streams (decompress)
   ↓
5. Parse content stream operators
   ↓
6. Extract character data with positions
   ↓
7. Cluster characters into words (DBSCAN)
   ↓
8. Cluster words into lines (DBSCAN)
   ↓
9. Detect columns (XY-Cut)
   ↓
10. Sort into reading order
   ↓
11. Format text with proper spacing
   ↓
12. Return extracted text
```

### Exporting to Markdown

```
1. User calls exporter.export_all(doc)
   ↓
2. For each page:
   ├─ Extract text blocks with metadata
   ├─ Detect bold sections (font analysis)
   ├─ Detect headings (size heuristics)
   ├─ Detect lists (indentation + markers)
   ↓
3. Format as Markdown:
   ├─ Headings: # ## ###
   ├─ Bold: **text**
   ├─ Lists: - item
   ├─ Links: [text](url)
   ↓
4. Concatenate pages with separators
   ↓
5. Return Markdown string
```

## Module Organization

```
src/
├── lib.rs              # Public API, re-exports
├── error.rs            # Error types
├── object.rs           # PDF object types
├── lexer.rs            # Tokenization
├── parser.rs           # Object parsing
├── document.rs         # High-level API
├── stream_decoder.rs   # Stream decompression
├── font.rs             # Font parsing
├── text.rs             # Text extraction
├── image.rs            # Image extraction
├── form.rs             # Form field extraction
├── bookmark.rs         # Bookmark extraction
├── annotation.rs       # Annotation extraction
│
├── layout/             # Layout analysis
│   ├── mod.rs
│   ├── dbscan.rs       # DBSCAN clustering
│   ├── xy_cut.rs       # XY-Cut segmentation
│   ├── font_clustering.rs
│   └── spatial_index.rs  # R*-tree
│
├── converters/         # Export formats
│   ├── mod.rs
│   ├── markdown.rs     # Markdown exporter
│   ├── html.rs         # HTML exporter
│   └── text.rs         # Plain text exporter
│
└── utils/              # Utilities
    ├── mod.rs
    ├── encoding.rs     # Character encoding
    └── geometry.rs     # Bounding box math

python/                 # Python bindings
├── src/
│   └── lib.rs         # PyO3 bindings
└── pyproject.toml

tests/                  # Integration tests
├── test_parsing.rs
├── test_text_extraction.rs
├── test_exporters.rs
└── fixtures/          # Test PDFs

benches/               # Performance benchmarks
└── parsing.rs

examples/              # Usage examples
├── basic.rs
├── batch_processing.rs
└── custom_exporter.rs
```

## Algorithm Details

### DBSCAN Clustering

**Pseudo-code:**
```
function dbscan(points, epsilon, minPts):
    clusters = []
    visited = set()

    for point in points:
        if point in visited:
            continue

        visited.add(point)
        neighbors = find_neighbors(point, epsilon)  # O(log n) with R*-tree

        if len(neighbors) < minPts:
            # Noise point
            continue

        # Start new cluster
        cluster = [point]
        queue = neighbors

        while queue:
            neighbor = queue.pop()
            if neighbor not in visited:
                visited.add(neighbor)
                new_neighbors = find_neighbors(neighbor, epsilon)

                if len(new_neighbors) >= minPts:
                    queue.extend(new_neighbors)

            if neighbor not in any cluster:
                cluster.append(neighbor)

        clusters.append(cluster)

    return clusters
```

**Optimization:** R*-tree spatial index allows O(log n) neighbor queries instead of O(n) linear scan.

### XY-Cut Segmentation

**Pseudo-code:**
```
function xy_cut(region, depth):
    if depth > MAX_DEPTH:
        return [region]

    # Project onto X-axis
    x_projection = project_horizontal(region)
    x_gaps = find_gaps(x_projection, threshold)

    if x_gaps:
        split_x = largest_gap(x_gaps)
        left, right = split_region(region, split_x, vertical=True)
        return xy_cut(left, depth+1) + xy_cut(right, depth+1)

    # Project onto Y-axis
    y_projection = project_vertical(region)
    y_gaps = find_gaps(y_projection, threshold)

    if y_gaps:
        split_y = largest_gap(y_gaps)
        top, bottom = split_region(region, split_y, vertical=False)
        return xy_cut(top, depth+1) + xy_cut(bottom, depth+1)

    # No split found
    return [region]
```

### Word Spacing Detection

**Algorithm:**
```
For each pair of adjacent characters (c1, c2):
    gap = c2.x - (c1.x + c1.width)
    threshold = 0.25 * c1.width

    if gap > threshold:
        insert_space()
```

**Dynamic threshold:** Uses character width for adaptive spacing detection.

## Performance Design

### Zero-Copy Parsing

**Technique:** Use byte slices (`&[u8]`) instead of copying data.

**Example:**
```rust
// Zero-copy token
pub enum Token<'a> {
    Name(&'a [u8]),
    String(&'a [u8]),
    // ...
}

// No allocation needed
let token = lexer.next_token()?;
```

### Memory-Mapped Files

**Technique:** Use `mmap` for large files.

**Benefits:**
- OS handles memory management
- Lazy loading (only accessed pages loaded)
- Shared memory across processes

### Object Caching

**Strategy:** Cache parsed objects by object ID.

**Benefits:**
- Avoid re-parsing referenced objects
- O(1) lookup for cross-references
- Memory bounded by unique object count

### Streaming

**Technique:** Process pages one at a time.

**Benefits:**
- Constant memory regardless of document size
- Can process 10 GB documents with <100 MB RAM

## Extension Points

### Custom Exporters

Implement the `Exporter` trait:

```rust
pub trait Exporter {
    fn export_page(&self, doc: &mut PdfDocument, page_num: u32) -> Result<String>;
    fn export_all(&self, doc: &mut PdfDocument) -> Result<String>;
}
```

### Custom Stream Filters

Implement the `StreamFilter` trait:

```rust
pub trait StreamFilter {
    fn decode(&self, data: &[u8], params: &DecodeParms) -> Result<Vec<u8>>;
}
```

### Custom Layout Analyzers

Replace or extend layout analysis:

```rust
pub trait LayoutAnalyzer {
    fn analyze(&self, page: &Page) -> Result<Layout>;
}
```

## Security Considerations

### Input Validation

**PDF as untrusted input:**
- Limit file size (default: 500 MB)
- Limit object count (default: 1M objects)
- Limit recursion depth (default: 100 levels)
- Validate array/dictionary sizes

### Decompression Bombs

**Protection:**
- Limit decompressed size to 10× compressed size
- Check compression ratio before full decompression
- Stream decompression where possible

### Memory Safety

**Rust guarantees:**
- No buffer overflows
- No use-after-free
- No data races
- Bounds checking on all array accesses

**Minimal `unsafe`:**
- Used sparingly (e.g., performance-critical paths)
- Thoroughly documented with safety invariants
- Reviewed by multiple developers

## Future Architecture

### Planned Improvements

**v1.x:**
- Stream API for ultra-low memory usage
- Table detection (smart heuristics)
- Rotated text handling

**v2.x:**
- ML-based layout analysis (optional)
- GPU acceleration (optional)
- Parallel page processing

**v3.x:**
- OCR integration (Tesseract)
- PDF/A validation
- Advanced form handling

## References

- **PDF Specification**: ISO 32000-1:2008 (PDF 1.7) - `docs/spec/pdf.md`
- **Architecture**: ARCHITECTURE.md - Detailed system design
- **DBSCAN**: Ester et al., "A Density-Based Algorithm for Discovering Clusters" (1996)
- **XY-Cut**: Nagy & Seth, "Hierarchical Representation of Optically Scanned Documents" (1984)
- **R*-tree**: Beckmann et al., "The R*-tree: An Efficient and Robust Access Method" (1990)

## Questions?

- **GitHub Issues**: https://github.com/yfedoseev/pdf_oxide/issues
- **Discussions**: https://github.com/yfedoseev/pdf_oxide/discussions
- **Documentation**: https://docs.rs/pdf_oxide
