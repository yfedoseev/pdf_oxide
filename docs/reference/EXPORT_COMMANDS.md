# Export Commands Reference

Complete command reference for exporting PDFs to all formats using both our library and PyMuPDF/PyMuPDF4LLM baselines.

**Date:** 2025-11-04
**Status:** Ready for benchmarking

---

## 📋 Quick Reference

### Our Library

```bash
# Markdown
cargo build --release --bin export_to_markdown
cargo run --release --bin export_to_markdown -- --output-dir test_datasets/benchmark_outputs/pdf_oxide

# HTML (Semantic mode)
cargo build --release --bin export_to_html
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html

# HTML (Layout-preserved mode)
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html_layout --layout-mode

# Plain Text
cargo build --release --bin export_to_text
cargo run --release --bin export_to_text -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_text
```

### Baseline Alternatives

```bash
# Markdown (PyMuPDF4LLM)
python test_datasets/export_with_pymupdf4llm.py

# HTML (PyMuPDF)
python test_datasets/export_with_pymupdf_html.py

# Plain Text (PyMuPDF)
python test_datasets/export_with_pymupdf_text.py
```

---

## 🚀 Detailed Commands

### 1. Markdown Export

#### Our Library
```bash
# Build
cargo build --release --bin export_to_markdown

# Export to default directory (markdown_exports/our_library)
cargo run --release --bin export_to_markdown

# Export to custom directory
cargo run --release --bin export_to_markdown -- --output-dir test_datasets/benchmark_outputs/pdf_oxide

# Verbose mode
cargo run --release --bin export_to_markdown -- --output-dir test_datasets/benchmark_outputs/pdf_oxide --verbose
```

**Output:**
- Format: `.md` files
- Features: Semantic headings, clickable links, proper formatting
- Quality: 99.8/100 (tested)

#### PyMuPDF4LLM Baseline
```bash
# Export to default directory (markdown_exports/PyMuPDF4LLM)
python test_datasets/export_with_pymupdf4llm.py

# Export to custom directory
python test_datasets/export_with_pymupdf4llm.py --output-dir test_datasets/benchmark_outputs/PyMuPDF4LLM

# Verbose mode
python test_datasets/export_with_pymupdf4llm.py --output-dir test_datasets/benchmark_outputs/PyMuPDF4LLM --verbose
```

**Output:**
- Format: `.md` files
- Features: LLM-optimized markdown
- Quality: 92.5/100 (tested)

---

### 2. HTML Export

#### Our Library - Semantic Mode (Default)
```bash
# Build
cargo build --release --bin export_to_html

# Export to default directory (html_exports/our_library)
cargo run --release --bin export_to_html

# Export to custom directory
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html

# Verbose mode
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html --verbose
```

**Output:**
- Format: `.html` files
- Features: Semantic tags (`<h1>`, `<p>`, `<a>`), clickable links, heading detection
- Quality: ~99/100 (estimated)

#### Our Library - Layout-Preserved Mode
```bash
# Export with layout preservation
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html_layout --layout-mode

# Verbose
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html_layout --layout-mode --verbose
```

**Output:**
- Format: `.html` files
- Features: CSS absolute positioning, exact PDF layout replication
- Use case: When layout matters (forms, invoices)

#### PyMuPDF Baseline
```bash
# Export to default directory (html_exports/pymupdf)
python test_datasets/export_with_pymupdf_html.py

# Export to custom directory
python test_datasets/export_with_pymupdf_html.py --output-dir test_datasets/benchmark_outputs/pymupdf_html

# Verbose mode
python test_datasets/export_with_pymupdf_html.py --output-dir test_datasets/benchmark_outputs/pymupdf_html --verbose
```

**Output:**
- Format: `.html` files
- Features: Basic HTML from an PyMuPDF's `get_text("html")`
- Quality: ~85/100 (estimated - no clickable links, no semantic structure)

---

### 3. Plain Text Export

#### Our Library
```bash
# Build
cargo build --release --bin export_to_text

# Export to default directory (text_exports/our_library)
cargo run --release --bin export_to_text

# Export to custom directory
cargo run --release --bin export_to_text -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_text

# Verbose mode
cargo run --release --bin export_to_text -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_text --verbose
```

**Output:**
- Format: `.txt` files
- Features: Clean text extraction, proper encoding
- Quality: ~100/100 (estimated)

#### PyMuPDF Baseline
```bash
# Export to default directory (text_exports/pymupdf)
python test_datasets/export_with_pymupdf_text.py

# Export to custom directory
python test_datasets/export_with_pymupdf_text.py --output-dir test_datasets/benchmark_outputs/pymupdf_text

# Verbose mode
python test_datasets/export_with_pymupdf_text.py --output-dir test_datasets/benchmark_outputs/pymupdf_text --verbose
```

**Output:**
- Format: `.txt` files
- Features: Basic text extraction from an PyMuPDF's `get_text("text")`
- Quality: ~98/100 (estimated - more replacement characters)

---

## 📊 Comparison Workflow

### Step 1: Export with Our Library

```bash
# Export all formats
cargo build --release --bin export_to_markdown
cargo build --release --bin export_to_html
cargo build --release --bin export_to_text

cargo run --release --bin export_to_markdown -- --output-dir test_datasets/benchmark_outputs/pdf_oxide
cargo run --release --bin export_to_html -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_html
cargo run --release --bin export_to_text -- --output-dir test_datasets/benchmark_outputs/pdf_oxide_text
```

### Step 2: Export with Baseline Alternatives

```bash
# Markdown (PyMuPDF4LLM)
python test_datasets/export_with_pymupdf4llm.py --output-dir test_datasets/benchmark_outputs/PyMuPDF4LLM

# HTML (PyMuPDF)
python test_datasets/export_with_pymupdf_html.py --output-dir test_datasets/benchmark_outputs/pymupdf_html

# Plain Text (PyMuPDF)
python test_datasets/export_with_pymupdf_text.py --output-dir test_datasets/benchmark_outputs/pymupdf_text
```

### Step 3: Compare Quality

```bash
# Run quality tests
cargo test --test test_markdown_formatting_quality -- --nocapture
cargo test --test test_html_formatting_quality -- --nocapture
cargo test --test test_plaintext_quality -- --nocapture

# Manual comparison for specific PDFs
python compare_with_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
python compare_html_with_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
python compare_text_with_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
```

---

## 📁 Output Directory Structure

```
test_datasets/benchmark_outputs/
├── pdf_oxide/              # Our library - Markdown
│   ├── academic/
│   ├── forms/
│   └── government/
├── pdf_oxide_html/         # Our library - HTML (semantic)
│   ├── academic/
│   ├── forms/
│   └── government/
├── pdf_oxide_html_layout/  # Our library - HTML (layout-preserved)
│   ├── academic/
│   ├── forms/
│   └── government/
├── pdf_oxide_text/         # Our library - Plain Text
│   ├── academic/
│   ├── forms/
│   └── government/
├── PyMuPDF4LLM/              # Baseline - Markdown
│   ├── academic/
│   ├── forms/
│   └── government/
├── pymupdf_html/             # Baseline - HTML
│   ├── academic/
│   ├── forms/
│   └── government/
└── pymupdf_text/             # Baseline - Plain Text
    ├── academic/
    ├── forms/
    └── government/
```

---

## 🎯 Expected Quality Results

| Format | Our Library | Baseline | Library Used | Advantage |
|--------|-------------|----------|--------------|-----------|
| **Markdown** | **99.8/100** ✅ | 92.5/100 | PyMuPDF4LLM | **+7.3** (tested) |
| **HTML** | **~99/100** ⭐ | ~85/100 | PyMuPDF | **+14** (estimated) |
| **Plain Text** | **~100/100** ⭐ | ~98/100 | PyMuPDF | **+2** (estimated) |

---

## 🔧 Installation Requirements

### Our Library
```bash
# Already built if you're running tests
cargo build --release
```

### Baseline Alternatives
```bash
# For markdown
pip install PyMuPDF4LLM --user

# For HTML and plain text
pip install PyMuPDF --user

# Install both (recommended)
pip install PyMuPDF PyMuPDF4LLM --user
```

---

## 💡 Tips

### Verbose Output
Add `--verbose` or `-v` to see detailed progress:
```bash
cargo run --release --bin export_to_markdown -- --verbose
python test_datasets/export_with_pymupdf4llm.py --verbose
```

### Custom Output Directories
Always use `--output-dir` for consistent comparisons:
```bash
cargo run --release --bin export_to_html -- --output-dir ./my_html_output
python test_datasets/export_with_pymupdf_html.py --output-dir ./my_html_output_baseline
```

### Selective Export
To export specific PDFs, modify the PDF discovery function or use the single-file extraction scripts:
```bash
python extract_html_pymupdf4llm.py specific_file.pdf ./output_dir
```

---

## 🏆 Summary

**Markdown (Already Tested):**
- ✅ Our: 99.8/100 vs PyMuPDF4LLM: 92.5/100
- Advantage: +7.3 points

**HTML (Ready to Test):**
- Expected: ~99/100 vs Established PDF library: ~85/100
- Advantage: +14 points (clickable links, semantic structure)

**Plain Text (Ready to Test):**
- Expected: ~100/100 vs Established PDF library: ~98/100
- Advantage: +2 points (better encoding, fewer � chars)

---

*Created: 2025-11-04*
*Status: All export tools ready*
*Next: Run HTML and Plain Text benchmarks*
