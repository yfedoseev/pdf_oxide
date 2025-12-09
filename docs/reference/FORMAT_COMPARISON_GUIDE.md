# Format Comparison Guide: HTML, Markdown, and Plain Text

This guide explains how to test and compare all three output formats (HTML, Markdown, Plain Text) between our PDF library and PyMuPDF/PyMuPDF4LLM baseline.

**Date:** 2025-11-04
**Status:** Ready for testing

---

## 📋 Overview

We now have comprehensive test suites and comparison tools for all three conversion formats:

1. **Markdown** - Semantic text with clickable links (already tested: 99.8/100)
2. **HTML** - Both semantic and layout-preserved modes (new tests)
3. **Plain Text** - Simple text extraction (new tests)

---

## 🧪 Test Suites

### 1. HTML Quality Tests

**File:** `tests/test_html_formatting_quality.rs`

**Tests:**
- `test_html_url_links_clickable` - Verifies URLs are `<a href>` links
- `test_html_email_links_clickable` - Verifies emails are `mailto:` links
- `test_html_semantic_structure` - Checks for proper `<h1>`, `<p>`, `<a>` tags
- `test_html_layout_preservation` - Validates CSS positioning in layout mode
- `test_overall_html_quality_score` - Comprehensive quality scoring (0-100)
- `test_html_link_format_differences` - Compares HTML vs Markdown link syntax
- `test_full_document_html_conversion` - Benchmark for full documents

**Run tests:**
```bash
# All HTML tests
cargo test --test test_html_formatting_quality -- --nocapture

# Specific test
cargo test --test test_html_formatting_quality test_html_url_links_clickable -- --nocapture

# With benchmark
cargo test --test test_html_formatting_quality --ignored -- --nocapture
```

### 2. Plain Text Quality Tests

**File:** `tests/test_plaintext_quality.rs`

**Tests:**
- `test_plaintext_url_preservation` - Verifies URLs are preserved
- `test_plaintext_email_preservation` - Verifies emails are preserved
- `test_plaintext_completeness` - Checks for complete content extraction
- `test_plaintext_encoding_quality` - Validates no replacement characters (�)
- `test_plaintext_reading_order` - Ensures proper reading order
- `test_overall_plaintext_quality_score` - Comprehensive quality scoring (0-100)
- `test_plaintext_vs_markdown_equivalence` - Compares with markdown content
- `test_full_document_plaintext_extraction` - Benchmark for full documents

**Run tests:**
```bash
# All plain text tests
cargo test --test test_plaintext_quality -- --nocapture

# Specific test
cargo test --test test_plaintext_quality test_plaintext_url_preservation -- --nocapture

# With benchmark
cargo test --test test_plaintext_quality --ignored -- --nocapture
```

### 3. Markdown Quality Tests (Existing)

**File:** `tests/test_markdown_formatting_quality.rs`

**Already tested - Quality Score: 99.8/100** ✅

```bash
cargo test --test test_markdown_formatting_quality -- --nocapture
```

---

## 🔬 PyMuPDF Baseline Extraction Scripts

### 1. HTML Extraction Baseline

**File:** `extract_html_pymupdf4llm.py`

**Usage:**
```bash
# Extract HTML from single PDF
python3 extract_html_pymupdf4llm.py test.pdf

# Extract and save to directory
python3 extract_html_pymupdf4llm.py test.pdf ./html_output

# Example with our test PDF
python3 extract_html_pymupdf4llm.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf \
  ./pymupdf_html_baseline
```

**Note:** Uses an PyMuPDF's `get_text("html")` method for HTML export.

### 2. Plain Text Extraction Baseline

**File:** `extract_text_pymupdf.py`

**Usage:**
```bash
# Extract plain text from single PDF
python3 extract_text_pymupdf.py test.pdf

# Extract and save to directory
python3 extract_text_pymupdf.py test.pdf ./text_output

# Example with our test PDF
python3 extract_text_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf \
  ./pymupdf_text_baseline
```

**Note:** Uses an PyMuPDF's `get_text("text")` method for plain text extraction.

### 3. Markdown Extraction Baseline (Existing)

**File:** `compare_with_pymupdf.py`

**Already implemented** ✅

---

## 📊 Comparison Scripts

### 1. HTML Quality Comparison

**File:** `compare_html_with_pymupdf.py`

**Usage:**
```bash
# Compare HTML quality for a PDF
python3 compare_html_with_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
```

**What it does:**
1. Extracts HTML using PyMuPDF 2. Analyzes HTML quality (clickable links, semantic structure, encoding)
3. Scores quality 0-100
4. Shows comparison template for our library's output

**Metrics:**
- Clickable URL links (20 points)
- Clickable mailto: links (20 points)
- Semantic structure (20 points)
- No garbled text (20 points)
- No replacement characters (20 points)

### 2. Plain Text Quality Comparison

**File:** `compare_text_with_pymupdf.py`

**Usage:**
```bash
# Compare plain text quality for a PDF
python3 compare_text_with_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
```

**What it does:**
1. Extracts plain text using PyMuPDF 2. Analyzes text quality (completeness, encoding, URLs, emails)
3. Scores quality 0-100
4. Saves baseline output for comparison

**Metrics:**
- Text completeness (25 points)
- No replacement characters (25 points)
- URLs preserved (25 points)
- Emails preserved (25 points)

### 3. Markdown Quality Comparison (Existing)

**File:** `compare_with_pymupdf.py`

**Already implemented and tested** ✅

**Result:** PDF Library 99.8/100 vs PyMuPDF4LLM 92.5/100 (+7.3 points)

---

## 🚀 Complete Workflow

### Step 1: Run Our Library's Tests

```bash
# Test HTML conversion
cargo test --test test_html_formatting_quality -- --nocapture > html_test_results.txt

# Test Plain Text conversion
cargo test --test test_plaintext_quality -- --nocapture > plaintext_test_results.txt

# Test Markdown conversion (already done)
cargo test --test test_markdown_formatting_quality -- --nocapture > markdown_test_results.txt
```

### Step 2: Extract PyMuPDF Baselines

```bash
# Create baseline directories
mkdir -p baseline_outputs/{html,text,markdown}

# Extract HTML baseline
python3 extract_html_pymupdf4llm.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf \
  baseline_outputs/html

# Extract Plain Text baseline
python3 extract_text_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf \
  baseline_outputs/text
```

### Step 3: Compare Quality

```bash
# Compare HTML quality
python3 compare_html_with_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf

# Compare Plain Text quality
python3 compare_text_with_pymupdf.py \
  test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
```

### Step 4: Analyze Results

Compare the quality scores from each format:

```
Expected Results (based on Markdown):
- HTML Quality: ~99/100 (similar to Markdown)
- Plain Text Quality: ~100/100 (simplest, most reliable)
- Markdown Quality: 99.8/100 (already verified) ✅
```

---

## 📈 Quality Metrics

### Scoring Criteria

**HTML (100 points):**
- Clickable URLs: 20 pts
- Clickable emails: 20 pts
- Semantic structure (`<h1>`, `<p>`, `<a>`): 20 pts
- No garbled text: 20 pts
- No replacement chars (�): 20 pts

**Plain Text (100 points):**
- Text completeness (>1000 chars): 25 pts
- No replacement chars (�): 25 pts
- URLs preserved: 25 pts
- Emails preserved: 25 pts

**Markdown (100 points):** *(already tested)*
- Clickable URLs: 20 pts
- Clickable emails: 20 pts
- Proper formatting (##, **): 20 pts
- No garbled text: 20 pts
- No replacement chars (�): 20 pts

---

## 🎯 Expected Quality Comparison

Based on our Markdown results (99.8/100 vs PyMuPDF4LLM's 92.5/100):

| Format | PDF Library | PyMuPDF Baseline | Advantage |
|--------|-------------|------------------|-----------|
| **Markdown** | **99.8/100** ✅ | 92.5/100 | **+7.3** (tested) |
| **HTML** | **~99/100** ⭐ | ~90/100 | **+9** (estimated) |
| **Plain Text** | **~100/100** ⭐ | ~98/100 | **+2** (estimated) |

**Reasoning:**
- HTML: Similar to Markdown (clickable links, semantic structure)
- Plain Text: Simpler extraction, fewer formatting issues
- All formats benefit from our superior encoding quality (89.3% vs 62.4% clean extraction)

---

## 🏆 Key Differentiators

### Our Library's Strengths

1. **Clickable Links** (HTML & Markdown)
   - URLs: `<a href="https://...">` or `[text](url)`
   - Emails: `<a href="mailto:...">` or `[email](mailto:...)`

2. **Layout Preservation** (HTML)
   - Semantic mode: `<h1>`, `<p>`, `<a>` tags
   - Layout mode: CSS positioning to match PDF

3. **Clean Encoding** (All Formats)
   - 89.3% clean extraction vs an PyMuPDF's 62.4%
   - Fewer � replacement characters
   - Better Unicode support

4. **Reading Order** (All Formats)
   - Left-to-right, top-to-bottom
   - Column-aware detection
   - PDF structure tree support

### an PyMuPDF's Approach

1. **HTML:** Basic `get_text("html")` with minimal semantic structure
2. **Plain Text:** Basic `get_text("text")` extraction
3. **Markdown:** PyMuPDF4LLM with column-by-column extraction (breaks layout)

---

## 📝 Usage Examples

### Extract HTML from Our Library

```rust
use pdf_oxide::PdfDocument;
use pdf_oxide::converters::ConversionOptions;

let mut doc = PdfDocument::open("paper.pdf")?;

// Semantic HTML
let options = ConversionOptions::default();
let html = doc.to_html(0, &options)?;

// Layout-preserved HTML
let layout_options = ConversionOptions {
    preserve_layout: true,
    ..Default::default()
};
let layout_html = doc.to_html(0, &layout_options)?;
```

### Extract Plain Text from Our Library

```rust
use pdf_oxide::PdfDocument;
use pdf_oxide::converters::ConversionOptions;

let mut doc = PdfDocument::open("paper.pdf")?;
let options = ConversionOptions::default();

// Single page
let text = doc.to_plain_text(0, &options)?;

// All pages
let all_text = doc.to_plain_text_all(&options)?;
```

---

## 🔍 Troubleshooting

### Tests Fail: PDF Not Found

```bash
# Check if test PDFs exist
ls -la test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf

# Download if missing (or use your own PDFs)
```

### PyMuPDF Not Installed

```bash
# Install PyMuPDF pip install PyMuPDF --user

# Or system-wide (may require sudo)
pip install PyMuPDF
```

### Comparison Script Errors

Make sure both libraries are installed:
```bash
pip install PyMuPDF PyMuPDF4LLM --user
```

---

## 📚 References

- **Markdown Comparison:** `PYMUPDF4LLM_COMPARISON.md` (99.8 vs 92.5)
- **Dense Grid Analysis:** `DENSE_GRID_COMPARISON.md`
- **Quick Reference:** `QUICK_REFERENCE_COMPARISON.md`
- **Fix #1 Summary:** `FIX1_FINAL_EXECUTIVE_SUMMARY.md`

---

**Created:** 2025-11-04
**Status:** Ready for comprehensive format testing
**Next Steps:** Run HTML and Plain Text tests to complete the comparison

---

## 🎯 Quick Start

```bash
# 1. Run all format tests
cargo test --test test_html_formatting_quality -- --nocapture
cargo test --test test_plaintext_quality -- --nocapture
cargo test --test test_markdown_formatting_quality -- --nocapture

# 2. Extract PyMuPDF baselines
python3 extract_html_pymupdf4llm.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf baseline_outputs/html
python3 extract_text_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf baseline_outputs/text

# 3. Compare quality
python3 compare_html_with_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf
python3 compare_text_with_pymupdf.py test_datasets/pdfs/academic/arxiv_2510.25332v1.pdf

# Done! 🎉
```
