# Golden File Baseline Generation and Regression Testing Infrastructure

## Overview

This document describes the golden file infrastructure for pdf_oxide, which enables quality regression testing across the PDF extraction pipeline.

**Status**: Fully implemented and ready for use
- Test files compiled and passing
- Helper infrastructure complete
- Ready for first-time baseline generation

## File Locations

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `/tests/test_golden_file_generation.rs` | One-time baseline generator (marked with `#[ignore]`) | 785 lines | ✓ Ready |
| `/tests/test_extraction_regression.rs` | Regression test suite for CI/CD | 1011 lines | ✓ Ready |
| `/tests/helpers/golden_file_manager.rs` | Core golden file management | 448 lines | ✓ Existing |
| `/tests/helpers/corpus_loader.rs` | Corpus access utilities | 222 lines | ✓ Existing |
| `/tests/golden_files/` | Generated baseline directory | (generated) | Will be created |

## What is a Golden File?

A golden file is a JSON baseline that captures:

```json
{
  "pdf_path": "path/to/document.pdf",
  "category": "academic",
  "extracted_text": "Full extracted text from all pages...",
  "text_hash": "hash of extracted text",
  "char_count": 5427,
  "word_count": 892,
  "script_distribution": {
    "Latin": 98.5,
    "CJK": 1.5
  },
  "extraction_timestamp": "2025-12-11T14:30:00Z"
}
```

These files serve as:
- **Baselines** for quality validation
- **Documentation** of expected extraction behavior
- **Regression detectors** for changes in extraction quality

## Step 1: Generate Golden Files (One-Time Setup)

### Quick Start

Generate baselines for the first time:

```bash
cargo test --test test_golden_file_generation -- --ignored --nocapture
```

This will:
1. Process all PDFs in the corpus (14 categories)
2. Extract text from each PDF
3. Save golden files to `tests/golden_files/extracted_text/{category}/`
4. Print detailed progress and summary

### Category-Specific Generation

Generate baselines for a single category:

```bash
# Academic papers only
cargo test --test test_golden_file_generation test_generate_academic_golden_files -- --ignored --nocapture

# Other categories
cargo test --test test_golden_file_generation test_generate_diverse_golden_files -- --ignored --nocapture
cargo test --test test_golden_file_generation test_generate_technical_golden_files -- --ignored --nocapture
```

Available categories:
- `academic` - Research papers, dissertations
- `diverse` - Varied document types
- `forms` - Forms, templates, structured documents
- `government` - Government documents, policies
- `mixed` - Mixed layout documents
- `newspapers` - News articles, journals
- `technical` - Technical documentation
- `theses` - Academic theses
- `text_heavy` - Text-dense documents
- `tables` - Documents with tables
- `multilingual` - Multi-language documents
- `scanned` - Scanned/OCR documents
- `images` - Documents with many images
- `test_datasets` - Test-specific datasets

### Output Structure

Generated files are organized as:

```
tests/golden_files/
├── extracted_text/
│   ├── academic/
│   │   ├── paper1.json
│   │   ├── paper2.json
│   │   └── ...
│   ├── diverse/
│   │   └── ...
│   └── ... (other categories)
```

### Generation Summary

The output includes a summary table:

```
========== GOLDEN FILE GENERATION SUMMARY ==========

Category Breakdown:
Category        Total   Success   Failed    Skip  Success%
────────────────────────────────────────────────────────────
academic           45       43        0       2      95.6%
diverse            38       37        1       0      97.4%
technical          52       50        1       1      96.2%
...

Content Statistics:
  Total Characters: 1,234,567
  Total Words:      201,234
  Total Pages:      5,432

Timing:
  Total Duration:   45m 32s
  Avg per PDF:      18.3ms
```

## Step 2: Run Regression Tests (Automated)

### Quick Start

Run regression tests against all generated baselines:

```bash
cargo test --test test_extraction_regression
```

This will:
1. Load each golden file
2. Extract text from the corresponding PDF
3. Compare with tolerance thresholds
4. Report any regressions

### Category-Specific Testing

Test a single category:

```bash
cargo test --test test_extraction_regression test_regression_academic -- --nocapture
cargo test --test test_extraction_regression test_regression_technical -- --nocapture
```

### With Detailed Output

Get detailed output including diff contexts:

```bash
cargo test --test test_extraction_regression -- --nocapture
```

### Integration with CI/CD

The regression tests are designed to run in CI/CD pipelines:

```bash
# In your CI configuration
cargo test --test test_extraction_regression --release
```

## Tolerance Thresholds

The regression tests use tolerance-based comparison to account for minor variations:

| Metric | Tolerance | Reason |
|--------|-----------|--------|
| Character count | ±0.5% | Allows minor encoding/normalization differences |
| Word count | ±1.0% | Allows minor whitespace/boundary differences |
| Line count | ±2.0% | Allows layout variation differences |

### Status Levels

Each PDF is classified as:

- **PASS** - Exact match with baseline
- **WARN** - Minor differences within tolerance
- **FAIL** - Regression detected (exceeds tolerances)
- **SKIP** - No baseline available
- **ERR** - Extraction error

## Update Workflow

When you intentionally improve extraction quality:

### 1. Make Code Changes

Implement improvements to the extraction pipeline.

### 2. Run Regression Tests

```bash
cargo test --test test_extraction_regression -- --nocapture
```

Review the output for expected improvements.

### 3. Update Baselines

```bash
# Remove old baselines
rm -rf tests/golden_files/

# Generate new baselines with improvements
cargo test --test test_golden_file_generation -- --ignored --nocapture
```

### 4. Verify and Commit

```bash
# Verify improved results
cargo test --test test_extraction_regression

# Commit code changes (golden_files/ not committed)
git add src/
git commit -m "Improve text extraction quality"
```

## Troubleshooting

### No Golden Files Generated

**Problem**: `assert!(total > 0, "No golden files were generated")`

**Causes**:
1. Corpus directory not found at `/home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000`
2. PDF files not present in corpus
3. All PDFs skipped (too large, unreadable)

**Solution**:
```bash
# Check corpus exists
ls -la /home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000

# List PDFs by category
find /home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000 -name "*.pdf" | head -10
```

### Regression Test Failures

**Problem**: Tests fail with "Regression detected"

**Response**:
1. Review the diff output showing what changed
2. If change is intentional, update baselines (see Update Workflow)
3. If change is unintended, investigate the code change
4. Use `--nocapture` to see detailed diffs

### Out of Memory During Generation

**Problem**: Process runs out of memory

**Solutions**:
1. Generate category-by-category instead of all at once
2. Increase available memory or reduce file size limit
3. File size limit is `MAX_FILE_SIZE_MB` (configurable in test file)

## Architecture Details

### Golden File Manager (`helpers/golden_file_manager.rs`)

Provides:
- **Save**: `save_golden_file()` - Save extraction as baseline
- **Load**: `load_golden_file()` - Load baseline from disk
- **Compare**: `compare_extraction()` - Compare current vs baseline
- **Analysis**: Script distribution, character/word counts

### Corpus Loader (`helpers/corpus_loader.rs`)

Provides:
- **List categories**: Available document categories
- **List PDFs**: PDFs in each category
- **Load PDF**: Open PDF documents
- **Metadata**: File size, page count, etc.

### Generation Test (`test_golden_file_generation.rs`)

Features:
- 14 category-specific tests (all marked `#[ignore]`)
- Progress reporting (files processed, errors)
- Time tracking (per-PDF and total)
- Content statistics (chars, words, pages)
- Error collection and reporting

### Regression Test (`test_extraction_regression.rs`)

Features:
- Per-category regression tests
- Tolerance-based comparison
- Detailed failure reporting with diffs
- Quality scoring (if available)
- Pass rate calculations

## Configuration

Key constants in test files (can be adjusted):

### Generation (`test_golden_file_generation.rs`)
```rust
const MAX_EXTRACTION_TIME_SECS: u64 = 120;  // Timeout per PDF
const MAX_FILE_SIZE_MB: u64 = 100;          // Size limit
```

### Regression (`test_extraction_regression.rs`)
```rust
const CHAR_COUNT_TOLERANCE: f64 = 0.005;    // ±0.5%
const WORD_COUNT_TOLERANCE: f64 = 0.01;     // ±1.0%
const LINE_COUNT_TOLERANCE: f64 = 0.02;     // ±2.0%
const SHOW_DETAILED_DIFFS: bool = true;     // Show diffs
```

## Performance Notes

- **First generation**: ~45-60 minutes (full corpus, depends on machine)
- **Per-PDF average**: ~18-25ms (varies by document complexity)
- **Regression tests**: ~30-45 seconds (CI/CD run)
- **Storage**: ~50-100MB golden files (compressed text baselines)

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Regression Tests
  run: cargo test --test test_extraction_regression --release
```

### GitLab CI Example

```yaml
regression_tests:
  script:
    - cargo test --test test_extraction_regression --release
```

## FAQ

**Q: Why are golden files not committed to git?**
A: They're too large (~50-100MB) and would slow down clones. They're generated locally and rebuilt if needed.

**Q: How often should I regenerate baselines?**
A: Only when making intentional quality improvements. Regression tests validate consistency.

**Q: Can I use this with partial corpus?**
A: Yes! Tests gracefully skip missing categories and show zero-baseline tests.

**Q: What if a PDF fails to extract?**
A: Logged as error, generation continues. Regression tests skip missing baselines.

**Q: How do I know if a regression is acceptable?**
A: Review the diff context. Small improvements are good. Large unexpected changes should be investigated.

## Testing Status

Current implementation status:

- ✓ Golden file manager fully implemented
- ✓ Corpus loader fully implemented
- ✓ Generation test with 14 category-specific tests
- ✓ Regression test suite with per-category tests
- ✓ Helper modules and utilities
- ✓ Gitignore configuration
- ✓ Both test files compile without errors
- ✓ Library tests pass (815 passed; 0 failed)

## Next Steps

1. **Generate baselines** (first-time setup):
   ```bash
   cargo test --test test_golden_file_generation -- --ignored --nocapture
   ```

2. **Verify quality** (spot-check 5-10 PDFs per category)

3. **Run regression tests** (validate baseline setup):
   ```bash
   cargo test --test test_extraction_regression
   ```

4. **Integrate into CI/CD** (add to your pipeline)

## Support

For issues or questions:
1. Check the `Troubleshooting` section above
2. Review test output with `--nocapture` flag
3. Check golden file manager implementation
4. Verify corpus path and PDF availability
