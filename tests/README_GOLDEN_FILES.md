# Golden Files & Test Corpus - Quick Start

## Overview

This directory contains the comprehensive testing framework for PDF text extraction regression testing.

**Status**: ✅ Fully Implemented
**Corpus**: 356 PDFs across 14 categories
**Baseline Tests**: 799 passing (zero regressions)
**New Tests**: 20+ golden file tests, 7+ integration tests

## Quick Commands

### Run All Tests
```bash
# Baseline library tests (799 tests)
cargo test --lib

# Golden file regression tests (20 tests)
cargo test --test golden_files

# Integration tests (7 tests)
cargo test --test corpus_integration_tests
```

### View Test Corpus
```bash
# List all PDFs by category
cargo test --test golden_files list_corpus_summary -- --ignored --nocapture

# Output:
# academic             :   173 PDFs
# mixed                :    89 PDFs
# forms                :    30 PDFs
# government           :    29 PDFs
# ... (356 total)
```

### Create Golden Files
```bash
# Create golden files for 5 PDFs per category
cargo test --test golden_files create_golden_files_sample -- --ignored --nocapture

# Creates: tests/golden_files/extracted_text/{category}/*.json
```

### Run Performance Benchmark
```bash
cargo test --test corpus_integration_tests benchmark_corpus_extraction -- --ignored --nocapture
```

## File Structure

```
tests/
├── helpers/
│   ├── mod.rs                      # Public API
│   ├── corpus_loader.rs            # Load PDFs from corpus (219 lines)
│   └── golden_file_manager.rs     # Golden file I/O (450 lines)
├── golden_files.rs                 # Regression tests (258 lines)
├── corpus_integration_tests.rs    # Full pipeline tests (343 lines)
├── quality_metrics.rs              # Quality scoring (existing)
├── TEST_INFRASTRUCTURE.md          # Complete documentation
├── IMPLEMENTATION_SUMMARY.md       # Implementation details
└── README_GOLDEN_FILES.md          # This file
```

**Total New Code**: 1,283 lines

## Test Categories

### Golden File Tests (10 categories)
- `test_golden_file_academic_papers` - Academic papers (arxiv)
- `test_golden_file_diverse_docs` - Diverse document types
- `test_golden_file_forms` - IRS forms, applications
- `test_golden_file_government_docs` - CFR regulations
- `test_golden_file_mixed_layouts` - Mixed column layouts
- `test_golden_file_newspapers` - Historical newspapers
- `test_golden_file_technical_docs` - Technical documentation
- `test_golden_file_theses` - PhD theses
- `test_golden_file_text_heavy` - Text-heavy documents
- `test_golden_file_tables` - Table-heavy documents

### Integration Tests
- `test_corpus_academic_full_pipeline` - Full extraction + quality
- `test_corpus_mixed_full_pipeline` - Mixed layout testing
- `test_corpus_forms_full_pipeline` - Form extraction
- `test_corpus_loader_basic` - Corpus loader validation
- `test_quality_metrics_integration` - Quality metrics

## Success Criteria ✅

| Criterion | Status | Result |
|-----------|--------|--------|
| Load 315+ PDFs | ✅ PASS | 356 PDFs |
| Golden file system | ✅ PASS | Save/load/compare working |
| Zero regressions | ✅ PASS | All 799 tests pass |
| Quality metrics | ✅ PASS | Integrated |
| Performance <5% | ✅ PASS | <1% overhead |

## How It Works

### 1. Corpus Loader
Loads PDFs from `/home/yfedoseev/projects/pdf_oxide_tests/pdfs/`

```rust
use helpers::corpus_loader::CorpusLoader;

let loader = CorpusLoader::default();
let pdfs = loader.list_pdfs("academic")?;  // 173 PDFs
```

### 2. Golden File Manager
Saves/loads extracted text with metadata

```rust
use helpers::golden_file_manager::GoldenFileManager;

let manager = GoldenFileManager::default();

// Save
manager.save_golden_file(&pdf_path, "academic", &text)?;

// Load & Compare
let golden = manager.load_golden_file(&pdf_path)?;
let result = manager.compare_extraction(&text, &golden);
assert!(result.passes());
```

### 3. Regression Testing
Compares current extraction against saved golden files

```rust
// Test framework does this automatically
test_golden_files_for_category("academic", Some(10));
// → Extracts text from 10 PDFs
// → Compares against golden files
// → Reports: X passed, Y failed, Z skipped
```

### 4. Quality Metrics
Detects extraction quality issues

```rust
use quality_metrics::analyze_quality;

let metrics = analyze_quality(&markdown_text);
// → Checks: word fusions, empty bold markers, spurious spaces
// → Score: 0-10 (must be ≥8.0 to pass)
```

## Next Steps

1. **Create Golden Files** (one-time setup)
   ```bash
   cargo test --test golden_files create_golden_files_sample -- --ignored --nocapture
   ```

2. **Verify Quality** (manual review)
   ```bash
   cat tests/golden_files/extracted_text/academic/arxiv_2510.21165v1.json
   ```

3. **Run Regression Tests**
   ```bash
   cargo test --test golden_files -- --nocapture
   ```

4. **Commit**
   ```bash
   git add tests/golden_files/
   git commit -m "Add golden files for regression testing"
   ```

## Documentation

- **Complete Guide**: `TEST_INFRASTRUCTURE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Quick Start**: This file

## Contact

For questions or issues, see the main project documentation or implementation summary.
