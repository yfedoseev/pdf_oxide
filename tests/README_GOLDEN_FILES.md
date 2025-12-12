# Golden Files & Test Corpus - Quick Start

## Overview

This directory contains the comprehensive testing framework for PDF text extraction regression testing.

**Status**: Fully Implemented
**Corpus**: 356 PDFs across 14 categories
**Baseline Tests**: 799 passing (zero regressions)
**New Tests**: 20+ golden file tests, 7+ integration tests, comprehensive regression suite

## Quick Commands

### Run All Tests
```bash
# Baseline library tests (799 tests)
cargo test --lib

# Golden file regression tests (20 tests)
cargo test --test golden_files

# Integration tests (7 tests)
cargo test --test corpus_integration_tests

# Comprehensive regression tests (12 category tests)
cargo test --test test_extraction_regression
```

### View Test Corpus
```bash
# List all PDFs by category with details
cargo test --test test_golden_file_generation test_list_corpus_detailed -- --ignored --nocapture

# Output:
# academic             :   173 PDFs
# mixed                :    89 PDFs
# forms                :    30 PDFs
# government           :    29 PDFs
# ... (356 total)
```

### Create Golden Files (Baseline Generation)
```bash
# Generate golden files for FULL corpus (all categories)
cargo test --test test_golden_file_generation test_generate_all_golden_files -- --ignored --nocapture

# Generate for a specific category only
cargo test --test test_golden_file_generation test_generate_academic_golden_files -- --ignored --nocapture

# Generate with quality validation (recommended for first-time setup)
cargo test --test test_golden_file_generation test_generate_with_quality_validation -- --ignored --nocapture

# Creates: tests/golden_files/extracted_text/{category}/*.json
```

### Run Regression Tests
```bash
# Quick regression check (5 PDFs per category, fast)
cargo test --test test_extraction_regression test_regression_quick

# Full regression test (all categories, up to 20 PDFs each)
cargo test --test test_extraction_regression test_regression_full_corpus -- --ignored --nocapture

# Test specific category
cargo test --test test_extraction_regression test_regression_academic -- --nocapture
```

### Run Performance Benchmark
```bash
cargo test --test corpus_integration_tests benchmark_corpus_extraction -- --ignored --nocapture
```

## File Structure

```
tests/
├── helpers/
│   ├── mod.rs                         # Public API
│   ├── corpus_loader.rs               # Load PDFs from corpus (219 lines)
│   └── golden_file_manager.rs         # Golden file I/O (450 lines)
├── test_golden_file_generation.rs     # Baseline generation (430 lines) - NEW
├── test_extraction_regression.rs      # Regression test suite (550 lines) - NEW
├── golden_files.rs                    # Original regression tests (258 lines)
├── corpus_integration_tests.rs        # Full pipeline tests (343 lines)
├── quality_metrics.rs                 # Quality scoring (459 lines)
├── TEST_INFRASTRUCTURE.md             # Complete documentation
├── IMPLEMENTATION_SUMMARY.md          # Implementation details
├── README_GOLDEN_FILES.md             # This file
└── golden_files/                      # Generated baselines (gitignored initially)
    └── extracted_text/
        ├── academic/*.json
        ├── diverse/*.json
        ├── forms/*.json
        └── ... (all categories)
```

**Total Infrastructure Code**: ~2,200 lines

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

## Baseline Workflow

### First-Time Setup (One-Time)

1. **Generate All Golden File Baselines**
   ```bash
   cargo test --test test_golden_file_generation test_generate_all_golden_files -- --ignored --nocapture
   ```
   This generates baselines for all 356 PDFs across 14 categories.
   Expected time: ~5-10 minutes depending on hardware.

2. **Verify Baseline Quality** (Recommended)
   ```bash
   # Run quality validation during generation
   cargo test --test test_golden_file_generation test_generate_with_quality_validation -- --ignored --nocapture
   ```
   This checks quality metrics for each PDF and warns about issues.

3. **Manual Spot-Check** (5-10 PDFs per category)
   ```bash
   # Review a sample golden file
   cat tests/golden_files/extracted_text/academic/*.json | head -100
   ```
   Verify extracted text looks correct for your corpus.

4. **Commit Baselines**
   ```bash
   git add tests/golden_files/
   git commit -m "Add golden file baselines for regression testing"
   ```

### Ongoing Regression Detection

Run after any code changes to detect regressions:

```bash
# Quick check (fast, CI-friendly)
cargo test --test test_extraction_regression test_regression_quick

# Full check (comprehensive)
cargo test --test test_extraction_regression test_regression_full_corpus -- --ignored --nocapture
```

### After Intentional Quality Improvements

When you make changes that intentionally improve extraction quality:

1. **Verify improvements are intentional**
2. **Update baselines for improved files**
   ```bash
   cargo test --test test_extraction_regression test_update_baselines_for_improved -- --ignored --nocapture
   ```
3. **Review and commit updated baselines**
   ```bash
   git diff tests/golden_files/
   git add tests/golden_files/
   git commit -m "Update baselines after quality improvement"
   ```

### Comparison Tolerances

The regression tests use these tolerances to allow minor variations:

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Character count | +/- 0.5% | Minor encoding differences |
| Word count | +/- 1.0% | Minor spacing variations |
| Line count | +/- 2.0% | Layout interpretation changes |

Failures beyond these tolerances indicate potential regressions.

## Documentation

- **Complete Guide**: `TEST_INFRASTRUCTURE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Quick Start**: This file

## Contact

For questions or issues, see the main project documentation or implementation summary.
