# Golden File Infrastructure - Quick Start Guide

## One-Minute Summary

pdf_oxide now has baseline generation and regression testing infrastructure:
- **Generate baselines**: `cargo test --test test_golden_file_generation -- --ignored --nocapture`
- **Run regression tests**: `cargo test --test test_extraction_regression`

Both files are fully implemented, compiled, and ready to use.

## Quick Commands

### 1. Generate Baselines (First Time Only)

```bash
# Full corpus (all 14 categories, ~45-60 minutes)
cargo test --test test_golden_file_generation -- --ignored --nocapture

# Single category (faster, ~3-5 minutes)
cargo test --test test_golden_file_generation test_generate_academic_golden_files -- --ignored --nocapture
```

Categories available:
- `academic`, `diverse`, `forms`, `government`, `mixed`, `newspapers`
- `technical`, `theses`, `text_heavy`, `tables`, `multilingual`, `scanned`
- `images`, `test_datasets`

### 2. Run Regression Tests (Automated)

```bash
# All categories
cargo test --test test_extraction_regression

# Single category with detailed output
cargo test --test test_extraction_regression test_regression_academic -- --nocapture

# With detailed diffs for failures
cargo test --test test_extraction_regression -- --nocapture
```

### 3. Update Baselines (After Improvements)

```bash
# Remove old baselines
rm -rf tests/golden_files/

# Generate new baselines
cargo test --test test_golden_file_generation -- --ignored --nocapture

# Verify improvements
cargo test --test test_extraction_regression

# Commit code changes (not golden files)
git add src/
git commit -m "Improve text extraction quality"
```

## File Locations

| File | Purpose |
|------|---------|
| `tests/test_golden_file_generation.rs` | Generate baselines (785 lines) |
| `tests/test_extraction_regression.rs` | Run regression tests (1011 lines) |
| `tests/helpers/golden_file_manager.rs` | Golden file management (pre-existing) |
| `tests/helpers/corpus_loader.rs` | Corpus access (pre-existing) |
| `tests/golden_files/` | Generated baselines (created by tests) |
| `GOLDEN_FILES_README.md` | Full documentation |

## Status

- ✓ Both test files created and compiling
- ✓ 14 generation tests (marked #[ignore])
- ✓ 11 automated regression tests
- ✓ Library tests: 815 passed, 0 failed
- ✓ Ready for use

## Tolerance Thresholds

Regression tests allow minor variations:
- **Character count**: ±0.5%
- **Word count**: ±1.0%
- **Line count**: ±2.0%

## Output Examples

### Generation Summary
```
========== GOLDEN FILE GENERATION SUMMARY ==========

Category Breakdown:
Category        Total   Success   Failed    Skip  Success%
────────────────────────────────────────────────────────────
academic           45       43        0       2      95.6%
diverse            38       37        1       0      97.4%
...

Content Statistics:
  Total Characters: 1,234,567
  Total Words:      201,234
  Total Pages:      5,432

Timing:
  Total Duration:   45m 32s
  Avg per PDF:      18.3ms
```

### Regression Results
```
[academic] Testing 45 PDFs against baselines...
  [1/45] [PASS] document1.pdf
  [2/45] [WARN] document2.pdf (chars: +0.2%, words: -0.1%)
  [3/45] [FAIL] document3.pdf - REGRESSION DETECTED
  ...
[academic] Complete: 43 pass, 1 warn, 1 fail, 0 skip, 0 error
```

## Test Attributes

### Generation Tests (Manual Execution)
```rust
#[test]
#[ignore]  // Must use --ignored flag to run
fn test_generate_all_golden_files() { ... }
```

### Regression Tests (Auto-Run)
```rust
#[test]  // Runs automatically with cargo test
fn test_regression_academic() { ... }
```

## Golden File Format

```json
{
  "pdf_path": "path/to/document.pdf",
  "category": "academic",
  "extracted_text": "Full extracted text...",
  "text_hash": "hash_value",
  "char_count": 5427,
  "word_count": 892,
  "script_distribution": {
    "Latin": 98.5,
    "CJK": 1.5
  },
  "extraction_timestamp": "2025-12-11T14:30:00Z"
}
```

## Corpus Path

Default: `/home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000`

Structure:
```
pdfs_1000/
├── academic/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
├── diverse/
│   └── ...
└── ... (12 more categories)
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Regression Tests
  run: cargo test --test test_extraction_regression --release
```

### GitLab CI
```yaml
regression_tests:
  script:
    - cargo test --test test_extraction_regression --release
```

## Troubleshooting

### No PDFs Found
Check corpus exists and contains PDFs:
```bash
ls -la /home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000
find /home/yfedoseev/projects/pdf_oxide_tests/pdfs_1000 -name "*.pdf" | head -5
```

### Out of Memory
Generate category-by-category instead of all at once:
```bash
cargo test --test test_golden_file_generation test_generate_academic_golden_files -- --ignored --nocapture
cargo test --test test_golden_file_generation test_generate_diverse_golden_files -- --ignored --nocapture
# ... continue for other categories
```

### Regression Failures
Review detailed output:
```bash
cargo test --test test_extraction_regression -- --nocapture
```

If regressions are intentional improvements:
1. Update baselines (see step 3 above)
2. Commit code changes
3. Tests will pass with new baselines

## Next Steps

1. **Generate baselines** (one-time, ~45-60 minutes):
   ```bash
   cargo test --test test_golden_file_generation -- --ignored --nocapture
   ```

2. **Spot-check quality** (5-10 PDFs per category):
   - Look at generated golden files
   - Verify extracted text looks correct

3. **Run regression tests** (validate setup):
   ```bash
   cargo test --test test_extraction_regression
   ```

4. **Integrate into CI/CD** (automated testing):
   - Add regression test command to your pipeline
   - Runs automatically on each commit

## Documentation

- **Full Guide**: See `GOLDEN_FILES_README.md` for comprehensive documentation
- **Quick Ref**: This file for common commands
- **Code**: Inline comments in test files explain implementation

## Support

For detailed information, see `GOLDEN_FILES_README.md`:
- Comprehensive usage guide
- Configuration options
- Troubleshooting guide
- FAQ section
- Performance notes
- CI/CD examples
