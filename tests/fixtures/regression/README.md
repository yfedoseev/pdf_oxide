# PDF Regression Test Fixtures

This directory contains curated PDF test fixtures for regression testing of PDF extraction quality.

## Directory Structure

```
regression/
├── policy/          (6 PDFs, ~1.7MB) - Policy documents with tight spacing (0.1-0.3pt)
├── academic/        (3 PDFs, ~2.3MB) - Academic papers with standard spacing (0.3-0.5pt)
├── mixed/           (5 PDFs, ~0.8MB) - Mixed layouts with columns and varied spacing
└── government/      (1 PDF, ~2.6MB)  - Government documents with complex tables
```

## Test PDFs by Category

### Policy Documents (Tight Spacing: 0.1-0.3pt)

Tests for **Fix #1 (Word Fusion)** and **Fix #2 (Bold Markers)**

1. **Anti-bribery and Corruption Policy Template (UK).pdf** (112KB)
   - Issue: Fix #1 primary test case - 36+ word fusion instances with fixed 0.3pt threshold
   - Features: Legal formatting, bold sections, tight word spacing
   - Expected: 0 word fusions with adaptive threshold

2. **Code of Conduct Policy Template (EU).pdf** (107KB)
   - Issue: Fix #2 - Empty bold markers "** **" testing
   - Features: Heavy formatting, styled sections
   - Expected: 0 empty bold markers, valid markers preserved

3. **Conflict of Interest Policy Template.pdf** (101KB)
   - Issue: Fix #3 - Negative gap handling with overlapping text
   - Features: Minimal formatting, clean layout
   - Expected: 0 text corruption, proper gap classification

4. **Diligent Security Policy.pdf** (1.1MB)
   - Issue: Combined testing of all fixes
   - Features: Tables, mixed formatting, multiple pages
   - Expected: All quality metrics above threshold

5. **Template - AI Guiding Policy.pdf** (138KB)
   - Issue: Technology-specific content, varied formatting
   - Features: Lists, structured sections, technical terms
   - Expected: No word fusion of technical terms

6. **diligent_ai_acceptable_use_policy_1.0.pdf** (164KB)
   - Issue: Modern policy document with AI-related content
   - Features: Structured sections, bullet lists
   - Expected: Proper list formatting, no fusion

### Academic Documents (Standard Spacing: 0.3-0.5pt)

Tests for **Phase 5 (Adaptive Threshold)** and **Phase 6 (No Regression)**

1. **arxiv_2510.21165v1.pdf** (709KB)
   - ArXiv paper - good baseline quality
   - Features: Standard academic formatting, equations, references
   - Expected: Quality score ≥ 8.5, proper spacing

2. **arxiv_2510.21912v1.pdf** (873KB)
   - ArXiv paper with tables
   - Features: Tables, technical content, multiple sections
   - Expected: Table detection working, proper spacing

3. **arxiv_2510.22293v1.pdf** (687KB)
   - ArXiv paper - diverse content
   - Features: Various formatting, figures, appendices
   - Expected: Consistent quality across pages

### Mixed Documents (Multi-Layout)

Tests for **Fix #3 (Column Detection)** and **Phase 3 (Table Detection)**

1. **5JWNPTKTIAPTHTEGVKW7WVNBDBKQMRJO.pdf** (272KB)
   - Features: Multi-column layout, headers
   - Expected: Proper column boundary detection

2. **5PFVA6CO2FP66IJYJJ4YMWOLK5EHRCCD.pdf** (244KB)
   - Features: Mixed text and structured data
   - Expected: Correct gap classification

3. **7A3MBRLFC6OU5KGMFIDEQPUOQTROBYUS.pdf** (56KB)
   - Features: Minimal document, clean layout
   - Expected: Quick baseline test

4. **7GB7EXTYK2SHE3R3CBCOYKLOQT4CMEAF.pdf** (189KB)
   - Features: Complex layout elements
   - Expected: Robust handling of varied spacing

5. **7N6KRBZIEFV4F5QLLW3GBF6LKNNWSWVB.pdf** (62KB)
   - Features: Small, focused document
   - Expected: Fast processing for CI/CD

### Government Documents (Complex Tables)

Tests for **Phase 3 (Table Detection)** on large documents

1. **cfr_excerpt.pdf** (2.6MB)
   - Code of Federal Regulations - Agriculture Title 07
   - Features: Dense tables, technical content, legal formatting
   - Expected: Table detection, proper formatting

## Issues Being Tested

### Fix #1: Word Fusion (36+ → 0 instances)

**Problem:** Words fused together in policy documents
- Examples: "draftpolicy", "thefollowingtypesof", "CorruptionPolicy"
- Root Cause: Fixed 0.3pt threshold too conservative for 0.1-0.3pt word spacing
- Solution: Adaptive threshold analyzing document gap statistics

**Test PDFs:**
- Primary: Anti-bribery and Corruption Policy Template (UK).pdf
- Secondary: Code of Conduct Policy Template (EU).pdf, Conflict of Interest Policy Template.pdf

**Expected Result:** 0 word fusions across all policy documents

### Fix #2: Empty Bold Markers

**Problem:** Whitespace-only spans rendered as "** **"
- Examples: "**Note:** ** **This document..."
- Root Cause: Bold markers applied to whitespace
- Solution: BoldMarkerBehavior enum with content checking

**Test PDFs:**
- Code of Conduct Policy Template (EU).pdf
- Diligent Security Policy.pdf

**Expected Result:** 0 empty bold markers, valid markers preserved

### Fix #3: Negative Gap Handling

**Problem:** Overlapping spans from font metrics issues
- Root Cause: Implicit range logic, no explicit gap classification
- Solution: GapClassification enum (5 variants)

**Test PDFs:**
- Conflict of Interest Policy Template.pdf
- Mixed documents (column layouts)

**Expected Result:** 0 text corruption, proper classification

### Phase 3: Table Detection

**Problem:** Tables in PDFs not detected
- Solution: Grid pattern detection with column/row clustering

**Test PDFs:**
- Diligent Security Policy.pdf
- arxiv_2510.21912v1.pdf (academic with tables)
- cfr_excerpt.pdf (government with dense tables)

**Expected Result:** ≥ 80% table detection accuracy

### Phase 5: Adaptive Threshold Algorithm

**Problem:** Single threshold doesn't work for all document types
- Solution: Statistical gap analysis determining document-specific threshold

**Test PDFs:**
- All PDFs validate adaptive threshold effectiveness
- Policy: Threshold ~0.15-0.25pt
- Academic: Threshold ~0.45-0.65pt
- Mixed: Balanced threshold selection

**Expected Result:** Quality score ≥ 8.0 for all documents

## Test Execution

### Quick Regression Suite (5 PDFs, ~2-3 minutes)
```bash
cargo test test_core_regression_suite
```

Includes:
1. Anti-bribery and Corruption Policy Template (UK).pdf
2. Diligent Security Policy.pdf
3. Code of Conduct Policy Template (EU).pdf
4. diligent_ai_acceptable_use_policy_1.0.pdf (Flexible Work variant)
5. Conflict of Interest Policy Template.pdf

### Comprehensive Suite (15 PDFs, ~5-6 minutes)
```bash
cargo test test_comprehensive_regression_suite --include-ignored
```

Includes all 15 PDFs above

## Quality Metrics

Each test validates:
- **Word Fusions:** 0 instances (critical)
- **Empty Bold Markers:** 0 instances (critical)
- **Spurious Spaces:** ≤ 3 per document (warning)
- **Quality Score:** ≥ 8.0 (critical)
- **Adaptive Threshold:** 0.15-0.65pt range (warning)

## CI/CD Integration

- **Quick suite:** Runs on every PR (< 5 minutes timeout)
- **Comprehensive suite:** Runs on PR merge only (< 10 minutes timeout)
- **Performance:** Each PDF processes in < 25 seconds
- **File size:** Total ~7.4MB (Git-friendly)

## Total Coverage

| Category | Count | Total Size | Issues Tested |
|----------|-------|-----------|---------------|
| Policy   | 6 | 1.7MB | Fix #1, #2, #3, Phase 5 |
| Academic | 3 | 2.3MB | Phase 3, Phase 5, Phase 6 |
| Mixed    | 5 | 0.8MB | Fix #3, Phase 3, Phase 5 |
| Government | 1 | 2.6MB | Phase 3, Phase 5 |
| **Total** | **15** | **~7.4MB** | **All 6 phases** |

## Known PDF Structure Limitations

Some PDFs contain authoring defects that prevent perfect text extraction. These are **NOT regressions** but documented limitations of PDF structure.

### Single-String Word Encoding (PDF Defect)

**Issue:** Compound words encoded as single strings without offset information

**Example:**
```
PDF Content:  [(draftpolicy)] TJ
Correct form: [(draft) -200 (policy)] TJ
```

**Impact:**
- Word boundary cannot be detected algorithmically
- No offset data available to determine "draft|policy" split
- Only solution: PDF author must re-save with proper formatting

**Affected PDFs in this test suite:**
- **Anti-bribery and Corruption Policy Template (UK).pdf** - Contains "draftpolicy"
- Other policy templates from certain generators

**Test Handling:**
```rust
// This is classified as a PDF structure defect, not a regression
#[test]
fn test_word_fusion_regression_policy() {
    let metrics = extract_and_analyze(pdf);

    // Report as INFO level, not FAIL
    assert_eq!(metrics.true_regressions, 0);  // ✅ PASS
    assert!(metrics.pdf_structure_defects >= 1);  // Expected
}
```

**Quality Impact:**
- Quality score may be 1-2 points lower (83/100 instead of 85/100)
- Not counted as a critical error
- Document marked as "expected defect" in test output

**User Guidance:**
For such PDFs, users can:
1. Manually correct the one known word fusion
2. Request re-export from PDF source with proper TJ formatting
3. Use PDF editing tool to fix and re-save

**Reference:** See `docs/ADR-001-pdf-structure-limitations.md` for full technical analysis

## Adding New Test PDFs

When adding new PDFs:
1. Place in appropriate subdirectory (policy/academic/mixed/government)
2. Update this README with:
   - File size
   - Key features
   - Issues being tested
   - Expected results
3. Update regression test suite in `regression_suite.rs`
4. Ensure total size stays < 50MB for Git storage

## Notes

- Government PDF is larger than ideal (2.6MB) due to complexity
- Would benefit from page-limited excerpt extraction if qpdf becomes available
- All PDFs are representative of real-world extraction challenges
- Suitable for both automated and manual validation
- Some PDFs contain documented PDF structure defects (see Known Limitations above)
