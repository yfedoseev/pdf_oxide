# PDF Spec Compliance Refactoring Plan

## Overview

This plan addresses the comprehensive refactoring needed to make pdf_oxide fully compliant with ISO 32000-1:2008 (PDF 1.7) specification while maintaining high-quality text extraction for all PDF files in `~/projects/pdf_oxide_tests`.

**Reference**: `docs/spec/pdf.md` (ISO 32000-1:2008)

---

## Phase 1: Audit and Classification

### 1.1 Non-Compliant Code Inventory

| Component | Location | Issue | Action |
|-----------|----------|-------|--------|
| **CamelCase word splitting** | `text.rs:1467-1475` | Linguistic heuristic, not PDF-defined | **ALREADY DISABLED** (Phase 10) |
| **Character transition heuristics** | `text.rs` (deleted) | Pattern recognition | **ALREADY DELETED** (Phase 10) |
| **Heading detection (font-size)** | `markdown.rs`, `html.rs`, `smart_analyzer.rs` | Semantic interpretation | **ALREADY REMOVED** (Phase 2) |
| **XY-Cut algorithm** | `markdown.rs:685-698` | Layout heuristic | **ALREADY DELETED** (Phase 2) |
| **Adaptive gap thresholding** | `gap_statistics.rs` | Statistical heuristic | KEEP (justified by geometry) |
| **Document type profiling** | `gap_statistics.rs:154-248` | Classification heuristic | **REMOVE** |
| **DBSCAN clustering** | `layout/clustering.rs` | ML-style approach | Feature-gated (OK) |
| **Column detection heuristic** | `document_analyzer.rs:133` | Position-based | **SIMPLIFY** |
| **Bold weight detection** | `layout/bold_validation.rs` | Statistical threshold | KEEP (font metrics) |
| **Debug span merging heuristics** | `debug_span_merging.rs` | Debug tracking | **SIMPLIFY** |

### 1.2 Spec-Compliant Components (Keep As-Is)

- PDF parsing: lexer, parser, xref (Section 7-8)
- Encryption: AES-256, RC4 (Section 7.6)
- ToUnicode CMap: Full support (Section 9.10)
- Text positioning: TJ/Tj operators (Section 9.3-9.4)
- Structure trees: Tagged PDF (Section 14.7)
- Font handling: Font dictionaries, CIDFont (Section 9.6-9.8)
- Geometric spacing: `geometric_spacing.rs` (pdfplumber-style)

---

## Phase 2: Remove Non-Compliant Code

### 2.1 Document Type Profiling (REMOVE)

**File**: `src/extractors/gap_statistics.rs`

**Current code** (lines 154-248):
- `DocumentType` enum with `policy_documents()`, `academic()`, `mixed()` profiles
- Profile-specific threshold multipliers

**Action**: Remove document type profiling; use single unified threshold approach.

**Rationale**: PDF spec doesn't define document types. Word boundary detection should use geometry only.

### 2.2 Debug Heuristic Tracking (SIMPLIFY)

**File**: `src/extractors/debug_span_merging.rs`

**Current code**:
- `SpaceInsertReason::Heuristic`
- `SpaceInsertReason::AdaptiveAndHeuristic`
- `needs_space_by_heuristic` field

**Action**: Remove heuristic-related variants; keep only:
- `SpaceInsertReason::TjOffset` (PDF spec: Section 9.4.4)
- `SpaceInsertReason::GeometricGap` (position-based)
- `SpaceInsertReason::AlreadyPresent`

### 2.3 Column Detection (SIMPLIFY)

**File**: `src/layout/document_analyzer.rs`

**Current code** (line 133):
```rust
// 4. Detect columns (rough heuristic based on horizontal gaps)
```

**Action**: Remove semantic column detection. Use only geometric position-based reading order.

---

## Phase 3: PDF-Compliant Word Boundary Detection

### 3.1 Unified Space Decision Pipeline

The spec-compliant approach uses TWO sources of space information:

1. **TJ Offset Values (Primary)** - PDF Spec Section 9.4.4
   - Negative offset in TJ array indicates positioning adjustment
   - Threshold: > 100 thousandths of em = word boundary

2. **Geometric Gap (Secondary)** - Position-based
   - Gap between consecutive spans > margin threshold
   - Margin = `word_margin * character_size`
   - Default `word_margin = 0.1` (matches pdfplumber/pdfminer.six)

### 3.2 Simplified TextExtractionConfig

**Current config** (complex):
```rust
pub struct TextExtractionConfig {
    pub space_insertion_threshold: f32,      // -120.0
    pub word_margin_ratio: f32,              // 0.1
    pub use_adaptive_tj_threshold: bool,     // true
    // ... more fields
}
```

**Spec-compliant config** (simplified):
```rust
pub struct TextExtractionConfig {
    /// TJ offset threshold (thousandths of em)
    /// Per PDF Spec Section 9.4.4, negative offsets > 100 indicate word space
    pub tj_threshold: f32,  // Default: -100.0

    /// Geometric word margin (fraction of character size)
    /// Matches pdfplumber/pdfminer.six LAParams
    pub word_margin: f32,   // Default: 0.1
}
```

### 3.3 Reading Order Algorithm

**Spec-compliant approach**:

1. **If Tagged PDF** (Section 14.7): Use structure tree's MCID sequence
2. **If Untagged**: Use geometric position-based sort (Y-then-X)

**Remove**: Column-aware sorting heuristics that guess layout.

---

## Phase 4: Testing Strategy

### 4.1 Test PDF Categories

All PDFs in `~/projects/pdf_oxide_tests/pdfs/`:
- `academic/` - ArXiv papers (multi-column, equations)
- `government/` - Official documents
- `technical/` - Technical manuals
- `diverse/` - Mixed types
- `forms/` - Interactive forms
- `mixed/` - Various layouts
- `newspapers/` - Multi-column layouts
- `theses/` - Academic theses

### 4.2 Quality Metrics

For each PDF, measure:
1. **No replacement characters** (U+FFFD count = 0)
2. **No fused words** (word count within 10% of expected)
3. **No extra spaces** (spurious space ratio < 1%)
4. **Readable content** (non-empty extraction)
5. **Preserved structure** (if Tagged PDF: structure preserved)

### 4.3 Test Commands

```bash
# Run all tests
cargo test

# Test specific PDF
cargo run --bin export_to_markdown -- ~/projects/pdf_oxide_tests/pdfs/academic/arxiv_*.pdf

# Batch validation
cargo run --bin validate_dataset -- ~/projects/pdf_oxide_tests/pdfs/
```

---

## Phase 5: Implementation Order

### Step 1: Simplify gap_statistics.rs
- [ ] Remove `DocumentType` enum
- [ ] Remove profile-specific configurations
- [ ] Keep only `AdaptiveThresholdConfig::default()`

### Step 2: Simplify debug_span_merging.rs
- [ ] Remove `Heuristic` and `AdaptiveAndHeuristic` variants
- [ ] Remove `needs_space_by_heuristic` field
- [ ] Update `SpaceDecision` to use only TJ/Geometric sources

### Step 3: Simplify TextExtractionConfig
- [ ] Remove deprecated fields
- [ ] Use only `tj_threshold` and `word_margin`
- [ ] Update all call sites

### Step 4: Simplify reading order
- [ ] Remove column detection heuristics in text.rs
- [ ] Use Tagged PDF structure when available
- [ ] Fall back to simple Y-then-X sort

### Step 5: Test and validate
- [ ] Run `cargo test` - all tests pass
- [ ] Run validation on all PDF categories
- [ ] Compare quality metrics before/after

---

## Phase 6: Rollback Plan

If quality degrades significantly:

1. **Restore geometric_spacing.rs** - already spec-compliant
2. **Adjust word_margin** - try 0.05 (tight) or 0.15 (loose)
3. **Re-enable adaptive threshold** - statistical but geometry-based

**Do NOT restore**:
- CamelCase splitting (linguistic heuristic)
- Character transition heuristics (pattern matching)
- Document type profiling (classification)

---

## Success Criteria

1. **All tests pass**: `cargo test` succeeds
2. **No regressions**: Quality metrics >= Phase 10 baseline
3. **Simplified code**: Remove ~500+ lines of heuristic code
4. **PDF spec alignment**: All space decisions traceable to Section 9.4.4 or geometry
5. **Validation passes**: All PDFs in test corpus parse successfully

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/extractors/gap_statistics.rs` | Remove DocumentType, simplify config |
| `src/extractors/debug_span_merging.rs` | Remove heuristic variants |
| `src/extractors/text.rs` | Simplify TextExtractionConfig, clean up comments |
| `src/layout/document_analyzer.rs` | Remove column detection heuristic |
| `src/converters/markdown.rs` | Remove XY-Cut comments |
| `src/converters/html.rs` | Remove heading detection comments |
| `CLAUDE.md` | Already updated with spec reference |

---

## Timeline

- **Step 1-2**: Remove non-compliant code structures
- **Step 3**: Simplify configuration
- **Step 4**: Update reading order logic
- **Step 5**: Test and validate

All changes should be incremental with testing after each step.
