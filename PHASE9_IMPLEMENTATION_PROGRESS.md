# Phase 9: Text Spacing Logic Simplification - Implementation Progress

## Status: IN PROGRESS (Steps A-B Complete, Step C In Progress)

**Last Updated**: 2025-12-05
**Current Phase**: Creating geometric spacing integration into text extraction pipeline

---

## Completed Work

### Step A: ✅ SpacingConfig Simplification (COMPLETE)
- Created new `src/extractors/geometric_spacing.rs` module (227 lines)
- Implemented `SpacingConfig` struct with single `word_margin: f32` parameter
- Replaced 5+ complex threshold parameters with one simple parameter
- Default word_margin = 0.1 (matches pdfminer.six)
- Added `SpacingConfig::tight()` and `SpacingConfig::loose()` factory methods

### Step B: ✅ Geometric Spacing Module (COMPLETE)
- Implemented `SpaceInsertion` struct (simple bool-based decision)
- Implemented `should_insert_space()` function with single geometric rule
- Algorithm: `if (prev.x1 + margin) < next.x0: insert_space()`
- Where: `margin = word_margin * max(prev.width, prev.height)`
- Removed all heuristics, confidence scoring, document-type awareness
- Added 6 comprehensive unit tests - ALL PASSING
  - test_clear_word_gap
  - test_tight_kerning
  - test_existing_boundary_space
  - test_word_margin_variations
  - test_exactly_at_margin
  - test_leading_space_in_next

### Infrastructure Fixes
- Fixed all TextSpan initializations across codebase to include `offset_semantic` field
- Updated extractors/mod.rs to export geometric_spacing module
- Code compiles without errors (3 unrelated warnings only)

---

## Current Work: Step C - Integration

### Phase 9.C.1: Simplify merge_adjacent_spans()
**Location**: `src/extractors/text.rs:1999`

**Current Status**:
- Function is ~160 lines with complex multi-rule logic
- Uses old `should_insert_space()` with 4 competing rules
- Document type adjustments (1.3x, 0.7x multipliers)
- Confidence scoring system
- Heuristic checks (CamelCase, number-letter)

**Target State**:
- ~30 lines of simple geometric logic
- Direct call to geometric_spacing::should_insert_space()
- No document type awareness
- No confidence scoring

### Phase 9.C.2: Replace SpanMergingConfig
**Location**: `src/extractors/text.rs`

**Current Structure**:
```rust
pub struct SpanMergingConfig {
    pub space_threshold_em_ratio: f32,           // REMOVE
    pub conservative_threshold_pt: f32,           // REMOVE
    pub column_boundary_threshold_pt: f32,       // KEEP
    pub severe_overlap_threshold_pt: f32,        // REMOVE
    pub use_adaptive_threshold: bool,            // REMOVE
    pub adaptive_config: Option<AdaptiveThresholdConfig>, // REMOVE
}
```

**Target Structure**:
```rust
pub struct SpanMergingConfig {
    pub word_margin: f32,                        // Single spacing parameter
    pub column_boundary_threshold_pt: f32,       // Keep for layout detection
}
```

---

## Planned Work: Steps D-I

### Step D: Remove Competing Rules (Pending)
- Remove SpaceSource enum (21-45 lines)
- Remove SpaceDecision struct (47-89 lines)
- Remove old should_insert_space() function (684-808 lines)
- Remove should_insert_space_heuristic() (3907-3930 lines)
- Remove get_adjusted_space_threshold() (1331-1350 lines)
- Remove calculate_adaptive_tj_threshold() (1422-1407 lines)
- Remove split_fused_words() (~150+ lines)
- Remove TJ offset space insertion logic from process_tj_array()

### Step E: Delete space_detection.rs (Pending)
- File location: `src/layout/space_detection.rs`
- Size: ~523 lines
- Components: SpaceDetectionEngine, 4 detector traits, all logic deleted

### Step F: Delete word_segmentation.rs (Pending)
- File location: `src/extractors/word_segmentation.rs`
- Size: ~1532 lines
- Components: Viterbi algorithm, 1000+ word dictionary, all deleted

### Step G: Simplify gap_statistics.rs (Pending)
- Remove DocumentType enum (757-945 lines)
- Remove DocumentProfile enum (1009-1127 lines)
- Keep GapStatistics and extract_gaps() for debugging only

### Step H: Update All Call Sites and Tests (Pending)
- Update all references to removed functions
- Update test expectations for quality metrics
- Update regression test suite
- Remove document-type-specific test branches

### Step I: Validate and Verify (Pending)
- Run full test suite
- Verify quality improvement from 4.3/10 → 8.5+/10
- Check policy PDFs for 0 spurious spaces
- Ensure academic PDFs maintain 6.0+/10 quality
- Confirm code reduction (~2900 lines removed, ~100 added)

---

## Implementation Challenges & Solutions

### Challenge 1: Circular Dependencies
- `text.rs` uses `gap_statistics::DocumentType`
- `gap_statistics.rs` uses `text.rs` types
- **Solution**: Remove DocumentType entirely, no circular reference

### Challenge 2: Large Refactoring Target
- `text.rs` is 4425 lines with deeply embedded spacing logic
- Multiple functions interdependent
- **Solution**: Phased approach - integrate geometric spacing alongside old code, then remove old code incrementally

### Challenge 3: Test Compatibility
- 650+ tests exist throughout codebase
- Many rely on old spacing behavior
- **Solution**: Create feature flag or gradual replacement to validate

---

## Expected Impact

### Code Reduction
| Component | Current Lines | Target Lines | Reduction |
|-----------|---------------|--------------|-----------|
| SpaceDecision/SpaceSource | 90 | 0 | -100% |
| should_insert_space() | 180 | 0 | -100% |
| space_detection.rs | 523 | 0 | -100% |
| word_segmentation.rs | 1532 | 0 | -100% |
| DocumentType/Profile | 400 | 0 | -100% |
| spacing logic total | ~2925 | ~80 | **-97%** |

### Quality Improvement
| Document Type | Current | Target | Metric |
|---------------|---------|--------|--------|
| Policy PDFs | 4.3/10 | 8.5+/10 | Spurious spaces removed |
| Academic PDFs | 6.0/10 | 8.0+/10 | Maintain quality |
| Mixed PDFs | 5.0/10 | 7.5+/10 | Overall improvement |

---

## Compilation Status

**Current**: ✅ Compiling successfully
```
   Compiling pdf_oxide v0.1.2
   Finished `dev` profile [unoptimized + debuginfo]
```

**Warnings** (unrelated to Phase 9):
- unused import: `normalize_horizontal_whitespace` in converters/markdown.rs
- unused variable: `group_text` in layout/bold_validation.rs
- dead code methods in text.rs (will be removed in Phase 9)

---

## Next Steps (Immediate)

1. ✅ Finalize geometric_spacing module integration
2. Simplify merge_adjacent_spans() to use new geometric logic
3. Update SpanMergingConfig to single word_margin parameter
4. Remove old should_insert_space() function
5. Test with existing test suite
6. Incrementally remove remaining old code

---

## Files Modified This Session

1. ✅ `src/extractors/geometric_spacing.rs` (NEW - 227 lines)
2. ✅ `src/extractors/mod.rs` (updated exports)
3. ✅ `src/layout/font_normalization.rs` (fixed TextSpan initializations)
4. ✅ `src/extractors/gap_statistics.rs` (fixed TextSpan initializations)
5. ✅ `src/extractors/text.rs` (fixed TextSpan initializations)
6. ✅ `src/converters/html.rs` (fixed TextSpan initializations)
7. ✅ `src/converters/markdown.rs` (fixed TextSpan initializations)

---

## Files Pending Modification

1. `src/extractors/text.rs` (Major refactoring in progress)
2. `src/extractors/gap_statistics.rs` (remove DocumentType/Profile)
3. `src/layout/space_detection.rs` (DELETE)
4. `src/extractors/word_segmentation.rs` (DELETE)
5. `src/layout/mod.rs` (remove exports)
6. `tests/quality_metrics.rs` (update thresholds)
7. `tests/regression_suite.rs` (update expectations)

---

## Success Criteria Status

- [ ] Code Reduction: Spacing logic ~2900 lines → ~80 lines
- [ ] Quality Improvement: 4.3/10 → 8.5+/10 average
- [ ] Policy PDFs: 0 spurious spaces (matching pdfplumber)
- [ ] Academic PDFs: No degradation (maintain 6.0+/10)
- [ ] All Regression Tests: Pass with new logic
- [ ] No Heuristics: Zero CamelCase/number-letter/confidence detection
- [ ] Single Config: One word_margin replaces all thresholds

---

**Estimated Completion**: 2-3 hours remaining
