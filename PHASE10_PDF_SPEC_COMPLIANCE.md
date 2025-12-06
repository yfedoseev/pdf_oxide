# Phase 10: PDF Specification Compliance Refactoring

**Goal**: Remove all non-spec-compliant heuristics and implement ONLY PDF ISO 32000-1:2008 spec-compliant solutions.

**Current State**: 3.4/10 quality (1/5 PDFs passing) with accumulated heuristics
**Target State**: Simpler, spec-aligned implementation that should perform better

---

## Specification Baseline (ISO 32000-1:2008)

### Permitted Signals for Word Boundary Detection

✅ **ALLOWED (In PDF Spec)**:
1. **TJ Array Offsets** (Section 9.4.3)
   - Negative offsets < -100 thousandths of em indicate word boundaries
   - Explicit positioning artifact from PDF author
   - Most reliable signal

2. **Geometric Gaps** (Section 9.4.4 - Text Space Details)
   - Gap between character bounding boxes
   - Font metrics: character width, word spacing, horizontal scaling
   - Position-based, measurable, reproducible

3. **Font Metrics** (Section 9.3 - Text State Parameters)
   - Word spacing (Tw) operator
   - Character spacing (Tc) operator
   - Horizontal scaling (Th) operator
   - Explicit PDF state

4. **Boundary Whitespace** (Section 9.4.3 - Text Showing)
   - Spaces already present in text strings
   - Explicit in PDF content stream
   - Should not be duplicated

❌ **NOT ALLOWED (Not in PDF Spec)**:
1. **CamelCase Detection** - linguistic heuristic, not positioning
2. **Character Pattern Heuristics** - not PDF-defined
3. **Confidence Scoring** - subjective, not spec-based
4. **Document Type Detection** - not PDF requirement
5. **Multiple Thresholds** - adds complexity without spec foundation
6. **Dictionary-based Word Segmentation** - external knowledge, not in PDF

---

## Implementation Changes

### 1. Remove `split_fused_words()` Function

**File**: `src/extractors/text.rs` (lines 2104-2204)
**Rationale**: CamelCase splitting is a heuristic trying to fix PDF authoring defects that the spec doesn't address.

**Action**: DELETE entirely
- Remove `split_fused_words()` function
- Remove `split_on_camelcase()` helper
- Remove call to split_fused_words() at line 1512

**Impact**:
- Accept that malformed PDFs (e.g., "helporganisationscraft" as single string without positioning) CANNOT be recovered
- This is PDF spec limitation, not our bug
- Aligns with pdfplumber philosophy

---

### 2. Simplify `should_insert_space()` Function

**File**: `src/extractors/text.rs` (lines 683-781)
**Current**: 4 rules with heuristics, confidence scoring, document-type adjustments
**Target**: 3 rules, spec-only signals

**NEW IMPLEMENTATION**:

```rust
fn should_insert_space(
    preceding_text: &str,
    following_text: &str,
    gap_pt: f32,
    font_size: f32,
    tj_offset_triggered: bool,
    _config: &SpanMergingConfig,  // Ignore config - use fixed thresholds
) -> SpaceDecision {
    // RULE 0: Boundary Space (Section 9.4.3)
    if has_boundary_space(preceding_text, following_text) {
        return SpaceDecision::no_space(SpaceSource::AlreadyPresent, 1.0);
    }

    // RULE 1: TJ Offset Signal (Section 9.4.3)
    // Most explicit PDF positioning signal
    if tj_offset_triggered {
        return SpaceDecision::insert(SpaceSource::TjOffset, 1.0);
    }

    // RULE 2: Geometric Gap + Font Metrics (Section 9.4.4)
    // Standard approach: (gap > font_size * 0.25)
    // 0.25 = typical word spacing em (pdfplumber standard)
    let geometric_threshold = font_size * 0.25;

    if gap_pt > geometric_threshold {
        return SpaceDecision::insert(SpaceSource::GeometricGap, 0.95);
    }

    // DEFAULT: No space
    SpaceDecision::no_space(SpaceSource::NoSpace, 1.0)
}
```

**Deletions**:
- Remove Rule 3 (Character heuristic) - lines 752-759
- Remove Rule 4 (Conservative threshold) - lines 763-772
- Remove `should_insert_space_heuristic()` function - line 3737
- Remove document-type adjustments - lines 709-716
- Remove dual-threshold logic - lines 721-723
- Remove confidence scoring variations

**Why**:
- Character heuristics have no basis in PDF spec
- Font metrics for geometric calculation are spec-defined
- Fixed threshold (0.25em) matches pdfplumber
- Simpler = fewer edge cases = more predictable

---

### 3. Remove Non-Spec Configuration

**Files**: `src/extractors/text.rs`, `SpanMergingConfig`

**Remove from config**:
- `document_type` detection - not in spec
- `space_threshold_em_ratio` - replace with constant 0.25
- `conservative_threshold_pt` - not needed with simplified logic
- All document-type-specific adjustments

**Keep**:
- `column_boundary_threshold_pt` - legitimate non-text element detection
- `severe_overlap_threshold_pt` - legitimate kerning detection

---

### 4. Update Documentation

**Add to code comments**: ISO 32000-1:2008 section references for all decisions

**Example**:
```rust
// Per ISO 32000-1:2008 Section 9.4.4:
// Text positioning is determined by the text matrix and glyph positioning.
// Gaps exceeding word spacing indicate word boundaries.
```

---

## Testing Strategy

### Baseline Comparison

**Current (Phase 9)**:
- Quality: 3.4/10 average
- ArXiv: 4.5/10 (spurious spaces from aggressive gaps)
- Code of Conduct: 0/10 (spacing issues)
- Anti-bribery: 0/10 (spacing issues)
- Diligent: 10/10 (clean PDF)
- Mixed: 7/10

**Expected (Phase 10 spec-compliant)**:
- Simpler logic should handle more cases correctly
- May lose "word fusion recovery" but gain reliability
- Should approach pdfplumber's 0.25em threshold performance

### Test Cases

Run `cargo test --test quality_metrics` on 5-PDF set:
```
✅ Diligent Security Policy (should remain 10/10)
❓ ArXiv Academic (current 4.5, expect improvement via geometric fix)
❓ Code of Conduct (current 0, expect improvement with simpler gaps)
❓ Anti-bribery (current 0, expect improvement with geometric fix)
❓ Mixed (current 7, expect stable or better)
```

---

## Execution Checklist

- [ ] **Step 1**: Comment out `split_fused_words()` call in extraction pipeline
- [ ] **Step 2**: Simplify `should_insert_space()` to 3 rules only
- [ ] **Step 3**: Remove `should_insert_space_heuristic()` function
- [ ] **Step 4**: Remove document-type detection code
- [ ] **Step 5**: Update SpanMergingConfig (remove unused fields or mark deprecated)
- [ ] **Step 6**: Build and test: `cargo build --release 2>&1 | grep -E "(error|warning)"`
- [ ] **Step 7**: Run quality metrics: `cargo test --test quality_metrics --release`
- [ ] **Step 8**: Document results in PHASE10_RESULTS.md
- [ ] **Step 9**: Compare with Phase 9 baseline
- [ ] **Step 10**: Commit with message "Phase 10: PDF spec compliance refactoring - remove non-spec heuristics"

---

## Philosophy

**Why This Matters**:

A spec-compliant implementation:
1. **Maintainable**: Future developers understand it's based on PDF spec, not ad-hoc heuristics
2. **Predictable**: All decisions trace to authoritative source (ISO 32000-1:2008)
3. **Defensible**: When PDF is malformed, we can say "PDF spec doesn't define this case"
4. **Comparable**: Can compare against pdfplumber, PyMuPDF etc. using same criteria

**Acceptance Criteria**:
- ✅ All code changes reference PDF spec section when explaining decisions
- ✅ No non-spec heuristics remain in decision logic
- ✅ Simpler code (fewer rules, fewer config options)
- ✅ Runs on same 5 PDFs with measurable quality metrics

---

## References

- **PDF Specification**: ISO 32000-1:2008 (PDF 1.7)
  - Section 9.3: Text State Parameters
  - Section 9.4.3: Text-Showing Operators (Tj, TJ)
  - Section 9.4.4: Text Space Details (positioning formula)

- **Baseline Reference**: pdfplumber 0.11.8
  - Uses 0.25em word_margin threshold
  - Achieves 0 spurious spaces on policy PDFs

- **Current Issues**:
  - EXTERNAL_COMPARISON_ANALYSIS.md shows pdfplumber outperforms on policy PDFs
  - CamelCase splitting can't handle all-lowercase fusions
  - Multiple thresholds add complexity without proven benefit
