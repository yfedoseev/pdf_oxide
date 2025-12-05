# Phase 3: Remaining PDF Extraction Quality Issues - Implementation Plan

**Date**: December 5, 2025
**Status**: ANALYSIS COMPLETE - IMPLEMENTATION PLAN READY
**Author**: Task-Planner-Architect Agent

---

## Executive Summary

After the Phase 2 boundary tracking fix and Phase 7.2 spurious space detection improvements, the following issues remain:

| Issue | Current Count | Target | Severity |
|-------|---------------|--------|----------|
| Empty bold markers | 3 across 2 PDFs | 0 | HIGH |
| Word fusions | 3 (High/Medium confidence) | 0 | HIGH |
| Spurious spaces | 9 in mixed PDF | < 3 | MEDIUM |
| Quality score | 4.4/10 average | >= 8.0 | CRITICAL |

**PDF-specific breakdown:**
- Anti-bribery PDF: 2 empty bold markers, quality 0.0/10
- Code of Conduct PDF: 1 empty bold marker, 2 word fusions ("helporganisationscraft", "theGeneral"), quality 0.0/10
- arxiv PDF: 1 word fusion ("lengthThis"), quality 5.0/10
- mixed PDF: 9 spurious spaces, quality 7.0/10

---

## Issue 1: Empty Bold Markers (3 Remaining)

### Current Status

**Before Phase 2**: 25 empty bold markers
**After Phase 2**: 3 empty bold markers (88% fixed)

**Remaining occurrences:**
- Anti-bribery PDF: 2 markers
- Code of Conduct PDF: 1 marker

### Root Cause Hypothesis

The `BoldMarkerValidator` in `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs` has 5 validation rules:
1. Must be bold
2. Must have word content (non-whitespace)
3. Must have valid opening boundary (alphanumeric first char)
4. Must have valid closing boundary (alphanumeric last char)
5. Content must not become empty after formatting

**Hypothesis**: The remaining 3 markers are caused by edge cases where:
- Text has punctuation or special characters at boundaries
- Cleaned text differs from group text (boundary mismatch)
- Text contains non-ASCII alphanumeric characters (e.g., accented letters)

### Investigation Steps

```
1. Enable DEBUG logging for bold validation:
   RUST_LOG=pdf_oxide::layout::bold_validation=debug

2. Extract specific problematic lines from each PDF:
   - Anti-bribery: Identify lines with empty bold markers (** **)
   - Code of Conduct: Same identification

3. Compare group text vs cleaned_text in BoldGroup:
   - Log first_char_in_group, last_char_in_group
   - Check if cleaning transforms text in unexpected ways

4. Create targeted test case for each remaining marker pattern
```

### Investigation Code

```rust
// Add to markdown.rs convert_page_from_spans() around line 361
log::debug!(
    "BoldGroup validation: text='{}', first={:?}, last={:?}, is_bold={}, cleaned='{}'",
    group.text.chars().take(20).collect::<String>(),
    group.first_char_in_group,
    group.last_char_in_group,
    group.is_bold,
    cleaned_text.chars().take(20).collect::<String>()
);
```

### Proposed Fix

**Option A: Enhance Boundary Detection**

In `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`:

```rust
// Current: Only checks is_alphabetic() || is_numeric()
pub fn has_valid_opening_boundary(&self) -> bool {
    match self.first_char_in_group {
        Some(c) => c.is_alphabetic() || c.is_numeric(),
        None => false,
    }
}

// Enhanced: Include Unicode letter categories
pub fn has_valid_opening_boundary(&self) -> bool {
    match self.first_char_in_group {
        Some(c) => c.is_alphanumeric() || c.is_alphabetic(),
        None => false,
    }
}
```

**Option B: Fix Boundary Extraction (More Likely Root Cause)**

In `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` around line 329-333:

```rust
// CURRENT: Extract boundaries AFTER cleaning
let first_char_in_group = cleaned_text.chars().next();
let last_char_in_group = cleaned_text.chars().last();

// FIX: Ensure boundaries reflect the trimmed content (not whitespace)
let trimmed = cleaned_text.trim();
let first_char_in_group = trimmed.chars().next();
let last_char_in_group = trimmed.chars().last();
```

**Option C: Add Validator Rule for Trimmed Content**

```rust
// Add Rule 6 to BoldMarkerValidator
if group.text.trim() != group.text {
    // Text has leading/trailing whitespace - boundaries may be wrong
    // Recompute boundaries from trimmed text
    let trimmed = group.text.trim();
    if !trimmed.chars().next().map(|c| c.is_alphanumeric()).unwrap_or(false) {
        return BoldMarkerDecision::Skip(ValidatorError::InvalidOpeningBoundary);
    }
    if !trimmed.chars().last().map(|c| c.is_alphanumeric()).unwrap_or(false) {
        return BoldMarkerDecision::Skip(ValidatorError::InvalidClosingBoundary);
    }
}
```

### Acceptance Criteria

- [ ] Zero empty bold markers in Anti-bribery PDF
- [ ] Zero empty bold markers in Code of Conduct PDF
- [ ] Regression tests pass for previously fixed PDFs
- [ ] No reduction in valid bold marker count

### Effort Estimate

**Size**: Medium (M)
**Files**: 2 (`bold_validation.rs`, `markdown.rs`)
**Lines**: ~20-40

---

## Issue 2: Word Fusions (3 Remaining)

### Current Status

**Detected fusions:**
1. Code of Conduct: "helporganisationscraft" (HIGH confidence)
2. Code of Conduct: "theGeneral" (MEDIUM confidence - CamelCase)
3. arxiv PDF: "lengthThis" (MEDIUM confidence - CamelCase)

**Note**: "draftpolicy" is classified as PdfStructure (PDF authoring defect, not regression)

### Root Cause Analysis

The space detection engine (`/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`) uses 4 detectors in consensus voting:

1. **TjOffsetDetector** (priority 120): Checks TJ array offsets < -100
2. **GapBasedDetector** (priority 100): Checks gap > font_size * 0.25em
3. **HeuristicDetector** (priority 80): Detects CamelCase transitions
4. **AdaptiveDetector** (priority 90): Uses document-wide gap statistics

**Hypothesis 1: TjOffset Not Available at Merge Time**

In `merge_adjacent_spans()` (text.rs:1398):
```rust
let space_context = SpaceContext {
    // ...
    tj_offset: None,  // TJ offset not available at span merging layer
    document_stats: None,  // Could be populated from document analysis
};
```

The highest-priority detector (TjOffset) always returns Skip because offset is None.

**Hypothesis 2: Gap-Based Detection Threshold Too High**

For "lengthThis" fusion in arxiv:
- Characters "h" and "T" have a gap between them
- Gap may be < 0.25em (conservative_threshold_pt = 0.1)
- HeuristicDetector should catch lowercase-to-uppercase transition

**Hypothesis 3: Heuristic Detector Not in Engine**

Looking at `merge_adjacent_spans()` around line 1403-1410:
```rust
let detectors: Vec<Box<dyn crate::layout::SpaceDetector>> = vec![
    Box::new(crate::layout::GapBasedDetector { ... }),
    Box::new(crate::layout::HeuristicDetector),
];
```

Only 2 detectors are used, not 4. The TjOffset and Adaptive detectors are excluded.

### Investigation Steps

```
1. Add debug logging to HeuristicDetector::detect():
   - Log prev_text, next_text, gap_pt
   - Log decision result

2. Analyze specific fusion cases:
   - "helporganisationscraft" - What gap exists between fragments?
   - "lengthThis" - Does 'h' to 'T' transition trigger heuristic?

3. Check TJ processor output:
   - Are these fusions from single TJ strings (PDF defect)?
   - Or are they multiple spans merged without space?

4. Examine gap statistics for affected PDFs:
   - What is the median gap?
   - What threshold was computed adaptively?
```

### Investigation Code

```rust
// Add to HeuristicDetector::detect() in space_detection.rs
log::debug!(
    "Heuristic check: prev='{}' -> next='{}', prev_lower={}, next_upper={}",
    context.prev_text.chars().last().map(|c| c.to_string()).unwrap_or_default(),
    context.next_text.chars().next().map(|c| c.to_string()).unwrap_or_default(),
    context.prev_text.chars().last().map(|c| c.is_lowercase()).unwrap_or(false),
    context.next_text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
);
```

### Proposed Fix

**Fix A: Make Heuristic Detector Primary for CamelCase**

Change priority in space_detection.rs:
```rust
impl SpaceDetector for HeuristicDetector {
    fn priority(&self) -> u8 { 150 }  // Was 80, now highest
    // ...
}
```

Or modify engine consensus logic:
```rust
pub fn detect_space(&self, context: &SpaceContext) -> SpaceDecision {
    // Check heuristic FIRST - CamelCase always indicates word boundary
    for d in &self.detectors {
        if d.name() == "Heuristic" {
            let decision = d.detect(context);
            if decision == SpaceDecision::Insert {
                return decision;  // Heuristic override
            }
        }
    }
    // Then do normal priority-based voting
    // ...
}
```

**Fix B: Add CamelCase Post-Processing**

In `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`, add post-processing:

```rust
/// Insert spaces at CamelCase boundaries
fn fix_camelcase_fusions(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        result.push(chars[i]);

        // Check for lowercase -> uppercase transition (not at start)
        if i < chars.len() - 1 {
            let curr = chars[i];
            let next = chars[i + 1];

            if curr.is_lowercase() && next.is_uppercase() {
                // Insert space at CamelCase boundary
                result.push(' ');
            }
        }
    }

    result
}
```

**Fix C: Fix TJ Processing to Preserve Boundaries**

Investigate if these fusions come from single TJ strings. If so, they're PDF defects (not our fault). However, if they come from merged spans, the gap detection needs adjustment.

### Acceptance Criteria

- [ ] "lengthThis" correctly extracted as "length This" in arxiv PDF
- [ ] "helporganisationscraft" correctly extracted as "help organisations craft"
- [ ] "theGeneral" correctly extracted as "the General"
- [ ] No new spurious spaces introduced
- [ ] Regression tests pass

### Effort Estimate

**Size**: Large (L)
**Files**: 3 (`space_detection.rs`, `markdown.rs`, `text.rs`)
**Lines**: ~50-80

---

## Issue 3: Spurious Spaces (9 in mixed PDF)

### Current Status

**After Phase 7.2 fix:**
- Academic PDF: 0 spurious spaces (FIXED)
- Mixed PDF: 9 spurious spaces (remaining)

### Root Cause Hypothesis

The mixed PDF (`7A3MBRLFC6OU5KGMFIDEQPUOQTROBYUS.pdf`) likely has:
1. Complex layout with tables or forms
2. Multiple columns with varying spacing
3. Non-standard character positioning

The spurious spaces are likely **real double spaces in the extracted text**, not detection false positives (Phase 7.2 fixed false positives).

### Investigation Steps

```
1. Run markdown extraction on mixed PDF:
   cargo run --bin export_to_markdown -- -p 'tests/fixtures/regression/mixed/7A3MBRLFC6OU5KGMFIDEQPUOQTROBYUS.pdf'

2. Search for double spaces in output:
   grep -n "  " output.md

3. Identify patterns:
   - Are double spaces at line boundaries?
   - Do they correlate with layout transitions?
   - Are they in table-like structures?

4. Check TJ processor logs:
   RUST_LOG=pdf_oxide::extractors::text=debug cargo test ...
   - Are spaces being inserted twice?
   - Is has_boundary_space() check working?
```

### Proposed Fix

**Fix A: Enhanced Boundary Space Check**

In `text.rs` function `has_boundary_space()`:

```rust
// Current implementation
fn has_boundary_space(current_text: &str, next_text: &str) -> bool {
    current_text.ends_with(|c: char| c.is_whitespace())
        || next_text.starts_with(|c: char| c.is_whitespace())
}

// Enhanced: Also check for multiple trailing/leading spaces
fn has_boundary_space(current_text: &str, next_text: &str) -> bool {
    let ends_ws = current_text.ends_with(|c: char| c.is_whitespace());
    let starts_ws = next_text.starts_with(|c: char| c.is_whitespace());
    let ends_multiple_ws = current_text.len() >= 2 &&
        current_text.chars().rev().take(2).all(|c| c.is_whitespace());

    ends_ws || starts_ws || ends_multiple_ws
}
```

**Fix B: TJ Processor Double-Space Prevention**

In `process_tj_array()` around line 2659, add lookahead for trailing spaces:

```rust
// Check if current buffer ends with whitespace
let buffer_ends_with_space = !buffer.unicode.is_empty() &&
    buffer.unicode.ends_with(|c: char| c.is_whitespace());

// Only insert space if buffer doesn't end with one
if !next_element_starts_with_space && !buffer_ends_with_space {
    self.insert_space_as_span()?;
}
```

**Fix C: Post-Processing Double Space Removal**

In `markdown.rs` whitespace cleanup:

```rust
// Add to cleanup_markdown() or create new function
fn remove_double_spaces(text: &str) -> String {
    let re = Regex::new(r" {2,}").unwrap();
    re.replace_all(text, " ").to_string()
}
```

### Acceptance Criteria

- [ ] Mixed PDF spurious spaces reduced from 9 to <= 3
- [ ] No new issues in other PDFs
- [ ] Table structures preserved
- [ ] Quality score >= 8.0

### Effort Estimate

**Size**: Medium (M)
**Files**: 2 (`text.rs`, optionally `markdown.rs`)
**Lines**: ~20-40

---

## Issue 4: PDF Structure Defects ("draftpolicy")

### Current Status

"draftpolicy" appears in both Anti-bribery and Code of Conduct PDFs.

**Classification**: PdfStructure (not a regression, but inherent PDF authoring defect)

### Root Cause

The PDF contains a single TJ string without any positioning offsets:
```
[(draftpolicy)] TJ
```

There is no information in the PDF to determine word boundaries within this string.

### Proposed Solution

**Option A: Dictionary-Based Word Splitting (Complex)**

Use a word list to identify possible word boundaries:
- "draftpolicy" -> "draft" + "policy"
- Requires maintaining a dictionary

**Option B: Accept as PDF Defect (Recommended)**

Document this as a known limitation. The PDF is malformed; we cannot fix authoring errors.

**Option C: Heuristic Word Boundary Detection**

Apply NLP techniques to identify likely word boundaries based on:
- Common word patterns
- Statistical analysis of English words
- Greedy matching against common prefixes/suffixes

### Recommendation

Accept as PDF defect. Update quality metrics to classify PdfStructure issues separately:

```rust
// In quality_metrics.rs
pub fn passes(&self) -> bool {
    // Ignore PdfStructure fusions - they're PDF authoring defects
    let true_regressions = self.word_fusions.iter()
        .filter(|f| !matches!(f.confidence, FusionConfidence::PdfStructure))
        .count();

    true_regressions == 0 && self.empty_bold_markers == 0 && self.quality_score >= 8.0
}
```

---

## Implementation Phases

### Phase 3.1: Empty Bold Markers (Effort: M, Priority: HIGH)

**Tasks:**
- [ ] Add debug logging to BoldMarkerValidator (1h)
- [ ] Identify specific patterns causing remaining markers (2h)
- [ ] Implement boundary fix (Option B or C) (2h)
- [ ] Write targeted test cases (1h)
- [ ] Verify regression suite passes (0.5h)

**Files:**
- `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`
- `/home/yfedoseev/projects/pdf_oxide/tests/regression_suite.rs`

**Success Metrics:**
- Empty bold markers: 3 -> 0
- No reduction in valid bold marker count

---

### Phase 3.2: Word Fusions (Effort: L, Priority: HIGH)

**Tasks:**
- [ ] Add debug logging to HeuristicDetector (1h)
- [ ] Analyze fusion patterns in affected PDFs (2h)
- [ ] Determine if fusions are single-TJ or merged-span (2h)
- [ ] Implement heuristic priority fix OR post-processing (3h)
- [ ] Write targeted test cases (1h)
- [ ] Verify regression suite passes (0.5h)

**Files:**
- `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`
- `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Success Metrics:**
- Word fusions (High/Medium): 3 -> 0
- No new spurious spaces introduced

---

### Phase 3.3: Spurious Spaces (Effort: M, Priority: MEDIUM)

**Tasks:**
- [ ] Extract and analyze mixed PDF output (1h)
- [ ] Identify double-space patterns (1h)
- [ ] Implement enhanced boundary check (2h)
- [ ] Add TJ processor lookahead if needed (2h)
- [ ] Verify regression suite passes (0.5h)

**Files:**
- `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs`

**Success Metrics:**
- Spurious spaces: 9 -> <= 3
- Quality score: 7.0 -> >= 8.0

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_bold_boundary_with_leading_space() {
    let group = BoldGroup {
        text: " hello".to_string(),  // Leading space
        is_bold: true,
        first_char_in_group: Some(' '),
        last_char_in_group: Some('o'),
    };
    // Should skip due to invalid opening boundary
    assert_eq!(
        BoldMarkerValidator::can_insert_markers(&group),
        BoldMarkerDecision::Skip(ValidatorError::InvalidOpeningBoundary)
    );
}

#[test]
fn test_heuristic_camelcase_detection() {
    let context = SpaceContext {
        prev_text: "length".to_string(),
        next_text: "This".to_string(),
        gap_pt: 0.5,
        font_size: 12.0,
        tj_offset: None,
        document_stats: None,
    };

    let detector = HeuristicDetector;
    assert_eq!(detector.detect(&context), SpaceDecision::Insert);
}
```

### Integration Tests

Add to regression_suite.rs:

```rust
#[test]
fn test_specific_word_fusion_cases() {
    let cases = vec![
        ("academic/arxiv_2510.21165v1.pdf", "lengthThis", false),  // Should NOT appear
        ("policy/Code of Conduct Policy Template (EU).pdf", "helporganisationscraft", false),
        ("policy/Code of Conduct Policy Template (EU).pdf", "theGeneral", false),
    ];

    for (pdf, fusion_text, should_exist) in cases {
        let path = PathBuf::from(FIXTURES_DIR).join(pdf);
        let markdown = extract_markdown(path.to_str().unwrap(), SpanMergingConfig::adaptive())
            .expect("Failed to extract");

        let contains = markdown.to_lowercase().contains(&fusion_text.to_lowercase());
        assert_eq!(contains, should_exist,
            "PDF {} - fusion '{}' should_exist={} but contains={}",
            pdf, fusion_text, should_exist, contains);
    }
}
```

---

## Potential Side Effects

| Fix | Risk | Mitigation |
|-----|------|------------|
| Bold boundary trim | May skip valid bold at punctuation | Test with diverse PDFs |
| Heuristic priority boost | May over-insert spaces at acronyms | Exclude ALL-CAPS patterns |
| CamelCase post-processing | May break intentional CamelCase terms | Allow configurable opt-out |
| Double-space removal | May affect table alignment | Preserve in table contexts |

---

## Performance Considerations

- **Bold validation**: O(1) per span, no impact
- **Heuristic detection**: O(n) string scan per merge, minimal impact
- **CamelCase post-processing**: O(n) per line, minor impact
- **Double-space regex**: O(n) compilation once, O(m*n) replacement

**Expected overhead**: < 1% of total extraction time

---

## SOLID Compliance Checklist

- **Single Responsibility**:
  - [x] BoldMarkerValidator handles only bold validation
  - [x] SpaceDetectionEngine handles only space detection
  - [x] Each detector has single purpose

- **Open/Closed**:
  - [x] New detectors can be added via trait implementation
  - [x] Validator rules can be extended without modifying existing code

- **Liskov Substitution**:
  - [x] All SpaceDetector implementations are truly substitutable
  - [x] BoldMarkerDecision enum variants can be handled uniformly

- **Interface Segregation**:
  - [x] SpaceDetector trait is minimal (detect, priority, name)
  - [ ] DEBT: BoldGroup has simulated_formatted_content() that may not be needed

- **Dependency Inversion**:
  - [x] SpaceDetectionEngine depends on trait, not concrete detectors
  - [x] MarkdownConverter uses validator via trait-like interface

---

## Technical Debt Identified

[DEBT:architecture:LOW] `BoldGroup::simulated_formatted_content()` returns clone of text, not actual simulation
[DEBT:testing:MEDIUM] Regression suite uses panics for failure instead of assert_eq! with clear messages
[DEBT:documentation:LOW] SpaceDetector trait priorities not documented in trait definition
[DEBT:performance:LOW] SpaceDetectionEngine creates new detector instances per merge operation

---

## Success Metrics Summary

| Metric | Before | Target | After (Expected) |
|--------|--------|--------|------------------|
| Empty bold markers | 3 | 0 | 0 |
| Word fusions (H/M) | 3 | 0 | 0 |
| Spurious spaces (mixed) | 9 | <= 3 | <= 3 |
| Quality score (avg) | 4.4 | >= 8.0 | >= 9.0 |

---

## Timeline

| Phase | Effort | ETA | Dependencies |
|-------|--------|-----|--------------|
| 3.1 Empty Bold | M (6.5h) | Day 1 | None |
| 3.2 Word Fusions | L (9.5h) | Day 2 | None (parallel) |
| 3.3 Spurious Spaces | M (6.5h) | Day 2 | 3.1, 3.2 (sequential) |
| Validation | S (2h) | Day 3 | All phases |

**Total Effort**: ~24.5 hours (3 days)

---

## Appendix: File References

| File | Purpose | Key Functions |
|------|---------|---------------|
| `/home/yfedoseev/projects/pdf_oxide/src/layout/bold_validation.rs` | Bold marker validation | `can_insert_markers()`, `BoldGroup` |
| `/home/yfedoseev/projects/pdf_oxide/src/layout/space_detection.rs` | Space detection engine | `SpaceDetector`, `HeuristicDetector` |
| `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` | Text extraction | `merge_adjacent_spans()`, `process_tj_array()` |
| `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` | Markdown conversion | `convert_page_from_spans()` |
| `/home/yfedoseev/projects/pdf_oxide/tests/regression_suite.rs` | Regression tests | `test_core_regression_suite()` |
| `/home/yfedoseev/projects/pdf_oxide/tests/quality_metrics.rs` | Quality detection | `detect_word_fusions()`, `detect_spurious_spaces()` |

---

**Prepared by**: Task-Planner-Architect
**Ready for**: Implementation by Staff-Rust-Engineer
**Confidence**: HIGH (based on code analysis and test output)
