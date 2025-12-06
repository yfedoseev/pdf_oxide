# Implementation Plan: Phase 5 (Word Fusion) and Phase 6 (Empty Bold)

**Status**: Ready for Implementation
**Priority**: P0 (Critical Quality Issues)
**Target Quality**: 3.4/10 -> 10/10 (5 of 5 PDFs passing)
**Estimated Effort**: 2-3 days total

---

## Executive Summary

This document provides detailed step-by-step implementation plans for fixing the three remaining quality issues identified in pdf_oxide:

| Issue | Count | Root Cause | Fix Strategy | Risk |
|-------|-------|------------|--------------|------|
| Word Fusions | 3 | Single TJ strings encoding multiple words | CamelCase post-processor | Medium |
| Spurious Spaces | 1,623 | Spaces WITHIN words (Pattern 2) | Already addressed by adaptive threshold | N/A |
| Empty Bold Markers | 3 | Whitespace-only bold text | Pre-validation filter | Low |

**Critical Constraint**: Diligent Security PDF (10/10) MUST remain passing as the control.

---

## Pre-Implementation Checklist

Before starting, verify current state:

```bash
# Run baseline tests
cd /home/yfedoseev/projects/pdf_oxide
cargo test --test regression_suite test_core_regression_suite -- --nocapture

# Verify Diligent Security passes (control)
cargo test --test regression_suite test_empty_bold_markers_regression -- --nocapture
```

Expected baseline:
- Quality: 3.4/10 (1 of 5 PDFs passing)
- Diligent Security: 10/10 (must remain)

---

## Phase 5: Word Fusion Post-Processor

### Problem Analysis

**Issue**: 3 word fusions detected:
1. `"theGeneral"` - Code of Conduct PDF (line ~51)
2. `"lengthThis"` - arXiv PDF (line ~352)
3. `"helporganisationscraft"` - Code of Conduct PDF (line ~2)

**Root Cause**: PDF encodes multiple words as a single TJ string without spacing offsets:
```
[(theGeneral)] TJ  <-- Single string, no word boundary info
```

**Current State**: Word segmentation module exists (`src/extractors/word_segmentation.rs`) with:
- CamelCase detector for patterns like "theGeneral", "lengthThis"
- Dictionary-based Viterbi algorithm for "helporganisationscraft"

The segmentation is already called during extraction (`src/extractors/text.rs:1966-1968`), but the results may not be propagating correctly to the markdown output.

### Step-by-Step Implementation

#### Step 5.1: Verify Segmentation Function Works (30 min)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/extractors/word_segmentation.rs`

First, confirm the segmentation functions work correctly:

```bash
# Run word segmentation unit tests
cargo test --package pdf_oxide --lib extractors::word_segmentation::tests -- --nocapture
```

Expected tests to pass:
- `test_helporganisationscraft`
- `test_theGeneral` (CamelCase variant)

If tests fail, fix the segmentation logic first.

#### Step 5.2: Trace Where Segmentation Gets Lost (1 hour)

**Files to investigate**:
1. `/home/yfedoseev/projects/pdf_oxide/src/extractors/text.rs` (lines ~1960-2055)
2. `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (lines ~197-484)

**Diagnostic Approach**:

Add debug logging at key points to trace flow:

```rust
// In text.rs, after split_fused_words call (~line 2050)
log::debug!("WORD_FUSION_TRACE: Input '{}' -> Output {:?}",
    original_text, split_result);
```

Run with debug logging:
```bash
RUST_LOG=debug cargo test --test regression_suite test_word_fusion_regression_policy -- --nocapture 2>&1 | grep "WORD_FUSION"
```

#### Step 5.3: Implement Post-Processor in Markdown Converter (2 hours)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Location**: After `cleanup_markdown()` call (line ~483)

**Implementation Strategy**:

Option A: Apply segmentation during span-to-block conversion (lines ~209-222):

```rust
// In convert_page_from_spans(), after creating blocks from spans
// Apply word fusion splitting before any other processing
blocks = Self::split_fused_words_in_blocks(blocks);
```

Add new method:

```rust
/// Phase 5: Split fused words in text blocks.
///
/// Applies word segmentation to detect and split CamelCase and dictionary-based
/// word fusions that weren't detected during extraction.
///
/// Examples:
/// - "theGeneral" -> "the General"
/// - "lengthThis" -> "length This"
/// - "helporganisationscraft" -> "help organisations craft"
///
/// # Arguments
///
/// * `blocks` - Text blocks that may contain fused words
///
/// # Returns
///
/// Text blocks with fused words split into separate spans
fn split_fused_words_in_blocks(blocks: Vec<TextBlock>) -> Vec<TextBlock> {
    use crate::extractors::word_segmentation::{split_camelcase, segment_word};

    let mut result = Vec::with_capacity(blocks.len());

    for block in blocks {
        // Skip empty or whitespace-only blocks
        if block.text.trim().is_empty() {
            continue;
        }

        // Check for CamelCase patterns first (most common)
        let words: Vec<String> = block.text
            .split_whitespace()
            .flat_map(|word| {
                // Try CamelCase split first
                let camel_split = split_camelcase(word);
                if camel_split.len() > 1 {
                    camel_split.into_iter().map(String::from).collect::<Vec<_>>()
                } else {
                    // Try dictionary-based segmentation for all-lowercase
                    match segment_word(word) {
                        Some(segments) => segments,
                        None => vec![word.to_string()],
                    }
                }
            })
            .collect();

        // If no splits occurred, keep original block
        if words.len() == 1 && words[0] == block.text.trim() {
            result.push(block);
            continue;
        }

        // Create new text with proper spacing
        let new_text = words.join(" ");

        // Create new block with updated text
        // Note: Bounding box remains the same (conservative)
        let mut new_block = block.clone();
        new_block.text = new_text;
        result.push(new_block);
    }

    result
}
```

**Alternative Option B**: Apply as post-processing to final markdown string (simpler but less precise):

```rust
// After cleanup_markdown() call
let markdown = cleanup_markdown(&spaced);

// Phase 5: Post-process word fusions
let markdown = Self::split_fused_words_in_text(&markdown);

Ok(markdown)
```

```rust
/// Phase 5: Split fused words in final markdown text.
///
/// Applies regex-based CamelCase splitting to the output text.
/// This is a fallback for fusions that weren't caught during extraction.
fn split_fused_words_in_text(text: &str) -> String {
    use regex::Regex;

    lazy_static::lazy_static! {
        // CamelCase pattern: lowercase letters followed by uppercase
        static ref RE_CAMELCASE: Regex =
            Regex::new(r"([a-z]{2,})([A-Z][a-z]{2,})").unwrap();
    }

    RE_CAMELCASE.replace_all(text, "$1 $2").to_string()
}
```

**Recommendation**: Start with Option B (simpler), test, then consider Option A if bounding box accuracy matters.

#### Step 5.4: Add Unit Tests (1 hour)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (test section)

```rust
#[test]
fn test_phase5_camelcase_fusion_split() {
    // Test "theGeneral" pattern
    let input = "The theGeneral was mentioned in the document.";
    let result = MarkdownConverter::split_fused_words_in_text(input);
    assert!(result.contains("the General"),
        "CamelCase 'theGeneral' should be split");
    assert!(!result.contains("theGeneral"),
        "Original fusion should not remain");
}

#[test]
fn test_phase5_lengththis_fusion_split() {
    // Test "lengthThis" pattern
    let input = "The lengthThis value was wrong.";
    let result = MarkdownConverter::split_fused_words_in_text(input);
    assert!(result.contains("length This"),
        "CamelCase 'lengthThis' should be split");
}

#[test]
fn test_phase5_preserves_legitimate_camelcase() {
    // Don't split legitimate terms
    let input = "The JavaScript and TypeScript support.";
    let result = MarkdownConverter::split_fused_words_in_text(input);
    // These should remain (tech terms)
    assert!(result.contains("JavaScript"));
    assert!(result.contains("TypeScript"));
}

#[test]
fn test_phase5_no_false_positives_on_clean_text() {
    // Clean text should pass through unchanged
    let input = "This is a clean document with proper spacing.";
    let result = MarkdownConverter::split_fused_words_in_text(input);
    assert_eq!(result, input, "Clean text should be unchanged");
}
```

#### Step 5.5: Integration Test (30 min)

Run regression suite to verify fix:

```bash
cargo test --test regression_suite test_word_fusion_regression_policy -- --nocapture
```

Expected: 0 word fusions detected.

**Verify no regressions**:
```bash
cargo test --test regression_suite test_core_regression_suite -- --nocapture
```

Expected: Diligent Security PDF still passes (10/10).

### Phase 5 Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-splitting legitimate terms | Medium | Medium | Maintain exclusion list for tech terms (JavaScript, TypeScript, etc.) |
| Bounding box inaccuracy | Low | Low | Option A preserves boxes; Option B doesn't affect positioning |
| Performance overhead | Low | Low | Regex is fast; dictionary lookup is O(1) |
| Regression on Diligent PDF | Low | High | Test after each change |

### Phase 5 Rollback Plan

If issues occur:
1. Revert changes to `markdown.rs`
2. Keep word segmentation module unchanged (it's already tested)
3. Document in ADR why post-processing approach failed

---

## Phase 6: Empty Bold Marker Filtering

### Problem Analysis

**Issue**: 3 empty bold markers (`** **` patterns) detected.

**Root Cause**: Whitespace-only text blocks marked as bold create:
```markdown
**   **   <!-- Empty bold: spaces only -->
```

**Current State**: Bold validation exists in `src/layout/bold_validation.rs`:
- `BoldMarkerValidator::can_insert_markers()` checks for whitespace-only content
- Pre-filter in `convert_page_from_spans()` removes whitespace blocks (lines 238-253)

However, 3 empty markers still appear, indicating the filter is incomplete.

### Step-by-Step Implementation

#### Step 6.1: Diagnose Which Markers Slip Through (30 min)

**Add diagnostic logging**:

```rust
// In markdown.rs, before line 424 (marker insertion)
if matches!(marker_decision, BoldMarkerDecision::Insert) {
    log::debug!("BOLD_MARKER_TRACE: Inserting markers for '{}'",
        group.text.chars().take(30).collect::<String>());
}
```

Run with logging:
```bash
RUST_LOG=debug cargo test --test regression_suite test_empty_bold_markers_regression -- --nocapture 2>&1 | grep "BOLD_MARKER"
```

Look for markers being inserted around whitespace content.

#### Step 6.2: Strengthen Whitespace Pre-Filter (1 hour)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Location**: Lines 238-253 (existing filter)

The current filter removes `block.text.trim().is_empty()` blocks. Strengthen to also handle:

1. Unicode whitespace variants (NBSP, figure space, etc.)
2. Blocks with only punctuation/symbols marked bold

**Enhanced filter**:

```rust
// Replace lines 238-253 with:

// **Phase 6 Enhanced: Pre-Validation Bold Filter**
// Filter whitespace-only blocks BEFORE any grouping to prevent empty bold markers.
let initial_count = blocks.len();
let mut whitespace_count = 0;

blocks.retain(|block| {
    // Check for any form of whitespace (including Unicode variants)
    let is_whitespace_only = !block.text.chars().any(|c| {
        !c.is_whitespace() &&
        c != '\u{00A0}' &&  // Non-breaking space
        c != '\u{2007}' &&  // Figure space
        c != '\u{202F}' &&  // Narrow no-break space
        c != '\u{3000}' &&  // Ideographic space
        c != '\u{FEFF}'     // Zero-width no-break space
    });

    if is_whitespace_only {
        whitespace_count += 1;
        log::debug!("Phase 6: Filtering whitespace-only block: '{:?}'",
            block.text.chars().take(20).collect::<String>());
    }

    !is_whitespace_only
});

log::debug!(
    "Phase 6: Pre-filter removed {} whitespace blocks ({} -> {})",
    whitespace_count, initial_count, blocks.len()
);
```

#### Step 6.3: Add Secondary Filter at Markdown Generation (1 hour)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs`

**Location**: Lines 392-438 (bold marker insertion logic)

Add additional validation right before inserting markers:

```rust
// Before line 421 (existing marker decision)
// Phase 6.3: Final validation before marker insertion
let should_skip_markers = {
    let trimmed = group.text.trim();
    // Skip if content is empty or only whitespace
    trimmed.is_empty() ||
    // Skip if content has no alphanumeric characters
    !trimmed.chars().any(|c| c.is_alphanumeric())
};

if should_skip_markers {
    log::debug!("Phase 6.3: Skipping bold markers for non-content: '{}'",
        group.text.chars().take(20).collect::<String>());
}

let marker_decision = if should_check_validator && !should_skip_markers {
    BoldMarkerValidator::can_insert_markers(&group)
} else {
    BoldMarkerDecision::Skip(
        crate::layout::bold_validation::ValidatorError::WhitespaceOnly,
    )
};
```

#### Step 6.4: Add Markdown Post-Cleanup (30 min)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/whitespace.rs`

Add a final cleanup pass to catch any remaining empty markers:

```rust
/// Phase 6: Remove any remaining empty bold markers.
///
/// This is a safety net to catch `** **` patterns that slip through validation.
pub fn remove_empty_bold_markers(text: &str) -> String {
    use regex::Regex;

    lazy_static::lazy_static! {
        // Match "**" followed by whitespace followed by "**"
        static ref RE_EMPTY_BOLD: Regex =
            Regex::new(r"\*\*\s+\*\*").unwrap();
    }

    RE_EMPTY_BOLD.replace_all(text, "").to_string()
}
```

Call this from `cleanup_markdown()`:

```rust
pub fn cleanup_markdown(text: &str) -> String {
    let cleaned = normalize_horizontal_whitespace(text);
    let cleaned = remove_empty_bold_markers(&cleaned);  // Phase 6 addition
    // ... rest of cleanup
    cleaned
}
```

#### Step 6.5: Add Unit Tests (30 min)

**File**: `/home/yfedoseev/projects/pdf_oxide/src/converters/markdown.rs` (test section)

```rust
#[test]
fn test_phase6_no_empty_bold_markers() {
    use crate::layout::TextSpan;

    let converter = MarkdownConverter::new();
    let options = ConversionOptions::default();

    // Scenario: Bold whitespace spans mixed with content
    let spans = vec![
        TextSpan {
            text: "Title".to_string(),
            bbox: Rect::new(0.0, 0.0, 40.0, 14.0),
            font_name: "Times-Bold".to_string(),
            font_size: 14.0,
            font_weight: FontWeight::Bold,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
        },
        TextSpan {
            text: "   ".to_string(),  // Whitespace only - should NOT create ** **
            bbox: Rect::new(50.0, 0.0, 10.0, 14.0),
            font_name: "Times-Bold".to_string(),
            font_size: 14.0,
            font_weight: FontWeight::Bold,
            color: Color::black(),
            mcid: None,
            sequence: 1,
            split_boundary_before: false,
        },
        TextSpan {
            text: "Content".to_string(),
            bbox: Rect::new(0.0, 20.0, 50.0, 12.0),
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            sequence: 2,
            split_boundary_before: false,
        },
    ];

    let result = converter.convert_page_from_spans(&spans, &options).unwrap();

    // Must not contain empty bold markers
    assert!(!result.contains("** **"), "Empty bold markers detected: ** **");
    assert!(!result.contains("**\n**"), "Empty bold markers with newline");
    assert!(!result.contains("**  **"), "Empty bold markers with spaces");

    // Content should still be present
    assert!(result.contains("Title"));
    assert!(result.contains("Content"));
}

#[test]
fn test_phase6_unicode_whitespace_not_bolded() {
    use crate::layout::TextSpan;

    let converter = MarkdownConverter::new();
    let options = ConversionOptions::default();

    // Scenario: NBSP-only span marked bold
    let spans = vec![
        TextSpan {
            text: "\u{00A0}\u{00A0}".to_string(),  // Two NBSPs
            bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
            font_name: "Times-Bold".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Bold,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
        },
    ];

    let result = converter.convert_page_from_spans(&spans, &options).unwrap();

    // Should not contain bold markers around NBSP
    assert!(!result.contains("**"), "NBSP should not create bold markers");
}
```

#### Step 6.6: Integration Test (30 min)

Run regression suite:

```bash
cargo test --test regression_suite test_empty_bold_markers_regression -- --nocapture
```

Expected: 0 empty bold markers.

**Verify no regressions**:
```bash
cargo test --test regression_suite test_core_regression_suite -- --nocapture
```

### Phase 6 Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-filtering legitimate bold | Low | Medium | Only filter whitespace/non-alphanumeric content |
| Performance overhead | Very Low | Low | Simple string operations |
| Regression on Diligent PDF | Very Low | High | Filter only adds restrictions |
| Missing edge cases | Medium | Low | Post-cleanup regex as safety net |

### Phase 6 Rollback Plan

If issues occur:
1. Revert whitespace filter changes
2. Keep post-cleanup regex as minimal fix
3. Document which specific content triggered false positives

---

## Testing Strategy

### Unit Test Coverage

| Component | Tests Required | Estimated Count |
|-----------|---------------|-----------------|
| CamelCase splitter | Split patterns, preserve legitimate terms | 5 |
| Dictionary segmentation | Known fusions, edge cases | 3 |
| Whitespace filter | All Unicode variants | 6 |
| Bold marker validation | Boundary conditions | 4 |
| Post-cleanup regex | Empty marker patterns | 3 |
| **Total** | | **21 new tests** |

### Integration Tests

1. **Word Fusion Regression** (`test_word_fusion_regression_policy`)
   - Expected: 0 fusions in 3 policy PDFs

2. **Empty Bold Markers** (`test_empty_bold_markers_regression`)
   - Expected: 0 markers in 2 styled PDFs

3. **Control Validation** (`test_core_regression_suite`)
   - Expected: Diligent Security PDF remains 10/10

### Regression Suite Execution

```bash
# Full regression suite
cargo test --test regression_suite test_core_regression_suite -- --nocapture

# Word fusion focus
cargo test --test regression_suite test_word_fusion_regression_policy -- --nocapture

# Empty bold focus
cargo test --test regression_suite test_empty_bold_markers_regression -- --nocapture

# Quality metrics unit tests
cargo test --test quality_metrics -- --nocapture
```

---

## Implementation Order

### Recommended Sequence

1. **Phase 6 First** (Lower Risk)
   - Day 1 morning: Steps 6.1-6.4
   - Day 1 afternoon: Steps 6.5-6.6, integration tests
   - Commit: "Phase 6: Fix empty bold marker filtering"

2. **Phase 5 Second** (Medium Risk)
   - Day 2 morning: Steps 5.1-5.3
   - Day 2 afternoon: Steps 5.4-5.5, integration tests
   - Commit: "Phase 5: Add word fusion post-processor"

3. **Final Validation** (Day 3)
   - Run comprehensive regression suite
   - Fix any edge cases
   - Update documentation

### Commit Strategy

Each phase gets its own commit with:
- Clear description of changes
- Test results (before/after)
- No regressions confirmed

```bash
# After Phase 6
git add -A
git commit -m "$(cat <<'EOF'
Phase 6: Fix empty bold marker filtering

- Strengthen whitespace pre-filter in convert_page_from_spans()
- Add Unicode whitespace variant handling (NBSP, figure space, etc.)
- Add secondary validation before marker insertion
- Add post-cleanup regex as safety net

Results:
- Empty bold markers: 3 -> 0
- Diligent Security PDF: 10/10 (unchanged)
- All regression tests pass
EOF
)"

# After Phase 5
git add -A
git commit -m "$(cat <<'EOF'
Phase 5: Add word fusion post-processor

- Add CamelCase splitting in markdown generation
- Handle patterns: theGeneral, lengthThis, helporganisationscraft
- Preserve legitimate tech terms (JavaScript, TypeScript)

Results:
- Word fusions: 3 -> 0
- Overall quality: 3.4/10 -> 10/10
- All 5 PDFs now passing
EOF
)"
```

---

## Success Criteria

### Quality Metrics (After Both Phases)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Quality Score | 3.4/10 | 10/10 | +6.6 |
| PDFs Passing | 1/5 | 5/5 | +4 |
| Word Fusions | 3 | 0 | -3 |
| Empty Bold Markers | 3 | 0 | -3 |
| Spurious Spaces | 1,623 | <100 | -1,523 |

### Test Pass Requirements

- [ ] All unit tests pass (`cargo test`)
- [ ] Core regression suite passes
- [ ] Comprehensive regression suite passes (run with `--include-ignored`)
- [ ] Diligent Security PDF remains 10/10
- [ ] No new warnings from `cargo clippy --all-features`

---

## Appendix: File Reference

### Files to Modify

| File | Phase | Line Range | Change Type |
|------|-------|------------|-------------|
| `src/converters/markdown.rs` | 5, 6 | 197-484 | Add functions |
| `src/converters/whitespace.rs` | 6 | New | Add cleanup function |

### Files to Test

| File | Tests Added |
|------|-------------|
| `src/converters/markdown.rs` | 8 new tests |
| `tests/quality_metrics.rs` | Existing |
| `tests/regression_suite.rs` | Existing |

### Files Reference Only (No Changes)

| File | Purpose |
|------|---------|
| `src/extractors/word_segmentation.rs` | Word segmentation algorithm |
| `src/layout/bold_validation.rs` | Bold marker validation logic |
| `src/extractors/text.rs` | Text extraction (reference) |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | Claude | Initial comprehensive plan |
