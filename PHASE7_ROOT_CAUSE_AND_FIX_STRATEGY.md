# Phase 7: Root Cause Analysis and Fixing Strategy

## Executive Summary

**Root Cause Identified**: The default threshold of 0.1pt is **massively too aggressive**, causing 136+ spurious spaces in academic PDFs and 38+ in policy PDFs. The adaptive threshold computation has fundamental flaws in handling different gap distributions.

**Key Finding**: We're measuring span boundary gaps (which are complex and document-dependent), not TJ operator offsets. The current single-threshold approach cannot handle the bimodal/multimodal distributions present in real PDFs.

**Solution**: Replace the simple threshold-based merging with a statistical approach that detects the gap between letter-spacing clusters and word-spacing clusters.

---

## Part 1: Root Cause Analysis (Detailed)

### Issue #1: Spurious Spaces (136 in Academic, 38 in Policy PDFs)

#### Academic PDF Gap Distribution (arxiv_2510.21165v1.pdf)
```
Total gaps: 265
Median gap: 3.3674pt
Gap distribution (histogram):
  -624pt to -12pt: 63 gaps (overlaps/kerning)
  3pt to 55pt: 202 gaps (NORMAL LETTER/WORD SPACING)
  55pt to 326pt: 0 gaps (GAP REGION!)
  326pt+: 2 gaps (line breaks)
```

**Analysis**:
- With default threshold (0.1pt): 194 gaps >= 0.1pt → 194 spaces inserted
- With adaptive threshold (1.0pt): Still 194 gaps >= 1.0pt → 194 spaces inserted
- Root cause: The 3-55pt region contains LEGITIMATE text spacing, not all gaps
- The real threshold should be **~55pt** (or higher) to avoid treating word spacing as breaks

#### Policy PDF Gap Distribution (Anti-bribery.pdf)
```
Total gaps: 62
Median gap: -89.9150pt (NEGATIVE!)
Gap distribution (histogram):
  -1148pt to -47pt: 54 gaps (text overlaps/kerning)
  -47pt to 43.7pt: 25 gaps (problematic region)
  43.7pt to 135.4pt: 1 gap
  135pt+: 1 gap
```

**Analysis**:
- Median is NEGATIVE due to text overlaps/kerning
- Adaptive threshold computation: -89.915pt * 1.5 = -134.8pt, clamped to 0.05pt
- With default threshold (0.1pt): 14 gaps >= 0.1pt → spaces
- Root cause: **Threshold computation formula is broken for negative medians**
- The `-47pt to 43.7pt` region has mixed semantics (some are overlaps, some are spaces)

### Issue #2: Empty Bold Markers (2-11 per PDF)

**Pattern**: `** **` appears in output where bold text boundaries are.

**Root Cause Hypothesis** (unconfirmed, pending investigation):
1. Text rendering with bold state changes creates whitespace-only spans
2. Bold marker wrapping (`**text**`) doesn't filter whitespace-only content
3. Results in `** **` (empty bold marker) in final output

**Evidence**:
- Occurs consistently across failing PDFs (6-11 instances each)
- Only clean PDF has 0 instances
- Suggests bold state tracking or span filtering issue

**Investigation Required**:
- Check TextExtractor/SpanMerger bold state handling
- Log which spans are whitespace-only
- Check markdown generation for empty text filtering

### Issue #3: Word Fusions (1-3 per PDF)

**Pattern**: Words incorrectly merged like "helporganisationscraft", "draftpolicy"

**Root Cause**:
1. Valid PDF structure defect (single-string TJ encoding) - documented in Phase 6
2. OR gaps between words not detected as spaces (threshold too high)

**Analysis**:
- Academic PDF: 0 fusions (threshold too low creating spaces everywhere)
- Policy PDFs: 1-2 fusions (mix of legitimate PDFs defects and threshold issues)
- Should be rare if threshold is correct

---

## Part 2: PDF Specification Alignment (ISO 32000-1:2008)

### Section 9.4.4: Text Positioning with TJ Operator

**PDF Spec Definition**:
```
TJ (string1 number string2 ...) Tj
```
- Strings are text to show
- Numbers are horizontal adjustments in **1/1000 em units**
- Spec does NOT define "word boundary threshold"

**Key Point**: The spec leaves word boundary detection as an **implementation detail**.

### Our Implementation Constraint

We extract spans (already positioned text), not raw TJ operators. This means:
1. We measure SPAN BOUNDARY GAPS (calculated from bbox.right - bbox.left)
2. These gaps are NOT directly TJ offset values
3. Gap distribution varies significantly by PDF author/creation tool
4. We must use **statistical analysis** to detect word boundaries

### Why Simple Thresholds Fail

**Bimodal Distributions**:
- Academic PDF: Clear bimodal distribution (letter spacing ~0-10pt vs word spacing ~10-55pt)
- Policy PDF: Multimodal with negative gaps (overlaps) + letter spacing + word spacing
- Single threshold cannot correctly separate all modes simultaneously

**Negative Gaps Problem**:
- Negative gaps represent text overlaps (kerning, subscripts, special positioning)
- These CANNOT be treated as word boundaries
- Adaptive threshold formula (median * multiplier) breaks when median is negative
- Result: Threshold gets clamped to minimum (0.05-0.2pt), defeating adaptation

---

## Part 3: Fixing Strategy

### Strategy 1: Improve Threshold Computation (Quick Fix)

**Goal**: Make the adaptive threshold work better without complete rewrite.

**Changes**:
1. **Fix negative gap handling**:
   ```rust
   // Use absolute median for formula to avoid negative values
   let abs_median = gaps.iter().map(|g| g.abs()).median();
   let threshold = abs_median * multiplier;
   ```

2. **Increase minimum threshold floor**:
   - Current: 0.05pt - 0.2pt (per config)
   - Proposed: 2.0pt - 3.0pt minimum
   - Rationale: 0.1pt is letter-spacing level, real word spacing is 3pt+

3. **Use percentile instead of median**:
   ```rust
   // Use 75th percentile instead of median (more conservative)
   let threshold = stats.p75 * multiplier;
   ```

4. **Filter negative gaps before computation**:
   ```rust
   // Only consider positive gaps for threshold
   let positive_gaps: Vec<f32> = gaps.iter().filter(|g| **g > 0.0).copied().collect();
   let threshold = if positive_gaps.is_empty() {
       default_threshold
   } else {
       compute_from_positive_gaps(&positive_gaps)
   };
   ```

**Pros**:
- Minimal code changes
- Maintains backward compatibility

**Cons**:
- Still uses single threshold (bimodal distributions will struggle)
- Percentile approach is heuristic-based

### Strategy 2: Bimodal Gap Detection (Robust Fix)

**Goal**: Detect the natural gap between letter-spacing and word-spacing clusters.

**Algorithm**:
1. Separate gaps into positive and negative
2. For positive gaps, sort and compute consecutive differences
3. Find the largest gap in the sorted sequence
4. Use that point as word boundary threshold

**Example (Academic PDF)**:
```
Sorted positive gaps: 3.0, 3.1, 3.2, ..., 55.1, 55.2 (202 gaps),
                      [big gap here], 326.9, 394.8
Consecutive differences: ~0.1pt increments, then JUMP of ~270pt
Threshold chosen: ~55-100pt (at the jump)
```

**Implementation**:
```rust
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> f32 {
    // Collect positive gaps only
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0)
        .collect();

    if gaps.len() < 10 {
        return 3.0; // fallback
    }

    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find largest consecutive gap
    let mut max_jump = 0.0;
    let mut threshold = 3.0;
    for i in 1..gaps.len() {
        let jump = gaps[i] - gaps[i-1];
        if jump > max_jump {
            max_jump = jump;
            threshold = (gaps[i] + gaps[i-1]) / 2.0;
        }
    }

    threshold
}
```

**Pros**:
- Adapts to document's actual gap structure
- Handles bimodal distributions naturally
- No magic constants (automatically finds the transition)

**Cons**:
- More complex implementation
- Edge cases (what if no clear bimodal distribution?)
- Requires testing on diverse PDFs

### Strategy 3: Multi-Mode Detection (Most Robust)

**Goal**: Detect multiple clustering modes and use Gaussian Mixture Model principles.

**Algorithm**:
1. Use gap statistics (mean, std_dev, IQR) to estimate number of modes
2. Group gaps into clusters (DBSCAN or k-means)
3. Use the threshold between largest two clusters
4. Handle negative gaps separately (kerning mode)

**This is complex** - reserve for Phase 8+ if needed.

---

## Part 4: Recommended Fix Path

### Immediate Fix (This Phase - Phase 7)

**Implement Strategy 2 (Bimodal Detection)** because:
1. Matches the statistical reality of gap distributions
2. Automatically adapts to different PDFs
3. No magic thresholds needed
4. Can be debugged and refined with real PDFs

**Steps**:
1. Implement `detect_word_boundary_threshold()` function in `gap_statistics.rs`
2. Use it in adaptive config computation
3. Add min threshold guard (2.0pt) to prevent degenerate cases
4. Test on all 5 regression PDFs
5. Verify spurious space counts drop significantly

### Implementation Plan

**File**: `src/extractors/gap_statistics.rs` (modified)

**Changes**:
```rust
/// Detect word boundary threshold by finding largest gap in sorted gap sequence
pub fn detect_word_boundary_from_gaps(spans: &[crate::layout::TextSpan]) -> f32 {
    // Implementation as above
}

/// Enhanced adaptive threshold computation
pub fn analyze_document_gaps(
    spans: &[crate::layout::TextSpan],
    config: Option<AdaptiveThresholdConfig>,
) -> ThresholdResult {
    // Collect gaps
    let gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .collect();

    if gaps.is_empty() {
        return ThresholdResult::default();
    }

    // Try bimodal detection first
    if let Some(threshold) = detect_word_boundary_from_gaps(spans) {
        if threshold > 2.0 {  // sanity check
            return ThresholdResult {
                threshold_pt: threshold,
                reason: format!("Bimodal detection: identified word boundary at {:.4}pt", threshold),
            };
        }
    }

    // Fallback to improved adaptive algorithm
    // ... (existing code with fixes for negative gaps, percentile-based)
}
```

**File**: `src/extractors/text.rs` (modified)

**Changes**:
- Update `SpanMergingConfig::adaptive()` to use new threshold computation
- Ensure `use_adaptive_threshold: true` is the default for better behavior
- Add logging for debugging threshold decisions

**File**: `tests/regression_suite.rs` (modified)

**Changes**:
- Update quality score thresholds based on new extraction quality
- Expect improvement: spurious_spaces < 5, empty_bold = 0
- Quality scores >= 8.0 for all PDFs

### Testing Plan

1. **Unit tests**: Test `detect_word_boundary_from_gaps()` on synthetic gap sets
2. **Integration tests**: Run on all 5 regression PDFs, verify:
   - Academic PDF: spurious_spaces < 5 (from 136)
   - Policy PDFs: spurious_spaces < 5 (from 38-43)
   - Quality scores: >= 8.0 for all
3. **Regression tests**: Ensure no new failures introduced

---

## Part 5: Expected Results

### Before Fix
```
Academic PDF (arxiv_2510.21165v1.pdf):
  Quality: 0.0/10.0
  Spurious spaces: 136
  Empty bold markers: 2

Policy PDF (Anti-bribery.pdf):
  Quality: 0.0/10.0
  Spurious spaces: 38
  Empty bold markers: 6
```

### After Fix (Target)
```
Academic PDF:
  Quality: >= 8.0/10.0
  Spurious spaces: 0-2
  Empty bold markers: 0

Policy PDF:
  Quality: >= 8.0/10.0
  Spurious spaces: 0-2
  Empty bold markers: 0
```

### Success Criteria

- ✅ All 5 regression tests pass (quality >= 8.0)
- ✅ Spurious spaces < 5 per PDF
- ✅ Empty bold markers = 0
- ✅ No word fusion regressions
- ✅ Average quality score >= 8.5/10.0

---

## Part 6: Alignment with PDF Specification

### ISO 32000-1:2008 Compliance

1. **Respects PDF encoding**: Uses extracted spans (which already respect TJ encoding)
2. **No over-specification**: Doesn't impose constraints not in spec
3. **Implementation-defined behavior**: Word boundary detection is left to implementation (per spec)
4. **Statistical approach**: Adapts to document's actual gap characteristics

### Key Principle

> "The PDF specification defines what positions text can be at, but not where word boundaries should be detected. Our implementation uses statistical analysis of actual gap distributions to make this determination - a valid and reasonable interpretation of the spec."

---

## Part 7: Additional Investigation Items

### Empty Bold Markers (2-11 per PDF)

**Next Steps**:
1. Add logging to bold state handling in TextExtractor
2. Check if whitespace-only spans are being generated during text extraction
3. Filter whitespace-only spans before bold wrapping
4. Consider: Should bold markers wrap empty text at all?

### Word Fusions

**Next Steps**:
1. After fixing spurious spaces, re-test for fusions
2. Document which are PDF structure defects vs extraction issues
3. Consider dictionary-based approach for common fusions (Phase 8+)

---

## Commit Strategy

1. **Commit 1**: Implement bimodal gap detection in `gap_statistics.rs`
2. **Commit 2**: Update threshold computation and adaptive config
3. **Commit 3**: Add logging and debugging helpers
4. **Commit 4**: Update regression tests with new expectations
5. **Commit 5**: Final validation and documentation

---

## References

- ISO 32000-1:2008 Section 9.4.4: Text Positioning with TJ
- PHASE7_DEBUG_STRATEGY.md: Initial investigation plan
- test analysis results: Bimodal gap distributions confirmed across all failing PDFs
- src/bin/analyze_gaps.rs: Gap distribution debugging tool

---

## Status

**Phase 7 Progress**:
- ✅ Root cause identified and analyzed
- ✅ PDF spec alignment verified
- ✅ Fixing strategy documented
- ⏳ Implementation pending

**Expected Timeline**:
- Bimodal detection implementation: 2-3 hours
- Testing and validation: 1-2 hours
- Total: Same-day completion if prioritized
