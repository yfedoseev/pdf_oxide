# Phase 7: Extraction Quality Debugging Strategy

## Executive Summary

The regression tests now execute properly (critical bug fix: `extract_spans_with_config()` implemented). However, extraction quality is poor with 4 of 5 PDFs failing.

**Test Results:**
- Diligent Security Policy: ✅ PASS (10.0/10.0)
- Anti-bribery Policy (UK): ❌ FAIL (0.0/10.0) - 11 empty bold, 39 spurious spaces
- Code of Conduct (EU): ❌ FAIL (0.0/10.0) - 10 empty bold, 46 spurious spaces, 2 word fusions
- Academic PDF: ❌ FAIL (0.0/10.0) - 2 empty bold, **136 spurious spaces**, 1 word fusion
- Mixed PDF: ❌ FAIL (0.0/10.0) - 4 empty bold, 118 spurious spaces

**Average Quality: 2.0/10.0** (Should be ≥ 8.0)

## Root Cause Analysis

### Issue #1: Spurious Spaces (136 in academic PDF!)

**Hypothesis:** The gap threshold is too aggressive, causing spaces to be inserted at tiny gaps that should not create word boundaries.

**Evidence:**
- Academic PDF has 136 spurious spaces (normal documents have 0-5)
- All failing PDFs have 39-136 spurious spaces
- Only passing PDF has 0 spurious spaces

**Theory:**
The `conservative_threshold_pt` was changed from 0.3pt to 0.1pt (comment: "Reverted from 0.3 after regression testing"). This means any gap ≥0.1pt triggers space insertion. With adaptive threshold, the threshold should be raised based on document gap distribution, but it might not be working correctly.

**PDF Spec Context (ISO 32000-1:2008):**
- TJ operator shows explicit horizontal offsets in 1/1000 em units
- No specification for "correct" word spacing threshold
- Implementation choice: must analyze document gaps to set threshold

### Issue #2: Empty Bold Markers (2-11 per PDF)

**Pattern:** `** **` appears in output (empty bold markers)

**Hypothesis:**
1. Text rendering with bold state changes is creating whitespace-only spans
2. Bold marker logic doesn't filter out whitespace-only spans before creating `**text**`
3. Results in `** **` in markdown

**Evidence:**
- Occurs consistently across all failing PDFs
- Only clean PDF has 0 empty markers
- Suggests bold state tracking issue in text extraction

### Issue #3: Word Fusions (1-2 in some PDFs)

**Pattern:** Words incorrectly merged like "helporganisationscraft", "lengthThis"

**Theory:**
- Gaps between words not being detected as spaces
- Adaptive threshold may be too high (gaps < threshold, no space inserted)
- OR gap analysis is failing entirely for some PDFs

## Investigation Plan

### Phase 1: Verify Adaptive Threshold is Being Applied

**Goal:** Confirm `use_adaptive_threshold: true` actually changes extraction output

**Steps:**
1. Create test that extracts same PDF with:
   - Default config (`use_adaptive_threshold: false`)
   - Adaptive config (`use_adaptive_threshold: true`)
2. Compare outputs to verify they're different
3. Log the computed threshold for each PDF

**Expected Results:**
- Different outputs confirm adaptive threshold is being applied
- Threshold values should be reasonable (0.15-0.5pt for normal documents)

### Phase 2: Analyze Gap Distributions

**Goal:** Understand what gaps are being measured and what thresholds are computed

**Steps:**
1. Create debug_gaps tool that:
   - Extracts spans from PDF
   - Measures all gaps between consecutive spans
   - Computes statistics (median, percentiles, IQR, std dev)
   - Shows computed adaptive threshold
   - Displays gap histogram/distribution
2. Run on all 5 test PDFs
3. Compare gap distributions for passing vs. failing PDFs

**Expected Results:**
- Passing PDF should have consistent, moderate gap sizes
- Failing PDFs should show abnormal distributions
- Computed thresholds should explain the differences

### Phase 3: Trace Merging Decisions

**Goal:** Understand which gaps are triggering space insertion

**Steps:**
1. Add detailed logging to `SpanMerger::merge_spans_into_text()`
2. Log every merge decision:
   - Gap size
   - Threshold being used
   - Decision (merge without space / insert space / break)
3. Focus on first 10-20 gap decisions for manual inspection
4. Compare logging output for passing vs. failing PDFs

**Expected Results:**
- Identify exactly where spurious spaces are being inserted
- See if threshold is too low
- See if gap measurement is correct

### Phase 4: Understand Empty Bold Markers

**Goal:** Find where whitespace-only bold markers are created

**Steps:**
1. Add logging to text rendering logic (FontWeight::Bold handling)
2. Log every bold state change:
   - Previous span
   - Current bold span (text content + length)
   - Next span
3. Identify cases where bold span is only whitespace
4. Look for patterns in failing PDFs

**Expected Results:**
- Find where `** **` is being created
- Determine if it's PDF authoring issue or extraction logic issue
- Decide if we should filter whitespace-only spans

## PDF Spec Alignment

### ISO 32000-1:2008 Text Positioning (Section 9.4.4)

**TJ Operator Definition:**
```
TJ (string1 number string2 ...) Tj
```
- Strings are shown at current text position
- Numbers are horizontal adjustments in 1/1000 em units (negative = move left)
- Positive number = space (kerning) between text

**Key Points:**
1. Spec does NOT define a threshold for space insertion
2. Implementation detail: when is a number "large enough" to insert space?
3. Typical word spacing: 0.25-0.33em = 250-330 units
4. Typical letter kerning: 0-50 units

**Our Implementation Constraint:**
- We extract spans (already positioned text)
- We measure gaps between span boundaries
- We must choose a threshold for "gap = space" vs "gap = kerning"
- This is a heuristic, not spec-defined

### Spec-Compliant Solution Approach

1. **Respect actual PDF encoding:** Use TJ offsets, not span boundary estimation
2. **Analyze document-specific gaps:** Each PDF has its own spacing characteristics
3. **Use robust statistics:** Median, percentiles more robust than mean for outliers
4. **Provide configurability:** Different document types (policy, academic, tables) need different thresholds
5. **Log decisions:** Make threshold choices auditable and debuggable

## Expected Root Cause

### Most Likely: Adaptive Threshold Computation Not Being Used

**Hypothesis:**
The threshold computation works (code exists, tests in gap_statistics.rs pass), but it's not being used in actual text extraction.

**Why:**
1. Default `use_adaptive_threshold: false` for backward compatibility
2. Regression tests use `SpanMergingConfig::adaptive()`
3. But the computed threshold might not be applied during span merging

**Fix:** Trace through the code path from `extract_spans_with_config()` → `TextExtractor::extract_text_spans()` → `SpanMerger` to verify adaptive threshold is applied.

### Secondary Hypothesis: Threshold Too Aggressive

**Even if applied, 0.1pt baseline might be causing issues:**
- Adaptive threshold is `median * multiplier`, clamped to [0.05, 1.0]
- If document median gap is 0.05-0.1pt (tight spacing documents), adaptive threshold might be 0.075-0.15pt
- Still too aggressive, creating spurious spaces
- Solution: increase multiplier or change base formula

## Deliverables

1. **Debug output:** Gap distributions and adaptive thresholds for each test PDF
2. **Root cause analysis:** Identify why spurious spaces occur
3. **Fixing strategy:** Specific approach aligned with PDF spec
4. **Implementation:** Code changes to resolve issues
5. **Validation:** Regression test passing with ≥ 8.0 quality scores

## Success Criteria

- [ ] Adaptive threshold is confirmed being applied
- [ ] Gap distributions analyzed and understood
- [ ] Root cause of spurious spaces identified
- [ ] Root cause of empty bold markers identified
- [ ] Fixing strategy documented and PDF-spec-aligned
- [ ] At least 3/5 PDFs passing regression tests
- [ ] Average quality score ≥ 6.0/10.0
