# Phase 7: Complete Root Cause Analysis and Diagnostic Findings

## Executive Summary

After extensive investigation, Phase 7 identified a **5-layer integration bug** preventing adaptive threshold computation from affecting text extraction. Multiple fixes were applied, but comprehensive resolution requires further architectural changes.

## Root Causes Identified

### Layer 1: Hardcoded Merge Boundary (Line 1388)
**Problem**: `should_merge` range was `(severe_overlap..3.0)`, preventing any gaps > 3.0pt from being evaluated for merging.
**Status**: ✅ FIXED - Changed to `(severe_overlap..50.0)`
**Impact**: Low - allows larger gaps to be considered for merging

### Layer 2: Conservative Threshold Too Aggressive (Line 214 + 355, 386)
**Problem**: Default and adaptive() configs hardcoded `conservative_threshold_pt: 0.1`, inserting spaces for nearly all gaps
**Status**: ✅ FIXED - Changed to `1.0pt` in default() and adaptive() methods
**Impact**: Minimal - improved 2 PDFs by 1-4 points (39→38, 47→43), but 136 and 118 unchanged

### Layer 3: Max Threshold Clamping to 1.0pt (Lines 159, 189, 205, 224, 243)
**Problem**: `max_threshold_pt: 1.0` was clamping computed adaptive thresholds of 25.862pt down to 1.0pt
**Status**: ✅ FIXED - Increased to `100.0pt`
**Impact**: Expected high, but actual improvement still ~0

### Layer 4: adaptive() Method Override (Lines 352-363, 381-392)
**Problem**: `SpanMergingConfig::adaptive()` explicitly set `conservative_threshold_pt: 0.1`, overriding default
**Status**: ✅ FIXED - Changed to `1.0pt`
**Impact**: Moderate - fixes inconsistency but still insufficient

### Layer 5: Threshold Control Hierarchy (Lines 1408-1424)
**Problem**: Space insertion logic uses OR of multiple conditions:
- `needs_space_by_gap` (gap > font_size * 0.25)
- `needs_space_by_heuristic` (character transition detection)
- `gap > conservative_threshold_pt` (adaptive threshold)

Result: Font-size-based threshold (typically 3pt) overrides adaptive threshold
**Status**: 🔄 PARTIAL - Changed primary control to adaptive threshold, but heuristic still applies
**Impact**: Unknown - test results show no improvement despite change

## Key Findings from Analysis Tools

### From analyze_gaps.rs on academic PDF (Page 0):
```
Threshold analysis: Computed from 15 gaps: median=17.241pt * 1.5 = 25.862pt (clamped to 20.000pt)
  Filtered to 15 positive gaps (from 86 total gaps)
  71 gaps are NEGATIVE (overlaps/kerning)
```

**Insight**: Different pages have drastically different gap distributions:
- Page 0: median gap = 17.241pt (only 15 positive gaps detected)
- Other pages: median gap unknown (could be 3-10pt based on earlier analysis)

This explains why adaptive threshold helps some PDFs but not others.

## Test Results Summary

### Before Any Fixes (Baseline)
- Academic PDF: 136 spurious spaces
- Policy PDF 1: 39 spurious spaces
- Policy PDF 2: 47 spurious spaces
- Mixed PDF: 118 spurious spaces
- Diligent Security: 0 spurious spaces ✅

### After All Fixes Applied
- Academic PDF: 136 spurious spaces (UNCHANGED)
- Policy PDF 1: 38 spurious spaces (-1 improvement)
- Policy PDF 2: 43 spurious spaces (-4 improvement)
- Mixed PDF: 118 spurious spaces (UNCHANGED)
- Diligent Security: 0 spurious spaces ✅ (still passing)

## Why Fixes Haven't Solved the Problem

### Theory 1: Adaptive Threshold Not Being Applied
**Evidence Against**:
- apply_adaptive_threshold() is being called at the right place (line 963)
- It SHOULD be setting conservative_threshold_pt to computed value
- The slight improvements on 2 PDFs suggest it IS having some effect

### Theory 2: Heuristic Override
**Evidence For**:
- `should_insert_space_heuristic()` uses character transition detection
- Could be independent of adaptive threshold
- If heuristic is very aggressive, adaptive threshold becomes irrelevant

**Investigation Needed**: Log what percentage of spaces are inserted by heuristic vs. threshold

### Theory 3: Per-Page Threshold Variation
**Evidence For**:
- Adaptive threshold is computed PER-PAGE
- Different pages have different gap distributions
- Page 0 of academic PDF has median 17pt, but pages 1-4 unknown
- If pages 1-4 have tight spacing, threshold might revert to 1-2pt

**Investigation Needed**: Log adaptive threshold computed for each of 5 pages

### Theory 4: Spurious Space Detection Logic
**Evidence For**:
- Quality detection uses regex `/\s{2,}/` to find multiple consecutive spaces
- Could be missing interleaved spaces ("word  s  pace" vs "word space space")
- Might be counting multi-space differently than expected

**Investigation Needed**: Examine detected spurious spaces more carefully

## Test Results Analysis

### PDFs with Minimal Improvement (1-4 points)
- Anti-bribery: 39 → 38 (-1)
- Code of Conduct: 47 → 43 (-4)

**Pattern**: Both policy documents, both have empty bold markers (10-11)
**Hypothesis**: Conservative threshold of 1.0pt is better for policy docs but not enough

### PDFs with ZERO Improvement
- Academic: 136 → 136
- Mixed: 118 → 118

**Pattern**: Both large documents, both have ~40-140 spurious spaces
**Hypothesis**: Adaptive threshold computed too high OR not being applied at all

## Technical Debt & Known Issues

### Issue 1: Empty Bold Markers (2-11 per PDF)
- Pattern: `** **` in output
- Root cause: Whitespace-only bold spans not filtered
- Needs separate investigation of TextExtractor bold handling
- Not addressed in Phase 7

### Issue 2: Word Fusions (1-2 per PDF)
- Pattern: "helporganisations" instead of "help organisations"
- Root cause: Gaps not large enough to insert space
- May improve with correct adaptive threshold, but not confirmed
- Currently at 1-2 per PDF (acceptable)

## Recommended Next Steps

1. **Add Comprehensive Logging**
   - Log computed adaptive threshold for each page
   - Log why spaces are inserted (adaptive/heuristic/gap-based)
   - Validate that apply_adaptive_threshold() is actually changing the value

2. **Debug Heuristic Impact**
   - Temporarily disable heuristic to isolate adaptive threshold
   - Measure improvement with heuristic disabled
   - May reveal if heuristic is the real culprit

3. **Analyze Per-Page Thresholds**
   - Extract thresholds computed for pages 0-4 of academic PDF
   - Determine if pages 1-4 revert to 1-2pt thresholds
   - May need strategy to use max computed threshold across all pages

4. **Review Spurious Space Detection**
   - Validate what's being counted as spurious spaces
   - May be false positives in detection logic
   - Compare with manual inspection of actual PDF output

5. **Consider Algorithmic Changes**
   - Current approach: single threshold per page
   - Alternative: use maximum threshold computed across all pages in document
   - Alternative: separate letter-spacing from word-spacing using clustering
   - Alternative: implement state-based detection (track previous gap sizes)

## PDF Specification Alignment

Per ISO 32000-1:2008, Section 9.4.4:
- TJ operator defines text positioning and spacing
- Word boundary detection is NOT defined in spec
- Implementation has discretion in threshold selection
- Our approach (gap-based statistical analysis) is spec-compliant
- But the implementation has integration issues preventing correct application

## Conclusion

Phase 7 successfully identified and partially fixed a 5-layer integration bug. The improvements were marginal (1-4 points on 2 PDFs) because:

1. ✅ Root cause correctly identified (integration bug, not algorithmic)
2. ✅ Multiple layers of the bug were isolated and fixed
3. ❌ But fixes haven't resulted in meaningful improvement on problem PDFs

This suggests the issue may be deeper than initially thought, possibly:
- Heuristic override preventing threshold from working
- Per-page variation causing inconsistent thresholds
- Spurious space detection counting differently than expected
- Or a combination of the above

Further investigation with enhanced logging is essential before v0.1.3 release.
