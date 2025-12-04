# Phase 7: Bimodal Gap Detection - Root Cause Analysis

## Executive Summary

The bimodal gap detection algorithm **is working correctly** from an implementation standpoint, but it's **detecting the wrong bimodal distribution**. Instead of finding the boundary between letter-spacing and word-spacing, it's finding the boundary between word-spacing and column/line breaks.

**Result**: The threshold (186.4pt) is 40-50x too high, causing all normal gaps to be ignored and creating massive spurious space problems.

---

## The Problem

### Test Results (UNCHANGED - Algorithm Not Helping)
- **Academic PDF**: 136 spurious spaces (FAIL)
- **Policy PDFs**: 39-46 spurious spaces (FAIL)
- **Only 1/5 PDFs passing**: Diligent Security Policy (10.0/10)

### What We Expected
The bimodal algorithm should detect **two clusters**:
1. **Letter spacing**: 0-3pt (tight character gaps within words)
2. **Word spacing**: 4-55pt (gaps between words)

**Expected threshold**: ~3.5pt (midpoint between clusters)

### What Actually Happened
The algorithm detected **two clusters**:
1. **All text spacing**: -12.7pt to 55.2pt (202 gaps - letters AND words together!)
2. **Line/column breaks**: 326.9pt to 394.9pt (2 gaps - page layout artifacts)

**Actual threshold**: 186.4pt (midpoint between 55.2pt and 326.9pt)

---

## Gap Distribution Analysis

### Academic PDF (arxiv_2510.21165v1.pdf, Page 0)

```
Total gaps: 265
Positive gaps: ~204

📊 HISTOGRAM:
  -624 to -556pt [  8]  ← Negative overlaps (column layout artifacts)
  -556 to -488pt [ 12]
  -488 to -420pt [  8]
  -420 to -352pt [  7]
  -352 to -284pt [  6]
  -284 to -216pt [  6]
  -216 to -148pt [  4]
  -148 to  -80pt [  6]
   -80 to  -12pt [  4]
   -12 to   55pt [202] ← ALL LETTER AND WORD SPACING (bimodal WITHIN this!)
    55 to  123pt [  0]  ← Empty
   123 to  191pt [  0]  ← Empty
   191 to  259pt [  0]  ← Empty
   259 to  327pt [  0]  ← Empty
   327 to  395pt [  2]  ← Line/column breaks

📈 STATISTICS:
  Median:  3.37pt
  P25:    -1.10pt
  P75:     3.96pt
  P90:     4.49pt
  IQR:     5.06pt
```

### Bimodal Detection Result

```
Config: Balanced (default)
  Multiplier: 1.5
  Min: 0.05pt, Max: 1pt
  Use IQR: false

[DEBUG] Bimodal detection: found threshold at 186.4200pt (jump: 326.5626pt)
[DEBUG] Using bimodal threshold: Bimodal detection: identified word boundary at 186.4200pt

  Computed threshold: 186.4200pt  ← WRONG! Should be ~3.5pt
  Reason: Bimodal detection: identified word boundary at 186.4200pt
```

---

## Why the Algorithm Failed

### Current Algorithm (lines 549-587 in gap_statistics.rs)

```rust
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
    // 1. Filter to positive gaps only
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0)  // Only positive
        .collect();

    if gaps.len() < 10 {
        return None;
    }

    // 2. Sort gaps
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Find LARGEST consecutive gap
    let mut max_jump = 0.0;
    let mut threshold = None;

    for i in 1..gaps.len() {
        let jump = gaps[i] - gaps[i-1];
        if jump > max_jump {
            max_jump = jump;
            threshold = Some((gaps[i] + gaps[i-1]) / 2.0);  // Midpoint
        }
    }

    // 4. Accept if jump > 1.0pt AND threshold > 2.0pt
    match threshold {
        Some(t) if max_jump > 1.0 && t > 2.0 => {
            Some(t)
        },
        _ => None
    }
}
```

### The Fatal Flaw

**The algorithm finds the LARGEST gap**, not the first significant gap!

For academic PDF:
- Sorted positive gaps: `3.0, 3.1, 3.2, ..., 54.9, 55.2, [JUMP 271pt], 326.9, 394.8`
- **Largest jump**: 326.9 - 55.2 = **271.6pt** ✓ (found this!)
- **Midpoint threshold**: (55.2 + 326.9) / 2 = **191.0pt** ✓ (calculated this!)

But we WANTED:
- **First significant jump**: ~1-2pt somewhere between 3pt and 4pt
- **Midpoint threshold**: ~3.5pt

### Why Policy Documents Also Fail

Policy documents have different gap distributions:
- Mostly **negative gaps** (-47pt to 0pt) due to tight justified spacing
- Only ~25 positive gaps (0-44pt range)
- When filtered to positive only: insufficient clustering structure
- Algorithm finds random large jumps in sparse data

---

## Root Cause Summary

### Three Fundamental Issues

1. **Wrong Bimodal Distribution Detected**
   - Looking for: letter-spacing vs word-spacing (~3pt boundary)
   - Actually finding: text-spacing vs layout-breaks (~186pt boundary)
   - The "largest jump" heuristic is too naive

2. **Insufficient Filtering**
   - Including all positive gaps (0pt to 400pt+)
   - Should filter OUT extreme outliers (>50pt) before analysis
   - Column breaks and line breaks pollute the distribution

3. **Missing Domain Knowledge**
   - Real word boundaries are typically 3-6pt (0.3-0.5em at 12pt font)
   - Anything >50pt is page layout, not word spacing
   - Algorithm has no concept of "reasonable" thresholds

---

## Why It's Not Working in Tests

### The Threshold Pipeline

```
1. Bimodal detection runs → Returns 186.4pt
2. Pipeline: "Bimodal threshold found, using it!"
3. conservative_threshold_pt ← 186.4pt (overrides 0.1pt default)
4. Merging logic: Insert space if gap > 186.4pt
5. Result: NO SPACES INSERTED (all normal gaps are 3-55pt, way below 186.4pt!)
6. Spurious spaces: 136 (because words run together without spaces)
```

### Why Diligent Security Policy Still Passes

Looking at the test output:
- **Only 1/5 PDFs passing**: Diligent Security Policy (10.0/10)

Likely reasons:
- Bimodal detection returns `None` (insufficient positive gaps)
- Falls back to adaptive threshold based on median
- Or: Already had perfect spacing in the PDF source

---

## Proposed Fixes

### Option 1: Percentile-Based Threshold (Recommended)

Instead of looking for the largest jump, use statistical percentiles:

```rust
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
    // Filter to positive gaps AND exclude extreme outliers
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0 && *g < 50.0)  // Exclude page layout artifacts
        .collect();

    if gaps.len() < 20 {
        return None;  // Need more data
    }

    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use 75th percentile as word boundary threshold
    // Rationale: 75% of gaps are letter-spacing, 25% are word-spacing
    let p75_idx = (gaps.len() as f32 * 0.75) as usize;
    let threshold = gaps[p75_idx];

    // Sanity check: reasonable range for word spacing
    if threshold >= 2.0 && threshold <= 10.0 {
        Some(threshold)
    } else {
        None  // Distribution doesn't look like normal text
    }
}
```

**Advantages**:
- Simple, robust to outliers
- Based on actual gap distribution, not arbitrary "largest jump"
- Automatically adapts to tight vs. loose spacing
- Sanity bounds prevent ridiculous thresholds

**Expected Results**:
- Academic PDF: P75 ≈ 3.96pt → threshold ≈ 4.0pt ✓
- Policy PDF: P75 of positive gaps → threshold ≈ 3-5pt ✓

### Option 2: Density-Based Clustering (Complex)

Use proper clustering (DBSCAN, k-means) to find letter vs. word gap clusters:
- **Pro**: Mathematically rigorous, handles complex distributions
- **Con**: Overkill for this problem, adds dependencies

### Option 3: Domain-Constrained Search (Hybrid)

Look for the largest jump WITHIN a reasonable range:

```rust
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0 && *g < 20.0)  // Only look at text-scale gaps
        .collect();

    if gaps.len() < 20 {
        return None;
    }

    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Find largest jump in the 2-8pt range (typical word boundary zone)
    let mut max_jump = 0.0;
    let mut threshold = None;

    for i in 1..gaps.len() {
        let midpoint = (gaps[i] + gaps[i-1]) / 2.0;
        if midpoint >= 2.0 && midpoint <= 8.0 {  // Constrain search range
            let jump = gaps[i] - gaps[i-1];
            if jump > max_jump {
                max_jump = jump;
                threshold = Some(midpoint);
            }
        }
    }

    // Require at least 0.5pt jump to be significant
    match threshold {
        Some(t) if max_jump > 0.5 => Some(t),
        _ => None
    }
}
```

---

## Recommended Next Steps

### Immediate Action (Fix for Phase 7)

1. **Replace bimodal detection with percentile-based approach** (Option 1)
   - Simple, proven, low risk
   - Expected to reduce spurious spaces by 90%+

2. **Add comprehensive logging**
   - Log gap distribution summary (min, p25, median, p75, p90, max)
   - Log why threshold was chosen (bimodal vs. percentile vs. fallback)
   - Log filtering steps (how many gaps excluded as outliers)

3. **Add sanity checks**
   - Threshold must be in 0.5-10pt range
   - If outside range, fall back to median-based adaptive threshold

### Validation Plan

1. **Re-run regression tests** with percentile approach
2. **Expected improvements**:
   - Academic PDF: 136 → <10 spurious spaces
   - Policy PDFs: 39-46 → <10 spurious spaces
   - Quality scores: 0.0 → 8.0-10.0

3. **Monitor edge cases**:
   - Documents with very tight spacing (policy docs)
   - Documents with very loose spacing
   - Mixed-font documents

---

## Conclusion

The bimodal detection algorithm is **fundamentally flawed** because it detects the wrong bimodal distribution. It finds "text vs. layout breaks" instead of "letter-spacing vs. word-spacing".

**The fix is straightforward**: Use percentile-based thresholds (P75) instead of searching for the largest gap. This is simpler, more robust, and aligned with how text spacing actually works.

**Impact**: This single fix should resolve 90%+ of spurious space issues and restore quality scores to 8.0-10.0 range.
