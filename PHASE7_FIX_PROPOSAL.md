# Phase 7: Bimodal Detection Fix Proposal

## Problem Summary

The bimodal gap detection algorithm finds the **wrong boundary**:
- **Expected**: Letter-spacing (2-3pt) vs. Word-spacing (4-55pt) → threshold ~3.5pt
- **Actual**: Text-spacing (all gaps 2-55pt) vs. Line-breaks (326-395pt) → threshold ~186pt

**Result**: Threshold 50x too high → all normal word spaces ignored → 136+ spurious spaces

---

## Proposed Fix: Percentile-Based Threshold

### Algorithm

Replace the "largest jump" search with a percentile-based approach:

```rust
/// Detect word boundary threshold using percentile-based gap analysis.
///
/// # Algorithm
///
/// 1. Filter gaps to positive values in reasonable range (0-50pt)
/// 2. Calculate P75 (75th percentile) as word boundary threshold
/// 3. Apply sanity bounds (2-10pt) to ensure reasonable results
///
/// # Rationale
///
/// In typical text, ~75% of gaps are letter-spacing (tight),
/// ~25% are word-spacing (wider). P75 naturally falls at the boundary.
///
/// # Returns
///
/// `Some(threshold)` if distribution looks like normal text,
/// `None` if insufficient data or unrealistic distribution.
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
    // Extract gaps and filter to text-scale values
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0 && *g < 50.0)  // Positive, exclude layout artifacts
        .collect();

    if gaps.len() < 20 {
        debug!("Bimodal detection: insufficient positive gaps ({} < 20)", gaps.len());
        return None;
    }

    // Sort for percentile calculation
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate 75th percentile
    let p75_idx = (gaps.len() as f32 * 0.75).floor() as usize;
    let p75 = gaps[p75_idx.min(gaps.len() - 1)];

    // Also get median and p90 for logging
    let median_idx = gaps.len() / 2;
    let p90_idx = (gaps.len() as f32 * 0.90).floor() as usize;
    let median = gaps[median_idx];
    let p90 = gaps[p90_idx.min(gaps.len() - 1)];

    debug!(
        "Bimodal detection: analyzed {} positive gaps (median={:.2}pt, p75={:.2}pt, p90={:.2}pt)",
        gaps.len(), median, p75, p90
    );

    // Sanity check: threshold should be in typical word-spacing range
    // Too low (<2pt): likely no clear word boundaries in data
    // Too high (>10pt): distribution doesn't look like normal text
    if p75 >= 2.0 && p75 <= 10.0 {
        debug!("Bimodal detection: using P75 threshold {:.2}pt", p75);
        Some(p75)
    } else {
        debug!(
            "Bimodal detection: P75 ({:.2}pt) outside reasonable range (2-10pt), skipping",
            p75
        );
        None
    }
}
```

### Expected Results

#### Academic PDF (arxiv_2510.21165v1.pdf)

**Current**:
- Gap distribution: 202 gaps in 3-55pt range
- P75 = 3.96pt (from histogram data)
- **Current threshold**: 186.4pt (WRONG - from largest jump)
- **Current spurious spaces**: 136

**After Fix**:
- Gap distribution: same (202 gaps in 3-55pt)
- P75 = 3.96pt ✓
- **New threshold**: 3.96pt ✓ (from percentile)
- **Expected spurious spaces**: <10

#### Policy PDFs

**Current**:
- Gap distribution: ~25 positive gaps in 0-44pt range
- Sparse data, mostly negative gaps
- **Current threshold**: varies (bimodal may return None)
- **Current spurious spaces**: 39-46

**After Fix**:
- Gap distribution: filter to positive gaps
- If <20 gaps: bimodal returns None → fallback to median adaptive
- If ≥20 gaps: P75 ≈ 3-5pt range
- **Expected spurious spaces**: <10

---

## Implementation Changes

### File: `/home/yfedoseev/projects/pdf_oxide/src/extractors/gap_statistics.rs`

**Lines to modify**: 549-587

**Change summary**:
1. Filter gaps to 0-50pt range (exclude layout artifacts)
2. Replace "largest jump" search with P75 calculation
3. Change sanity bounds from (>1.0pt jump, >2.0pt threshold) to (2.0pt ≤ threshold ≤ 10.0pt)
4. Add detailed logging for gap distribution analysis

### Code Diff

```diff
 fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
-    // Extract gaps
+    // Extract gaps and filter to text-scale values (exclude page layout artifacts)
     let mut gaps: Vec<f32> = spans.windows(2)
         .map(|w| w[1].bbox.left() - w[0].bbox.right())
-        .filter(|g| *g > 0.0)  // Only positive gaps
+        .filter(|g| *g > 0.0 && *g < 50.0)  // Positive gaps in text range
         .collect();

-    if gaps.len() < 10 {
-        return None;  // Not enough data for bimodal detection
+    if gaps.len() < 20 {
+        debug!("Bimodal detection: insufficient positive gaps ({} < 20)", gaps.len());
+        return None;
     }

     // Sort gaps
     gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

-    // Find largest consecutive gap
-    let mut max_jump = 0.0;
-    let mut threshold = None;
-
-    for i in 1..gaps.len() {
-        let jump = gaps[i] - gaps[i-1];
-        if jump > max_jump {
-            max_jump = jump;
-            // Use midpoint between the two sides of the gap as threshold
-            threshold = Some((gaps[i] + gaps[i-1]) / 2.0);
-        }
-    }
+    // Calculate 75th percentile as word boundary threshold
+    // Rationale: In typical text, ~75% of gaps are letter-spacing, ~25% are word-spacing
+    let p75_idx = (gaps.len() as f32 * 0.75).floor() as usize;
+    let p75 = gaps[p75_idx.min(gaps.len() - 1)];
+
+    // Also get median and p90 for logging
+    let median_idx = gaps.len() / 2;
+    let p90_idx = (gaps.len() as f32 * 0.90).floor() as usize;
+    let median = gaps[median_idx];
+    let p90 = gaps[p90_idx.min(gaps.len() - 1)];
+
+    debug!(
+        "Bimodal detection: analyzed {} positive gaps (median={:.2}pt, p75={:.2}pt, p90={:.2}pt)",
+        gaps.len(), median, p75, p90
+    );

-    // Only accept if we found a significant gap (> 1pt) and threshold is reasonable
-    match threshold {
-        Some(t) if max_jump > 1.0 && t > 2.0 => {
-            debug!("Bimodal detection: found threshold at {:.4}pt (jump: {:.4}pt)", t, max_jump);
-            Some(t)
-        },
-        _ => {
-            debug!("Bimodal detection: no clear bimodal gap found");
-            None
-        }
+    // Sanity check: threshold should be in typical word-spacing range
+    if p75 >= 2.0 && p75 <= 10.0 {
+        debug!("Bimodal detection: using P75 threshold {:.2}pt", p75);
+        Some(p75)
+    } else {
+        debug!(
+            "Bimodal detection: P75 ({:.2}pt) outside reasonable range (2-10pt), skipping",
+            p75
+        );
+        None
     }
 }
```

---

## Testing Plan

### 1. Unit Tests

Add tests to verify percentile calculation:

```rust
#[test]
fn test_bimodal_detection_academic_gaps() {
    // Simulate academic PDF gap distribution
    let gaps = vec![
        3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,  // Letter spacing (50%)
        4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,  // Word spacing (50%)
    ];

    let spans = create_spans_with_gaps(&gaps);
    let threshold = detect_word_boundary_threshold(&spans);

    // P75 of 20 values = index 15 = 4.5pt
    assert!(threshold.is_some());
    assert!((threshold.unwrap() - 4.5).abs() < 0.1);
}

#[test]
fn test_bimodal_detection_filters_outliers() {
    // Include some column breaks that should be filtered out
    let gaps = vec![
        3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
        4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
        326.9, 394.8,  // Should be filtered out (>50pt)
    ];

    let spans = create_spans_with_gaps(&gaps);
    let threshold = detect_word_boundary_threshold(&spans);

    // Should still get ~4.5pt, not affected by outliers
    assert!(threshold.is_some());
    assert!((threshold.unwrap() - 4.5).abs() < 0.1);
}

#[test]
fn test_bimodal_detection_rejects_unrealistic_p75() {
    // All gaps very small (no clear word spacing)
    let gaps = vec![0.5; 30];  // 30 identical tiny gaps

    let spans = create_spans_with_gaps(&gaps);
    let threshold = detect_word_boundary_threshold(&spans);

    // P75 = 0.5pt, below 2.0pt minimum → should reject
    assert!(threshold.is_none());
}
```

### 2. Integration Tests

Run regression suite:

```bash
cargo test --test regression_suite
```

**Expected before fix**:
- Failed PDFs: 4/5
- Average quality: 2.0/10.0
- Academic spurious spaces: 136

**Expected after fix**:
- Failed PDFs: 0/5 (or 1/5 if edge cases)
- Average quality: 8.5-9.5/10.0
- Academic spurious spaces: <10

### 3. Real-World Validation

Test on the actual PDFs:

```bash
# Academic PDF
cargo run --bin analyze_gaps -- tests/fixtures/regression/academic/arxiv_2510.21165v1.pdf

# Policy PDF
cargo run --bin analyze_gaps -- "tests/fixtures/regression/policy/Anti-bribery and Corruption Policy Template (UK).pdf"
```

**Look for in output**:
- `Bimodal detection: analyzed X positive gaps (median=Y, p75=Z, p90=W)`
- `Bimodal detection: using P75 threshold Zpt` (where Z is 2-10pt)
- Final markdown: minimal spurious spaces in word boundaries

---

## Fallback Strategy

If percentile approach doesn't work as expected:

### Alternative: Median + IQR

Use median + 1.5*IQR as threshold:

```rust
let median = gaps[median_idx];
let p25_idx = (gaps.len() as f32 * 0.25).floor() as usize;
let p75_idx = (gaps.len() as f32 * 0.75).floor() as usize;
let iqr = gaps[p75_idx] - gaps[p25_idx];

let threshold = median + 1.5 * iqr;

if threshold >= 2.0 && threshold <= 10.0 {
    Some(threshold)
} else {
    None
}
```

This is more conservative but may work better for distributions with high variability.

---

## Risk Assessment

### Low Risk ✅

- Change is isolated to one function (`detect_word_boundary_threshold`)
- Fallback to existing adaptive threshold if bimodal detection fails
- Sanity bounds prevent catastrophic threshold values
- All existing tests should still pass (they don't rely on specific bimodal behavior)

### Potential Edge Cases

1. **Very tight spacing** (all gaps <2pt):
   - P75 < 2.0pt → bimodal returns None → falls back to median adaptive ✓

2. **Very loose spacing** (all gaps >10pt):
   - P75 > 10.0pt → bimodal returns None → falls back to median adaptive ✓

3. **Insufficient positive gaps** (<20):
   - bimodal returns None → falls back to median adaptive ✓

4. **Mixed fonts/sizes**:
   - P75 will be weighted average across all gaps
   - May be slightly off but still in reasonable range (2-10pt)
   - Better than current approach (186pt!)

---

## Success Criteria

### Phase 7 Completion Requirements

1. ✅ **Bimodal detection uses percentile approach** (P75)
2. ✅ **Sanity bounds prevent unrealistic thresholds** (2-10pt)
3. ✅ **Academic PDF spurious spaces**: 136 → <10 (>90% reduction)
4. ✅ **Policy PDF spurious spaces**: 39-46 → <10 (>75% reduction)
5. ✅ **Regression suite**: 4/5 failed → 0/5 failed
6. ✅ **Average quality score**: 2.0 → 8.5+ (>325% improvement)

### Definition of Done

- [ ] Implementation complete in `gap_statistics.rs`
- [ ] Unit tests added and passing
- [ ] Integration tests passing (regression suite)
- [ ] Real-world validation on 5 test PDFs
- [ ] Documentation updated (algorithm description, examples)
- [ ] Phase 7 completion report written

---

## Timeline Estimate

- **Implementation**: 30 minutes (straightforward algorithm replacement)
- **Testing**: 1 hour (unit tests + regression suite + real PDFs)
- **Documentation**: 30 minutes (update comments, write completion report)
- **Total**: ~2 hours

---

## Conclusion

The fix is **simple, low-risk, and high-impact**:
- Replace "largest jump" with "P75 percentile"
- Add sanity bounds (2-10pt)
- Filter out layout artifacts (>50pt)

**Expected outcome**: 90%+ reduction in spurious spaces, quality scores restored to 8.5-10.0 range.

This aligns with the original Phase 5/6/7 goals of adaptive threshold detection while fixing the fundamental algorithm flaw.
