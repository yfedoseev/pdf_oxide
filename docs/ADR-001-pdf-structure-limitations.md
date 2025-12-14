# ADR-001: PDF Structure Limitations in Text Extraction

**Status:** Accepted
**Date:** 2024-12-03 (Updated: 2025-12-14)
**Context:** TJ operator offset analysis for word fusion detection
**Stakeholders:** Core extraction team, Quality assurance
**Version:** v0.2.0+

---

## Problem Statement

During Phase 5 development, we implemented TJ (Text Show) operator offset collection to detect and prevent word fusion issues in PDF text extraction. While this approach successfully addresses most word fusion cases, we discovered a class of word fusions that **cannot be resolved by offset analysis alone**.

### The Issue: PDF Author Defect

Some PDF creation tools encode compound words (e.g., "draftpolicy", "thefollowingtypesof") as **single text strings without offsets between word components**:

```
PDF Content Stream:  [(draftpolicy)] TJ
Expected behavior:   Two words with boundary info: [(draft) OFFSET (policy)] TJ
Actual behavior:     One string with zero boundary info
```

**Example from real PDF:**
- **Document:** Anti-bribery and Corruption Policy Template (UK).pdf
- **Span:** "N ote: This document is a draftpolicy and is subject..."
- **TJ Operator:** `[(draftpolicy)]` (single string, no offset data)
- **Outcome:** No algorithmic way to determine "draft|policy" boundary

### Root Cause

This is a **PDF authoring defect**, not an extractor bug. The PDF specification (ISO 32000-1:2008) defines:

> The TJ operator displays text with positioning adjustments between array elements. Numeric values encode inter-glyph spacing (negative = more space).

When a PDF creation tool generates `[(singleword)]` instead of `[(word1) -200 (word2)]`, it removes the **only authoritative source of word boundary information**. The extractor cannot synthesize this information from:

- **Geometric gaps:** Not available—single continuous string
- **Font metrics:** No word boundary information in fonts
- **Heuristics:** Dictionary splitting, NLP, OCR-like techniques are unreliable and language-dependent

---

## Decision

**Accept this limitation and document it.** Specifically:

1. **Classify** such issues as "PDF structure defects" rather than extractor regressions
2. **Document** the limitation in user-facing guides
3. **Distinguish** defects from true regressions in quality metrics
4. **Preserve** ISO 32000-1:2008 compliance and avoiding false positives

### Why NOT Other Approaches

#### Option A: Dictionary-Based Word Splitting ❌
- **Risk:** Over-engineering for edge cases (few PDFs have this issue)
- **Problem:** CamelCase names ("iPhone", "myValue") would be incorrectly split
- **Language:** English-specific, breaks for non-Latin scripts
- **False Positives:** "network" ≠ "net" + "work"; "feedback" ≠ "feed" + "back"

#### Option C: Enhanced TJ Heuristics ❌
- **Problem:** Cannot solve single-string encoding (no offset data exists)
- **Approach Fails:** Even with perfect heuristics, if `TJ = [(word)]`, no offset exists to detect

#### Option D: Optional Post-Processing 🔶
- **Status:** Noted for future v1.0+ as **optional enhancement**
- **Timeline:** Out of scope for v0.1.3
- **Design:** Language-aware word segmentation hook (future)

---

## Technical Details

### TJ Operator Offset Collection (Phases 4-5)

We implemented comprehensive TJ offset analysis:

```rust
pub struct TextSpan {
    pub text: String,
    pub bbox: Rect,
    // ... other fields ...
    pub tj_offsets: Vec<f32>,  // TJ operator offsets (1/1000 em units)
}

pub fn extract_gaps_with_tj(spans: &[TextSpan]) -> Vec<f32> {
    for i in 0..spans.len() - 1 {
        let current = &spans[i];
        if !current.tj_offsets.is_empty() {
            // Use TJ offset as primary source
            let last_offset = current.tj_offsets[current.tj_offsets.len() - 1];
            let gap = -(last_offset as f32) * (current.font_size / 1000.0);
            gaps.push(gap);
        } else {
            // Fall back to geometric gap
            gaps.push(next_left - current_right);
        }
    }
}
```

### When TJ Offset Analysis Succeeds

✅ `[(draft) -200 (policy)] TJ` → Detects boundary
✅ `[(word1) -150 (word2)] TJ` → Correctly merged or separate
✅ `[(text) 50 (more)] TJ` → Geometric fallback works

### When TJ Offset Analysis Fails

❌ `[(draftpolicy)] TJ` → **No offset data, cannot detect boundary**
❌ Single monolithic string encoding compound words

---

## Quality Metrics Classification

Updated `FusionConfidence` enum:

```rust
pub enum FusionConfidence {
    High,           // Clear offset data, high confidence
    Medium,         // Ambiguous geometric gaps, moderate confidence
    Low,            // Heuristic-only, low confidence
    PdfStructure,   // PDF authoring defect, not an extractor bug
}
```

**In regression tests:**

```rust
// This is NOT a regression—it's a PDF defect
#[test]
fn test_word_fusion_regression_policy() {
    let pdf = "Anti-bribery and Corruption Policy Template (UK).pdf";
    let metrics = extract_and_analyze(pdf);

    // Report defects as INFO, not FAIL
    assert_eq!(metrics.critical_errors, 0, "No critical errors");

    // PDF defects are expected and documented
    assert!(metrics.pdf_structure_defects >= 1);
}
```

---

## Implementation Plan

**Timeline:** ~95 minutes (Phase 6)

### Step 1: Verify Compilation ✅
- TJ offset collection already integrated
- No compilation errors

### Step 2: Document Limitation (30 min)
- ✅ Create `docs/ADR-001-pdf-structure-limitations.md` (this file)
- ✅ Create `docs/pdf-quality-guide.md` (user-facing guide)
- ✅ Update `tests/fixtures/regression/README.md`
- ✅ Update root `README.md`

### Step 3: Update Quality Metrics (20 min)
- Update `tests/quality_metrics.rs` with `FusionConfidence::PdfStructure`
- Add `pdf_structure_defects` field to `QualityMetrics`
- Update detection functions

### Step 4: Update Regression Suite (15 min)
- Modify `tests/regression_suite.rs` test assertions
- Distinguish defects from regressions
- Update output formatting

### Step 5: Run Tests & Validate (10 min)
- `cargo test --test regression_suite --release`
- Verify all PDFs pass
- Confirm "draftpolicy" reported as PDF defect

### Step 6: Commit (5 min)
- Create git commit with clear message
- Push to feature branch

---

## Success Criteria

✅ All 15 regression PDFs pass tests
✅ "draftpolicy" defect properly classified (INFO level, not FAIL)
✅ Quality scores ≥ 8.0/10.0 across all test PDFs
✅ No new regressions introduced
✅ ISO 32000-1:2008 compliance maintained
✅ False positives eliminated

---

## Future Enhancements (v1.0+)

### Optional: Language-Aware Post-Processing Hook

For future versions, consider optional word segmentation:

```rust
pub trait WordSegmentationHook: Send + Sync {
    fn segment(&self, text: &str, context: &SegmentationContext)
        -> Vec<(String, usize)>;  // (word, start_offset)
}

// Implementation examples
pub struct EnglishWordSegmenter { /* ... */ }
pub struct CustomNLPSegmenter { /* ... */ }
```

**Design Notes:**
- Opt-in (default: disabled)
- Language-specific implementations
- Configurable confidence thresholds
- Validation against test corpus before deployment

---

## References

- **ISO 32000-1:2008:** PDF Specification, Section 9.4.3 (Text Show Array)
- **Phase 4-5 Work:** TJ Offset Collection Implementation
- **Test Data:** Real PDF corpus from academic and policy documents
- **Related Issues:** Fix #1 (word fusion regression prevention)

---

## Decision Record Approval

| Role | Name | Date | Sign-Off |
|------|------|------|----------|
| Maintainer | PDF Oxide Team | 2024-12-03 | ✅ Accepted |
| QA Lead | (TBD) | | |
| Architect | (TBD) | | |

---

## Changelog

- **2024-12-03:** Initial ADR created after TJ offset analysis
- **Phase 5:** TJ offset collection implemented
- **Phase 6:** Classification and documentation formalized
