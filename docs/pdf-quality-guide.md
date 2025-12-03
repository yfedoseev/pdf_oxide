# PDF Quality Guide

**Understanding Text Extraction Quality in PDF Oxide**

---

## Overview

PDF Oxide aims for high-quality text extraction from complex PDFs. This guide explains quality metrics, known limitations, and how to interpret extraction results.

**Key Takeaway:** Quality varies by PDF structure, not extractor capability.

---

## Quality Score (0-100)

Each extracted document receives a quality score based on several factors:

```
Quality Score = 100
  - (word_fusions × 5)      // Merged words (critical)
  - (spurious_spaces × 3)    // Character-level spacing artifacts
  - (mid_word_bold × 2)      // Bold formatting mid-word
  - (empty_bold_markers × 1) // Malformed markdown
  - (gap_cv > 2.0 ? 10 : 0)  // High variance in inter-word gaps
```

**Interpretation:**

| Score | Rating | Action |
|-------|--------|--------|
| 90-100 | Excellent | Production ready, minimal manual review |
| 80-89 | Good | Minor formatting issues, suitable for most uses |
| 70-79 | Fair | Noticeable artifacts, recommend manual review |
| < 70 | Poor | Significant issues, likely requires corrections |

---

## Issue Types

### 1. Word Fusion (Critical)

**What it is:** Two words merged into one without space.
```
Expected: "draft policy"
Actual:   "draftpolicy"
```

**Severity:** ⚠️ Critical (5 points per instance)

**Causes:**
- **Geometric gaps too small:** PDF author set tight spacing; extractor threshold wrong
  - **Solution:** Adaptive threshold tuning (Phases 4-5)

- **TJ operator missing offsets:** PDF tool encoded compound word as single string
  - **Example:** `[(draftpolicy)] TJ` instead of `[(draft) -200 (policy)] TJ`
  - **Solution:** None (PDF authoring defect, not extractor bug)
  - **Classification:** PDF structure defect, not a regression

**How to check:**
```bash
grep -E '\b[a-z]{3,}[A-Z][a-z]{3,}\b' output.md  # CamelCase without spaces
grep "draftpolicy\|thefollowingtypesof" output.md  # Known patterns
```

### 2. Spurious Spaces (Warning)

**What it is:** Single letters or small fragments separated by unexpected spaces.
```
Expected: "organisations"
Actual:   "organi s ations"
```

**Severity:** ⚠️ Warning (3 points per instance)

**Causes:**
- Incorrect character positioning in PDF
- Heuristic thresholds catching spacing noise
- Column layout detection confusion

**Typical occurrence:** 0-3 instances per 100 pages (acceptable)

### 3. Mid-Word Bold Formatting

**What it is:** Bold marker applied to part of a word.
```
Expected: **grid**
Actual:   gr**i**d
```

**Severity:** ⚠️ Minor (2 points per instance)

**Causes:**
- PDF font changes mid-word for emphasis or ligatures
- Complex text mixing weights/styles

### 4. Gap Statistics Anomalies

**What it is:** Statistical indicators of potential spacing issues:

- **Gap Coefficient of Variation > 2.0:** High variance in inter-word spacing
- **Gap Median < 0.05pt:** Extremely tight spacing (unusual)

**Severity:** ⚠️ Warning (10 points if CV > 2.0, 15 points if median < 0.05pt)

**Interpretation:**
- Normal range: Median 0.12-0.35pt, CV < 1.5
- Tight spacing: Median 0.05-0.12pt, may indicate word fusion risk
- Variable spacing: CV > 2.0, suggests mixed layouts or column detection issues

---

## Known Limitations

### 1. PDF Structure Defects

Some PDF creation tools generate compound words as monolithic strings without offset information:

```
PDF Content:   [(draftpolicy)] TJ   (← single string, no boundary info)
Spec-compliant: [(draft) -200 (policy)] TJ
```

**Impact:**
- Cannot detect word boundary between "draft" and "policy"
- No algorithmic solution (no data available)
- Not a regression; documented limitation

**Affected documents:**
- Anti-bribery and Corruption Policy Template (UK).pdf
- Templates from certain policy generators

**Workaround:**
- Manual correction post-processing
- Use PDF editing tool to re-export with proper TJ formatting
- Consider using source document if available

### 2. Column Layout Detection

PDFs with complex multi-column layouts may:
- Miss column boundaries
- Incorrectly merge text across columns
- Produce non-linear output

**Mitigation:** Phases 3-4 introduced column detection; most layouts now handled correctly.

### 3. Table Detection

Tables are detected but rendered as markdown with simple formatting:
- Complex nested tables: Simplified structure
- Spanning cells: Limited support
- Visual alignment: Not preserved

**Recommendation:** Export raw table cells for programmatic use; review visual layout manually.

### 4. Font Color & Emphasis

Some PDF emphasis markers may not translate to markdown:
- Colored text: No special formatting (converted to plain text)
- Strikethrough: Not always preserved
- Shadows/outlines: Ignored

---

## Quality Metrics in Tests

### Test Classification

**Regression Tests:**
```rust
#[test]
fn test_word_fusion_regression_policy() {
    let pdf = "Anti-bribery and Corruption Policy Template (UK).pdf";
    let metrics = extract_and_analyze(pdf);

    // Critical: No true regressions allowed
    assert_eq!(metrics.critical_errors, 0);

    // Expected: Document PDF defects separately
    assert!(metrics.pdf_structure_defects >= 1);

    // Quality still good despite defects
    assert!(metrics.quality_score >= 8.0);
}
```

**Confidence Levels:**

```rust
pub enum FusionConfidence {
    High,         // Clear TJ offset data; high confidence word boundary exists
    Medium,       // Geometric gap analysis; moderate confidence
    Low,          // Heuristic-only; low confidence
    PdfStructure, // PDF authoring defect; not an extractor issue
}
```

---

## Improving Quality

### For Library Users

**1. Configure Extraction Parameters**

```rust
use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::SpanMergingConfig;

let config = SpanMergingConfig::adaptive();  // Recommended (v0.1.3+)
let doc = PdfDocument::open("document.pdf")?;
let spans = doc.extract_spans_with_config(page, config)?;
```

**2. Post-Process Markdown**

```bash
# Detect word fusions
grep -oE '\b[a-z]{3,}[A-Z][a-z]{3,}\b' output.md

# Fix common patterns manually
sed -i 's/draftpolicy/draft policy/g' output.md
sed -i 's/thefollowingtypesof/the following types of/g' output.md
```

**3. Validate Before Use**

Always visually spot-check a sample of generated markdown, especially:
- First page of document
- Any page with quality score < 80
- Pages with detected word fusions or spacing artifacts

### For PDF Authors

**Create extraction-friendly PDFs:**

1. **Use proper TJ formatting:** Text show arrays with explicit offsets
   ```
   ✅ [(word1) -200 (word2)] TJ
   ❌ [(word1word2)] TJ
   ```

2. **Avoid tight spacing:** Use normal inter-word gaps (0.15-0.35pt typical)
   ```
   ✅ Normal spacing: readable, extractable
   ❌ Kerned spacing: may be misinterpreted
   ```

3. **Use standard fonts:** Avoids embedding issues
   ```
   ✅ Helvetica, Times-Roman, DejaVuSans
   ❌ Custom embedded fonts with unusual metrics
   ```

4. **Export cleanly:** Use modern PDF tools
   - LibreOffice Writer
   - Microsoft Word (recent versions)
   - Professional tools (InDesign, LaTeX)

5. **Avoid:** Low-quality scans, OCR'd PDFs (limited text layer)

---

## Examples

### Good Quality PDF

**Metrics:**
- Quality Score: **97/100**
- Word Fusions: 0
- Spurious Spaces: 0
- Gap Median: 0.18pt
- Gap CV: 1.1

**Characteristics:**
- Clear fonts, proper spacing
- Well-structured sections
- Metadata present (title, author)
- Modern authoring tool

---

### Fair Quality PDF (With Known Defect)

**Metrics:**
- Quality Score: **83/100**
- Word Fusions: 1 (PDF structure defect)
- Spurious Spaces: 0
- Gap Median: 0.22pt
- Gap CV: 1.3

**Characteristics:**
- One compound word ("draftpolicy") from PDF tool
- Otherwise excellent formatting
- Issue documented as PDF structure defect
- Not a regression; expected and handled

**Recommendation:** Acceptable for use; manually fix the one known word fusion.

---

### Poor Quality PDF

**Metrics:**
- Quality Score: **58/100**
- Word Fusions: 8
- Spurious Spaces: 4
- Gap Median: 0.08pt
- Gap CV: 2.8

**Characteristics:**
- Extremely tight, variable spacing
- Multiple word fusions beyond PDF structure defects
- Likely OCR'd or low-quality source
- Significant manual correction needed

**Recommendation:** Review extraction approach; consider:
- Re-scanning if available
- Requesting native PDF from source
- Manual reconstruction from page images

---

## Troubleshooting

### "I'm seeing word fusions"

1. **Check the PDF:** Is it a known limitation?
   - Is it the "draftpolicy" case?
   - Check: `cargo test test_word_fusion_regression_policy --release`

2. **Check your configuration:**
   ```rust
   // Use adaptive (recommended)
   let config = SpanMergingConfig::adaptive();

   // NOT conservative or aggressive
   let config = SpanMergingConfig::conservative();  // Too strict
   ```

3. **Check the quality score:**
   - Quality < 70? Likely PDF quality issue, not extractor
   - Quality > 85? Bug report if still seeing fusions

### "Quality score is low"

1. **Examine the breakdown:**
   ```rust
   if metrics.word_fusions > 5 {
       println!("Likely PDF structure issues");
   }
   if metrics.gap_cv > 2.0 {
       println!("Likely layout/formatting complexity");
   }
   ```

2. **Assess the PDF:**
   - Is it a scan or OCR'd?
   - Does it have complex multi-column layout?
   - Is it professionally authored?

3. **Consider alternatives:**
   - Export from source format if available
   - Request higher-quality PDF from source
   - Use specialized tool for PDF type (scans → OCR tool)

### "I'm getting different results in different runs"

PDF Oxide is deterministic. If results differ:

1. **Check for randomness:**
   - Ensure `RUSTFLAGS` not set to random features
   - Verify `Cargo.lock` is committed
   - Use same CPU/memory (performance but not correctness)

2. **Check PDF modifications:**
   - Are you modifying the PDF file?
   - Is it being auto-updated?
   - Try with a fresh copy

3. **Report as bug** with:
   - PDF file (if not sensitive)
   - Extraction code
   - OS/Rust version
   - Multiple run comparison

---

## References

- **Adaptive Threshold:** Phase 5 documentation, `docs/PHASE_5_ADAPTIVE_THRESHOLD_PLAN.md`
- **PDF Spec:** ISO 32000-1:2008, Section 9.4.3 (Text Show Array)
- **Known Limitations:** `docs/ADR-001-pdf-structure-limitations.md`
- **Architecture:** `docs/ARCHITECTURE.md`

---

## FAQ

**Q: Why can't you just split compound words with a dictionary?**
A: Causes false positives (network → net+work, feedback → feed+back). Also language-specific and breaks on names (iPhone, CamelCase). Better to classify as PDF defect and handle separately if needed.

**Q: Will you support this in future versions?**
A: Yes, v1.0+ may include optional language-aware word segmentation. See `docs/ADR-001-pdf-structure-limitations.md` for planned approach.

**Q: How can I help improve quality?**
A: Send PDFs with unexpected results. Include quality metrics output and what you expected. We continuously improve detection algorithms based on real-world data.

**Q: Is word fusion a regression?**
A: Not always. If caused by PDF structure defect (single-string encoding), it's documented. If caused by threshold/detection bug, it's a regression. Our tests distinguish both.

---

## Version History

| Version | Changes |
|---------|---------|
| v0.1.3+ | Adaptive threshold (Phase 5), TJ offset collection (Phase 4), PDF structure defect classification |
| v0.1.2 | Conservative span merging, basic word fusion detection |
| v0.1.1 | Initial extraction, geometric gap analysis |
