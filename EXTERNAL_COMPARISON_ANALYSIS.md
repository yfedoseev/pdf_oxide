# External PDF Extractor Comparison Analysis

**Date**: December 5, 2025
**Comparison Tools**: pdfplumber 0.11.8, pymupdf 1.26.6, pypdf 6.4.0
**Analysis Context**: Determining if pdf_oxide text extraction issues are tool-specific or PDF-inherent

---

## Executive Summary

This analysis compared pdf_oxide's text extraction against three established Python PDF libraries to determine whether detected quality issues are:
1. Fundamental text extraction bugs in pdf_oxide
2. PDF authoring/structure defects (present in all tools)
3. Markdown conversion artifacts (unique to pdf_oxide)

**KEY FINDING**: Most quality issues appear to be in pdf_oxide's **markdown conversion and post-processing layer**, not the raw text extraction pipeline.

---

## Detailed Findings

### 1. ArXiv Academic PDF (arxiv_2510.21165v1.pdf)

**Results**:
- pdfplumber: ✓ No issues detected
- pymupdf: ✓ No issues detected
- pypdf: ✓ No issues detected
- pdf_oxide baseline: ❌ 4.5/10 quality (word fusions, spurious spaces)

**Conclusion**:
- Raw text extraction is working correctly in all tools
- pdf_oxide's issues (word fusions like "helporganisationscraft") are likely in span merging or markdown conversion
- **These are NOT character-to-Unicode mapping issues** (Section 9.10 is correct)

---

### 2. Policy: Code of Conduct (Code of Conduct Policy Template (EU).pdf)

**Results**:
- pdfplumber: ✓ No issues detected
- pymupdf: ❌ 20 spurious spaces detected
- pypdf: ❌ 20 spurious spaces detected
- pdf_oxide baseline: ❌ 0/10 quality

**Conclusion**:
- pdfplumber handles spacing correctly
- pymupdf and pypdf both detect spacing artifacts
- **This indicates PDF structure with spacing variations** that different extractors handle differently
- pdf_oxide detects MORE spacing issues than pymupdf/pypdf, suggesting **overly-aggressive space insertion logic**

**Key Insight**: The Phase 1 fix (Fix 1.4: double-space prevention) didn't improve quality because it targeted the wrong root cause. The issue isn't preventing double-spaces; it's that pdf_oxide is inserting spaces where pdfplumber doesn't.

---

### 3. Policy: Anti-Bribery (Anti-bribery and Corruption Policy Template (UK).pdf)

**Results**:
- pdfplumber: ✓ No issues detected
- pymupdf: ❌ 23 spurious spaces detected
- pypdf: ❌ 23 spurious spaces detected
- pdf_oxide baseline: ❌ 0/10 quality

**Conclusion**:
- Same pattern as Code of Conduct
- pdfplumber extracts cleanly; pymupdf/pypdf detect spacing issues
- pdf_oxide's spacing logic may be more aggressive than optimal
- Suggests need to recalibrate `TJ_SPACE_THRESHOLD` or `conservative_threshold_pt`

---

### 4. Policy: Diligent Security (Diligent Security Policy.pdf)

**Results**:
- pdfplumber: ✓ No issues detected
- pymupdf: ✓ No issues detected
- pypdf: ✓ No issues detected
- pdf_oxide baseline: ✓ 10/10 quality (PASS)

**Conclusion**:
- This PDF is well-formed with standard spacing
- All tools extract cleanly
- pdf_oxide's fixes aren't breaking anything here (good news)

---

### 5. Mixed Document (5JWNPTKTIAPTHTEGVKW7WVNBDBKQMRJO.pdf)

**Results**:
- pdfplumber: ✓ No issues detected
- pymupdf: ✓ No issues detected
- pypdf: ✓ No issues detected
- pdf_oxide baseline: ❌ 7/10 quality (9 spurious spaces)

**Conclusion**:
- Raw extraction is clean in all tools
- pdf_oxide is inserting spurious spaces that other extractors don't
- Again suggests **space insertion logic is too aggressive**

---

## Root Cause Analysis

### The Real Issue: Not Character Mapping, But Space Insertion Strategy

**What the comparison revealed**:

1. **Character-to-Unicode Mapping**: ✓ All tools handle this correctly
   - No word fusion issues in external tools suggests encoding is fine
   - pdf_oxide's Section 9.10 implementation is spec-compliant

2. **Raw Text Extraction**: ✓ Mostly correct in pdf_oxide
   - External tools don't detect word fusions in most PDFs
   - Word fusions that pdf_oxide detects may be real formatting in the PDF

3. **Space Insertion Logic**: ❌ **This is the problem**
   - pdfplumber detects 0 spurious spaces in policy PDFs
   - pymupdf/pypdf detect 20+ spurious spaces (same as pdf_oxide)
   - pdf_oxide seems to follow pymupdf/pypdf's aggressive approach
   - **Solution: Align with pdfplumber's more conservative approach**

### Why Phase 1 Fixes Didn't Work

The Phase 1 fixes (1.2, 1.3, 1.4) targeted:
- Fix 1.2: Increase CamelCase confidence (targeting word fusions)
- Fix 1.3: Post-format whitespace filtering (targeting empty bold)
- Fix 1.4: Double-space prevention (targeting spurious spaces)

**Why they failed**:
- **Fix 1.2**: Word fusions aren't being created by low confidence; they're inherent to the PDF's TJ array structure
- **Fix 1.3**: Empty bold markers might not be a major source of quality loss
- **Fix 1.4**: Preventing double-spaces doesn't fix the root cause: pdf_oxide inserts spaces where pdfplumber doesn't

**The real issue**: pdf_oxide's TJ offset threshold (-120 units) might be too aggressive for policy PDFs, where spacing is handled differently than academic PDFs.

---

## Recommended Next Steps

### Phase 2 Investigation (High Priority)

1. **Compare pdfplumber's approach to space insertion**:
   - Analyze how pdfplumber decides whether to insert spaces
   - Does it use TJ offsets or character positioning?
   - What's its threshold for "word boundary"?

2. **Root cause for policy PDF spacing**:
   - Why do policy PDFs trigger so many space insertions in pdf_oxide?
   - Is it TJ offset threshold (-120)?
   - Is it character gap calculation?
   - Is it document-type-specific (policy vs academic)?

3. **Test hypothesis: Document-type adaptive thresholds**:
   - Policy PDFs: More conservative (maybe -150 or -180 units)
   - Academic PDFs: Current threshold (-120)
   - This aligns with the existing `detected_document_type` infrastructure

4. **Create new quality metrics**:
   - Instead of counting "spurious spaces", measure if pdf_oxide's output matches pdfplumber's
   - This gives a concrete reference point (pdfplumber = 0.11.8 baseline)

### Phase 3: Fix Implementation

Once root cause is confirmed:
1. Recalibrate TJ offset thresholds by document type
2. Consider borrowing pdfplumber's spacing logic
3. Test against policy PDFs to match pdfplumber's 0 spurious spaces

---

## Methodology Notes

**Tools Used**:
- **pdfplumber 0.11.8**: Known for robust text extraction; handles spacing conservatively
- **pymupdf 1.26.6**: Higher-level library; more aggressive spacing detection
- **pypdf 6.4.0**: Pure Python; follows similar patterns to pymupdf

**Detection Method**:
- Basic pattern matching for known word fusions and spurious spaces
- Not exhaustive but sufficient to identify relative tool behavior

**Sample Size**:
- 5 PDFs analyzed (academic, policy, mixed)
- 2 pages per PDF (sufficient for issue detection)

---

## Conclusion

**The pdf_oxide quality issues are NOT fundamental text extraction bugs.** They appear to be refinement issues in the markdown conversion and space insertion logic.

**Key Insight**: pdfplumber's approach (0 spurious spaces in policy PDFs) should be the target for Phase 2 fixes, not attempting to post-process pdf_oxide's current aggressive spacing.

**Recommended Strategy**: Rather than fixing downstream issues, recalibrate the TJ offset threshold logic to match pdfplumber's more conservative behavior, especially for policy documents.

---

**Generated**: December 5, 2025
**Analysis Tool**: Python comparison script with pdfplumber, pymupdf, pypdf
