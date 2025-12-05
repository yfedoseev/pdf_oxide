# Real Examples of Quality Issues Found

## 1. Word Fusion Examples (CRITICAL)

### Example 1: RFC_2616_HTTP_1_1.pdf (Worst Case)
**Original PDF text should read:**
> "The request-header field specifies the Internet host..."
> "An intermediary program which acts as both a server..."
> "It builds on the discipline of reference provided by..."

**Actual extracted text:**
> "inThe request-header field specifies..."
> "otherAn intermediary program which acts..."
> "aIt builds on the discipline of reference..."

**Pattern:** Article or short word fused to next word
- "in" + "The" → "inThe"
- "other" + "An" → "otherAn"  
- "a" + "It" → "aIt"

**Impact:** Makes text difficult to read and breaks semantic meaning

---

### Example 2: NASA_Apollo_11_Preliminary_Science_Report.pdf
**Original should read:**
> "There were about two billion people on Earth"
> "Partners will be able to share facilities on Freedom to facilitate basic research in"
> "Look at them on Earth"

**Actual extracted:**
> "wereabouttwobillionpeopleonEarth"
> "partnerswillbeabletosharefacilitiesonFreedomtofacilitatebasicresearchin"
> "themonEarth"

**Impact:** SEVERE - entire sentences concatenated without spaces

---

### Example 3: Academic Paper arxiv_2510.26793v1.pdf
**Example fusions found:**
> "performsThe" (should be: "performs The")
> "performedThe" (should be: "performed The")
> "areSyntax" (should be: "are Syntax")
> "parsersExisting" (should be: "parsers Existing")

**Pattern:** Verb + article/noun fusions at sentence/clause boundaries

---

## 2. Empty Bold Markers (MAJOR)

### Example: arxiv_2510.26793v1.pdf
**Line 409:**
```markdown
**Dataset ** **Two**
```

**Should be:**
```markdown
**Dataset Two**
```

**Pattern:** Bold marker inserted in middle of phrase with space, creating empty marker pair

---

## 3. Excessive Spacing (MAJOR)

### Example: Government CFR Document (2.4 MB output)
**Statistics:**
- 2,278 instances of multiple consecutive spaces in one document
- Often 5-10 spaces between words in tables of contents
- Column alignments creating massive gaps

**Sample (table of contents):**
```
Section 1.1          Page 5
Section 1.2                    Page 15
Section 1.3                              Page 25
```

**Issue:** Tab stops or column alignments converted to literal spaces

---

### Example: Newspaper IA_02620150R.nlm.nih.gov.pdf
**Statistics:** 4,133 instances of excessive spacing

**Likely cause:** Multi-column layout - columns separated by large gaps
```
Column 1 text here          Column 2 text here
More text in col 1          More text in col 2
```

---

## 4. Missing Spaces After Punctuation (MAJOR)

### Example: RFC_2616_HTTP_1_1.pdf
**Actual text:**
> "affects neither the requesting client nor the origin server, except to improve performance.When a cache is"

**Should be:**
> "affects neither the requesting client nor the origin server, except to improve performance. When a cache is"

**Pattern:** Period directly followed by capital letter (sentence boundary)

---

### Example: IRS Form 706
**Extracted text shows patterns like:**
> "Form 706.Estate Tax Return"
> "Part 1.Decedent and Executor Information"
> "Schedule A.Real Estate"

**Should be:**
> "Form 706. Estate Tax Return"
> "Part 1. Decedent and Executor Information"
> "Schedule A. Real Estate"

---

## 5. Broken Bold Formatting (MAJOR)

### Example: Technical Document arxiv_2312.00001.pdf
**Found 702 instances of unclosed bold markers**

**Sample:**
```markdown
**Theorem 1.Let X be a random variable with distribution F.
If E[X] exists, then...

**Proof:We proceed by induction on n.
For n=1, the result is trivial...
```

**Issue:** Bold markers opened but never closed across line breaks

---

## 6. False Positives: Form Field Names

### Example: IRS Forms (f1040es, Form 706, fw9)
**Detected as "word fusion":**
- topmostSubform
- Table_RecordEstimated
- Line1[0]
- f8_1[0]

**Reality:** These are legitimate camelCase form field names from PDF AcroForm structure

**Form field table example:**
```markdown
| Field Name | Value |
|------------|-------|
| topmostSubform[0].Page8[0].f8_1[0] | *[empty]* |
| topmostSubform[0].Page8[0].f8_2[0] | *[empty]* |
```

**Note:** These should be whitelisted, not treated as errors

---

## 7. Table Detection (WORKING)

### Example: Government CFR Documents
**Successfully detected 33 tables**

**Sample table output:**
```markdown
| Part | Title | Page |
|------|-------|------|
| 1 | General Provisions | 5 |
| 2 | Administrative Regulations | 15 |
| 3 | Implementation Rules | 25 |
```

**Quality:** Good - tables properly formatted in markdown

---

## 8. Header Detection (WORKING)

### Example: Academic Papers
**Successfully detected headers:**
```markdown
**1 Introduction**
**2 Related Work**
**3 Methodology**
**3.1 Experimental Setup**
**3.2 Data Collection**
**4 Results**
```

**Quality:** Good - hierarchical structure preserved

---

## Visual Comparison: Best vs Worst

### BEST: Mixed Content (7.5/10 Quality)
```markdown
# Extracted from: ZZLARWOCNXAHCS25AGWDA2UPFRV3G6TU.pdf

**Contract Agreement**

This agreement is made between Party A and Party B on the date
specified below. The terms and conditions are as follows:

1. Scope of Work
2. Payment Terms
3. Deliverables
```

**Assessment:** Clean, readable, minimal issues

---

### WORST: RFC_2616_HTTP_1_1.pdf (3.5/10 Quality)
```markdown
.....143.1.......143.2
 ..........143.2.1  ....153.2.2 
 .........153.2.3  ...........153.3
 .......163.3.1  ..............163.3.2 
 ......163.4 ...........163.4.1 
 ...........173.583.6

---

NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be 
interpreted as described inThe key words "MUST", "MUST NOT", "REQUIRED"

of other clients. Requests are serviced internally or by passing them on, 
with possible translation, to otherAn intermediary program which acts as 
both a server and a client

deletion. A cache stores cacheable responses in order to reduce the 
response time and network bandwidthA program's local store of response 
messages and the subsystem
```

**Assessment:** 
- Table of contents garbled with dots
- Multiple word fusions per paragraph
- Excessive spacing in some areas
- Nearly unreadable

---

## Summary Statistics

### Issue Prevalence

| Issue Type | Worst Case | Average | Best Case |
|------------|-----------|---------|-----------|
| Word Fusions | 400 (RFC) | 65 | 0 |
| Empty Bold | 438 (tech) | 57 | 0 |
| Excessive Spacing | 4,133 (news) | 929 | 0 |
| Missing Space After Punct | 965 (form) | 552 | 0 |
| Broken Bold | 702 (tech) | 238 | 1 |

### Quality by Complexity

| Document Type | Typical Quality | Issues |
|--------------|----------------|--------|
| Simple single-column | 8-9/10 | Minimal |
| Academic papers | 7-8/10 | Moderate word fusion |
| Forms | 6-7/10 | Spacing, field names |
| Multi-column layouts | 4-6/10 | Severe spacing |
| Complex technical specs | 3-5/10 | Multiple severe issues |

---

## Recommendations Based on Examples

### Priority 1: Fix Word Fusion
**Target documents for testing:**
1. RFC_2616_HTTP_1_1.pdf (400 fusions - worst case)
2. IRS_Form_706_2024.pdf (332 fusions)
3. Berkeley_Thesis_Security_1.pdf (233 fusions)

**Expected pattern to fix:**
- Short word + capital letter at word boundary
- Article/preposition fusions: "inThe", "aIt", "otherAn"
- Verb + article: "performsThe", "areSyntax"

### Priority 2: Normalize Spacing
**Target documents:**
1. IA_02620150R.nlm.nih.gov.pdf (4,133 excessive spaces)
2. CFR_2024_Title29_Vol1_Labor.pdf (2,530 excessive spaces)

**Fix approach:**
- Collapse multiple spaces to single space in body text
- Preserve table alignment
- Detect column boundaries

### Priority 3: Remove Empty Bold Markers
**Simple regex fix:**
```
s/\*\*\s*\*\*//g
```

**Should resolve 1,472 instances across 17 files**

---

**Examples Document Created:** December 4, 2025
**Purpose:** Provide concrete evidence for quality report findings
**Usage:** Reference for debugging and validation
