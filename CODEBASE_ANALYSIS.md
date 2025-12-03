# PDF Extraction Codebase Analysis & Fix Strategy

**Analysis Date:** 2025-12-02
**Branch:** fix/pdf-format-handling
**Goal:** Identify architectural root causes and implement proper fixes

---

## Executive Summary

The pdf_oxide codebase has **sophisticated text extraction** with multiple safeguards:
- Gap-based space detection (gap > 0.1pt threshold)
- Heuristic-based space insertion (CamelCase, digit-letter transitions)
- Font weight detection (priority 1-4 system)
- Bold text preservation with word boundary checking
- Span merging and deduplication

**However**, our test PDFs show issues indicating the problem is likely in:
1. **PDF-specific character positioning** (unusual coordinate systems, font handling)
2. **Span boundary detection** when font changes occur mid-line
3. **Markdown rendering logic** not preserving formatting across style boundaries

---

## Data Flow Analysis

### Pipeline: PDF → TextSpans → TextBlocks → Markdown

```
PDF Content Stream
    ↓
[text.rs:614] extract_text_spans()
    ├─ Parse PDF operators (Tj, TJ, Td, Tm)
    ├─ Track GraphicsState (matrix, font, color)
    ├─ Build TjBuffer per operator sequence
    ├─ Detect spaces from TJ offsets
    └─ Create TextSpan per buffer (contains full text + position + styling)
    ↓
[text.rs:732-965] Post-processing
    ├─ sort_spans_by_reading_order() - handles multi-column PDFs
    ├─ deduplicate_overlapping_spans() - removes duplicate renderings
    └─ merge_adjacent_spans() - combines spans with gap < 3pt
        └─ Space insertion logic:
            - needs_space_by_gap = gap > font_size * 0.25
            - needs_space_by_heuristic = CamelCase || digit-letter
            - Final: needs_space = gap > 0.1pt || heuristic || gap > threshold
    ↓
Vec<TextSpan> (spans with merged text, proper spacing, bold detection)
    ↓
[markdown.rs:191] convert_page_from_spans()
    ├─ Convert spans → TextBlocks (preserving all fields)
    ├─ Sort blocks by Y then X (reading order)
    ├─ merge_adjacent_char_spans() - character-level merging
    ├─ detect_headings() - cluster font sizes
    └─ Format lines with bold markers:
        - Group consecutive blocks by bold status
        - Check word boundaries for valid bold positions
        - Insert "**" only if both opening & closing valid
    ↓
Markdown Output
```

---

## Root Cause Analysis: Issue by Issue

### ISSUE 1: Extra Spaces in Words ("organi s ations")

**Symptom:** Character sequences with unusual spacing produce spurious spaces mid-word

**Location:** `src/extractors/text.rs:1044-1059` - Space insertion in `merge_adjacent_spans()`

**Current Logic:**
```rust
let space_threshold = current.font_size * 0.25;  // ~3pt for 12pt font
let needs_space = gap > space_threshold
                || needs_space_by_heuristic
                || gap > 0.1;  // VERY aggressive
```

**Root Cause Analysis:**
- The `gap > 0.1` condition is **too aggressive** for policy documents
- Policy PDFs have unusual font metrics (multiple fonts, font substitutions)
- A gap of 0.1pt appears when:
  - Font scaling differences (text_matrix.d != 1.0)
  - Non-standard character widths in embedded fonts
  - Font transitions (Times → Times-Bold)
  - Kerning tables with unusual metrics

**Example from Privacy Policy:**
```
Current: "organi s ations"

What's happening:
- "organi" extracted as TextSpan 1
- " s " extracted as TextSpan 2 (or character with position gap)
- "ations" extracted as TextSpan 3
- Gap between spans 1-2: ~0.15pt (> 0.1) → space inserted
- This suggests PDF has spacing around "s" for some reason
  (possibly: font substitution, glyph variant, encoding issue)
```

**Fix Strategy:**
- **Increase minimum gap threshold from 0.1pt to 0.5pt** for conservative approach
- **Add PDF analysis debug mode** to identify spacing patterns before fix
- **Context-aware thresholds**: different thresholds for policy vs. academic PDFs
- **Font transition detection**: don't insert space if font changed mid-word

### ISSUE 2: Fused Words ("thefollowingtypesof")

**Symptom:** Adjacent words with no visible gap get concatenated

**Location:** Same `merge_adjacent_spans()` function

**Root Cause Analysis:**
- This is the **inverse** of Issue 1
- Spans with gap ≤ 0.1pt are merged without space
- But the gap calculation might be wrong for certain PDF structures

**Gap Calculation** (text.rs line 991):
```rust
let gap = span.bbox.x - (current.bbox.x + current.bbox.width);
```

**Issues:**
- `bbox.width` is calculated from character widths
- Character widths come from font widths table (or defaults)
- Font substitution or embedded fonts may have wrong widths
- Result: gap calculation is off, words get fused

**Example:**
```
"the following"

In PDF:
- "the" bbox: x=0, width=12.5
- "following" bbox: x=11.8, width=45

Calculated gap: 11.8 - (0 + 12.5) = -0.7pt (NEGATIVE!)

When gap < 0 → spans overlap → merge without space
Even though visually they're separate
```

**Fix Strategy:**
- **Debug gap calculation** with real PDFs
- **Add safeguards**: if gap < 0, treat as adjacent overlap, still insert space
- **Validate bbox.width**: check against actual character positions
- **Font width verification**: log when calculated width mismatches visual position

### ISSUE 3: Bold Text Loss ("Accesscontrol:Enforce")

**Symptom:** Bold formatting markers disappear, mixed bold/regular text becomes monolithic

**Location:** `src/converters/markdown.rs:280-337` - Bold grouping and marker insertion

**Current Logic** (markdown.rs line 284-288):
```rust
// Find all consecutive blocks with same bold status
let mut j = i + 1;
while j < line_indices.len() && blocks[line_indices[j]].is_bold == is_bold {
    j += 1;
}
// Groups blocks where blocks[i..j] all have same bold status
```

**Root Cause:**
- Bold detection works correctly at span level (FontInfo.is_bold())
- But the markdown converter **requires bold status to be consistent across the group**
- When a line has: **"Access control:"** + regular "Enforce"
  - Block 1: "Access control:" (is_bold=true)
  - Block 2: "Enforce" (is_bold=false)
  - Loop finds j=1 (only one consecutive bold block)
  - Then renders: "**Access control:**" + "Enforce"
  - Which should work...

**Wait, let me reconsider.** If Block 1 is "Access control:" with is_bold=true, it should render as "**Access control:**". Let me check what our test showed:

From test output: `"Accesscontrol:Enforce"` - NO bold markers at all

This suggests:
1. Either `is_bold` is false when it shouldn't be
2. Or the word boundary check is failing
3. Or the blocks aren't being created correctly from spans

**Word Boundary Check** (markdown.rs line 316):
```rust
let can_insert_open = should_insert_bold_marker(prev_char, first_char_in_group);
let can_insert_close = should_insert_bold_marker(last_char_in_group, next_char_after_group);
let should_insert_markers = is_bold && can_insert_open && can_insert_close;
```

This checks if markers can be inserted around the group. If `prev_char` is space and `first_char` is 'A', should it insert?

**Fix Strategy:**
- **Debug: Add logging to span conversion**
  - Log each span: text, is_bold, font_name, font_weight
  - Trace through markdown conversion
  - Identify where bold information is lost
- **Hypothesis to test**: PDF fonts have unusual names/weights that don't trigger bold detection
- **Potential fix**: Improve font weight detection (priorities 1-4 in font_dict.rs)

---

### ISSUE 4: Table Detection Missing

**Symptom:** Tables rendered as plain text without column structure

**Location:** `src/converters/markdown.rs` - No table detection algorithm exists

**Expected Structure** (from IT Security Policy):
```
Original Table:
┌──────────┬─────────────────┐
│   Role   │  Responsibility │
├──────────┼─────────────────┤
│Executive │ Oversee...      │
│Security  │ Lead impl...    │
│IT dept   │ Configure...    │
└──────────┴─────────────────┘

Current Output:
Role Responsibility
Executive Oversee...
Security Lead impl...
IT dept Configure...

Expected Markdown:
| Role | Responsibility |
|------|-----------------|
| Executive | Oversee strategy |
| Security officer | Lead implementation |
```

**Root Cause:**
- No table detection algorithm implemented
- Tables in PDFs are complex: can be drawn with graphics (lines), or text with alignment

**Fix Strategy:**
- **Create new module:** `src/extractors/table_detector.rs`
- **Algorithm:**
  1. Detect column boundaries from X-coordinate clustering
  2. Detect row boundaries from Y-coordinate clustering
  3. Check if blocks form a grid pattern
  4. Extract text for each cell (block union)
  5. Generate markdown table format
- **Integration point:** In `convert_page_from_spans()`, detect tables before converting to markdown
- **Complexity:** Medium (no dependencies on ML features)

---

## Architecture: Key Insights

### 1. Span-Based vs. Character-Based Extraction

**Current approach uses BOTH:**
- **TextSpan** (line-level): text + bbox + styling, created during content stream parsing
- **TextChar** (character-level): individual chars with per-character styling
- **TextBlock** (group-level): union of chars or spans with dominant styling

**Problem:**
- Markdown converter converts Spans → Blocks (line 203-216)
- This loses per-character styling information
- Bold marker placement doesn't account for mid-span style changes

**Why?**
- Span = complete text string from one operator sequence
- If an operator sequence includes "Access control" in bold, it's all bold
- If next operator sequence has "Enforce" in regular, it's all regular
- The text is split correctly, but markdown grouping is per-block

### 2. GraphicsState Tracking

The code tracks font, size, color, matrix position throughout extraction, but:
- **Matrix positioning can be unusual** in policy PDFs
- **Text matrix (Tm operator) sets absolute position** but PDFs may use relative positioning
- **Font transitions** mid-line might not be handled correctly when calculating bounding boxes

### 3. Gap Calculation Fragility

The gap calculation assumes:
```
gap = next.bbox.x - (current.bbox.x + current.bbox.width)
```

But this breaks if:
- `current.bbox.width` is calculated incorrectly (font metrics)
- Spans are created by different operator sequences with different positioning logic
- Matrix transformations affect positioning differently for different fonts

---

## Implementation Plan: Proper Fixes

### PHASE 1: DIAGNOSIS (Foundation)

**Step 1.1: Add comprehensive logging**
```rust
// In src/extractors/text.rs::merge_adjacent_spans()
// Log for each merge decision:
// - Current span: text, font, size, bold status
// - Next span: text, font, size, bold status
// - Gap calculation: gap value, threshold, decision (merge/no-merge)
// - Space insertion: gap condition, heuristic condition, final decision

// In src/converters/markdown.rs::convert_page_from_spans()
// Log for each block:
// - Text, position (x, y), font, size, is_bold
// - Grouping decision: why grouped with previous/next
// - Bold marker decision: can_insert_open, can_insert_close
```

**Step 1.2: Extract sample PDFs with logging**
```bash
RUST_LOG=debug cargo run --example debug_extraction -- \
  ~/projects/pdf_oxide_new_docs/Privacy*.pdf > /tmp/extraction_logs.txt
```

**Step 1.3: Analyze logs to identify exact issues**
- Find where spacing gets inserted incorrectly
- Find where spacing is missing
- Find where bold markers aren't being inserted

---

### PHASE 2: HIGH-PRIORITY FIXES

**Fix #1: Conservative gap threshold**

**File:** `src/extractors/text.rs:1059`

**Current:**
```rust
let needs_space = needs_space_by_gap || needs_space_by_heuristic || gap > 0.1;
```

**Proposed:**
```rust
// Only insert space for significant gaps or clear heuristic matches
// gap > 0.1 is too aggressive and causes spurious spaces in multi-font PDFs
let conservative_threshold = 0.3;  // Only gaps > 0.3pt get aggressive treatment
let needs_space = needs_space_by_gap
                || (needs_space_by_heuristic && gap > 0.0)
                || (gap > conservative_threshold);

// Log space insertion decisions for policy documents
if ENABLE_DEBUG_LOGGING {
    if gap > 0.0 && gap < conservative_threshold && !needs_space_by_gap {
        eprintln!("Skipping space for small gap {:.2}pt: '{}' + '{}'",
                  gap, current.text, span.text);
    }
}
```

**Why:** Eliminates spurious spaces in multi-font documents like policy PDFs

---

**Fix #2: Negative gap handling**

**File:** `src/extractors/text.rs:991`

**Current:**
```rust
let gap = span.bbox.x - (current.bbox.x + current.bbox.width);
// If gap is negative, spans overlap - but code doesn't handle this case!
let should_merge = same_line && (-0.5..3.0).contains(&gap) && !large_gap_indicates_column;
```

**Proposed:**
```rust
let gap = span.bbox.x - (current.bbox.x + current.bbox.width);

// Handle negative gaps (overlapping spans) - common with font metrics issues
if gap < 0.0 {
    // Overlapping spans likely due to font width miscalculation
    // Still merge, but ALWAYS insert space (they shouldn't overlap)
    log::warn!(
        "Overlapping spans detected (gap={:.2}pt): '{}' + '{}' - inserting space",
        gap, current.text, span.text
    );
    // Continue to merge with space insertion
}

let same_line = (current.bbox.y - span.bbox.y).abs() < line_tolerance;
let large_gap = gap > 5.0;  // Column boundary
let should_merge = same_line && (-0.5..3.0).contains(&gap) && !large_gap;
```

**Why:** Prevents word fusion when font metrics are off

---

**Fix #3: Font transition detection**

**File:** `src/extractors/text.rs:1010-1027`

**Current:** No special handling for font changes mid-merge

**Proposed:**
```rust
// Before deciding on space insertion, check if fonts changed
let font_changed = current.font_name != span.font_name;
let font_size_changed = (current.font_size - span.font_size).abs() > 0.5;
let bold_status_changed = current.font_weight.is_bold() != span.font_weight.is_bold();

if font_changed || font_size_changed || bold_status_changed {
    // Font transition detected - be conservative with space insertion
    // These transitions often have unusual gap values due to font metrics
    log::debug!(
        "Font transition: '{}' ({}, {}pt, {}) → '{}' ({}, {}pt, {})",
        current.text, current.font_name, current.font_size,
        if current.font_weight.is_bold() { "bold" } else { "regular" },
        span.text, span.font_name, span.font_size,
        if span.font_weight.is_bold() { "bold" } else { "regular" }
    );

    // For font transitions, use stricter space threshold
    let transition_threshold = font_size * 0.15;  // Even stricter: 1.5% not 2.5%
    let needs_space = gap > transition_threshold || needs_space_by_heuristic;
    // Continue with merge using needs_space...
} else {
    // Standard merge logic
    let needs_space = needs_space_by_gap || needs_space_by_heuristic || gap > 0.1;
}
```

**Why:** Handles multi-font policy documents properly

---

**Fix #4: Improve bold detection for embedded fonts**

**File:** `src/fonts/font_dict.rs:823-905`

**Current priorities:**
1. FontWeight field from descriptor
2. ForceBold flag
3. Font name regex matching
4. StemV heuristic

**Add Priority 0:** Check previous span's bold status as context

**Proposed addition to `get_font_weight()`:**
```rust
// NEW PRIORITY 0: Check font family context
// Some PDFs embed fonts with non-standard names that still clearly indicate weight
// Examples: "ABCDEF+TT1111Bold", "CustomFont-BoldMT"
// Look for patterns even if not in standard name patterns

if base_font.contains('-') {
    let parts: Vec<&str> = base_font.split('-').collect();
    if parts.len() > 1 {
        let weight_part = parts.last().unwrap_or(&"");
        if weight_part.eq_ignore_ascii_case("Bold")
            || weight_part.eq_ignore_ascii_case("BoldMT")
            || weight_part.contains("Bold") {
            return FontWeight::Bold;
        }
    }
}

// Also check embedded font names like "ABCDEF+TimesNewRoman-Bold"
if let Some(plus_pos) = base_font.find('+') {
    let embedded_part = &base_font[plus_pos+1..];
    if embedded_part.contains("Bold") || embedded_part.contains("bold") {
        return FontWeight::Bold;
    }
}
```

**Why:** Policy PDFs use custom embedded fonts; this improves detection accuracy

---

### PHASE 3: TABLE DETECTION

**New file:** `src/extractors/table_detector.rs`

**Create table detection algorithm:**
```rust
pub struct TableDetector {
    x_tolerance: f32,  // Pixels tolerance for column alignment
    y_tolerance: f32,  // Pixels tolerance for row alignment
}

impl TableDetector {
    pub fn detect_tables(&self, blocks: &[TextBlock]) -> Vec<Table> {
        // 1. Cluster blocks by X coordinate (column detection)
        let x_clusters = self.cluster_by_x(blocks);

        // 2. Cluster blocks by Y coordinate (row detection)
        let y_clusters = self.cluster_by_y(blocks);

        // 3. Check if pattern is grid-like
        if self.is_grid_pattern(&x_clusters, &y_clusters) {
            // 4. Extract table cells and return
            vec![self.extract_table(&x_clusters, &y_clusters)]
        } else {
            vec![]
        }
    }

    fn cluster_by_x(&self, blocks: &[TextBlock]) -> Vec<Vec<usize>> {
        // Use histogram of X positions to find column boundaries
    }

    fn cluster_by_y(&self, blocks: &[TextBlock]) -> Vec<Vec<usize>> {
        // Use histogram of Y positions to find row boundaries
    }

    fn is_grid_pattern(&self, x_clusters: &[Vec<usize>], y_clusters: &[Vec<usize>]) -> bool {
        // Check if number of cells matches grid structure
        // For a real table: len(x_clusters) * len(y_clusters) ≈ number of blocks
    }
}
```

**Integration:** Call `detect_tables()` in `markdown.rs::convert_page_from_spans()` before rendering

---

## Testing Strategy

### Unit Test Coverage

**File:** `tests/test_markdown_extraction_quality.rs`

Tests to add based on root cause analysis:

```rust
#[test]
fn test_gap_calculation_with_font_substitution() {
    // Test: gap calculation when font metrics are off
    // Ensure negative gaps are handled correctly
}

#[test]
fn test_font_transition_space_insertion() {
    // Test: space insertion at font transitions
    // Bold → Regular, different sizes, etc.
}

#[test]
fn test_aggressive_gap_threshold_in_multifont_docs() {
    // Test: gap > 0.1pt doesn't trigger in close font transitions
}

#[test]
fn test_bold_detection_with_embedded_fonts() {
    // Test: bold detection for custom embedded font names
}

#[test]
fn test_table_grid_detection() {
    // Test: table detection algorithm identifies grid patterns
}
```

### Integration Testing

1. Extract real PDFs with logging
2. Verify gap calculations are correct
3. Verify span merging produces correct text
4. Verify markdown output has proper formatting

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/extractors/text.rs` | Gap threshold, font transition, negative gap handling | P0 |
| `src/fonts/font_dict.rs` | Improve bold detection for embedded fonts | P0 |
| `src/extractors/table_detector.rs` | NEW: Table detection algorithm | P1 |
| `src/converters/markdown.rs` | Integrate table detection | P1 |
| `tests/test_markdown_extraction_quality.rs` | Add new test cases | P0 |
| `SPAN_SPACING_INVESTIGATION.md` | Document findings (already exists) | Reference |

---

## Validation Metrics

After implementing fixes:

- [ ] **No spurious spaces:** Gaps between 0.1-0.3pt don't create unwanted spaces
- [ ] **Font transitions preserve spacing:** "organi s ations" → "organisations"
- [ ] **Negative gaps handled:** Overlapping spans still have spaces
- [ ] **Bold text preserved:** "**Access control:**" stays bolded
- [ ] **Tables detected:** Policy tables render as markdown tables
- [ ] **All tests pass:** test_markdown_extraction_quality suite passes
- [ ] **Sample PDFs improved:** All 6 sample PDFs extract cleanly

---

## Summary

The pdf_oxide codebase is **well-architected** with sophisticated extraction logic. The issues we found are not architectural flaws but rather edge cases:

1. **Gap threshold too aggressive** for multi-font policy documents (0.1pt)
2. **Font metrics handling** when fonts transition mid-word
3. **Table detection** not implemented (requires new module)
4. **Bold detection** could improve for embedded fonts

**All fixes are surgical and localized** - no major refactoring needed. The existing test suite and logging infrastructure make it straightforward to validate changes.

