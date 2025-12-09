# PDF Spec Word Boundary Analysis

## The Core Problem

Per ISO 32000-1:2008 Section 9.4.3 NOTE 6:
> "The identification of what constitutes a word is **unrelated** to how the text happens to be grouped into show strings. The division into show strings has no semantic significance."

This means the PDF spec does NOT define word boundaries. We must infer them from available signals.

## Current Implementation Signals

| Signal | Source | Threshold | Confidence |
|--------|--------|-----------|------------|
| TJ Offset | Section 9.4.3 | -120 thousandths | 1.0 |
| Geometric Gap | Section 9.4.4 | 0.05 * font_size | 0.95 |
| Boundary Whitespace | Text strings | Existing space | 1.0 |

## PDF Spec Signals We Could Add

### 1. Word Spacing (Tw) as Semantic Signal
**Spec Reference**: Section 9.3.3

Current behavior: `Tw` is only used for width calculation.

Per spec:
> "Word spacing shall work the same way as character spacing but shall apply only to the ASCII SPACE character (0x20)."

**Proposed Enhancement**:
- When `Tw > 0`, the PDF writer intended word separation
- This is a **semantic signal** that a space character means "word boundary"
- We could use `Tw > 0` as an indicator that spans with 0x20 between them are separate words

**Impact**: Low - most PDFs with fused words don't use Tw.

### 2. ToUnicode Space Detection
**Spec Reference**: Section 9.10

When a character code maps to Unicode U+0020 (SPACE) via ToUnicode CMap, it's definitively a word boundary.

**Current Status**: We already convert to Unicode but may not be treating spaces specially.

**Proposed Enhancement**:
- When processing text, if any character maps to U+0020 (or other space Unicode), treat it as a definitive word boundary
- This should already work if space is preserved in text strings

**Impact**: Should already work. Verify implementation.

### 3. ActualText Entries
**Spec Reference**: Section 14.9.4

`ActualText` provides exact replacement text for structure elements. This is the most authoritative source.

**Current Status**: Not checked.

**Proposed Enhancement**:
- When a structure element has `ActualText`, use it directly instead of extracting from content
- This would perfectly handle cases where PDF producer encodes proper text

**Impact**: High for tagged PDFs, none for untagged.

### 4. Adaptive TJ Threshold
**Spec Reference**: Section 9.4.3

TJ offsets are in "thousandths of a unit of text space" (thousandths of em).

Current threshold: -120 (static)

**Typical values observed**:
- Kerning: -10 to -50 (brings characters closer)
- Word boundary: -150 to -300 (creates gap)
- Some PDFs use smaller offsets for word boundaries

**Proposed Enhancement**:
- Lower threshold to -80 or -60 for tighter PDFs
- Or use adaptive threshold based on font's space glyph width
- `threshold = -0.3 * space_glyph_width * 1000` (30% of space width)

**Impact**: Medium - helps PDFs using small TJ offsets.

### 5. Font Space Width Analysis
**Spec Reference**: Section 9.6.1 (Font Descriptors)

The font's actual space glyph width can inform our threshold decisions.

**Proposed Enhancement**:
- Extract the width of glyph 0x20 (SPACE) from font
- Use this to calculate adaptive thresholds:
  - TJ threshold: `-space_width * 0.3` (30% of space)
  - Geometric threshold: `space_width * 0.2` (20% of space)

**Impact**: High - makes thresholds font-aware.

## Analysis of Problem PDFs

### diligent_security_policy_8.6.pdf
This PDF has 181 CamelCase fusions like "comCopyright", "SecurityFull".

**Root Cause Investigation Needed**:
1. What TJ offsets are used between fused words?
2. Does the PDF use Tw?
3. Does it have ToUnicode mappings?
4. Is it tagged (has ActualText)?

### Anti-bribery Policy Template (EU).pdf
This PDF now extracts correctly with 0.05 threshold.

**Working signals**: Geometric gaps are sufficient.

## Recommended Improvements (Priority Order)

### Priority 1: Debug Logging for Problem PDFs
Add logging to understand WHY words are fused:
```rust
log::debug!(
    "Word boundary check: gap={:.2}pt, threshold={:.2}pt, tj_offset={}, font_size={:.1}",
    gap, threshold, tj_offset, font_size
);
```

### Priority 2: Font-Aware Thresholds
Calculate thresholds based on actual font metrics:
```rust
fn calculate_space_based_threshold(&self, font: &FontInfo) -> f32 {
    let space_width = font.get_glyph_width(0x20);
    // Word boundary = gap > 20% of space width
    space_width * font_size / 1000.0 * 0.2
}
```

### Priority 3: Lower TJ Offset Threshold
Try -80 instead of -120 for tighter PDFs:
```rust
const TJ_WORD_BOUNDARY_THRESHOLD: f32 = -80.0; // thousandths of em
```

### Priority 4: ActualText Support
For tagged PDFs, use ActualText when available:
```rust
if let Some(actual_text) = structure_element.get("ActualText") {
    return actual_text.as_string();
}
```

## Conclusion

The PDF spec does NOT define word boundaries. We're doing the best we can with:
1. TJ offsets (explicit positioning)
2. Geometric gaps (measurement-based)
3. Boundary whitespace (content-based)

For extremely tight PDFs like `diligent_security_policy_8.6.pdf`, the only solutions are:
1. Lower thresholds (risk: false positives)
2. Font-aware thresholds (more accurate)
3. ActualText support (for tagged PDFs)

The fundamental limitation is that **some PDFs simply don't encode word boundaries** in any recoverable form.
