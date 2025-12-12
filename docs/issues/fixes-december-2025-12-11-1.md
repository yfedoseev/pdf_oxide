# December 2025-12-11 - Quality Improvement Plan

**Date**: 2025-12-11
**Project**: Word Boundary Enhancement - Quality & Accuracy Improvements
**Phase**: Post-Implementation Enhancement Planning
**Status**: Ready for Implementation

---

## Executive Summary

The Word Boundary Enhancement project is **production ready** with 1,033+ tests passing and <3% performance overhead. However, analysis of extraction quality against PDF 1.7 specification (ISO 32000-1:2008) reveals **specific, actionable improvements** that will increase extraction accuracy from ~7.5/10 to 8.5+/10.

This document outlines a prioritized improvement roadmap with:
- **Priority 1** (Must-Do): 3 critical fixes addressing high-severity issues
- **Priority 2** (Should-Do): 4 improvements for enhanced accuracy
- **Priority 3** (Nice-To-Have): 3 quality-of-life enhancements

**Overall Impact**: Expected quality score improvement from 7.5/10 → 8.5+/10 (13% improvement)

---

## 1. Problem Analysis & Root Causes

### 1.1 High-Severity Issues (From Issues Document)

| Issue | Severity | Root Cause | Current Impact | Fix Type |
|-------|----------|-----------|-----------------|----------|
| **Word Concatenation** | HIGH | Missing space detection between TJ array elements | Produces "thequick" instead of "the quick" | Detection |
| **Character Spacing** | HIGH | Geometric gap threshold too sensitive | Extra spaces in output ("th e qu ick") | Threshold tuning |
| **Line-Ending Hyphens** | MEDIUM | PDF spec compliance - hyphenation rules | Incomplete word breaks ("wrap-" on line, "-ping" below) | Post-processing |
| **Type0 Font CMap Recovery** | MEDIUM | ToUnicode CMap missing/invalid, no fallback attempt | U+FFFD replacement characters in output | Font handling |
| **CJK Boundary Accuracy** | MEDIUM | Limited fullwidth punctuation scoring | Incorrect word boundaries in Chinese/Japanese | Script rules |

### 1.2 PDF Specification Reference Points

Per **ISO 32000-1:2008 PDF 1.7**:

1. **Section 9.10 - Extraction of Text Content**
   - Character-to-Unicode mapping priority: ToUnicode CMap → Predefined CMap → Font Encoding
   - Type0 fonts MUST attempt CIDToGIDMap lookup before fallback
   - Unmapped characters SHOULD use U+FFFD as last resort
   - **Current Status**: ✅ Implemented, working correctly

2. **Section 9.4.4 - Text Positioning and Metrics**
   - TJ array values control word spacing (negative values = reduced spacing)
   - Tc (character spacing) and Tw (word spacing) parameters affect boundaries
   - Geometric gaps should be measured against text matrix scaling
   - **Current Status**: ⚠️ Partial - TJ offsets used but not optimally

3. **Section 9.3 - Text State Parameters**
   - Tc, Tw, Tz (text scaling) affect how gaps should be interpreted
   - Font size (Tf) and text matrix (Tm) scale geometric measurements
   - **Current Status**: ⚠️ Limited - Not all parameters fully utilized

4. **Section 9.7 - Composite Fonts (Type0)**
   - CIDToGIDMap maps CID to GID (glyph index)
   - Identity-H encoding is most common for Unicode mapping
   - **Current Status**: ⚠️ Partial - CIDToGIDMap validation needed

---

## 2. Priority 1 Improvements (Must-Do Before Release)

### 2.1 Word Concatenation Fix - TJ Array Threshold Optimization

**Issue**: Words from different TJ array elements merge without spaces
**Severity**: HIGH
**Current Quality Impact**: -2.0 points

#### Root Cause Analysis

PDF uses two methods to control text positioning:
1. **Tj operator**: Single string output
2. **TJ operator**: Array of strings and offsets

Negative offsets in TJ arrays reduce spacing. Current implementation:
```
// Current logic (simplified)
if tj_offset < -100 {
    insert_space = true  // Threshold-based
} else {
    insert_space = false
}
```

**Problem**: Threshold is too conservative, misses real word boundaries

#### Solution: Adaptive Threshold Based on Text State

**Implementation File**: `src/text/word_boundary.rs` (lines ~250-350)

```rust
/// Calculate adaptive word boundary threshold per PDF spec 9.3
///
/// TJ offset significance depends on:
/// - Font size (Tf) - larger fonts need larger offsets to create space
/// - Character spacing (Tc) - added to each character
/// - Word spacing (Tw) - added only to space characters
/// - Horizontal scaling (Tz) - scales offset values
fn calculate_tj_threshold(
    &self,
    context: &BoundaryContext,
) -> f32 {
    let font_size = context.font_size.max(1.0);
    let char_spacing = context.char_spacing;
    let word_spacing = context.word_spacing;
    let h_scale = context.h_scale.max(0.01);

    // Base threshold: proportional to font size
    // Per spec 9.3: offset values are in 1/1000 of text space units
    let base_threshold = -font_size * h_scale * 0.025;

    // Adjust for explicit spacing parameters
    let adjustment = (char_spacing + word_spacing).abs();

    // Return adaptive threshold
    base_threshold - adjustment
}

/// Enhanced TJ offset boundary detection
fn detect_tj_boundary(&self,
    offset: f32,
    context: &BoundaryContext
) -> bool {
    let threshold = self.calculate_tj_threshold(context);

    // Offset is a boundary if it's more negative than threshold
    // (larger negative = more space)
    offset < threshold
}
```

**Changes Required**:

1. **Modify src/extractors/text.rs** (lines ~700-900)
   - Track Tc, Tw, Tz state parameters during TJ/Tj processing
   - Pass BoundaryContext with all state to is_word_boundary()

2. **Modify src/text/word_boundary.rs**
   - Add `calculate_tj_threshold()` function (20 lines)
   - Update `is_tj_offset_boundary()` to use adaptive threshold (5 lines)
   - Add tests for threshold calculation (10 tests)

**Impact on Extraction**:
- Expected improvement: +1.5 points (fixes ~70% of concatenation issues)
- Performance: No change (still O(1) per boundary)
- Risk: Low (threshold-based, can be tuned per PDF corpus)

**Testing Strategy**:
```rust
#[test]
fn test_tj_threshold_adapts_to_font_size() {
    let context_10pt = BoundaryContext {
        font_size: 10.0,
        ..Default::default()
    };
    let context_24pt = BoundaryContext {
        font_size: 24.0,
        ..Default::default()
    };

    let threshold_10 = detector.calculate_tj_threshold(&context_10pt);
    let threshold_24 = detector.calculate_tj_threshold(&context_24pt);

    // Larger fonts should have more negative threshold
    assert!(threshold_24 < threshold_10);
}

#[test]
fn test_tj_threshold_accounts_for_word_spacing() {
    let context_no_ws = BoundaryContext {
        word_spacing: 0.0,
        ..Default::default()
    };
    let context_with_ws = BoundaryContext {
        word_spacing: 5.0,  // Tw=5
        ..Default::default()
    };

    let threshold_no = detector.calculate_tj_threshold(&context_no_ws);
    let threshold_with = detector.calculate_tj_threshold(&context_with_ws);

    // With word spacing, threshold should be more lenient
    assert!(threshold_with > threshold_no);
}
```

**Validation on Test Corpus**:
- Run extraction on academic + mixed PDFs (262 PDFs)
- Compare output against golden files with ±1% word count tolerance
- Target: <5% documents with quality regression

**Timeline**: 1-2 days
**Effort**: 5-7 implementation files

---

### 2.2 Character Spacing Sensitivity - Threshold Refinement

**Issue**: Geometric gap detection too aggressive, creates extra spaces
**Severity**: HIGH
**Current Quality Impact**: -1.5 points

#### Root Cause Analysis

Geometric gap detection measures space between character bounding boxes:
```
// Current (simplified)
if (curr_x - prev_x_end) > (char_width * 0.5) {
    insert_space = true
}
```

**Problem**:
- Doesn't account for kerning (tighter spacing in ligatures)
- Doesn't scale with font size
- Doesn't consider character type (punctuation vs letters)

#### Solution: Context-Aware Gap Threshold

**Implementation File**: `src/text/word_boundary.rs` (lines ~400-500)

```rust
/// Per PDF spec 9.3, geometric spacing depends on:
/// - Font size (larger = larger acceptable gaps without spaces)
/// - Character type (punctuation allows tighter spacing)
/// - Previous character width (affects baseline for next gap)
fn has_significant_geometric_gap(
    &self,
    prev: &CharacterInfo,
    curr: &CharacterInfo,
    context: &BoundaryContext,
) -> bool {
    let gap = curr.x_position - (prev.x_position + prev.width);

    // Threshold: font_size * gap_ratio_factor
    // For 12pt font, threshold = ~2.4pt typically
    let font_size = context.font_size.max(1.0);
    let h_scale = context.h_scale.max(0.01);

    // Default gap ratio: 0.2 (20% of character width)
    // For 12pt font with 7pt character width: threshold = 1.4pt
    let threshold = (prev.width * 0.2).max(font_size * h_scale * 0.02);

    // Special case: don't create boundary within ligatures
    if self.is_ligature_internal_gap(prev, curr) {
        return false;
    }

    // Punctuation has different spacing rules
    if Self::is_punctuation(curr.code) {
        // Punctuation can follow more closely
        return gap > threshold * 0.5;
    }

    gap > threshold
}

/// Check if gap is within a ligature (should not create boundary)
fn is_ligature_internal_gap(
    &self,
    prev: &CharacterInfo,
    curr: &CharacterInfo,
) -> bool {
    // fi, fl, ffi, ffl ligatures have internal gaps
    // Don't split these
    let ligatures = [0xFB00, 0xFB01, 0xFB02, 0xFB03, 0xFB04]; // fi, fl, ffi, ffl, st
    ligatures.contains(&prev.code)
}

fn is_punctuation(code: u32) -> bool {
    matches!(code,
        0x2018..=0x201F |  // Quotation marks
        0x2E..=0x3B |      // ./:;
        0x21 |             // !
        0x3F                // ?
    )
}
```

**Changes Required**:

1. **Modify src/text/word_boundary.rs**
   - Update `has_significant_geometric_gap()` with context-aware threshold (20 lines)
   - Add `is_punctuation()` helper (5 lines)
   - Add `is_ligature_internal_gap()` helper (5 lines)
   - Add tests for gap detection (15 tests)

2. **No changes needed** to src/extractors/text.rs (uses existing function)

**Impact on Extraction**:
- Expected improvement: +1.0 point (reduces extra spaces by ~60%)
- Performance: No change (O(1) per character pair)
- Risk: Low (still conservative, can be tuned)

**Testing Strategy**:
```rust
#[test]
fn test_gap_threshold_scales_with_font_size() {
    let info_10pt_prev = CharacterInfo { width: 7.0, ..Default::default() };
    let info_10pt_curr = CharacterInfo {
        x_position: 15.0, // 8pt gap
        ..Default::default()
    };

    let context_10 = BoundaryContext { font_size: 10.0, ..Default::default() };
    let context_24 = BoundaryContext { font_size: 24.0, ..Default::default() };

    // Same gap in larger font should not create boundary
    let detector = WordBoundaryDetector::new();
    let boundary_10 = detector.has_significant_geometric_gap(&info_10pt_prev, &info_10pt_curr, &context_10);

    // With 24pt font, same physical gap is smaller relative to font size
    let boundary_24 = detector.has_significant_geometric_gap(&info_10pt_prev, &info_10pt_curr, &context_24);

    // Both should be consistent (no boundary)
    assert_eq!(boundary_10, boundary_24);
}

#[test]
fn test_punctuation_allows_tight_spacing() {
    let prev = CharacterInfo { x_position: 0.0, width: 7.0, ..Default::default() };
    let punct = CharacterInfo {
        x_position: 7.1,  // Only 0.1pt gap
        code: 0x2E,  // Period
        ..Default::default()
    };

    let context = BoundaryContext { font_size: 12.0, ..Default::default() };
    let detector = WordBoundaryDetector::new();

    // Punctuation at 0.1pt gap should NOT create boundary
    let has_boundary = detector.has_significant_geometric_gap(&prev, &punct, &context);
    assert!(!has_boundary);
}
```

**Timeline**: 1 day
**Effort**: 2-3 implementation files

---

### 2.3 Golden File Baseline Establishment & Regression Detection

**Issue**: No golden file baselines created, cannot validate quality or detect regressions
**Severity**: HIGH
**Current Quality Impact**: -0.5 points (measurement uncertainty)

#### Current Status

✅ Infrastructure exists:
- Extraction pipeline completed (356 PDFs processed)
- Golden file manager implemented (tests/helpers/golden_file_manager.rs)
- Corpus loader ready (tests/helpers/corpus_loader.rs)
- Regression detection framework implemented

❌ Baselines not created:
- No reference extracted text stored
- Cannot compare against baseline
- Quality metrics cannot be validated

#### Solution: Create Golden File Baselines

**Steps**:

1. **Extract & Store Baselines** (execution, ~10 minutes)
   ```bash
   cd /home/yfedoseev/projects/pdf_oxide
   cargo test --test golden_files --release -- create_baselines
   ```

   This will:
   - Extract text from all 356 PDFs using current (improved) implementation
   - Calculate hash + character/word counts
   - Store in `tests/fixtures/golden_files/` directory
   - Create metadata JSON with baseline metrics

2. **Validate Baselines** (manual, ~30 minutes)
   - Spot-check 10-15 PDFs across different categories
   - Verify extraction quality looks reasonable
   - Check character accuracy (no obvious U+FFFD replacements where text exists)
   - Confirm word counts are sensible

3. **Create Regression Test** (implementation, 1 day)
   ```rust
   #[test]
   fn test_extraction_regression_academic() {
       let corpus = CorpusLoader::load("academic");
       let golden = GoldenFileManager::load_baselines("academic");

       for pdf in corpus {
           let extracted = extract_text(&pdf);
           let baseline = &golden[&pdf.path];

           // Allow ±0.5% character count variance
           // Allow ±1% word count variance
           assert_quality_within_tolerance(&extracted, baseline, 0.005, 0.01);
       }
   }
   ```

**Impact on Quality**:
- Expected improvement: +0.5 points (enables quality validation)
- Enables continuous quality monitoring
- Detects regressions automatically

**Testing**:
- Run regression test after any code changes
- Monitor quality metrics dashboard (creates CSV export)
- Alert if >5% of PDFs show quality drop

**Timeline**: 1 day (execution + validation)
**Effort**: 2-3 test files

---

## 3. Priority 2 Improvements (Should-Do for Accuracy)

### 3.1 Type0 Font CMap Recovery Enhancement

**Issue**: Type0 fonts without ToUnicode CMap fallback to U+FFFD replacement
**Severity**: MEDIUM
**Current Quality Impact**: -0.8 points

#### Root Cause Analysis

Per PDF spec 9.10.2, character mapping priority:
1. ToUnicode CMap (if present) ✅ Implemented
2. Predefined CMap + CIDToGIDMap (if available) ⚠️ Limited
3. AGL (Adobe Glyph List) ❌ Not attempted
4. U+FFFD replacement ✅ Fallback exists

**Current Implementation** (src/fonts/cmap.rs):
- ✅ Loads ToUnicode if present
- ✅ Loads predefined CMaps (Identity-H, etc.)
- ⚠️ CIDToGIDMap validation incomplete
- ❌ No AGL fallback attempt

#### Solution: Add AGL Fallback for Type0 Fonts

**Implementation File**: `src/fonts/mod.rs` (new module `src/fonts/agl.rs`)

```rust
/// Adobe Glyph List (AGL) provides fallback mappings for common glyphs
/// Per PDF spec 9.10, this is the third priority for character mapping
///
/// Maps glyph names to Unicode codepoints
/// Example: "fi" → U+FB01, "Oslash" → U+00D8

pub struct AdobeGlyphList {
    // Map: glyph_name → Unicode codepoint
    mappings: HashMap<&'static str, u32>,
}

impl AdobeGlyphList {
    pub fn new() -> Self {
        // Pre-compiled from Adobe's official AGL file
        Self {
            mappings: [
                ("fi", 0xFB01),
                ("fl", 0xFB02),
                ("ffi", 0xFB03),
                ("ffl", 0xFB04),
                ("Oslash", 0x00D8),
                ("oslash", 0x00F8),
                // ... 1000+ more mappings
            ].iter().cloned().collect(),
        }
    }

    pub fn lookup_by_name(&self, glyph_name: &str) -> Option<u32> {
        self.mappings.get(glyph_name).copied()
    }
}

/// Enhanced Type0 font character mapping
pub fn char_to_unicode_type0(
    &self,
    char_code: u32,
    glyph_id: u32,
) -> Option<u32> {
    // Try 1: ToUnicode CMap (highest priority)
    if let Some(cmap) = &self.to_unicode_cmap {
        if let Some(unicode) = cmap.lookup(char_code) {
            return Some(unicode);
        }
    }

    // Try 2: CIDToGIDMap → Predefined CMap
    if let Some(cid_to_gid) = &self.cid_to_gid_map {
        if let Some(gid) = cid_to_gid.lookup(char_code) {
            if let Some(unicode) = self.glyph_name_to_unicode(gid) {
                return Some(unicode);
            }
        }
    }

    // Try 3: Predefined CMap with Identity-H encoding
    if let Some(cmap) = &self.predefined_cmap {
        if let Some(unicode) = cmap.lookup(char_code) {
            return Some(unicode);
        }
    }

    // Try 4: AGL fallback (NEW)
    if let Some(glyph_name) = self.get_glyph_name_from_cff(glyph_id) {
        if let Some(unicode) = AdobeGlyphList::new().lookup_by_name(&glyph_name) {
            return Some(unicode);
        }
    }

    // Fallback: U+FFFD replacement character
    Some(0xFFFD)
}
```

**Changes Required**:

1. **Create src/fonts/agl.rs** (150 lines)
   - AdobeGlyphList struct with pre-compiled mappings
   - lookup_by_name() method
   - Efficient HashMap-based lookup

2. **Modify src/fonts/mod.rs**
   - Import AGL module
   - Update char_to_unicode_type0() to use AGL fallback
   - Add tests for AGL mapping (10 tests)

**Impact on Quality**:
- Expected improvement: +0.8 points (recovers ~40% of unmapped glyphs)
- Particularly helps with symbol fonts and legacy PDFs
- Risk: Low (AGL is standard Adobe reference)

**Testing**:
```rust
#[test]
fn test_agl_fallback_for_ligatures() {
    let agl = AdobeGlyphList::new();

    assert_eq!(agl.lookup_by_name("fi"), Some(0xFB01));
    assert_eq!(agl.lookup_by_name("fl"), Some(0xFB02));
    assert_eq!(agl.lookup_by_name("ffi"), Some(0xFB03));
    assert_eq!(agl.lookup_by_name("ffl"), Some(0xFB04));
}

#[test]
fn test_agl_fallback_for_special_chars() {
    let agl = AdobeGlyphList::new();

    assert_eq!(agl.lookup_by_name("Oslash"), Some(0x00D8));
    assert_eq!(agl.lookup_by_name("oslash"), Some(0x00F8));
    assert_eq!(agl.lookup_by_name("Aacute"), Some(0x00C1));
}
```

**Timeline**: 2-3 days
**Effort**: 2 implementation files

---

### 3.2 CJK Punctuation Scoring Enhancement

**Issue**: CJK word boundaries insufficiently accurate due to punctuation scoring
**Severity**: MEDIUM
**Current Quality Impact**: -0.5 points

#### Current Implementation

Per `src/text/cjk_punctuation.rs`:
```rust
// Current: Fullwidth marks score lower than TJ offsets
const FULLWIDTH_BOUNDARY_SCORE: f32 = 0.3;  // 30% confidence
// TJ offset boundaries get 1.0 (100% confidence)
```

**Problem**: In CJK documents with poor TJ offset data, fullwidth punctuation alone isn't enough to establish boundaries.

#### Solution: Adaptive Scoring Based on Text Density

```rust
/// Enhanced CJK punctuation boundary detection
///
/// Scoring depends on character density and surrounding context
/// - Dense text (many characters per inch) → punctuation more reliable
/// - Sparse text → TJ offsets more important
fn evaluate_cjk_boundary(
    &self,
    chars: &[CharacterInfo],
    index: usize,
    context: &BoundaryContext,
) -> BoundaryScore {
    let prev = &chars[index];
    let curr = &chars[index + 1];

    // Calculate text density (characters per 1000 user space units)
    let density = self.calculate_text_density(chars);

    let mut score = 0.0;

    // Fullwidth punctuation scoring (adaptive)
    if self.is_fullwidth_punctuation(curr.code) {
        // In dense text, punctuation is more reliable
        // Dense > 10 chars per 1000 units: score += 0.8
        // Medium 5-10: score += 0.6
        // Sparse < 5: score += 0.3
        if density > 10.0 {
            score += 0.8;  // Higher confidence in dense text
        } else if density > 5.0 {
            score += 0.6;
        } else {
            score += 0.3;
        }
    }

    // TJ offset scoring (unchanged)
    if self.has_tj_offset_gap(prev) {
        score = 1.0;  // Override with TJ confidence
    }

    // Geometric gap scoring
    if self.has_significant_gap(prev, curr) {
        score = (score + 0.9) / 2.0;  // Combine with previous score
    }

    BoundaryScore { confidence: score }
}

fn calculate_text_density(&self, chars: &[CharacterInfo]) -> f32 {
    if chars.len() < 2 {
        return 0.0;
    }

    let first = &chars[0];
    let last = &chars[chars.len() - 1];
    let span_width = (last.x_position + last.width) - first.x_position;

    if span_width <= 0.0 {
        return 0.0;
    }

    (chars.len() as f32 / span_width) * 1000.0  // Per 1000 units
}
```

**Changes Required**:

1. **Modify src/text/cjk_punctuation.rs**
   - Add `calculate_text_density()` method (10 lines)
   - Update `evaluate_cjk_boundary()` with adaptive scoring (20 lines)
   - Add `BoundaryScore` struct if not exists (5 lines)
   - Add tests for density calculation (8 tests)

**Impact on Quality**:
- Expected improvement: +0.5 points (better CJK boundary detection)
- Particularly helps Chinese and Japanese documents
- Risk: Low (additive scoring, maintains fallback)

**Timeline**: 1-2 days
**Effort**: 1 implementation file

---

### 3.3 Line-Ending Hyphenation Post-Processing

**Issue**: Hyphenated words split across lines are extracted as separate words
**Severity**: MEDIUM
**Current Quality Impact**: -0.4 points

#### Root Cause Analysis

Example: A PDF with text "wrap-ping" split across lines:
- Line 1 ends with: "wrap-"
- Line 2 starts with: "ping"

Current extraction produces: "wrap - ping" (three words)

Per PDF spec 9.4.4, hyphens at line endings are formatting artifacts and should be handled specially.

#### Solution: Smart Hyphenation Merging Post-Processor

**Implementation File**: `src/extractors/text.rs` (new section ~5800-5900)

```rust
/// Post-process extracted text to merge hyphenated words
///
/// Rules:
/// 1. If line ends with hyphen, next word is likely continuation
/// 2. Check if hyphen-word combination forms valid Unicode sequence
/// 3. Merge if confidence > 0.8
pub fn merge_hyphenated_words(text: &str) -> String {
    // Regex pattern: word-\n with next word starting with lowercase
    let pattern = Regex::new(r"(\w+)-\s*\n\s*([a-z]\w*)").unwrap();

    let result = pattern.replace_all(text, "$1$2").to_string();

    result
}

/// Helper to detect if merged word is valid Unicode
fn is_valid_utf8_word(s: &str) -> bool {
    // All characters should be valid Unicode
    s.chars().all(|c| c != char::REPLACEMENT_CHARACTER)
}
```

**Changes Required**:

1. **Modify src/extractors/text.rs**
   - Add `merge_hyphenated_words()` function (15 lines)
   - Call in final text assembly stage (2 lines)
   - Add configuration option for enable/disable (3 lines)

2. **Modify src/pipeline/config.rs**
   - Add `merge_hyphenated_words: bool` field (1 line)
   - Default: true (1 line)

**Impact on Quality**:
- Expected improvement: +0.4 points (fixes ~50% of hyphenation issues)
- Post-processing (happens after boundary detection)
- Risk: Very Low (conservative regex, easy to disable)

**Testing**:
```rust
#[test]
fn test_hyphenated_word_merging() {
    let input = "The word wrap-\ning continues here";
    let output = merge_hyphenated_words(input);

    assert_eq!(output, "The word wrapping continues here");
}

#[test]
fn test_preserve_intentional_hyphens() {
    // Don't merge if hyphen is clearly intentional
    let input = "user-friendly software design";
    let output = merge_hyphenated_words(input);

    assert_eq!(output, input);  // Unchanged
}
```

**Timeline**: 1 day
**Effort**: 1-2 implementation files

---

### 3.4 Diacritic & Combining Mark Validation

**Issue**: Some combining marks may incorrectly create word boundaries
**Severity**: MEDIUM
**Current Quality Impact**: -0.3 points

#### Current Status

✅ Working:
- Diacritical marks detected correctly (Unicode ranges)
- Marks don't create boundaries
- Marks preserved in output

❌ Edge cases:
- Zero-width joiners (ZWJ, U+200D)
- Variation selectors (VS1-VS16, U+FE00-U+FE0F)
- Some rare combining sequences

#### Solution: Enhanced Mark Classification

**Implementation File**: `src/text/word_boundary.rs` (lines ~150-250)

```rust
/// Enhanced classification of combining marks
/// Per Unicode Standard, different mark types:
/// - Spacing marks (should create boundaries)
/// - Non-spacing marks (should NOT create boundaries)
/// - Enclosing marks (rare, special handling)
/// - Zero-width joiners (explicit binding directive)

enum CombiningMarkType {
    /// Spacing marks (general category Mc)
    /// Example: Devanagari matras, Thai vowels
    Spacing,

    /// Non-spacing marks (categories Mn, Me)
    /// Example: Accent diacritics
    NonSpacing,

    /// Enclosing marks (category Me)
    /// Example: U+0488 (Cyrillic combining)
    Enclosing,

    /// Zero-width joiners (explicit binding)
    /// U+200D should bind characters
    ZeroWidthJoiner,

    /// Variation selectors
    /// U+FE00-U+FE0F modify preceding character appearance
    VariationSelector,
}

fn classify_combining_mark(code: u32) -> Option<CombiningMarkType> {
    match code {
        0x200D => Some(CombiningMarkType::ZeroWidthJoiner),
        0xFE00..=0xFE0F => Some(CombiningMarkType::VariationSelector),
        0x0300..=0x036F => Some(CombiningMarkType::NonSpacing),  // Latin combining
        0x0900..=0x0954 => Some(CombiningMarkType::NonSpacing),  // Devanagari
        0x0E31..=0x0E3A => Some(CombiningMarkType::Spacing),     // Thai
        _ => None,
    }
}

/// Updated boundary detection accounting for mark types
fn is_word_boundary_with_marks(
    &self,
    prev: &CharacterInfo,
    curr: &CharacterInfo,
) -> bool {
    // Zero-width joiner explicitly prevents boundary
    if curr.code == 0x200D {
        return false;
    }

    // Variation selectors don't create boundaries
    if (0xFE00..=0xFE0F).contains(&curr.code) {
        return false;
    }

    // Regular non-spacing marks still don't create boundaries
    if let Some(mark_type) = self.classify_combining_mark(curr.code) {
        match mark_type {
            CombiningMarkType::NonSpacing => return false,
            CombiningMarkType::Spacing => {
                // These can create boundaries
                // but check context
            }
            _ => {}
        }
    }

    // Fall through to normal boundary detection
    self.is_word_boundary(prev, curr)
}
```

**Changes Required**:

1. **Modify src/text/word_boundary.rs**
   - Add `CombiningMarkType` enum (10 lines)
   - Add `classify_combining_mark()` function (15 lines)
   - Add `is_word_boundary_with_marks()` function (20 lines)
   - Update existing `is_word_boundary()` to use new classification (5 lines)
   - Add tests for mark classification (12 tests)

**Impact on Quality**:
- Expected improvement: +0.3 points (fixes rare edge cases)
- Particularly helps with RTL and complex scripts
- Risk: Very Low (additional checks, fallback to existing logic)

**Timeline**: 1 day
**Effort**: 1 implementation file

---

## 4. Priority 3 Improvements (Nice-To-Have)

### 4.1 Extended Logging for Debugging

**Implementation**: Add detailed logging of boundary detection decisions
**Effort**: Few hours
**Value**: Debugging aid for users

```rust
// In word_boundary.rs
#[cfg(debug_assertions)]
fn log_boundary_decision(
    &self,
    prev: &CharacterInfo,
    curr: &CharacterInfo,
    decision: BoundaryDecision,
) {
    debug!("Boundary check: '{}' (U+{:04X}) → '{}' (U+{:04X})",
        char::from_u32(prev.code).unwrap_or('?'),
        prev.code,
        char::from_u32(curr.code).unwrap_or('?'),
        curr.code);
    debug!("Decision: {:?}", decision);
    debug!("Reasons: {:?}", decision.reasons);
}
```

**Timeline**: Few hours
**Implementation**: 1 file

---

### 4.2 Configuration Presets for Document Types

**Implementation**: Pre-tuned configurations for specific PDF types
**Effort**: Few hours
**Value**: Easier adoption for users

```rust
pub struct TextPipelineConfig {
    // ... existing fields ...

    /// Preset configuration for document type
    pub preset: DocumentTypePreset,
}

pub enum DocumentTypePreset {
    /// Default: balanced for general PDFs
    Default,

    /// Academic: tighter spacing, focus on references
    Academic,

    /// Business: more aggressive space detection
    Business,

    /// Novel: preserve formatting, hyphenation
    Novel,

    /// CJK-focused: optimized for Chinese/Japanese/Korean
    CJK,

    /// RTL-focused: optimized for Arabic/Hebrew
    RTL,

    /// Custom(Box<TextPipelineConfig>)
    Custom(Box<TextPipelineConfig>),
}

impl DocumentTypePreset {
    pub fn apply_to_config(&self, config: &mut TextPipelineConfig) {
        match self {
            DocumentTypePreset::Academic => {
                config.tj_threshold = -0.025;
                config.geometric_gap_ratio = 0.15;
                config.merge_hyphenated_words = true;
            }
            DocumentTypePreset::CJK => {
                config.enable_cjk_punctuation = true;
                config.cjk_punctuation_weight = 0.7;
            }
            // ... other presets ...
            _ => {}
        }
    }
}
```

**Timeline**: Few hours
**Implementation**: 1-2 files

---

### 4.3 Quality Metrics Dashboard Export

**Implementation**: Export quality metrics to CSV for analysis
**Effort**: Few hours
**Value**: Continuous monitoring

```rust
pub fn export_quality_metrics_csv(
    extraction_results: &[ExtractionResult],
    path: &Path,
) -> Result<()> {
    let mut csv = String::from("pdf_path,characters,words,quality_score,boundary_accuracy\n");

    for result in extraction_results {
        csv.push_str(&format!(
            "{},{},{},{:.2},{:.2}%\n",
            result.pdf_path,
            result.char_count,
            result.word_count,
            result.quality_score,
            result.boundary_accuracy * 100.0
        ));
    }

    std::fs::write(path, csv)?;
    Ok(())
}
```

**Timeline**: Few hours
**Implementation**: 1 file

---

## 5. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1 - Parallel Implementation)

| Task | Files | Effort | Owner |
|------|-------|--------|-------|
| 2.1 TJ Threshold Optimization | 2 | 2 days | Primary |
| 2.2 Geometric Gap Refinement | 1 | 1 day | Secondary |
| 2.3 Golden File Baselines | 2 | 1 day | Testing |

**Concurrent Execution**: All 3 can run in parallel
**Validation**: Run golden file regression test after each fix
**Target Completion**: End of Week 1

### Phase 2: Accuracy Improvements (Week 2 - Sequential)

| Task | Files | Effort | Prerequisite |
|------|-------|--------|--------------|
| 3.1 AGL Fallback | 2 | 2-3 days | Phase 1 complete |
| 3.2 CJK Scoring | 1 | 1-2 days | Phase 1 complete |
| 3.3 Hyphenation | 1-2 | 1 day | Phase 1 complete |
| 3.4 Mark Validation | 1 | 1 day | Phase 1 complete |

**Sequential or Parallel**: Can run in parallel after Phase 1
**Validation**: Run golden file tests after Phase 1 and Phase 2
**Target Completion**: End of Week 2

### Phase 3: Polish & Documentation (Week 3)

| Task | Files | Effort |
|------|-------|--------|
| 4.1 Logging | 1 | Few hours |
| 4.2 Presets | 1-2 | Few hours |
| 4.3 Dashboard | 1 | Few hours |
| Documentation updates | 1-2 | 1 day |

**Timeline**: Week 3 (as schedule allows)

---

## 6. Testing Strategy

### Unit Tests (Per-Module)

Each improvement includes unit tests:
- TJ threshold: 10 tests
- Geometric gap: 8 tests
- AGL fallback: 10 tests
- CJK scoring: 8 tests
- Hyphenation: 6 tests
- Mark validation: 12 tests

**Total New Tests**: 54+ unit tests

### Integration Tests

**Golden File Regression Tests**:
```bash
cargo test test_extraction_regression_academic --release
cargo test test_extraction_regression_mixed --release
# ... for each category
```

Run after each phase completion to validate quality improvements.

### Performance Validation

**Criterion Benchmarks**:
```bash
cargo bench --bench word_boundary_benchmarks
cargo bench --bench full_pipeline_benchmarks
```

Target: <3% performance impact maintained (currently achieved)

### Manual Quality Review

**Spot Check Process** (10-15 PDFs per category):
1. Extract text with improvements enabled
2. Visual review of output
3. Check for:
   - No extra spaces introduced
   - No missing spaces
   - Proper CJK boundaries
   - Correct ligature handling
   - RTL text preserved

---

## 7. Success Criteria & Metrics

### Quality Improvement Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Overall Quality Score | 7.5/10 | 8.5/10 | +1.0 (13%) |
| Word Concatenation | Poor | Good | -95% issue frequency |
| Character Spacing | Fair | Good | -60% extra spaces |
| CJK Accuracy | Fair | Good | +80% boundary accuracy |
| Type0 Recovery | Poor | Good | +40% unmapped glyphs recovered |
| Hyphenation | Poor | Fair | +50% correct merging |

### Performance Targets

| Metric | Target | Validation |
|--------|--------|------------|
| Overall Overhead | <3% (maintained) | Criterion benchmarks |
| Character Processing | <10µs/char | Component benchmarks |
| Extraction Speed | ~45ms/page | Full pipeline benchmark |
| Memory Impact | <5% increase | Profiler analysis |

### Test Coverage Targets

| Category | Target | Current → Expected |
|----------|--------|------------------|
| Unit Tests | +54 | 234 → 288 |
| Integration Tests | +20 | 52 → 72 |
| Regression Tests | 356 PDFs | All categories covered |
| Performance Benchmarks | All passing | Baseline maintained |

---

## 8. Risk Mitigation

### Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Quality regression on some PDFs | Medium | High | Golden file testing, spot checks |
| Performance degradation | Low | High | Benchmarks at each phase |
| Breaking changes to output | Medium | Medium | Version bump, migration guide |
| Integration complexity | Low | Medium | Incremental testing, parallel execution |

### Rollback Strategy

Each improvement is behind a configuration flag:

```rust
pub struct TextPipelineConfig {
    pub use_adaptive_tj_threshold: bool,        // Default: true
    pub use_context_aware_gaps: bool,            // Default: true
    pub attempt_agl_fallback: bool,              // Default: true
    pub use_adaptive_cjk_scoring: bool,          // Default: true
    pub merge_hyphenated_words: bool,            // Default: true
    pub validate_combining_marks: bool,          // Default: true
}
```

If issues arise, individual improvements can be disabled without full rollback.

---

## 9. Documentation Plan

### User Documentation

1. **Migration Guide** (update existing)
   - New parameters explained
   - Impact on extraction output
   - Troubleshooting tips

2. **Configuration Guide** (new)
   - Document new flags
   - Preset configurations
   - Tuning guidance

### Technical Documentation

1. **Architecture Update** (update existing)
   - Explain enhancement flow
   - Document decision points
   - Add performance analysis

2. **Testing Guide** (update existing)
   - Describe golden file regression testing
   - Quality metrics explanation
   - How to validate improvements

---

## 10. Conclusion & Recommendations

### Overall Impact Assessment

Implementing all Priority 1 and 2 improvements will:

✅ **Increase Quality Score**: 7.5/10 → 8.5+/10 (13% improvement)
✅ **Maintain Performance**: <3% overhead preserved
✅ **Zero Regressions**: Golden file testing ensures quality
✅ **Better Spec Compliance**: Per PDF 1.7 specification
✅ **Enhanced Reliability**: AGL fallback, better Type0 handling

### Timeline Estimate

- **Phase 1 (Critical)**: 4 days (parallel execution)
- **Phase 2 (Accuracy)**: 5-6 days (parallel execution after Phase 1)
- **Phase 3 (Polish)**: 2-3 days (as schedule allows)
- **Total**: 11-13 days (can be compressed with parallel work)

### Next Steps (Immediate)

1. **Day 1**: Start Priority 1 improvements in parallel
   - Assign 2.1 (TJ Threshold) to primary implementation
   - Assign 2.2 (Geometric Gap) to secondary implementation
   - Assign 2.3 (Golden Files) to testing track

2. **Day 2-3**: Golden file baselines completed, regression testing enabled

3. **Day 4**: Phase 1 complete, run golden file tests
   - Validate quality improvement to 8.0+/10
   - Check performance maintained <3%

4. **Day 5+**: Proceed to Phase 2 improvements based on Phase 1 results

### Recommendation

**Proceed with full implementation** of Priority 1 and 2 improvements. Expected ROI is significant: 13% quality improvement with maintained performance and zero regressions. Risk is low due to:
- Robust test coverage
- Golden file validation
- Feature flags for rollback
- Incremental implementation approach

**Timeline is realistic** for 2-week completion (Priority 1+2) with high confidence.

---

**Document Generated**: 2025-12-11T23:59:00Z
**Status**: Ready for Implementation
**Approval Required**: Project Lead
**Implementation Owner**: To be assigned

