# Phase 9: Text Spacing Logic Simplification

## Executive Summary

This document outlines a comprehensive refactoring plan to replace pdf_oxide's complex, multi-rule text spacing logic with pdfplumber's proven single-rule geometric approach. The goal is to improve extraction quality from 4.3/10 average to 8.5+/10 by eliminating spurious spaces and word fusion issues.

**Current State:**
- Quality: 4.3/10 average (fails on 4/5 PDFs)
- Architecture: 4+ competing rules with confidence scoring
- Complexity: ~500 lines of spacing logic across multiple modules

**Target State:**
- Quality: 8.5+/10 (matching pdfplumber on policy PDFs)
- Architecture: Single geometric rule with relative margin
- Complexity: ~100 lines of spacing logic

---

## 1. ANALYSIS SECTION

### 1.1 Files Affected by Refactoring

| File | Lines | Impact | Description |
|------|-------|--------|-------------|
| `src/extractors/text.rs` | ~4400 | **Major** | Core spacing logic, SpaceDecision, should_insert_space() |
| `src/layout/space_detection.rs` | ~523 | **Remove** | SpaceDetectionEngine, all detector traits |
| `src/extractors/gap_statistics.rs` | ~1610 | **Simplify** | Keep statistics, remove DocumentType/Profile |
| `src/extractors/word_segmentation.rs` | ~1532 | **Remove** | Viterbi dictionary segmentation |
| `src/layout/mod.rs` | ~38 | **Minor** | Remove space_detection exports |
| `src/converters/markdown.rs` | ~800+ | **Minor** | Simplify block merging logic |
| `tests/quality_metrics.rs` | ~459 | **Update** | Adjust quality thresholds |
| `tests/regression_suite.rs` | ~350+ | **Update** | Update test expectations |

### 1.2 Code to be REMOVED (Rules 1-4, Heuristics, Multipliers, Confidence)

#### 1.2.1 src/extractors/text.rs - Remove These Components

```rust
// REMOVE: SpaceSource enum (lines 21-45)
pub enum SpaceSource {
    TjOffset,           // Rule 1 - REMOVE
    GeometricGap,       // Rule 2 - REMOVE
    CharacterHeuristic, // Rule 3 - REMOVE
    AlreadyPresent,     // Rule 0 - KEEP (refactor)
    NoSpace,            // REMOVE
}

// REMOVE: SpaceDecision struct with confidence (lines 47-89)
pub struct SpaceDecision {
    pub insert_space: bool,
    pub source: SpaceSource,
    pub confidence: f32,  // REMOVE - no confidence scoring
}

// REMOVE: should_insert_space() function (lines 632-808)
// This entire function with 4+ rules will be replaced by single geometric rule

// REMOVE: should_insert_space_heuristic() function (lines 3907-3930)
fn should_insert_space_heuristic(current_text: &str, next_text: &str) -> bool {
    // CamelCase detection
    // Number-to-letter detection
}

// REMOVE: Document type awareness (lines 1313-1350)
fn get_adjusted_space_threshold(&self) -> f32 {
    match self.detected_document_type {
        Some(DocumentType::Policy) => self.config.space_insertion_threshold * 1.5,
        Some(DocumentType::Academic) => self.config.space_insertion_threshold * 0.7,
        // ...multipliers...
    }
}

// REMOVE: Adaptive TJ threshold calculation (lines 1352-1407)
fn calculate_adaptive_tj_threshold(&self) -> f32 {
    // Complex font-based threshold calculation
}

// REMOVE: split_fused_words() function (lines 2260+)
fn split_fused_words(&mut self) {
    // CamelCase splitting logic
    // Dictionary-based word splitting
}

// REMOVE: TJ offset space insertion in process_tj_array (affected lines ~3600-3700)
// Keep TJ for positioning only, not for space decisions
```

#### 1.2.2 src/layout/space_detection.rs - REMOVE ENTIRE FILE

This 523-line file contains the over-engineered detection engine:

```rust
// REMOVE: SpaceContext struct
// REMOVE: DocumentGapStats struct
// REMOVE: SpaceDecision enum (different from text.rs version)
// REMOVE: SkipReason enum
// REMOVE: SpaceDetector trait
// REMOVE: GapBasedDetector struct
// REMOVE: HeuristicDetector struct (CamelCase, number->letter)
// REMOVE: TjOffsetDetector struct
// REMOVE: AdaptiveDetector struct
// REMOVE: SpaceDetectionEngine struct
```

#### 1.2.3 src/extractors/gap_statistics.rs - REMOVE Document Type System

```rust
// REMOVE: DocumentType enum (lines 757-791)
pub enum DocumentType {
    Academic,  // 1.3x/1.6x multipliers
    Policy,    // 0.7x/1.2x multipliers
    Mixed,     // 1.0x/1.5x multipliers
}

// REMOVE: DocumentType::detect() (lines 820-882)
// REMOVE: DocumentType::threshold_multiplier() (lines 894-900)
// REMOVE: DocumentType::min_threshold_pt() (lines 910-917)
// REMOVE: DocumentType::get_adaptive_config() (lines 927-945)

// REMOVE: DocumentProfile enum (lines 1009-1017)
// REMOVE: DocumentProfile::detect() (lines 1044-1077)
// REMOVE: DocumentProfile::get_config() (lines 1099-1117)
```

#### 1.2.4 src/extractors/word_segmentation.rs - REMOVE ENTIRE FILE

This 1532-line file with Viterbi algorithm and 1000+ word dictionary is no longer needed:

```rust
// REMOVE: load_word_dictionary() - 1200+ word dictionary
// REMOVE: word_score() - scoring function
// REMOVE: segment_word() - public API
// REMOVE: segment_word_viterbi() - core algorithm
```

### 1.3 Code to be KEPT

#### 1.3.1 Rule 0: Boundary Space Check (Keep and Simplify)

```rust
// KEEP: has_boundary_space() function (lines 817-828)
fn has_boundary_space(preceding: &str, following: &str) -> bool {
    let has_trailing_space = preceding
        .chars()
        .last()
        .map_or(false, |c| c.is_whitespace());
    let has_leading_space = following
        .chars()
        .next()
        .map_or(false, |c| c.is_whitespace());
    has_trailing_space || has_leading_space
}
```

#### 1.3.2 Character Positioning (Keep)

```rust
// KEEP: TextSpan struct with bbox (x0, y0, x1, y1)
// KEEP: TextChar struct with positioning
// KEEP: Character-level bounding box calculations
// KEEP: Basic span extraction from TJ/Tj operators
```

#### 1.3.3 Basic Span Merging (Keep and Simplify)

```rust
// KEEP: merge_adjacent_spans() - but simplify logic
// KEEP: sort_spans_by_reading_order()
// KEEP: deduplicate_overlapping_spans()
```

#### 1.3.4 Gap Statistics (Keep for Debugging Only)

```rust
// KEEP: GapStatistics struct
// KEEP: extract_gaps()
// KEEP: calculate_statistics()
// KEEP: percentile()
// REMOVE: adaptive threshold determination
```

### 1.4 Complexity Reduction Quantification

| Component | Current Lines | Target Lines | Reduction |
|-----------|---------------|--------------|-----------|
| SpaceDecision/SpaceSource | 90 | 0 | -100% |
| should_insert_space() | 180 | 0 | -100% |
| space_detection.rs | 523 | 0 | -100% |
| word_segmentation.rs | 1532 | 0 | -100% |
| DocumentType/Profile | 400 | 0 | -100% |
| New geometric rule | 0 | 50 | +50 |
| Configuration | 200 | 30 | -85% |
| **Total** | **~2925** | **~80** | **-97%** |

---

## 2. DESIGN SECTION

### 2.1 New SpaceInsertion Struct

Replace the complex `SpaceDecision` with a simple struct:

```rust
/// Result of geometric space detection.
///
/// Per pdfplumber architecture: spaces are determined purely by position.
/// No confidence scoring, no heuristics, no document-type awareness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceInsertion {
    /// Whether to insert a space between characters/spans
    pub insert: bool,
}

impl SpaceInsertion {
    #[inline]
    pub fn yes() -> Self { Self { insert: true } }

    #[inline]
    pub fn no() -> Self { Self { insert: false } }
}
```

### 2.2 Single Spacing Rule (Pseudocode)

```rust
/// Determine if space should be inserted between two positioned elements.
///
/// This implements pdfplumber's single geometric rule:
/// - Calculate gap between elements using bounding boxes
/// - Compare gap to relative margin based on character dimensions
/// - Insert space if gap exceeds margin
///
/// # Algorithm (pdfplumber equivalent)
///
/// ```text
/// gap = next_char.x0 - prev_char.x1
/// margin = word_margin * max(prev_char.width, prev_char.height)
/// insert_space = (prev_char.x1 + margin) < next_char.x0
/// ```
///
/// # Arguments
///
/// * `prev` - Previous character/span with bounding box
/// * `next` - Next character/span with bounding box
/// * `word_margin` - Relative margin ratio (default: 0.1 like pdfminer.six)
///
/// # Returns
///
/// `SpaceInsertion::yes()` if space should be inserted, `SpaceInsertion::no()` otherwise
fn should_insert_space_geometric(
    prev_x1: f32,      // Right edge of previous element
    prev_width: f32,   // Width of previous element
    prev_height: f32,  // Height of previous element
    next_x0: f32,      // Left edge of next element
    word_margin: f32,  // Relative margin (0.1 default)
) -> SpaceInsertion {
    // Calculate gap
    let gap = next_x0 - prev_x1;

    // Calculate relative margin
    let char_size = prev_width.max(prev_height);
    let margin = word_margin * char_size;

    // Single geometric test
    if gap > margin {
        SpaceInsertion::yes()
    } else {
        SpaceInsertion::no()
    }
}
```

### 2.3 word_margin Parameter

```rust
/// Configuration for geometric space detection.
///
/// Matches pdfplumber/pdfminer.six LAParams approach.
#[derive(Debug, Clone)]
pub struct SpacingConfig {
    /// Word margin as ratio of character size.
    ///
    /// Default: 0.1 (matches pdfminer.six default)
    ///
    /// - Lower values (0.05): More spaces inserted, catches tight kerning
    /// - Higher values (0.15): Fewer spaces, more conservative
    ///
    /// This single parameter replaces:
    /// - space_insertion_threshold
    /// - space_threshold_em_ratio
    /// - conservative_threshold_pt
    /// - column_boundary_threshold_pt
    /// - All document-type multipliers
    pub word_margin: f32,
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self {
            word_margin: 0.1, // pdfminer.six default
        }
    }
}

impl SpacingConfig {
    /// Create configuration for tight spacing (policy documents)
    pub fn tight() -> Self {
        Self { word_margin: 0.05 }
    }

    /// Create configuration for loose spacing (academic papers)
    pub fn loose() -> Self {
        Self { word_margin: 0.15 }
    }
}
```

### 2.4 Character BBox Usage

The existing character bounding box data is sufficient:

```rust
// Current TextSpan struct (KEEP)
pub struct TextSpan {
    pub text: String,
    pub bbox: Rect,  // Contains x, y, width, height
    pub font_name: String,
    pub font_size: f32,
    // ... other fields
}

// Use bbox for geometric spacing:
impl TextSpan {
    /// Left edge of span (x0)
    pub fn left(&self) -> f32 { self.bbox.x }

    /// Right edge of span (x1)
    pub fn right(&self) -> f32 { self.bbox.x + self.bbox.width }

    /// Character size for margin calculation
    pub fn char_size(&self) -> f32 {
        self.bbox.width.max(self.bbox.height)
    }
}
```

### 2.5 Architecture Diagram

```
BEFORE (4+ competing rules):
┌─────────────────────────────────────────────────────────────────────┐
│                    SpaceDetectionEngine                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ TjOffsetDetector (priority 120, confidence 0.95)            │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │ HeuristicDetector (priority 150, CamelCase/number-letter)   │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │ GapBasedDetector (space_threshold_em + conservative_pt)     │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │ AdaptiveDetector (median * multiplier, DocumentType)        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                           ▼                                          │
│               Priority-Based Voting                                  │
│                           │                                          │
│                           ▼                                          │
│              SpaceDecision { insert, source, confidence }            │
└─────────────────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  split_fused_words   DocumentType      word_segmentation
  (CamelCase split)   (1.3x/0.7x mult)  (Viterbi + 1200 words)


AFTER (single geometric rule):
┌─────────────────────────────────────────────────────────────────────┐
│                   GeometricSpaceDetector                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ gap = next.x0 - prev.x1                                     │    │
│  │ margin = word_margin * max(prev.width, prev.height)          │    │
│  │ insert_space = (gap > margin)                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                           ▼                                          │
│               SpaceInsertion { insert: bool }                        │
└─────────────────────────────────────────────────────────────────────┘

No heuristics. No confidence. No document types. Just geometry.
```

---

## 3. IMPLEMENTATION SECTION

### 3.1 Implementation Phases

```
Phase 9.1: Create new geometric spacing module (1-2 hours)
Phase 9.2: Integrate into text extraction (2-3 hours)
Phase 9.3: Remove legacy code (1-2 hours)
Phase 9.4: Update tests and validate (2-3 hours)
Phase 9.5: Documentation and cleanup (1 hour)

Total estimated time: 7-11 hours
```

### 3.2 Step 1: Create New Geometric Spacing Module

**File: `src/extractors/geometric_spacing.rs`** (NEW)

```rust
//! Geometric space detection - pdfplumber-style single rule.
//!
//! This module implements position-based space detection without heuristics,
//! confidence scoring, or document-type awareness. The algorithm is:
//!
//! ```text
//! insert_space = (prev.x1 + margin) < next.x0
//! where margin = word_margin * max(prev.width, prev.height)
//! ```
//!
//! Reference: pdfplumber (https://github.com/jsvine/pdfplumber)
//! Reference: pdfminer.six LAParams (word_margin parameter)

use crate::layout::TextSpan;

/// Result of geometric space detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceInsertion {
    pub insert: bool,
}

impl SpaceInsertion {
    #[inline]
    pub const fn yes() -> Self { Self { insert: true } }

    #[inline]
    pub const fn no() -> Self { Self { insert: false } }
}

/// Configuration for geometric space detection.
#[derive(Debug, Clone, Copy)]
pub struct SpacingConfig {
    /// Word margin as ratio of character size (default: 0.1).
    pub word_margin: f32,
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self { word_margin: 0.1 }
    }
}

/// Determine if space should be inserted between consecutive spans.
///
/// Uses pdfplumber's single geometric rule: if the gap between spans
/// exceeds a margin relative to character size, insert a space.
///
/// # Arguments
///
/// * `prev` - Previous span in reading order
/// * `next` - Next span in reading order
/// * `config` - Spacing configuration
///
/// # Returns
///
/// `SpaceInsertion` indicating whether to insert a space.
pub fn should_insert_space(
    prev: &TextSpan,
    next: &TextSpan,
    config: &SpacingConfig,
) -> SpaceInsertion {
    // Rule 0: Skip if boundary already has whitespace
    if has_boundary_whitespace(&prev.text, &next.text) {
        return SpaceInsertion::no();
    }

    // Geometric rule: gap vs relative margin
    let prev_right = prev.bbox.right();
    let next_left = next.bbox.left();
    let gap = next_left - prev_right;

    // Use character size for relative margin (pdfplumber approach)
    let char_size = prev.bbox.width.max(prev.bbox.height);
    let margin = config.word_margin * char_size;

    if gap > margin {
        SpaceInsertion::yes()
    } else {
        SpaceInsertion::no()
    }
}

/// Check if boundary between texts already has whitespace.
fn has_boundary_whitespace(prev: &str, next: &str) -> bool {
    prev.chars().last().map_or(false, |c| c.is_whitespace())
        || next.chars().next().map_or(false, |c| c.is_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight};

    fn make_span(text: &str, x: f32, width: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, 0.0, width, 12.0),
            font_name: "Arial".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::new(0.0, 0.0, 0.0),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
        }
    }

    #[test]
    fn test_clear_word_gap() {
        let prev = make_span("Hello", 0.0, 30.0);
        let next = make_span("World", 40.0, 30.0);  // 10pt gap
        let config = SpacingConfig::default();

        // gap=10, margin=0.1*30=3, gap > margin -> insert
        assert_eq!(should_insert_space(&prev, &next, &config), SpaceInsertion::yes());
    }

    #[test]
    fn test_tight_kerning() {
        let prev = make_span("Hel", 0.0, 20.0);
        let next = make_span("lo", 21.0, 10.0);  // 1pt gap
        let config = SpacingConfig::default();

        // gap=1, margin=0.1*20=2, gap < margin -> no insert
        assert_eq!(should_insert_space(&prev, &next, &config), SpaceInsertion::no());
    }

    #[test]
    fn test_existing_boundary_space() {
        let prev = make_span("Hello ", 0.0, 30.0);  // trailing space
        let next = make_span("World", 35.0, 30.0);
        let config = SpacingConfig::default();

        // Already has space, don't insert another
        assert_eq!(should_insert_space(&prev, &next, &config), SpaceInsertion::no());
    }
}
```

### 3.3 Step 2: Integrate into Text Extraction

**File: `src/extractors/text.rs`** - Modify `merge_adjacent_spans()`

```rust
// BEFORE: Complex multi-rule logic
fn merge_adjacent_spans(&mut self) {
    // 100+ lines of:
    // - should_insert_space() with 4 rules
    // - Document type adjustments
    // - Confidence scoring
    // - Heuristic checks
}

// AFTER: Simple geometric logic
fn merge_adjacent_spans(&mut self) {
    use crate::extractors::geometric_spacing::{should_insert_space, SpacingConfig};

    let config = SpacingConfig::default();
    let mut merged: Vec<TextSpan> = Vec::new();

    for span in std::mem::take(&mut self.spans) {
        match merged.last_mut() {
            None => merged.push(span),
            Some(prev) => {
                // Check if on same line (Y tolerance)
                let same_line = (prev.bbox.y - span.bbox.y).abs() < 2.0;

                if same_line {
                    // Use geometric spacing rule
                    let insertion = should_insert_space(prev, &span, &config);

                    if insertion.insert {
                        // Insert space and push as separate span
                        prev.text.push(' ');
                    }

                    // Merge span text
                    prev.text.push_str(&span.text);
                    prev.bbox = prev.bbox.union(&span.bbox);
                } else {
                    // Different line, push as new span
                    merged.push(span);
                }
            }
        }
    }

    self.spans = merged;
}
```

### 3.4 Step 3: Replace SpanMergingConfig

**File: `src/extractors/text.rs`** - Simplify configuration

```rust
// BEFORE: Complex configuration with multiple thresholds
pub struct SpanMergingConfig {
    pub space_threshold_em_ratio: f32,      // REMOVE
    pub conservative_threshold_pt: f32,      // REMOVE
    pub column_boundary_threshold_pt: f32,   // REMOVE (or keep for column detection only)
    pub severe_overlap_threshold_pt: f32,    // REMOVE
    pub use_adaptive_threshold: bool,        // REMOVE
    pub adaptive_config: Option<AdaptiveThresholdConfig>, // REMOVE
}

// AFTER: Single parameter
pub struct SpanMergingConfig {
    /// Word margin ratio for space detection (default: 0.1)
    pub word_margin: f32,

    /// Column boundary threshold for layout detection (keep)
    pub column_boundary_threshold_pt: f32,
}

impl Default for SpanMergingConfig {
    fn default() -> Self {
        Self {
            word_margin: 0.1,
            column_boundary_threshold_pt: 5.0,
        }
    }
}
```

### 3.5 Step 4: Remove Document-Type Multipliers

**File: `src/extractors/text.rs`** - Remove all DocumentType references

```rust
// REMOVE: detected_document_type field from TextExtractor
// REMOVE: get_adjusted_space_threshold()
// REMOVE: All imports from gap_statistics related to DocumentType

// REMOVE from extract_text_spans():
// - self.apply_adaptive_threshold()
// - Document type detection logic
```

### 3.6 Step 5: Remove Heuristic-Based Rules

**File: `src/extractors/text.rs`** - Remove CamelCase and number-letter detection

```rust
// REMOVE: should_insert_space_heuristic() function entirely
// REMOVE: split_fused_words() function entirely
// REMOVE: All CamelCase pattern matching

// The geometric rule handles these naturally:
// - "theGeneral" -> If PDF provides proper positioning, geometric gap will exist
// - If no gap exists, the PDF itself is malformed (not our problem to fix)
```

### 3.7 Step 6: Remove Confidence Scoring System

All confidence scoring is eliminated:

```rust
// REMOVE: SpaceDecision.confidence field
// REMOVE: SpaceSource enum
// REMOVE: All confidence comparisons and logging
// REMOVE: Priority-based voting in SpaceDetectionEngine
```

### 3.8 Step 7: Simplify TJ Offset Processing

**File: `src/extractors/text.rs`** - Keep TJ for positioning only

```rust
// BEFORE: TJ offsets trigger space insertion
fn process_tj_array(&mut self, elements: &[TextElement]) {
    for element in elements {
        match element {
            TextElement::Offset(offset) => {
                if *offset < self.calculate_adaptive_tj_threshold() {
                    // Insert space span  <- REMOVE this logic
                }
            }
            // ...
        }
    }
}

// AFTER: TJ offsets only update text position
fn process_tj_array(&mut self, elements: &[TextElement]) {
    for element in elements {
        match element {
            TextElement::Offset(offset) => {
                // Only update text matrix position
                self.state_stack.current_mut().text_matrix.e +=
                    (*offset / 1000.0) * self.state_stack.current().font_size;
                // NO space insertion here - let geometric detector handle it
            }
            TextElement::String(bytes) => {
                // Extract span with bounding box
                self.extract_span_from_string(bytes)?;
            }
        }
    }
}
```

---

## 4. TESTING SECTION

### 4.1 Quality Improvement Targets

| Document Type | Current | Target | Metric |
|---------------|---------|--------|--------|
| Policy PDFs | 4.3/10 | 8.5+/10 | Spurious spaces |
| Academic PDFs | 6.0/10 | 8.0+/10 | Word fusion |
| Mixed PDFs | 5.0/10 | 7.5+/10 | Overall quality |
| **Average** | **4.3/10** | **8.5+/10** | **All metrics** |

### 4.2 Specific Test Cases

#### 4.2.1 Policy PDF Tests (Previously Failing)

```rust
#[test]
fn test_policy_no_spurious_spaces() {
    // Anti-bribery and Corruption Policy Template (UK).pdf
    // Expected: 0 spurious spaces (matching pdfplumber)
    let markdown = extract_markdown("policy/Anti-bribery*.pdf");
    let metrics = analyze_quality(&markdown);

    assert_eq!(metrics.spurious_spaces.len(), 0,
        "Policy PDFs should have 0 spurious spaces with geometric spacing");
    assert!(metrics.quality_score >= 8.5);
}

#[test]
fn test_draftpolicy_no_fusion() {
    // "draftpolicy" should appear as "draft policy" if PDF has proper positioning
    // If PDF encodes as single string, accept as PDF defect (not our bug)
    let markdown = extract_markdown("policy/Code of Conduct*.pdf");

    // Count "draftpolicy" occurrences - should be 0 or marked as PDF defect
    let fusion_count = markdown.matches("draftpolicy").count();
    let defect_count = count_pdf_structure_defects(&markdown);

    assert!(fusion_count == 0 || fusion_count == defect_count,
        "Word fusions should be 0 or attributed to PDF structure defects");
}
```

#### 4.2.2 Academic PDF Tests (Should Not Degrade)

```rust
#[test]
fn test_academic_maintains_quality() {
    // arxiv_2510.21165v1.pdf
    let markdown = extract_markdown("academic/arxiv*.pdf");
    let metrics = analyze_quality(&markdown);

    // Academic should not degrade from current ~6.0/10
    assert!(metrics.quality_score >= 6.0,
        "Academic PDF quality should not degrade");

    // Word fusion should remain at 0
    assert_eq!(metrics.word_fusions.len(), 0);
}
```

#### 4.2.3 Empty Bold Marker Tests

```rust
#[test]
fn test_no_empty_bold_markers() {
    // Code of Conduct Policy Template (EU).pdf
    let markdown = extract_markdown("policy/Code of Conduct*.pdf");
    let metrics = analyze_quality(&markdown);

    assert_eq!(metrics.empty_bold_markers, 0,
        "Empty bold markers should be 0");
}
```

#### 4.2.4 Geometric Spacing Unit Tests

```rust
#[test]
fn test_geometric_spacing_basic() {
    let config = SpacingConfig::default();  // word_margin = 0.1

    // Clear word gap: 10pt gap, 30pt char width -> margin=3pt
    assert!(should_insert_space_basic(30.0, 40.0, 30.0, &config).insert);

    // Tight kerning: 1pt gap, 20pt char width -> margin=2pt
    assert!(!should_insert_space_basic(20.0, 21.0, 20.0, &config).insert);

    // Edge case: exactly at margin
    assert!(!should_insert_space_basic(20.0, 22.0, 20.0, &config).insert);
    assert!(should_insert_space_basic(20.0, 22.1, 20.0, &config).insert);
}

#[test]
fn test_word_margin_variations() {
    let tight = SpacingConfig { word_margin: 0.05 };
    let loose = SpacingConfig { word_margin: 0.15 };

    // Same 3pt gap, 30pt char
    // tight: margin=1.5pt, gap > margin -> insert
    // loose: margin=4.5pt, gap < margin -> no insert

    assert!(should_insert_space_basic(30.0, 33.0, 30.0, &tight).insert);
    assert!(!should_insert_space_basic(30.0, 33.0, 30.0, &loose).insert);
}
```

### 4.3 Regression Test Updates

**File: `tests/regression_suite.rs`**

```rust
// UPDATE: Remove document-type-specific assertions
// UPDATE: Use single geometric config for all tests
// UPDATE: Adjust quality thresholds

fn extract_markdown(pdf_path: &str, config: SpanMergingConfig) -> String {
    // Use single config for all document types
    // No more SpanMergingConfig::adaptive() vs SpanMergingConfig::legacy()
}

fn run_regression_tests(pdfs: &[&str], mode: TestMode) {
    // Simplify: single quality check, no document-type branching
    for pdf_name in pdfs {
        let markdown = extract_markdown(pdf_path, SpanMergingConfig::default());
        let metrics = analyze_quality(&markdown);

        // Single unified quality gate
        assert!(metrics.quality_score >= 8.0);
        assert_eq!(metrics.empty_bold_markers, 0);
        // Allow PDF structure defects, block true regressions
    }
}
```

---

## 5. RISKS & TRADE-OFFS

### 5.1 Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Edge cases break | Medium | Medium | Comprehensive test suite before removal |
| Academic PDFs degrade | Low | High | Run full regression before merge |
| Word fusion increases | Low | High | Geometric rule handles most cases |
| Performance regression | Very Low | Low | Simpler code = faster execution |

### 5.2 Trade-offs

#### Simplicity vs Feature-Rich

**ACCEPTING**: Single geometric rule
**REJECTING**:
- CamelCase heuristic detection
- Number-to-letter heuristic
- Document-type-aware thresholds
- Viterbi word segmentation
- Confidence scoring

**RATIONALE**: pdfplumber achieves 0 spurious spaces without these features. Our heuristics are causing more harm than good.

#### Determinism vs Adaptability

**ACCEPTING**: Fixed `word_margin` parameter
**REJECTING**: Adaptive threshold calculation based on document statistics

**RATIONALE**: Adaptive thresholds introduce unpredictability. A well-tuned fixed parameter (0.1) works across document types.

### 5.3 Mitigation Strategy

1. **Before removing any code**: Run full regression suite
2. **Create feature flag**: `LEGACY_SPACING=1` to enable old code during transition
3. **Preserve old code**: Move to `_deprecated/` directory initially
4. **Phased rollout**:
   - Phase 9.1-9.2: Add new code alongside old
   - Phase 9.3: Disable old code (keep present)
   - Phase 9.4: Validate with tests
   - Phase 9.5: Remove old code after validation

---

## 6. IMPLEMENTATION CHECKLIST

### 6.1 File Changes

- [ ] **NEW** `src/extractors/geometric_spacing.rs` - Single geometric rule
- [ ] **MODIFY** `src/extractors/text.rs` - Integrate geometric spacing
- [ ] **MODIFY** `src/extractors/mod.rs` - Export new module
- [ ] **MODIFY** `src/layout/mod.rs` - Remove space_detection exports
- [ ] **DELETE** `src/layout/space_detection.rs` - Entire file
- [ ] **DELETE** `src/extractors/word_segmentation.rs` - Entire file
- [ ] **MODIFY** `src/extractors/gap_statistics.rs` - Remove DocumentType/Profile
- [ ] **MODIFY** `src/converters/markdown.rs` - Simplify if needed
- [ ] **MODIFY** `tests/quality_metrics.rs` - Update thresholds
- [ ] **MODIFY** `tests/regression_suite.rs` - Update expectations

### 6.2 Code Removal List

| File | Item | Lines |
|------|------|-------|
| text.rs | SpaceSource enum | 21-45 |
| text.rs | SpaceDecision struct | 47-89 |
| text.rs | should_insert_space() | 632-808 |
| text.rs | should_insert_space_heuristic() | 3907-3930 |
| text.rs | get_adjusted_space_threshold() | 1313-1350 |
| text.rs | calculate_adaptive_tj_threshold() | 1352-1407 |
| text.rs | split_fused_words() | 2260+ |
| text.rs | detected_document_type field | 1222 |
| space_detection.rs | Entire file | 1-523 |
| word_segmentation.rs | Entire file | 1-1532 |
| gap_statistics.rs | DocumentType enum | 757-945 |
| gap_statistics.rs | DocumentProfile enum | 1009-1127 |

### 6.3 New Function Signatures

```rust
// New module: src/extractors/geometric_spacing.rs

/// Simple space detection result (no confidence)
pub struct SpaceInsertion { pub insert: bool }

/// Minimal configuration (single parameter)
pub struct SpacingConfig { pub word_margin: f32 }

/// Main API - geometric space detection
pub fn should_insert_space(
    prev: &TextSpan,
    next: &TextSpan,
    config: &SpacingConfig,
) -> SpaceInsertion;

/// Helper - boundary whitespace check (from old has_boundary_space)
fn has_boundary_whitespace(prev: &str, next: &str) -> bool;
```

### 6.4 Test Updates

- [ ] Add `test_geometric_spacing_basic()` - Unit tests for new logic
- [ ] Add `test_word_margin_variations()` - Config variations
- [ ] Update `test_policy_no_spurious_spaces()` - Expect 0 spurious
- [ ] Update `test_draftpolicy_no_fusion()` - Accept PDF defects
- [ ] Update `test_academic_maintains_quality()` - No degradation
- [ ] Update `test_core_regression_suite()` - Single config
- [ ] Remove `test_document_type_*` tests - No longer relevant

### 6.5 Documentation Updates

- [ ] Update `README.md` - Simplified spacing section
- [ ] Update `IMPLEMENTATION_ROADMAP.md` - Add Phase 9
- [ ] Archive `PHASE1_*.md` through `PHASE8_*.md` references
- [ ] Create `PHASE9_COMPLETION_REPORT.md` after implementation

---

## 7. SUCCESS CRITERIA

Phase 9 is complete when:

1. **Code Reduction**: Spacing logic reduced from ~2925 lines to ~80 lines (-97%)
2. **Quality Improvement**: Average quality score increases from 4.3/10 to 8.5+/10
3. **Policy PDFs**: 0 spurious spaces (matching pdfplumber)
4. **Academic PDFs**: No quality degradation (maintain 6.0+/10)
5. **Tests Pass**: All regression tests pass with new geometric logic
6. **No Heuristics**: Zero CamelCase, number-letter, or confidence-based detection
7. **Single Config**: One `word_margin` parameter replaces all previous thresholds

---

## Appendix A: pdfplumber Reference

pdfplumber's word extraction uses `pdfminer.six` LAParams:

```python
# pdfplumber defaults
LAParams(
    line_overlap=0.5,
    char_margin=2.0,
    word_margin=0.1,  # <- This is the key parameter
    line_margin=0.5,
    boxes_flow=0.5,
    detect_vertical=False,
    all_texts=False,
)
```

The `word_margin` parameter controls space insertion:
- Value: Ratio of character width/height
- Default: 0.1 (10% of character size)
- Behavior: If gap > word_margin * char_size, insert space

This single parameter achieves 0 spurious spaces on policy PDFs.

---

## Appendix B: Comparison with Current Rules

| Rule | Current Approach | pdfplumber Approach |
|------|------------------|---------------------|
| TJ Offset | Insert space if offset < -120 | Position update only |
| Geometric Gap | space_threshold_em + conservative_pt | word_margin * char_size |
| CamelCase | Detect and split | Not needed |
| Number-Letter | Detect and split | Not needed |
| Document Type | 1.3x/0.7x multipliers | Not needed |
| Confidence | 0.5-0.95 scores | Not needed |
| Adaptive | median * multiplier | Not needed |

**Key Insight**: pdfplumber's simplicity works because:
1. PDF text positioning is usually accurate
2. When it's not, heuristics often make things worse
3. A consistent geometric rule produces predictable results
