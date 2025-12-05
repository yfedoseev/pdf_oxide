# Phase 2: Unified Space Detection and Bold Rendering Architecture

## Executive Summary

Phase 1 addressed symptoms of the empty bold markers and word fusion issues through targeted fixes (adaptive thresholds, whitespace filtering, punctuation spacing). However, the root cause persists: **three decoupled layers make independent space/formatting decisions**.

This document defines Phase 2, a comprehensive architectural redesign applying SOLID principles to create:
1. **Unified Space Detection Engine** - Single source of truth for space decisions (SRP)
2. **Font Weight Normalization** - Propagate bold across word boundaries, never on spaces (DIP)
3. **Conservative Bold Rendering** - Validate bold markers require word content (OCP)

**Scope**: Approximately 3-4 weeks of focused development across 27 tasks in 6 phases.

---

## Problem Analysis

### Current Architecture (Flawed)

```
PDF Content Stream
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 1: TJ Processing (text.rs:2643-2677)                     │
│ - Creates space spans from TJ offset thresholds                │
│ - Space inherits bold from graphics state (WRONG per PDF spec) │
│ - insert_space_as_span() hardcodes FontWeight::Normal          │
│   (partial fix, but spacing still created as separate spans)   │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 2: Span Merging (text.rs:1347-1511)                      │
│ - Uses SpanMergingConfig thresholds                            │
│ - Adaptive gap statistics (gap_statistics.rs)                  │
│ - should_insert_space_heuristic() for character transitions    │
│ - has_boundary_space() prevents double spaces                  │
│ - Inserts space into merged text, inherits formatting          │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: Markdown Rendering (markdown.rs:290-370)              │
│ - Groups consecutive blocks by bold status                     │
│ - is_content_block() checks for non-whitespace                 │
│ - should_insert_bold_marker() validates boundaries             │
│ - Filtering happens AFTER content cleaning (too late!)         │
└────────────────────────────────────────────────────────────────┘
```

### Root Causes

| Issue | Root Cause | Current Code Location |
|-------|------------|----------------------|
| Empty bold markers (`** **`) | Space spans inherit bold from TJ state, grouped with content | text.rs:2753 (hardcoded Normal, but separate spans still grouped) |
| Word fusions | Three gap thresholds contradict; heuristics applied inconsistently | text.rs:1400-1417 (gap vs heuristic vs conservative) |
| Spurious spaces | Space decision made too early (TJ layer) before full context | text.rs:2643-2677 (space_insertion_threshold) |

### PDF Specification Reference

**ISO 32000-1:2008, Section 9.4.4, NOTE 6**:
> "Text strings shall be as long as possible, since spaces between strings may be implemented differently in different text showing operators."

**Interpretation**: Spaces are positioning artifacts from TJ offsets, not semantic content. They should:
1. Never carry formatting attributes (bold, italic)
2. Be decided holistically, not at TJ processing time
3. Serve only as word delimiters in final output

---

## Phase 2 Architecture

### 2.1: Unified Space Detection Engine

**Principle**: Single Responsibility (SRP) + Dependency Inversion (DIP)

Create a new module `src/layout/space_detection.rs` that centralizes ALL space decisions.

#### Interface Design

```rust
/// Result of space detection analysis.
#[derive(Debug, Clone)]
pub struct SpaceDecision {
    /// Whether to insert a space between two text elements.
    pub insert_space: bool,
    /// Confidence level of the decision.
    pub confidence: SpaceConfidence,
    /// Detection method that produced this decision.
    pub method: SpaceDetectionMethod,
    /// Position in document (for debugging).
    pub position: SpacePosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceConfidence {
    /// High confidence - clear word boundary.
    High,
    /// Medium confidence - likely word boundary.
    Medium,
    /// Low confidence - may be kerning or word boundary.
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceDetectionMethod {
    /// Gap exceeds threshold.
    GapBased,
    /// Character transition heuristic (CamelCase, number-letter).
    Heuristic,
    /// Adaptive threshold from document statistics.
    Adaptive,
    /// TJ offset exceeds threshold.
    TjOffset,
    /// Combined methods agree.
    Consensus,
}

/// Position context for space decision.
#[derive(Debug, Clone)]
pub struct SpacePosition {
    pub left_text: String,
    pub right_text: String,
    pub gap_pt: f32,
    pub line_y: f32,
}

/// Trait for space detection strategies (Open/Closed principle).
pub trait SpaceDetector: Send + Sync {
    /// Analyze gap between two text elements.
    fn detect(&self, context: &SpaceContext) -> SpaceDecision;

    /// Priority for this detector (higher = checked first).
    fn priority(&self) -> u8;

    /// Name for debugging.
    fn name(&self) -> &'static str;
}

/// Context passed to space detectors.
#[derive(Debug)]
pub struct SpaceContext<'a> {
    /// Text of left element.
    pub left_text: &'a str,
    /// Text of right element.
    pub right_text: &'a str,
    /// Gap in PDF points.
    pub gap_pt: f32,
    /// Font size of left element.
    pub left_font_size: f32,
    /// Font size of right element.
    pub right_font_size: f32,
    /// Document-wide gap statistics (if available).
    pub gap_stats: Option<&'a GapStatistics>,
    /// TJ offset that caused this gap (if applicable).
    pub tj_offset: Option<f32>,
}

/// Unified space detection engine.
pub struct SpaceDetectionEngine {
    detectors: Vec<Box<dyn SpaceDetector>>,
    config: SpaceDetectionConfig,
}

impl SpaceDetectionEngine {
    /// Create engine with default detectors.
    pub fn new() -> Self;

    /// Create engine with custom configuration.
    pub fn with_config(config: SpaceDetectionConfig) -> Self;

    /// Add a custom detector (Open/Closed principle).
    pub fn add_detector(&mut self, detector: Box<dyn SpaceDetector>);

    /// Make authoritative space decision.
    pub fn decide(&self, context: &SpaceContext) -> SpaceDecision;
}
```

#### Built-in Detectors

```rust
/// Gap-based detection using geometric spacing.
pub struct GapBasedDetector {
    pub threshold_em: f32,  // Default: 0.25
}

/// Heuristic-based detection for character transitions.
pub struct HeuristicDetector;

/// Adaptive detection using document statistics.
pub struct AdaptiveDetector {
    pub config: AdaptiveThresholdConfig,
}

/// TJ offset-based detection for PDF operators.
pub struct TjOffsetDetector {
    pub threshold: f32,  // Default: -120.0
}

/// Consensus detector that requires agreement.
pub struct ConsensusDetector {
    pub min_agreement: usize,  // Default: 2
}
```

### 2.2: Font Weight Normalization

**Principle**: Single Responsibility (SRP) + Separation of Concerns

Font weight should be tracked separately from text content, applied only to word-containing spans.

#### Data Structure Changes

```rust
/// Enhanced TextSpan with formatting metadata.
#[derive(Debug, Clone)]
pub struct TextSpan {
    pub text: String,
    pub bbox: Rect,
    pub font_name: String,
    pub font_size: f32,
    /// Raw font weight from graphics state.
    pub raw_font_weight: FontWeight,
    /// Effective font weight (may differ for spaces).
    pub effective_font_weight: FontWeight,
    pub color: Color,
    pub mcid: Option<u32>,
    pub sequence: usize,
    /// New: Span type for formatting decisions.
    pub span_type: SpanType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanType {
    /// Word content (letters, numbers, punctuation).
    Word,
    /// Whitespace separator (should not carry bold).
    Space,
    /// Mixed content (rare edge case).
    Mixed,
}

impl TextSpan {
    /// Check if this span should participate in bold grouping.
    pub fn should_render_bold(&self) -> bool {
        self.span_type == SpanType::Word && self.effective_font_weight.is_bold()
    }

    /// Normalize font weight based on span type.
    pub fn normalize_font_weight(&mut self) {
        if self.span_type == SpanType::Space {
            // Spaces never carry bold formatting
            self.effective_font_weight = FontWeight::Normal;
        }
    }
}
```

#### Normalization Pipeline

```rust
/// Font weight normalization pass.
pub struct FontWeightNormalizer;

impl FontWeightNormalizer {
    /// Normalize font weights across a span sequence.
    ///
    /// Rules:
    /// 1. Space spans always have Normal weight
    /// 2. Bold should apply to complete words
    /// 3. Adjacent bold words in same font should stay bold
    pub fn normalize(spans: &mut [TextSpan]) {
        for span in spans.iter_mut() {
            span.normalize_font_weight();
        }
    }

    /// Propagate bold across word boundaries.
    ///
    /// When a word is split across multiple spans, ensure consistent bold.
    pub fn propagate_bold_to_word_boundaries(spans: &mut [TextSpan]) {
        // Implementation: track word boundaries, ensure all spans
        // in a word have consistent bold status
    }
}
```

### 2.3: Conservative Bold Boundary Rules

**Principle**: Open/Closed (OCP) - New validation rules without modifying rendering

#### Validation Layer

```rust
/// Validator for bold marker insertion.
pub struct BoldMarkerValidator;

impl BoldMarkerValidator {
    /// Check if bold markers can be inserted for a text group.
    pub fn can_insert_markers(group: &BoldGroup) -> BoldMarkerDecision {
        // Rule 1: Must have non-whitespace content
        if !group.has_word_content() {
            return BoldMarkerDecision::Skip(SkipReason::WhitespaceOnly);
        }

        // Rule 2: Opening position must be at word boundary
        if !group.has_valid_opening_boundary() {
            return BoldMarkerDecision::Skip(SkipReason::InvalidOpenBoundary);
        }

        // Rule 3: Closing position must be at word boundary
        if !group.has_valid_closing_boundary() {
            return BoldMarkerDecision::Skip(SkipReason::InvalidCloseBoundary);
        }

        // Rule 4: Content must remain non-empty after cleaning
        if group.cleaned_text().trim().is_empty() {
            return BoldMarkerDecision::Skip(SkipReason::EmptyAfterCleaning);
        }

        BoldMarkerDecision::Insert
    }
}

#[derive(Debug, Clone)]
pub enum BoldMarkerDecision {
    Insert,
    Skip(SkipReason),
}

#[derive(Debug, Clone)]
pub enum SkipReason {
    WhitespaceOnly,
    InvalidOpenBoundary,
    InvalidCloseBoundary,
    EmptyAfterCleaning,
}

/// Group of spans to render with bold formatting.
#[derive(Debug)]
pub struct BoldGroup<'a> {
    spans: &'a [TextSpan],
    prev_char: Option<char>,
    next_char: Option<char>,
}

impl<'a> BoldGroup<'a> {
    pub fn has_word_content(&self) -> bool {
        self.spans.iter().any(|s| s.span_type == SpanType::Word)
    }

    pub fn has_valid_opening_boundary(&self) -> bool {
        // Check prev_char and first char of group
        let first_char = self.spans.iter()
            .filter(|s| s.span_type == SpanType::Word)
            .flat_map(|s| s.text.chars())
            .next();

        should_insert_bold_marker(self.prev_char, first_char)
    }

    pub fn has_valid_closing_boundary(&self) -> bool {
        // Check last char of group and next_char
        let last_char = self.spans.iter()
            .filter(|s| s.span_type == SpanType::Word)
            .flat_map(|s| s.text.chars())
            .last();

        should_insert_bold_marker(last_char, self.next_char)
    }

    pub fn cleaned_text(&self) -> String {
        self.spans.iter()
            .map(|s| s.text.as_str())
            .collect()
    }
}
```

---

## Data Flow (New Architecture)

```
PDF Content Stream
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ TJ Processing Layer (SIMPLIFIED)                               │
│ - Extract text content with raw positioning                    │
│ - Mark spans as Word or Space type                             │
│ - Do NOT make space insertion decisions                        │
│ - Preserve TJ offsets for later analysis                       │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ NEW: Unified Space Detection Engine                            │
│ - Analyze all gaps using multiple strategies                   │
│ - Gap-based + Heuristic + Adaptive + TJ offset                 │
│ - Single authoritative decision per gap                        │
│ - Output: list of SpaceDecision with positions                 │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ NEW: Font Weight Normalization                                 │
│ - Mark Space spans as Normal weight                            │
│ - Propagate bold to complete words                             │
│ - Set effective_font_weight based on span_type                 │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Span Merging Layer (SIMPLIFIED)                                │
│ - Use SpaceDecision results (no independent logic)             │
│ - Merge spans based on detection engine output                 │
│ - Preserve span_type through merging                           │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Markdown Rendering Layer (SIMPLIFIED)                          │
│ - Use effective_font_weight for grouping                       │
│ - Use BoldMarkerValidator before insertion                     │
│ - Skip groups that fail validation                             │
└────────────────────────────────────────────────────────────────┘
```

---

## File Modifications

### New Files

| File | Purpose | LOC Estimate |
|------|---------|--------------|
| `src/layout/space_detection.rs` | Unified space detection engine | 400-500 |
| `src/layout/font_normalization.rs` | Font weight normalization | 150-200 |
| `src/layout/bold_validation.rs` | Bold marker validation | 100-150 |

### Modified Files

| File | Changes | Complexity |
|------|---------|------------|
| `src/extractors/text.rs` | Remove independent space logic, add span_type, integrate engine | High |
| `src/converters/markdown.rs` | Use BoldMarkerValidator, simplify grouping | Medium |
| `src/layout/text_block.rs` | Add SpanType, effective_font_weight | Low |
| `src/layout/mod.rs` | Export new modules | Low |

---

## Migration Strategy

### Phase 2a: Foundation (Week 1)
1. Create `space_detection.rs` with trait and basic detectors
2. Add `SpanType` to `TextSpan`
3. Add `effective_font_weight` to `TextSpan`
4. Update `insert_space_as_span()` to set `span_type = SpanType::Space`

### Phase 2b: Integration (Week 2)
1. Create `FontWeightNormalizer` and integrate into extraction
2. Create `BoldMarkerValidator` and integrate into markdown rendering
3. Modify `merge_adjacent_spans()` to use detection engine
4. Update tests to verify new behavior

### Phase 2c: Cleanup (Week 3)
1. Remove duplicate space detection logic from `merge_adjacent_spans()`
2. Remove `is_content_block()` checks that are now redundant
3. Simplify `should_insert_bold_marker()` to use validator
4. Update documentation

### Phase 2d: Validation (Week 4)
1. Run full regression suite
2. Verify 0 empty bold markers across all test PDFs
3. Verify 0 new word fusions (existing PDF structure issues exempted)
4. Performance benchmarking (target: <5% overhead)

---

## Technical Debt Identified

### Current Debt (to be resolved in Phase 2)

| ID | Category | Severity | Description | Resolution |
|----|----------|----------|-------------|------------|
| DEBT-001 | Architecture | HIGH | Three independent space detection layers | Unified SpaceDetectionEngine |
| DEBT-002 | Architecture | HIGH | Space spans inherit bold from graphics state | SpanType with normalization |
| DEBT-003 | Performance | MEDIUM | Gap analysis repeated at multiple layers | Single-pass detection |
| DEBT-004 | Maintenance | MEDIUM | Duplicate heuristics in text.rs and markdown.rs | Centralized in space_detection.rs |

### New Debt (introduced by Phase 2)

| ID | Category | Severity | Description | Mitigation |
|----|----------|----------|-------------|------------|
| DEBT-005 | Complexity | LOW | Additional abstraction layer for space detection | Well-documented trait hierarchy |
| DEBT-006 | Performance | LOW | Extra pass for font normalization | Can be combined with span creation |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Regression in existing PDFs | Medium | High | Comprehensive regression suite with 15 PDFs |
| Performance degradation | Low | Medium | Benchmark before/after; target <5% overhead |
| Scope creep | Medium | Medium | Clear phase boundaries with deliverables |
| Breaking API changes | Low | High | Maintain backward compatibility with deprecated aliases |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Empty bold markers per PDF | 2-10 | 0 | Regression suite |
| Word fusions (High/Medium confidence) | 1-3 | 0 | Regression suite |
| Quality score | 5.0-8.0 | 9.0+ | Quality metrics |
| Test pass rate | 1/5 (20%) | 5/5 (100%) | Regression suite |
| Performance overhead | - | <5% | Benchmark suite |

---

## Appendix: Current Code References

### TJ Processing (text.rs:2643-2677)
```rust
TextElement::Offset(offset) => {
    if *offset < self.config.space_insertion_threshold {
        self.flush_tj_buffer(&buffer)?;
        // Phase 7.2 Fix: Check if next element starts with space
        let next_element_starts_with_space = ...;
        if !next_element_starts_with_space {
            self.insert_space_as_span()?;  // Creates space span
        }
        buffer = TjBuffer::new(self.state_stack.current(), self.current_mcid);
    }
    self.advance_position_for_offset(*offset)?;
}
```

### Span Merging (text.rs:1400-1423)
```rust
let needs_space_by_gap = gap > space_threshold;
let needs_space_by_heuristic = should_insert_space_heuristic(&current.text, &span.text);
let gap_wants_space = needs_space_by_gap
    || needs_space_by_heuristic
    || gap > self.merging_config.conservative_threshold_pt;
let already_has_space = has_boundary_space(&current.text, &span.text);
let needs_space = gap_wants_space && !already_has_space;
```

### Markdown Rendering (markdown.rs:330-354)
```rust
let can_insert_open = should_insert_bold_marker(prev_char, first_char_in_group);
let can_insert_close = should_insert_bold_marker(last_char_in_group, next_char_after_group);
let should_render_bold_markers = match options.bold_marker_behavior {
    BoldMarkerBehavior::Aggressive => true,
    BoldMarkerBehavior::Conservative => is_content_block(&group_text),
};
let final_is_whitespace_only = cleaned_text.trim().is_empty();
let should_insert_markers = is_bold && can_insert_open && can_insert_close
    && should_render_bold_markers && !final_is_whitespace_only;
```

---

## Next Steps

1. Review and approve this design document
2. Begin Phase 2a implementation (space_detection.rs foundation)
3. Add new module structure to `src/layout/mod.rs`
4. Create initial test suite for SpaceDetector trait
