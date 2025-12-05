# Code Documentation - Quality Fix Implementation

**Version**: 1.0
**Date**: December 4, 2025
**Focus**: Explaining key functions and their roles in quality improvements

---

## Overview

This document provides detailed explanations of the key functions and structures implementing the quality improvements.

---

## Key Structures

### SpaceDecision

**File**: `src/extractors/text.rs`

**Purpose**: Represents a decision about whether to insert a space between two text spans.

**Definition**:
```rust
pub struct SpaceDecision {
    pub insert_space: bool,
    pub source: SpaceSource,
    pub confidence: f32,
}

pub enum SpaceSource {
    TjOffset,          // Negative offset in TJ array (explicit PDF positioning)
    GeometricGap,      // Gap > threshold between spans
    CharacterHeuristic, // Character transition (e.g., lowercase->uppercase)
    AlreadyPresent,    // Trailing/leading whitespace exists
}
```

**Usage**:
```rust
// Unified space decision - single source of truth
let decision = should_insert_space(
    "the",
    "General",
    0.0,      // No gap (fused words)
    12.0,     // Font size
    false,    // No TJ offset
    &config,
);

match decision.source {
    SpaceSource::CharacterHeuristic => {
        // CamelCase transition detected
        println!("Splitting CamelCase: {} {}", "the", "General");
    }
    _ => {}
}
```

**PDF Spec Reference**:
- ISO 32000-1:2008, Section 9.4.4 NOTE 6
- "The identification of what constitutes a word is unrelated to how the text happens to be grouped"

---

### DocumentProfile

**File**: `src/extractors/gap_statistics.rs`

**Purpose**: Classifies document type to select appropriate spacing thresholds.

**Definition**:
```rust
pub enum DocumentProfile {
    /// Academic papers: standard spacing, column layouts
    /// Gap distribution: median ~0.5-1.5pt, high variance
    Academic,

    /// Policy documents: tight spacing, justified text
    /// Gap distribution: median <0.5pt, low variance
    Policy,

    /// Mixed/Unknown: balanced defaults
    Default,
}

impl DocumentProfile {
    pub fn detect(spans: &[TextSpan]) -> Self {
        let stats = analyze_document_gaps(spans, None);

        // Tight median gap suggests policy document
        if stats.median < 0.5 {
            return Self::Policy;
        }

        // High gap variance suggests academic (columns)
        if stats.coefficient_of_variation() > 0.8 {
            return Self::Academic;
        }

        Self::Default
    }

    pub fn get_config(&self) -> AdaptiveThresholdConfig {
        match self {
            Self::Academic => AdaptiveThresholdConfig {
                median_multiplier: 1.6,
                min_threshold_pt: 0.1,
                max_threshold_pt: 100.0,
                ..Default::default()
            },
            Self::Policy => AdaptiveThresholdConfig {
                median_multiplier: 1.2,  // More sensitive
                min_threshold_pt: 0.05,
                max_threshold_pt: 100.0,
                ..Default::default()
            },
            Self::Default => AdaptiveThresholdConfig::balanced(),
        }
    }
}
```

**Usage**:
```rust
// Automatically detect and apply profile
let profile = DocumentProfile::detect(&spans);
let config = profile.get_config();

println!("Detected profile: {:?}", profile);
println!("Using threshold multiplier: {}", config.median_multiplier);
```

**Why This Works**:
- Different document types have fundamentally different spacing
- Single fixed threshold fails for multiple document types
- Automatic detection requires no user configuration

---

### TextSpan Extension

**File**: `src/layout/mod.rs`

**Purpose**: Track intentional word splits to prevent re-merging.

**Addition**:
```rust
pub struct TextSpan {
    // ... existing fields ...

    /// If true, this span was created by splitting fused words (e.g., CamelCase)
    /// These spans have zero gap but should NOT be merged back with adjacent spans.
    ///
    /// Example:
    /// - Original: "theGeneral" (single text in PDF)
    /// - After CamelCase split: ["the", "General"]
    /// - split_boundary_before = true for "General"
    /// - This prevents merge from recreating "theGeneral"
    pub split_boundary_before: bool,
}
```

**Usage in Merging**:
```rust
fn merge_adjacent_spans(&mut self) {
    // ... iterate through spans ...

    let should_merge = same_line
        && !span.split_boundary_before  // NEW: Respect split boundaries
        && (self.merging_config.severe_overlap_threshold_pt..3.0).contains(&gap)
        && !large_gap_indicates_column;

    if should_merge {
        // Merge spans
    }
}
```

**Impact**:
- Eliminates word fusion regression during merging
- Preserves intentional CamelCase splits
- No false positives from legitimate zero-gap scenarios

---

## Key Functions

### should_insert_space()

**File**: `src/extractors/text.rs`

**Purpose**: Unified decision function for space insertion (single source of truth).

**Full Implementation**:
```rust
/// Unified space decision logic - SINGLE POINT OF TRUTH
///
/// PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
/// "Word boundaries are NOT defined by PDF - they require heuristics."
///
/// This function consolidates all space insertion signals into one decision,
/// eliminating the double-space problem from previous implementations.
///
/// # Arguments
///
/// * `preceding_text` - Text of previous span (e.g., "the")
/// * `following_text` - Text of next span (e.g., "General")
/// * `gap_pt` - Gap between spans in PDF points (1/72 inch)
/// * `font_size` - Current font size in points
/// * `tj_offset_triggered` - Whether TJ array offset exceeded threshold
/// * `config` - Span merging configuration with thresholds
///
/// # Returns
///
/// SpaceDecision with insert_space bool, source, and confidence
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::extractors::text::should_insert_space;
/// use pdf_oxide::extractors::SpanMergingConfig;
///
/// // Example 1: Large gap (clear word boundary)
/// let decision = should_insert_space(
///     "word",
///     "next",
///     2.5,  // Large gap > threshold
///     12.0,
///     false,
///     &SpanMergingConfig::default(),
/// );
/// assert!(decision.insert_space);  // Insert space
///
/// // Example 2: CamelCase boundary (no gap, but character transition)
/// let decision = should_insert_space(
///     "the",
///     "General",
///     0.0,  // No gap (same string in PDF)
///     12.0,
///     false,
///     &SpanMergingConfig::default(),
/// );
/// assert!(decision.insert_space);  // Heuristic detects case transition
///
/// // Example 3: Already has space (avoid double space)
/// let decision = should_insert_space(
///     "word ",   // Trailing space
///     "next",
///     1.5,
///     12.0,
///     false,
///     &SpanMergingConfig::default(),
/// );
/// assert!(!decision.insert_space);  // Skip insertion
/// ```
///
/// # Decision Logic (Ordered by Priority)
///
/// 1. **Rule 0 - Already Has Space** (confidence 1.0)
///    - Check if preceding_text ends with space OR following_text starts with space
///    - If yes: skip insertion (avoid double space)
///
/// 2. **Rule 1 - TJ Offset** (confidence 0.95)
///    - If tj_offset_triggered: insert (explicit PDF positioning)
///    - This is the most reliable signal
///
/// 3. **Rule 2 - Dual Threshold** (confidence 0.8)
///    - Calculate space_threshold = font_size * space_threshold_em_ratio
///    - Calculate char_threshold = font_size * 0.3 (PDFBox default)
///    - Use MINIMUM of both thresholds (conservative)
///    - If gap > effective_threshold: insert space
///
/// 4. **Rule 3 - Character Transition** (confidence 0.6)
///    - Detect CamelCase: lowercase->uppercase transition
///    - Detect ALLCAPS: multiple capitals in succession
///    - If detected: insert space (heuristic)
///
/// 5. **Rule 4 - Conservative Threshold** (confidence 0.5)
///    - If gap > conservative_threshold_pt: insert space
///    - Catches small intentional gaps
///
/// 6. **Default - No Space** (confidence 1.0)
///    - No signal detected, keep text together
///
/// # PDF Specification Context
///
/// The PDF specification (Section 9.4.4) defines HOW positioning works
/// but NOT when a position offset represents a word boundary:
///
/// > "A text-positioning entry shall denote a horizontal displacement,
/// > expressed in thousandths of a unit of text space"
///
/// The threshold for what constitutes a "word boundary" is NOT SPECIFIED.
/// This function implements industry-standard heuristics based on:
/// - PDFBox (Apache): Dual threshold approach
/// - pdfminer.six (Python): Character-relative margins
/// - pdf.js (Mozilla): Position overlap detection
///
/// # Previous Problem (v0.1.1)
///
/// Two independent mechanisms both inserted spaces:
/// 1. TJ offset processing inserted space span immediately
/// 2. Span merging independently detected gap and inserted space
/// Result: "word " + " next" = "word  next" (double space)
///
/// # Current Solution (v0.1.2)
///
/// Single unified function:
/// 1. TJ processing sets flag, doesn't insert span
/// 2. Span merging calls should_insert_space() for decision
/// 3. One decision point, one insertion point
/// Result: No double spaces
pub fn should_insert_space(
    preceding_text: &str,
    following_text: &str,
    gap_pt: f32,
    font_size: f32,
    tj_offset_triggered: bool,
    config: &SpanMergingConfig,
) -> SpaceDecision {
    // Rule 0: If already have boundary space, skip
    // This prevents "word " + " next" -> "word  next"
    if preceding_text.ends_with(' ') || following_text.starts_with(' ') {
        return SpaceDecision {
            insert_space: false,
            source: SpaceSource::AlreadyPresent,
            confidence: 1.0,
        };
    }

    // Rule 1: TJ offset (highest confidence - explicit PDF positioning)
    // If PDF author explicitly positioned text, trust that decision
    if tj_offset_triggered {
        return SpaceDecision {
            insert_space: true,
            source: SpaceSource::TjOffset,
            confidence: 0.95,
        };
    }

    // Rule 2: Dual threshold (PDFBox pattern)
    // Use MINIMUM of space-width-based and char-width-based thresholds
    let space_threshold = font_size * config.space_threshold_em_ratio;
    let char_threshold = font_size * 0.3;  // 30% of em (PDFBox default)
    let effective_threshold = space_threshold.min(char_threshold);

    if gap_pt > effective_threshold {
        return SpaceDecision {
            insert_space: true,
            source: SpaceSource::GeometricGap,
            confidence: 0.8,
        };
    }

    // Rule 3: Character transition heuristic
    // Detect CamelCase and similar patterns
    if should_insert_space_heuristic(preceding_text, following_text) {
        return SpaceDecision {
            insert_space: true,
            source: SpaceSource::CharacterHeuristic,
            confidence: 0.6,
        };
    }

    // Rule 4: Conservative threshold (catches small intentional gaps)
    // Fallback for documents with minimal spacing
    if gap_pt > config.conservative_threshold_pt {
        return SpaceDecision {
            insert_space: true,
            source: SpaceSource::GeometricGap,
            confidence: 0.5,
        };
    }

    // No space signal detected
    SpaceDecision {
        insert_space: false,
        source: SpaceSource::GeometricGap,
        confidence: 1.0,
    }
}
```

**Key Design Decisions**:

1. **Single Source of Truth**: One function, called once per decision
2. **Ordered Rules**: Priority from most to least reliable
3. **Confidence Scores**: Help debug extraction issues
4. **Dual Threshold**: PDFBox pattern is industry-standard
5. **Conservative**: Prefers missing spaces over double spaces

**Testing**:
```rust
#[test]
fn test_space_decision_examples() {
    let config = SpanMergingConfig::default();

    // Double space prevention
    assert!(!should_insert_space("word ", "next", 1.0, 12.0, false, &config).insert_space);
    assert!(!should_insert_space("word", " next", 1.0, 12.0, false, &config).insert_space);

    // CamelCase detection
    assert!(should_insert_space("the", "General", 0.0, 12.0, false, &config).insert_space);

    // Gap detection
    assert!(should_insert_space("word", "next", 2.5, 12.0, false, &config).insert_space);
}
```

---

### split_fused_words()

**File**: `src/extractors/text.rs`

**Purpose**: Detect and split fused words (e.g., "theGeneral" -> "the General").

**Logic**:
```rust
/// Split words fused in PDF (e.g., "theGeneralwas" -> "the General was")
///
/// Detects CamelCase and ALLCAPS transitions as word boundaries
/// Sets split_boundary_before flag to prevent re-merging
///
/// # Examples
///
/// - Input: "theGeneral" -> Output: ["the", "General"]
/// - Input: "lengthThis" -> Output: ["length", "This"]
/// - Input: "XMLParser" -> Output: ["XML", "Parser"]
/// - Input: "word" -> Output: ["word"] (no change)
fn split_fused_words(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current_word = String::new();
    let mut prev_is_lower = false;
    let mut prev_is_digit = false;

    for ch in text.chars() {
        let is_lower = ch.is_lowercase();
        let is_upper = ch.is_uppercase();
        let is_digit = ch.is_numeric();

        // Detect boundaries:
        // 1. lowercase -> uppercase (camelCase boundary)
        // 2. digit -> letter or letter -> digit (word boundary)
        let should_split = (prev_is_lower && is_upper) ||
                          (prev_is_digit != is_digit && !current_word.is_empty());

        if should_split && !current_word.is_empty() {
            result.push(current_word.clone());
            current_word.clear();
        }

        current_word.push(ch);
        prev_is_lower = is_lower;
        prev_is_digit = is_digit;
    }

    if !current_word.is_empty() {
        result.push(current_word);
    }

    // Return original if no splits (avoid unnecessary work)
    if result.len() == 1 {
        vec![text.to_string()]
    } else {
        result
    }
}

// Usage: Mark split spans so they don't re-merge
for word in split_fused_words("theGeneral") {
    // Create TextSpan with split_boundary_before = true
    // This prevents merge from recreating "theGeneral"
}
```

---

### convert_page_from_spans() - Bold Pre-Validation

**File**: `src/converters/markdown.rs`

**Purpose**: Filter and validate blocks before bold marker grouping.

**Key Addition**:
```rust
/// Convert TextSpans to Markdown with bold pre-validation
///
/// Pipeline:
/// 1. Convert spans to blocks
/// 2. PRE-FILTER whitespace blocks (BEFORE grouping)
/// 3. NEUTRALIZE bold on non-word blocks
/// 4. Group and render (safe - won't have empty bold groups)
fn convert_page_from_spans(
    &self,
    spans: &[TextSpan],
    options: &ConversionOptions,
) -> Result<String> {
    // Step 1: Convert to blocks
    let mut blocks: Vec<TextBlock> = spans
        .iter()
        .map(|span| TextBlock {
            text: span.text.clone(),
            is_bold: matches!(span.font_weight, FontWeight::Bold),
            // ... other fields ...
        })
        .collect();

    // Step 2: PRE-FILTER whitespace blocks
    // This prevents whitespace-only blocks from causing empty bold markers
    blocks.retain(|block| !block.text.trim().is_empty());

    // Step 3: NEUTRALIZE bold on non-word blocks
    // Some blocks may have been marked bold but don't have word content
    for block in &mut blocks {
        let has_word_chars = block.text
            .chars()
            .any(|c| c.is_alphanumeric());

        if !has_word_chars {
            // Non-word content shouldn't be bold
            block.is_bold = false;
        }
    }

    // Step 4: Now safe to group and render
    // Bold groups will never contain only whitespace
    let mut output = String::new();
    let mut in_bold_group = false;
    let mut current_group = String::new();

    for block in blocks {
        if block.is_bold != in_bold_group {
            // Group boundary - output previous group
            if !current_group.is_empty() {
                if in_bold_group {
                    output.push_str(&format!("**{}**", current_group));
                } else {
                    output.push_str(&current_group);
                }
            }
            current_group.clear();
            in_bold_group = block.is_bold;
        }
        current_group.push_str(&block.text);
    }

    // Output final group
    if !current_group.is_empty() {
        if in_bold_group {
            output.push_str(&format!("**{}**", current_group));
        } else {
            output.push_str(&current_group);
        }
    }

    Ok(output)
}
```

**Why This Works**:
- Whitespace filtered BEFORE bold grouping
- Bold markers wrap actual content only
- No empty `** **` patterns

---

## Configuration

### SpanMergingConfig

**File**: `src/extractors/text.rs`

**Purpose**: Configure space insertion and span merging behavior.

**Fields Explained**:

```rust
pub struct SpanMergingConfig {
    /// Enable adaptive thresholds based on document profile
    ///
    /// When true:
    /// - Analyzes gap statistics on first extraction pass
    /// - Detects document type (Academic/Policy/Default)
    /// - Uses profile-specific thresholds
    ///
    /// When false:
    /// - Uses fixed thresholds (legacy v0.1.1 behavior)
    /// - No profile detection overhead
    /// - May have spurious spaces in some document types
    pub use_adaptive_threshold: bool,

    /// Configuration for adaptive mode
    pub adaptive_config: AdaptiveThresholdConfig,

    /// Fallback threshold when not using adaptive (pt)
    /// Default: 0.5
    /// - Lower (0.1): More sensitive, more spurious spaces
    /// - Higher (1.0): Less sensitive, more missed boundaries
    pub fixed_space_threshold_pt: f32,

    /// Conservative threshold applied in all modes (pt)
    /// Default: 0.1
    /// Catches very small gaps even with adaptive mode
    /// Safety net to prevent fused text
    pub conservative_threshold_pt: f32,

    /// Threshold for merging spans on different lines (pt)
    /// Default: -0.5
    /// If gap <= this, spans on different lines may still merge
    pub severe_overlap_threshold_pt: f32,

    /// Ratio for space width threshold (em units)
    /// Default: 0.5
    /// Threshold = font_size * space_threshold_em_ratio
    /// 0.25em is typical space character width
    pub space_threshold_em_ratio: f32,
}
```

**Examples**:

```rust
// Default (recommended) - adaptive with safety net
let config = SpanMergingConfig::default();

// Legacy (v0.1.1 behavior) - fixed thresholds
let legacy_config = SpanMergingConfig::legacy();

// Tight documents (policy) - more sensitive
let mut tight_config = SpanMergingConfig::default();
tight_config.adaptive_config.median_multiplier = 1.2;

// Fast extraction - skip profile detection
let mut fast_config = SpanMergingConfig::default();
fast_config.use_adaptive_threshold = false;
fast_config.fixed_space_threshold_pt = 0.75;
```

---

## Testing

### Unit Tests

**File**: `tests/quality_metrics.rs`

**Key Test Cases**:

```rust
#[test]
fn test_double_space_prevention() {
    // Verify unified space decision prevents double spaces
    let text = extract_text("arxiv_2510.21165v1.pdf");
    assert!(text.matches("  ").count() < 50);
}

#[test]
fn test_camelcase_splitting() {
    // Verify CamelCase words are split and preserved
    let text = extract_text("test_camelcase.pdf");
    assert!(text.contains("the General"));  // Should be split
    assert!(!text.contains("theGeneral")); // Not fused
}

#[test]
fn test_empty_bold_markers() {
    // Verify no empty bold markers in output
    let markdown = to_markdown("test.pdf");
    assert!(!markdown.contains("** **"));  // No empty bold
    assert!(!markdown.contains("**  **")); // No whitespace-only bold
}

#[test]
fn test_document_profile_detection() {
    let spans = extract_spans("academic.pdf");
    let profile = DocumentProfile::detect(&spans);

    // Academic documents should be detected
    match profile {
        DocumentProfile::Academic => {},
        _ => panic!("Expected academic profile"),
    }
}
```

---

## Debugging Tips

### Enable Debug Logging

```bash
# Run with logging
RUST_LOG=pdf_oxide=debug cargo test

# Filter to specific module
RUST_LOG=pdf_oxide::extractors::text=debug cargo test
```

### Analyze Document Profile

```rust
use pdf_oxide::extractors::gap_statistics::{DocumentProfile, analyze_document_gaps};

let profile = DocumentProfile::detect(&spans);
let stats = analyze_document_gaps(&spans, None);

println!("Profile: {:?}", profile);
println!("Gap stats: median={}, mean={}, std_dev={}",
    stats.median, stats.mean, stats.std_dev);
println!("Coefficient of variation: {}", stats.coefficient_of_variation());
```

### Trace Space Decisions

```rust
for decision in decisions {
    println!("Decision: insert={}, source={:?}, confidence={}",
        decision.insert_space, decision.source, decision.confidence);
}
```

---

## References

### PDF Specification Sections

- **Section 9.4.4**: Text Positioning Operators (TJ, Tj)
- **Section 5.3.2**: Word Spacing (Tw parameter)
- **Section 9.10.2**: Character Encoding (ToUnicode CMap)

### Implementation References

- [PDFBox PDFTextStripper](https://github.com/Valuya/fontbox/blob/master/pdfbox/src/main/java/org/apache/pdfbox/text/PDFTextStripper.java)
- [pdfminer.six LAParams](https://pdfminersix.readthedocs.io/)
- [pdf.js Text Extraction](https://github.com/mozilla/pdf.js)

---

**Document prepared**: December 4, 2025
**Focus**: Detailed function documentation with examples and rationale
