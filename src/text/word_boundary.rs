//! ISO 32000-1:2008 Section 9.4.4 Word Boundary Detection
//!
//! This module implements specification-compliant word boundary detection for PDF text extraction.
//! Word boundaries are identified through multiple mechanisms defined in the PDF specification:
//!
//! 1. **TJ Array Offsets** (Section 9.4.4): Character-level spacing information from text positioning
//! 2. **Geometric Positioning** (Section 9.4): Layout-based word breaking through character positions
//! 3. **Space Characters** (Section 5.3.2): Explicit word separators (U+0020 and variants)
//! 4. **Font Metrics** (Section 9.3): Character width, font size, and scaling adjustments
//! 5. **Script-Aware Detection**: CJK text, custom encodings, and special characters
//!
//! # Specification References
//!
//! - ISO 32000-1:2008 Section 9.4: Text Objects
//! - ISO 32000-1:2008 Section 9.4.3: Text Positioning Operators
//! - ISO 32000-1:2008 Section 9.4.4: Text Objects and Word Spacing
//! - ISO 32000-1:2008 Section 9.3: Text State Parameters (Tc, Tw, Tz, TL)
//! - ISO 32000-1:2008 Section 9.6-9.8: Font Metrics

/// Information about a character in the text stream for boundary detection.
///
/// This type captures all the information needed to determine word boundaries
/// per PDF specification Section 9.4.4.
#[derive(Clone, Debug)]
pub struct CharacterInfo {
    /// Unicode code point of the character
    pub code: u32,

    /// Glyph ID in the font (if available)
    pub glyph_id: Option<u16>,

    /// Character width in text space units (thousandths of em)
    pub width: f32,

    /// X position (horizontal) in text space
    pub x_position: f32,

    /// TJ array offset value (in thousandths of em) - negative = extra space
    /// Per spec: Negative values in TJ array increase spacing between characters
    pub tj_offset: Option<i32>,

    /// Current font size in points
    pub font_size: f32,
}

/// Context information for word boundary detection.
///
/// Provides the font metrics and text state parameters that influence
/// how word boundaries are determined (per Section 9.3).
#[derive(Clone, Debug)]
pub struct BoundaryContext {
    /// Font size (Tf parameter in text state)
    pub font_size: f32,

    /// Horizontal scaling percentage (Tz parameter, default 100.0)
    pub horizontal_scaling: f32,

    /// Word spacing adjustment (Tw parameter, added after space character)
    pub word_spacing: f32,

    /// Character spacing adjustment (Tc parameter, added after every character)
    pub char_spacing: f32,
}

impl BoundaryContext {
    /// Create a new boundary context with default text state parameters.
    pub fn new(font_size: f32) -> Self {
        Self {
            font_size,
            horizontal_scaling: 100.0,
            word_spacing: 0.0,
            char_spacing: 0.0,
        }
    }

    /// Get the effective font size accounting for horizontal scaling
    fn effective_font_size(&self) -> f32 {
        self.font_size * (self.horizontal_scaling / 100.0)
    }
}

/// Main word boundary detection engine.
///
/// Implements the specification-compliant word boundary detection algorithm
/// that considers TJ offsets, geometric spacing, and font metrics.
#[derive(Debug)]
pub struct WordBoundaryDetector {
    /// Threshold for TJ offset values that indicate word boundaries
    /// Default: -100 (representing 0.1em in thousand-units of em)
    tj_offset_threshold: i32,

    /// Ratio of font size to use as geometric gap threshold
    /// Default: 0.3 (30% of font size indicates a word boundary)
    geometric_gap_ratio: f32,

    /// Enable CJK-aware boundary detection
    cjk_enabled: bool,
}

impl Default for WordBoundaryDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl WordBoundaryDetector {
    /// Create a new word boundary detector with default settings.
    pub fn new() -> Self {
        Self {
            tj_offset_threshold: -100,
            // Geometric gap threshold: 80% of font size
            // This is conservative enough to avoid false positives from normal character spacing
            // but sensitive enough to detect actual word breaks
            geometric_gap_ratio: 0.8,
            cjk_enabled: true,
        }
    }

    /// Set the TJ offset threshold for boundary detection.
    ///
    /// Negative values in TJ arrays that are more negative than this threshold
    /// are considered word boundaries. Default: -100
    pub fn with_tj_threshold(mut self, threshold: i32) -> Self {
        self.tj_offset_threshold = threshold;
        self
    }

    /// Set the geometric gap ratio as a fraction of font size.
    ///
    /// Gaps between characters larger than (font_size * ratio) are considered
    /// word boundaries. Default: 0.3
    pub fn with_geometric_gap_ratio(mut self, ratio: f32) -> Self {
        self.geometric_gap_ratio = ratio;
        self
    }

    /// Enable or disable CJK-aware word boundary detection.
    pub fn with_cjk_enabled(mut self, enabled: bool) -> Self {
        self.cjk_enabled = enabled;
        self
    }

    /// Detect word boundaries in a character stream.
    ///
    /// Returns a vector of indices where word boundaries occur.
    /// A boundary at index `i` means there is a word break between
    /// characters at indices `i-1` and `i`.
    ///
    /// Per ISO 32000-1:2008 Section 9.4.4, word boundaries are determined by:
    /// 1. Space characters (U+0020, U+200B)
    /// 2. TJ array offset signals (negative values below threshold)
    /// 3. Geometric gaps exceeding font-size relative threshold
    /// 4. CJK character transitions
    ///
    /// # Arguments
    ///
    /// * `characters` - Sequence of characters with positioning information
    /// * `context` - Font metrics and text state parameters
    ///
    /// # Returns
    ///
    /// Vector of indices where word boundaries occur (between characters)
    pub fn detect_word_boundaries(
        &self,
        characters: &[CharacterInfo],
        context: &BoundaryContext,
    ) -> Vec<usize> {
        if characters.is_empty() {
            return Vec::new();
        }

        let mut boundaries = Vec::new();

        for i in 1..characters.len() {
            let prev_char = &characters[i - 1];
            let curr_char = &characters[i];

            if self.is_word_boundary(prev_char, curr_char, context) {
                boundaries.push(i);
            }
        }

        boundaries
    }

    /// Determine if a word boundary exists between two consecutive characters.
    ///
    /// Implements the specification rules per ISO 32000-1:2008 Section 9.4.4:
    ///
    /// 1. **Space characters** (U+0020, U+200B): Always create boundaries
    /// 2. **TJ array offsets**: Negative values below threshold indicate spacing
    /// 3. **Geometric gaps**: Gaps larger than font-size-relative threshold
    /// 4. **CJK characters**: Each non-punctuation CJK character creates boundary
    ///
    /// # Arguments
    ///
    /// * `prev_char` - Previous character in the stream
    /// * `curr_char` - Current character
    /// * `context` - Font metrics and text state
    ///
    /// # Returns
    ///
    /// `true` if a word boundary should be placed between these characters
    fn is_word_boundary(
        &self,
        prev_char: &CharacterInfo,
        curr_char: &CharacterInfo,
        context: &BoundaryContext,
    ) -> bool {
        // Rule 1: ASCII space (U+0020) or zero-width space (U+200B)
        if prev_char.code == 0x20 || prev_char.code == 0x200B {
            return true;
        }

        // Rule 2: TJ array offset signals explicit spacing
        if let Some(tj_offset) = prev_char.tj_offset {
            if tj_offset < self.tj_offset_threshold {
                return true;
            }
        }

        // Rule 3: Geometric spacing detection
        if self.has_significant_geometric_gap(prev_char, curr_char, context) {
            return true;
        }

        // Rule 4: CJK character boundaries (if enabled)
        if self.cjk_enabled
            && self.is_cjk_character(prev_char.code)
            && !self.is_cjk_punctuation(prev_char.code)
        {
            return true;
        }

        false
    }

    /// Check if there is a significant geometric gap between two characters.
    ///
    /// Per Section 9.4, character positions and widths determine visual spacing.
    /// A gap larger than the threshold (font_size * ratio) indicates a word boundary.
    fn has_significant_geometric_gap(
        &self,
        prev_char: &CharacterInfo,
        curr_char: &CharacterInfo,
        context: &BoundaryContext,
    ) -> bool {
        // Calculate the expected end position of previous character
        let prev_end_x = prev_char.x_position + prev_char.width;

        // Calculate actual gap
        let gap = curr_char.x_position - prev_end_x;

        // Threshold is relative to font size (accounting for horizontal scaling)
        let threshold = context.effective_font_size() * self.geometric_gap_ratio;

        gap > threshold
    }

    /// Check if a character code represents a CJK (Chinese/Japanese/Korean) character.
    ///
    /// CJK Unicode ranges per Unicode Standard:
    /// - CJK Unified Ideographs: U+4E00-U+9FFF
    /// - CJK Unified Ideographs Extension A: U+3400-U+4DBF
    /// - CJK Unified Ideographs Extension B and beyond: higher ranges
    /// - Hiragana: U+3040-U+309F
    /// - Katakana: U+30A0-U+30FF
    fn is_cjk_character(&self, code: u32) -> bool {
        matches!(
            code,
            0x3040..=0x309F   // Hiragana
            | 0x30A0..=0x30FF // Katakana
            | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0x20000..=0x2A6DF // CJK Unified Ideographs Extension B
            | 0x2A700..=0x2B73F // CJK Unified Ideographs Extension C
            | 0x2B740..=0x2B81F // CJK Unified Ideographs Extension D
            | 0x2B820..=0x2CEAF // CJK Unified Ideographs Extension E
            | 0x2CEB0..=0x2EBEF // CJK Unified Ideographs Extension F
        )
    }

    /// Check if a character is CJK punctuation that attaches to words.
    ///
    /// CJK punctuation like ideographic commas and periods attach to the preceding
    /// word and should not create boundaries.
    fn is_cjk_punctuation(&self, code: u32) -> bool {
        matches!(
            code,
            0x3001 // IDEOGRAPHIC COMMA
            | 0x3002 // IDEOGRAPHIC FULL STOP
            | 0x3008 // LEFT ANGLE BRACKET
            | 0x3009 // RIGHT ANGLE BRACKET
            | 0x300A // LEFT DOUBLE ANGLE BRACKET
            | 0x300B // RIGHT DOUBLE ANGLE BRACKET
            | 0x300C // LEFT CORNER BRACKET
            | 0x300D // RIGHT CORNER BRACKET
            | 0x300E // LEFT WHITE CORNER BRACKET
            | 0x300F // RIGHT WHITE CORNER BRACKET
            | 0x3010 // LEFT BLACK LENTICULAR BRACKET
            | 0x3011 // RIGHT BLACK LENTICULAR BRACKET
            | 0x3014 // LEFT TORTOISE SHELL BRACKET
            | 0x3015 // RIGHT TORTOISE SHELL BRACKET
        )
    }
}

/// Detect word boundaries in a character stream.
///
/// This is a convenience function that creates a detector with default settings
/// and performs boundary detection in one call.
///
/// # Arguments
///
/// * `characters` - Sequence of characters with positioning information
/// * `context` - Font metrics and text state parameters
///
/// # Returns
///
/// Vector of indices where word boundaries occur
pub fn detect_word_boundaries(
    characters: &[CharacterInfo],
    context: &BoundaryContext,
) -> Vec<usize> {
    let detector = WordBoundaryDetector::new();
    detector.detect_word_boundaries(characters, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_space_detection() {
        let characters = vec![
            CharacterInfo {
                code: 0x48,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'e'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
            }, // SPACE
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(6),
                width: 0.7,
                x_position: 16.2,
                tj_offset: None,
                font_size: 12.0,
            }, // 'W'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Space character at index 2 creates boundary at index 3
        assert!(boundaries.contains(&3));
    }

    #[test]
    fn test_tj_offset_threshold() {
        let characters = vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'T'
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 6.0,
                tj_offset: Some(-200),
                font_size: 12.0,
            }, // '-' with large negative offset
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'o'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // TJ offset at index 1 creates boundary at index 2
        assert!(boundaries.contains(&2));
    }

    #[test]
    fn test_geometric_gap_detection() {
        let characters = vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'T'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'e'
            CharacterInfo {
                code: 0x78,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
            }, // 'x'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
            }, // 't'
            // Gap of ~11.1 units (much larger than threshold ~3.6)
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(5),
                width: 0.5,
                x_position: 27.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'B'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Gap between 't' (ends at 15.9) and 'B' (at 27.0) is 11.1 units > threshold (3.6)
        // This creates a boundary at index 4 (the 'B' character)
        assert!(boundaries.contains(&4), "Expected boundary at index 4, got: {:?}", boundaries);
    }

    #[test]
    fn test_cjk_character_boundaries() {
        let characters = vec![
            CharacterInfo {
                code: 0x4E2D,
                glyph_id: Some(1),
                width: 1.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // CJK UNIFIED IDEOGRAPH
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(2),
                width: 1.0,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
            }, // CJK UNIFIED IDEOGRAPH
            CharacterInfo {
                code: 0x5B57,
                glyph_id: Some(3),
                width: 1.0,
                x_position: 24.0,
                tj_offset: None,
                font_size: 12.0,
            }, // CJK UNIFIED IDEOGRAPH
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Each CJK character creates a boundary after it
        // Character 0 -> boundary at 1, Character 1 -> boundary at 2
        assert!(boundaries.contains(&1), "Expected boundary at index 1");
        assert!(boundaries.contains(&2), "Expected boundary at index 2");
    }

    #[test]
    fn test_zero_width_space() {
        let characters = vec![
            CharacterInfo {
                code: 0x6E,
                glyph_id: Some(1),
                width: 0.4,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'n'
            CharacterInfo {
                code: 0x200B,
                glyph_id: Some(2),
                width: 0.0,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
            }, // ZERO WIDTH SPACE
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
            }, // 'r'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Zero-width space creates boundary
        assert!(boundaries.contains(&2));
    }

    #[test]
    fn test_horizontal_scaling_affects_gap_threshold() {
        // Create a gap that's on the threshold boundary
        // Gap = 7.5 units
        // At 100% scaling (font size 12): threshold = 12 * 0.8 = 9.6, gap < threshold = no boundary
        // At 75% scaling (font size 9): threshold = 9 * 0.8 = 7.2, gap > threshold = boundary!
        let characters = vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'A' ends at 0.5
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.5,
                x_position: 8.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'B' starts at 8.0
        ];

        // With 100% scaling, gap (7.5) doesn't exceed threshold (9.6)
        let mut context = BoundaryContext::new(12.0);
        context.horizontal_scaling = 100.0;
        let boundaries_normal =
            WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // With 75% scaling, gap (7.5) exceeds threshold (7.2)
        context.horizontal_scaling = 75.0;
        let boundaries_scaled =
            WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Scaling affects the effective threshold, so results should differ
        // Normal: no boundary, Scaled: boundary at index 1
        assert!(boundaries_normal.is_empty(), "Should have no boundaries at 100% scaling");
        assert!(boundaries_scaled.contains(&1), "Should have boundary at 75% scaling");
    }

    #[test]
    fn test_detect_word_boundaries_ascii_space() {
        // Test that ASCII space creates word boundary
        let characters = vec![
            CharacterInfo {
                code: 0x48,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'e'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
            }, // SPACE
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(6),
                width: 0.7,
                x_position: 16.2,
                tj_offset: None,
                font_size: 12.0,
            }, // 'W'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Space character at index 2 creates boundary at index 3
        assert!(boundaries.contains(&3), "Should have boundary after space");
    }

    #[test]
    fn test_detect_word_boundaries_tj_offset() {
        // Test that large negative TJ offset creates boundary
        let characters = vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'T'
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 6.0,
                tj_offset: Some(-200),
                font_size: 12.0,
            }, // '-' with large negative offset
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
            }, // 'o'
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // TJ offset at index 1 creates boundary at index 2
        assert!(boundaries.contains(&2), "Should have boundary after large TJ offset");
    }

    #[test]
    fn test_detect_word_boundaries_cjk() {
        // Test that CJK characters create boundaries
        let characters = vec![
            CharacterInfo {
                code: 0x4E2D,
                glyph_id: Some(1),
                width: 1.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
            }, // CJK character
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(2),
                width: 1.0,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
            }, // CJK character
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Each CJK character should create a boundary
        assert!(boundaries.contains(&1), "Should have boundary after first CJK character");
    }
}
