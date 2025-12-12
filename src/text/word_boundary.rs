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

use crate::text::cjk_punctuation;
use crate::text::complex_script_detector::{
    ComplexScript, detect_complex_script, handle_devanagari_boundary, handle_indic_boundary,
    handle_khmer_boundary, handle_thai_boundary,
};
use crate::text::rtl_detector::should_split_at_rtl_boundary;
use crate::text::script_detector::{
    DocumentLanguage, detect_cjk_script, handle_japanese_text, handle_korean_text,
    should_split_on_script_transition,
};

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

    /// Whether this character is a ligature (U+FB00-U+FB04)
    /// Week 2 Day 6: Ligature Expansion Enhancement (2A)
    pub is_ligature: bool,

    /// Original ligature character if this was split from a ligature
    /// Used for debugging and tracking ligature expansion
    /// Week 2 Day 6: Ligature Expansion Enhancement (2A)
    pub original_ligature: Option<char>,

    /// Whether this character is protected from word boundary splitting
    /// Week 2 Day 7: Email/URL Pattern Preservation (2C)
    ///
    /// When true, word boundary detection will skip creating boundaries
    /// before or after this character. Used to preserve email addresses
    /// (user@example.com) and URLs (http://example.com) as single tokens.
    pub protected_from_split: bool,
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

/// Document script profile for optimization.
///
/// OPTIMIZATION (Issue #1 fix): Detect document primary script once,
/// then skip unnecessary script detection functions for faster boundary detection.
///
/// When documents contain only Latin text, we skip RTL and CJK detection entirely.
/// When documents are CJK-dominant, we skip RTL detection.
/// This reduces function call overhead from millions per batch to thousands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentScript {
    /// Latin-only document (ASCII + extended Latin)
    /// Fast path: only check space, TJ offset, geometric gap
    Latin,

    /// CJK-dominant document (Chinese, Japanese, Korean)
    /// Skip RTL detection, use optimized CJK path
    CJK,

    /// Right-to-left dominant (Arabic, Hebrew)
    /// Skip CJK detection, use optimized RTL path
    RTL,

    /// Complex scripts (Devanagari, Thai, Khmer, etc.)
    /// Use specialized complex script detection
    Complex,

    /// Mixed scripts or unknown
    /// Check all detection functions (slowest path)
    Mixed,
}

impl DocumentScript {
    /// Detect document script profile by sampling first 1000 characters.
    ///
    /// This optimization reduces boundary detection overhead by skipping
    /// unnecessary script detection for documents with known script profiles.
    ///
    /// PERFORMANCE: O(min(n, 1000)) sampling, executed once per extraction
    pub fn detect_from_characters(characters: &[CharacterInfo]) -> Self {
        if characters.is_empty() {
            return Self::Latin; // Default to Latin for empty documents
        }

        let mut has_rtl = false;
        let mut has_cjk = false;
        let mut has_complex = false;
        let sample_size = characters.len().min(1000);

        // Sample first 1000 characters to classify document
        for ch in &characters[..sample_size] {
            // Check for RTL (fast range check)
            if (0x0590..=0x08FF).contains(&ch.code) || (0xFB1D..=0xFDFF).contains(&ch.code) {
                has_rtl = true;
            }

            // Check for CJK (fast range checks - common ranges first)
            if (0x4E00..=0x9FFF).contains(&ch.code) // Han
                || (0x3040..=0x309F).contains(&ch.code) // Hiragana
                || (0x30A0..=0x30FF).contains(&ch.code) // Katakana
                || (0xAC00..=0xD7AF).contains(&ch.code)
            {
                // Hangul
                has_cjk = true;
            }

            // Check for complex scripts
            if (0x0900..=0x097F).contains(&ch.code) // Devanagari
                || (0x0E00..=0x0E7F).contains(&ch.code) // Thai
                || (0x1780..=0x17FF).contains(&ch.code)
            {
                // Khmer
                has_complex = true;
            }
        }

        // Decision tree: classify based on what we found
        match (has_rtl, has_cjk, has_complex) {
            (false, false, false) => Self::Latin, // Pure Latin (fast path)
            (false, true, _) => Self::CJK,        // CJK-dominant (skip RTL)
            (true, false, _) => Self::RTL,        // RTL-dominant (skip CJK)
            (_, _, true) => Self::Complex,        // Complex scripts present
            _ => Self::Mixed,                     // Mixed scripts
        }
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

    /// Enable script-aware transition detection
    detect_script_transitions: bool,

    /// Document language context (if known)
    document_language: Option<DocumentLanguage>,

    /// Detected document script profile (Issue #1 optimization)
    /// Cached at detector creation to skip unnecessary detection functions
    primary_script: DocumentScript,
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
            detect_script_transitions: true,
            document_language: None,
            primary_script: DocumentScript::Mixed, // Default to Mixed, will be set by caller
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

    /// Enable or disable script-aware transition detection.
    ///
    /// When enabled, the detector will analyze script transitions (e.g., Hiragana→Katakana)
    /// and apply language-specific rules for word boundaries.
    pub fn with_script_detection(mut self, enabled: bool) -> Self {
        self.detect_script_transitions = enabled;
        self
    }

    /// Set the document language context.
    ///
    /// This helps apply appropriate script transition rules:
    /// - Japanese: Allow Han↔Kana transitions
    /// - Korean: Allow Hangul↔Hanja transitions
    /// - Chinese: Use conservative Han character boundaries
    pub fn with_document_language(mut self, lang: DocumentLanguage) -> Self {
        self.document_language = Some(lang);
        self
    }

    /// Set the document script profile (Issue #1 optimization).
    ///
    /// When set, the detector will skip unnecessary script detection functions
    /// for documents with known script profiles, significantly improving performance.
    pub fn with_document_script(mut self, script: DocumentScript) -> Self {
        self.primary_script = script;
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
    /// 4. **CJK script transitions**: Script-aware word boundaries
    /// 5. **CJK characters**: Each non-punctuation CJK character creates boundary (legacy)
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
        // Week 2 Day 7 (2C): Skip boundaries in protected contexts (emails, URLs)
        if prev_char.protected_from_split || curr_char.protected_from_split {
            return false;
        }

        // Rule 1: ASCII space (U+0020) or zero-width space (U+200B)
        if prev_char.code == 0x20 || prev_char.code == 0x200B {
            return true;
        }

        // OPTIMIZATION (Issue #1): Use script-aware dispatch to avoid unnecessary function calls
        // This reduces millions of function calls per batch by skipping detection for known script types
        match self.primary_script {
            // Fast path: Latin-only documents - skip RTL and CJK detection entirely
            DocumentScript::Latin => self.is_word_boundary_basic(prev_char, curr_char, context),

            // CJK path: Skip RTL detection, use only CJK detection
            DocumentScript::CJK => {
                if self.detect_script_transitions {
                    if let Some(decision) = self.should_split_at_cjk_boundary(prev_char, curr_char)
                    {
                        return decision;
                    }
                }
                self.is_word_boundary_basic(prev_char, curr_char, context)
            },

            // RTL path: Skip CJK detection, use only RTL detection
            DocumentScript::RTL => {
                if let Some(decision) =
                    should_split_at_rtl_boundary(prev_char, curr_char, Some(context))
                {
                    return decision;
                }
                self.is_word_boundary_basic(prev_char, curr_char, context)
            },

            // Complex script path: Use complex script detection, skip RTL/CJK
            DocumentScript::Complex => {
                if let Some(decision) =
                    self.should_split_at_complex_script_boundary(prev_char, curr_char)
                {
                    return decision;
                }
                self.is_word_boundary_basic(prev_char, curr_char, context)
            },

            // Mixed path: Check all detection functions (original behavior)
            DocumentScript::Mixed => {
                // Week 2 Day 10: RTL (Arabic/Hebrew) boundary detection
                if let Some(decision) =
                    should_split_at_rtl_boundary(prev_char, curr_char, Some(context))
                {
                    return decision;
                }

                // Week 2 Days 8-9 (3A-3D): CJK script-aware boundaries
                if self.detect_script_transitions {
                    if let Some(decision) = self.should_split_at_cjk_boundary(prev_char, curr_char)
                    {
                        return decision;
                    }
                }

                // Week 3 Days 11-12: Complex script boundary detection
                if let Some(decision) =
                    self.should_split_at_complex_script_boundary(prev_char, curr_char)
                {
                    return decision;
                }

                self.is_word_boundary_basic(prev_char, curr_char, context)
            },
        }
    }

    /// Basic boundary detection used by all script paths.
    ///
    /// This contains the core TJ offset and geometric gap checks
    /// that apply to all scripts.
    fn is_word_boundary_basic(
        &self,
        prev_char: &CharacterInfo,
        curr_char: &CharacterInfo,
        context: &BoundaryContext,
    ) -> bool {
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

        // Rule 4: CJK character boundaries (legacy, if enabled but script detection disabled)
        if self.cjk_enabled
            && !self.detect_script_transitions
            && self.is_cjk_character(prev_char.code)
            && !self.is_cjk_punctuation(prev_char.code)
        {
            return true;
        }

        false
    }

    /// Determine if a complex script boundary should be created.
    ///
    /// This implements Week 3 Days 11-12 Complex Script support:
    /// - Devanagari virama and matras
    /// - Thai tone marks and vowel modifiers
    /// - Khmer COENG and vowels
    /// - Indic scripts (Tamil, Telugu, Kannada, Malayalam) diacritics
    ///
    /// # Arguments
    ///
    /// * `prev_char` - Previous character information
    /// * `curr_char` - Current character information
    ///
    /// # Returns
    ///
    /// - `Some(true)` - Must create boundary
    /// - `Some(false)` - Must not create boundary
    /// - `None` - Use other signals (TJ offset, geometry)
    fn should_split_at_complex_script_boundary(
        &self,
        prev_char: &CharacterInfo,
        curr_char: &CharacterInfo,
    ) -> Option<bool> {
        let prev_script = detect_complex_script(prev_char.code);
        let curr_script = detect_complex_script(curr_char.code);

        // If neither is complex script, not our concern
        if prev_script.is_none() && curr_script.is_none() {
            return None;
        }

        // Apply script-specific rules based on which scripts are involved
        match (prev_script, curr_script) {
            // Devanagari boundaries
            (Some(ComplexScript::Devanagari), _) | (_, Some(ComplexScript::Devanagari)) => {
                handle_devanagari_boundary(prev_char, curr_char)
            },
            // Thai boundaries
            (Some(ComplexScript::Thai), _) | (_, Some(ComplexScript::Thai)) => {
                handle_thai_boundary(prev_char, curr_char)
            },
            // Khmer boundaries
            (Some(ComplexScript::Khmer), _) | (_, Some(ComplexScript::Khmer)) => {
                handle_khmer_boundary(prev_char, curr_char)
            },
            // South Asian Indic scripts (Tamil, Telugu, Kannada, Malayalam)
            (Some(ComplexScript::Tamil), _)
            | (_, Some(ComplexScript::Tamil))
            | (Some(ComplexScript::Telugu), _)
            | (_, Some(ComplexScript::Telugu))
            | (Some(ComplexScript::Kannada), _)
            | (_, Some(ComplexScript::Kannada))
            | (Some(ComplexScript::Malayalam), _)
            | (_, Some(ComplexScript::Malayalam))
            | (Some(ComplexScript::Bengali), _)
            | (_, Some(ComplexScript::Bengali)) => handle_indic_boundary(prev_char, curr_char),
            // Other complex scripts - use conservative default (let other signals decide)
            _ => None,
        }
    }

    /// Determine if a CJK boundary should be created based on script analysis.
    ///
    /// This implements Week 2 Days 8-9 CJK script support:
    /// - CJK punctuation detection
    /// - Script type detection
    /// - Language-specific transition rules
    /// - Japanese modifier handling
    ///
    /// # Arguments
    ///
    /// * `prev_char` - Previous character information
    /// * `curr_char` - Current character information
    ///
    /// # Returns
    ///
    /// - `Some(true)` - Must create boundary
    /// - `Some(false)` - Must not create boundary
    /// - `None` - Use other signals (TJ offset, geometry)
    fn should_split_at_cjk_boundary(
        &self,
        prev_char: &CharacterInfo,
        curr_char: &CharacterInfo,
    ) -> Option<bool> {
        // Check CJK punctuation (always creates boundary with high confidence)
        let prev_punctuation_score =
            cjk_punctuation::get_cjk_punctuation_boundary_score(prev_char.code);
        if prev_punctuation_score >= 0.9 {
            // Sentence-ending and enumeration punctuation create boundaries
            return Some(true);
        }

        // Detect scripts for both characters
        let prev_script = detect_cjk_script(prev_char.code);
        let curr_script = detect_cjk_script(curr_char.code);

        // If neither character is CJK, not our concern
        if prev_script.is_none() && curr_script.is_none() {
            return None;
        }

        // Apply language-specific rules
        match self.document_language {
            Some(DocumentLanguage::Japanese) => {
                handle_japanese_text(prev_char, curr_char, prev_script, curr_script)
            },
            Some(DocumentLanguage::Korean) => {
                handle_korean_text(prev_char, curr_char, prev_script, curr_script)
            },
            Some(DocumentLanguage::Chinese) | None => {
                // Chinese or unknown: use script transition analysis
                should_split_on_script_transition(prev_script, curr_script, self.document_language)
            },
        }
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(6),
                width: 0.7,
                x_position: 16.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 6.0,
                tj_offset: Some(-200),
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // '-' with large negative offset
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x78,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'x'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
            // Gap of ~11.1 units (much larger than threshold ~3.6)
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(5),
                width: 0.5,
                x_position: 27.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK UNIFIED IDEOGRAPH
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(2),
                width: 1.0,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK UNIFIED IDEOGRAPH
            CharacterInfo {
                code: 0x5B57,
                glyph_id: Some(3),
                width: 1.0,
                x_position: 24.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'n'
            CharacterInfo {
                code: 0x200B,
                glyph_id: Some(2),
                width: 0.0,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // ZERO WIDTH SPACE
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'A' ends at 0.5
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.5,
                x_position: 8.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(6),
                width: 0.7,
                x_position: 16.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 6.0,
                tj_offset: Some(-200),
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // '-' with large negative offset
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
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
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK character
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(2),
                width: 1.0,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK character
        ];

        let context = BoundaryContext::new(12.0);
        let boundaries = WordBoundaryDetector::new().detect_word_boundaries(&characters, &context);

        // Each CJK character should create a boundary
        assert!(boundaries.contains(&1), "Should have boundary after first CJK character");
    }
}
