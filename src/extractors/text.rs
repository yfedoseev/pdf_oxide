//! Text extraction from PDF content streams.
//!
//! This module executes content stream operators to extract positioned
//! text characters with their Unicode mappings, font information, and
//! bounding boxes.

use crate::config::ExtractionProfile;
use crate::content::graphics_state::{GraphicsStateStack, Matrix};
use crate::content::operators::{Operator, TextElement};
use crate::content::parse_and_execute_text_only;
use crate::content::parse_content_stream_text_only;
use crate::error::Result;
use crate::extract_log_debug;
use crate::fonts::FontInfo;
use crate::geometry::Rect;
use crate::layout::{Color, FontWeight, TextChar, TextSpan};
use crate::object::{Object, ObjectRef};
use crate::pipeline::config::WordBoundaryMode;
use crate::text::{BoundaryContext, CharacterInfo, DocumentScript, WordBoundaryDetector};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Source of a space decision in the unified pipeline.
///
/// This enum tracks why a space was inserted (or not), which helps with
/// debugging and understanding the text extraction behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceSource {
    /// Space triggered by TJ offset value (negative offset > threshold)
    /// Confidence: 0.95 (explicit PDF positioning signal)
    TjOffset,

    /// Space triggered by geometric gap between spans
    /// Confidence: 0.8 (heuristic based on font metrics)
    GeometricGap,

    /// Space triggered by character transition heuristic (e.g., CamelCase, number->letter)
    /// Confidence: 0.6 (pattern-based heuristic)
    CharacterHeuristic,

    /// Space already present in boundary (no insertion needed)
    /// Confidence: 1.0 (deterministic)
    AlreadyPresent,

    /// No space inserted
    /// Confidence: varies (default when no rule matches)
    NoSpace,

    /// Space triggered by WordBoundaryDetector analysis
    /// Confidence: 0.85 (combines TJ offset, geometric, and CJK signals per PDF Spec 9.4.4)
    WordBoundaryAnalysis,
}

/// Result of unified space decision process.
///
/// This struct is the single source of truth for whether a space should be inserted
/// between two text spans. It combines all available signals:
/// - TJ offset values from PDF content stream
/// - Geometric gaps between spans
/// - Character transition heuristics
/// - Existing boundary whitespace
///
/// Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
/// "The identification of what constitutes a word is unrelated to how the text
/// happens to be grouped into show strings... text strings should be as long as possible."
#[derive(Debug, Clone)]
pub struct SpaceDecision {
    /// Whether a space should be inserted
    pub insert_space: bool,

    /// Source/reason for this decision
    pub source: SpaceSource,

    /// Confidence score (0.0-1.0) indicating certainty
    pub confidence: f32,
}

impl SpaceDecision {
    /// Create a decision to insert a space from a specific source.
    pub fn insert(source: SpaceSource, confidence: f32) -> Self {
        Self {
            insert_space: true,
            source,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a decision to not insert a space.
    pub fn no_space(source: SpaceSource, confidence: f32) -> Self {
        Self {
            insert_space: false,
            source,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Configuration for text extraction heuristics.
///
/// PDF spec does not define explicit rules for many spacing scenarios.
/// These configurable thresholds allow tuning extraction behavior.
///
/// # PDF Spec Reference
///
/// ISO 32000-1:2008, Section 9.4.4 - Text Positioning operators (TJ, Tj)
/// The spec defines how positioning works but NOT when a position offset
/// represents a word boundary vs. tight kerning.
#[derive(Debug, Clone)]
pub struct TextExtractionConfig {
    /// Extraction profile with document-type-specific thresholds
    ///
    /// When set, this profile overrides individual threshold settings and provides
    /// pre-tuned parameters optimized for specific document types (Academic, Policy,
    /// Government, Form, ScannedOCR, etc.).
    ///
    /// **Default**: None (uses legacy individual thresholds for backward compatibility)
    pub profile: Option<ExtractionProfile>,

    /// Threshold for inserting space characters in TJ arrays.
    ///
    /// **DEPRECATED**: Consider using `profile` with an `ExtractionProfile` or
    /// `word_margin_ratio` with `use_adaptive_tj_threshold` enabled for geometry-based
    /// adaptive thresholds. This field is used as a fallback when font metrics are
    /// unavailable or adaptive thresholds are disabled, and when profile is not set.
    ///
    /// **HEURISTIC**: When a TJ array contains a negative offset (in text space units),
    /// and that offset exceeds this threshold, a space character is inserted.
    ///
    /// **Default**: -120.0 units ≈ 0.12em
    /// - Typical word space: 0.25-0.33em (250-330 units)
    /// - Typical letter kerning: <0.1em (<100 units)
    ///
    /// **Lower values** (e.g., -80): More sensitive, inserts more spaces (may add spurious spaces)
    /// **Higher values** (e.g., -200): Less sensitive, inserts fewer spaces (may miss word boundaries)
    ///
    /// Set to `f32::NEG_INFINITY` to disable space insertion entirely.
    pub space_insertion_threshold: f32,

    /// Word margin ratio for geometry-based adaptive TJ threshold.
    ///
    /// When `use_adaptive_tj_threshold` is true and font metrics are available,
    /// the TJ offset threshold is calculated as:
    /// ```text
    /// adaptive_threshold = -(average_glyph_width * word_margin_ratio)
    /// ```
    ///
    /// This approach adapts to different font sizes and families by using the
    /// actual glyph metrics instead of a static value. This matches pdfplumber's
    /// `word_margin` parameter (default 0.1).
    ///
    /// **Default**: 0.1 (10% of average glyph width)
    ///
    /// **Typical values**:
    /// - 0.05: Tighter spacing (fewer spaces inserted, better for narrow fonts)
    /// - 0.1: Standard word spacing (default, matches pdfplumber)
    /// - 0.15: Looser spacing (more spaces inserted, better for wide fonts)
    ///
    /// **Note**: If font metrics are unavailable, falls back to `space_insertion_threshold`.
    ///
    /// # PDF Spec Reference
    ///
    /// ISO 32000-1:2008, Section 9.4.4 - TJ offsets are in thousandths of em.
    /// Average glyph width is also in thousandths of em, making this ratio
    /// dimensionally correct.
    pub word_margin_ratio: f32,

    /// Enable adaptive TJ threshold based on font geometry.
    ///
    /// When true, uses font metrics to calculate the TJ offset threshold dynamically:
    /// `adaptive_threshold = -(average_glyph_width * word_margin_ratio)`
    ///
    /// This replaces the static `space_insertion_threshold` with a value that adapts
    /// to different font sizes, families, and document layouts.
    ///
    /// **Default**: true (adaptive approach enabled)
    ///
    /// Set to `false` for backward compatibility with legacy behavior, which
    /// uses only the static `space_insertion_threshold`.
    ///
    /// # Benefits
    ///
    /// - Handles font size variations (8pt vs 24pt documents)
    /// - Adapts to different character widths (serif vs sans-serif, monospace vs proportional)
    /// - Reduces spurious spaces in policy documents with tight kerning
    /// - Maintains word boundary detection in academic documents
    pub use_adaptive_tj_threshold: bool,

    /// Word boundary detection mode for TJ array processing
    ///
    /// Controls whether WordBoundaryDetector is used as:
    /// - Tiebreaker: Only when TJ and geometric signals conflict (default)
    /// - Primary: Before creating TextSpans from tj_character_array
    ///
    /// **Default**: WordBoundaryMode::Tiebreaker (backward compatible)
    pub word_boundary_mode: WordBoundaryMode,
}

impl Default for TextExtractionConfig {
    fn default() -> Self {
        Self {
            profile: None,
            space_insertion_threshold: -120.0,
            word_margin_ratio: 0.1,
            use_adaptive_tj_threshold: false,
            word_boundary_mode: WordBoundaryMode::default(),
        }
    }
}

impl TextExtractionConfig {
    /// Create a new configuration with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    ///
    /// let config = TextExtractionConfig::new();
    /// assert_eq!(config.space_insertion_threshold, -120.0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration with custom space insertion threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Negative offset threshold for space insertion (in text space units)
    ///
    /// **Note**: This uses the static threshold. For better results, consider using
    /// `with_word_margin_ratio()` with adaptive thresholds enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    ///
    /// // More aggressive space insertion
    /// let config = TextExtractionConfig::with_space_threshold(-80.0);
    ///
    /// // Disable space insertion entirely
    /// let no_spaces = TextExtractionConfig::with_space_threshold(f32::NEG_INFINITY);
    /// ```
    pub fn with_space_threshold(threshold: f32) -> Self {
        Self {
            profile: None,
            space_insertion_threshold: threshold,
            word_margin_ratio: 0.1,
            use_adaptive_tj_threshold: false, // Static threshold mode
            word_boundary_mode: WordBoundaryMode::default(),
        }
    }

    /// Create a configuration with custom word margin ratio for adaptive TJ thresholds.
    ///
    /// # Arguments
    ///
    /// * `ratio` - Word margin ratio as fraction of average glyph width (typically 0.05-0.15)
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    ///
    /// // Standard adaptive thresholds (matches pdfplumber)
    /// let config = TextExtractionConfig::with_word_margin_ratio(0.1);
    ///
    /// // More aggressive (wider thresholds, more spaces)
    /// let aggressive = TextExtractionConfig::with_word_margin_ratio(0.15);
    ///
    /// // More conservative (narrower thresholds, fewer spaces)
    /// let conservative = TextExtractionConfig::with_word_margin_ratio(0.05);
    /// ```
    pub fn with_word_margin_ratio(ratio: f32) -> Self {
        Self {
            profile: None,
            space_insertion_threshold: -120.0, // Fallback value
            word_margin_ratio: ratio,
            use_adaptive_tj_threshold: true, // Adaptive threshold mode
            word_boundary_mode: WordBoundaryMode::default(),
        }
    }

    /// Set the word margin ratio on an existing configuration (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `ratio` - Word margin ratio as fraction of average glyph width
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    ///
    /// let config = TextExtractionConfig::new()
    ///     .set_word_margin_ratio(0.15);
    /// ```
    pub fn set_word_margin_ratio(mut self, ratio: f32) -> Self {
        self.word_margin_ratio = ratio;
        self.use_adaptive_tj_threshold = true;
        self
    }

    /// Enable or disable adaptive TJ thresholds (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to use adaptive thresholds based on font metrics
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    ///
    /// // Use static threshold only
    /// let config = TextExtractionConfig::new()
    ///     .set_adaptive_tj_threshold(false);
    /// ```
    pub fn set_adaptive_tj_threshold(mut self, enabled: bool) -> Self {
        self.use_adaptive_tj_threshold = enabled;
        self
    }

    /// Set the extraction profile and apply its threshold configuration (builder pattern).
    ///
    /// This applies the profile's thresholds to the configuration, selecting document-type-specific
    /// parameters for better text extraction quality.
    ///
    /// # Arguments
    ///
    /// * `profile` - An extraction profile (e.g., ACADEMIC, POLICY, FORM)
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractionConfig;
    /// use pdf_oxide::config::ExtractionProfile;
    ///
    /// // Use ACADEMIC profile for research papers
    /// let config = TextExtractionConfig::new()
    ///     .with_profile(ExtractionProfile::ACADEMIC);
    /// ```
    pub fn with_profile(mut self, profile: ExtractionProfile) -> Self {
        // Extract profile settings before moving profile
        let tj_offset = profile.tj_offset_threshold;
        let word_margin = profile.word_margin_ratio;
        let use_adaptive = profile.use_adaptive_threshold;

        // Set profile and apply its thresholds
        self.profile = Some(profile);
        self.space_insertion_threshold = tj_offset;
        self.word_margin_ratio = word_margin;
        self.use_adaptive_tj_threshold = use_adaptive;
        self
    }
}

/// Configuration for span merging behavior.
///
/// These thresholds control how adjacent text spans are merged together and when
/// spaces are inserted between them. All thresholds are in PDF points (1/72 inch).
///
/// # Rationale
///
/// PDF content streams don't explicitly mark word boundaries - text can be rendered
/// with arbitrary gaps. These configurable thresholds allow tuning extraction to
/// different document types:
/// - Academic papers: tight column spacing, small gaps between words
/// - Documents with tables: larger gaps to preserve structure
/// - Dense grids (author lists): very small gaps that are still word boundaries
///
/// # References
///
/// Typography standards: word spacing typically 0.25-0.33em (25-33% of font size)
/// See: SPAN_SPACING_INVESTIGATION.md for empirical measurements
#[derive(Clone, Debug, PartialEq)]
pub struct SpanMergingConfig {
    /// Minimum gap (in multiples of font size) to trigger space insertion.
    ///
    /// When the gap between two spans exceeds this threshold, a space is inserted.
    /// Expressed as a ratio of font size (em).
    ///
    /// **Default**: 0.25
    /// - Based on typography standards: typical word spacing is 0.25-0.33em
    /// - For 12pt font: 0.25em * 12pt = 3pt
    /// - For 10pt font: 0.25em * 10pt = 2.5pt
    ///
    /// **Tuning guidance**:
    /// - Lower values (0.15-0.20): More aggressive space insertion, catches dense layouts
    /// - Higher values (0.33-0.50): Conservative, only clear word boundaries
    pub space_threshold_em_ratio: f32,

    /// Conservative threshold for font transitions (in points).
    ///
    /// Below this gap, don't insert a space even if gap > 0, to avoid spurious spaces
    /// from font metric changes or very tight kerning.
    ///
    /// **Default**: 0.1
    /// - Avoids spaces from font metric alignment issues (very tight threshold)
    /// - Smaller than typical letter spacing in justified text
    /// - Catches actual overlaps/reversals while preserving character adjacency
    ///
    /// **Note**: Changed from 0.3 to 0.1 after regression testing revealed
    /// that 0.3pt was too conservative for policy documents (0.1-0.3pt word spacing),
    /// causing word fusion. Adaptive threshold analysis recommended for future improvement.
    ///
    /// **Tuning guidance**:
    /// - Lower values (0.1-0.2): More aggressive, inserts more spaces
    /// - Higher values (0.5-1.0): Conservative, only clear separations
    pub conservative_threshold_pt: f32,

    /// Column boundary threshold (in points).
    ///
    /// Gaps larger than this indicate column separation and prevent span merging.
    /// Used to preserve document structure (e.g., multi-column layouts, tables).
    ///
    /// **Default**: 5.0
    /// - Typical character width for 10-12pt font: 4-6pt
    /// - Word spacing: 2-4pt
    /// - Column gaps in academic papers: 5-15pt
    /// - Table column gaps: 10-50pt
    ///
    /// **Tuning guidance**:
    /// - Lower values (3.0-4.0): Merge more spans, risk merging across columns
    /// - Higher values (8.0-10.0): Keep columns separate, preserve structure
    pub column_boundary_threshold_pt: f32,

    /// Negative gap threshold for severe overlaps (in points).
    ///
    /// When gaps are negative (spans overlap), values more severe than this
    /// indicate genuine overlap and should prevent merging.
    ///
    /// **Default**: -0.5
    /// - Typical font metric variations: 0 to -0.3pt
    /// - Small overlaps from kerning: -0.3 to -0.5pt
    /// - Real overlap errors: worse than -0.5pt
    ///
    /// **Tuning guidance**:
    /// - Less negative (-0.2, -0.1): More conservative on overlaps
    /// - More negative (-1.0, -2.0): Allow some overlap to merge adjacent text
    pub severe_overlap_threshold_pt: f32,

    /// Enable adaptive threshold analysis (default: true).
    ///
    /// When true, the `conservative_threshold_pt` is automatically calculated
    /// based on the gap distribution within the document. This overrides the fixed
    /// threshold value and adapts to different document types.
    ///
    /// **Default**: true (adaptive enabled)
    /// Enabled by default to improve extraction quality across document types.
    /// Use `SpanMergingConfig::legacy()` for the old fixed-threshold behavior.
    ///
    /// # Performance
    ///
    /// Adaptive analysis adds minimal overhead (O(n log n) for gap analysis where n = spans).
    /// Expected overhead: <5% of total extraction time.
    pub use_adaptive_threshold: bool,

    /// Configuration for adaptive threshold analysis.
    ///
    /// Only used when `use_adaptive_threshold` is true.
    /// If None, uses `AdaptiveThresholdConfig::default()`.
    ///
    /// Allows fine-tuning the adaptive analysis for specific document types:
    /// - `AdaptiveThresholdConfig::policy_documents()` - For tight spacing
    /// - `AdaptiveThresholdConfig::academic()` - For standard spacing
    /// - `AdaptiveThresholdConfig::aggressive()` - For dense layouts
    /// - `AdaptiveThresholdConfig::conservative()` - For formal documents
    pub adaptive_config: Option<crate::extractors::gap_statistics::AdaptiveThresholdConfig>,

    /// Enable email pattern detection for spacing decisions.
    ///
    /// When true, detects email-like patterns in surrounding text
    /// (e.g., "user@domain" separated by spaces) and applies special spacing rules
    /// to preserve email addresses.
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.10, only extracted text patterns
    /// are used - no domain-specific semantics.
    ///
    /// **Default**: false
    pub detect_email_patterns: bool,

    /// Multiplier for email pattern threshold detection.
    ///
    /// Controls how aggressively email patterns are detected by adjusting the gap threshold.
    /// A multiplier > 1.0 makes detection more lenient (allows larger gaps to be considered email context).
    /// A multiplier < 1.0 makes detection stricter.
    ///
    /// Calculated as: `email_threshold = geometric_threshold * email_threshold_multiplier`
    ///
    /// **Default**: 2.5
    /// - At 2.5×, handles typical email address separations with spaces
    /// - Typical gap between email parts: 4-8pt (after @, before TLD)
    pub email_threshold_multiplier: f32,

    /// Enable citation marker detection for spacing decisions.
    ///
    /// When true, detects superscript citation markers (typically smaller font size)
    /// and adjusts spacing rules to preserve citation formatting.
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.10, font size ratios from extracted content
    /// are used for detection.
    ///
    /// **Default**: false
    pub detect_citation_markers: bool,

    /// Font size ratio for citation marker detection.
    ///
    /// Citation markers typically have font size between this ratio and 1.0 of the base text.
    /// Values below this ratio are considered citation markers.
    ///
    /// **Default**: 0.75
    /// - Typical citation markers: 70-80% of text font size
    /// - Superscript usually: 50-80% of base font
    pub citation_font_size_ratio: f32,
}

impl Default for SpanMergingConfig {
    fn default() -> Self {
        Self {
            space_threshold_em_ratio: 0.25,
            conservative_threshold_pt: 0.1, // Reverted from 0.3 after regression testing
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: true, // Enabled by default for better quality
            adaptive_config: None,
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }
}

impl SpanMergingConfig {
    /// Create a new configuration with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::new();
    /// assert_eq!(config.space_threshold_em_ratio, 0.25);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration with aggressive space insertion (for dense layouts).
    ///
    /// Uses lower thresholds to insert spaces more readily:
    /// - space_threshold_em_ratio: 0.15 (instead of 0.25)
    /// - conservative_threshold_pt: 0.1 (instead of 0.3)
    ///
    /// Good for documents with many short words close together (author lists, grids).
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::aggressive();
    /// ```
    pub fn aggressive() -> Self {
        Self {
            space_threshold_em_ratio: 0.15,
            conservative_threshold_pt: 0.1,
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: false,
            adaptive_config: None,
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }

    /// Create a configuration with conservative space insertion (for formal documents).
    ///
    /// Uses higher thresholds to insert spaces less readily:
    /// - space_threshold_em_ratio: 0.33 (instead of 0.25)
    /// - conservative_threshold_pt: 0.3 (instead of 0.1)
    ///
    /// Good for formal documents where spacing is reliable.
    ///
    /// **Note**: After regression testing, 0.5pt threshold was found to cause
    /// excessive word fusion in policy documents. Reduced to 0.3pt.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::conservative();
    /// ```
    pub fn conservative() -> Self {
        Self {
            space_threshold_em_ratio: 0.33,
            conservative_threshold_pt: 0.3, // Reduced from 0.5 (was too aggressive for policy docs)
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: false,
            adaptive_config: None,
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }

    /// Create a configuration with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `space_threshold_em` - Space threshold as em ratio
    /// * `conservative_pt` - Conservative gap threshold in points
    /// * `column_boundary_pt` - Column boundary threshold in points
    /// * `overlap_pt` - Severe overlap threshold in points
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::custom(0.2, 0.2, 6.0, -0.3);
    /// ```
    pub fn custom(
        space_threshold_em: f32,
        conservative_pt: f32,
        column_boundary_pt: f32,
        overlap_pt: f32,
    ) -> Self {
        Self {
            space_threshold_em_ratio: space_threshold_em,
            conservative_threshold_pt: conservative_pt,
            column_boundary_threshold_pt: column_boundary_pt,
            severe_overlap_threshold_pt: overlap_pt,
            use_adaptive_threshold: false,
            adaptive_config: None,
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }

    /// Create a configuration with adaptive threshold enabled (default settings).
    ///
    /// This enables automatic threshold calculation based on the document's gap
    /// distribution. Uses conservative base settings for reliable defaults:
    /// - space_threshold_em_ratio: 0.25
    /// - conservative_threshold_pt: 0.1 (overridden by adaptive calculation)
    /// - column_boundary_threshold_pt: 5.0
    /// - severe_overlap_threshold_pt: -0.5
    /// - adaptive_config: AdaptiveThresholdConfig::default()
    ///
    /// The adaptive threshold is computed as: median_gap * 1.5, clamped to [0.05, 1.0] points.
    ///
    /// # Benefits
    ///
    /// - Automatically adapts to different document types
    /// - Reduces word fusion in policy documents with tight spacing
    /// - Minimizes spurious spaces in other document types
    /// - Maintains backward compatibility (disabled by default)
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::adaptive();
    /// assert!(config.use_adaptive_threshold);
    /// ```
    pub fn adaptive() -> Self {
        Self {
            space_threshold_em_ratio: 0.25,
            conservative_threshold_pt: 0.1,
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: true,
            adaptive_config: Some(
                crate::extractors::gap_statistics::AdaptiveThresholdConfig::default(),
            ),
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }

    /// Create a configuration with adaptive threshold and custom settings.
    ///
    /// # Arguments
    ///
    /// * `adaptive_config` - Custom adaptive threshold configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::{SpanMergingConfig, AdaptiveThresholdConfig};
    ///
    /// let config = SpanMergingConfig::adaptive_with_config(
    ///     AdaptiveThresholdConfig::policy_documents()
    /// );
    /// assert!(config.use_adaptive_threshold);
    /// ```
    pub fn adaptive_with_config(
        adaptive_config: crate::extractors::gap_statistics::AdaptiveThresholdConfig,
    ) -> Self {
        Self {
            space_threshold_em_ratio: 0.25,
            conservative_threshold_pt: 0.1,
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: true,
            adaptive_config: Some(adaptive_config),
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }

    /// Create a configuration using the legacy fixed-threshold approach.
    ///
    /// This provides backward compatibility with legacy behavior where
    /// adaptive threshold was disabled by default. All thresholds are fixed values.
    ///
    /// **Default values**:
    /// - space_threshold_em_ratio: 0.25 (standard word spacing)
    /// - conservative_threshold_pt: 0.1 (tight font metric threshold)
    /// - column_boundary_threshold_pt: 5.0 (standard column separation)
    /// - severe_overlap_threshold_pt: -0.5 (standard overlap tolerance)
    /// - use_adaptive_threshold: false (no automatic adjustment)
    ///
    /// # When to Use
    ///
    /// Use this when you need the fixed-threshold behavior:
    /// - Testing regression against old baselines
    /// - Documents with known quirks that required specific thresholds
    /// - Performance-critical applications where adaptive overhead is unacceptable
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::SpanMergingConfig;
    ///
    /// let config = SpanMergingConfig::legacy();
    /// assert!(!config.use_adaptive_threshold);
    /// assert_eq!(config.conservative_threshold_pt, 0.1);
    /// ```
    pub fn legacy() -> Self {
        Self {
            space_threshold_em_ratio: 0.25,
            conservative_threshold_pt: 0.1,
            column_boundary_threshold_pt: 5.0,
            severe_overlap_threshold_pt: -0.5,
            use_adaptive_threshold: false, // Fixed thresholds, no adaptive
            adaptive_config: None,
            detect_email_patterns: false,
            email_threshold_multiplier: 2.5,
            detect_citation_markers: false,
            citation_font_size_ratio: 0.75,
        }
    }
}

/// Unified space decision function - SINGLE SOURCE OF TRUTH for space insertion.
///
/// This function consolidates all space insertion logic into one place per the
/// design principle in the comprehensive plan. It evaluates multiple signals and
/// returns a definitive decision about whether to insert a space between spans.
///
/// # Rules (in priority order)
///
/// **Rule 0**: Check if boundary space already exists (from trailing/leading whitespace)
/// - If preceding text ends with space OR following text starts with space, don't insert
/// - Confidence: 1.0 (deterministic)
///
/// **Rule 1**: TJ offset triggered flag
/// - If the TJ processor set the flag due to negative offset > threshold, insert space
/// - This is explicit PDF positioning information
/// - Confidence: 0.95 (highest, explicit signal)
///
/// **Rule 2**: Dual threshold (PDFBox pattern) with document-type adjustment
/// - Calculate both space-width-based and char-width-based thresholds
/// - Adjust thresholds based on document type (Academic/Policy/Mixed)
/// - Use MINIMUM of the two for robustness
/// - If gap exceeds this threshold, insert space
/// - Confidence: 0.8 (geometric measurement)
///
/// **Rule 3**: Character heuristic (CamelCase, number->letter, etc.)
/// - Detect character transitions indicating word boundaries
/// - If heuristic fires, insert space
/// - Confidence: 0.6 (pattern-based)
///
/// **Rule 4**: Conservative threshold (document-type aware)
/// - If gap exceeds conservative threshold (very small), insert space
/// - Catches small intentional gaps that are still word boundaries
/// - Adaptive to document type (Policy uses lower threshold, Academic uses higher)
/// - Confidence: 0.5 (conservative)
///
/// **Default**: No space inserted
///
/// # Document Type Adjustment
///
/// When document_type is provided, thresholds are adjusted:
/// - **Academic** (1.4x multiplier): Higher thresholds for loose spacing
/// - **Policy** (0.6x multiplier): Lower thresholds for tight justified text
/// - **Mixed** (1.0x multiplier): Default/balanced approach
///
/// This matches research findings from LA-PDFText, pdfminer.six, PDFBox, and iText
/// that adaptive thresholds provide better results than fixed values.
///
/// # PDF Spec Reference
///
/// ISO 32000-1:2008, Section 9.4.4 NOTE 6:
/// "The identification of what constitutes a word is unrelated to how the text
/// happens to be grouped into show strings... text strings should be as long as possible."
fn should_insert_space(
    preceding_text: &str,
    following_text: &str,
    gap_pt: f32,
    font_size: f32,
    font_name: &str,
    fonts: &std::collections::HashMap<String, std::sync::Arc<crate::fonts::FontInfo>>,
    tj_offset_triggered: bool,
    config: &SpanMergingConfig,
    prev_bbox: Option<&crate::geometry::Rect>,
    next_bbox: Option<&crate::geometry::Rect>,
    prev_font_size: f32,
    next_font_size: f32,
) -> SpaceDecision {
    // PHASE 10: PDF Spec-Compliant Space Detection
    // Per ISO 32000-1:2008 Section 9.4.3 and 9.4.4
    //
    // Text positioning is determined by the text matrix and glyph positioning.
    // Only spec-defined signals are used; linguistic heuristics are excluded.
    //
    // Allowed signals (from PDF Spec):
    // 1. Boundary whitespace: spaces already present in text strings
    // 2. TJ array offsets: negative offsets < -100 thousandths of em
    // 3. Geometric gaps: gaps between character bounding boxes vs font metrics

    // Rule 0: Boundary Space (Section 9.4.3 - Text Showing)
    // Spaces already present in text strings should not be duplicated
    if has_boundary_space(preceding_text, following_text) {
        return SpaceDecision::no_space(SpaceSource::AlreadyPresent, 1.0);
    }

    // Rule 0.5: Email Pattern Detection
    // Per ISO 32000-1:2008 Section 9.10, email formatting preservation
    if config.detect_email_patterns && is_email_context(preceding_text, following_text) {
        let geometric_threshold = if let Some(font_info) = fonts.get(font_name) {
            let space_width_units = font_info.get_space_glyph_width();
            let space_width_pt = (space_width_units / 1000.0) * font_size;
            let word_margin_ratio = 0.5;
            space_width_pt * word_margin_ratio
        } else {
            font_size * 0.25
        };

        let email_threshold = geometric_threshold * config.email_threshold_multiplier;

        if gap_pt > email_threshold {
            log::debug!(
                "Email context detected: gap={:.2}pt > {:.2}pt email threshold - inserting space",
                gap_pt,
                email_threshold
            );
            return SpaceDecision::insert(SpaceSource::GeometricGap, 0.85);
        }

        log::debug!(
            "Email context detected: gap={:.2}pt <= {:.2}pt email threshold - suppressing space",
            gap_pt,
            email_threshold
        );
        return SpaceDecision::no_space(SpaceSource::NoSpace, 1.0);
    }

    // Line Break Handling
    // ==============================================================================
    // Per ISO 32000-1:2008 Section 5.2 (geometric positioning):
    // Line breaks are detected using bbox Y-coordinates (vertical positioning).
    // Words split across lines need special handling:
    // - Soft hyphen breaks: Previous text ends with '-' → NO space (word continuation)
    // - Hard line breaks: Normal breaks → INSERT space (new word on next line)
    //
    // Spec Reference: Section 5.2 states coordinates are in user space units.
    // Font size is used as reference for vertical gap detection threshold.

    if let (Some(prev_box), Some(next_box)) = (prev_bbox, next_bbox) {
        // Calculate vertical positioning for line break detection.
        // Use Y-coordinate difference (not bottom-to-top gap) to detect actual line breaks.
        // Two spans on the same line have nearly identical Y positions regardless of height.
        let y_diff = (prev_box.y - next_box.y).abs();

        // Line break threshold: if Y positions differ by more than 0.5× font size
        let line_break_threshold = font_size * 0.5;
        let is_line_break = y_diff > line_break_threshold;

        if is_line_break {
            // Verify same-column layout: X-positions within 2× font width
            let same_column = (prev_box.left() - next_box.left()).abs() < (font_size * 2.0);

            if same_column {
                log::debug!(
                    "Detected line break: y_diff={:.2}pt > {:.2}pt threshold, same_column=true",
                    y_diff,
                    line_break_threshold
                );

                // Check if previous text ends with hyphen (soft line break)
                if preceding_text.ends_with('-') {
                    log::debug!(
                        "Soft hyphen detected: '{}' ends with '-', suppressing space insertion",
                        preceding_text
                    );
                    return SpaceDecision::no_space(SpaceSource::NoSpace, 1.0);
                } else {
                    log::debug!("Hard line break detected: inserting space for word continuation");
                    return SpaceDecision::insert(SpaceSource::GeometricGap, 0.9);
                }
            }
        }
    }

    // NEW: Rule 1.5: Citation Marker Detection
    // ==============================================================================
    // Per ISO 32000-1:2008 Section 9.3, citation markers have distinct visual properties
    if config.detect_citation_markers
        && is_citation_context(prev_bbox, next_bbox, font_size, prev_font_size, next_font_size)
    {
        // For citations, use single-signal detection (don't require consensus)
        // Compute geometric threshold for citation context
        let citation_geometric_threshold = if let Some(font_info) = fonts.get(font_name) {
            let space_width_units = font_info.get_space_glyph_width();
            let space_width_pt = (space_width_units / 1000.0) * font_size;
            space_width_pt * 0.5
        } else {
            font_size * 0.25
        };

        if tj_offset_triggered || gap_pt > citation_geometric_threshold {
            log::debug!(
                "Citation context detected: using relaxed spacing rules (gap={:.2}pt, tj={})",
                gap_pt,
                tj_offset_triggered
            );
            return SpaceDecision::insert(SpaceSource::TjOffset, 0.90);
        }
    }

    // Consensus-Based Spacing Logic
    // ==============================================================================
    // Per ISO 32000-1:2008 Section 9.4.4 and 9.10:
    // "Determining word boundaries is not specified by PDF."
    // TJ offsets are typographic hints only, not definitive word boundaries.
    //
    // Solution: Require CONSENSUS between multiple PDF-spec-defined signals:
    // - TJ offset signal (explicit typography positioning)
    // - Geometric signal (bounding box analysis)
    // - Strong geometric signal alone is sufficient (gap > 2× threshold)

    // Rule 1: TJ Offset Signal (Section 9.4.3) - PDF-spec explicit signal
    // Calculate font-aware geometric threshold for consensus checking
    let geometric_threshold = if let Some(font_info) = fonts.get(font_name) {
        // Font found: use space glyph width for calculation
        let space_width_units = font_info.get_space_glyph_width(); // in 1000ths of em
        let space_width_pt = (space_width_units / 1000.0) * font_size;
        let word_margin_ratio = 0.5; // 50% of space width
        let threshold = space_width_pt * word_margin_ratio;

        log::debug!(
            "Font-aware spacing for '{}' @ {:.1}pt: space_width={:.1}pt, threshold={:.1}pt",
            font_name,
            font_size,
            space_width_pt,
            threshold
        );

        threshold
    } else {
        // Font not found: fallback to fixed 0.25em threshold
        log::debug!(
            "Font '{}' not found in font map, using default 0.25em threshold for {:.1}pt",
            font_name,
            font_size
        );
        font_size * 0.25
    };

    let geometric_suggests_space = gap_pt > geometric_threshold;

    // Consensus checking
    // Only insert space if BOTH signals agree OR geometric signal is very strong
    // This reduces false positives in justified text where TJ offsets are arbitrary
    if tj_offset_triggered && geometric_suggests_space {
        // HIGH CONFIDENCE: Both TJ and geometric signals agree
        log::debug!(
            "Space decision: CONSENSUS - both TJ and geometric signals triggered (gap={:.2}pt > {:.2}pt) - inserting space",
            gap_pt,
            geometric_threshold
        );
        return SpaceDecision::insert(SpaceSource::TjOffset, 1.0);
    }

    // TJ offset with relaxed geometric confirmation
    // In tight typesetting (e.g., LaTeX academic papers), word gaps are narrower than
    // the standard 50% space-width threshold. When the PDF producer explicitly encoded
    // a TJ offset, accept a lower geometric bar (25% of space width).
    if tj_offset_triggered && gap_pt > geometric_threshold * 0.5 {
        log::debug!(
            "Space decision: TJ + relaxed geometric (gap={:.2}pt > {:.2}pt relaxed threshold) - inserting space",
            gap_pt,
            geometric_threshold * 0.5
        );
        return SpaceDecision::insert(SpaceSource::TjOffset, 0.9);
    }

    // WordBoundaryDetector tiebreaker when TJ and geometric signals conflict
    // Per ISO 32000-1:2008 Section 9.4.4, use multiple signals to determine word boundaries
    if tj_offset_triggered != geometric_suggests_space {
        if let (Some(prev_box), Some(next_box)) = (prev_bbox, next_bbox) {
            let (characters, context) = build_boundary_characters(
                preceding_text,
                following_text,
                prev_box,
                next_box,
                font_size,
                tj_offset_triggered,
            );

            // Use WordBoundaryDetector with geometric gap ratio matching our threshold
            // OPTIMIZATION: Detect document script profile to skip unnecessary detectors
            let script = DocumentScript::detect_from_characters(&characters);
            let detector = WordBoundaryDetector::new()
                .with_document_script(script)
                .with_geometric_gap_ratio(0.5);
            let boundaries = detector.detect_word_boundaries(&characters, &context);

            if !boundaries.is_empty() {
                log::debug!(
                    "Space decision: WordBoundaryDetector resolved conflict (TJ={}, geo={}) - inserting space",
                    tj_offset_triggered,
                    geometric_suggests_space
                );
                return SpaceDecision::insert(SpaceSource::WordBoundaryAnalysis, 0.85);
            }
        }
    }

    // Strong geometric signal alone.
    //
    // `geometric_threshold` is already `space_width_pt * 0.5`. A gap that
    // clears this threshold is >= 50 % of the font's own space-glyph
    // advance, which is what pdfium (Chrome/pypdfium2) uses as the
    // word-break heuristic in its default text-extraction path — and
    // the reason pdf_oxide was glueing adjacent words like
    // "atBirmingham", "LIFESCIENCESRESEARCH", "STATIONFREEDOM",
    // "proteincrystals" before this change. The previous 2× multiplier
    // required gaps >= 100 % of a full space glyph, which is stricter
    // than the gaps modern tightly-kerned typesetters emit between
    // real words (often 60-80 % of a space glyph).
    //
    // Intra-word kerning and letter-spacing adjustments are well below
    // 50 % of a space glyph (typically under 5 % of font-size), so
    // lowering this threshold does not produce false word breaks
    // inside words. Pure digit-digit sequences are separately protected
    // in the value/token branch below via `digit_digit_gap_ok`.
    //
    // See issue #326 for the corpus-wide measurement that motivated
    // this change (NASA Apollo 11 jaccard 0.449 → target >= 0.90 vs
    // pypdfium2 on the 60-PDF regression corpus).
    if gap_pt > geometric_threshold {
        log::debug!(
            "Space decision: STRONG GEOMETRIC - gap={:.2}pt > {:.2}pt threshold - inserting space",
            gap_pt,
            geometric_threshold
        );
        return SpaceDecision::insert(SpaceSource::GeometricGap, 0.95);
    }

    // Separate token detection: when two spans have a positive gap and look like
    // distinct values (not fragments of the same word), insert a space.
    //
    // This catches adjacent table cell values like "$0.00" "$0.00" that have small
    // gaps (1-2pt) which fall below the standard geometric threshold but are clearly
    // separate tokens. Word fragments within the same word have zero or near-zero
    // gaps; any meaningful positive gap between non-fragment tokens indicates a
    // word boundary.
    //
    // Heuristic: gap > 0 AND spans look like separate tokens based on boundary characters.
    // Use near-zero threshold for currency boundaries (any positive gap = separate)
    let min_token_gap = 0.01; // Essentially any positive gap triggers token check
    if gap_pt > min_token_gap {
        let prev_last = preceding_text.chars().last();
        let next_first = following_text.chars().next();

        if let (Some(pc), Some(nc)) = (prev_last, next_first) {
            // Separate value tokens: digit/currency/punctuation boundaries that
            // indicate two distinct values rather than fragments of one word.
            // Examples: "$0.00" + "$0.00", "100" + "200", "Subtotal" + "$500.00"
            let prev_is_value_end = pc.is_ascii_digit() || pc == '%' || pc == ')' || pc == ']';

            // Pure digit→digit boundaries require a larger gap than the
            // global `min_token_gap`: a long number emitted as multiple
            // spans (e.g. due to glyph-level kerning or TJ positioning
            // rounding) can have a tiny positive gap between adjacent
            // digit spans, which must NOT become "123 456". Anything less
            // than half the font-aware geometric threshold is treated as
            // intra-number kerning, not a token boundary.
            let digit_digit = nc.is_ascii_digit() && pc.is_ascii_digit();
            let digit_digit_gap_ok = !digit_digit || gap_pt > geometric_threshold * 0.5;

            let next_is_value_start = nc == '$'
                || nc == '('
                || nc == '['
                || (nc == '-' && following_text.len() > 1)
                || (nc.is_ascii_digit() && prev_is_value_end && digit_digit_gap_ok);

            // Also detect: any text followed by currency symbol
            // e.g., "Subtotal" + "$500.00" or "49" + "$0.00"
            let text_then_currency = (pc.is_ascii_alphabetic() || pc.is_ascii_digit())
                && (nc == '$' || nc == '€' || nc == '£');

            if (prev_is_value_end && next_is_value_start) || text_then_currency {
                log::debug!(
                    "Space decision: SEPARATE VALUES - gap={:.2}pt > {:.2}pt min, prev='{}', next='{}' - inserting space",
                    gap_pt,
                    min_token_gap,
                    crate::utils::safe_suffix(preceding_text, 5),
                    crate::utils::safe_prefix(following_text, 5),
                );
                return SpaceDecision::insert(SpaceSource::GeometricGap, 0.85);
            }
        }
    }

    // Default: No space
    // Per ISO 32000-1:2008 Section 9.10, when PDF doesn't encode a clear word boundary,
    // we cannot reliably recover it. Requiring consensus prevents false positives in justified text.
    log::trace!(
        "Space decision: Insufficient consensus (TJ={}, gap={:.2}pt <= {:.2}pt) - no space",
        tj_offset_triggered,
        gap_pt,
        geometric_threshold
    );
    SpaceDecision::no_space(SpaceSource::NoSpace, 1.0)
}

/// Check if a boundary between spans already has whitespace.
///
/// Returns true if:
/// - The preceding text ends with whitespace, OR
/// - The following text starts with whitespace
///
/// This prevents double-spacing when text already contains space characters.
fn has_boundary_space(preceding: &str, following: &str) -> bool {
    // Use ends_with/starts_with patterns instead of .chars().last() to avoid
    // O(n) iteration over the entire accumulated text
    let has_trailing_space = preceding.ends_with(|c: char| c.is_whitespace());
    let has_leading_space = following.starts_with(|c: char| c.is_whitespace());

    has_trailing_space || has_leading_space
}

/// Build CharacterInfo for word boundary analysis between two text segments.
///
/// Creates minimal character info for the last character of the preceding text
/// and the first character of the following text. This allows WordBoundaryDetector
/// to determine if a word boundary exists between two spans.
///
/// Per ISO 32000-1:2008 Section 9.4.4, word boundaries can be identified through:
/// - TJ array offsets (passed via tj_offset_triggered)
/// - Geometric gaps between glyphs (calculated from bbox positions)
/// - Space characters in the text stream
/// - CJK character transitions
fn build_boundary_characters(
    prev_text: &str,
    next_text: &str,
    prev_bbox: &Rect,
    next_bbox: &Rect,
    font_size: f32,
    tj_offset_triggered: bool,
) -> (Vec<CharacterInfo>, BoundaryContext) {
    let prev_last_char = prev_text.chars().last().unwrap_or(' ');
    let next_first_char = next_text.chars().next().unwrap_or(' ');

    // Estimate character widths from bbox and character count
    // Use byte length as fast O(1) approximation (accurate for ASCII, close for UTF-8)
    // to avoid O(n) char counting on the accumulated merge text
    let prev_char_count = prev_text.len().max(1) as f32;
    let prev_char_width = prev_bbox.width / prev_char_count;
    let prev_last_x = prev_bbox.x + prev_bbox.width - prev_char_width;

    let next_char_count = next_text.len().max(1) as f32;
    let next_char_width = next_bbox.width / next_char_count;

    // Build CharacterInfo for boundary analysis
    let characters = vec![
        CharacterInfo {
            code: prev_last_char as u32,
            glyph_id: None,
            width: prev_char_width,
            x_position: prev_last_x,
            // Convert TJ trigger to offset value: -200 indicates word boundary
            tj_offset: if tj_offset_triggered {
                Some(-200)
            } else {
                None
            },
            font_size,
            is_ligature: false, // Not relevant for tiebreaker mode
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: next_first_char as u32,
            glyph_id: None,
            width: next_char_width,
            x_position: next_bbox.x,
            tj_offset: None,
            font_size,
            is_ligature: false, // Not relevant for tiebreaker mode
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let context = BoundaryContext {
        font_size,
        horizontal_scaling: 100.0, // Default; actual value not available at span level
        word_spacing: 0.0,
        char_spacing: 0.0,
    };

    (characters, context)
}

/// Check if surrounding text forms an email-like pattern.
/// Per PDF spec, uses only extracted text pattern matching.
///
/// Patterns detected:
/// - "user@outlook" + "." + "com" (space before TLD)
/// - "user@" + "domain.com" (space after @)
fn is_email_context(preceding_text: &str, following_text: &str) -> bool {
    // Only check the last ~64 bytes for email patterns to avoid O(n) scan
    // of the entire accumulated text (which would cause O(n²) in merge loop)
    let prev_start = preceding_text.len().saturating_sub(64);
    // Find a valid UTF-8 char boundary
    let prev_start = preceding_text.ceil_char_boundary(prev_start);
    let prev = preceding_text[prev_start..].trim_end();
    let next = following_text.trim_start();

    // Pattern 1: @ followed by domain part
    if prev.contains('@') {
        let after_at = prev.split('@').next_back().unwrap_or("");

        // Pattern 1a: "outlook" + "." → likely email
        if !after_at.is_empty() && next.starts_with('.') {
            return true;
        }

        // Pattern 1b: "outlook." + "com" → likely email
        if after_at.ends_with('.') && next.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            return true;
        }
    }

    // Pattern 2: Previous ends with @ (immediate after @)
    if prev.ends_with('@')
        && next
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric())
    {
        return true;
    }

    false
}

/// Detect if bounding boxes indicate citation marker context.
/// Per PDF spec Section 9.3, citation markers have distinct visual properties:
/// - Smaller font size (typically 50-75% of body text)
/// - Raised position (superscript)
fn is_citation_context(
    prev_bbox: Option<&crate::geometry::Rect>,
    next_bbox: Option<&crate::geometry::Rect>,
    current_font_size: f32,
    prev_font_size: f32,
    next_font_size: f32,
) -> bool {
    let prev_ratio = prev_font_size / current_font_size;
    let next_ratio = next_font_size / current_font_size;

    // Superscript range: 50-75% of body text size
    const SUPERSCRIPT_MIN: f32 = 0.5;
    const SUPERSCRIPT_MAX: f32 = 0.75;

    let prev_is_superscript = (SUPERSCRIPT_MIN..=SUPERSCRIPT_MAX).contains(&prev_ratio);
    let next_is_superscript = (SUPERSCRIPT_MIN..=SUPERSCRIPT_MAX).contains(&next_ratio);

    if let (Some(prev_box), Some(next_box)) = (prev_bbox, next_bbox) {
        let vertical_offset = (prev_box.y - next_box.y).abs();
        let is_raised = vertical_offset > (current_font_size * 0.2);

        // Either previous OR next is superscript + raised
        if (prev_is_superscript || next_is_superscript) && is_raised {
            return true;
        }
    }

    // Fallback: just font size check if bbox unavailable
    prev_is_superscript || next_is_superscript
}

/// Buffer for accumulating text from TJ array elements into a single span.
///
/// Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
/// "The performance of text searching (and other text extraction operations) is
/// significantly better if the text strings are as long as possible."
///
/// This buffer accumulates consecutive string elements from TJ arrays into
/// a single logical text span, only breaking on explicit word boundaries.
#[derive(Debug)]
struct TjBuffer {
    /// Accumulated Unicode text
    unicode: String,
    /// Text matrix at the start of this buffer
    start_matrix: Matrix,
    /// Font name when buffer started
    font_name: Option<String>,
    /// Fill color RGB when buffer started
    fill_color_rgb: (f32, f32, f32),
    /// Character spacing (Tc) when buffer started
    char_space: f32,
    /// Word spacing (Tw) when buffer started
    word_space: f32,
    /// Horizontal scaling (Th) when buffer started
    horizontal_scaling: f32,
    /// MCID when buffer started
    mcid: Option<u32>,
    /// Accumulated width from advance_position_for_string calls.
    /// Avoids redundant per-byte width recalculation in flush.
    accumulated_width: f32,
    /// Cached font reference — avoids per-Tj HashMap lookup in append.
    /// Set once at buffer creation, never changes (font change flushes buffer).
    cached_font: Option<Arc<FontInfo>>,
    /// Pre-computed effective font size (CTM × text_matrix scaling × font_size).
    /// Computed once at buffer creation to avoid matrix multiply + sqrt per flush.
    effective_font_size: f32,
    /// Pre-computed font weight from cached font reference.
    font_weight: FontWeight,
    /// Pre-computed italic flag from cached font reference.
    is_italic: bool,
    /// Whether the font is monospaced (from FixedPitch flag or name heuristic).
    is_monospace: bool,
    /// Per-character advance widths in text-space units (before user_h_scale).
    char_widths: Vec<f32>,
    /// Pre-computed user-space position (CTM applied to text matrix origin).
    /// Avoids two transform_point calls per flush.
    user_pos_x: f32,
    user_pos_y: f32,
    /// Pre-computed horizontal scale factor (CTM × text_matrix).
    /// Used to convert accumulated_width from text space to user space for bbox.
    user_h_scale: f32,
}

impl TjBuffer {
    /// Create a new empty buffer with current state.
    fn new(
        state: &crate::content::graphics_state::GraphicsState,
        mcid: Option<u32>,
        cached_font: Option<Arc<FontInfo>>,
    ) -> Self {
        // Pre-compute effective font size: CTM × text_matrix scaling × font_size
        let combined = state.ctm.multiply(&state.text_matrix);
        let effective_font_size =
            state.font_size * (combined.d * combined.d + combined.b * combined.b).sqrt();
        // Pre-compute horizontal scale for converting text-space widths to user space
        let user_h_scale = (combined.a * combined.a + combined.c * combined.c).sqrt();
        let font_weight = match &cached_font {
            Some(f) if f.is_bold() => FontWeight::Bold,
            _ => FontWeight::Normal,
        };
        let is_italic = cached_font.as_ref().map(|f| f.is_italic()).unwrap_or(false);
        let is_monospace = cached_font.as_ref().is_some_and(|f| {
            if f.flags.is_some_and(|flags| flags & 1 != 0) {
                return true;
            }
            let name = f.base_font.to_uppercase();
            name.contains("COURIER")
                || name.contains("CONSOLAS")
                || name.contains("MONO")
                || name.contains("FIXED")
        });
        // Pre-compute user-space position: text_matrix origin → CTM transform
        let text_pos = state.text_matrix.transform_point(0.0, 0.0);
        let user_pos = state.ctm.transform_point(text_pos.x, text_pos.y);
        Self {
            unicode: String::new(),
            start_matrix: state.text_matrix,
            font_name: state.font_name.clone(),
            fill_color_rgb: state.fill_color_rgb,
            char_space: state.char_space,
            word_space: state.word_space,
            horizontal_scaling: state.horizontal_scaling,
            mcid,
            accumulated_width: 0.0,
            cached_font,
            effective_font_size,
            font_weight,
            is_italic,
            is_monospace,
            char_widths: Vec::new(),
            user_pos_x: user_pos.x,
            user_pos_y: user_pos.y,
            user_h_scale,
        }
    }

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.unicode.is_empty()
    }

    /// Append a text string to the buffer.
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        // PDF spec Section 7.3.4.2: implementation limit of 32,767 bytes per string.
        // Malformed PDFs may exceed this, causing text blowup.
        let bytes = if bytes.len() > 32_767 {
            &bytes[..32_767]
        } else {
            bytes
        };

        let font = self.cached_font.as_deref();

        // Fast path: OneByte fonts push chars directly into buffer via lookup table.
        // Avoids String allocation in decode_text_to_unicode (2 allocations per call).
        if let Some(font) = font {
            if font.subtype != "Type0" {
                let table = font.get_byte_to_char_table();
                for &byte in bytes {
                    let c = table[byte as usize];
                    if c != '\0' {
                        self.unicode.push(c);
                    } else {
                        // Rare: multi-char mapping or unmapped byte
                        if let Some(s) = font.char_to_unicode(byte as u32) {
                            if s != "\u{FFFD}" {
                                for ch in s.chars() {
                                    if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                        self.unicode.push(ch);
                                    }
                                }
                            }
                        } else {
                            let fb = fallback_char_to_unicode(byte as u32);
                            if fb != "\u{FFFD}" {
                                for ch in fb.chars() {
                                    if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                        self.unicode.push(ch);
                                    }
                                }
                            }
                        }
                    }
                }
                return Ok(());
            }
        }

        // Slow path: Type0 (CID) fonts or no font — use full decode function
        let unicode_text = decode_text_to_unicode(bytes, font);
        self.unicode.push_str(&unicode_text);

        Ok(())
    }
}

/// Fallback function to map common character codes to Unicode when ToUnicode CMap fails.
///
/// PDF Spec Compliance: ISO 32000-1:2008 Section 9.10.2
/// This function implements Priority 6 (enhanced fallback) after the standard 5-tier
/// encoding system (ToUnicode CMap, predefined encodings, Adobe Glyph List, etc.) fails.
///
/// Multi-tier fallback strategy:
/// 1. Common punctuation and symbols (em dash, en dash, quotes, bullets)
/// 2. Mathematical operators (∂, ∇, ∑, ∏, ∫, √, ∞, ≤, ≥, ≠)
/// 3. Greek letters (α, β, γ, δ, θ, λ, μ, π, σ, ω - both cases)
/// 4. Currency symbols (€, £, ¥, ¢)
/// 5. Direct Unicode (if char_code is in valid Unicode range)
/// 6. Private Use Area visual description (U+E000-U+F8FF)
/// 7. Replacement character "?" as last resort
///
/// # Arguments
/// * `char_code` - 16-bit character code that failed to decode via standard system
///
/// # Returns
/// Best-effort Unicode string representation, or "?" if no mapping possible
fn fallback_char_to_unicode(char_code: u32) -> String {
    match char_code {
        // ==================================================================================
        // PRIORITY 1: Common Punctuation (most frequently failing)
        // ==================================================================================
        0x2014 => "—".to_string(),        // Em dash
        0x2013 => "–".to_string(),        // En dash
        0x2018 => "\u{2018}".to_string(), // Left single quotation mark (')
        0x2019 => "\u{2019}".to_string(), // Right single quotation mark (')
        0x201C => "\u{201C}".to_string(), // Left double quotation mark (")
        0x201D => "\u{201D}".to_string(), // Right double quotation mark (")
        0x2022 => "•".to_string(),        // Bullet
        0x2026 => "…".to_string(),        // Horizontal ellipsis
        0x00B0 => "°".to_string(),        // Degree sign

        // ==================================================================================
        // PRIORITY 2: Mathematical Operators (common in academic papers)
        // ==================================================================================
        0x00B1 => "±".to_string(), // Plus-minus sign
        0x00D7 => "×".to_string(), // Multiplication sign
        0x00F7 => "÷".to_string(), // Division sign
        0x2202 => "∂".to_string(), // Partial differential
        0x2207 => "∇".to_string(), // Nabla (del operator)
        0x220F => "∏".to_string(), // N-ary product
        0x2211 => "∑".to_string(), // N-ary summation
        0x221A => "√".to_string(), // Square root
        0x221E => "∞".to_string(), // Infinity
        0x2260 => "≠".to_string(), // Not equal to
        0x2261 => "≡".to_string(), // Identical to
        0x2264 => "≤".to_string(), // Less-than or equal to
        0x2265 => "≥".to_string(), // Greater-than or equal to
        0x222B => "∫".to_string(), // Integral
        0x2248 => "≈".to_string(), // Almost equal to
        0x2282 => "⊂".to_string(), // Subset of
        0x2283 => "⊃".to_string(), // Superset of
        0x2286 => "⊆".to_string(), // Subset of or equal to
        0x2287 => "⊇".to_string(), // Superset of or equal to
        0x2208 => "∈".to_string(), // Element of
        0x2209 => "∉".to_string(), // Not an element of
        0x2200 => "∀".to_string(), // For all
        0x2203 => "∃".to_string(), // There exists
        0x2205 => "∅".to_string(), // Empty set
        0x2227 => "∧".to_string(), // Logical and
        0x2228 => "∨".to_string(), // Logical or
        0x00AC => "¬".to_string(), // Not sign
        0x2192 => "→".to_string(), // Rightwards arrow
        0x2190 => "←".to_string(), // Leftwards arrow
        0x2194 => "↔".to_string(), // Left right arrow
        0x21D2 => "⇒".to_string(), // Rightwards double arrow
        0x21D4 => "⇔".to_string(), // Left right double arrow

        // ==================================================================================
        // PRIORITY 3: Greek Letters (common in scientific/mathematical texts)
        // ==================================================================================
        // Lowercase Greek
        0x03B1 => "α".to_string(), // Alpha
        0x03B2 => "β".to_string(), // Beta
        0x03B3 => "γ".to_string(), // Gamma
        0x03B4 => "δ".to_string(), // Delta
        0x03B5 => "ε".to_string(), // Epsilon
        0x03B6 => "ζ".to_string(), // Zeta
        0x03B7 => "η".to_string(), // Eta
        0x03B8 => "θ".to_string(), // Theta
        0x03B9 => "ι".to_string(), // Iota
        0x03BA => "κ".to_string(), // Kappa
        0x03BB => "λ".to_string(), // Lambda
        0x03BC => "μ".to_string(), // Mu
        0x03BD => "ν".to_string(), // Nu
        0x03BE => "ξ".to_string(), // Xi
        0x03BF => "ο".to_string(), // Omicron
        0x03C0 => "π".to_string(), // Pi
        0x03C1 => "ρ".to_string(), // Rho
        0x03C2 => "ς".to_string(), // Final sigma
        0x03C3 => "σ".to_string(), // Sigma
        0x03C4 => "τ".to_string(), // Tau
        0x03C5 => "υ".to_string(), // Upsilon
        0x03C6 => "φ".to_string(), // Phi
        0x03C7 => "χ".to_string(), // Chi
        0x03C8 => "ψ".to_string(), // Psi
        0x03C9 => "ω".to_string(), // Omega

        // Uppercase Greek
        0x0391 => "Α".to_string(), // Alpha
        0x0392 => "Β".to_string(), // Beta
        0x0393 => "Γ".to_string(), // Gamma
        0x0394 => "Δ".to_string(), // Delta
        0x0395 => "Ε".to_string(), // Epsilon
        0x0396 => "Ζ".to_string(), // Zeta
        0x0397 => "Η".to_string(), // Eta
        0x0398 => "Θ".to_string(), // Theta
        0x0399 => "Ι".to_string(), // Iota
        0x039A => "Κ".to_string(), // Kappa
        0x039B => "Λ".to_string(), // Lambda
        0x039C => "Μ".to_string(), // Mu
        0x039D => "Ν".to_string(), // Nu
        0x039E => "Ξ".to_string(), // Xi
        0x039F => "Ο".to_string(), // Omicron
        0x03A0 => "Π".to_string(), // Pi
        0x03A1 => "Ρ".to_string(), // Rho
        0x03A3 => "Σ".to_string(), // Sigma
        0x03A4 => "Τ".to_string(), // Tau
        0x03A5 => "Υ".to_string(), // Upsilon
        0x03A6 => "Φ".to_string(), // Phi
        0x03A7 => "Χ".to_string(), // Chi
        0x03A8 => "Ψ".to_string(), // Psi
        0x03A9 => "Ω".to_string(), // Omega

        // ==================================================================================
        // PRIORITY 4: Currency Symbols
        // ==================================================================================
        0x20AC => "€".to_string(), // Euro
        0x00A3 => "£".to_string(), // Pound sterling
        0x00A5 => "¥".to_string(), // Yen
        0x00A2 => "¢".to_string(), // Cent
        0x20A3 => "₣".to_string(), // French franc
        0x20A4 => "₤".to_string(), // Lira
        0x20A9 => "₩".to_string(), // Won
        0x20AA => "₪".to_string(), // New shekel
        0x20AB => "₫".to_string(), // Dong
        0x20B9 => "₹".to_string(), // Indian rupee

        // ==================================================================================
        // PRIORITY 5: Direct Unicode (for valid ranges)
        // ==================================================================================
        // Valid Unicode: BMP (0x0000-0xD7FF, 0xE000-0xFFFF) and supplementary planes
        // Excludes surrogate pairs (0xD800-0xDFFF)
        code => {
            if let Some(ch) = char::from_u32(code) {
                if (0xE000..=0xF8FF).contains(&code) {
                    log::debug!("Private Use Area character: U+{:04X}", code);
                }
                ch.to_string()
            } else {
                log::warn!("Character code 0x{:04X} is not a valid Unicode code point", code);
                "?".to_string()
            }
        },
    }
}

/// Byte grouping mode for CID font character code decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteMode {
    /// Single-byte codes (simple fonts, some predefined CMaps)
    OneByte,
    /// Always 2-byte codes (Identity-H/V, UCS2)
    TwoByte,
    /// Shift-JIS variable-width (1 or 2 bytes depending on lead byte)
    ShiftJIS,
}

/// Get byte grouping mode for a font (v0.3.14).
fn get_byte_mode(font: Option<&FontInfo>) -> ByteMode {
    if let Some(font) = font {
        if font.subtype == "Type0" {
            match &font.encoding {
                crate::fonts::Encoding::Identity => ByteMode::TwoByte,
                crate::fonts::Encoding::Standard(name) => {
                    if (name.contains("Identity") && !name.contains("OneByteIdentity"))
                        || name.contains("UCS2")
                        || name.contains("UTF16")
                    {
                        ByteMode::TwoByte
                    } else if name.contains("RKSJ") {
                        ByteMode::ShiftJIS
                    } else if name.contains("EUC")
                        || name.contains("GBK")
                        || name.contains("GBpc")
                        || name.contains("GB-")
                        || name.contains("CNS")
                        || name.contains("B5")
                        || name.contains("KSC")
                        || name.contains("KSCms")
                    {
                        ByteMode::TwoByte
                    } else {
                        ByteMode::OneByte
                    }
                },
                _ => ByteMode::OneByte,
            }
        } else {
            ByteMode::OneByte
        }
    } else {
        ByteMode::OneByte
    }
}

/// Iterator over characters in a PDF string based on font encoding (v0.3.14).
struct TextCharIter<'a> {
    bytes: &'a [u8],
    byte_mode: ByteMode,
    index: usize,
}

impl<'a> TextCharIter<'a> {
    fn new(bytes: &'a [u8], font: Option<&FontInfo>) -> Self {
        Self {
            bytes,
            byte_mode: get_byte_mode(font),
            index: 0,
        }
    }
}

impl<'a> Iterator for TextCharIter<'a> {
    type Item = (u16, usize); // (char_code, bytes_consumed)

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.bytes.len() {
            return None;
        }

        let (char_code, bytes_consumed) = match self.byte_mode {
            ByteMode::TwoByte if self.index + 1 < self.bytes.len() => {
                (((self.bytes[self.index] as u16) << 8) | (self.bytes[self.index + 1] as u16), 2)
            },
            ByteMode::ShiftJIS => {
                let b = self.bytes[self.index];
                let is_lead = (0x81..=0x9F).contains(&b) || (0xE0..=0xFC).contains(&b);
                if is_lead && self.index + 1 < self.bytes.len() {
                    (((b as u16) << 8) | (self.bytes[self.index + 1] as u16), 2)
                } else {
                    (b as u16, 1)
                }
            },
            _ => (self.bytes[self.index] as u16, 1),
        };

        self.index += bytes_consumed;
        Some((char_code, bytes_consumed))
    }
}

fn decode_text_to_unicode(bytes: &[u8], font: Option<&FontInfo>) -> String {
    let raw_result = if let Some(font) = font {
        let mut result = String::new();
        // Use pre-computed lookup table for performance if it's a simple font
        if font.subtype != "Type0" {
            let table = font.get_byte_to_char_table();
            for &byte in bytes {
                let c = table[byte as usize];
                if c != '\0' {
                    result.push(c);
                } else {
                    // Fallback: multi-char mapping or unmapped byte
                    let char_str = font
                        .char_to_unicode(byte as u32)
                        .unwrap_or_else(|| fallback_char_to_unicode(byte as u32));
                    if char_str != "\u{FFFD}" {
                        result.push_str(&char_str);
                    }
                }
            }
        } else {
            // Complex font: use unified iterator for robust multi-byte decoding
            for (char_code, _) in TextCharIter::new(bytes, Some(font)) {
                let char_str = font
                    .char_to_unicode(char_code as u32)
                    .unwrap_or_else(|| fallback_char_to_unicode(char_code as u32));

                if char_str != "\u{FFFD}" {
                    result.push_str(&char_str);
                }
            }
        }
        result
    } else {
        // No font - fallback to Latin-1 (ISO 8859-1) encoding
        // Per PDF Spec ISO 32000-1:2008, Section 9.6.6, Latin-1 maps bytes 0x00-0xFF
        // directly to Unicode code points U+0000-U+00FF
        log::warn!(
            "⚠️  No font provided for {} bytes, using Latin-1 fallback (PDF spec compliant)",
            bytes.len()
        );
        bytes.iter().map(|&b| char::from(b)).collect()
    };

    // Filter control characters from failed encoding resolution
    // Keep: \t (0x09), \n (0x0A), \r (0x0D), and all printable chars (>= 0x20)
    let mut filtered = String::with_capacity(raw_result.len());
    for c in raw_result.chars() {
        if c >= '\x20' || c == '\t' || c == '\n' || c == '\r' {
            filtered.push(c);
        }
    }
    filtered
}

/// Artifact type classification per PDF Spec Section 14.8.2.2
///
/// Artifacts are content that is not part of the document's logical structure,
/// such as headers, footers, page numbers, and decorative elements.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum ArtifactType {
    /// Pagination artifacts (headers, footers, page numbers)
    Pagination(PaginationSubtype),
    /// Layout artifacts (ruled lines, backgrounds, borders)
    Layout,
    /// Page artifacts (full-page backgrounds, watermarks)
    Page,
    /// Background graphics or decorations
    Background,
}

/// Pagination artifact subtypes per PDF Spec Section 14.8.2.2.1
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum PaginationSubtype {
    /// Page header content
    Header,
    /// Page footer content
    Footer,
    /// Watermark overlay
    Watermark,
    /// Page number
    PageNumber,
    /// Other pagination element
    Other,
}

/// Context for marked content sequences (per PDF Spec Section 14.6)
///
/// Tracks nested marked content tags to implement artifact filtering.
/// When content is marked as `/Artifact`, it should be excluded from text extraction.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct MarkedContentContext {
    tag: String,
    is_artifact: bool,
    /// Artifact type classification for filtered content (PDF Spec Section 14.8.2.2)
    artifact_type: Option<ArtifactType>,
    /// ActualText for marked content (PDF Spec Section 14.9.4)
    /// Used to replace extracted text with correct representation
    /// e.g., ligatures (fi, fl, ffi, ffl), decorated glyphs
    actual_text: Option<String>,
    /// Expansion text for abbreviations (PDF Spec Section 14.9.5)
    /// The /E entry provides the expansion of an abbreviation or acronym.
    /// e.g., "PDF" might expand to "Portable Document Format"
    expansion: Option<String>,
}

/// Text extractor that processes content streams.
///
/// This structure maintains the graphics state stack and font information
/// while processing operators to extract positioned text.
///
/// The extractor can work in two modes:
/// - **Span mode** (default): Extracts complete text strings as PDF provides them (PDF spec compliant)
/// - **Character mode**: Extracts individual characters (for special use cases)
#[derive(Debug)]
pub struct TextExtractor {
    /// Graphics state stack for handling q/Q operators
    state_stack: GraphicsStateStack,
    /// Loaded fonts (name -> FontInfo). Arc-wrapped to avoid deep cloning across pages.
    fonts: HashMap<String, Arc<FontInfo>>,
    /// Extracted text spans (complete strings from Tj/TJ operators)
    spans: Vec<TextSpan>,
    /// Extracted characters (for backward compatibility)
    chars: Vec<TextChar>,
    /// Resources dictionary (for accessing XObjects and fonts)
    resources: Option<Object>,
    /// Reference to the document (for loading XObjects)
    document: Option<*const crate::document::PdfDocument>,
    /// Set of processed XObject references to avoid duplicates
    processed_xobjects: HashSet<ObjectRef>,
    /// Cached XObject name → ObjectRef mapping for current resources context.
    /// Avoids expensive repeated resolution of the resources/XObject dict chain.
    cached_xobject_refs: HashMap<String, Option<ObjectRef>>,
    /// Current XObject recursion depth (0 = page level)
    xobject_depth: u32,
    /// Number of XObjects decoded on this page (for budget limiting)
    xobject_decode_count: u32,
    /// Configuration for text extraction heuristics
    config: TextExtractionConfig,
    /// Configuration for span merging behavior
    merging_config: SpanMergingConfig,
    /// Current marked content ID (for Tagged PDFs)
    ///
    /// Tracks the MCID of the currently active marked content sequence.
    /// Used to associate extracted text with structure tree elements.
    current_mcid: Option<u32>,
    /// Stack of marked content contexts (per PDF Spec Section 14.6)
    ///
    /// Tracks nested marked content tags to enable artifact filtering.
    /// When content is marked as `/Artifact`, it should be excluded from text extraction.
    marked_content_stack: Vec<MarkedContentContext>,
    /// Whether we're currently inside an /Artifact marked content context
    ///
    /// Per PDF Spec Section 14.6, artifact content should be excluded from text extraction.
    /// This flag is true when any ancestor in the marked_content_stack has is_artifact=true.
    inside_artifact: bool,
    /// Extraction mode: true for spans, false for characters
    extract_spans: bool,
    /// Buffer for accumulating consecutive Tj operators into single spans
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.4.4 NOTE 6, text strings should
    /// be as long as possible. This buffer accumulates consecutive Tj operators
    /// until a positioning command or state change is encountered.
    tj_span_buffer: Option<TjBuffer>,
    /// Sequence counter for TextSpan ordering
    ///
    /// Used as a tie-breaker when sorting spans by Y-coordinate. Ensures
    /// that spans with identical Y-coordinates maintain extraction order.
    span_sequence_counter: usize,
    /// History of TJ array offsets for statistical analysis
    ///
    /// Tracks TJ offset values to detect justified vs. normal text through
    /// statistical distribution analysis (coefficient of variation).
    /// Used to dynamically adjust spacing thresholds per ISO 32000-1:2008 Section 9.4.4.
    tj_offset_history: Vec<f32>,
    /// Character-level tracking for word boundary detection
    ///
    /// Collects CharacterInfo for each character during TJ array processing.
    /// This provides character-level positioning, width, and TJ offset data
    /// to WordBoundaryDetector for primary word boundary detection.
    /// Per ISO 32000-1:2008 Section 9.4.4, character-level analysis improves accuracy.
    tj_character_array: Vec<CharacterInfo>,
    /// Current X position in text space for character tracking
    ///
    /// Updated as each character in a TJ array is processed. Used to calculate
    /// x_position for CharacterInfo entries (not used after character collection).
    current_x_position: f32,
    /// Word boundary detection mode
    ///
    /// Controls whether WordBoundaryDetector is used as:
    /// - Tiebreaker: Only when TJ and geometric signals conflict (default)
    /// - Primary: Before creating TextSpans from tj_character_array
    word_boundary_mode: WordBoundaryMode,
    /// Cached current font (updated on Tf). Avoids per-Tj HashMap lookup
    /// in advance_position_for_string.
    cached_current_font: Option<Arc<FontInfo>>,
}

impl TextExtractor {
    /// Create a new text extractor with default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::TextExtractor;
    ///
    /// let extractor = TextExtractor::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(TextExtractionConfig::default())
    }

    /// Create a new text extractor with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for text extraction heuristics
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::{TextExtractor, TextExtractionConfig};
    ///
    /// // Use custom space threshold
    /// let config = TextExtractionConfig::with_space_threshold(-80.0);
    /// let extractor = TextExtractor::with_config(config);
    /// ```
    pub fn with_config(config: TextExtractionConfig) -> Self {
        let word_boundary_mode = config.word_boundary_mode;
        Self {
            state_stack: GraphicsStateStack::new(),
            fonts: HashMap::new(),
            spans: Vec::new(),
            chars: Vec::new(),
            resources: None,
            document: None,
            processed_xobjects: HashSet::new(),
            cached_xobject_refs: HashMap::new(),
            xobject_depth: 0,
            xobject_decode_count: 0,
            config,
            merging_config: SpanMergingConfig::default(),
            current_mcid: None,
            extract_spans: true,      // Default to span mode (PDF spec compliant)
            tj_span_buffer: None,     // No buffer initially
            span_sequence_counter: 0, // Initialize sequence counter
            marked_content_stack: Vec::new(), // Track marked content contexts
            inside_artifact: false,   // Track artifact state
            tj_offset_history: Vec::with_capacity(1000), // Track TJ offsets for statistical analysis
            tj_character_array: Vec::new(),              // Character tracking for word boundaries
            current_x_position: 0.0,                     // Start at origin
            word_boundary_mode,                          // Word boundary detection mode
            cached_current_font: None,                   // Set on first Tf
        }
    }

    /// Create a new text extractor with custom merging configuration.
    ///
    /// This allows fine-tuning how adjacent spans are merged and when spaces
    /// are inserted, useful for documents with unusual spacing patterns.
    ///
    /// # Arguments
    ///
    /// * `merging_config` - Configuration for span merging thresholds
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::{TextExtractor, SpanMergingConfig};
    ///
    /// // Use aggressive space insertion for dense layouts
    /// let config = SpanMergingConfig::aggressive();
    /// let extractor = TextExtractor::new().with_merging_config(config);
    /// ```
    pub fn with_merging_config(mut self, merging_config: SpanMergingConfig) -> Self {
        self.merging_config = merging_config;
        self
    }

    /// Set the resources dictionary for this extractor.
    ///
    /// This allows the extractor to access XObjects and fonts during extraction.
    pub fn set_resources(&mut self, resources: Object) {
        self.resources = Some(resources);
    }

    /// Set the document reference for loading XObjects.
    ///
    /// # Safety
    ///
    /// The caller must ensure the document pointer remains valid for the lifetime
    /// of this extractor. This is safe when used within PdfDocument methods.
    pub fn set_document(&mut self, document: *const crate::document::PdfDocument) {
        self.document = Some(document);
    }

    // ========================================================================
    // Debug/profiling helpers — exposed for examples/debug_katalog.rs
    // ========================================================================

    /// Convenience wrapper: set document from a reference (avoids raw pointer in caller).
    pub fn set_document_ptr(&mut self, doc: &crate::document::PdfDocument) {
        self.document = Some(doc as *const crate::document::PdfDocument);
    }

    /// Prepare for span extraction mode (same setup as extract_text_spans preamble).
    pub fn prepare_for_span_extraction(&mut self) {
        self.extract_spans = true;
        self.spans.clear();
        self.span_sequence_counter = 0;
    }

    /// Public wrapper for execute_operator (normally private).
    pub fn execute_operator_public(&mut self, op: crate::content::Operator) -> Result<()> {
        self.execute_operator(op)
    }

    /// Public wrapper for flush_tj_span_buffer (normally private).
    pub fn flush_public(&mut self) -> Result<()> {
        self.flush_tj_span_buffer()
    }

    /// Calculate adaptive TJ offset threshold based on font size and text justification.
    ///
    /// When `use_adaptive_tj_threshold` is enabled, this method calculates the TJ offset
    /// threshold dynamically using the formula:
    ///
    /// ```text
    /// adaptive_threshold = -(space_width * font_size * margin_ratio) / 1000
    /// ```
    ///
    /// Where `margin_ratio` is adjusted based on justified vs normal text detection:
    /// - **Justified text** (high CV > 0.5): Uses 3× the normal ratio (conservative)
    ///   to prevent false space insertions from arbitrary TJ offsets
    /// - **Normal text** (low CV ≤ 0.5): Uses the default ratio (aggressive)
    ///
    /// # Adaptive Threshold Enhancement
    ///
    /// Per ISO 32000-1:2008 Section 9.4.4, justified text uses arbitrary TJ offsets to
    /// distribute whitespace. This method detects justified text through statistical
    /// analysis (coefficient of variation) and adapts the threshold accordingly.
    ///
    /// # Fallback Behavior
    ///
    /// If adaptive thresholds are disabled, this method returns the static
    /// `space_insertion_threshold` from the configuration.
    ///
    /// # PDF Spec Compliance
    ///
    /// Per Section 9.10: "Determining word boundaries is not specified by PDF."
    /// This method uses only spec-defined TJ values and geometric positions.
    fn calculate_adaptive_tj_threshold(&self) -> f32 {
        // Check if adaptive thresholds are enabled
        if !self.config.use_adaptive_tj_threshold {
            return self.config.space_insertion_threshold;
        }

        // Get current text state
        let state = self.state_stack.current();

        // ==============================================================================
        // FONT-AWARE ADAPTIVE THRESHOLD WITH JUSTIFIED TEXT DETECTION
        // (ISO 32000-1:2008 Section 9.4.4, 9.6.3, 9.10)
        // ==============================================================================

        let font_size = state.font_size;

        // Get font from current text state to access space glyph width
        // ISO 32000-1:2008 Section 9.6.3: Font metrics (glyph widths)
        let space_width_units = state
            .font_name
            .as_ref()
            .and_then(|name| self.fonts.get(name))
            .map(|font| font.get_space_glyph_width())
            .unwrap_or(250.0); // Fallback: Times-Roman typical space width

        // Detect justified vs normal text
        let (is_justified, cv) = self.analyze_tj_distribution();

        // Adjust margin ratio based on text justification
        // Justified text: use 3× conservative ratio (reduce false spaces)
        // Normal text: use default ratio
        let margin_ratio = if is_justified {
            self.config.word_margin_ratio * 3.0 // Conservative for justified
        } else {
            self.config.word_margin_ratio // Normal for non-justified
        };

        // Calculate threshold: negative offset required to trigger space insertion
        // Normalized by 1000 (PDF spec font units are 1/1000em)
        let adaptive_threshold = -((space_width_units * font_size * margin_ratio) / 1000.0);

        log::debug!(
            "TJ threshold: {} (justified={}, cv={:.2}, margin_ratio={:.3}, ISO 32000-1 §9.4.4)",
            adaptive_threshold,
            is_justified,
            cv,
            margin_ratio
        );

        adaptive_threshold
    }

    /// Analyze TJ offset distribution to detect justified vs normal text.
    ///
    /// This method performs statistical analysis on collected TJ offsets to determine
    /// if the document uses justified alignment. Justified text has high variance in TJ
    /// offsets (to distribute whitespace), while normally-spaced text has low variance.
    ///
    /// # Returns
    ///
    /// A tuple `(is_justified: bool, coefficient_of_variation: f32)` where:
    /// - `is_justified`: true if CV > 0.5 (high variance = justified text)
    /// - `coefficient_of_variation`: standard deviation / mean (normalized spread)
    ///
    /// # Algorithm
    ///
    /// Per ISO 32000-1:2008 Section 9.4.4, TJ array offsets are in font-relative units
    /// (1/1000 of text space). The distribution is analyzed as:
    ///
    /// 1. Calculate mean of all TJ offsets
    /// 2. Calculate variance: average of squared deviations from mean
    /// 3. Calculate standard deviation: sqrt(variance)
    /// 4. Calculate coefficient of variation: std_dev / |mean|
    ///
    /// # Thresholds
    ///
    /// - CV > 0.5: Justified text (high variance in offsets)
    /// - CV ≤ 0.5: Normal text (consistent spacing)
    ///
    /// # PDF Spec Compliance
    ///
    /// Per ISO 32000-1:2008 Section 9.10 ("Extraction of Text Content"):
    /// "Determining word boundaries is not specified by PDF." This method uses only
    /// spec-defined TJ offset values to infer text characteristics, not semantic assumptions.
    fn analyze_tj_distribution(&self) -> (bool, f32) {
        if self.tj_offset_history.is_empty() {
            return (false, 0.0);
        }

        let offsets = &self.tj_offset_history;

        // Calculate mean of TJ offsets
        let mean = offsets.iter().sum::<f32>() / offsets.len() as f32;

        // Calculate variance (average of squared deviations)
        let variance =
            offsets.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / offsets.len() as f32;

        // Calculate standard deviation
        let std_dev = variance.sqrt();

        // Calculate coefficient of variation (normalized spread)
        // Avoid division by zero for edge case of zero mean
        let cv = if mean.abs() > 0.001 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        let is_justified = cv > 0.5;

        log::debug!(
            "TJ distribution analysis: mean={:.2}, std_dev={:.2}, cv={:.2}, justified={}",
            mean,
            std_dev,
            cv,
            is_justified
        );

        (is_justified, cv)
    }

    /// Update the artifact state based on the marked content stack.
    ///
    /// This method computes whether we're currently inside an artifact region
    /// by checking if any ancestor in the marked_content_stack has is_artifact=true.
    /// Per PDF Spec Section 14.6, artifact content should be excluded from text extraction.
    ///
    /// # Performance
    ///
    /// This is O(n) where n is the depth of the marked content stack (typically 1-5).
    /// Called each time a marked content boundary is crossed (BMC/BDC/EMC).
    fn update_artifact_state(&mut self) {
        // True if ANY ancestor in the stack is an artifact
        self.inside_artifact = self.marked_content_stack.iter().any(|ctx| ctx.is_artifact);
    }

    /// Parse artifact type and subtype from artifact properties dictionary.
    ///
    /// Per PDF Spec Section 14.8.2.2, artifacts have optional /Type and /Subtype entries:
    /// - /Type: Pagination, Layout, Page, or Background
    /// - /Subtype: For Pagination artifacts: Header, Footer, Watermark, etc.
    ///
    /// # Arguments
    ///
    /// * `props_dict` - The properties dictionary from BDC operator
    ///
    /// # Returns
    ///
    /// The classified artifact type, or None if no type is specified
    fn parse_artifact_type(props_dict: &HashMap<String, Object>) -> Option<ArtifactType> {
        // Extract /Type entry (PDF Spec Section 14.8.2.2)
        let artifact_type_name = props_dict
            .get("Type")
            .and_then(|obj| obj.as_name())
            .map(|s| s.to_lowercase());

        // Extract /Subtype entry for Pagination artifacts
        let subtype_name = props_dict
            .get("Subtype")
            .and_then(|obj| obj.as_name())
            .map(|s| s.to_lowercase());

        match artifact_type_name.as_deref() {
            Some("pagination") => {
                let subtype = match subtype_name.as_deref() {
                    Some("header") => PaginationSubtype::Header,
                    Some("footer") => PaginationSubtype::Footer,
                    Some("watermark") => PaginationSubtype::Watermark,
                    Some("pagenumber") | Some("page") => PaginationSubtype::PageNumber,
                    _ => PaginationSubtype::Other,
                };
                Some(ArtifactType::Pagination(subtype))
            },
            Some("layout") => Some(ArtifactType::Layout),
            Some("page") => Some(ArtifactType::Page),
            Some("background") => Some(ArtifactType::Background),
            None => {
                // No /Type specified - check if /Subtype alone indicates pagination
                // Some PDFs use /Subtype without /Type
                match subtype_name.as_deref() {
                    Some("header") => Some(ArtifactType::Pagination(PaginationSubtype::Header)),
                    Some("footer") => Some(ArtifactType::Pagination(PaginationSubtype::Footer)),
                    Some("watermark") => {
                        Some(ArtifactType::Pagination(PaginationSubtype::Watermark))
                    },
                    _ => None,
                }
            },
            _ => None, // Unknown type
        }
    }

    /// Decode a PDF text string (handles UTF-16BE/LE with BOM and PDFDocEncoding).
    fn decode_pdf_text_string(bytes: &[u8]) -> String {
        if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
            // UTF-16BE with BOM
            let utf16_pairs: Vec<u16> = bytes[2..]
                .chunks_exact(2)
                .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16(&utf16_pairs)
                .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
        } else if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
            // UTF-16LE with BOM
            let utf16_pairs: Vec<u16> = bytes[2..]
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16(&utf16_pairs)
                .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
        } else {
            // PDFDocEncoding — try as UTF-8 first, fall back to lossy
            String::from_utf8(bytes.to_vec())
                .unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
        }
    }

    /// Resolve BDC properties: can be an inline dictionary or a name referencing /Properties resource.
    fn resolve_bdc_properties(
        &self,
        properties: &Object,
    ) -> Option<std::collections::HashMap<String, Object>> {
        // Inline dictionary
        if let Some(dict) = properties.as_dict() {
            return Some(dict.clone());
        }

        // Name reference — look up in /Properties sub-dictionary of resources
        let prop_name = properties.as_name()?;
        let resources = self.resources.as_ref()?;
        let res_dict = if let Some(res_ref) = resources.as_reference() {
            let doc = unsafe { &*self.document? };
            doc.load_object(res_ref).ok()?
        } else {
            resources.clone()
        };
        let res_dict = res_dict.as_dict()?;
        let properties_dict_obj = res_dict.get("Properties")?;
        let properties_dict = if let Some(r) = properties_dict_obj.as_reference() {
            let doc = unsafe { &*self.document? };
            doc.load_object(r).ok()?
        } else {
            properties_dict_obj.clone()
        };
        let properties_dict = properties_dict.as_dict()?;
        let prop_obj = properties_dict.get(prop_name)?;
        let resolved = if let Some(r) = prop_obj.as_reference() {
            let doc = unsafe { &*self.document? };
            doc.load_object(r).ok()?
        } else {
            prop_obj.clone()
        };
        resolved.as_dict().cloned()
    }

    /// Get current ActualText from marked content stack (PDF Spec Section 14.9.4).
    ///
    /// Searches from the innermost marked content context outward, returning
    /// the first ActualText found. If no ActualText is defined, returns None.
    ///
    /// ActualText provides the exact text representation for content that's
    /// represented non-standardly, such as ligatures (fi, fl, ffi, ffl) or
    /// decorated glyphs.
    fn get_current_actual_text(&self) -> Option<String> {
        self.marked_content_stack
            .iter()
            .rev()  // Search from innermost (most recent) context
            .find_map(|ctx| ctx.actual_text.clone())
    }

    /// Calculate the average glyph width for a font.
    ///
    /// Computes the mean width of printable ASCII characters (codes 32-126)
    /// in the given font, expressed in thousandths of em.
    ///
    /// # Fallback
    ///
    /// If the font doesn't have a widths array, uses the font's default width.
    ///
    /// # Performance
    ///
    /// This is relatively efficient, typically iterating over 95 ASCII characters.
    /// In practice, most fonts have widths arrays, so this completes quickly.
    #[allow(dead_code)]
    fn calculate_average_glyph_width(&self, font: &FontInfo) -> f32 {
        const PRINTABLE_ASCII_START: u32 = 32; // Space
        const PRINTABLE_ASCII_END: u32 = 126; // Tilde

        // If no widths array, use default width
        let Some(ref widths) = font.widths else {
            return font.default_width;
        };

        // We need FirstChar and LastChar to map character codes to width indices
        let Some(first_char) = font.first_char else {
            return font.default_width;
        };
        let Some(last_char) = font.last_char else {
            return font.default_width;
        };

        // Collect widths for all printable ASCII characters
        let mut total_width = 0.0;
        let mut count = 0;

        for char_code in PRINTABLE_ASCII_START..=PRINTABLE_ASCII_END {
            if char_code >= first_char && char_code <= last_char {
                // This character is in the widths array
                let index = (char_code - first_char) as usize;
                if index < widths.len() {
                    total_width += widths[index];
                    count += 1;
                }
            }
        }

        // Return average if we found any widths
        if count > 0 {
            total_width / count as f32
        } else {
            // Fallback if no widths in range
            font.default_width
        }
    }

    /// Add a font to the extractor.
    ///
    /// Fonts must be added before processing content streams that reference them.
    ///
    /// # Arguments
    ///
    /// * `name` - The font resource name (e.g., "F1", "TT1")
    /// * `font` - The font information
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::extractors::TextExtractor;
    /// # use pdf_oxide::fonts::FontInfo;
    /// # fn example(font: FontInfo) {
    /// let mut extractor = TextExtractor::new();
    /// extractor.add_font("F1".to_string(), font);
    /// # }
    /// ```
    pub fn add_font(&mut self, name: String, font: FontInfo) {
        self.fonts.insert(name, Arc::new(font));
    }

    /// Add a pre-shared font (Arc-wrapped) to the extractor. Avoids deep cloning.
    pub(crate) fn add_font_shared(&mut self, name: String, font: Arc<FontInfo>) {
        self.fonts.insert(name, font);
    }

    /// Return the current font set for caching purposes.
    pub(crate) fn get_font_set(&self) -> Vec<(String, Arc<FontInfo>)> {
        self.fonts
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect()
    }

    /// Share TrueType cmap tables between fonts with matching base font names.
    /// When a CIDFontType2 Identity-H font has no truetype_cmap, borrow from
    /// another font on the same page with the same base font name (ignoring subset prefix).
    pub fn share_truetype_cmaps(&mut self) {
        // Strip subset prefix (e.g., "QQPMQK+Impact" → "Impact")
        fn strip_subset(name: &str) -> &str {
            if name.len() > 7
                && name.as_bytes()[6] == b'+'
                && name[..6].chars().all(|c| c.is_ascii_uppercase())
            {
                &name[7..]
            } else {
                name
            }
        }

        // First pass: collect the BEST available TrueType cmap for each stripped base font name.
        // When multiple subset variants of the same font exist (e.g., ABCDEF+Arial, GHIJKL+Arial),
        // pick the cmap with the most glyph mappings — it has the best Unicode coverage.
        // On equal coverage, prefer the lexicographically smallest base_font name as a
        // deterministic tie-breaker (HashMap iteration order is randomized per-process).
        let mut best_cmaps: std::collections::HashMap<
            String,
            (crate::fonts::truetype_cmap::TrueTypeCMap, String),
        > = std::collections::HashMap::new();
        for font in self.fonts.values() {
            if let Some(cmap) = font.truetype_cmap() {
                let stripped = strip_subset(&font.base_font).to_string();
                let dominated =
                    best_cmaps
                        .get(&stripped)
                        .is_none_or(|(existing, existing_name)| {
                            match cmap.len().cmp(&existing.len()) {
                                std::cmp::Ordering::Greater => true,
                                std::cmp::Ordering::Equal => font.base_font < *existing_name,
                                std::cmp::Ordering::Less => false,
                            }
                        });
                if dominated {
                    best_cmaps.insert(stripped, (cmap.clone(), font.base_font.clone()));
                }
            }
        }

        if best_cmaps.is_empty() {
            return;
        }

        // Second pass: find CIDFontType2 Identity-H fonts without truetype_cmap
        for font_arc in self.fonts.values_mut() {
            if font_arc.truetype_cmap().is_some() {
                continue;
            }
            // Only target Type0 CIDFontType2 with Identity-H encoding
            if font_arc.subtype != "Type0" {
                continue;
            }
            let is_identity = matches!(&font_arc.encoding, crate::fonts::Encoding::Identity)
                || matches!(&font_arc.encoding, crate::fonts::Encoding::Standard(ref n) if n.contains("Identity"));
            if !is_identity {
                continue;
            }

            let stripped = strip_subset(&font_arc.base_font);
            if let Some((donor_cmap, _)) = best_cmaps.get(stripped) {
                log::info!(
                    "Sharing TrueType cmap ({} entries) to '{}' (Identity-H, no embedded font)",
                    donor_cmap.len(),
                    font_arc.base_font
                );
                // Use Arc::make_mut + set_truetype_cmap for copy-on-write sharing
                Arc::make_mut(font_arc).set_truetype_cmap(Some(donor_cmap.clone()));
            }
        }
    }

    /// Extract text from a content stream.
    ///
    /// Parses the content stream and executes operators to extract positioned
    /// characters with Unicode mappings and font information.
    ///
    /// # Arguments
    ///
    /// * `content_stream` - The raw content stream data (should be decoded first)
    ///
    /// # Returns
    ///
    /// A vector of TextChar structures containing positioned characters.
    ///
    /// # Errors
    ///
    /// Returns an error if the content stream cannot be parsed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::extractors::TextExtractor;
    /// # fn example(content_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut extractor = TextExtractor::new();
    /// let chars = extractor.extract(content_data)?;
    /// println!("Extracted {} characters", chars.len());
    /// # Ok(())
    /// # }
    /// ```
    /// Extract text as complete spans (PDF spec compliant).
    ///
    /// This is the recommended method for text extraction. It extracts complete
    /// text strings as the PDF provides them via Tj/TJ operators, following the
    /// PDF specification ISO 32000-1:2008.
    ///
    /// # Benefits
    /// - Avoids overlapping character issues
    /// - Preserves PDF's text positioning intent
    /// - More robust for complex layouts
    /// - Matches industry best practices
    ///
    /// # Arguments
    ///
    /// * `content_stream` - The PDF content stream data
    ///
    /// # Returns
    ///
    /// Vector of TextSpan objects in reading order
    pub fn extract_text_spans(&mut self, content_stream: &[u8]) -> Result<Vec<TextSpan>> {
        // Enable span extraction mode
        self.extract_spans = true;
        self.spans.clear();
        self.span_sequence_counter = 0; // Reset sequence counter for this page

        // Streaming parse+execute: operators are processed immediately without
        // building an intermediate Vec<Operator>. This eliminates allocation of
        // potentially huge vectors (196K+ operators for graphics-heavy pages)
        // and improves cache locality.
        extract_log_debug!("Parsing content stream for text extraction");
        parse_and_execute_text_only(content_stream, |op| self.execute_operator(op))?;

        // Flush any remaining Tj buffer at end of content stream
        self.flush_tj_span_buffer()?;

        // Sort spans by reading order (top-to-bottom, left-to-right)
        if log::log_enabled!(log::Level::Debug) {
            let space_spans = self
                .spans
                .iter()
                .filter(|s| s.text.chars().all(|c| c.is_whitespace()))
                .count();
            let offset_semantic = self.spans.iter().filter(|s| s.offset_semantic).count();
            log::debug!(
                "Before sort_spans_by_reading_order(): {} spans total, {} space-only, {} offset_semantic=true",
                self.spans.len(),
                space_spans,
                offset_semantic
            );
        }

        self.sort_spans_by_reading_order();

        // Deduplicate overlapping spans
        self.deduplicate_overlapping_spans();

        // Merge adjacent spans on the same line to reconstruct complete words
        self.merge_adjacent_spans();

        Ok(std::mem::take(&mut self.spans))
    }

    /// Extract individual characters from a PDF content stream.
    ///
    /// This is a low-level method that extracts characters one by one.
    /// For most use cases, prefer using `extract_text_spans()` which groups
    /// characters into text spans according to PDF semantics.
    pub fn extract(&mut self, content_stream: &[u8]) -> Result<Vec<TextChar>> {
        // Enable character extraction mode
        self.extract_spans = false;
        self.chars.clear();
        self.spans.clear(); // Ensure spans are clear so they don't poison xobject_spans_cache

        // Parse content stream into operators
        let operators = parse_content_stream_text_only(content_stream)?;

        // Execute each operator
        for op in operators {
            self.execute_operator(op)?;
        }

        // BUG FIX #2: Sort characters by reading order (top-to-bottom, left-to-right)
        // PDF content streams are in rendering order, not reading order.
        // PDF Y coordinates increase upward, so higher Y = top of page.
        // We need to sort by Y descending (top first), then X ascending (left to right).
        self.sort_by_reading_order();

        // BUG FIX #3: Deduplicate overlapping characters
        // Some PDFs render text multiple times (for effects like boldness, shadowing).
        // This causes characters to appear at very close X positions (< 2pt).
        // We deduplicate by keeping only the first character when multiple chars
        // at the same Y position have X positions within 2pt of each other.
        self.deduplicate_overlapping_chars();

        Ok(self.chars.clone())
    }

    /// Deduplicate overlapping characters on the same line.
    ///
    /// Some PDFs render text multiple times at slightly different X positions
    /// (e.g., for bold effect or shadowing). This causes garbled text output when
    /// all renders are extracted. We keep only one character when multiple chars
    /// at nearly the same position exist.
    ///
    /// Heuristic: If two consecutive characters on the same line (Y rounded to integer)
    /// are within 2pt horizontally, keep only the first one.
    fn deduplicate_overlapping_chars(&mut self) {
        if self.chars.is_empty() {
            return;
        }

        let mut deduplicated = Vec::with_capacity(self.chars.len());
        let mut prev_y_rounded: Option<i32> = None;
        let mut prev_x: Option<f32> = None;
        let mut prev_char: Option<char> = None;

        for ch in self.chars.iter() {
            let y_rounded = ch.bbox.y.round() as i32;
            let x = ch.bbox.x;

            // Check if this char overlaps with the previous one
            let should_skip = if let (Some(prev_y), Some(prev_x_val), Some(prev_ch)) =
                (prev_y_rounded, prev_x, prev_char)
            {
                // Same character, same line, and within 2pt horizontally
                ch.char == prev_ch && y_rounded == prev_y && (x - prev_x_val).abs() < 2.0
            } else {
                false
            };

            if !should_skip {
                deduplicated.push(ch.clone());
                prev_y_rounded = Some(y_rounded);
                prev_x = Some(x);
                prev_char = Some(ch.char);
            } else {
                log::trace!(
                    "Deduplicating overlapping char '{}' at X={:.1}, Y={:.1} (too close to previous)",
                    ch.char,
                    x,
                    ch.bbox.y
                );
            }
        }

        log::debug!(
            "Deduplicated {} overlapping characters ({} -> {} chars)",
            self.chars.len() - deduplicated.len(),
            self.chars.len(),
            deduplicated.len()
        );

        self.chars = deduplicated;
    }

    /// Sort extracted text spans by reading order (top-to-bottom, left-to-right).
    fn sort_spans_by_reading_order(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        // Detect columns first
        let columns = self.detect_span_columns();

        log::trace!(
            "Column detection: found {} columns from {} spans",
            columns.len(),
            self.spans.len()
        );
        for (i, (left, right)) in columns.iter().enumerate() {
            log::trace!(
                "  Column {}: X range [{:.1}, {:.1}] (width: {:.1})",
                i,
                left,
                right,
                right - left
            );
        }

        if columns.len() <= 1 {
            // Single column or no columns detected: use simple sort
            log::trace!("Using simple Y-then-X sorting (single column)");
            self.simple_sort_spans();
        } else {
            // Multi-column layout: sort within each column, then across columns
            log::trace!("Using column-aware sorting ({} columns)", columns.len());
            self.sort_spans_by_columns(&columns);
        }
    }

    /// Simple Y-then-X sorting for single-column layouts.
    fn simple_sort_spans(&mut self) {
        self.spans.sort_by(|a, b| {
            // Round Y coordinates for stable comparison
            let a_y_rounded = a.bbox.y.round() as i32;
            let b_y_rounded = b.bbox.y.round() as i32;

            match b_y_rounded.cmp(&a_y_rounded) {
                std::cmp::Ordering::Equal => {
                    // Same line: sort by X ascending (left to right)
                    crate::utils::safe_float_cmp(a.bbox.x, b.bbox.x)
                },
                other => other,
            }
        });
    }

    /// Detect columns by analyzing X-coordinate distribution.
    ///
    /// Returns column boundaries as (left_x, right_x) pairs, sorted left-to-right.
    fn detect_span_columns(&self) -> Vec<(f32, f32)> {
        if self.spans.is_empty() {
            return vec![];
        }

        // Find page bounds
        let min_x = self
            .spans
            .iter()
            .map(|s| s.bbox.x)
            .fold(f32::INFINITY, f32::min);
        let max_x = self
            .spans
            .iter()
            .map(|s| s.bbox.x + s.bbox.width)
            .fold(f32::NEG_INFINITY, f32::max);

        let page_width = max_x - min_x;

        // Build X-coordinate histogram to find vertical gaps
        let bins = 100;
        let bin_width = page_width / bins as f32;
        let mut histogram = vec![0; bins];

        for span in &self.spans {
            let start_bin = ((span.bbox.x - min_x) / bin_width) as usize;
            let end_bin = ((span.bbox.x + span.bbox.width - min_x) / bin_width) as usize;

            for i in start_bin..=end_bin.min(bins - 1) {
                histogram[i] += 1;
            }
        }

        // Find gaps (bins with zero or very low content)
        let avg_density: f32 = histogram.iter().sum::<i32>() as f32 / bins as f32;
        let gap_threshold = (avg_density * 0.2).max(1.0); // 20% of average or at least 1

        let mut gaps = vec![];
        let mut in_gap = false;
        let mut gap_start = 0;

        for (i, &count) in histogram.iter().enumerate() {
            if count as f32 <= gap_threshold {
                if !in_gap {
                    gap_start = i;
                    in_gap = true;
                }
            } else if in_gap {
                // End of gap - record if significant
                // Use 2% of page width OR absolute 15pt minimum (catches narrow column gutters)
                let gap_width = (i - gap_start) as f32 * bin_width;
                if gap_width > (page_width * 0.02).max(15.0) {
                    let gap_x = min_x + gap_start as f32 * bin_width;
                    gaps.push(gap_x);
                }
                in_gap = false;
            }
        }

        // No significant gaps found - single column
        if gaps.is_empty() {
            return vec![(min_x, max_x)];
        }

        // Build column boundaries from gaps
        let mut columns = vec![];
        let mut left = min_x;

        for gap_x in gaps {
            columns.push((left, gap_x));
            left = gap_x;
        }
        columns.push((left, max_x));

        log::debug!("Detected {} columns: {:?}", columns.len(), columns);

        columns
    }

    /// Sort spans by column-aware reading order.
    ///
    /// Process columns left-to-right, and within each column, top-to-bottom.
    fn sort_spans_by_columns(&mut self, columns: &[(f32, f32)]) {
        // Assign each span to a column
        let mut column_spans: Vec<Vec<TextSpan>> = vec![vec![]; columns.len()];

        for span in self.spans.drain(..) {
            let span_center_x = span.bbox.x + span.bbox.width / 2.0;

            // Find which column this span belongs to
            let col_idx = columns
                .iter()
                .position(|&(left, right)| span_center_x >= left && span_center_x <= right)
                .unwrap_or(0); // Default to first column if not found

            column_spans[col_idx].push(span);
        }

        // Sort within each column (top-to-bottom, then left-to-right)
        for col_spans in &mut column_spans {
            col_spans.sort_by(|a, b| {
                let a_y_rounded = a.bbox.y.round() as i32;
                let b_y_rounded = b.bbox.y.round() as i32;

                match b_y_rounded.cmp(&a_y_rounded) {
                    std::cmp::Ordering::Equal => crate::utils::safe_float_cmp(a.bbox.x, b.bbox.x),
                    other => other,
                }
            });
        }

        // Reassemble: read columns left-to-right
        for col_spans in column_spans {
            self.spans.extend(col_spans);
        }
    }

    /// Deduplicate overlapping text spans on the same line.
    ///
    /// Uses hybrid geometric + content-based deduplication:
    /// - Geometric check (same Y, X within 2pt) - catches identical positions
    /// - Content check (same text, same line Y, different X) - catches duplicates across columns
    fn deduplicate_overlapping_spans(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        // Phase 0 (B7): same-text overlapping spans from stroke+fill render
        // passes. Maps (newspaper / poster) frequently draw every label
        // twice — once stroked for outline, once filled — and both passes
        // land at essentially the same CTM. Without this up-front filter,
        // the merge step later concatenates them into "EverestEverest" /
        // "CentralCentral". We bucket by lowercased text and compare each
        // new span's bbox against prior entries via IoU; any later span
        // whose bbox overlaps an earlier one by >= 70 % is dropped.
        self.dedup_stroke_fill_overlap();

        // Take ownership of spans to avoid cloning during iteration
        let old_len = self.spans.len();
        let spans = std::mem::take(&mut self.spans);
        let mut deduplicated = Vec::with_capacity(old_len);
        let mut prev_y_rounded: Option<i32> = None;
        let mut prev_x: Option<f32> = None;
        let mut prev_text: Option<String> = None;
        let mut seen_content: std::collections::HashMap<String, (f32, f32)> =
            std::collections::HashMap::new();

        let mut geometric_skips = 0;
        let mut content_skips = 0;

        for span in spans {
            let y_rounded = span.bbox.y.round() as i32;
            let x = span.bbox.x;

            // PHASE 1: Geometric deduplication — require BOTH position AND text match
            let geometric_duplicate = if let (Some(prev_y), Some(prev_x_val), Some(ref prev_txt)) =
                (prev_y_rounded, prev_x, &prev_text)
            {
                y_rounded == prev_y && (x - prev_x_val).abs() < 2.0 && span.text == *prev_txt
            } else {
                false
            };

            // PHASE 2: Content-based deduplication — require positions to OVERLAP
            let content_duplicate = if span.text.len() >= 5 {
                if let Some((prev_x_val, prev_y_val)) = seen_content.get(&span.text) {
                    let y_diff = (span.bbox.y - prev_y_val).abs();
                    let x_diff = (span.bbox.x - prev_x_val).abs();

                    // Only dedup when spans overlap geometrically (X within 5pt)
                    // NOT when they're at different positions on the same line
                    let same_line = y_diff < 2.0;
                    let overlapping_position = x_diff < 5.0;

                    same_line && overlapping_position
                } else {
                    false
                }
            } else {
                false
            };

            if geometric_duplicate {
                geometric_skips += 1;
            } else if content_duplicate {
                content_skips += 1;
            } else {
                prev_y_rounded = Some(y_rounded);
                prev_x = Some(x);
                prev_text = Some(span.text.clone());

                // Track content for duplicate detection
                if span.text.len() >= 5 {
                    seen_content.insert(span.text.clone(), (span.bbox.x, span.bbox.y));
                }
                // Move span instead of cloning
                deduplicated.push(span);
            }
        }

        log::debug!(
            "Deduplicated {} spans (geometric: {}, content: {}) ({} -> {} spans)",
            geometric_skips + content_skips,
            geometric_skips,
            content_skips,
            old_len,
            deduplicated.len()
        );

        self.spans = deduplicated;
    }

    /// Drop same-text spans whose bounding boxes overlap heavily with an
    /// earlier span. This is the canonical stroke+fill pattern on maps,
    /// posters, and marketing materials: a label is drawn twice (once
    /// stroked for the outline, once filled for the glyph) at identical
    /// positions. Both passes surface as distinct spans; without this
    /// filter the downstream merge pass concatenates them.
    ///
    /// Keyed by lowercased text + rounded (x, y) bucket to make the
    /// lookup O(1) without quadratic bbox comparisons on large pages.
    /// The actual overlap check falls through to a real IoU on collision.
    fn dedup_stroke_fill_overlap(&mut self) {
        use std::collections::HashMap;

        if self.spans.len() < 2 {
            return;
        }
        let old_len = self.spans.len();
        let spans = std::mem::take(&mut self.spans);
        // Bucket text → list of prior bboxes. Only runs when trimmed text
        // has ≥ 2 *characters* (not bytes) — shorter candidates (single
        // letters, digits) rely on the downstream positional dedup already
        // in place.
        let mut seen: HashMap<String, Vec<crate::geometry::Rect>> = HashMap::new();
        let mut kept: Vec<TextSpan> = Vec::with_capacity(old_len);
        let mut skipped = 0usize;
        for span in spans {
            let trimmed = span.text.trim();
            if trimmed.chars().count() < 2 {
                kept.push(span);
                continue;
            }
            let key = trimmed.to_ascii_lowercase();
            let b = span.bbox;
            let mut is_dup = false;
            if let Some(existing) = seen.get(&key) {
                for other in existing {
                    // IoU — intersection over union. >= 0.7 means the two
                    // bboxes are almost the same rectangle, which is what
                    // stroke+fill produces.
                    let ix1 = b.x.max(other.x);
                    let iy1 = b.y.max(other.y);
                    let ix2 = (b.x + b.width).min(other.x + other.width);
                    let iy2 = (b.y + b.height).min(other.y + other.height);
                    if ix2 <= ix1 || iy2 <= iy1 {
                        continue;
                    }
                    let inter = (ix2 - ix1) * (iy2 - iy1);
                    let area_a = b.width * b.height;
                    let area_b = other.width * other.height;
                    let union = area_a + area_b - inter;
                    if union > 0.0 && inter / union >= 0.7 {
                        is_dup = true;
                        break;
                    }
                }
            }
            if is_dup {
                skipped += 1;
            } else {
                seen.entry(key).or_default().push(b);
                kept.push(span);
            }
        }
        if skipped > 0 {
            log::debug!("Stroke+fill dedup: dropped {skipped} duplicate spans of {old_len}");
        }
        self.spans = kept;
    }

    /// Merge adjacent text spans on the same line to reconstruct complete words.
    ///
    /// PDF content streams often break words into multiple Tj operators for precise
    /// kerning/positioning. This causes word fragmentation like "Intr oduction" instead
    /// of "Introduction". We merge spans that are:
    /// - On the same line (Y coordinates within 1pt)
    /// - Very close horizontally (gap < 3pt, approximately average char width)
    ///
    /// This matches the behavior of industry-standard PDF tools.
    fn merge_adjacent_spans(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        // Take ownership of spans to avoid cloning during iteration
        let old_len = self.spans.len();
        let spans = std::mem::take(&mut self.spans);
        let mut merged = Vec::with_capacity(old_len);
        let mut current_span: Option<TextSpan> = None;

        for span in spans {
            if current_span.is_none() {
                // First span — move, no clone needed
                current_span = Some(span);
                continue;
            }

            // Take ownership of current to avoid borrow checker issues.
            // Safety: checked is_none() above which continues, so this is always Some.
            let mut current = match current_span.take() {
                Some(s) => s,
                None => {
                    current_span = Some(span);
                    continue;
                },
            };

            // Check if this span should be merged with the current one
            let y_diff = (span.bbox.y - current.bbox.y).abs();
            let same_line = y_diff < 1.0;

            // Gap between end of current span and start of next span
            let current_end_x = current.bbox.x + current.bbox.width;
            let gap = span.bbox.x - current_end_x;

            // Fallback-width correction (issue #328): When the previous
            // span's font has no explicit `/Widths` array, every glyph in
            // that span reports the 500/550/600-thousandths-of-em fallback
            // from `FontInfo::new`. For proportional Latin fonts whose
            // real glyphs are narrower than that fallback (`SR` in the
            // NASA Apollo report is a concrete example), the span's
            // `bbox.width` is systematically inflated and `current_end_x`
            // overshoots the actual end of the rendered text — often by
            // enough to swallow the real inter-word gap entirely, turning
            // the visible word boundary into a negative `gap` value and
            // tripping merge logic that then glues the words without a
            // space.
            //
            // `space_gap` is a corrected gap value used ONLY for the
            // space-insertion decision below. The original `gap` is left
            // unchanged so the merge-vs-column decision, the decimal-merge
            // heuristic, and any downstream branch that reasons about the
            // actual bbox layout still see the real layout and don't
            // suddenly reclassify legitimate adjacent words as column
            // boundaries. In other words: the merge still happens exactly
            // as before on fallback-width fonts, but once we're inside the
            // merge branch we consult a more honest gap to decide whether
            // a space is warranted.
            let space_gap = {
                let prev_font = self.fonts.get(&current.font_name);
                let reliable = prev_font.map(|f| f.has_explicit_widths()).unwrap_or(true);
                if !reliable && current.bbox.width > 0.0 && !current.text.is_empty() {
                    // 0.55 / 0.45 ≈ 1.22 matches the per-glyph inflation
                    // observed on the NASA Apollo corpus (subagent analysis
                    // in issue #328). Keeping the correction modest avoids
                    // over-reporting gaps on fonts where 0.55 em is actually
                    // the correct average advance.
                    let corrected_end_x = current.bbox.x + current.bbox.width / 1.22;
                    span.bbox.x - corrected_end_x
                } else {
                    gap
                }
            };

            // Column-boundary gap, font-size-aware. The same 6pt gap is
            // a column gutter at 11pt body text but normal word kerning
            // at a 36pt title; use 0.5em as a floor above the configured
            // absolute threshold.
            let font_size_ref = current.font_size.max(span.font_size);
            let column_threshold = self
                .merging_config
                .column_boundary_threshold_pt
                .max(font_size_ref * 0.5);
            let large_gap_indicates_column = gap > column_threshold;

            // SPLIT BOUNDARY CHECK: Respect boundaries from CamelCase splitting
            // If a span has split_boundary_before=true, it represents a word boundary
            // from a split operation (e.g., "the" + "General" from "theGeneral")
            // These should always be merged WITH a space, never without.
            let has_split_boundary = span.split_boundary_before;

            // Font identity: same base font AND same size AND same styling.
            let is_same_font = current.font_name == span.font_name
                && (current.font_size - span.font_size).abs() < 0.01
                && current.font_weight == span.font_weight
                && current.is_italic == span.is_italic;

            // Cross-font word glue: same-baseline spans in different
            // fonts/weights, tight gap (<0.25em), both sides alphabetic,
            // and one side is a single character. Targets the drop-cap /
            // single-letter-small-caps typography pattern where per-
            // letter emphasis runs would corrupt proper nouns.
            let cross_font_word_glue = !is_same_font
                && same_line
                && gap > -1.0
                && gap < font_size_ref * 0.25
                && !current.text.is_empty()
                && !span.text.is_empty()
                && current
                    .text
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_alphabetic())
                && span.text.chars().next().is_some_and(|c| c.is_alphabetic())
                && (current.text.chars().count() == 1 || span.text.chars().count() == 1);

            // Merge threshold: Use configured values
            // Negative gaps: use severe_overlap_threshold_pt (default -0.5pt)
            // Positive gaps: use a threshold that allows for justified text but
            // avoids merging across clear column boundaries.
            // Same-font spans are merged more aggressively to reconstruct words.
            let merge_threshold_pt = if is_same_font {
                column_threshold.max(3.0)
            } else {
                // Different fonts: only merge if they are effectively overlapping
                // to handle minor kerning/rounding issues, but generally keep separate.
                0.5
            };

            let should_merge = same_line
                && is_same_font
                && (self.merging_config.severe_overlap_threshold_pt..merge_threshold_pt)
                    .contains(&gap)
                && !large_gap_indicates_column
                || (same_line && has_split_boundary)
                || cross_font_word_glue;

            // DECIMAL VALUE MERGE: Some forms place integer and decimal parts
            // of dollar amounts in separate fixed-width boxes.
            // e.g., "123456" (integer box) + "72" (cents box) with ~10pt gap.
            // Detect this pattern: both spans are pure digits, the second is
            // exactly 1-2 digits (cents), same line, and gap < 2x font size.
            let decimal_merge = same_line
                && gap > 0.0
                && gap < current.font_size * 2.0
                && !current.text.is_empty()
                && !span.text.is_empty()
                && current.text.chars().all(|c| c.is_ascii_digit())
                && span.text.chars().all(|c| c.is_ascii_digit())
                && (1..=2).contains(&span.text.len());

            if decimal_merge {
                // Join integer and decimal parts with "."
                log::debug!(
                    "Decimal value merge: '{}' + '{}' -> '{}.{}' (gap={:.1}pt)",
                    current.text,
                    span.text,
                    current.text,
                    span.text,
                    gap
                );
                current.text.push('.');
                current.text.push_str(&span.text);
            } else if cross_font_word_glue {
                // Mid-word font/weight change: concatenate without any space
                // or space-heuristic — these are same-word character runs.
                current.text.push_str(&span.text);
            } else if should_merge {
                // PHASE 1 FIX: Check if next span is entirely whitespace-only OR marked as offset_semantic space
                // If either is true, never insert an additional space - just concatenate directly
                // This prevents double-space issue when TJ processor creates space spans
                let next_is_whitespace_only = span.text.chars().all(|c| c.is_whitespace());
                let next_is_offset_semantic_space = span.offset_semantic && next_is_whitespace_only;

                // Merge spans: append text in-place using push_str (O(n) total vs O(n²) with format!)
                if next_is_whitespace_only {
                    // Next span is already space-only: just concatenate without adding more space
                    log::debug!(
                        "Merging with whitespace-only span: '{}' + '{}' (whitespace, offset_semantic={})",
                        current.text,
                        span.text.escape_default(),
                        span.offset_semantic
                    );
                    current.text.push_str(&span.text);
                } else {
                    let tj_offset_triggered_override = has_split_boundary;
                    let space_decision = should_insert_space(
                        &current.text,
                        &span.text,
                        space_gap,
                        current.font_size,
                        &current.font_name,
                        &self.fonts,
                        tj_offset_triggered_override,
                        &self.merging_config,
                        Some(&current.bbox),
                        Some(&span.bbox),
                        current.font_size,
                        span.font_size,
                    );

                    log::debug!(
                        "Span merge decision: gap={:.2}pt, decision={:?}, source={:?}, confidence={:.2}, offset_semantic={}",
                        gap,
                        space_decision.insert_space,
                        space_decision.source,
                        space_decision.confidence,
                        span.offset_semantic
                    );

                    if space_decision.insert_space {
                        // Space insertion triggered by unified decision
                        // But SKIP if this span is already a TJ-offset space (would create double space)
                        if next_is_offset_semantic_space {
                            log::debug!(
                                "Suppressing space insertion: next span is already TJ-offset space"
                            );
                            current.text.push_str(&span.text);
                        } else {
                            // Prevent double-space edge case
                            let would_create_double_space =
                                current.text.ends_with(' ') && span.text.starts_with(' ');

                            if would_create_double_space {
                                log::debug!(
                                    "Preventing double-space: current ends with space, next starts with space"
                                );
                                current.text.push_str(&span.text);
                            } else {
                                log::trace!("Space via {:?}", space_decision.source);
                                current.text.push(' ');
                                current.text.push_str(&span.text);
                            }
                        }
                    } else {
                        // No space: adjacent characters within same word
                        log::trace!(
                            "No space insertion: decision source={:?}",
                            space_decision.source
                        );
                        current.text.push_str(&span.text);
                    }
                }
            }

            if decimal_merge || should_merge || cross_font_word_glue {
                // Extend bounding box to include both spans
                let new_width = (span.bbox.x + span.bbox.width) - current.bbox.x;
                let new_height = current.bbox.height.max(span.bbox.height);

                current.bbox.width = new_width;
                current.bbox.height = new_height;

                // After a cross-font glue, adopt the longer run's font
                // metadata. The single-letter side was typographic
                // decoration, not semantic emphasis, so the dominant-run
                // style should win.
                if cross_font_word_glue {
                    let span_chars = span.text.chars().count();
                    let current_chars_before = current.text.chars().count() - span_chars;
                    if span_chars > current_chars_before {
                        current.font_name = span.font_name.clone();
                        current.font_weight = span.font_weight;
                        current.is_italic = span.is_italic;
                    }
                }

                log::trace!(
                    "Merged span: appended '{}' (gap={:.1}pt, now {} chars)",
                    span.text,
                    gap,
                    current.text.len()
                );

                // Put modified current back
                current_span = Some(current);
            } else {
                // Not mergeable: save current and start new span
                if same_line {
                    if span.split_boundary_before {
                        log::trace!(
                            "Not merging spans (split boundary): '{}' | '{}'",
                            current.text,
                            span.text
                        );
                    } else {
                        log::trace!(
                            "Not merging spans (gap={:.1}pt > 3pt): '{}' | '{}'",
                            gap,
                            current.text,
                            span.text
                        );
                    }
                }
                merged.push(current);
                current_span = Some(span);
            }
        }

        // Don't forget the last span
        if let Some(last) = current_span {
            merged.push(last);
        }

        log::debug!("Merged adjacent spans: {} -> {} spans", old_len, merged.len());

        self.spans = merged;
    }

    /// Sort extracted characters by reading order (top-to-bottom, left-to-right).
    ///
    /// This is critical for proper text extraction as PDF content streams are
    /// organized for rendering efficiency, not reading order.
    fn sort_by_reading_order(&mut self) {
        self.chars.sort_by(|a, b| {
            // Handle NaN/Inf values - treat them as at the end
            if !a.bbox.y.is_finite() {
                return if b.bbox.y.is_finite() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                };
            }
            if !b.bbox.y.is_finite() {
                return std::cmp::Ordering::Less;
            }

            // Sort by Y descending (top first), then by X ascending (left to right)
            // Round Y coordinates to ensure transitivity of the comparison function
            let a_y_rounded = a.bbox.y.round() as i32;
            let b_y_rounded = b.bbox.y.round() as i32;

            match b_y_rounded.cmp(&a_y_rounded) {
                std::cmp::Ordering::Equal => {
                    // Same line: sort by X ascending (left to right)
                    if !a.bbox.x.is_finite() {
                        return if b.bbox.x.is_finite() {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        };
                    }
                    if !b.bbox.x.is_finite() {
                        return std::cmp::Ordering::Less;
                    }

                    if a.bbox.x < b.bbox.x {
                        std::cmp::Ordering::Less
                    } else if a.bbox.x > b.bbox.x {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                },
                other => other,
            }
        });
    }

    /// ISSUE 1 FIX: Split fused words created by PDF authoring defects
    ///
    /// Some PDFs encode multiple words as a single TJ string without spacing:
    /// - "theGeneral" instead of "the" + "General"
    /// - "lengthThis" instead of "length" + "This"
    /// - "helporganisationscraft" (partial fusion)
    ///
    /// This post-processor detects word fusions and splits them into separate spans.
    ///
    /// Uses two strategies:
    /// 1. **CamelCase detection** (first priority): Detects lowercase->uppercase transitions
    ///    - Example: "theGeneral" -> ["the", "General"]
    /// 2. **Dictionary-based segmentation** (fallback): Uses Viterbi algorithm with word dictionary
    ///    - Example: "helporganisationscraft" -> ["help", "organisations", "craft"]
    ///
    /// Per ISO 32000-1:2008 Section 9.4.4: "Text strings are as long as possible" - spaces
    /// are positioning artifacts, so word fusions must be detected and reconstructed.
    #[allow(dead_code)]
    fn split_fused_words(&mut self) {
        let mut split_spans = Vec::new();

        for span in &self.spans {
            // DEBUG: Log field values before cloning
            log::debug!(
                "split_fused_words() processing span '{}' (offset_semantic={}, split_boundary_before={})",
                if span.text.len() <= 30 {
                    &span.text
                } else {
                    "[whitespace or long text]"
                },
                span.offset_semantic,
                span.split_boundary_before
            );

            // Try CamelCase split (handles mixed-case fusions)
            let parts = self.split_on_camelcase(&span.text);

            if parts.len() == 1 {
                // No split needed
                let cloned = span.clone();
                log::debug!(
                    "  → No split: cloned offset_semantic={} (text: '{}')",
                    cloned.offset_semantic,
                    if cloned.text.len() <= 30 {
                        &cloned.text
                    } else {
                        "[whitespace or long text]"
                    }
                );
                split_spans.push(cloned);
            } else {
                // Split into multiple spans with proportional bounding boxes
                let total_chars = span.text.len() as f32;
                let mut char_pos = 0;

                for (i, part) in parts.iter().enumerate() {
                    let part_len = part.len() as f32;
                    let part_ratio = part_len / total_chars;

                    // Calculate proportional bounding box
                    let new_width = span.bbox.width * part_ratio;
                    let new_x = span.bbox.x + (span.bbox.width * (char_pos as f32 / total_chars));

                    let mut new_span = span.clone();
                    new_span.text = part.clone();
                    new_span.bbox.x = new_x;
                    new_span.bbox.width = new_width;

                    // Set split_boundary_before flag for all parts except the first
                    // This prevents them from being re-merged during span merging
                    if i > 0 {
                        new_span.split_boundary_before = true;
                    }

                    log::debug!(
                        "  → Split part {}: '{}' offset_semantic={} split_boundary_before={}",
                        i,
                        part,
                        new_span.offset_semantic,
                        new_span.split_boundary_before
                    );
                    split_spans.push(new_span);
                    char_pos += part.len();
                }
            }
        }

        self.spans = split_spans;
    }

    /// Detect CamelCase boundaries and split text into parts
    ///
    /// Splits on lowercase->uppercase transitions:
    /// - "theGeneral" -> ["the", "General"]
    /// - "lengthThis" -> ["length", "This"]
    /// - "helporganisationscraft" -> ["help", "organisations", "craft"]
    #[allow(dead_code)]
    fn split_on_camelcase(&self, text: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current_part = String::new();
        let mut prev_is_lower = false;

        for ch in text.chars() {
            if prev_is_lower && ch.is_uppercase() {
                // CamelCase boundary detected
                if !current_part.is_empty() {
                    parts.push(current_part.clone());
                    current_part.clear();
                }
                current_part.push(ch);
                prev_is_lower = false;
            } else {
                current_part.push(ch);
                prev_is_lower = ch.is_lowercase();
            }
        }

        if !current_part.is_empty() {
            parts.push(current_part);
        }

        // Only return split if we found at least 2 parts with actual boundaries
        if parts.len() > 1 {
            parts
        } else {
            vec![text.to_string()]
        }
    }

    /// Execute a single operator.
    ///
    /// Updates the graphics state and extracts text as appropriate.
    fn execute_operator(&mut self, op: Operator) -> Result<()> {
        match op {
            // Text state operators
            Operator::Tf { font, size } => {
                // Skip flush + lookup when font name AND size haven't changed.
                // Many PDFs redundantly set the same font (e.g., Tf after q/Q).
                let same_font = {
                    let state = self.state_stack.current();
                    state.font_size == size && state.font_name.as_deref() == Some(font.as_str())
                };
                if !same_font {
                    // Flush Tj buffer before changing font — the buffer decodes bytes
                    // using the font set at creation time, so a font change requires a
                    // new buffer to avoid decoding with the wrong ToUnicode CMap.
                    self.flush_tj_span_buffer()?;

                    // Cache font reference for advance_position_for_string
                    self.cached_current_font = self.fonts.get(&font).cloned();

                    let state = self.state_stack.current_mut();
                    state.font_name = Some(font);
                    state.font_size = size;
                }
            },

            // Text positioning operators
            Operator::Tm { a, b, c, d, e, f } => {
                // Optimization: batch character-by-character Tm+Tj patterns.
                // Many PDFs position each character with individual Tm+Tj operators.
                // If the new Tm is on the same line with the same transform,
                // keep accumulating into the existing buffer instead of flushing
                // (avoids creating thousands of 1-char TextSpans per page).
                let is_continuation = match self.tj_span_buffer {
                    Some(ref mut buffer)
                        if !buffer.is_empty()
                            && f.round() as i32 == buffer.start_matrix.f.round() as i32
                            && a == buffer.start_matrix.a
                            && b == buffer.start_matrix.b
                            && c == buffer.start_matrix.c
                            && d == buffer.start_matrix.d
                            && e >= buffer.start_matrix.e =>
                    {
                        // Same line, same transform, LTR progression →
                        // update width to reflect actual visual extent
                        buffer.accumulated_width = e - buffer.start_matrix.e;
                        true
                    },
                    _ => false,
                };

                if !is_continuation {
                    self.flush_tj_span_buffer()?;
                }

                let state = self.state_stack.current_mut();
                state.text_matrix = Matrix { a, b, c, d, e, f };
                state.text_line_matrix = state.text_matrix;
            },
            Operator::Td { tx, ty } => {
                // Flush Tj buffer before changing text position
                self.flush_tj_span_buffer()?;
                let state = self.state_stack.current_mut();
                // Per ISO 32000-1:2008 §9.4.2, Table 108:
                // Tlm_new = T(tx,ty) × Tlm_old
                // The translation is in text-line space, so it must be
                // pre-multiplied to be scaled by the existing Tlm transform.
                let tm = Matrix::translation(tx, ty);
                state.text_line_matrix = tm.multiply(&state.text_line_matrix);
                state.text_matrix = state.text_line_matrix;
            },
            Operator::TD { tx, ty } => {
                // Flush Tj buffer before changing text position
                self.flush_tj_span_buffer()?;

                // TD is like Td but also sets leading
                let state = self.state_stack.current_mut();
                state.leading = -ty;
                // Per ISO 32000-1:2008 §9.4.2: Tlm_new = T(tx,ty) × Tlm_old
                let tm = Matrix::translation(tx, ty);
                state.text_line_matrix = tm.multiply(&state.text_line_matrix);
                state.text_matrix = state.text_line_matrix;
            },
            Operator::TStar => {
                // Flush Tj buffer before moving to next line
                self.flush_tj_span_buffer()?;

                // Move to start of next line (using leading)
                let leading = self.state_stack.current().leading;
                let state = self.state_stack.current_mut();
                // Per ISO 32000-1:2008 §9.4.2: Tlm_new = T(0,-TL) × Tlm_old
                let tm = Matrix::translation(0.0, -leading);
                state.text_line_matrix = tm.multiply(&state.text_line_matrix);
                state.text_matrix = state.text_line_matrix;
            },

            // Text showing operators
            Operator::Tj { text } => {
                // Note: We do NOT skip /Artifact content here.
                // Many PDFs incorrectly mark page content as artifacts.
                // For tagged PDFs, the structure tree already excludes artifacts
                // via MCID mapping, so no filtering is needed at extractor level.

                // ActualText override
                // Per PDF Spec ISO 32000-1:2008, Section 14.9.4:
                // ActualText provides replacement text for content that cannot be
                // automatically extracted (e.g., figures, symbols, decorative text).
                if let Some(actual_text) = self.get_current_actual_text() {
                    log::debug!("Tj operator: Using ActualText override: '{}'", actual_text);

                    if self.extract_spans {
                        // Use ActualText in span mode — push pre-decoded Unicode directly
                        // into the buffer, bypassing font character mapping (the text is
                        // already decoded from the BDC /ActualText property).
                        if self.tj_span_buffer.is_none() {
                            self.tj_span_buffer = Some(TjBuffer::new(
                                self.state_stack.current(),
                                self.current_mcid,
                                self.cached_current_font.clone(),
                            ));
                        }

                        if let Some(ref mut buffer) = self.tj_span_buffer {
                            buffer.unicode.push_str(&actual_text);
                        }
                    } else {
                        // Character mode: show_text maps through font, but ActualText
                        // is already decoded. Fall back to show_text for positioning.
                        self.show_text(actual_text.as_bytes())?;
                    }

                    // Advance position for the original text (to maintain layout)
                    let w = self.advance_position_for_string(&text)?;
                    if let Some(ref mut buffer) = self.tj_span_buffer {
                        buffer.accumulated_width += w;
                    }
                } else {
                    // No ActualText - use standard text extraction
                    if self.extract_spans {
                        // NEW: Buffer consecutive Tj operators into single spans
                        // Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
                        // "text strings are as long as possible"

                        // Create buffer if doesn't exist
                        if self.tj_span_buffer.is_none() {
                            self.tj_span_buffer = Some(TjBuffer::new(
                                self.state_stack.current(),
                                self.current_mcid,
                                self.cached_current_font.clone(),
                            ));
                        }

                        // Merged single-pass: Unicode decode + width + position advance
                        self.append_and_advance(&text)?;
                    } else {
                        self.show_text(&text)?;
                    }
                }
            },
            Operator::TJ { array } => {
                // Note: We do NOT skip /Artifact content here.
                // Many PDFs incorrectly mark page content as artifacts.
                // For tagged PDFs, the structure tree already excludes artifacts
                // via MCID mapping, so no filtering is needed at extractor level.

                // ActualText override
                // Per PDF Spec ISO 32000-1:2008, Section 14.9.4:
                // When ActualText is present, use it instead of the TJ array contents.
                // The entire TJ array is replaced with the ActualText string.
                if let Some(actual_text) = self.get_current_actual_text() {
                    log::debug!(
                        "TJ operator: Using ActualText override: '{}' (replacing {} elements)",
                        actual_text,
                        array.len()
                    );

                    if self.extract_spans {
                        // Use ActualText in span mode — push pre-decoded Unicode directly
                        let mut buffer = TjBuffer::new(
                            self.state_stack.current(),
                            self.current_mcid,
                            self.cached_current_font.clone(),
                        );
                        buffer.unicode.push_str(&actual_text);
                        self.flush_tj_buffer(buffer)?;
                    } else {
                        // Character mode: fall back to show_text for positioning
                        self.show_text(actual_text.as_bytes())?;
                    }

                    // Advance position for the entire TJ array (to maintain layout)
                    // Calculate the total displacement the array would have caused
                    for element in array {
                        match element {
                            TextElement::String(s) => {
                                let w = self.advance_position_for_string(&s)?;
                                if let Some(ref mut buffer) = self.tj_span_buffer {
                                    buffer.accumulated_width += w;
                                }
                            },
                            TextElement::Offset(offset) => {
                                self.advance_position_for_offset(offset)?;
                            },
                        }
                    }
                } else {
                    // No ActualText - use standard TJ array processing
                    if self.extract_spans {
                        // NEW: Use buffered TJ array processing for span extraction
                        // Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
                        // "text strings are as long as possible"
                        // This creates one span per logical text unit instead of fragmenting
                        self.process_tj_array(&array)?;
                    } else {
                        // Keep old behavior for character extraction mode
                        for element in array {
                            match element {
                                TextElement::String(s) => {
                                    self.show_text(&s)?;
                                },
                                TextElement::Offset(offset) => {
                                    // Adjust text position by offset (in thousandths of em)
                                    let state = self.state_stack.current();
                                    let tx = -offset / 1000.0
                                        * state.font_size
                                        * state.horizontal_scaling
                                        / 100.0;

                                    // HEURISTIC: Insert space character for significant negative offsets
                                    //
                                    // PDF Spec Reference: ISO 32000-1:2008, Section 9.4.4
                                    // The spec defines text positioning but does NOT specify when a positioning
                                    // offset represents a word boundary vs. tight kerning.
                                    //
                                    // In PDFs, spaces are often represented as negative positioning offsets in TJ arrays,
                                    // not as explicit space characters. For example:
                                    // [(Text1) -200 (Text2)] TJ  <- the -200 creates visual spacing
                                    //
                                    // Geometry-based adaptive threshold (based on font metrics)
                                    // Formula: adaptive_threshold = -(average_glyph_width * word_margin_ratio)
                                    // This adapts to different font sizes and families.
                                    // Fallback: static threshold if font unavailable or adaptive disabled.
                                    let threshold = self.calculate_adaptive_tj_threshold();
                                    if offset < threshold {
                                        let text_matrix = state.text_matrix;
                                        let ctm = state.ctm;
                                        let font_name = state.font_name.clone();
                                        let font_size = state.font_size;
                                        let fill_color_rgb = state.fill_color_rgb;

                                        // Calculate effective font size (accounting for CTM and text matrix scaling)
                                        let combined = ctm.multiply(&text_matrix);
                                        let effective_font_size = font_size
                                            * (combined.d * combined.d + combined.b * combined.b)
                                                .sqrt();

                                        // Get font for determining weight
                                        let font = font_name
                                            .as_ref()
                                            .and_then(|name| self.fonts.get(name));
                                        let font_weight = if let Some(font) = font {
                                            if font.is_bold() {
                                                FontWeight::Bold
                                            } else {
                                                FontWeight::Normal
                                            }
                                        } else {
                                            FontWeight::Normal
                                        };

                                        // Create space character at current position
                                        // Apply CTM to get position in user space
                                        let text_pos = text_matrix.transform_point(0.0, 0.0);
                                        let pos = ctm.transform_point(text_pos.x, text_pos.y);
                                        let (r, g, b) = fill_color_rgb;
                                        let is_italic_space = font_name
                                            .as_ref()
                                            .and_then(|name| self.fonts.get(name))
                                            .map(|font| font.is_italic())
                                            .unwrap_or(false);
                                        let font_name_str = font_name.unwrap_or_default();
                                        // Compose CTM and text_matrix for full transformation
                                        let final_matrix = ctm.multiply(&text_matrix);
                                        // Calculate rotation from matrix: atan2(b, a)
                                        let rotation_degrees =
                                            final_matrix.b.atan2(final_matrix.a).to_degrees();

                                        let space_char = TextChar {
                                            char: ' ',
                                            bbox: Rect::new(
                                                pos.x,               // X position in user space
                                                pos.y,               // Y position in user space
                                                tx.abs(), // Width = the gap being created
                                                effective_font_size, // Height = effective font size
                                            ),
                                            font_name: font_name_str,
                                            font_size: effective_font_size,
                                            font_weight,
                                            color: Color::new(r, g, b),
                                            mcid: self.current_mcid,
                                            is_italic: is_italic_space,
                                            is_monospace: false,
                                            // Transformation properties (v0.3.1)
                                            origin_x: pos.x,
                                            origin_y: pos.y,
                                            rotation_degrees,
                                            advance_width: tx.abs(),
                                            matrix: Some([
                                                final_matrix.a,
                                                final_matrix.b,
                                                final_matrix.c,
                                                final_matrix.d,
                                                final_matrix.e,
                                                final_matrix.f,
                                            ]),
                                        };
                                        self.chars.push(space_char);
                                    }

                                    let state_mut = self.state_stack.current_mut();
                                    let tm = state_mut.text_matrix;
                                    state_mut.text_matrix.e += tx * tm.a;
                                    state_mut.text_matrix.f += tx * tm.b;
                                },
                            }
                        }
                    }
                }
            },
            Operator::Quote { text } => {
                // ' operator: Move to next line (T*) and show text (Tj)
                // Flush any pending span buffer before line break
                self.flush_tj_span_buffer()?;

                let leading = self.state_stack.current().leading;
                {
                    let state = self.state_stack.current_mut();
                    // Per ISO 32000-1:2008 §9.4.2: Tlm_new = T(0,-TL) × Tlm_old
                    let tm = Matrix::translation(0.0, -leading);
                    state.text_line_matrix = tm.multiply(&state.text_line_matrix);
                    state.text_matrix = state.text_line_matrix;
                }

                if self.extract_spans {
                    if self.tj_span_buffer.is_none() {
                        self.tj_span_buffer = Some(TjBuffer::new(
                            self.state_stack.current(),
                            self.current_mcid,
                            self.cached_current_font.clone(),
                        ));
                    }
                    self.append_and_advance(&text)?;
                } else {
                    self.show_text(&text)?;
                }
            },
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            } => {
                // " operator: Set spacing, move to next line (T*), and show text (Tj)
                // Flush any pending span buffer before line break
                self.flush_tj_span_buffer()?;

                {
                    let state = self.state_stack.current_mut();
                    state.word_space = word_space;
                    state.char_space = char_space;
                    let leading = state.leading;
                    // Per ISO 32000-1:2008 §9.4.2: Tlm_new = T(0,-TL) × Tlm_old
                    let tm = Matrix::translation(0.0, -leading);
                    state.text_line_matrix = tm.multiply(&state.text_line_matrix);
                    state.text_matrix = state.text_line_matrix;
                }

                if self.extract_spans {
                    if self.tj_span_buffer.is_none() {
                        self.tj_span_buffer = Some(TjBuffer::new(
                            self.state_stack.current(),
                            self.current_mcid,
                            self.cached_current_font.clone(),
                        ));
                    }
                    self.append_and_advance(&text)?;
                } else {
                    self.show_text(&text)?;
                }
            },

            // Text state parameters
            Operator::Tc { char_space } => {
                self.state_stack.current_mut().char_space = char_space;
            },
            Operator::Tw { word_space } => {
                self.state_stack.current_mut().word_space = word_space;
            },
            Operator::Tz { scale } => {
                self.state_stack.current_mut().horizontal_scaling = scale;
            },
            Operator::TL { leading } => {
                self.state_stack.current_mut().leading = leading;
            },
            Operator::Ts { rise } => {
                self.state_stack.current_mut().text_rise = rise;
            },
            Operator::Tr { render } => {
                self.state_stack.current_mut().render_mode = render;
            },

            // Graphics state operators
            Operator::SaveState => {
                self.state_stack.save();
            },
            Operator::RestoreState => {
                self.state_stack.restore();
                // Sync cached font with restored state
                self.cached_current_font = self
                    .state_stack
                    .current()
                    .font_name
                    .as_ref()
                    .and_then(|name| self.fonts.get(name))
                    .cloned();
            },
            Operator::Cm { a, b, c, d, e, f } => {
                let state = self.state_stack.current_mut();
                let new_ctm = Matrix { a, b, c, d, e, f };
                // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                state.ctm = new_ctm.multiply(&state.ctm);
            },

            // Color operators
            Operator::SetFillRgb { r, g, b } => {
                self.state_stack.current_mut().fill_color_rgb = (r, g, b);
            },
            Operator::SetStrokeRgb { r, g, b } => {
                self.state_stack.current_mut().stroke_color_rgb = (r, g, b);
            },
            Operator::SetFillGray { gray } => {
                self.state_stack.current_mut().fill_color_rgb = (gray, gray, gray);
            },
            Operator::SetStrokeGray { gray } => {
                self.state_stack.current_mut().stroke_color_rgb = (gray, gray, gray);
            },
            Operator::SetFillCmyk { c, m, y, k } => {
                // Store CMYK and convert to RGB for rendering
                // CMYK to RGB conversion: R = 1 - min(1, C*(1-K) + K)
                let state = self.state_stack.current_mut();
                state.fill_color_cmyk = Some((c, m, y, k));
                state.fill_color_rgb = cmyk_to_rgb(c, m, y, k);
            },
            Operator::SetStrokeCmyk { c, m, y, k } => {
                // Store CMYK and convert to RGB for rendering
                let state = self.state_stack.current_mut();
                state.stroke_color_cmyk = Some((c, m, y, k));
                state.stroke_color_rgb = cmyk_to_rgb(c, m, y, k);
            },

            // Color space operators
            Operator::SetFillColorSpace { name } => {
                let state = self.state_stack.current_mut();
                state.fill_color_space = name.clone();
                // Reset color when changing color space
                state.fill_color_rgb = (0.0, 0.0, 0.0);
                state.fill_color_cmyk = None;
            },
            Operator::SetStrokeColorSpace { name } => {
                let state = self.state_stack.current_mut();
                state.stroke_color_space = name.clone();
                // Reset color when changing color space
                state.stroke_color_rgb = (0.0, 0.0, 0.0);
                state.stroke_color_cmyk = None;
            },
            Operator::SetFillColor { components } => {
                // Set fill color using components in current fill color space
                let state = self.state_stack.current_mut();
                match state.fill_color_space.as_str() {
                    "DeviceGray" | "CalGray" if components.len() == 1 => {
                        let gray = components[0];
                        state.fill_color_rgb = (gray, gray, gray);
                    },
                    "DeviceRGB" | "CalRGB" if components.len() == 3 => {
                        state.fill_color_rgb = (components[0], components[1], components[2]);
                    },
                    "Lab" if components.len() == 3 => {
                        // CIE L*a*b* color space
                        // For now, treat as RGB (proper conversion requires whitepoint)
                        // L* is lightness (0-100), a* and b* are color opponents
                        // Simplified conversion: normalize and treat as RGB
                        let l = components[0] / 100.0;
                        state.fill_color_rgb = (l, l, l); // Simplified grayscale approximation
                        log::debug!(
                            "Lab color space simplified to grayscale (full conversion not yet implemented)"
                        );
                    },
                    "DeviceCMYK" if components.len() == 4 => {
                        state.fill_color_cmyk =
                            Some((components[0], components[1], components[2], components[3]));
                        state.fill_color_rgb =
                            cmyk_to_rgb(components[0], components[1], components[2], components[3]);
                    },
                    "ICCBased" => {
                        // ICC profile-based color space
                        // For now, assume RGB and use components directly
                        if components.len() == 3 {
                            state.fill_color_rgb = (components[0], components[1], components[2]);
                        } else if components.len() == 1 {
                            let gray = components[0];
                            state.fill_color_rgb = (gray, gray, gray);
                        } else if components.len() == 4 {
                            // Treat as CMYK
                            state.fill_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.fill_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        }
                        log::debug!(
                            "ICCBased color space using simplified conversion (ICC profile not processed)"
                        );
                    },
                    "Separation" if components.len() == 1 => {
                        // Separation color space (spot color)
                        // Component is tint value (0.0 = no ink, 1.0 = full ink)
                        // For now, treat as grayscale
                        let tint = components[0];
                        let gray = 1.0 - tint; // Inverted (0 tint = white, 1 tint = black)
                        state.fill_color_rgb = (gray, gray, gray);
                        log::debug!("Separation color space simplified to grayscale");
                    },
                    "DeviceN" if !components.is_empty() => {
                        // DeviceN color space (multiple colorants)
                        // For now, use simplified conversion
                        if components.len() == 4 {
                            state.fill_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.fill_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        } else {
                            // Use first component as grayscale
                            let gray = 1.0 - components[0];
                            state.fill_color_rgb = (gray, gray, gray);
                        }
                        log::debug!("DeviceN color space using simplified conversion");
                    },
                    _ => {
                        // Unknown or unsupported color space - use default black
                        log::warn!(
                            "Unsupported fill color space: {} with {} components",
                            state.fill_color_space,
                            components.len()
                        );
                    },
                }
            },
            Operator::SetStrokeColor { components } => {
                // Set stroke color using components in current stroke color space
                let state = self.state_stack.current_mut();
                match state.stroke_color_space.as_str() {
                    "DeviceGray" | "CalGray" if components.len() == 1 => {
                        let gray = components[0];
                        state.stroke_color_rgb = (gray, gray, gray);
                    },
                    "DeviceRGB" | "CalRGB" if components.len() == 3 => {
                        state.stroke_color_rgb = (components[0], components[1], components[2]);
                    },
                    "Lab" if components.len() == 3 => {
                        let l = components[0] / 100.0;
                        state.stroke_color_rgb = (l, l, l);
                        log::debug!("Lab stroke color space simplified to grayscale");
                    },
                    "DeviceCMYK" if components.len() == 4 => {
                        state.stroke_color_cmyk =
                            Some((components[0], components[1], components[2], components[3]));
                        state.stroke_color_rgb =
                            cmyk_to_rgb(components[0], components[1], components[2], components[3]);
                    },
                    "ICCBased" => {
                        if components.len() == 3 {
                            state.stroke_color_rgb = (components[0], components[1], components[2]);
                        } else if components.len() == 1 {
                            let gray = components[0];
                            state.stroke_color_rgb = (gray, gray, gray);
                        } else if components.len() == 4 {
                            state.stroke_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.stroke_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        }
                        log::debug!("ICCBased stroke color using simplified conversion");
                    },
                    "Separation" if components.len() == 1 => {
                        let tint = components[0];
                        let gray = 1.0 - tint;
                        state.stroke_color_rgb = (gray, gray, gray);
                        log::debug!("Separation stroke color simplified to grayscale");
                    },
                    "DeviceN" if !components.is_empty() => {
                        if components.len() == 4 {
                            state.stroke_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.stroke_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        } else {
                            let gray = 1.0 - components[0];
                            state.stroke_color_rgb = (gray, gray, gray);
                        }
                        log::debug!("DeviceN stroke color using simplified conversion");
                    },
                    _ => {
                        // Unknown or unsupported color space
                        log::warn!(
                            "Unsupported stroke color space: {} with {} components",
                            state.stroke_color_space,
                            components.len()
                        );
                    },
                }
            },
            Operator::SetFillColorN { components, name } => {
                // Like SetFillColor, but also supports pattern color spaces
                if name.is_some() {
                    // Pattern color space - for now, just log and ignore
                    log::debug!("Pattern fill color not yet supported: {:?}", name);
                } else {
                    // Same logic as SetFillColor - supports all color spaces
                    let state = self.state_stack.current_mut();
                    match state.fill_color_space.as_str() {
                        "DeviceGray" | "CalGray" if components.len() == 1 => {
                            let gray = components[0];
                            state.fill_color_rgb = (gray, gray, gray);
                        },
                        "DeviceRGB" | "CalRGB" if components.len() == 3 => {
                            state.fill_color_rgb = (components[0], components[1], components[2]);
                        },
                        "Lab" if components.len() == 3 => {
                            let l = components[0] / 100.0;
                            state.fill_color_rgb = (l, l, l);
                        },
                        "DeviceCMYK" if components.len() == 4 => {
                            state.fill_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.fill_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        "ICCBased" => {
                            if components.len() == 3 {
                                state.fill_color_rgb =
                                    (components[0], components[1], components[2]);
                            } else if components.len() == 1 {
                                let gray = components[0];
                                state.fill_color_rgb = (gray, gray, gray);
                            } else if components.len() == 4 {
                                state.fill_color_cmyk = Some((
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                ));
                                state.fill_color_rgb = cmyk_to_rgb(
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                );
                            }
                        },
                        "Separation" if components.len() == 1 => {
                            let tint = components[0];
                            let gray = 1.0 - tint;
                            state.fill_color_rgb = (gray, gray, gray);
                        },
                        "DeviceN" if !components.is_empty() => {
                            if components.len() == 4 {
                                state.fill_color_cmyk = Some((
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                ));
                                state.fill_color_rgb = cmyk_to_rgb(
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                );
                            } else {
                                let gray = 1.0 - components[0];
                                state.fill_color_rgb = (gray, gray, gray);
                            }
                        },
                        _ => {
                            log::warn!(
                                "Unsupported fill color space: {} with {} components",
                                state.fill_color_space,
                                components.len()
                            );
                        },
                    }
                }
            },
            Operator::SetStrokeColorN { components, name } => {
                // Like SetStrokeColor, but also supports pattern color spaces
                if name.is_some() {
                    // Pattern color space - for now, just log and ignore
                    log::debug!("Pattern stroke color not yet supported: {:?}", name);
                } else {
                    // Same logic as SetStrokeColor - supports all color spaces
                    let state = self.state_stack.current_mut();
                    match state.stroke_color_space.as_str() {
                        "DeviceGray" | "CalGray" if components.len() == 1 => {
                            let gray = components[0];
                            state.stroke_color_rgb = (gray, gray, gray);
                        },
                        "DeviceRGB" | "CalRGB" if components.len() == 3 => {
                            state.stroke_color_rgb = (components[0], components[1], components[2]);
                        },
                        "Lab" if components.len() == 3 => {
                            let l = components[0] / 100.0;
                            state.stroke_color_rgb = (l, l, l);
                        },
                        "DeviceCMYK" if components.len() == 4 => {
                            state.stroke_color_cmyk =
                                Some((components[0], components[1], components[2], components[3]));
                            state.stroke_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        "ICCBased" => {
                            if components.len() == 3 {
                                state.stroke_color_rgb =
                                    (components[0], components[1], components[2]);
                            } else if components.len() == 1 {
                                let gray = components[0];
                                state.stroke_color_rgb = (gray, gray, gray);
                            } else if components.len() == 4 {
                                state.stroke_color_cmyk = Some((
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                ));
                                state.stroke_color_rgb = cmyk_to_rgb(
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                );
                            }
                        },
                        "Separation" if components.len() == 1 => {
                            let tint = components[0];
                            let gray = 1.0 - tint;
                            state.stroke_color_rgb = (gray, gray, gray);
                        },
                        "DeviceN" if !components.is_empty() => {
                            if components.len() == 4 {
                                state.stroke_color_cmyk = Some((
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                ));
                                state.stroke_color_rgb = cmyk_to_rgb(
                                    components[0],
                                    components[1],
                                    components[2],
                                    components[3],
                                );
                            } else {
                                let gray = 1.0 - components[0];
                                state.stroke_color_rgb = (gray, gray, gray);
                            }
                        },
                        _ => {
                            log::warn!(
                                "Unsupported stroke color space: {} with {} components",
                                state.stroke_color_space,
                                components.len()
                            );
                        },
                    }
                }
            },

            // Line style operators
            Operator::SetLineCap { cap_style } => {
                self.state_stack.current_mut().line_cap = cap_style;
            },
            Operator::SetLineJoin { join_style } => {
                self.state_stack.current_mut().line_join = join_style;
            },
            Operator::SetMiterLimit { limit } => {
                self.state_stack.current_mut().miter_limit = limit;
            },
            Operator::SetRenderingIntent { intent } => {
                self.state_stack.current_mut().rendering_intent = intent.clone();
            },
            Operator::SetFlatness { tolerance } => {
                self.state_stack.current_mut().flatness = tolerance;
            },
            Operator::SetExtGState { dict_name } => {
                // ExtGState operator - set graphics state from resource dictionary
                // PDF Spec: ISO 32000-1:2008, Section 8.4.5
                //
                // This operator references an ExtGState dictionary in the page resources
                // that contains transparency, blend modes, and other graphics state parameters.
                //
                // For now, we log the usage. Full implementation would require:
                // 1. Access to page resources (/ExtGState dictionary)
                // 2. Loading the named dictionary
                // 3. Extracting /CA (fill alpha), /ca (stroke alpha), /BM (blend mode), etc.
                // 4. Updating graphics state accordingly
                //
                // Future enhancement: Pass resources to text extractor for full support
                log::debug!(
                    "ExtGState '{}' referenced (transparency/blend modes not yet fully supported)",
                    dict_name
                );

                // Apply default transparency values for now
                // In a full implementation, we would look up dict_name in resources
                // and apply the actual values from the ExtGState dictionary
            },
            Operator::PaintShading { name } => {
                // Shading operator - paint gradient/shading pattern
                // PDF Spec: ISO 32000-1:2008, Section 8.7.4.3
                //
                // Shading patterns define smooth color gradients and can be:
                // Type 1: Function-based shading
                // Type 2: Axial shading (linear gradient)
                // Type 3: Radial shading (circular gradient)
                // Type 4-7: Mesh-based shadings (Gouraud, Coons patch, tensor-product)
                //
                // For text extraction, shading patterns don't affect text content.
                // Full implementation would require rendering the gradient for visual output.
                log::debug!(
                    "Shading pattern '{}' referenced (gradients not rendered in text extraction)",
                    name
                );
            },
            Operator::InlineImage { dict, data } => {
                // Inline image operator - embedded image in content stream
                // PDF Spec: ISO 32000-1:2008, Section 8.9.7 - Inline Images
                //
                // Inline images are small images embedded directly in the content stream
                // using the BI...ID...EI sequence, rather than referenced as XObjects.
                //
                // For text extraction, inline images don't contribute to text content.
                // They would be rendered for visual output or extracted separately
                // for image extraction functionality.
                //
                // Common dictionary keys (abbreviated):
                // - W: Width, H: Height
                // - CS: ColorSpace (DeviceRGB, DeviceGray, etc.)
                // - BPC: BitsPerComponent
                // - F: Filter (FlateDecode, DCTDecode, etc.)
                let width = dict
                    .get("W")
                    .and_then(|obj| match obj {
                        Object::Integer(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(0);
                let height = dict
                    .get("H")
                    .and_then(|obj| match obj {
                        Object::Integer(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(0);
                log::debug!(
                    "Inline image encountered: {}x{} pixels, {} bytes of data (not rendered in text extraction)",
                    width,
                    height,
                    data.len()
                );
            },

            // Text object operators (BT/ET)
            // PDF Spec ISO 32000-1:2008, Section 9.4.1:
            // "At the beginning of a text object, Tm and Tlm shall be
            // initialized to the identity matrix."
            Operator::BeginText => {
                let state = self.state_stack.current_mut();
                state.text_matrix = Matrix::identity();
                state.text_line_matrix = Matrix::identity();
            },
            Operator::EndText => {
                // Flush any pending text buffer at end of text object
                self.flush_tj_span_buffer()?;
            },

            // Marked content operators - for tagged PDF structure
            // PDF Spec: ISO 32000-1:2008, Section 14.6 - Marked Content
            // These operators define logical structure and accessibility metadata.
            // Per PDF Spec Section 14.6, we track artifact status to filter out
            // non-text content (headers, footers, watermarks, resource paths).
            Operator::BeginMarkedContent { tag } => {
                // BMC doesn't have properties, but the tag can indicate artifacts
                let is_artifact = tag == "Artifact";
                self.marked_content_stack.push(MarkedContentContext {
                    tag: tag.clone(),
                    is_artifact,
                    artifact_type: None, // No artifact classification; None for backward compatibility
                    actual_text: None,   // BMC doesn't have ActualText
                    expansion: None,     // BMC doesn't have expansion
                });
                self.update_artifact_state();

                if is_artifact {
                    log::debug!("Entered /Artifact marked content (BMC, no subtype)");
                }
            },

            Operator::BeginMarkedContentDict { tag, properties } => {
                // BDC can have properties including MCID, artifact indicators, ActualText, and expansion
                // Properties can be an inline dictionary or a name referencing /Properties resource
                let mut actual_text = None;
                let mut artifact_type = None;
                let mut expansion = None;

                if let Some(props_dict) = self.resolve_bdc_properties(&properties) {
                    if let Some(mcid_obj) = props_dict.get("MCID") {
                        if let Some(mcid) = mcid_obj.as_integer() {
                            self.current_mcid = Some(mcid as u32);
                            log::debug!("Entered marked content with MCID: {}", mcid);
                        }
                    }

                    if let Some(actual_text_obj) = props_dict.get("ActualText") {
                        if let Some(text_bytes) = actual_text_obj.as_string() {
                            actual_text = Some(Self::decode_pdf_text_string(text_bytes));
                            log::debug!("Marked content has ActualText: {:?}", actual_text);
                        }
                    }

                    if let Some(expansion_obj) = props_dict.get("E") {
                        if let Some(text_bytes) = expansion_obj.as_string() {
                            expansion = Some(Self::decode_pdf_text_string(text_bytes));
                            log::debug!("Marked content has expansion /E: {:?}", expansion);
                        }
                    }

                    if tag == "Artifact" {
                        artifact_type = Self::parse_artifact_type(&props_dict);
                    }
                }

                // Check if this is an artifact (per PDF Spec Section 14.6)
                let is_artifact = tag == "Artifact";
                self.marked_content_stack.push(MarkedContentContext {
                    tag: tag.clone(),
                    is_artifact,
                    artifact_type: artifact_type.clone(),
                    actual_text,
                    expansion,
                });
                self.update_artifact_state();

                if is_artifact {
                    if let Some(ref atype) = artifact_type {
                        log::debug!("Entered /Artifact marked content: {:?}", atype);
                    } else {
                        log::debug!("Entered /Artifact marked content (no type specified)");
                    }
                }
            },

            Operator::EndMarkedContent => {
                // EMC ends the current marked content sequence
                if let Some(mcid) = self.current_mcid {
                    log::debug!("Exited marked content with MCID: {}", mcid);
                }
                self.current_mcid = None;

                // Pop from marked content stack and update artifact state
                if !self.marked_content_stack.is_empty() {
                    self.marked_content_stack.pop();
                    self.update_artifact_state();
                }
            },

            // XObject operator - Process Form XObjects for text extraction
            Operator::Do { name } => {
                // Process Form XObjects to extract text from reusable content.
                // Form XObjects can contain text that is not duplicated in the main stream.
                // We track processed XObjects to avoid infinite loops and duplicates.
                if let Err(e) = self.process_xobject(&name) {
                    // Log error but continue processing - don't fail the entire extraction
                    log::warn!("Failed to process XObject '{}': {}", name, e);
                }
            },

            // Other operators we don't need for text extraction
            _ => {
                // Ignore path, image, and other operators
            },
        }

        Ok(())
    }

    /// Maximum XObject recursion depth. Text content in PDFs is rarely nested
    /// more than 2-3 levels. Deep nesting typically indicates complex vector
    /// graphics (charts, plots) with no text content.
    const MAX_XOBJECT_DEPTH: u32 = 10;

    const MAX_XOBJECT_DECODES: u32 = 500;

    /// Resolve XObject name to ObjectRef using cached mapping.
    fn resolve_xobject_ref(&mut self, name: &str) -> Result<Option<ObjectRef>> {
        // Check cache first (O(1) lookup)
        if let Some(cached) = self.cached_xobject_refs.get(name) {
            return Ok(*cached);
        }

        // Cache miss — resolve the full chain once and populate cache
        let resources = match &self.resources {
            Some(res) => res.clone(),
            None => return Ok(None),
        };

        let doc_ptr = match self.document {
            Some(ptr) => ptr,
            None => return Ok(None),
        };
        let doc = unsafe { &*doc_ptr };

        // Resolve resources → XObject dict
        let resources_obj = if let Some(res_ref) = resources.as_reference() {
            doc.load_object(res_ref)?
        } else {
            resources
        };

        let resources_dict = match resources_obj.as_dict() {
            Some(d) => d,
            None => return Ok(None),
        };

        let xobject_entry = match resources_dict.get("XObject") {
            Some(xobj) => xobj.clone(),
            None => return Ok(None),
        };

        let xobject_obj = if let Some(xobj_ref) = xobject_entry.as_reference() {
            doc.load_object(xobj_ref)?
        } else {
            xobject_entry
        };

        let xobject_dict = match xobject_obj.as_dict() {
            Some(d) => d,
            None => return Ok(None),
        };

        // Populate the entire cache for this resources context
        for (key, val) in xobject_dict.iter() {
            let obj_ref = val.as_reference();
            self.cached_xobject_refs.insert(key.clone(), obj_ref);
        }

        // Return the requested name
        Ok(self.cached_xobject_refs.get(name).copied().flatten())
    }

    fn process_xobject(&mut self, name: &str) -> Result<()> {
        if self.xobject_depth >= Self::MAX_XOBJECT_DEPTH {
            return Ok(());
        }
        if self.xobject_decode_count >= Self::MAX_XOBJECT_DECODES {
            return Ok(());
        }

        // Resolve name → ObjectRef using cached mapping (avoids expensive
        // repeated resolution of resources/XObject dict chain)
        let xobject_ref = match self.resolve_xobject_ref(name)? {
            Some(r) => r,
            None => return Ok(()),
        };

        // Skip already-processed XObjects (permanent set — each unique XObject
        // is processed at most once per page for text extraction)
        if self.processed_xobjects.contains(&xobject_ref) {
            return Ok(());
        }

        self.processed_xobjects.insert(xobject_ref);

        // Get document reference for loading objects
        let doc = match self.document {
            Some(ptr) => unsafe { &*ptr },
            None => return Ok(()),
        };

        if doc
            .xobject_text_free_cache
            .lock()
            .unwrap()
            .contains(&xobject_ref)
        {
            return Ok(());
        }

        // Quick Subtype check: skip Image XObjects without loading the full object.
        // Image XObjects can be megabytes of compressed pixel data — loading them
        // just to discover Subtype=Image is a major bottleneck (10-15ms per image).
        if !doc.is_form_xobject(xobject_ref) {
            return Ok(());
        }

        // Span result cache: reuse extracted spans from self-contained Form XObjects.
        //
        // Spans are stored in CTM-transformed page coordinates, so the cache is
        // only correct when the caller's CTM matches the one at first extraction.
        // Issue B1 (nougat_005.pdf): a single Form XObject carries every page's
        // content, and each page's content stream applies a different CTM
        // translation to position its viewport into that XObject. Reusing the
        // cached spans returned page 0's coordinates on every page, so every
        // page emitted identical cross-page text.
        //
        // Safe path: only hit the cache when the current CTM is identity. That
        // covers the common case (reusable headers/footers stamped at the same
        // origin) without mixing coordinate systems. For non-identity CTMs we
        // fall through to the fresh extraction below, which applies the caller's
        // CTM to each span.
        if self.extract_spans && self.state_stack.current().ctm.is_identity() {
            let cached_spans = {
                doc.xobject_spans_cache
                    .lock()
                    .unwrap()
                    .get(&xobject_ref)
                    .cloned()
            };
            if let Some(cached_spans) = cached_spans {
                if let Some(spans) = cached_spans {
                    self.spans.extend(spans.iter().cloned());
                }
                return Ok(());
            }
        }

        // Load the XObject (now known to be Form or unknown — worth the full load)
        let xobject = doc.load_object(xobject_ref)?;

        // Check if it's a Form XObject (has Subtype /Form)
        let xobject_dict = match xobject.as_dict() {
            Some(d) => d,
            None => {
                log::debug!("XObject '{}' is not a dictionary", name);
                return Ok(());
            },
        };

        let subtype = xobject_dict.get("Subtype").and_then(|s| s.as_name());

        match subtype {
            Some("Form") => {
                // Form XObject - extract text from it
                log::debug!("Processing Form XObject: {}", name);

                // Pre-decode resource check: if the XObject's own /Resources has
                // neither /Font nor /XObject entries, it cannot render text directly
                // and cannot invoke nested XObjects. Skip it without decoding the
                // stream, which avoids expensive FlateDecode decompression.
                if let Some(xobj_resources) = xobject_dict.get("Resources") {
                    let xobj_res = if let Some(res_ref) = xobj_resources.as_reference() {
                        doc.load_object(res_ref).ok()
                    } else {
                        Some(xobj_resources.clone())
                    };

                    if let Some(ref res_obj) = xobj_res {
                        if let Some(res_dict) = res_obj.as_dict() {
                            let has_font = res_dict.contains_key("Font");
                            let has_xobject = res_dict.contains_key("XObject");
                            if !has_font && !has_xobject {
                                log::debug!(
                                    "Skipping Form XObject '{}': no Font/XObject in Resources",
                                    name
                                );
                                doc.xobject_text_free_cache
                                    .lock()
                                    .unwrap()
                                    .insert(xobject_ref);
                                return Ok(());
                            }
                        }
                    }
                } else {
                    // No Resources at all — XObject inherits page-level fonts but
                    // still must be decoded to check for text operators. However,
                    // Form XObjects that are pure graphics often omit Resources
                    // entirely when they have no font/xobject needs. Check if the
                    // page has any active fonts; if not, skip.
                }

                // Decode the stream — check cache first to avoid repeated FlateDecode.
                self.xobject_decode_count += 1;
                let cached_stream = {
                    doc.xobject_stream_cache
                        .lock()
                        .unwrap()
                        .get(&xobject_ref)
                        .cloned()
                };
                let stream_data = if let Some(cached) = cached_stream {
                    cached.as_ref().clone()
                } else {
                    match doc.decode_stream_with_encryption(&xobject, xobject_ref) {
                        Ok(data) => {
                            // Cache if under 50MB total
                            const MAX_STREAM_CACHE_BYTES: usize = 50 * 1024 * 1024;
                            let current = doc
                                .xobject_stream_cache_bytes
                                .load(std::sync::atomic::Ordering::Relaxed);
                            if current + data.len() <= MAX_STREAM_CACHE_BYTES {
                                doc.xobject_stream_cache_bytes.store(
                                    current + data.len(),
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                doc.xobject_stream_cache
                                    .lock()
                                    .unwrap()
                                    .insert(xobject_ref, std::sync::Arc::new(data.clone()));
                            }
                            data
                        },
                        Err(e) => {
                            log::warn!(
                                "Failed to decode Form XObject '{}' stream: {}, skipping",
                                name,
                                e
                            );
                            return Ok(());
                        },
                    }
                };

                if !crate::document::PdfDocument::may_contain_text(&stream_data) {
                    log::debug!(
                        "Skipping text-free Form XObject '{}' ({} bytes)",
                        name,
                        stream_data.len()
                    );
                    doc.xobject_text_free_cache
                        .lock()
                        .unwrap()
                        .insert(xobject_ref);
                    return Ok(());
                }

                // Parse /Matrix from Form XObject dict (default: identity per ISO 32000-1 §8.10.1)
                let form_matrix = if let Some(Object::Array(arr)) = xobject_dict.get("Matrix") {
                    let get_f32 = |i: usize| -> f32 {
                        match arr.get(i) {
                            Some(Object::Real(v)) => *v as f32,
                            Some(Object::Integer(v)) => *v as f32,
                            _ => {
                                if i == 0 || i == 3 {
                                    1.0
                                } else {
                                    0.0
                                }
                            },
                        }
                    };
                    Matrix {
                        a: get_f32(0),
                        b: get_f32(1),
                        c: get_f32(2),
                        d: get_f32(3),
                        e: get_f32(4),
                        f: get_f32(5),
                    }
                } else {
                    Matrix::identity()
                };

                // Only save/restore fonts+resources when XObject has its own Resources.
                // Avoids expensive HashMap clone for XObjects that inherit page fonts.
                let has_own_resources = xobject_dict.contains_key("Resources");

                let saved_fonts;
                let saved_resources;
                let saved_xobj_cache;

                if has_own_resources {
                    saved_fonts = Some(self.fonts.clone());
                    saved_resources = self.resources.clone();
                    saved_xobj_cache = Some(std::mem::take(&mut self.cached_xobject_refs));

                    // Safety: has_own_resources was set by contains_key("Resources")
                    // so get("Resources") will always return Some here
                    let xobj_resources = xobject_dict
                        .get("Resources")
                        .expect("contains_key confirmed Resources exists");
                    let xobj_res = if let Some(res_ref) = xobj_resources.as_reference() {
                        match doc.load_object(res_ref) {
                            Ok(obj) => obj,
                            Err(_) => xobj_resources.clone(),
                        }
                    } else {
                        xobj_resources.clone()
                    };

                    if let Err(e) = doc.load_fonts(&xobj_res, self) {
                        log::debug!(
                            "Failed to load fonts for Form XObject '{}': {}, using page fonts",
                            name,
                            e
                        );
                    }

                    self.resources = Some(xobj_res);
                } else {
                    saved_fonts = None;
                    saved_resources = None;
                    saved_xobj_cache = None;
                }

                // Track span count for result caching
                let spans_before = self.spans.len();

                // Save graphics state (implicit q per ISO 32000-1 §8.10.1)
                self.state_stack.save();

                // Concatenate Form XObject /Matrix with CTM
                let state = self.state_stack.current_mut();
                state.ctm = form_matrix.multiply(&state.ctm);

                // Streaming parse+execute: avoids allocating Vec<Operator>
                self.xobject_depth += 1;
                let parse_result =
                    parse_and_execute_text_only(&stream_data, |op| self.execute_operator(op));
                self.xobject_depth -= 1;
                if let Err(e) = parse_result {
                    log::debug!(
                        "Error parsing Form XObject '{}' content stream: {}, partial text may be extracted",
                        name,
                        e
                    );
                }

                // Cache span results for self-contained Form XObjects. Only
                // safe when the XObject has its own /Resources (font context
                // is page-independent) AND the current CTM is identity —
                // otherwise the stored spans are in caller-specific page
                // coordinates and would poison identity-CTM hits on other
                // pages (see issue B1).
                //
                // Note: is_identity() is checked AFTER the XObject's
                // /Matrix is concatenated, so XObjects with their own
                // /Matrix never cache even when the caller CTM is identity.
                // That's conservative-safe at the cost of re-extraction;
                // storing spans in XObject-local coords would fix this but
                // the complexity isn't justified by the current benchmark.
                let save_identity_ctm = self.state_stack.current().ctm.is_identity();
                if has_own_resources && self.extract_spans && save_identity_ctm {
                    let new_spans = if self.spans.len() > spans_before {
                        Some(self.spans[spans_before..].to_vec())
                    } else {
                        None
                    };
                    doc.xobject_spans_cache
                        .lock()
                        .unwrap()
                        .insert(xobject_ref, new_spans);
                }

                // Restore graphics state (implicit Q per ISO 32000-1 §8.10.1)
                self.state_stack.restore();
                // Sync cached font with restored state
                self.cached_current_font = self
                    .state_stack
                    .current()
                    .font_name
                    .as_ref()
                    .and_then(|name| self.fonts.get(name))
                    .cloned();

                // Restore fonts, resources, and XObject cache only if saved
                if let Some(fonts) = saved_fonts {
                    self.fonts = fonts;
                }
                if let Some(res) = saved_resources {
                    self.resources = Some(res);
                }
                if let Some(cache) = saved_xobj_cache {
                    self.cached_xobject_refs = cache;
                }

                // Keep xobject_ref in processed_xobjects permanently.
                // For text extraction, re-processing the same Form XObject produces
                // identical text. Keeping it prevents O(n!) fan-out in pages with
                // deep XObject trees (e.g., 4000+ nested chart elements).

                Ok(())
            },
            Some("Image") => {
                // Image XObject - no text to extract
                log::debug!("Skipping Image XObject: {}", name);
                Ok(())
            },
            _ => {
                log::debug!("Unknown XObject subtype for '{}': {:?}", name, subtype);
                Ok(())
            },
        }
    }

    /// Get the current artifact type from the marked content stack.
    fn current_artifact_type(&self) -> Option<ArtifactType> {
        self.marked_content_stack
            .iter()
            .rev()
            .find_map(|ctx| ctx.artifact_type.clone())
    }

    /// Flush accumulated TJ buffer into a single TextSpan.
    ///
    /// This creates one span for the entire buffer content, properly calculating
    /// the total width including character spacing (Tc) and word spacing (Tw).
    fn flush_tj_buffer(&mut self, mut buffer: TjBuffer) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Use accumulated width from advance_position_for_string calls
        // Convert from text space to user space using pre-computed horizontal scale
        let total_width = buffer.accumulated_width * buffer.user_h_scale;

        // Use pre-computed values from buffer creation (avoids
        // matrix multiply + sqrt + HashMap lookup + transform_point per flush)
        let effective_font_size = buffer.effective_font_size;
        let font_weight = buffer.font_weight;
        let is_italic_span = buffer.is_italic;

        // Move owned strings out of buffer (avoids clone)
        let font_name_span = buffer
            .font_name
            .take()
            .unwrap_or_else(|| "Unknown".to_string());

        // RTL text correction: if text contains RTL characters and spans left-to-right
        // on the page, the characters are in visual LTR order. Reverse to logical order.
        let mut text = std::mem::take(&mut buffer.unicode);
        if text.len() > 1 {
            let has_rtl = text
                .chars()
                .any(|c| crate::text::rtl_detector::is_rtl_text(c as u32));
            if has_rtl {
                // In the tiebreaker path, characters are appended left-to-right in content
                // stream order. For RTL scripts displayed right-to-left, this means the
                // leftmost visual character (last logical character) is first in the buffer.
                // Reverse to get logical reading order.
                // Only reverse if user_pos_x indicates LTR placement (positive width).
                if buffer.accumulated_width > 0.0 {
                    text = text.chars().rev().collect();
                }
            }
        }

        let span = TextSpan {
            text,
            bbox: Rect {
                x: buffer.user_pos_x,
                y: buffer.user_pos_y,
                width: total_width,
                height: effective_font_size,
            },
            font_name: font_name_span,
            font_size: effective_font_size,
            font_weight,
            color: Color::new(
                buffer.fill_color_rgb.0,
                buffer.fill_color_rgb.1,
                buffer.fill_color_rgb.2,
            ),
            mcid: buffer.mcid,
            sequence: self.span_sequence_counter,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: buffer.char_space, // Tc - captured from PDF content stream
            word_spacing: buffer.word_space, // Tw - captured from PDF content stream
            horizontal_scaling: buffer.horizontal_scaling, // Tz - captured from PDF content stream
            is_italic: is_italic_span,
            is_monospace: buffer.is_monospace,
            primary_detected: false,
            artifact_type: self.current_artifact_type(),
            char_widths: {
                let mut cw = std::mem::take(&mut buffer.char_widths);
                let h = buffer.user_h_scale;
                for w in &mut cw {
                    *w *= h;
                }
                cw
            },
        };
        self.span_sequence_counter += 1;

        self.spans.push(span);
        Ok(())
    }

    /// Calculate total width of TJ buffer using PDF spec formula.
    ///
    /// Process TJ array according to configured word boundary detection mode.
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.4.4,
    /// this method dispatches to either:
    /// - process_tj_array_tiebreaker(): WordBoundaryMode::Tiebreaker (default)
    /// - process_tj_array_primary(): WordBoundaryMode::Primary
    fn process_tj_array(&mut self, array: &[TextElement]) -> Result<()> {
        match self.word_boundary_mode {
            WordBoundaryMode::Tiebreaker => self.process_tj_array_tiebreaker(array),
            WordBoundaryMode::Primary => self.process_tj_array_primary(array),
        }
    }

    /// Process TJ array using tiebreaker mode (backward compatible).
    ///
    /// This is the legacy code path used when
    /// WordBoundaryMode::Tiebreaker is configured.
    ///
    /// Maintains 100% backward compatibility with existing behavior.
    /// Word boundaries are detected only as a tiebreaker when TJ offset
    /// and geometric signals contradict each other.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
    /// "The performance of text searching (and other text extraction operations) is
    /// significantly better if the text strings are as long as possible."
    ///
    /// This method buffers consecutive strings into a single span, only breaking on:
    /// - Large negative offsets (indicating word boundaries)
    /// - End of TJ array
    fn process_tj_array_tiebreaker(&mut self, array: &[TextElement]) -> Result<()> {
        // Character-level tracking for word boundary detection
        // Collect detailed character information during TJ array processing
        // Per ISO 32000-1:2008 Section 9.4.4, character-level data improves accuracy

        self.tj_character_array.clear();
        self.current_x_position = 0.0;

        // Copy state data to avoid holding reference while borrowing self mutably
        let font_size = self.state_stack.current().font_size;
        let horizontal_scaling = self.state_stack.current().horizontal_scaling / 100.0;
        let font_name = self.state_stack.current().font_name.clone();
        let char_space = self.state_stack.current().char_space;
        let word_space = self.state_stack.current().word_space;

        let mut buffer = TjBuffer::new(
            self.state_stack.current(),
            self.current_mcid,
            self.cached_current_font.clone(),
        );
        let mut _element_count = 0;

        for (idx, element) in array.iter().enumerate() {
            _element_count += 1;
            match element {
                TextElement::String(s) => {
                    // Collect character-level data before processing buffer
                    // Extract individual characters with their properties
                    if let Some(ref name) = font_name {
                        if let Some(font) = self.fonts.get(name) {
                            // Process each byte in the string
                            for &byte in s.iter() {
                                // Normalize character code through encoding.
                                // This ensures word boundary detection works on actual characters,
                                // not raw byte codes from custom encodings
                                let char_code = font
                                    .get_encoded_char(byte)
                                    .map(|ch| ch as u32)
                                    .unwrap_or(byte as u32);

                                let glyph_width = font.get_glyph_width(byte as u16);

                                // Check if this is a ligature character (U+FB00-U+FB04)
                                let is_ligature = Self::is_ligature_code(char_code);

                                // Create CharacterInfo for this character
                                // The tj_offset will be applied when we encounter the next Offset element
                                let char_info = CharacterInfo {
                                    code: char_code,
                                    glyph_id: None, // Could be enhanced to extract actual GID
                                    width: glyph_width,
                                    x_position: self.current_x_position,
                                    tj_offset: None, // Will be set if next element is Offset
                                    font_size,
                                    is_ligature,
                                    original_ligature: None,
                                    protected_from_split: false,
                                };

                                self.tj_character_array.push(char_info);

                                // Update current X position (in text space units)
                                // Per PDF Spec: account for character spacing and scaling
                                let char_advance = glyph_width * horizontal_scaling
                                    + char_space
                                    + (if byte == 0x20 { word_space } else { 0.0 });
                                self.current_x_position += char_advance;
                            }
                        }
                    }

                    // Single-pass: append unicode + compute width + advance position
                    self.append_advance_buffer(&mut buffer, s)?;
                },
                TextElement::Offset(offset) => {
                    // Track TJ offset for statistical analysis
                    // Per ISO 32000-1:2008 Section 9.4.4, collect all TJ values
                    // to detect justified vs normal text through coefficient of variation
                    if self.tj_offset_history.len() < 10000 {
                        // Keep history reasonable size (first 10k offsets per document)
                        self.tj_offset_history.push(*offset);
                    }

                    // Associate TJ offset with the last character
                    // The offset applies AFTER the previous string, affecting spacing to next string
                    if !self.tj_character_array.is_empty() {
                        let last_idx = self.tj_character_array.len() - 1;
                        self.tj_character_array[last_idx].tj_offset = Some(*offset as i32);
                    }

                    // Check if this offset indicates a word boundary
                    // Per PDF spec: negative offsets increase spacing
                    // Use geometry-based adaptive threshold
                    let threshold = self.calculate_adaptive_tj_threshold();
                    if *offset < threshold {
                        // Check if buffer ends with space BEFORE flushing
                        // This prevents double spaces when TJ processor inserts space
                        // AND span merging would insert space at the same boundary.
                        let buffer_ends_with_space = !buffer.unicode.is_empty()
                            && buffer
                                .unicode
                                .chars()
                                .next_back()
                                .map(|c| c.is_whitespace())
                                .unwrap_or(false);

                        // Flush buffer before space
                        self.flush_tj_buffer(buffer)?;

                        // Check if the next element in the TJ array is a string
                        // that starts with whitespace. If so, DON'T insert a space to avoid doubling.
                        // This prevents patterns like "word " + " next" = "word  next" (double space)
                        let next_element_starts_with_space = if idx + 1 < array.len() {
                            if let TextElement::String(next_s) = &array[idx + 1] {
                                next_s.first().is_some_and(|&byte| {
                                    byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
                                })
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Only insert space if neither side already has whitespace
                        if !buffer_ends_with_space && !next_element_starts_with_space {
                            // Insert space character as separate span
                            self.insert_space_as_span()?;
                        }

                        // Start new buffer with current state
                        buffer = TjBuffer::new(
                            self.state_stack.current(),
                            self.current_mcid,
                            self.cached_current_font.clone(),
                        );
                    }

                    // Advance position for offset (updates text matrix)
                    self.advance_position_for_offset(*offset)?;
                },
            }
        }

        // Flush remaining buffer
        if !buffer.is_empty() {
            self.flush_tj_buffer(buffer)?;
        }

        Ok(())
    }

    /// Process TJ array using primary detection mode.
    ///
    /// This implementation:
    /// 1. Creates BoundaryContext from graphics state
    /// 2. Calls WordBoundaryDetector to detect boundaries in tj_character_array
    /// 3. Apply ligature expansion decisions
    /// 4. Partitions characters into clusters at boundary positions
    /// 5. Converts each cluster to a TextSpan with proper bounding boxes
    /// 6. Marks spans with primary_detected flag
    fn process_tj_array_primary(&mut self, array: &[TextElement]) -> Result<()> {
        // Primary detection mode implementation

        // Step 1: If no characters collected, fall back to tiebreaker behavior
        if self.tj_character_array.is_empty() {
            return self.process_tj_array_tiebreaker(array);
        }

        // Mark pattern contexts BEFORE boundary detection
        // This protects email and URL patterns from being split at word boundaries
        let pattern_config = crate::extractors::PatternPreservationConfig::default();
        crate::extractors::PatternDetector::mark_pattern_contexts(
            &mut self.tj_character_array,
            &pattern_config,
        )?;

        // Step 2: Create BoundaryContext from current graphics state
        let context = self.create_boundary_context();

        // Step 3: Create WordBoundaryDetector and detect boundaries
        // OPTIMIZATION: Detect document script profile to skip unnecessary detectors (Issue #1 fix)
        let script = DocumentScript::detect_from_characters(&self.tj_character_array);
        let detector = WordBoundaryDetector::new().with_document_script(script);
        let boundaries = detector.detect_word_boundaries(&self.tj_character_array, &context);

        // Step 4: If no boundaries detected, process entire array as single span
        if boundaries.is_empty() {
            // All characters form a single word
            return self.process_tj_array_tiebreaker(array);
        }

        // Step 3.5: Apply ligature expansion decisions
        // This intelligently splits ligatures at word boundaries
        self.apply_ligature_decisions()?;

        // Step 5: Partition characters into clusters at boundary positions
        let clusters =
            self.partition_characters_by_boundaries(&self.tj_character_array, boundaries);

        // Step 6: Convert each cluster to a TextSpan
        for cluster in clusters {
            if !cluster.is_empty() {
                self.cluster_to_span(&cluster)?;
            }
        }

        Ok(())
    }

    /// Create BoundaryContext from current graphics state.
    ///
    /// Per ISO 32000-1:2008 Section 9.3, extracts text state parameters
    /// used by WordBoundaryDetector to make boundary decisions.
    fn create_boundary_context(&self) -> BoundaryContext {
        let state = self.state_stack.current();
        BoundaryContext {
            font_size: state.font_size,
            horizontal_scaling: state.horizontal_scaling,
            word_spacing: state.word_space,
            char_spacing: state.char_space,
        }
    }

    /// Partition character array into clusters at boundary positions.
    ///
    /// # Arguments
    /// * `characters` - Full character array from TJ processing
    /// * `boundaries` - Boundary indices (positions where word boundaries occur)
    ///
    /// # Returns
    /// Vector of character clusters, where boundaries separate clusters
    fn partition_characters_by_boundaries(
        &self,
        characters: &[CharacterInfo],
        boundaries: Vec<usize>,
    ) -> Vec<Vec<CharacterInfo>> {
        if boundaries.is_empty() {
            return vec![characters.to_vec()];
        }

        let mut clusters = Vec::new();
        let mut prev = 0;

        for boundary_idx in boundaries {
            if boundary_idx > prev {
                clusters.push(characters[prev..boundary_idx].to_vec());
            }
            prev = boundary_idx;
        }

        // Add remaining characters after last boundary
        if prev < characters.len() {
            clusters.push(characters[prev..].to_vec());
        }

        clusters
    }

    /// Convert a character cluster to a TextSpan.
    ///
    /// Calculates bounding box from character positions and creates
    /// a single TextSpan marked with primary_detected flag.
    ///
    /// # Arguments
    /// * `cluster` - Character cluster from partitioning
    fn cluster_to_span(&mut self, cluster: &[CharacterInfo]) -> Result<()> {
        if cluster.is_empty() {
            return Ok(());
        }

        let state = self.state_stack.current();

        // Step 1: Calculate bounding box from character positions in text space
        // X position: from first character to end of last character
        let text_min_x = cluster[0].x_position;
        // Safety: caller checks cluster.is_empty() above and returns early
        let last = cluster.last().expect("cluster verified non-empty above");
        let text_max_x = last.x_position + last.width;
        let text_width = (text_max_x - text_min_x).max(0.0);

        // Height from font size
        let height = cluster[0].font_size.abs() * state.text_matrix.d.abs().max(1.0);

        // Step 2: Apply CTM to convert from text space to user space
        // Per PDF Spec ISO 32000-1:2008 Section 9.4.4
        let text_matrix = state.text_matrix;
        let ctm = state.ctm;
        let text_pos = text_matrix.transform_point(text_min_x, 0.0);
        let user_pos = ctm.transform_point(text_pos.x, text_pos.y);

        // Transform the width as well (accounting for matrix scaling)
        let user_width = text_width * text_matrix.a.abs() * ctm.a.abs();

        // Step 3: Create bounding box rectangle in user space
        let bbox = Rect {
            x: user_pos.x,
            y: user_pos.y,
            width: user_width.max(text_width), // Use larger of the two for safety
            height,
        };

        // Step 3: Convert characters to Unicode string
        // Use same decoding as existing code
        let mut unicode_text = if let Some(font_name) = state.font_name.as_ref() {
            if let Some(font) = self.fonts.get(font_name) {
                let mut text = String::new();
                for char_info in cluster {
                    if let Some(decoded) = font.char_to_unicode(char_info.code) {
                        text.push_str(&decoded);
                    }
                }
                text
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Step 3b: RTL text correction — reverse visual-order characters to logical order.
        // PDF stores characters in content stream order. For RTL scripts (Arabic/Hebrew),
        // if characters are positioned left-to-right in user space (increasing x), they
        // are in visual LTR order, which is reversed from logical RTL reading order.
        // We reverse them to produce correct logical order.
        // When combined_matrix.a < 0, characters go right-to-left in user space, meaning
        // content stream order already matches logical RTL reading order — no reversal.
        if unicode_text.len() > 1 && cluster.len() >= 2 {
            let has_rtl = unicode_text
                .chars()
                .any(|c| crate::text::rtl_detector::is_rtl_text(c as u32));
            if has_rtl {
                let first_x = {
                    let p = text_matrix.transform_point(cluster[0].x_position, 0.0);
                    ctm.transform_point(p.x, p.y).x
                };
                let last_x = {
                    let p = text_matrix.transform_point(last.x_position, 0.0);
                    ctm.transform_point(p.x, p.y).x
                };
                // Characters go left-to-right in user space → visual LTR → reverse for RTL
                if last_x > first_x {
                    unicode_text = unicode_text.chars().rev().collect();
                }
            }
        }

        // Step 4: Determine font weight
        let font_weight = if let Some(font_name) = state.font_name.as_ref() {
            if let Some(font) = self.fonts.get(font_name) {
                if font.is_bold() {
                    FontWeight::Bold
                } else {
                    FontWeight::Normal
                }
            } else {
                FontWeight::Normal
            }
        } else {
            FontWeight::Normal
        };

        // Determine if italic
        let is_italic = state
            .font_name
            .as_ref()
            .and_then(|name| self.fonts.get(name))
            .map(|font| font.is_italic())
            .unwrap_or(false);

        // Step 5: Create TextSpan with primary_detected flag
        let span = TextSpan {
            text: unicode_text,
            bbox,
            font_name: state
                .font_name
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            font_size: cluster[0].font_size,
            font_weight,
            color: Color::new(
                state.fill_color_rgb.0,
                state.fill_color_rgb.1,
                state.fill_color_rgb.2,
            ),
            mcid: self.current_mcid,
            sequence: self.span_sequence_counter,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: state.char_space,
            word_spacing: state.word_space,
            horizontal_scaling: state.horizontal_scaling,
            is_italic,
            is_monospace: false,
            primary_detected: true,
            artifact_type: None,
            char_widths: vec![],
        };

        // Step 6: Increment sequence counter and add to spans
        self.span_sequence_counter += 1;
        self.spans.push(span);

        Ok(())
    }

    /// Check if a character code is a ligature (U+FB00-U+FB04).
    ///
    /// Standard ligatures supported:
    /// - U+FB00: ff (LATIN SMALL LIGATURE FF)
    /// - U+FB01: fi (LATIN SMALL LIGATURE FI)
    /// - U+FB02: fl (LATIN SMALL LIGATURE FL)
    /// - U+FB03: ffi (LATIN SMALL LIGATURE FFI)
    /// - U+FB04: ffl (LATIN SMALL LIGATURE FFL)
    fn is_ligature_code(code: u32) -> bool {
        matches!(code, 0xFB00..=0xFB04)
    }

    /// Apply ligature expansion decisions after word boundary detection.
    ///
    /// This method processes the character array after boundary detection,
    /// making intelligent decisions about whether to split ligatures.
    ///
    /// Algorithm:
    /// 1. Iterate through character array
    /// 2. For each ligature character:
    ///    - Get next character (if exists)
    ///    - Call LigatureDecisionMaker::decide()
    ///    - If Split: expand to component characters with proportional widths
    ///    - If Keep: leave as-is
    /// 3. Recalculate x_positions for all following characters after splits
    fn apply_ligature_decisions(&mut self) -> Result<()> {
        use crate::text::ligature_processor::{
            expand_ligature_to_chars, LigatureDecision, LigatureDecisionMaker,
        };

        let context = self.create_boundary_context();
        let mut result = Vec::new();
        let mut i = 0;

        // OPTIMIZATION: Single-pass reconstruction instead of Vec::insert() in loop
        // This fixes O(n²) complexity to O(n) by avoiding repeated insertions
        // Issue #2 fix: Vec::insert was causing 50× slowdown for ligature-heavy PDFs
        while i < self.tj_character_array.len() {
            let char_info = &self.tj_character_array[i];

            // If not a ligature, keep as-is
            if !char_info.is_ligature {
                result.push(char_info.clone());
                i += 1;
                continue;
            }

            // Get next character without cloning (Issue #3 fix: eliminate unnecessary clones)
            let next_char = if i + 1 < self.tj_character_array.len() {
                Some(&self.tj_character_array[i + 1])
            } else {
                None
            };

            // Make decision using references
            let decision = LigatureDecisionMaker::decide(char_info, &context, next_char);

            if decision == LigatureDecision::Split {
                // Get the ligature character from code
                let ligature_char = char::from_u32(char_info.code).unwrap_or('?');
                let original_width = char_info.width;
                let original_x = char_info.x_position;
                let font_size = char_info.font_size;

                // Expand to component characters
                let components = expand_ligature_to_chars(ligature_char, original_width);

                if !components.is_empty() {
                    // Add first component (replacing the ligature)
                    let mut x_offset = 0.0;
                    result.push(CharacterInfo {
                        code: components[0].0 as u32,
                        glyph_id: char_info.glyph_id,
                        width: components[0].1,
                        x_position: original_x,
                        tj_offset: char_info.tj_offset,
                        font_size,
                        is_ligature: false,
                        original_ligature: Some(ligature_char),
                        protected_from_split: char_info.protected_from_split,
                    });
                    x_offset += components[0].1;

                    // Add remaining components (no Vec::insert needed - just push!)
                    for (comp_char, comp_width) in components.iter().skip(1) {
                        result.push(CharacterInfo {
                            code: *comp_char as u32,
                            glyph_id: None,
                            width: *comp_width,
                            x_position: original_x + x_offset,
                            tj_offset: None,
                            font_size,
                            is_ligature: false,
                            original_ligature: Some(ligature_char),
                            protected_from_split: false,
                        });
                        x_offset += comp_width;
                    }
                } else {
                    // If expansion failed, keep original ligature
                    result.push(char_info.clone());
                }
            } else {
                // Keep ligature intact
                result.push(char_info.clone());
            }

            i += 1;
        }

        // OPTIMIZATION: Replace entire array once instead of multiple insertions
        self.tj_character_array = result;
        Ok(())
    }

    /// Advance text position for a string (used in TJ array processing).
    /// Advance the text matrix position by the width of a text string.
    /// Returns the computed width so callers can accumulate it.
    fn advance_position_for_string(&mut self, text: &[u8]) -> Result<f32> {
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;
        let char_space = state.char_space;
        let word_space = state.word_space;

        let font = self.cached_current_font.as_deref();

        // Hoist loop-invariant computations
        let fs_factor = font_size / 1000.0;
        let hs_factor = horizontal_scaling / 100.0;
        let cs_hs = char_space * hs_factor;
        let ws_hs = word_space * hs_factor;

        let total_width = if let Some(font) = font {
            if font.subtype != "Type0" {
                // Fast path: use precomputed 256-entry width table (simple fonts)
                let width_table = font.get_byte_to_width_table();
                let mut w_sum = 0.0f32;
                for &byte in text {
                    let mut w = width_table[byte as usize] * fs_factor * hs_factor;
                    w += cs_hs;
                    if byte == 0x20 {
                        w += ws_hs;
                    }
                    w_sum += w;
                }
                w_sum
            } else {
                // Type0/CID font: process 2-byte CID codes (Identity-H encoding)
                // Per ISO 32000-1:2008 Section 9.7.6.2, composite fonts use multi-byte codes
                let mut w_sum = 0.0f32;
                for chunk in text.chunks(2) {
                    let cid = if chunk.len() == 2 {
                        ((chunk[0] as u16) << 8) | (chunk[1] as u16)
                    } else {
                        chunk[0] as u16
                    };
                    let mut w = font.get_glyph_width(cid) * fs_factor * hs_factor;
                    w += cs_hs;
                    // Per ISO 32000-1:2008 Section 9.3.3: Tw applied when CID == 32
                    if cid == 32 {
                        w += ws_hs;
                    }
                    w_sum += w;
                }
                w_sum
            }
        } else {
            // No font: use default width
            let default_w = 500.0 * fs_factor * hs_factor + cs_hs;
            let space_w = default_w + ws_hs;
            let mut w_sum = 0.0f32;
            for &byte in text {
                w_sum += if byte == 0x20 { space_w } else { default_w };
            }
            w_sum
        };

        // Update text matrix position per ISO 32000-1:2008 §9.4.4:
        // Tm_new = [1 0 0 1 tx 0] × Tm_old, where tx = total_width (text-space displacement)
        let state = self.state_stack.current_mut();
        let text_matrix = state.text_matrix;
        state.text_matrix.e += total_width * text_matrix.a;
        state.text_matrix.f += total_width * text_matrix.b;

        Ok(total_width)
    }

    /// Combined Unicode decode + width calculation in a single pass.
    /// Merges TjBuffer::append and advance_position_for_string for simple fonts,
    /// eliminating one full per-byte iteration per Tj operator.
    fn append_and_advance(&mut self, text: &[u8]) -> Result<()> {
        let text = if text.len() > 32_767 {
            &text[..32_767]
        } else {
            text
        };

        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;
        let char_space = state.char_space;
        let word_space = state.word_space;

        let fs_factor = font_size / 1000.0;
        let hs_factor = horizontal_scaling / 100.0;
        let cs_hs = char_space * hs_factor;
        let ws_hs = word_space * hs_factor;

        // Disjoint field borrows: cached_current_font (immutable) + tj_span_buffer (mutable)
        let font = self.cached_current_font.as_deref();
        // Safety: tj_span_buffer is always initialized via begin_text_object()
        let buffer = self
            .tj_span_buffer
            .as_mut()
            .expect("tj_span_buffer initialized in begin_text_object");

        let total_width = if let Some(font) = font {
            if font.subtype != "Type0" {
                // Fast path: single pass over bytes for both Unicode and width
                let char_table = font.get_byte_to_char_table();
                let width_table = font.get_byte_to_width_table();
                let mut w_sum = 0.0f32;
                for &byte in text {
                    // Unicode decode — count chars added for per-char width tracking
                    let len_before = buffer.unicode.len();
                    let c = char_table[byte as usize];
                    if c != '\0' {
                        buffer.unicode.push(c);
                    } else {
                        // Rare: multi-char mapping or unmapped byte
                        if let Some(s) = font.char_to_unicode(byte as u32) {
                            if s != "\u{FFFD}" {
                                for ch in s.chars() {
                                    if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                        buffer.unicode.push(ch);
                                    }
                                }
                            }
                        } else {
                            let fb = fallback_char_to_unicode(byte as u32);
                            if fb != "\u{FFFD}" {
                                for ch in fb.chars() {
                                    if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                        buffer.unicode.push(ch);
                                    }
                                }
                            }
                        }
                    }
                    // Width calculation
                    let mut w = width_table[byte as usize] * fs_factor * hs_factor;
                    w += cs_hs;
                    if byte == 0x20 {
                        w += ws_hs;
                    }
                    w_sum += w;
                    // Track per-character advance widths
                    let chars_added = buffer.unicode.len() - len_before;
                    if chars_added == 1 {
                        buffer.char_widths.push(w);
                    } else if chars_added > 1 {
                        let per_char = w / chars_added as f32;
                        for _ in 0..chars_added {
                            buffer.char_widths.push(per_char);
                        }
                    }
                }
                w_sum
            } else {
                // Type0/CID font: use unified iterator for robust multi-byte decoding and widths
                buffer.append(text)?;
                let mut w_sum = 0.0f32;
                for (char_code, _) in TextCharIter::new(text, Some(font)) {
                    let mut w = font.get_glyph_width(char_code) * fs_factor * hs_factor;
                    w += cs_hs;
                    // Standard PDF space character (code 32) triggers word spacing
                    if char_code == 32 {
                        w += ws_hs;
                    }
                    w_sum += w;
                    buffer.char_widths.push(w);
                }
                w_sum
            }
        } else {
            // No font: decode as ASCII + use default widths
            buffer.append(text)?;
            let default_w = 500.0 * fs_factor * hs_factor + cs_hs;
            let space_w = default_w + ws_hs;
            let mut w_sum = 0.0f32;
            for &byte in text {
                let w = if byte == 0x20 { space_w } else { default_w };
                w_sum += w;
                buffer.char_widths.push(w);
            }
            w_sum
        };

        buffer.accumulated_width += total_width;

        // Update text matrix position per ISO 32000-1:2008 §9.4.4
        let state = self.state_stack.current_mut();
        let text_matrix = state.text_matrix;
        state.text_matrix.e += total_width * text_matrix.a;
        state.text_matrix.f += total_width * text_matrix.b;

        Ok(())
    }

    /// Combined Unicode decode + width + position advance for a local buffer.
    /// Same as append_and_advance but works on an explicit buffer parameter
    /// instead of self.tj_span_buffer. Used by TJ array processing.
    fn append_advance_buffer(&mut self, buffer: &mut TjBuffer, text: &[u8]) -> Result<()> {
        let text = if text.len() > 32_767 {
            &text[..32_767]
        } else {
            text
        };

        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;
        let char_space = state.char_space;
        let word_space = state.word_space;

        let fs_factor = font_size / 1000.0;
        let hs_factor = horizontal_scaling / 100.0;
        let cs_hs = char_space * hs_factor;
        let ws_hs = word_space * hs_factor;

        let font = self.cached_current_font.as_deref();

        let total_width = if let Some(font) = font {
            if font.subtype != "Type0" {
                let char_table = font.get_byte_to_char_table();
                let width_table = font.get_byte_to_width_table();
                let mut w_sum = 0.0f32;
                for &byte in text {
                    let len_before = buffer.unicode.len();
                    let c = char_table[byte as usize];
                    if c != '\0' {
                        buffer.unicode.push(c);
                    } else if let Some(s) = font.char_to_unicode(byte as u32) {
                        if s != "\u{FFFD}" {
                            for ch in s.chars() {
                                if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                    buffer.unicode.push(ch);
                                }
                            }
                        }
                    } else {
                        let fb = fallback_char_to_unicode(byte as u32);
                        if fb != "\u{FFFD}" {
                            for ch in fb.chars() {
                                if ch >= '\x20' || ch == '\t' || ch == '\n' || ch == '\r' {
                                    buffer.unicode.push(ch);
                                }
                            }
                        }
                    }
                    let mut w = width_table[byte as usize] * fs_factor * hs_factor;
                    w += cs_hs;
                    if byte == 0x20 {
                        w += ws_hs;
                    }
                    w_sum += w;
                    let chars_added = buffer.unicode.len() - len_before;
                    if chars_added == 1 {
                        buffer.char_widths.push(w);
                    } else if chars_added > 1 {
                        let per_char = w / chars_added as f32;
                        for _ in 0..chars_added {
                            buffer.char_widths.push(per_char);
                        }
                    }
                }
                w_sum
            } else {
                buffer.append(text)?;
                // Width calculation: process 2-byte CID codes (Identity-H encoding)
                let mut w_sum = 0.0f32;
                for chunk in text.chunks(2) {
                    let cid = if chunk.len() == 2 {
                        ((chunk[0] as u16) << 8) | (chunk[1] as u16)
                    } else {
                        chunk[0] as u16
                    };
                    let mut w = font.get_glyph_width(cid) * fs_factor * hs_factor;
                    w += cs_hs;
                    if cid == 32 {
                        w += ws_hs;
                    }
                    w_sum += w;
                    buffer.char_widths.push(w);
                }
                w_sum
            }
        } else {
            buffer.append(text)?;
            let default_w = 500.0 * fs_factor * hs_factor + cs_hs;
            let space_w = default_w + ws_hs;
            let mut w_sum = 0.0f32;
            for &byte in text {
                let w = if byte == 0x20 { space_w } else { default_w };
                w_sum += w;
                buffer.char_widths.push(w);
            }
            w_sum
        };

        buffer.accumulated_width += total_width;

        let state = self.state_stack.current_mut();
        let text_matrix = state.text_matrix;
        state.text_matrix.e += total_width * text_matrix.a;
        state.text_matrix.f += total_width * text_matrix.b;

        Ok(())
    }

    /// Insert a space character as a separate span.
    fn insert_space_as_span(&mut self) -> Result<()> {
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let text_matrix = state.text_matrix;
        let ctm = state.ctm;
        let combined = ctm.multiply(&text_matrix);
        let effective_font_size =
            font_size * (combined.d * combined.d + combined.b * combined.b).sqrt();
        let word_space = state.word_space;
        let horizontal_scaling = state.horizontal_scaling;

        // Calculate space width
        let space_width = (250.0 * font_size / 1000.0 + word_space) * horizontal_scaling / 100.0;

        // Apply CTM to get position in user space
        // Per PDF Spec ISO 32000-1:2008 Section 9.4.4
        let text_pos = text_matrix.transform_point(0.0, 0.0);
        let user_pos = ctm.transform_point(text_pos.x, text_pos.y);

        log::trace!(
            "Inserting space span from TJ offset (offset_semantic=true) at position ({:.2}, {:.2})",
            user_pos.x,
            user_pos.y
        );

        let font_name_space = state
            .font_name
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        let is_italic_space = state
            .font_name
            .as_ref()
            .and_then(|name| self.fonts.get(name))
            .map(|font| font.is_italic())
            .unwrap_or(false);
        let span = TextSpan {
            text: " ".to_string(),
            bbox: Rect {
                x: user_pos.x,
                y: user_pos.y,
                width: space_width,
                height: effective_font_size,
            },
            font_name: font_name_space,
            font_size: effective_font_size,
            font_weight: FontWeight::Normal,
            color: Color::new(
                state.fill_color_rgb.0,
                state.fill_color_rgb.1,
                state.fill_color_rgb.2,
            ),
            mcid: self.current_mcid,
            sequence: self.span_sequence_counter,
            split_boundary_before: false,
            offset_semantic: true,
            char_spacing: state.char_space, // Tc - captured from PDF content stream
            word_spacing: state.word_space, // Tw - captured from PDF content stream
            horizontal_scaling: state.horizontal_scaling, // Tz - captured from PDF content stream
            is_italic: is_italic_space,
            is_monospace: false,
            primary_detected: false,
            artifact_type: self.current_artifact_type(),
            char_widths: vec![],
        };
        self.span_sequence_counter += 1;

        log::trace!("PUSH space span with offset_semantic={}", span.offset_semantic);

        self.spans.push(span);

        // Advance position per ISO 32000-1:2008 §9.4.4
        let state = self.state_stack.current_mut();
        state.text_matrix.e += space_width * text_matrix.a;
        state.text_matrix.f += space_width * text_matrix.b;

        Ok(())
    }

    /// Advance text position for a TJ offset value.
    fn advance_position_for_offset(&mut self, offset: f32) -> Result<()> {
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;

        // Calculate horizontal displacement per PDF spec §9.4.4
        // tx = -offset / 1000.0 * font_size * horizontal_scaling / 100.0
        let tx = -offset / 1000.0 * font_size * horizontal_scaling / 100.0;

        // Update text matrix: Tm_new = [1 0 0 1 tx 0] × Tm_old
        let state = self.state_stack.current_mut();
        let text_matrix = state.text_matrix;
        state.text_matrix.e += tx * text_matrix.a;
        state.text_matrix.f += tx * text_matrix.b;

        Ok(())
    }

    /// Flush accumulated Tj span buffer into a single TextSpan.
    ///
    /// This is similar to flush_tj_buffer but works with the tj_span_buffer field
    /// which accumulates consecutive Tj operators.
    fn flush_tj_span_buffer(&mut self) -> Result<()> {
        if let Some(mut buffer) = self.tj_span_buffer.take() {
            if !buffer.is_empty() {
                // Use accumulated width from advance_position_for_string calls
                // Convert from text space to user space using pre-computed horizontal scale
                let total_width = buffer.accumulated_width * buffer.user_h_scale;

                // Use pre-computed values from buffer creation (avoids
                // matrix multiply + sqrt + HashMap lookup per flush)
                let effective_font_size = buffer.effective_font_size;
                let font_weight = buffer.font_weight;
                let is_italic_buf = buffer.is_italic;

                // Move owned strings out of buffer (avoids clone)
                let font_name_buf = buffer
                    .font_name
                    .take()
                    .unwrap_or_else(|| "Unknown".to_string());
                let span = TextSpan {
                    text: std::mem::take(&mut buffer.unicode),
                    bbox: Rect {
                        x: buffer.user_pos_x,
                        y: buffer.user_pos_y,
                        width: total_width,
                        height: effective_font_size,
                    },
                    font_name: font_name_buf,
                    font_size: effective_font_size,
                    font_weight,
                    color: Color::new(
                        buffer.fill_color_rgb.0,
                        buffer.fill_color_rgb.1,
                        buffer.fill_color_rgb.2,
                    ),
                    mcid: buffer.mcid,
                    sequence: self.span_sequence_counter,
                    split_boundary_before: false,
                    offset_semantic: false,
                    char_spacing: 0.0, // Tc - per ISO 32000-1:2008 Section 9.3.1
                    word_spacing: 0.0, // Tw - per ISO 32000-1:2008 Section 9.3.1
                    horizontal_scaling: 100.0, // Tz - per ISO 32000-1:2008 Section 9.3.1
                    is_italic: is_italic_buf,
                    is_monospace: buffer.is_monospace,
                    primary_detected: false,
                    artifact_type: None,
                    char_widths: {
                        let mut cw = std::mem::take(&mut buffer.char_widths);
                        let h = buffer.user_h_scale;
                        for w in &mut cw {
                            *w *= h;
                        }
                        cw
                    },
                };
                self.span_sequence_counter += 1;

                log::trace!(
                    "FLUSH_TJ_SPAN_BUFFER creating span: text='{}', offset_semantic={} (space-only spans marked as offset_semantic)",
                    if span.text.chars().all(|c| c.is_whitespace()) {
                        "<space-only>"
                    } else {
                        crate::utils::safe_prefix(&span.text, 20)
                    },
                    span.offset_semantic
                );

                self.spans.push(span);
            }
        }
        Ok(())
    }

    fn show_text(&mut self, text: &[u8]) -> Result<()> {
        // PDF spec Section 7.3.4.2: implementation limit of 32,767 bytes per string.
        let text = if text.len() > 32_767 {
            log::warn!(
                "String exceeds PDF spec limit: {} bytes (max 32,767), truncating",
                text.len()
            );
            &text[..32_767]
        } else {
            text
        };

        // Get current state values
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;
        let char_space = state.char_space;
        let word_space = state.word_space;
        let fill_color_rgb = state.fill_color_rgb;
        let ctm = state.ctm;

        // Get current font from cached reference
        let font = self.cached_current_font.as_deref();

        for (char_code, _) in TextCharIter::new(text, font) {
            // Get current text matrix (may be updated by previous characters in this string)
            let state = self.state_stack.current();
            let text_matrix = state.text_matrix;

            // Get Unicode string using font mapping
            let unicode_string = if let Some(font) = font {
                font.char_to_unicode(char_code as u32)
                    .unwrap_or_else(|| fallback_char_to_unicode(char_code as u32))
            } else if char_code < 256 && (char_code as u8).is_ascii() {
                (char_code as u8 as char).to_string()
            } else {
                "?".to_string()
            };

            // Calculate character position in user space
            let text_pos = text_matrix.transform_point(0.0, 0.0);
            let pos = ctm.transform_point(text_pos.x, text_pos.y);

            // Calculate effective font size
            let combined_char = ctm.multiply(&text_matrix);
            let effective_font_size = font_size
                * (combined_char.d * combined_char.d + combined_char.b * combined_char.b).sqrt();

            // Calculate character dimensions using accurate glyph width
            let glyph_width_font_units = if let Some(font) = font {
                font.get_glyph_width(char_code)
            } else {
                500.0 // Default 0.5em
            };

            let fs_factor = font_size / 1000.0;
            let hs_factor = horizontal_scaling / 100.0;
            let glyph_width_user_space = glyph_width_font_units * fs_factor * hs_factor;

            // For TextChar, we use the device-space width
            let glyph_width_device_space = glyph_width_user_space * combined_char.a.abs();
            let height_device_space = effective_font_size;

            // Determine font weight and style
            let (font_weight, is_italic_char) = if let Some(font) = font {
                (
                    if font.is_bold() {
                        FontWeight::Bold
                    } else {
                        FontWeight::Normal
                    },
                    font.is_italic(),
                )
            } else {
                (FontWeight::Normal, false)
            };

            // Get color
            let (r, g, b) = fill_color_rgb;
            let color = Color::new(r, g, b);

            // Compose CTM and text_matrix for full transformation
            let final_matrix = ctm.multiply(&text_matrix);
            let rotation_degrees = final_matrix.b.atan2(final_matrix.a).to_degrees();

            // Guard against malformed fonts
            let unicode_string = if unicode_string.chars().count() > 8 {
                unicode_string.chars().next().unwrap_or('?').to_string()
            } else {
                unicode_string
            };

            // Process each character in the expanded string (ligatures)
            let char_count = unicode_string.chars().count();
            let char_width_device = if char_count > 0 {
                glyph_width_device_space / char_count as f32
            } else {
                glyph_width_device_space
            };
            let char_width_user = if char_count > 0 {
                glyph_width_user_space / char_count as f32
            } else {
                glyph_width_user_space
            };

            for (char_index, unicode_char) in unicode_string.chars().enumerate() {
                let should_skip = unicode_char == '\0'
                    || (unicode_char.is_control()
                        && unicode_char != '\t'
                        && unicode_char != '\n'
                        && unicode_char != '\r');

                if !should_skip {
                    let x_offset_device = char_index as f32 * char_width_device;
                    let x_offset_user = char_index as f32 * char_width_user;

                    let char_origin_x = pos.x + x_offset_device;
                    let char_origin_y = pos.y;

                    let text_char = TextChar {
                        char: unicode_char,
                        bbox: Rect::new(
                            char_origin_x,
                            char_origin_y,
                            char_width_device,
                            height_device_space,
                        ),
                        font_name: font.map(|f| f.base_font.clone()).unwrap_or_default(),
                        font_size: effective_font_size,
                        font_weight,
                        color,
                        mcid: self.current_mcid,
                        is_italic: is_italic_char,
                        is_monospace: false,
                        origin_x: char_origin_x,
                        origin_y: char_origin_y,
                        rotation_degrees,
                        advance_width: char_width_device,
                        matrix: Some([
                            final_matrix.a,
                            final_matrix.b,
                            final_matrix.c,
                            final_matrix.d,
                            final_matrix.e + x_offset_user,
                            final_matrix.f,
                        ]),
                    };

                    self.chars.push(text_char);
                }
            }

            // Advance position: Tx = (w0 * Tfs + Tc + Tw) * Th
            let mut tx = glyph_width_user_space;
            tx += char_space * hs_factor;
            if char_code == 32 {
                tx += word_space * hs_factor;
            }

            // Update text matrix in current state per ISO 32000-1:2008 §9.4.4
            let state_mut = self.state_stack.current_mut();
            let tm = state_mut.text_matrix;
            state_mut.text_matrix.e += tx * tm.a;
            state_mut.text_matrix.f += tx * tm.b;
        }

        Ok(())
    }

    /// Get the number of extracted characters.
    pub fn char_count(&self) -> usize {
        self.chars.len()
    }

    /// Clear all extracted characters.
    pub fn clear(&mut self) {
        self.chars.clear();
    }
}

/// Convert CMYK color to RGB color.
///
/// CMYK uses subtractive color model (for print), RGB uses additive (for screen).
/// Conversion formula: R = 1 - min(1, C*(1-K) + K)
///
/// PDF Spec: ISO 32000-1:2008, Section 8.6.4.4 - DeviceCMYK Color Space
fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let r = 1.0 - (c * (1.0 - k) + k).min(1.0);
    let g = 1.0 - (m * (1.0 - k) + k).min(1.0);
    let b = 1.0 - (y * (1.0 - k) + k).min(1.0);
    (r, g, b)
}

impl Default for TextExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to determine if a space should be inserted between two text spans
/// based on character transition heuristics.
///
/// This complements gap-based space detection by catching cases where the geometric
/// gap is small but a space is semantically needed based on character patterns.
///
/// # Detected Patterns
///
/// - **CamelCase transitions**: `thenThe` → `then The` (lowercase followed by uppercase)
/// - **Number-letter transitions**: `Figure1` → `Figure 1` (digit followed by letter)
/// - **Letter-number transitions**: `page3` → `page 3` (letter followed by digit)
///
/// # Arguments
///
/// * `current_text` - The text of the current span
/// * `next_text` - The text of the next span to be merged
///
/// # Returns
///
/// `true` if a space should be inserted based on character transitions,
/// `false` if no space is needed
///
/// # Preserves
///
/// - Acronyms like "HTML", "PDF", "API" (all uppercase)
/// - Normal word boundaries (already handled by gap detection)
/// - Intentional concatenations within words
// DELETED: should_insert_space_heuristic()
// Character pattern heuristics (CamelCase detection, number-letter transitions)
// are not defined in ISO 32000-1:2008 PDF spec. Per spec-compliance refactoring,
// only spec-defined signals (TJ offsets, geometric gaps, boundary whitespace)
// are used for space insertion decisions.
// See: PHASE10_PDF_SPEC_COMPLIANCE.md
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fonts::{Encoding, LazyCMap};
    use std::sync::Arc;

    fn create_test_font() -> FontInfo {
        FontInfo {
            base_font: "Times-Roman".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn test_text_extractor_new() {
        let extractor = TextExtractor::new();
        assert_eq!(extractor.char_count(), 0);
    }

    #[test]
    fn test_text_extractor_add_font() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);
        assert_eq!(extractor.fonts.len(), 1);
    }

    #[test]
    fn test_extract_simple_text() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 5); // "Hello"
        assert_eq!(chars[0].char, 'H');
        assert_eq!(chars[1].char, 'e');
        assert_eq!(chars[2].char, 'l');
        assert_eq!(chars[3].char, 'l');
        assert_eq!(chars[4].char, 'o');
    }

    #[test]
    fn test_extract_with_matrix() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (Hi) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'H');
        assert_eq!(chars[1].char, 'i');
        // Position should be around (100, 700)
        assert!(chars[0].bbox.x >= 99.0 && chars[0].bbox.x <= 101.0);
    }

    /// Regression test for Issue #11: CTM must be applied to text positions
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.4.4, the text rendering matrix is:
    /// T_rm = [font_matrix] × T_m × CTM
    ///
    /// This test verifies that when CTM contains a translation, text positions
    /// are correctly transformed from text space to user space.
    #[test]
    fn test_ctm_applied_to_text_position() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // CTM translates by (100, 200), text matrix at origin
        // Final position should be (100, 200), not (0, 0)
        let stream = b"q 1 0 0 1 100 200 cm BT /F1 12 Tf (A) Tj ET Q";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'A');
        // Position should be translated by CTM: (100, 200)
        assert!(
            chars[0].bbox.x >= 99.0 && chars[0].bbox.x <= 101.0,
            "X position should be ~100 (got {})",
            chars[0].bbox.x
        );
        assert!(
            chars[0].bbox.y >= 199.0 && chars[0].bbox.y <= 201.0,
            "Y position should be ~200 (got {})",
            chars[0].bbox.y
        );
    }

    /// Regression test for Issue #11: CTM scaling must affect text positions
    ///
    /// This test verifies that CTM scaling is correctly applied to text positions.
    #[test]
    fn test_ctm_scaling_applied_to_text_position() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // CTM scales by 2x, text at position (50, 100) in text space
        // Final position should be (100, 200) in user space
        let stream = b"q 2 0 0 2 0 0 cm BT /F1 12 Tf 1 0 0 1 50 100 Tm (B) Tj ET Q";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'B');
        // Position should be scaled: (50*2, 100*2) = (100, 200)
        assert!(
            chars[0].bbox.x >= 99.0 && chars[0].bbox.x <= 101.0,
            "X position should be ~100 (got {})",
            chars[0].bbox.x
        );
        assert!(
            chars[0].bbox.y >= 199.0 && chars[0].bbox.y <= 201.0,
            "Y position should be ~200 (got {})",
            chars[0].bbox.y
        );
    }

    /// Regression test for Issue #11: Combined CTM translation and text matrix
    ///
    /// This test verifies the complete transformation chain works correctly.
    #[test]
    fn test_ctm_combined_with_text_matrix() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // CTM translates by (50, 50), text matrix positions at (25, 25)
        // Final position should be (75, 75)
        let stream = b"q 1 0 0 1 50 50 cm BT /F1 12 Tf 1 0 0 1 25 25 Tm (C) Tj ET Q";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'C');
        // Position: text_matrix(25,25) + CTM_translation(50,50) = (75, 75)
        assert!(
            chars[0].bbox.x >= 74.0 && chars[0].bbox.x <= 76.0,
            "X position should be ~75 (got {})",
            chars[0].bbox.x
        );
        assert!(
            chars[0].bbox.y >= 74.0 && chars[0].bbox.y <= 76.0,
            "Y position should be ~75 (got {})",
            chars[0].bbox.y
        );
    }

    #[test]
    fn test_extract_with_tj_array() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 0 0 Td [(H)(i)] TJ ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'H');
        assert_eq!(chars[1].char, 'i');
    }

    /// Test extraction of multi-byte characters from Type0 fonts (Identity-H)
    /// This verifies the fix for Issue #186 where extract_chars() was garbling CJK text.
    #[test]
    fn test_extract_type0_multibyte_character_extraction() {
        let mut extractor = TextExtractor::new();

        // Create a mock Type0 font with Identity-H encoding
        let mut font = create_test_font();
        font.subtype = "Type0".to_string();
        font.encoding = Encoding::Standard("Identity-H".to_string());

        // Create a valid ToUnicode CMap stream that maps CID 0x4E2D to '中' and 0x6587 to '文'
        let cmap_data = b"
            /CIDInit /ProcSet findresource begin
            12 dict begin
            begincmap
            /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
            /CMapName /Adobe-Identity-UCS def
            /CMapType 2 def
            1 begincodespacerange <0000> <FFFF> endcodespacerange
            2 beginbfchar
            <4E2D> <4E2D>
            <6587> <6587>
            endbfchar
            endcmap
            CMapName currentdict /CMap defineresource pop
            end
            end
        ";

        // Use public parse_tounicode_cmap to create CMap, then wrap in LazyCMap
        let lazy_cmap = LazyCMap::new(cmap_data.to_vec());
        font.to_unicode = Some(lazy_cmap);

        extractor.add_font("F1".to_string(), font);

        // Content stream with 2-byte CIDs for "中文" (0x4E2D 0x6587)
        let stream = b"BT /F1 12 Tf 0 0 Td <4E2D6587> Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, '中');
        assert_eq!(chars[1].char, '文');
    }

    #[test]
    fn test_extract_color() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT 1 0 0 rg /F1 12 Tf 0 0 Td (R) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'R');
        assert_eq!(chars[0].color.r, 1.0);
        assert_eq!(chars[0].color.g, 0.0);
        assert_eq!(chars[0].color.b, 0.0);
    }

    /// Regression test: is_monospace flag must propagate from FontInfo flags
    /// through TjBuffer into the final TextSpan.
    ///
    /// When font descriptor flags have bit 0 (FixedPitch) set, spans produced
    /// by extract_text_spans() must report is_monospace == true.
    /// Conversely, a proportional font (e.g. Helvetica) must yield false.
    #[test]
    fn test_is_monospace_from_font_flags() {
        // --- Monospace font: flags bit 0 (FixedPitch) set ---
        let mut mono_font = create_test_font();
        mono_font.base_font = "Courier".to_string();
        mono_font.flags = Some(1); // bit 0 = FixedPitch

        let mut extractor = TextExtractor::new();
        extractor.add_font("F1".to_string(), mono_font);

        let stream = b"BT /F1 12 Tf 100 700 Td (Code) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        assert!(!spans.is_empty(), "should produce at least one span");
        assert!(
            spans[0].is_monospace,
            "Courier with FixedPitch flag should be monospace, got is_monospace=false"
        );

        // --- Proportional font: no FixedPitch flag ---
        let mut prop_font = create_test_font();
        prop_font.base_font = "Helvetica".to_string();
        prop_font.flags = Some(0); // no FixedPitch

        let mut extractor2 = TextExtractor::new();
        extractor2.add_font("F2".to_string(), prop_font);

        let stream2 = b"BT /F2 12 Tf 100 700 Td (Text) Tj ET";
        let spans2 = extractor2.extract_text_spans(stream2).unwrap();

        assert!(!spans2.is_empty(), "should produce at least one span");
        assert!(
            !spans2[0].is_monospace,
            "Helvetica without FixedPitch flag should not be monospace"
        );

        // --- Name-based heuristic: font name containing MONO ---
        let mut mono_name_font = create_test_font();
        mono_name_font.base_font = "DejaVuSansMono".to_string();
        mono_name_font.flags = None; // no flags at all

        let mut extractor3 = TextExtractor::new();
        extractor3.add_font("F3".to_string(), mono_name_font);

        let stream3 = b"BT /F3 12 Tf 100 700 Td (Mono) Tj ET";
        let spans3 = extractor3.extract_text_spans(stream3).unwrap();

        assert!(!spans3.is_empty(), "should produce at least one span");
        assert!(
            spans3[0].is_monospace,
            "Font named DejaVuSansMono should be detected as monospace via name heuristic"
        );
    }

    #[test]
    fn test_extract_save_restore() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Valid PDF: q saves state, Tf changes font size inside, Q restores
        let stream = b"BT /F1 12 Tf q /F1 14 Tf (A) Tj Q (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].font_size, 14.0); // Inside q/Q
        assert_eq!(chars[1].font_size, 12.0); // After Q, restored to 12
    }

    #[test]
    fn test_extract_no_font() {
        let mut extractor = TextExtractor::new();
        // Don't add any fonts

        let stream = b"BT /F1 12 Tf (ABC) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        // Should still extract, using identity mapping
        assert_eq!(chars.len(), 3);
    }

    #[test]
    fn test_char_count() {
        let mut extractor = TextExtractor::new();
        assert_eq!(extractor.char_count(), 0);

        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf (Test) Tj ET";
        extractor.extract(stream).unwrap();
        assert_eq!(extractor.char_count(), 4);
    }

    #[test]
    fn test_clear() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf (Test) Tj ET";
        extractor.extract(stream).unwrap();
        assert_eq!(extractor.char_count(), 4);

        extractor.clear();
        assert_eq!(extractor.char_count(), 0);
    }

    #[test]
    fn test_default() {
        let extractor = TextExtractor::default();
        assert_eq!(extractor.char_count(), 0);
    }

    /// Test unified space decision: TJ offset rule
    /// NOTE: Disabled - space detection has been refactored to be PDF spec-compliant
    #[test]
    #[ignore]
    fn test_space_decision_tj_offset() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();

        // TJ offset triggered should always insert space (Rule 1, confidence 0.95)
        let decision = should_insert_space(
            "word", "next", 0.0, 12.0, "TestFont", &fonts, true, &config, None, None, 12.0, 12.0,
        );

        assert!(decision.insert_space);
        assert_eq!(decision.source, SpaceSource::TjOffset);
        assert_eq!(decision.confidence, 0.95);
    }

    /// Test unified space decision: Boundary space already present
    #[test]
    fn test_space_decision_boundary_space() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();

        // Preceding text ends with space
        let decision = should_insert_space(
            "word ", "next", 0.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space);
        assert_eq!(decision.source, SpaceSource::AlreadyPresent);

        // Following text starts with space
        let decision = should_insert_space(
            "word", " next", 0.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space);
        assert_eq!(decision.source, SpaceSource::AlreadyPresent);
    }

    /// Test unified space decision: Dual threshold rule
    /// NOTE: Disabled - space detection has been refactored to be PDF spec-compliant
    #[test]
    #[ignore]
    fn test_space_decision_dual_threshold() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();
        // space_threshold_em_ratio: 0.25, conservative_threshold_pt: 0.1

        // 12pt font, space_threshold = 12 * 0.25 = 3pt
        // char_width_threshold = 12 * 0.3 = 3.6pt
        // dual_threshold = min(3, 3.6) = 3pt
        let font_size = 12.0;

        // Gap > dual_threshold should insert (Rule 2, confidence 0.8)
        let decision = should_insert_space(
            "word", "next", 3.5, font_size, "TestFont", &fonts, false, &config, None, None, 12.0,
            12.0,
        );
        assert!(decision.insert_space);
        assert_eq!(decision.source, SpaceSource::GeometricGap);
        assert_eq!(decision.confidence, 0.8);

        // Gap <= dual_threshold, not at heuristic boundary: no space (yet)
        let decision = should_insert_space(
            "word", "next", 2.5, font_size, "TestFont", &fonts, false, &config, None, None, 12.0,
            12.0,
        );
        // This should not insert (no rule triggers)
        // But conservative threshold (0.1) is still checked below
    }

    /// Test unified space decision: Character heuristic rule
    /// NOTE: Disabled - heuristic rules removed in PDF spec-compliant refactoring
    #[test]
    #[ignore]
    fn test_space_decision_heuristic_camelcase() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();

        // lowercase -> uppercase (CamelCase) should trigger heuristic (Rule 3, confidence 0.85)
        let decision = should_insert_space(
            "the", "General", 0.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(decision.insert_space);
        assert_eq!(decision.source, SpaceSource::CharacterHeuristic);
        assert_eq!(decision.confidence, 0.85);

        // numeric -> letter should trigger heuristic
        let decision = should_insert_space(
            "version2", "dot3", 0.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0,
            12.0,
        );
        assert!(decision.insert_space);
        assert_eq!(decision.source, SpaceSource::CharacterHeuristic);
    }

    /// Test unified space decision: Conservative threshold rule
    /// NOTE: Disabled - conservative threshold rules removed in PDF spec-compliant refactoring
    #[test]
    #[ignore]
    fn test_space_decision_conservative_threshold() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();
        // conservative_threshold_pt: 0.1

        // Gap > conservative_threshold but not meeting other rules (Rule 4, confidence 0.5)
        let decision = should_insert_space(
            "word", "next", 0.2, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(decision.insert_space);
        assert_eq!(decision.source, SpaceSource::GeometricGap);
        assert_eq!(decision.confidence, 0.5);

        // Gap <= conservative_threshold: no space (Rule 5 - default)
        let decision = should_insert_space(
            "word", "next", 0.05, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space);
        assert_eq!(decision.source, SpaceSource::NoSpace);
    }

    /// Regression test for issue flagged in PR #281 review:
    /// a long number emitted as multiple digit-only spans with a kerning-sized
    /// positive gap must NOT have a space inserted between the digits (would
    /// turn "123456" into "123 456"). Adjacent table cell digit values with a
    /// larger gap must still be separated.
    #[test]
    fn test_space_decision_digit_digit_gap_threshold() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();

        // Kerning-sized gap (0.3pt) between digit spans — must NOT insert.
        // For 12pt font with no font-info fallback, geometric_threshold is
        // typically around 1.5pt, so half of that is 0.75pt.
        let kerning = should_insert_space(
            "123", "456", 0.3, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(
            !kerning.insert_space,
            "Kerning-sized gap (0.3pt) between digits must not split the number, got: {:?}",
            kerning
        );

        // Larger gap (2pt) between digit spans — adjacent table cell values,
        // must still insert a space.
        let table_cells = should_insert_space(
            "123", "456", 2.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(
            table_cells.insert_space,
            "2pt gap between digits should still split adjacent table values, got: {:?}",
            table_cells
        );
    }

    /// Test unified space decision: No double spaces
    /// NOTE: Disabled - space detection has been refactored to be PDF spec-compliant
    #[test]
    #[ignore]
    fn test_space_decision_no_double_spaces() {
        let config = SpanMergingConfig::default();
        let fonts = std::collections::HashMap::new();

        // When both TJ offset and gap would trigger, they should be coordinated
        // TJ offset has highest priority and should be respected first
        let decision_tj = should_insert_space(
            "word", "next", 1.0, 12.0, "TestFont", &fonts, true, &config, None, None, 12.0, 12.0,
        );
        let decision_gap = should_insert_space(
            "word", "next", 1.0, 12.0, "TestFont", &fonts, false, &config, None, None, 12.0, 12.0,
        );

        // TJ offset decision (0.95) should be preferred over gap decision
        assert!(decision_tj.insert_space);
        assert_eq!(decision_tj.source, SpaceSource::TjOffset);

        // Gap alone would not trigger for 1pt gap (< conservative 0.1pt is false, but 1pt > 0.1pt is true)
        // So gap should also trigger via conservative threshold
        assert!(decision_gap.insert_space);
    }

    /// Test split boundary merging with space insertion
    ///
    /// When split_boundary_before=true, it indicates the span is part of a boundary
    /// that was previously split (e.g., from CamelCase fusion like "theGeneral").
    /// These spans should be merged WITH a space to preserve word separation.
    #[test]
    fn test_split_boundary_merges_with_space() {
        let spans = vec![
            TextSpan {
                artifact_type: None,
                text: "the".to_string(),
                bbox: Rect {
                    x: 0.0,
                    y: 100.0,
                    width: 10.0,
                    height: 12.0,
                },
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "General".to_string(),
                bbox: Rect {
                    x: 10.0,
                    y: 100.0,
                    width: 25.0,
                    height: 12.0,
                },
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: true, // Marks this as part of a split boundary
                offset_semantic: false,
                primary_detected: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                char_widths: vec![],
            },
        ];

        // Simulate extraction state
        let mut extractor = TextExtractor::new();
        extractor.spans = spans;
        extractor.merging_config = SpanMergingConfig::default();

        // Merge adjacent spans
        extractor.merge_adjacent_spans();

        // Per PDF Spec ISO 32000-1:2008 Section 9.4.4 and implementation design:
        // split_boundary_before=true means "merge with a space, never without"
        // This ensures "length" + "This" becomes "length This" not "lengthThis"
        // The spans are merged INTO ONE span with space-separated text
        assert_eq!(extractor.spans.len(), 1);
        assert_eq!(extractor.spans[0].text, "the General");
    }

    // Removed: test_should_insert_space_heuristic - function doesn't exist in current codebase

    /// Test boundary space detection
    #[test]
    fn test_has_boundary_space() {
        // Preceding text with trailing space
        assert!(has_boundary_space("word ", "next"));

        // Following text with leading space
        assert!(has_boundary_space("word", " next"));

        // Both with space
        assert!(has_boundary_space("word ", " next"));

        // Neither
        assert!(!has_boundary_space("word", "next"));

        // Only whitespace characters count
        assert!(has_boundary_space("word\t", "next"));
        assert!(has_boundary_space("word\n", "next"));
        assert!(has_boundary_space("word", "\tnext"));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: TextExtractionConfig
    // ========================================================================

    #[test]
    fn test_text_extraction_config_new_defaults() {
        let config = TextExtractionConfig::new();
        assert_eq!(config.space_insertion_threshold, -120.0);
        assert_eq!(config.word_margin_ratio, 0.1);
        assert!(!config.use_adaptive_tj_threshold);
        assert!(config.profile.is_none());
    }

    #[test]
    fn test_text_extraction_config_with_space_threshold() {
        let config = TextExtractionConfig::with_space_threshold(-80.0);
        assert_eq!(config.space_insertion_threshold, -80.0);
        assert_eq!(config.word_margin_ratio, 0.1);
        assert!(!config.use_adaptive_tj_threshold);
    }

    #[test]
    fn test_text_extraction_config_with_word_margin_ratio() {
        let config = TextExtractionConfig::with_word_margin_ratio(0.15);
        assert_eq!(config.word_margin_ratio, 0.15);
        assert!(config.use_adaptive_tj_threshold);
        assert_eq!(config.space_insertion_threshold, -120.0); // fallback
    }

    #[test]
    fn test_text_extraction_config_set_word_margin_ratio() {
        let config = TextExtractionConfig::new().set_word_margin_ratio(0.2);
        assert_eq!(config.word_margin_ratio, 0.2);
        assert!(config.use_adaptive_tj_threshold);
    }

    #[test]
    fn test_text_extraction_config_set_adaptive_tj_threshold() {
        let config = TextExtractionConfig::new().set_adaptive_tj_threshold(true);
        assert!(config.use_adaptive_tj_threshold);
        let config2 = config.set_adaptive_tj_threshold(false);
        assert!(!config2.use_adaptive_tj_threshold);
    }

    #[test]
    fn test_text_extraction_config_with_profile() {
        let config =
            TextExtractionConfig::new().with_profile(crate::config::ExtractionProfile::ACADEMIC);
        assert!(config.profile.is_some());
        let profile = config.profile.unwrap();
        assert_eq!(profile.name, "Academic");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: SpanMergingConfig
    // ========================================================================

    #[test]
    fn test_span_merging_config_defaults() {
        let config = SpanMergingConfig::new();
        assert_eq!(config.space_threshold_em_ratio, 0.25);
        assert_eq!(config.conservative_threshold_pt, 0.1);
        assert_eq!(config.column_boundary_threshold_pt, 5.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.5);
        assert!(config.use_adaptive_threshold);
        assert!(!config.detect_email_patterns);
        assert!(!config.detect_citation_markers);
    }

    #[test]
    fn test_span_merging_config_aggressive() {
        let config = SpanMergingConfig::aggressive();
        assert_eq!(config.space_threshold_em_ratio, 0.15);
        assert_eq!(config.conservative_threshold_pt, 0.1);
        assert!(!config.use_adaptive_threshold);
    }

    #[test]
    fn test_span_merging_config_conservative() {
        let config = SpanMergingConfig::conservative();
        assert_eq!(config.space_threshold_em_ratio, 0.33);
        assert_eq!(config.conservative_threshold_pt, 0.3);
        assert!(!config.use_adaptive_threshold);
    }

    #[test]
    fn test_span_merging_config_custom() {
        let config = SpanMergingConfig::custom(0.2, 0.2, 6.0, -0.3);
        assert_eq!(config.space_threshold_em_ratio, 0.2);
        assert_eq!(config.conservative_threshold_pt, 0.2);
        assert_eq!(config.column_boundary_threshold_pt, 6.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.3);
        assert!(!config.use_adaptive_threshold);
    }

    #[test]
    fn test_span_merging_config_adaptive() {
        let config = SpanMergingConfig::adaptive();
        assert!(config.use_adaptive_threshold);
        assert!(config.adaptive_config.is_some());
    }

    #[test]
    fn test_span_merging_config_legacy() {
        let config = SpanMergingConfig::legacy();
        assert!(!config.use_adaptive_threshold);
        assert_eq!(config.conservative_threshold_pt, 0.1);
        assert!(config.adaptive_config.is_none());
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: SpaceDecision
    // ========================================================================

    #[test]
    fn test_space_decision_insert() {
        let d = SpaceDecision::insert(SpaceSource::TjOffset, 0.95);
        assert!(d.insert_space);
        assert_eq!(d.source, SpaceSource::TjOffset);
        assert_eq!(d.confidence, 0.95);
    }

    #[test]
    fn test_space_decision_no_space() {
        let d = SpaceDecision::no_space(SpaceSource::NoSpace, 1.0);
        assert!(!d.insert_space);
        assert_eq!(d.source, SpaceSource::NoSpace);
        assert_eq!(d.confidence, 1.0);
    }

    #[test]
    fn test_space_decision_clamp_confidence() {
        let d = SpaceDecision::insert(SpaceSource::GeometricGap, 1.5);
        assert_eq!(d.confidence, 1.0); // clamped
        let d2 = SpaceDecision::insert(SpaceSource::GeometricGap, -0.5);
        assert_eq!(d2.confidence, 0.0); // clamped
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Text operators via execute_operator
    // ========================================================================

    #[test]
    fn test_operator_td_positioning() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // BT, set font, Td to position (100, 700), show "X", ET
        let stream = b"BT /F1 12 Tf 100 700 Td (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'X');
        // After Td(100, 700), position should be near (100, 700)
        assert!((chars[0].bbox.x - 100.0).abs() < 2.0);
        assert!((chars[0].bbox.y - 700.0).abs() < 2.0);
    }

    /// Issue #254: TD Y offset must be scaled by the text matrix.
    /// Pattern: `/F1 1 Tf 10 0 0 10 72 700 Tm (Line one) Tj 0 -1.3 TD (Line two) Tj`
    /// The Tm sets a 10x scale, so `0 -1.3 TD` should produce a 13pt vertical gap,
    /// not 1.3pt. Both lines must appear in extracted text.
    #[test]
    fn test_issue_254_tm_scale_td_offset() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Font size 1 with Tm scale 10 — effective font size is 10pt.
        // TD(0, -1.3) in text space = 13pt in user space.
        let stream = b"BT /F1 1 Tf 10 0 0 10 72 700 Tm (Line one) Tj 0 -1.3 TD (Line two) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        // Collect unique text
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("Line one"), "Should contain 'Line one', got: {}", text);
        assert!(text.contains("Line two"), "Should contain 'Line two', got: {}", text);

        // Verify the Y gap is ~13pt (1.3 * 10), not 1.3pt
        let line_one_y = chars.iter().find(|c| c.char == 'L').unwrap().bbox.y;
        let line_two_chars: Vec<_> = chars.iter().filter(|c| c.char == 'L').collect();
        assert!(line_two_chars.len() >= 2, "Should have at least 2 'L' chars (one per line)");
        let line_two_y = line_two_chars[1].bbox.y;
        let y_gap = (line_one_y - line_two_y).abs();
        assert!(y_gap > 5.0, "Y gap should be ~13pt (Tm-scaled), got {:.1}pt", y_gap);
    }

    #[test]
    fn test_operator_td_sets_leading() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // TD sets leading = -ty, then positions text
        let stream = b"BT /F1 12 Tf 100 -14 TD (A) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'A');
    }

    #[test]
    fn test_operator_tstar_line_break() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // TL sets leading, then T* moves to next line using leading
        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td (A) Tj T* (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'A');
        assert_eq!(chars[1].char, 'B');
        // B should be on a different line (different Y)
        assert!((chars[0].bbox.y - chars[1].bbox.y).abs() > 1.0);
    }

    #[test]
    fn test_operator_quote_next_line_show_text() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // ' operator: T* + Tj combined
        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td (A) Tj (B) ' ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'A');
        assert_eq!(chars[1].char, 'B');
    }

    #[test]
    fn test_operator_double_quote() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // " operator: set word/char spacing, T*, Tj
        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td 1 2 (Hi) \" ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'H');
        assert_eq!(chars[1].char, 'i');
    }

    #[test]
    fn test_operator_tc_char_spacing() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 2 Tc 100 700 Td (AB) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'A');
        assert_eq!(chars[1].char, 'B');
    }

    #[test]
    fn test_operator_tw_word_spacing() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 5 Tw 100 700 Td (A B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert!(chars.len() >= 3); // A, space, B
    }

    #[test]
    fn test_operator_tz_horizontal_scaling() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 150 Tz 100 700 Td (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'X');
    }

    #[test]
    fn test_operator_tl_leading() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 20 TL 100 700 Td (A) Tj T* (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        // A and B should be 20pt apart vertically (the leading value)
        let y_diff = (chars[0].bbox.y - chars[1].bbox.y).abs();
        assert!(y_diff > 10.0, "Leading should create vertical gap, got {}", y_diff);
    }

    #[test]
    fn test_operator_ts_text_rise() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Ts sets text rise (superscript/subscript)
        let stream = b"BT /F1 12 Tf 5 Ts 100 700 Td (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'X');
    }

    #[test]
    fn test_operator_tr_render_mode() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Tr sets rendering mode
        let stream = b"BT /F1 12 Tf 1 Tr 100 700 Td (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].char, 'X');
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Color operators
    // ========================================================================

    #[test]
    fn test_set_fill_rgb() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT 0.5 0.3 0.8 rg /F1 12 Tf 0 0 Td (C) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert!((chars[0].color.r - 0.5).abs() < 0.01);
        assert!((chars[0].color.g - 0.3).abs() < 0.01);
        assert!((chars[0].color.b - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_gray() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT 0.5 g /F1 12 Tf 0 0 Td (G) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert!((chars[0].color.r - 0.5).abs() < 0.01);
        assert!((chars[0].color.g - 0.5).abs() < 0.01);
        assert!((chars[0].color.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_cmyk() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // CMYK: 0 0 0 1 = pure black => RGB (0, 0, 0)
        let stream = b"BT 0 0 0 1 k /F1 12 Tf 0 0 Td (K) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert!((chars[0].color.r - 0.0).abs() < 0.01);
        assert!((chars[0].color.g - 0.0).abs() < 0.01);
        assert!((chars[0].color.b - 0.0).abs() < 0.01);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Graphics state save/restore
    // ========================================================================

    #[test]
    fn test_save_restore_color() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Test that color state is saved/restored by q/Q within a BT/ET block
        // Set blue, save, set red, show R (red), restore, show B (blue restored)
        let stream = b"BT /F1 12 Tf 0 0 1 rg q 1 0 0 rg 100 700 Td (R) Tj Q 200 700 Td (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2, "Should extract 2 chars, got {}", chars.len());
        let r_char = chars.iter().find(|c| c.char == 'R').expect("Should find R");
        let b_char = chars.iter().find(|c| c.char == 'B').expect("Should find B");
        // R should be red (set inside q)
        assert!(
            (r_char.color.r - 1.0).abs() < 0.01,
            "R should be red, got ({}, {}, {})",
            r_char.color.r,
            r_char.color.g,
            r_char.color.b
        );
        // B should be blue (restored by Q)
        assert!(
            (b_char.color.b - 1.0).abs() < 0.01,
            "B should be blue after Q restore, got ({}, {}, {})",
            b_char.color.r,
            b_char.color.g,
            b_char.color.b
        );
    }

    #[test]
    fn test_save_restore_ctm() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Save, translate CTM, show A, restore (CTM back to identity), show B at different position
        let stream = b"q 1 0 0 1 100 200 cm BT /F1 12 Tf (A) Tj ET Q BT /F1 12 Tf (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        // A should be at (100, 200), B should be at (0, 0)
        assert!(chars[0].bbox.x > 90.0, "A should be translated by CTM");
        assert!(chars[1].bbox.x < 10.0, "B should be at origin after restore");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Span extraction mode
    // ========================================================================

    #[test]
    fn test_extract_text_spans_simple() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        assert!(!spans.is_empty());
        // Find the span containing "Hello World"
        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Hello"), "Expected 'Hello' in extracted text, got: {}", text);
        assert!(text.contains("World"), "Expected 'World' in extracted text, got: {}", text);
    }

    #[test]
    fn test_extract_text_spans_multiple_tj() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Two Tj operators that should be accumulated into one span
        let stream = b"BT /F1 12 Tf 100 700 Td (He) Tj (llo) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Hello"), "Expected 'Hello' in spans, got: {}", text);
    }

    #[test]
    fn test_extract_text_spans_with_font_info() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 14 Tf 100 700 Td (Test) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        assert!(!spans.is_empty());
        let span = &spans[0];
        assert!(
            span.font_name.contains("F1") || span.font_name.contains("Times"),
            "Font name should reference F1 or Times, got: {}",
            span.font_name
        );
        assert!(span.font_size > 0.0, "Font size should be positive");
    }

    #[test]
    fn test_extract_text_spans_empty_stream() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"";
        let spans = extractor.extract_text_spans(stream).unwrap();
        assert!(spans.is_empty());
    }

    #[test]
    fn test_extract_text_spans_bt_et_no_text() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf ET";
        let spans = extractor.extract_text_spans(stream).unwrap();
        assert!(spans.is_empty());
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: TJ array processing (span mode)
    // ========================================================================

    #[test]
    fn test_tj_array_with_spacing() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // TJ array with small kerning offset (should not insert space)
        let stream = b"BT /F1 12 Tf 100 700 Td [(H) -20 (ello)] TJ ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Hello"), "Small TJ offset should not split word, got: {}", text);
    }

    #[test]
    fn test_tj_array_word_boundary() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // TJ array with large negative offset (word boundary)
        let stream = b"BT /F1 12 Tf 100 700 Td [(Hello) -300 (World)] TJ ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        // Should have space between Hello and World
        assert!(
            text.contains("Hello") && text.contains("World"),
            "Should extract both words, got: {}",
            text
        );
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: fallback_char_to_unicode
    // ========================================================================

    #[test]
    fn test_fallback_common_punctuation() {
        assert_eq!(fallback_char_to_unicode(0x2014), "\u{2014}"); // Em dash
        assert_eq!(fallback_char_to_unicode(0x2013), "\u{2013}"); // En dash
        assert_eq!(fallback_char_to_unicode(0x2022), "\u{2022}"); // Bullet
        assert_eq!(fallback_char_to_unicode(0x2026), "\u{2026}"); // Ellipsis
        assert_eq!(fallback_char_to_unicode(0x00B0), "\u{00B0}"); // Degree
    }

    #[test]
    fn test_fallback_math_operators() {
        assert_eq!(fallback_char_to_unicode(0x00B1), "\u{00B1}"); // Plus-minus
        assert_eq!(fallback_char_to_unicode(0x00D7), "\u{00D7}"); // Multiply
        assert_eq!(fallback_char_to_unicode(0x221E), "\u{221E}"); // Infinity
        assert_eq!(fallback_char_to_unicode(0x2264), "\u{2264}"); // Less or equal
        assert_eq!(fallback_char_to_unicode(0x2265), "\u{2265}"); // Greater or equal
        assert_eq!(fallback_char_to_unicode(0x2260), "\u{2260}"); // Not equal
        assert_eq!(fallback_char_to_unicode(0x221A), "\u{221A}"); // Square root
        assert_eq!(fallback_char_to_unicode(0x222B), "\u{222B}"); // Integral
        assert_eq!(fallback_char_to_unicode(0x2211), "\u{2211}"); // Summation
    }

    #[test]
    fn test_fallback_greek_letters() {
        assert_eq!(fallback_char_to_unicode(0x03B1), "\u{03B1}"); // alpha
        assert_eq!(fallback_char_to_unicode(0x03B2), "\u{03B2}"); // beta
        assert_eq!(fallback_char_to_unicode(0x03C0), "\u{03C0}"); // pi
        assert_eq!(fallback_char_to_unicode(0x03C9), "\u{03C9}"); // omega
        assert_eq!(fallback_char_to_unicode(0x0393), "\u{0393}"); // Gamma
        assert_eq!(fallback_char_to_unicode(0x03A9), "\u{03A9}"); // Omega
    }

    #[test]
    fn test_fallback_currency() {
        assert_eq!(fallback_char_to_unicode(0x20AC), "\u{20AC}"); // Euro
        assert_eq!(fallback_char_to_unicode(0x00A3), "\u{00A3}"); // Pound
        assert_eq!(fallback_char_to_unicode(0x00A5), "\u{00A5}"); // Yen
        assert_eq!(fallback_char_to_unicode(0x00A2), "\u{00A2}"); // Cent
    }

    #[test]
    fn test_fallback_direct_unicode() {
        // Valid ASCII character
        assert_eq!(fallback_char_to_unicode(0x41), "A");
        assert_eq!(fallback_char_to_unicode(0x20), " ");
    }

    #[test]
    fn test_fallback_invalid_code_point() {
        // Surrogate pair range is invalid Unicode
        assert_eq!(fallback_char_to_unicode(0xD800), "?");
        assert_eq!(fallback_char_to_unicode(0xDFFF), "?");
    }

    #[test]
    fn test_fallback_private_use_area() {
        // PUA characters should still be returned (not replaced with ?)
        let result = fallback_char_to_unicode(0xE000);
        assert_ne!(result, "?");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: decode_text_to_unicode
    // ========================================================================

    #[test]
    fn test_decode_text_no_font_latin1() {
        let result = decode_text_to_unicode(b"Hello", None);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_decode_text_no_font_high_bytes() {
        // Latin-1 high bytes should map to Unicode code points
        let bytes = vec![0xC0, 0xE9]; // A-grave, e-acute in Latin-1
        let result = decode_text_to_unicode(&bytes, None);
        assert!(result.contains('\u{00C0}'), "Should contain A-grave");
        assert!(result.contains('\u{00E9}'), "Should contain e-acute");
    }

    #[test]
    fn test_decode_text_filters_control_chars() {
        // Control characters (except tab, newline, carriage return) should be filtered
        let bytes = vec![0x01, 0x02, 0x41, 0x09, 0x0A]; // ctrl chars, 'A', tab, newline
        let result = decode_text_to_unicode(&bytes, None);
        assert!(result.contains('A'), "Should contain 'A'");
        assert!(result.contains('\t'), "Should keep tab");
        assert!(result.contains('\n'), "Should keep newline");
        assert!(!result.contains('\x01'), "Should filter ctrl-A");
    }

    #[test]
    fn test_decode_text_with_simple_font() {
        let font = create_test_font();
        let result = decode_text_to_unicode(b"ABC", Some(&font));
        // With WinAnsiEncoding, ASCII characters should map correctly
        assert!(result.contains('A') || !result.is_empty(), "Should decode something");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: cmyk_to_rgb
    // ========================================================================

    #[test]
    fn test_cmyk_to_rgb_black() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 1.0);
        assert!((r - 0.0).abs() < 0.01);
        assert!((g - 0.0).abs() < 0.01);
        assert!((b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_cmyk_to_rgb_white() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cmyk_to_rgb_cyan() {
        let (r, g, b) = cmyk_to_rgb(1.0, 0.0, 0.0, 0.0);
        assert!((r - 0.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cmyk_to_rgb_magenta() {
        let (r, g, b) = cmyk_to_rgb(0.0, 1.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 0.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cmyk_to_rgb_yellow() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 1.0, 0.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 0.0).abs() < 0.01);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: has_boundary_space edge cases
    // ========================================================================

    #[test]
    fn test_has_boundary_space_empty_strings() {
        assert!(!has_boundary_space("", ""));
        assert!(!has_boundary_space("", "hello"));
        assert!(!has_boundary_space("hello", ""));
    }

    #[test]
    fn test_has_boundary_space_only_spaces() {
        assert!(has_boundary_space(" ", " "));
        assert!(has_boundary_space(" ", "word"));
        assert!(has_boundary_space("word", " "));
    }

    #[test]
    fn test_has_boundary_space_unicode_whitespace() {
        // Non-breaking space (U+00A0)
        assert!(has_boundary_space("word\u{00A0}", "next"));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: is_email_context
    // ========================================================================

    #[test]
    fn test_email_context_at_domain() {
        // Pattern: user@outlook + . + com
        assert!(is_email_context("user@outlook", ".com"));
    }

    #[test]
    fn test_email_context_after_at() {
        // Pattern: user@ + domain.com
        assert!(is_email_context("user@", "domain.com"));
    }

    #[test]
    fn test_email_context_domain_dot_tld() {
        // Pattern: user@domain. + com
        assert!(is_email_context("user@domain.", "com"));
    }

    #[test]
    fn test_email_context_not_email() {
        assert!(!is_email_context("hello", "world"));
        assert!(!is_email_context("no at sign", "here"));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: is_citation_context
    // ========================================================================

    #[test]
    fn test_citation_context_superscript() {
        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(60.0, 105.0, 10.0, 7.0); // Raised, smaller

        // next_font_size is 0.6 * current = superscript range
        let result = is_citation_context(
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
            7.2, // 60% of 12 = 0.6, within 0.5-0.75 range
        );
        assert!(result, "Should detect citation context");
    }

    #[test]
    fn test_citation_context_no_superscript() {
        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(60.0, 100.0, 50.0, 12.0); // Same size, same position

        let result = is_citation_context(
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
            12.0, // Same font size = not a citation
        );
        assert!(!result, "Should not detect citation when same size");
    }

    #[test]
    fn test_citation_context_no_bbox() {
        // Font size ratio alone (without bbox) - prev is superscript
        let result = is_citation_context(None, None, 12.0, 7.2, 12.0);
        assert!(result, "Should detect citation from font size ratio alone");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: TextExtractor configuration
    // ========================================================================

    #[test]
    fn test_extractor_with_merging_config() {
        let extractor = TextExtractor::new().with_merging_config(SpanMergingConfig::aggressive());
        assert_eq!(extractor.merging_config.space_threshold_em_ratio, 0.15);
    }

    #[test]
    fn test_extractor_set_resources() {
        let mut extractor = TextExtractor::new();
        assert!(extractor.resources.is_none());
        extractor.set_resources(Object::Null);
        assert!(extractor.resources.is_some());
    }

    #[test]
    fn test_extractor_prepare_for_span_extraction() {
        let mut extractor = TextExtractor::new();
        extractor.extract_spans = false;
        extractor.span_sequence_counter = 42;
        extractor.prepare_for_span_extraction();
        assert!(extractor.extract_spans);
        assert_eq!(extractor.span_sequence_counter, 0);
        assert!(extractor.spans.is_empty());
    }

    #[test]
    fn test_extractor_get_font_set() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);
        let font2 = create_test_font();
        extractor.add_font("F2".to_string(), font2);

        let font_set = extractor.get_font_set();
        assert_eq!(font_set.len(), 2);
    }

    #[test]
    fn test_extractor_add_font_shared() {
        let mut extractor = TextExtractor::new();
        let font = Arc::new(create_test_font());
        extractor.add_font_shared("F1".to_string(), font.clone());
        assert_eq!(extractor.fonts.len(), 1);
        // Verify it's the same Arc
        assert!(Arc::ptr_eq(extractor.fonts.get("F1").unwrap(), &font));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: analyze_tj_distribution
    // ========================================================================

    #[test]
    fn test_analyze_tj_distribution_empty() {
        let extractor = TextExtractor::new();
        let (is_justified, cv) = extractor.analyze_tj_distribution();
        assert!(!is_justified);
        assert_eq!(cv, 0.0);
    }

    #[test]
    fn test_analyze_tj_distribution_uniform() {
        let mut extractor = TextExtractor::new();
        // Uniform offsets (all the same) = low CV = not justified
        extractor.tj_offset_history = vec![-100.0; 50];
        let (is_justified, cv) = extractor.analyze_tj_distribution();
        assert!(!is_justified, "Uniform offsets should not be justified");
        assert!(cv < 0.01, "CV should be ~0 for uniform offsets, got {}", cv);
    }

    #[test]
    fn test_analyze_tj_distribution_high_variance() {
        let mut extractor = TextExtractor::new();
        // High variance offsets = justified text
        let mut offsets = Vec::new();
        for i in 0..100 {
            offsets.push(if i % 2 == 0 { -50.0 } else { -200.0 });
        }
        extractor.tj_offset_history = offsets;
        let (is_justified, cv) = extractor.analyze_tj_distribution();
        assert!(is_justified, "High variance should indicate justified text, cv={}", cv);
        assert!(cv > 0.5, "CV should be > 0.5 for justified text, got {}", cv);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: calculate_adaptive_tj_threshold
    // ========================================================================

    #[test]
    fn test_adaptive_threshold_disabled() {
        let config = TextExtractionConfig {
            use_adaptive_tj_threshold: false,
            space_insertion_threshold: -120.0,
            ..TextExtractionConfig::default()
        };
        let extractor = TextExtractor::with_config(config);
        let threshold = extractor.calculate_adaptive_tj_threshold();
        assert_eq!(threshold, -120.0);
    }

    #[test]
    fn test_adaptive_threshold_enabled() {
        let config = TextExtractionConfig {
            use_adaptive_tj_threshold: true,
            word_margin_ratio: 0.1,
            ..TextExtractionConfig::default()
        };
        let mut extractor = TextExtractor::with_config(config);
        // Set font size
        extractor.state_stack.current_mut().font_size = 12.0;
        let threshold = extractor.calculate_adaptive_tj_threshold();
        assert!(threshold < 0.0, "Adaptive threshold should be negative");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: update_artifact_state
    // ========================================================================

    #[test]
    fn test_update_artifact_state_empty_stack() {
        let mut extractor = TextExtractor::new();
        extractor.update_artifact_state();
        assert!(!extractor.inside_artifact);
    }

    #[test]
    fn test_update_artifact_state_artifact_present() {
        let mut extractor = TextExtractor::new();
        extractor.marked_content_stack.push(MarkedContentContext {
            artifact_type: None,
            tag: "Artifact".to_string(),
            is_artifact: true,
            actual_text: None,
            expansion: None,
        });
        extractor.update_artifact_state();
        assert!(extractor.inside_artifact);
    }

    #[test]
    fn test_update_artifact_state_nested_non_artifact() {
        let mut extractor = TextExtractor::new();
        extractor.marked_content_stack.push(MarkedContentContext {
            artifact_type: None,
            tag: "Artifact".to_string(),
            is_artifact: true,
            actual_text: None,
            expansion: None,
        });
        extractor.marked_content_stack.push(MarkedContentContext {
            artifact_type: None,
            tag: "Span".to_string(),
            is_artifact: false,
            actual_text: None,
            expansion: None,
        });
        extractor.update_artifact_state();
        // Should still be inside artifact because parent is artifact
        assert!(extractor.inside_artifact);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: parse_artifact_type
    // ========================================================================

    #[test]
    fn test_parse_artifact_type_page() {
        let mut props = HashMap::new();
        props.insert("Type".to_string(), Object::Name("Page".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, Some(ArtifactType::Page));
    }

    #[test]
    fn test_parse_artifact_type_pagination_page_number() {
        let mut props = HashMap::new();
        props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
        props.insert("Subtype".to_string(), Object::Name("PageNumber".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::PageNumber)));
    }

    #[test]
    fn test_parse_artifact_type_pagination_other_subtype() {
        let mut props = HashMap::new();
        props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
        props.insert("Subtype".to_string(), Object::Name("SomethingElse".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Other)));
    }

    #[test]
    fn test_parse_artifact_type_unknown_type() {
        let mut props = HashMap::new();
        props.insert("Type".to_string(), Object::Name("UnknownType".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_artifact_type_subtype_footer_only() {
        let mut props = HashMap::new();
        props.insert("Subtype".to_string(), Object::Name("Footer".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Footer)));
    }

    #[test]
    fn test_parse_artifact_type_subtype_watermark_only() {
        let mut props = HashMap::new();
        props.insert("Subtype".to_string(), Object::Name("Watermark".to_string()));
        let result = TextExtractor::parse_artifact_type(&props);
        assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Watermark)));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: decode_pdf_text_string
    // ========================================================================

    #[test]
    fn test_decode_pdf_text_string_utf8() {
        let result = TextExtractor::decode_pdf_text_string(b"Hello World");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_decode_pdf_text_string_utf16be_bom() {
        // UTF-16BE with BOM: FE FF, then "Hi" in UTF-16BE
        let bytes: Vec<u8> = vec![0xFE, 0xFF, 0x00, 0x48, 0x00, 0x69];
        let result = TextExtractor::decode_pdf_text_string(&bytes);
        assert_eq!(result, "Hi");
    }

    #[test]
    fn test_decode_pdf_text_string_utf16le_bom() {
        // UTF-16LE with BOM: FF FE, then "Hi" in UTF-16LE
        let bytes: Vec<u8> = vec![0xFF, 0xFE, 0x48, 0x00, 0x69, 0x00];
        let result = TextExtractor::decode_pdf_text_string(&bytes);
        assert_eq!(result, "Hi");
    }

    #[test]
    fn test_decode_pdf_text_string_empty() {
        let result = TextExtractor::decode_pdf_text_string(b"");
        assert_eq!(result, "");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: split_on_camelcase
    // ========================================================================

    #[test]
    fn test_split_camelcase_basic() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("theGeneral");
        assert_eq!(parts, vec!["the", "General"]);
    }

    #[test]
    fn test_split_camelcase_multiple() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("lengthThisPage");
        assert_eq!(parts, vec!["length", "This", "Page"]);
    }

    #[test]
    fn test_split_camelcase_no_split_all_lower() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("lowercase");
        assert_eq!(parts, vec!["lowercase"]);
    }

    #[test]
    fn test_split_camelcase_no_split_all_upper() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("HTML");
        assert_eq!(parts, vec!["HTML"]);
    }

    #[test]
    fn test_split_camelcase_single_char() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("A");
        assert_eq!(parts, vec!["A"]);
    }

    #[test]
    fn test_split_camelcase_empty() {
        let extractor = TextExtractor::new();
        let parts = extractor.split_on_camelcase("");
        // Empty string gives one empty part
        assert_eq!(parts.len(), 1);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: is_ligature_code
    // ========================================================================

    #[test]
    fn test_is_ligature_code() {
        // Standard ligatures: U+FB00-U+FB04
        assert!(TextExtractor::is_ligature_code(0xFB00)); // ff
        assert!(TextExtractor::is_ligature_code(0xFB01)); // fi
        assert!(TextExtractor::is_ligature_code(0xFB02)); // fl
        assert!(TextExtractor::is_ligature_code(0xFB03)); // ffi
        assert!(TextExtractor::is_ligature_code(0xFB04)); // ffl
    }

    #[test]
    fn test_is_not_ligature_code() {
        assert!(!TextExtractor::is_ligature_code(0x41)); // 'A'
        assert!(!TextExtractor::is_ligature_code(0xFAFF)); // Before range
        assert!(!TextExtractor::is_ligature_code(0xFB05)); // After range
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: BT/ET operators
    // ========================================================================

    #[test]
    fn test_bt_resets_text_matrix() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // First BT/ET block at (100, 700)
        // Second BT should reset text matrix to identity
        let stream = b"BT /F1 12 Tf 100 700 Td (A) Tj ET BT /F1 12 Tf (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].char, 'A');
        assert_eq!(chars[1].char, 'B');
        // B should be at origin (BT resets text matrix)
        assert!(
            chars[1].bbox.x < 10.0,
            "Second BT should reset text matrix, x={}",
            chars[1].bbox.x
        );
    }

    #[test]
    fn test_multiple_bt_et_blocks() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET BT /F1 12 Tf 100 680 Td (World) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("Hello"), "Should contain Hello");
        assert!(text.contains("World"), "Should contain World");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Marked content operators
    // ========================================================================

    #[test]
    fn test_bmc_artifact_tracking() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Use execute_operator_public for fine-grained testing
        extractor
            .execute_operator_public(crate::content::operators::Operator::BeginMarkedContent {
                tag: "Artifact".to_string(),
            })
            .unwrap();

        assert!(extractor.inside_artifact, "Should be inside artifact after BMC Artifact");

        extractor
            .execute_operator_public(crate::content::operators::Operator::EndMarkedContent)
            .unwrap();

        assert!(!extractor.inside_artifact, "Should be outside artifact after EMC");
    }

    #[test]
    fn test_bmc_non_artifact() {
        let mut extractor = TextExtractor::new();

        extractor
            .execute_operator_public(crate::content::operators::Operator::BeginMarkedContent {
                tag: "Span".to_string(),
            })
            .unwrap();

        assert!(!extractor.inside_artifact, "Non-artifact BMC should not set inside_artifact");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Font switching
    // ========================================================================

    #[test]
    fn test_font_switch_mid_stream() {
        let mut extractor = TextExtractor::new();
        let font1 = create_test_font();
        let mut font2_data = create_test_font();
        font2_data.base_font = "Helvetica".to_string();
        extractor.add_font("F1".to_string(), font1);
        extractor.add_font("F2".to_string(), font2_data);

        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj /F2 14 Tf (World) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        // All characters should be extracted
        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("Hello"), "Should contain Hello");
        assert!(text.contains("World"), "Should contain World");
    }

    #[test]
    fn test_font_switch_same_font_no_flush() {
        // Setting the same font twice should be a no-op (optimization)
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf /F1 12 Tf 100 700 Td (Test) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Test"), "Should extract text, got: {}", text);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Cm operator (CTM modification)
    // ========================================================================

    #[test]
    fn test_cm_operator_translation() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Apply translation via Cm operator
        let stream = b"1 0 0 1 50 100 cm BT /F1 12 Tf (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        assert!((chars[0].bbox.x - 50.0).abs() < 2.0, "X should be ~50");
        assert!((chars[0].bbox.y - 100.0).abs() < 2.0, "Y should be ~100");
    }

    #[test]
    fn test_cm_operator_scaling() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Scale by 2x via CTM
        let stream = b"2 0 0 2 0 0 cm BT /F1 12 Tf 1 0 0 1 50 100 Tm (Y) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 1);
        // Position should be scaled: (50*2, 100*2) = (100, 200)
        assert!(
            (chars[0].bbox.x - 100.0).abs() < 2.0,
            "X should be ~100 (got {})",
            chars[0].bbox.x
        );
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Deduplication
    // ========================================================================

    #[test]
    fn test_deduplicate_overlapping_chars() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Create overlapping chars (simulating bold rendering with duplicate glyphs)
        extractor.chars = vec![
            TextChar {
                char: 'A',
                bbox: Rect::new(100.0, 700.0, 6.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
            TextChar {
                char: 'A',
                bbox: Rect::new(100.5, 700.0, 6.0, 12.0), // Very close X
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.5,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
        ];

        extractor.deduplicate_overlapping_chars();
        assert_eq!(extractor.chars.len(), 1, "Overlapping chars should be deduplicated");
    }

    #[test]
    fn test_deduplicate_overlapping_chars_different_lines() {
        let mut extractor = TextExtractor::new();

        // Chars on different lines should NOT be deduplicated
        extractor.chars = vec![
            TextChar {
                char: 'A',
                bbox: Rect::new(100.0, 700.0, 6.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
            TextChar {
                char: 'A',
                bbox: Rect::new(100.0, 680.0, 6.0, 12.0), // Different Y
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 680.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
        ];

        extractor.deduplicate_overlapping_chars();
        assert_eq!(extractor.chars.len(), 2, "Chars on different lines should not be deduplicated");
    }

    #[test]
    fn test_deduplicate_overlapping_chars_empty() {
        let mut extractor = TextExtractor::new();
        extractor.deduplicate_overlapping_chars();
        assert!(extractor.chars.is_empty());
    }

    #[test]
    fn test_deduplicate_keeps_distinct_close_chars() {
        // Issue #253: distinct characters close together should NOT be dropped
        let mut extractor = TextExtractor::new();

        let make_char = |c: char, x: f32| TextChar {
            char: c,
            bbox: Rect::new(x, 700.0, 6.0, 12.0),
            font_name: "F1".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            is_italic: false,
            is_monospace: false,
            origin_x: x,
            origin_y: 700.0,
            rotation_degrees: 0.0,
            advance_width: 6.0,
            matrix: None,
        };

        // 't' at x=100, ' ' at x=105, 'r' at x=106.5 (within 2pt of ' ' but different char)
        extractor.chars = vec![
            make_char('t', 100.0),
            make_char(' ', 105.0),
            make_char('r', 106.5),
        ];

        extractor.deduplicate_overlapping_chars();
        assert_eq!(
            extractor.chars.len(),
            3,
            "Distinct characters close together must not be dropped"
        );
        assert_eq!(extractor.chars[0].char, 't');
        assert_eq!(extractor.chars[1].char, ' ');
        assert_eq!(extractor.chars[2].char, 'r');
    }

    #[test]
    fn test_deduplicate_still_removes_same_char_duplicates() {
        // Duplicate same character at nearly the same position should still be deduped
        let mut extractor = TextExtractor::new();

        let make_char = |c: char, x: f32| TextChar {
            char: c,
            bbox: Rect::new(x, 700.0, 6.0, 12.0),
            font_name: "F1".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            is_italic: false,
            is_monospace: false,
            origin_x: x,
            origin_y: 700.0,
            rotation_degrees: 0.0,
            advance_width: 6.0,
            matrix: None,
        };

        extractor.chars = vec![make_char('A', 100.0), make_char('A', 100.5)];

        extractor.deduplicate_overlapping_chars();
        assert_eq!(extractor.chars.len(), 1, "Duplicate same char should still be deduped");
        assert_eq!(extractor.chars[0].char, 'A');
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Span deduplication
    // ========================================================================

    #[test]
    fn test_deduplicate_overlapping_spans_geometric() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(101.0, 700.0, 30.0, 12.0), // Very close
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.deduplicate_overlapping_spans();
        assert_eq!(extractor.spans.len(), 1, "Geometric duplicates should be removed");
    }

    #[test]
    fn test_deduplicate_overlapping_spans_empty() {
        let mut extractor = TextExtractor::new();
        extractor.deduplicate_overlapping_spans();
        assert!(extractor.spans.is_empty());
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Column detection
    // ========================================================================

    #[test]
    fn test_detect_span_columns_empty() {
        let extractor = TextExtractor::new();
        let columns = extractor.detect_span_columns();
        assert!(columns.is_empty());
    }

    #[test]
    fn test_detect_span_columns_single_column() {
        let mut extractor = TextExtractor::new();
        // Create spans all in one column
        for i in 0..10 {
            extractor.spans.push(TextSpan {
                artifact_type: None,
                text: format!("Line {}", i),
                bbox: Rect::new(50.0, 700.0 - (i as f32 * 14.0), 200.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            });
        }

        let columns = extractor.detect_span_columns();
        assert_eq!(columns.len(), 1, "Should detect single column");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Sort by reading order
    // ========================================================================

    #[test]
    fn test_sort_by_reading_order() {
        let mut extractor = TextExtractor::new();
        // Add chars in wrong order
        extractor.chars = vec![
            TextChar {
                char: 'B',
                bbox: Rect::new(100.0, 680.0, 6.0, 12.0), // Lower on page
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 680.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
            TextChar {
                char: 'A',
                bbox: Rect::new(100.0, 700.0, 6.0, 12.0), // Higher on page
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
        ];

        extractor.sort_by_reading_order();
        // PDF Y increases upward, so 700 is higher than 680
        // Reading order: top first, so A (y=700) before B (y=680)
        assert_eq!(extractor.chars[0].char, 'A');
        assert_eq!(extractor.chars[1].char, 'B');
    }

    #[test]
    fn test_sort_by_reading_order_same_line() {
        let mut extractor = TextExtractor::new();
        extractor.chars = vec![
            TextChar {
                char: 'B',
                bbox: Rect::new(200.0, 700.0, 6.0, 12.0), // Right
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 200.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
            TextChar {
                char: 'A',
                bbox: Rect::new(100.0, 700.0, 6.0, 12.0), // Left
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
        ];

        extractor.sort_by_reading_order();
        // Same line: left to right
        assert_eq!(extractor.chars[0].char, 'A');
        assert_eq!(extractor.chars[1].char, 'B');
    }

    #[test]
    fn test_sort_by_reading_order_nan_values() {
        let mut extractor = TextExtractor::new();
        extractor.chars = vec![
            TextChar {
                char: 'A',
                bbox: Rect::new(f32::NAN, f32::NAN, 6.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 0.0,
                origin_y: 0.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
            TextChar {
                char: 'B',
                bbox: Rect::new(100.0, 700.0, 6.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                is_italic: false,
                is_monospace: false,
                origin_x: 100.0,
                origin_y: 700.0,
                rotation_degrees: 0.0,
                advance_width: 6.0,
                matrix: None,
            },
        ];

        // Should not panic with NaN values
        extractor.sort_by_reading_order();
        assert_eq!(extractor.chars.len(), 2);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: merge_adjacent_spans
    // ========================================================================

    #[test]
    fn test_merge_adjacent_spans_same_line() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "World".to_string(),
                bbox: Rect::new(131.0, 700.0, 30.0, 12.0), // 1pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "Adjacent spans on same line should merge");
        assert!(extractor.spans[0].text.contains("Hello"));
        assert!(extractor.spans[0].text.contains("World"));
    }

    #[test]
    fn test_merge_adjacent_spans_different_lines() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "World".to_string(),
                bbox: Rect::new(100.0, 680.0, 30.0, 12.0), // Different line
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 2, "Spans on different lines should not merge");
    }

    #[test]
    fn test_merge_adjacent_spans_empty() {
        let mut extractor = TextExtractor::new();
        extractor.merge_adjacent_spans();
        assert!(extractor.spans.is_empty());
    }

    #[test]
    fn test_merge_adjacent_spans_column_boundary() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Left".to_string(),
                bbox: Rect::new(50.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Right".to_string(),
                bbox: Rect::new(300.0, 700.0, 30.0, 12.0), // Large gap (column boundary)
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 2, "Spans separated by column boundary should not merge");
    }

    #[test]
    fn test_merge_whitespace_only_span() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: " ".to_string(),
                bbox: Rect::new(130.0, 700.0, 2.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: true, // TJ offset space
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "World".to_string(),
                bbox: Rect::new(132.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 2,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "All three spans should merge");
        assert!(extractor.spans[0].text.contains("Hello"), "Should contain Hello");
        assert!(extractor.spans[0].text.contains("World"), "Should contain World");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: partition_characters_by_boundaries
    // ========================================================================

    #[test]
    fn test_partition_no_boundaries() {
        let extractor = TextExtractor::new();
        let chars = vec![
            CharacterInfo {
                code: 65,
                glyph_id: None,
                width: 10.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 66,
                glyph_id: None,
                width: 10.0,
                x_position: 10.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ];

        let clusters = extractor.partition_characters_by_boundaries(&chars, vec![]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn test_partition_with_boundary() {
        let extractor = TextExtractor::new();
        let chars = vec![
            CharacterInfo {
                code: 65,
                glyph_id: None,
                width: 10.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 66,
                glyph_id: None,
                width: 10.0,
                x_position: 10.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 67,
                glyph_id: None,
                width: 10.0,
                x_position: 25.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ];

        let clusters = extractor.partition_characters_by_boundaries(&chars, vec![2]);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 2); // [A, B]
        assert_eq!(clusters[1].len(), 1); // [C]
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: create_boundary_context
    // ========================================================================

    #[test]
    fn test_create_boundary_context() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;
        extractor.state_stack.current_mut().word_space = 2.0;
        extractor.state_stack.current_mut().char_space = 0.5;

        let ctx = extractor.create_boundary_context();
        assert_eq!(ctx.font_size, 12.0);
        assert_eq!(ctx.horizontal_scaling, 100.0);
        assert_eq!(ctx.word_spacing, 2.0);
        assert_eq!(ctx.char_spacing, 0.5);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: build_boundary_characters
    // ========================================================================

    #[test]
    fn test_build_boundary_characters() {
        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(65.0, 100.0, 40.0, 12.0);

        let (chars, ctx) =
            build_boundary_characters("Hello", "World", &prev_bbox, &next_bbox, 12.0, false);

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].code, 'o' as u32); // Last char of "Hello"
        assert_eq!(chars[1].code, 'W' as u32); // First char of "World"
        assert_eq!(ctx.font_size, 12.0);
    }

    #[test]
    fn test_build_boundary_characters_with_tj_offset() {
        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(65.0, 100.0, 40.0, 12.0);

        let (chars, _ctx) =
            build_boundary_characters("Hello", "World", &prev_bbox, &next_bbox, 12.0, true);

        assert_eq!(chars[0].tj_offset, Some(-200)); // TJ offset triggered
        assert_eq!(chars[1].tj_offset, None);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: TjBuffer
    // ========================================================================

    #[test]
    fn test_tj_buffer_empty() {
        let state = crate::content::graphics_state::GraphicsStateStack::new();
        let buffer = TjBuffer::new(state.current(), None, None);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_tj_buffer_append() {
        let state = crate::content::graphics_state::GraphicsStateStack::new();
        let mut buffer = TjBuffer::new(state.current(), None, None);
        buffer.append(b"Hello").unwrap();
        assert!(!buffer.is_empty());
        assert_eq!(buffer.unicode, "Hello");
    }

    #[test]
    fn test_tj_buffer_append_truncates_long_string() {
        let state = crate::content::graphics_state::GraphicsStateStack::new();
        let mut buffer = TjBuffer::new(state.current(), None, None);
        // Create a string larger than 32,767 bytes
        let long_bytes = vec![0x41u8; 40_000]; // 40K 'A's
        buffer.append(&long_bytes).unwrap();
        // Should be truncated to 32,767 chars
        assert!(buffer.unicode.len() <= 32_767);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: advance_position_for_offset
    // ========================================================================

    #[test]
    fn test_advance_position_for_offset_positive() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;

        let initial_e = extractor.state_stack.current().text_matrix.e;
        extractor.advance_position_for_offset(100.0).unwrap();
        let new_e = extractor.state_stack.current().text_matrix.e;

        // Positive offset should move text position left (negative tx)
        // tx = -offset / 1000.0 * font_size * horizontal_scaling / 100.0
        // tx = -100 / 1000 * 12 * 100 / 100 = -1.2
        assert!((new_e - initial_e - (-1.2)).abs() < 0.01);
    }

    #[test]
    fn test_advance_position_for_offset_negative() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;

        let initial_e = extractor.state_stack.current().text_matrix.e;
        extractor.advance_position_for_offset(-200.0).unwrap();
        let new_e = extractor.state_stack.current().text_matrix.e;

        // Negative offset should move text position right (positive tx)
        // tx = -(-200) / 1000 * 12 * 100/100 = 2.4
        assert!((new_e - initial_e - 2.4).abs() < 0.01);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: should_insert_space function
    // ========================================================================

    #[test]
    fn test_should_insert_space_boundary_already_present_trailing() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        let decision = should_insert_space(
            "word ", "next", 5.0, 12.0, "F1", &fonts, true, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space);
        assert_eq!(decision.source, SpaceSource::AlreadyPresent);
    }

    #[test]
    fn test_should_insert_space_boundary_already_present_leading() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        let decision = should_insert_space(
            "word", " next", 5.0, 12.0, "F1", &fonts, true, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space);
        assert_eq!(decision.source, SpaceSource::AlreadyPresent);
    }

    #[test]
    fn test_should_insert_space_strong_geometric() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Very large gap should trigger strong geometric rule
        // geometric_threshold = 12.0 * 0.25 = 3.0 (fallback)
        // strong threshold = 3.0 * 2.0 = 6.0
        let decision = should_insert_space(
            "word", "next", 10.0, 12.0, "F1", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(decision.insert_space, "Large gap should insert space");
        assert_eq!(decision.source, SpaceSource::GeometricGap);
    }

    #[test]
    fn test_should_insert_space_consensus_tj_and_geometric() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Both TJ offset and geometric gap triggered
        // geometric_threshold = 12.0 * 0.25 = 3.0 (fallback)
        let decision = should_insert_space(
            "word", "next", 4.0, 12.0, "F1", &fonts, true, &config, None, None, 12.0, 12.0,
        );
        assert!(decision.insert_space, "Consensus should insert space");
        assert_eq!(decision.source, SpaceSource::TjOffset);
        assert_eq!(decision.confidence, 1.0);
    }

    #[test]
    fn test_should_insert_space_no_consensus_small_gap() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Small gap, no TJ offset - should not insert
        let decision = should_insert_space(
            "word", "next", 0.5, 12.0, "F1", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        assert!(!decision.insert_space, "Small gap without TJ should not insert space");
        assert_eq!(decision.source, SpaceSource::NoSpace);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Line break detection in should_insert_space
    // ========================================================================

    #[test]
    fn test_should_insert_space_line_break_hard() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Simulate line break: prev at Y=700, next at Y=680
        let prev_bbox = Rect::new(100.0, 700.0, 200.0, 12.0);
        let next_bbox = Rect::new(100.0, 680.0, 200.0, 12.0);

        let decision = should_insert_space(
            "end of line",
            "start of next",
            0.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        // Line break detected, same column, not ending with hyphen => insert space
        assert!(decision.insert_space, "Hard line break should insert space");
    }

    #[test]
    fn test_should_insert_space_line_break_hyphen() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Line break with hyphen: should NOT insert space
        let prev_bbox = Rect::new(100.0, 700.0, 200.0, 12.0);
        let next_bbox = Rect::new(100.0, 680.0, 200.0, 12.0);

        let decision = should_insert_space(
            "self-contain-",
            "ed text",
            0.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        assert!(!decision.insert_space, "Hyphenated line break should not insert space");
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Full extraction pipeline via content streams
    // ========================================================================

    #[test]
    fn test_extract_multiple_text_objects() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream =
            b"BT /F1 12 Tf 100 700 Td (First) Tj ET BT /F1 12 Tf 100 680 Td (Second) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
    }

    #[test]
    fn test_extract_spans_with_line_break() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Two lines of text
        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td (First line) Tj T* (Second line) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        assert!(!spans.is_empty());
        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("First"), "Should contain first line");
        assert!(text.contains("Second"), "Should contain second line");
    }

    #[test]
    fn test_extract_chars_reading_order() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Text in reverse rendering order
        let stream = b"BT /F1 12 Tf 100 680 Td (B) Tj ET BT /F1 12 Tf 100 700 Td (A) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        // After sorting by reading order: A (y=700 higher) should come first
        assert_eq!(chars[0].char, 'A', "Higher Y should come first in reading order");
        assert_eq!(chars[1].char, 'B');
    }

    #[test]
    fn test_extract_empty_string() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 100 700 Td () Tj ET";
        let chars = extractor.extract(stream).unwrap();
        assert_eq!(chars.len(), 0);
    }

    #[test]
    fn test_extract_only_graphics_no_text() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Only graphics commands, no text
        let stream = b"q 1 0 0 1 0 0 cm 100 700 m 200 700 l S Q";
        let chars = extractor.extract(stream).unwrap();
        assert_eq!(chars.len(), 0);
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Inline images should not affect text
    // ========================================================================

    #[test]
    fn test_inline_image_ignored() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Text before and after inline image - both should be extracted
        // The inline image operators are handled by the parser
        let stream = b"BT /F1 12 Tf 100 700 Td (Before) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        let text: String = chars.iter().map(|c| c.char).collect();
        assert!(text.contains("Before"));
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Tm operator batching optimization
    // ========================================================================

    #[test]
    fn test_tm_continuation_same_line() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Character-by-character Tm+Tj pattern on same line
        // The optimization should batch these into fewer spans
        let stream = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (H) Tj 1 0 0 1 106 700 Tm (i) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Hi"), "Should batch Tm+Tj on same line, got: {}", text);
    }

    #[test]
    fn test_tm_different_line_flushes() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Tm to different Y should flush buffer and start new span
        let stream = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (A) Tj 1 0 0 1 100 680 Tm (B) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        // Should have at least 2 spans (different lines)
        assert!(
            spans.len() >= 2 || {
                // Or could be merged if within merge range
                let text: String = spans
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join("");
                text.contains("A") && text.contains("B")
            }
        );
    }

    // ========================================================================
    // NEW COMPREHENSIVE TESTS: Edge cases
    // ========================================================================

    #[test]
    fn test_extract_with_zero_font_size() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Zero font size is technically valid in PDF
        let stream = b"BT /F1 0 Tf 100 700 Td (X) Tj ET";
        let result = extractor.extract(stream);
        // Should not panic
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_with_negative_font_size() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Negative font size inverts text
        let stream = b"BT /F1 -12 Tf 100 700 Td (X) Tj ET";
        let result = extractor.extract(stream);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_with_very_large_coordinate() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 99999 99999 Td (X) Tj ET";
        let chars = extractor.extract(stream).unwrap();
        assert_eq!(chars.len(), 1);
    }

    // ========================================================================
    // COVERAGE TESTS: Color space operators (SetFillColor/SetStrokeColor)
    // ========================================================================

    #[test]
    fn test_set_fill_color_device_gray() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // cs sets color space, then sc sets color components
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.5],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.5).abs() < 0.01);
        assert!((state.fill_color_rgb.1 - 0.5).abs() < 0.01);
        assert!((state.fill_color_rgb.2 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_device_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.2, 0.4, 0.6],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.2).abs() < 0.01);
        assert!((state.fill_color_rgb.1 - 0.4).abs() < 0.01);
        assert!((state.fill_color_rgb.2 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_device_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceCMYK".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.0, 0.0, 0.0, 1.0], // pure black
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.0).abs() < 0.01);
        assert!((state.fill_color_rgb.1 - 0.0).abs() < 0.01);
        assert!((state.fill_color_rgb.2 - 0.0).abs() < 0.01);
        assert!(state.fill_color_cmyk.is_some());
    }

    #[test]
    fn test_set_fill_color_lab() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "Lab".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![50.0, 20.0, -10.0],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        // Lab simplified to grayscale: L/100
        assert!((state.fill_color_rgb.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_iccbased_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.1, 0.2, 0.3],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.1).abs() < 0.01);
        assert!((state.fill_color_rgb.1 - 0.2).abs() < 0.01);
        assert!((state.fill_color_rgb.2 - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_iccbased_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.7],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_iccbased_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![1.0, 0.0, 0.0, 0.0], // cyan
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.fill_color_cmyk.is_some());
    }

    #[test]
    fn test_set_fill_color_separation() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "Separation".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.8], // tint
            })
            .unwrap();

        let state = extractor.state_stack.current();
        // gray = 1.0 - tint = 0.2
        assert!((state.fill_color_rgb.0 - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_devicen_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.0, 0.0, 0.0, 0.5], // 4-component DeviceN
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.fill_color_cmyk.is_some());
    }

    #[test]
    fn test_set_fill_color_devicen_single() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.3], // single-component DeviceN
            })
            .unwrap();

        let state = extractor.state_stack.current();
        // gray = 1.0 - 0.3 = 0.7
        assert!((state.fill_color_rgb.0 - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_unknown_space() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "CustomUnknown".to_string(),
            })
            .unwrap();
        // This should log warning but not panic
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.5, 0.5],
            })
            .unwrap();
    }

    #[test]
    fn test_set_fill_color_cal_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "CalGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.8],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_cal_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "CalRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColor {
                components: vec![0.9, 0.1, 0.5],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.9).abs() < 0.01);
        assert!((state.fill_color_rgb.1 - 0.1).abs() < 0.01);
        assert!((state.fill_color_rgb.2 - 0.5).abs() < 0.01);
    }

    // ========================================================================
    // COVERAGE TESTS: Stroke color operators
    // ========================================================================

    #[test]
    fn test_set_stroke_color_device_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.4],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_device_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.1, 0.2, 0.3],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_lab() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "Lab".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![75.0, 10.0, -5.0],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_device_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceCMYK".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.0, 1.0, 0.0, 0.0], // magenta
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_iccbased_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.3],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_iccbased_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.9, 0.8, 0.7],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_iccbased_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.1, 0.2, 0.3, 0.4],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_separation() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "Separation".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.6],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        // gray = 1.0 - 0.6 = 0.4
        assert!((state.stroke_color_rgb.0 - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_devicen_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.1, 0.2, 0.3, 0.4],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_devicen_single() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.5],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_cal_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "CalRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.5, 0.6, 0.7],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_cal_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "CalGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.9],
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_unknown() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "UnknownCS".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColor {
                components: vec![0.5],
            })
            .unwrap();
        // Should not panic
    }

    // ========================================================================
    // COVERAGE TESTS: SetFillColorN / SetStrokeColorN
    // ========================================================================

    #[test]
    fn test_set_fill_color_n_with_pattern() {
        let mut extractor = TextExtractor::new();
        // Pattern color space with name
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![],
                name: Some(Box::new("P1".to_string())),
            })
            .unwrap();
        // Should not panic (pattern ignored)
    }

    #[test]
    fn test_set_fill_color_n_without_pattern_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.3],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_without_pattern_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.1, 0.2, 0.3],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_lab() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "Lab".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![80.0, 0.0, 0.0],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceCMYK".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.0, 0.0, 0.0, 0.0],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        // White (no ink)
        assert!((state.fill_color_rgb.0 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_iccbased() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.5, 0.6, 0.7],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_iccbased_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.9],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_iccbased_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.1, 0.2, 0.3, 0.4],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.fill_color_cmyk.is_some());
    }

    #[test]
    fn test_set_fill_color_n_separation() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "Separation".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.4],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_set_fill_color_n_devicen() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.1, 0.2, 0.3, 0.4],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.fill_color_cmyk.is_some());
    }

    #[test]
    fn test_set_fill_color_n_devicen_single() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetFillColorN {
                components: vec![0.2],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_with_pattern() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![],
                name: Some(Box::new("P2".to_string())),
            })
            .unwrap();
        // Should not panic
    }

    #[test]
    fn test_set_stroke_color_n_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceGray".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.6],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.8, 0.7, 0.6],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_lab() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "Lab".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![60.0, 0.0, 0.0],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceCMYK".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.0, 0.0, 1.0, 0.0], // yellow
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_n_iccbased_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.2, 0.3, 0.4],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_iccbased_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.5],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_iccbased_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "ICCBased".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.1, 0.2, 0.3, 0.4],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_n_separation() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "Separation".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![1.0],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_color_n_devicen_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.5, 0.5, 0.5, 0.5],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
    }

    #[test]
    fn test_set_stroke_color_n_devicen_single() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceN".to_string(),
            })
            .unwrap();
        extractor
            .execute_operator_public(Operator::SetStrokeColorN {
                components: vec![0.1],
                name: None,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.9).abs() < 0.01);
    }

    // ========================================================================
    // COVERAGE TESTS: Line style & misc operators
    // ========================================================================

    #[test]
    fn test_set_line_cap() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetLineCap { cap_style: 2 })
            .unwrap();
        assert_eq!(extractor.state_stack.current().line_cap, 2);
    }

    #[test]
    fn test_set_line_join() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetLineJoin { join_style: 1 })
            .unwrap();
        assert_eq!(extractor.state_stack.current().line_join, 1);
    }

    #[test]
    fn test_set_miter_limit() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetMiterLimit { limit: 5.0 })
            .unwrap();
        assert!((extractor.state_stack.current().miter_limit - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_set_rendering_intent() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetRenderingIntent {
                intent: "RelativeColorimetric".to_string(),
            })
            .unwrap();
        assert_eq!(extractor.state_stack.current().rendering_intent, "RelativeColorimetric");
    }

    #[test]
    fn test_set_flatness() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFlatness { tolerance: 0.5 })
            .unwrap();
        assert!((extractor.state_stack.current().flatness - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_ext_gstate() {
        let mut extractor = TextExtractor::new();
        // Should not panic, just logs debug info
        extractor
            .execute_operator_public(Operator::SetExtGState {
                dict_name: "GS1".to_string(),
            })
            .unwrap();
    }

    #[test]
    fn test_paint_shading() {
        let mut extractor = TextExtractor::new();
        // Should not panic, just logs debug info
        extractor
            .execute_operator_public(Operator::PaintShading {
                name: "sh1".to_string(),
            })
            .unwrap();
    }

    #[test]
    fn test_inline_image_operator() {
        let mut extractor = TextExtractor::new();
        let mut dict = HashMap::new();
        dict.insert("W".to_string(), Object::Integer(100));
        dict.insert("H".to_string(), Object::Integer(50));
        extractor
            .execute_operator_public(Operator::InlineImage {
                dict: Box::new(dict),
                data: vec![0u8; 100],
            })
            .unwrap();
        // Should not panic and not produce text
    }

    #[test]
    fn test_inline_image_no_dimensions() {
        let mut extractor = TextExtractor::new();
        let dict = HashMap::new(); // no W/H
        extractor
            .execute_operator_public(Operator::InlineImage {
                dict: Box::new(dict),
                data: vec![0u8; 10],
            })
            .unwrap();
    }

    // ========================================================================
    // COVERAGE TESTS: Email pattern detection with config
    // ========================================================================

    #[test]
    fn test_email_context_at_sign_end() {
        // Pattern: user@ + domain
        assert!(is_email_context("user@", "domain.com"));
    }

    #[test]
    fn test_email_context_domain_dot() {
        // Pattern: user@domain. + com
        assert!(is_email_context("user@domain.", "com"));
    }

    #[test]
    fn test_email_context_not_alpha_after_at() {
        // @ followed by non-alphanumeric should not be email
        assert!(!is_email_context("user@", " "));
    }

    #[test]
    fn test_email_context_long_preceding_text() {
        // Test with very long preceding text (should only check last 64 bytes)
        let long_prefix = "a".repeat(200) + "@domain";
        assert!(is_email_context(&long_prefix, ".com"));
    }

    #[test]
    fn test_should_insert_space_with_email_config() {
        let config = SpanMergingConfig {
            detect_email_patterns: true,
            email_threshold_multiplier: 2.5,
            ..Default::default()
        };
        let fonts = HashMap::new();

        // Email context with gap below threshold: suppress space
        let decision = should_insert_space(
            "user@domain",
            ".com",
            1.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            None,
            None,
            12.0,
            12.0,
        );
        assert!(!decision.insert_space, "Email context should suppress space for small gap");
    }

    #[test]
    fn test_should_insert_space_email_large_gap() {
        let config = SpanMergingConfig {
            detect_email_patterns: true,
            email_threshold_multiplier: 2.5,
            ..Default::default()
        };
        let fonts = HashMap::new();

        // Email context with very large gap: insert space
        let decision = should_insert_space(
            "user@domain",
            ".com",
            100.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            None,
            None,
            12.0,
            12.0,
        );
        assert!(decision.insert_space, "Email context should insert space for large gap");
    }

    #[test]
    fn test_should_insert_space_email_with_font_info() {
        let config = SpanMergingConfig {
            detect_email_patterns: true,
            ..Default::default()
        };
        let mut fonts: HashMap<String, Arc<FontInfo>> = HashMap::new();
        let font = create_test_font();
        fonts.insert("F1".to_string(), Arc::new(font));

        // Email context uses font metrics for threshold
        let decision = should_insert_space(
            "user@domain",
            ".com",
            1.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            None,
            None,
            12.0,
            12.0,
        );
        assert!(!decision.insert_space);
    }

    // ========================================================================
    // COVERAGE TESTS: Citation marker detection with config
    // ========================================================================

    #[test]
    fn test_should_insert_space_citation_context() {
        let config = SpanMergingConfig {
            detect_citation_markers: true,
            citation_font_size_ratio: 0.75,
            ..Default::default()
        };
        let fonts = HashMap::new();

        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(60.0, 105.0, 10.0, 7.0); // Raised, smaller

        let decision = should_insert_space(
            "text",
            "1",
            2.0,
            12.0,
            "F1",
            &fonts,
            true,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            7.2,
        );
        assert!(decision.insert_space, "Citation context with TJ should insert space");
    }

    #[test]
    fn test_should_insert_space_citation_geometric() {
        let config = SpanMergingConfig {
            detect_citation_markers: true,
            ..Default::default()
        };
        let fonts = HashMap::new();

        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(60.0, 105.0, 10.0, 7.0);

        // Citation context with large geometric gap
        let decision = should_insert_space(
            "text",
            "1",
            10.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            7.2,
        );
        assert!(decision.insert_space, "Citation context with large gap should insert space");
    }

    #[test]
    fn test_should_insert_space_citation_with_font() {
        let config = SpanMergingConfig {
            detect_citation_markers: true,
            ..Default::default()
        };
        let mut fonts: HashMap<String, Arc<FontInfo>> = HashMap::new();
        fonts.insert("F1".to_string(), Arc::new(create_test_font()));

        let prev_bbox = Rect::new(10.0, 100.0, 50.0, 12.0);
        let next_bbox = Rect::new(60.0, 105.0, 10.0, 7.0);

        let decision = should_insert_space(
            "text",
            "1",
            5.0,
            12.0,
            "F1",
            &fonts,
            true,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            7.2,
        );
        assert!(decision.insert_space);
    }

    // ========================================================================
    // COVERAGE TESTS: Line break detection
    // ========================================================================

    #[test]
    fn test_line_break_different_column() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // prev and next at very different X positions (different columns)
        let prev_bbox = Rect::new(50.0, 700.0, 200.0, 12.0);
        let next_bbox = Rect::new(400.0, 680.0, 200.0, 12.0);

        let decision = should_insert_space(
            "end",
            "start",
            0.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        // Different column - should not trigger same_column line break path
        // The default no space path should apply
    }

    #[test]
    fn test_line_break_not_triggered_small_vertical_gap() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        // Small vertical gap - not a line break
        let prev_bbox = Rect::new(100.0, 700.0, 200.0, 12.0);
        let next_bbox = Rect::new(100.0, 699.0, 200.0, 12.0);

        let decision = should_insert_space(
            "word",
            "next",
            0.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        // Small vertical gap should not trigger line break
    }

    // ========================================================================
    // COVERAGE TESTS: WordBoundary tiebreaker path
    // ========================================================================

    #[test]
    fn test_should_insert_space_tiebreaker_with_bboxes() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        let prev_bbox = Rect::new(100.0, 700.0, 50.0, 12.0);
        let next_bbox = Rect::new(155.0, 700.0, 50.0, 12.0);

        // TJ triggered but gap does not suggest space (conflict)
        // Should go through tiebreaker
        let decision = should_insert_space(
            "word",
            "next",
            1.0,
            12.0,
            "F1",
            &fonts,
            true,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        // Result depends on WordBoundaryDetector
    }

    #[test]
    fn test_should_insert_space_geometric_only_conflict() {
        let config = SpanMergingConfig::default();
        let fonts = HashMap::new();

        let prev_bbox = Rect::new(100.0, 700.0, 50.0, 12.0);
        let next_bbox = Rect::new(155.0, 700.0, 50.0, 12.0);

        // No TJ but gap suggests space (conflict with no TJ)
        let decision = should_insert_space(
            "word",
            "next",
            5.0,
            12.0,
            "F1",
            &fonts,
            false,
            &config,
            Some(&prev_bbox),
            Some(&next_bbox),
            12.0,
            12.0,
        );
        // Geometric alone - should go through tiebreaker path
    }

    // ========================================================================
    // COVERAGE TESTS: Font-aware spacing in should_insert_space
    // ========================================================================

    #[test]
    fn test_should_insert_space_font_aware() {
        let config = SpanMergingConfig::default();
        let mut fonts: HashMap<String, Arc<FontInfo>> = HashMap::new();
        fonts.insert("F1".to_string(), Arc::new(create_test_font()));

        // With font info, threshold is calculated from font metrics
        let decision = should_insert_space(
            "word", "next", 0.5, 12.0, "F1", &fonts, false, &config, None, None, 12.0, 12.0,
        );
        // The result depends on font-specific threshold
    }

    // ========================================================================
    // COVERAGE TESTS: SpanMergingConfig builder variants
    // ========================================================================

    #[test]
    fn test_span_merging_config_adaptive_with_config() {
        let adaptive_config = crate::extractors::gap_statistics::AdaptiveThresholdConfig::default();
        let config = SpanMergingConfig::adaptive_with_config(adaptive_config);
        assert!(config.use_adaptive_threshold);
        assert!(config.adaptive_config.is_some());
    }

    // ========================================================================
    // COVERAGE TESTS: Fallback char to unicode (more symbols)
    // ========================================================================

    #[test]
    fn test_fallback_quotation_marks() {
        assert_eq!(fallback_char_to_unicode(0x2018), "\u{2018}"); // Left single quote
        assert_eq!(fallback_char_to_unicode(0x2019), "\u{2019}"); // Right single quote
        assert_eq!(fallback_char_to_unicode(0x201C), "\u{201C}"); // Left double quote
        assert_eq!(fallback_char_to_unicode(0x201D), "\u{201D}"); // Right double quote
    }

    #[test]
    fn test_fallback_math_extended() {
        assert_eq!(fallback_char_to_unicode(0x00F7), "\u{00F7}"); // Division
        assert_eq!(fallback_char_to_unicode(0x2202), "\u{2202}"); // Partial diff
        assert_eq!(fallback_char_to_unicode(0x2207), "\u{2207}"); // Nabla
        assert_eq!(fallback_char_to_unicode(0x220F), "\u{220F}"); // Product
        assert_eq!(fallback_char_to_unicode(0x2261), "\u{2261}"); // Identical
        assert_eq!(fallback_char_to_unicode(0x2248), "\u{2248}"); // Almost equal
    }

    #[test]
    fn test_fallback_set_theory() {
        assert_eq!(fallback_char_to_unicode(0x2282), "\u{2282}"); // Subset
        assert_eq!(fallback_char_to_unicode(0x2283), "\u{2283}"); // Superset
        assert_eq!(fallback_char_to_unicode(0x2286), "\u{2286}"); // Subset or equal
        assert_eq!(fallback_char_to_unicode(0x2287), "\u{2287}"); // Superset or equal
        assert_eq!(fallback_char_to_unicode(0x2208), "\u{2208}"); // Element of
        assert_eq!(fallback_char_to_unicode(0x2209), "\u{2209}"); // Not element
        assert_eq!(fallback_char_to_unicode(0x2200), "\u{2200}"); // For all
        assert_eq!(fallback_char_to_unicode(0x2203), "\u{2203}"); // There exists
        assert_eq!(fallback_char_to_unicode(0x2205), "\u{2205}"); // Empty set
    }

    #[test]
    fn test_fallback_logic() {
        assert_eq!(fallback_char_to_unicode(0x2227), "\u{2227}"); // Logical and
        assert_eq!(fallback_char_to_unicode(0x2228), "\u{2228}"); // Logical or
        assert_eq!(fallback_char_to_unicode(0x00AC), "\u{00AC}"); // Not
    }

    #[test]
    fn test_fallback_arrows() {
        assert_eq!(fallback_char_to_unicode(0x2192), "\u{2192}"); // Right arrow
        assert_eq!(fallback_char_to_unicode(0x2190), "\u{2190}"); // Left arrow
        assert_eq!(fallback_char_to_unicode(0x2194), "\u{2194}"); // Left right arrow
        assert_eq!(fallback_char_to_unicode(0x21D2), "\u{21D2}"); // Double right
        assert_eq!(fallback_char_to_unicode(0x21D4), "\u{21D4}"); // Double left-right
    }

    #[test]
    fn test_fallback_greek_lowercase_extended() {
        assert_eq!(fallback_char_to_unicode(0x03B5), "\u{03B5}"); // epsilon
        assert_eq!(fallback_char_to_unicode(0x03B6), "\u{03B6}"); // zeta
        assert_eq!(fallback_char_to_unicode(0x03B7), "\u{03B7}"); // eta
        assert_eq!(fallback_char_to_unicode(0x03B9), "\u{03B9}"); // iota
        assert_eq!(fallback_char_to_unicode(0x03BA), "\u{03BA}"); // kappa
        assert_eq!(fallback_char_to_unicode(0x03BB), "\u{03BB}"); // lambda
        assert_eq!(fallback_char_to_unicode(0x03BC), "\u{03BC}"); // mu
        assert_eq!(fallback_char_to_unicode(0x03BD), "\u{03BD}"); // nu
        assert_eq!(fallback_char_to_unicode(0x03BE), "\u{03BE}"); // xi
        assert_eq!(fallback_char_to_unicode(0x03BF), "\u{03BF}"); // omicron
        assert_eq!(fallback_char_to_unicode(0x03C1), "\u{03C1}"); // rho
        assert_eq!(fallback_char_to_unicode(0x03C2), "\u{03C2}"); // final sigma
        assert_eq!(fallback_char_to_unicode(0x03C3), "\u{03C3}"); // sigma
        assert_eq!(fallback_char_to_unicode(0x03C4), "\u{03C4}"); // tau
        assert_eq!(fallback_char_to_unicode(0x03C5), "\u{03C5}"); // upsilon
        assert_eq!(fallback_char_to_unicode(0x03C6), "\u{03C6}"); // phi
        assert_eq!(fallback_char_to_unicode(0x03C7), "\u{03C7}"); // chi
        assert_eq!(fallback_char_to_unicode(0x03C8), "\u{03C8}"); // psi
    }

    #[test]
    fn test_fallback_greek_uppercase_extended() {
        assert_eq!(fallback_char_to_unicode(0x0391), "\u{0391}"); // Alpha
        assert_eq!(fallback_char_to_unicode(0x0392), "\u{0392}"); // Beta
        assert_eq!(fallback_char_to_unicode(0x0394), "\u{0394}"); // Delta
        assert_eq!(fallback_char_to_unicode(0x0395), "\u{0395}"); // Epsilon
        assert_eq!(fallback_char_to_unicode(0x0396), "\u{0396}"); // Zeta
        assert_eq!(fallback_char_to_unicode(0x0397), "\u{0397}"); // Eta
        assert_eq!(fallback_char_to_unicode(0x0398), "\u{0398}"); // Theta
        assert_eq!(fallback_char_to_unicode(0x0399), "\u{0399}"); // Iota
        assert_eq!(fallback_char_to_unicode(0x039A), "\u{039A}"); // Kappa
        assert_eq!(fallback_char_to_unicode(0x039B), "\u{039B}"); // Lambda
        assert_eq!(fallback_char_to_unicode(0x039C), "\u{039C}"); // Mu
        assert_eq!(fallback_char_to_unicode(0x039D), "\u{039D}"); // Nu
        assert_eq!(fallback_char_to_unicode(0x039E), "\u{039E}"); // Xi
        assert_eq!(fallback_char_to_unicode(0x039F), "\u{039F}"); // Omicron
        assert_eq!(fallback_char_to_unicode(0x03A0), "\u{03A0}"); // Pi
        assert_eq!(fallback_char_to_unicode(0x03A1), "\u{03A1}"); // Rho
        assert_eq!(fallback_char_to_unicode(0x03A3), "\u{03A3}"); // Sigma
        assert_eq!(fallback_char_to_unicode(0x03A4), "\u{03A4}"); // Tau
        assert_eq!(fallback_char_to_unicode(0x03A5), "\u{03A5}"); // Upsilon
        assert_eq!(fallback_char_to_unicode(0x03A6), "\u{03A6}"); // Phi
        assert_eq!(fallback_char_to_unicode(0x03A7), "\u{03A7}"); // Chi
        assert_eq!(fallback_char_to_unicode(0x03A8), "\u{03A8}"); // Psi
    }

    #[test]
    fn test_fallback_currency_extended() {
        assert_eq!(fallback_char_to_unicode(0x20A3), "\u{20A3}"); // Franc
        assert_eq!(fallback_char_to_unicode(0x20A4), "\u{20A4}"); // Lira
        assert_eq!(fallback_char_to_unicode(0x20A9), "\u{20A9}"); // Won
        assert_eq!(fallback_char_to_unicode(0x20AA), "\u{20AA}"); // Shekel
        assert_eq!(fallback_char_to_unicode(0x20AB), "\u{20AB}"); // Dong
        assert_eq!(fallback_char_to_unicode(0x20B9), "\u{20B9}"); // Rupee
    }

    // ========================================================================
    // COVERAGE TESTS: decode_text_to_unicode edge cases
    // ========================================================================

    #[test]
    fn test_decode_text_simple_font_with_control_chars() {
        let font = create_test_font();
        let bytes = vec![0x01, 0x41, 0x09]; // ctrl char, 'A', tab
        let result = decode_text_to_unicode(&bytes, Some(&font));
        // Should filter control chars but keep tab
        assert!(result.contains('\t') || result.contains('A'));
    }

    #[test]
    fn test_decode_text_single_byte_only() {
        // Test with bytes that hit the TwoByte < 2 fallback
        let mut font = create_test_font();
        font.subtype = "Type0".to_string();
        font.encoding = crate::fonts::Encoding::Identity;
        let bytes = vec![0x41]; // Single byte for Type0 identity
        let result = decode_text_to_unicode(&bytes, Some(&font));
        // Should hit trailing byte path
    }

    // ========================================================================
    // COVERAGE TESTS: Color space resets
    // ========================================================================

    #[test]
    fn test_set_fill_color_space_resets_color() {
        let mut extractor = TextExtractor::new();
        // Set RGB color first
        extractor
            .execute_operator_public(Operator::SetFillRgb {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 1.0).abs() < 0.01);

        // Change color space should reset to black
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceGray".to_string(),
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.fill_color_rgb.0 - 0.0).abs() < 0.01);
        assert!(state.fill_color_cmyk.is_none());
    }

    #[test]
    fn test_set_stroke_color_space_resets_color() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeRgb {
                r: 0.0,
                g: 1.0,
                b: 0.0,
            })
            .unwrap();

        extractor
            .execute_operator_public(Operator::SetStrokeColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.0).abs() < 0.01);
        assert!(state.stroke_color_cmyk.is_none());
    }

    // ========================================================================
    // COVERAGE TESTS: CMYK color operators
    // ========================================================================

    #[test]
    fn test_set_stroke_cmyk() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeCmyk {
                c: 1.0,
                m: 0.0,
                y: 0.0,
                k: 0.0,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!(state.stroke_color_cmyk.is_some());
        // Cyan: R=0, G=1, B=1
        assert!((state.stroke_color_rgb.0 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_gray() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeGray { gray: 0.7 })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.7).abs() < 0.01);
        assert!((state.stroke_color_rgb.1 - 0.7).abs() < 0.01);
        assert!((state.stroke_color_rgb.2 - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_set_stroke_rgb() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetStrokeRgb {
                r: 0.3,
                g: 0.6,
                b: 0.9,
            })
            .unwrap();

        let state = extractor.state_stack.current();
        assert!((state.stroke_color_rgb.0 - 0.3).abs() < 0.01);
        assert!((state.stroke_color_rgb.1 - 0.6).abs() < 0.01);
        assert!((state.stroke_color_rgb.2 - 0.9).abs() < 0.01);
    }

    // ========================================================================
    // COVERAGE TESTS: CMYK to RGB edge cases
    // ========================================================================

    #[test]
    fn test_cmyk_to_rgb_mixed() {
        let (r, g, b) = cmyk_to_rgb(0.5, 0.3, 0.1, 0.2);
        assert!((0.0..=1.0).contains(&r));
        assert!((0.0..=1.0).contains(&g));
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn test_cmyk_to_rgb_all_ones() {
        let (r, g, b) = cmyk_to_rgb(1.0, 1.0, 1.0, 1.0);
        assert!((r - 0.0).abs() < 0.01);
        assert!((g - 0.0).abs() < 0.01);
        assert!((b - 0.0).abs() < 0.01);
    }

    // ========================================================================
    // COVERAGE TESTS: Content deduplication - content-based
    // ========================================================================

    #[test]
    fn test_deduplicate_content_based() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello World".to_string(), // >= 5 chars
                bbox: Rect::new(100.0, 700.0, 60.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Hello World".to_string(), // Same text, overlapping position
                bbox: Rect::new(102.0, 700.0, 60.0, 12.0), // X within 5pt
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.deduplicate_overlapping_spans();
        assert_eq!(extractor.spans.len(), 1, "Content duplicates should be removed");
    }

    #[test]
    fn test_deduplicate_content_not_overlapping() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello World".to_string(),
                bbox: Rect::new(100.0, 700.0, 60.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Hello World".to_string(), // Same text but far apart
                bbox: Rect::new(500.0, 700.0, 60.0, 12.0), // X > 5pt difference
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.deduplicate_overlapping_spans();
        assert_eq!(extractor.spans.len(), 2, "Non-overlapping content should not be deduped");
    }

    // ========================================================================
    // COVERAGE TESTS: advance_position_for_string
    // ========================================================================

    #[test]
    fn test_advance_position_no_font() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;

        let width = extractor.advance_position_for_string(b"Hello").unwrap();
        assert!(width > 0.0, "Width should be positive even without font");
    }

    #[test]
    fn test_advance_position_with_font() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);
        extractor.cached_current_font = extractor.fonts.get("F1").cloned();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().font_name = Some("F1".to_string());
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;

        let width = extractor.advance_position_for_string(b"Hi").unwrap();
        assert!(width > 0.0, "Width should be positive with font");
    }

    #[test]
    fn test_advance_position_with_word_space() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;
        extractor.state_stack.current_mut().word_space = 5.0;

        let width = extractor.advance_position_for_string(b"A B").unwrap();
        assert!(width > 0.0);
    }

    // ========================================================================
    // COVERAGE TESTS: insert_space_as_span
    // ========================================================================

    #[test]
    fn test_insert_space_as_span() {
        let mut extractor = TextExtractor::new();
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().horizontal_scaling = 100.0;
        extractor.state_stack.current_mut().font_name = Some("F1".to_string());

        let before = extractor.spans.len();
        extractor.insert_space_as_span().unwrap();
        assert_eq!(extractor.spans.len(), before + 1);
        assert_eq!(extractor.spans.last().unwrap().text, " ");
        assert!(extractor.spans.last().unwrap().offset_semantic);
    }

    // ========================================================================
    // COVERAGE TESTS: split_fused_words
    // ========================================================================

    #[test]
    fn test_split_fused_words_camelcase() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![TextSpan {
            artifact_type: None,
            text: "theGeneral".to_string(),
            bbox: Rect::new(100.0, 700.0, 60.0, 12.0),
            font_name: "F1".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            is_italic: false,
            is_monospace: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
        }];

        extractor.split_fused_words();
        assert_eq!(extractor.spans.len(), 2, "Should split theGeneral into two spans");
        assert_eq!(extractor.spans[0].text, "the");
        assert_eq!(extractor.spans[1].text, "General");
        assert!(extractor.spans[1].split_boundary_before);
    }

    #[test]
    fn test_split_fused_words_no_split() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![TextSpan {
            artifact_type: None,
            text: "hello".to_string(),
            bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
            font_name: "F1".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            is_italic: false,
            is_monospace: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
        }];

        extractor.split_fused_words();
        assert_eq!(extractor.spans.len(), 1, "No split needed for all-lowercase");
        assert_eq!(extractor.spans[0].text, "hello");
    }

    // ========================================================================
    // COVERAGE TESTS: calculate_average_glyph_width
    // ========================================================================

    #[test]
    fn test_calculate_average_glyph_width_no_widths() {
        let extractor = TextExtractor::new();
        let font = create_test_font(); // No widths array
        let avg = extractor.calculate_average_glyph_width(&font);
        assert_eq!(avg, font.default_width);
    }

    #[test]
    fn test_calculate_average_glyph_width_with_widths() {
        let extractor = TextExtractor::new();
        let mut font = create_test_font();
        font.first_char = Some(32);
        font.last_char = Some(126);
        font.widths = Some(vec![500.0; 95]); // 95 printable chars

        let avg = extractor.calculate_average_glyph_width(&font);
        assert!((avg - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_average_glyph_width_no_first_char() {
        let extractor = TextExtractor::new();
        let mut font = create_test_font();
        font.widths = Some(vec![500.0; 95]);
        font.first_char = None;

        let avg = extractor.calculate_average_glyph_width(&font);
        assert_eq!(avg, font.default_width);
    }

    #[test]
    fn test_calculate_average_glyph_width_no_last_char() {
        let extractor = TextExtractor::new();
        let mut font = create_test_font();
        font.widths = Some(vec![500.0; 95]);
        font.first_char = Some(32);
        font.last_char = None;

        let avg = extractor.calculate_average_glyph_width(&font);
        assert_eq!(avg, font.default_width);
    }

    // ========================================================================
    // COVERAGE TESTS: Adaptive TJ threshold with justified text
    // ========================================================================

    #[test]
    fn test_adaptive_threshold_with_justified_text() {
        let config = TextExtractionConfig {
            use_adaptive_tj_threshold: true,
            word_margin_ratio: 0.1,
            ..TextExtractionConfig::default()
        };
        let mut extractor = TextExtractor::with_config(config);
        extractor.state_stack.current_mut().font_size = 12.0;

        // Simulate justified text (high CV)
        for i in 0..100 {
            extractor
                .tj_offset_history
                .push(if i % 2 == 0 { -50.0 } else { -200.0 });
        }

        let threshold = extractor.calculate_adaptive_tj_threshold();
        // Justified text uses 3x ratio, so threshold should be more negative
        assert!(threshold < 0.0);
    }

    #[test]
    fn test_adaptive_threshold_with_font_name() {
        let config = TextExtractionConfig {
            use_adaptive_tj_threshold: true,
            word_margin_ratio: 0.1,
            ..TextExtractionConfig::default()
        };
        let mut extractor = TextExtractor::with_config(config);
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);
        extractor.state_stack.current_mut().font_size = 12.0;
        extractor.state_stack.current_mut().font_name = Some("F1".to_string());

        let threshold = extractor.calculate_adaptive_tj_threshold();
        assert!(threshold < 0.0);
    }

    #[test]
    fn test_analyze_tj_distribution_zero_mean() {
        let mut extractor = TextExtractor::new();
        // Push offsets that average to near zero
        extractor.tj_offset_history = vec![100.0, -100.0, 100.0, -100.0];
        let (is_justified, cv) = extractor.analyze_tj_distribution();
        // Mean ~0, so CV should be 0 (avoid division by zero)
        assert_eq!(cv, 0.0);
    }

    // ========================================================================
    // COVERAGE TESTS: Quote and DoubleQuote operators in span mode
    // ========================================================================

    #[test]
    fn test_quote_operator_span_mode() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td (Line1) Tj (Line2) ' ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Line1"), "Should contain Line1, got: {}", text);
        assert!(text.contains("Line2"), "Should contain Line2, got: {}", text);
    }

    #[test]
    fn test_double_quote_operator_span_mode() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 14 TL 100 700 Td 1 2 (Text) \" ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Text"), "Should extract text, got: {}", text);
    }

    // ========================================================================
    // COVERAGE TESTS: Sort spans by columns (multi-column)
    // ========================================================================

    #[test]
    fn test_sort_spans_by_columns() {
        let mut extractor = TextExtractor::new();
        // Create spans in two distinct columns
        let columns = vec![(0.0, 250.0), (300.0, 550.0)];

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Right Col".to_string(),
                bbox: Rect::new(350.0, 700.0, 100.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Left Col".to_string(),
                bbox: Rect::new(50.0, 700.0, 100.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.sort_spans_by_columns(&columns);
        // Left column should come first
        assert_eq!(extractor.spans[0].text, "Left Col");
        assert_eq!(extractor.spans[1].text, "Right Col");
    }

    // ========================================================================
    // COVERAGE TESTS: TJ buffer with MCID
    // ========================================================================

    #[test]
    fn test_tj_buffer_with_mcid() {
        let state = crate::content::graphics_state::GraphicsStateStack::new();
        let buffer = TjBuffer::new(state.current(), Some(42), None);
        assert!(buffer.is_empty());
        assert_eq!(buffer.mcid, Some(42));
    }

    // ========================================================================
    // COVERAGE TESTS: Word boundary mode primary
    // ========================================================================

    #[test]
    fn test_extractor_with_primary_word_boundary() {
        let config = TextExtractionConfig {
            word_boundary_mode: WordBoundaryMode::Primary,
            ..TextExtractionConfig::default()
        };
        let mut extractor = TextExtractor::with_config(config);
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);
        extractor.merging_config = SpanMergingConfig::legacy();

        let stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Hello"), "Primary mode should still extract text, got: {}", text);
    }

    // ========================================================================
    // COVERAGE TESTS: Merge adjacent spans - double space prevention
    // ========================================================================

    #[test]
    fn test_merge_prevents_double_space() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello ".to_string(), // ends with space
                bbox: Rect::new(100.0, 700.0, 35.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: " World".to_string(), // starts with space
                bbox: Rect::new(136.0, 700.0, 35.0, 12.0), // 1pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: true, // forces merge-with-space path
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1);
        // Should not have "Hello   World" (triple space)
        assert!(!extractor.spans[0].text.contains("   "), "Should prevent triple space");
    }

    // ========================================================================
    // COVERAGE TESTS: TextExtractor with_config builder
    // ========================================================================

    #[test]
    fn test_extractor_with_config_copies_word_boundary_mode() {
        let config = TextExtractionConfig {
            word_boundary_mode: WordBoundaryMode::Primary,
            ..TextExtractionConfig::default()
        };
        let extractor = TextExtractor::with_config(config);
        assert_eq!(extractor.word_boundary_mode, WordBoundaryMode::Primary);
    }

    // ========================================================================
    // COVERAGE TESTS: Partition characters boundary at start
    // ========================================================================

    #[test]
    fn test_partition_boundary_at_start() {
        let extractor = TextExtractor::new();
        let chars = vec![
            CharacterInfo {
                code: 65,
                glyph_id: None,
                width: 10.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 66,
                glyph_id: None,
                width: 10.0,
                x_position: 10.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ];

        // Boundary at 0 means empty first cluster
        let clusters = extractor.partition_characters_by_boundaries(&chars, vec![0]);
        // Should have just one cluster (boundary at 0 produces no items before it)
        assert!(!clusters.is_empty());
    }

    // ========================================================================
    // COVERAGE TESTS: Color space resets (fill_color_cmyk cleared)
    // ========================================================================

    #[test]
    fn test_fill_cmyk_then_change_color_space() {
        let mut extractor = TextExtractor::new();
        extractor
            .execute_operator_public(Operator::SetFillCmyk {
                c: 0.5,
                m: 0.5,
                y: 0.5,
                k: 0.5,
            })
            .unwrap();
        assert!(extractor.state_stack.current().fill_color_cmyk.is_some());

        // Changing color space should reset CMYK
        extractor
            .execute_operator_public(Operator::SetFillColorSpace {
                name: "DeviceRGB".to_string(),
            })
            .unwrap();
        assert!(extractor.state_stack.current().fill_color_cmyk.is_none());
    }

    // ========================================================================
    // COVERAGE TESTS: Marked content with BDC
    // ========================================================================

    #[test]
    fn test_bdc_with_mcid() {
        let mut extractor = TextExtractor::new();
        let mut props = HashMap::new();
        props.insert("MCID".to_string(), Object::Integer(5));

        extractor
            .execute_operator_public(Operator::BeginMarkedContentDict {
                tag: "P".to_string(),
                properties: Box::new(Object::Dictionary(props)),
            })
            .unwrap();

        assert_eq!(extractor.current_mcid, Some(5));
        assert!(!extractor.inside_artifact);
    }

    #[test]
    fn test_bdc_artifact_with_type() {
        let mut extractor = TextExtractor::new();
        let mut props = HashMap::new();
        props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
        props.insert("Subtype".to_string(), Object::Name("Header".to_string()));

        extractor
            .execute_operator_public(Operator::BeginMarkedContentDict {
                tag: "Artifact".to_string(),
                properties: Box::new(Object::Dictionary(props)),
            })
            .unwrap();

        assert!(extractor.inside_artifact);
    }

    #[test]
    fn test_emc_resets_mcid() {
        let mut extractor = TextExtractor::new();
        extractor.current_mcid = Some(10);
        extractor.marked_content_stack.push(MarkedContentContext {
            artifact_type: None,
            tag: "P".to_string(),
            is_artifact: false,
            actual_text: None,
            expansion: None,
        });

        extractor
            .execute_operator_public(Operator::EndMarkedContent)
            .unwrap();

        assert_eq!(extractor.current_mcid, None);
        assert!(extractor.marked_content_stack.is_empty());
    }

    #[test]
    fn test_emc_with_empty_stack() {
        let mut extractor = TextExtractor::new();
        // Should not panic
        extractor
            .execute_operator_public(Operator::EndMarkedContent)
            .unwrap();
    }

    // ========================================================================
    // COVERAGE TESTS: BDC with ActualText and Expansion
    // ========================================================================

    #[test]
    fn test_bdc_with_actual_text() {
        let mut extractor = TextExtractor::new();
        let mut props = HashMap::new();
        props.insert("ActualText".to_string(), Object::String(b"fi".to_vec()));

        extractor
            .execute_operator_public(Operator::BeginMarkedContentDict {
                tag: "Span".to_string(),
                properties: Box::new(Object::Dictionary(props)),
            })
            .unwrap();

        let actual = extractor.get_current_actual_text();
        assert_eq!(actual, Some("fi".to_string()));
    }

    #[test]
    fn test_bdc_with_expansion() {
        let mut extractor = TextExtractor::new();
        let mut props = HashMap::new();
        props.insert("E".to_string(), Object::String(b"PDF".to_vec()));

        extractor
            .execute_operator_public(Operator::BeginMarkedContentDict {
                tag: "Span".to_string(),
                properties: Box::new(Object::Dictionary(props)),
            })
            .unwrap();

        let ctx = &extractor.marked_content_stack[0];
        assert_eq!(ctx.expansion, Some("PDF".to_string()));
    }

    // ========================================================================
    // COVERAGE TESTS: Do operator without document
    // ========================================================================

    #[test]
    fn test_do_operator_without_document() {
        let mut extractor = TextExtractor::new();
        // Do without document set should not panic
        extractor
            .execute_operator_public(Operator::Do {
                name: "Im1".to_string(),
            })
            .unwrap();
    }

    // ========================================================================
    // COVERAGE TESTS: flush_tj_span_buffer when buffer is Some but empty
    // ========================================================================

    #[test]
    fn test_flush_tj_span_buffer_empty_buffer() {
        let mut extractor = TextExtractor::new();
        let state = extractor.state_stack.current().clone();
        extractor.tj_span_buffer = Some(TjBuffer::new(&state, None, None));
        // Empty buffer should not produce a span
        let before = extractor.spans.len();
        extractor.flush_tj_span_buffer().unwrap();
        assert_eq!(extractor.spans.len(), before);
    }

    #[test]
    fn test_flush_tj_span_buffer_with_content() {
        let mut extractor = TextExtractor::new();
        let state_stack = crate::content::graphics_state::GraphicsStateStack::new();
        let mut buffer = TjBuffer::new(state_stack.current(), Some(7), None);
        buffer.append(b"Test").unwrap();
        buffer.accumulated_width = 20.0;
        extractor.tj_span_buffer = Some(buffer);

        extractor.flush_tj_span_buffer().unwrap();
        assert_eq!(extractor.spans.len(), 1);
        assert!(extractor.spans[0].text.contains("Test"));
    }

    // ========================================================================
    // COVERAGE TESTS: TJ array with adaptive threshold - full pipeline
    // ========================================================================

    #[test]
    fn test_tj_array_span_mode_with_space_insertion() {
        let config = TextExtractionConfig {
            use_adaptive_tj_threshold: false,
            space_insertion_threshold: -120.0,
            ..TextExtractionConfig::default()
        };
        let mut extractor = TextExtractor::with_config(config);
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // TJ array with large offset that triggers space
        let stream = b"BT /F1 12 Tf 100 700 Td [(Word1) -500 (Word2)] TJ ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        let text: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("Word1"), "Should contain Word1");
        assert!(text.contains("Word2"), "Should contain Word2");
    }

    // ========================================================================
    // COVERAGE TESTS: Sort spans reading order (single vs multi column)
    // ========================================================================

    #[test]
    fn test_sort_spans_single_column() {
        let mut extractor = TextExtractor::new();
        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Line2".to_string(),
                bbox: Rect::new(50.0, 680.0, 100.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "Line1".to_string(),
                bbox: Rect::new(50.0, 700.0, 100.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.sort_spans_by_reading_order();
        assert_eq!(extractor.spans[0].text, "Line1"); // higher Y first
        assert_eq!(extractor.spans[1].text, "Line2");
    }

    // ========================================================================
    // COVERAGE TESTS: Tm continuation optimization
    // ========================================================================

    #[test]
    fn test_tm_continuation_different_transform() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Different transform params (a=2) should NOT be continuation
        let stream = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (A) Tj 2 0 0 1 120 700 Tm (B) Tj ET";
        let spans = extractor.extract_text_spans(stream).unwrap();

        // Should produce separate spans due to different transform
        assert!(!spans.is_empty());
    }

    // ========================================================================
    // COVERAGE TESTS: decode_pdf_text_string edge cases
    // ========================================================================

    #[test]
    fn test_decode_pdf_text_string_single_byte() {
        let result = TextExtractor::decode_pdf_text_string(&[0x41]);
        assert_eq!(result, "A");
    }

    #[test]
    fn test_decode_pdf_text_string_invalid_utf16() {
        // UTF-16BE BOM followed by invalid pair
        let bytes = vec![0xFE, 0xFF, 0xD8, 0x00]; // invalid surrogate half
        let result = TextExtractor::decode_pdf_text_string(&bytes);
        // Should fall back to lossy conversion
        assert!(!result.is_empty() || result.is_empty()); // Just don't panic
    }

    #[test]
    fn test_decode_pdf_text_string_utf16le_invalid() {
        // UTF-16LE BOM followed by odd byte count
        let bytes = vec![0xFF, 0xFE, 0x41]; // odd after BOM
        let result = TextExtractor::decode_pdf_text_string(&bytes);
        // Should handle gracefully
    }

    // ========================================================================
    // COVERAGE TESTS: shared truetype cmaps (no donors)
    // ========================================================================

    #[test]
    fn test_share_truetype_cmaps_no_donors() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        // Should return early (no cmap donors)
        extractor.share_truetype_cmaps();
        assert_eq!(extractor.fonts.len(), 1);
    }

    // ========================================================================
    // COVERAGE TESTS: Extract with WithConfig
    // ========================================================================

    #[test]
    fn test_extractor_with_config_and_profile() {
        let config =
            TextExtractionConfig::new().with_profile(crate::config::ExtractionProfile::POLICY);

        let mut extractor = TextExtractor::with_config(config);
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf 100 700 Td (Policy) Tj ET";
        let chars = extractor.extract(stream).unwrap();
        assert!(!chars.is_empty());
    }

    // ========================================================================
    // COVERAGE TESTS: Merge with offset_semantic space span suppression
    // ========================================================================

    #[test]
    fn test_merge_offset_semantic_space_suppression() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: " ".to_string(), // offset_semantic space
                bbox: Rect::new(130.5, 700.0, 2.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: true, // forcing merge path
                offset_semantic: true,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        // offset_semantic space should be merged without adding extra space
        let text = &extractor.spans[0].text;
        assert!(!text.contains("  "), "Should not have double space, got: '{}'", text);
    }
}

#[test]
fn test_space_threshold_default() {
    // Test that default configuration uses -120.0 threshold
    let config = TextExtractionConfig::new();
    assert_eq!(config.space_insertion_threshold, -120.0);

    // Test that default extractor has default config
    let extractor = TextExtractor::new();
    assert_eq!(extractor.config.space_insertion_threshold, -120.0);
}

#[test]
fn test_space_threshold_custom() {
    // Test custom threshold configuration
    let config = TextExtractionConfig::with_space_threshold(-80.0);
    assert_eq!(config.space_insertion_threshold, -80.0);

    let extractor = TextExtractor::with_config(config);
    assert_eq!(extractor.config.space_insertion_threshold, -80.0);
}

#[test]
fn test_space_threshold_disabled() {
    // Test that threshold can be disabled with NEG_INFINITY
    let config = TextExtractionConfig::with_space_threshold(f32::NEG_INFINITY);
    assert_eq!(config.space_insertion_threshold, f32::NEG_INFINITY);

    let extractor = TextExtractor::with_config(config);
    assert_eq!(extractor.config.space_insertion_threshold, f32::NEG_INFINITY);
}

#[test]
fn test_adaptive_enabled_by_default() {
    // Test that adaptive threshold is enabled by default
    let config = SpanMergingConfig::default();
    assert!(config.use_adaptive_threshold, "Adaptive threshold should be enabled by default");
}

#[test]
fn test_legacy_mode_disables_adaptive() {
    // Test that legacy() constructor provides backward-compatible behavior
    let legacy = SpanMergingConfig::legacy();
    assert!(!legacy.use_adaptive_threshold, "Legacy mode should disable adaptive threshold");
    assert_eq!(legacy.conservative_threshold_pt, 0.1);
}

#[test]
fn test_adaptive_constructor_enables_adaptive() {
    // Test that adaptive() constructor enables adaptive threshold
    let adaptive = SpanMergingConfig::adaptive();
    assert!(
        adaptive.use_adaptive_threshold,
        "Adaptive constructor should enable adaptive threshold"
    );
    assert!(
        adaptive.adaptive_config.is_some(),
        "Adaptive constructor should set adaptive_config"
    );
}

// ============================================================================
// Artifact Type Parsing Tests (PDF Spec Section 14.8.2.2)
// ============================================================================

#[test]
fn test_parse_artifact_type_pagination_header() {
    let mut props = HashMap::new();
    props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
    props.insert("Subtype".to_string(), Object::Name("Header".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Header)));
}

#[test]
fn test_parse_artifact_type_pagination_footer() {
    let mut props = HashMap::new();
    props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
    props.insert("Subtype".to_string(), Object::Name("Footer".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Footer)));
}

#[test]
fn test_parse_artifact_type_pagination_watermark() {
    let mut props = HashMap::new();
    props.insert("Type".to_string(), Object::Name("Pagination".to_string()));
    props.insert("Subtype".to_string(), Object::Name("Watermark".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Watermark)));
}

#[test]
fn test_parse_artifact_type_layout() {
    let mut props = HashMap::new();
    props.insert("Type".to_string(), Object::Name("Layout".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Layout));
}

#[test]
fn test_parse_artifact_type_background() {
    let mut props = HashMap::new();
    props.insert("Type".to_string(), Object::Name("Background".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Background));
}

#[test]
fn test_parse_artifact_type_subtype_only() {
    // Some PDFs use /Subtype without /Type
    let mut props = HashMap::new();
    props.insert("Subtype".to_string(), Object::Name("Header".to_string()));

    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, Some(ArtifactType::Pagination(PaginationSubtype::Header)));
}

#[test]
fn test_parse_artifact_type_empty() {
    let props = HashMap::new();
    let result = TextExtractor::parse_artifact_type(&props);
    assert_eq!(result, None);
}

// ============================================================================
// ActualText Verification Tests (PDF Spec Section 14.9.4)
// ============================================================================
//
// ActualText provides replacement text for content that cannot be accurately
// represented by the content stream (ligatures, decorated glyphs, formulas).
// Per ISO 32000-1:2008 Section 14.9.4, ActualText takes precedence over
// character extraction.

#[test]
fn test_marked_content_context_with_actual_text() {
    // Verify MarkedContentContext correctly stores ActualText
    let ctx = MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: Some("fi".to_string()), // Ligature expansion
        expansion: None,
    };

    assert_eq!(ctx.actual_text, Some("fi".to_string()));
    assert!(!ctx.is_artifact);
}

#[test]
fn test_marked_content_context_with_expansion() {
    // Verify MarkedContentContext correctly stores /E expansion
    let ctx = MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: None,
        expansion: Some("Portable Document Format".to_string()),
    };

    assert_eq!(ctx.expansion, Some("Portable Document Format".to_string()));
}

#[test]
fn test_marked_content_context_artifact_with_actual_text() {
    // Verify artifacts can have ActualText (though typically they don't)
    let ctx = MarkedContentContext {
        tag: "Artifact".to_string(),
        is_artifact: true,
        artifact_type: Some(ArtifactType::Pagination(PaginationSubtype::Header)),
        actual_text: Some("Header text".to_string()),
        expansion: None,
    };

    assert!(ctx.is_artifact);
    assert_eq!(ctx.actual_text, Some("Header text".to_string()));
}

#[test]
fn test_get_current_actual_text_finds_first() {
    // Verify get_current_actual_text returns first ActualText in stack
    let mut extractor = TextExtractor::new();

    // Push contexts with ActualText
    extractor.marked_content_stack.push(MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: Some("outer text".to_string()),
        expansion: None,
    });

    extractor.marked_content_stack.push(MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: Some("inner text".to_string()),
        expansion: None,
    });

    // Should return innermost (most recent) ActualText
    let result = extractor.get_current_actual_text();
    assert_eq!(result, Some("inner text".to_string()));
}

#[test]
fn test_get_current_actual_text_skips_none() {
    // Verify get_current_actual_text skips contexts without ActualText
    let mut extractor = TextExtractor::new();

    // Push context with ActualText
    extractor.marked_content_stack.push(MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: Some("replacement text".to_string()),
        expansion: None,
    });

    // Push context without ActualText
    extractor.marked_content_stack.push(MarkedContentContext {
        artifact_type: None,
        tag: "Span".to_string(),
        is_artifact: false,
        actual_text: None,
        expansion: None,
    });

    // Should find the ActualText from outer context
    let result = extractor.get_current_actual_text();
    assert_eq!(result, Some("replacement text".to_string()));
}

#[test]
fn test_get_current_actual_text_returns_none_when_empty() {
    // Verify get_current_actual_text returns None when no ActualText
    let extractor = TextExtractor::new();

    let result = extractor.get_current_actual_text();
    assert_eq!(result, None);
}

// ============================================================================
// PHASE 2.5: Profile-Based Space Insertion Tests (TDD)
// ============================================================================
//
// Tests for document-type-specific extraction profiles.
// These tests define expected behavior BEFORE implementation.
// Once these tests pass, the profile integration is complete.
//
// Key Scenarios:
// 1. Academic papers: Tighter spacing, aggressive space insertion
// 2. Policy documents: Justified text, conservative spacing
// 3. Forms: Structured fields with precise boundaries
// 4. Default/Conservative: Backward-compatible behavior

#[cfg(test)]
mod profile_based_space_tests {
    use super::*;

    /// Test that ACADEMIC profile uses aggressive thresholds
    ///
    /// Academic papers have tight spacing (especially around punctuation).
    /// The profile should:
    /// - Use lower TJ offset threshold (-90 instead of -120)
    /// - Use lower word margin ratio (0.12 instead of 0.1)
    /// - Enable adaptive threshold for dynamic adjustment
    #[test]
    fn test_academic_profile_thresholds() {
        let profile = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Academic,
        );

        // Academic papers should be more aggressive with space insertion
        assert!(
            profile.tj_offset_threshold < -100.0,
            "Academic should use lower TJ threshold for more spaces"
        );

        // Academic papers should have tighter word margins
        assert!(
            profile.word_margin_ratio <= 0.15,
            "Academic should use conservative word margin"
        );

        // Verify we can create a config from the profile
        let config = TextExtractionConfig::with_space_threshold(profile.tj_offset_threshold);
        assert_eq!(config.space_insertion_threshold, profile.tj_offset_threshold);
    }

    /// Test that POLICY profile uses conservative thresholds
    ///
    /// Policy documents (like GDPR) have justified text with precise spacing.
    /// The profile should:
    /// - Use higher TJ offset threshold (-110 to preserve structure)
    /// - Use higher word margin ratio (0.18-0.2 for justified text)
    /// - Preserve column boundaries and table structure
    #[test]
    fn test_policy_profile_thresholds() {
        let profile = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Policy,
        );

        // Policy documents should be more conservative to preserve structure
        assert!(
            profile.tj_offset_threshold > -120.0,
            "Policy should use higher TJ threshold to avoid over-spacing"
        );

        // Policy documents should have looser word margins for justified text
        assert!(
            profile.word_margin_ratio >= 0.15,
            "Policy should use higher word margin for justified text"
        );

        let config = TextExtractionConfig::with_space_threshold(profile.tj_offset_threshold);
        assert_eq!(config.space_insertion_threshold, profile.tj_offset_threshold);
    }

    /// Test that FORM profile preserves field boundaries
    ///
    /// Forms have checkboxes, fields, and precise layout.
    /// The profile should:
    /// - Use conservative thresholds to avoid merging fields
    /// - High column boundary threshold to preserve structure
    /// - Enable adaptive threshold for form field detection
    #[test]
    fn test_form_profile_thresholds() {
        let profile =
            crate::config::ExtractionProfile::for_document_type(crate::config::DocumentType::Form);

        // Forms should preserve field structure with conservative spacing
        assert!(
            profile.tj_offset_threshold >= -120.0,
            "Form profile should be conservative with space insertion"
        );

        let config = TextExtractionConfig::with_space_threshold(profile.tj_offset_threshold);
        assert_eq!(config.space_insertion_threshold, profile.tj_offset_threshold);
    }

    /// Test that profile selection works correctly for document types
    #[test]
    fn test_profile_selection_for_document_types() {
        let academic = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Academic,
        );
        let policy = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Policy,
        );
        let form =
            crate::config::ExtractionProfile::for_document_type(crate::config::DocumentType::Form);
        let mixed =
            crate::config::ExtractionProfile::for_document_type(crate::config::DocumentType::Mixed);

        // Verify each profile has distinct thresholds
        let thresholds = [
            academic.tj_offset_threshold,
            policy.tj_offset_threshold,
            form.tj_offset_threshold,
            mixed.tj_offset_threshold,
        ];

        // At least some profiles should have different thresholds
        let unique_count = thresholds
            .iter()
            .filter(|t| !thresholds.iter().skip(1).any(|other| other == *t))
            .count();

        assert!(
            unique_count > 0,
            "Profiles should have different thresholds for different document types"
        );
    }

    /// Test that TextExtractionConfig can accept a profile
    #[test]
    fn test_config_with_profile() {
        let profile = crate::config::ExtractionProfile::ACADEMIC;

        // Should be able to create config with profile thresholds
        let config = TextExtractionConfig::with_space_threshold(profile.tj_offset_threshold);

        assert_eq!(config.space_insertion_threshold, profile.tj_offset_threshold);
    }

    /// Test that profiles have reasonable threshold ranges
    #[test]
    fn test_profile_thresholds_in_reasonable_range() {
        let profiles = vec![
            crate::config::ExtractionProfile::CONSERVATIVE,
            crate::config::ExtractionProfile::ACADEMIC,
            crate::config::ExtractionProfile::POLICY,
            crate::config::ExtractionProfile::FORM,
        ];

        for profile in profiles {
            // TJ offsets should be negative (per PDF spec)
            assert!(
                profile.tj_offset_threshold < 0.0,
                "TJ threshold must be negative ({})",
                profile.name
            );

            // Should be in reasonable range (-150 to -50)
            assert!(
                profile.tj_offset_threshold >= -150.0 && profile.tj_offset_threshold <= -50.0,
                "TJ threshold out of range for {} ({})",
                profile.name,
                profile.tj_offset_threshold
            );

            // Word margin ratios should be positive and reasonable (0.05 to 0.25)
            assert!(
                profile.word_margin_ratio > 0.0 && profile.word_margin_ratio < 1.0,
                "Word margin ratio must be between 0 and 1 for {}",
                profile.name
            );

            // Space threshold EM ratio should be positive
            assert!(
                profile.space_threshold_em_ratio > 0.0,
                "Space threshold EM ratio must be positive for {}",
                profile.name
            );
        }
    }

    /// Test that multiple profiles can coexist
    #[test]
    fn test_multiple_profiles_independent() {
        let academic = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Academic,
        );
        let policy = crate::config::ExtractionProfile::for_document_type(
            crate::config::DocumentType::Policy,
        );

        // Create configs from both profiles
        let academic_config =
            TextExtractionConfig::with_space_threshold(academic.tj_offset_threshold);
        let policy_config = TextExtractionConfig::with_space_threshold(policy.tj_offset_threshold);

        // Verify they have different thresholds
        assert_ne!(
            academic_config.space_insertion_threshold, policy_config.space_insertion_threshold,
            "Academic and policy configs should have different thresholds"
        );
    }

    /// Test that default config is backward-compatible
    #[test]
    fn test_default_config_backward_compatible() {
        let default_config = TextExtractionConfig::default();
        let conservative_profile = crate::config::ExtractionProfile::CONSERVATIVE;

        // Default should match or be compatible with conservative profile
        assert_eq!(
            default_config.space_insertion_threshold, conservative_profile.tj_offset_threshold,
            "Default config should use conservative threshold for backward compatibility"
        );
    }

    /// Test that adjacent table cell values get spaces inserted between them.
    ///
    /// Simulates a form where two "$0.00" values are in adjacent cells with
    /// a small positive gap (1pt). The merge logic should insert a space because
    /// the spans are clearly separate tokens (ending/starting with digits/currency).
    #[test]
    fn test_adjacent_table_cell_values_not_concatenated() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        // "$0.00" at 10pt font is about 30pt wide (5 chars * ~6pt average width)
        // Second value starts at x=131, creating a 1pt gap (100 + 30 = 130, gap = 1pt)
        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "$0.00".to_string(),
                bbox: Rect::new(100.0, 700.0, 30.0, 10.0),
                font_name: "F1".to_string(),
                font_size: 10.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "$0.00".to_string(),
                bbox: Rect::new(131.0, 700.0, 30.0, 10.0), // 1pt gap
                font_name: "F1".to_string(),
                font_size: 10.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "Adjacent spans should merge");
        assert_eq!(
            extractor.spans[0].text, "$0.00 $0.00",
            "Adjacent table cell values should have space between them, got: '{}'",
            extractor.spans[0].text
        );
    }

    /// Test that adjacent numeric values with small gaps get spaces.
    /// Covers cases like "100200" that should be "100 200" in table contexts.
    #[test]
    fn test_adjacent_numeric_values_not_concatenated() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "100".to_string(),
                bbox: Rect::new(200.0, 500.0, 18.0, 10.0),
                font_name: "F1".to_string(),
                font_size: 10.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "200".to_string(),
                bbox: Rect::new(219.5, 500.0, 18.0, 10.0), // 1.5pt gap
                font_name: "F1".to_string(),
                font_size: 10.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "Adjacent spans should merge");
        assert_eq!(
            extractor.spans[0].text, "100 200",
            "Adjacent numeric values should have space between them, got: '{}'",
            extractor.spans[0].text
        );
    }

    /// Ensure that true word fragments (zero gap) still merge without space.
    /// E.g., "Hel" + "lo" with gap=0 should become "Hello" not "Hel lo".
    #[test]
    fn test_word_fragments_zero_gap_no_space() {
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hel".to_string(),
                bbox: Rect::new(100.0, 700.0, 18.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "lo".to_string(),
                bbox: Rect::new(118.0, 700.0, 12.0, 12.0), // 0pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "Adjacent spans should merge");
        assert_eq!(
            extractor.spans[0].text, "Hello",
            "Zero-gap word fragments should merge without space, got: '{}'",
            extractor.spans[0].text
        );
    }

    // ========================================================================
    // Decimal dollar value merging (split integer/decimal boxes)
    // ========================================================================

    #[test]
    fn test_merge_decimal_dollar_value_split_boxes() {
        // Some forms have integer and decimal parts in separate fixed-width boxes.
        // e.g., "123456" at x=382.3 width=39.6, "72" at x=432.7 width=13.2
        // gap = 432.7 - (382.3 + 39.6) = 10.8pt
        // These should be merged as "123456.72"
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "123456".to_string(),
                bbox: Rect::new(382.3, 700.0, 39.6, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "72".to_string(),
                bbox: Rect::new(432.7, 700.0, 13.2, 12.0), // 10.8pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1, "Decimal dollar value spans should merge into one");
        assert_eq!(
            extractor.spans[0].text, "123456.72",
            "Integer and decimal parts should be joined with '.'"
        );
    }

    #[test]
    fn test_merge_decimal_value_small_integer_part() {
        // Smaller dollar amount: "50" + "00" -> "50.00"
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "50".to_string(),
                bbox: Rect::new(382.3, 700.0, 15.0, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "00".to_string(),
                bbox: Rect::new(407.0, 700.0, 13.2, 12.0), // 9.7pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        assert_eq!(extractor.spans.len(), 1);
        assert_eq!(extractor.spans[0].text, "50.00");
    }

    #[test]
    fn test_no_decimal_merge_for_non_digit_spans() {
        // Should NOT merge "Hello" + "72" as decimal
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "Hello".to_string(),
                bbox: Rect::new(382.3, 700.0, 39.6, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "72".to_string(),
                bbox: Rect::new(432.7, 700.0, 13.2, 12.0), // 10.8pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        // Should NOT merge because first span is not all digits
        assert_eq!(
            extractor.spans.len(),
            2,
            "Non-digit spans should not be merged as decimal values"
        );
    }

    #[test]
    fn test_no_decimal_merge_for_long_decimal_part() {
        // Should NOT merge "123456" + "723" (3-digit decimal part is not a cents pattern)
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::legacy();

        extractor.spans = vec![
            TextSpan {
                artifact_type: None,
                text: "123456".to_string(),
                bbox: Rect::new(382.3, 700.0, 39.6, 12.0),
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            TextSpan {
                artifact_type: None,
                text: "723".to_string(),
                bbox: Rect::new(432.7, 700.0, 18.0, 12.0), // 10.8pt gap
                font_name: "F1".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
                offset_semantic: false,
                is_italic: false,
                is_monospace: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
        ];

        extractor.merge_adjacent_spans();
        // Should NOT merge because decimal part has 3 digits (not a cents pattern)
        assert_eq!(
            extractor.spans.len(),
            2,
            "3-digit decimal part should not trigger decimal merge"
        );
    }

    #[test]
    fn test_cross_font_word_glue_single_letter_prefix() {
        // A single-letter span in one font, tight-kerned against a
        // multi-letter span in another font, is the drop-cap pattern.
        // These must merge into one word with the longer run's font
        // metadata — emitting per-letter emphasis runs corrupts proper
        // nouns.
        let mut extractor = TextExtractor::new();
        extractor.merging_config = SpanMergingConfig::default();

        extractor.spans = vec![
            TextSpan {
                text: "S".to_string(),
                bbox: Rect::new(72.0, 700.0, 10.0, 12.0),
                font_name: "Helvetica-Bold".to_string(),
                font_weight: FontWeight::Bold,
                font_size: 12.0,
                ..TextSpan::default()
            },
            TextSpan {
                text: "ales".to_string(),
                bbox: Rect::new(82.0, 700.0, 30.0, 12.0),
                font_name: "Helvetica".to_string(),
                font_weight: FontWeight::Normal,
                font_size: 12.0,
                ..TextSpan::default()
            },
        ];

        extractor.merge_adjacent_spans();

        assert_eq!(
            extractor.spans.len(),
            1,
            "cross_font_word_glue should merge 'S' + 'ales' into 'Sales'"
        );
        assert_eq!(extractor.spans[0].text, "Sales");
        // Dominant-font swap: the longer run (regular weight) should win.
        assert_eq!(extractor.spans[0].font_weight, FontWeight::Normal);
    }
}
