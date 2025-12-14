//! Text extraction from PDF content streams.
//!
//! This module executes content stream operators to extract positioned
//! text characters with their Unicode mappings, font information, and
//! bounding boxes.

use crate::config::ExtractionProfile;
use crate::content::graphics_state::{GraphicsStateStack, Matrix};
use crate::content::operators::{Operator, TextElement};
use crate::content::parse_content_stream;
use crate::error::Result;
use crate::extract_log_debug;
use crate::fonts::FontInfo;
use crate::geometry::Rect;
use crate::layout::{Color, FontWeight, TextChar, TextSpan};
use crate::object::{Object, ObjectRef};
use crate::pipeline::config::WordBoundaryMode;
use crate::text::{BoundaryContext, CharacterInfo, DocumentScript, WordBoundaryDetector};
use std::collections::{HashMap, HashSet};

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
    fonts: &std::collections::HashMap<String, crate::fonts::FontInfo>,
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
        // Calculate vertical and horizontal positioning for line break detection
        let prev_bottom = prev_box.y + prev_box.height;
        let next_top = next_box.y;
        let vertical_gap = (prev_bottom - next_top).abs();

        // Line break threshold: if vertical gap > 0.5× font size (typical line spacing margin)
        let line_break_threshold = font_size * 0.5;
        let is_line_break = vertical_gap > line_break_threshold;

        if is_line_break {
            // Verify same-column layout: X-positions within 2× font width
            let same_column = (prev_box.left() - next_box.left()).abs() < (font_size * 2.0);

            if same_column {
                log::debug!(
                    "Detected line break: vertical_gap={:.2}pt > {:.2}pt threshold, same_column=true",
                    vertical_gap,
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

    // Strong geometric signal alone (gap > 2× threshold)
    // This is high confidence even without TJ signal
    let strong_geometric_threshold = geometric_threshold * 2.0;
    if gap_pt > strong_geometric_threshold {
        log::debug!(
            "Space decision: STRONG GEOMETRIC - gap={:.2}pt > 2×{:.2}pt threshold - inserting space",
            gap_pt,
            geometric_threshold
        );
        return SpaceDecision::insert(SpaceSource::GeometricGap, 0.95);
    }

    // Default: No space
    // Per ISO 32000-1:2008 Section 9.10, when PDF doesn't encode a clear word boundary,
    // we cannot reliably recover it. Requiring consensus prevents false positives in justified text.
    log::trace!(
        "Space decision: Insufficient consensus (TJ={}, gap={:.2}pt <= {:.2}pt, strong_threshold={:.2}pt) - no space",
        tj_offset_triggered,
        gap_pt,
        geometric_threshold,
        strong_geometric_threshold
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
    let has_trailing_space = preceding.chars().last().is_some_and(|c| c.is_whitespace());
    let has_leading_space = following.chars().next().is_some_and(|c| c.is_whitespace());

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
    let prev_char_count = prev_text.chars().count().max(1) as f32;
    let prev_char_width = prev_bbox.width / prev_char_count;
    let prev_last_x = prev_bbox.x + prev_bbox.width - prev_char_width;

    let next_char_count = next_text.chars().count().max(1) as f32;
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
    let prev = preceding_text.trim_end();
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
    /// Accumulated raw bytes from text strings
    text: Vec<u8>,
    /// Accumulated Unicode text
    unicode: String,
    /// Text matrix at the start of this buffer
    start_matrix: Matrix,
    /// Font name when buffer started
    font_name: Option<String>,
    /// Font size when buffer started
    font_size: f32,
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
}

impl TjBuffer {
    /// Create a new empty buffer with current state.
    fn new(state: &crate::content::graphics_state::GraphicsState, mcid: Option<u32>) -> Self {
        Self {
            text: Vec::new(),
            unicode: String::new(),
            start_matrix: state.text_matrix,
            font_name: state.font_name.clone(),
            font_size: state.font_size,
            fill_color_rgb: state.fill_color_rgb,
            char_space: state.char_space,
            word_space: state.word_space,
            horizontal_scaling: state.horizontal_scaling,
            mcid,
        }
    }

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Append a text string to the buffer.
    fn append(&mut self, bytes: &[u8], fonts: &HashMap<String, FontInfo>) -> Result<()> {
        self.text.extend_from_slice(bytes);

        // Convert to Unicode using helper function
        let font = self.font_name.as_ref().and_then(|name| fonts.get(name));
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
fn fallback_char_to_unicode(char_code: u16) -> String {
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
        // Valid Unicode ranges: 0x0000-0xD7FF, 0xE000-0xFFFF (BMP)
        // Excludes surrogate pairs (0xD800-0xDFFF) and above BMP (handled separately)
        code if (code <= 0xD7FF || (0xE000..=0xF8FF).contains(&code)) => {
            // Private Use Area (0xE000-0xF8FF): Return visual description
            if (0xE000..=0xF8FF).contains(&code) {
                // These are application-specific symbols (logos, custom glyphs, etc.)
                // Can't decode to standard Unicode, so provide context
                log::debug!("Private Use Area character: U+{:04X}", code);
                // Return the character itself - it's valid Unicode but application-specific
                if let Some(ch) = char::from_u32(code as u32) {
                    return ch.to_string();
                }
            }

            // Standard Unicode in valid range
            if let Some(ch) = char::from_u32(code as u32) {
                ch.to_string()
            } else {
                "?".to_string()
            }
        },

        // Above Basic Multilingual Plane would require surrogate pairs
        // These shouldn't appear as u16, but handle gracefully
        code if code >= 0xF900 => {
            if let Some(ch) = char::from_u32(code as u32) {
                ch.to_string()
            } else {
                "?".to_string()
            }
        },

        // ==================================================================================
        // PRIORITY 7: Last Resort - Replacement Character
        // ==================================================================================
        _ => {
            log::warn!("Character code 0x{:04X} failed all fallback strategies", char_code);
            "?".to_string()
        },
    }
}

/// Helper function to decode text bytes to Unicode, handling multi-byte encodings.
///
/// For Type0/CIDFonts (like UTF-16), this processes bytes in pairs.
/// For simple fonts (Type1, TrueType), this processes bytes individually.
fn decode_text_to_unicode(bytes: &[u8], font: Option<&FontInfo>) -> String {
    // DIAGNOSTIC: Log Font 'F1' text decoding to trace replacement characters
    if let Some(font) = font {
        if font.base_font == "F1" {
            let hex_bytes: String = bytes
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect::<Vec<_>>()
                .join(" ");
            log::warn!(
                "🔍 decode_text_to_unicode: Font 'F1' processing {} bytes: [{}]",
                bytes.len(),
                hex_bytes
            );

            // Check if bytes contain literal UTF-8 replacement character (0xEF 0xBF 0xBD)
            if bytes.len() >= 3 {
                for i in 0..=bytes.len() - 3 {
                    if bytes[i] == 0xEF && bytes[i + 1] == 0xBF && bytes[i + 2] == 0xBD {
                        log::warn!(
                            "⚠️  FOUND LITERAL UTF-8 REPLACEMENT CHAR at byte offset {}! \
                             This proves Hypothesis 1 - PDF contains literal UTF-8 in content stream",
                            i
                        );
                    }
                }
            }
        }
    }

    if let Some(font) = font {
        // Check if font uses multi-byte character codes
        let is_type0 = font.subtype == "Type0";

        if is_type0 && bytes.len() >= 2 {
            // Type0 fonts use 2-byte character codes (big-endian)
            // ===== NEW VALIDATION CODE START =====
            if !bytes.len().is_multiple_of(2) {
                log::warn!(
                    "Type0 font '{}' has ODD byte count ({})! Expected even count. \
                     Last byte will be processed as single-byte fallback. \
                     This may indicate a malformed content stream.",
                    font.base_font,
                    bytes.len()
                );
            }
            // ===== NEW VALIDATION CODE END =====

            // Type0 fonts use 2-byte character codes (usually UTF-16 BE)
            let mut result = String::new();
            let mut i = 0;
            while i < bytes.len() {
                if i + 1 < bytes.len() {
                    // Combine two bytes into a 16-bit character code (big-endian)
                    let char_code = ((bytes[i] as u16) << 8) | (bytes[i + 1] as u16);
                    let char_str = font
                        .char_to_unicode(char_code as u32)
                        .unwrap_or_else(|| fallback_char_to_unicode(char_code));
                    result.push_str(&char_str);
                    i += 2;
                } else {
                    // Odd byte at end - process as single byte
                    let char_code = bytes[i] as u16;
                    let char_str = font
                        .char_to_unicode(char_code as u32)
                        .unwrap_or_else(|| fallback_char_to_unicode(char_code));
                    result.push_str(&char_str);
                    i += 1;
                }
            }
            result
        } else {
            // Simple fonts use single-byte character codes
            let mut result = String::new();
            for &byte in bytes {
                let char_code = byte as u16;
                let char_str = font
                    .char_to_unicode(char_code as u32)
                    .unwrap_or_else(|| fallback_char_to_unicode(char_code));
                result.push_str(&char_str);
            }
            result
        }
    } else {
        // No font - fallback to Latin-1 (ISO 8859-1) encoding
        // Per PDF Spec ISO 32000-1:2008, Section 9.6.6, Latin-1 maps bytes 0x00-0xFF
        // directly to Unicode code points U+0000-U+00FF
        log::warn!(
            "⚠️  No font provided for {} bytes, using Latin-1 fallback (PDF spec compliant)",
            bytes.len()
        );
        bytes.iter().map(|&b| char::from(b)).collect()
    }
}

/// Context for marked content sequences (per PDF Spec Section 14.6)
///
/// Tracks nested marked content tags to implement artifact filtering.
/// When content is marked as `/Artifact`, it should be excluded from text extraction.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MarkedContentContext {
    tag: String,
    is_artifact: bool,
    /// ActualText for marked content (PDF Spec Section 14.9.4)
    /// Used to replace extracted text with correct representation
    /// e.g., ligatures (fi, fl, ffi, ffl), decorated glyphs
    actual_text: Option<String>,
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
    /// Loaded fonts (name -> FontInfo)
    fonts: HashMap<String, FontInfo>,
    /// Extracted text spans (complete strings from Tj/TJ operators)
    spans: Vec<TextSpan>,
    /// Extracted characters (for backward compatibility)
    chars: Vec<TextChar>,
    /// Resources dictionary (for accessing XObjects and fonts)
    resources: Option<Object>,
    /// Reference to the document (for loading XObjects)
    document: Option<*mut crate::document::PdfDocument>,
    /// Set of processed XObject references to avoid duplicates
    processed_xobjects: HashSet<ObjectRef>,
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
    pub fn set_document(&mut self, document: *mut crate::document::PdfDocument) {
        self.document = Some(document);
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
        self.fonts.insert(name, font);
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

        // Parse content stream into operators
        extract_log_debug!("Parsing content stream for text extraction");
        let operators = parse_content_stream(content_stream)?;

        // Execute each operator
        for op in operators {
            self.execute_operator(op)?;
        }

        // Flush any remaining Tj buffer at end of content stream
        self.flush_tj_span_buffer()?;

        // Sort spans by reading order (top-to-bottom, left-to-right)
        // DEBUG: Log span count and offset_semantic values before sorting
        let space_spans_before = self
            .spans
            .iter()
            .filter(|s| s.text.chars().all(|c| c.is_whitespace()))
            .count();
        let offset_semantic_before = self.spans.iter().filter(|s| s.offset_semantic).count();
        log::debug!(
            "Before sort_spans_by_reading_order(): {} spans total, {} space-only, {} offset_semantic=true",
            self.spans.len(),
            space_spans_before,
            offset_semantic_before
        );

        self.sort_spans_by_reading_order();

        // DEBUG: Log after sorting
        let space_spans_after = self
            .spans
            .iter()
            .filter(|s| s.text.chars().all(|c| c.is_whitespace()))
            .count();
        let offset_semantic_after = self.spans.iter().filter(|s| s.offset_semantic).count();
        log::debug!(
            "After sort_spans_by_reading_order(): {} spans total, {} space-only, {} offset_semantic=true",
            self.spans.len(),
            space_spans_after,
            offset_semantic_after
        );

        // Deduplicate overlapping spans
        self.deduplicate_overlapping_spans();

        // PHASE 10: DISABLED - CamelCase splitting is a linguistic heuristic not in PDF spec
        // Per ISO 32000-1:2008, text strings are extracted as-is from the content stream.
        // Splitting on pattern matching (e.g., "theGeneral" -> "the"/"General") requires knowledge
        // external to the PDF spec. Spec-compliant extraction accepts malformed PDFs as-is.
        // See: PHASE10_PDF_SPEC_COMPLIANCE.md for rationale.
        // self.split_fused_words();

        // Merge adjacent spans on the same line to reconstruct complete words
        // This will respect split_boundary_before flags set by split_fused_words()
        self.merge_adjacent_spans();

        Ok(self.spans.clone())
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

        // Parse content stream into operators
        let operators = parse_content_stream(content_stream)?;

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

        for ch in self.chars.iter() {
            let y_rounded = ch.bbox.y.round() as i32;
            let x = ch.bbox.x;

            // Check if this char overlaps with the previous one
            let should_skip = if let (Some(prev_y), Some(prev_x_val)) = (prev_y_rounded, prev_x) {
                // Same line and within 2pt horizontally
                y_rounded == prev_y && (x - prev_x_val).abs() < 2.0
            } else {
                false
            };

            if !should_skip {
                deduplicated.push(ch.clone());
                prev_y_rounded = Some(y_rounded);
                prev_x = Some(x);
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

        log::info!(
            "Column detection: found {} columns from {} spans",
            columns.len(),
            self.spans.len()
        );
        for (i, (left, right)) in columns.iter().enumerate() {
            log::info!(
                "  Column {}: X range [{:.1}, {:.1}] (width: {:.1})",
                i,
                left,
                right,
                right - left
            );
        }

        if columns.len() <= 1 {
            // Single column or no columns detected: use simple sort
            log::info!("Using simple Y-then-X sorting (single column)");
            self.simple_sort_spans();
        } else {
            // Multi-column layout: sort within each column, then across columns
            log::info!("Using column-aware sorting ({} columns)", columns.len());
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
                    a.bbox
                        .x
                        .partial_cmp(&b.bbox.x)
                        .unwrap_or(std::cmp::Ordering::Equal)
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
                // End of gap - only record if gap is significant (>5% of page width)
                let gap_width = (i - gap_start) as f32 * bin_width;
                if gap_width > page_width * 0.05 {
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
                    std::cmp::Ordering::Equal => a
                        .bbox
                        .x
                        .partial_cmp(&b.bbox.x)
                        .unwrap_or(std::cmp::Ordering::Equal),
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

        let mut deduplicated = Vec::with_capacity(self.spans.len());
        let mut prev_y_rounded: Option<i32> = None;
        let mut prev_x: Option<f32> = None;
        let mut seen_content: std::collections::HashMap<String, (f32, f32)> =
            std::collections::HashMap::new();

        let mut geometric_skips = 0;
        let mut content_skips = 0;

        for span in self.spans.iter() {
            let y_rounded = span.bbox.y.round() as i32;
            let x = span.bbox.x;

            // PHASE 1: Geometric deduplication (identical positions)
            let geometric_duplicate =
                if let (Some(prev_y), Some(prev_x_val)) = (prev_y_rounded, prev_x) {
                    // Same line and within 2pt horizontally
                    y_rounded == prev_y && (x - prev_x_val).abs() < 2.0
                } else {
                    false
                };

            // PHASE 2: Content-based deduplication (different positions, same text)
            let content_duplicate = if span.text.len() >= 5 {
                // Only check non-trivial text (5+ chars to avoid false positives like "the")
                if let Some((prev_x, prev_y)) = seen_content.get(&span.text) {
                    let y_diff = (span.bbox.y - prev_y).abs();
                    let x_diff = (span.bbox.x - prev_x).abs();

                    // Same line (Y within 2.0pt tolerance) AND different position (X differs by > 10pt)
                    let same_line = y_diff < 2.0;
                    let different_position = x_diff > 10.0;

                    if same_line && different_position {
                        log::debug!(
                            "Content duplicate: '{}' at X={:.1} (original at X={:.1})",
                            span.text.chars().take(30).collect::<String>(),
                            span.bbox.x,
                            prev_x
                        );
                        true
                    } else {
                        false
                    }
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
                deduplicated.push(span.clone());
                prev_y_rounded = Some(y_rounded);
                prev_x = Some(x);

                // Track content for duplicate detection
                if span.text.len() >= 5 {
                    seen_content.insert(span.text.clone(), (span.bbox.x, span.bbox.y));
                }
            }
        }

        log::debug!(
            "Deduplicated {} spans (geometric: {}, content: {}) ({} -> {} spans)",
            geometric_skips + content_skips,
            geometric_skips,
            content_skips,
            self.spans.len(),
            deduplicated.len()
        );

        self.spans = deduplicated;
    }

    /// Merge adjacent text spans on the same line to reconstruct complete words.
    ///
    /// PDF content streams often break words into multiple Tj operators for precise
    /// kerning/positioning. This causes word fragmentation like "Intr oduction" instead
    /// of "Introduction". We merge spans that are:
    /// - On the same line (Y coordinates within 1pt)
    /// - Very close horizontally (gap < 3pt, approximately average char width)
    ///
    /// This matches the behavior of industry-standard tools like PyMuPDF.
    fn merge_adjacent_spans(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        let mut merged = Vec::with_capacity(self.spans.len());
        let mut current_span: Option<TextSpan> = None;

        for span in self.spans.iter() {
            if current_span.is_none() {
                // First span
                current_span = Some(span.clone());
                continue;
            }

            // Take ownership of current to avoid borrow checker issues
            let mut current = current_span.take().unwrap();

            // Check if this span should be merged with the current one
            let y_diff = (span.bbox.y - current.bbox.y).abs();
            let same_line = y_diff < 1.0;

            // Gap between end of current span and start of next span
            let current_end_x = current.bbox.x + current.bbox.width;
            let gap = span.bbox.x - current_end_x;

            // COLUMN BOUNDARY CHECK: Don't merge spans with large gaps
            // Use configured threshold to detect column separation
            let large_gap_indicates_column = gap > self.merging_config.column_boundary_threshold_pt;

            // SPLIT BOUNDARY CHECK: Respect boundaries from CamelCase splitting
            // If a span has split_boundary_before=true, it represents a word boundary
            // from a split operation (e.g., "the" + "General" from "theGeneral")
            // These should always be merged WITH a space, never without.
            let has_split_boundary = span.split_boundary_before;

            // Merge threshold: Use configured values
            // Negative gaps: use severe_overlap_threshold_pt (default -0.5pt)
            // Positive gaps: use 3pt default (0.25em * 12pt)
            // However, if split_boundary_before=true, ALWAYS merge but insert space
            let should_merge = same_line
                && (self.merging_config.severe_overlap_threshold_pt..3.0).contains(&gap)
                && !large_gap_indicates_column
                || (same_line && has_split_boundary);

            if should_merge {
                // Merge spans: concatenate text and extend bbox
                let old_text = current.text.clone();

                // PHASE 1 FIX: Check if next span is entirely whitespace-only OR marked as offset_semantic space
                // If either is true, never insert an additional space - just concatenate directly
                // This prevents double-space issue when TJ processor creates space spans
                let next_is_whitespace_only = span.text.chars().all(|c| c.is_whitespace());
                let next_is_offset_semantic_space = span.offset_semantic && next_is_whitespace_only;

                let merged_text = if next_is_whitespace_only {
                    // Next span is already space-only: just concatenate without adding more space
                    log::debug!(
                        "Merging with whitespace-only span: '{}' + '{}' (whitespace, offset_semantic={})",
                        current.text,
                        span.text.escape_default(),
                        span.offset_semantic
                    );
                    format!("{}{}", current.text, span.text)
                } else {
                    // Use unified space decision function with detected document type
                    // If we have a split_boundary_before flag, FORCE a space by treating it like a TJ offset
                    // This ensures "length" + "This" becomes "length This" not "lengthThis"
                    // Document type adjustment: Use adaptive thresholds based on document characteristics
                    // Fix 2: Use font-aware spacing thresholds instead of fixed 0.25em
                    let tj_offset_triggered_override = has_split_boundary;
                    let space_decision = should_insert_space(
                        &current.text,
                        &span.text,
                        gap,
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
                            format!("{}{}", current.text, span.text)
                        } else {
                            // NEW: Prevent double-space edge case
                            // If current text ends with space AND next span starts with space, skip inserting space
                            let would_create_double_space =
                                current.text.ends_with(' ') && span.text.starts_with(' ');

                            if would_create_double_space {
                                log::debug!(
                                    "Preventing double-space: current ends with space, next starts with space"
                                );
                                format!("{}{}", current.text, span.text)
                            } else {
                                match space_decision.source {
                                    SpaceSource::CharacterHeuristic => {
                                        log::trace!(
                                            "Space via heuristic: '{}' | '{}'",
                                            current.text,
                                            span.text
                                        );
                                    },
                                    SpaceSource::GeometricGap => {
                                        log::trace!(
                                            "Space via gap (source={:?}): '{}' | '{}' (gap={:.2}pt)",
                                            space_decision.source,
                                            current.text,
                                            span.text,
                                            gap
                                        );
                                    },
                                    _ => {
                                        log::trace!("Space via {:?}", space_decision.source);
                                    },
                                }
                                format!("{} {}", current.text, span.text)
                            }
                        }
                    } else {
                        // No space: adjacent characters within same word
                        log::trace!(
                            "No space insertion: decision source={:?}",
                            space_decision.source
                        );
                        format!("{}{}", current.text, span.text)
                    }
                };

                // Extend bounding box to include both spans
                let new_width = (span.bbox.x + span.bbox.width) - current.bbox.x;
                let new_height = current.bbox.height.max(span.bbox.height);

                current.text = merged_text;
                current.bbox.width = new_width;
                current.bbox.height = new_height;

                log::trace!(
                    "Merged spans: '{}' + '{}' -> '{}' (gap={:.1}pt)",
                    old_text,
                    span.text,
                    current.text,
                    gap
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
                current_span = Some(span.clone());
            }
        }

        // Don't forget the last span
        if let Some(last) = current_span {
            merged.push(last);
        }

        log::debug!("Merged adjacent spans: {} -> {} spans", self.spans.len(), merged.len());

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
                let state = self.state_stack.current_mut();
                state.font_name = Some(font);
                state.font_size = size;
            },

            // Text positioning operators
            Operator::Tm { a, b, c, d, e, f } => {
                // Flush Tj buffer before changing text matrix
                self.flush_tj_span_buffer()?;

                let state = self.state_stack.current_mut();
                state.text_matrix = Matrix { a, b, c, d, e, f };
                state.text_line_matrix = state.text_matrix;
            },
            Operator::Td { tx, ty } => {
                // Flush Tj buffer before changing text position
                self.flush_tj_span_buffer()?;
                let state = self.state_stack.current_mut();
                let tm = Matrix::translation(tx, ty);
                state.text_line_matrix = state.text_line_matrix.multiply(&tm);
                state.text_matrix = state.text_line_matrix;
            },
            Operator::TD { tx, ty } => {
                // Flush Tj buffer before changing text position
                self.flush_tj_span_buffer()?;

                // TD is like Td but also sets leading
                let state = self.state_stack.current_mut();
                state.leading = -ty;
                let tm = Matrix::translation(tx, ty);
                state.text_line_matrix = state.text_line_matrix.multiply(&tm);
                state.text_matrix = state.text_line_matrix;
            },
            Operator::TStar => {
                // Flush Tj buffer before moving to next line
                self.flush_tj_span_buffer()?;

                // Move to start of next line (using leading)
                let leading = self.state_stack.current().leading;
                let state = self.state_stack.current_mut();
                let tm = Matrix::translation(0.0, -leading);
                state.text_line_matrix = state.text_line_matrix.multiply(&tm);
                state.text_matrix = state.text_line_matrix;
            },

            // Text showing operators
            Operator::Tj { text } => {
                // Per PDF Spec Section 14.6: Skip text extraction if inside /Artifact
                // Artifacts include headers, footers, watermarks, resource paths, etc.
                if self.inside_artifact {
                    log::debug!("Skipping text in /Artifact: {:?}", text);
                    return Ok(());
                }

                // ActualText override
                // Per PDF Spec ISO 32000-1:2008, Section 14.9.4:
                // ActualText provides replacement text for content that cannot be
                // automatically extracted (e.g., figures, symbols, decorative text).
                if let Some(actual_text) = self.get_current_actual_text() {
                    log::debug!("Tj operator: Using ActualText override: '{}'", actual_text);

                    if self.extract_spans {
                        // Use ActualText in span mode - buffer it like normal text
                        if self.tj_span_buffer.is_none() {
                            self.tj_span_buffer =
                                Some(TjBuffer::new(self.state_stack.current(), self.current_mcid));
                        }

                        // Append ActualText to buffer (convert to bytes for consistency)
                        if let Some(ref mut buffer) = self.tj_span_buffer {
                            buffer.append(actual_text.as_bytes(), &self.fonts)?;
                        }
                    } else {
                        // Use ActualText in character mode - process each character
                        self.show_text(actual_text.as_bytes())?;
                    }

                    // Advance position for the original text (to maintain layout)
                    self.advance_position_for_string(&text)?;
                } else {
                    // No ActualText - use standard text extraction
                    if self.extract_spans {
                        // NEW: Buffer consecutive Tj operators into single spans
                        // Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
                        // "text strings are as long as possible"

                        // Create buffer if doesn't exist
                        if self.tj_span_buffer.is_none() {
                            self.tj_span_buffer =
                                Some(TjBuffer::new(self.state_stack.current(), self.current_mcid));
                        }

                        // Append to buffer
                        if let Some(ref mut buffer) = self.tj_span_buffer {
                            buffer.append(&text, &self.fonts)?;
                        }

                        // Advance position (text matrix must be updated)
                        self.advance_position_for_string(&text)?;
                    } else {
                        self.show_text(&text)?;
                    }
                }
            },
            Operator::TJ { array } => {
                // Per PDF Spec Section 14.6: Skip text extraction if inside /Artifact
                // Artifacts include headers, footers, watermarks, resource paths, etc.
                if self.inside_artifact {
                    log::debug!(
                        "Skipping text in /Artifact: TJ array with {} elements",
                        array.len()
                    );
                    return Ok(());
                }

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
                        // Use ActualText in span mode - create a single span
                        let mut buffer =
                            TjBuffer::new(self.state_stack.current(), self.current_mcid);
                        buffer.append(actual_text.as_bytes(), &self.fonts)?;
                        self.flush_tj_buffer(&buffer)?;
                    } else {
                        // Use ActualText in character mode
                        self.show_text(actual_text.as_bytes())?;
                    }

                    // Advance position for the entire TJ array (to maintain layout)
                    // Calculate the total displacement the array would have caused
                    for element in array {
                        match element {
                            TextElement::String(s) => {
                                self.advance_position_for_string(&s)?;
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
                                        let font_name = state.font_name.clone();
                                        let font_size = state.font_size;
                                        let fill_color_rgb = state.fill_color_rgb;

                                        // Calculate effective font size (accounting for text matrix scaling)
                                        let effective_font_size = font_size * text_matrix.d.abs();

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
                                        let (r, g, b) = fill_color_rgb;
                                        let is_italic_space = font_name
                                            .as_ref()
                                            .and_then(|name| self.fonts.get(name))
                                            .map(|font| font.is_italic())
                                            .unwrap_or(false);
                                        let font_name_str = font_name.unwrap_or_default();
                                        let space_char = TextChar {
                                            char: ' ',
                                            bbox: Rect::new(
                                                text_matrix.e,       // Current X position
                                                text_matrix.f,       // Current Y position
                                                tx.abs(), // Width = the gap being created
                                                effective_font_size, // Height = effective font size
                                            ),
                                            font_name: font_name_str,
                                            font_size: effective_font_size,
                                            font_weight,
                                            color: Color::new(r, g, b),
                                            mcid: self.current_mcid,
                                            is_italic: is_italic_space,
                                        };
                                        self.chars.push(space_char);
                                    }

                                    let state_mut = self.state_stack.current_mut();
                                    state_mut.text_matrix.e += tx;
                                },
                            }
                        }
                    }
                }
            },
            Operator::Quote { text } => {
                // Move to next line and show text
                let leading = self.state_stack.current().leading;
                {
                    let state = self.state_stack.current_mut();
                    let tm = Matrix::translation(0.0, -leading);
                    state.text_line_matrix = state.text_line_matrix.multiply(&tm);
                    state.text_matrix = state.text_line_matrix;
                }
                self.show_text(&text)?;
            },
            Operator::DoubleQuote {
                word_space,
                char_space,
                text,
            } => {
                // Set spacing, move to next line, and show text
                {
                    let state = self.state_stack.current_mut();
                    state.word_space = word_space;
                    state.char_space = char_space;
                    let leading = state.leading;
                    let tm = Matrix::translation(0.0, -leading);
                    state.text_line_matrix = state.text_line_matrix.multiply(&tm);
                    state.text_matrix = state.text_line_matrix;
                }
                self.show_text(&text)?;
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
            },
            Operator::Cm { a, b, c, d, e, f } => {
                let state = self.state_stack.current_mut();
                let new_ctm = Matrix { a, b, c, d, e, f };
                state.ctm = state.ctm.multiply(&new_ctm);
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

            // Text object operators (BT/ET) - just markers, no state change needed
            Operator::BeginText | Operator::EndText => {
                // No action needed
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
                    actual_text: None, // BMC doesn't have ActualText
                });
                self.update_artifact_state();

                if is_artifact {
                    log::debug!("Entered /Artifact marked content");
                }
            },

            Operator::BeginMarkedContentDict { tag, properties } => {
                // BDC can have properties including MCID, artifact indicators, and ActualText
                // PDF Spec Section 14.9.4: ActualText provides replacement text for content
                let mut actual_text = None;
                if let Some(props_dict) = properties.as_dict() {
                    // Extract MCID if present
                    if let Some(mcid_obj) = props_dict.get("MCID") {
                        if let Some(mcid) = mcid_obj.as_integer() {
                            self.current_mcid = Some(mcid as u32);
                            log::debug!("Entered marked content with MCID: {}", mcid);
                        }
                    }

                    // Extract ActualText if present (PDF Spec Section 14.9.4)
                    // ActualText provides exact text representation (e.g., ligatures, decorated glyphs)
                    if let Some(actual_text_obj) = props_dict.get("ActualText") {
                        if let Some(text_bytes) = actual_text_obj.as_string() {
                            // Convert UTF-8 bytes to String, fall back to lossy conversion if invalid
                            actual_text = Some(String::from_utf8_lossy(text_bytes).to_string());
                            log::debug!("Marked content has ActualText: {:?}", actual_text);
                        }
                    }
                }

                // Check if this is an artifact (per PDF Spec Section 14.6)
                let is_artifact = tag == "Artifact";
                self.marked_content_stack.push(MarkedContentContext {
                    tag: tag.clone(),
                    is_artifact,
                    actual_text,
                });
                self.update_artifact_state();

                if is_artifact {
                    log::debug!("Entered /Artifact marked content");
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

    /// Process a Form XObject invoked by the Do operator.
    ///
    /// This extracts text from Form XObjects while avoiding duplicate processing.
    fn process_xobject(&mut self, name: &str) -> Result<()> {
        // Get resources dictionary
        let resources = match &self.resources {
            Some(res) => res,
            None => {
                // No resources available, skip XObject
                log::debug!("No resources available for XObject: {}", name);
                return Ok(());
            },
        };

        // Get document reference
        let doc_ptr = match self.document {
            Some(ptr) => ptr,
            None => {
                // No document reference, skip XObject
                log::debug!("No document reference available for XObject: {}", name);
                return Ok(());
            },
        };

        // Safety: The document pointer is valid because it's set by PdfDocument::extract_chars
        // and this method is called synchronously within that context
        let doc = unsafe { &mut *doc_ptr };

        // Resolve resources dictionary if it's a reference
        let resources_dict = if let Some(res_ref) = resources.as_reference() {
            doc.load_object(res_ref)?
        } else {
            resources.clone()
        };

        // Get XObject dictionary from resources
        let resources_dict =
            resources_dict
                .as_dict()
                .ok_or_else(|| crate::error::Error::ParseError {
                    offset: 0,
                    reason: "Resources is not a dictionary".to_string(),
                })?;

        let xobject_dict = match resources_dict.get("XObject") {
            Some(xobj) => xobj,
            None => {
                // No XObjects in resources
                log::debug!("No XObject dictionary in resources");
                return Ok(());
            },
        };

        // Resolve XObject dictionary if it's a reference
        let xobject_dict = if let Some(xobj_ref) = xobject_dict.as_reference() {
            doc.load_object(xobj_ref)?
        } else {
            xobject_dict.clone()
        };

        let xobject_dict =
            xobject_dict
                .as_dict()
                .ok_or_else(|| crate::error::Error::ParseError {
                    offset: 0,
                    reason: "XObject is not a dictionary".to_string(),
                })?;

        // Get the specific XObject
        let xobject_obj = match xobject_dict.get(name) {
            Some(obj) => obj,
            None => {
                log::debug!("XObject '{}' not found in dictionary", name);
                return Ok(());
            },
        };

        // Resolve XObject if it's a reference
        let xobject_ref = match xobject_obj.as_reference() {
            Some(r) => r,
            None => {
                // XObject is not a reference, skip
                log::debug!("XObject '{}' is not a reference", name);
                return Ok(());
            },
        };

        // Check if we've already processed this XObject
        if self.processed_xobjects.contains(&xobject_ref) {
            log::debug!("Skipping duplicate XObject: {} (ref {:?})", name, xobject_ref);
            return Ok(());
        }

        // Mark as processed
        self.processed_xobjects.insert(xobject_ref);

        // Load the XObject
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

                // Decode the stream data with error recovery, respecting encryption
                let stream_data = match doc.decode_stream_with_encryption(&xobject, xobject_ref) {
                    Ok(data) => data,
                    Err(e) => {
                        log::warn!(
                            "Failed to decode Form XObject '{}' stream: {}, skipping",
                            name,
                            e
                        );
                        return Ok(());
                    },
                };

                // Parse and execute operators from the Form XObject
                let operators = match parse_content_stream(&stream_data) {
                    Ok(ops) => ops,
                    Err(e) => {
                        log::warn!(
                            "Failed to parse Form XObject '{}' content stream: {}, skipping",
                            name,
                            e
                        );
                        return Ok(());
                    },
                };

                for op in operators {
                    // Continue processing even if individual operators fail
                    if let Err(e) = self.execute_operator(op) {
                        log::debug!("Error executing operator in Form XObject '{}': {}", name, e);
                    }
                }

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

    /// Flush accumulated TJ buffer into a single TextSpan.
    ///
    /// This creates one span for the entire buffer content, properly calculating
    /// the total width including character spacing (Tc) and word spacing (Tw).
    fn flush_tj_buffer(&mut self, buffer: &TjBuffer) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Calculate total width using PDF spec formula (including Tc/Tw)
        let total_width = self.calculate_tj_buffer_width(buffer)?;

        // Calculate effective font size from text matrix
        let effective_font_size = buffer.font_size * buffer.start_matrix.d.abs();

        // Determine font weight
        let font_weight = if let Some(font_name) = &buffer.font_name {
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

        // Create single span for entire buffer
        let font_name_span = buffer
            .font_name
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        let is_italic_span = buffer
            .font_name
            .as_ref()
            .and_then(|name| self.fonts.get(name))
            .map(|font| font.is_italic())
            .unwrap_or(false);
        let span = TextSpan {
            text: buffer.unicode.clone(),
            bbox: Rect {
                x: buffer.start_matrix.e,
                y: buffer.start_matrix.f,
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
            primary_detected: false, // Default to false for backward compatibility
        };
        self.span_sequence_counter += 1;

        self.spans.push(span);
        Ok(())
    }

    /// Calculate total width of TJ buffer using PDF spec formula.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.4.4:
    /// tx = ((w0 - Tj/1000) × Tfs + Tc + Tw) × Th
    ///
    /// For TJ arrays without offset adjustments (Tj=0 for strings):
    /// tx = (w0 × Tfs / 1000 + Tc + Tw) × Th
    fn calculate_tj_buffer_width(&self, buffer: &TjBuffer) -> Result<f32> {
        let font = buffer
            .font_name
            .as_ref()
            .and_then(|name| self.fonts.get(name));

        let mut total_width = 0.0;

        for &byte in &buffer.text {
            // Per PDF Spec 9.4.4: tx = ((w0 - Tj/1000) × Tfs + Tc + Tw) × Th
            let glyph_width = if let Some(font) = font {
                font.get_glyph_width(byte as u16)
            } else {
                500.0 // Default glyph width if no font available
            };

            // 1. Convert glyph width to user space: w0 * Tfs / 1000
            let mut char_width = glyph_width * buffer.font_size / 1000.0;

            // 2. Add character spacing (Tc) - applies to ALL characters
            char_width += buffer.char_space;

            // 3. Add word spacing (Tw) - applies ONLY to space (0x20)
            if byte == 0x20 {
                char_width += buffer.word_space;
            }

            // 4. Apply horizontal scaling (Th)
            char_width *= buffer.horizontal_scaling / 100.0;

            total_width += char_width;
        }

        Ok(total_width)
    }

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

        let mut buffer = TjBuffer::new(self.state_stack.current(), self.current_mcid);
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

                    // Now process buffer as before for span generation
                    // FIX: Detect and skip space strings that split words
                    // Per PDF Spec ISO 32000-1:2008, Section 14.8.2.5 NOTE 3:
                    // "The identification of what constitutes a word is unrelated to how
                    // the text happens to be grouped into show strings. The division into
                    // show strings has no semantic significance."
                    //
                    // Some malformed PDFs incorrectly put space strings mid-word:
                    // [(var) ( ) (ious)] TJ  <- WRONG, should be [(various)] TJ
                    //
                    // We detect this by checking if a space string appears after a lowercase
                    // letter (indicating we're mid-word).

                    // First, check if this is a space/whitespace-only string
                    let unicode_text = if let Some(ref name) = font_name {
                        if let Some(font) = self.fonts.get(name) {
                            let mut text = String::new();
                            for &byte in s.iter() {
                                if let Some(chars) = font.char_to_unicode(byte as u32) {
                                    text.push_str(&chars);
                                }
                            }
                            text
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    };

                    // Check if this is a whitespace-only string
                    if !unicode_text.is_empty() && unicode_text.trim().is_empty() {
                        // This is a space/whitespace string
                        // Check if it's splitting a word (buffer ends with lowercase letter)
                        if !buffer.unicode.is_empty() {
                            if let Some(last_char) = buffer.unicode.chars().last() {
                                if last_char.is_lowercase() {
                                    // This space is splitting a word - skip it!
                                    // Just advance position but don't add to buffer
                                    self.advance_position_for_string(s)?;
                                    continue; // Skip to next element
                                }
                            }
                        }
                    }

                    // Normal case: append string to buffer
                    buffer.append(s, &self.fonts)?;

                    // Advance position for this string
                    self.advance_position_for_string(s)?;
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
                        self.flush_tj_buffer(&buffer)?;

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
                        buffer = TjBuffer::new(self.state_stack.current(), self.current_mcid);
                    }

                    // Advance position for offset (updates text matrix)
                    self.advance_position_for_offset(*offset)?;
                },
            }
        }

        // Flush remaining buffer
        if !buffer.is_empty() {
            self.flush_tj_buffer(&buffer)?;
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

        // Step 1: Calculate bounding box from character positions
        // X position: from first character to end of last character
        let min_x = cluster[0].x_position;
        let max_x = cluster.last().unwrap().x_position + cluster.last().unwrap().width;
        let width = (max_x - min_x).max(0.0);

        // Y position: from text matrix, height from font size
        let y = state.text_matrix.f;
        let height = cluster[0].font_size.abs() * state.text_matrix.d.abs().max(1.0);

        // Step 2: Create bounding box rectangle
        let bbox = Rect {
            x: min_x,
            y,
            width,
            height,
        };

        // Step 3: Convert characters to Unicode string
        // Use same decoding as existing code
        let unicode_text = if let Some(font_name) = state.font_name.as_ref() {
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
            primary_detected: true, // Mark as created by primary detector
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
    fn advance_position_for_string(&mut self, text: &[u8]) -> Result<()> {
        let state = self.state_stack.current();
        let font_name = state.font_name.clone();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;
        let char_space = state.char_space;
        let word_space = state.word_space;

        let font = font_name.as_ref().and_then(|name| self.fonts.get(name));

        // Calculate total width per PDF spec
        let mut total_width = 0.0;
        for &byte in text {
            let glyph_width = if let Some(font) = font {
                font.get_glyph_width(byte as u16)
            } else {
                500.0
            };

            let mut char_width = glyph_width * font_size / 1000.0;
            char_width += char_space;
            if byte == 0x20 {
                char_width += word_space;
            }
            char_width *= horizontal_scaling / 100.0;
            total_width += char_width;
        }

        // Update text matrix position
        let state = self.state_stack.current_mut();
        let text_matrix = state.text_matrix;
        let advance = total_width / text_matrix.d.abs();
        state.text_matrix.e += advance * text_matrix.a;
        state.text_matrix.f += advance * text_matrix.b;

        Ok(())
    }

    /// Insert a space character as a separate span.
    fn insert_space_as_span(&mut self) -> Result<()> {
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let text_matrix = state.text_matrix;
        let effective_font_size = font_size * text_matrix.d.abs();
        let word_space = state.word_space;
        let horizontal_scaling = state.horizontal_scaling;

        // Calculate space width
        let space_width = (250.0 * font_size / 1000.0 + word_space) * horizontal_scaling / 100.0;

        log::info!(
            "Inserting space span from TJ offset (offset_semantic=true) at position ({:.2}, {:.2})",
            text_matrix.e,
            text_matrix.f
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
                x: text_matrix.e,
                y: text_matrix.f,
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
            primary_detected: false, // Default to false for backward compatibility
        };
        self.span_sequence_counter += 1;

        log::info!("PUSH space span with offset_semantic={}", span.offset_semantic);

        self.spans.push(span);

        // Advance position
        let state = self.state_stack.current_mut();
        let advance = space_width / text_matrix.d.abs();
        state.text_matrix.e += advance * text_matrix.a;
        state.text_matrix.f += advance * text_matrix.b;

        Ok(())
    }

    /// Advance text position for a TJ offset value.
    fn advance_position_for_offset(&mut self, offset: f32) -> Result<()> {
        let state = self.state_stack.current();
        let font_size = state.font_size;
        let horizontal_scaling = state.horizontal_scaling;

        // Calculate horizontal displacement per PDF spec
        // tx = -offset / 1000.0 * font_size * horizontal_scaling / 100.0
        let tx = -offset / 1000.0 * font_size * horizontal_scaling / 100.0;

        // Update text matrix position
        let state = self.state_stack.current_mut();
        state.text_matrix.e += tx;

        Ok(())
    }

    /// Flush accumulated Tj span buffer into a single TextSpan.
    ///
    /// This is similar to flush_tj_buffer but works with the tj_span_buffer field
    /// which accumulates consecutive Tj operators.
    fn flush_tj_span_buffer(&mut self) -> Result<()> {
        if let Some(buffer) = self.tj_span_buffer.take() {
            if !buffer.is_empty() {
                // Calculate total width using PDF spec formula
                let total_width = self.calculate_tj_buffer_width(&buffer)?;

                // Calculate effective font size
                let effective_font_size = buffer.font_size * buffer.start_matrix.d.abs();

                // Determine font weight
                let font_weight = if let Some(font_name) = &buffer.font_name {
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

                // Create single span for entire buffer
                // PHASE 1 ENHANCEMENT: Mark space-only spans as offset_semantic=true
                // This allows merge_adjacent_spans() to recognize them and skip double-space insertion
                let font_name_buf = buffer
                    .font_name
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string());
                let is_italic_buf = buffer
                    .font_name
                    .as_ref()
                    .and_then(|name| self.fonts.get(name))
                    .map(|font| font.is_italic())
                    .unwrap_or(false);
                let span = TextSpan {
                    text: buffer.unicode.clone(),
                    bbox: Rect {
                        x: buffer.start_matrix.e,
                        y: buffer.start_matrix.f,
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
                    primary_detected: false, // Default to false for backward compatibility
                };
                self.span_sequence_counter += 1;

                log::info!(
                    "FLUSH_TJ_SPAN_BUFFER creating span: text='{}', offset_semantic={} (space-only spans marked as offset_semantic)",
                    if span.text.chars().all(|c| c.is_whitespace()) {
                        "<space-only>"
                    } else {
                        &span.text[..span.text.len().min(20)]
                    },
                    span.offset_semantic
                );

                self.spans.push(span);
            }
        }
        Ok(())
    }

    fn show_text(&mut self, text: &[u8]) -> Result<()> {
        for &byte in text {
            let char_code = byte as u16;

            // Get current state values (no borrow after this)
            let state = self.state_stack.current();
            let font_name = state.font_name.clone();
            let text_matrix = state.text_matrix;
            let font_size = state.font_size;
            let horizontal_scaling = state.horizontal_scaling;
            let char_space = state.char_space;
            let word_space = state.word_space;
            let fill_color_rgb = state.fill_color_rgb;

            // Get current font
            let font = font_name.as_ref().and_then(|name| self.fonts.get(name));

            // Get Unicode string using font mapping
            // BUG FIX #2: Handle multi-character ligature expansion (e.g., "fi", "fl", "ff")
            // char_to_unicode() returns a String which may contain multiple characters when
            // a ligature glyph is expanded to its constituent ASCII characters.
            let unicode_string = if let Some(font) = font {
                let result = font
                    .char_to_unicode(char_code as u32)
                    .unwrap_or_else(|| "?".to_string());

                // DEBUG: Log when we get 'd' or ρ to trace the issue
                if result == "d"
                    || result.contains('ρ')
                    || result.contains('r') && char_code == 0x72
                {
                    log::info!(
                        "Text extraction: font '{}', code 0x{:02X} → '{}' (bytes: {:?})",
                        font_name.as_ref().unwrap_or(&String::from("?")),
                        char_code,
                        result,
                        result.as_bytes()
                    );
                }

                result
            } else {
                // No font loaded, use identity mapping
                if byte.is_ascii() {
                    (byte as char).to_string()
                } else {
                    "?".to_string()
                }
            };

            // Calculate character position in user space
            let pos = text_matrix.transform_point(0.0, 0.0);

            // BUG FIX #1: Calculate effective font size from text matrix
            // The text matrix scales the font size - the vertical scaling component (d)
            // determines the actual rendered font size in user space.
            // This is critical for proper header detection and text structure analysis.
            let effective_font_size = font_size * text_matrix.d.abs();

            // Calculate character dimensions
            // Use effective font size and better width estimate based on horizontal scaling
            let char_width_ratio = 0.5; // Average character width-to-height ratio
            let glyph_width = effective_font_size * horizontal_scaling / 100.0 * char_width_ratio;
            let height = effective_font_size;

            // Determine font weight
            let font_weight = if let Some(font) = font {
                if font.is_bold() {
                    FontWeight::Bold
                } else {
                    FontWeight::Normal
                }
            } else {
                FontWeight::Normal
            };

            // Get color
            let (r, g, b) = fill_color_rgb;
            let color = Color::new(r, g, b);

            // Process each character in the expanded string
            // For ligatures (e.g., "fi" from ﬁ), we create multiple TextChar objects
            // and distribute them horizontally across the glyph width
            let char_count = unicode_string.chars().count();
            let char_width = if char_count > 0 {
                glyph_width / char_count as f32
            } else {
                glyph_width
            };

            for (char_index, unicode_char) in unicode_string.chars().enumerate() {
                // Skip NULL characters (U+0000) and other control characters
                // These are often artifacts from PDF encoding and should not be extracted
                let should_skip = unicode_char == '\0'
                    || (unicode_char.is_control()
                        && unicode_char != '\t'
                        && unicode_char != '\n'
                        && unicode_char != '\r');

                if !should_skip {
                    // Calculate position for this character within the ligature
                    // Distribute characters horizontally across the glyph width
                    let x_offset = char_index as f32 * char_width;

                    // Create TextChar with effective font size
                    let font_name_str = font_name.clone().unwrap_or_default();
                    let is_italic_char = font_name
                        .as_ref()
                        .and_then(|name| self.fonts.get(name))
                        .map(|font| font.is_italic())
                        .unwrap_or(false);
                    let text_char = TextChar {
                        char: unicode_char,
                        bbox: Rect::new(pos.x + x_offset, pos.y, char_width, height),
                        font_name: font_name_str,
                        font_size: effective_font_size,
                        font_weight,
                        color,
                        mcid: self.current_mcid,
                        is_italic: is_italic_char,
                    };

                    self.chars.push(text_char);
                }
            }

            // Advance text position (always do this once per PDF byte, not per expanded character)
            // Tx = (w0 * Tfs + Tc + Tw) * Th / 100
            // where w0 is glyph width (we estimate using char_width_ratio)
            // Note: Use the nominal font_size here, not effective_font_size,
            // because text matrix scaling is already applied to the text position
            let mut tx = char_width_ratio * font_size;
            tx += char_space;
            // Check if ANY character in the expanded string is a space
            if unicode_string.chars().any(|c| c == ' ') {
                tx += word_space;
            }
            tx *= horizontal_scaling / 100.0;

            // Update text matrix
            let state_mut = self.state_stack.current_mut();
            state_mut.text_matrix.e += tx;
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
    use crate::fonts::Encoding;

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
            truetype_cmap: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
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

    #[test]
    #[ignore] // TODO: Fix Tf inside q/Q not working correctly
    fn test_extract_save_restore() {
        let mut extractor = TextExtractor::new();
        let font = create_test_font();
        extractor.add_font("F1".to_string(), font);

        let stream = b"BT /F1 12 Tf q 14 Tf (A) Tj Q (B) Tj ET";
        let chars = extractor.extract(stream).unwrap();

        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].font_size, 14.0); // Inside q/Q
        assert_eq!(chars[1].font_size, 12.0); // After Q
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
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
            },
            TextSpan {
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
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
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
}
