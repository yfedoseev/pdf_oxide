//! Text block representation for layout analysis.
//!
//! This module defines structures for representing text elements in a PDF document
//! with their geometric and styling information.

use crate::extractors::text::ArtifactType;
use crate::geometry::{Point, Rect};
use crate::structure::McidScope;
use std::collections::HashMap;

/// A text span (complete string from a Tj/TJ operator).
///
/// This represents text as the PDF specification provides it - complete strings
/// from text showing operators, not individual characters. This is the correct
/// approach per PDF spec ISO 32000-1:2008.
///
/// Extracting complete strings instead of individual characters:
/// - Avoids overlapping character issues
/// - Preserves PDF's text positioning intent
/// - Matches industry best practices
/// - More robust for complex layouts
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct TextSpan {
    /// The complete text string
    pub text: String,
    /// Bounding box of the entire span in PDF coordinates (points)
    pub bbox: Rect,
    /// Font name/family
    pub font_name: String,
    /// Font size in points
    pub font_size: f32,
    /// Font weight (normal or bold)
    pub font_weight: FontWeight,
    /// Font style: italic or normal
    pub is_italic: bool,
    /// Whether the font is monospaced (from PDF font descriptor FixedPitch flag)
    pub is_monospace: bool,
    /// Text color
    pub color: Color,
    /// Marked Content ID (for Tagged PDFs)
    pub mcid: Option<u32>,
    /// Content-stream scope of [`Self::mcid`] (ISO 32000-1:2008 §14.7.4.3).
    ///
    /// MCIDs are scoped to a single content stream — page, Form
    /// XObject, or Tiling Pattern — not to a page globally. When this
    /// span's `mcid` was emitted inside a Form XObject's content
    /// stream, `mcid_scope` is `Form(<form_ref>)`; inside a Tiling
    /// Pattern, `Pattern(<pattern_ref>)`; otherwise `Page(page_index)`
    /// for the page that owns the top-level content stream the span
    /// came from.
    ///
    /// The struct-tree `/ActualText` applier keys lookups by
    /// `(mcid_scope, mcid)` so two Form XObjects on the same page that
    /// each carry MCID 0 do not collide and overwrite each other's
    /// replacements.
    ///
    /// `None` for spans extracted before page-index attribution
    /// completes (e.g. mid-extraction internal spans) or for synthetic
    /// test fixtures.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mcid_scope: Option<McidScope>,
    /// Extraction sequence number
    pub sequence: usize,
    /// If true, this span was created by splitting fused words
    pub split_boundary_before: bool,
    /// If true, this span was created by the TJ processor as a space
    pub offset_semantic: bool,
    /// Character spacing (Tc parameter)
    pub char_spacing: f32,
    /// Word spacing (Tw parameter)
    pub word_spacing: f32,
    /// Horizontal scaling (Tz parameter)
    pub horizontal_scaling: f32,
    /// If true, was created by WordBoundaryDetector primary detection.
    pub primary_detected: bool,
    /// Artifact type classification for filtered content (PDF Spec Section 14.8.2.2)
    pub artifact_type: Option<ArtifactType>,
    /// Per-character advance widths in user-space points.
    /// When non-empty and matching text length, to_chars() uses these
    /// for accurate per-glyph bounding boxes instead of uniform division.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub char_widths: Vec<f32>,
    /// Heading level (1-6) when this span belongs to a document heading.
    /// Populated either from the source PDF's structure tree
    /// (`StructRole::Heading(n)`) or from a font-size-ratio heuristic when
    /// the PDF is untagged. Layout-preserving DOCX export uses this to
    /// emit `<w:pStyle w:val="HeadingN"/>` so the output document
    /// preserves heading semantics for accessibility, navigation, and
    /// outline panes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_level: Option<u8>,
    /// Display rotation of the run in degrees, from `atan2(b, a)` of the composed
    /// text rendering matrix (`T_m × CTM`, ISO 32000-1 §9.4.4), normalised so a
    /// near-quadrant angle snaps to `0` / `90` / `180` / `-90`. `0.0` for ordinary
    /// horizontal text. Reading order segregates non-zero-rotation runs out of the
    /// horizontal flow so they are ordered as their own blocks rather than
    /// interleaved (the axis-aligned assumptions in the row-band / XY-cut sort do
    /// not hold for rotated text).
    #[serde(skip_serializing_if = "is_zero_f32", default)]
    pub rotation_degrees: f32,
    /// Writing mode under which the glyphs in this span were emitted.
    ///
    /// `0` = horizontal (the overwhelming default), `1` = vertical (tategaki
    /// / lateral CJK). Set from `GraphicsState::text_wmode` when the span
    /// is constructed, so each span carries its own writing-mode metadata
    /// even on mixed-mode pages (e.g. horizontal headings above vertical
    /// body copy). The reading-order sort consults this to advance
    /// downward-then-right-to-left within blocks of vertical spans while
    /// leaving horizontal spans on their existing top-to-bottom,
    /// left-to-right path.
    #[serde(skip_serializing_if = "is_zero_u8", default)]
    pub wmode: u8,
}

/// serde skip helper: omit a `0` writing mode (horizontal, the common case)
/// from serialized output so existing fixtures stay unchanged.
pub(crate) fn is_zero_u8(v: &u8) -> bool {
    *v == 0
}

/// serde skip helper: omit a `0.0` rotation (the overwhelming common case) from
/// serialized output so existing fixtures stay unchanged.
pub(crate) fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

impl Default for TextSpan {
    fn default() -> Self {
        Self {
            text: String::new(),
            bbox: Rect::default(),
            font_name: "Helvetica".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            mcid_scope: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            artifact_type: None,
            char_widths: Vec::new(),
            heading_level: None,
            rotation_degrees: 0.0,
            wmode: 0,
        }
    }
}

impl TextSpan {
    /// Decompose the span into individual characters.
    pub fn to_chars(&self) -> Vec<TextChar> {
        let char_count = self.text.chars().count();
        if char_count == 0 {
            return Vec::new();
        }

        // Use per-character widths when available and matching text length;
        // otherwise fall back to uniform division (backward compatible).
        if self.char_widths.len() == char_count {
            let mut x = self.bbox.x;
            self.text
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    let w = self.char_widths[i];
                    let char_x = x;
                    x += w;
                    TextChar {
                        char: c,
                        bbox: Rect::new(char_x, self.bbox.y, w, self.bbox.height),
                        font_name: self.font_name.clone(),
                        font_size: self.font_size,
                        font_weight: self.font_weight,
                        is_italic: self.is_italic,
                        is_monospace: self.is_monospace,
                        color: self.color,
                        mcid: self.mcid,
                        origin_x: char_x,
                        origin_y: self.bbox.y,
                        rotation_degrees: 0.0,
                        advance_width: w,
                        rendered_advance: w,
                        ascent: 0.95 * self.font_size,
                        descent: -0.35 * self.font_size,
                        matrix: Some([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    }
                })
                .collect()
        } else {
            let char_width = self.bbox.width / (char_count as f32);
            self.text
                .chars()
                .enumerate()
                .map(|(i, c)| TextChar {
                    char: c,
                    bbox: Rect::new(
                        self.bbox.x + (i as f32) * char_width,
                        self.bbox.y,
                        char_width,
                        self.bbox.height,
                    ),
                    font_name: self.font_name.clone(),
                    font_size: self.font_size,
                    font_weight: self.font_weight,
                    is_italic: self.is_italic,
                    is_monospace: self.is_monospace,
                    color: self.color,
                    mcid: self.mcid,
                    origin_x: self.bbox.x + (i as f32) * char_width,
                    origin_y: self.bbox.y,
                    rotation_degrees: 0.0,
                    advance_width: char_width,
                    rendered_advance: char_width,
                    ascent: 0.95 * self.font_size,
                    descent: -0.35 * self.font_size,
                    matrix: Some([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                })
                .collect()
        }
    }
}

/// A single character with its position and styling.
///
/// NOTE: This is kept for backward compatibility and special use cases.
/// For normal text extraction, prefer TextSpan which represents complete
/// text strings as the PDF provides them.
///
/// ## Transformation Properties (v0.3.1+)
///
/// TextChar now includes transformation information for precise text positioning:
/// - `origin_x`, `origin_y`: Baseline position (where the character sits)
/// - `rotation_degrees`: Text rotation angle
/// - `advance_width`: Horizontal distance to next character
/// - `matrix`: Full 6-element transformation matrix for advanced use cases
///
/// These properties match industry standards.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct TextChar {
    /// The character itself
    pub char: char,
    /// Bounding box of the character
    pub bbox: Rect,
    /// Font name/family
    pub font_name: String,
    /// Font size in points
    pub font_size: f32,
    /// Font weight (normal or bold)
    pub font_weight: FontWeight,
    /// Font style: italic or normal
    pub is_italic: bool,
    /// Whether the font is monospaced (from PDF font descriptor FixedPitch flag)
    pub is_monospace: bool,
    /// Text color
    pub color: Color,
    /// Marked Content ID (for Tagged PDFs)
    ///
    /// This field stores the MCID if this character was extracted within
    /// a marked content sequence in a Tagged PDF.
    pub mcid: Option<u32>,

    // === Transformation properties (v0.3.1, Issue #27) ===
    /// Baseline origin X coordinate.
    ///
    /// This is the X position where the character's baseline starts,
    /// which is the standard reference point for text positioning in PDFs.
    /// Unlike bbox.x which is the left edge of the glyph, origin_x is
    /// the typographic origin point.
    pub origin_x: f32,

    /// Baseline origin Y coordinate.
    ///
    /// This is the Y position of the character's baseline. For horizontal
    /// text, this is where the bottom of letters like 'a', 'x' sit, while
    /// letters with descenders like 'g', 'y' extend below this line.
    pub origin_y: f32,

    /// Rotation angle in degrees (0-360, clockwise from horizontal).
    ///
    /// Calculated from the text transformation matrix using atan2(b, a).
    /// - 0° = normal horizontal text (left to right)
    /// - 90° = vertical text (top to bottom)
    /// - 180° = upside down text
    /// - 270° = vertical text (bottom to top)
    pub rotation_degrees: f32,

    /// Horizontal advance width (distance to next character position).
    ///
    /// Glyph advance width from font metrics (device space).
    ///
    /// This is the advance for the glyph shape only — it does **not** include
    /// character spacing (Tc), word spacing (Tw), or TJ array adjustments.
    /// For word-boundary detection and the full cursor advance including all
    /// spacing, use [`Self::rendered_advance`].
    pub advance_width: f32,

    /// Actual rendered advance to the next character's origin (device space).
    ///
    /// This is the per-glyph cursor advance including character spacing (Tc)
    /// and word spacing (Tw for U+0020), per the PDF spec Tx formula:
    /// `(w0 × Tfs / 1000 + Tc + Tw) × Th` converted to device space.
    ///
    /// TJ array adjustments between strings are **not** folded into this
    /// field.  They are emitted as separate synthetic-space [`TextChar`]s
    /// inserted between the glyphs they affect, so the overall cursor
    /// displacement is correctly represented by walking the full char list.
    ///
    /// Equivalent to Poppler's `dx` argument in `drawChar`.
    ///
    /// For the last character on a line this falls back to `advance_width`.
    /// Use this field (not `advance_width`) to detect word boundaries:
    /// a gap `next.origin_x − (this.origin_x + this.rendered_advance) > threshold`
    /// reliably identifies inter-word spacing.
    pub rendered_advance: f32,

    /// Distance from the baseline to the top of the typographic glyph box (device space).
    ///
    /// From the font descriptor `/Ascent`; falls back to Adobe AFM values for the 14
    /// standard PDF fonts, then to 0.95 × font_size (Poppler's default).
    ///
    /// `bbox.height` is the full em square and does not reflect the font's actual cap
    /// height. Use `origin_y + ascent` for the glyph's true top edge.
    pub ascent: f32,

    /// Distance from the baseline to the bottom of the typographic glyph box (device space, negative).
    ///
    /// From the font descriptor `/Descent`; falls back to Adobe AFM values for the 14
    /// standard PDF fonts, then to −0.35 × font_size (Poppler's default).
    ///
    /// `bbox` does not represent the descender region at all (its origin is the
    /// baseline). Use `origin_y + descent` for the glyph's true bottom edge.
    pub descent: f32,

    /// Full transformation matrix [a, b, c, d, e, f].
    ///
    /// The composed text matrix (CTM × Tm) that transforms this character
    /// from text space to device space. Provides complete transformation
    /// info for advanced use cases like re-rendering or precise positioning.
    ///
    /// Matrix layout:
    /// ```text
    /// [ a  b  0 ]
    /// [ c  d  0 ]
    /// [ e  f  1 ]
    /// ```
    /// Where (a,d) = scaling, (b,c) = rotation/skew, (e,f) = translation.
    pub matrix: Option<[f32; 6]>,
}

impl Default for TextChar {
    fn default() -> Self {
        Self {
            char: ' ',
            bbox: Rect::default(),
            font_name: "Helvetica".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            origin_x: 0.0,
            origin_y: 0.0,
            rotation_degrees: 0.0,
            advance_width: 0.0,
            rendered_advance: 0.0,
            ascent: 0.95 * 12.0,
            descent: -0.35 * 12.0,
            matrix: Some([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        }
    }
}

impl TextChar {
    /// Get the rotation angle in radians.
    pub fn rotation_radians(&self) -> f32 {
        self.rotation_degrees.to_radians()
    }

    /// Check if this character is rotated (non-zero rotation).
    pub fn is_rotated(&self) -> bool {
        self.rotation_degrees.abs() > 0.01
    }

    /// Set the transformation matrix and update derived values.
    ///
    /// This method sets the full transformation matrix and automatically
    /// calculates the rotation angle and origin from the matrix components.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 6-element transformation matrix [a, b, c, d, e, f]
    pub fn with_matrix(mut self, matrix: [f32; 6]) -> Self {
        self.matrix = Some(matrix);
        // Extract origin from translation components
        self.origin_x = matrix[4];
        self.origin_y = matrix[5];
        // Calculate rotation from matrix: atan2(b, a)
        self.rotation_degrees = matrix[1].atan2(matrix[0]).to_degrees();
        self
    }

    /// Get the transformation matrix, computing from basic values if not stored.
    ///
    /// If the matrix was stored during extraction, returns it directly.
    /// Otherwise, reconstructs a basic matrix from origin and rotation.
    ///
    /// # Returns
    ///
    /// A 6-element transformation matrix [a, b, c, d, e, f]
    pub fn get_matrix(&self) -> [f32; 6] {
        if let Some(m) = self.matrix {
            m
        } else {
            // Reconstruct matrix from rotation and origin
            let rad = self.rotation_radians();
            let cos_r = rad.cos();
            let sin_r = rad.sin();
            [cos_r, sin_r, -sin_r, cos_r, self.origin_x, self.origin_y]
        }
    }

    /// Create a simple TextChar with default transformation values.
    ///
    /// This is a convenience constructor for creating TextChar instances
    /// when transformation data is not available (e.g., programmatic creation).
    /// The origin defaults to the bbox position, rotation to 0, and
    /// advance_width to the bbox width.
    pub fn simple(char: char, bbox: Rect, font_name: String, font_size: f32) -> Self {
        Self {
            char,
            bbox,
            font_name,
            font_size,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            origin_x: bbox.x,
            origin_y: bbox.y,
            rotation_degrees: 0.0,
            advance_width: bbox.width,
            rendered_advance: bbox.width,
            ascent: 0.95 * font_size,
            descent: -0.35 * font_size,
            matrix: None,
        }
    }
}

/// Font weight classification following PDF spec numeric scale.
///
/// PDF Spec: ISO 32000-1:2008, Table 122 - FontDescriptor
/// Values: 100-900 where 400 = normal, 700 = bold
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, Default)]
#[repr(u16)]
pub enum FontWeight {
    /// Thin (100)
    Thin = 100,
    /// Extra Light (200)
    ExtraLight = 200,
    /// Light (300)
    Light = 300,
    /// Normal (400) - default weight
    #[default]
    Normal = 400,
    /// Medium (500)
    Medium = 500,
    /// Semi Bold (600)
    SemiBold = 600,
    /// Bold (700) - standard bold weight
    Bold = 700,
    /// Extra Bold (800)
    ExtraBold = 800,
    /// Black (900) - heaviest weight
    Black = 900,
}

impl FontWeight {
    /// Check if this weight is considered bold (>= 600).
    ///
    /// Per PDF spec, weights 600+ are semi-bold or bolder.
    pub fn is_bold(&self) -> bool {
        *self as u16 >= 600
    }

    /// Create FontWeight from PDF numeric value.
    ///
    /// Rounds to nearest standard weight value.
    pub fn from_pdf_value(value: i32) -> Self {
        match value {
            ..=150 => FontWeight::Thin,
            151..=250 => FontWeight::ExtraLight,
            251..=350 => FontWeight::Light,
            351..=450 => FontWeight::Normal,
            451..=550 => FontWeight::Medium,
            551..=650 => FontWeight::SemiBold,
            651..=750 => FontWeight::Bold,
            751..=850 => FontWeight::ExtraBold,
            851.. => FontWeight::Black,
        }
    }

    /// Get the numeric PDF value for this weight.
    pub fn to_pdf_value(&self) -> u16 {
        *self as u16
    }
}

/// RGB color representation.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, Default)]
pub struct Color {
    /// Red channel (0.0 - 1.0)
    pub r: f32,
    /// Green channel (0.0 - 1.0)
    pub g: f32,
    /// Blue channel (0.0 - 1.0)
    pub b: f32,
}

impl Color {
    /// Create a new color.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create a black color.
    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create a white color.
    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

/// Complete text extraction result for a single page.
///
/// Single-call API that provides spans, per-character data, and page dimensions.
/// The `chars` field is derived from spans via `TextSpan::to_chars()`, using
/// font-metric widths when available for accurate per-glyph bounding boxes.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct PageText {
    /// Text spans in reading order.
    pub spans: Vec<TextSpan>,
    /// Per-character data derived from spans (uses font metric widths when available).
    pub chars: Vec<TextChar>,
    /// Page width in PDF points.
    pub page_width: f32,
    /// Page height in PDF points.
    pub page_height: f32,
}

/// A text block (word, line, or paragraph).
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct TextBlock {
    /// Characters in this block
    pub chars: Vec<TextChar>,
    /// Bounding box of the entire block
    pub bbox: Rect,
    /// Text content
    pub text: String,
    /// Average font size
    pub avg_font_size: f32,
    /// Dominant font name
    pub dominant_font: String,
    /// Whether the block contains bold text
    pub is_bold: bool,
    /// Whether the block contains italic text
    pub is_italic: bool,
    /// Marked Content ID (for Tagged PDFs)
    pub mcid: Option<u32>,
}

impl TextBlock {
    /// Create a text block from a collection of characters.
    ///
    /// This computes the bounding box, text content, average font size,
    /// and dominant font from the character data.
    ///
    /// # Panics
    ///
    /// Panics if the `chars` vector is empty.
    pub fn from_chars(chars: Vec<TextChar>) -> Self {
        assert!(!chars.is_empty(), "Cannot create TextBlock from empty chars");

        // Collect text directly
        let text: String = chars.iter().map(|c| c.char).collect();

        // Compute bounding box as union of all character bboxes
        let bbox = chars
            .iter()
            .map(|c| c.bbox)
            .fold(chars[0].bbox, |acc, r| acc.union(&r));

        let avg_font_size = chars.iter().map(|c| c.font_size).sum::<f32>() / chars.len() as f32;

        // Find dominant font (most common)
        let mut font_counts = HashMap::new();
        for c in &chars {
            *font_counts.entry(c.font_name.clone()).or_insert(0) += 1;
        }
        let dominant_font = font_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(font, _)| font.clone())
            .unwrap_or_default();

        let is_bold = chars.iter().any(|c| c.font_weight.is_bold());
        let is_italic = chars.iter().any(|c| c.is_italic);

        // Determine MCID for the block
        let mcid = chars
            .first()
            .and_then(|c| c.mcid)
            .filter(|&first_mcid| chars.iter().all(|c| c.mcid == Some(first_mcid)));

        Self {
            chars,
            bbox,
            text,
            avg_font_size,
            dominant_font,
            is_bold,
            is_italic,
            mcid,
        }
    }

    /// Get the center point of the text block.
    pub fn center(&self) -> Point {
        self.bbox.center()
    }

    /// Check if this block is horizontally aligned with another block.
    pub fn is_horizontally_aligned(&self, other: &TextBlock, tolerance: f32) -> bool {
        (self.bbox.y - other.bbox.y).abs() < tolerance
    }

    /// Check if this block is vertically aligned with another block.
    pub fn is_vertically_aligned(&self, other: &TextBlock, tolerance: f32) -> bool {
        (self.bbox.x - other.bbox.x).abs() < tolerance
    }
}

/// A word is a semantic unit of text (alias for TextBlock).
pub type Word = TextBlock;

/// A line of text containing multiple words.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct TextLine {
    /// Words in this line
    pub words: Vec<Word>,
    /// Bounding box of the entire line
    pub bbox: Rect,
    /// Complete text content of the line (words joined by spaces)
    pub text: String,
}

impl TextLine {
    /// Create a new TextLine from a list of words.
    ///
    /// # Panics
    ///
    /// Panics if the `words` vector is empty.
    pub fn new(words: Vec<Word>) -> Self {
        assert!(!words.is_empty(), "Cannot create TextLine from empty words");

        // Compute bounding box as union of all word bboxes
        let bbox = words
            .iter()
            .map(|w| w.bbox)
            .fold(words[0].bbox, |acc, r| acc.union(&r));

        // Join word text with spaces
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Self { words, bbox, text }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_char(c: char, x: f32, y: f32) -> TextChar {
        let bbox = Rect::new(x, y, 10.0, 12.0);
        TextChar {
            char: c,
            bbox,
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            origin_x: bbox.x,
            origin_y: bbox.y,
            rotation_degrees: 0.0,
            advance_width: bbox.width,
            rendered_advance: bbox.width,
            ascent: 0.95 * 12.0,
            descent: -0.35 * 12.0,
            matrix: None,
        }
    }

    #[test]
    fn test_text_block_from_chars() {
        let chars = vec![
            mock_char('H', 0.0, 0.0),
            mock_char('e', 10.0, 0.0),
            mock_char('l', 20.0, 0.0),
            mock_char('l', 30.0, 0.0),
            mock_char('o', 40.0, 0.0),
        ];

        let block = TextBlock::from_chars(chars);
        assert_eq!(block.text, "Hello");
        assert_eq!(block.avg_font_size, 12.0);
    }

    #[test]
    fn test_text_span_is_monospace_default() {
        let span = TextSpan::default();
        assert!(!span.is_monospace, "Default spans should not be monospace");
    }

    #[test]
    fn test_text_span_is_monospace_set() {
        let span = TextSpan {
            is_monospace: true,
            text: "AB".to_string(),
            bbox: Rect::new(0.0, 0.0, 20.0, 12.0),
            ..TextSpan::default()
        };
        assert!(span.is_monospace);

        // to_chars should propagate is_monospace
        let chars = span.to_chars();
        for c in &chars {
            assert!(c.is_monospace, "TextChar should inherit is_monospace from span");
        }
    }

    #[test]
    fn test_text_char_is_monospace() {
        let c = TextChar {
            char: 'A',
            bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
            font_name: "Courier".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: true,
            color: Color::black(),
            mcid: None,
            origin_x: 0.0,
            origin_y: 0.0,
            rotation_degrees: 0.0,
            advance_width: 10.0,
            rendered_advance: 10.0,
            ascent: 0.95 * 12.0,
            descent: -0.35 * 12.0,
            matrix: None,
        };
        assert!(c.is_monospace);
    }

    #[test]
    fn test_to_chars_uses_char_widths_when_available() {
        let span = TextSpan {
            text: "AB".to_string(),
            bbox: Rect::new(10.0, 20.0, 30.0, 12.0),
            char_widths: vec![10.0, 20.0],
            ..TextSpan::default()
        };
        let chars = span.to_chars();
        assert_eq!(chars.len(), 2);
        // First char: x=10, width=10
        assert!((chars[0].bbox.x - 10.0).abs() < 0.001);
        assert!((chars[0].bbox.width - 10.0).abs() < 0.001);
        assert!((chars[0].advance_width - 10.0).abs() < 0.001);
        // Second char: x=20, width=20
        assert!((chars[1].bbox.x - 20.0).abs() < 0.001);
        assert!((chars[1].bbox.width - 20.0).abs() < 0.001);
        assert!((chars[1].advance_width - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_to_chars_falls_back_to_uniform_when_no_widths() {
        let span = TextSpan {
            text: "AB".to_string(),
            bbox: Rect::new(10.0, 20.0, 30.0, 12.0),
            // char_widths left empty (default)
            ..TextSpan::default()
        };
        let chars = span.to_chars();
        assert_eq!(chars.len(), 2);
        // Uniform division: 30.0 / 2 = 15.0 each
        assert!((chars[0].bbox.width - 15.0).abs() < 0.001);
        assert!((chars[1].bbox.width - 15.0).abs() < 0.001);
        assert!((chars[0].bbox.x - 10.0).abs() < 0.001);
        assert!((chars[1].bbox.x - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_to_chars_handles_mismatched_widths_gracefully() {
        let span = TextSpan {
            text: "ABC".to_string(),
            bbox: Rect::new(0.0, 0.0, 30.0, 12.0),
            char_widths: vec![5.0, 10.0], // only 2 widths for 3 chars
            ..TextSpan::default()
        };
        let chars = span.to_chars();
        assert_eq!(chars.len(), 3);
        // Should fall back to uniform: 30.0 / 3 = 10.0 each
        assert!((chars[0].bbox.width - 10.0).abs() < 0.001);
        assert!((chars[1].bbox.width - 10.0).abs() < 0.001);
        assert!((chars[2].bbox.width - 10.0).abs() < 0.001);
    }
}
