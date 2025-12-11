//! Text block representation for layout analysis.
//!
//! This module defines structures for representing text elements in a PDF document
//! with their geometric and styling information.

use crate::geometry::{Point, Rect};
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
/// - Matches industry best practices (PyMuPDF, etc.)
/// - More robust for complex layouts
#[derive(Debug, Clone)]
pub struct TextSpan {
    /// The complete text string
    pub text: String,
    /// Bounding box of the entire span
    pub bbox: Rect,
    /// Font name/family
    pub font_name: String,
    /// Font size in points
    pub font_size: f32,
    /// Font weight (normal or bold)
    pub font_weight: FontWeight,
    /// Font style: italic or normal
    pub is_italic: bool,
    /// Text color
    pub color: Color,
    /// Marked Content ID (for Tagged PDFs)
    pub mcid: Option<u32>,
    /// Extraction sequence number (used as tie-breaker for Y-coordinate sorting)
    ///
    /// When PDFs use unusual coordinate systems where many spans have identical
    /// Y coordinates (e.g., EU GDPR PDF with Y=0.0 for multiple spans), this
    /// sequence number preserves the original extraction order from the content
    /// stream, which often reflects the intended reading order per PDF spec.
    pub sequence: usize,
    /// If true, this span was created by splitting fused words and should not be re-merged.
    ///
    /// When CamelCase splitting creates separate spans from a single fused word
    /// (e.g., "theGeneral" -> "the" + "General"), this flag prevents them from
    /// being re-merged during the span merging phase, even if the gap is 0pt.
    /// This preserves the split intent and prevents word fusion regressions.
    pub split_boundary_before: bool,
    /// If true, this span was created by the TJ processor as a space from a negative offset.
    ///
    /// Per PDF spec ISO 32000-1:2008 Section 9.4.4, negative offsets in TJ arrays
    /// indicate word boundaries where spaces should be inserted. This flag marks
    /// those automatically generated space spans so merge logic can avoid double-spacing.
    pub offset_semantic: bool,
    /// Character spacing (Tc parameter) per ISO 32000-1:2008 Section 9.3.1.
    ///
    /// Tc is added after each character during text positioning. Default value is 0.
    /// This value is used for text justification detection (Phase 3.5).
    pub char_spacing: f32,
    /// Word spacing (Tw parameter) per ISO 32000-1:2008 Section 9.3.1.
    ///
    /// Tw is added after space characters (U+0020) during text positioning. Default is 0.
    /// This value is critical for text justification detection - the variance in Tw
    /// values across a line indicates the degree of justification applied.
    pub word_spacing: f32,
    /// Horizontal scaling (Tz parameter) per ISO 32000-1:2008 Section 9.3.1.
    ///
    /// Tz scales all character widths and word spacing. Value is in percent (e.g., 100 = 100%).
    /// Default value is 100.0. Used for justification detection and layout analysis.
    pub horizontal_scaling: f32,
    /// If true, was created by WordBoundaryDetector primary detection.
    ///
    /// Used to mark spans created by primary detection mode so they
    /// are not re-merged during the span merging phase.
    /// Default is false for backward compatibility.
    pub primary_detected: bool,
}

/// A single character with its position and styling.
///
/// NOTE: This is kept for backward compatibility and special use cases.
/// For normal text extraction, prefer TextSpan which represents complete
/// text strings as the PDF provides them.
#[derive(Debug, Clone)]
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
    /// Text color
    pub color: Color,
    /// Marked Content ID (for Tagged PDFs)
    ///
    /// This field stores the MCID if this character was extracted within
    /// a marked content sequence in a Tagged PDF.
    pub mcid: Option<u32>,
}

/// Font weight classification following PDF spec numeric scale.
///
/// PDF Spec: ISO 32000-1:2008, Table 122 - FontDescriptor
/// Values: 100-900 where 400 = normal, 700 = bold
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u16)]
#[derive(Default)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::layout::Color;
    ///
    /// let black = Color::new(0.0, 0.0, 0.0);
    /// let white = Color::new(1.0, 1.0, 1.0);
    /// let red = Color::new(1.0, 0.0, 0.0);
    /// ```ignore
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

/// A text block (word, line, or paragraph).
#[derive(Debug, Clone)]
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
    ///
    /// This field stores the MCID (Marked Content ID) if this text block
    /// belongs to a marked content sequence in a Tagged PDF. The MCID can
    /// be used to determine reading order via the structure tree.
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::geometry::Rect;
    /// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color};
    ///
    /// let chars = vec![
    ///     TextChar {
    ///         char: 'H',
    ///         bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    ///     TextChar {
    ///         char: 'i',
    ///         bbox: Rect::new(10.0, 0.0, 5.0, 12.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    ///
    /// let block = TextBlock::from_chars(chars);
    /// assert_eq!(block.text, "Hi");
    /// assert_eq!(block.avg_font_size, 12.0);
    /// ```ignore
    pub fn from_chars(chars: Vec<TextChar>) -> Self {
        assert!(!chars.is_empty(), "Cannot create TextBlock from empty chars");

        // Collect text directly (word spacing is handled at markdown level)
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
        // Use the MCID of the first character if all chars have the same MCID
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::geometry::Rect;
    /// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color};
    ///
    /// let chars = vec![
    ///     TextChar {
    ///         char: 'A',
    ///         bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    ///
    /// let block = TextBlock::from_chars(chars);
    /// let center = block.center();
    /// assert_eq!(center.x, 50.0);
    /// assert_eq!(center.y, 25.0);
    /// ```ignore
    pub fn center(&self) -> Point {
        self.bbox.center()
    }

    /// Check if this block is horizontally aligned with another block.
    ///
    /// Two blocks are considered horizontally aligned if their Y coordinates
    /// are within the specified tolerance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::geometry::Rect;
    /// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color};
    ///
    /// let chars1 = vec![
    ///     TextChar {
    ///         char: 'A',
    ///         bbox: Rect::new(0.0, 0.0, 10.0, 10.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    /// let chars2 = vec![
    ///     TextChar {
    ///         char: 'B',
    ///         bbox: Rect::new(50.0, 1.0, 10.0, 10.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    ///
    /// let block1 = TextBlock::from_chars(chars1);
    /// let block2 = TextBlock::from_chars(chars2);
    ///
    /// assert!(block1.is_horizontally_aligned(&block2, 5.0));
    /// assert!(!block1.is_horizontally_aligned(&block2, 0.5));
    /// ```ignore
    pub fn is_horizontally_aligned(&self, other: &TextBlock, tolerance: f32) -> bool {
        (self.bbox.y - other.bbox.y).abs() < tolerance
    }

    /// Check if this block is vertically aligned with another block.
    ///
    /// Two blocks are considered vertically aligned if their X coordinates
    /// are within the specified tolerance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::geometry::Rect;
    /// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color};
    ///
    /// let chars1 = vec![
    ///     TextChar {
    ///         char: 'A',
    ///         bbox: Rect::new(0.0, 0.0, 10.0, 10.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    /// let chars2 = vec![
    ///     TextChar {
    ///         char: 'B',
    ///         bbox: Rect::new(1.0, 50.0, 10.0, 10.0),
    ///         font_name: "Times".to_string(),
    ///         font_size: 12.0,
    ///         font_weight: FontWeight::Normal,
    ///         is_italic: false,
    ///         color: Color::black(),
    ///     },
    /// ];
    ///
    /// let block1 = TextBlock::from_chars(chars1);
    /// let block2 = TextBlock::from_chars(chars2);
    ///
    /// assert!(block1.is_vertically_aligned(&block2, 5.0));
    /// assert!(!block1.is_vertically_aligned(&block2, 0.5));
    /// ```ignore
    pub fn is_vertically_aligned(&self, other: &TextBlock, tolerance: f32) -> bool {
        (self.bbox.x - other.bbox.x).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_char(c: char, x: f32, y: f32) -> TextChar {
        TextChar {
            char: c,
            bbox: Rect::new(x, y, 10.0, 12.0),
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
        }
    }

    #[test]
    fn test_color_creation() {
        let black = Color::black();
        assert_eq!(black.r, 0.0);
        assert_eq!(black.g, 0.0);
        assert_eq!(black.b, 0.0);

        let white = Color::white();
        assert_eq!(white.r, 1.0);
        assert_eq!(white.g, 1.0);
        assert_eq!(white.b, 1.0);

        let red = Color::new(1.0, 0.0, 0.0);
        assert_eq!(red.r, 1.0);
        assert_eq!(red.g, 0.0);
        assert_eq!(red.b, 0.0);
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
        assert_eq!(block.dominant_font, "Times");
        assert!(!block.is_bold);
    }

    #[test]
    fn test_text_block_bold_detection() {
        let chars = vec![
            TextChar {
                char: 'B',
                bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
                font_name: "Times".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Bold,
                is_italic: false,
                color: Color::black(),
                mcid: None,
            },
            mock_char('o', 10.0, 0.0),
            mock_char('l', 20.0, 0.0),
            mock_char('d', 30.0, 0.0),
        ];

        let block = TextBlock::from_chars(chars);
        assert_eq!(block.text, "Bold");
        assert!(block.is_bold);
    }

    #[test]
    fn test_text_block_center() {
        let chars = vec![TextChar {
            char: 'A',
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
        }];

        let block = TextBlock::from_chars(chars);
        let center = block.center();
        assert_eq!(center.x, 50.0);
        assert_eq!(center.y, 25.0);
    }

    #[test]
    fn test_horizontal_alignment() {
        let chars1 = vec![mock_char('A', 0.0, 0.0)];
        let chars2 = vec![mock_char('B', 50.0, 2.0)];
        let chars3 = vec![mock_char('C', 100.0, 20.0)];

        let block1 = TextBlock::from_chars(chars1);
        let block2 = TextBlock::from_chars(chars2);
        let block3 = TextBlock::from_chars(chars3);

        assert!(block1.is_horizontally_aligned(&block2, 5.0));
        assert!(!block1.is_horizontally_aligned(&block3, 5.0));
    }

    #[test]
    fn test_vertical_alignment() {
        let chars1 = vec![mock_char('A', 0.0, 0.0)];
        let chars2 = vec![mock_char('B', 2.0, 50.0)];
        let chars3 = vec![mock_char('C', 20.0, 100.0)];

        let block1 = TextBlock::from_chars(chars1);
        let block2 = TextBlock::from_chars(chars2);
        let block3 = TextBlock::from_chars(chars3);

        assert!(block1.is_vertically_aligned(&block2, 5.0));
        assert!(!block1.is_vertically_aligned(&block3, 5.0));
    }
}
