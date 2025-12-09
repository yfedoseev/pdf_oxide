//! Text content element types.
//!
//! This module provides the `TextContent` type and related structures
//! for representing text in PDFs.

use crate::geometry::Rect;
use crate::layout::{Color, FontWeight, TextSpan};

/// Text content that can be extracted from or written to a PDF.
///
/// This is the unified text representation for both reading and writing.
/// Unlike `TextSpan` which is extraction-focused, `TextContent` is designed
/// to work symmetrically for both directions.
#[derive(Debug, Clone)]
pub struct TextContent {
    /// The text string
    pub text: String,
    /// Bounding box of the text
    pub bbox: Rect,
    /// Font specification
    pub font: FontSpec,
    /// Text styling (bold, italic, color, etc.)
    pub style: TextStyle,
    /// Reading order index (for extraction) or write order (for generation)
    pub reading_order: Option<usize>,
}

impl TextContent {
    /// Create a new text content element.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::elements::{TextContent, FontSpec, TextStyle};
    /// use pdf_oxide::geometry::Rect;
    ///
    /// let text = TextContent::new(
    ///     "Hello, World!",
    ///     Rect::new(72.0, 720.0, 100.0, 12.0),
    ///     FontSpec::default(),
    ///     TextStyle::default(),
    /// );
    /// ```
    pub fn new(text: impl Into<String>, bbox: Rect, font: FontSpec, style: TextStyle) -> Self {
        Self {
            text: text.into(),
            bbox,
            font,
            style,
            reading_order: None,
        }
    }

    /// Create text content with reading order.
    pub fn with_reading_order(mut self, order: usize) -> Self {
        self.reading_order = Some(order);
        self
    }

    /// Check if this text is bold.
    pub fn is_bold(&self) -> bool {
        self.style.weight.is_bold()
    }

    /// Check if this text is italic.
    pub fn is_italic(&self) -> bool {
        self.style.italic
    }

    /// Get the font size in points.
    pub fn font_size(&self) -> f32 {
        self.font.size
    }
}

/// Convert from TextSpan (extraction result) to TextContent (unified representation).
impl From<TextSpan> for TextContent {
    fn from(span: TextSpan) -> Self {
        TextContent {
            text: span.text,
            bbox: span.bbox,
            font: FontSpec {
                name: span.font_name,
                size: span.font_size,
            },
            style: TextStyle {
                weight: span.font_weight,
                italic: false, // TextSpan doesn't track italic separately
                color: span.color,
                underline: false,
                strikethrough: false,
            },
            reading_order: Some(span.sequence),
        }
    }
}

/// Convert from TextContent to TextSpan (for backward compatibility).
impl From<TextContent> for TextSpan {
    fn from(content: TextContent) -> Self {
        TextSpan {
            text: content.text,
            bbox: content.bbox,
            font_name: content.font.name,
            font_size: content.font.size,
            font_weight: content.style.weight,
            color: content.style.color,
            mcid: None,
            sequence: content.reading_order.unwrap_or(0),
            split_boundary_before: false,
            offset_semantic: false,
        }
    }
}

/// Font specification for text rendering.
///
/// Contains the minimal font information needed for both
/// extraction and generation of PDF text.
#[derive(Debug, Clone)]
pub struct FontSpec {
    /// Font name/family (e.g., "Times-Roman", "Helvetica-Bold")
    pub name: String,
    /// Font size in points
    pub size: f32,
}

impl FontSpec {
    /// Create a new font specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::elements::FontSpec;
    ///
    /// let font = FontSpec::new("Helvetica", 12.0);
    /// ```
    pub fn new(name: impl Into<String>, size: f32) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }

    /// Create a Helvetica font spec with given size.
    pub fn helvetica(size: f32) -> Self {
        Self::new("Helvetica", size)
    }

    /// Create a Times Roman font spec with given size.
    pub fn times(size: f32) -> Self {
        Self::new("Times-Roman", size)
    }

    /// Create a Courier font spec with given size.
    pub fn courier(size: f32) -> Self {
        Self::new("Courier", size)
    }
}

impl Default for FontSpec {
    fn default() -> Self {
        Self {
            name: "Helvetica".to_string(),
            size: 12.0,
        }
    }
}

/// Font style classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontStyle {
    /// Normal (upright) style
    #[default]
    Normal,
    /// Italic style
    Italic,
    /// Oblique style (slanted but not true italic)
    Oblique,
}

/// Text styling information.
///
/// Contains visual styling properties that can be applied to text.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Font weight (normal, bold, etc.)
    pub weight: FontWeight,
    /// Whether text is italicized
    pub italic: bool,
    /// Text color
    pub color: Color,
    /// Whether text is underlined
    pub underline: bool,
    /// Whether text has strikethrough
    pub strikethrough: bool,
}

impl TextStyle {
    /// Create a new text style with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create bold text style.
    pub fn bold() -> Self {
        Self {
            weight: FontWeight::Bold,
            ..Default::default()
        }
    }

    /// Create italic text style.
    pub fn italic() -> Self {
        Self {
            italic: true,
            ..Default::default()
        }
    }

    /// Create bold italic text style.
    pub fn bold_italic() -> Self {
        Self {
            weight: FontWeight::Bold,
            italic: true,
            ..Default::default()
        }
    }

    /// Set the font weight.
    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Set italic style.
    pub fn with_italic(mut self, italic: bool) -> Self {
        self.italic = italic;
        self
    }

    /// Set text color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            weight: FontWeight::Normal,
            italic: false,
            color: Color::black(),
            underline: false,
            strikethrough: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_content_creation() {
        let text = TextContent::new(
            "Hello",
            Rect::new(0.0, 0.0, 50.0, 12.0),
            FontSpec::default(),
            TextStyle::default(),
        );

        assert_eq!(text.text, "Hello");
        assert_eq!(text.font_size(), 12.0);
        assert!(!text.is_bold());
        assert!(!text.is_italic());
    }

    #[test]
    fn test_text_content_with_reading_order() {
        let text = TextContent::new(
            "First",
            Rect::new(0.0, 0.0, 50.0, 12.0),
            FontSpec::default(),
            TextStyle::default(),
        )
        .with_reading_order(5);

        assert_eq!(text.reading_order, Some(5));
    }

    #[test]
    fn test_font_spec_presets() {
        let helvetica = FontSpec::helvetica(14.0);
        assert_eq!(helvetica.name, "Helvetica");
        assert_eq!(helvetica.size, 14.0);

        let times = FontSpec::times(12.0);
        assert_eq!(times.name, "Times-Roman");

        let courier = FontSpec::courier(10.0);
        assert_eq!(courier.name, "Courier");
    }

    #[test]
    fn test_text_style_presets() {
        let bold = TextStyle::bold();
        assert!(bold.weight.is_bold());
        assert!(!bold.italic);

        let italic = TextStyle::italic();
        assert!(!italic.weight.is_bold());
        assert!(italic.italic);

        let bold_italic = TextStyle::bold_italic();
        assert!(bold_italic.weight.is_bold());
        assert!(bold_italic.italic);
    }

    #[test]
    fn test_text_span_conversion() {
        let span = TextSpan {
            text: "Test".to_string(),
            bbox: Rect::new(10.0, 20.0, 40.0, 12.0),
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Bold,
            color: Color::black(),
            mcid: None,
            sequence: 3,
            split_boundary_before: false,
            offset_semantic: false,
        };

        let content: TextContent = span.into();

        assert_eq!(content.text, "Test");
        assert_eq!(content.font.name, "Times");
        assert_eq!(content.font.size, 12.0);
        assert!(content.is_bold());
        assert_eq!(content.reading_order, Some(3));
    }

    #[test]
    fn test_text_content_to_span_conversion() {
        let content = TextContent {
            text: "Converted".to_string(),
            bbox: Rect::new(0.0, 0.0, 80.0, 14.0),
            font: FontSpec::new("Helvetica", 14.0),
            style: TextStyle::bold(),
            reading_order: Some(7),
        };

        let span: TextSpan = content.into();

        assert_eq!(span.text, "Converted");
        assert_eq!(span.font_name, "Helvetica");
        assert_eq!(span.font_size, 14.0);
        assert!(span.font_weight.is_bold());
        assert_eq!(span.sequence, 7);
    }
}
