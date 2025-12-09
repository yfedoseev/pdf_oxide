//! PDF content stream builder.
//!
//! Builds PDF content streams containing graphics and text operators
//! according to PDF specification ISO 32000-1:2008 Section 8-9.

use crate::elements::{ContentElement, PathContent, PathOperation, TextContent};
use crate::error::Result;
use crate::layout::Color;
use std::io::Write;

/// Operations that can be added to a content stream.
#[derive(Debug, Clone)]
pub enum ContentStreamOp {
    /// Save graphics state (q)
    SaveState,
    /// Restore graphics state (Q)
    RestoreState,
    /// Set transformation matrix (cm)
    Transform(f32, f32, f32, f32, f32, f32),
    /// Begin text object (BT)
    BeginText,
    /// End text object (ET)
    EndText,
    /// Set font and size (Tf)
    SetFont(String, f32),
    /// Move text position (Td)
    MoveText(f32, f32),
    /// Set text matrix (Tm)
    SetTextMatrix(f32, f32, f32, f32, f32, f32),
    /// Show text (Tj)
    ShowText(String),
    /// Show text with positioning (TJ)
    ShowTextArray(Vec<TextArrayItem>),
    /// Set character spacing (Tc)
    SetCharacterSpacing(f32),
    /// Set word spacing (Tw)
    SetWordSpacing(f32),
    /// Set text leading (TL)
    SetTextLeading(f32),
    /// Move to next line (T*)
    NextLine,
    /// Set fill color RGB (rg)
    SetFillColorRGB(f32, f32, f32),
    /// Set stroke color RGB (RG)
    SetStrokeColorRGB(f32, f32, f32),
    /// Set fill color gray (g)
    SetFillColorGray(f32),
    /// Set stroke color gray (G)
    SetStrokeColorGray(f32),
    /// Set line width (w)
    SetLineWidth(f32),
    /// Move to (m)
    MoveTo(f32, f32),
    /// Line to (l)
    LineTo(f32, f32),
    /// Curve to (c)
    CurveTo(f32, f32, f32, f32, f32, f32),
    /// Rectangle (re)
    Rectangle(f32, f32, f32, f32),
    /// Close path (h)
    ClosePath,
    /// Stroke (S)
    Stroke,
    /// Fill (f)
    Fill,
    /// Fill and stroke (B)
    FillStroke,
    /// Close and stroke (s)
    CloseStroke,
    /// End path without filling/stroking (n)
    EndPath,
    /// Raw operator (for extensibility)
    Raw(String),
}

/// Item in a TJ array (text or positioning adjustment).
#[derive(Debug, Clone)]
pub enum TextArrayItem {
    /// Text string
    Text(String),
    /// Positioning adjustment (negative = move right, positive = move left)
    Adjustment(f32),
}

/// Builder for PDF content streams.
///
/// Creates the byte sequence for a PDF content stream from operations
/// or ContentElements.
#[derive(Debug, Default)]
pub struct ContentStreamBuilder {
    /// Operations in the stream
    operations: Vec<ContentStreamOp>,
    /// Current font name
    current_font: Option<String>,
    /// Current font size
    current_font_size: f32,
    /// Whether we're in a text object
    in_text_object: bool,
}

impl ContentStreamBuilder {
    /// Create a new content stream builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an operation to the stream.
    pub fn op(&mut self, op: ContentStreamOp) -> &mut Self {
        self.operations.push(op);
        self
    }

    /// Add multiple operations.
    pub fn ops(&mut self, ops: impl IntoIterator<Item = ContentStreamOp>) -> &mut Self {
        self.operations.extend(ops);
        self
    }

    /// Begin a text object.
    pub fn begin_text(&mut self) -> &mut Self {
        if !self.in_text_object {
            self.op(ContentStreamOp::BeginText);
            self.in_text_object = true;
        }
        self
    }

    /// End a text object.
    pub fn end_text(&mut self) -> &mut Self {
        if self.in_text_object {
            self.op(ContentStreamOp::EndText);
            self.in_text_object = false;
        }
        self
    }

    /// Set font for text operations.
    pub fn set_font(&mut self, font_name: &str, size: f32) -> &mut Self {
        if self.current_font.as_deref() != Some(font_name) || self.current_font_size != size {
            self.op(ContentStreamOp::SetFont(font_name.to_string(), size));
            self.current_font = Some(font_name.to_string());
            self.current_font_size = size;
        }
        self
    }

    /// Add text at a position.
    pub fn text(&mut self, text: &str, x: f32, y: f32) -> &mut Self {
        self.begin_text();
        self.op(ContentStreamOp::SetTextMatrix(1.0, 0.0, 0.0, 1.0, x, y));
        self.op(ContentStreamOp::ShowText(text.to_string()));
        self
    }

    /// Set fill color.
    pub fn fill_color(&mut self, color: Color) -> &mut Self {
        self.op(ContentStreamOp::SetFillColorRGB(color.r, color.g, color.b))
    }

    /// Set stroke color.
    pub fn stroke_color(&mut self, color: Color) -> &mut Self {
        self.op(ContentStreamOp::SetStrokeColorRGB(color.r, color.g, color.b))
    }

    /// Draw a rectangle.
    pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32) -> &mut Self {
        self.op(ContentStreamOp::Rectangle(x, y, width, height))
    }

    /// Stroke the current path.
    pub fn stroke(&mut self) -> &mut Self {
        self.op(ContentStreamOp::Stroke)
    }

    /// Fill the current path.
    pub fn fill(&mut self) -> &mut Self {
        self.op(ContentStreamOp::Fill)
    }

    /// Add a ContentElement to the stream.
    pub fn add_element(&mut self, element: &ContentElement) -> &mut Self {
        match element {
            ContentElement::Text(text) => self.add_text_content(text),
            ContentElement::Path(path) => self.add_path_content(path),
            ContentElement::Image(_) => self, // Images require XObject - skip for now
            ContentElement::Structure(_) => self, // Structure doesn't generate content stream ops
        }
    }

    /// Add text content element.
    fn add_text_content(&mut self, text: &TextContent) -> &mut Self {
        self.begin_text();

        // Set color if not black
        if text.style.color.r != 0.0 || text.style.color.g != 0.0 || text.style.color.b != 0.0 {
            self.fill_color(text.style.color);
        }

        // Set font
        let font_name = self.map_font_name(&text.font.name, text.style.weight.is_bold());
        self.set_font(&font_name, text.font.size);

        // Position and show text
        self.op(ContentStreamOp::SetTextMatrix(1.0, 0.0, 0.0, 1.0, text.bbox.x, text.bbox.y));
        self.op(ContentStreamOp::ShowText(text.text.clone()));

        self
    }

    /// Map a font name to a PDF base font name.
    fn map_font_name(&self, name: &str, bold: bool) -> String {
        let base = match name.to_lowercase().as_str() {
            "helvetica" | "arial" | "sans-serif" => "Helvetica",
            "times" | "times-roman" | "times new roman" | "serif" => "Times-Roman",
            "courier" | "courier new" | "monospace" => "Courier",
            _ => "Helvetica",
        };

        if bold {
            format!("{}-Bold", base)
        } else {
            base.to_string()
        }
    }

    /// Add path content element.
    fn add_path_content(&mut self, path: &PathContent) -> &mut Self {
        // End any text object first
        self.end_text();

        // Set stroke properties
        if let Some(color) = path.stroke_color {
            self.stroke_color(color);
        }
        if let Some(color) = path.fill_color {
            self.fill_color(color);
        }
        self.op(ContentStreamOp::SetLineWidth(path.stroke_width));

        // Add path operations
        for op in &path.operations {
            match op {
                PathOperation::MoveTo(x, y) => {
                    self.op(ContentStreamOp::MoveTo(*x, *y));
                },
                PathOperation::LineTo(x, y) => {
                    self.op(ContentStreamOp::LineTo(*x, *y));
                },
                PathOperation::CurveTo(x1, y1, x2, y2, x3, y3) => {
                    self.op(ContentStreamOp::CurveTo(*x1, *y1, *x2, *y2, *x3, *y3));
                },
                PathOperation::Rectangle(x, y, w, h) => {
                    self.op(ContentStreamOp::Rectangle(*x, *y, *w, *h));
                },
                PathOperation::ClosePath => {
                    self.op(ContentStreamOp::ClosePath);
                },
            }
        }

        // Apply stroke/fill
        match (path.stroke_color.is_some(), path.fill_color.is_some()) {
            (true, true) => self.op(ContentStreamOp::FillStroke),
            (true, false) => self.op(ContentStreamOp::Stroke),
            (false, true) => self.op(ContentStreamOp::Fill),
            (false, false) => self.op(ContentStreamOp::EndPath),
        };

        self
    }

    /// Build multiple elements into the stream.
    pub fn add_elements(&mut self, elements: &[ContentElement]) -> &mut Self {
        for element in elements {
            self.add_element(element);
        }
        // Make sure to end any open text object
        self.end_text();
        self
    }

    /// Build the content stream to bytes.
    pub fn build(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();

        for op in &self.operations {
            self.write_op(&mut buf, op)?;
            writeln!(buf)?;
        }

        Ok(buf)
    }

    /// Write a single operation to the buffer.
    fn write_op<W: Write>(&self, w: &mut W, op: &ContentStreamOp) -> std::io::Result<()> {
        match op {
            ContentStreamOp::SaveState => write!(w, "q"),
            ContentStreamOp::RestoreState => write!(w, "Q"),
            ContentStreamOp::Transform(a, b, c, d, e, f) => {
                write!(w, "{} {} {} {} {} {} cm", a, b, c, d, e, f)
            },
            ContentStreamOp::BeginText => write!(w, "BT"),
            ContentStreamOp::EndText => write!(w, "ET"),
            ContentStreamOp::SetFont(name, size) => write!(w, "/{} {} Tf", name, size),
            ContentStreamOp::MoveText(tx, ty) => write!(w, "{} {} Td", tx, ty),
            ContentStreamOp::SetTextMatrix(a, b, c, d, e, f) => {
                write!(w, "{} {} {} {} {} {} Tm", a, b, c, d, e, f)
            },
            ContentStreamOp::ShowText(text) => {
                write!(w, "(")?;
                self.write_escaped_string(w, text)?;
                write!(w, ") Tj")
            },
            ContentStreamOp::ShowTextArray(items) => {
                write!(w, "[")?;
                for item in items {
                    match item {
                        TextArrayItem::Text(t) => {
                            write!(w, "(")?;
                            self.write_escaped_string(w, t)?;
                            write!(w, ")")?;
                        },
                        TextArrayItem::Adjustment(adj) => {
                            write!(w, "{}", adj)?;
                        },
                    }
                    write!(w, " ")?;
                }
                write!(w, "] TJ")
            },
            ContentStreamOp::SetCharacterSpacing(spacing) => write!(w, "{} Tc", spacing),
            ContentStreamOp::SetWordSpacing(spacing) => write!(w, "{} Tw", spacing),
            ContentStreamOp::SetTextLeading(leading) => write!(w, "{} TL", leading),
            ContentStreamOp::NextLine => write!(w, "T*"),
            ContentStreamOp::SetFillColorRGB(r, g, b) => write!(w, "{} {} {} rg", r, g, b),
            ContentStreamOp::SetStrokeColorRGB(r, g, b) => write!(w, "{} {} {} RG", r, g, b),
            ContentStreamOp::SetFillColorGray(g) => write!(w, "{} g", g),
            ContentStreamOp::SetStrokeColorGray(g) => write!(w, "{} G", g),
            ContentStreamOp::SetLineWidth(width) => write!(w, "{} w", width),
            ContentStreamOp::MoveTo(x, y) => write!(w, "{} {} m", x, y),
            ContentStreamOp::LineTo(x, y) => write!(w, "{} {} l", x, y),
            ContentStreamOp::CurveTo(x1, y1, x2, y2, x3, y3) => {
                write!(w, "{} {} {} {} {} {} c", x1, y1, x2, y2, x3, y3)
            },
            ContentStreamOp::Rectangle(x, y, w_val, h) => {
                write!(w, "{} {} {} {} re", x, y, w_val, h)
            },
            ContentStreamOp::ClosePath => write!(w, "h"),
            ContentStreamOp::Stroke => write!(w, "S"),
            ContentStreamOp::Fill => write!(w, "f"),
            ContentStreamOp::FillStroke => write!(w, "B"),
            ContentStreamOp::CloseStroke => write!(w, "s"),
            ContentStreamOp::EndPath => write!(w, "n"),
            ContentStreamOp::Raw(raw) => write!(w, "{}", raw),
        }
    }

    /// Write an escaped PDF string.
    fn write_escaped_string<W: Write>(&self, w: &mut W, text: &str) -> std::io::Result<()> {
        for byte in text.bytes() {
            match byte {
                b'(' => write!(w, "\\(")?,
                b')' => write!(w, "\\)")?,
                b'\\' => write!(w, "\\\\")?,
                b'\n' => write!(w, "\\n")?,
                b'\r' => write!(w, "\\r")?,
                b'\t' => write!(w, "\\t")?,
                _ => w.write_all(&[byte])?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::{FontSpec, TextStyle};
    use crate::geometry::Rect;

    #[test]
    fn test_simple_text() {
        let mut builder = ContentStreamBuilder::new();
        builder
            .begin_text()
            .set_font("Helvetica", 12.0)
            .text("Hello, World!", 72.0, 720.0)
            .end_text();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("BT"));
        assert!(content.contains("/Helvetica 12 Tf"));
        assert!(content.contains("(Hello, World!) Tj"));
        assert!(content.contains("ET"));
    }

    #[test]
    fn test_text_content_element() {
        let text_content = TextContent {
            text: "Test".to_string(),
            bbox: Rect::new(100.0, 700.0, 50.0, 12.0),
            font: FontSpec::new("Helvetica", 12.0),
            style: TextStyle::default(),
            reading_order: Some(0),
        };

        let mut builder = ContentStreamBuilder::new();
        builder.add_element(&ContentElement::Text(text_content));
        builder.end_text();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("BT"));
        assert!(content.contains("100 700"));
        assert!(content.contains("(Test) Tj"));
        assert!(content.contains("ET"));
    }

    #[test]
    fn test_path_operations() {
        let mut builder = ContentStreamBuilder::new();
        builder
            .stroke_color(Color::black())
            .op(ContentStreamOp::SetLineWidth(1.0))
            .op(ContentStreamOp::MoveTo(0.0, 0.0))
            .op(ContentStreamOp::LineTo(100.0, 100.0))
            .stroke();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("0 0 0 RG"));
        assert!(content.contains("1 w"));
        assert!(content.contains("0 0 m"));
        assert!(content.contains("100 100 l"));
        assert!(content.contains("S"));
    }

    #[test]
    fn test_rectangle() {
        let mut builder = ContentStreamBuilder::new();
        builder.rect(72.0, 72.0, 468.0, 648.0).stroke();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("72 72 468 648 re"));
        assert!(content.contains("S"));
    }

    #[test]
    fn test_escaped_text() {
        let mut builder = ContentStreamBuilder::new();
        builder
            .begin_text()
            .set_font("Helvetica", 12.0)
            .text("Text with (parens) and \\backslash", 72.0, 720.0)
            .end_text();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("\\(parens\\)"));
        assert!(content.contains("\\\\backslash"));
    }

    #[test]
    fn test_font_mapping() {
        let builder = ContentStreamBuilder::new();

        assert_eq!(builder.map_font_name("Arial", false), "Helvetica");
        assert_eq!(builder.map_font_name("Arial", true), "Helvetica-Bold");
        assert_eq!(builder.map_font_name("Times New Roman", false), "Times-Roman");
        assert_eq!(builder.map_font_name("Courier", false), "Courier");
    }
}
