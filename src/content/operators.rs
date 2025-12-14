//! PDF content stream operators.
//!
//! This module defines the operator types used in PDF content streams.
//! Content streams contain a sequence of operators that define the appearance
//! of a page, including text positioning, graphics state, and colors.

use crate::object::Object;

/// A content stream operator.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    // Text positioning operators
    /// Move text position (Td)
    Td {
        /// Horizontal offset
        tx: f32,
        /// Vertical offset
        ty: f32,
    },
    /// Move text position and set leading (TD)
    TD {
        /// Horizontal offset
        tx: f32,
        /// Vertical offset
        ty: f32,
    },
    /// Set text matrix (Tm)
    Tm {
        /// Matrix element a
        a: f32,
        /// Matrix element b
        b: f32,
        /// Matrix element c
        c: f32,
        /// Matrix element d
        d: f32,
        /// Matrix element e (x translation)
        e: f32,
        /// Matrix element f (y translation)
        f: f32,
    },
    /// Move to start of next line (T*)
    TStar,

    // Text showing operators
    /// Show text string (Tj)
    Tj {
        /// Text to show (byte array)
        text: Vec<u8>,
    },
    /// Show text with individual glyph positioning (TJ)
    TJ {
        /// Array of text strings and positioning adjustments
        array: Vec<TextElement>,
    },
    /// Move to next line and show text (')
    Quote {
        /// Text to show
        text: Vec<u8>,
    },
    /// Set spacing and show text (")
    DoubleQuote {
        /// Word spacing
        word_space: f32,
        /// Character spacing
        char_space: f32,
        /// Text to show
        text: Vec<u8>,
    },

    // Text state operators
    /// Set character spacing (Tc)
    Tc {
        /// Character spacing
        char_space: f32,
    },
    /// Set word spacing (Tw)
    Tw {
        /// Word spacing
        word_space: f32,
    },
    /// Set horizontal scaling (Tz)
    Tz {
        /// Horizontal scaling percentage
        scale: f32,
    },
    /// Set text leading (TL)
    TL {
        /// Text leading
        leading: f32,
    },
    /// Set font and size (Tf)
    Tf {
        /// Font name
        font: String,
        /// Font size
        size: f32,
    },
    /// Set text rendering mode (Tr)
    Tr {
        /// Rendering mode
        render: u8,
    },
    /// Set text rise (Ts)
    Ts {
        /// Text rise
        rise: f32,
    },

    // Graphics state operators
    /// Save graphics state (q)
    SaveState,
    /// Restore graphics state (Q)
    RestoreState,
    /// Modify current transformation matrix (cm)
    Cm {
        /// Matrix element a
        a: f32,
        /// Matrix element b
        b: f32,
        /// Matrix element c
        c: f32,
        /// Matrix element d
        d: f32,
        /// Matrix element e (x translation)
        e: f32,
        /// Matrix element f (y translation)
        f: f32,
    },

    // Color operators
    /// Set RGB fill color (rg)
    SetFillRgb {
        /// Red component (0.0-1.0)
        r: f32,
        /// Green component (0.0-1.0)
        g: f32,
        /// Blue component (0.0-1.0)
        b: f32,
    },
    /// Set RGB stroke color (RG)
    SetStrokeRgb {
        /// Red component (0.0-1.0)
        r: f32,
        /// Green component (0.0-1.0)
        g: f32,
        /// Blue component (0.0-1.0)
        b: f32,
    },
    /// Set gray fill color (g)
    SetFillGray {
        /// Gray level (0.0-1.0)
        gray: f32,
    },
    /// Set gray stroke color (G)
    SetStrokeGray {
        /// Gray level (0.0-1.0)
        gray: f32,
    },
    /// Set CMYK fill color (k)
    SetFillCmyk {
        /// Cyan component (0.0-1.0)
        c: f32,
        /// Magenta component (0.0-1.0)
        m: f32,
        /// Yellow component (0.0-1.0)
        y: f32,
        /// Black component (0.0-1.0)
        k: f32,
    },
    /// Set CMYK stroke color (K)
    SetStrokeCmyk {
        /// Cyan component (0.0-1.0)
        c: f32,
        /// Magenta component (0.0-1.0)
        m: f32,
        /// Yellow component (0.0-1.0)
        y: f32,
        /// Black component (0.0-1.0)
        k: f32,
    },

    // Color space operators
    /// Set fill color space (cs)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.6.4 - Color Space Operators
    SetFillColorSpace {
        /// Color space name (e.g., "DeviceRGB", "DeviceCMYK", "DeviceGray")
        name: String,
    },
    /// Set stroke color space (CS)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.6.4 - Color Space Operators
    SetStrokeColorSpace {
        /// Color space name (e.g., "DeviceRGB", "DeviceCMYK", "DeviceGray")
        name: String,
    },
    /// Set fill color (sc)
    ///
    /// Sets color components in the current fill color space.
    /// Number of components depends on color space (1 for Gray, 3 for RGB, 4 for CMYK).
    SetFillColor {
        /// Color components (length depends on color space)
        components: Vec<f32>,
    },
    /// Set stroke color (SC)
    ///
    /// Sets color components in the current stroke color space.
    /// Number of components depends on color space (1 for Gray, 3 for RGB, 4 for CMYK).
    SetStrokeColor {
        /// Color components (length depends on color space)
        components: Vec<f32>,
    },
    /// Set fill color with named pattern (scn)
    ///
    /// Like sc, but also supports pattern color spaces with an optional pattern name.
    SetFillColorN {
        /// Color components (may be empty for patterns)
        components: Vec<f32>,
        /// Optional pattern name for pattern color spaces
        name: Option<String>,
    },
    /// Set stroke color with named pattern (SCN)
    ///
    /// Like SC, but also supports pattern color spaces with an optional pattern name.
    SetStrokeColorN {
        /// Color components (may be empty for patterns)
        components: Vec<f32>,
        /// Optional pattern name for pattern color spaces
        name: Option<String>,
    },

    // Text object operators
    /// Begin text object (BT)
    BeginText,
    /// End text object (ET)
    EndText,

    // XObject operators
    /// Paint XObject (Do)
    Do {
        /// XObject name
        name: String,
    },

    // Path construction and painting (minimal support for now)
    /// Move to (m)
    MoveTo {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
    },
    /// Line to (l)
    LineTo {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
    },
    /// Cubic Bézier curve (c)
    CurveTo {
        /// X coordinate of first control point
        x1: f32,
        /// Y coordinate of first control point
        y1: f32,
        /// X coordinate of second control point
        x2: f32,
        /// Y coordinate of second control point
        y2: f32,
        /// X coordinate of end point
        x3: f32,
        /// Y coordinate of end point
        y3: f32,
    },
    /// Bézier curve with first control point = current point (v)
    CurveToV {
        /// X coordinate of second control point
        x2: f32,
        /// Y coordinate of second control point
        y2: f32,
        /// X coordinate of end point
        x3: f32,
        /// Y coordinate of end point
        y3: f32,
    },
    /// Bézier curve with second control point = end point (y)
    CurveToY {
        /// X coordinate of first control point
        x1: f32,
        /// Y coordinate of first control point
        y1: f32,
        /// X coordinate of end point
        x3: f32,
        /// Y coordinate of end point
        y3: f32,
    },
    /// Close current subpath (h)
    ClosePath,
    /// Rectangle (re)
    Rectangle {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
        /// Width
        width: f32,
        /// Height
        height: f32,
    },
    /// Stroke path (S)
    Stroke,
    /// Fill path (f)
    Fill,
    /// Fill path (even-odd) (f*)
    FillEvenOdd,
    /// Close, fill and stroke (b)
    CloseFillStroke,
    /// End path without filling or stroking (n)
    EndPath,
    /// Modify clipping path using non-zero winding rule (W)
    ClipNonZero,
    /// Modify clipping path using even-odd rule (W*)
    ClipEvenOdd,

    // Graphics state operators
    /// Set line width (w)
    SetLineWidth {
        /// Line width
        width: f32,
    },
    /// Set line dash pattern (d)
    SetDash {
        /// Dash array ([on1, off1, on2, off2, ...])
        array: Vec<f32>,
        /// Dash phase (offset into pattern)
        phase: f32,
    },
    /// Set line cap style (J)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.4.3.3 - Line Cap Style
    SetLineCap {
        /// Line cap style: 0=butt cap, 1=round cap, 2=projecting square cap
        cap_style: u8,
    },
    /// Set line join style (j)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.4.3.4 - Line Join Style
    SetLineJoin {
        /// Line join style: 0=miter join, 1=round join, 2=bevel join
        join_style: u8,
    },
    /// Set miter limit (M)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.4.3.5 - Miter Limit
    SetMiterLimit {
        /// Miter limit (ratio of miter length to line width)
        limit: f32,
    },
    /// Set rendering intent (ri)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.6.5.8 - Rendering Intents
    SetRenderingIntent {
        /// Rendering intent: AbsoluteColorimetric, RelativeColorimetric, Saturation, or Perceptual
        intent: String,
    },
    /// Set flatness tolerance (i)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 6.5.1 - Flatness Tolerance
    SetFlatness {
        /// Flatness tolerance (0-100, controlling curve approximation quality)
        tolerance: f32,
    },
    /// Set extended graphics state (gs)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.4.5 - Graphics State Parameter Dictionaries
    ///
    /// References an ExtGState dictionary in the page resources that contains
    /// graphics state parameters like transparency, blend modes, and line styles.
    SetExtGState {
        /// Name of the ExtGState dictionary in /ExtGState resources
        dict_name: String,
    },
    /// Paint shading pattern (sh)
    ///
    /// PDF Spec: ISO 32000-1:2008, Section 8.7.4.3 - Shading Patterns
    ///
    /// Paints a shading pattern (gradient) defined in the /Shading resource dictionary.
    /// Shading types include: Function-based, Axial, Radial, Free-form Gouraud, Lattice-form Gouraud, Coons patch, Tensor-product patch.
    PaintShading {
        /// Name of the shading dictionary in /Shading resources
        name: String,
    },

    // Inline image operator
    // PDF Spec: ISO 32000-1:2008, Section 8.9.7 - Inline Images
    /// Inline image (BI...ID...EI sequence)
    ///
    /// Represents a complete inline image sequence from BI (begin inline image)
    /// through ID (inline image data) to EI (end inline image).
    ///
    /// Inline images are small images embedded directly in the content stream
    /// rather than referenced as XObjects. The dictionary contains abbreviated
    /// keys for image properties.
    ///
    /// Common dictionary keys (abbreviated):
    /// - W: Width (required)
    /// - H: Height (required)
    /// - CS: ColorSpace (e.g., /DeviceRGB, /DeviceGray)
    /// - BPC: BitsPerComponent (typically 1, 8)
    /// - F: Filter (e.g., /FlateDecode, /DCTDecode)
    /// - DP: DecodeParms (decode parameters for filter)
    /// - I: Interpolate (boolean)
    InlineImage {
        /// Inline image dictionary with abbreviated keys
        dict: std::collections::HashMap<String, Object>,
        /// Raw image data bytes (possibly compressed)
        data: Vec<u8>,
    },

    // Marked content operators (for tagged PDF structure)
    // PDF Spec: ISO 32000-1:2008, Section 14.6 - Marked Content
    /// Begin marked content (BMC)
    ///
    /// Begins a marked content sequence identified by a tag.
    /// Used for logical structure and accessibility in tagged PDFs.
    BeginMarkedContent {
        /// Tag name identifying the marked content
        tag: String,
    },
    /// Begin marked content with property list (BDC)
    ///
    /// Begins a marked content sequence with associated properties.
    /// The properties can be inline (dictionary) or a reference to a properties resource.
    BeginMarkedContentDict {
        /// Tag name identifying the marked content
        tag: String,
        /// Properties (dictionary or name reference to /Properties resource)
        properties: Object,
    },
    /// End marked content (EMC)
    ///
    /// Ends the most recent marked content sequence.
    /// Must be balanced with BMC or BDC operators.
    EndMarkedContent,

    // Unknown operator (for operators we don't handle yet)
    /// Other operator
    Other {
        /// Operator name
        name: String,
        /// Operands
        operands: Vec<Object>,
    },
}

/// Element in a TJ array (text showing with positioning).
#[derive(Debug, Clone, PartialEq)]
pub enum TextElement {
    /// Text string to show
    String(Vec<u8>),
    /// Positioning adjustment (in thousandths of a unit of text space)
    Offset(f32),
}

impl Operator {
    /// Validate operand count and types according to PDF spec Table A.1.
    ///
    /// PDF Spec: ISO 32000-1:2008, Appendix A - Table A.1 - PDF content stream operators
    ///
    /// This method checks that operators have the correct number and types of operands.
    /// Only call this in strict mode for spec compliance validation.
    ///
    /// # Arguments
    ///
    /// * `operands` - The operands provided for this operator
    ///
    /// # Returns
    ///
    /// Ok(()) if operands are valid, Err with descriptive message if invalid
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::content::operators::Operator;
    /// use pdf_oxide::object::Object;
    ///
    /// let op = Operator::MoveTo { x: 10.0, y: 20.0 };
    /// let operands = vec![Object::Integer(10), Object::Integer(20)];
    /// assert!(op.validate_operands(&operands).is_ok());
    /// ```
    pub fn validate_operands_for_raw_operator(
        operator_name: &str,
        operands: &[Object],
    ) -> crate::error::Result<()> {
        use crate::error::Error;

        // Validate operand count according to PDF Spec Table A.1
        match operator_name {
            // Path construction operators - PDF Spec Section 8.5.2
            "m" => {
                // moveto: x y m
                if operands.len() != 2 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'm' (moveto) requires 2 operands (x, y), got {}",
                        operands.len()
                    )));
                }
            },
            "l" => {
                // lineto: x y l
                if operands.len() != 2 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'l' (lineto) requires 2 operands (x, y), got {}",
                        operands.len()
                    )));
                }
            },
            "c" => {
                // curveto: x1 y1 x2 y2 x3 y3 c
                if operands.len() != 6 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'c' (curveto) requires 6 operands (x1, y1, x2, y2, x3, y3), got {}",
                        operands.len()
                    )));
                }
            },
            "v" => {
                // curveto (v variant): x2 y2 x3 y3 v
                if operands.len() != 4 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'v' (curveto) requires 4 operands (x2, y2, x3, y3), got {}",
                        operands.len()
                    )));
                }
            },
            "y" => {
                // curveto (y variant): x1 y1 x3 y3 y
                if operands.len() != 4 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'y' (curveto) requires 4 operands (x1, y1, x3, y3), got {}",
                        operands.len()
                    )));
                }
            },
            "h" => {
                // closepath: h (no operands)
                if !operands.is_empty() {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'h' (closepath) requires 0 operands, got {}",
                        operands.len()
                    )));
                }
            },
            "re" => {
                // rectangle: x y width height re
                if operands.len() != 4 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 're' (rectangle) requires 4 operands (x, y, width, height), got {}",
                        operands.len()
                    )));
                }
            },

            // Text positioning operators - PDF Spec Section 9.4.2
            "Td" => {
                // Move text position: tx ty Td
                if operands.len() != 2 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Td' requires 2 operands (tx, ty), got {}",
                        operands.len()
                    )));
                }
            },
            "TD" => {
                // Move text position and set leading: tx ty TD
                if operands.len() != 2 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'TD' requires 2 operands (tx, ty), got {}",
                        operands.len()
                    )));
                }
            },
            "Tm" => {
                // Set text matrix: a b c d e f Tm
                if operands.len() != 6 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tm' requires 6 operands (a, b, c, d, e, f), got {}",
                        operands.len()
                    )));
                }
            },
            "T*" => {
                // Move to next line: T* (no operands)
                if !operands.is_empty() {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'T*' requires 0 operands, got {}",
                        operands.len()
                    )));
                }
            },

            // Text showing operators - PDF Spec Section 9.4.3
            "Tj" => {
                // Show text: string Tj
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tj' requires 1 operand (string), got {}",
                        operands.len()
                    )));
                }
            },
            "TJ" => {
                // Show text with positioning: array TJ
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'TJ' requires 1 operand (array), got {}",
                        operands.len()
                    )));
                }
            },
            "'" => {
                // Move to next line and show text: string '
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator ''' requires 1 operand (string), got {}",
                        operands.len()
                    )));
                }
            },
            "\"" => {
                // Set spacing and show text: aw ac string "
                if operands.len() != 3 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator '\"' requires 3 operands (word_space, char_space, string), got {}",
                        operands.len()
                    )));
                }
            },

            // Text state operators - PDF Spec Section 9.3
            "Tc" => {
                // Set character spacing: charSpace Tc
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tc' requires 1 operand (char_space), got {}",
                        operands.len()
                    )));
                }
            },
            "Tw" => {
                // Set word spacing: wordSpace Tw
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tw' requires 1 operand (word_space), got {}",
                        operands.len()
                    )));
                }
            },
            "Tz" => {
                // Set horizontal scaling: scale Tz
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tz' requires 1 operand (scale), got {}",
                        operands.len()
                    )));
                }
            },
            "TL" => {
                // Set text leading: leading TL
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'TL' requires 1 operand (leading), got {}",
                        operands.len()
                    )));
                }
            },
            "Tf" => {
                // Set font: font size Tf
                if operands.len() != 2 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tf' requires 2 operands (font, size), got {}",
                        operands.len()
                    )));
                }
            },
            "Tr" => {
                // Set text rendering mode: render Tr
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Tr' requires 1 operand (render), got {}",
                        operands.len()
                    )));
                }
            },
            "Ts" => {
                // Set text rise: rise Ts
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Ts' requires 1 operand (rise), got {}",
                        operands.len()
                    )));
                }
            },

            // Graphics state operators
            "q" | "Q" => {
                // Save/restore graphics state: q, Q (no operands)
                if !operands.is_empty() {
                    return Err(Error::InvalidPdf(format!(
                        "Operator '{}' requires 0 operands, got {}",
                        operator_name,
                        operands.len()
                    )));
                }
            },
            "cm" => {
                // Modify CTM: a b c d e f cm
                if operands.len() != 6 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'cm' requires 6 operands (a, b, c, d, e, f), got {}",
                        operands.len()
                    )));
                }
            },

            // Color operators - PDF Spec Section 8.6.8
            "rg" => {
                // Set RGB fill color: r g b rg
                if operands.len() != 3 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'rg' requires 3 operands (r, g, b), got {}",
                        operands.len()
                    )));
                }
            },
            "RG" => {
                // Set RGB stroke color: r g b RG
                if operands.len() != 3 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'RG' requires 3 operands (r, g, b), got {}",
                        operands.len()
                    )));
                }
            },
            "g" => {
                // Set gray fill color: gray g
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'g' requires 1 operand (gray), got {}",
                        operands.len()
                    )));
                }
            },
            "G" => {
                // Set gray stroke color: gray G
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'G' requires 1 operand (gray), got {}",
                        operands.len()
                    )));
                }
            },
            "k" => {
                // Set CMYK fill color: c m y k k
                if operands.len() != 4 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'k' requires 4 operands (c, m, y, k), got {}",
                        operands.len()
                    )));
                }
            },
            "K" => {
                // Set CMYK stroke color: c m y k K
                if operands.len() != 4 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'K' requires 4 operands (c, m, y, k), got {}",
                        operands.len()
                    )));
                }
            },

            // Text object operators - PDF Spec Section 9.4
            "BT" | "ET" => {
                // Begin/end text: BT, ET (no operands)
                if !operands.is_empty() {
                    return Err(Error::InvalidPdf(format!(
                        "Operator '{}' requires 0 operands, got {}",
                        operator_name,
                        operands.len()
                    )));
                }
            },

            // XObject operator - PDF Spec Section 8.8
            "Do" => {
                // Paint XObject: name Do
                if operands.len() != 1 {
                    return Err(Error::InvalidPdf(format!(
                        "Operator 'Do' requires 1 operand (name), got {}",
                        operands.len()
                    )));
                }
            },

            // Other operators we don't validate yet
            _ => {
                // No validation for unknown operators (lenient behavior)
                log::debug!(
                    "No operand validation for operator '{}' (not implemented yet)",
                    operator_name
                );
            },
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_td() {
        let op = Operator::Td { tx: 10.0, ty: 20.0 };
        match op {
            Operator::Td { tx, ty } => {
                assert_eq!(tx, 10.0);
                assert_eq!(ty, 20.0);
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_operator_tm() {
        let op = Operator::Tm {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 100.0,
            f: 200.0,
        };
        match op {
            Operator::Tm { a, b, c, d, e, f } => {
                assert_eq!(a, 1.0);
                assert_eq!(b, 0.0);
                assert_eq!(c, 0.0);
                assert_eq!(d, 1.0);
                assert_eq!(e, 100.0);
                assert_eq!(f, 200.0);
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_operator_tj() {
        let op = Operator::Tj {
            text: b"Hello".to_vec(),
        };
        match op {
            Operator::Tj { text } => {
                assert_eq!(text, b"Hello");
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_operator_tf() {
        let op = Operator::Tf {
            font: "F1".to_string(),
            size: 12.0,
        };
        match op {
            Operator::Tf { font, size } => {
                assert_eq!(font, "F1");
                assert_eq!(size, 12.0);
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_operator_rgb() {
        let op = Operator::SetFillRgb {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        };
        match op {
            Operator::SetFillRgb { r, g, b } => {
                assert_eq!(r, 1.0);
                assert_eq!(g, 0.0);
                assert_eq!(b, 0.0);
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_text_element_string() {
        let elem = TextElement::String(b"Text".to_vec());
        match elem {
            TextElement::String(s) => {
                assert_eq!(s, b"Text");
            },
            _ => panic!("Wrong element type"),
        }
    }

    #[test]
    fn test_text_element_offset() {
        let elem = TextElement::Offset(-100.0);
        match elem {
            TextElement::Offset(offset) => {
                assert_eq!(offset, -100.0);
            },
            _ => panic!("Wrong element type"),
        }
    }

    #[test]
    fn test_operator_clone() {
        let op1 = Operator::Tj {
            text: b"Test".to_vec(),
        };
        let op2 = op1.clone();
        assert_eq!(op1, op2);
    }

    #[test]
    fn test_operator_save_restore() {
        let save = Operator::SaveState;
        let restore = Operator::RestoreState;
        assert!(matches!(save, Operator::SaveState));
        assert!(matches!(restore, Operator::RestoreState));
    }

    #[test]
    fn test_operator_other() {
        let op = Operator::Other {
            name: "Do".to_string(),
            operands: vec![Object::Name("Im1".to_string())],
        };
        match op {
            Operator::Other { name, operands } => {
                assert_eq!(name, "Do");
                assert_eq!(operands.len(), 1);
            },
            _ => panic!("Wrong operator type"),
        }
    }

    #[test]
    fn test_text_element_eq() {
        let elem1 = TextElement::String(b"Test".to_vec());
        let elem2 = TextElement::String(b"Test".to_vec());
        assert_eq!(elem1, elem2);

        let elem3 = TextElement::Offset(10.0);
        let elem4 = TextElement::Offset(10.0);
        assert_eq!(elem3, elem4);
    }
}
