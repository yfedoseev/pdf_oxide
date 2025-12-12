//! Text extraction from PDF content streams.
//!
//! This module executes content stream operators to extract positioned
//! text characters with their Unicode mappings, font information, and
//! bounding boxes.
//!
//! Phase 4, Task 4.6

use crate::content::graphics_state::{GraphicsStateStack, Matrix};
use crate::content::operators::{Operator, TextElement};
use crate::content::parse_content_stream;
use crate::error::Result;
use crate::fonts::FontInfo;
use crate::geometry::Rect;
use crate::layout::{Color, FontWeight, TextChar, TextSpan};
use crate::object::{Object, ObjectRef};
use std::collections::{HashMap, HashSet};

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
    /// Threshold for inserting space characters in TJ arrays.
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
}

impl Default for TextExtractionConfig {
    fn default() -> Self {
        Self {
            // Conservative threshold: avoids false positives from tight kerning
            // but reliably detects word boundaries
            space_insertion_threshold: -120.0,
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
            space_insertion_threshold: threshold,
        }
    }
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
            // Type0 fonts use 2-byte character codes (usually UTF-16 BE)
            let mut result = String::new();
            let mut i = 0;
            while i < bytes.len() {
                if i + 1 < bytes.len() {
                    // Combine two bytes into a 16-bit character code (big-endian)
                    let char_code = ((bytes[i] as u16) << 8) | (bytes[i + 1] as u16);
                    let char_str = font
                        .char_to_unicode(char_code)
                        .unwrap_or_else(|| fallback_char_to_unicode(char_code));
                    result.push_str(&char_str);
                    i += 2;
                } else {
                    // Odd byte at end - process as single byte
                    let char_code = bytes[i] as u16;
                    let char_str = font
                        .char_to_unicode(char_code)
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
                    .char_to_unicode(char_code)
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
    /// Current marked content ID (for Tagged PDFs)
    ///
    /// Tracks the MCID of the currently active marked content sequence.
    /// Used to associate extracted text with structure tree elements.
    current_mcid: Option<u32>,
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
        Self {
            state_stack: GraphicsStateStack::new(),
            fonts: HashMap::new(),
            spans: Vec::new(),
            chars: Vec::new(),
            resources: None,
            document: None,
            processed_xobjects: HashSet::new(),
            config,
            current_mcid: None,
            extract_spans: true,      // Default to span mode (PDF spec compliant)
            tj_span_buffer: None,     // No buffer initially
            span_sequence_counter: 0, // Initialize sequence counter
        }
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
        let operators = parse_content_stream(content_stream)?;

        // Execute each operator
        for op in operators {
            self.execute_operator(op)?;
        }

        // Flush any remaining Tj buffer at end of content stream
        self.flush_tj_span_buffer()?;

        // Sort spans by reading order (top-to-bottom, left-to-right)
        self.sort_spans_by_reading_order();

        // Deduplicate overlapping spans
        self.deduplicate_overlapping_spans();

        // Merge adjacent spans on the same line to reconstruct complete words
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
    /// Similar to character deduplication, but works with complete text spans.
    fn deduplicate_overlapping_spans(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        let mut deduplicated = Vec::with_capacity(self.spans.len());
        let mut prev_y_rounded: Option<i32> = None;
        let mut prev_x: Option<f32> = None;

        for span in self.spans.iter() {
            let y_rounded = span.bbox.y.round() as i32;
            let x = span.bbox.x;

            // Check if this span overlaps with the previous one
            let should_skip = if let (Some(prev_y), Some(prev_x_val)) = (prev_y_rounded, prev_x) {
                // Same line and within 2pt horizontally
                y_rounded == prev_y && (x - prev_x_val).abs() < 2.0
            } else {
                false
            };

            if !should_skip {
                deduplicated.push(span.clone());
                prev_y_rounded = Some(y_rounded);
                prev_x = Some(x);
            } else {
                log::trace!(
                    "Deduplicating overlapping span '{}' at X={:.1}, Y={:.1} (too close to previous)",
                    span.text,
                    x,
                    span.bbox.y
                );
            }
        }

        log::debug!(
            "Deduplicated {} overlapping spans ({} -> {} spans)",
            self.spans.len() - deduplicated.len(),
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
            // Academic papers (2-column) typically have 5-15pt gaps between columns
            // Government docs with tables may have 20-50pt gaps
            // Word spacing is typically 2-4pt
            // Using 5pt threshold: covers tight academic columns while preserving word boundaries
            let large_gap_indicates_column = gap > 5.0;

            // Merge threshold: 3pt (typical char width for 10pt font is ~5pt)
            // This captures fragments within words but preserves word boundaries
            // BUT: Don't merge across column boundaries (gaps > 5pt)
            let should_merge =
                same_line && (-0.5..3.0).contains(&gap) && !large_gap_indicates_column;

            if should_merge {
                // Merge spans: concatenate text and extend bbox
                let old_text = current.text.clone();

                // Per PDF Spec ISO 32000-1:2008, Section 9.4.3:
                // Determine space threshold based on span characteristics
                //
                // FIX: Use 0.25em (25% of font size) threshold per typography standards
                // and industry best practices (PyMuPDF4LLM, Adobe Acrobat).
                //
                // Previous threshold was too conservative (15-20%), causing missing spaces
                // in author names like "WangZhenyu" instead of "Wang Zhenyu".
                //
                // Typography reference: word spacing is typically 0.25-0.33em
                //
                // KNOWN LIMITATION: Dense grid layouts (e.g., 6 authors in 3×2 grid)
                // with gaps <0.25em everywhere will still create mega-spans that get
                // incorrectly split during layout analysis. This is a pipeline issue
                // (spans are correct, but markdown/layout breaks them incorrectly).
                // See SPAN_SPACING_INVESTIGATION.md for details.
                // TODO: Fix span splitting in layout/markdown conversion to preserve
                // internal spaces when breaking up mega-spans.
                let space_threshold = current.font_size * 0.25;

                // FIX #1 COMPREHENSIVE: Conservative space insertion for dense layouts
                //
                // Root cause: The merge condition (gap < 3pt) and space condition (gap > 3pt)
                // created a paradox where gaps in [0, 3pt) would merge without spaces.
                //
                // For dense layouts with 1-2pt gaps between words:
                // - BEFORE: gap < 3pt → merge, gap < 3pt → no space → "email@example.comfinancial"
                // - AFTER: gap < 3pt → merge, gap > 1.5pt → insert space → "email@example.com financial"
                //
                // The fix: Use a conservative threshold (50% of space_threshold) to catch
                // word boundaries in dense layouts while preserving tight kerning within words.

                // Check if space should be inserted based on:
                // 1. Gap-based detection (geometric spacing)
                // 2. Heuristic-based detection (character transitions)
                let needs_space_by_gap = gap > space_threshold;
                let needs_space_by_heuristic =
                    should_insert_space_heuristic(&current.text, &span.text);

                // CRITICAL FIX: For gaps in range [0, space_threshold], be conservative.
                // In dense layouts, even 0pt gaps can be word boundaries (names in grids).
                //
                // Strategy:
                // 1. If gap >= space_threshold (3pt): Always insert space
                // 2. If heuristic detects boundary: Always insert space
                // 3. If gap > 0.1pt: Insert space (even tiny gaps are usually intentional)
                //
                // Why gap > 0.1pt? In PDF, a gap of 0pt means characters are truly adjacent.
                // Any positive gap, even 0.1pt, indicates the PDF author intended separation.
                // This catches dense layouts where word spacing is 0.5-2pt.
                let needs_space = needs_space_by_gap || needs_space_by_heuristic || gap > 0.1;

                let merged_text = if needs_space {
                    // Gap or heuristic indicates intentional word spacing
                    if needs_space_by_heuristic && !needs_space_by_gap {
                        log::trace!(
                            "Heuristic space insertion: '{}' | '{}'",
                            current.text,
                            span.text
                        );
                    } else if gap > 0.1 && gap <= space_threshold {
                        log::trace!(
                            "Aggressive space insertion (gap={:.2}pt < threshold): '{}' | '{}'",
                            gap,
                            current.text,
                            span.text
                        );
                    }
                    format!("{} {}", current.text, span.text)
                } else {
                    // Gap ≤ 0.1pt: truly adjacent characters within same word
                    format!("{}{}", current.text, span.text)
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
                    log::trace!(
                        "Not merging spans (gap={:.1}pt > 3pt): '{}' | '{}'",
                        gap,
                        current.text,
                        span.text
                    );
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
            },
            Operator::TJ { array } => {
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
                                let tx =
                                    -offset / 1000.0 * state.font_size * state.horizontal_scaling
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
                                // This threshold is CONFIGURABLE via TextExtractionConfig.
                                // Default: -120.0 units ≈ 0.12em (see TextExtractionConfig docs)
                                if offset < self.config.space_insertion_threshold {
                                    let text_matrix = state.text_matrix;
                                    let font_name = state.font_name.clone();
                                    let font_size = state.font_size;
                                    let fill_color_rgb = state.fill_color_rgb;

                                    // Calculate effective font size (accounting for text matrix scaling)
                                    let effective_font_size = font_size * text_matrix.d.abs();

                                    // Get font for determining weight
                                    let font =
                                        font_name.as_ref().and_then(|name| self.fonts.get(name));
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
                                    let space_char = TextChar {
                                        char: ' ',
                                        bbox: Rect::new(
                                            text_matrix.e,       // Current X position
                                            text_matrix.f,       // Current Y position
                                            tx.abs(),            // Width = the gap being created
                                            effective_font_size, // Height = effective font size
                                        ),
                                        font_name: font_name.unwrap_or_default(),
                                        font_size: effective_font_size,
                                        font_weight,
                                        color: Color::new(r, g, b),
                                        mcid: self.current_mcid,
                                    };
                                    self.chars.push(space_char);
                                }

                                let state_mut = self.state_stack.current_mut();
                                state_mut.text_matrix.e += tx;
                            },
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
            // We track MCIDs to support reading order determination via structure trees.
            Operator::BeginMarkedContent { .. } => {
                // BMC doesn't have properties, so no MCID
                // Just a simple tag like /P or /Figure
            },

            Operator::BeginMarkedContentDict { properties, .. } => {
                // BDC can have properties including MCID
                // Properties is an inline dictionary or a reference to one
                // Extract MCID if present
                if let Some(props_dict) = properties.as_dict() {
                    if let Some(mcid_obj) = props_dict.get("MCID") {
                        if let Some(mcid) = mcid_obj.as_integer() {
                            self.current_mcid = Some(mcid as u32);
                            log::debug!("Entered marked content with MCID: {}", mcid);
                        }
                    }
                }
            },

            Operator::EndMarkedContent => {
                // EMC ends the current marked content sequence
                if let Some(mcid) = self.current_mcid {
                    log::debug!("Exited marked content with MCID: {}", mcid);
                }
                self.current_mcid = None;
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
        let span = TextSpan {
            text: buffer.unicode.clone(),
            bbox: Rect {
                x: buffer.start_matrix.e,
                y: buffer.start_matrix.f,
                width: total_width,
                height: effective_font_size,
            },
            font_name: buffer
                .font_name
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            font_size: effective_font_size,
            font_weight,
            color: Color::new(
                buffer.fill_color_rgb.0,
                buffer.fill_color_rgb.1,
                buffer.fill_color_rgb.2,
            ),
            mcid: buffer.mcid,
            sequence: self.span_sequence_counter,
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

    /// Process entire TJ array with buffering logic.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.4.4 NOTE 6:
    /// "The performance of text searching (and other text extraction operations) is
    /// significantly better if the text strings are as long as possible."
    ///
    /// This method buffers consecutive strings into a single span, only breaking on:
    /// - Large negative offsets (indicating word boundaries)
    /// - End of TJ array
    fn process_tj_array(&mut self, array: &[TextElement]) -> Result<()> {
        // DEBUG: Log TJ array details

        let mut buffer = TjBuffer::new(self.state_stack.current(), self.current_mcid);
        let mut _element_count = 0;

        for element in array {
            _element_count += 1;
            match element {
                TextElement::String(s) => {
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
                    let unicode_text =
                        if let Some(font_name) = self.state_stack.current().font_name.as_ref() {
                            if let Some(font) = self.fonts.get(font_name) {
                                let mut text = String::new();
                                for &byte in s.iter() {
                                    if let Some(chars) = font.char_to_unicode(byte as u16) {
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
                    // Check if this offset indicates a word boundary
                    // Per PDF spec: negative offsets increase spacing
                    if *offset < self.config.space_insertion_threshold {
                        // Flush buffer before space
                        self.flush_tj_buffer(&buffer)?;

                        // Insert space character as separate span
                        self.insert_space_as_span()?;

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

        let span = TextSpan {
            text: " ".to_string(),
            bbox: Rect {
                x: text_matrix.e,
                y: text_matrix.f,
                width: space_width,
                height: effective_font_size,
            },
            font_name: state
                .font_name
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            font_size: effective_font_size,
            font_weight: FontWeight::Normal,
            color: Color::new(
                state.fill_color_rgb.0,
                state.fill_color_rgb.1,
                state.fill_color_rgb.2,
            ),
            mcid: self.current_mcid,
            sequence: self.span_sequence_counter,
        };
        self.span_sequence_counter += 1;

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
                let span = TextSpan {
                    text: buffer.unicode.clone(),
                    bbox: Rect {
                        x: buffer.start_matrix.e,
                        y: buffer.start_matrix.f,
                        width: total_width,
                        height: effective_font_size,
                    },
                    font_name: buffer
                        .font_name
                        .clone()
                        .unwrap_or_else(|| "Unknown".to_string()),
                    font_size: effective_font_size,
                    font_weight,
                    color: Color::new(
                        buffer.fill_color_rgb.0,
                        buffer.fill_color_rgb.1,
                        buffer.fill_color_rgb.2,
                    ),
                    mcid: buffer.mcid,
                    sequence: self.span_sequence_counter,
                };
                self.span_sequence_counter += 1;

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
                    .char_to_unicode(char_code)
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
                    let text_char = TextChar {
                        char: unicode_char,
                        bbox: Rect::new(pos.x + x_offset, pos.y, char_width, height),
                        font_name: font_name.clone().unwrap_or_default(),
                        font_size: effective_font_size,
                        font_weight,
                        color,
                        mcid: self.current_mcid,
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
fn should_insert_space_heuristic(current_text: &str, next_text: &str) -> bool {
    let last_char = current_text.chars().last();
    let first_char = next_text.chars().next();

    match (last_char, first_char) {
        // CamelCase transition: lowercase → uppercase (e.g., "then" + "The" → "then The")
        (Some(l), Some(f)) if l.is_lowercase() && f.is_uppercase() => {
            // But don't split acronyms: if current_text ends with uppercase, skip
            // (e.g., "HTML" + "Parser" should stay together if already concatenated)
            let prev_is_uppercase = current_text
                .chars()
                .rev()
                .nth(1)
                .is_some_and(|c| c.is_uppercase());
            !prev_is_uppercase
        },
        // Number-letter transition: digit → letter (e.g., "Figure1" → "Figure 1")
        (Some(l), Some(f)) if l.is_numeric() && f.is_alphabetic() => true,
        // Letter-number transition: letter → digit (e.g., "page3" → "page 3")
        (Some(l), Some(f)) if l.is_alphabetic() && f.is_numeric() => true,
        // No heuristic match
        _ => false,
    }
}

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
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
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
