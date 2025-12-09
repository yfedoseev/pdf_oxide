//! Font management for PDF generation.
//!
//! This module provides font metrics and management for accurate
//! text positioning in generated PDFs.

use std::collections::HashMap;

/// Font manager for PDF generation.
///
/// Manages fonts and provides metrics for accurate text layout.
/// Currently supports PDF Base-14 fonts with standard metrics.
#[derive(Debug, Clone)]
pub struct FontManager {
    /// Registered fonts (name -> font info)
    fonts: HashMap<String, FontInfo>,
    /// Font ID counter for resource naming
    next_font_id: u32,
}

impl FontManager {
    /// Create a new font manager with Base-14 fonts.
    pub fn new() -> Self {
        let mut manager = Self {
            fonts: HashMap::new(),
            next_font_id: 1,
        };

        // Register PDF Base-14 fonts
        manager.register_base14_fonts();
        manager
    }

    /// Register the PDF Base-14 standard fonts.
    fn register_base14_fonts(&mut self) {
        // Helvetica family
        self.register_font(FontInfo::base14(
            "Helvetica",
            FontFamily::Helvetica,
            FontWeight::Normal,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Helvetica-Bold",
            FontFamily::Helvetica,
            FontWeight::Bold,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Helvetica-Oblique",
            FontFamily::Helvetica,
            FontWeight::Normal,
            true,
        ));
        self.register_font(FontInfo::base14(
            "Helvetica-BoldOblique",
            FontFamily::Helvetica,
            FontWeight::Bold,
            true,
        ));

        // Times family
        self.register_font(FontInfo::base14(
            "Times-Roman",
            FontFamily::Times,
            FontWeight::Normal,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Times-Bold",
            FontFamily::Times,
            FontWeight::Bold,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Times-Italic",
            FontFamily::Times,
            FontWeight::Normal,
            true,
        ));
        self.register_font(FontInfo::base14(
            "Times-BoldItalic",
            FontFamily::Times,
            FontWeight::Bold,
            true,
        ));

        // Courier family
        self.register_font(FontInfo::base14(
            "Courier",
            FontFamily::Courier,
            FontWeight::Normal,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Courier-Bold",
            FontFamily::Courier,
            FontWeight::Bold,
            false,
        ));
        self.register_font(FontInfo::base14(
            "Courier-Oblique",
            FontFamily::Courier,
            FontWeight::Normal,
            true,
        ));
        self.register_font(FontInfo::base14(
            "Courier-BoldOblique",
            FontFamily::Courier,
            FontWeight::Bold,
            true,
        ));

        // Symbol and ZapfDingbats
        self.register_font(FontInfo::base14_symbol("Symbol"));
        self.register_font(FontInfo::base14_symbol("ZapfDingbats"));
    }

    /// Register a font.
    fn register_font(&mut self, font: FontInfo) {
        self.fonts.insert(font.name.clone(), font);
    }

    /// Get font info by name.
    pub fn get_font(&self, name: &str) -> Option<&FontInfo> {
        self.fonts.get(name)
    }

    /// Get font info, falling back to Helvetica if not found.
    pub fn get_font_or_default(&self, name: &str) -> &FontInfo {
        self.fonts.get(name).unwrap_or_else(|| {
            self.fonts
                .get("Helvetica")
                .expect("Helvetica must be registered")
        })
    }

    /// Calculate the width of a string in the given font at the given size.
    ///
    /// Returns width in points.
    pub fn text_width(&self, text: &str, font_name: &str, font_size: f32) -> f32 {
        let font = self.get_font_or_default(font_name);
        font.text_width(text, font_size)
    }

    /// Calculate the width of a single character.
    pub fn char_width(&self, ch: char, font_name: &str, font_size: f32) -> f32 {
        let font = self.get_font_or_default(font_name);
        font.char_width(ch) * font_size / 1000.0
    }

    /// Get the next available font resource ID.
    pub fn next_font_resource_id(&mut self) -> String {
        let id = format!("F{}", self.next_font_id);
        self.next_font_id += 1;
        id
    }

    /// Check if a font name corresponds to a Base-14 font.
    pub fn is_base14(&self, name: &str) -> bool {
        self.fonts.get(name).map(|f| f.is_base14).unwrap_or(false)
    }

    /// Get all registered font names.
    pub fn font_names(&self) -> Vec<&str> {
        self.fonts.keys().map(|s| s.as_str()).collect()
    }

    /// Select the best matching font for the given criteria.
    pub fn select_font(&self, family: FontFamily, weight: FontWeight, italic: bool) -> &str {
        let name = match (family, weight, italic) {
            (FontFamily::Helvetica, FontWeight::Normal, false) => "Helvetica",
            (FontFamily::Helvetica, FontWeight::Bold, false) => "Helvetica-Bold",
            (FontFamily::Helvetica, FontWeight::Normal, true) => "Helvetica-Oblique",
            (FontFamily::Helvetica, FontWeight::Bold, true) => "Helvetica-BoldOblique",
            (FontFamily::Times, FontWeight::Normal, false) => "Times-Roman",
            (FontFamily::Times, FontWeight::Bold, false) => "Times-Bold",
            (FontFamily::Times, FontWeight::Normal, true) => "Times-Italic",
            (FontFamily::Times, FontWeight::Bold, true) => "Times-BoldItalic",
            (FontFamily::Courier, FontWeight::Normal, false) => "Courier",
            (FontFamily::Courier, FontWeight::Bold, false) => "Courier-Bold",
            (FontFamily::Courier, FontWeight::Normal, true) => "Courier-Oblique",
            (FontFamily::Courier, FontWeight::Bold, true) => "Courier-BoldOblique",
            (FontFamily::Symbol, _, _) => "Symbol",
            (FontFamily::ZapfDingbats, _, _) => "ZapfDingbats",
        };
        name
    }
}

impl Default for FontManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Font family classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontFamily {
    /// Helvetica (sans-serif)
    Helvetica,
    /// Times (serif)
    Times,
    /// Courier (monospace)
    Courier,
    /// Symbol
    Symbol,
    /// ZapfDingbats
    ZapfDingbats,
}

/// Font weight classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FontWeight {
    /// Normal weight
    #[default]
    Normal,
    /// Bold weight
    Bold,
}

/// Information about a font.
#[derive(Debug, Clone)]
pub struct FontInfo {
    /// Font name (e.g., "Helvetica-Bold")
    pub name: String,
    /// Font family
    pub family: FontFamily,
    /// Font weight
    pub weight: FontWeight,
    /// Whether the font is italic/oblique
    pub italic: bool,
    /// Whether this is a Base-14 font
    pub is_base14: bool,
    /// Character widths (glyph index -> width in 1/1000 of font size)
    widths: FontWidths,
    /// Ascender height (above baseline)
    pub ascender: f32,
    /// Descender depth (below baseline, negative)
    pub descender: f32,
    /// Line gap (extra space between lines)
    pub line_gap: f32,
    /// Cap height (height of capital letters)
    pub cap_height: f32,
    /// x-height (height of lowercase x)
    pub x_height: f32,
}

impl FontInfo {
    /// Create a Base-14 font info.
    fn base14(name: &str, family: FontFamily, weight: FontWeight, italic: bool) -> Self {
        let widths = FontWidths::for_base14(name);
        let metrics = base14_metrics(name);

        Self {
            name: name.to_string(),
            family,
            weight,
            italic,
            is_base14: true,
            widths,
            ascender: metrics.0,
            descender: metrics.1,
            line_gap: metrics.2,
            cap_height: metrics.3,
            x_height: metrics.4,
        }
    }

    /// Create a Base-14 symbol font.
    fn base14_symbol(name: &str) -> Self {
        Self {
            name: name.to_string(),
            family: if name == "Symbol" {
                FontFamily::Symbol
            } else {
                FontFamily::ZapfDingbats
            },
            weight: FontWeight::Normal,
            italic: false,
            is_base14: true,
            widths: FontWidths::Symbol,
            ascender: 800.0,
            descender: -200.0,
            line_gap: 0.0,
            cap_height: 700.0,
            x_height: 500.0,
        }
    }

    /// Calculate the width of text in this font.
    ///
    /// Returns width in points for the given font size.
    pub fn text_width(&self, text: &str, font_size: f32) -> f32 {
        let width_units: f32 = text.chars().map(|c| self.char_width(c)).sum();
        width_units * font_size / 1000.0
    }

    /// Get the width of a single character in font units (1/1000 of em).
    pub fn char_width(&self, ch: char) -> f32 {
        self.widths.width_for_char(ch)
    }

    /// Get the line height for this font at the given size.
    pub fn line_height(&self, font_size: f32) -> f32 {
        (self.ascender - self.descender + self.line_gap) * font_size / 1000.0
    }

    /// Get recommended line spacing multiplier.
    pub fn line_spacing_factor(&self) -> f32 {
        1.2 // Standard 120% line height
    }
}

/// Font width data.
#[derive(Debug, Clone)]
enum FontWidths {
    /// Proportional font with per-character widths
    Proportional(HashMap<char, f32>),
    /// Monospace font with fixed width
    Monospace(f32),
    /// Symbol font (use default width)
    Symbol,
}

impl FontWidths {
    /// Get widths for a Base-14 font.
    fn for_base14(name: &str) -> Self {
        match name {
            "Courier" | "Courier-Bold" | "Courier-Oblique" | "Courier-BoldOblique" => {
                FontWidths::Monospace(600.0)
            },
            "Symbol" | "ZapfDingbats" => FontWidths::Symbol,
            _ => FontWidths::Proportional(get_base14_widths(name)),
        }
    }

    /// Get width for a character.
    fn width_for_char(&self, ch: char) -> f32 {
        match self {
            FontWidths::Proportional(widths) => {
                *widths.get(&ch).unwrap_or(&500.0) // Default to 500 for unknown chars
            },
            FontWidths::Monospace(width) => *width,
            FontWidths::Symbol => 500.0,
        }
    }
}

/// Get metrics for a Base-14 font: (ascender, descender, line_gap, cap_height, x_height)
fn base14_metrics(name: &str) -> (f32, f32, f32, f32, f32) {
    match name {
        "Helvetica" | "Helvetica-Oblique" => (718.0, -207.0, 0.0, 718.0, 523.0),
        "Helvetica-Bold" | "Helvetica-BoldOblique" => (718.0, -207.0, 0.0, 718.0, 532.0),
        "Times-Roman" | "Times-Italic" => (683.0, -217.0, 0.0, 662.0, 450.0),
        "Times-Bold" | "Times-BoldItalic" => (676.0, -205.0, 0.0, 676.0, 461.0),
        "Courier" | "Courier-Oblique" => (629.0, -157.0, 0.0, 562.0, 426.0),
        "Courier-Bold" | "Courier-BoldOblique" => (626.0, -142.0, 0.0, 562.0, 439.0),
        _ => (750.0, -250.0, 0.0, 700.0, 500.0), // Default metrics
    }
}

/// Get character widths for Base-14 proportional fonts.
///
/// These are standard PostScript/PDF metrics in units of 1/1000 em.
fn get_base14_widths(name: &str) -> HashMap<char, f32> {
    let mut widths = HashMap::new();

    // Common ASCII characters with approximate standard widths
    // These are based on standard PostScript font metrics

    let (space_w, period_w, comma_w, hyphen_w, colon_w) = match name {
        "Helvetica" | "Helvetica-Oblique" => (278.0, 278.0, 278.0, 333.0, 278.0),
        "Helvetica-Bold" | "Helvetica-BoldOblique" => (278.0, 278.0, 278.0, 333.0, 333.0),
        "Times-Roman" | "Times-Italic" => (250.0, 250.0, 250.0, 333.0, 278.0),
        "Times-Bold" | "Times-BoldItalic" => (250.0, 250.0, 250.0, 333.0, 333.0),
        _ => (250.0, 250.0, 250.0, 333.0, 278.0),
    };

    // Whitespace and punctuation
    widths.insert(' ', space_w);
    widths.insert('.', period_w);
    widths.insert(',', comma_w);
    widths.insert('-', hyphen_w);
    widths.insert(':', colon_w);
    widths.insert(';', 278.0);
    widths.insert('!', 333.0);
    widths.insert('?', 500.0);
    widths.insert('\'', 222.0);
    widths.insert('"', 400.0);
    widths.insert('(', 333.0);
    widths.insert(')', 333.0);
    widths.insert('[', 333.0);
    widths.insert(']', 333.0);
    widths.insert('{', 333.0);
    widths.insert('}', 333.0);
    widths.insert('/', 278.0);
    widths.insert('\\', 278.0);
    widths.insert('@', 800.0);
    widths.insert('#', 556.0);
    widths.insert('$', 556.0);
    widths.insert('%', 889.0);
    widths.insert('^', 500.0);
    widths.insert('&', 722.0);
    widths.insert('*', 389.0);
    widths.insert('+', 584.0);
    widths.insert('=', 584.0);
    widths.insert('<', 584.0);
    widths.insert('>', 584.0);
    widths.insert('|', 280.0);
    widths.insert('`', 333.0);
    widths.insert('~', 584.0);
    widths.insert('_', 556.0);

    // Numbers (fairly consistent across fonts)
    for digit in '0'..='9' {
        widths.insert(digit, 556.0);
    }

    // Uppercase letters - Helvetica-style widths
    let uppercase_widths = match name {
        "Helvetica" | "Helvetica-Oblique" => [
            ('A', 722.0),
            ('B', 722.0),
            ('C', 722.0),
            ('D', 722.0),
            ('E', 667.0),
            ('F', 611.0),
            ('G', 778.0),
            ('H', 722.0),
            ('I', 278.0),
            ('J', 556.0),
            ('K', 722.0),
            ('L', 611.0),
            ('M', 833.0),
            ('N', 722.0),
            ('O', 778.0),
            ('P', 667.0),
            ('Q', 778.0),
            ('R', 722.0),
            ('S', 667.0),
            ('T', 611.0),
            ('U', 722.0),
            ('V', 667.0),
            ('W', 944.0),
            ('X', 667.0),
            ('Y', 667.0),
            ('Z', 611.0),
        ],
        "Helvetica-Bold" | "Helvetica-BoldOblique" => [
            ('A', 722.0),
            ('B', 722.0),
            ('C', 722.0),
            ('D', 722.0),
            ('E', 667.0),
            ('F', 611.0),
            ('G', 778.0),
            ('H', 722.0),
            ('I', 278.0),
            ('J', 556.0),
            ('K', 722.0),
            ('L', 611.0),
            ('M', 833.0),
            ('N', 722.0),
            ('O', 778.0),
            ('P', 667.0),
            ('Q', 778.0),
            ('R', 722.0),
            ('S', 667.0),
            ('T', 611.0),
            ('U', 722.0),
            ('V', 667.0),
            ('W', 944.0),
            ('X', 667.0),
            ('Y', 667.0),
            ('Z', 611.0),
        ],
        "Times-Roman" | "Times-Italic" => [
            ('A', 722.0),
            ('B', 667.0),
            ('C', 667.0),
            ('D', 722.0),
            ('E', 611.0),
            ('F', 556.0),
            ('G', 722.0),
            ('H', 722.0),
            ('I', 333.0),
            ('J', 389.0),
            ('K', 722.0),
            ('L', 611.0),
            ('M', 889.0),
            ('N', 722.0),
            ('O', 722.0),
            ('P', 556.0),
            ('Q', 722.0),
            ('R', 667.0),
            ('S', 556.0),
            ('T', 611.0),
            ('U', 722.0),
            ('V', 722.0),
            ('W', 944.0),
            ('X', 722.0),
            ('Y', 722.0),
            ('Z', 611.0),
        ],
        "Times-Bold" | "Times-BoldItalic" => [
            ('A', 722.0),
            ('B', 667.0),
            ('C', 722.0),
            ('D', 722.0),
            ('E', 667.0),
            ('F', 611.0),
            ('G', 778.0),
            ('H', 778.0),
            ('I', 389.0),
            ('J', 500.0),
            ('K', 778.0),
            ('L', 667.0),
            ('M', 944.0),
            ('N', 722.0),
            ('O', 778.0),
            ('P', 611.0),
            ('Q', 778.0),
            ('R', 722.0),
            ('S', 556.0),
            ('T', 667.0),
            ('U', 722.0),
            ('V', 722.0),
            ('W', 1000.0),
            ('X', 722.0),
            ('Y', 722.0),
            ('Z', 667.0),
        ],
        _ => [
            ('A', 722.0),
            ('B', 667.0),
            ('C', 667.0),
            ('D', 722.0),
            ('E', 611.0),
            ('F', 556.0),
            ('G', 722.0),
            ('H', 722.0),
            ('I', 333.0),
            ('J', 389.0),
            ('K', 722.0),
            ('L', 611.0),
            ('M', 889.0),
            ('N', 722.0),
            ('O', 722.0),
            ('P', 556.0),
            ('Q', 722.0),
            ('R', 667.0),
            ('S', 556.0),
            ('T', 611.0),
            ('U', 722.0),
            ('V', 722.0),
            ('W', 944.0),
            ('X', 722.0),
            ('Y', 722.0),
            ('Z', 611.0),
        ],
    };

    for (ch, w) in uppercase_widths {
        widths.insert(ch, w);
    }

    // Lowercase letters - Helvetica-style widths
    let lowercase_widths = match name {
        "Helvetica" | "Helvetica-Oblique" => [
            ('a', 556.0),
            ('b', 611.0),
            ('c', 556.0),
            ('d', 611.0),
            ('e', 556.0),
            ('f', 278.0),
            ('g', 611.0),
            ('h', 611.0),
            ('i', 222.0),
            ('j', 222.0),
            ('k', 556.0),
            ('l', 222.0),
            ('m', 833.0),
            ('n', 611.0),
            ('o', 611.0),
            ('p', 611.0),
            ('q', 611.0),
            ('r', 389.0),
            ('s', 556.0),
            ('t', 333.0),
            ('u', 611.0),
            ('v', 556.0),
            ('w', 778.0),
            ('x', 556.0),
            ('y', 556.0),
            ('z', 500.0),
        ],
        "Helvetica-Bold" | "Helvetica-BoldOblique" => [
            ('a', 556.0),
            ('b', 611.0),
            ('c', 556.0),
            ('d', 611.0),
            ('e', 556.0),
            ('f', 333.0),
            ('g', 611.0),
            ('h', 611.0),
            ('i', 278.0),
            ('j', 278.0),
            ('k', 556.0),
            ('l', 278.0),
            ('m', 889.0),
            ('n', 611.0),
            ('o', 611.0),
            ('p', 611.0),
            ('q', 611.0),
            ('r', 389.0),
            ('s', 556.0),
            ('t', 333.0),
            ('u', 611.0),
            ('v', 556.0),
            ('w', 778.0),
            ('x', 556.0),
            ('y', 556.0),
            ('z', 500.0),
        ],
        "Times-Roman" | "Times-Italic" => [
            ('a', 444.0),
            ('b', 500.0),
            ('c', 444.0),
            ('d', 500.0),
            ('e', 444.0),
            ('f', 333.0),
            ('g', 500.0),
            ('h', 500.0),
            ('i', 278.0),
            ('j', 278.0),
            ('k', 500.0),
            ('l', 278.0),
            ('m', 778.0),
            ('n', 500.0),
            ('o', 500.0),
            ('p', 500.0),
            ('q', 500.0),
            ('r', 333.0),
            ('s', 389.0),
            ('t', 278.0),
            ('u', 500.0),
            ('v', 500.0),
            ('w', 722.0),
            ('x', 500.0),
            ('y', 500.0),
            ('z', 444.0),
        ],
        "Times-Bold" | "Times-BoldItalic" => [
            ('a', 500.0),
            ('b', 556.0),
            ('c', 444.0),
            ('d', 556.0),
            ('e', 444.0),
            ('f', 333.0),
            ('g', 500.0),
            ('h', 556.0),
            ('i', 278.0),
            ('j', 333.0),
            ('k', 556.0),
            ('l', 278.0),
            ('m', 833.0),
            ('n', 556.0),
            ('o', 500.0),
            ('p', 556.0),
            ('q', 556.0),
            ('r', 444.0),
            ('s', 389.0),
            ('t', 333.0),
            ('u', 556.0),
            ('v', 500.0),
            ('w', 722.0),
            ('x', 500.0),
            ('y', 500.0),
            ('z', 444.0),
        ],
        _ => [
            ('a', 444.0),
            ('b', 500.0),
            ('c', 444.0),
            ('d', 500.0),
            ('e', 444.0),
            ('f', 333.0),
            ('g', 500.0),
            ('h', 500.0),
            ('i', 278.0),
            ('j', 278.0),
            ('k', 500.0),
            ('l', 278.0),
            ('m', 778.0),
            ('n', 500.0),
            ('o', 500.0),
            ('p', 500.0),
            ('q', 500.0),
            ('r', 333.0),
            ('s', 389.0),
            ('t', 278.0),
            ('u', 500.0),
            ('v', 500.0),
            ('w', 722.0),
            ('x', 500.0),
            ('y', 500.0),
            ('z', 444.0),
        ],
    };

    for (ch, w) in lowercase_widths {
        widths.insert(ch, w);
    }

    widths
}

/// Text layout helper for calculating text positioning.
#[derive(Debug)]
pub struct TextLayout {
    /// Font manager reference
    font_manager: FontManager,
}

impl TextLayout {
    /// Create a new text layout helper.
    pub fn new() -> Self {
        Self {
            font_manager: FontManager::new(),
        }
    }

    /// Create with a specific font manager.
    pub fn with_font_manager(font_manager: FontManager) -> Self {
        Self { font_manager }
    }

    /// Calculate wrapped lines for text within a given width.
    ///
    /// Returns a vector of (line_text, line_width) pairs.
    pub fn wrap_text(
        &self,
        text: &str,
        font_name: &str,
        font_size: f32,
        max_width: f32,
    ) -> Vec<(String, f32)> {
        let mut lines = Vec::new();
        let mut current_line = String::new();
        let mut current_width = 0.0;
        let space_width = self.font_manager.char_width(' ', font_name, font_size);

        for word in text.split_whitespace() {
            let word_width = self.font_manager.text_width(word, font_name, font_size);

            if current_line.is_empty() {
                // First word on line
                current_line = word.to_string();
                current_width = word_width;
            } else if current_width + space_width + word_width <= max_width {
                // Word fits on current line
                current_line.push(' ');
                current_line.push_str(word);
                current_width += space_width + word_width;
            } else {
                // Word doesn't fit, start new line
                lines.push((current_line, current_width));
                current_line = word.to_string();
                current_width = word_width;
            }
        }

        // Don't forget the last line
        if !current_line.is_empty() {
            lines.push((current_line, current_width));
        }

        if lines.is_empty() {
            lines.push((String::new(), 0.0));
        }

        lines
    }

    /// Calculate the bounding box dimensions for wrapped text.
    pub fn text_bounds(
        &self,
        text: &str,
        font_name: &str,
        font_size: f32,
        max_width: f32,
    ) -> (f32, f32) {
        let lines = self.wrap_text(text, font_name, font_size, max_width);
        let font = self.font_manager.get_font_or_default(font_name);
        let line_height = font.line_height(font_size) * font.line_spacing_factor();

        let max_line_width = lines.iter().map(|(_, w)| *w).fold(0.0_f32, f32::max);
        let total_height = lines.len() as f32 * line_height;

        (max_line_width, total_height)
    }

    /// Get the font manager.
    pub fn font_manager(&self) -> &FontManager {
        &self.font_manager
    }
}

impl Default for TextLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_manager_creation() {
        let manager = FontManager::new();
        assert!(manager.get_font("Helvetica").is_some());
        assert!(manager.get_font("Times-Roman").is_some());
        assert!(manager.get_font("Courier").is_some());
    }

    #[test]
    fn test_base14_fonts() {
        let manager = FontManager::new();
        assert!(manager.is_base14("Helvetica"));
        assert!(manager.is_base14("Helvetica-Bold"));
        assert!(manager.is_base14("Times-Roman"));
        assert!(manager.is_base14("Courier"));
        assert!(!manager.is_base14("Arial")); // Not a Base-14 font
    }

    #[test]
    fn test_text_width_calculation() {
        let manager = FontManager::new();

        // "Hello" in Helvetica 12pt
        let width = manager.text_width("Hello", "Helvetica", 12.0);
        assert!(width > 0.0);
        assert!(width < 100.0); // Reasonable range for 5 chars at 12pt
    }

    #[test]
    fn test_monospace_consistency() {
        let manager = FontManager::new();

        // All characters should have the same width in Courier
        let w1 = manager.char_width('i', "Courier", 12.0);
        let w2 = manager.char_width('m', "Courier", 12.0);
        let w3 = manager.char_width('W', "Courier", 12.0);

        assert!((w1 - w2).abs() < 0.001);
        assert!((w2 - w3).abs() < 0.001);
    }

    #[test]
    fn test_proportional_variance() {
        let manager = FontManager::new();

        // 'i' should be narrower than 'W' in Helvetica
        let w_i = manager.char_width('i', "Helvetica", 12.0);
        let w_w = manager.char_width('W', "Helvetica", 12.0);

        assert!(w_i < w_w);
    }

    #[test]
    fn test_font_selection() {
        let manager = FontManager::new();

        assert_eq!(
            manager.select_font(FontFamily::Helvetica, FontWeight::Normal, false),
            "Helvetica"
        );
        assert_eq!(
            manager.select_font(FontFamily::Helvetica, FontWeight::Bold, false),
            "Helvetica-Bold"
        );
        assert_eq!(
            manager.select_font(FontFamily::Times, FontWeight::Normal, true),
            "Times-Italic"
        );
        assert_eq!(
            manager.select_font(FontFamily::Courier, FontWeight::Bold, true),
            "Courier-BoldOblique"
        );
    }

    #[test]
    fn test_font_metrics() {
        let manager = FontManager::new();
        let font = manager.get_font("Helvetica").unwrap();

        assert!(font.ascender > 0.0);
        assert!(font.descender < 0.0);
        assert!(font.cap_height > 0.0);
        assert!(font.x_height > 0.0);
        assert!(font.x_height < font.cap_height);
    }

    #[test]
    fn test_line_height() {
        let manager = FontManager::new();
        let font = manager.get_font("Helvetica").unwrap();

        let line_height = font.line_height(12.0);
        // Raw line height is the em-square height (ascender - descender)
        // For Helvetica: (718 - (-207)) * 12 / 1000 = 11.1 points
        assert!(line_height > 10.0);
        assert!(line_height < 15.0);

        // With spacing factor (1.2), we get a comfortable reading line height
        let visual_line_height = line_height * font.line_spacing_factor();
        assert!(visual_line_height > 12.0); // Should be > font size with spacing
    }

    #[test]
    fn test_text_layout_wrap() {
        let layout = TextLayout::new();

        let text = "The quick brown fox jumps over the lazy dog";
        let lines = layout.wrap_text(text, "Helvetica", 12.0, 100.0);

        assert!(lines.len() > 1); // Should wrap into multiple lines
        for (line, width) in &lines {
            assert!(!line.is_empty() || lines.len() == 1);
            assert!(*width <= 100.0 || line.split_whitespace().count() == 1);
        }
    }

    #[test]
    fn test_text_bounds() {
        let layout = TextLayout::new();

        let text = "Hello World";
        let (width, height) = layout.text_bounds(text, "Helvetica", 12.0, 1000.0);

        assert!(width > 0.0);
        assert!(height > 0.0);
    }

    #[test]
    fn test_empty_text() {
        let layout = TextLayout::new();
        let lines = layout.wrap_text("", "Helvetica", 12.0, 100.0);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].0.is_empty());
    }
}
