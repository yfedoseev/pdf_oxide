//! Character-to-Unicode mapping with priority-based fallback chain.
//!
//! Implements ISO 32000-1:2008 Section 9.10.2 Character-to-Unicode Mapping Priorities:
//! 1. ToUnicode CMap (highest priority)
//! 2. Adobe Glyph List (fallback 1)
//! 3. Predefined CMaps (fallback 2)
//! 4. ActualText attribute (fallback 3)
//! 5. Font encoding (lowest priority)
//!
//! This module provides a unified interface for character mapping that respects
//! the spec-defined priority order.

use super::adobe_glyph_list::ADOBE_GLYPH_LIST;
use super::cmap::CMap;
use std::collections::HashMap;

/// Character-to-Unicode mapper with priority-based fallback chain.
///
/// Implements the PDF spec's 5-level priority order for character-to-Unicode mapping.
/// This ensures characters are mapped correctly even in PDFs with custom encodings,
/// symbol fonts, or missing ToUnicode CMaps.
///
/// # Example
///
/// ```no_run
/// use pdf_oxide::fonts::character_mapper::CharacterMapper;
///
/// let mut mapper = CharacterMapper::new();
///
/// // Set ToUnicode CMap (priority 1)
/// let mut tounicode = std::collections::HashMap::new();
/// tounicode.insert(0x41, "A".to_string());
/// mapper.set_tounicode_cmap(Some(tounicode));
///
/// // Character 0x41 maps to "A" from ToUnicode
/// assert_eq!(mapper.map_character(0x41), Some("A".to_string()));
///
/// // Character 0x42 not in ToUnicode, falls back to Adobe Glyph List -> "B"
/// assert_eq!(mapper.map_character(0x42), Some("B".to_string()));
/// ```
#[derive(Clone)]
pub struct CharacterMapper {
    /// Priority 1: ToUnicode CMap (explicit character code to Unicode mapping)
    tounicode_cmap: Option<CMap>,

    /// Priority 5: Font encoding (character code to glyph name or character)
    font_encoding: Option<HashMap<u32, char>>,
}

impl CharacterMapper {
    /// Create a new character mapper with no mappings set.
    pub fn new() -> Self {
        Self {
            tounicode_cmap: None,
            font_encoding: None,
        }
    }

    /// Set the ToUnicode CMap (Priority 1 - highest).
    ///
    /// The ToUnicode CMap provides explicit character code to Unicode mappings
    /// from the PDF file. This has the highest priority in the mapping chain.
    ///
    /// # Arguments
    /// * `cmap` - The ToUnicode CMap, or None to remove it
    pub fn set_tounicode_cmap(&mut self, cmap: Option<CMap>) {
        self.tounicode_cmap = cmap;
    }

    /// Set the font encoding (Priority 5 - lowest).
    ///
    /// Font encoding provides a fallback mapping from character codes to characters.
    /// This is only used if higher-priority mappings are not available.
    ///
    /// # Arguments
    /// * `encoding` - HashMap mapping character codes to characters, or None to remove it
    pub fn set_font_encoding(&mut self, encoding: Option<HashMap<u32, char>>) {
        self.font_encoding = encoding;
    }

    /// Map a character code to a Unicode string using the priority chain.
    ///
    /// Implements the PDF spec's priority order:
    /// 1. ToUnicode CMap - if present and has mapping
    /// 2. Adobe Glyph List - fallback to standard glyph names
    /// 3. Predefined CMaps - (not yet implemented)
    /// 4. ActualText - (not yet implemented)
    /// 5. Font encoding - lowest priority
    ///
    /// # Arguments
    /// * `code` - The character code to map (typically 0-255 for simple fonts, up to 0xFFFF for CID)
    ///
    /// # Returns
    /// * `Some(string)` - The mapped Unicode character(s)
    /// * `None` - No mapping found in any priority level
    ///
    /// # Spec Reference
    /// ISO 32000-1:2008, Section 9.10.2 - Character-to-Unicode Mapping Priorities
    pub fn map_character(&self, code: u32) -> Option<String> {
        // Priority 1: ToUnicode CMap
        if let Some(ref cmap) = self.tounicode_cmap {
            if let Some(mapped) = cmap.get(&code) {
                return Some(mapped.clone());
            }
        }

        // Priority 2: Adobe Glyph List (standard glyph for code)
        if let Some(glyph_name) = self.code_to_glyph_name(code) {
            if let Some(unicode_str) = self.map_glyph_name_internal(&glyph_name) {
                return Some(unicode_str);
            }
        }

        // Priority 3: Predefined CMaps (not yet implemented - would go here)

        // Priority 4: ActualText (not yet implemented - would go here)

        // Priority 5: Font encoding
        if let Some(ref encoding) = self.font_encoding {
            if let Some(&ch) = encoding.get(&code) {
                return Some(ch.to_string());
            }
        }

        // No mapping found
        None
    }

    /// Map a glyph name to its Unicode representation.
    ///
    /// Uses the Adobe Glyph List to find the Unicode character(s) for a named glyph.
    /// This is the public interface for glyph name mapping.
    ///
    /// # Arguments
    /// * `glyph_name` - Name of the glyph (e.g., "A", "ampersand", "fi")
    ///
    /// # Returns
    /// * `Some(string)` - The Unicode character(s) for this glyph
    /// * `None` - Glyph name not found
    pub fn map_glyph_name(&self, glyph_name: &str) -> Option<String> {
        self.map_glyph_name_internal(glyph_name)
    }

    /// Internal helper for glyph name mapping.
    fn map_glyph_name_internal(&self, glyph_name: &str) -> Option<String> {
        // Look up in Adobe Glyph List
        ADOBE_GLYPH_LIST.get(glyph_name).map(|&ch| ch.to_string())
    }

    /// Convert a character code to a glyph name using standard mappings.
    ///
    /// For ASCII range (0x20-0x7E), this maps directly to character names.
    /// For other ranges, this uses predefined mappings or returns None.
    fn code_to_glyph_name(&self, code: u32) -> Option<String> {
        match code {
            // ASCII printable range
            0x20 => Some("space".to_string()),
            0x21 => Some("exclam".to_string()),
            0x22 => Some("quotedbl".to_string()),
            0x23 => Some("numbersign".to_string()),
            0x24 => Some("dollar".to_string()),
            0x25 => Some("percent".to_string()),
            0x26 => Some("ampersand".to_string()),
            0x27 => Some("quoteright".to_string()),
            0x28 => Some("parenleft".to_string()),
            0x29 => Some("parenright".to_string()),
            0x2A => Some("asterisk".to_string()),
            0x2B => Some("plus".to_string()),
            0x2C => Some("comma".to_string()),
            0x2D => Some("hyphen".to_string()),
            0x2E => Some("period".to_string()),
            0x2F => Some("slash".to_string()),

            // Digits 0-9 use glyph names "zero" through "nine"
            0x30 => Some("zero".to_string()),
            0x31 => Some("one".to_string()),
            0x32 => Some("two".to_string()),
            0x33 => Some("three".to_string()),
            0x34 => Some("four".to_string()),
            0x35 => Some("five".to_string()),
            0x36 => Some("six".to_string()),
            0x37 => Some("seven".to_string()),
            0x38 => Some("eight".to_string()),
            0x39 => Some("nine".to_string()),

            0x3A => Some("colon".to_string()),
            0x3B => Some("semicolon".to_string()),
            0x3C => Some("less".to_string()),
            0x3D => Some("equal".to_string()),
            0x3E => Some("greater".to_string()),
            0x3F => Some("question".to_string()),
            0x40 => Some("at".to_string()),

            // Uppercase A-Z
            0x41..=0x5A => {
                let ch = (code - 0x41) as u8 + b'A';
                Some((ch as char).to_string())
            },

            0x5B => Some("bracketleft".to_string()),
            0x5C => Some("backslash".to_string()),
            0x5D => Some("bracketright".to_string()),
            0x5E => Some("asciicircum".to_string()),
            0x5F => Some("underscore".to_string()),
            0x60 => Some("grave".to_string()),

            // Lowercase a-z
            0x61..=0x7A => {
                let ch = (code - 0x61) as u8 + b'a';
                Some((ch as char).to_string())
            },

            0x7B => Some("braceleft".to_string()),
            0x7C => Some("bar".to_string()),
            0x7D => Some("braceright".to_string()),
            0x7E => Some("asciitilde".to_string()),

            // Extended ASCII and beyond - would need more sophisticated mapping
            _ => None,
        }
    }
}

impl Default for CharacterMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn test_ascii_glyph_names() {
        let mapper = CharacterMapper::new();

        // Test ASCII character to glyph name conversion
        assert_eq!(mapper.code_to_glyph_name(0x20), Some("space".to_string()));
        assert_eq!(mapper.code_to_glyph_name(0x41), Some("A".to_string()));
        assert_eq!(mapper.code_to_glyph_name(0x61), Some("a".to_string()));
    }

    #[test]
    fn test_glyph_name_lookup() {
        let mapper = CharacterMapper::new();

        // Test that Adobe Glyph List lookups work
        assert!(mapper.map_glyph_name("A").is_some());
        assert!(mapper.map_glyph_name("space").is_some());
    }
}
