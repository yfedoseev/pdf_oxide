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
/// ```
/// use pdf_oxide::fonts::character_mapper::CharacterMapper;
///
/// let mapper = CharacterMapper::new();
///
/// // Character 0x41 (ASCII 'A') maps via Adobe Glyph List to "A"
/// assert_eq!(mapper.map_character(0x41), Some("A".to_string()));
///
/// // Character 0x42 (ASCII 'B') maps via Adobe Glyph List to "B"
/// assert_eq!(mapper.map_character(0x42), Some("B".to_string()));
///
/// // Character 0x20 (space) maps via Adobe Glyph List to " "
/// assert_eq!(mapper.map_character(0x20), Some(" ".to_string()));
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

        // No mapping found - return U+FFFD replacement character per PDF Spec 9.10.2
        Some("\u{FFFD}".to_string())
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
    /// For extended ASCII (0x80-0xFF), uses WinAnsiEncoding fallback.
    /// For other ranges, returns None.
    fn code_to_glyph_name(&self, code: u32) -> Option<String> {
        // Try standard ASCII first
        if code <= 0x7E {
            return self.code_to_glyph_name_ascii(code);
        }

        // Try extended ASCII (WinAnsiEncoding) fallback
        self.code_to_glyph_name_extended(code)
    }

    /// Map ASCII range (0x20-0x7E) to standard glyph names.
    fn code_to_glyph_name_ascii(&self, code: u32) -> Option<String> {
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

            _ => None,
        }
    }

    /// Map extended ASCII range (0x80-0xFF) to glyph names using WinAnsiEncoding.
    ///
    /// This implements the WinAnsiEncoding (Windows-1252) character mappings
    /// which are commonly used as a fallback in PDF documents.
    /// Per PDF Spec ISO 32000-1:2008 Section 9.6.6.1.
    pub fn code_to_glyph_name_extended(&self, code: u32) -> Option<String> {
        // WinAnsiEncoding (Windows-1252) mappings for 0x80-0xFF range
        // Maps character codes to standard glyph names per Adobe Glyph List
        match code {
            // 0x80-0x8F: Special WinAnsiEncoding characters
            0x80 => Some("Euro".to_string()), // € (U+20AC)
            0x81 => None,                     // Undefined in WinAnsiEncoding
            0x82 => Some("quotesinglbase".to_string()), // ‚ (U+201A)
            0x83 => Some("florin".to_string()), // ƒ (U+0192)
            0x84 => Some("quotedblbase".to_string()), // „ (U+201E)
            0x85 => Some("ellipsis".to_string()), // … (U+2026)
            0x86 => Some("dagger".to_string()), // † (U+2020)
            0x87 => Some("daggerdbl".to_string()), // ‡ (U+2021)
            0x88 => Some("circumflex".to_string()), // ˆ (U+02C6)
            0x89 => Some("perthousand".to_string()), // ‰ (U+2030)
            0x8A => Some("Scaron".to_string()), // Š (U+0160)
            0x8B => Some("guilsinglleft".to_string()), // ‹ (U+2039)
            0x8C => Some("OEligature".to_string()), // Œ (U+0152)
            0x8D => None,                     // Undefined
            0x8E => Some("Zcaron".to_string()), // Ž (U+017D)
            0x8F => None,                     // Undefined

            // 0x90-0x9F: More WinAnsiEncoding specials
            0x90 => None,                               // Undefined
            0x91 => Some("quoteleft".to_string()),      // ' (U+2018)
            0x92 => Some("quoteright".to_string()),     // ' (U+2019)
            0x93 => Some("quotedblleft".to_string()),   // " (U+201C)
            0x94 => Some("quotedblright".to_string()),  // " (U+201D)
            0x95 => Some("bullet".to_string()),         // • (U+2022)
            0x96 => Some("endash".to_string()),         // – (U+2013) - COMMON: en-dash
            0x97 => Some("emdash".to_string()),         // — (U+2014) - COMMON: em-dash
            0x98 => Some("tilde".to_string()),          // ˜ (U+02DC)
            0x99 => Some("trademark".to_string()),      // ™ (U+2122) - COMMON: trademark
            0x9A => Some("scaron".to_string()),         // š (U+0161)
            0x9B => Some("guilsinglright".to_string()), // › (U+203A)
            0x9C => Some("oeligature".to_string()),     // œ (U+0153)
            0x9D => None,                               // Undefined
            0x9E => Some("zcaron".to_string()),         // ž (U+017E)
            0x9F => Some("ydieresis".to_string()),      // Ÿ (U+0178)

            // 0xA0-0xFF: Latin-1 Supplement (ISO-8859-1 compatible)
            0xA0 => Some("space".to_string()), // Non-breaking space (U+00A0)
            0xA1 => Some("exclamdown".to_string()), // ¡ (U+00A1)
            0xA2 => Some("cent".to_string()),  // ¢ (U+00A2)
            0xA3 => Some("sterling".to_string()), // £ (U+00A3) - COMMON: pound sign
            0xA4 => Some("currency".to_string()), // ¤ (U+00A4)
            0xA5 => Some("yen".to_string()),   // ¥ (U+00A5)
            0xA6 => Some("brokenbar".to_string()), // ¦ (U+00A6)
            0xA7 => Some("section".to_string()), // § (U+00A7)
            0xA8 => Some("dieresis".to_string()), // ¨ (U+00A8)
            0xA9 => Some("copyright".to_string()), // © (U+00A9) - COMMON: copyright
            0xAA => Some("ordfeminine".to_string()), // ª (U+00AA)
            0xAB => Some("guillemotleft".to_string()), // « (U+00AB)
            0xAC => Some("logicalnot".to_string()), // ¬ (U+00AC)
            0xAD => Some("hyphen".to_string()), // Soft hyphen (U+00AD)
            0xAE => Some("registered".to_string()), // ® (U+00AE) - COMMON: registered
            0xAF => Some("macron".to_string()), // ¯ (U+00AF)
            0xB0 => Some("degree".to_string()), // ° (U+00B0) - COMMON: degree
            0xB1 => Some("plusminus".to_string()), // ± (U+00B1)
            0xB2 => Some("twosuperior".to_string()), // ² (U+00B2)
            0xB3 => Some("threesuperior".to_string()), // ³ (U+00B3)
            0xB4 => Some("acute".to_string()), // ´ (U+00B4)
            0xB5 => Some("mu".to_string()),    // µ (U+00B5)
            0xB6 => Some("paragraph".to_string()), // ¶ (U+00B6)
            0xB7 => Some("periodcentered".to_string()), // · (U+00B7)
            0xB8 => Some("cedilla".to_string()), // ¸ (U+00B8)
            0xB9 => Some("onesuperior".to_string()), // ¹ (U+00B9)
            0xBA => Some("ordmasculine".to_string()), // º (U+00BA)
            0xBB => Some("guillemotright".to_string()), // » (U+00BB)
            0xBC => Some("onequarter".to_string()), // ¼ (U+00BC)
            0xBD => Some("onehalf".to_string()), // ½ (U+00BD)
            0xBE => Some("threequarters".to_string()), // ¾ (U+00BE)
            0xBF => Some("questiondown".to_string()), // ¿ (U+00BF)

            // 0xC0-0xFF: Accented uppercase and lowercase letters
            0xC0 => Some("Agrave".to_string()),      // À (U+00C0)
            0xC1 => Some("Aacute".to_string()),      // Á (U+00C1)
            0xC2 => Some("Acircumflex".to_string()), // Â (U+00C2)
            0xC3 => Some("Atilde".to_string()),      // Ã (U+00C3)
            0xC4 => Some("Adieresis".to_string()),   // Ä (U+00C4)
            0xC5 => Some("Aring".to_string()),       // Å (U+00C5)
            0xC6 => Some("AEligature".to_string()),  // Æ (U+00C6)
            0xC7 => Some("Ccedilla".to_string()),    // Ç (U+00C7)
            0xC8 => Some("Egrave".to_string()),      // È (U+00C8)
            0xC9 => Some("Eacute".to_string()),      // É (U+00C9)
            0xCA => Some("Ecircumflex".to_string()), // Ê (U+00CA)
            0xCB => Some("Edieresis".to_string()),   // Ë (U+00CB)
            0xCC => Some("Igrave".to_string()),      // Ì (U+00CC)
            0xCD => Some("Iacute".to_string()),      // Í (U+00CD)
            0xCE => Some("Icircumflex".to_string()), // Î (U+00CE)
            0xCF => Some("Idieresis".to_string()),   // Ï (U+00CF)
            0xD0 => Some("Eth".to_string()),         // Ð (U+00D0)
            0xD1 => Some("Ntilde".to_string()),      // Ñ (U+00D1)
            0xD2 => Some("Ograve".to_string()),      // Ò (U+00D2)
            0xD3 => Some("Oacute".to_string()),      // Ó (U+00D3)
            0xD4 => Some("Ocircumflex".to_string()), // Ô (U+00D4)
            0xD5 => Some("Otilde".to_string()),      // Õ (U+00D5)
            0xD6 => Some("Odieresis".to_string()),   // Ö (U+00D6)
            0xD7 => Some("multiply".to_string()),    // × (U+00D7)
            0xD8 => Some("Oslash".to_string()),      // Ø (U+00D8)
            0xD9 => Some("Ugrave".to_string()),      // Ù (U+00D9)
            0xDA => Some("Uacute".to_string()),      // Ú (U+00DA)
            0xDB => Some("Ucircumflex".to_string()), // Û (U+00DB)
            0xDC => Some("Udieresis".to_string()),   // Ü (U+00DC)
            0xDD => Some("Yacute".to_string()),      // Ý (U+00DD)
            0xDE => Some("Thorn".to_string()),       // Þ (U+00DE)
            0xDF => Some("germandbls".to_string()),  // ß (U+00DF)
            0xE0 => Some("agrave".to_string()),      // à (U+00E0)
            0xE1 => Some("aacute".to_string()),      // á (U+00E1)
            0xE2 => Some("acircumflex".to_string()), // â (U+00E2)
            0xE3 => Some("atilde".to_string()),      // ã (U+00E3)
            0xE4 => Some("adieresis".to_string()),   // ä (U+00E4) - COMMON: German a-umlaut
            0xE5 => Some("aring".to_string()),       // å (U+00E5)
            0xE6 => Some("aeligature".to_string()),  // æ (U+00E6)
            0xE7 => Some("ccedilla".to_string()),    // ç (U+00E7) - COMMON: French c-cedilla
            0xE8 => Some("egrave".to_string()),      // è (U+00E8)
            0xE9 => Some("eacute".to_string()),      // é (U+00E9) - COMMON: French e-acute
            0xEA => Some("ecircumflex".to_string()), // ê (U+00EA)
            0xEB => Some("edieresis".to_string()),   // ë (U+00EB)
            0xEC => Some("igrave".to_string()),      // ì (U+00EC)
            0xED => Some("iacute".to_string()),      // í (U+00ED)
            0xEE => Some("icircumflex".to_string()), // î (U+00EE)
            0xEF => Some("idieresis".to_string()),   // ï (U+00EF)
            0xF0 => Some("eth".to_string()),         // ð (U+00F0)
            0xF1 => Some("ntilde".to_string()),      // ñ (U+00F1)
            0xF2 => Some("ograve".to_string()),      // ò (U+00F2)
            0xF3 => Some("oacute".to_string()),      // ó (U+00F3)
            0xF4 => Some("ocircumflex".to_string()), // ô (U+00F4)
            0xF5 => Some("otilde".to_string()),      // õ (U+00F5)
            0xF6 => Some("odieresis".to_string()),   // ö (U+00F6)
            0xF7 => Some("divide".to_string()),      // ÷ (U+00F7)
            0xF8 => Some("oslash".to_string()),      // ø (U+00F8)
            0xF9 => Some("ugrave".to_string()),      // ù (U+00F9)
            0xFA => Some("uacute".to_string()),      // ú (U+00FA)
            0xFB => Some("ucircumflex".to_string()), // û (U+00FB)
            0xFC => Some("udieresis".to_string()),   // ü (U+00FC) - COMMON: German u-umlaut
            0xFD => Some("yacute".to_string()),      // ý (U+00FD)
            0xFE => Some("thorn".to_string()),       // þ (U+00FE)
            0xFF => Some("ydieresis".to_string()),   // ÿ (U+00FF)

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
