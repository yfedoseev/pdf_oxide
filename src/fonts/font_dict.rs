//! Font dictionary parsing.
//!
//! This module handles parsing of PDF font dictionaries and encoding information.
//! Fonts in PDF can have various encodings, and the ToUnicode CMap provides the
//! most accurate character-to-Unicode mapping.
//!
//! Phase 4, Task 4.5

use crate::document::PdfDocument;
use crate::error::{Error, Result};
use crate::fonts::cmap::{CMap, parse_tounicode_cmap};
use crate::layout::text_block::FontWeight;
use crate::object::Object;
use std::collections::HashMap;
use std::sync::Arc;

/// Font information extracted from a PDF font dictionary.
#[derive(Debug, Clone)]
pub struct FontInfo {
    /// Base font name (e.g., "Times-Roman", "Helvetica-Bold")
    pub base_font: String,
    /// Font subtype (e.g., "Type1", "TrueType", "Type0")
    pub subtype: String,
    /// Encoding information
    pub encoding: Encoding,
    /// ToUnicode CMap (character code to Unicode mapping)
    pub to_unicode: Option<CMap>,
    /// Font weight from FontDescriptor (400 = normal, 700 = bold)
    pub font_weight: Option<i32>,
    /// Font descriptor flags (bit field)
    /// Bit 1: FixedPitch, Bit 2: Serif, Bit 3: Symbolic, Bit 4: Script,
    /// Bit 6: Nonsymbolic, Bit 7: Italic
    /// PDF Spec: ISO 32000-1:2008, Table 5.20
    pub flags: Option<i32>,
    /// Stem thickness (vertical) from FontDescriptor (used for weight inference)
    /// PDF Spec: ISO 32000-1:2008, Section 9.6.2
    /// Typical values: <80 = light, 80-110 = normal/medium, >110 = bold
    pub stem_v: Option<f32>,
    /// Embedded TrueType font data (from FontFile2 stream)
    /// Shared via Arc to avoid expensive cloning
    pub embedded_font_data: Option<Arc<Vec<u8>>>,
    /// Character widths in 1000ths of em (PDF units)
    /// For simple fonts (Type1, TrueType): array indexed by (char_code - first_char)
    /// PDF Spec: ISO 32000-1:2008, Section 9.7.4
    pub widths: Option<Vec<f32>>,
    /// First character code covered by widths array
    /// Used to map character codes to width array indices
    pub first_char: Option<u32>,
    /// Last character code covered by widths array
    pub last_char: Option<u32>,
    /// Default width for characters not in widths array (in 1000ths of em)
    /// Typical values: 500-600 for proportional fonts, 600 for monospace
    pub default_width: f32,
}

/// Font encoding types.
#[derive(Debug, Clone)]
pub enum Encoding {
    /// Standard PDF encoding (WinAnsiEncoding, MacRomanEncoding, etc.)
    Standard(String),
    /// Custom encoding with explicit character mappings
    Custom(HashMap<u8, char>),
    /// Identity encoding (typically used for CID fonts)
    Identity,
}

impl FontInfo {
    /// Parse font information from a font dictionary object.
    ///
    /// # Arguments
    ///
    /// * `dict` - The font dictionary object (should be a Dictionary or Stream)
    /// * `doc` - The PDF document (needed to load referenced objects)
    ///
    /// # Returns
    ///
    /// A FontInfo struct containing the parsed font information.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The object is not a dictionary
    /// - Required font dictionary entries are missing or invalid
    /// - Referenced objects cannot be loaded
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::document::PdfDocument;
    /// use pdf_oxide::fonts::FontInfo;
    /// use pdf_oxide::object::ObjectRef;
    ///
    /// # fn example(mut doc: PdfDocument, font_ref: ObjectRef) -> Result<(), Box<dyn std::error::Error>> {
    /// let font_obj = doc.load_object(font_ref)?;
    /// let font_info = FontInfo::from_dict(&font_obj, &mut doc)?;
    /// println!("Font: {}", font_info.base_font);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_dict(dict: &Object, doc: &mut PdfDocument) -> Result<Self> {
        let font_dict = dict.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "Font object is not a dictionary".to_string(),
        })?;

        // Extract BaseFont (required)
        let base_font = font_dict
            .get("BaseFont")
            .and_then(|obj| obj.as_name())
            .unwrap_or("Unknown")
            .to_string();

        // Extract Subtype (required)
        let subtype = font_dict
            .get("Subtype")
            .and_then(|obj| obj.as_name())
            .unwrap_or("Unknown")
            .to_string();

        // Log Type 3 fonts for Phase 7C tracking
        if subtype == "Type3" {
            log::warn!(
                "Font '{}' is Type 3 - may require special glyph name mapping (Phase 7C)",
                base_font
            );
        }

        // Parse FontDescriptor FIRST to get font flags (needed for encoding decision)
        // PDF Spec: ISO 32000-1:2008, Section 9.6.2 - Font Descriptor
        let (font_weight, flags, stem_v, embedded_font_data) = if let Some(descriptor_ref) =
            font_dict
                .get("FontDescriptor")
                .and_then(|obj| obj.as_reference())
        {
            // Load the FontDescriptor object
            if let Ok(descriptor_obj) = doc.load_object(descriptor_ref) {
                if let Some(descriptor_dict) = descriptor_obj.as_dict() {
                    let weight = descriptor_dict
                        .get("FontWeight")
                        .and_then(|weight_obj| weight_obj.as_integer())
                        .map(|w| w as i32);

                    let descriptor_flags = descriptor_dict
                        .get("Flags")
                        .and_then(|flags_obj| flags_obj.as_integer())
                        .map(|f| f as i32);

                    let stem_v_value = descriptor_dict.get("StemV").and_then(|sv_obj| {
                        sv_obj
                            .as_real()
                            .map(|r| r as f32)
                            .or_else(|| sv_obj.as_integer().map(|i| i as f32))
                    });

                    // Load embedded font data from FontFile2 (TrueType), FontFile (Type 1), or FontFile3 (CFF/OpenType)
                    let embedded_font = if let Some(ff2_obj) = descriptor_dict.get("FontFile2") {
                        log::info!("Font '{}' has FontFile2 entry", base_font);
                        ff2_obj
                            .as_reference()
                            .and_then(|ff2_ref| {
                                doc.load_object(ff2_ref).ok().map(|obj| (obj, ff2_ref))
                            })
                            .and_then(|(ff2_stream, ff2_ref)| {
                                doc.decode_stream_with_encryption(&ff2_stream, ff2_ref).ok()
                            })
                            .map(|data| {
                                log::info!(
                                    "Font '{}' loaded embedded TrueType font ({} bytes)",
                                    base_font,
                                    data.len()
                                );
                                Arc::new(data)
                            })
                    } else if let Some(ff3_obj) = descriptor_dict.get("FontFile3") {
                        log::info!("Font '{}' has FontFile3 entry (CFF/OpenType)", base_font);
                        ff3_obj
                            .as_reference()
                            .and_then(|ff3_ref| {
                                doc.load_object(ff3_ref).ok().map(|obj| (obj, ff3_ref))
                            })
                            .and_then(|(ff3_stream, ff3_ref)| {
                                doc.decode_stream_with_encryption(&ff3_stream, ff3_ref).ok()
                            })
                            .map(|data| {
                                log::info!(
                                    "Font '{}' loaded embedded CFF/OpenType font ({} bytes)",
                                    base_font,
                                    data.len()
                                );
                                Arc::new(data)
                            })
                    } else if descriptor_dict.get("FontFile").is_some() {
                        log::info!(
                            "Font '{}' has FontFile entry (Type 1 - not supported for cmap)",
                            base_font
                        );
                        None
                    } else {
                        log::debug!("Font '{}' has no embedded font data", base_font);
                        None
                    };

                    (weight, descriptor_flags, stem_v_value, embedded_font)
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            }
        } else {
            (None, None, None, None)
        };

        // Helper function to check if font is symbolic (bit 3 set)
        let is_symbolic_font = |flags_opt: Option<i32>| -> bool {
            if let Some(flags_value) = flags_opt {
                const SYMBOLIC_BIT: i32 = 1 << 2; // Bit 3
                (flags_value & SYMBOLIC_BIT) != 0
            } else {
                // Fallback: check font name
                let name_lower = base_font.to_lowercase();
                name_lower.contains("symbol")
                    || name_lower.contains("zapf")
                    || name_lower.contains("dingbat")
            }
        };

        // Parse encoding (now that we have flags)
        // PDF Spec: ISO 32000-1:2008, Section 9.6.6.1
        // "For symbolic fonts, the Encoding entry is ignored"
        let encoding = if let Some(enc_obj) = font_dict.get("Encoding") {
            // Dereference if it's a reference
            let resolved_enc_obj = if let Some(obj_ref) = enc_obj.as_reference() {
                doc.load_object(obj_ref)?
            } else {
                enc_obj.clone()
            };

            if is_symbolic_font(flags) {
                log::debug!(
                    "Font '{}' is symbolic (Flags={:?}) - /Encoding entry will be IGNORED per PDF spec",
                    base_font,
                    flags
                );
                // For symbolic fonts, ignore /Encoding and use built-in encoding
                // This will be handled in char_to_unicode() Priority 2
                Encoding::Standard("StandardEncoding".to_string()) // Placeholder, not actually used
            } else {
                log::debug!("Font '{}' using /Encoding entry", base_font);
                Self::parse_encoding(&resolved_enc_obj, doc)?
            }
        } else {
            // No /Encoding entry
            if is_symbolic_font(flags) {
                log::debug!(
                    "Font '{}' is symbolic with no /Encoding - will use built-in encoding (Symbol/ZapfDingbats)",
                    base_font
                );
                // Placeholder - actual encoding determined by font name in char_to_unicode()
                Encoding::Standard("SymbolicBuiltIn".to_string())
            } else {
                log::debug!(
                    "Font '{}' has no /Encoding entry - defaulting to StandardEncoding",
                    base_font
                );
                Encoding::Standard("StandardEncoding".to_string())
            }
        };

        // Parse ToUnicode CMap if present
        let to_unicode = if let Some(cmap_ref) = font_dict
            .get("ToUnicode")
            .and_then(|obj| obj.as_reference())
        {
            let cmap_opt = doc
                .load_object(cmap_ref)
                .ok()
                .and_then(|cmap_obj| doc.decode_stream_with_encryption(&cmap_obj, cmap_ref).ok())
                .and_then(|decoded| parse_tounicode_cmap(&decoded).ok());

            if let Some(ref cmap) = cmap_opt {
                log::info!(
                    "ToUnicode CMap loaded for font '{}': {} mappings found",
                    base_font,
                    cmap.len()
                );
                // Log first 5 mappings for debugging
                for (code, unicode) in cmap.iter().take(5) {
                    log::debug!("  0x{:04X} → {:?}", code, unicode);
                }
            } else {
                log::warn!("Failed to parse ToUnicode CMap for font '{}'", base_font);
            }
            cmap_opt
        } else {
            if subtype == "Type0" {
                log::warn!("Type0 font '{}' has no ToUnicode entry!", base_font);
            }
            None
        };

        // Parse /Widths array for glyph width information
        // PDF Spec: ISO 32000-1:2008, Section 9.7.4 - Font Widths
        //
        // For simple fonts (Type1, TrueType), widths are specified as an array
        // of integers in 1000ths of em, indexed from FirstChar to LastChar.
        //
        // Note: Type0 (CID) fonts use a different /W array format (not yet implemented)
        let (widths, first_char, last_char) = if subtype != "Type0" {
            // Try to parse /Widths array
            let widths_opt = font_dict.get("Widths").and_then(|widths_obj| {
                // Handle both direct arrays and references
                let resolved = if let Some(ref_obj) = widths_obj.as_reference() {
                    doc.load_object(ref_obj).ok()?
                } else {
                    widths_obj.clone()
                };

                resolved.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|obj| {
                            // Widths can be integers or reals
                            obj.as_integer()
                                .map(|i| i as f32)
                                .or_else(|| obj.as_real().map(|r| r as f32))
                        })
                        .collect::<Vec<f32>>()
                })
            });

            let first = font_dict
                .get("FirstChar")
                .and_then(|obj| obj.as_integer())
                .map(|i| i as u32);

            let last = font_dict
                .get("LastChar")
                .and_then(|obj| obj.as_integer())
                .map(|i| i as u32);

            if widths_opt.is_some() {
                log::debug!(
                    "Font '{}': parsed {} widths (FirstChar={:?}, LastChar={:?})",
                    base_font,
                    widths_opt.as_ref().map(|w| w.len()).unwrap_or(0),
                    first,
                    last
                );
            } else {
                log::debug!("Font '{}': no /Widths array found, will use default width", base_font);
            }

            (widths_opt, first, last)
        } else {
            log::debug!("Font '{}': Type0 font, /W array parsing not yet implemented", base_font);
            (None, None, None)
        };

        // Set default width based on font characteristics
        // PDF Spec: Typical values are 500-600 for proportional fonts, ~600 for monospace
        let default_width = if let Some(flags_val) = flags {
            const FIXED_PITCH_BIT: i32 = 1 << 0; // Bit 1
            if (flags_val & FIXED_PITCH_BIT) != 0 {
                600.0 // Monospace font
            } else {
                500.0 // Proportional font
            }
        } else {
            // No flags, use middle-ground default
            550.0
        };

        Ok(FontInfo {
            base_font,
            subtype,
            encoding,
            to_unicode,
            font_weight,
            flags,
            stem_v,
            embedded_font_data,
            widths,
            first_char,
            last_char,
            default_width,
        })
    }

    /// Parse encoding from an encoding object.
    ///
    /// Handles both named encodings (e.g., /WinAnsiEncoding) and encoding dictionaries
    /// with /Differences arrays that override specific character codes.
    ///
    /// # PDF Spec Reference
    ///
    /// ISO 32000-1:2008, Section 9.6.6.2 - Character Encoding
    ///
    /// A /Differences array has the format:
    /// ```pdf
    /// /Encoding <<
    ///     /BaseEncoding /WinAnsiEncoding
    ///     /Differences [code1 /name1 /name2 ... codeN /nameN ...]
    /// >>
    /// ```
    ///
    /// Where integers specify starting codes, and names specify glyphs for consecutive codes.
    fn parse_encoding(enc_obj: &Object, _doc: &mut PdfDocument) -> Result<Encoding> {
        // Encoding can be either a name or a dictionary
        if let Some(name) = enc_obj.as_name() {
            // Standard encoding names
            match name {
                "WinAnsiEncoding" => Ok(Encoding::Standard("WinAnsiEncoding".to_string())),
                "MacRomanEncoding" => Ok(Encoding::Standard("MacRomanEncoding".to_string())),
                "MacExpertEncoding" => Ok(Encoding::Standard("MacExpertEncoding".to_string())),
                "Identity-H" | "Identity-V" => Ok(Encoding::Identity),
                _ => Ok(Encoding::Standard(name.to_string())),
            }
        } else if let Some(dict) = enc_obj.as_dict() {
            // Custom encoding dictionary - parse /Differences array

            // Step 1: Get base encoding (if specified)
            let mut encoding_map: HashMap<u8, char> = if let Some(base_enc_obj) =
                dict.get("BaseEncoding")
            {
                if let Some(base_name) = base_enc_obj.as_name() {
                    // Build initial encoding from base encoding
                    let mut map = HashMap::new();
                    for code in 0u8..=255 {
                        if let Some(unicode_str) = standard_encoding_lookup(base_name, code) {
                            // Convert the first character of the unicode string
                            if let Some(ch) = unicode_str.chars().next() {
                                map.insert(code, ch);
                            }
                        }
                    }
                    map
                } else {
                    HashMap::new()
                }
            } else {
                // No base encoding specified - start with StandardEncoding as default
                let mut map = HashMap::new();
                for code in 0u8..=255 {
                    if let Some(unicode_str) = standard_encoding_lookup("StandardEncoding", code) {
                        if let Some(ch) = unicode_str.chars().next() {
                            map.insert(code, ch);
                        }
                    }
                }
                map
            };

            // Step 2: Apply /Differences array if present
            if let Some(differences_obj) = dict.get("Differences") {
                log::info!("Found /Differences array in encoding dictionary");
                if let Some(diff_array) = differences_obj.as_array() {
                    log::info!("/Differences array has {} items", diff_array.len());
                    let mut current_code: u32 = 0;

                    for item in diff_array {
                        match item {
                            Object::Integer(code) => {
                                // New starting code
                                current_code = *code as u32;
                            },
                            Object::Name(glyph_name) => {
                                // Log ALL glyphs for code 0x64 (even if lookup fails)
                                if current_code == 0x64 {
                                    log::info!(
                                        "/Differences: code 0x64 has glyph name /{}",
                                        glyph_name
                                    );
                                }

                                // Map glyph name to Unicode character
                                if let Some(unicode_char) = glyph_name_to_unicode(glyph_name) {
                                    if current_code <= 255 {
                                        encoding_map.insert(current_code as u8, unicode_char);
                                        // Log ligature mappings AND code 0x64 (for rho debugging)
                                        if is_ligature_char(unicode_char) || current_code == 0x64 {
                                            log::info!(
                                                "/Differences: code {} → /{} → '{}' (U+{:04X})",
                                                current_code,
                                                glyph_name,
                                                unicode_char,
                                                unicode_char as u32
                                            );
                                        }
                                    } else {
                                        log::warn!(
                                            "Character code {} in /Differences array exceeds u8 range",
                                            current_code
                                        );
                                    }
                                } else if current_code == 0x64 {
                                    log::warn!(
                                        "/Differences: code 0x64 glyph name /{} NOT FOUND in glyph_name_to_unicode lookup table",
                                        glyph_name
                                    );
                                } else {
                                    log::debug!(
                                        "Unknown glyph name '{}' at code {} in /Differences array",
                                        glyph_name,
                                        current_code
                                    );
                                }
                                current_code += 1;
                            },
                            _ => {
                                // Invalid item in /Differences array - skip
                                log::warn!("Unexpected item in /Differences array: {:?}", item);
                            },
                        }
                    }

                    log::debug!(
                        "Parsed /Differences array with {} custom mappings",
                        encoding_map.len()
                    );
                } else {
                    log::warn!("/Differences is not an array");
                }
            }

            // If we have custom mappings, return Custom encoding
            if !encoding_map.is_empty() {
                // Log ligature mappings for debugging
                for (code, ch) in &encoding_map {
                    if is_ligature_char(*ch) {
                        log::debug!(
                            "Custom encoding has ligature: code {} → '{}' (U+{:04X})",
                            code,
                            ch,
                            *ch as u32
                        );
                    }
                }
                Ok(Encoding::Custom(encoding_map))
            } else {
                // Fallback to StandardEncoding if no differences were parsed
                Ok(Encoding::Standard("StandardEncoding".to_string()))
            }
        } else {
            Ok(Encoding::Standard("StandardEncoding".to_string()))
        }
    }

    /// Map a character code to a Unicode string.
    ///
    /// Priority:
    /// 1. ToUnicode CMap (most accurate)
    /// 2. Built-in encoding
    /// 3. Symbol font encoding (for Symbol/ZapfDingbats fonts)
    /// 4. Ligature expansion (for ligature characters)
    /// 5. Identity mapping (as fallback)
    ///
    /// # Arguments
    ///
    /// * `char_code` - The character code from the PDF content stream
    ///
    /// # Returns
    ///
    /// The Unicode string for this character, or None if no mapping exists.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::fonts::FontInfo;
    /// # fn example(font: &FontInfo) {
    /// if let Some(unicode) = font.char_to_unicode(0x41) {
    ///     println!("Character: {}", unicode); // Should print "A"
    /// }
    /// # }
    /// ```
    /// Convert a character code to Unicode string.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.10.2 "Mapping Character Codes to Unicode Values":
    ///
    /// Priority order (STRICTLY FOLLOWED):
    /// 1. ToUnicode CMap (if present) - highest priority, NO EXCEPTIONS
    /// 2. Predefined encodings for simple fonts with standard glyphs
    /// 3. Font descriptor's symbolic flag + built-in encoding (e.g., Symbol, ZapfDingbats)
    /// 4. Font's /Encoding + /Differences
    ///
    /// IMPORTANT: We do NOT apply heuristics to override ToUnicode. If the PDF has
    /// a buggy ToUnicode CMap, that is a PDF authoring error, not our responsibility
    /// to "fix" by guessing what the author meant.
    /// Get glyph width for a character code.
    ///
    /// Returns width in 1000ths of em (PDF units) per PDF Spec ISO 32000-1:2008, Section 9.7.4.
    /// Must be multiplied by (font_size / 1000) to get actual width in user space units.
    ///
    /// # Arguments
    ///
    /// * `char_code` - Character code from PDF content stream (e.g., byte value from Tj/TJ operator)
    ///
    /// # Returns
    ///
    /// Width in 1000ths of em. Returns `default_width` if the character code is not
    /// in the widths array or if widths are not available for this font.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pdf_oxide::fonts::FontInfo;
    ///
    /// # fn example(font: &FontInfo) {
    /// // Get width for character 'A' (code 65)
    /// let width = font.get_glyph_width(65);
    /// let font_size = 12.0;
    /// let actual_width = width * font_size / 1000.0;
    /// println!("Width of 'A' at 12pt: {:.2}pt", actual_width);
    /// # }
    /// ```
    pub fn get_glyph_width(&self, char_code: u16) -> f32 {
        if let Some(widths) = &self.widths {
            if let Some(first_char) = self.first_char {
                let index = char_code as i32 - first_char as i32;
                if index >= 0 && (index as usize) < widths.len() {
                    return widths[index as usize];
                }
            }
        }
        self.default_width
    }

    /// Convert a character code to Unicode string.
    ///
    /// This method looks up the character code in the font's encoding tables
    /// (ToUnicode CMap, built-in encoding, or glyph name mappings) and returns
    /// the corresponding Unicode string if found.
    pub fn char_to_unicode(&self, char_code: u16) -> Option<String> {
        // Convert u16 to u32 for CMap lookup (supports multi-byte codes)
        let char_code_u32 = char_code as u32;

        // ==================================================================================
        // PRIORITY 1: ToUnicode CMap (PDF Spec Section 9.10.2, Method 1)
        // ==================================================================================
        // "If the font dictionary contains a ToUnicode CMap, use that CMap to convert
        // the character code to Unicode."
        //
        // QUALITY HEURISTIC: Skip U+FFFD (replacement character) mappings.
        // Some PDF authoring tools write U+FFFD in ToUnicode CMaps when they can't
        // determine the correct Unicode value. This is effectively saying "I don't know".
        // We treat U+FFFD mappings the same as missing entries and fall back to Priority 2.
        //
        // This matches industry practice (PyMuPDF) and fixes 57 PDFs (16%) with en-dash issues.
        // See ENDASH_ISSUE_ROOT_CAUSE.md for full analysis.
        if let Some(cmap) = &self.to_unicode {
            if let Some(unicode) = cmap.get(&char_code_u32) {
                // Skip U+FFFD mappings - treat as missing entry
                if unicode == "\u{FFFD}" {
                    log::warn!(
                        "ToUnicode CMap has U+FFFD for code 0x{:02X} in font '{}' - falling back to Priority 2",
                        char_code,
                        self.base_font
                    );
                    // Fall through to Priority 2 (predefined encodings)
                } else {
                    log::debug!(
                        "ToUnicode CMap: font='{}' code=0x{:02X} → '{}'",
                        self.base_font,
                        char_code,
                        unicode
                    );
                    return Some(unicode.clone());
                }
            } else {
                // DIAGNOSTIC: Log when ToUnicode CMap exists but lookup fails
                log::warn!(
                    "ToUnicode CMap MISS: font='{}' subtype='{}' code=0x{:04X} (cmap has {} entries)",
                    self.base_font,
                    self.subtype,
                    char_code,
                    cmap.len()
                );
            }
        } else {
            // DIAGNOSTIC: Log when ToUnicode CMap is missing
            if self.subtype == "Type0" {
                log::warn!(
                    "Type0 font '{}' missing ToUnicode CMap! This will cause character scrambling.",
                    self.base_font
                );
            }
        }

        // ==================================================================================
        // PRIORITY 2: Predefined Encodings (PDF Spec Section 9.10.2, Method 2)
        // ==================================================================================
        // For symbolic fonts (Flags bit 3 set), the PDF spec requires us to IGNORE any
        // /Encoding entry and use the font's built-in encoding directly.
        //
        // PDF Spec ISO 32000-1:2008, Section 9.6.6.1:
        // "For symbolic fonts, the Encoding entry is ignored; characters are mapped directly
        // using their character codes to glyphs in the font."
        //
        // Common symbolic fonts: Symbol (Greek/math), ZapfDingbats (decorative)
        if self.is_symbolic() {
            let font_name_lower = self.base_font.to_lowercase();

            // Symbol font: Maps character codes to Greek letters and mathematical symbols
            // Standard encoding defined in PDF spec Annex D.4
            if font_name_lower.contains("symbol") {
                if let Some(unicode_char) = symbol_encoding_lookup(char_code as u8) {
                    log::debug!(
                        "Symbolic font '{}': code 0x{:02X} → '{}' (U+{:04X}) [using Symbol encoding]",
                        self.base_font,
                        char_code,
                        unicode_char,
                        unicode_char as u32
                    );
                    return Some(unicode_char.to_string());
                }
            }
            // ZapfDingbats font: Maps character codes to decorative symbols
            // Standard encoding defined in PDF spec Annex D.5
            else if font_name_lower.contains("zapf") || font_name_lower.contains("dingbat") {
                if let Some(unicode_char) = zapf_dingbats_encoding_lookup(char_code as u8) {
                    log::debug!(
                        "Symbolic font '{}': code 0x{:02X} → '{}' (U+{:04X}) [using ZapfDingbats encoding]",
                        self.base_font,
                        char_code,
                        unicode_char,
                        unicode_char as u32
                    );
                    return Some(unicode_char.to_string());
                }
            }

            // For other symbolic fonts without specific encoding, fall through to /Encoding
            // (though spec says to ignore /Encoding, some PDFs may still work with it)
        }

        // ==================================================================================
        // PRIORITY 3: Font's /Encoding Entry (PDF Spec Section 9.10.2, Method 3)
        // ==================================================================================
        // For non-symbolic fonts, use the /Encoding entry which can be:
        // - A predefined encoding name (e.g., WinAnsiEncoding, MacRomanEncoding)
        // - A custom encoding dictionary with /BaseEncoding and /Differences array
        //
        // The /Differences array allows overriding specific character codes with custom
        // glyph names, which are then mapped to Unicode via the Adobe Glyph List (AGL).
        match &self.encoding {
            Encoding::Standard(name) => {
                // Predefined encodings: StandardEncoding, WinAnsiEncoding, MacRomanEncoding, etc.
                if let Some(unicode) = standard_encoding_lookup(name, char_code as u8) {
                    log::debug!(
                        "Standard encoding '{}': code 0x{:02X} → '{}'",
                        name,
                        char_code,
                        unicode
                    );
                    return Some(unicode);
                }
            },
            Encoding::Custom(map) => {
                // Custom encoding with /Differences array
                // Maps character code → glyph name → Unicode (via AGL)
                if let Some(&custom_char) = map.get(&(char_code as u8)) {
                    log::debug!(
                        "Custom encoding: code 0x{:02X} → '{}' (U+{:04X})",
                        char_code,
                        custom_char,
                        custom_char as u32
                    );

                    // Handle ligatures (ff, fi, fl, ffi, ffl) by expanding to component characters
                    // This is NOT in the PDF spec but improves text extraction usability
                    if is_ligature_char(custom_char) {
                        if let Some(expanded) = expand_ligature_char(custom_char) {
                            return Some(expanded.to_string());
                        }
                    }

                    return Some(custom_char.to_string());
                }
            },
            Encoding::Identity => {
                // Identity-H or Identity-V encoding for CID fonts
                // Character code is used directly as Unicode value
                if let Some(ch) = char::from_u32(char_code as u32) {
                    log::debug!(
                        "Identity encoding: code 0x{:02X} → '{}' (U+{:04X})",
                        char_code,
                        ch,
                        ch as u32
                    );
                    return Some(ch.to_string());
                }
            },
        }

        // ==================================================================================
        // PRIORITY 4: Fallback - No Mapping Found
        // ==================================================================================
        // If we reach here, the character is either:
        // - A control character (0x00-0x1F, 0x7F-0x9F) - intentionally omitted
        // - A character code outside all known encodings
        // - From a malformed PDF missing encoding information
        //
        // Control characters don't have visible representations, so returning None
        // (which becomes empty string) is more appropriate than returning � (U+FFFD).
        log::debug!(
            "No Unicode mapping for font '{}' code=0x{:02X} (symbolic={}, encoding={:?}) - likely control char",
            self.base_font,
            char_code,
            self.is_symbolic(),
            self.encoding
        );
        None
    }

    /// Determine the font weight using a comprehensive cascade of PDF spec methods.
    ///
    /// Priority order per PDF Spec ISO 32000-1:2008:
    /// 1. FontWeight field from FontDescriptor (Table 122) - MOST RELIABLE
    /// 2. ForceBold flag (bit 19) from Flags field (Table 123)
    /// 3. Font name heuristics (fallback for legacy PDFs)
    /// 4. StemV analysis (stem thickness correlates with weight)
    ///
    /// # Returns
    ///
    /// FontWeight enum value (Thin to Black scale)
    ///
    /// # PDF Spec References
    ///
    /// - Table 122 (page 456): FontWeight values 100-900
    /// - Table 123 (page 457): ForceBold flag at bit 19 (0x80000)
    /// - Section 9.6.2: StemV field interpretation
    pub fn get_font_weight(&self) -> FontWeight {
        // ==================================================================================
        // PRIORITY 1: FontWeight Field (PDF Spec Table 122)
        // ==================================================================================
        // Most reliable method. If present, use directly.
        if let Some(weight_value) = self.font_weight {
            return FontWeight::from_pdf_value(weight_value);
        }

        // ==================================================================================
        // PRIORITY 2: ForceBold Flag (PDF Spec Table 123, Bit 19)
        // ==================================================================================
        // If ForceBold flag is set, font is explicitly bold.
        // Bit 19 = 0x80000 (524288 decimal)
        if let Some(flags_value) = self.flags {
            const FORCE_BOLD_BIT: i32 = 0x80000; // Bit 19 = 524288
            if (flags_value & FORCE_BOLD_BIT) != 0 {
                log::debug!("Font '{}': ForceBold flag set (bit 19) → Bold", self.base_font);
                return FontWeight::Bold;
            }
        }

        // ==================================================================================
        // PRIORITY 3: Font Name Heuristics
        // ==================================================================================
        // Fallback for fonts without FontDescriptor or with missing fields.
        // Checks for bold-indicating keywords in the font name.
        let name_lower = self.base_font.to_lowercase();

        // Check for explicit weight keywords in order of strength
        if name_lower.contains("black") || name_lower.contains("heavy") {
            return FontWeight::Black; // 900
        }
        if name_lower.contains("extrabold") || name_lower.contains("ultrabold") {
            return FontWeight::ExtraBold; // 800
        }
        if name_lower.contains("bold") {
            // Distinguish between "SemiBold" and "Bold"
            if name_lower.contains("semibold") || name_lower.contains("demibold") {
                return FontWeight::SemiBold; // 600
            }
            return FontWeight::Bold; // 700
        }
        if name_lower.contains("medium") {
            return FontWeight::Medium; // 500
        }
        if name_lower.contains("light") {
            if name_lower.contains("extralight") || name_lower.contains("ultralight") {
                return FontWeight::ExtraLight; // 200
            }
            return FontWeight::Light; // 300
        }
        if name_lower.contains("thin") {
            return FontWeight::Thin; // 100
        }

        // ==================================================================================
        // PRIORITY 4: StemV Analysis (EXPERIMENTAL)
        // ==================================================================================
        // StemV measures vertical stem thickness. Empirically:
        // - StemV > 110: Usually bold (700+)
        // - StemV 80-110: Medium (500-600)
        // - StemV < 80: Normal or lighter (400-)
        //
        // NOTE: This is a heuristic and may not be reliable for all fonts.
        // PDF spec does not mandate this correlation.
        if let Some(stem_v) = self.stem_v {
            log::debug!("Font '{}': Using StemV analysis (StemV={})", self.base_font, stem_v);

            if stem_v > 110.0 {
                return FontWeight::Bold; // 700
            } else if stem_v >= 80.0 {
                return FontWeight::Medium; // 500
            }
            // If StemV < 80, continue to default (Normal)
        }

        // ==================================================================================
        // DEFAULT: Normal Weight (400)
        // ==================================================================================
        // If no other method yields a weight, assume normal.
        FontWeight::Normal
    }

    /// Check if this font is bold (convenience method).
    ///
    /// Returns true if font weight is SemiBold (600) or higher.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if font.is_bold() {
    ///     // Apply bold markdown formatting
    /// }
    /// ```
    pub fn is_bold(&self) -> bool {
        self.get_font_weight().is_bold()
    }

    /// Check if this font is likely italic based on the font name.
    ///
    /// This is a heuristic check looking for "Italic" or "Oblique" in the font name.
    pub fn is_italic(&self) -> bool {
        let name_lower = self.base_font.to_lowercase();
        name_lower.contains("italic") || name_lower.contains("oblique")
    }

    /// Check if this is a symbolic font based on FontDescriptor flags.
    ///
    /// Symbolic fonts (bit 3 set in /Flags) contain glyphs outside the Adobe standard
    /// Latin character set. For symbolic fonts, the PDF spec requires ignoring any
    /// Encoding entry and using direct character code mapping to the font's built-in encoding.
    ///
    /// Common symbolic fonts: Symbol, ZapfDingbats
    ///
    /// PDF Spec: ISO 32000-1:2008, Table 5.20 - Font descriptor flags
    /// Bit 3: Symbolic - Font contains glyphs outside Adobe standard Latin character set
    /// Bit 6: Nonsymbolic - Font uses Adobe standard Latin character set (mutually exclusive with bit 3)
    pub fn is_symbolic(&self) -> bool {
        // Priority 1: Check FontDescriptor /Flags bit 3
        if let Some(flags_value) = self.flags {
            // Bit 3 = 0x04 (1 << 2, since bits are numbered starting at 1 in PDF spec)
            const SYMBOLIC_BIT: i32 = 1 << 2; // Bit 3
            return (flags_value & SYMBOLIC_BIT) != 0;
        }

        // Priority 2: Fallback to font name heuristic
        let name_lower = self.base_font.to_lowercase();
        name_lower.contains("symbol")
            || name_lower.contains("zapf")
            || name_lower.contains("dingbat")
    }
}

/// Map a PDF glyph name to a Unicode character.
///
/// This function implements the Adobe Glyph List (AGL) specification,
/// which defines standard mappings from PostScript glyph names to Unicode.
/// This is essential for parsing /Differences arrays in custom encodings.
///
/// # Arguments
///
/// * `glyph_name` - The PostScript glyph name (e.g., "bullet", "emdash", "Aacute")
///
/// # Returns
///
/// The corresponding Unicode character, or None if the glyph name is not recognized.
///
/// # References
///
/// - Adobe Glyph List Specification: https://github.com/adobe-type-tools/agl-specification
/// - PDF 32000-1:2008, Section 9.6.6.2 (Differences Arrays)
///
/// # Examples
///
/// ```ignore
/// # use pdf_oxide::fonts::font_dict::glyph_name_to_unicode;
/// assert_eq!(glyph_name_to_unicode("bullet"), Some('•'));
/// assert_eq!(glyph_name_to_unicode("emdash"), Some('—'));
/// assert_eq!(glyph_name_to_unicode("A"), Some('A'));
/// assert_eq!(glyph_name_to_unicode("unknown"), None);
/// ```ignore
fn glyph_name_to_unicode(glyph_name: &str) -> Option<char> {
    // Priority 1: Adobe Glyph List (AGL) lookup - O(1) with perfect hash
    // PDF Spec: ISO 32000-1:2008, Section 9.10.2
    if let Some(&unicode_char) = super::adobe_glyph_list::ADOBE_GLYPH_LIST.get(glyph_name) {
        return Some(unicode_char);
    }

    // Priority 2: Parse "uniXXXX" format (e.g., uni0041 -> A)
    // Common in custom fonts and font subsets
    if glyph_name.starts_with("uni") && glyph_name.len() == 7 {
        if let Ok(code_point) = u32::from_str_radix(&glyph_name[3..], 16) {
            if let Some(c) = char::from_u32(code_point) {
                return Some(c);
            }
        }
    }

    // Priority 3: Parse "uXXXX" format (e.g., u0041 -> A)
    // Alternative format used by some PDF generators
    if glyph_name.starts_with('u') && glyph_name.len() >= 5 {
        if let Ok(code_point) = u32::from_str_radix(&glyph_name[1..], 16) {
            if let Some(c) = char::from_u32(code_point) {
                return Some(c);
            }
        }
    }

    // Unknown glyph name - not in AGL and not a recognized format
    log::debug!("Unknown glyph name not in Adobe Glyph List: '{}'", glyph_name);
    None
}

// Removed old implementation - replaced with compact AGL lookup above
// Old code: ~350 lines of match arms with ~200 hardcoded glyphs
// New code: 4281 glyphs from official Adobe Glyph List via perfect hash map
#[allow(dead_code)]
fn _old_glyph_name_to_unicode_removed() {
    // This function body intentionally left empty.
    // The old match-based implementation has been replaced with
    // a lookup in the complete Adobe Glyph List static map.
    // See super::adobe_glyph_list::ADOBE_GLYPH_LIST for the new implementation.
}

// Old implementation removed - was 350+ lines of hardcoded match arms
// Now using complete Adobe Glyph List with 4281 entries from adobe_glyph_list module

/// Check if a character is a ligature.
///
/// This function identifies Unicode ligature characters (U+FB00 to U+FB06)
/// that are commonly used in PDFs for typographic ligatures.
///
/// # Arguments
///
/// * `c` - The character to check
///
/// # Returns
///
/// `true` if the character is a ligature, `false` otherwise.
///
/// # Examples
///
/// ```ignore
/// # use pdf_oxide::fonts::font_dict::is_ligature_char;
/// assert_eq!(is_ligature_char('ﬁ'), true);  // U+FB01
/// assert_eq!(is_ligature_char('ﬂ'), true);  // U+FB02
/// assert_eq!(is_ligature_char('A'), false);
/// ```ignore
fn is_ligature_char(c: char) -> bool {
    matches!(
        c,
        'ﬁ' |  // fi - U+FB01
        'ﬂ' |  // fl - U+FB02
        'ﬀ' |  // ff - U+FB00
        'ﬃ' |  // ffi - U+FB03
        'ﬄ' // ffl - U+FB04
    )
}

/// Expand a ligature character to its ASCII equivalent.
///
/// This function handles the Unicode ligature characters (U+FB00 to U+FB06)
/// and expands them to their multi-character ASCII equivalents.
///
/// # Arguments
///
/// * `c` - The character to potentially expand
///
/// # Returns
///
/// The expanded string if `c` is a ligature, None otherwise.
///
/// # Examples
///
/// ```ignore
/// # use pdf_oxide::fonts::font_dict::expand_ligature_char;
/// assert_eq!(expand_ligature_char('ﬁ'), Some("fi"));
/// assert_eq!(expand_ligature_char('ﬂ'), Some("fl"));
/// assert_eq!(expand_ligature_char('A'), None);
/// ```ignore
fn expand_ligature_char(c: char) -> Option<&'static str> {
    match c {
        'ﬁ' => Some("fi"),  // U+FB01
        'ﬂ' => Some("fl"),  // U+FB02
        'ﬀ' => Some("ff"),  // U+FB00
        'ﬃ' => Some("ffi"), // U+FB03
        'ﬄ' => Some("ffl"), // U+FB04
        _ => None,
    }
}

/// Look up a character in the Adobe Symbol font encoding.
///
/// This function implements the Symbol font encoding table as defined in
/// PDF Specification Appendix D.4 (ISO 32000-1:2008, pages 996-997).
///
/// Symbol font is used extensively in mathematical and scientific documents
/// for Greek letters, mathematical operators, and special symbols.
///
/// # Arguments
///
/// * `code` - The character code (0-255)
///
/// # Returns
///
/// The corresponding Unicode character, or None if not in the encoding.
///
/// # References
///
/// - PDF 32000-1:2008, Appendix D.4 - Symbol Encoding
///
/// # Examples
///
/// ```ignore
/// # use pdf_oxide::fonts::font_dict::symbol_encoding_lookup;
/// assert_eq!(symbol_encoding_lookup(0x72), Some('ρ')); // rho
/// assert_eq!(symbol_encoding_lookup(0x61), Some('α')); // alpha
/// assert_eq!(symbol_encoding_lookup(0xF2), Some('∫')); // integral
/// ```ignore
fn symbol_encoding_lookup(code: u8) -> Option<char> {
    match code {
        // Greek lowercase letters
        0x61 => Some('α'), // alpha
        0x62 => Some('β'), // beta
        0x63 => Some('χ'), // chi
        0x64 => Some('δ'), // delta
        0x65 => Some('ε'), // epsilon
        0x66 => Some('φ'), // phi
        0x67 => Some('γ'), // gamma
        0x68 => Some('η'), // eta
        0x69 => Some('ι'), // iota
        0x6A => Some('ϕ'), // phi1 (variant)
        0x6B => Some('κ'), // kappa
        0x6C => Some('λ'), // lambda
        0x6D => Some('μ'), // mu
        0x6E => Some('ν'), // nu
        0x6F => Some('ο'), // omicron
        0x70 => Some('π'), // pi
        0x71 => Some('θ'), // theta
        0x72 => Some('ρ'), // rho ← THE IMPORTANT ONE for Pearson's ρ!
        0x73 => Some('σ'), // sigma
        0x74 => Some('τ'), // tau
        0x75 => Some('υ'), // upsilon
        0x76 => Some('ϖ'), // omega1 (variant pi)
        0x77 => Some('ω'), // omega
        0x78 => Some('ξ'), // xi
        0x79 => Some('ψ'), // psi
        0x7A => Some('ζ'), // zeta

        // Greek uppercase letters
        0x41 => Some('Α'), // Alpha
        0x42 => Some('Β'), // Beta
        0x43 => Some('Χ'), // Chi
        0x44 => Some('Δ'), // Delta
        0x45 => Some('Ε'), // Epsilon
        0x46 => Some('Φ'), // Phi
        0x47 => Some('Γ'), // Gamma
        0x48 => Some('Η'), // Eta
        0x49 => Some('Ι'), // Iota
        0x4B => Some('Κ'), // Kappa
        0x4C => Some('Λ'), // Lambda
        0x4D => Some('Μ'), // Mu
        0x4E => Some('Ν'), // Nu
        0x4F => Some('Ο'), // Omicron
        0x50 => Some('Π'), // Pi
        0x51 => Some('Θ'), // Theta
        0x52 => Some('Ρ'), // Rho
        0x53 => Some('Σ'), // Sigma
        0x54 => Some('Τ'), // Tau
        0x55 => Some('Υ'), // Upsilon
        0x57 => Some('Ω'), // Omega
        0x58 => Some('Ξ'), // Xi
        0x59 => Some('Ψ'), // Psi
        0x5A => Some('Ζ'), // Zeta

        // Mathematical operators
        0xB1 => Some('±'), // plusminus
        0xB4 => Some('÷'), // divide
        0xB5 => Some('∞'), // infinity
        0xB6 => Some('∂'), // partialdiff
        0xB7 => Some('•'), // bullet
        0xB9 => Some('≠'), // notequal
        0xBA => Some('≡'), // equivalence
        0xBB => Some('≈'), // approxequal
        0xBC => Some('…'), // ellipsis
        0xBE => Some('⊥'), // perpendicular
        0xBF => Some('⊙'), // circleplus

        0xD0 => Some('°'), // degree
        0xD1 => Some('∇'), // gradient (nabla)
        0xD2 => Some('¬'), // logicalnot
        0xD3 => Some('∧'), // logicaland
        0xD4 => Some('∨'), // logicalor
        0xD5 => Some('∏'), // product ← Product symbol!
        0xD6 => Some('√'), // radical ← Square root!
        0xD7 => Some('⋅'), // dotmath
        0xD8 => Some('⊕'), // circleplus
        0xD9 => Some('⊗'), // circletimes

        0xDA => Some('∈'), // element
        0xDB => Some('∉'), // notelement
        0xDC => Some('∠'), // angle
        0xDD => Some('∇'), // gradient
        0xDE => Some('®'), // registered
        0xDF => Some('©'), // copyright
        0xE0 => Some('™'), // trademark

        0xE1 => Some('∑'), // summation ← Summation symbol!
        0xE2 => Some('⊂'), // propersubset
        0xE3 => Some('⊃'), // propersuperset
        0xE4 => Some('⊆'), // reflexsubset
        0xE5 => Some('⊇'), // reflexsuperset
        0xE6 => Some('∪'), // union
        0xE7 => Some('∩'), // intersection
        0xE8 => Some('∀'), // universal
        0xE9 => Some('∃'), // existential
        0xEA => Some('¬'), // logicalnot

        0xF1 => Some('〈'), // angleleft
        0xF2 => Some('∫'),  // integral ← Integral symbol!
        0xF3 => Some('⌠'),  // integraltp
        0xF4 => Some('⌡'),  // integralbt
        0xF5 => Some('⊓'),  // square intersection
        0xF6 => Some('⊔'),  // square union
        0xF7 => Some('〉'), // angleright

        // Basic punctuation and symbols (overlap with ASCII)
        0x20 => Some(' '), // space
        0x21 => Some('!'), // exclam
        0x22 => Some('∀'), // universal (sometimes mapped here)
        0x23 => Some('#'), // numbersign
        0x24 => Some('∃'), // existential (sometimes mapped here)
        0x25 => Some('%'), // percent
        0x26 => Some('&'), // ampersand
        0x27 => Some('∋'), // suchthat
        0x28 => Some('('), // parenleft
        0x29 => Some(')'), // parenright
        0x2A => Some('∗'), // asteriskmath
        0x2B => Some('+'), // plus
        0x2C => Some(','), // comma
        0x2D => Some('−'), // minus
        0x2E => Some('.'), // period
        0x2F => Some('/'), // slash

        // Digits 0-9 (0x30-0x39) map to themselves
        0x30..=0x39 => Some(code as char),

        0x3A => Some(':'), // colon
        0x3B => Some(';'), // semicolon
        0x3C => Some('<'), // less
        0x3D => Some('='), // equal
        0x3E => Some('>'), // greater
        0x3F => Some('?'), // question

        0x40 => Some('≅'), // congruent

        // Brackets and arrows
        0x5B => Some('['), // bracketleft
        0x5C => Some('∴'), // therefore
        0x5D => Some(']'), // bracketright
        0x5E => Some('⊥'), // perpendicular
        0x5F => Some('_'), // underscore

        0x7B => Some('{'), // braceleft
        0x7C => Some('|'), // bar
        0x7D => Some('}'), // braceright
        0x7E => Some('∼'), // similar

        _ => None,
    }
}

/// Look up a character in the Adobe ZapfDingbats font encoding.
///
/// This function implements a subset of the ZapfDingbats font encoding table
/// as defined in PDF Specification Appendix D.5 (ISO 32000-1:2008, page 998).
///
/// ZapfDingbats font is used for ornamental symbols, arrows, and decorative characters.
///
/// # Arguments
///
/// * `code` - The character code (0-255)
///
/// # Returns
///
/// The corresponding Unicode character, or None if not in the encoding.
///
/// # References
///
/// - PDF 32000-1:2008, Appendix D.5 - ZapfDingbats Encoding
fn zapf_dingbats_encoding_lookup(code: u8) -> Option<char> {
    match code {
        0x20 => Some(' '), // space
        0x21 => Some('✁'), // scissors
        0x22 => Some('✂'), // scissors (filled)
        0x23 => Some('✃'), // scissors (outline)
        0x24 => Some('✄'), // scissors (small)
        0x25 => Some('☎'), // telephone
        0x26 => Some('✆'), // telephone (filled)
        0x27 => Some('✇'), // tape drive
        0x28 => Some('✈'), // airplane
        0x29 => Some('✉'), // envelope
        0x2A => Some('☛'), // hand pointing right
        0x2B => Some('☞'), // hand pointing right (filled)
        0x2C => Some('✌'), // victory hand
        0x2D => Some('✍'), // writing hand
        0x2E => Some('✎'), // pencil
        0x2F => Some('✏'), // pencil (filled)

        0x30 => Some('✐'), // pen nib
        0x31 => Some('✑'), // pen nib (filled)
        0x32 => Some('✒'), // pen nib (outline)
        0x33 => Some('✓'), // checkmark
        0x34 => Some('✔'), // checkmark (bold)
        0x35 => Some('✕'), // multiplication X
        0x36 => Some('✖'), // multiplication X (heavy)
        0x37 => Some('✗'), // ballot X
        0x38 => Some('✘'), // ballot X (heavy)
        0x39 => Some('✙'), // outlined Greek cross
        0x3A => Some('✚'), // heavy Greek cross
        0x3B => Some('✛'), // open center cross
        0x3C => Some('✜'), // heavy open center cross
        0x3D => Some('✝'), // Latin cross
        0x3E => Some('✞'), // Latin cross (shadowed)
        0x3F => Some('✟'), // Latin cross (outline)

        // Common symbols
        0x40 => Some('✠'), // Maltese cross
        0x41 => Some('✡'), // Star of David
        0x42 => Some('✢'), // four teardrop-spoked asterisk
        0x43 => Some('✣'), // four balloon-spoked asterisk
        0x44 => Some('✤'), // heavy four balloon-spoked asterisk
        0x45 => Some('✥'), // four club-spoked asterisk
        0x46 => Some('✦'), // black four pointed star
        0x47 => Some('✧'), // white four pointed star
        0x48 => Some('★'), // black star
        0x49 => Some('✩'), // outlined black star
        0x4A => Some('✪'), // circled white star
        0x4B => Some('✫'), // circled black star
        0x4C => Some('✬'), // shadowed white star
        0x4D => Some('✭'), // heavy asterisk
        0x4E => Some('✮'), // eight spoke asterisk
        0x4F => Some('✯'), // eight pointed black star

        // More ornaments
        0x50 => Some('✰'), // eight pointed pinwheel star
        0x51 => Some('✱'), // heavy eight pointed pinwheel star
        0x52 => Some('✲'), // eight pointed star
        0x53 => Some('✳'), // eight pointed star (outlined)
        0x54 => Some('✴'), // eight pointed star (heavy)
        0x55 => Some('✵'), // six pointed black star
        0x56 => Some('✶'), // six pointed star
        0x57 => Some('✷'), // eight pointed star (black)
        0x58 => Some('✸'), // heavy eight pointed star
        0x59 => Some('✹'), // twelve pointed black star
        0x5A => Some('✺'), // sixteen pointed star
        0x5B => Some('✻'), // teardrop-spoked asterisk
        0x5C => Some('✼'), // open center teardrop-spoked asterisk
        0x5D => Some('✽'), // heavy teardrop-spoked asterisk
        0x5E => Some('✾'), // six petalled black and white florette
        0x5F => Some('✿'), // black florette

        // Geometric shapes
        0x60 => Some('❀'), // white florette
        0x61 => Some('❁'), // eight petalled outlined black florette
        0x62 => Some('❂'), // circled open center eight pointed star
        0x63 => Some('❃'), // heavy teardrop-spoked pinwheel asterisk
        0x64 => Some('❄'), // snowflake
        0x65 => Some('❅'), // tight trifoliate snowflake
        0x66 => Some('❆'), // heavy chevron snowflake
        0x67 => Some('❇'), // sparkle
        0x68 => Some('❈'), // heavy sparkle
        0x69 => Some('❉'), // balloon-spoked asterisk
        0x6A => Some('❊'), // eight teardrop-spoked propeller asterisk
        0x6B => Some('❋'), // heavy eight teardrop-spoked propeller asterisk

        // Arrows
        0x6C => Some('●'), // black circle
        0x6D => Some('○'), // white circle
        0x6E => Some('❍'), // shadowed white circle
        0x6F => Some('■'), // black square
        0x70 => Some('□'), // white square
        0x71 => Some('▢'), // white square with rounded corners
        0x72 => Some('▣'), // white square containing black small square
        0x73 => Some('▤'), // square with horizontal fill
        0x74 => Some('▥'), // square with vertical fill
        0x75 => Some('▦'), // square with orthogonal crosshatch fill
        0x76 => Some('▧'), // square with upper left to lower right fill
        0x77 => Some('▨'), // square with upper right to lower left fill
        0x78 => Some('▩'), // square with diagonal crosshatch fill
        0x79 => Some('▪'), // black small square
        0x7A => Some('▫'), // white small square

        _ => None,
    }
}

/// Look up a character in PDFDocEncoding.
///
/// PDFDocEncoding is a superset of ISO Latin-1 used as the default encoding
/// for PDF text strings and metadata (bookmarks, annotations, document info).
///
/// Codes 0-127 are identical to ASCII.
/// Codes 128-159 have special mappings (different from ISO Latin-1).
/// Codes 160-255 are identical to ISO Latin-1.
///
/// # PDF Spec Reference
///
/// ISO 32000-1:2008, Appendix D.2, Table D.2, page 994
///
/// # Arguments
///
/// * `code` - The byte code to look up (0-255)
///
/// # Returns
///
/// The Unicode character for this code, or None for undefined codes
pub fn pdfdoc_encoding_lookup(code: u8) -> Option<char> {
    match code {
        // ASCII range (0-127)
        0x00..=0x7F => Some(code as char),

        // PDFDocEncoding special range (128-159)
        0x80 => Some('•'),        // bullet
        0x81 => Some('†'),        // dagger
        0x82 => Some('‡'),        // daggerdbl
        0x83 => Some('…'),        // ellipsis
        0x84 => Some('—'),        // emdash
        0x85 => Some('–'),        // endash
        0x86 => Some('ƒ'),        // florin
        0x87 => Some('⁄'),        // fraction
        0x88 => Some('‹'),        // guilsinglleft
        0x89 => Some('›'),        // guilsinglright
        0x8A => Some('−'),        // minus (different from hyphen!)
        0x8B => Some('‰'),        // perthousand
        0x8C => Some('„'),        // quotedblbase
        0x8D => Some('"'),        // quotedblleft
        0x8E => Some('"'),        // quotedblright
        0x8F => Some('\u{2018}'), // quoteleft (left single quotation mark)
        0x90 => Some('\u{2019}'), // quoteright (right single quotation mark)
        0x91 => Some('‚'),        // quotesinglbase
        0x92 => Some('™'),        // trademark
        0x93 => Some('ﬁ'),        // fi ligature
        0x94 => Some('ﬂ'),        // fl ligature
        0x95 => Some('Ł'),        // Lslash
        0x96 => Some('Œ'),        // OE
        0x97 => Some('Š'),        // Scaron
        0x98 => Some('Ÿ'),        // Ydieresis
        0x99 => Some('Ž'),        // Zcaron
        0x9A => Some('ı'),        // dotlessi
        0x9B => Some('ł'),        // lslash
        0x9C => Some('œ'),        // oe
        0x9D => Some('š'),        // scaron
        0x9E => Some('ž'),        // zcaron
        0x9F => None,             // undefined

        // ISO Latin-1 range (160-255) - direct mapping
        0xA0..=0xFF => Some(code as char),
    }
}

/// Look up a character in a standard PDF encoding.
///
/// This function provides support for standard PDF encodings including
/// PDFDocEncoding, WinAnsiEncoding, StandardEncoding, and MacRomanEncoding.
///
/// # Arguments
///
/// * `encoding` - The encoding name (e.g., "WinAnsiEncoding", "PDFDocEncoding")
/// * `code` - The character code (0-255)
///
/// # Returns
///
/// The Unicode string for this character, or None if not in the encoding.
fn standard_encoding_lookup(encoding: &str, code: u8) -> Option<String> {
    match encoding {
        "PDFDocEncoding" => {
            // PDFDocEncoding: superset of ISO Latin-1 with special 128-159 range
            pdfdoc_encoding_lookup(code).map(|c| c.to_string())
        },
        "WinAnsiEncoding" => {
            // ASCII printable range (32-126)
            if (32..=126).contains(&code) {
                return Some((code as char).to_string());
            }

            // WinAnsiEncoding extended range (128-255)
            // Based on Windows-1252 encoding
            let unicode = match code {
                0x80 => '\u{20AC}', // Euro sign
                0x82 => '\u{201A}', // Single low-9 quotation mark
                0x83 => '\u{0192}', // Latin small letter f with hook
                0x84 => '\u{201E}', // Double low-9 quotation mark
                0x85 => '\u{2026}', // Horizontal ellipsis
                0x86 => '\u{2020}', // Dagger
                0x87 => '\u{2021}', // Double dagger
                0x88 => '\u{02C6}', // Modifier letter circumflex accent
                0x89 => '\u{2030}', // Per mille sign
                0x8A => '\u{0160}', // Latin capital letter S with caron
                0x8B => '\u{2039}', // Single left-pointing angle quotation mark
                0x8C => '\u{0152}', // Latin capital ligature OE
                0x8E => '\u{017D}', // Latin capital letter Z with caron
                0x91 => '\u{2018}', // Left single quotation mark
                0x92 => '\u{2019}', // Right single quotation mark
                0x93 => '\u{201C}', // Left double quotation mark
                0x94 => '\u{201D}', // Right double quotation mark
                0x95 => '\u{2022}', // Bullet
                0x96 => '\u{2013}', // En dash
                0x97 => '\u{2014}', // Em dash
                0x98 => '\u{02DC}', // Small tilde
                0x99 => '\u{2122}', // Trade mark sign
                0x9A => '\u{0161}', // Latin small letter s with caron
                0x9B => '\u{203A}', // Single right-pointing angle quotation mark
                0x9C => '\u{0153}', // Latin small ligature oe
                0x9E => '\u{017E}', // Latin small letter z with caron
                0x9F => '\u{0178}', // Latin capital letter Y with diaeresis
                // 0xA0-0xFF: Direct mapping to Unicode (ISO-8859-1)
                _ if code >= 0xA0 => char::from_u32(code as u32)?,
                _ => return None,
            };
            Some(unicode.to_string())
        },
        "StandardEncoding" => {
            // StandardEncoding is mostly like WinAnsi for the ASCII range
            if (32..=126).contains(&code) {
                Some((code as char).to_string())
            } else if code >= 0xA0 {
                // Extended range uses ISO Latin-1 for most characters
                char::from_u32(code as u32).map(|c| c.to_string())
            } else {
                None
            }
        },
        "MacRomanEncoding" => {
            // ASCII range is the same
            if (32..=126).contains(&code) {
                Some((code as char).to_string())
            } else {
                // MacRoman extended characters
                // Partial implementation focusing on common punctuation
                // PDF Spec: ISO 32000-1:2008, Annex D.2
                let unicode = match code {
                    0x80 => '\u{00C4}', // Adieresis
                    0x81 => '\u{00C5}', // Aring
                    0x82 => '\u{00C7}', // Ccedilla
                    0x83 => '\u{00C9}', // Eacute
                    0x84 => '\u{00D1}', // Ntilde
                    0x85 => '\u{00D6}', // Odieresis
                    0x86 => '\u{00DC}', // Udieresis
                    0x87 => '\u{00E1}', // aacute
                    0x88 => '\u{00E0}', // agrave
                    0x89 => '\u{00E2}', // acircumflex
                    0x8A => '\u{00E4}', // adieresis
                    0x8B => '\u{00E3}', // atilde
                    0x8C => '\u{00E5}', // aring
                    0x8D => '\u{00E7}', // ccedilla
                    0x8E => '\u{00E9}', // eacute
                    0x8F => '\u{00E8}', // egrave
                    0x90 => '\u{00EA}', // ecircumflex
                    0x91 => '\u{00EB}', // edieresis
                    0x92 => '\u{00ED}', // iacute
                    0x93 => '\u{00EC}', // igrave
                    0x94 => '\u{00EE}', // icircumflex
                    0x95 => '\u{00EF}', // idieresis
                    0x96 => '\u{00F1}', // ntilde
                    0x97 => '\u{00F3}', // oacute
                    0x98 => '\u{00F2}', // ograve
                    0x99 => '\u{00F4}', // ocircumflex
                    0x9A => '\u{00F6}', // odieresis
                    0x9B => '\u{00F5}', // otilde
                    0x9C => '\u{00FA}', // uacute
                    0x9D => '\u{00F9}', // ugrave
                    0x9E => '\u{00FB}', // ucircumflex
                    0x9F => '\u{00FC}', // udieresis
                    0xD0 => '\u{2013}', // EN DASH (this is the key fix!)
                    0xD1 => '\u{2014}', // EM DASH
                    0xD2 => '\u{201C}', // LEFT DOUBLE QUOTATION MARK
                    0xD3 => '\u{201D}', // RIGHT DOUBLE QUOTATION MARK
                    0xD4 => '\u{2018}', // LEFT SINGLE QUOTATION MARK
                    0xD5 => '\u{2019}', // RIGHT SINGLE QUOTATION MARK
                    0xE0 => '\u{2020}', // dagger
                    0xE1 => '\u{00B0}', // degree
                    _ if code >= 0xA0 => char::from_u32(code as u32)?,
                    _ => return None,
                };
                Some(unicode.to_string())
            }
        },
        _ => {
            // Unknown encoding, try identity mapping for ASCII
            if code.is_ascii() && code >= 32 {
                Some((code as char).to_string())
            } else {
                None
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_encoding_ascii() {
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", b'A'), Some("A".to_string()));
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", b'Z'), Some("Z".to_string()));
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", b'0'), Some("0".to_string()));
    }

    #[test]
    fn test_standard_encoding_space() {
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", b' '), Some(" ".to_string()));
    }

    #[test]
    fn test_font_info_is_bold() {
        let font = FontInfo {
            base_font: "Times-Bold".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: Some(700),
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };
        assert!(font.is_bold());

        let font2 = FontInfo {
            base_font: "Helvetica".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: Some(400),
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };
        assert!(!font2.is_bold());
    }

    #[test]
    fn test_font_info_is_italic() {
        let font = FontInfo {
            base_font: "Times-Italic".to_string(),
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
        };
        assert!(font.is_italic());

        let font2 = FontInfo {
            base_font: "Courier-Oblique".to_string(),
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
        };
        assert!(font2.is_italic());
    }

    #[test]
    fn test_char_to_unicode_with_tounicode() {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "X".to_string()); // Custom mapping

        let font = FontInfo {
            base_font: "CustomFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: Some(cmap),
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        // Should use ToUnicode mapping (priority)
        assert_eq!(font.char_to_unicode(0x41), Some("X".to_string()));
        // Should fall back to standard encoding
        assert_eq!(font.char_to_unicode(0x42), Some("B".to_string()));
    }

    #[test]
    fn test_char_to_unicode_standard_encoding() {
        let font = FontInfo {
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
        };

        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font.char_to_unicode(0x20), Some(" ".to_string()));
    }

    #[test]
    fn test_char_to_unicode_identity() {
        let font = FontInfo {
            base_font: "CIDFont".to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font.char_to_unicode(0x263A), Some("☺".to_string()));
    }

    #[test]
    fn test_encoding_clone() {
        let enc = Encoding::Standard("WinAnsiEncoding".to_string());
        let enc2 = enc.clone();
        match enc2 {
            Encoding::Standard(name) => assert_eq!(name, "WinAnsiEncoding"),
            _ => panic!("Wrong encoding type"),
        }
    }

    #[test]
    fn test_font_info_clone() {
        let font = FontInfo {
            base_font: "Test".to_string(),
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
        };

        let font2 = font.clone();
        assert_eq!(font2.base_font, "Test");
    }

    #[test]
    fn test_glyph_name_to_unicode_basic() {
        assert_eq!(glyph_name_to_unicode("A"), Some('A'));
        assert_eq!(glyph_name_to_unicode("a"), Some('a'));
        assert_eq!(glyph_name_to_unicode("zero"), Some('0'));
        assert_eq!(glyph_name_to_unicode("nine"), Some('9'));
    }

    #[test]
    fn test_glyph_name_to_unicode_punctuation() {
        assert_eq!(glyph_name_to_unicode("space"), Some(' '));
        assert_eq!(glyph_name_to_unicode("quotesingle"), Some('\''));
        assert_eq!(glyph_name_to_unicode("grave"), Some('`'));
        assert_eq!(glyph_name_to_unicode("hyphen"), Some('-'));
        // Official AGL: "minus" maps to U+2212 (MINUS SIGN), not U+002D (HYPHEN-MINUS)
        assert_eq!(glyph_name_to_unicode("minus"), Some('−'));
    }

    #[test]
    fn test_glyph_name_to_unicode_special() {
        assert_eq!(glyph_name_to_unicode("bullet"), Some('•'));
        assert_eq!(glyph_name_to_unicode("dagger"), Some('†'));
        assert_eq!(glyph_name_to_unicode("daggerdbl"), Some('‡'));
        assert_eq!(glyph_name_to_unicode("ellipsis"), Some('…'));
        assert_eq!(glyph_name_to_unicode("emdash"), Some('—'));
        assert_eq!(glyph_name_to_unicode("endash"), Some('–'));
    }

    #[test]
    fn test_glyph_name_to_unicode_quotes() {
        assert_eq!(glyph_name_to_unicode("quotesinglbase"), Some('‚'));
        assert_eq!(glyph_name_to_unicode("quotedblbase"), Some('„'));
        // Official AGL uses proper curly quotes, not straight quotes
        assert_eq!(glyph_name_to_unicode("quotedblleft"), Some('\u{201C}')); // LEFT DOUBLE QUOTATION MARK
        assert_eq!(glyph_name_to_unicode("quotedblright"), Some('\u{201D}')); // RIGHT DOUBLE QUOTATION MARK
        assert_eq!(glyph_name_to_unicode("quoteleft"), Some('\u{2018}'));
        assert_eq!(glyph_name_to_unicode("quoteright"), Some('\u{2019}'));
    }

    #[test]
    fn test_glyph_name_to_unicode_accented() {
        assert_eq!(glyph_name_to_unicode("Aacute"), Some('Á'));
        assert_eq!(glyph_name_to_unicode("aacute"), Some('á'));
        assert_eq!(glyph_name_to_unicode("Ntilde"), Some('Ñ'));
        assert_eq!(glyph_name_to_unicode("ntilde"), Some('ñ'));
    }

    #[test]
    fn test_glyph_name_to_unicode_currency() {
        assert_eq!(glyph_name_to_unicode("Euro"), Some('€'));
        assert_eq!(glyph_name_to_unicode("sterling"), Some('£'));
        assert_eq!(glyph_name_to_unicode("yen"), Some('¥'));
        assert_eq!(glyph_name_to_unicode("cent"), Some('¢'));
    }

    #[test]
    fn test_glyph_name_to_unicode_ligatures() {
        assert_eq!(glyph_name_to_unicode("fi"), Some('ﬁ'));
        assert_eq!(glyph_name_to_unicode("fl"), Some('ﬂ'));
        assert_eq!(glyph_name_to_unicode("ffi"), Some('ﬃ'));
    }

    #[test]
    fn test_glyph_name_to_unicode_uni_xxxx() {
        // Test uni format (4 hex digits)
        assert_eq!(glyph_name_to_unicode("uni0041"), Some('A'));
        assert_eq!(glyph_name_to_unicode("uni2022"), Some('•'));
    }

    #[test]
    fn test_glyph_name_to_unicode_u_xxxx() {
        // Test u format (variable hex digits)
        assert_eq!(glyph_name_to_unicode("u0041"), Some('A'));
        assert_eq!(glyph_name_to_unicode("u2022"), Some('•'));
    }

    #[test]
    fn test_glyph_name_to_unicode_unknown() {
        assert_eq!(glyph_name_to_unicode("unknownglyph"), None);
        assert_eq!(glyph_name_to_unicode(""), None);
    }

    #[test]
    fn test_char_to_unicode_custom_encoding() {
        // Create a custom encoding map
        let mut custom_map = HashMap::new();
        custom_map.insert(0x41, 'X'); // A -> X
        custom_map.insert(0x42, '•'); // B -> bullet

        let font = FontInfo {
            base_font: "CustomFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Custom(custom_map),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        // Should use custom encoding
        assert_eq!(font.char_to_unicode(0x41), Some("X".to_string()));
        assert_eq!(font.char_to_unicode(0x42), Some("•".to_string()));
        // Unmapped character should return None
        assert_eq!(font.char_to_unicode(0x43), None);
    }

    /// Integration Test 1: ForceBold flag detection (PDF Spec Table 123, bit 19)
    #[test]
    fn test_get_font_weight_force_bold_flag() {
        // Test ForceBold flag set (bit 19 = 0x80000 = 524288)
        let font_with_force_bold = FontInfo {
            base_font: "Helvetica".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,    // No explicit weight
            flags: Some(0x80000), // ForceBold flag set
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_with_force_bold.get_font_weight(), FontWeight::Bold);
        assert!(font_with_force_bold.is_bold());

        // Test without ForceBold flag
        let font_without_force_bold = FontInfo {
            base_font: "Helvetica".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: Some(0x40000), // Different flag, NOT ForceBold
            stem_v: None,
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_without_force_bold.get_font_weight(), FontWeight::Normal);
        assert!(!font_without_force_bold.is_bold());
    }

    /// Integration Test 2: StemV analysis for weight inference
    #[test]
    fn test_get_font_weight_stem_v_analysis() {
        // Test StemV > 110 → Bold
        let font_heavy_stem = FontInfo {
            base_font: "UnknownFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: Some(120.0), // Heavy stem
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_heavy_stem.get_font_weight(), FontWeight::Bold);
        assert!(font_heavy_stem.is_bold());

        // Test StemV 80-110 → Medium
        let font_medium_stem = FontInfo {
            base_font: "UnknownFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: Some(95.0), // Medium stem
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_medium_stem.get_font_weight(), FontWeight::Medium);
        assert!(!font_medium_stem.is_bold());

        // Test StemV < 80 → Normal
        let font_light_stem = FontInfo {
            base_font: "UnknownFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: Some(70.0), // Light stem
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_light_stem.get_font_weight(), FontWeight::Normal);
        assert!(!font_light_stem.is_bold());
    }

    /// Integration Test 3: Priority cascade (FontWeight > ForceBold > Name > StemV)
    #[test]
    fn test_get_font_weight_priority_cascade() {
        // Priority 1: Explicit FontWeight field overrides everything
        let font_explicit = FontInfo {
            base_font: "Helvetica-Bold".to_string(), // Name says Bold
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: Some(300), // But explicit weight is Light
            flags: Some(0x80000),   // ForceBold flag set
            stem_v: Some(120.0),    // Heavy stem
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_explicit.get_font_weight(), FontWeight::Light);
        assert!(!font_explicit.is_bold());

        // Priority 2: ForceBold overrides name and StemV
        let font_force_bold = FontInfo {
            base_font: "Helvetica".to_string(), // Name says Normal
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,    // No explicit weight
            flags: Some(0x80000), // ForceBold flag set
            stem_v: Some(70.0),   // Light stem
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_force_bold.get_font_weight(), FontWeight::Bold);
        assert!(font_force_bold.is_bold());

        // Priority 3: Name heuristics override StemV
        let font_name = FontInfo {
            base_font: "Helvetica-Bold".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: Some(70.0), // Light stem, but name says Bold
            embedded_font_data: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
        };

        assert_eq!(font_name.get_font_weight(), FontWeight::Bold);
        assert!(font_name.is_bold());
    }

    /// Integration Test 4: Name heuristics for all weight categories
    #[test]
    fn test_get_font_weight_name_heuristics() {
        // Test Black/Heavy
        let font_black = FontInfo {
            base_font: "Helvetica-Black".to_string(),
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
        };
        assert_eq!(font_black.get_font_weight(), FontWeight::Black);
        assert!(font_black.is_bold());

        // Test ExtraBold
        let font_extrabold = FontInfo {
            base_font: "Arial-ExtraBold".to_string(),
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
        };
        assert_eq!(font_extrabold.get_font_weight(), FontWeight::ExtraBold);
        assert!(font_extrabold.is_bold());

        // Test Bold (but not SemiBold)
        let font_bold = FontInfo {
            base_font: "TimesNewRoman-Bold".to_string(),
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
        };
        assert_eq!(font_bold.get_font_weight(), FontWeight::Bold);
        assert!(font_bold.is_bold());

        // Test SemiBold
        let font_semibold = FontInfo {
            base_font: "Arial-SemiBold".to_string(),
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
        };
        assert_eq!(font_semibold.get_font_weight(), FontWeight::SemiBold);
        assert!(font_semibold.is_bold());

        // Test Medium
        let font_medium = FontInfo {
            base_font: "Roboto-Medium".to_string(),
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
        };
        assert_eq!(font_medium.get_font_weight(), FontWeight::Medium);
        assert!(!font_medium.is_bold());

        // Test Light (but not ExtraLight)
        let font_light = FontInfo {
            base_font: "Helvetica-Light".to_string(),
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
        };
        assert_eq!(font_light.get_font_weight(), FontWeight::Light);
        assert!(!font_light.is_bold());

        // Test ExtraLight
        let font_extralight = FontInfo {
            base_font: "Roboto-ExtraLight".to_string(),
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
        };
        assert_eq!(font_extralight.get_font_weight(), FontWeight::ExtraLight);
        assert!(!font_extralight.is_bold());

        // Test Thin
        let font_thin = FontInfo {
            base_font: "HelveticaNeue-Thin".to_string(),
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
        };
        assert_eq!(font_thin.get_font_weight(), FontWeight::Thin);
        assert!(!font_thin.is_bold());

        // Test Normal/Regular (no weight keywords)
        let font_normal = FontInfo {
            base_font: "Helvetica".to_string(),
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
        };
        assert_eq!(font_normal.get_font_weight(), FontWeight::Normal);
        assert!(!font_normal.is_bold());
    }
}
