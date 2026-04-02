//! Font dictionary parsing.
//!
//! This module handles parsing of PDF font dictionaries and encoding information.
//! Fonts in PDF can have various encodings, and the ToUnicode CMap provides the
//! most accurate character-to-Unicode mapping.

use super::adobe_glyph_list::ADOBE_GLYPH_LIST;
use crate::document::PdfDocument;
use crate::error::{Error, Result};
use crate::fonts::cmap::LazyCMap;
use crate::fonts::TrueTypeCMap;
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
    /// Lazily parsed on first character lookup for improved performance
    pub to_unicode: Option<LazyCMap>,
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
    /// Lazily-extracted TrueType cmap table (GID to Unicode mappings).
    /// Used as fallback when ToUnicode CMap is missing.
    /// Initialized on first access via `truetype_cmap()` accessor to avoid
    /// the 10-25ms per-font extraction cost when ToUnicode resolves all chars.
    pub truetype_cmap: std::sync::OnceLock<Option<TrueTypeCMap>>,
    /// Whether this font has an embedded TrueType font (FontFile2).
    /// Controls whether lazy truetype_cmap extraction is attempted.
    pub is_truetype_font: bool,
    /// CID to GID mapping (Type0 fonts only, Phase 3)
    /// Converts Character IDs in the PDF to Glyph IDs in the embedded font
    /// Used to look up Unicode values via the TrueType cmap table
    /// Phase 3: Enables CFF/OpenType support via CIDToGIDMap parsing
    pub cid_to_gid_map: Option<CIDToGIDMap>,
    /// CIDFont character collection info (Type0 fonts only)
    /// Identifies the character set (e.g., Adobe-Japan1, Adobe-GB1)
    pub cid_system_info: Option<CIDSystemInfo>,
    /// CIDFont subtype ("CIDFontType0" for CFF, "CIDFontType2" for TrueType)
    pub cid_font_type: Option<String>,
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
    /// CID to width mapping for Type0 (CIDFont) fonts
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.3
    /// Widths in 1000ths of em. Uses HashMap for sparse CID distributions.
    pub cid_widths: Option<HashMap<u16, f32>>,
    /// Default width for CIDs not in cid_widths (Type0 fonts only)
    /// Per PDF Spec: default is 1000 if /DW not specified
    pub cid_default_width: f32,
    /// Multi-character encoding map for compound glyph names (e.g. f_f → "ff")
    /// Stores mappings from character code to multi-char strings
    pub multi_char_map: HashMap<u8, String>,
    /// CFF byte_code → glyph_id mapping for embedded CFF subset fonts.
    /// Allows direct glyph rendering without Unicode cmap.
    pub cff_gid_map: Option<HashMap<u8, u16>>,
    /// Pre-computed byte→char lookup for simple (non-Type0) fonts.
    /// Index by byte value (0-255). '\0' means "use full char_to_unicode fallback".
    /// Built lazily on first text decode. Avoids per-byte HashMap lookups.
    pub byte_to_char_table: std::sync::OnceLock<[char; 256]>,
    /// Pre-computed byte→width lookup for simple (non-Type0) fonts.
    /// Index by byte value (0-255). Built lazily on first advance_position call.
    /// Eliminates per-byte bounds check and subtraction in get_glyph_width.
    pub byte_to_width_table: std::sync::OnceLock<[f32; 256]>,
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

/// CID to GID mapping for Type 2 CIDFonts (TrueType-based)
/// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.2
///
/// This mapping converts Character IDs (CIDs) in the PDF document to Glyph IDs (GIDs)
/// in the embedded TrueType font, which can then be mapped to Unicode via the cmap table.
#[derive(Debug, Clone)]
pub enum CIDToGIDMap {
    /// Identity mapping: CID == GID (default, most common)
    /// Used when each character ID directly corresponds to a glyph ID
    Identity,

    /// Explicit mapping: CID → GID via uint16 stream
    /// Stream format: GID at bytes [2*CID, 2*CID+1], big-endian
    /// Used for non-standard glyph ID assignments
    Explicit(Vec<u16>),
}

impl CIDToGIDMap {
    /// Convert a Character ID (CID) to a Glyph ID (GID) using this mapping.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.2:
    /// - Identity mapping: CID == GID (most common, default)
    /// - Explicit mapping: Use uint16 array lookup
    ///
    /// # Arguments
    ///
    /// * `cid` - The Character ID from the PDF document
    ///
    /// # Returns
    ///
    /// The corresponding Glyph ID in the embedded font
    pub fn get_gid(&self, cid: u16) -> u16 {
        match self {
            CIDToGIDMap::Identity => cid,
            CIDToGIDMap::Explicit(gid_array) => {
                if (cid as usize) < gid_array.len() {
                    gid_array[cid as usize]
                } else {
                    // Out of range - fall back to identity mapping
                    cid
                }
            },
        }
    }
}

/// CIDFont character collection identifier
/// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.2
///
/// Identifies which character encoding the CIDFont uses, such as:
/// - Adobe-Japan1: Japanese text
/// - Adobe-GB1: Simplified Chinese
/// - Adobe-CNS1: Traditional Chinese
/// - Adobe-Korea1: Korean
#[derive(Debug, Clone)]
pub struct CIDSystemInfo {
    /// Registry name (typically "Adobe")
    pub registry: String,

    /// Ordering string (e.g., "Japan1", "GB1", "CNS1", "Korea1")
    pub ordering: String,

    /// Supplement number (version of the character collection)
    pub supplement: i32,
}

impl FontInfo {
    /// Get the TrueType cmap, lazily extracting it on first access.
    /// Returns `None` if the font is not TrueType or has no embedded data.
    pub fn truetype_cmap(&self) -> Option<&TrueTypeCMap> {
        self.truetype_cmap
            .get_or_init(|| {
                if !self.is_truetype_font {
                    return None;
                }
                let font_data = self.embedded_font_data.as_ref()?;
                if font_data.is_empty() {
                    return None;
                }
                match TrueTypeCMap::from_font_data(font_data) {
                    Ok(cmap) if !cmap.is_empty() => {
                        log::info!(
                            "Lazy-extracted TrueType cmap for font '{}': {} mappings",
                            self.base_font,
                            cmap.len()
                        );
                        Some(cmap)
                    },
                    Ok(_) => None,
                    Err(e) => {
                        log::warn!(
                            "Font '{}': TrueType cmap extraction failed: {}",
                            self.base_font,
                            e
                        );
                        None
                    },
                }
            })
            .as_ref()
    }

    /// Set the TrueType cmap directly (used by share_truetype_cmaps and tests).
    pub fn set_truetype_cmap(&mut self, cmap: Option<TrueTypeCMap>) {
        self.truetype_cmap = std::sync::OnceLock::new();
        if let Some(c) = cmap {
            let _ = self.truetype_cmap.set(Some(c));
        } else {
            let _ = self.truetype_cmap.set(None);
        }
    }

    /// Check if a TrueType cmap is available (either already extracted or extractable).
    pub fn has_truetype_cmap(&self) -> bool {
        self.truetype_cmap().is_some()
    }

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
    /// # fn example(doc: PdfDocument, font_ref: ObjectRef) -> Result<(), Box<dyn std::error::Error>> {
    /// let font_obj = doc.load_object(font_ref)?;
    /// let font_info = FontInfo::from_dict(&font_obj, &doc)?;
    /// println!("Font: {}", font_info.base_font);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_dict(dict: &Object, doc: &PdfDocument) -> Result<Self> {
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

        // Log Type 3 fonts - may require special glyph name mapping
        if subtype == "Type3" {
            log::warn!("Font '{}' is Type 3 - may require special glyph name mapping", base_font);
        }

        // Parse FontDescriptor FIRST to get font flags (needed for encoding decision)
        // PDF Spec: ISO 32000-1:2008, Section 9.6.2 - Font Descriptor
        let (font_weight, flags, stem_v, mut embedded_font_data, is_truetype_font) =
            if let Some(descriptor_ref) = font_dict
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
                        // IMPORTANT: Track whether font is TrueType or CFF - only TrueType fonts have cmaps!
                        let (embedded_font, is_truetype_font) =
                            if let Some(ff2_obj) = descriptor_dict.get("FontFile2") {
                                log::info!("Font '{}' has FontFile2 entry (TrueType)", base_font);
                                let font_data = ff2_obj
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
                                    });
                                (font_data, true) // TrueType - can have cmaps
                            } else if let Some(ff3_obj) = descriptor_dict.get("FontFile3") {
                                log::info!(
                            "Font '{}' has FontFile3 entry (CFF/OpenType - no TrueType cmap)",
                            base_font
                        );
                                let font_data = ff3_obj
                                    .as_reference()
                                    .and_then(|ff3_ref| {
                                        doc.load_object(ff3_ref).ok().map(|obj| (obj, ff3_ref))
                                    })
                                    .and_then(|(ff3_stream, ff3_ref)| {
                                        doc.decode_stream_with_encryption(&ff3_stream, ff3_ref).ok()
                                    })
                                    .map(|data| {
                                        // Wrap raw CFF in OpenType container for ttf-parser
                                        let data =
                                            if !data.is_empty() && data[0] == 1 && data.len() > 4 {
                                                log::info!(
                                        "Font '{}': Wrapping raw CFF in OpenType ({} bytes)",
                                        base_font,
                                        data.len()
                                    );
                                                wrap_cff_in_opentype(&data)
                                            } else {
                                                log::info!(
                                        "Font '{}' loaded embedded CFF/OpenType font ({} bytes)",
                                        base_font,
                                        data.len()
                                    );
                                                data
                                            };
                                        Arc::new(data)
                                    });
                                (font_data, false) // CFF - no TrueType cmap
                            } else if let Some(ff_obj) = descriptor_dict.get("FontFile") {
                                log::info!("Font '{}' has FontFile entry (Type 1)", base_font);
                                let font_data = ff_obj
                                    .as_reference()
                                    .and_then(|ff_ref| {
                                        doc.load_object(ff_ref).ok().map(|obj| (obj, ff_ref))
                                    })
                                    .and_then(|(ff_stream, ff_ref)| {
                                        doc.decode_stream_with_encryption(&ff_stream, ff_ref).ok()
                                    })
                                    .map(|data| {
                                        log::info!(
                                            "Font '{}' loaded embedded Type 1 font ({} bytes)",
                                            base_font,
                                            data.len()
                                        );
                                        Arc::new(data)
                                    });
                                (font_data, false) // Type 1 - no TrueType cmap
                            } else {
                                log::debug!("Font '{}' has no embedded font data", base_font);
                                (None, false)
                            };

                        (weight, descriptor_flags, stem_v_value, embedded_font, is_truetype_font)
                    } else {
                        (None, None, None, None, false)
                    }
                } else {
                    (None, None, None, None, false)
                }
            } else {
                (None, None, None, None, false)
            };

        // TrueType cmap extraction is now LAZY — deferred until first access via
        // truetype_cmap() accessor. This saves 10-25ms per font when ToUnicode CMap
        // (Priority 1) resolves all characters, making the cmap unnecessary.
        // The is_truetype_font flag is recorded here for the lazy accessor to use.

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
        //
        // However, many PDF generators (LaTeX, LibreOffice, etc.) incorrectly set the
        // Symbolic flag on non-symbolic fonts. When an explicit /Encoding entry exists,
        // we always parse it — real-world PDF viewers (MuPDF, poppler, pdf.js) do the same.
        // The Symbolic flag only controls behavior when NO /Encoding is present.
        // Pre-parse font program encoding (needed for /Differences base encoding per PDF spec)
        let font_program_enc_cache: Option<HashMap<u8, char>> =
            if let Some(font_data) = &embedded_font_data {
                if subtype == "Type1" || subtype == "MMType1" {
                    super::type1_encoding::parse_type1_encoding(font_data)
                } else {
                    super::cff_encoding::parse_cff_encoding(font_data)
                }
            } else {
                None
            };

        let (encoding, diff_multi_char_map) = if let Some(enc_obj) = font_dict.get("Encoding") {
            let resolved_enc_obj = if let Some(obj_ref) = enc_obj.as_reference() {
                doc.load_object(obj_ref)?
            } else {
                enc_obj.clone()
            };

            if is_symbolic_font(flags) {
                log::debug!(
                    "Font '{}' is symbolic (Flags={:?}) but has /Encoding — parsing it anyway (common in LaTeX/LibreOffice PDFs)",
                    base_font,
                    flags
                );
            } else {
                log::debug!("Font '{}' using /Encoding entry", base_font);
            }
            let (mut parsed_enc, mut multi_map) =
                Self::parse_encoding(&resolved_enc_obj, doc, font_program_enc_cache.as_ref())?;

            // When /Encoding is a named encoding (e.g., /WinAnsiEncoding) AND the font
            // has an embedded program, merge the font program's encoding. This handles
            // fonts where the program maps glyphs to non-standard code positions
            // (e.g., space at 0xCA) that the named encoding maps differently.
            // The font program's mappings override the standard encoding.
            if matches!(parsed_enc, Encoding::Standard(_)) {
                if let Some(prog_enc) = &font_program_enc_cache {
                    log::info!(
                        "Font '{}': merging {} font program encoding entries with {}",
                        base_font,
                        prog_enc.len(),
                        match &parsed_enc {
                            Encoding::Standard(n) => n.as_str(),
                            _ => "custom",
                        }
                    );
                    // Build Custom map: start with standard encoding, overlay font program
                    let std_name = match &parsed_enc {
                        Encoding::Standard(n) => n.clone(),
                        _ => "StandardEncoding".to_string(),
                    };
                    let mut custom_map: HashMap<u8, char> = HashMap::new();
                    for code in 0u8..=255 {
                        if let Some(unicode_str) = standard_encoding_lookup(&std_name, code) {
                            if let Some(ch) = unicode_str.chars().next() {
                                custom_map.insert(code, ch);
                            }
                        }
                    }
                    // Font program overrides
                    for (&code, &ch) in prog_enc {
                        custom_map.insert(code, ch);
                        if is_ligature_char(ch) {
                            if let Some(expanded) = expand_ligature_char(ch) {
                                multi_map.insert(code, expanded.to_string());
                            }
                        }
                    }
                    parsed_enc = Encoding::Custom(custom_map);
                }
            }

            (parsed_enc, multi_map)
        } else {
            // No /Encoding entry — use font program's built-in encoding if available
            if let Some(prog_enc) = font_program_enc_cache {
                log::info!(
                    "Font '{}' using built-in font program encoding ({} mappings)",
                    base_font,
                    prog_enc.len()
                );
                let mut multi_map: HashMap<u8, String> = HashMap::new();
                for (&code, &ch) in &prog_enc {
                    if is_ligature_char(ch) {
                        if let Some(expanded) = expand_ligature_char(ch) {
                            multi_map.insert(code, expanded.to_string());
                        }
                    }
                }
                (Encoding::Custom(prog_enc), multi_map)
            } else if is_symbolic_font(flags) {
                log::debug!(
                    "Font '{}' is symbolic with no /Encoding - will use built-in encoding (Symbol/ZapfDingbats)",
                    base_font
                );
                (Encoding::Standard("SymbolicBuiltIn".to_string()), HashMap::new())
            } else {
                log::debug!(
                    "Font '{}' has no /Encoding entry - defaulting to StandardEncoding",
                    base_font
                );
                (Encoding::Standard("StandardEncoding".to_string()), HashMap::new())
            }
        };

        // Parse ToUnicode CMap if present (Phase 5.1: Lazy Loading)
        // The CMap stream is stored raw and parsed only on first character lookup
        let to_unicode = if let Some(cmap_ref) = font_dict
            .get("ToUnicode")
            .and_then(|obj| obj.as_reference())
        {
            let stream_opt = match doc.load_object(cmap_ref) {
                Ok(cmap_obj) => {
                    match doc.decode_stream_with_encryption(&cmap_obj, cmap_ref) {
                        Ok(data) => Some(data),
                        Err(e) => {
                            log::warn!(
                                "Font '{}': Failed to decrypt/decode ToUnicode CMap stream {:?}: {}",
                                base_font, cmap_ref, e
                            );
                            None
                        },
                    }
                },
                Err(e) => {
                    log::warn!(
                        "Font '{}': Failed to load ToUnicode CMap object {:?}: {}",
                        base_font,
                        cmap_ref,
                        e
                    );
                    None
                },
            };

            if let Some(stream_bytes) = stream_opt {
                // Store raw bytes for lazy parsing — LazyCMap handles errors on first access.
                // Skipping eager validation avoids parsing every CMap twice.
                log::info!(
                    "ToUnicode CMap stream loaded for font '{}': {} bytes (lazy parsing enabled)",
                    base_font,
                    stream_bytes.len()
                );
                Some(LazyCMap::new(stream_bytes))
            } else {
                // Specific error already logged above in the match arms
                None
            }
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
            // Type0 fonts use /W and /DW arrays parsed via parse_descendant_fonts
            log::debug!("Font '{}': Type0 font, widths parsed from CIDFont /W array", base_font);
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

        // Phase 3: Parse DescendantFonts for Type0 fonts
        let (
            cid_to_gid_map,
            cid_system_info,
            cid_font_type,
            cid_widths,
            cid_default_width,
            descendant_tt_cmap,
        ) = if subtype == "Type0" {
            match Self::parse_descendant_fonts(font_dict, &base_font, doc) {
                Ok((map, info, ftype, widths, dw, tt_cmap, desc_embedded)) => {
                    log::info!(
                            "Font '{}': Parsed DescendantFonts - CIDFontType={}, CIDSystemInfo={}-{}, widths={}, embedded={}",
                            base_font,
                            ftype.as_ref().unwrap_or(&"Unknown".to_string()),
                            info.as_ref()
                                .map(|s| s.registry.as_str())
                                .unwrap_or("Unknown"),
                            info.as_ref()
                                .map(|s| s.ordering.as_str())
                                .unwrap_or("Unknown"),
                            widths.as_ref().map(|m| m.len()).unwrap_or(0),
                            desc_embedded.is_some()
                        );
                    // Use embedded font data from CIDFont descendant if top-level didn't have it
                    if desc_embedded.is_some() && embedded_font_data.is_none() {
                        embedded_font_data = desc_embedded;
                    }
                    (map, info, ftype, widths, dw, tt_cmap)
                },
                Err(e) => {
                    log::warn!(
                        "Font '{}': Failed to parse DescendantFonts: {}. Using Identity fallback.",
                        base_font,
                        e
                    );
                    (Some(CIDToGIDMap::Identity), None, None, None, 1000.0, None)
                },
            }
        } else {
            (None, None, None, None, 1000.0, None)
        };

        // Pre-populate OnceLock with descendant's TrueType cmap if available.
        // Otherwise leave it for lazy extraction from embedded_font_data.
        let truetype_cmap_lock = std::sync::OnceLock::new();
        if let Some(desc_cmap) = descendant_tt_cmap {
            let _ = truetype_cmap_lock.set(Some(desc_cmap));
        }

        // Parse CFF GID mapping ONLY for simple (non-Type0) fonts with embedded CFF data.
        // Type0/CID fonts use Identity-H encoding and CIDToGIDMap, not CFF Standard Encoding.
        let cff_gid_map = if subtype != "Type0" {
            embedded_font_data.as_ref().and_then(|data| {
                super::cff_encoding::parse_cff_gid_mapping(data).inspect(|map| {
                    log::debug!(
                        "Font '{}': parsed CFF GID mapping ({} entries)",
                        base_font,
                        map.len()
                    );
                })
            })
        } else {
            None
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
            truetype_cmap: truetype_cmap_lock,
            is_truetype_font,
            cid_to_gid_map,
            cid_system_info,
            cid_font_type,
            widths,
            first_char,
            last_char,
            default_width,
            cid_widths,
            cid_default_width,
            cff_gid_map,
            multi_char_map: diff_multi_char_map,
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        })
    }

    /// Parse encoding from an encoding object.
    ///
    /// Phase 3: Parse CIDSystemInfo from CIDFont dictionary
    /// Extracts Registry, Ordering, and Supplement for character collection identification
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.3
    fn parse_cidsysteminfo(
        cidfont_dict: &HashMap<String, Object>,
        doc: &PdfDocument,
    ) -> Result<CIDSystemInfo> {
        let sysinfo_obj = cidfont_dict
            .get("CIDSystemInfo")
            .ok_or_else(|| Error::ParseError {
                offset: 0,
                reason: "CIDFont missing required /CIDSystemInfo entry".to_string(),
            })?;

        // Resolve reference if needed
        let resolved = if let Some(ref_obj) = sysinfo_obj.as_reference() {
            doc.load_object(ref_obj)?
        } else {
            sysinfo_obj.clone()
        };

        let sysinfo_dict = resolved.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: "CIDSystemInfo is not a dictionary".to_string(),
        })?;

        let registry = sysinfo_dict
            .get("Registry")
            .and_then(|obj| obj.as_string())
            .map(|s| String::from_utf8_lossy(s).to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let ordering = sysinfo_dict
            .get("Ordering")
            .and_then(|obj| obj.as_string())
            .map(|s| String::from_utf8_lossy(s).to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let supplement = sysinfo_dict
            .get("Supplement")
            .and_then(|obj| obj.as_integer())
            .unwrap_or(0) as i32;

        log::debug!(
            "CIDSystemInfo parsed: Registry={}, Ordering={}, Supplement={}",
            registry,
            ordering,
            supplement
        );

        Ok(CIDSystemInfo {
            registry,
            ordering,
            supplement,
        })
    }

    /// Phase 3: Parse DescendantFonts array for Type0 fonts
    /// Extracts CIDFont dictionary and related information
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.1
    ///
    /// Returns: (CIDToGIDMap, CIDSystemInfo, CIDFontType, CIDWidths, DefaultWidth)
    fn parse_descendant_fonts(
        font_dict: &HashMap<String, Object>,
        base_font: &str,
        doc: &PdfDocument,
    ) -> Result<(
        Option<CIDToGIDMap>,
        Option<CIDSystemInfo>,
        Option<String>,
        Option<HashMap<u16, f32>>,
        f32,
        Option<TrueTypeCMap>, // TrueType cmap from descendant's embedded font
        Option<Arc<Vec<u8>>>, // Embedded font data from CIDFont's FontDescriptor
    )> {
        let descendant_obj = font_dict
            .get("DescendantFonts")
            .ok_or_else(|| Error::ParseError {
                offset: 0,
                reason: format!(
                    "Type0 font '{}' missing required /DescendantFonts entry",
                    base_font
                ),
            })?;

        // Resolve reference if needed
        let resolved = if let Some(ref_obj) = descendant_obj.as_reference() {
            doc.load_object(ref_obj)?
        } else {
            descendant_obj.clone()
        };

        let array = resolved.as_array().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: format!("Type0 font '{}': DescendantFonts is not an array", base_font),
        })?;

        if array.is_empty() {
            return Err(Error::ParseError {
                offset: 0,
                reason: format!(
                    "Type0 font '{}': DescendantFonts array is empty - must have at least 1 element",
                    base_font
                ),
            });
        }

        // Use first element (PDF spec: "Usually contains a single element")
        if array.len() > 1 {
            log::warn!(
                "Font '{}': DescendantFonts array has {} elements, using first",
                base_font,
                array.len()
            );
        }

        let cidfont_ref = array[0].as_reference().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: format!("Type0 font '{}': DescendantFonts[0] is not a reference", base_font),
        })?;

        let cidfont_obj = doc.load_object(cidfont_ref)?;
        let cidfont_dict = cidfont_obj.as_dict().ok_or_else(|| Error::ParseError {
            offset: 0,
            reason: format!("Type0 font '{}': CIDFont is not a dictionary", base_font),
        })?;

        // Get CIDFont subtype (required: CIDFontType0 or CIDFontType2)
        let cid_font_type = cidfont_dict
            .get("Subtype")
            .and_then(|obj| obj.as_name())
            .ok_or_else(|| Error::ParseError {
                offset: 0,
                reason: format!("Type0 font '{}': CIDFont missing required /Subtype", base_font),
            })?
            .to_string();

        // Validate subtype
        if cid_font_type != "CIDFontType0" && cid_font_type != "CIDFontType2" {
            return Err(Error::ParseError {
                offset: 0,
                reason: format!(
                    "Type0 font '{}': Invalid CIDFontType '{}' (must be CIDFontType0 or CIDFontType2)",
                    base_font, cid_font_type
                ),
            });
        }

        // Parse CIDSystemInfo (required for all CIDFonts)
        let cid_system_info = match Self::parse_cidsysteminfo(cidfont_dict, doc) {
            Ok(info) => Some(info),
            Err(e) => {
                log::warn!(
                    "Font '{}': Failed to parse CIDSystemInfo: {}. Continuing with None.",
                    base_font,
                    e
                );
                None
            },
        };

        // Parse CIDToGIDMap (only for CIDFontType2 - TrueType-based)
        let cid_to_gid_map = if cid_font_type == "CIDFontType2" {
            match cidfont_dict.get("CIDToGIDMap") {
                None => {
                    // Default to Identity if not specified
                    log::debug!(
                        "Font '{}': CIDToGIDMap not specified, defaulting to Identity",
                        base_font
                    );
                    Some(CIDToGIDMap::Identity)
                },
                Some(cidtogid_obj) => {
                    // Handle Name object "/Identity"
                    if let Some(name) = cidtogid_obj.as_name() {
                        if name == "Identity" {
                            log::debug!("Font '{}': CIDToGIDMap is Identity", base_font);
                            Some(CIDToGIDMap::Identity)
                        } else {
                            log::warn!(
                                "Font '{}': Invalid CIDToGIDMap name '{}' (only 'Identity' is valid as name)",
                                base_font,
                                name
                            );
                            Some(CIDToGIDMap::Identity) // Fallback
                        }
                    } else if let Some(stream_ref) = cidtogid_obj.as_reference() {
                        // Handle Stream object (binary uint16 array)
                        match doc.load_object(stream_ref) {
                            Ok(stream_obj) => {
                                match doc.decode_stream_with_encryption(&stream_obj, stream_ref) {
                                    Ok(stream_data) => {
                                        // Validate stream length (must be even)
                                        if stream_data.len() % 2 != 0 {
                                            log::warn!(
                                            "Font '{}': CIDToGIDMap stream has odd length {} (must be even). Using Identity fallback.",
                                            base_font,
                                            stream_data.len()
                                        );
                                            Some(CIDToGIDMap::Identity)
                                        } else if stream_data.is_empty() {
                                            log::warn!(
                                            "Font '{}': CIDToGIDMap stream is empty. Using Identity fallback.",
                                            base_font
                                        );
                                            Some(CIDToGIDMap::Identity)
                                        } else {
                                            // Parse big-endian uint16 array
                                            let num_entries = stream_data.len() / 2;
                                            let mut map = Vec::with_capacity(num_entries);
                                            for i in 0..num_entries {
                                                let gid = u16::from_be_bytes([
                                                    stream_data[i * 2],
                                                    stream_data[i * 2 + 1],
                                                ]);
                                                map.push(gid);
                                            }
                                            log::debug!(
                                            "Font '{}': Loaded explicit CIDToGIDMap with {} entries",
                                            base_font,
                                            num_entries
                                        );
                                            Some(CIDToGIDMap::Explicit(map))
                                        }
                                    },
                                    Err(e) => {
                                        log::warn!(
                                        "Font '{}': CIDToGIDMap stream decode failed: {}. Using Identity fallback.",
                                        base_font,
                                        e
                                    );
                                        Some(CIDToGIDMap::Identity)
                                    },
                                }
                            },
                            Err(e) => {
                                log::warn!(
                                    "Font '{}': CIDToGIDMap stream object load failed: {}. Using Identity fallback.",
                                    base_font,
                                    e
                                );
                                Some(CIDToGIDMap::Identity)
                            },
                        }
                    } else {
                        log::warn!(
                            "Font '{}': CIDToGIDMap is neither Name nor Stream reference. Using Identity fallback.",
                            base_font
                        );
                        Some(CIDToGIDMap::Identity)
                    }
                },
            }
        } else {
            // CIDFontType0 (CFF/OpenType) doesn't use CIDToGIDMap
            log::debug!(
                "Font '{}': CIDFontType0 (CFF/OpenType) - no CIDToGIDMap needed",
                base_font
            );
            None
        };

        // Parse /DW (default width for CIDs) - PDF Spec Section 9.7.4.3
        // Default is 1000 if not specified
        let cid_default_width = cidfont_dict
            .get("DW")
            .and_then(|obj| {
                // Resolve indirect reference if needed
                let resolved = if let Some(r) = obj.as_reference() {
                    doc.load_object(r).ok()
                } else {
                    Some(obj.clone())
                };
                resolved.and_then(|o| match &o {
                    Object::Integer(i) => Some(*i as f32),
                    Object::Real(r) => Some(*r as f32),
                    _ => None,
                })
            })
            .unwrap_or(1000.0);

        // Parse /W array (CID widths) - PDF Spec Section 9.7.4.3
        // Resolve /W reference if needed before parsing (common for large arrays)
        let resolved_cidfont_dict = if let Some(w_obj) = cidfont_dict.get("W") {
            if let Some(r) = w_obj.as_reference() {
                match doc.load_object(r) {
                    Ok(resolved) => {
                        let mut dict_clone = cidfont_dict.clone();
                        dict_clone.insert("W".to_string(), resolved);
                        std::borrow::Cow::Owned(dict_clone)
                    },
                    Err(e) => {
                        log::warn!("Font '{}': Failed to resolve /W reference: {}", base_font, e);
                        std::borrow::Cow::Borrowed(cidfont_dict)
                    },
                }
            } else {
                std::borrow::Cow::Borrowed(cidfont_dict)
            }
        } else {
            std::borrow::Cow::Borrowed(cidfont_dict)
        };
        let cid_widths = Self::parse_cid_widths(&resolved_cidfont_dict, base_font);

        if cid_widths.is_some() {
            log::debug!(
                "Font '{}': Parsed CID widths - {} entries, default width {}",
                base_font,
                cid_widths.as_ref().map(|m| m.len()).unwrap_or(0),
                cid_default_width
            );
        }

        // Extract TrueType cmap from descendant's FontDescriptor if available.
        // Type0 parent fonts often have no embedded data — it's on the CIDFont.
        let descendant_tt_cmap = if cid_font_type == "CIDFontType2" {
            Self::extract_truetype_cmap_from_descriptor(cidfont_dict, base_font, doc)
        } else {
            None
        };

        // Extract embedded font data from CIDFont's FontDescriptor.
        // Per PDF spec, embedded font programs for Type0 fonts live on the
        // CIDFont descendant's FontDescriptor, not on the Type0 wrapper.
        let descendant_embedded =
            Self::extract_embedded_font_from_descriptor(cidfont_dict, base_font, doc);

        Ok((
            cid_to_gid_map,
            cid_system_info,
            Some(cid_font_type),
            cid_widths,
            cid_default_width,
            descendant_tt_cmap,
            descendant_embedded,
        ))
    }

    /// Extract TrueType cmap from a font dictionary's /FontDescriptor /FontFile2.
    fn extract_truetype_cmap_from_descriptor(
        font_dict: &HashMap<String, Object>,
        base_font: &str,
        doc: &PdfDocument,
    ) -> Option<TrueTypeCMap> {
        let desc_obj = font_dict.get("FontDescriptor")?;
        let desc = if let Some(r) = desc_obj.as_reference() {
            doc.load_object(r).ok()?
        } else {
            desc_obj.clone()
        };
        let desc_dict = desc.as_dict()?;
        let ff2_obj = desc_dict.get("FontFile2")?;
        let ff2_ref = ff2_obj.as_reference()?;
        let ff2_stream = match doc.load_object(ff2_ref) {
            Ok(obj) => obj,
            Err(e) => {
                log::warn!(
                    "Font '{}': Failed to load FontFile2 object {:?}: {}",
                    base_font,
                    ff2_ref,
                    e
                );
                return None;
            },
        };
        let font_data = match doc.decode_stream_with_encryption(&ff2_stream, ff2_ref) {
            Ok(data) => data,
            Err(e) => {
                log::warn!(
                    "Font '{}': Failed to decrypt/decode FontFile2 stream {:?}: {}",
                    base_font,
                    ff2_ref,
                    e
                );
                return None;
            },
        };
        if font_data.is_empty() {
            return None;
        }
        match TrueTypeCMap::from_font_data(&font_data) {
            Ok(cmap) if !cmap.is_empty() => {
                log::info!(
                    "Font '{}': Extracted TrueType cmap from descendant CIDFont ({} mappings)",
                    base_font,
                    cmap.len()
                );
                Some(cmap)
            },
            _ => None,
        }
    }

    /// Extract embedded font data from a font dictionary's /FontDescriptor.
    /// Checks FontFile2 (TrueType), FontFile3 (CFF/OpenType), and FontFile (Type 1).
    fn extract_embedded_font_from_descriptor(
        font_dict: &HashMap<String, Object>,
        base_font: &str,
        doc: &PdfDocument,
    ) -> Option<Arc<Vec<u8>>> {
        let desc_obj = font_dict.get("FontDescriptor")?;
        let desc = if let Some(r) = desc_obj.as_reference() {
            doc.load_object(r).ok()?
        } else {
            desc_obj.clone()
        };
        let desc_dict = desc.as_dict()?;

        // Try FontFile2 (TrueType), FontFile3 (CFF/OpenType), FontFile (Type 1)
        let font_file_keys = ["FontFile2", "FontFile3", "FontFile"];
        for key in &font_file_keys {
            if let Some(ff_obj) = desc_dict.get(*key) {
                let ff_ref = match ff_obj.as_reference() {
                    Some(r) => r,
                    None => continue,
                };
                let ff_stream = match doc.load_object(ff_ref) {
                    Ok(obj) => obj,
                    Err(e) => {
                        log::warn!(
                            "Font '{}': Failed to load {} {:?}: {}",
                            base_font,
                            key,
                            ff_ref,
                            e
                        );
                        continue;
                    },
                };
                let font_data = match doc.decode_stream_with_encryption(&ff_stream, ff_ref) {
                    Ok(data) => data,
                    Err(e) => {
                        log::warn!("Font '{}': Failed to decode {} stream: {}", base_font, key, e);
                        continue;
                    },
                };
                if !font_data.is_empty() {
                    // If this is raw CFF data (FontFile3), wrap it in an OpenType
                    // container so ttf-parser can parse it.
                    let font_data = if *key == "FontFile3" && !font_data.is_empty()
                        && font_data[0] == 1 // CFF version 1
                        && font_data.len() > 4
                    {
                        log::info!(
                            "Font '{}': Wrapping raw CFF in OpenType container ({} bytes)",
                            base_font,
                            font_data.len()
                        );
                        wrap_cff_in_opentype(&font_data)
                    } else {
                        font_data
                    };
                    log::info!(
                        "Font '{}': Extracted embedded font from {} ({} bytes)",
                        base_font,
                        key,
                        font_data.len()
                    );
                    return Some(Arc::new(font_data));
                }
            }
        }
        None
    }
}

/// Wrap raw CFF font data in a minimal OpenType container so ttf-parser can parse it.
/// Creates an OpenType font with `head` and `CFF ` tables (both required by ttf-parser).
fn wrap_cff_in_opentype(cff_data: &[u8]) -> Vec<u8> {
    let num_tables: u16 = 4; // CFF + head + hhea + maxp
    let search_range: u16 = 32; // largest power of 2 <= numTables*16 = 64 → 32 (2 tables)
    let entry_selector: u16 = 2;
    let range_shift: u16 = (num_tables * 16) - search_range;

    // Minimal head table (54 bytes) — OpenType spec required fields
    let head_table: [u8; 54] = [
        0x00, 0x01, 0x00, 0x00, // majorVersion=1, minorVersion=0
        0x00, 0x01, 0x00, 0x00, // fontRevision=1.0
        0x00, 0x00, 0x00, 0x00, // checksumAdjustment (0, will be ignored)
        0x5F, 0x0F, 0x3C, 0xF5, // magicNumber
        0x00, 0x0B, // flags (baseline at y=0, lsb at x=0, etc)
        0x03, 0xE8, // unitsPerEm = 1000
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // created (0)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // modified (0)
        0xFF, 0x38, // xMin = -200
        0xFF, 0x38, // yMin = -200
        0x03, 0xE8, // xMax = 1000
        0x03, 0xE8, // yMax = 1000
        0x00, 0x00, // macStyle
        0x00, 0x08, // lowestRecPPEM = 8
        0x00, 0x02, // fontDirectionHint
        0x00, 0x01, // indexToLocFormat = 1 (long)
        0x00, 0x00, // glyphDataFormat
    ];

    // Minimal hhea table (36 bytes)
    let hhea_table: [u8; 36] = [
        0x00, 0x01, 0x00, 0x00, // majorVersion=1, minorVersion=0
        0x03, 0x20, // ascender = 800
        0xFF, 0x38, // descender = -200
        0x00, 0x00, // lineGap = 0
        0x04, 0x00, // advanceWidthMax = 1024
        0x00, 0x00, // minLeftSideBearing = 0
        0x00, 0x00, // minRightSideBearing = 0
        0x04, 0x00, // xMaxExtent = 1024
        0x00, 0x01, // caretSlopeRise = 1
        0x00, 0x00, // caretSlopeRun = 0
        0x00, 0x00, // caretOffset = 0
        0x00, 0x00, // reserved
        0x00, 0x00, // reserved
        0x00, 0x00, // reserved
        0x00, 0x00, // reserved
        0x00, 0x00, // metricDataFormat = 0
        0x01, 0x00, // numberOfHMetrics = 256
    ];

    // Minimal maxp table (6 bytes for CFF fonts — version 0.5)
    let maxp_table: [u8; 6] = [
        0x00, 0x00, 0x50, 0x00, // version = 0.5 (CFF)
        0x01, 0x00, // numGlyphs = 256
    ];

    // Layout: offset table (12) + 4 table records (64) = 76 bytes header
    let header_size: u32 = 12 + (num_tables as u32) * 16;
    // Place tables: head, hhea, maxp, CFF (alphabetical by tag within each group)
    let head_offset = (header_size + 3) & !3;
    let head_len = head_table.len() as u32;
    let hhea_offset = ((head_offset + head_len) + 3) & !3;
    let hhea_len = hhea_table.len() as u32;
    let maxp_offset = ((hhea_offset + hhea_len) + 3) & !3;
    let maxp_len = maxp_table.len() as u32;
    let cff_offset = ((maxp_offset + maxp_len) + 3) & !3;
    let cff_len = cff_data.len() as u32;

    fn table_checksum(data: &[u8]) -> u32 {
        let mut sum: u32 = 0;
        for chunk in data.chunks(4) {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            sum = sum.wrapping_add(u32::from_be_bytes(bytes));
        }
        sum
    }

    let mut out = Vec::with_capacity((cff_offset + cff_len) as usize);

    // Offset table (12 bytes)
    out.extend_from_slice(b"OTTO");
    out.extend_from_slice(&num_tables.to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());

    // Table record: CFF (alphabetical order: CFF before head)
    out.extend_from_slice(b"CFF ");
    out.extend_from_slice(&table_checksum(cff_data).to_be_bytes());
    out.extend_from_slice(&cff_offset.to_be_bytes());
    out.extend_from_slice(&cff_len.to_be_bytes());

    // Table record: head
    out.extend_from_slice(b"head");
    out.extend_from_slice(&table_checksum(&head_table).to_be_bytes());
    out.extend_from_slice(&head_offset.to_be_bytes());
    out.extend_from_slice(&head_len.to_be_bytes());

    // Table record: hhea
    out.extend_from_slice(b"hhea");
    out.extend_from_slice(&table_checksum(&hhea_table).to_be_bytes());
    out.extend_from_slice(&hhea_offset.to_be_bytes());
    out.extend_from_slice(&hhea_len.to_be_bytes());

    // Table record: maxp
    out.extend_from_slice(b"maxp");
    out.extend_from_slice(&table_checksum(&maxp_table).to_be_bytes());
    out.extend_from_slice(&maxp_offset.to_be_bytes());
    out.extend_from_slice(&maxp_len.to_be_bytes());

    // head table data
    while out.len() < head_offset as usize {
        out.push(0);
    }
    out.extend_from_slice(&head_table);

    // hhea table data
    while out.len() < hhea_offset as usize {
        out.push(0);
    }
    out.extend_from_slice(&hhea_table);

    // maxp table data
    while out.len() < maxp_offset as usize {
        out.push(0);
    }
    out.extend_from_slice(&maxp_table);

    // Pad to CFF offset
    while out.len() < cff_offset as usize {
        out.push(0);
    }

    // CFF data
    out.extend_from_slice(cff_data);

    out
}

impl FontInfo {
    /// Parse CIDFont /W array for glyph widths.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.3, the /W array has two formats:
    /// - `c [w1 w2 ... wn]` - CID c has width w1, c+1 has width w2, etc.
    /// - `cfirst clast w` - CIDs from cfirst to clast all have width w
    ///
    /// These formats can be mixed in a single array.
    ///
    /// # Example /W array
    /// ```pdf
    /// /W [
    ///   1 [500 600 700]     % CID 1=500, CID 2=600, CID 3=700
    ///   100 200 300         % CIDs 100-200 all have width 300
    /// ]
    /// ```
    fn parse_cid_widths(
        cidfont_dict: &HashMap<String, Object>,
        base_font: &str,
    ) -> Option<HashMap<u16, f32>> {
        let w_obj = cidfont_dict.get("W")?;
        let w_array = w_obj.as_array()?;

        if w_array.is_empty() {
            return None;
        }

        let mut widths: HashMap<u16, f32> = HashMap::new();
        let mut i = 0;

        while i < w_array.len() {
            // First element must be a CID (integer)
            let cid_start = match &w_array[i] {
                Object::Integer(c) => *c as u16,
                _ => {
                    log::warn!(
                        "Font '{}': /W array element {} is not an integer, skipping",
                        base_font,
                        i
                    );
                    i += 1;
                    continue;
                },
            };
            i += 1;

            if i >= w_array.len() {
                break;
            }

            // Second element is either:
            // - An array of widths (format: c [w1 w2 ...])
            // - An integer CID end (format: cfirst clast w)
            match &w_array[i] {
                Object::Array(width_array) => {
                    // Format: c [w1 w2 ... wn]
                    for (j, width_obj) in width_array.iter().enumerate() {
                        let width = match width_obj {
                            Object::Integer(w) => *w as f32,
                            Object::Real(w) => *w as f32,
                            _ => continue,
                        };
                        let cid = cid_start.saturating_add(j as u16);
                        widths.insert(cid, width);
                    }
                    i += 1;
                },
                Object::Integer(cid_end) => {
                    // Format: cfirst clast w
                    let cid_end = *cid_end as u16;
                    i += 1;

                    if i >= w_array.len() {
                        log::warn!(
                            "Font '{}': /W array missing width for CID range {}-{}",
                            base_font,
                            cid_start,
                            cid_end
                        );
                        break;
                    }

                    let width = match &w_array[i] {
                        Object::Integer(w) => *w as f32,
                        Object::Real(w) => *w as f32,
                        _ => {
                            log::warn!(
                                "Font '{}': /W array has invalid width for CID range {}-{}",
                                base_font,
                                cid_start,
                                cid_end
                            );
                            i += 1;
                            continue;
                        },
                    };
                    i += 1;

                    // Apply width to all CIDs in range
                    for cid in cid_start..=cid_end {
                        widths.insert(cid, width);
                    }
                },
                _ => {
                    log::warn!(
                        "Font '{}': /W array has unexpected element type after CID {}",
                        base_font,
                        cid_start
                    );
                    i += 1;
                },
            }
        }

        if widths.is_empty() {
            None
        } else {
            Some(widths)
        }
    }

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
    fn parse_encoding(
        enc_obj: &Object,
        doc: &PdfDocument,
        font_program_encoding: Option<&HashMap<u8, char>>,
    ) -> Result<(Encoding, HashMap<u8, String>)> {
        let empty_map = HashMap::new();
        // Encoding can be either a name or a dictionary
        if let Some(name) = enc_obj.as_name() {
            // Standard encoding names
            match name {
                "WinAnsiEncoding" => {
                    Ok((Encoding::Standard("WinAnsiEncoding".to_string()), empty_map))
                },
                "MacRomanEncoding" => {
                    Ok((Encoding::Standard("MacRomanEncoding".to_string()), empty_map))
                },
                "MacExpertEncoding" => {
                    Ok((Encoding::Standard("MacExpertEncoding".to_string()), empty_map))
                },
                "Identity-H" | "Identity-V" => Ok((Encoding::Identity, empty_map)),
                _ => Ok((Encoding::Standard(name.to_string()), empty_map)),
            }
        } else if let Some(dict) = enc_obj.as_dict() {
            // Check if this is a CMap stream (Type0 font encoding reference)
            // Per PDF Spec §9.7.5.2, Type0 fonts can reference a CMap stream
            // via /Encoding. For known Adobe character collections (Japan1, GB1,
            // CNS1, Korea1), these define charcode→CID identity mappings and we
            // can resolve CIDs via predefined CID-to-Unicode tables.
            // For custom CMaps (e.g., "Prince-ArialMT-H"), we preserve the default
            // behavior since we can't parse arbitrary CMap programs yet.
            if let Some(cmap_name) = dict.get("CMapName").and_then(|n| n.as_name()) {
                // Check for Adobe standard character collection CMaps.
                // These are named like "Adobe-Japan1-2", "Adobe-Korea1-0", etc.
                // For these collections, charcode→CID is identity, and we can
                // resolve CID→Unicode via predefined tables.
                let is_adobe_collection = cmap_name.starts_with("Adobe-")
                    && (cmap_name.contains("Japan")
                        || cmap_name.contains("GB")
                        || cmap_name.contains("CNS")
                        || cmap_name.contains("Korea"));
                if is_adobe_collection {
                    log::debug!(
                        "Encoding is Adobe CMap stream (CMapName={:?}), treating as Identity",
                        cmap_name
                    );
                    return Ok((Encoding::Identity, HashMap::new()));
                }
                // For predefined PDF CMaps like "Identity-H", "Identity-V"
                if cmap_name == "Identity-H" || cmap_name == "Identity-V" {
                    return Ok((Encoding::Identity, HashMap::new()));
                }
                // Custom CMap streams (e.g., "Prince-ArialMT-H", "OneByteIdentityH")
                log::debug!(
                    "Encoding is custom CMap stream (CMapName={:?}), treating as Standard",
                    cmap_name
                );
                return Ok((Encoding::Standard(cmap_name.to_string()), HashMap::new()));
            }

            // Custom encoding dictionary - parse /Differences array
            let mut multi_char_map: HashMap<u8, String> = HashMap::new();

            // Step 1: Get base encoding (if specified)
            let mut encoding_map: HashMap<u8, char> = if let Some(base_enc_obj) =
                dict.get("BaseEncoding")
            {
                // Resolve indirect reference for /BaseEncoding
                let resolved_base = if let Some(obj_ref) = base_enc_obj.as_reference() {
                    doc.load_object(obj_ref).ok()
                } else {
                    None
                };
                let base_obj = resolved_base.as_ref().unwrap_or(base_enc_obj);

                if let Some(base_name) = base_obj.as_name() {
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
            } else if let Some(prog_enc) = font_program_encoding {
                // PDF Spec ISO 32000-1:2008, Section 9.6.6.1:
                // "If BaseEncoding is absent and the font has a built-in encoding,
                // the built-in encoding shall be used as the base encoding."
                prog_enc.clone()
            } else {
                // No base encoding specified and no font program - use StandardEncoding as default
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

                // Resolve indirect reference for /Differences itself
                let resolved_diff = if let Some(obj_ref) = differences_obj.as_reference() {
                    doc.load_object(obj_ref).ok()
                } else {
                    None
                };
                let diff_obj = resolved_diff.as_ref().unwrap_or(differences_obj);

                if let Some(diff_array) = diff_obj.as_array() {
                    log::info!("/Differences array has {} items", diff_array.len());
                    let mut current_code: u32 = 0;

                    for item in diff_array {
                        // Resolve indirect references within the array
                        let resolved_item = if let Some(obj_ref) = item.as_reference() {
                            doc.load_object(obj_ref).ok()
                        } else {
                            None
                        };
                        let actual_item = resolved_item.as_ref().unwrap_or(item);

                        match actual_item {
                            Object::Integer(code) => {
                                // New starting code
                                current_code = *code as u32;
                            },
                            Object::Name(glyph_name) => {
                                // Map glyph name to Unicode character(s)
                                if let Some(unicode_char) = glyph_name_to_unicode(glyph_name) {
                                    if current_code <= 255 {
                                        encoding_map.insert(current_code as u8, unicode_char);
                                        if is_ligature_char(unicode_char) {
                                            log::info!(
                                                "/Differences: code {} → /{} → '{}' (U+{:04X})",
                                                current_code,
                                                glyph_name,
                                                unicode_char,
                                                unicode_char as u32
                                            );
                                        }
                                    }
                                } else if let Some(unicode_string) =
                                    glyph_name_to_unicode_string(glyph_name)
                                {
                                    // Compound glyph name (e.g. f_f → "ff", f_f_i → "ffi")
                                    if current_code <= 255 {
                                        multi_char_map
                                            .insert(current_code as u8, unicode_string.clone());
                                        log::info!(
                                            "/Differences: code {} → /{} → {:?} (compound)",
                                            current_code,
                                            glyph_name,
                                            unicode_string
                                        );
                                    }
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
                                log::warn!(
                                    "Unexpected item in /Differences array: {:?}",
                                    actual_item
                                );
                            },
                        }
                    }

                    log::debug!(
                        "Parsed /Differences array with {} custom mappings",
                        encoding_map.len()
                    );
                } else {
                    log::warn!("/Differences is not an array: {:?}", diff_obj);
                }
            }

            if !encoding_map.is_empty() || !multi_char_map.is_empty() {
                Ok((Encoding::Custom(encoding_map), multi_char_map))
            } else {
                Ok((Encoding::Standard("StandardEncoding".to_string()), HashMap::new()))
            }
        } else {
            Ok((Encoding::Standard("StandardEncoding".to_string()), HashMap::new()))
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
        // For Type0 (CID) fonts, check cid_widths first
        // The char_code is the CID for these fonts
        if let Some(cid_widths) = &self.cid_widths {
            if let Some(&width) = cid_widths.get(&char_code) {
                return width;
            }
            // CID not in /W array, use /DW default
            return self.cid_default_width;
        }

        // For simple fonts, use the widths array
        if let Some(widths) = &self.widths {
            if let Some(first_char) = self.first_char {
                let index = char_code as i32 - first_char as i32;
                if index >= 0 && (index as usize) < widths.len() {
                    return widths[index as usize];
                }
            }
        }
        // For standard 14 fonts without /Widths, use built-in metrics
        if let Some(w) = self.get_standard_font_width(char_code) {
            return w;
        }
        self.default_width
    }

    /// Look up width from standard 14 font metrics when /Widths array is absent.
    fn get_standard_font_width(&self, char_code: u16) -> Option<f32> {
        // Only apply for standard 14 fonts (no embedded data, no widths array)
        if self.widths.is_some() || self.embedded_font_data.is_some() {
            return None;
        }
        let name = &self.base_font;
        let is_times = name.contains("Times");
        let is_helvetica = name.contains("Helvetica") || name.contains("Arial");
        let is_courier = name.contains("Courier");

        if !is_times && !is_helvetica && !is_courier {
            return None;
        }

        if is_courier {
            return Some(600.0); // Monospace
        }

        let code = char_code as u8;
        // Times-Roman standard widths (most common chars, PDF spec Appendix D)
        if is_times {
            return Some(match code {
                32 => 250.0,
                33 => 333.0,
                34 => 408.0,
                35 => 500.0,
                36 => 500.0,
                37 => 833.0,
                38 => 778.0,
                39 => 333.0,
                40 => 333.0,
                41 => 333.0,
                42 => 500.0,
                43 => 564.0,
                44 => 250.0,
                45 => 333.0,
                46 => 250.0,
                47 => 278.0,
                48 => 500.0,
                49 => 500.0,
                50 => 500.0,
                51 => 500.0,
                52 => 500.0,
                53 => 500.0,
                54 => 500.0,
                55 => 500.0,
                56 => 500.0,
                57 => 500.0,
                58 => 278.0,
                59 => 278.0,
                60 => 564.0,
                61 => 564.0,
                62 => 564.0,
                63 => 444.0,
                64 => 921.0,
                65 => 722.0,
                66 => 667.0,
                67 => 667.0,
                68 => 722.0,
                69 => 611.0,
                70 => 556.0,
                71 => 722.0,
                72 => 722.0,
                73 => 333.0,
                74 => 389.0,
                75 => 722.0,
                76 => 611.0,
                77 => 889.0,
                78 => 722.0,
                79 => 722.0,
                80 => 556.0,
                81 => 722.0,
                82 => 667.0,
                83 => 556.0,
                84 => 611.0,
                85 => 722.0,
                86 => 722.0,
                87 => 944.0,
                88 => 722.0,
                89 => 722.0,
                90 => 611.0,
                91 => 333.0,
                92 => 278.0,
                93 => 333.0,
                97 => 444.0,
                98 => 500.0,
                99 => 444.0,
                100 => 500.0,
                101 => 444.0,
                102 => 333.0,
                103 => 500.0,
                104 => 500.0,
                105 => 278.0,
                106 => 278.0,
                107 => 500.0,
                108 => 278.0,
                109 => 778.0,
                110 => 500.0,
                111 => 500.0,
                112 => 500.0,
                113 => 500.0,
                114 => 333.0,
                115 => 389.0,
                116 => 278.0,
                117 => 500.0,
                118 => 500.0,
                119 => 722.0,
                120 => 500.0,
                121 => 500.0,
                122 => 444.0,
                _ => return None,
            });
        }

        // Helvetica standard widths
        if is_helvetica {
            return Some(match code {
                32 => 278.0,
                33 => 278.0,
                34 => 355.0,
                44 => 278.0,
                45 => 333.0,
                46 => 278.0,
                47 => 278.0,
                48..=57 => 556.0, // digits
                58 => 278.0,
                59 => 278.0,
                65 => 667.0,
                66 => 667.0,
                67 => 722.0,
                68 => 722.0,
                69 => 667.0,
                70 => 611.0,
                71 => 778.0,
                72 => 722.0,
                73 => 278.0,
                74 => 500.0,
                75 => 667.0,
                76 => 556.0,
                77 => 833.0,
                78 => 722.0,
                79 => 778.0,
                80 => 667.0,
                81 => 778.0,
                82 => 722.0,
                83 => 667.0,
                84 => 611.0,
                85 => 722.0,
                86 => 667.0,
                87 => 944.0,
                88 => 667.0,
                89 => 667.0,
                90 => 611.0,
                97 => 556.0,
                98 => 556.0,
                99 => 500.0,
                100 => 556.0,
                101 => 556.0,
                102 => 278.0,
                103 => 556.0,
                104 => 556.0,
                105 => 222.0,
                106 => 222.0,
                107 => 500.0,
                108 => 222.0,
                109 => 833.0,
                110 => 556.0,
                111 => 556.0,
                112 => 556.0,
                113 => 556.0,
                114 => 333.0,
                115 => 500.0,
                116 => 278.0,
                117 => 556.0,
                118 => 500.0,
                119 => 722.0,
                120 => 500.0,
                121 => 500.0,
                122 => 444.0,
                _ => return None,
            });
        }
        None
    }

    /// Get the width of the space glyph (U+0020) in font units.
    ///
    /// Returns the width in 1000ths of em per PDF spec Section 9.7.4.
    /// Used for font-aware spacing threshold calculations.
    ///
    /// Per PDF Spec Section 9.4.4, word spacing should be based on actual font metrics
    /// rather than fixed ratios. This method returns the actual space glyph width,
    /// which is used to compute adaptive TJ offset thresholds that account for
    /// different font sizes and families.
    ///
    /// # Returns
    ///
    /// The width of the space character (code 0x20) in 1000ths of em,
    /// or the font's default width if the space glyph is not defined.
    pub fn get_space_glyph_width(&self) -> f32 {
        // Space character is always code 0x20 (32) in PDF
        self.get_glyph_width(0x20)
    }

    /// Map a Glyph ID (GID) to a standard PostScript glyph name.
    ///
    /// This is used as a fallback for Type0 fonts without ToUnicode CMaps.
    /// For ASCII range GIDs (32-126), maps to standard PostScript glyph names
    /// that can be looked up in the Adobe Glyph List.
    ///
    /// Phase 1.2: Adobe Glyph List Fallback
    ///
    /// # Arguments
    ///
    /// * `gid` - The Glyph ID to map (typically 0x20-0x7E for ASCII)
    ///
    /// # Returns
    ///
    /// The standard glyph name if GID is in the ASCII range, None otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(FontInfo::gid_to_standard_glyph_name(0x41), Some("A"));
    /// assert_eq!(FontInfo::gid_to_standard_glyph_name(0x20), Some("space"));
    /// assert_eq!(FontInfo::gid_to_standard_glyph_name(0xFFFF), None);
    /// ```
    pub fn gid_to_standard_glyph_name(gid: u16) -> Option<&'static str> {
        // Map GIDs to standard PostScript glyph names across multiple ranges:
        // - ASCII printable range (0x20-0x7E)
        // - Extended Latin / Windows-1252 range (0x80-0xFF)
        // - Latin-1 Supplement range (0xA0-0xFF)
        match gid {
            // Control characters and whitespace (32-33)
            0x20 => Some("space"),
            0x21 => Some("exclam"),
            0x22 => Some("quotedbl"),
            0x23 => Some("numbersign"),
            0x24 => Some("dollar"),
            0x25 => Some("percent"),
            0x26 => Some("ampersand"),
            0x27 => Some("quoteright"),
            0x28 => Some("parenleft"),
            0x29 => Some("parenright"),
            0x2A => Some("asterisk"),
            0x2B => Some("plus"),
            0x2C => Some("comma"),
            0x2D => Some("hyphen"),
            0x2E => Some("period"),
            0x2F => Some("slash"),
            // Digits (48-57)
            0x30 => Some("zero"),
            0x31 => Some("one"),
            0x32 => Some("two"),
            0x33 => Some("three"),
            0x34 => Some("four"),
            0x35 => Some("five"),
            0x36 => Some("six"),
            0x37 => Some("seven"),
            0x38 => Some("eight"),
            0x39 => Some("nine"),
            // Punctuation (58-64)
            0x3A => Some("colon"),
            0x3B => Some("semicolon"),
            0x3C => Some("less"),
            0x3D => Some("equal"),
            0x3E => Some("greater"),
            0x3F => Some("question"),
            0x40 => Some("at"),
            // Uppercase letters (65-90)
            0x41 => Some("A"),
            0x42 => Some("B"),
            0x43 => Some("C"),
            0x44 => Some("D"),
            0x45 => Some("E"),
            0x46 => Some("F"),
            0x47 => Some("G"),
            0x48 => Some("H"),
            0x49 => Some("I"),
            0x4A => Some("J"),
            0x4B => Some("K"),
            0x4C => Some("L"),
            0x4D => Some("M"),
            0x4E => Some("N"),
            0x4F => Some("O"),
            0x50 => Some("P"),
            0x51 => Some("Q"),
            0x52 => Some("R"),
            0x53 => Some("S"),
            0x54 => Some("T"),
            0x55 => Some("U"),
            0x56 => Some("V"),
            0x57 => Some("W"),
            0x58 => Some("X"),
            0x59 => Some("Y"),
            0x5A => Some("Z"),
            // Brackets (91-96)
            0x5B => Some("bracketleft"),
            0x5C => Some("backslash"),
            0x5D => Some("bracketright"),
            0x5E => Some("asciicircum"),
            0x5F => Some("underscore"),
            0x60 => Some("quoteleft"),
            // Lowercase letters (97-122)
            0x61 => Some("a"),
            0x62 => Some("b"),
            0x63 => Some("c"),
            0x64 => Some("d"),
            0x65 => Some("e"),
            0x66 => Some("f"),
            0x67 => Some("g"),
            0x68 => Some("h"),
            0x69 => Some("i"),
            0x6A => Some("j"),
            0x6B => Some("k"),
            0x6C => Some("l"),
            0x6D => Some("m"),
            0x6E => Some("n"),
            0x6F => Some("o"),
            0x70 => Some("p"),
            0x71 => Some("q"),
            0x72 => Some("r"),
            0x73 => Some("s"),
            0x74 => Some("t"),
            0x75 => Some("u"),
            0x76 => Some("v"),
            0x77 => Some("w"),
            0x78 => Some("x"),
            0x79 => Some("y"),
            0x7A => Some("z"),
            // Braces (123-126)
            0x7B => Some("braceleft"),
            0x7C => Some("bar"),
            0x7D => Some("braceright"),
            0x7E => Some("asciitilde"),

            // ==================================================================================
            // Extended Latin / Windows-1252 range (0x80-0xFF)
            // ==================================================================================
            // These mappings cover the extended ASCII characters commonly found in Western
            // European PDFs. When a Type0 font with Identity CMap encounters these GIDs,
            // we map them to their standard PostScript glyph names for AGL lookup.
            //
            // Per PDF Spec ISO 32000-1:2008 Section 9.10.2, when ToUnicode CMap is unavailable,
            // readers may use glyph name lookup as a fallback mechanism.

            // 0x80-0x8F: Windows-1252 extended control characters and symbols
            0x80 => Some("euro"), // U+20AC (Euro sign)
            // 0x81: undefined in Windows-1252
            0x82 => Some("quotesinglbase"), // U+201A (Single low quotation mark)
            0x83 => Some("florin"),         // U+0192 (Latin small letter f with hook)
            0x84 => Some("quotedblbase"),   // U+201E (Double low quotation mark)
            0x85 => Some("ellipsis"),       // U+2026 (Horizontal ellipsis)
            0x86 => Some("dagger"),         // U+2020 (Dagger)
            0x87 => Some("daggerdbl"),      // U+2021 (Double dagger)
            0x88 => Some("circumflex"),     // U+02C6 (Modifier letter circumflex accent)
            0x89 => Some("perthousand"),    // U+2030 (Per mille sign)
            0x8A => Some("Scaron"),         // U+0160 (Latin capital letter S with caron)
            0x8B => Some("guilsinglleft"),  // U+2039 (Single left-pointing angle quotation mark)
            0x8C => Some("OE"),             // U+0152 (Latin capital ligature OE)
            // 0x8D: undefined in Windows-1252
            0x8E => Some("Zcaron"), // U+017D (Latin capital letter Z with caron)
            // 0x8F: undefined in Windows-1252

            // 0x90-0x9F: Windows-1252 smart quotes, dashes, and accents
            // 0x90: undefined in Windows-1252
            0x91 => Some("quoteleft"), // U+2018 (Left single quotation mark)
            0x92 => Some("quoteright"), // U+2019 (Right single quotation mark)
            0x93 => Some("quotedblleft"), // U+201C (Left double quotation mark)
            0x94 => Some("quotedblright"), // U+201D (Right double quotation mark)
            0x95 => Some("bullet"),    // U+2022 (Bullet)
            0x96 => Some("endash"),    // U+2013 (En dash)
            0x97 => Some("emdash"),    // U+2014 (Em dash)
            0x98 => Some("tilde"),     // U+02DC (Small tilde)
            0x99 => Some("trademark"), // U+2122 (Trade mark sign)
            0x9A => Some("scaron"),    // U+0161 (Latin small letter s with caron)
            0x9B => Some("guilsinglright"), // U+203A (Single right-pointing angle quotation mark)
            0x9C => Some("oe"),        // U+0153 (Latin small ligature oe)
            // 0x9D: undefined in Windows-1252
            0x9E => Some("zcaron"), // U+017E (Latin small letter z with caron)
            0x9F => Some("Ydieresis"), // U+0178 (Latin capital letter Y with diaeresis)

            // 0xA0-0xFF: Latin-1 Supplement (ISO 8859-1)
            // Most of these are direct character mappings (À-ÿ)
            // Implement programmatically for common characters and fallback to glyph name generation
            0xA0 => Some("space"),          // U+00A0 (No-break space)
            0xA1 => Some("exclamdown"),     // U+00A1 (Inverted exclamation mark)
            0xA2 => Some("cent"),           // U+00A2 (Cent sign)
            0xA3 => Some("sterling"),       // U+00A3 (Pound sign)
            0xA4 => Some("currency"),       // U+00A4 (Currency sign)
            0xA5 => Some("yen"),            // U+00A5 (Yen sign)
            0xA6 => Some("brokenbar"),      // U+00A6 (Broken bar)
            0xA7 => Some("section"),        // U+00A7 (Section sign)
            0xA8 => Some("dieresis"),       // U+00A8 (Diaeresis)
            0xA9 => Some("copyright"),      // U+00A9 (Copyright sign)
            0xAA => Some("ordfeminine"),    // U+00AA (Feminine ordinal indicator)
            0xAB => Some("guillemotleft"),  // U+00AB (Left-pointing double angle quotation mark)
            0xAC => Some("logicalnot"),     // U+00AC (Not sign)
            0xAD => Some("uni00AD"),        // U+00AD (Soft hyphen)
            0xAE => Some("registered"),     // U+00AE (Registered sign)
            0xAF => Some("macron"),         // U+00AF (Macron)
            0xB0 => Some("degree"),         // U+00B0 (Degree sign)
            0xB1 => Some("plusminus"),      // U+00B1 (Plus-minus sign)
            0xB2 => Some("twosuperior"),    // U+00B2 (Superscript two)
            0xB3 => Some("threesuperior"),  // U+00B3 (Superscript three)
            0xB4 => Some("acute"),          // U+00B4 (Acute accent)
            0xB5 => Some("mu"),             // U+00B5 (Micro sign)
            0xB6 => Some("paragraph"),      // U+00B6 (Pilcrow)
            0xB7 => Some("middot"),         // U+00B7 (Middle dot)
            0xB8 => Some("cedilla"),        // U+00B8 (Cedilla)
            0xB9 => Some("onesuperior"),    // U+00B9 (Superscript one)
            0xBA => Some("ordmasculine"),   // U+00BA (Masculine ordinal indicator)
            0xBB => Some("guillemotright"), // U+00BB (Right-pointing double angle quotation mark)
            0xBC => Some("onequarter"),     // U+00BC (Vulgar fraction one quarter)
            0xBD => Some("onehalf"),        // U+00BD (Vulgar fraction one half)
            0xBE => Some("threequarters"),  // U+00BE (Vulgar fraction three quarters)
            0xBF => Some("questiondown"),   // U+00BF (Inverted question mark)

            // 0xC0-0xFE: Latin-1 Supplement letters (À-þ)
            // These map directly to their Unicode equivalents via standard PostScript names
            // Format: glyph name is the Unicode character itself (e.g., "Agrave" for U+00C0)
            0xC0 => Some("Agrave"), // U+00C0 (Latin capital letter A with grave)
            0xC1 => Some("Aacute"), // U+00C1 (Latin capital letter A with acute)
            0xC2 => Some("Acircumflex"), // U+00C2 (Latin capital letter A with circumflex)
            0xC3 => Some("Atilde"), // U+00C3 (Latin capital letter A with tilde)
            0xC4 => Some("Adieresis"), // U+00C4 (Latin capital letter A with diaeresis)
            0xC5 => Some("Aring"),  // U+00C5 (Latin capital letter A with ring above)
            0xC6 => Some("AE"),     // U+00C6 (Latin capital letter AE)
            0xC7 => Some("Ccedilla"), // U+00C7 (Latin capital letter C with cedilla)
            0xC8 => Some("Egrave"), // U+00C8 (Latin capital letter E with grave)
            0xC9 => Some("Eacute"), // U+00C9 (Latin capital letter E with acute)
            0xCA => Some("Ecircumflex"), // U+00CA (Latin capital letter E with circumflex)
            0xCB => Some("Edieresis"), // U+00CB (Latin capital letter E with diaeresis)
            0xCC => Some("Igrave"), // U+00CC (Latin capital letter I with grave)
            0xCD => Some("Iacute"), // U+00CD (Latin capital letter I with acute)
            0xCE => Some("Icircumflex"), // U+00CE (Latin capital letter I with circumflex)
            0xCF => Some("Idieresis"), // U+00CF (Latin capital letter I with diaeresis)
            0xD0 => Some("Eth"),    // U+00D0 (Latin capital letter Eth)
            0xD1 => Some("Ntilde"), // U+00D1 (Latin capital letter N with tilde)
            0xD2 => Some("Ograve"), // U+00D2 (Latin capital letter O with grave)
            0xD3 => Some("Oacute"), // U+00D3 (Latin capital letter O with acute)
            0xD4 => Some("Ocircumflex"), // U+00D4 (Latin capital letter O with circumflex)
            0xD5 => Some("Otilde"), // U+00D5 (Latin capital letter O with tilde)
            0xD6 => Some("Odieresis"), // U+00D6 (Latin capital letter O with diaeresis)
            0xD7 => Some("multiply"), // U+00D7 (Multiplication sign)
            0xD8 => Some("Oslash"), // U+00D8 (Latin capital letter O with stroke)
            0xD9 => Some("Ugrave"), // U+00D9 (Latin capital letter U with grave)
            0xDA => Some("Uacute"), // U+00DA (Latin capital letter U with acute)
            0xDB => Some("Ucircumflex"), // U+00DB (Latin capital letter U with circumflex)
            0xDC => Some("Udieresis"), // U+00DC (Latin capital letter U with diaeresis)
            0xDD => Some("Yacute"), // U+00DD (Latin capital letter Y with acute)
            0xDE => Some("Thorn"),  // U+00DE (Latin capital letter Thorn)
            0xDF => Some("germandbls"), // U+00DF (Latin small letter sharp s)
            0xE0 => Some("agrave"), // U+00E0 (Latin small letter a with grave)
            0xE1 => Some("aacute"), // U+00E1 (Latin small letter a with acute)
            0xE2 => Some("acircumflex"), // U+00E2 (Latin small letter a with circumflex)
            0xE3 => Some("atilde"), // U+00E3 (Latin small letter a with tilde)
            0xE4 => Some("adieresis"), // U+00E4 (Latin small letter a with diaeresis)
            0xE5 => Some("aring"),  // U+00E5 (Latin small letter a with ring above)
            0xE6 => Some("ae"),     // U+00E6 (Latin small letter ae)
            0xE7 => Some("ccedilla"), // U+00E7 (Latin small letter c with cedilla)
            0xE8 => Some("egrave"), // U+00E8 (Latin small letter e with grave)
            0xE9 => Some("eacute"), // U+00E9 (Latin small letter e with acute)
            0xEA => Some("ecircumflex"), // U+00EA (Latin small letter e with circumflex)
            0xEB => Some("edieresis"), // U+00EB (Latin small letter e with diaeresis)
            0xEC => Some("igrave"), // U+00EC (Latin small letter i with grave)
            0xED => Some("iacute"), // U+00ED (Latin small letter i with acute)
            0xEE => Some("icircumflex"), // U+00EE (Latin small letter i with circumflex)
            0xEF => Some("idieresis"), // U+00EF (Latin small letter i with diaeresis)
            0xF0 => Some("eth"),    // U+00F0 (Latin small letter eth)
            0xF1 => Some("ntilde"), // U+00F1 (Latin small letter n with tilde)
            0xF2 => Some("ograve"), // U+00F2 (Latin small letter o with grave)
            0xF3 => Some("oacute"), // U+00F3 (Latin small letter o with acute)
            0xF4 => Some("ocircumflex"), // U+00F4 (Latin small letter o with circumflex)
            0xF5 => Some("otilde"), // U+00F5 (Latin small letter o with tilde)
            0xF6 => Some("odieresis"), // U+00F6 (Latin small letter o with diaeresis)
            0xF7 => Some("divide"), // U+00F7 (Division sign)
            0xF8 => Some("oslash"), // U+00F8 (Latin small letter o with stroke)
            0xF9 => Some("ugrave"), // U+00F9 (Latin small letter u with grave)
            0xFA => Some("uacute"), // U+00FA (Latin small letter u with acute)
            0xFB => Some("ucircumflex"), // U+00FB (Latin small letter u with circumflex)
            0xFC => Some("udieresis"), // U+00FC (Latin small letter u with diaeresis)
            0xFD => Some("yacute"), // U+00FD (Latin small letter y with acute)
            0xFE => Some("thorn"),  // U+00FE (Latin small letter thorn)
            0xFF => Some("ydieresis"), // U+00FF (Latin small letter y with diaeresis)

            // All other GIDs not in the supported ranges
            _ => None,
        }
    }

    /// Get the pre-computed byte→char lookup table for OneByte (simple) fonts.
    /// Built lazily on first call by running `char_to_unicode` for all 256 byte values.
    /// Returns a 256-element array: non-'\0' = single printable char, '\0' = needs fallback.
    /// Control chars (except tab/newline/cr), multi-char, and \u{FFFD} are stored as '\0'.
    pub fn get_byte_to_char_table(&self) -> &[char; 256] {
        self.byte_to_char_table.get_or_init(|| {
            let mut tbl = ['\0'; 256];
            for i in 0..=255u8 {
                if let Some(s) = self.char_to_unicode(i as u32) {
                    let mut chars = s.chars();
                    if let Some(c) = chars.next() {
                        if chars.next().is_none()
                            && c != '\u{FFFD}'
                            && (c >= '\x20' || c == '\t' || c == '\n' || c == '\r')
                        {
                            tbl[i as usize] = c;
                        }
                        // Multi-char, replacement, or control char: leave as '\0'
                    }
                }
            }
            tbl
        })
    }

    /// Pre-computed byte→width lookup for simple (non-Type0) fonts.
    /// Returns a 256-entry array where index i = glyph width for byte i.
    /// Eliminates per-byte bounds check and subtraction in advance_position.
    #[inline]
    pub fn get_byte_to_width_table(&self) -> &[f32; 256] {
        self.byte_to_width_table.get_or_init(|| {
            let mut tbl = [self.default_width; 256];
            if let Some(widths) = &self.widths {
                if let Some(first_char) = self.first_char {
                    for (idx, &w) in widths.iter().enumerate() {
                        let code = first_char as usize + idx;
                        if code < 256 {
                            tbl[code] = w;
                        }
                    }
                }
            }
            tbl
        })
    }

    /// Convert a character code to Unicode string.
    ///
    /// This method looks up the character code in the font's encoding tables
    /// (ToUnicode CMap, built-in encoding, or glyph name mappings) and returns
    /// the corresponding Unicode string if found.
    pub fn char_to_unicode(&self, char_code: u32) -> Option<String> {
        // char_code is now u32 to support 4-byte character codes (0x00000000-0xFFFFFFFF)
        // This is backward compatible - u16 values are automatically promoted to u32

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
        //
        // Phase 5.1: With lazy loading, the CMap is parsed on first access here
        if let Some(lazy_cmap) = &self.to_unicode {
            // Get the parsed CMap - this triggers lazy parsing on first access
            if let Some(cmap) = lazy_cmap.get() {
                if let Some(unicode) = cmap.get(&char_code) {
                    // Skip U+FFFD mappings - treat as missing entry
                    if unicode == "\u{FFFD}" {
                        log::warn!(
                            "ToUnicode CMap has U+FFFD for code 0x{:02X} in font '{}' - falling back to Priority 2",
                            char_code,
                            self.base_font
                        );
                        // Fall through to Priority 2 (predefined encodings)
                    } else if unicode
                        .chars()
                        .all(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r')
                        && !unicode.is_empty()
                    {
                        // ToUnicode mapped to control characters (U+0000-U+001F excluding
                        // whitespace). This is a broken CMap — fall through to glyph name
                        // resolution or other fallbacks.
                        log::warn!(
                            "ToUnicode CMap maps code 0x{:04X} to control char(s) in font '{}' - falling back",
                            char_code,
                            self.base_font
                        );
                        // Fall through to Priority 2
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
                // Lazy CMap parsing failed
                log::warn!(
                    "Failed to parse lazy CMap for font '{}' - will fall back to Priority 2",
                    self.base_font
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
        // PRIORITY 2: Predefined CMaps (PDF Spec Section 9.7.5.2)
        // ==================================================================================
        // Phase 3.1: Identity-H/Identity-V Predefined CMap Support
        //
        // For CID-keyed fonts (Type0 subtype), predefined CMaps provide character mapping
        // when no ToUnicode CMap is present. This is critical for CJK PDFs using standard
        // Adobe CID collections (Adobe-Identity, Adobe-GB1, Adobe-Japan1, etc.)
        //
        // Identity-H/Identity-V: The simplest predefined CMap
        // - Maps 2-byte CID directly to 2-byte Unicode code point: CID == Unicode
        // - Used with ANY font when encoding is "Identity-H" or "Identity-V"
        // - Per PDF Spec ISO 32000-1:2008 Section 9.7.5.2
        //
        // Examples:
        // - CID 0x4E00 → U+4E00 (CJK UNIFIED IDEOGRAPH "一" in Chinese/Japanese)
        // - CID 0x0041 → U+0041 (Latin Capital Letter A)
        //
        // NOTE: Identity-H/V is actually handled by checking the encoding field.
        // It is checked here for Type0 fonts to ensure it happens before other fallbacks.
        if self.subtype == "Type0" {
            if let Encoding::Standard(ref encoding_name) = self.encoding {
                if encoding_name == "Identity-H"
                    || encoding_name == "Identity-V"
                    || encoding_name.contains("UCS2")
                    || encoding_name.contains("UTF16")
                {
                    // For Identity-H/V: CID value IS the Unicode code point (2-byte)
                    // Valid Unicode range for 2-byte CID: 0x0000 to 0xFFFF
                    // (Standard Unicode BMP - Basic Multilingual Plane)
                    // Since char_code is u16, it's always in range [0x0000, 0xFFFF]
                    //
                    // IMPORTANT: Per PDF Spec 9.10.2, Type0 fonts require either:
                    // 1. A ToUnicode CMap, OR
                    // 2. A predefined CMap (which requires CIDSystemInfo)
                    //
                    // If neither exists, we should NOT treat Identity-H/V as valid for Type0.
                    // This prevents "identity" treatment when there's no proper CIDSystemInfo.
                    if self.cid_system_info.is_some() {
                        // For Adobe-Identity ordering, CIDs are glyph indices (GIDs),
                        // NOT Unicode code points. Try the embedded TrueType cmap first.
                        let is_identity_ordering = self
                            .cid_system_info
                            .as_ref()
                            .map(|info| info.ordering == "Identity")
                            .unwrap_or(false);

                        if is_identity_ordering {
                            // Try TrueType cmap: CID → GID → Unicode
                            if let Some(tt_cmap) = self.truetype_cmap() {
                                let gid = if let Some(ref cid_to_gid) = self.cid_to_gid_map {
                                    cid_to_gid.get_gid(char_code as u16)
                                } else {
                                    char_code as u16
                                };
                                if let Some(unicode_char) = tt_cmap.get_unicode(gid) {
                                    return Some(unicode_char.to_string());
                                }
                            }
                        }

                        // For UCS2/UTF16 encodings, char codes ARE Unicode values directly.
                        // For Identity-H/V with non-Identity ordering (e.g., Adobe-GB1),
                        // char codes are CIDs that need CID-to-Unicode lookup.
                        let is_ucs2_or_utf16 =
                            encoding_name.contains("UCS2") || encoding_name.contains("UTF16");
                        let is_non_identity_ordering = self
                            .cid_system_info
                            .as_ref()
                            .map(|info| info.ordering != "Identity")
                            .unwrap_or(false);

                        if !is_ucs2_or_utf16 && is_non_identity_ordering {
                            // Identity-H/V with CJK collection: CIDs are NOT Unicode!
                            if let Some(unicode_codepoint) = lookup_predefined_cmap(
                                encoding_name,
                                &self.cid_system_info,
                                char_code as u16,
                            ) {
                                if let Some(unicode_char) = char::from_u32(unicode_codepoint) {
                                    return Some(unicode_char.to_string());
                                }
                            }
                            // CID lookup failed — fall through to Priority 2b and beyond
                        } else {
                            // UCS2/UTF16 or Adobe-Identity: char code == Unicode
                            if let Some(unicode_char) = char::from_u32(char_code) {
                                if !unicode_char.is_control() || unicode_char == ' ' {
                                    return Some(unicode_char.to_string());
                                }
                            }
                        }
                    } else {
                        // No CIDSystemInfo — use CID-as-Unicode as last resort.
                        // Many PDF generators assign CID values equal to Unicode code points
                        // even without proper CIDSystemInfo. MuPDF uses this fallback.
                        if let Some(unicode_char) = char::from_u32(char_code) {
                            if !unicode_char.is_control() || unicode_char == ' ' {
                                log::debug!(
                                    "Identity-H/V CID-as-Unicode fallback (no CIDSystemInfo): font='{}' CID=0x{:04X} → '{}' (U+{:04X})",
                                    self.base_font,
                                    char_code,
                                    unicode_char,
                                    unicode_char as u32
                                );
                                return Some(unicode_char.to_string());
                            }
                        }
                        log::debug!(
                            "Type0 font '{}' with {} encoding: CID 0x{:04X} is not a valid Unicode code point",
                            self.base_font,
                            encoding_name,
                            char_code
                        );
                    }
                }
            }
        }

        // ==================================================================================
        // PRIORITY 2a: Shift-JIS (RKSJ) direct decoding
        // ==================================================================================
        // For fonts using 90ms-RKSJ-H/V encoding, the char_code is a Shift-JIS value
        // (after byte grouping in decode_text_to_unicode). Convert directly to Unicode.
        if self.subtype == "Type0" {
            if let Encoding::Standard(ref enc) = self.encoding {
                if enc.contains("RKSJ") {
                    if let Some(unicode_char) = shift_jis_to_unicode(char_code as u16) {
                        return Some(unicode_char.to_string());
                    }
                }
            }
        }

        // ==================================================================================
        // PRIORITY 2b: Unicode-based Predefined CMaps (Phase 3.2)
        // ==================================================================================
        // For Type0 fonts with CIDSystemInfo (Adobe-Japan1, Adobe-GB1, etc.) that don't have
        // ToUnicode CMaps. Use predefined CID-to-Unicode mappings based on the character
        // collection ordering.
        //
        // Per PDF Spec ISO 32000-1:2008 Section 9.7.5.2:
        // "Predefined CMaps can be used for CID-keyed fonts without embedded ToUnicode CMaps"
        //
        // The encoding (Identity, CMap stream, etc.) maps byte codes → CIDs, but
        // CID → Unicode mapping is determined by the CIDSystemInfo ordering alone.
        if self.subtype == "Type0" {
            // Extract encoding name for the lookup (used for specific CMap matching)
            let enc_name = match &self.encoding {
                Encoding::Standard(name) => name.clone(),
                Encoding::Identity => "Identity-H".to_string(),
                Encoding::Custom(_) => String::new(),
            };
            if let Some(unicode_codepoint) =
                lookup_predefined_cmap(&enc_name, &self.cid_system_info, char_code as u16)
            {
                if let Some(unicode_char) = char::from_u32(unicode_codepoint) {
                    log::debug!(
                        "Predefined CMap mapping: CID 0x{:04X} → '{}' (U+{:04X}) in font '{}'",
                        char_code,
                        unicode_char,
                        unicode_codepoint,
                        self.base_font
                    );
                    return Some(unicode_char.to_string());
                }
            }
        }

        // ==================================================================================
        // PRIORITY 1.5: Ligature Expansion (Unicode Ligature Characters)
        // ==================================================================================
        // Check if this character code is a Unicode ligature character (U+FB00-U+FB04).
        // Ligatures should be expanded to their component characters for better text extraction.
        //
        // This is placed early (after ToUnicode but before other fallbacks) because:
        // - Some PDFs may map ligature character codes through ToUnicode CMaps
        // - If no ToUnicode mapping exists, we still want to expand ligatures
        // - Ligature expansion is a Unicode standard (ISO 32000-1:2008 Section 9.10)
        //
        // Ligatures supported:
        // - U+FB00: ff (LATIN SMALL LIGATURE FF)
        // - U+FB01: fi (LATIN SMALL LIGATURE FI)
        // - U+FB02: fl (LATIN SMALL LIGATURE FL)
        // - U+FB03: ffi (LATIN SMALL LIGATURE FFI)
        // - U+FB04: ffl (LATIN SMALL LIGATURE FFL)
        if let Some(expanded) = expand_ligature_char_code(char_code as u16) {
            log::debug!(
                "Ligature expansion: font='{}' code=0x{:04X} → '{}'",
                self.base_font,
                char_code,
                expanded
            );
            return Some(expanded.to_string());
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
                // Check for Identity-H and Identity-V encodings (common for Type0 fonts)
                if name == "Identity-H" || name == "Identity-V" {
                    // NOTE: Type0 fonts with Identity-H/V are handled at Priority 2 (predefined CMaps)
                    // above, so this code path is only reached for simple fonts (Type1, TrueType).
                    // Type0 fonts will have already returned at Priority 2 if the CID is valid Unicode.
                    if self.subtype == "Type0" {
                        // Priority 2 didn't map this CID. Use CID-as-Unicode fallback.
                        if let Some(unicode_char) = char::from_u32(char_code) {
                            if !unicode_char.is_control() || unicode_char == ' ' {
                                log::debug!(
                                    "Type0 font '{}' {} encoding Priority 3 CID-as-Unicode: CID 0x{:04X} → '{}' (U+{:04X})",
                                    self.base_font,
                                    name,
                                    char_code,
                                    unicode_char,
                                    unicode_char as u32
                                );
                                return Some(unicode_char.to_string());
                            }
                        }
                        return Some("\u{FFFD}".to_string());
                    }
                    // For simple fonts, Identity encoding is valid
                    if let Some(ch) = char::from_u32(char_code) {
                        return Some(ch.to_string());
                    }
                }

                // For TrueType subset fonts with no /Encoding, character codes are often
                // GIDs (glyph indices), not standard encoding values. Per PDF Spec 9.6.5.4,
                // when no /Encoding exists and the font has a (3,1) cmap, character codes
                // map through the cmap. Try TrueType cmap first for these fonts.
                if (self.subtype == "TrueType" || self.subtype == "Type1")
                    && name == "StandardEncoding"
                {
                    if let Some(tt_cmap) = self.truetype_cmap() {
                        if let Some(unicode_char) = tt_cmap.get_unicode(char_code as u16) {
                            return Some(unicode_char.to_string());
                        }
                    }
                }

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
                // Check multi_char_map for compound glyph names (e.g., f_f → "ff")
                if let Some(multi_str) = self.multi_char_map.get(&(char_code as u8)) {
                    return Some(multi_str.clone());
                }
            },
            Encoding::Identity => {
                // CRITICAL: Identity encoding assumes char_code == Unicode.
                // This is ONLY valid for simple fonts, NOT Type0/CID fonts.
                // Per PDF Spec ISO 32000-1:2008 Section 9.7.6.3:
                // "Type0 fonts REQUIRE ToUnicode CMaps for proper character mapping"

                if self.subtype == "Type0" {
                    // Type0 fonts: character codes are CID (glyph indices), NOT Unicode
                    // Per PDF Spec ISO 32000-1:2008 Section 9.7.4.2, when no ToUnicode CMap exists,
                    // conforming readers SHALL use the TrueType font's internal "cmap" table as fallback.
                    // This requires translating CID → GID via the CIDToGIDMap, then looking up Unicode.

                    if let Some(tt_cmap) = self.truetype_cmap() {
                        // Translate CID → GID using the CIDToGIDMap
                        // Note: CIDToGIDMap only works with u16 CIDs (2-byte codes)
                        // For CIDs > 0xFFFF, we skip CIDToGIDMap and use char_code as GID if it fits in u16
                        let gid = if char_code <= 0xFFFF {
                            if let Some(ref cid_to_gid) = self.cid_to_gid_map {
                                cid_to_gid.get_gid(char_code as u16)
                            } else {
                                // No explicit mapping - assume Identity (CID == GID)
                                char_code as u16
                            }
                        } else {
                            // Large CID (> 0xFFFF) - cannot use CIDToGIDMap
                            // GIDs are typically u16, so large CIDs won't map correctly
                            log::debug!(
                                "CID 0x{:X} in font '{}' is too large (> 0xFFFF) for CIDToGIDMap - skipping TrueType cmap",
                                char_code,
                                self.base_font
                            );
                            // Return early to skip TrueType cmap lookup for large CIDs
                            return None;
                        };

                        if let Some(unicode_char) = tt_cmap.get_unicode(gid) {
                            log::debug!(
                                "TrueType cmap fallback SUCCESS: font='{}' CID=0x{:04X} (GID={}) → '{}' (U+{:04X})",
                                self.base_font,
                                char_code,
                                gid,
                                unicode_char,
                                unicode_char as u32
                            );
                            return Some(unicode_char.to_string());
                        } else {
                            log::debug!(
                                "TrueType cmap: GID {} not found in font '{}' (CID 0x{:04X} mapped via {})",
                                gid,
                                self.base_font,
                                char_code,
                                if self.cid_to_gid_map.is_some() {
                                    "explicit CIDToGIDMap"
                                } else {
                                    "Identity mapping"
                                }
                            );
                        }
                    }

                    // ==================================================================================
                    // PRIORITY 5: Adobe Glyph List Fallback (Phase 1.2)
                    // ==================================================================================
                    // When TrueType cmap fails (or is not available), try Adobe Glyph List fallback.
                    // This handles Type0 fonts with standard glyph names (e.g., Aptos, LMRoman)
                    // that don't have ToUnicode CMaps or embedded TrueType fonts.
                    //
                    // Process: CID → GID (via CIDToGIDMap) → Glyph Name → Unicode (via AGL)
                    //
                    // IMPORTANT: Only apply AGL fallback if a CIDToGIDMap is explicitly defined
                    // (even if it's Identity). This distinguishes between:
                    // - Type0 fonts with proper CIDToGIDMap (may have standard glyphs)
                    // - Malformed Type0 fonts without CIDToGIDMap (unlikely to work)
                    //
                    // Per PDF Spec ISO 32000-1:2008 Section 9.10.2:
                    // "If a ToUnicode CMap is not available, conforming readers may fall back
                    // to predefined encodings and glyph name lookup."

                    if let Some(ref cid_to_gid) = self.cid_to_gid_map {
                        // CIDToGIDMap only works with u16 CIDs (2-byte codes)
                        if char_code > 0xFFFF {
                            log::debug!(
                                "CID 0x{:X} in font '{}' is too large (> 0xFFFF) for CIDToGIDMap AGL fallback - skipping",
                                char_code,
                                self.base_font
                            );
                            // Fall through to continue fallback attempts
                        } else {
                            let gid = cid_to_gid.get_gid(char_code as u16);

                            if let Some(glyph_name) = Self::gid_to_standard_glyph_name(gid) {
                                if let Some(&unicode_char) = ADOBE_GLYPH_LIST.get(glyph_name) {
                                    log::debug!(
                                        "Adobe Glyph List fallback SUCCESS: font='{}' CID=0x{:04X} (GID={}) → glyph '{}' → '{}' (U+{:04X})",
                                        self.base_font,
                                        char_code,
                                        gid,
                                        glyph_name,
                                        unicode_char,
                                        unicode_char as u32
                                    );
                                    return Some(unicode_char.to_string());
                                }
                            }
                        }
                    }

                    // All standard fallbacks exhausted (no TrueType cmap, no Adobe Glyph List match).
                    // Use CID-as-Unicode fallback: many PDF generators assign CID values equal
                    // to Unicode code points. This matches MuPDF behavior.
                    if let Some(unicode_char) = char::from_u32(char_code) {
                        if !unicode_char.is_control() || unicode_char == ' ' {
                            log::debug!(
                                "Type0 font '{}' Identity encoding CID-as-Unicode fallback: CID 0x{:04X} → '{}' (U+{:04X})",
                                self.base_font,
                                char_code,
                                unicode_char,
                                unicode_char as u32
                            );
                            return Some(unicode_char.to_string());
                        }
                    }
                    log::warn!(
                        "Type0 font '{}' using Identity encoding: CID 0x{:04X} could not be mapped to Unicode. \
                         Embedded font: {} bytes.",
                        self.base_font,
                        char_code,
                        self.embedded_font_data
                            .as_ref()
                            .map(|d| d.len())
                            .unwrap_or(0)
                    );
                    return Some("\u{FFFD}".to_string());
                }

                // For simple fonts (Type1, TrueType), Identity encoding MAY be valid
                if let Some(ch) = char::from_u32(char_code) {
                    log::debug!(
                        "Identity encoding (simple font '{}'): code 0x{:02X} → '{}' (U+{:04X})",
                        self.base_font,
                        char_code,
                        ch,
                        ch as u32
                    );
                    return Some(ch.to_string());
                }
            },
        }

        // ==================================================================================
        // PRIORITY 4: TrueType cmap fallback for simple fonts
        // ==================================================================================
        // When all encoding-based lookups fail, try the embedded TrueType cmap as a last
        // resort. For subset fonts, character codes may be GIDs that the encoding table
        // doesn't cover. The cmap provides GID → Unicode mapping.
        if self.subtype != "Type0" {
            if let Some(tt_cmap) = self.truetype_cmap() {
                if let Some(unicode_char) = tt_cmap.get_unicode(char_code as u16) {
                    return Some(unicode_char.to_string());
                }
            }
        }

        // ==================================================================================
        // PRIORITY 5: Fallback - No Mapping Found
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

    /// Get character from encoding (custom or standard).
    ///
    /// Custom encoding support
    ///
    /// This method normalizes a raw character code through the font's encoding,
    /// converting it to the actual Unicode character. This ensures word boundary
    /// detection works on real characters, not raw byte codes.
    ///
    /// Per PDF Spec ISO 32000-1:2008, Section 9.6.6:
    /// - Custom encodings with /Differences override standard encodings
    /// - Standard encodings have well-defined mappings
    /// - Identity encoding passes codes through as-is
    ///
    /// # Arguments
    ///
    /// * `code` - The raw byte value from the PDF content stream
    ///
    /// # Returns
    ///
    /// The normalized Unicode character, or None if no mapping exists
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::fonts::FontInfo;
    ///
    /// let font_info = /* ... load font ... */;
    /// if let Some(ch) = font_info.get_encoded_char(0x64) {
    ///     println!("Code 0x64 maps to: {}", ch);
    /// }
    /// ```
    pub fn get_encoded_char(&self, code: u8) -> Option<char> {
        match &self.encoding {
            Encoding::Custom(mappings) => {
                // Custom encoding: use explicit character mappings
                mappings.get(&code).copied()
            },
            Encoding::Standard(_encoding_name) => {
                // Standard encoding: for now, assume ToUnicode CMap handles this
                // If we need explicit standard encoding tables, add them here
                // For basic ASCII range, we can pass through
                if code < 128 {
                    Some(code as char)
                } else {
                    None
                }
            },
            Encoding::Identity => {
                // Identity encoding: code == Unicode (for CID fonts)
                // For single-byte codes, treat as Unicode
                if code < 128 {
                    Some(code as char)
                } else {
                    None
                }
            },
        }
    }

    /// Check if font has custom encoding.
    ///
    /// Custom encoding support
    ///
    /// Returns true if the font uses a custom encoding with /Differences array,
    /// which overrides standard encoding for specific character codes.
    ///
    /// # Returns
    ///
    /// true if the font has a custom encoding, false otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::fonts::FontInfo;
    ///
    /// let font_info = /* ... load font ... */;
    /// if font_info.has_custom_encoding() {
    ///     println!("Font uses custom encoding");
    /// }
    /// ```
    pub fn has_custom_encoding(&self) -> bool {
        matches!(self.encoding, Encoding::Custom(_))
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
///
/// Extended glyph names from TeX/math fonts (MSAM, MSBM, Computer Modern, etc.)
/// not present in the standard Adobe Glyph List.
static TEX_MATH_GLYPH_NAMES: phf::Map<&'static str, char> = phf::phf_map! {
    // AMS math symbols (MSAM10, MSBM10)
    "square" => '\u{25A1}',           // WHITE SQUARE
    "squaredot" => '\u{22A1}',        // SQUARED DOT OPERATOR
    "blacksquare" => '\u{25A0}',      // BLACK SQUARE
    "dblarrowup" => '\u{21C8}',       // UPWARDS PAIRED ARROWS
    "dblarrowdwn" => '\u{21CA}',      // DOWNWARDS PAIRED ARROWS
    "dblarrowleft" => '\u{21C7}',     // LEFTWARDS PAIRED ARROWS
    "dblarrowright" => '\u{21C9}',    // RIGHTWARDS PAIRED ARROWS
    "triangle" => '\u{25B3}',         // WHITE UP-POINTING TRIANGLE
    "triangledown" => '\u{25BD}',     // WHITE DOWN-POINTING TRIANGLE
    "triangleleft" => '\u{25C1}',     // WHITE LEFT-POINTING TRIANGLE
    "triangleright" => '\u{25B7}',    // WHITE RIGHT-POINTING TRIANGLE
    "blacktriangle" => '\u{25B2}',    // BLACK UP-POINTING TRIANGLE
    "blacktriangledown" => '\u{25BC}',// BLACK DOWN-POINTING TRIANGLE
    "blacktriangleleft" => '\u{25C0}',// BLACK LEFT-POINTING TRIANGLE
    "blacktriangleright" => '\u{25B6}',// BLACK RIGHT-POINTING TRIANGLE
    "diamond" => '\u{25C7}',          // WHITE DIAMOND
    "blackdiamond" => '\u{25C6}',     // BLACK DIAMOND
    "circle" => '\u{25CB}',           // WHITE CIRCLE
    "bullet1" => '\u{2219}',          // BULLET OPERATOR
    "star" => '\u{22C6}',             // STAR OPERATOR
    "bigstar" => '\u{2605}',          // BLACK STAR
    "checkmark" => '\u{2713}',        // CHECK MARK
    "maltese" => '\u{2720}',          // MALTESE CROSS
    // TeX arrows
    "arrowleft" => '\u{2190}',        // LEFTWARDS ARROW
    "arrowright" => '\u{2192}',       // RIGHTWARDS ARROW
    "arrowup" => '\u{2191}',          // UPWARDS ARROW
    "arrowdown" => '\u{2193}',        // DOWNWARDS ARROW
    "arrowboth" => '\u{2194}',        // LEFT RIGHT ARROW
    "arrowdblup" => '\u{21D1}',       // UPWARDS DOUBLE ARROW
    "arrowdbldown" => '\u{21D3}',     // DOWNWARDS DOUBLE ARROW
    "arrowdblleft" => '\u{21D0}',     // LEFTWARDS DOUBLE ARROW
    "arrowdblright" => '\u{21D2}',    // RIGHTWARDS DOUBLE ARROW
    "arrowdblboth" => '\u{21D4}',     // LEFT RIGHT DOUBLE ARROW
    // TeX math operators
    "langle" => '\u{27E8}',           // MATHEMATICAL LEFT ANGLE BRACKET
    "rangle" => '\u{27E9}',           // MATHEMATICAL RIGHT ANGLE BRACKET
    "lfloor" => '\u{230A}',           // LEFT FLOOR
    "rfloor" => '\u{230B}',           // RIGHT FLOOR
    "lceil" => '\u{2308}',            // LEFT CEILING
    "rceil" => '\u{2309}',            // RIGHT CEILING
    "emptyset" => '\u{2205}',         // EMPTY SET
    "infty" => '\u{221E}',            // INFINITY (alias)
    "nabla" => '\u{2207}',            // NABLA
    "partial" => '\u{2202}',          // PARTIAL DIFFERENTIAL
    "forall" => '\u{2200}',           // FOR ALL
    "exists" => '\u{2203}',           // THERE EXISTS
    "neg" => '\u{00AC}',              // NOT SIGN
    "backslash" => '\u{005C}',        // REVERSE SOLIDUS
    "prime" => '\u{2032}',            // PRIME
    "natural" => '\u{266E}',          // MUSIC NATURAL SIGN
    "flat" => '\u{266D}',             // MUSIC FLAT SIGN
    "sharp" => '\u{266F}',            // MUSIC SHARP SIGN
};

/// Convert a Shift-JIS encoded byte sequence (1 or 2 bytes) to a Unicode character.
/// Uses the encoding_rs crate for correct, complete Shift-JIS decoding.
fn shift_jis_to_unicode(code: u16) -> Option<char> {
    let bytes = if code <= 0xFF {
        vec![code as u8]
    } else {
        vec![(code >> 8) as u8, (code & 0xFF) as u8]
    };
    let (decoded, _, had_errors) = encoding_rs::SHIFT_JIS.decode(&bytes);
    if had_errors {
        return None;
    }
    let mut chars = decoded.chars();
    let c = chars.next()?;
    // Ensure only one character was produced
    if chars.next().is_some() {
        return None;
    }
    Some(c)
}

pub(crate) fn glyph_name_to_unicode(glyph_name: &str) -> Option<char> {
    // Priority 1: Adobe Glyph List (AGL) lookup - O(1) with perfect hash
    // PDF Spec: ISO 32000-1:2008, Section 9.10.2
    if let Some(&unicode_char) = super::adobe_glyph_list::ADOBE_GLYPH_LIST.get(glyph_name) {
        return Some(unicode_char);
    }

    // Priority 1b: Extended glyph names from TeX/math fonts (MSAM, MSBM, etc.)
    // These are well-known glyph names not in the standard AGL but common in
    // academic/mathematical PDFs generated by TeX/LaTeX.
    if let Some(&unicode_char) = TEX_MATH_GLYPH_NAMES.get(glyph_name) {
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

    // Priority 4: Underscore-delimited compound glyph names (AGL spec section 2)
    // e.g. "f_f" → 'f'+'f', "f_i" → 'f'+'i', "T_h" → 'T'+'h'
    // Return the first component character for single-char return type
    if glyph_name.contains('_') {
        let parts: Vec<&str> = glyph_name.split('_').collect();
        if let Some(first) = parts.first() {
            if let Some(&ch) = super::adobe_glyph_list::ADOBE_GLYPH_LIST.get(*first) {
                return Some(ch);
            }
        }
    }

    // Unknown glyph name - not in AGL and not a recognized format
    log::debug!("Unknown glyph name not in Adobe Glyph List: '{}'", glyph_name);
    None
}

/// Resolve a glyph name to a Unicode string, handling compound names.
///
/// Like `glyph_name_to_unicode` but returns a full String for compound glyph names
/// (underscore-delimited per AGL spec, e.g. "f_f" → "ff", "f_f_i" → "ffi").
pub(crate) fn glyph_name_to_unicode_string(glyph_name: &str) -> Option<String> {
    // Try single char lookup first
    if let Some(ch) = glyph_name_to_unicode(glyph_name) {
        return Some(ch.to_string());
    }

    // Handle underscore-delimited compound names (AGL spec section 2)
    if glyph_name.contains('_') {
        let mut result = String::new();
        for part in glyph_name.split('_') {
            if let Some(ch) = glyph_name_to_unicode(part) {
                result.push(ch);
            } else {
                return None; // If any component is unknown, fail entirely
            }
        }
        if !result.is_empty() {
            return Some(result);
        }
    }

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
        'ﬀ' |  // ff  - U+FB00
        'ﬁ' |  // fi  - U+FB01
        'ﬂ' |  // fl  - U+FB02
        'ﬃ' |  // ffi - U+FB03
        'ﬄ' |  // ffl - U+FB04
        'ﬅ' |  // st (long s + t) - U+FB05
        'ﬆ' // st - U+FB06
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
        'ﬀ' => Some("ff"),  // U+FB00
        'ﬁ' => Some("fi"),  // U+FB01
        'ﬂ' => Some("fl"),  // U+FB02
        'ﬃ' => Some("ffi"), // U+FB03
        'ﬄ' => Some("ffl"), // U+FB04
        'ﬅ' => Some("st"),  // U+FB05 (long s + t)
        'ﬆ' => Some("st"),  // U+FB06
        _ => None,
    }
}

/// Expand a Unicode ligature character code to its ASCII equivalent.
///
/// This function handles the Unicode ligature character codes (U+FB00 to U+FB04)
/// and expands them to their multi-character ASCII equivalents.
///
/// This is the u16 character code variant, used in the character mapping priority chain
/// where character codes come as u16 values directly from the PDF.
///
/// # Arguments
///
/// * `char_code` - The character code (as u16) to potentially expand
///
/// # Returns
///
/// The expanded string if `char_code` is a ligature, None otherwise.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(FontInfo::expand_ligature_char_code(0xFB01), Some("fi"));
/// assert_eq!(FontInfo::expand_ligature_char_code(0xFB02), Some("fl"));
/// assert_eq!(FontInfo::expand_ligature_char_code(0x0041), None); // 'A'
/// ```
fn expand_ligature_char_code(char_code: u16) -> Option<&'static str> {
    match char_code {
        0xFB00 => Some("ff"),  // LATIN SMALL LIGATURE FF
        0xFB01 => Some("fi"),  // LATIN SMALL LIGATURE FI
        0xFB02 => Some("fl"),  // LATIN SMALL LIGATURE FL
        0xFB03 => Some("ffi"), // LATIN SMALL LIGATURE FFI
        0xFB04 => Some("ffl"), // LATIN SMALL LIGATURE FFL
        0xFB05 => Some("st"),  // LATIN SMALL LIGATURE LONG S T
        0xFB06 => Some("st"),  // LATIN SMALL LIGATURE ST
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
            // PostScript StandardEncoding per PDF Spec ISO 32000-1:2008, Annex D, Table D.1
            // NOTE: StandardEncoding differs significantly from ISO-8859-1 in the 0xA0-0xFF range.
            // Using ISO-8859-1 fallback here would produce wrong characters for ligatures,
            // smart quotes, accents, and other typographic characters.
            if (32..=126).contains(&code) {
                Some((code as char).to_string())
            } else {
                let unicode = match code {
                    // 0xA0-0xAF
                    0xA1 => '\u{00A1}', // exclamdown
                    0xA2 => '\u{00A2}', // cent
                    0xA3 => '\u{00A3}', // sterling
                    0xA4 => '\u{2044}', // fraction (NOT currency ¤)
                    0xA5 => '\u{00A5}', // yen
                    0xA6 => '\u{0192}', // florin (NOT broken bar)
                    0xA7 => '\u{00A7}', // section
                    0xA8 => '\u{00A4}', // currency (NOT dieresis)
                    0xA9 => '\u{0027}', // quotesingle (NOT copyright)
                    0xAA => '\u{201C}', // quotedblleft (NOT ordfeminine)
                    0xAB => '\u{00AB}', // guillemotleft
                    0xAC => '\u{2039}', // guilsinglleft (NOT not-sign)
                    0xAD => '\u{203A}', // guilsinglright (NOT soft-hyphen)
                    0xAE => '\u{FB01}', // fi ligature (NOT registered)
                    0xAF => '\u{FB02}', // fl ligature (NOT macron)
                    // 0xB0-0xBF
                    0xB1 => '\u{2013}', // endash (NOT plus-minus)
                    0xB2 => '\u{2020}', // dagger (NOT superscript 2)
                    0xB3 => '\u{2021}', // daggerdbl (NOT superscript 3)
                    0xB4 => '\u{00B7}', // periodcentered (NOT acute accent)
                    0xB6 => '\u{00B6}', // paragraph
                    0xB7 => '\u{2022}', // bullet (NOT middle dot)
                    0xB8 => '\u{201A}', // quotesinglbase (NOT cedilla)
                    0xB9 => '\u{201E}', // quotedblbase (NOT superscript 1)
                    0xBA => '\u{201D}', // quotedblright (NOT ordmasculine)
                    0xBB => '\u{00BB}', // guillemotright
                    0xBC => '\u{2026}', // ellipsis (NOT one quarter)
                    0xBD => '\u{2030}', // perthousand (NOT one half)
                    0xBF => '\u{00BF}', // questiondown
                    // 0xC0-0xCF — accent marks and modifiers
                    0xC1 => '\u{0060}', // grave (NOT A-grave)
                    0xC2 => '\u{00B4}', // acute (NOT A-circumflex)
                    0xC3 => '\u{02C6}', // circumflex (NOT A-tilde)
                    0xC4 => '\u{02DC}', // tilde (NOT A-dieresis)
                    0xC5 => '\u{00AF}', // macron (NOT A-ring)
                    0xC6 => '\u{02D8}', // breve (NOT AE)
                    0xC7 => '\u{02D9}', // dotaccent (NOT C-cedilla)
                    0xC8 => '\u{00A8}', // dieresis (NOT E-grave)
                    0xCA => '\u{02DA}', // ring (NOT E-circumflex)
                    0xCB => '\u{00B8}', // cedilla (NOT E-dieresis)
                    0xCD => '\u{02DD}', // hungarumlaut (NOT I-acute)
                    0xCE => '\u{02DB}', // ogonek (NOT I-circumflex)
                    0xCF => '\u{02C7}', // caron (NOT I-dieresis)
                    // 0xD0 — em dash
                    0xD0 => '\u{2014}', // emdash (NOT Eth)
                    // 0xE0-0xEF — uppercase special chars
                    0xE1 => '\u{00C6}', // AE (NOT a-acute)
                    0xE3 => '\u{00AA}', // ordfeminine (NOT a-tilde)
                    0xE8 => '\u{0141}', // Lslash (NOT e-grave)
                    0xE9 => '\u{00D8}', // Oslash (NOT e-acute)
                    0xEA => '\u{0152}', // OE (NOT e-circumflex)
                    0xEB => '\u{00BA}', // ordmasculine (NOT e-dieresis)
                    // 0xF0-0xFF — lowercase special chars
                    0xF1 => '\u{00E6}', // ae (NOT n-tilde)
                    0xF5 => '\u{0131}', // dotlessi (NOT o-tilde)
                    0xF8 => '\u{0142}', // lslash (NOT o-stroke)
                    0xF9 => '\u{00F8}', // oslash (NOT u-grave)
                    0xFA => '\u{0153}', // oe (NOT u-acute)
                    0xFB => '\u{00DF}', // germandbls (NOT u-circumflex)
                    _ => return None,
                };
                Some(unicode.to_string())
            }
        },
        "MacRomanEncoding" => {
            // ASCII range is the same
            if (32..=126).contains(&code) {
                Some((code as char).to_string())
            } else {
                // Complete Mac OS Roman encoding per PDF Spec ISO 32000-1:2008, Annex D, Table D.2
                let unicode = match code {
                    // 0x80-0x9F: Accented letters
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
                    // 0xA0-0xBF: Symbols and punctuation (NOT Latin-1!)
                    0xA0 => '\u{2020}', // dagger (NOT NBSP)
                    0xA1 => '\u{00B0}', // degree (NOT inverted exclamation)
                    0xA2 => '\u{00A2}', // cent
                    0xA3 => '\u{00A3}', // sterling
                    0xA4 => '\u{00A7}', // section (NOT currency sign)
                    0xA5 => '\u{2022}', // bullet (NOT yen)
                    0xA6 => '\u{00B6}', // paragraph (NOT broken bar)
                    0xA7 => '\u{00DF}', // germandbls (NOT section)
                    0xA8 => '\u{00AE}', // registered (NOT dieresis)
                    0xA9 => '\u{00A9}', // copyright
                    0xAA => '\u{2122}', // trademark (NOT ordfeminine)
                    0xAB => '\u{00B4}', // acute (NOT guillemotleft)
                    0xAC => '\u{00A8}', // dieresis (NOT logical not)
                    0xAD => '\u{2260}', // notequal (NOT soft hyphen)
                    0xAE => '\u{00C6}', // AE (NOT registered)
                    0xAF => '\u{00D8}', // Oslash (NOT macron)
                    0xB0 => '\u{221E}', // infinity (NOT degree)
                    0xB1 => '\u{00B1}', // plusminus
                    0xB2 => '\u{2264}', // lessequal (NOT superscript 2)
                    0xB3 => '\u{2265}', // greaterequal (NOT superscript 3)
                    0xB4 => '\u{00A5}', // yen (NOT acute)
                    0xB5 => '\u{00B5}', // mu
                    0xB6 => '\u{2202}', // partialdiff (NOT paragraph)
                    0xB7 => '\u{2211}', // summation (NOT middle dot)
                    0xB8 => '\u{220F}', // product (NOT cedilla)
                    0xB9 => '\u{03C0}', // pi (NOT superscript 1)
                    0xBA => '\u{222B}', // integral (NOT ordmasculine)
                    0xBB => '\u{00AA}', // ordfeminine (NOT guillemotright)
                    0xBC => '\u{00BA}', // ordmasculine (NOT one quarter)
                    0xBD => '\u{2126}', // Omega (NOT one half)
                    0xBE => '\u{00E6}', // ae (NOT three quarters)
                    0xBF => '\u{00F8}', // oslash (NOT inverted question)
                    // 0xC0-0xCF: More symbols and accented capitals
                    0xC0 => '\u{00BF}', // questiondown
                    0xC1 => '\u{00A1}', // exclamdown
                    0xC2 => '\u{00AC}', // logicalnot
                    0xC3 => '\u{221A}', // radical
                    0xC4 => '\u{0192}', // florin
                    0xC5 => '\u{2248}', // approxequal
                    0xC6 => '\u{2206}', // Delta
                    0xC7 => '\u{00AB}', // guillemotleft
                    0xC8 => '\u{00BB}', // guillemotright
                    0xC9 => '\u{2026}', // ellipsis
                    0xCA => '\u{00A0}', // nonbreakingspace
                    0xCB => '\u{00C0}', // Agrave
                    0xCC => '\u{00C3}', // Atilde
                    0xCD => '\u{00D5}', // Otilde
                    0xCE => '\u{0152}', // OE
                    0xCF => '\u{0153}', // oe
                    // 0xD0-0xDF: Dashes, quotes, ligatures
                    0xD0 => '\u{2013}', // endash
                    0xD1 => '\u{2014}', // emdash
                    0xD2 => '\u{201C}', // quotedblleft
                    0xD3 => '\u{201D}', // quotedblright
                    0xD4 => '\u{2018}', // quoteleft
                    0xD5 => '\u{2019}', // quoteright
                    0xD6 => '\u{00F7}', // divide
                    0xD7 => '\u{25CA}', // lozenge
                    0xD8 => '\u{00FF}', // ydieresis
                    0xD9 => '\u{0178}', // Ydieresis
                    0xDA => '\u{2044}', // fraction
                    0xDB => '\u{20AC}', // Euro
                    0xDC => '\u{2039}', // guilsinglleft
                    0xDD => '\u{203A}', // guilsinglright
                    0xDE => '\u{FB01}', // fi ligature
                    0xDF => '\u{FB02}', // fl ligature
                    // 0xE0-0xEF: More symbols and accented capitals
                    0xE0 => '\u{2021}', // daggerdbl
                    0xE1 => '\u{00B7}', // periodcentered
                    0xE2 => '\u{201A}', // quotesinglbase
                    0xE3 => '\u{201E}', // quotedblbase
                    0xE4 => '\u{2030}', // perthousand
                    0xE5 => '\u{00C2}', // Acircumflex
                    0xE6 => '\u{00CA}', // Ecircumflex
                    0xE7 => '\u{00C1}', // Aacute
                    0xE8 => '\u{00CB}', // Edieresis
                    0xE9 => '\u{00C8}', // Egrave
                    0xEA => '\u{00CD}', // Iacute
                    0xEB => '\u{00CE}', // Icircumflex
                    0xEC => '\u{00CF}', // Idieresis
                    0xED => '\u{00CC}', // Igrave
                    0xEE => '\u{00D3}', // Oacute
                    0xEF => '\u{00D4}', // Ocircumflex
                    // 0xF0-0xFF: More accented and special chars
                    0xF0 => '\u{F8FF}', // Apple logo (private use area)
                    0xF1 => '\u{00D2}', // Ograve
                    0xF2 => '\u{00DA}', // Uacute
                    0xF3 => '\u{00DB}', // Ucircumflex
                    0xF4 => '\u{00D9}', // Ugrave
                    0xF5 => '\u{0131}', // dotlessi
                    0xF6 => '\u{02C6}', // circumflex
                    0xF7 => '\u{02DC}', // tilde
                    0xF8 => '\u{00AF}', // macron
                    0xF9 => '\u{02D8}', // breve
                    0xFA => '\u{02D9}', // dotaccent
                    0xFB => '\u{02DA}', // ring
                    0xFC => '\u{00B8}', // cedilla
                    0xFD => '\u{02DD}', // hungarumlaut
                    0xFE => '\u{02DB}', // ogonek
                    0xFF => '\u{02C7}', // caron
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

/// Lookup Unicode code point for a CID in a predefined Unicode-based CMap.
///
/// Predefined CMaps for CJK fonts map CID values from Adobe character collections to Unicode.
/// Per PDF Spec ISO 32000-1:2008 Section 9.7.5.2.
///
/// # Arguments
///
/// * `cmap_name` - The predefined CMap name (e.g., "UniGB-UCS2-H")
/// * `cid_system_info` - The CIDSystemInfo identifying the character collection
/// * `cid` - The Character ID (CID) to look up
///
/// # Returns
///
/// The corresponding Unicode code point, or None if not found.
///
/// # Predefined CMaps Supported
///
/// - UniGB-UCS2-H: Adobe-GB1 (Simplified Chinese)
/// - UniJIS-UCS2-H: Adobe-Japan1 (Japanese)
/// - UniCNS-UCS2-H: Adobe-CNS1 (Traditional Chinese)
/// - UniKS-UCS2-H: Adobe-Korea1 (Korean)
fn lookup_predefined_cmap(
    cmap_name: &str,
    cid_system_info: &Option<CIDSystemInfo>,
    cid: u16,
) -> Option<u32> {
    // Verify that we have CIDSystemInfo to match against the CMap
    let system_info = cid_system_info.as_ref()?;

    // Route to the appropriate CMap lookup based on name and character collection
    match (cmap_name, system_info.ordering.as_str()) {
        ("UniGB-UCS2-H", "GB1") => lookup_adobe_gb1_to_unicode(cid),
        ("UniJIS-UCS2-H", "Japan1") => lookup_adobe_japan1_to_unicode(cid),
        ("UniCNS-UCS2-H", "CNS1") => lookup_adobe_cns1_to_unicode(cid),
        ("UniKS-UCS2-H", "Korea1") => lookup_adobe_korea1_to_unicode(cid),
        // Fallback: match by CIDSystemInfo ordering alone.
        // Some PDFs use encoding CMaps with custom names (e.g., "Adobe-Japan1-2")
        // that are identity mappings (charcode == CID). The CID→Unicode lookup
        // should still work based on the character collection ordering.
        (_, "GB1") => lookup_adobe_gb1_to_unicode(cid),
        (_, "Japan1") => lookup_adobe_japan1_to_unicode(cid),
        (_, "CNS1") => lookup_adobe_cns1_to_unicode(cid),
        (_, "Korea1") => lookup_adobe_korea1_to_unicode(cid),
        _ => None,
    }
}

/// Map CID from Adobe-GB1 character collection to Unicode.
///
/// Adobe-GB1 contains Simplified Chinese characters from GB 2312 and extensions.
/// Reference: Adobe Technical Note #5079 (Adobe-GB1-4 Character Collection)
fn lookup_adobe_gb1_to_unicode(cid: u16) -> Option<u32> {
    crate::fonts::cid_mappings::lookup_adobe_gb1(cid)
}

/// Map CID from Adobe-Japan1 character collection to Unicode.
///
/// Adobe-Japan1 contains Japanese characters from JIS X 0208, JIS X 0212, etc.
/// Reference: Adobe Technical Note #5078 (Adobe-Japan1-4 Character Collection)
fn lookup_adobe_japan1_to_unicode(cid: u16) -> Option<u32> {
    crate::fonts::cid_mappings::lookup_adobe_japan1(cid)
}

/// Map CID from Adobe-CNS1 character collection to Unicode.
///
/// Adobe-CNS1 contains Traditional Chinese characters from CNS 11643 and extensions.
/// Reference: Adobe Technical Note #5080 (Adobe-CNS1-4 Character Collection)
fn lookup_adobe_cns1_to_unicode(cid: u16) -> Option<u32> {
    crate::fonts::cid_mappings::lookup_adobe_cns1(cid)
}

/// Map CID from Adobe-Korea1 character collection to Unicode.
///
/// Adobe-Korea1 contains Korean characters from KS X 1001 and KS X 1002.
/// Reference: Adobe Technical Note #5093 (Adobe-Korea1-2 Character Collection)
fn lookup_adobe_korea1_to_unicode(cid: u16) -> Option<u32> {
    crate::fonts::cid_mappings::lookup_adobe_korea1(cid)
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };
        assert!(font2.is_italic());
    }

    #[test]
    fn test_char_to_unicode_with_tounicode() {
        // Create a simple CMap with one custom mapping
        let cmap_data = b"beginbfchar\n<0041> <0058>\nendbfchar"; // Map 0x41 to 'X'

        let font = FontInfo {
            base_font: "CustomFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: Some(LazyCMap::new(cmap_data.to_vec())),
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font.char_to_unicode(0x20), Some(" ".to_string()));
    }

    #[test]
    fn test_char_to_unicode_identity() {
        // Test Type0 font WITHOUT ToUnicode - should return U+FFFD per PDF Spec 9.10.2
        let font_type0 = FontInfo {
            base_font: "CIDFont".to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Type0 without ToUnicode should use CID-as-Unicode fallback
        assert_eq!(font_type0.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font_type0.char_to_unicode(0x263A), Some("\u{263A}".to_string()));

        // Test Type1 font WITH Identity encoding - should work correctly
        let font_type1 = FontInfo {
            base_font: "TimesRoman".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Simple fonts (Type1) CAN use Identity encoding for valid Unicode codes
        assert_eq!(font_type1.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font_type1.char_to_unicode(0x263A), Some("☺".to_string()));
    }

    #[test]
    fn test_lookup_predefined_cmap_adobe_gb1() {
        // Test Adobe-GB1 (Simplified Chinese) CMap lookup
        let cid_system_info = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "GB1".to_string(),
            supplement: 2,
        });

        // Test ASCII from CID (CID 34 -> 'A')
        assert_eq!(lookup_predefined_cmap("UniGB-UCS2-H", &cid_system_info, 34), Some(0x41));

        // Test known CJK mapping (CID 4559 -> U+4E2D "中")
        assert_eq!(lookup_predefined_cmap("UniGB-UCS2-H", &cid_system_info, 4559), Some(0x4E2D));

        // Test unknown CID
        assert_eq!(lookup_predefined_cmap("UniGB-UCS2-H", &cid_system_info, 50000), None);

        // Test without CIDSystemInfo (should return None)
        assert_eq!(lookup_predefined_cmap("UniGB-UCS2-H", &None, 34), None);
    }

    #[test]
    fn test_lookup_predefined_cmap_adobe_japan1() {
        // Test Adobe-Japan1 (Japanese) CMap lookup
        let cid_system_info = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Japan1".to_string(),
            supplement: 4,
        });

        // Test ASCII from CID (CID 34 -> 'A')
        assert_eq!(lookup_predefined_cmap("UniJIS-UCS2-H", &cid_system_info, 34), Some(0x41));

        // Test Hiragana from CID (CID 843 -> あ U+3042)
        assert_eq!(lookup_predefined_cmap("UniJIS-UCS2-H", &cid_system_info, 843), Some(0x3042));

        // Test unknown CID
        assert_eq!(lookup_predefined_cmap("UniJIS-UCS2-H", &cid_system_info, 50000), None);
    }

    #[test]
    fn test_lookup_predefined_cmap_adobe_cns1() {
        // Test Adobe-CNS1 (Traditional Chinese) CMap lookup
        let cid_system_info = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "CNS1".to_string(),
            supplement: 3,
        });

        // Test ASCII from CID (CID 34 -> 'A')
        assert_eq!(lookup_predefined_cmap("UniCNS-UCS2-H", &cid_system_info, 34), Some(0x41));

        // Test CJK from CID (CID 595 -> 一 U+4E00)
        assert_eq!(lookup_predefined_cmap("UniCNS-UCS2-H", &cid_system_info, 595), Some(0x4E00));
    }

    #[test]
    fn test_lookup_predefined_cmap_adobe_korea1() {
        // Test Adobe-Korea1 (Korean) CMap lookup
        let cid_system_info = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Korea1".to_string(),
            supplement: 1,
        });

        // Test ASCII from CID (CID 34 -> 'A')
        assert_eq!(lookup_predefined_cmap("UniKS-UCS2-H", &cid_system_info, 34), Some(0x41));

        // Test Hangul from CID (CID 1086 -> 가 U+AC00)
        assert_eq!(lookup_predefined_cmap("UniKS-UCS2-H", &cid_system_info, 1086), Some(0xAC00));
    }

    #[test]
    fn test_lookup_predefined_cmap_wrong_ordering() {
        // Test that lookup fails if CIDSystemInfo ordering doesn't match
        let cid_system_info_wrong = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "WrongOrdering".to_string(),
            supplement: 1,
        });

        // Should return None because ordering doesn't match
        assert_eq!(lookup_predefined_cmap("UniGB-UCS2-H", &cid_system_info_wrong, 34), None);
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
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
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };
        assert_eq!(font_normal.get_font_weight(), FontWeight::Normal);
        assert!(!font_normal.is_bold());
    }

    /// Test CIDToGIDMap Identity mapping
    /// Per PDF Spec ISO 32000-1:2008, Section 9.7.4.2
    #[test]
    fn test_cid_to_gid_identity() {
        let identity_map = CIDToGIDMap::Identity;

        // In identity mapping, CID == GID
        assert_eq!(identity_map.get_gid(0), 0);
        assert_eq!(identity_map.get_gid(100), 100);
        assert_eq!(identity_map.get_gid(0xFFFF), 0xFFFF);
    }

    /// Test CIDToGIDMap Explicit mapping
    /// Verifies that explicit GID arrays are looked up correctly
    #[test]
    fn test_cid_to_gid_explicit() {
        // Create explicit mapping: CID 0→10, CID 1→20, CID 2→30
        let gid_array = vec![10, 20, 30];
        let explicit_map = CIDToGIDMap::Explicit(gid_array);

        assert_eq!(explicit_map.get_gid(0), 10);
        assert_eq!(explicit_map.get_gid(1), 20);
        assert_eq!(explicit_map.get_gid(2), 30);

        // Out of range - falls back to identity
        assert_eq!(explicit_map.get_gid(3), 3);
        assert_eq!(explicit_map.get_gid(100), 100);
    }

    // ==================================================================================
    // Extended Latin AGL Fallback Tests
    // ==================================================================================
    // These tests verify that Type0 fonts with Identity CMap can recover unmapped
    // characters using the Adobe Glyph List fallback for extended Latin characters
    // (0x80-0xFF range).

    #[test]
    fn test_gid_to_glyph_name_ascii_range() {
        // Verify ASCII printable range (0x20-0x7E) is still working
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x20), Some("space"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x41), Some("A"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x61), Some("a"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x30), Some("zero"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x7E), Some("asciitilde"));
    }

    #[test]
    fn test_gid_to_glyph_name_windows1252_symbols() {
        // Test Windows-1252 extended symbols (0x80-0x9F range)
        // These are commonly found in Western European PDFs

        // Currency and special symbols
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x80), Some("euro"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x83), Some("florin"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x85), Some("ellipsis"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x8C), Some("OE"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x9C), Some("oe"));

        // Diacritical marks
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x8A), Some("Scaron"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x9A), Some("scaron"));

        // Smart quotes and dashes
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x91), Some("quoteleft"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x92), Some("quoteright"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x93), Some("quotedblleft"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x94), Some("quotedblright"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x96), Some("endash"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x97), Some("emdash"));
    }

    #[test]
    fn test_gid_to_glyph_name_latin1_supplement() {
        // Test Latin-1 Supplement range (0xA0-0xFF)
        // These cover Western European languages (French, Spanish, German, etc.)

        // Currency and symbols
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xA2), Some("cent"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xA3), Some("sterling"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xA4), Some("currency"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xA5), Some("yen"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xA9), Some("copyright"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xAE), Some("registered"));

        // Math symbols
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xB0), Some("degree"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xB1), Some("plusminus"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xD7), Some("multiply"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xF7), Some("divide"));
    }

    #[test]
    fn test_gid_to_glyph_name_uppercase_accented() {
        // Test uppercase Latin letters with diacritical marks
        // These are essential for French (accented A, E), Spanish (N with tilde), German (Umlaut)
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC0), Some("Agrave"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC1), Some("Aacute"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC2), Some("Acircumflex"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC3), Some("Atilde"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC4), Some("Adieresis"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC5), Some("Aring"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC6), Some("AE"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xC7), Some("Ccedilla"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xD1), Some("Ntilde"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xD6), Some("Odieresis"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xDC), Some("Udieresis"));
    }

    #[test]
    fn test_gid_to_glyph_name_lowercase_accented() {
        // Test lowercase Latin letters with diacritical marks
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE0), Some("agrave"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE1), Some("aacute"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE2), Some("acircumflex"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE3), Some("atilde"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE4), Some("adieresis"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE5), Some("aring"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE6), Some("ae"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xE7), Some("ccedilla"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xF1), Some("ntilde"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xF6), Some("odieresis"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xFC), Some("udieresis"));
    }

    #[test]
    fn test_gid_to_glyph_name_special_characters() {
        // Test ordinal indicators and special characters
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xAA), Some("ordfeminine"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xBA), Some("ordmasculine"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xB2), Some("twosuperior"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xB3), Some("threesuperior"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xB9), Some("onesuperior"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xBC), Some("onequarter"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xBD), Some("onehalf"));
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xBE), Some("threequarters"));
    }

    #[test]
    fn test_gid_to_glyph_name_undefined_codes() {
        // Test that undefined codes in Windows-1252 return None
        // (0x81, 0x8D, 0x8F, 0x90, 0x9D are undefined)
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x81), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x8D), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x8F), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x90), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x9D), None);
    }

    #[test]
    fn test_gid_to_glyph_name_out_of_range() {
        // Test that GIDs outside supported ranges return None
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x100), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x1000), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xFFFF), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x0000), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x0001), None);
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x001F), None);
    }

    #[test]
    fn test_agl_fallback_euro_sign() {
        // Test that CID 0x80 (Euro sign) maps through AGL correctly
        // This is a real-world case: Type0 fonts without ToUnicode often need Euro mapping
        let glyph_name =
            FontInfo::gid_to_standard_glyph_name(0x80).expect("0x80 should map to euro");
        assert_eq!(glyph_name, "euro");

        // Verify the glyph exists in AGL
        assert!(ADOBE_GLYPH_LIST.get(glyph_name).is_some());

        // Verify it maps to the correct Unicode
        if let Some(&unicode_char) = ADOBE_GLYPH_LIST.get(glyph_name) {
            assert_eq!(unicode_char as u32, 0x20AC); // Euro sign U+20AC
        }
    }

    #[test]
    fn test_agl_fallback_extended_latin_coverage() {
        // Test that all common extended Latin characters have AGL mappings
        // This ensures the implementation works end-to-end through AGL lookup
        let test_cases = vec![
            (0x80, "euro", 0x20AC),           // Euro sign
            (0x82, "quotesinglbase", 0x201A), // Single low quote
            (0x83, "florin", 0x0192),         // f with hook
            (0x84, "quotedblbase", 0x201E),   // Double low quote
            (0x85, "ellipsis", 0x2026),       // Ellipsis
            (0xA9, "copyright", 0x00A9),      // Copyright
            (0xAE, "registered", 0x00AE),     // Registered
            (0xB0, "degree", 0x00B0),         // Degree
            (0xC1, "Aacute", 0x00C1),         // A acute
            (0xE1, "aacute", 0x00E1),         // a acute
        ];

        for (gid, expected_glyph, expected_unicode) in test_cases {
            // Step 1: GID -> Glyph name
            let glyph_name = FontInfo::gid_to_standard_glyph_name(gid as u16)
                .unwrap_or_else(|| panic!("GID 0x{:02X} should map to a glyph name", gid));
            assert_eq!(glyph_name, expected_glyph);

            // Step 2: Glyph name -> Unicode (via AGL)
            if let Some(&unicode_char) = ADOBE_GLYPH_LIST.get(glyph_name) {
                assert_eq!(unicode_char as u32, expected_unicode);
            } else {
                panic!("Glyph '{}' should exist in Adobe Glyph List", glyph_name);
            }
        }
    }

    #[test]
    fn test_agl_fallback_priority_integration() {
        // Integration test: Verify AGL fallback would activate for unmapped Type0 CIDs
        // This simulates the Priority 5 fallback in char_to_unicode()
        //
        // Scenario:
        // 1. Type0 font with Identity-H CMap
        // 2. No ToUnicode CMap
        // 3. No TrueType cmap
        // 4. CID 0xC1 (Á - A with acute accent) - common in Spanish/French documents
        //
        // Expected: CID 0xC1 -> GID 0xC1 -> "Aacute" -> U+00C1

        let glyph_name =
            FontInfo::gid_to_standard_glyph_name(0xC1).expect("GID 0xC1 should map to Aacute");
        assert_eq!(glyph_name, "Aacute");

        // Verify AGL has the mapping
        assert!(ADOBE_GLYPH_LIST.get("Aacute").is_some());

        // Verify correct Unicode
        if let Some(&unicode_char) = ADOBE_GLYPH_LIST.get("Aacute") {
            let result = unicode_char.to_string();
            assert_eq!(unicode_char as u32, 0x00C1);
            assert!(!result.is_empty());
        }
    }

    // =============================================================================
    // Type 0 /W Array (CID Width) Tests - PDF Spec 9.7.4.3
    // =============================================================================

    #[test]
    fn test_get_glyph_width_uses_cid_widths() {
        // Test that get_glyph_width properly uses cid_widths for Type0 fonts
        let mut cid_widths = HashMap::new();
        cid_widths.insert(1u16, 500.0f32);
        cid_widths.insert(2u16, 600.0f32);
        cid_widths.insert(3u16, 700.0f32);

        let font = FontInfo {
            base_font: "CIDFont".to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 1000.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: Some(cid_widths),
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Widths from cid_widths
        assert_eq!(font.get_glyph_width(1), 500.0);
        assert_eq!(font.get_glyph_width(2), 600.0);
        assert_eq!(font.get_glyph_width(3), 700.0);

        // CID not in cid_widths should return cid_default_width
        assert_eq!(font.get_glyph_width(100), 1000.0);
    }

    #[test]
    fn test_get_glyph_width_cid_default_width() {
        // Test that cid_default_width is used when CID is not in cid_widths
        let mut cid_widths = HashMap::new();
        cid_widths.insert(1u16, 500.0f32);

        let font = FontInfo {
            base_font: "CIDFont".to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0, // Simple font default
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: Some(cid_widths),
            cid_default_width: 800.0, // CID default width
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // CID 1 has explicit width
        assert_eq!(font.get_glyph_width(1), 500.0);

        // Other CIDs use cid_default_width (not default_width)
        assert_eq!(font.get_glyph_width(2), 800.0);
        assert_eq!(font.get_glyph_width(999), 800.0);
    }

    #[test]
    fn test_get_glyph_width_no_cid_widths_uses_default() {
        // Test that fonts without cid_widths fall back to default_width
        let font = FontInfo {
            base_font: "SimpleFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 600.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // All CIDs use default_width when no cid_widths and no widths array
        assert_eq!(font.get_glyph_width(1), 600.0);
        assert_eq!(font.get_glyph_width(65), 600.0);
    }

    #[test]
    fn test_cid_widths_large_range() {
        // Test CID widths with a large range of values (simulating real CJK fonts)
        let mut cid_widths = HashMap::new();
        // Simulate /W array: [1 100 1000] - CIDs 1-100 all have width 1000
        for cid in 1u16..=100 {
            cid_widths.insert(cid, 1000.0f32);
        }
        // Add some individual widths
        cid_widths.insert(200, 500.0);
        cid_widths.insert(201, 600.0);

        let font = FontInfo {
            base_font: "CJKFont".to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0,
            cid_to_gid_map: None,
            cid_system_info: Some(CIDSystemInfo {
                registry: "Adobe".to_string(),
                ordering: "Japan1".to_string(),
                supplement: 4,
            }),
            cid_font_type: Some("CIDFontType2".to_string()),
            cid_widths: Some(cid_widths),
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Range test
        assert_eq!(font.get_glyph_width(1), 1000.0);
        assert_eq!(font.get_glyph_width(50), 1000.0);
        assert_eq!(font.get_glyph_width(100), 1000.0);

        // Individual widths
        assert_eq!(font.get_glyph_width(200), 500.0);
        assert_eq!(font.get_glyph_width(201), 600.0);

        // Default for unlisted CIDs
        assert_eq!(font.get_glyph_width(300), 1000.0);
    }

    // =========================================================================
    // Helper: create a minimal FontInfo for testing (reduces boilerplate)
    // =========================================================================
    fn make_font(overrides: impl FnOnce(&mut FontInfo)) -> FontInfo {
        let mut f = FontInfo {
            base_font: "TestFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };
        overrides(&mut f);
        f
    }

    // =========================================================================
    // parse_cid_widths — unit tests for the /W array parser
    // =========================================================================

    #[test]
    fn test_parse_cid_widths_array_format() {
        // Format: c [w1 w2 ... wn]
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Integer(10), // start CID
                Object::Array(vec![
                    Object::Integer(500),
                    Object::Integer(600),
                    Object::Integer(700),
                ]),
            ]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        assert_eq!(widths.get(&10), Some(&500.0));
        assert_eq!(widths.get(&11), Some(&600.0));
        assert_eq!(widths.get(&12), Some(&700.0));
        assert_eq!(widths.get(&13), None);
    }

    #[test]
    fn test_parse_cid_widths_range_format() {
        // Format: cfirst clast w
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Integer(100),
                Object::Integer(105),
                Object::Integer(300),
            ]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        for cid in 100..=105 {
            assert_eq!(widths.get(&cid), Some(&300.0), "CID {} should be 300", cid);
        }
        assert_eq!(widths.get(&106), None);
    }

    #[test]
    fn test_parse_cid_widths_mixed_formats() {
        // Mix array-format and range-format in one /W array
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                // Array format
                Object::Integer(1),
                Object::Array(vec![Object::Integer(200), Object::Integer(300)]),
                // Range format
                Object::Integer(50),
                Object::Integer(52),
                Object::Integer(400),
            ]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        assert_eq!(widths.get(&1), Some(&200.0));
        assert_eq!(widths.get(&2), Some(&300.0));
        assert_eq!(widths.get(&50), Some(&400.0));
        assert_eq!(widths.get(&51), Some(&400.0));
        assert_eq!(widths.get(&52), Some(&400.0));
    }

    #[test]
    fn test_parse_cid_widths_real_values() {
        // Widths specified as Real (float) values
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![Object::Integer(5), Object::Array(vec![Object::Real(123.5)])]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        assert_eq!(widths.get(&5), Some(&123.5));
    }

    #[test]
    fn test_parse_cid_widths_empty_array() {
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert("W".to_string(), Object::Array(vec![]));
        assert!(FontInfo::parse_cid_widths(&dict, "Test").is_none());
    }

    #[test]
    fn test_parse_cid_widths_missing_w() {
        let dict: HashMap<String, Object> = HashMap::new();
        assert!(FontInfo::parse_cid_widths(&dict, "Test").is_none());
    }

    #[test]
    fn test_parse_cid_widths_non_integer_start() {
        // First element is not an integer — should skip
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Name("bad".to_string()),
                Object::Integer(10),
                Object::Array(vec![Object::Integer(500)]),
            ]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        assert_eq!(widths.get(&10), Some(&500.0));
    }

    #[test]
    fn test_parse_cid_widths_truncated_range() {
        // Range format with missing width — should just stop
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Integer(10),
                Object::Integer(15),
                // missing width
            ]),
        );
        assert!(FontInfo::parse_cid_widths(&dict, "Test").is_none());
    }

    #[test]
    fn test_parse_cid_widths_unexpected_second_element() {
        // Second element after CID is neither Array nor Integer
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![Object::Integer(10), Object::Name("bad".to_string())]),
        );
        // Should produce empty widths
        assert!(FontInfo::parse_cid_widths(&dict, "Test").is_none());
    }

    #[test]
    fn test_parse_cid_widths_range_with_bad_width() {
        // Range format where the width value is not a number
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Integer(1),
                Object::Integer(3),
                Object::Name("notanumber".to_string()),
            ]),
        );
        // Bad width for range, should skip and produce no widths
        assert!(FontInfo::parse_cid_widths(&dict, "Test").is_none());
    }

    #[test]
    fn test_parse_cid_widths_range_with_real_width() {
        let mut dict: HashMap<String, Object> = HashMap::new();
        dict.insert(
            "W".to_string(),
            Object::Array(vec![
                Object::Integer(10),
                Object::Integer(12),
                Object::Real(750.5),
            ]),
        );
        let widths = FontInfo::parse_cid_widths(&dict, "Test").unwrap();
        assert_eq!(widths.get(&10), Some(&750.5));
        assert_eq!(widths.get(&11), Some(&750.5));
        assert_eq!(widths.get(&12), Some(&750.5));
    }

    // =========================================================================
    // shift_jis_to_unicode
    // =========================================================================

    #[test]
    fn test_shift_jis_single_byte_ascii() {
        // Single-byte ASCII should decode normally
        assert_eq!(shift_jis_to_unicode(0x41), Some('A'));
        assert_eq!(shift_jis_to_unicode(0x20), Some(' '));
    }

    #[test]
    fn test_shift_jis_two_byte_katakana() {
        // 0x8341 is Shift-JIS for katakana "ア" (U+30A2)
        assert_eq!(shift_jis_to_unicode(0x8341), Some('ア'));
    }

    #[test]
    fn test_shift_jis_invalid() {
        // 0xFFFF is not a valid Shift-JIS sequence
        assert_eq!(shift_jis_to_unicode(0xFFFF), None);
    }

    // =========================================================================
    // standard_encoding_lookup — extended coverage
    // =========================================================================

    #[test]
    fn test_standard_encoding_lookup_standard_encoding_ascii() {
        assert_eq!(standard_encoding_lookup("StandardEncoding", b'A'), Some("A".to_string()));
        assert_eq!(standard_encoding_lookup("StandardEncoding", b' '), Some(" ".to_string()));
    }

    #[test]
    fn test_standard_encoding_lookup_standard_encoding_extended() {
        // StandardEncoding 0xAE → fi ligature (U+FB01)
        assert_eq!(
            standard_encoding_lookup("StandardEncoding", 0xAE),
            Some("\u{FB01}".to_string())
        );
        // 0xD0 → emdash (U+2014)
        assert_eq!(
            standard_encoding_lookup("StandardEncoding", 0xD0),
            Some("\u{2014}".to_string())
        );
        // 0xA1 → exclamdown
        assert_eq!(
            standard_encoding_lookup("StandardEncoding", 0xA1),
            Some("\u{00A1}".to_string())
        );
    }

    #[test]
    fn test_standard_encoding_lookup_standard_encoding_unmapped() {
        // 0x00 is in the control range, outside 32..=126
        assert_eq!(standard_encoding_lookup("StandardEncoding", 0x00), None);
        // 0xB0 is not mapped in StandardEncoding
        assert_eq!(standard_encoding_lookup("StandardEncoding", 0xB0), None);
    }

    #[test]
    fn test_standard_encoding_lookup_macroman_ascii() {
        assert_eq!(standard_encoding_lookup("MacRomanEncoding", b'Z'), Some("Z".to_string()));
    }

    #[test]
    fn test_standard_encoding_lookup_macroman_extended() {
        // 0x80 → Adieresis (U+00C4)
        assert_eq!(
            standard_encoding_lookup("MacRomanEncoding", 0x80),
            Some("\u{00C4}".to_string())
        );
        // 0xD0 → endash (U+2013)
        assert_eq!(
            standard_encoding_lookup("MacRomanEncoding", 0xD0),
            Some("\u{2013}".to_string())
        );
        // 0xCA → NBSP (U+00A0)
        assert_eq!(
            standard_encoding_lookup("MacRomanEncoding", 0xCA),
            Some("\u{00A0}".to_string())
        );
        // 0xF0 → Apple logo (private use U+F8FF)
        assert_eq!(
            standard_encoding_lookup("MacRomanEncoding", 0xF0),
            Some("\u{F8FF}".to_string())
        );
    }

    #[test]
    fn test_standard_encoding_lookup_macroman_unmapped() {
        // 0x00 is control range
        assert_eq!(standard_encoding_lookup("MacRomanEncoding", 0x00), None);
    }

    #[test]
    fn test_standard_encoding_lookup_winansi_extended() {
        // 0x80 → Euro sign (U+20AC)
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x80), Some("\u{20AC}".to_string()));
        // 0x96 → En dash (U+2013)
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x96), Some("\u{2013}".to_string()));
        // 0xA0 → NBSP direct ISO-8859-1 mapping
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0xA0), Some("\u{00A0}".to_string()));
    }

    #[test]
    fn test_standard_encoding_lookup_winansi_undefined_holes() {
        // 0x81 is undefined in WinAnsi/Windows-1252
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x81), None);
        // 0x8D is undefined
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x8D), None);
    }

    #[test]
    fn test_standard_encoding_lookup_pdfdoc() {
        // 0x80 → bullet (U+2022)
        assert_eq!(standard_encoding_lookup("PDFDocEncoding", 0x80), Some("\u{2022}".to_string()));
        // 0x84 → emdash (U+2014)
        assert_eq!(standard_encoding_lookup("PDFDocEncoding", 0x84), Some("\u{2014}".to_string()));
        // ASCII range
        assert_eq!(standard_encoding_lookup("PDFDocEncoding", b'B'), Some("B".to_string()));
    }

    #[test]
    fn test_standard_encoding_lookup_unknown_encoding() {
        // Unknown encoding: ASCII passthrough for printable chars
        assert_eq!(standard_encoding_lookup("SomeWeirdEncoding", b'X'), Some("X".to_string()));
        // Non-printable or < 32 → None
        assert_eq!(standard_encoding_lookup("SomeWeirdEncoding", 0x01), None);
        // High byte → None (not ASCII)
        assert_eq!(standard_encoding_lookup("SomeWeirdEncoding", 0x80), None);
    }

    // =========================================================================
    // pdfdoc_encoding_lookup
    // =========================================================================

    #[test]
    fn test_pdfdoc_encoding_ascii_range() {
        assert_eq!(pdfdoc_encoding_lookup(0x00), Some('\0'));
        assert_eq!(pdfdoc_encoding_lookup(0x41), Some('A'));
        assert_eq!(pdfdoc_encoding_lookup(0x7F), Some('\x7F'));
    }

    #[test]
    fn test_pdfdoc_encoding_special_range() {
        assert_eq!(pdfdoc_encoding_lookup(0x80), Some('\u{2022}')); // bullet
        assert_eq!(pdfdoc_encoding_lookup(0x85), Some('\u{2013}')); // endash
        assert_eq!(pdfdoc_encoding_lookup(0x93), Some('\u{FB01}')); // fi ligature
        assert_eq!(pdfdoc_encoding_lookup(0x94), Some('\u{FB02}')); // fl ligature
        assert_eq!(pdfdoc_encoding_lookup(0x92), Some('\u{2122}')); // trademark
    }

    #[test]
    fn test_pdfdoc_encoding_undefined() {
        assert_eq!(pdfdoc_encoding_lookup(0x9F), None);
    }

    #[test]
    fn test_pdfdoc_encoding_latin1_range() {
        assert_eq!(pdfdoc_encoding_lookup(0xA0), Some('\u{00A0}')); // NBSP
        assert_eq!(pdfdoc_encoding_lookup(0xFF), Some('\u{00FF}')); // ydieresis
    }

    // =========================================================================
    // symbol_encoding_lookup — extended coverage
    // =========================================================================

    #[test]
    fn test_symbol_encoding_greek_lowercase() {
        assert_eq!(symbol_encoding_lookup(0x61), Some('α'));
        assert_eq!(symbol_encoding_lookup(0x62), Some('β'));
        assert_eq!(symbol_encoding_lookup(0x67), Some('γ'));
        assert_eq!(symbol_encoding_lookup(0x72), Some('ρ'));
        assert_eq!(symbol_encoding_lookup(0x77), Some('ω'));
    }

    #[test]
    fn test_symbol_encoding_greek_uppercase() {
        assert_eq!(symbol_encoding_lookup(0x44), Some('Δ'));
        assert_eq!(symbol_encoding_lookup(0x53), Some('Σ'));
        assert_eq!(symbol_encoding_lookup(0x57), Some('Ω'));
    }

    #[test]
    fn test_symbol_encoding_math_operators() {
        assert_eq!(symbol_encoding_lookup(0xE1), Some('∑')); // summation
        assert_eq!(symbol_encoding_lookup(0xF2), Some('∫')); // integral
        assert_eq!(symbol_encoding_lookup(0xD6), Some('√')); // radical
        assert_eq!(symbol_encoding_lookup(0xB1), Some('±')); // plusminus
        assert_eq!(symbol_encoding_lookup(0xB9), Some('≠')); // notequal
    }

    #[test]
    fn test_symbol_encoding_digits() {
        // Digits 0x30-0x39 map to themselves
        assert_eq!(symbol_encoding_lookup(0x30), Some('0'));
        assert_eq!(symbol_encoding_lookup(0x39), Some('9'));
    }

    #[test]
    fn test_symbol_encoding_punctuation() {
        assert_eq!(symbol_encoding_lookup(0x20), Some(' '));
        assert_eq!(symbol_encoding_lookup(0x2B), Some('+'));
        assert_eq!(symbol_encoding_lookup(0x2D), Some('−')); // minus (not hyphen)
    }

    #[test]
    fn test_symbol_encoding_unmapped() {
        assert_eq!(symbol_encoding_lookup(0x00), None);
        assert_eq!(symbol_encoding_lookup(0x01), None);
    }

    // =========================================================================
    // zapf_dingbats_encoding_lookup — extended coverage
    // =========================================================================

    #[test]
    fn test_zapf_dingbats_common() {
        assert_eq!(zapf_dingbats_encoding_lookup(0x20), Some(' '));
        assert_eq!(zapf_dingbats_encoding_lookup(0x21), Some('✁'));
        assert_eq!(zapf_dingbats_encoding_lookup(0x33), Some('✓')); // checkmark
        assert_eq!(zapf_dingbats_encoding_lookup(0x34), Some('✔')); // bold checkmark
        assert_eq!(zapf_dingbats_encoding_lookup(0x48), Some('★')); // black star
    }

    #[test]
    fn test_zapf_dingbats_geometric() {
        assert_eq!(zapf_dingbats_encoding_lookup(0x6C), Some('●')); // black circle
        assert_eq!(zapf_dingbats_encoding_lookup(0x6F), Some('■')); // black square
    }

    #[test]
    fn test_zapf_dingbats_unmapped() {
        assert_eq!(zapf_dingbats_encoding_lookup(0x00), None);
        assert_eq!(zapf_dingbats_encoding_lookup(0xFF), None);
    }

    // =========================================================================
    // glyph_name_to_unicode — extended edge cases
    // =========================================================================

    #[test]
    fn test_glyph_name_to_unicode_tex_math() {
        assert_eq!(glyph_name_to_unicode("square"), Some('\u{25A1}'));
        assert_eq!(glyph_name_to_unicode("emptyset"), Some('\u{2205}'));
        assert_eq!(glyph_name_to_unicode("infty"), Some('\u{221E}'));
        assert_eq!(glyph_name_to_unicode("nabla"), Some('\u{2207}'));
        assert_eq!(glyph_name_to_unicode("forall"), Some('\u{2200}'));
        assert_eq!(glyph_name_to_unicode("checkmark"), Some('\u{2713}'));
    }

    #[test]
    fn test_glyph_name_to_unicode_underscore_compound() {
        // "f_f" should return first component 'f' via AGL
        assert_eq!(glyph_name_to_unicode("f_f"), Some('f'));
        // "T_h" should return first component 'T' via AGL
        assert_eq!(glyph_name_to_unicode("T_h"), Some('T'));
    }

    #[test]
    fn test_glyph_name_to_unicode_uni_format_edge_cases() {
        // Too short (not 7 chars total)
        assert_eq!(glyph_name_to_unicode("uni004"), None);
        // Invalid hex
        assert_eq!(glyph_name_to_unicode("uniZZZZ"), None);
    }

    #[test]
    fn test_glyph_name_to_unicode_u_format_long() {
        // u1F600 = grinning face emoji
        assert_eq!(glyph_name_to_unicode("u1F600"), Some('\u{1F600}'));
    }

    // =========================================================================
    // glyph_name_to_unicode_string — compound names
    // =========================================================================

    #[test]
    fn test_glyph_name_to_unicode_string_simple() {
        // Single char should just return it as string
        assert_eq!(glyph_name_to_unicode_string("A"), Some("A".to_string()));
    }

    #[test]
    fn test_glyph_name_to_unicode_string_compound_ff() {
        // glyph_name_to_unicode("f_f") returns Some('f') — first component via AGL
        // So glyph_name_to_unicode_string wraps it as "f" (single-char short-circuit)
        assert_eq!(glyph_name_to_unicode_string("f_f"), Some("f".to_string()));
    }

    #[test]
    fn test_glyph_name_to_unicode_string_compound_all_known() {
        // Use a compound name where each component is known individually.
        // "T_h" → glyph_name_to_unicode finds 'T' (first component) → returns "T"
        assert_eq!(glyph_name_to_unicode_string("T_h"), Some("T".to_string()));
    }

    #[test]
    fn test_glyph_name_to_unicode_string_compound_unknown_part() {
        // "f_unknownglyph" — glyph_name_to_unicode finds 'f' (first component via underscore rule)
        // So it returns Some("f") not None
        assert_eq!(glyph_name_to_unicode_string("f_unknownglyph"), Some("f".to_string()));
    }

    #[test]
    fn test_glyph_name_to_unicode_string_totally_unknown_compound() {
        // Both parts unknown — should return None
        assert_eq!(glyph_name_to_unicode_string("xyzzy_plugh"), None);
    }

    #[test]
    fn test_glyph_name_to_unicode_string_unknown() {
        assert_eq!(glyph_name_to_unicode_string("totallyunknown"), None);
    }

    // =========================================================================
    // is_ligature_char and expand_ligature_char
    // =========================================================================

    #[test]
    fn test_is_ligature_char_all_variants() {
        assert!(is_ligature_char('\u{FB00}')); // ff
        assert!(is_ligature_char('\u{FB01}')); // fi
        assert!(is_ligature_char('\u{FB02}')); // fl
        assert!(is_ligature_char('\u{FB03}')); // ffi
        assert!(is_ligature_char('\u{FB04}')); // ffl
        assert!(is_ligature_char('\u{FB05}')); // long s + t
        assert!(is_ligature_char('\u{FB06}')); // st
        assert!(!is_ligature_char('A'));
        assert!(!is_ligature_char(' '));
    }

    #[test]
    fn test_expand_ligature_char_all_variants() {
        assert_eq!(expand_ligature_char('\u{FB00}'), Some("ff"));
        assert_eq!(expand_ligature_char('\u{FB01}'), Some("fi"));
        assert_eq!(expand_ligature_char('\u{FB02}'), Some("fl"));
        assert_eq!(expand_ligature_char('\u{FB03}'), Some("ffi"));
        assert_eq!(expand_ligature_char('\u{FB04}'), Some("ffl"));
        assert_eq!(expand_ligature_char('\u{FB05}'), Some("st"));
        assert_eq!(expand_ligature_char('\u{FB06}'), Some("st"));
        assert_eq!(expand_ligature_char('x'), None);
    }

    // =========================================================================
    // expand_ligature_char_code
    // =========================================================================

    #[test]
    fn test_expand_ligature_char_code_all_variants() {
        assert_eq!(expand_ligature_char_code(0xFB00), Some("ff"));
        assert_eq!(expand_ligature_char_code(0xFB01), Some("fi"));
        assert_eq!(expand_ligature_char_code(0xFB02), Some("fl"));
        assert_eq!(expand_ligature_char_code(0xFB03), Some("ffi"));
        assert_eq!(expand_ligature_char_code(0xFB04), Some("ffl"));
        assert_eq!(expand_ligature_char_code(0xFB05), Some("st"));
        assert_eq!(expand_ligature_char_code(0xFB06), Some("st"));
        assert_eq!(expand_ligature_char_code(0x0041), None);
        assert_eq!(expand_ligature_char_code(0x0000), None);
    }

    // =========================================================================
    // get_glyph_width — simple font widths array
    // =========================================================================

    #[test]
    fn test_get_glyph_width_simple_font_widths_array() {
        let font = make_font(|f| {
            f.widths = Some(vec![200.0, 300.0, 400.0, 500.0]);
            f.first_char = Some(65); // 'A'
            f.last_char = Some(68); // 'D'
            f.default_width = 600.0;
        });
        assert_eq!(font.get_glyph_width(65), 200.0); // 'A'
        assert_eq!(font.get_glyph_width(66), 300.0); // 'B'
        assert_eq!(font.get_glyph_width(67), 400.0); // 'C'
        assert_eq!(font.get_glyph_width(68), 500.0); // 'D'
                                                     // Out of range → default_width
        assert_eq!(font.get_glyph_width(64), 600.0);
        assert_eq!(font.get_glyph_width(69), 600.0);
    }

    #[test]
    fn test_get_glyph_width_below_first_char() {
        let font = make_font(|f| {
            f.widths = Some(vec![250.0]);
            f.first_char = Some(100);
            f.last_char = Some(100);
            f.default_width = 777.0;
        });
        // char_code < first_char → negative index → default
        assert_eq!(font.get_glyph_width(50), 777.0);
    }

    #[test]
    fn test_get_glyph_width_no_widths_no_cid() {
        let font = make_font(|f| {
            f.default_width = 550.0;
        });
        assert_eq!(font.get_glyph_width(65), 550.0);
    }

    // =========================================================================
    // get_space_glyph_width
    // =========================================================================

    #[test]
    fn test_get_space_glyph_width_from_array() {
        let font = make_font(|f| {
            f.widths = Some(vec![250.0]); // only one entry
            f.first_char = Some(32); // space = 0x20 = 32
            f.last_char = Some(32);
        });
        assert_eq!(font.get_space_glyph_width(), 250.0);
    }

    #[test]
    fn test_get_space_glyph_width_default() {
        let font = make_font(|f| {
            f.default_width = 333.0;
        });
        assert_eq!(font.get_space_glyph_width(), 333.0);
    }

    // =========================================================================
    // is_symbolic — flags and name-based detection
    // =========================================================================

    #[test]
    fn test_is_symbolic_flag_set() {
        let font = make_font(|f| {
            f.flags = Some(0x04); // bit 3 set
        });
        assert!(font.is_symbolic());
    }

    #[test]
    fn test_is_symbolic_flag_not_set() {
        let font = make_font(|f| {
            f.flags = Some(0x20); // nonsymbolic bit only
        });
        assert!(!font.is_symbolic());
    }

    #[test]
    fn test_is_symbolic_no_flags_symbol_name() {
        let font = make_font(|f| {
            f.base_font = "Symbol".to_string();
        });
        assert!(font.is_symbolic());
    }

    #[test]
    fn test_is_symbolic_no_flags_zapf_name() {
        let font = make_font(|f| {
            f.base_font = "ZapfDingbats".to_string();
        });
        assert!(font.is_symbolic());
    }

    #[test]
    fn test_is_symbolic_no_flags_normal_name() {
        let font = make_font(|f| {
            f.base_font = "Helvetica".to_string();
        });
        assert!(!font.is_symbolic());
    }

    // =========================================================================
    // get_encoded_char
    // =========================================================================

    #[test]
    fn test_get_encoded_char_custom() {
        let mut map = HashMap::new();
        map.insert(0x41, 'X');
        map.insert(0x42, 'Y');
        let font = make_font(|f| {
            f.encoding = Encoding::Custom(map);
        });
        assert_eq!(font.get_encoded_char(0x41), Some('X'));
        assert_eq!(font.get_encoded_char(0x42), Some('Y'));
        assert_eq!(font.get_encoded_char(0x43), None);
    }

    #[test]
    fn test_get_encoded_char_standard_ascii() {
        let font = make_font(|f| {
            f.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
        });
        assert_eq!(font.get_encoded_char(0x41), Some('A'));
        assert_eq!(font.get_encoded_char(0x20), Some(' '));
        // High byte → None (>= 128)
        assert_eq!(font.get_encoded_char(0x80), None);
    }

    #[test]
    fn test_get_encoded_char_identity_ascii() {
        let font = make_font(|f| {
            f.encoding = Encoding::Identity;
        });
        assert_eq!(font.get_encoded_char(0x41), Some('A'));
        assert_eq!(font.get_encoded_char(0x80), None);
    }

    // =========================================================================
    // has_custom_encoding
    // =========================================================================

    #[test]
    fn test_has_custom_encoding_true() {
        let font = make_font(|f| {
            f.encoding = Encoding::Custom(HashMap::new());
        });
        assert!(font.has_custom_encoding());
    }

    #[test]
    fn test_has_custom_encoding_false_standard() {
        let font = make_font(|_| {});
        assert!(!font.has_custom_encoding());
    }

    #[test]
    fn test_has_custom_encoding_false_identity() {
        let font = make_font(|f| {
            f.encoding = Encoding::Identity;
        });
        assert!(!font.has_custom_encoding());
    }

    // =========================================================================
    // char_to_unicode — Symbol font path
    // =========================================================================

    #[test]
    fn test_char_to_unicode_symbol_font() {
        let font = make_font(|f| {
            f.base_font = "Symbol".to_string();
            f.flags = Some(0x04); // Symbolic
            f.encoding = Encoding::Standard("SymbolicBuiltIn".to_string());
        });
        // alpha
        assert_eq!(font.char_to_unicode(0x61), Some("α".to_string()));
        // Sigma
        assert_eq!(font.char_to_unicode(0x53), Some("Σ".to_string()));
        // integral
        assert_eq!(font.char_to_unicode(0xF2), Some("∫".to_string()));
    }

    #[test]
    fn test_char_to_unicode_zapfdingbats_font() {
        let font = make_font(|f| {
            f.base_font = "ZapfDingbats".to_string();
            f.flags = Some(0x04); // Symbolic
            f.encoding = Encoding::Standard("SymbolicBuiltIn".to_string());
        });
        // checkmark
        assert_eq!(font.char_to_unicode(0x33), Some("✓".to_string()));
        // star
        assert_eq!(font.char_to_unicode(0x48), Some("★".to_string()));
    }

    // =========================================================================
    // char_to_unicode — ligature expansion path
    // =========================================================================

    #[test]
    fn test_char_to_unicode_ligature_expansion() {
        // Ligature codes come through when no ToUnicode and not in encoding
        let font = make_font(|f| {
            f.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
        });
        // 0xFB01 (fi ligature) should expand to "fi"
        assert_eq!(font.char_to_unicode(0xFB01), Some("fi".to_string()));
        // 0xFB03 (ffi ligature) should expand to "ffi"
        assert_eq!(font.char_to_unicode(0xFB03), Some("ffi".to_string()));
    }

    // =========================================================================
    // char_to_unicode — custom encoding with ligature
    // =========================================================================

    #[test]
    fn test_char_to_unicode_custom_encoding_with_ligature() {
        let mut custom = HashMap::new();
        custom.insert(0x01, '\u{FB01}'); // fi ligature
        let font = make_font(|f| {
            f.encoding = Encoding::Custom(custom);
        });
        // Should expand ligature
        assert_eq!(font.char_to_unicode(0x01), Some("fi".to_string()));
    }

    #[test]
    fn test_char_to_unicode_custom_encoding_multi_char_map() {
        let font = make_font(|f| {
            f.encoding = Encoding::Custom(HashMap::new());
            f.multi_char_map.insert(0x01, "ff".to_string());
        });
        assert_eq!(font.char_to_unicode(0x01), Some("ff".to_string()));
    }

    // =========================================================================
    // char_to_unicode — ToUnicode with U+FFFD (replacement character skip)
    // =========================================================================

    #[test]
    fn test_char_to_unicode_tounicode_fffd_fallback() {
        // A ToUnicode mapping to U+FFFD should be treated as missing
        // and fall back to encoding
        let cmap_data = b"beginbfchar\n<0041> <FFFD>\nendbfchar";
        let font = make_font(|f| {
            f.to_unicode = Some(LazyCMap::new(cmap_data.to_vec()));
            f.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
        });
        // Should fall through FFFD and use WinAnsi → 'A'
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    #[test]
    fn test_char_to_unicode_tounicode_control_char_fallback() {
        // A ToUnicode mapping to control characters should be treated as missing
        let cmap_data = b"beginbfchar\n<0041> <0001>\nendbfchar";
        let font = make_font(|f| {
            f.to_unicode = Some(LazyCMap::new(cmap_data.to_vec()));
            f.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
        });
        // Control char mapping → fallback to WinAnsi → 'A'
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Type0 with Identity-H and CIDSystemInfo
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_h_with_sysinfo() {
        let font = make_font(|f| {
            f.base_font = "CIDFont+F1".to_string();
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("Identity-H".to_string());
            f.cid_system_info = Some(CIDSystemInfo {
                registry: "Adobe".to_string(),
                ordering: "Identity".to_string(),
                supplement: 0,
            });
            f.cid_to_gid_map = Some(CIDToGIDMap::Identity);
        });
        // Adobe-Identity with no TrueType cmap → CID-as-Unicode fallback
        // For non-control Unicode code points, char_code == Unicode
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font.char_to_unicode(0x4E2D), Some("\u{4E2D}".to_string())); // 中
    }

    #[test]
    fn test_char_to_unicode_type0_identity_h_no_sysinfo() {
        let font = make_font(|f| {
            f.base_font = "CIDFont+F2".to_string();
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("Identity-H".to_string());
        });
        // No CIDSystemInfo → CID-as-Unicode last resort
        assert_eq!(font.char_to_unicode(0x42), Some("B".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Type0 with Identity encoding (not Standard)
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_encoding_cid_as_unicode() {
        let font = make_font(|f| {
            f.base_font = "MyCIDFont".to_string();
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Identity;
            // No TrueType cmap, no CIDToGIDMap → CID-as-Unicode fallback
        });
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    #[test]
    fn test_char_to_unicode_type0_identity_encoding_control_char() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Identity;
        });
        // Control char (0x01) should return FFFD because CID-as-Unicode skips controls
        // but the last resort returns FFFD
        let result = font.char_to_unicode(0x01);
        assert_eq!(result, Some("\u{FFFD}".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Identity encoding for simple (non-Type0) fonts
    // =========================================================================

    #[test]
    fn test_char_to_unicode_simple_font_identity() {
        let font = make_font(|f| {
            f.subtype = "Type1".to_string();
            f.encoding = Encoding::Identity;
        });
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        assert_eq!(font.char_to_unicode(0x263A), Some("☺".to_string()));
    }

    // =========================================================================
    // char_to_unicode — TrueType StandardEncoding fallback
    // =========================================================================

    #[test]
    fn test_char_to_unicode_truetype_standard_encoding_ascii() {
        let font = make_font(|f| {
            f.subtype = "TrueType".to_string();
            f.encoding = Encoding::Standard("StandardEncoding".to_string());
        });
        // Should use standard encoding lookup for ASCII
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    // =========================================================================
    // char_to_unicode — MacRomanEncoding
    // =========================================================================

    #[test]
    fn test_char_to_unicode_macroman_extended() {
        let font = make_font(|f| {
            f.encoding = Encoding::Standard("MacRomanEncoding".to_string());
        });
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        // 0x80 → Adieresis (Ä)
        assert_eq!(font.char_to_unicode(0x80), Some("\u{00C4}".to_string()));
    }

    // =========================================================================
    // get_font_weight — DemiBold name heuristic
    // =========================================================================

    #[test]
    fn test_get_font_weight_demibold() {
        let font = make_font(|f| {
            f.base_font = "MyFont-DemiBold".to_string();
        });
        assert_eq!(font.get_font_weight(), FontWeight::SemiBold);
    }

    #[test]
    fn test_get_font_weight_heavy() {
        let font = make_font(|f| {
            f.base_font = "MyFont-Heavy".to_string();
        });
        assert_eq!(font.get_font_weight(), FontWeight::Black);
    }

    #[test]
    fn test_get_font_weight_ultrabold() {
        let font = make_font(|f| {
            f.base_font = "MyFont-UltraBold".to_string();
        });
        assert_eq!(font.get_font_weight(), FontWeight::ExtraBold);
    }

    #[test]
    fn test_get_font_weight_ultralight() {
        let font = make_font(|f| {
            f.base_font = "MyFont-UltraLight".to_string();
        });
        assert_eq!(font.get_font_weight(), FontWeight::ExtraLight);
    }

    // =========================================================================
    // get_byte_to_char_table
    // =========================================================================

    #[test]
    fn test_get_byte_to_char_table_basic() {
        let font = make_font(|f| {
            f.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
        });
        let table = font.get_byte_to_char_table();
        // ASCII 'A' (0x41 = 65) should be 'A'
        assert_eq!(table[0x41], 'A');
        // space (0x20 = 32)
        assert_eq!(table[0x20], ' ');
        // Control chars (except tab/newline/cr) should be '\0'
        assert_eq!(table[0x01], '\0');
    }

    #[test]
    fn test_get_byte_to_char_table_tab_newline_passthrough() {
        let font = make_font(|f| {
            let mut custom = HashMap::new();
            custom.insert(0x09u8, '\t');
            custom.insert(0x0Au8, '\n');
            custom.insert(0x0Du8, '\r');
            f.encoding = Encoding::Custom(custom);
        });
        let table = font.get_byte_to_char_table();
        assert_eq!(table[0x09], '\t');
        assert_eq!(table[0x0A], '\n');
        assert_eq!(table[0x0D], '\r');
    }

    // =========================================================================
    // get_byte_to_width_table
    // =========================================================================

    #[test]
    fn test_get_byte_to_width_table_basic() {
        let font = make_font(|f| {
            f.widths = Some(vec![200.0, 300.0, 400.0]);
            f.first_char = Some(65); // 'A'
            f.default_width = 500.0;
        });
        let table = font.get_byte_to_width_table();
        assert_eq!(table[65], 200.0);
        assert_eq!(table[66], 300.0);
        assert_eq!(table[67], 400.0);
        // Unmapped code uses default
        assert_eq!(table[0], 500.0);
        assert_eq!(table[100], 500.0);
    }

    #[test]
    fn test_get_byte_to_width_table_no_widths() {
        let font = make_font(|f| {
            f.default_width = 600.0;
        });
        let table = font.get_byte_to_width_table();
        // All entries should be default_width
        for &w in table.iter() {
            assert_eq!(w, 600.0);
        }
    }

    // =========================================================================
    // lookup_predefined_cmap — fallback by ordering alone
    // =========================================================================

    #[test]
    fn test_lookup_predefined_cmap_ordering_fallback_gb1() {
        // Even with non-standard CMap name, ordering "GB1" should work
        let sysinfo = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "GB1".to_string(),
            supplement: 2,
        });
        assert_eq!(lookup_predefined_cmap("SomeCustomCMap", &sysinfo, 34), Some(0x41));
    }

    #[test]
    fn test_lookup_predefined_cmap_ordering_fallback_japan1() {
        let sysinfo = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Japan1".to_string(),
            supplement: 4,
        });
        assert_eq!(lookup_predefined_cmap("CustomJapanCMap", &sysinfo, 34), Some(0x41));
    }

    #[test]
    fn test_lookup_predefined_cmap_ordering_fallback_cns1() {
        let sysinfo = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "CNS1".to_string(),
            supplement: 3,
        });
        assert_eq!(lookup_predefined_cmap("CustomCNSCMap", &sysinfo, 34), Some(0x41));
    }

    #[test]
    fn test_lookup_predefined_cmap_ordering_fallback_korea1() {
        let sysinfo = Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Korea1".to_string(),
            supplement: 1,
        });
        assert_eq!(lookup_predefined_cmap("CustomKoreaCMap", &sysinfo, 34), Some(0x41));
    }

    #[test]
    fn test_lookup_predefined_cmap_unknown_ordering() {
        let sysinfo = Some(CIDSystemInfo {
            registry: "Custom".to_string(),
            ordering: "Unknown".to_string(),
            supplement: 0,
        });
        assert_eq!(lookup_predefined_cmap("AnyCMap", &sysinfo, 34), None);
    }

    // =========================================================================
    // truetype_cmap() accessor — non-TrueType font
    // =========================================================================

    #[test]
    fn test_truetype_cmap_not_truetype() {
        let font = make_font(|f| {
            f.is_truetype_font = false;
            f.embedded_font_data = None;
        });
        assert!(font.truetype_cmap().is_none());
    }

    #[test]
    fn test_truetype_cmap_truetype_no_data() {
        let font = make_font(|f| {
            f.is_truetype_font = true;
            f.embedded_font_data = None;
        });
        assert!(font.truetype_cmap().is_none());
    }

    #[test]
    fn test_truetype_cmap_truetype_empty_data() {
        let font = make_font(|f| {
            f.is_truetype_font = true;
            f.embedded_font_data = Some(Arc::new(vec![]));
        });
        assert!(font.truetype_cmap().is_none());
    }

    #[test]
    fn test_truetype_cmap_truetype_invalid_data() {
        let font = make_font(|f| {
            f.is_truetype_font = true;
            f.embedded_font_data = Some(Arc::new(vec![0xFF, 0xFF, 0xFF, 0xFF]));
        });
        // Invalid font data → extraction fails → None
        assert!(font.truetype_cmap().is_none());
    }

    #[test]
    fn test_has_truetype_cmap_no_data() {
        let font = make_font(|f| {
            f.is_truetype_font = false;
        });
        assert!(!font.has_truetype_cmap());
    }

    // =========================================================================
    // set_truetype_cmap
    // =========================================================================

    #[test]
    fn test_set_truetype_cmap_to_none() {
        let mut font = make_font(|_| {});
        font.set_truetype_cmap(None);
        assert!(font.truetype_cmap().is_none());
    }

    // =========================================================================
    // CIDToGIDMap edge cases
    // =========================================================================

    #[test]
    fn test_cid_to_gid_explicit_empty() {
        let map = CIDToGIDMap::Explicit(vec![]);
        // Empty array → all fall back to identity
        assert_eq!(map.get_gid(0), 0);
        assert_eq!(map.get_gid(100), 100);
    }

    #[test]
    fn test_cid_to_gid_explicit_boundary() {
        let map = CIDToGIDMap::Explicit(vec![99, 88]);
        assert_eq!(map.get_gid(0), 99);
        assert_eq!(map.get_gid(1), 88);
        // index 2 is out of bounds → identity
        assert_eq!(map.get_gid(2), 2);
    }

    #[test]
    fn test_cid_to_gid_identity_max() {
        let map = CIDToGIDMap::Identity;
        assert_eq!(map.get_gid(u16::MAX), u16::MAX);
    }

    // =========================================================================
    // char_to_unicode — Type0 Identity encoding with CIDToGIDMap + AGL fallback
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_agl_fallback() {
        let font = make_font(|f| {
            f.base_font = "SubsetFont+F3".to_string();
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Identity;
            f.cid_to_gid_map = Some(CIDToGIDMap::Identity);
            // No TrueType cmap → AGL fallback path
        });
        // CID 0x41 → GID 0x41 → glyph name "A" → AGL → 'A'
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Type0 RKSJ (Shift-JIS) path
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_rksj() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("90ms-RKSJ-H".to_string());
        });
        // ASCII char through Shift-JIS
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Type0 Identity-H/V at Priority 3 fallback
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_v() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("Identity-V".to_string());
        });
        // No CIDSystemInfo → CID-as-Unicode last resort
        assert_eq!(font.char_to_unicode(0x42), Some("B".to_string()));
    }

    // =========================================================================
    // char_to_unicode — unknown encoding for simple font
    // =========================================================================

    #[test]
    fn test_char_to_unicode_unknown_standard_encoding() {
        let font = make_font(|f| {
            f.encoding = Encoding::Standard("SomeRandomEncoding".to_string());
        });
        // Unknown encoding falls back to ASCII passthrough for printable
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
        // Non-ASCII will return None from standard_encoding_lookup
        assert_eq!(font.char_to_unicode(0x80), None);
    }

    // =========================================================================
    // Encoding enum Debug/Clone
    // =========================================================================

    #[test]
    fn test_encoding_identity_clone() {
        let enc = Encoding::Identity;
        let enc2 = enc.clone();
        assert!(matches!(enc2, Encoding::Identity));
    }

    #[test]
    fn test_encoding_custom_clone() {
        let mut map = HashMap::new();
        map.insert(1u8, 'X');
        let enc = Encoding::Custom(map);
        let enc2 = enc.clone();
        match enc2 {
            Encoding::Custom(m) => assert_eq!(m.get(&1), Some(&'X')),
            _ => panic!("Wrong encoding type"),
        }
    }

    #[test]
    fn test_encoding_debug() {
        let enc = Encoding::Standard("WinAnsiEncoding".to_string());
        let debug = format!("{:?}", enc);
        assert!(debug.contains("WinAnsiEncoding"));
    }

    // =========================================================================
    // CIDSystemInfo clone/debug
    // =========================================================================

    #[test]
    fn test_cidsysteminfo_clone() {
        let info = CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Japan1".to_string(),
            supplement: 6,
        };
        let info2 = info.clone();
        assert_eq!(info2.registry, "Adobe");
        assert_eq!(info2.ordering, "Japan1");
        assert_eq!(info2.supplement, 6);
    }

    #[test]
    fn test_cidsysteminfo_debug() {
        let info = CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "GB1".to_string(),
            supplement: 2,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("Adobe"));
        assert!(debug.contains("GB1"));
    }

    // =========================================================================
    // CIDToGIDMap clone/debug
    // =========================================================================

    #[test]
    fn test_cidtogidmap_clone() {
        let map = CIDToGIDMap::Explicit(vec![1, 2, 3]);
        let map2 = map.clone();
        assert_eq!(map2.get_gid(0), 1);
    }

    #[test]
    fn test_cidtogidmap_debug() {
        let map = CIDToGIDMap::Identity;
        let debug = format!("{:?}", map);
        assert!(debug.contains("Identity"));
    }

    // =========================================================================
    // char_to_unicode — Type0 Identity encoding with large CID (> 0xFFFF)
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_large_cid() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Identity;
            f.cid_to_gid_map = Some(CIDToGIDMap::Identity);
        });
        // CID > 0xFFFF: TrueType cmap lookup returns early with None,
        // AGL fallback also returns early for large CIDs,
        // then CID-as-Unicode fallback kicks in: 0x10000 is valid Unicode (Linear B Syllable B008 A)
        assert_eq!(font.char_to_unicode(0x10000), Some("\u{10000}".to_string()));
        // But a CID that maps to a control character should return FFFD
        assert_eq!(font.char_to_unicode(0x01), Some("\u{FFFD}".to_string()));
    }

    // =========================================================================
    // char_to_unicode — Type0 predefined CMap fallback (Priority 2b)
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_predefined_cmap_japan1() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Identity; // Will be tried at priority 2b
            f.cid_system_info = Some(CIDSystemInfo {
                registry: "Adobe".to_string(),
                ordering: "Japan1".to_string(),
                supplement: 4,
            });
        });
        // CID 843 → Hiragana あ (U+3042) via predefined Japan1 CMap
        assert_eq!(font.char_to_unicode(843), Some("\u{3042}".to_string()));
    }

    // =========================================================================
    // gid_to_standard_glyph_name — boundary checks
    // =========================================================================

    #[test]
    fn test_gid_to_standard_glyph_name_boundary_values() {
        // First valid entry
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x20), Some("space"));
        // Last valid in basic ASCII
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x7E), Some("asciitilde"));
        // Just before first valid
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x1F), None);
        // 0x7F (DEL) is not mapped
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0x7F), None);
        // Last valid entry
        assert_eq!(FontInfo::gid_to_standard_glyph_name(0xFF), Some("ydieresis"));
    }

    // =========================================================================
    // glyph_name_to_unicode — AGL completeness spot checks
    // =========================================================================

    #[test]
    fn test_glyph_name_to_unicode_math_symbols() {
        assert_eq!(glyph_name_to_unicode("infinity"), Some('∞'));
        assert_eq!(glyph_name_to_unicode("notequal"), Some('≠'));
        assert_eq!(glyph_name_to_unicode("lessequal"), Some('≤'));
        assert_eq!(glyph_name_to_unicode("greaterequal"), Some('≥'));
    }

    #[test]
    fn test_glyph_name_to_unicode_german_sharp_s() {
        assert_eq!(glyph_name_to_unicode("germandbls"), Some('ß'));
    }

    #[test]
    fn test_glyph_name_to_unicode_copyright_registered() {
        assert_eq!(glyph_name_to_unicode("copyright"), Some('©'));
        assert_eq!(glyph_name_to_unicode("registered"), Some('®'));
        assert_eq!(glyph_name_to_unicode("trademark"), Some('™'));
    }

    // =========================================================================
    // char_to_unicode — Type0 Identity-H with non-Identity ordering (CJK)
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_identity_h_cjk_ordering() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("Identity-H".to_string());
            f.cid_system_info = Some(CIDSystemInfo {
                registry: "Adobe".to_string(),
                ordering: "Japan1".to_string(),
                supplement: 4,
            });
        });
        // Non-identity ordering with Identity-H: CIDs are NOT Unicode
        // Should use predefined CMap lookup
        // CID 843 → Hiragana あ (U+3042)
        assert_eq!(font.char_to_unicode(843), Some("\u{3042}".to_string()));
    }

    // =========================================================================
    // char_to_unicode — UCS2/UTF16 encoding variant
    // =========================================================================

    #[test]
    fn test_char_to_unicode_type0_ucs2_encoding() {
        let font = make_font(|f| {
            f.subtype = "Type0".to_string();
            f.encoding = Encoding::Standard("UniJIS-UCS2-H".to_string());
            f.cid_system_info = Some(CIDSystemInfo {
                registry: "Adobe".to_string(),
                ordering: "Identity".to_string(),
                supplement: 0,
            });
        });
        // UCS2 encoding: char_code IS the Unicode value
        assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    }

    // =========================================================================
    // Standard encoding control char handling
    // =========================================================================

    #[test]
    fn test_standard_encoding_winansi_control_range() {
        // Codes 0-31 are control range — WinAnsi doesn't map these
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x00), None);
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x01), None);
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x1F), None);
    }

    // =========================================================================
    // WinAnsi full extended range spot checks
    // =========================================================================

    #[test]
    fn test_standard_encoding_winansi_full_extended() {
        // 0x85 → Horizontal ellipsis (U+2026)
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x85), Some("\u{2026}".to_string()));
        // 0x99 → Trade mark sign (U+2122)
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0x99), Some("\u{2122}".to_string()));
        // 0xFF → Latin small letter y with diaeresis
        assert_eq!(standard_encoding_lookup("WinAnsiEncoding", 0xFF), Some("\u{00FF}".to_string()));
    }

    // ==========================================
    // wrap_cff_in_opentype tests
    // ==========================================

    #[test]
    fn test_wrap_cff_in_opentype_header() {
        // Minimal CFF data (version 1.0, hdrSize=4, offSize=1)
        let cff = vec![1, 0, 4, 1, 0, 0, 0, 0];
        let otf = super::wrap_cff_in_opentype(&cff);

        // Must start with 'OTTO' tag
        assert_eq!(&otf[0..4], b"OTTO");
        // numTables = 4 (CFF, head, hhea, maxp)
        assert_eq!(u16::from_be_bytes([otf[4], otf[5]]), 4);
        // Must contain the CFF data at some offset
        assert!(otf.windows(cff.len()).any(|w| w == &cff[..]));
    }

    #[test]
    fn test_wrap_cff_in_opentype_contains_required_tables() {
        let cff = vec![1, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let otf = super::wrap_cff_in_opentype(&cff);

        // Check all 4 required table tags exist in the table directory
        // Table directory starts at offset 12, each record is 16 bytes
        let mut found_tables = Vec::new();
        for i in 0..4 {
            let offset = 12 + i * 16;
            let tag = std::str::from_utf8(&otf[offset..offset + 4]).unwrap_or("????");
            found_tables.push(tag.to_string());
        }
        found_tables.sort();
        assert_eq!(found_tables, vec!["CFF ", "head", "hhea", "maxp"]);
    }

    #[test]
    fn test_wrap_cff_in_opentype_parseable() {
        // Create a minimal but valid CFF font stub
        let cff = vec![1, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let otf = super::wrap_cff_in_opentype(&cff);

        // ttf-parser should be able to parse the header (head + hhea + maxp)
        // without panicking, even if CFF data is minimal
        let result = ttf_parser::Face::parse(&otf, 0);
        // May fail on CFF content but should not panic on table parsing
        // The fact that it doesn't panic is the test
        let _ = result;
    }

    // ==========================================
    // get_standard_font_width tests
    // ==========================================

    #[test]
    fn test_standard_font_width_times_roman() {
        let font = FontInfo {
            base_font: "Times-Roman".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: None, // No widths → should use standard metrics
            first_char: None,
            last_char: None,
            default_width: 500.0,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // 'A' = 722 in Times-Roman (not the default 500)
        assert_eq!(font.get_glyph_width(65), 722.0);
        // 'i' = 278 (narrow)
        assert_eq!(font.get_glyph_width(105), 278.0);
        // space = 250
        assert_eq!(font.get_glyph_width(32), 250.0);
        // 'm' = 778 (wide)
        assert_eq!(font.get_glyph_width(109), 778.0);
    }

    #[test]
    fn test_standard_font_width_courier_monospace() {
        let font = FontInfo {
            base_font: "Courier".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Courier is monospace — all chars 600
        assert_eq!(font.get_glyph_width(65), 600.0); // A
        assert_eq!(font.get_glyph_width(105), 600.0); // i
        assert_eq!(font.get_glyph_width(32), 600.0); // space
    }

    #[test]
    fn test_standard_font_width_not_applied_with_widths_array() {
        let font = FontInfo {
            base_font: "Times-Roman".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: Some(vec![999.0]), // Has explicit widths
            first_char: Some(65),      // Starting at 'A'
            last_char: Some(65),
            default_width: 500.0,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Should use explicit width (999), not standard Times width (722)
        assert_eq!(font.get_glyph_width(65), 999.0);
    }

    #[test]
    fn test_standard_font_width_not_applied_to_unknown_font() {
        let font = FontInfo {
            base_font: "MyCustomFont".to_string(),
            subtype: "Type1".to_string(),
            encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
            to_unicode: None,
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: std::sync::OnceLock::new(),
            is_truetype_font: false,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0,
            cid_widths: None,
            cid_default_width: 1000.0,
            cff_gid_map: None,
            multi_char_map: HashMap::new(),
            byte_to_char_table: std::sync::OnceLock::new(),
            byte_to_width_table: std::sync::OnceLock::new(),
        };

        // Unknown font → should fall back to default_width (500)
        assert_eq!(font.get_glyph_width(65), 500.0);
    }
}
