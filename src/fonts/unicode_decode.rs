//! The one glyph decoder shared by every text surface.
//!
//! `decode_text_to_unicode` and its helpers (`fallback_char_to_unicode`,
//! `get_byte_mode`, `TextCharIter`) used to exist twice — an extraction copy
//! in `extractors/text.rs` and a rendering copy in
//! `rendering/text_rasterizer.rs` — and the copies drifted in both
//! directions: UTF-8 codespace CMaps and `preserve_unmapped_glyphs` existed
//! only in extraction; ligature decomposition and dropped-glyph accounting
//! only in rendering. A Type0 font with a UTF-8 CMap extracted right and
//! rendered garbage; a ligature rendered right and extracted lossy. This
//! module is the union; the two genuine policy differences are expressed in
//! [`DecodePolicy`], not by forking the decoder.

use std::collections::HashMap;

use crate::fonts::truetype_cmap::TrueTypeCMap;
use crate::fonts::FontInfo;

/// Per-surface decoding policy — the only intended differences between the
/// extraction and rendering paths.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DecodePolicy {
    /// Keep U+FFFD for unmapped codes instead of dropping them.
    /// Extraction honours the `preserve_unmapped_glyphs()` toggle; rendering
    /// always drops (a U+FFFD has no glyph to paint).
    pub preserve_unmapped: bool,
    /// Expand presentation-form ligature code points (fi, fl, ffi, …) into
    /// component letters. The rasterizer needs this so the shaper doesn't
    /// drop the cluster (issue #331); extraction decomposes downstream in
    /// `ligature_processor`, so it must stay off here or chars would change.
    pub decompose_ligatures: bool,
}

/// Counts glyphs a decode dropped so the loss is reported, not silent (#991).
#[derive(Debug, Default)]
pub(crate) struct GlyphDropTally {
    count: usize,
    first: Option<(&'static str, u32, u16)>,
}

impl GlyphDropTally {
    pub(crate) fn record(&mut self, reason: &'static str, char_code: u32, gid: u16) {
        self.count += 1;
        self.first.get_or_insert((reason, char_code, gid));
    }

    pub(crate) fn report(&self, font_name: &str) {
        let Some((reason, char_code, gid)) = self.first else {
            return;
        };
        let message = format!(
            "font '{font_name}' painted nothing for {} glyph(s) while advancing the cursor; \
             first was code 0x{char_code:X} (glyph {gid}): {reason}. The page renders with \
             a gap that reads as whitespace downstream.",
            self.count
        );
        log::warn!(target: "pdf_oxide::fonts", "{message}");
        crate::extractors::warnings::push_global_warning(crate::extractors::warnings::Warning {
            category: crate::extractors::warnings::WarningCategory::GlyphDropped,
            page: None,
            message,
            spec_section: Some("9.6.6"),
        });
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
pub(crate) fn fallback_char_to_unicode(char_code: u32) -> String {
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
        // Valid Unicode: BMP (0x0000-0xD7FF, 0xE000-0xFFFF) and supplementary planes
        // Excludes surrogate pairs (0xD800-0xDFFF)
        code => {
            if let Some(ch) = char::from_u32(code) {
                if (0xE000..=0xF8FF).contains(&code) {
                    log::debug!("Private Use Area character: U+{:04X}", code);
                }
                ch.to_string()
            } else {
                log::warn!("Character code 0x{:04X} is not a valid Unicode code point", code);
                "?".to_string()
            }
        },
    }
}

/// Byte grouping mode for CID font character code decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteMode {
    /// Single-byte codes (simple fonts, some predefined CMaps)
    OneByte,
    /// Always 2-byte codes (Identity-H/V, UCS2)
    TwoByte,
    /// Shift-JIS variable-width (1 or 2 bytes depending on lead byte)
    ShiftJIS,
}

/// True when a Type0 font's `/Encoding` is a UTF-8 (variable-width) CMap —
/// `Uni-Utf8-H` (embedded, pdf.js issue18117) or the Adobe predefined
/// `UniGB-UTF8-H` / `UniCNS-UTF8-H` / `UniJIS-UTF8-H` / `UniKS-UTF8-H` family.
/// Such codes are 1–4 bytes and must be segmented by UTF-8 lead-byte rules
/// (see `decode_text_to_unicode`), not the fixed 1/2-byte `ByteMode`. Matching
/// on the CMap name keeps the change isolated to these fonts. See #610.
pub(crate) fn font_has_utf8_cmap(font: &FontInfo) -> bool {
    if font.subtype != "Type0" {
        return false;
    }
    if let crate::fonts::Encoding::Standard(name) = &font.encoding {
        let lower = name.to_ascii_lowercase();
        lower.contains("utf8") || lower.contains("utf-8")
    } else {
        false
    }
}

/// Get byte grouping mode for a font (v0.3.14).
pub(crate) fn get_byte_mode(font: Option<&FontInfo>) -> ByteMode {
    if let Some(font) = font {
        if font.subtype == "Type0" {
            // If the ToUnicode CMap declares a 2-byte codespace range, always use
            // TwoByte mode regardless of the encoding name. This handles CJK fonts
            // whose /Encoding name is a custom CMap stream that doesn't match the
            // well-known keyword patterns below (e.g. "H", "V", "UniCNS-H", …).
            // See PDF Spec §9.7.5 — `begincodespacerange` is authoritative.
            if let Some(ref lazy_cmap) = font.to_unicode {
                if lazy_cmap.code_width() == 2 {
                    return ByteMode::TwoByte;
                }
            }

            match &font.encoding {
                crate::fonts::Encoding::Identity => ByteMode::TwoByte,
                crate::fonts::Encoding::Standard(name) => {
                    if (name.contains("Identity") && !name.contains("OneByteIdentity"))
                        || name.contains("UCS2")
                        || name.contains("UTF16")
                        // CORPUS-3: bare Adobe predefined horizontal/vertical CMaps
                        // ("H"/"V", e.g. Adobe-Japan1-H) are 2-byte by definition;
                        // without this they were read single-byte → CJK garbage
                        // ("あいうえお" → "CACCCECGCI" on noembed-jis7).
                        || name == "H"
                        || name == "V"
                    {
                        ByteMode::TwoByte
                    } else if name.contains("RKSJ") {
                        ByteMode::ShiftJIS
                    } else if name.contains("EUC")
                        || name.contains("GBK")
                        || name.contains("GBpc")
                        || name.contains("GB-")
                        || name.contains("CNS")
                        || name.contains("B5")
                        || name.contains("KSC")
                        || name.contains("KSCms")
                    {
                        ByteMode::TwoByte
                    } else {
                        ByteMode::OneByte
                    }
                },
                _ => ByteMode::OneByte,
            }
        } else {
            ByteMode::OneByte
        }
    } else {
        ByteMode::OneByte
    }
}

/// Iterator over characters in a PDF string based on font encoding (v0.3.14).
pub(crate) struct TextCharIter<'a> {
    bytes: &'a [u8],
    byte_mode: ByteMode,
    index: usize,
}

impl<'a> TextCharIter<'a> {
    pub(crate) fn new(bytes: &'a [u8], font: Option<&FontInfo>) -> Self {
        Self {
            bytes,
            byte_mode: get_byte_mode(font),
            index: 0,
        }
    }
}

impl<'a> Iterator for TextCharIter<'a> {
    type Item = (u16, usize); // (char_code, bytes_consumed)

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.bytes.len() {
            return None;
        }

        let (char_code, bytes_consumed) = match self.byte_mode {
            ByteMode::TwoByte if self.index + 1 < self.bytes.len() => {
                (((self.bytes[self.index] as u16) << 8) | (self.bytes[self.index + 1] as u16), 2)
            },
            ByteMode::ShiftJIS => {
                let b = self.bytes[self.index];
                let is_lead = (0x81..=0x9F).contains(&b) || (0xE0..=0xFC).contains(&b);
                if is_lead && self.index + 1 < self.bytes.len() {
                    (((b as u16) << 8) | (self.bytes[self.index + 1] as u16), 2)
                } else {
                    (b as u16, 1)
                }
            },
            _ => (self.bytes[self.index] as u16, 1),
        };

        self.index += bytes_consumed;
        Some((char_code, bytes_consumed))
    }
}

pub(crate) fn decode_text_to_unicode(
    bytes: &[u8],
    font: Option<&FontInfo>,
    policy: DecodePolicy,
    mut drops: Option<&mut GlyphDropTally>,
) -> String {
    let raw_result = if let Some(font) = font {
        let mut result = String::new();
        // Use pre-computed lookup table for performance if it's a simple font
        if font.subtype != "Type0" {
            let table = font.get_byte_to_char_table();
            for &byte in bytes {
                let c = table[byte as usize];
                if c != '\0' {
                    result.push(c);
                } else {
                    // Fallback: multi-char mapping or unmapped byte
                    let char_str = font
                        .char_to_unicode(byte as u32)
                        .unwrap_or_else(|| fallback_char_to_unicode(byte as u32));
                    if char_str != "\u{FFFD}" || policy.preserve_unmapped {
                        result.push_str(&char_str);
                    } else if let Some(tally) = drops.as_deref_mut() {
                        tally.record("no Unicode mapping", byte as u32, 0);
                    }
                }
            }
        } else if font_has_utf8_cmap(font) {
            // Type0 font whose /Encoding is an embedded CMap with a UTF-8
            // (variable-width) codespace — e.g. `Uni-Utf8-H` (pdf.js
            // issue18117) and the Adobe predefined `Uni*-UTF8-H` family.
            // Codes are 1–4 bytes segmented by UTF-8 lead-byte rules, which
            // exceed the u16 of `TextCharIter`. Segment here into u32 codes
            // and resolve via the (present) ToUnicode CMap, which is keyed by
            // the same multi-byte codes. Isolated to UTF-8-CMap fonts: every
            // other font keeps the path below unchanged. See #610.
            let n = bytes.len();
            let mut i = 0;
            while i < n {
                let lead = bytes[i];
                let width = match lead {
                    0x00..=0x7F => 1,
                    0xC0..=0xDF => 2,
                    0xE0..=0xEF => 3,
                    0xF0..=0xF7 => 4,
                    _ => 1, // invalid lead byte → consume one, avoids stalling
                }
                .min(n - i);
                let mut code: u32 = 0;
                for &b in &bytes[i..i + width] {
                    code = (code << 8) | b as u32;
                }
                let char_str = font
                    .char_to_unicode(code)
                    .unwrap_or_else(|| fallback_char_to_unicode(code));
                if char_str != "\u{FFFD}" || policy.preserve_unmapped {
                    result.push_str(&char_str);
                } else if let Some(tally) = drops.as_deref_mut() {
                    tally.record("no Unicode mapping", code, 0);
                }
                i += width;
            }
        } else {
            // Complex font: use unified iterator for robust multi-byte decoding
            for (char_code, _) in TextCharIter::new(bytes, Some(font)) {
                let char_str = font
                    .char_to_unicode(char_code as u32)
                    .unwrap_or_else(|| fallback_char_to_unicode(char_code as u32));

                if char_str != "\u{FFFD}" || policy.preserve_unmapped {
                    result.push_str(&char_str);
                } else if let Some(tally) = drops.as_deref_mut() {
                    tally.record("no Unicode mapping", char_code as u32, 0);
                }
            }
        }
        result
    } else {
        // No font - fallback to Latin-1 (ISO 8859-1) encoding
        // Per PDF Spec ISO 32000-1:2008, Section 9.6.6, Latin-1 maps bytes 0x00-0xFF
        // directly to Unicode code points U+0000-U+00FF
        log::warn!(
            "⚠️  No font provided for {} bytes, using Latin-1 fallback (PDF spec compliant)",
            bytes.len()
        );
        bytes.iter().map(|&b| char::from(b)).collect()
    };

    // Filter control characters from failed encoding resolution
    // Keep: \t (0x09), \n (0x0A), \r (0x0D), and all printable chars (>= 0x20)
    let mut filtered = String::with_capacity(raw_result.len());
    for c in raw_result.chars() {
        if c < '\x20' && c != '\t' && c != '\n' && c != '\r' {
            continue;
        }
        if policy.decompose_ligatures {
            if let Some(components) = crate::text::ligature_processor::get_ligature_components(c) {
                filtered.push_str(components);
                continue;
            }
        }
        filtered.push(c);
    }
    filtered
}

/// Strip a subset-tag prefix from a base font name
/// (e.g., `"QQPMQK+Impact"` → `"Impact"`). Only the spec-shaped form —
/// six uppercase letters and a `+` — is treated as a tag, so fonts whose
/// real name contains `+` are left alone.
pub(crate) fn strip_subset_prefix(name: &str) -> &str {
    if name.len() > 7
        && name.as_bytes()[6] == b'+'
        && name[..6].chars().all(|c| c.is_ascii_uppercase())
    {
        &name[7..]
    } else {
        name
    }
}

/// The best available TrueType cmap for each stripped base font name.
///
/// When multiple subset variants of the same font exist (`ABCDEF+Arial`,
/// `GHIJKL+Arial`), pick the cmap with the most glyph mappings — it has the
/// best Unicode coverage. On equal coverage, prefer the lexicographically
/// smallest `base_font` as a deterministic tie-breaker (HashMap iteration
/// order is randomized per-process). Callers donate these cmaps to
/// Identity-H fonts that lack one; the write loop stays at the call site
/// because the two font tables store `FontInfo` differently (`Arc` vs
/// owned).
pub(crate) fn best_truetype_cmaps<'a>(
    fonts: impl Iterator<Item = &'a FontInfo>,
) -> HashMap<String, (TrueTypeCMap, String)> {
    let mut best: HashMap<String, (TrueTypeCMap, String)> = HashMap::new();
    for font in fonts {
        if let Some(cmap) = font.truetype_cmap() {
            let stripped = strip_subset_prefix(&font.base_font).to_string();
            let dominated = best
                .get(&stripped)
                .is_none_or(|(existing, existing_name)| match cmap.len().cmp(&existing.len()) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Equal => font.base_font < *existing_name,
                    std::cmp::Ordering::Less => false,
                });
            if dominated {
                best.insert(stripped, (cmap.clone(), font.base_font.clone()));
            }
        }
    }
    best
}
