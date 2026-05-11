//! Text rasterizer - renders PDF text using tiny-skia.
//!
//! Text rendering in PDF is complex because:
//! - Fonts may be embedded or use standard PDF fonts
//! - Character encoding varies (identity-H, MacRoman, custom ToUnicode, etc.)
#![allow(clippy::collapsible_if, clippy::vec_box)]
//! - Glyph positioning is explicit via TJ arrays
//!
//! This module provides a text rendering implementation that:
//! - Uses system fonts as fallback when embedded fonts aren't available
//! - Renders text using rustybuzz for shaping and tiny-skia for drawing glyph paths

use super::create_fill_paint;
use crate::content::operators::TextElement;
use crate::content::GraphicsState;
use crate::document::PdfDocument;
use crate::error::{Error, Result};
use crate::object::Object;
use std::collections::HashMap;
use std::sync::Arc;

use tiny_skia::{Paint, PathBuilder, Pixmap, Transform};
use ttf_parser::OutlineBuilder;

/// Outline builder that converts ttf-parser paths to tiny-skia paths.
struct SkiaOutlineBuilder<'a>(&'a mut PathBuilder);

impl<'a> OutlineBuilder for SkiaOutlineBuilder<'a> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(x, y);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.0.quad_to(x1, y1, x, y);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.0.cubic_to(x1, y1, x2, y2, x, y);
    }
    fn close(&mut self) {
        self.0.close();
    }
}

/// Classify an embedded font's cmap tables in a single parse pass and cache
/// the result keyed on the Arc pointer (stable for the document lifetime).
///
/// Returns `(is_byte_indexed_only, has_unicode_cmap)`:
/// - `is_byte_indexed_only`: only Macintosh byte-indexed cmap present → use
///   `render_cid_direct` rather than Unicode shaping.
/// - `has_unicode_cmap`: a Unicode/Windows cmap is present → Unicode shaping
///   is likely to produce non-.notdef glyphs; use `render_unicode_text`.
///
/// Before this cache existed, every `render_text` call with an embedded font
/// ran `has_byte_indexed_cmap` (one `ttf_parser::Face::parse`) **plus** a full
/// `rustybuzz::shape` probe on the current text — so 1000 text segments on one
/// page paid for 1000 parse + 1000 shape calls on the same font bytes. Now
/// those are collapsed to one parse per embedded font per process.
static EMBEDDED_FONT_CLASS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, (bool, bool)>>,
> = std::sync::OnceLock::new();

fn classify_embedded_font(data: &Arc<Vec<u8>>) -> (bool, bool) {
    // Use the Arc's inner-pointer value as a stable key (two Arc::clones of the
    // same allocation share the same raw pointer, so this is always a cache hit
    // after the first call for any given font binary).
    let key = Arc::as_ptr(data) as usize;
    let cache =
        EMBEDDED_FONT_CLASS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&v) = guard.get(&key) {
            return v;
        }
    }
    let result = (|| {
        let face = ttf_parser::Face::parse(data, 0).ok()?;
        let cmap = face.tables().cmap?;
        let mut saw_byte_indexed = false;
        let mut saw_unicode = false;
        for sub in cmap.subtables {
            use ttf_parser::PlatformId;
            match sub.platform_id {
                PlatformId::Unicode => saw_unicode = true,
                PlatformId::Windows if sub.encoding_id == 1 || sub.encoding_id == 10 => {
                    saw_unicode = true;
                },
                PlatformId::Macintosh if sub.encoding_id == 0 => saw_byte_indexed = true,
                _ => {},
            }
        }
        Some((saw_byte_indexed && !saw_unicode, saw_unicode))
    })()
    .unwrap_or((false, false));
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, result);
    result
}

/// Resolve a single PDF content byte to a GID by consulting the font's
/// own cmap subtables. Prefers a byte-indexed (Macintosh Roman) subtable
/// when present; falls back to ttf-parser's default Unicode resolution
/// for ASCII-range bytes if no byte-indexed subtable exists.
fn cmap_byte_to_gid(face: &ttf_parser::Face, byte: u8) -> Option<u16> {
    if let Some(cmap) = face.tables().cmap {
        for sub in cmap.subtables {
            use ttf_parser::PlatformId;
            if matches!(sub.platform_id, PlatformId::Macintosh) && sub.encoding_id == 0 {
                if let Some(gid) = sub.glyph_index(byte as u32) {
                    return Some(gid.0);
                }
            }
        }
    }
    face.glyph_index(byte as char).map(|g| g.0)
}

/// Process-wide cache for the system font database.
///
/// `fontdb::Database::load_system_fonts()` walks every font directory on
/// the host and parses each face it finds, which typically takes several
/// seconds on first call. Before this cache was introduced, every
/// `TextRasterizer::new()` (and therefore every `PageRenderer::new()`)
/// paid that cost, and callers who constructed a fresh `PageRenderer`
/// per page — which is the obvious first-draft usage from the Python /
/// CLI surface — hit the scan once per page. A cold-cache ORAFOL 5400
/// render took ~4.1 s on a warm machine for a single page because of
/// this. See issue #331.
///
/// Switching to a process-wide `OnceLock<Arc<fontdb::Database>>` loads
/// the database exactly once per process, and every subsequent
/// `TextRasterizer` constructor takes a cheap `Arc::clone`. Wrapping
/// in `Arc` is important so that the cache is still cheaply shareable
/// across `TextRasterizer` instances in different rendering contexts
/// without re-copying the full parsed font metadata. Callers that want
/// a private / modified database can still construct one by hand and
/// bypass this cache via `TextRasterizer::with_fontdb()`.
static SYSTEM_FONTDB: std::sync::OnceLock<std::sync::Arc<fontdb::Database>> =
    std::sync::OnceLock::new();

fn system_fontdb() -> std::sync::Arc<fontdb::Database> {
    SYSTEM_FONTDB
        .get_or_init(|| {
            let mut db = fontdb::Database::new();
            db.load_system_fonts();
            std::sync::Arc::new(db)
        })
        .clone()
}

/// Process-wide cache mapping fontdb::ID → (font bytes, face index).
///
/// Without this cache, `load_font_data` calls `with_face_data(...to_vec())`
/// which clones the entire font binary (often 300–500 KB for Liberation Serif
/// or Times New Roman) on every `render_text` call. A two-page text PDF can
/// trigger hundreds of such clones per render pass. This cache reduces each
/// subsequent access to a cheap `Arc::clone`.
static FONT_BYTES_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<fontdb::ID, (Arc<Vec<u8>>, u32)>>,
> = std::sync::OnceLock::new();

fn cached_font_bytes(id: fontdb::ID, db: &fontdb::Database) -> Option<(Arc<Vec<u8>>, u32)> {
    let cache =
        FONT_BYTES_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = guard.get(&id) {
            return Some(entry.clone());
        }
    }
    let mut result: Option<(Arc<Vec<u8>>, u32)> = None;
    db.with_face_data(id, |data, index| {
        result = Some((Arc::new(data.to_vec()), index));
    });
    if let Some(ref entry) = result {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(id, entry.clone());
    }
    result
}

/// Parsed font faces cached by fontdb ID.
///
/// `rustybuzz::Face` and `ttf_parser::Face` both borrow the backing bytes.
/// We use a self-referential pattern (backed Arc keeps bytes alive) with
/// unsafe 'static transmute so we can store them in a process-wide cache
/// and reuse them across hundreds of render_text calls for the same font.
///
/// # Safety
/// Both face fields borrow `_data`'s heap allocation (not the Arc pointer
/// itself, so no double-free on Arc drop). The fields are only ever
/// accessed while `_data` is alive — i.e., while this struct exists.
/// Because the struct is behind `Arc`, it lives at least as long as any
/// caller that holds a clone of that Arc.
struct CachedFace {
    _data: Arc<Vec<u8>>,
    rb_face: rustybuzz::Face<'static>,
    ttf_face: ttf_parser::Face<'static>,
    pub units_per_em: f32,
}

// SAFETY: rustybuzz::Face and ttf_parser::Face only borrow immutable bytes.
unsafe impl Send for CachedFace {}
unsafe impl Sync for CachedFace {}

impl CachedFace {
    fn new(data: Arc<Vec<u8>>, index: u32) -> Option<Self> {
        let rb_face: rustybuzz::Face<'_> = rustybuzz::Face::from_slice(&data, index)?;
        let ttf_face: ttf_parser::Face<'_> = ttf_parser::Face::parse(&data, index).ok()?;
        let units_per_em = ttf_face.units_per_em() as f32;
        // SAFETY: both faces borrow the data slice. We store an Arc to that
        // data in `_data`, ensuring the bytes stay alive for this struct's lifetime.
        let rb_face: rustybuzz::Face<'static> = unsafe { std::mem::transmute(rb_face) };
        let ttf_face: ttf_parser::Face<'static> = unsafe { std::mem::transmute(ttf_face) };
        Some(CachedFace {
            _data: data,
            rb_face,
            ttf_face,
            units_per_em,
        })
    }
}

static FACE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(fontdb::ID, u32), Arc<CachedFace>>>,
> = std::sync::OnceLock::new();

fn cached_face(id: fontdb::ID, data: Arc<Vec<u8>>, index: u32) -> Option<Arc<CachedFace>> {
    let cache = FACE_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = guard.get(&(id, index)) {
            return Some(entry.clone());
        }
    }
    let face = CachedFace::new(data, index)?;
    let arc = Arc::new(face);
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.insert((id, index), arc.clone());
    Some(arc)
}

/// Process-wide CJK fallback font — loaded once per process, shared by all
/// TextRasterizer instances.
///
/// Before this cache, every glyph that fell through to the CJK path called
/// `load_cjk_fallback()`, which iterated 7+ fontdb queries and cloned a
/// 10–20 MB Noto CJK binary. Now that work is done exactly once.
static CJK_FALLBACK: std::sync::OnceLock<Option<(fontdb::ID, Arc<Vec<u8>>, u32)>> =
    std::sync::OnceLock::new();

fn get_cjk_fallback_cached(db: &fontdb::Database) -> Option<(fontdb::ID, Arc<Vec<u8>>, u32)> {
    CJK_FALLBACK
        .get_or_init(|| {
            let prioritized_variants = [
                "Noto Sans CJK SC",
                "Noto Serif CJK SC",
                "Droid Sans Fallback",
                "SimSun",
                "WenQuanYi Micro Hei",
                "Noto Sans CJK JP",
                "Noto Serif CJK JP",
            ];
            for variant in prioritized_variants {
                let query = fontdb::Query {
                    families: &[fontdb::Family::Name(variant)],
                    weight: fontdb::Weight::NORMAL,
                    stretch: fontdb::Stretch::Normal,
                    style: fontdb::Style::Normal,
                };
                if let Some(id) = db.query(&query) {
                    if let Some((arc, idx)) = cached_font_bytes(id, db) {
                        log::debug!(
                            "CJK fallback: matched '{}', idx={}, size={} bytes",
                            variant,
                            idx,
                            arc.len()
                        );
                        return Some((id, arc, idx));
                    }
                }
            }
            let query = fontdb::Query {
                families: &[fontdb::Family::SansSerif],
                weight: fontdb::Weight::NORMAL,
                stretch: fontdb::Stretch::Normal,
                style: fontdb::Style::Normal,
            };
            if let Some(id) = db.query(&query) {
                if let Some((arc, idx)) = cached_font_bytes(id, db) {
                    return Some((id, arc, idx));
                }
            }
            None
        })
        .as_ref()
        .map(|(id, arc, idx)| (*id, Arc::clone(arc), *idx))
}

/// Rasterizer for PDF text operations.
pub struct TextRasterizer {
    /// Font database for system font fallback.
    ///
    /// Shared across rasterizers via a process-wide `OnceLock` cache so
    /// we don't re-scan the system font directories on every new
    /// `PageRenderer`. See the `SYSTEM_FONTDB` docstring for the
    /// measurement that motivated the switch.
    fontdb: std::sync::Arc<fontdb::Database>,
}

impl TextRasterizer {
    /// Create a new text rasterizer using the cached system font database.
    pub fn new() -> Self {
        Self {
            fontdb: system_fontdb(),
        }
    }

    /// Construct with a caller-supplied font database. Bypasses the
    /// process-wide cache — useful for tests or callers that need to
    /// pre-populate the database with non-system fonts.
    #[allow(dead_code)]
    pub fn with_fontdb(fontdb: std::sync::Arc<fontdb::Database>) -> Self {
        Self { fontdb }
    }

    /// Render a text string (Tj operator).
    /// Returns the total horizontal advance in PDF points.
    #[allow(unused_variables)]
    pub fn render_text(
        &self,
        pixmap: &mut Pixmap,
        text: &[u8],
        base_transform: Transform,
        gs: &GraphicsState,
        _resources: &Object,
        doc: &PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
        font_cache: &HashMap<String, Arc<crate::fonts::FontInfo>>,
    ) -> Result<f32> {
        // Get font info from cache
        let font_info = if let Some(font_name) = &gs.font_name {
            font_cache.get(font_name).cloned()
        } else {
            None
        };

        // Convert raw PDF bytes to Unicode string using font encoding
        let unicode_text = self.decode_text_to_unicode(text, font_info.as_deref());
        log::debug!("Decoded text: '{}' (font={:?})", unicode_text, gs.font_name);

        // Create paint from fill color
        let mut paint = create_fill_paint(gs, "Normal");
        // Text rendering mode 3 = invisible text (used for searchable OCR layers)
        if gs.render_mode == 3 {
            paint.set_color(tiny_skia::Color::from_rgba(0.0, 0.0, 0.0, 0.0).unwrap());
        }

        // Find and load font - prioritize embedded font data
        let pdf_font_name = gs.font_name.as_deref().unwrap_or("Helvetica");
        let font_data_and_index: Option<(Option<fontdb::ID>, Arc<Vec<u8>>, u32, bool)> =
            if let Some(ref info) = font_info {
                if let Some(ref embedded) = info.embedded_font_data {
                    // Simple (non-Type0) TrueType subsets whose sole cmap subtable
                    // is a byte-indexed table must be rendered by feeding the raw
                    // PDF content bytes to the embedded cmap directly — the PDF
                    // byte is the cmap input under the font's declared encoding
                    // (ISO 32000-1 §9.6.6.4). Unicode shaping against these fonts
                    // is unreliable: even if a space or punctuation happens to
                    // share a codepoint with a cmap key, shaping for letters
                    // resolves to .notdef and the system-font fallback picks up
                    // unrelated glyphs. Bypass the Unicode shaping path entirely
                    // for this subtype so the byte→GID route is taken for every
                    // `Tj` / `TJ` call, not just the ones whose decoded Unicode
                    // happens to miss the cmap.
                    // Classify the embedded font's cmap tables once per Arc lifetime;
                    // subsequent calls for the same font bytes are a cheap HashMap hit.
                    let (is_byte_indexed, has_unicode_cmap) = classify_embedded_font(embedded);
                    if info.subtype != "Type0" && is_byte_indexed {
                        log::debug!(
                        "Using embedded font '{}' with byte-indexed cmap (simple TrueType subset)",
                        info.base_font
                    );
                        return self.render_cid_direct(
                            pixmap,
                            text,
                            info,
                            embedded,
                            0,
                            &paint,
                            base_transform,
                            gs,
                            clip_mask,
                        );
                    }

                    if has_unicode_cmap {
                        log::debug!("Using embedded font data for '{}'", info.base_font);
                        Some((None, Arc::clone(embedded), 0, false))
                    } else if info.subtype == "Type0"
                        && info.cid_to_gid_map.is_some()
                        && info.cid_font_type.as_deref() == Some("CIDFontType2")
                    {
                        // CIDFontType2 (TrueType) with CIDToGIDMap — use direct GID rendering
                        // Note: CIDFontType0 (CFF) requires a CFF parser which ttf-parser doesn't handle
                        // for raw CFF data (FontFile3), so those fall back to system fonts.
                        log::debug!(
                            "Using embedded font '{}' with CIDToGIDMap (CIDFontType2)",
                            info.base_font
                        );
                        Some((None, Arc::clone(embedded), 0, true))
                    } else if info.cff_gid_map.is_some() {
                        // CFF font with byte→GID mapping — use direct rendering
                        log::debug!(
                            "Using embedded CFF font '{}' with direct GID mapping",
                            info.base_font
                        );
                        Some((None, Arc::clone(embedded), 0, true))
                    } else {
                        log::debug!(
                            "Embedded font '{}' lacks usable cmap, falling back to system font",
                            info.base_font
                        );
                        self.load_font_data(&info.base_font)
                            .map(|(id, d, i)| (Some(id), d, i, false))
                    }
                } else {
                    self.load_font_data(&info.base_font)
                        .map(|(id, d, i)| (Some(id), d, i, false))
                }
            } else {
                self.load_font_data(pdf_font_name)
                    .map(|(id, d, i)| (Some(id), d, i, false))
            };

        if let Some((font_id, font_data, index, use_cid_to_gid)) = font_data_and_index {
            if use_cid_to_gid {
                // Direct CIDToGIDMap/CFF rendering — bypass rustybuzz, use ttf-parser for glyph outlines
                match self.render_cid_direct(
                    pixmap,
                    text,
                    font_info.as_deref().unwrap(),
                    &font_data,
                    index,
                    &paint,
                    base_transform,
                    gs,
                    clip_mask,
                ) {
                    Ok(advance) => return Ok(advance),
                    Err(e) => {
                        // Fall back to system font if embedded parsing fails
                        log::warn!(
                            "Direct CID/CFF rendering failed: {}, falling back to system font",
                            e
                        );
                        if let Some((fb_id, fallback_data, fallback_idx)) =
                            self.load_font_data(pdf_font_name)
                        {
                            return self.render_unicode_text(
                                pixmap,
                                &unicode_text,
                                text,
                                font_info.as_deref(),
                                Some(fb_id),
                                fallback_data,
                                fallback_idx,
                                &paint,
                                base_transform,
                                gs,
                                clip_mask,
                                pdf_font_name,
                                false,
                            );
                        }
                    },
                }
            }
            Ok(self.render_unicode_text(
                pixmap,
                &unicode_text,
                text, // raw bytes
                font_info.as_deref(),
                font_id,
                font_data,
                index,
                &paint,
                base_transform,
                gs,
                clip_mask,
                pdf_font_name,
                true, // allow_fallback
            )?)
        } else {
            let font_name = font_info
                .as_ref()
                .map(|i| i.base_font.as_str())
                .unwrap_or("unknown");
            log::warn!(
                "No font found for '{}', text may render incorrectly. \
                 Install common fonts (e.g., liberation-fonts, dejavu-fonts, or noto-fonts).",
                font_name
            );
            // Fallback to simple rendering if font not found
            Ok(self.render_text_fallback(
                pixmap,
                &unicode_text,
                &paint,
                base_transform,
                gs,
                clip_mask,
            )?)
        }
    }

    /// Decode raw PDF text bytes to a Unicode string based on font type.
    fn decode_text_to_unicode(
        &self,
        bytes: &[u8],
        font: Option<&crate::fonts::FontInfo>,
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
                        if char_str != "\u{FFFD}" {
                            result.push_str(&char_str);
                        }
                    }
                }
            } else {
                // Complex font: use unified iterator for robust multi-byte decoding
                for (char_code, _) in TextCharIter::new(bytes, Some(font)) {
                    let char_str = font
                        .char_to_unicode(char_code as u32)
                        .unwrap_or_else(|| fallback_char_to_unicode(char_code as u32));

                    if char_str != "\u{FFFD}" {
                        result.push_str(&char_str);
                    }
                }
            }
            result
        } else {
            // No font - fallback to Latin-1 (ISO 8859-1) encoding
            bytes.iter().map(|&b| char::from(b)).collect()
        };

        // Filter control characters from failed encoding resolution,
        // and expand presentation-form ligature code points (fi, fl, ffi,
        // ffl, st, ct, …) into their component letters so the shaper
        // passes the cluster through as ordinary glyphs instead of
        // dropping it or producing a lone box. `extract_text` already
        // does this on the extraction path via
        // `ligature_processor::get_ligature_components`; without the
        // same decomposition on the render path, words like
        // "Efficient" rasterize as "Effi  ert" because the shaper can't
        // resolve the ligature cluster against the fallback system
        // font. See issue #331 (R2).
        let mut filtered = String::with_capacity(raw_result.len());
        for c in raw_result.chars() {
            if c < '\x20' && c != '\t' && c != '\n' && c != '\r' {
                continue;
            }
            if let Some(components) = crate::text::ligature_processor::get_ligature_components(c) {
                filtered.push_str(components);
            } else {
                filtered.push(c);
            }
        }
        filtered
    }

    /// Render a TJ array (text with positioning adjustments).
    /// Returns the total horizontal advance in PDF points.
    pub fn render_tj_array(
        &self,
        pixmap: &mut Pixmap,
        array: &[TextElement],
        base_transform: Transform,
        gs: &GraphicsState,
        resources: &Object,
        doc: &PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
        font_cache: &HashMap<String, Arc<crate::fonts::FontInfo>>,
    ) -> Result<f32> {
        let mut current_gs = gs.clone();
        let mut total_advance: f32 = 0.0;

        for element in array {
            match element {
                TextElement::String(text) => {
                    let advance = self.render_text(
                        pixmap,
                        text,
                        base_transform,
                        &current_gs,
                        resources,
                        doc,
                        clip_mask,
                        font_cache,
                    )?;

                    // Advance text position in text space: Tm' = T(advance, 0) * Tm
                    let advance_matrix = crate::content::Matrix::translation(advance, 0.0);
                    current_gs.text_matrix = advance_matrix.multiply(&current_gs.text_matrix);
                    total_advance += advance;
                },
                TextElement::Offset(offset) => {
                    // PDF offsets are in 1/1000th of a unit, and positive shifts text to the left
                    let shift = (-offset / 1000.0) * current_gs.font_size;
                    let advance_matrix = crate::content::Matrix::translation(shift, 0.0);
                    current_gs.text_matrix = advance_matrix.multiply(&current_gs.text_matrix);
                    total_advance += shift;
                },
            }
        }
        Ok(total_advance)
    }

    /// Get font info for a specific font name from resources.
    #[allow(dead_code)]
    fn get_font_info(
        &self,
        doc: &PdfDocument,
        resources: &Object,
        font_name: &str,
    ) -> Result<crate::fonts::FontInfo> {
        if let Object::Dictionary(res_dict) = resources {
            if let Some(Object::Dictionary(fonts)) = res_dict.get("Font") {
                if let Some(font_ref) = fonts.get(font_name) {
                    let font_obj = doc.resolve_object(font_ref)?;
                    let info = crate::fonts::FontInfo::from_dict(&font_obj, doc)?;
                    log::debug!("Resolved font '{}': subtype={}, encoding={:?}, has_to_unicode={}, has_embedded={}", 
                        info.base_font, info.subtype, info.encoding, info.to_unicode.is_some(), info.embedded_font_data.is_some());
                    return Ok(info);
                }
            }
        }
        Err(Error::InvalidPdf(format!("Font {} not found", font_name)))
    }

    /// Find and load font data from system. Returns a `fontdb::ID` alongside
    /// the `Arc`-wrapped bytes so callers can look up the parsed-face cache.
    fn load_font_data(&self, pdf_font_name: &str) -> Option<(fontdb::ID, Arc<Vec<u8>>, u32)> {
        // Strip subset prefix (e.g., "ABCDEF+FontName" -> "FontName")
        let clean_name = if let Some(plus_idx) = pdf_font_name.find('+') {
            &pdf_font_name[plus_idx + 1..]
        } else {
            pdf_font_name
        };

        // Handle common CJK names and encoding markers
        let is_cjk_probability = clean_name.contains("GB2312") 
            || clean_name.contains("Identity")
            || clean_name.contains("楷体") 
            || clean_name.contains("æ¥·ä½") // Mojibake variant
            || clean_name.contains("宋体")
            || clean_name.contains("å®\u{008b}ä½") // Mojibake variant
            || clean_name.contains("黑体")
            || clean_name.contains("é»\u{0091}ä½") // Mojibake variant
            || clean_name.contains("FangSong")
            || clean_name.contains("SimSun")
            || clean_name.contains("SimHei")
            || clean_name.contains("KaiTi")
            || pdf_font_name == "F1";

        let final_name = if clean_name.contains("楷体")
            || clean_name.contains("æ¥·ä½")
            || clean_name.contains("KaiTi")
        {
            "KaiTi"
        } else if clean_name.contains("宋体")
            || clean_name.contains("å®\u{008b}ä½")
            || clean_name.contains("SimSun")
        {
            "SimSun"
        } else if clean_name.contains("黑体")
            || clean_name.contains("é»\u{0091}ä½")
            || clean_name.contains("SimHei")
        {
            "SimHei"
        } else {
            clean_name
        };

        // Map well-known PDF/LaTeX font names to system font equivalents
        let mut variants = vec![final_name.to_string()];

        // URW/TeX font mappings to URW base35 system fonts
        if clean_name.contains("URWPalladioL") || clean_name.contains("Palatino") {
            variants.insert(0, "P052".to_string());
            variants.push("Palatino Linotype".to_string());
            variants.push("TeX Gyre Pagella".to_string());
        } else if clean_name.contains("NimbusRomNo9L") || clean_name.contains("NimbusRoman") {
            variants.insert(0, "Nimbus Roman".to_string());
            variants.push("Times New Roman".to_string());
        } else if clean_name.contains("NimbusSanL") || clean_name.contains("NimbusSans") {
            variants.insert(0, "Nimbus Sans".to_string());
            variants.push("Arial".to_string());
        } else if clean_name.contains("NimbusMonL") || clean_name.contains("NimbusMono") {
            variants.insert(0, "Nimbus Mono PS".to_string());
            variants.push("Courier New".to_string());
        } else if clean_name.contains("CMSS")
            || clean_name.contains("CMR")
            || clean_name.contains("CMBX")
        {
            // Computer Modern fonts (LaTeX) — use Latin Modern or serif fallback
            variants.push("Latin Modern Roman".to_string());
            variants.push("Computer Modern".to_string());
        } else if clean_name.contains("URWBookmanL") || clean_name.contains("Bookman") {
            variants.insert(0, "Bookman URW".to_string());
        } else if clean_name.contains("CenturySchL") || clean_name.contains("NewCentury") {
            variants.insert(0, "C059".to_string());
        } else if clean_name.contains("URWChanceryL") || clean_name.contains("Chancery") {
            variants.insert(0, "Z003".to_string());
        }

        if is_cjk_probability {
            variants.push("Noto Sans CJK SC".to_string());
            variants.push("Noto Serif CJK SC".to_string());
            variants.push("WenQuanYi Micro Hei".to_string());
            variants.push("Droid Sans Fallback".to_string());
        }

        // Generic fallbacks — detect serif vs sans-serif
        let is_serif = clean_name.contains("Roman")
            || clean_name.contains("Serif")
            || clean_name.contains("Times")
            || clean_name.contains("Palladio")
            || clean_name.contains("Palatino")
            || clean_name.contains("Bookman")
            || clean_name.contains("Garamond")
            || clean_name.contains("Century")
            || clean_name.contains("Georgia")
            || clean_name.contains("CMR")
            || clean_name.contains("CMBX")
            || clean_name.contains("CMTI");
        if is_serif {
            variants.push("Times New Roman".to_string());
            variants.push("Liberation Serif".to_string());
            variants.push("DejaVu Serif".to_string());
        }
        variants.push("Arial".to_string());
        variants.push("Helvetica".to_string());
        variants.push("Liberation Sans".to_string());
        variants.push("DejaVu Sans".to_string());
        variants.push("Noto Sans".to_string());
        variants.push("FreeSans".to_string());

        let weight = if pdf_font_name.contains("Bold") || pdf_font_name.contains("Black") {
            fontdb::Weight::BOLD
        } else {
            fontdb::Weight::NORMAL
        };

        let style = if pdf_font_name.contains("Italic") || pdf_font_name.contains("Oblique") {
            fontdb::Style::Italic
        } else {
            fontdb::Style::Normal
        };

        for variant in variants {
            let families = [
                fontdb::Family::Name(&variant),
                fontdb::Family::Serif,
                fontdb::Family::SansSerif,
            ];
            let query = fontdb::Query {
                families: &families,
                weight,
                stretch: fontdb::Stretch::Normal,
                style,
            };

            if let Some(id) = self.font_db().query(&query) {
                if let Some((arc_data, index)) = cached_font_bytes(id, self.font_db()) {
                    log::debug!(
                        "Matched system font for {}: variant={}, index={}, size={} bytes",
                        pdf_font_name,
                        variant,
                        index,
                        arc_data.len()
                    );
                    return Some((id, arc_data, index));
                }
            }
        }
        log::debug!(
            "No system font matched for '{}' after trying all fallback variants",
            pdf_font_name
        );
        None
    }

    /// Access the font database.
    fn font_db(&self) -> &fontdb::Database {
        &self.fontdb
    }

    /// Render Unicode text using shaped glyphs.
    /// Returns the total horizontal advance in PDF points.
    fn render_unicode_text(
        &self,
        pixmap: &mut Pixmap,
        text: &str,
        bytes: &[u8],
        font_info: Option<&crate::fonts::FontInfo>,
        font_id: Option<fontdb::ID>,
        font_data: Arc<Vec<u8>>,
        index: u32,
        paint: &Paint,
        base_transform: Transform,
        gs: &GraphicsState,
        clip_mask: Option<&tiny_skia::Mask>,
        pdf_font_name: &str,
        allow_fallback: bool,
    ) -> Result<f32> {
        let font_size = gs.font_size;
        let h_scale = gs.horizontal_scaling / 100.0;

        // 1. Resolve faces — prefer process-wide cache to avoid re-parsing font tables
        //    on every text segment.  Embedded fonts (font_id == None) are not cached
        //    because they are unique per-PDF and typically only rendered once.
        let cached_arc: Option<Arc<CachedFace>> =
            font_id.and_then(|id| cached_face(id, Arc::clone(&font_data), index));

        // Storage for locally-created faces when there is no cache entry
        // (embedded fonts, first-ever render of a system font).
        let _local_rb: Option<rustybuzz::Face<'_>>;
        let _local_ttf: Option<ttf_parser::Face<'_>>;

        let rb_face_ref: &rustybuzz::Face<'_>;
        let ttf_face_ref: &ttf_parser::Face<'_>;
        let units_per_em: f32;

        if let Some(ref c) = cached_arc {
            _local_rb = None;
            _local_ttf = None;
            rb_face_ref = &c.rb_face;
            ttf_face_ref = &c.ttf_face;
            units_per_em = c.units_per_em;
        } else {
            let rb_opt = rustybuzz::Face::from_slice(&font_data, index);
            if rb_opt.is_none() {
                if allow_fallback {
                    log::warn!("Failed to create rustybuzz face from embedded data for '{}', falling back to system font", pdf_font_name);
                    if let Some((fb_id, fallback_data, fallback_index)) =
                        self.load_font_data(pdf_font_name)
                    {
                        return self.render_unicode_text(
                            pixmap,
                            text,
                            bytes,
                            font_info,
                            Some(fb_id),
                            fallback_data,
                            fallback_index,
                            paint,
                            base_transform,
                            gs,
                            clip_mask,
                            pdf_font_name,
                            false, // don't allow infinite fallback
                        );
                    }
                }
                return self.render_text_fallback(
                    pixmap,
                    text,
                    paint,
                    base_transform,
                    gs,
                    clip_mask,
                );
            }
            _local_rb = rb_opt;
            _local_ttf = ttf_parser::Face::parse(&font_data, index).ok();
            if _local_ttf.is_none() {
                return Err(Error::InvalidPdf(format!("Failed to parse font: {}", pdf_font_name)));
            }
            rb_face_ref = _local_rb.as_ref().unwrap();
            ttf_face_ref = _local_ttf.as_ref().unwrap();
            units_per_em = ttf_face_ref.units_per_em() as f32;
        }

        // 2. Buffer setup
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);

        // Explicitly set script and direction for better CJK shaping
        if text
            .chars()
            .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF)
        {
            if let Some(script) = rustybuzz::Script::from_iso15924_tag(
                rustybuzz::ttf_parser::Tag::from_bytes(b"Hani"),
            ) {
                buffer.set_script(script);
            }
        }
        buffer.set_direction(rustybuzz::Direction::LeftToRight);

        // 3. Shape the text
        let glyphs = rustybuzz::shape(rb_face_ref, &[], buffer);
        let info = glyphs.glyph_infos();
        let pos = glyphs.glyph_positions();

        let scale = font_size / units_per_em;
        log::debug!(
            "render_unicode_text: pdf_font={}, units_per_em={}, font_size={}, scale={}",
            pdf_font_name,
            units_per_em,
            font_size,
            scale
        );

        // 4. Transform setup - include full text matrix [Tm]
        let text_transform = Transform::from_row(
            gs.text_matrix.a,
            gs.text_matrix.b,
            gs.text_matrix.c,
            gs.text_matrix.d,
            gs.text_matrix.e,
            gs.text_matrix.f,
        );
        // Transform from text space to pixel space: P_pixel = base_transform * text_transform * P_text
        let combined_base = base_transform.pre_concat(text_transform);

        let mut x_cursor: f32 = 0.0; // In text space units
        let mut last_fallback_cluster: Option<usize> = None;

        // Pre-resolve CIDs for Type0 fonts using our iterator
        let cids: Vec<u16> = if let Some(info) = font_info {
            if info.subtype == "Type0" {
                TextCharIter::new(bytes, Some(info))
                    .map(|(cid, _)| cid)
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Build mapping from Unicode byte offset → character index for correct CID lookup.
        // Rustybuzz clusters are byte offsets into the Unicode string, but we need
        // the character index to map to the corresponding CID.
        let cluster_to_char_idx: HashMap<usize, usize> = text
            .char_indices()
            .enumerate()
            .map(|(char_idx, (byte_offset, _))| (byte_offset, char_idx))
            .collect();

        // 5. Iterate through shaped glyphs
        for i in 0..info.len() {
            let glyph_id = info[i].glyph_id;
            let cluster = info[i].cluster as usize;

            // Get character at this cluster (byte offset)
            let char_at_pos = text[cluster..].chars().next().unwrap_or(' ');

            // Map cluster (Unicode byte offset) to character index
            let char_idx = cluster_to_char_idx.get(&cluster).copied().unwrap_or(0);

            // Determine how many *source* characters this glyph represents.
            // For normal 1:1 glyphs, cluster_chars == 1. For shaped
            // ligatures like the "ffi" glyph (#331 R2), one glyph covers
            // multiple characters and rustybuzz reports them with the
            // same cluster index on every glyph of the cluster. Since we
            // advance the output cursor by the sum of the PDF-declared
            // widths of the *source* characters (per PDF §9.2.4 text-
            // showing advance), we must add the widths of every source
            // character in the ligature cluster to the cursor, not just
            // the first character's width. Otherwise a ligature glyph
            // draws wide but only advances by one character's worth, and
            // subsequent glyphs overwrite the tail of the ligature —
            // exactly the `Efficient` → `Effi ert` symptom reported in
            // #331 on arxiv-style LaTeX-embedded fonts.
            let next_cluster_byte: usize = info
                .get(i + 1)
                .map(|n| n.cluster as usize)
                .unwrap_or(text.len());
            let cluster_chars: usize = text[cluster..next_cluster_byte.min(text.len())]
                .chars()
                .count()
                .max(1);

            // PDF Spec: tx = ((w0 * Tfs) + Tc + Tw) * Th
            // Priority:
            // 1. Explicit /W or /DW from FontInfo (in 1000ths of em),
            //    summed across every source character in the cluster
            //    so ligatures advance by the full cluster's width.
            // 2. Shaped advance from rustybuzz (fallback, already
            //    reflects the ligature's real width because it comes
            //    from the font's horizontal metrics table).
            let pdf_width = if let Some(font_info_ref) = font_info {
                let mut sum = 0.0_f32;
                for k in 0..cluster_chars {
                    let idx = char_idx + k;
                    let char_code = if font_info_ref.subtype == "Type0" {
                        *cids.get(idx).unwrap_or(&0)
                    } else {
                        *bytes.get(idx).unwrap_or(&0) as u16
                    };
                    sum += font_info_ref.get_glyph_width(char_code);
                }
                sum
            } else {
                // No FontInfo, use shaped advance
                pos[i].x_advance as f32 / font_size * 1000.0
            };

            let x_advance = pdf_width * font_size / 1000.0;
            let x_offset = pos[i].x_offset as f32 / units_per_em * font_size;
            let y_offset = pos[i].y_offset as f32 / units_per_em * font_size;

            let mut x_advance_override: Option<f32> = None;

            // Try to get glyph from primary font
            let mut pb = PathBuilder::new();
            let mut builder = SkiaOutlineBuilder(&mut pb);
            let mut has_outline = ttf_face_ref
                .outline_glyph(ttf_parser::GlyphId(glyph_id as u16), &mut builder)
                .is_some();

            if has_outline && glyph_id != 0 {
                if let Some(path) = pb.finish() {
                    let glyph_transform = combined_base
                        .pre_translate((x_cursor + x_offset) * h_scale, y_offset + gs.text_rise)
                        .pre_scale(scale, scale);

                    pixmap.fill_path(
                        &path,
                        paint,
                        tiny_skia::FillRule::Winding,
                        glyph_transform,
                        clip_mask,
                    );
                }
            } else {
                // FALLBACK PATH: If primary font fails, use the cluster offset to find the original character
                // char_at_pos already retrieved above using byte offset

                // Skip empty glyphs for spaces
                if char_at_pos.is_whitespace() {
                    x_cursor += x_advance;
                    x_cursor += gs.char_space;
                    if char_at_pos == ' ' {
                        x_cursor += gs.word_space;
                    }
                    continue;
                }

                // IMPORTANT: Only render fallback character ONCE per cluster
                if last_fallback_cluster == Some(cluster) {
                    x_cursor += x_advance;
                    continue;
                }
                last_fallback_cluster = Some(cluster);

                // Try to find character in fallback CJK fonts.
                // get_cjk_fallback_cached() hits a process-wide OnceLock after the
                // first call — no fontdb queries or font clones on subsequent glyphs.
                if let Some((cjk_id, cjk_arc, cjk_index)) = get_cjk_fallback_cached(self.font_db())
                {
                    if let Some(cjk_cached) = cached_face(cjk_id, cjk_arc, cjk_index) {
                        if let Some(cjk_glyph_id) = cjk_cached.ttf_face.glyph_index(char_at_pos) {
                            let mut cjk_pb = PathBuilder::new();
                            let mut cjk_builder = SkiaOutlineBuilder(&mut cjk_pb);
                            if cjk_cached
                                .ttf_face
                                .outline_glyph(cjk_glyph_id, &mut cjk_builder)
                                .is_some()
                            {
                                if let Some(cjk_path) = cjk_pb.finish() {
                                    let cjk_scale = font_size / cjk_cached.units_per_em;
                                    let cjk_transform = combined_base
                                        .pre_translate(
                                            (x_cursor + x_offset) * h_scale,
                                            y_offset + gs.text_rise,
                                        )
                                        .pre_scale(cjk_scale, -cjk_scale);
                                    pixmap.fill_path(
                                        &cjk_path,
                                        paint,
                                        tiny_skia::FillRule::Winding,
                                        cjk_transform,
                                        clip_mask,
                                    );
                                    has_outline = true;

                                    if let Some(adv) =
                                        cjk_cached.ttf_face.glyph_hor_advance(cjk_glyph_id)
                                    {
                                        x_advance_override =
                                            Some(adv as f32 / cjk_cached.units_per_em * font_size);
                                    }
                                }
                            }
                        }
                    }
                }

                if !has_outline {
                    log::debug!(
                        "No glyph outline found for char='{}' (0x{:X})",
                        char_at_pos,
                        char_at_pos as u32
                    );
                }
            }

            // Advance cursor in text space
            // PDF spec: tx = ((w0 * Tfs) + Tc + Tw) * Th
            // Note: x_advance already includes w0 * Tfs
            x_cursor += x_advance_override.unwrap_or(x_advance);

            // Add character spacing (Tc)
            x_cursor += gs.char_space;

            if char_at_pos == ' ' {
                // Add word spacing (Tw) for space characters
                x_cursor += gs.word_space;
            }
        }

        Ok(x_cursor)
    }
    /// Render text using direct CID-to-GID mapping, bypassing rustybuzz shaping.
    /// Used for CID subset fonts that have embedded data but no usable Unicode cmap.
    /// Per PDF spec section 9.7.4, CIDToGIDMap maps CIDs to glyph indices in the TrueType font.
    fn render_cid_direct(
        &self,
        pixmap: &mut Pixmap,
        bytes: &[u8],
        font_info: &crate::fonts::FontInfo,
        font_data: &[u8],
        index: u32,
        paint: &Paint,
        base_transform: Transform,
        gs: &GraphicsState,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<f32> {
        let font_size = gs.font_size;
        let h_scale = gs.horizontal_scaling / 100.0;

        let ttf_face = ttf_parser::Face::parse(font_data, index)
            .map_err(|e| Error::InvalidPdf(format!("Failed to parse embedded font: {}", e)))?;
        let units_per_em = ttf_face.units_per_em() as f32;
        let scale = font_size / units_per_em;

        let text_transform = Transform::from_row(
            gs.text_matrix.a,
            gs.text_matrix.b,
            gs.text_matrix.c,
            gs.text_matrix.d,
            gs.text_matrix.e,
            gs.text_matrix.f,
        );
        let combined_base = base_transform.pre_concat(text_transform);

        let mut x_cursor: f32 = 0.0;

        // Iterate over character codes from the raw bytes
        for (char_code, _bytes_consumed) in TextCharIter::new(bytes, Some(font_info)) {
            // Map character code to GID based on font type:
            // - CIDFontType2: CIDToGIDMap maps CID → GID
            // - CFF simple font: cff_gid_map maps byte → GID
            // - Simple TrueType: consult the embedded font's cmap directly
            //   (the PDF content byte is the cmap input under the font's
            //   declared encoding; ISO 32000-1 §9.6.6.4).
            // - Default: identity mapping
            let gid = if let Some(cff_map) = &font_info.cff_gid_map {
                *cff_map.get(&(char_code as u8)).unwrap_or(&0)
            } else if font_info.subtype != "Type0" && font_info.cid_to_gid_map.is_none() {
                cmap_byte_to_gid(&ttf_face, char_code as u8).unwrap_or(0)
            } else {
                match &font_info.cid_to_gid_map {
                    Some(crate::fonts::CIDToGIDMap::Identity) => char_code,
                    Some(crate::fonts::CIDToGIDMap::Explicit(map)) => {
                        *map.get(char_code as usize).unwrap_or(&0)
                    },
                    None => char_code,
                }
            };
            let cid = char_code; // For width lookup

            // Get width from PDF metrics
            let pdf_width = font_info.get_glyph_width(cid);
            let x_advance = pdf_width * font_size / 1000.0;

            // Get Unicode character for space/word-space detection.
            // Use '\0' as the sentinel for "no mapping" so that bytes without a
            // Unicode entry (e.g. ligatures and accented chars in symbolic TrueType
            // fonts that use the Mac Roman cmap path) are not silently treated as
            // spaces and dropped from the rendered output.
            let char_str = font_info.char_to_unicode(cid as u32).unwrap_or_default();
            let char_at_pos = char_str.chars().next().unwrap_or('\0');

            // Draw glyph outline
            if gid != 0 || char_at_pos.is_whitespace() {
                if !char_at_pos.is_whitespace() {
                    let mut pb = PathBuilder::new();
                    let mut builder = SkiaOutlineBuilder(&mut pb);
                    if ttf_face
                        .outline_glyph(ttf_parser::GlyphId(gid), &mut builder)
                        .is_some()
                    {
                        if let Some(path) = pb.finish() {
                            let glyph_transform = combined_base
                                .pre_translate(x_cursor * h_scale, gs.text_rise)
                                .pre_scale(scale, scale);
                            pixmap.fill_path(
                                &path,
                                paint,
                                tiny_skia::FillRule::Winding,
                                glyph_transform,
                                clip_mask,
                            );
                        }
                    }
                }
            }

            x_cursor += x_advance;
            x_cursor += gs.char_space;
            if char_at_pos == ' ' {
                x_cursor += gs.word_space;
            }
        }

        Ok(x_cursor)
    }

    /// Fallback simple rendering if no font found.
    /// Returns the total horizontal advance in PDF points.
    fn render_text_fallback(
        &self,
        pixmap: &mut Pixmap,
        text: &str,
        paint: &Paint,
        base_transform: Transform,
        gs: &GraphicsState,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<f32> {
        // Just draw rectangles for now as very last resort
        let font_size = gs.font_size;
        let char_width = font_size * 0.6;
        let mut x_cursor: f32 = 0.0;
        let h_scale = gs.horizontal_scaling / 100.0;

        let text_transform = Transform::from_row(
            gs.text_matrix.a,
            gs.text_matrix.b,
            gs.text_matrix.c,
            gs.text_matrix.d,
            gs.text_matrix.e,
            gs.text_matrix.f,
        );
        let transform = base_transform.pre_concat(text_transform);

        for c in text.chars() {
            if !c.is_whitespace() {
                let mut pb = PathBuilder::new();
                if let Some(rect) = tiny_skia::Rect::from_xywh(
                    x_cursor * h_scale,
                    0.0,
                    char_width * 0.8,
                    font_size * 0.8,
                ) {
                    pb.push_rect(rect);
                    if let Some(path) = pb.finish() {
                        pixmap.fill_path(
                            &path,
                            paint,
                            tiny_skia::FillRule::Winding,
                            transform,
                            clip_mask,
                        );
                    }
                }
            }

            x_cursor += (char_width + gs.char_space) / h_scale;
            if c == ' ' {
                x_cursor += gs.word_space / h_scale;
            }
        }

        Ok(x_cursor * h_scale)
    }
}

/// Byte grouping mode for CID font character code decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteMode {
    /// Single-byte codes (simple fonts, some predefined CMaps)
    OneByte,
    /// Always 2-byte codes (Identity-H/V, UCS2)
    TwoByte,
    /// Shift-JIS variable-width (1 or 2 bytes depending on lead byte)
    ShiftJIS,
}

/// Get byte grouping mode for a font.
fn get_byte_mode(font: Option<&crate::fonts::FontInfo>) -> ByteMode {
    if let Some(font) = font {
        if font.subtype == "Type0" {
            match &font.encoding {
                crate::fonts::Encoding::Identity => ByteMode::TwoByte,
                crate::fonts::Encoding::Standard(name) => {
                    if (name.contains("Identity") && !name.contains("OneByteIdentity"))
                        || name.contains("UCS2")
                        || name.contains("UTF16")
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

/// Iterator over characters in a PDF string based on font encoding.
struct TextCharIter<'a> {
    bytes: &'a [u8],
    byte_mode: ByteMode,
    index: usize,
}

impl<'a> TextCharIter<'a> {
    fn new(bytes: &'a [u8], font: Option<&crate::fonts::FontInfo>) -> Self {
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

/// Fallback function to map common character codes to Unicode when ToUnicode CMap fails.
fn fallback_char_to_unicode(char_code: u32) -> String {
    match char_code {
        0x2014 => "—".to_string(),
        0x2013 => "–".to_string(),
        0x2018 => "\u{2018}".to_string(),
        0x2019 => "\u{2019}".to_string(),
        0x201C => "\u{201C}".to_string(),
        0x201D => "\u{201D}".to_string(),
        0x2022 => "•".to_string(),
        0x2026 => "…".to_string(),
        0x00B0 => "°".to_string(),
        0x00B1 => "±".to_string(),
        0x00D7 => "×".to_string(),
        0x00F7 => "÷".to_string(),
        0x2202 => "∂".to_string(),
        0x2207 => "∇".to_string(),
        0x220F => "∏".to_string(),
        0x2211 => "∑".to_string(),
        0x221A => "√".to_string(),
        0x221E => "∞".to_string(),
        0x2260 => "≠".to_string(),
        0x2261 => "≡".to_string(),
        0x2264 => "≤".to_string(),
        0x2265 => "≥".to_string(),
        code => {
            if let Some(ch) = char::from_u32(code) {
                ch.to_string()
            } else {
                "\u{FFFD}".to_string()
            }
        },
    }
}

impl Default for TextRasterizer {
    fn default() -> Self {
        Self::new()
    }
}
