//! Undecodable text layer diagnostic.
//!
//! A common subsetting pattern produces a Type0/Identity-H font whose text
//! layer carries no glyph→Unicode information at all: `CIDSystemInfo`
//! ordering `Identity`, `CIDToGIDMap /Identity` (content-stream codes are raw
//! glyph indices), no `/ToUnicode` CMap, and an embedded CIDFontType2 program
//! stripped of both its `cmap` and `post` tables. The page renders fine, but
//! no conformant extractor can recover the text — every mapping path in ISO
//! 32000-1 §9.10.2 is severed. pdf_oxide (like every other reader) falls back
//! to CID-as-Unicode, which faithfully echoes the raw glyph indices.
//!
//! These tests pin the diagnostic for that condition:
//! - `FontInfo::has_undecodable_text_layer()` fires on exactly the severed
//!   shape above, and stays silent on every recoverable variant — a usable
//!   embedded `cmap` (the recoverable byte-as-GID subset shape), usable `post` glyph
//!   names, or a usable `/ToUnicode`.
//! - Extraction emits a structured `undecodable_text_layer` warning (page-
//!   scoped, once per font) while leaving the text output byte-identical —
//!   the raw CID sequence stays available to callers who want it.
//!
//! Fixtures: `tests/fixtures/fonts/synthetic_severed_nocmap_nopost.ttf` and
//! `synthetic_severed_nocmap_withpost.ttf` are fully synthetic (fontTools
//! `FontBuilder`, 91 blank glyphs `.notdef`,`g1`..`g90`, unitsPerEm 1000,
//! advance 600), carrying only `glyf`/`loca`/`head`/`hhea`/`hmtx`/`maxp`
//! (+ format-2 `post` for the second). No third-party document or glyph
//! outlines are involved; the PDFs are built in-test.

use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::fonts::{CIDSystemInfo, CIDToGIDMap, Encoding, FontInfo};
use pdf_oxide::PdfDocument;
use std::collections::HashMap;
use std::sync::Arc;

const SEVERED_TTF: &str = "tests/fixtures/fonts/synthetic_severed_nocmap_nopost.ttf";
const POST_ONLY_TTF: &str = "tests/fixtures/fonts/synthetic_severed_nocmap_withpost.ttf";
const CMAP_TTF: &str = "tests/fixtures/fonts/DejaVuSans.ttf";
const CMAP_NOPOST_TTF: &str = "tests/fixtures/fonts/synthetic_cmap_nopost.ttf";

/// A `FontInfo` in the exact severed shape: Type0 / Identity encoding /
/// Identity ordering / CIDToGIDMap Identity / no ToUnicode, with the given
/// embedded font program.
fn identity_font_with_program(font_data: Option<Vec<u8>>) -> FontInfo {
    FontInfo {
        base_font: "AAAAAA+Severed".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: std::sync::OnceLock::new(),
        embedded_glyph_names: std::sync::OnceLock::new(),
        is_truetype_font: font_data.is_some(),
        embedded_font_data: font_data.map(Arc::new),
        cid_to_gid_map: Some(CIDToGIDMap::Identity),
        cid_system_info: Some(CIDSystemInfo {
            registry: "Adobe".to_string(),
            ordering: "Identity".to_string(),
            supplement: 0,
        }),
        cid_font_type: None,
        cid_widths: None,
        cid_default_width: 600.0,
        has_explicit_dw: true,
        font_weight: None,
        flags: None,
        stem_v: None,
        ascent: 0.8,
        descent: -0.2,
        widths: None,
        first_char: None,
        last_char: None,
        font_matrix_a: 0.001,
        default_width: 600.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
        diff_glyph_names: HashMap::new(),
        wmode: 0,
        cid_vertical_metrics: None,
        cid_default_vertical_metrics: pdf_oxide::fonts::VerticalMetrics::SPEC_DEFAULT,
        cjk_substitution: None,
        type0_unicode_memo: Arc::new(std::sync::Mutex::new(HashMap::new())),
    }
}

fn read_fixture(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

// ---------------------------------------------------------------------------
// Predicate unit tests
// ---------------------------------------------------------------------------

#[test]
fn predicate_fires_on_fully_severed_font() {
    let font = identity_font_with_program(Some(read_fixture(SEVERED_TTF)));
    assert!(
        font.has_undecodable_text_layer(),
        "Type0/Identity-H, Identity ordering, no ToUnicode, embedded program \
         without cmap/post must be reported undecodable"
    );
}

#[test]
fn predicate_silent_when_embedded_cmap_present() {
    // The recoverable byte-as-GID subset shape: the embedded program
    // retains a usable `cmap`, so GID→Unicode inversion works.
    let font = identity_font_with_program(Some(read_fixture(CMAP_TTF)));
    assert!(
        !font.has_undecodable_text_layer(),
        "a font with a usable embedded cmap is recoverable — no diagnostic"
    );
}

#[test]
fn predicate_silent_when_post_glyph_names_present() {
    let font = identity_font_with_program(Some(read_fixture(POST_ONLY_TTF)));
    assert!(
        !font.has_undecodable_text_layer(),
        "a font with post glyph names retains a recovery path — no diagnostic"
    );
}

#[test]
fn predicate_silent_without_embedded_program() {
    let font = identity_font_with_program(None);
    assert!(
        !font.has_undecodable_text_layer(),
        "without an embedded program the severed condition cannot be proven"
    );
    let empty = identity_font_with_program(Some(Vec::new()));
    assert!(!empty.has_undecodable_text_layer(), "zero-byte program is not provably severed");
}

#[test]
fn predicate_silent_for_explicit_cid_to_gid_stream() {
    let mut font = identity_font_with_program(Some(read_fixture(SEVERED_TTF)));
    font.cid_to_gid_map = Some(CIDToGIDMap::Explicit(vec![0, 1, 2]));
    assert!(
        !font.has_undecodable_text_layer(),
        "diagnostic is scoped to CIDToGIDMap /Identity (codes are raw GIDs)"
    );
}

#[test]
fn predicate_silent_for_non_identity_ordering() {
    let mut font = identity_font_with_program(Some(read_fixture(SEVERED_TTF)));
    font.cid_system_info = Some(CIDSystemInfo {
        registry: "Adobe".to_string(),
        ordering: "Japan1".to_string(),
        supplement: 6,
    });
    assert!(
        !font.has_undecodable_text_layer(),
        "a known collection ordering has a predefined CID→Unicode path"
    );
    font.cid_system_info = None;
    assert!(
        !font.has_undecodable_text_layer(),
        "missing CIDSystemInfo is not the proven-severed shape"
    );
}

#[test]
fn predicate_silent_when_descendant_program_kept_cmap_but_accessor_missed_it() {
    // A CIDFontType2 descendant embedded via /FontFile3 (OpenType wrapper)
    // never sets `is_truetype_font`, so the lazy cmap accessor bails without
    // inspecting the program. The predicate must still see the live cmap in
    // the table directory and stay silent — the file is not severed.
    let mut font = identity_font_with_program(Some(read_fixture(CMAP_NOPOST_TTF)));
    font.is_truetype_font = false;
    font.cid_font_type = Some("CIDFontType2".to_string());
    assert!(
        !font.has_undecodable_text_layer(),
        "a program whose table directory contains a cmap is never severed, \
         even when the decode cascade's accessor did not read it"
    );
}

#[test]
fn predicate_silent_when_program_is_unparseable() {
    // Garbage bytes: severedness cannot be proven from an unreadable program.
    let font = identity_font_with_program(Some(vec![0xDE; 256]));
    assert!(!font.has_undecodable_text_layer());
}

#[test]
fn predicate_fires_when_cid_to_gid_map_is_absent() {
    // /CIDToGIDMap is optional; per the spec its absence means Identity, so
    // an otherwise-severed font without the key is still severed.
    let mut font = identity_font_with_program(Some(read_fixture(SEVERED_TTF)));
    font.cid_to_gid_map = None;
    assert!(font.has_undecodable_text_layer());
}

#[test]
fn predicate_silent_for_simple_fonts() {
    let mut font = identity_font_with_program(Some(read_fixture(SEVERED_TTF)));
    font.subtype = "TrueType".to_string();
    font.encoding = Encoding::Standard("WinAnsiEncoding".to_string());
    assert!(!font.has_undecodable_text_layer());
}

#[test]
fn warning_category_string_is_stable() {
    assert_eq!(WarningCategory::UndecodableTextLayer.as_str(), "undecodable_text_layer");
}

// ---------------------------------------------------------------------------
// Extraction integration tests
// ---------------------------------------------------------------------------

/// Wrap raw bytes as a `<< /Length … >> stream … endstream` object body.
fn stream_obj(bytes: &[u8]) -> Vec<u8> {
    let mut v = format!("<< /Length {} >>\nstream\n", bytes.len()).into_bytes();
    v.extend_from_slice(bytes);
    v.extend_from_slice(b"\nendstream");
    v
}

/// Wrap an embedded font program as a `/Length1`-carrying stream object body.
fn font_stream_obj(font_bytes: &[u8]) -> Vec<u8> {
    let mut v =
        format!("<< /Length {} /Length1 {} >>\nstream\n", font_bytes.len(), font_bytes.len())
            .into_bytes();
    v.extend_from_slice(font_bytes);
    v.extend_from_slice(b"\nendstream");
    v
}

/// Assemble numbered objects into a complete PDF (header, xref, trailer).
fn assemble_pdf(objs: &[Vec<u8>]) -> Vec<u8> {
    let mut out: Vec<u8> = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF",
            objs.len() + 1
        )
        .as_bytes(),
    );
    out
}

/// The Type0 → CIDFontType2 → FontDescriptor → FontFile2 object chain for
/// the severed Identity-H font, numbered from `first`: the Type0 wrapper at
/// `first`, descendant at `first + 1`, descriptor at `first + 2`, and the
/// font program at `first + 3`.
fn identity_h_font_objs(
    font_bytes: &[u8],
    first: usize,
    tounicode_ref: Option<usize>,
) -> Vec<Vec<u8>> {
    let basefont = "AAAAAA+Severed";
    let tounicode_entry = tounicode_ref
        .map(|r| format!(" /ToUnicode {r} 0 R"))
        .unwrap_or_default();
    vec![
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{basefont} /Encoding /Identity-H \
             /DescendantFonts [{} 0 R]{tounicode_entry} >>",
            first + 1
        )
        .into_bytes(),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{basefont} \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor {} 0 R /DW 600 /CIDToGIDMap /Identity >>",
            first + 2
        )
        .into_bytes(),
        format!(
            "<< /Type /FontDescriptor /FontName /{basefont} /Flags 4 \
             /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
             /CapHeight 700 /StemV 80 /FontFile2 {} 0 R >>",
            first + 3
        )
        .into_bytes(),
        font_stream_obj(font_bytes),
    ]
}

/// Build a one-page PDF showing `<00480049> Tj` ("HI" when the raw GIDs are
/// read as codepoints) in a Type0/Identity-H font embedding `font_bytes`.
/// Mirrors the in-the-wild reproducer for the severed pattern, minus the
/// third-party font.
fn build_identity_h_pdf(font_bytes: &[u8], with_tounicode: bool) -> Vec<u8> {
    let content = b"BT /F1 24 Tf 72 700 Td <00480049> Tj ET";

    let cmap = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         2 beginbfchar\n<0048> <0048>\n<0049> <0049>\nendbfchar\n\
         endcmap\nend\nend";

    let mut objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_obj(content),
    ];
    // Font chain at objects 5-8; the ToUnicode CMap stream, when present,
    // follows at object 9.
    objs.extend(identity_h_font_objs(font_bytes, 5, with_tounicode.then_some(9)));
    if with_tounicode {
        objs.push(stream_obj(cmap.as_bytes()));
    }
    assemble_pdf(&objs)
}

/// Two pages: page 0 has no text at all; page 1 shows `<00480049> Tj` in the
/// severed Identity-H font. Used to prove page attribution of the warning.
fn build_two_page_identity_h_pdf(font_bytes: &[u8]) -> Vec<u8> {
    let content = b"BT /F1 24 Tf 72 700 Td <00480049> Tj ET";

    let mut objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>"
            .to_vec(),
        stream_obj(b""),
        stream_obj(content),
    ];
    // Font chain at objects 7-10.
    objs.extend(identity_h_font_objs(font_bytes, 7, None));
    assemble_pdf(&objs)
}

fn undecodable_warnings(doc: &PdfDocument) -> Vec<pdf_oxide::extractors::warnings::Warning> {
    doc.structured_warnings()
        .into_iter()
        .filter(|w| w.category == WarningCategory::UndecodableTextLayer)
        .collect()
}

#[test]
fn severed_page_keeps_raw_cid_text_and_emits_warning() {
    let pdf = build_identity_h_pdf(&read_fixture(SEVERED_TTF), false);
    let doc = PdfDocument::from_bytes(pdf).expect("parse pdf");
    let text = doc.extract_text(0).expect("extract");

    // Text output is untouched: the raw glyph indices 0x48 0x49 echo as "HI".
    assert_eq!(text.trim(), "HI", "text output must remain the raw CID sequence");

    let warnings = undecodable_warnings(&doc);
    assert_eq!(
        warnings.len(),
        1,
        "exactly one undecodable_text_layer warning per page/font, got: {warnings:?}"
    );
    let w = &warnings[0];
    assert_eq!(w.page, Some(0), "warning must carry the page index");
    assert!(
        w.message.contains("AAAAAA+Severed"),
        "warning must name the font: {}",
        w.message
    );
    assert_eq!(w.spec_section, Some("9.10.2"));

    // A second pass over the same page states the same fact — it must not
    // produce a second warning.
    let _ = doc.extract_text(0).expect("second extract");
    assert_eq!(
        undecodable_warnings(&doc).len(),
        1,
        "re-extracting the page must not duplicate the warning"
    );
}

#[test]
fn recoverable_embedded_cmap_page_stays_silent() {
    // Same document shape, but the embedded program keeps its cmap (the
    // byte-as-GID recovery). The diagnostic must not fire.
    let pdf = build_identity_h_pdf(&read_fixture(CMAP_TTF), false);
    let doc = PdfDocument::from_bytes(pdf).expect("parse pdf");
    let _ = doc.extract_text(0).expect("extract");
    assert!(
        undecodable_warnings(&doc).is_empty(),
        "recoverable byte-as-GID font must not be flagged undecodable"
    );
}

#[test]
fn warning_page_attribution_is_exact() {
    // The severed font is used only on page 1. Every page-parameterized
    // extraction API must attribute the warning to that page — never to a
    // defaulted page 0 — and the same page/font fact reported through
    // different APIs stays a single warning.
    let pdf = build_two_page_identity_h_pdf(&read_fixture(SEVERED_TTF));
    let doc = PdfDocument::from_bytes(pdf).expect("parse pdf");

    let _ = doc.extract_chars(1).expect("chars");
    let warnings = undecodable_warnings(&doc);
    assert_eq!(warnings.len(), 1, "char-level API surfaces the warning: {warnings:?}");
    assert_eq!(warnings[0].page, Some(1), "char-level API attributes the true page");

    let _ = doc
        .extract_spans_with_config(1, pdf_oxide::extractors::SpanMergingConfig::default())
        .expect("spans");
    let _ = doc.extract_text(0).expect("page 0");
    let _ = doc.extract_text(1).expect("page 1");
    let warnings = undecodable_warnings(&doc);
    assert_eq!(
        warnings.len(),
        1,
        "one warning for the one severed page, across all APIs: {warnings:?}"
    );
    assert_eq!(warnings[0].page, Some(1));
}

#[test]
fn empty_tounicode_page_still_fires() {
    // A present-but-empty /ToUnicode maps nothing — the decode cascade
    // treats it as absent, and so must the diagnostic.
    let empty_cmap = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         endcmap\nend\nend";
    let font_bytes = read_fixture(SEVERED_TTF);
    let content = b"BT /F1 24 Tf 72 700 Td <00480049> Tj ET";
    let mut objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_obj(content),
    ];
    objs.extend(identity_h_font_objs(&font_bytes, 5, Some(9)));
    objs.push(stream_obj(empty_cmap.as_bytes()));
    let doc = PdfDocument::from_bytes(assemble_pdf(&objs)).expect("parse pdf");
    let text = doc.extract_text(0).expect("extract");
    assert_eq!(text.trim(), "HI", "raw CID echo is unchanged");
    assert_eq!(
        undecodable_warnings(&doc).len(),
        1,
        "empty ToUnicode does not mask the condition"
    );
}

#[test]
fn tounicode_page_stays_silent() {
    // A /ToUnicode CMap maps the codes; the layer is decodable.
    let pdf = build_identity_h_pdf(&read_fixture(SEVERED_TTF), true);
    let doc = PdfDocument::from_bytes(pdf).expect("parse pdf");
    let text = doc.extract_text(0).expect("extract");
    assert_eq!(text.trim(), "HI");
    assert!(
        undecodable_warnings(&doc).is_empty(),
        "a font with a usable ToUnicode CMap must not be flagged undecodable"
    );
}
