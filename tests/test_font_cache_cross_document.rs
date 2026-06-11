//! Cross-document font-cache collision regression (completes #595, #597, #598).
//!
//! The process-global font cache (`fonts::global_cache`) is keyed by a font
//! *identity hash*. The #595 hardening folds the `/ToUnicode` *reference*
//! (object id/gen) into that hash and keeps *canonical* subset fonts
//! (`AAAAAA+`) out of the cache. A non-canonical subset tag such as
//! `/CIDFont+F1` falls outside that exclusion, and template-emitted PDFs reuse
//! the same `/ToUnicode` object number, so the reference-keyed hash can still
//! match for two genuinely different fonts — the later document is then served
//! the earlier font's parsed `FontInfo`, and its glyphs decode through the
//! wrong `/ToUnicode` and come out garbled. Folding the stream's bytes (not
//! just its reference) distinguishes them and closes this case.
//!
//! Both PDFs here are built in memory (per the repo's no-binary-fixtures
//! convention) and are byte-for-byte identical except for the CID→Unicode
//! mapping: same `/BaseFont` (`/CIDFont+F1`, the non-canonical subset tag some
//! real generators emit), same object numbers, same width metrics — only the
//! `/ToUnicode` stream and the matching content-stream CIDs differ. That is the
//! exact shape that triggered the leak.
//!
//! Oracle: correct text contains the header `SUMMARY`; a font decoded through
//! another document's `/ToUnicode` does not.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::fonts::global_cache::{clear_global_font_cache, global_font_cache_stats};
use std::sync::Mutex;

/// Serializes the two tests in this binary: both assert against the
/// process-global cache, so they must not run concurrently.
static CACHE_LOCK: Mutex<()> = Mutex::new(());

/// Lines rendered on the single page. The content is fabricated and trivial;
/// only the presence of `SUMMARY` matters to the oracle.
const LINES: &[&str] = &[
    "SUMMARY",
    "Synthetic document for the font-cache regression.",
    "Text is recoverable only via the ToUnicode CMap.",
];

/// Build a minimal non-embedded Type0/Identity-H PDF in memory.
///
/// Every document shares one object layout and `/BaseFont` name, so their cheap
/// identity hashes collide. `cid_base` shifts the (otherwise sequential) glyph
/// indices, mirroring a real subset font whose CIDs are arbitrary indices
/// unrelated to Unicode and recoverable only through `/ToUnicode`. Two
/// documents built with different `cid_base` therefore carry byte-different
/// `/ToUnicode` streams and content-stream CIDs while remaining identical in
/// every field the pre-fix key looked at.
fn build_type0_pdf(cid_base: u16, cid_to_gid: Option<&[u8]>) -> Vec<u8> {
    // Distinct characters in first-appearance order; CID = cid_base + index.
    let mut chars: Vec<char> = Vec::new();
    for ch in LINES.iter().flat_map(|l| l.chars()) {
        if !chars.contains(&ch) {
            chars.push(ch);
        }
    }
    let cid = |ch: char| -> u16 {
        let idx = chars.iter().position(|&c| c == ch).unwrap() as u16;
        cid_base + idx
    };

    // Content stream: 2-byte CIDs, one `Tj` per line.
    let mut content = String::from("BT\n/F1 13 Tf\n15 TL\n40 770 Td\n");
    for line in LINES {
        let hex: String = line.chars().map(|ch| format!("{:04X}", cid(ch))).collect();
        content.push_str(&format!("<{hex}> Tj\nT*\n"));
    }
    content.push_str("ET");

    // ToUnicode CMap inverting the CID→Unicode mapping.
    let bfchar: String = chars
        .iter()
        .map(|&ch| format!("<{:04X}> <{:04X}>", cid(ch), ch as u32))
        .collect::<Vec<_>>()
        .join("\n");
    let cmap = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{}\nendbfchar\n\
         endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend",
        chars.len(),
        bfchar
    );

    // /CIDToGIDMap defaults to the `/Identity` name; `cid_to_gid` switches it to
    // the stream form (object 9) so a test can vary its bytes.
    let cid_to_gid_entry = if cid_to_gid.is_some() {
        "/CIDToGIDMap 9 0 R"
    } else {
        "/CIDToGIDMap /Identity"
    };

    let mut objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()).into_bytes(),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /CIDFont+F1 /Encoding /Identity-H \
          /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
            .to_vec(),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /CIDFont+F1 \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 7 0 R /DW 500 {cid_to_gid_entry} >>"
        )
        .into_bytes(),
        // No /FontFile* — non-embedded, like the real garbled documents.
        b"<< /Type /FontDescriptor /FontName /CIDFont+F1 /Flags 4 \
          /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
          /CapHeight 700 /StemV 80 /MissingWidth 500 >>"
            .to_vec(),
        format!("<< /Length {} >>\nstream\n{cmap}\nendstream", cmap.len()).into_bytes(),
    ];
    if let Some(map) = cid_to_gid {
        let mut obj = format!("<< /Length {} >>\nstream\n", map.len()).into_bytes();
        obj.extend_from_slice(map);
        obj.extend_from_slice(b"\nendstream");
        objs.push(obj);
    }

    // Assemble with a byte-accurate xref table.
    let mut out: Vec<u8> = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_off = out.len();
    let size = objs.len() + 1;
    out.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF").as_bytes(),
    );
    out
}

fn extract_first_page(bytes: Vec<u8>) -> String {
    let doc = PdfDocument::from_bytes(bytes).expect("parse synthetic PDF");
    doc.extract_text(0).expect("extract page 0")
}

/// Several documents that share a `/BaseFont` name but map glyphs differently
/// must each decode through their own `/ToUnicode`, even when processed
/// back-to-back in one process without clearing the cache between them.
#[test]
fn distinct_tounicode_fonts_do_not_collide_across_documents() {
    let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_global_font_cache();

    // Distinct CID bases ⇒ distinct ToUnicode streams. The first document
    // primes the global cache; before the fix, every later one inherited its
    // mapping. The bases are arbitrary, only mutually distinct.
    let bases = [3u16, 1000, 2000, 40000];
    let mut garbled = Vec::new();
    for base in bases {
        let text = extract_first_page(build_type0_pdf(base, None));
        if !text.contains("SUMMARY") {
            let preview: String = text.chars().take(48).collect();
            garbled.push(format!("cid_base={base}: {preview:?}"));
        }
    }

    assert!(
        garbled.is_empty(),
        "{} of {} same-named fonts decoded through another document's ToUnicode \
         (cross-document font-cache collision):\n  {}",
        garbled.len(),
        bases.len(),
        garbled.join("\n  ")
    );
}

/// The precise key must not regress the dedup the global cache exists for:
/// a byte-identical font (different document) is a cache *hit* with no new
/// entry, while a font with a different `/ToUnicode` gets its own entry.
#[test]
fn identical_fonts_dedup_while_distinct_fonts_get_separate_entries() {
    let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_global_font_cache();
    assert_eq!(global_font_cache_stats().0, 0, "cache should be empty after clear");

    // Each document defines exactly one cross-document-shareable Type0 font, so
    // the cache grows by one entry per *distinct* font.
    assert!(extract_first_page(build_type0_pdf(3, None)).contains("SUMMARY"));
    let after_first = global_font_cache_stats().0;
    assert_eq!(after_first, 1, "first document inserts exactly one font");

    // Same bytes, brand-new PdfDocument: must hit the global cache, not reinsert.
    assert!(extract_first_page(build_type0_pdf(3, None)).contains("SUMMARY"));
    assert_eq!(
        global_font_cache_stats().0,
        after_first,
        "an identical font must hit the global cache rather than re-insert"
    );

    // Different ToUnicode: must get its own entry (the absence of which was the
    // collision bug) and decode correctly.
    assert!(extract_first_page(build_type0_pdf(2000, None)).contains("SUMMARY"));
    assert_eq!(
        global_font_cache_stats().0,
        after_first + 1,
        "a font with a different ToUnicode must not alias the cached one"
    );
}

/// A *stream*-form `/CIDToGIDMap` remaps CID→glyph (ISO 32000-1 §9.7.4.3), so
/// two embedded CIDFontType2 fonts identical in name, `/ToUnicode`, and metrics
/// but differing in that stream are not interchangeable and must get separate
/// cache entries. (PR #733 review: the `/Identity` name, the default, still
/// folds nothing — that case is covered by the tests above.)
#[test]
fn stream_cid_to_gid_map_distinguishes_otherwise_identical_fonts() {
    let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_global_font_cache();

    // Same cid_base ⇒ identical /ToUnicode and content; the ONLY difference is
    // the /CIDToGIDMap stream. Sized to cover every CID used (2 bytes/CID), GIDs
    // kept < 0x80 so it is well-formed map data.
    let distinct = LINES
        .iter()
        .flat_map(|l| l.chars())
        .fold(Vec::new(), |mut v, c| {
            if !v.contains(&c) {
                v.push(c);
            }
            v
        });
    let len = 2 * (0x21 + distinct.len());
    let map_a: Vec<u8> = (0..len).map(|i| (i % 0x40) as u8).collect();
    let mut map_b = map_a.clone();
    *map_b.last_mut().unwrap() ^= 0x01; // differ by a single byte

    assert!(extract_first_page(build_type0_pdf(0x21, Some(&map_a))).contains("SUMMARY"));
    let after_a = global_font_cache_stats().0;
    assert!(extract_first_page(build_type0_pdf(0x21, Some(&map_b))).contains("SUMMARY"));
    assert_eq!(
        global_font_cache_stats().0,
        after_a + 1,
        "fonts differing only in a stream /CIDToGIDMap must not alias in the cache"
    );
}
