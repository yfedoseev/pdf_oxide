//! Regression: a Type0 / Identity-H / CIDFontType2 font whose content-stream
//! character codes are a *constant offset* from the Unicode values they stand
//! for, recoverable only through the font's `/ToUnicode` CMap.
//!
//! Some producers subset a TrueType face and assign CIDs (== Identity-H codes,
//! == GIDs) that bear no relation to any standard encoding — here each code is
//! `Unicode − 29`, so `Z` (U+005A) is drawn as code 0x3D, `E` (U+0045) as
//! 0x28, `K` (U+004B) as 0x2E, and so on. A reader that ignores the ToUnicode
//! CMap and falls through to a base text encoding emits the raw low bytes as
//! Latin-1, producing constant-offset mojibake (`ZEKAT…` → `=(.$7…`).
//!
//! ISO 32000-1:2008 §9.10.2 ("Mapping Character Codes to Unicode Values") gives
//! the ToUnicode CMap as the **first**, highest-priority method, and §9.10.2
//! explicitly *excepts* Identity-H / Identity-V composite fonts from the
//! predefined-CMap (registry-ordering-UCS2) fallback — so for an Identity-H
//! font the ToUnicode CMap is the only spec-sanctioned recovery path. When it
//! is present it shall be used; the offset garble must never surface.
//!
//! The fixture is 100% synthetic — no third-party file, no embedded glyph
//! program (text extraction is driven entirely by the ToUnicode CMap, not by
//! the outlines). The canonical subset BaseFont tag keeps the font
//! document-local so font-cache behaviour cannot leak across tests.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Heading text exercised by the fixture. Uppercase Latin + spaces keep every
/// `Unicode − 29` code in the printable-ASCII range, so the would-be garble is
/// itself printable and unambiguous to assert against.
const HEADING: &str = "ZEKAT VE FITIR SADAKASI";

/// Per-character code offset below the Unicode scalar value. 29 reproduces the
/// observed signature exactly: Z→'=', E→'(', K→'.', A→'$', T→'7'.
const OFFSET: u32 = 29;

/// Build a single-page PDF that draws `HEADING` with a Type0/Identity-H font
/// whose 2-byte codes are `Unicode − OFFSET`. When `with_tounicode` is true the
/// font carries a `/ToUnicode` CMap mapping each code back to its true Unicode
/// value; when false the font omits it, so only the offset codes survive.
fn build_offset_coded_pdf(with_tounicode: bool) -> Vec<u8> {
    let size = 24.0f32;
    let dw_units = 600u32;

    // Unique characters in first-seen order; code(c) = (c as u32) - OFFSET.
    let mut chars: Vec<char> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ch in HEADING.chars() {
        if seen.insert(ch) {
            chars.push(ch);
        }
    }
    let code = |c: char| c as u32 - OFFSET;

    let adv = dw_units as f32 / 1000.0 * size;
    let mut content = format!("BT\n/F1 {size} Tf\n1 0 0 1 40 720 Tm\n");
    for ch in HEADING.chars() {
        content.push_str(&format!("<{:04X}> Tj\n{adv:.3} 0 Td\n", code(ch)));
    }
    content.push_str("ET\n");
    let content_b = content.into_bytes();

    let bf: String = chars
        .iter()
        .map(|&ch| format!("<{:04X}> <{:04X}>", code(ch), ch as u32))
        .collect::<Vec<_>>()
        .join("\n");
    let cmap = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{bf}\nendbfchar\nendcmap\nend\nend",
        chars.len()
    );
    let cmap_b = cmap.into_bytes();

    let tounicode_entry = if with_tounicode {
        " /ToUnicode 8 0 R"
    } else {
        ""
    };

    let basefont = "AAAAAA+Sub";
    let objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_string(),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content_b.len(),
            String::from_utf8_lossy(&content_b)
        ),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{basefont} /Encoding /Identity-H \
             /DescendantFonts [6 0 R]{tounicode_entry} >>"
        ),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{basefont} \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 7 0 R /DW {dw_units} /CIDToGIDMap /Identity >>"
        ),
        format!(
            "<< /Type /FontDescriptor /FontName /{basefont} /Flags 4 \
             /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
             /CapHeight 700 /StemV 80 /MissingWidth {dw_units} >>"
        ),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            cmap_b.len(),
            String::from_utf8_lossy(&cmap_b)
        ),
    ];

    let mut out: Vec<u8> = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
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

fn extract(pdf: &[u8]) -> String {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("parse pdf");
    let opts = ConversionOptions::default();
    let pages = doc.page_count().expect("page count");
    (0..pages)
        .map(|i| doc.to_plain_text(i, &opts).expect("to_plain_text"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The first three distinct glyphs (Z,E,K) garble to "=(." under a
/// base-encoding fall-through — the canonical signature of ToUnicode being
/// ignored.
const GARBLE_SIGNATURE: &str = "=(.";

/// The ToUnicode CMap must recover the true heading; the constant-offset garble
/// must not appear.
#[test]
fn identity_h_offset_codes_resolve_via_tounicode() {
    let text = extract(&build_offset_coded_pdf(true));

    assert!(
        text.contains(HEADING),
        "Identity-H font with a ToUnicode CMap must extract {HEADING:?}; got {text:?}"
    );
    assert!(
        !text.contains(GARBLE_SIGNATURE),
        "constant-offset mojibake leaked — ToUnicode CMap was not consulted; got {text:?}"
    );
}

/// Control: with the ToUnicode CMap removed the offset codes have no
/// spec-sanctioned recovery for an Identity-H font, so the heading cannot be
/// reconstructed. This proves the fixture genuinely depends on the CMap and the
/// positive test above is not passing for an unrelated reason.
#[test]
fn identity_h_offset_codes_without_tounicode_do_not_recover_heading() {
    let text = extract(&build_offset_coded_pdf(false));

    assert!(
        !text.contains(HEADING),
        "without a ToUnicode CMap the offset codes must not reconstruct {HEADING:?}; got {text:?}"
    );
}
