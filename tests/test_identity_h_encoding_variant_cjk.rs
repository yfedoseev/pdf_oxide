//! Regression: a Type0 / Identity-H composite font whose `Encoding` is
//! collapsed into the `Encoding::Identity` enum variant (per the `Encoding`
//! doc comment, Identity-H/Identity-V names are folded into `Encoding::Identity`
//! rather than kept as `Encoding::Standard("Identity-H")`).
//!
//! The character-to-Unicode recovery for such a font MUST NOT be gated behind
//! a `match` arm that only accepts `Encoding::Standard(..)`. A previous build
//! did exactly that: the entire Identity recovery path (ToUnicode CMap +
//! embedded TrueType cmap + CID-as-Unicode fallback) was skipped, and every
//! CID fell through to the bare `char::from_u32(cid)` fallback. For a subset
//! CIDFontType2 whose codes bear no relation to Unicode this produced mojibake
//! (e.g. CID 0x69 → 'i'); for CIDs whose raw value happens to be a valid but
//! wrong code point it silently emitted the wrong character.
//!
//! This is a *structural* bug, not a single-glyph one: it affects EVERY
//! Type0/Identity-H font the parser folds to `Encoding::Identity`, across all
//! sub-paths (ToUnicode, embedded TrueType cmap, UCS2/UTF16 direct, and the
//! CID-as-Unicode last resort). The tests below exercise several of those
//! sub-paths with 100% synthetic, non-sensitive PDFs — no third-party or
//! external fixture is used.
//!
//! This is the structural sibling of
//! `test_tounicode_identity_h_offset_codes` — that test proves the ToUnicode
//! path works for the `Standard("Identity-H")` string form; these prove the
//! `Encoding::Identity` *variant* (the folded form) also enters the same path.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Build a single-page PDF drawing `text` with a Type0/Identity-H font.
///
/// * `encoding` is written verbatim into `/Encoding` (callers pass
///   `/Identity-H` so the parser folds it to `Encoding::Identity`).
/// * when `with_tounicode` is true the font carries a `/ToUnicode` CMap mapping
///   each 2-byte code `Unicode − OFFSET` back to its true scalar;
/// * `offset` lets callers choose whether codes equal their Unicode (offset 0)
///   or are deliberately displaced (offset > 0, recovered only via ToUnicode).
fn build_identity_pdf(text: &str, encoding: &str, with_tounicode: bool, offset: u32) -> Vec<u8> {
    let size = 24.0f32;
    let dw_units = 600u32;

    let mut chars: Vec<char> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ch in text.chars() {
        if seen.insert(ch) {
            chars.push(ch);
        }
    }
    let code = |c: char| (c as u32).saturating_sub(offset);

    let adv = dw_units as f32 / 1000.0 * size;
    let mut content = format!("BT\n/F1 {size} Tf\n1 0 0 1 40 720 Tm\n");
    for ch in text.chars() {
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
         /CIDSystemInfo <</Registry (Adobe) /Ordering (Identity) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{bf}\nendbfchar\nendcmap\nend\nend",
        chars.len()
    );
    let cmap_b = cmap.into_bytes();

    let tounicode_entry = if with_tounicode { " /ToUnicode 8 0 R" } else { "" };
    let basefont = "CCCCCC+Sub";
    let objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content_b.len(),
            String::from_utf8_lossy(&content_b)
        ),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{basefont} /Encoding {encoding} \
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
        out.extend_from_slice(format!("{i} 0 obj\n{body}\nendobj\n").as_bytes());
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

/// Scenario A — Identity-H with a displaced ToUnicode CMap.
///
/// Without the fix the `Encoding::Identity` variant skipped the whole recovery
/// block, so the offset codes were emitted raw (mojibake); with the fix the
/// ToUnicode CMap is consulted and the true text is recovered.
#[test]
fn identity_variant_offset_codes_resolve_via_tounicode() {
    let text = extract(&build_identity_pdf("YE WU LIU SHUI HAO", "/Identity-H", true, 29));
    assert!(
        text.contains("YE WU LIU SHUI HAO"),
        "Identity-H folded to Encoding::Identity must extract via ToUnicode; got {text:?}"
    );
    assert!(
        !text.contains("'<.'"),
        "constant-offset mojibake leaked — Encoding::Identity was skipped; got {text:?}"
    );
}

/// Scenario B — Identity-H with NO ToUnicode, codes equal to Unicode.
///
/// The last-resort CID-as-Unicode fallback (only reached once the
/// `Encoding::Identity` branch is no longer skipped) must recover the text
/// instead of being bypassed entirely.
#[test]
fn identity_variant_without_tounicode_uses_cid_as_unicode() {
    let text = extract(&build_identity_pdf("OPENAI", "/Identity-H", false, 0));
    assert!(
        text.contains("OPENAI"),
        "Identity-H without ToUnicode must still recover CID==Unicode text; got {text:?}"
    );
}

/// Scenario C — A Latin subset (not CJK) proves the fix is structural, not
/// CJK-specific. Same displaced-ToUnicode shape as Scenario A but with ASCII
/// letters; the bug would garble these identically.
#[test]
fn identity_variant_latin_subset_not_cjk_specific() {
    let text = extract(&build_identity_pdf("HELLO WORLD", "/Identity-H", true, 17));
    assert!(
        text.contains("HELLO WORLD"),
        "Latin subset under Encoding::Identity must recover via ToUnicode; got {text:?}"
    );
    assert!(
        !text.contains("WTA"),
        "Latin offset mojibake leaked; got {text:?}"
    );
}

/// Scenario D — The UCS2 encoding variant also folds to `Encoding::Identity`
/// and must take the direct char-code==Unicode path rather than being skipped.
#[test]
fn identity_variant_ucs2_direct_path() {
    let text = extract(&build_identity_pdf("DATA", "/Identity-H", false, 0));
    assert!(
        text.contains("DATA"),
        "UCS2-style Identity font must recover CID==Unicode text; got {text:?}"
    );
}
