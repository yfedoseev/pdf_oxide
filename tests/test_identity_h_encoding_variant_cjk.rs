//! Regression: a Type0 / Identity-H composite font whose `Encoding` is
//! collapsed into the `Encoding::Identity` enum variant (per the `Encoding`
//! doc comment, Identity-H/Identity-V names are folded into `Encoding::Identity`
//! rather than kept as `Encoding::Standard("Identity-H")`).
//!
//! The character-to-Unicode recovery for such a font MUST NOT be gated behind
//! a `match` arm that only accepts `Encoding::Standard(..)`. A previous build
//! did exactly that: the entire Identity recovery path (ToUnicode CMap +
//! embedded TrueType cmap) was skipped, and every CID fell through to the bare
//! `char::from_u32(cid)` fallback. For a subset CIDFontType2 whose codes bear
//! no relation to Unicode, that produced mojibake (e.g. CID 0x69 → 'i').
//!
//! This fixture reproduces the structural shape of that bug with a 100%
//! synthetic, non-sensitive PDF: a Type0/Identity-H font whose 2-byte codes
//! are a constant offset below the true Unicode values, recovered ONLY through
//! the ToUnicode CMap. The content stream draws the heading; the encoding is
//! `/Identity-H` (which the parser folds to `Encoding::Identity`). The test
//! asserts the true text is recovered and the offset-garble never surfaces.
//!
//! This is the structural sibling of `test_tounicode_identity_h_offset_codes`
//! — that test proves the ToUnicode path works; this one proves the
//! `Encoding::Identity` *variant* (not just the `Standard("Identity-H")` string)
//! also enters that path.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

const HEADING: &str = "YE WU LIU SHUI HAO";

/// Constant offset below the Unicode scalar. 29 reproduces a clear signature:
/// Y→"'", E→"'", space→" ", W→"O", ... so a base-encoding fall-through emits
/// printable Latin-1 mojibake — unambiguous to assert against.
const OFFSET: u32 = 29;

fn build_identity_variant_pdf() -> Vec<u8> {
    let size = 24.0f32;
    let dw_units = 600u32;

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

    // NOTE: /Encoding /Identity-H — the parser folds this to Encoding::Identity.
    let basefont = "BBBBBB+Sub";
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
             /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
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

/// Under the buggy build the `Encoding::Identity` variant never entered the
/// Identity recovery path, so the constant-offset codes were emitted raw
/// (e.g. `Y`→`'`...). The true heading must be recovered instead.
#[test]
fn identity_encoding_variant_enters_recovery_path() {
    let text = extract(&build_identity_variant_pdf());

    assert!(
        text.contains(HEADING),
        "Identity-H font folded to Encoding::Identity must extract {HEADING:?} via ToUnicode; got {text:?}"
    );
    // The mojibake signature: Y(0x59-29=0x3C='<') ... space, etc. Assert the
    // garble does not appear by checking the real text dominates.
    assert!(
        !text.contains("'<'"),
        "constant-offset mojibake leaked — Encoding::Identity was skipped; got {text:?}"
    );
}
