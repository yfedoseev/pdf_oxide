//! Regression test: inter-word spaces on tightly-spaced PDFs.
//!
//! Some producers (notably resume generators) draw text one glyph per `Tj`
//! with incremental `Td` moves and emit NO space glyph — inter-word gaps are
//! just slightly larger `Td` offsets (0.3–0.8 × glyph advance). oxide used to
//! concatenate words in this case ("JOHN DOE" -> "JOHNDOE",
//! "Master of Science" -> "MasterofScience") while PyMuPDF and poppler
//! `pdftotext` both infer the spaces correctly, so oxide was the outlier
//! against its own calibration reference.
//!
//! Two root causes, both fixed:
//!   1. `FontInfo::get_space_glyph_width` returned a CID font's /DW default
//!      width (often 0.5 em) as the "space advance", inflating the word-gap
//!      threshold so tight gaps fell below it.
//!   2. The intra-word kerning guard in `should_insert_space` suppressed
//!      lowercase→lowercase boundaries up to 2.4 × the (already inflated)
//!      threshold — far wider than any real kerning — swallowing genuine
//!      tight word gaps.
//!
//! These fixtures are 100 % synthetic (no PII). The canonical subset BaseFont
//! tag (`AAAAAA+`) keeps the font document-local so unrelated font-cache
//! behaviour cannot interfere.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

const LINES: &[&[&str]] = &[
    &["JOHN", "DOE", "Boston", "MA"],
    &["Master", "of", "Science", "in", "Information", "Systems"],
    &[
        "Results",
        "driven",
        "engineer",
        "with",
        "expertise",
        "in",
        "data",
    ],
];

/// Build a Type0/CIDFontType2 PDF that draws `LINES` one glyph per `Tj`, with a
/// per-glyph advance equal to the declared width (so intra-word gap == 0) and
/// an inter-word `Td` gap of `word_gap_factor × advance` (no space glyph).
///
/// `space_w`: when `Some(w)`, declare an explicit `/W` width for code 0x20 — the
/// "embedded font that carries a real space-glyph width" case (real resumes).
/// When `None`, the font has no space entry and oxide must not mistake /DW for
/// the space advance — the non-embedded fixture case.
fn build(word_gap_factor: f32, space_w: Option<u32>) -> Vec<u8> {
    let size = 24.0f32;
    let dw_units = 500u32;

    // Unique glyphs in first-seen order; arbitrary CIDs from `cid_base` below.
    let mut chars: Vec<char> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ln in LINES {
        for w in *ln {
            for ch in w.chars() {
                if seen.insert(ch) {
                    chars.push(ch);
                }
            }
        }
    }
    // Reserve CID 32 (= code 0x20) for the declared space glyph in the
    // `space_w.is_some()` variant so no real word glyph collides with it
    // (the space's declared width would otherwise mismatch the Td advance and
    // open a spurious intra-word gap). The non-embedded variant has no /W, so
    // base 3 is safe.
    let cid_base: u32 = if space_w.is_some() { 33 } else { 3 };
    let cid: std::collections::HashMap<char, u32> = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u32 + cid_base))
        .collect();

    let adv = dw_units as f32 / 1000.0 * size; // == declared width => 0 intra-word gap
    let word_gap = word_gap_factor * adv;

    let mut content = String::new();
    let mut y = 760.0f32;
    for ln in LINES {
        content.push_str(&format!("BT\n/F1 {size} Tf\n1 0 0 1 40 {y:.2} Tm\n"));
        for (wi, word) in ln.iter().enumerate() {
            for ch in word.chars() {
                content.push_str(&format!("<{:04X}> Tj\n{adv:.3} 0 Td\n", cid[&ch]));
            }
            if wi != ln.len() - 1 {
                content.push_str(&format!("{word_gap:.3} 0 Td\n")); // inter-word gap, NO space glyph
            }
        }
        content.push_str("ET\n");
        y -= size * 1.6;
    }
    let content_b = content.into_bytes();

    let bf: String = chars
        .iter()
        .map(|&ch| format!("<{:04X}> <{:04X}>", cid[&ch], ch as u32))
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

    let w_entry = match space_w {
        Some(w) => format!(" /W [32 [{w}]]"),
        None => String::new(),
    };

    let basefont = "AAAAAA+Hv";
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
             /FontDescriptor 7 0 R /DW {dw_units}{w_entry} /CIDToGIDMap /Identity >>"
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

/// Non-embedded font (no space glyph, only /DW): a 0.4–0.6 × advance gap is a
/// real word break that must be recovered, matching PyMuPDF / pdftotext.
#[test]
fn tight_word_gaps_no_space_glyph_are_spaced() {
    for factor in [0.4_f32, 0.6] {
        let text = extract(&build(factor, None));
        assert!(
            text.contains("JOHN DOE"),
            "gap factor {factor}: expected 'JOHN DOE', got: {text:?}"
        );
        assert!(
            text.contains("Boston MA"),
            "gap factor {factor}: expected 'Boston MA', got: {text:?}"
        );
        assert!(
            text.contains("Master of Science in Information Systems"),
            "gap factor {factor}: expected 'Master of Science in Information Systems', got: {text:?}"
        );
        // Lowercase→lowercase word boundaries are the regression-prone case.
        assert!(
            text.contains("Results driven engineer with expertise in data"),
            "gap factor {factor}: lowercase word boundaries must be spaced, got: {text:?}"
        );
    }
}

/// Embedded font that DOES declare a space-glyph width (/W 32 [277] ≈ 0.277 em)
/// yet still positions words via Td gaps (the real-resume case). The fix must
/// cover this path too. A 0.6 × advance gap (≈ 0.3 em) is comfortably above
/// intra-word kerning and must be recovered.
///
/// Note: a 0.4 × advance gap (≈ 0.2 em) against a 0.277-em space is only
/// ~0.72 × the space width — that sits in the zone where a real word gap is
/// not separable from aggressive letter-spacing by magnitude alone (see the
/// guard comment in `should_insert_space`), so it is intentionally not
/// asserted here.
#[test]
fn tight_word_gaps_with_declared_space_width_are_spaced() {
    let factor = 0.6_f32;
    let text = extract(&build(factor, Some(277)));
    assert!(
        text.contains("JOHN DOE")
            && text.contains("Master of Science in Information Systems")
            && text.contains("Results driven engineer with expertise in data"),
        "declared-space gap factor {factor}: words must be spaced, got: {text:?}"
    );
}

/// Lower bound: a 0.2 × advance gap is genuinely too small to be a word break
/// (PyMuPDF / pdftotext also keep it glued). Guard against the fix over-firing
/// and shattering tight intra-line text into spurious words.
#[test]
fn sub_threshold_gaps_stay_glued() {
    let text = extract(&build(0.2, None));
    assert!(
        text.contains("JOHNDOE"),
        "0.2x gap must NOT be treated as a word break, got: {text:?}"
    );
}
