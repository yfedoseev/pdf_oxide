//! Word spacing (`Tw`) must apply only to the single-byte character code 32
//! (ISO 32000-1:2008 §9.3.3): "Word spacing shall be applied to every
//! occurrence of the single-byte character code 32 in a string... Word
//! spacing shall not apply to occurrences of the byte value 32 in
//! multiple-byte codes." A Type0/Identity-H font's codes are always 2 bytes,
//! so CID 32 (content-stream bytes `<0020>`) must never receive `Tw`, even
//! though its Unicode mapping (or raw numeric value) happens to be 32 — a
//! real embedded glyph at that code point must not be over-advanced as if
//! it were a word gap.
//!
//! The fixture is 100% synthetic: a 3-glyph Identity-H run (`A`, CID 32
//! mapped to U+0020 via `/ToUnicode`, `B`) with an explicit uniform `/W`
//! width, compared under `Tw = 0` and `Tw = 50` — the span's total advance
//! width must be identical in both, proving the multi-byte CID 32 did not
//! absorb the word-spacing addend.

use pdf_oxide::PdfDocument;

/// Build a single-page PDF drawing 3 Identity-H CIDs (`0041`, `0020`, `0042`)
/// each with declared width 500/1000 em, with the given `Tw` value active.
fn build_identity_h_pdf(word_space: f32) -> Vec<u8> {
    let size = 24.0f32;
    let w_units = 500u32;

    let content =
        format!("{word_space} Tw\nBT\n/F1 {size} Tf\n1 0 0 1 40 720 Tm\n<004100200042> Tj\nET\n");
    let content_b = content.into_bytes();

    // /ToUnicode: CID 0x0041 -> 'A', CID 0x0020 -> U+0020 (space), CID 0x0042 -> 'B'.
    let cmap = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         3 beginbfchar\n<0041> <0041>\n<0020> <0020>\n<0042> <0042>\nendbfchar\nendcmap\nend\nend";
    let cmap_b = cmap.as_bytes();

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
             /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
        ),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{basefont} \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 7 0 R /DW {w_units} \
             /W [65 [{w_units}] 32 [{w_units}] 66 [{w_units}]] /CIDToGIDMap /Identity >>"
        ),
        format!(
            "<< /Type /FontDescriptor /FontName /{basefont} /Flags 4 \
             /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
             /CapHeight 700 /StemV 80 /MissingWidth {w_units} >>"
        ),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            cmap_b.len(),
            String::from_utf8_lossy(cmap_b)
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

/// Total advance width of the single Tj span on page 0 — directly reflects
/// whatever the extractor accumulated across all 3 CIDs, including any
/// (incorrectly) applied Tw.
fn span_width(pdf: &[u8]) -> f32 {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("parse pdf");
    let spans = doc.extract_spans(0).expect("extract_spans");
    assert_eq!(spans.len(), 1, "expected exactly 1 span, got {:?}", spans);
    spans[0].bbox.width
}

#[test]
fn word_spacing_does_not_shift_glyph_after_multibyte_cid_32() {
    let w_no_tw = span_width(&build_identity_h_pdf(0.0));
    let w_with_tw = span_width(&build_identity_h_pdf(50.0));

    assert!(
        (w_no_tw - w_with_tw).abs() < 0.01,
        "Tw must not apply to the 2-byte CID 32 glyph: no-Tw width={}, Tw=50 width={}",
        w_no_tw,
        w_with_tw
    );
}

/// Control: the same word-spacing value DOES shift a simple (single-byte)
/// font's real space character, proving the fix does not disable Tw wholesale.
#[test]
fn word_spacing_shifts_glyph_after_single_byte_space() {
    fn build_simple_pdf(word_space: f32) -> Vec<u8> {
        let content = format!("{word_space} Tw\nBT\n/F1 24 Tf\n1 0 0 1 40 720 Tm\n(A B) Tj\nET\n");
        let content_b = content.into_bytes();
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
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
                .to_string(),
        ];
        let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::with_capacity(objs.len());
        for (i, body) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref_pos = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes(),
        );
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

    let w_no_tw = span_width(&build_simple_pdf(0.0));
    let w_with_tw = span_width(&build_simple_pdf(50.0));

    assert!(
        w_with_tw - w_no_tw > 40.0,
        "Tw=50 must widen the span containing a single-byte space by ~50pt: \
         no-Tw width={}, Tw=50 width={}",
        w_no_tw,
        w_with_tw
    );
}
