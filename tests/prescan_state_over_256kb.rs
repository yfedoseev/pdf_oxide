//! Text state must survive the >256 KB prescan boundary.
//!
//! Streams past 256 KB route through the SIMD prescan, which parses each
//! text region in isolation. State set before a region — the font, its
//! size, the fill colour, a `"` operator's spacing operands — belongs to
//! the region all the same. The fixture sets state early, crosses the
//! boundary with ~280 KB of path filler, and draws text that never restates
//! it; the extracted spans must carry the state from before the boundary.
//!
//! No crawled corpus document exercises this boundary, so the fixture is
//! built in code and asserts its own reachability: the content stream must
//! actually exceed 256 * 1024 bytes or the test proves nothing.

use pdf_oxide::PdfDocument;

/// One-page PDF whose content stream crosses the prescan threshold.
/// Layout: red 24 pt text state + one marker line, ~280 KB of path filler,
/// then two text blocks that rely on earlier state:
/// - "After the big stream" with no `Tf` after the boundary;
/// - a `"` (quote) operator with word spacing 40 set by its own operand.
fn big_stream_pdf() -> Vec<u8> {
    big_stream_pdf_with_quote_spacing("40")
}

/// The same page with the `"` operator's word-spacing operand substituted:
/// the differential pair for the injection test below.
fn big_stream_pdf_with_quote_spacing(aw: &str) -> Vec<u8> {
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(
        b"BT /F1 24 Tf 1 0 0 rg 1 0 0 1 100 700 Tm (Before the big stream) Tj ET\n",
    );
    for _ in 0..20000 {
        content.extend_from_slice(b"0 0 m 1 1 l S\n");
    }
    content.extend_from_slice(b"BT 1 0 0 1 100 600 Tm (After the big stream) Tj ET\n");
    content.extend_from_slice(
        format!(
            "BT /F1 12 Tf 1 0 0 1 100 500 Tm (first line) Tj 0 -20 TD {aw} 2 (quote line) \" ET\n"
        )
        .as_bytes(),
    );
    // Path filler between text blocks, so the trailing block sits in its own
    // prescan region and can only see the quote's Tw through state injection.
    for _ in 0..500 {
        content.extend_from_slice(b"0 0 m 1 1 l S\n");
    }
    content.extend_from_slice(b"BT /F1 12 Tf 1 0 0 1 100 400 Tm (space separated words) Tj ET\n");
    assert!(content.len() > 256 * 1024, "fixture must cross the prescan threshold");

    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 6];
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    {
        let mut obj = |buf: &mut Vec<u8>, off: &mut [usize; 6], id: usize, body: &str| {
            off[id] = buf.len();
            buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
        };
        obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
        obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
        obj(
            &mut buf,
            &mut off,
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        );
        off[4] = buf.len();
        buf.extend_from_slice(
            format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes(),
        );
        buf.extend_from_slice(&content);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
        obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    }
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn text_state_survives_the_prescan_boundary() {
    let doc = PdfDocument::from_bytes(big_stream_pdf()).expect("fixture parses");
    let text = doc.extract_text(0).expect("extract_text");
    assert!(
        text.contains("After the big stream"),
        "text after the boundary must extract: {text:?}"
    );

    let spans = doc.extract_spans(0).expect("extract_spans");
    let after = spans
        .iter()
        .find(|s| s.text.contains("After the big stream"))
        .expect("span after the boundary");
    assert!(
        (after.font_size - 24.0).abs() < 0.01,
        "the 24 pt Tf from before the boundary must reach the span, got {}",
        after.font_size
    );
    assert!(
        after.color.r > 0.9 && after.color.g < 0.1,
        "the red fill from before the boundary must reach the span, got {:?}",
        after.color
    );
}

/// The `"` operator's Tw operand is persistent state: text drawn in a later
/// prescan region must still be spaced by it. The pair differs only in the
/// operand (0 vs 40), so the trailing span — two word gaps — must widen by
/// exactly 80 pt; without operand tracking through the prescan the two
/// fixtures render identically.
#[test]
fn quote_operator_spacing_survives_the_prescan_boundary() {
    let width_of_trailing = |aw: &str| -> f32 {
        let doc =
            PdfDocument::from_bytes(big_stream_pdf_with_quote_spacing(aw)).expect("fixture parses");
        let spans = doc.extract_spans(0).expect("extract_spans");
        spans
            .iter()
            .filter(|s| s.text.contains("space separated words"))
            .map(|s| s.bbox.width)
            .fold(0.0, f32::max)
    };
    let narrow = width_of_trailing("0");
    let wide = width_of_trailing("40");
    assert!(
        (wide - narrow - 80.0).abs() < 1.0,
        "40 pt of word spacing across two spaces must widen the trailing span by 80 pt: \
         aw=0 width {narrow}, aw=40 width {wide}"
    );
}
