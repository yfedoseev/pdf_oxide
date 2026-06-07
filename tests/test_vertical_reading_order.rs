//! Tategaki reading-order regression test.
//!
//! Builds a synthetic PDF with two vertical columns at distinct X positions:
//!
//!     Column A (rightmost): X≈500, glyphs at y = 700, 680, 660
//!     Column B (leftmost):  X≈300, glyphs at y = 700, 680, 660
//!
//! Vertical Japanese reads right-to-left across columns, top-to-bottom
//! within each column. The extractor's reading-order sort must order
//! column A's three glyphs first (in descending y), then column B's three
//! glyphs (also in descending y).
//!
//! The PDF emits each glyph in its own `BT … Tm Tj … ET` block so that the
//! text matrix is set explicitly per glyph — this lets us pin glyph
//! positions in the synthetic content stream without needing real vertical
//! font metrics for layout.

use pdf_oxide::document::PdfDocument;

fn build_tategaki_two_columns_pdf() -> Vec<u8> {
    // ToUnicode CMap mapping CIDs 1..=6 to ASCII 'A'..'F'.
    let cmap = b"\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
6 beginbfchar
<0001> <0041>
<0002> <0042>
<0003> <0043>
<0004> <0044>
<0005> <0045>
<0006> <0046>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end";

    // Reading order (right column first, top first within column):
    //   A → B → C  (column A, top→bottom)
    //   D → E → F  (column B, top→bottom)
    // Content places each glyph at the correct (x, y). Column A on the
    // right (x=500), column B on the left (x=300). Y descends within each
    // column (700, 680, 660). Tm operator sets absolute position before
    // each Tj so the spans land exactly where we want them, independent
    // of the per-glyph advance math.
    //
    // Note: we use `Tm 1 0 0 1 X Y` (identity scale, translate to X,Y)
    // before each Tj. The Tj itself shows a single CID.
    let content = b"BT /F1 12 Tf \
        1 0 0 1 500 700 Tm <0001> Tj \
        1 0 0 1 500 680 Tm <0002> Tj \
        1 0 0 1 500 660 Tm <0003> Tj \
        1 0 0 1 300 700 Tm <0004> Tj \
        1 0 0 1 300 680 Tm <0005> Tj \
        1 0 0 1 300 660 Tm <0006> Tj \
        ET";

    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let o1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    let o2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    let o3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] \
          /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
    );
    let o4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj << /Length {} >> stream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let o5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestFont \
          /Encoding /Identity-V /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
    );
    let o6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont \
          /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
          /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );
    let o7 = pdf.len();
    pdf.extend_from_slice(format!("7 0 obj << /Length {} >> stream\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for off in [o1, o2, o3, o4, o5, o6, o7] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size 8 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref).as_bytes(),
    );

    pdf
}

/// All six spans must carry `wmode == 1` because the font's /Encoding is
/// /Identity-V. This is the per-span tag the reading-order sort consults.
#[test]
fn vertical_spans_tag_wmode_1() {
    let pdf = build_tategaki_two_columns_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse tategaki PDF");
    let spans = doc.extract_spans(0).expect("extract spans");
    assert_eq!(
        spans.len(),
        6,
        "expected 6 glyph spans (2 columns x 3 glyphs); got {} (texts={:?})",
        spans.len(),
        spans.iter().map(|s| s.text.clone()).collect::<Vec<_>>()
    );
    for s in &spans {
        assert_eq!(
            s.wmode, 1,
            "every span emitted under /Identity-V must have wmode=1; got {:?} for {:?}",
            s.wmode, s.text
        );
    }
}

/// Reading order: A, B, C, D, E, F — right column top-down first, then
/// left column top-down. This pins the tategaki sort behavior independent
/// of any horizontal-column heuristic.
#[test]
fn vertical_reading_order_is_right_to_left_top_to_bottom() {
    let pdf = build_tategaki_two_columns_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse tategaki PDF");
    let spans = doc.extract_spans(0).expect("extract spans");
    let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(
        combined,
        "ABCDEF",
        "tategaki reading order should yield right-column-first, top-down within column. \
         got {:?} from spans at positions {:?}",
        combined,
        spans
            .iter()
            .map(|s| (s.text.clone(), s.bbox.x, s.bbox.y))
            .collect::<Vec<_>>()
    );
}
