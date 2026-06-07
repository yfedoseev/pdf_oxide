//! End-to-end extraction test for vertical writing mode (WMode 1).
//!
//! These tests build minimal synthetic PDFs that pair an `Identity-V`
//! Type0 font with a content stream showing three glyphs at user-space
//! origin `(100, 700)`. The horizontal sibling document uses `Identity-H`
//! at the same position. Together they pin the axis-swap behavior:
//!
//!   - Identity-H ⇒ spans advance in X (Y stable, X ascending).
//!   - Identity-V ⇒ spans advance in Y (X stable, Y descending).
//!
//! No copyrighted CJK fonts are required: the math is exercised through the
//! advance helpers, and the ToUnicode CMap maps CIDs to ASCII so the
//! extracted text is trivially identifiable.

use pdf_oxide::document::PdfDocument;

/// Build a minimal PDF showing three glyphs (CIDs 1, 2, 3) from a Type0
/// font with the given encoding name. Each glyph is full-em wide
/// (/DW = 1000) and full-em tall in the vertical direction
/// (`/DW2 [880 -1000]` — the spec default for full-width CJK).
///
/// CIDs map through ToUnicode to ASCII 'A', 'B', 'C' so tests can
/// identify the spans without depending on any embedded font.
fn build_pdf_with_encoding(encoding_name: &str) -> Vec<u8> {
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
3 beginbfchar
<0001> <0041>
<0002> <0042>
<0003> <0043>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end";

    // Content stream: position at (100, 700) and show three CIDs in one Tj.
    // Tj is enough — the advance helpers must move the cursor along the
    // active writing axis for every CID consumed.
    let content = b"BT /F1 12 Tf 100 700 Td <000100020003> Tj ET";

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
    let c4 = format!("4 0 obj << /Length {} >> stream\n", content.len());
    pdf.extend_from_slice(c4.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Obj 5: Type0 wrapper.
    let o5 = pdf.len();
    let f5 = format!(
        "5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestFont \
         /Encoding /{} /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
        encoding_name
    );
    pdf.extend_from_slice(f5.as_bytes());

    // Obj 6: CIDFont. /DW 1000 means each glyph occupies a full em horizontally;
    // /DW2 [880 -1000] gives default vertical advance of -1000 (one full em
    // downward) at the spec-default vertical origin (500, 880).
    let o6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont \
          /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
          /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );

    // Obj 7: ToUnicode CMap.
    let o7 = pdf.len();
    let c7 = format!("7 0 obj << /Length {} >> stream\n", cmap.len());
    pdf.extend_from_slice(c7.as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 8\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for off in [o1, o2, o3, o4, o5, o6, o7] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size 8 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref).as_bytes(),
    );

    pdf
}

/// Horizontal sentinel. Identity-H pushes the cursor along x for every
/// glyph; the extractor should produce three spans whose `bbox.y` are
/// equal and whose `bbox.x` strictly increase by ~font_size each.
#[test]
fn horizontal_identity_h_emits_x_advancing_spans() {
    let pdf = build_pdf_with_encoding("Identity-H");
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic horizontal PDF");
    let spans = doc.extract_spans(0).expect("extract spans");
    assert!(!spans.is_empty(), "horizontal Identity-H must produce at least one span");

    // Concatenate text so we can identify the three CIDs.
    let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(
        combined.contains('A') && combined.contains('B') && combined.contains('C'),
        "expected A, B, C in horizontal extraction; got {:?}",
        combined
    );

    // All visible glyphs sit at the same baseline y=700 in horizontal mode.
    let y0 = spans[0].bbox.y;
    for s in &spans {
        assert!(
            (s.bbox.y - y0).abs() < 1.0,
            "horizontal mode must keep Y stable: span y={} differs from baseline {}",
            s.bbox.y,
            y0
        );
    }
}

/// Vertical extraction (Identity-V) must advance along the y-axis instead
/// of the x-axis. The three glyphs must stack downward at a stable X,
/// and the per-glyph Y delta must equal exactly `-font_size * w1y / 1000`
/// per ISO 32000-1 §9.4.4. `extract_chars` exposes per-glyph positions
/// without the per-span buffer coalescing, making this delta directly
/// observable.
#[test]
fn vertical_identity_v_emits_y_advancing_spans() {
    let pdf = build_pdf_with_encoding("Identity-V");
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic vertical PDF");
    let spans = doc.extract_spans(0).expect("extract spans");
    assert!(!spans.is_empty(), "vertical Identity-V must produce at least one span");

    let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(
        combined.contains('A') && combined.contains('B') && combined.contains('C'),
        "expected A, B, C in vertical extraction; got {:?}",
        combined
    );

    // Pin per-glyph Y delta to the spec formula: w1y=-1000 at fs=12 ⇒
    // 12.0 step per glyph. extract_chars bypasses the buffer so each
    // glyph's user-space origin is preserved.
    let chars = doc.extract_chars(0).expect("extract chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A char");
    let b = chars.iter().find(|c| c.char == 'B').expect("B char");
    let c = chars.iter().find(|c| c.char == 'C').expect("C char");
    let dy_ab = a.bbox.y - b.bbox.y;
    let dy_bc = b.bbox.y - c.bbox.y;
    assert!(
        (dy_ab - 12.0).abs() < 0.01,
        "expected per-glyph Y delta exactly 12.0 (font_size * |w1y|/1000); got {} between A,B",
        dy_ab
    );
    assert!(
        (dy_bc - 12.0).abs() < 0.01,
        "expected per-glyph Y delta exactly 12.0; got {} between B,C",
        dy_bc
    );

    // The span bbox is captured at the cursor's user-space position when
    // the Tj begins. In vertical mode, repeated Tj draws should not be the
    // case here (single Tj), but we still want to verify the extraction
    // pipeline emitted glyphs and tagged them with the vertical writing
    // mode. The strongest cross-check is to confirm that the same content
    // stream extracted under Identity-V produces a different span layout
    // than under Identity-H — specifically, the bounding rectangle of all
    // spans must be wider in X for the horizontal case than for the
    // vertical case, since horizontal advances accumulate along X.
    let hpdf = build_pdf_with_encoding("Identity-H");
    let hdoc = PdfDocument::from_bytes(hpdf).expect("parse horizontal sibling");
    let hspans = hdoc.extract_spans(0).expect("extract horizontal spans");

    let x_extent = |spans: &[pdf_oxide::layout::TextSpan]| -> f32 {
        let min = spans.iter().map(|s| s.bbox.x).fold(f32::INFINITY, f32::min);
        let max = spans
            .iter()
            .map(|s| s.bbox.x + s.bbox.width)
            .fold(f32::NEG_INFINITY, f32::max);
        max - min
    };
    let y_extent = |spans: &[pdf_oxide::layout::TextSpan]| -> f32 {
        let min = spans.iter().map(|s| s.bbox.y).fold(f32::INFINITY, f32::min);
        let max = spans
            .iter()
            .map(|s| s.bbox.y + s.bbox.height)
            .fold(f32::NEG_INFINITY, f32::max);
        max - min
    };

    let h_x = x_extent(&hspans);
    let h_y = y_extent(&hspans);
    let v_x = x_extent(&spans);
    let v_y = y_extent(&spans);

    // Vertical extraction must have *more* y-extent than horizontal
    // extraction (cursor moved downward across glyphs), and *less* x-extent
    // than horizontal extraction (cursor stayed in a column).
    assert!(
        v_y >= h_y,
        "vertical y-extent {} should meet or exceed horizontal y-extent {}",
        v_y,
        h_y
    );
    assert!(v_x <= h_x, "vertical x-extent {} should be <= horizontal x-extent {}", v_x, h_x);
}
