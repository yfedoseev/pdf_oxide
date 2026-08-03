//! A rotated run must be able to report where the text sits on the page.
//!
//! A run drawn with `Tm [0 1 -1 0]` advances up the page, so its page-space box
//! is tall and narrow. `bbox` carries the run's own extents instead — `width`
//! is the advance along the writing axis whatever the rotation — so a
//! vertically drawn run's `bbox` is wide and short.
//!
//! `bbox` keeps that meaning: it is the frame every existing consumer reads,
//! and redefining it would move every word box on every rotated page. The
//! page-space rectangle is exposed as a derived accessor instead.

use pdf_oxide::document::PdfDocument;

/// One 90°-rotated run of two words, drawn up the page from (200, 200).
fn rotated_run_pdf() -> Vec<u8> {
    let content = b"BT /F1 10 Tf 0 1 -1 0 200 200 Tm (Alpha Bravo) Tj ET".to_vec();
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

#[test]
fn rotated_run_reports_a_page_space_box() {
    let doc = PdfDocument::from_bytes(rotated_run_pdf()).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");

    let rotated: Vec<_> = spans.iter().filter(|s| s.rotation_degrees != 0.0).collect();
    assert!(!rotated.is_empty(), "fixture produced no rotated spans");

    for s in rotated {
        let page = s.page_bbox();
        assert!(
            page.height > page.width,
            "run drawn vertically reports a horizontal page box: {:?} is {}x{}",
            s.text,
            page.width,
            page.height
        );
        // The run's own extents are unchanged — this is an added view, not a
        // redefinition, so nothing reading `bbox` today moves.
        assert!(
            s.bbox.width > s.bbox.height,
            "run's own extents changed: {:?} is {}x{}",
            s.text,
            s.bbox.width,
            s.bbox.height
        );
        // Rotating a rectangle preserves its area, so the two views must agree
        // on how much page the run covers.
        let (a, b) = (page.width * page.height, s.bbox.width * s.bbox.height);
        assert!(
            (a - b).abs() <= 0.01 * b.max(1.0),
            "page box area {a} does not match run extents area {b}"
        );
    }
}

/// The accessor is a no-op on upright text, which is what keeps every
/// unrotated page byte-identical.
#[test]
fn upright_run_page_box_equals_its_bbox() {
    let content = b"BT /F1 10 Tf 1 0 0 1 100 700 Tm (Alpha Bravo) Tj ET".to_vec();
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    let doc = PdfDocument::from_bytes(pdf).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");
    assert!(!spans.is_empty(), "fixture produced no spans");

    for s in &spans {
        assert_eq!(s.rotation_degrees, 0.0, "fixture span is not upright");
        let page = s.page_bbox();
        assert_eq!(
            (page.x, page.y, page.width, page.height),
            (s.bbox.x, s.bbox.y, s.bbox.width, s.bbox.height),
            "page_bbox moved an upright run"
        );
    }
}

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

/// A subscript inside a rotated run sits off the run's writing axis by the
/// subscript drop, not by a line height. It must stay attached to the formula
/// it belongs to: `N`, a smaller `2`, and `O` drawn as three runs of one
/// rotated label are one chemical formula, not three fragments separated by
/// unrelated page content.
fn rotated_formula_with_subscript_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // A 90-degree label "N2O" whose middle glyph is dropped below the baseline,
    // the shape a rotated chart axis uses. Under a 90-degree matrix the drop is
    // a displacement along -x, i.e. PERPENDICULAR to the +y writing axis — the
    // exact quantity the continuation test measures.
    //
    // One BT/ET and one Tf size throughout, deliberately: `ET` flushes the run
    // buffer and so does a Tf size change, either of which would leave the
    // buffer empty at the next Tm and make the continuation test unreachable —
    // the assertion below would then hold no matter what that test decided.
    content.extend_from_slice(b"BT /F1 18 Tf\n");
    content.extend_from_slice(b"0 1 -1 0 246 244 Tm (N) Tj\n");
    content.extend_from_slice(b"0 1 -1 0 242 258 Tm (2) Tj\n");
    content.extend_from_slice(b"0 1 -1 0 246 266 Tm (O) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// A baseline drop inside a rotated run is a sub-glyph perpendicular offset, not
/// a line break: the formula must not be split apart by the continuation test.
#[test]
fn rotated_subscript_formula_stays_contiguous() {
    let doc = PdfDocument::from_bytes(rotated_formula_with_subscript_pdf()).expect("parse fixture");
    let text = doc.extract_text(0).expect("extract text");
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join("");
    assert!(flat.contains("N2O"), "rotated subscripted formula came apart: {text:?}");
}
