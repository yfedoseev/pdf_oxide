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

const EPS: f32 = 0.05;

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
        // Position, not just shape: `Tm [0 1 -1 0 200 200]` runs the baseline
        // up the page at x = 200 with glyphs extending LEFT one font size, so
        // the hull is [190, 200] × [origin_y, origin_y + advance]. Pins the
        // pivot and both axis signs — a perpendicular-axis sign flip would
        // land the box at [200, 210], on the wrong side of the baseline.
        assert!(
            (page.x - 190.0).abs() < EPS && (page.x - (s.bbox.x - s.bbox.height)).abs() < EPS,
            "page box left edge {} is not one font size left of the baseline",
            page.x
        );
        assert!(
            (page.y - s.bbox.y).abs() < EPS,
            "page box bottom {} does not sit at the run origin y {}",
            page.y,
            s.bbox.y
        );
        assert!(
            (page.width - s.bbox.height).abs() < EPS && (page.height - s.bbox.width).abs() < EPS,
            "page box {}x{} did not swap the run extents {}x{}",
            page.width,
            page.height,
            s.bbox.width,
            s.bbox.height
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

/// The same sideways run on a `/Rotate 90` page — #806's sideways-landscape
/// class. `postprocess_spans` has already rewritten `bbox` into the displayed
/// frame (recorded in `page_rotation_applied`) while keeping
/// `rotation_degrees` at the raw content angle, so `page_bbox` must build its
/// hull in the displayed frame instead of re-rotating the mapped rect.
///
/// Displayed (clockwise 90°: display-x = raw-y, display-y = 612 − raw-x) the
/// run reads horizontally from (200, 412), glyphs one font size above.
#[test]
fn rotate90_page_span_page_box_is_not_double_transformed() {
    let content = b"BT /F1 10 Tf 0 1 -1 0 200 200 Tm (Alpha Bravo) Tj ET".to_vec();
    let pdf = build_minimal_pdf_raw(
        &content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");

    let rotated: Vec<_> = spans.iter().filter(|s| s.rotation_degrees != 0.0).collect();
    assert!(!rotated.is_empty(), "fixture produced no rotated spans");

    for s in rotated {
        assert_eq!(
            s.page_rotation_applied, 90,
            "span bbox was not mapped into the displayed frame"
        );
        let page = s.page_bbox();
        // Baseline of the displayed horizontal run: y = 612 − 200 = 412, at
        // the mapped rect's top edge (the image of the run origin).
        assert!(
            (page.y - 412.0).abs() < EPS && (page.y - (s.bbox.y + s.bbox.height)).abs() < EPS,
            "page box bottom {} is not on the displayed baseline 412",
            page.y
        );
        assert!(
            (page.x - s.bbox.x).abs() < EPS,
            "page box left edge {} does not start at the displayed run origin {}",
            page.x,
            s.bbox.x
        );
        assert!(
            (page.height - 10.0).abs() < EPS && (page.width - s.bbox.height).abs() < EPS,
            "displayed run is horizontal: expected {}x10 box, got {}x{}",
            s.bbox.height,
            page.width,
            page.height
        );
    }
}

/// `Tm [0 1 1 0]` carries the same 90° angle as a clean rotation but a
/// negative determinant: the glyph-up axis is the writing axis *reflected*,
/// landing on (+1, 0). The span's `mirrored` flag tells `page_bbox` to
/// reflect the across-axis, so the box sits on the glyph side of the
/// baseline — [200, 210], not [190, 200].
#[test]
fn mirrored_run_page_box_sits_on_the_glyph_side_of_the_baseline() {
    let content = b"BT /F1 10 Tf 0 1 1 0 200 200 Tm (Alpha Bravo) Tj ET".to_vec();
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    let doc = PdfDocument::from_bytes(pdf).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");

    let rotated: Vec<_> = spans.iter().filter(|s| s.rotation_degrees != 0.0).collect();
    assert!(!rotated.is_empty(), "fixture produced no rotated spans");

    for s in rotated {
        assert!(s.mirrored, "negative-determinant run lost its mirror flag");
        let page = s.page_bbox();
        assert!(
            (page.x - 200.0).abs() < EPS && (page.x - s.bbox.x).abs() < EPS,
            "mirrored run's box left edge {} is on the wrong side of the x = 200 baseline",
            page.x
        );
        assert!(
            (page.y - s.bbox.y).abs() < EPS,
            "page box bottom {} does not sit at the run origin y {}",
            page.y,
            s.bbox.y
        );
        assert!(
            (page.width - s.bbox.height).abs() < EPS && (page.height - s.bbox.width).abs() < EPS,
            "page box {}x{} did not swap the run extents {}x{}",
            page.width,
            page.height,
            s.bbox.width,
            s.bbox.height
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
