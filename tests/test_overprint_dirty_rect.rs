//! Dirty-rect bounding for the §11.7.4 overprint after-paint pass.
//!
//! `apply_overprint_after_paint` historically snapshotted and re-scanned
//! the full page pixmap on every paint operator with `/OP`/`/op` active.
//! These probes pin the rect-bounded behaviour: the scan must touch no
//! more than a conservative device-space bound of the painted geometry,
//! while unboundable paints (shadings, which fill the whole clip) keep
//! the full-page scan as a safe fallback.
//!
//! The probe counter (`PageRenderer::overprint_scanned_pixels`, gated on
//! `test-support`) counts pixels iterated by the overprint scan during
//! the most recent `render_page*` call. Counters, not wall-clock: exact
//! and immune to machine noise (same rationale as the ICC cache-count
//! probes in the round-3 QA pass).
//!
//! Byte equivalence with the pre-rect behaviour is established by the
//! corpus A/B harness (full-page vs rect builds must render 100%
//! byte-identical pages); these tests pin the *bounding* contract so a
//! regression back to full-page scanning surfaces loudly.

#![cfg(all(feature = "rendering", feature = "icc", feature = "test-support"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{PageRenderer, RenderOptions};

/// Minimal single-page PDF: 100×100pt MediaBox, given resources and
/// content stream, no OutputIntent (plain RGB composite render).
fn build_pdf(content: &str, resources_inner: &str) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << {} >> /Contents 4 0 R >>\nendobj\n",
        resources_inner
    );
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

const OP_GS: &str = "/ExtGState << /Ov << /Type /ExtGState /OP true /op true >> >>";

/// Render page 0 at 72 DPI (100×100pt → 139×139 px at the renderer's
/// default scaling) and return (scanned, total_pixels).
fn scanned_for(pdf: Vec<u8>) -> (u64, u64) {
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF must parse");
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72));
    let img = renderer.render_page(&doc, 0).expect("render");
    let total = (img.width as u64) * (img.height as u64);
    (renderer.overprint_scanned_pixels(), total)
}

/// A 20×20pt DeviceCMYK fill with /OP true on a 100×100pt page must
/// scan only a rect-bounded neighbourhood of the square — not the page.
/// Generous 25%-of-page ceiling: the rect is ~4% of the page, the old
/// full-page scan is 100%.
#[test]
fn small_cmyk_fill_scan_is_rect_bounded() {
    let content = "/Ov gs 0 0.5 0 0 k 10 10 20 20 re f";
    let (scanned, total) = scanned_for(build_pdf(content, OP_GS));
    assert!(scanned > 0, "overprint pass must run for a /OP true DeviceCMYK fill");
    assert!(
        scanned <= total / 4,
        "overprint scan must be rect-bounded: scanned {scanned} of {total} pixels \
         for a 20×20pt square on a 100×100pt page"
    );
}

/// A short Tj with /OP true must scan only the glyph-bounded rect.
#[test]
fn small_text_scan_is_rect_bounded() {
    let resources = format!(
        "{} /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >>",
        OP_GS
    );
    let content = "/Ov gs 0 0.5 0 0 k BT /F1 12 Tf 20 50 Td (Hi) Tj ET";
    let (scanned, total) = scanned_for(build_pdf(content, &resources));
    assert!(scanned > 0, "overprint pass must run for a /OP true Tj paint");
    assert!(
        scanned <= total / 4,
        "text overprint scan must be glyph-bounded: scanned {scanned} of {total} pixels \
         for a 12pt two-glyph string on a 100×100pt page"
    );
}

/// A stroked path with /OP true must scan a rect that includes the
/// stroke expansion but stays far below the page.
#[test]
fn small_stroke_scan_is_rect_bounded() {
    let content = "/Ov gs 0 0.5 0 0 K 4 w 20 20 m 40 40 l S";
    let (scanned, total) = scanned_for(build_pdf(content, OP_GS));
    assert!(scanned > 0, "overprint pass must run for a /OP true stroke");
    assert!(
        scanned <= total / 4,
        "stroke overprint scan must be rect-bounded: scanned {scanned} of {total} pixels"
    );
}

/// `sh` paints the entire current clip region — there is no provable
/// paint bbox, so the pass must keep the full-page scan (safe fallback,
/// identical to the historical behaviour).
#[test]
fn shading_scan_falls_back_to_full_page() {
    let resources = format!(
        "{} /Shading << /Sh0 << /ShadingType 2 /ColorSpace /DeviceGray \
         /Coords [0 0 100 100] /Function << /FunctionType 2 /Domain [0 1] \
         /C0 [0] /C1 [1] /N 1 >> >> >>",
        OP_GS
    );
    let content = "/Ov gs /Sh0 sh";
    let (scanned, total) = scanned_for(build_pdf(content, &resources));
    assert_eq!(
        scanned, total,
        "sh has no provable paint bbox; the overprint scan must remain full-page"
    );
}

/// Without /OP the gate is closed: no snapshot, no scan. Pins that the
/// rect machinery adds zero work to non-overprint documents.
#[test]
fn no_overprint_means_no_scan() {
    let content = "0 0.5 0 0 k 10 10 20 20 re f";
    let (scanned, _) = scanned_for(build_pdf(content, ""));
    assert_eq!(scanned, 0, "no /OP in the ExtGState ⇒ the overprint pass must not run");
}
