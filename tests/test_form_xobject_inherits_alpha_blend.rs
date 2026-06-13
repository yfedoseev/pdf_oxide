//! Regression test: a Form XObject's content must inherit the constant
//! alpha (`ca`) and blend mode in effect at the `Do` operator.
//!
//! Per ISO 32000-1:2008 §8.10.1, invoking a form XObject is equivalent to
//! splicing its content stream into the page at the point of the `Do`,
//! within the current graphics state. The non-stroking constant alpha and
//! the blend mode set by an `ExtGState` before the `Do` therefore apply to
//! everything the form paints.
//!
//! The defect this pins: the renderer rendered form content with a fresh
//! default graphics state (opaque, Normal blend), so a form painted under
//! `ca 0.5` (or `/BM /Multiply`) was composited fully opaque — an inner
//! image / fill completely overwrote the backdrop instead of blending with
//! it. In a real spot-colour artwork this manifested as a watercolour
//! "swoosh" rendering the wrong hue: the top layer (a form-wrapped image
//! set to Multiply at 0.8) clobbered the layer beneath instead of
//! multiplying into it.
#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{ImageFormat, RenderOptions};

/// Build a PDF where:
/// - A solid black rectangle covers the lower-left quadrant (the backdrop).
/// - A Form XObject that paints a solid WHITE rectangle over the same area
///   is invoked under an ExtGState with `/ca 0.5` (50% constant alpha).
///
/// Correct compositing: white at 50% over black → mid-gray (~128).
/// The bug paints the form opaque → pure white (255).
fn build_pdf_with_form_under_half_alpha() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]\n\
           /Contents 4 0 R\n\
           /Resources << /XObject << /Fm0 5 0 R >> /ExtGState << /GSa 6 0 R >> >> >>\nendobj\n\n",
    );

    // Page content:
    //   1. Paint a black 100x100 rect at (0,0) — the backdrop.
    //   2. Set GSa (ca 0.5) then invoke the form, which paints white over it.
    let page_content = b"0 0 0 rg 0 0 100 100 re f\n/GSa gs\n/Fm0 Do";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject painting a solid white rect over (0,0)-(100,100).
    // The form's own graphics state is the default (Normal, opaque); the
    // 50% comes solely from the caller's GSa.
    let form_stream = b"1 1 1 rg 0 0 100 100 re f";
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 200 200]\n\
            /Resources << >>\n/Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_hdr.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 6: ExtGState with 50% non-stroking constant alpha.
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /ExtGState /ca 0.5 /CA 0.5 >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

/// Average straight-alpha luminance of a pixel region (un-premultiplied).
fn sample_gray(data: &[u8], width: u32, x: u32, y: u32, w: u32, h: u32) -> f32 {
    let mut sum = 0f32;
    let mut count = 0f32;
    for py in y..(y + h) {
        for px in x..(x + w) {
            let idx = ((py * width + px) * 4) as usize;
            if idx + 3 < data.len() {
                let a = data[idx + 3] as f32;
                if a > 0.0 {
                    let scale = 255.0 / a;
                    sum += (data[idx] as f32 * scale).min(255.0);
                    count += 1.0;
                }
            }
        }
    }
    if count == 0.0 {
        0.0
    } else {
        sum / count
    }
}

#[test]
fn form_xobject_content_inherits_caller_constant_alpha() {
    let pdf = build_pdf_with_form_under_half_alpha();
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");

    let mut options = RenderOptions::with_dpi(72);
    options.format = ImageFormat::RawRgba8;
    let img = pdf_oxide::rendering::render_page(&doc, 0, &options).expect("render");

    // The black+white overlap region is the lower-left quadrant. At 72 DPI
    // the 100x100 PDF rect maps 1:1 to device pixels; device y is flipped,
    // so the rect occupies the bottom of the 200x200 image. Sample a patch
    // well inside it.
    let gray = sample_gray(&img.data, img.width, 25, 125, 50, 50);

    // White (255) over black (0) at ca=0.5 composites to ~128. The bug
    // paints the form opaque, yielding ~255. Require the result to be a
    // genuine mid-gray blend, not opaque white.
    assert!(
        gray < 200.0,
        "form painted under ca=0.5 must blend with the backdrop \
         (expected ~128 mid-gray); got luminance {gray:.1} — the form's \
         content is being composited opaque, ignoring the caller's alpha"
    );
    assert!(
        gray > 80.0,
        "blended result should be near mid-gray (~128), not near-black; \
         got luminance {gray:.1}"
    );
}
