//! Overprint must not erase a Separation paint on a composite RGB render.
//!
//! ISO 32000-1:2008 §11.7.3: a Separation or DeviceN source may address the
//! device's process colorants "as if they were spot colours" only when the
//! group inherits the output device's native colour space. Otherwise "the
//! Separation or DeviceN colour space **shall be converted to its alternate
//! colour space**", and §11.7.4.3 NOTE 2 then reads that alternate as the
//! current colour space for Table 149 — its "any process colour space" row,
//! `B = c_s`. Table 149 NOTE 1 says the same from the other side: the group's
//! process components "cannot be treated as if they were spot colours in a
//! Separation or DeviceN colour space".
//!
//! With no CMYK sidecar the composite pixmap *is* the group colour space, and
//! it is RGB. Applying Table 149 row 3 there preserved the backdrop on all
//! four process lanes with no spot lane in existence to receive `c_s`, so the
//! paint vanished: two scholarly documents rendered their whole body text
//! invisibly, at coverage 0.00009 and 0.00286, where MuPDF, pdfium, poppler
//! and Ghostscript all paint them.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

/// A 100x100 page that fills a rectangle through
/// `[/Separation /Black <DeviceRGB alternate> <tint transform>]` at tint 1.0,
/// with overprint enabled in nonzero mode. No `/OutputIntents`, so no CMYK
/// sidecar exists and the render is a plain composite.
fn separation_overprint_pdf(with_overprint: bool) -> Vec<u8> {
    let gs = if with_overprint { "/GS0 gs " } else { "" };
    let content = format!("{gs}/CS0 cs 1 scn 10 10 80 80 re f");
    // Tint 1.0 -> black. A type-2 exponential function from one input to three
    // DeviceRGB outputs: C0 white, C1 black.
    let tint = "<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 0 0] /N 1 \
                /Range [0 1 0 1 0 1] >>";

    let mut pdf: Vec<u8> = Vec::new();
    let mut off = [0usize; 8];
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.4\n");
    off[1] = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    push!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Contents 4 0 R /Resources << /ColorSpace << /CS0 5 0 R >> \
         /ExtGState << /GS0 6 0 R >> >> >>\nendobj\n"
    );
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(content.as_bytes());
    push!("\nendstream\nendobj\n");
    off[5] = pdf.len();
    push!(format!("5 0 obj\n[ /Separation /Black /DeviceRGB {tint} ]\nendobj\n"));
    off[6] = pdf.len();
    push!("6 0 obj\n<< /Type /ExtGState /OP true /op true /OPM 1 >>\nendobj\n");

    let xref = pdf.len();
    push!("xref\n0 7\n0000000000 65535 f \r\n");
    for id in 1..=6 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

/// Ink coverage of the rendered page.
fn coverage(pdf: Vec<u8>) -> f64 {
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgba8();
    let n = px.pixels().len() as f64;
    let inked = px
        .pixels()
        .filter(|p| p[3] > 0 && (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3 < 250)
        .count();
    inked as f64 / n
}

#[test]
fn overprint_does_not_erase_a_separation_fill_on_a_composite_render() {
    let with = coverage(separation_overprint_pdf(true));
    // The rect is 80x80 on a 100x100 page: 64% of it.
    assert!(
        with > 0.55,
        "a Separation fill must survive overprint on a composite render \
         (§11.7.3 reverts it to its alternate space); covered {with:.5}"
    );
}

/// Control: the identical page without the overprint ExtGState. If the two
/// disagree, overprint is still altering a paint it has no authority over.
#[test]
fn overprint_makes_no_difference_to_a_separation_fill_here() {
    let with = coverage(separation_overprint_pdf(true));
    let without = coverage(separation_overprint_pdf(false));
    assert!(
        (with - without).abs() < 0.01,
        "enabling overprint changed a composite Separation fill: \
         {with:.5} with, {without:.5} without"
    );
}
