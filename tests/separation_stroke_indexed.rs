//! Regression tests for two fixes that fell out of unifying the renderer's
//! colour-operator resolution into one shared path
//! ([`pdf_oxide::rendering::color_resolve::resolve_color_to_rgb`]):
//!
//! 1. **Stroke Separation/DeviceN** — the `SC`/`SCN` (stroke) operators
//!    previously had no Separation/DeviceN arm (only fill did), so a stroked
//!    spot colour fell back to `grey = 1 - tint`, i.e. solid black at full tint.
//!    Now stroke routes through the same tint-transform resolution as fill.
//! 2. **Indexed colour space** — the renderer's Indexed handling was a crude
//!    `index / 255` grayscale guess; it now performs a real palette lookup.
//!
//! The PDFs are built in-memory at a tiny 10×10 MediaBox (no committed binary
//! fixtures) so the tests stay self-contained and cheap to render.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page_fit, ImageFormat, RenderOptions};

/// Assemble a minimal PDF with a classic xref table from 1-indexed object
/// bodies. Object 1 must be the catalog (it is used as `/Root`). Offsets are
/// computed as the file is built, so the bodies can be any valid object text.
fn build_pdf(objects: &[String]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body.as_bytes());
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = out.len();
    let size = objects.len() + 1;
    out.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<</Size {size}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF").as_bytes(),
    );
    out
}

/// A content stream object (`<</Length N>>stream … endstream`) with the byte
/// length computed from `content`.
fn stream_obj(content: &str) -> String {
    format!(
        "<</Length {}>>\nstream\n{content}\nendstream",
        content.len()
    )
}

/// Render the given PDF into a 50×50 raster and return its centre pixel RGB.
fn centre_pixel(pdf: &[u8]) -> (u8, u8, u8) {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open pdf");
    let options = RenderOptions::with_dpi(72).as_raw();
    let image = render_page_fit(&doc, 0, 50, 50, &options).expect("render");
    assert_eq!(image.format, ImageFormat::RawRgba8);
    let (cx, cy) = (image.width / 2, image.height / 2);
    let i = ((cy * image.width + cx) * 4) as usize;
    (image.data[i], image.data[i + 1], image.data[i + 2])
}

/// A full-tint Separation *stroke* (`/CsTest CS 1 SCN`) over a thick line
/// through the page centre. The tint transform maps tint 1.0 → CMYK(0.1,0,0.15,0)
/// → light green. Before the fix the stroke path ignored Separation and rendered
/// `grey = 1 - 1 = 0` (black); now it resolves the tint transform like fill.
#[test]
fn separation_stroke_is_not_black() {
    let pdf = build_pdf(&[
        "<</Type/Catalog/Pages 2 0 R>>".to_string(),
        "<</Type/Pages/Count 1/Kids[3 0 R]>>".to_string(),
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]\
         /Contents 4 0 R/Resources<</ColorSpace<</CsTest[/Separation/Spot/DeviceCMYK 5 0 R]>>>>>>"
            .to_string(),
        stream_obj("/CsTest CS 1 SCN 10 w 0 5 m 10 5 l S"),
        "<</FunctionType 2/Domain[0 1]/C0[0 0 0 0]/C1[0.1 0 0.15 0]/N 1>>".to_string(),
    ]);
    let (r, g, b) = centre_pixel(&pdf);
    assert!(
        r > 150 && g > 150 && b > 150,
        "Separation stroke rendered dark (r={r}, g={g}, b={b}) — tint transform not applied to SCN?",
    );
    assert!(g >= r && g >= b, "expected green-dominant stroke, got ({r},{g},{b})");
}

/// An Indexed colour space `[/Indexed /DeviceRGB 1 <00FF00 FF0000>]` filled at
/// index 1, which must look up palette entry 1 = pure red (FF0000). The old
/// `index / 255` grayscale guess produced near-black `(1/255)` grey instead.
#[test]
fn indexed_fill_does_palette_lookup() {
    let pdf = build_pdf(&[
        "<</Type/Catalog/Pages 2 0 R>>".to_string(),
        "<</Type/Pages/Count 1/Kids[3 0 R]>>".to_string(),
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]\
         /Contents 4 0 R/Resources<</ColorSpace<</CsIdx[/Indexed/DeviceRGB 1<00FF00FF0000>]>>>>>>"
            .to_string(),
        stream_obj("/CsIdx cs 1 sc 0 0 10 10 re f"),
    ]);
    let (r, g, b) = centre_pixel(&pdf);
    assert!(
        r > 200 && g < 80 && b < 80,
        "Indexed fill at index 1 expected red (FF0000), got ({r},{g},{b})",
    );
}
