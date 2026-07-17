//! A Separation colour space whose ALTERNATE colour space is an indirect
//! reference (e.g. `[/Separation /Spot 6 0 R 5 0 R]` with `6 0 obj =
//! /DeviceRGB`) must project the evaluated tint transform through that
//! alternate. `resolve_separation_or_devicen` previously inspected
//! `alt_cs_obj` (the raw, UNRESOLVED array element) via `.as_name()` and
//! `if let Object::Array(_) = ...`; both fail on a bare `Reference`, so
//! control fell through to `first_as_gray`, turning a pure-red spot fill
//! into gray/white.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// Minimal one-page PDF whose whole page is filled at tint 1.0 with a
/// Separation colour space `[/Separation /Spot 6 0 R 5 0 R]`, where object 6
/// is an INDIRECT reference to the bare name `/DeviceRGB` (the alternate
/// colour space) and object 5 is a Type 2 tint transform mapping tint 1.0 to
/// pure red `[1 0 0]`.
fn indirect_alt_cs_pdf() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
         /Resources << /ColorSpace << /CS0 [/Separation /Spot 6 0 R 5 0 R] >> >> \
         /Contents 4 0 R >>",
    );
    let content = b"/CS0 cs 1 scn 0 0 200 200 re f\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Type 2 (exponential interpolation) tint transform: tint 1.0 -> [1 0 0].
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0] /C1 [1 0 0] >>",
    );

    // The alternate colour space itself, as an INDIRECT reference to a bare
    // name object - this is the shape that previously defeated .as_name().
    obj(&mut buf, &mut off, 6, "/DeviceRGB");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// Render page 0 as raw RGBA and return the centre pixel.
fn centre_pixel(pdf: Vec<u8>) -> [u8; 4] {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let mut opts = RenderOptions::default();
    opts.dpi = 72;
    opts.format = pdf_oxide::rendering::ImageFormat::RawRgba8;
    let img = render_page(&doc, 0, &opts).expect("render");
    let (w, h) = (img.width as usize, img.height as usize);
    let at = (h / 2 * w + w / 2) * 4;
    [
        img.data[at],
        img.data[at + 1],
        img.data[at + 2],
        img.data[at + 3],
    ]
}

#[test]
fn indirect_alternate_colour_space_renders_true_colour_not_gray() {
    let [r, g, b, _a] = centre_pixel(indirect_alt_cs_pdf());
    // The old fallback (first_as_gray on the ALREADY-EVALUATED altspace
    // values [1, 0, 0]) took the first component as gray -> gray(1.0) =
    // WHITE, not red. The alternate is really /DeviceRGB, so the resolved
    // tint transform output [1 0 0] must render pure red.
    assert!(
        r > 200 && g < 80 && b < 80,
        "expected red spot fill via indirect /DeviceRGB alternate, got rgb({r},{g},{b}) \
         - indirect alt_cs_obj not resolved?"
    );
}
