//! Separation (spot-colour) fills whose tint transform is a Type 0 (sampled)
//! or Type 3 (stitching) function must render in their REAL alternate-space
//! colour. Previously only Type 2 and Type 4 tint transforms were evaluated;
//! Types 0 and 3 fell back to a `gray = 1 - tint` heuristic, so e.g. PANTONE
//! spot fills defined with sampled tint transforms rendered near-black.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// Minimal one-page PDF whose whole page is filled with tint 1.0 of a
/// Separation colour space using the given tint-transform function object
/// (object 5). `func_body` is the complete `5 0 obj ... endobj` text (dict
/// functions) or None to use a Type 0 SAMPLED stream mapping
/// tint 0 -> white, tint 1 -> pure red in a DeviceRGB alternate.
fn spot_pdf(func_obj5: Option<&str>) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
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
         /Resources << /ColorSpace << /CS0 [/Separation /Spot /DeviceRGB 5 0 R] >> >> \
         /Contents 4 0 R >>",
    );
    let content = b"/CS0 cs 1 scn 0 0 200 200 re f\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    match func_obj5 {
        Some(body) => obj(&mut buf, &mut off, 5, body),
        None => {
            // Type 0 sampled: Domain [0 1], Range [0 1 0 1 0 1], Size [2],
            // 8-bit samples: t=0 -> (FF FF FF) white, t=1 -> (FF 00 00) red.
            let samples: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00];
            off[5] = buf.len();
            buf.extend_from_slice(
                format!(
                    "5 0 obj\n<< /FunctionType 0 /Domain [0 1] /Range [0 1 0 1 0 1] \
                     /Size [2] /BitsPerSample 8 /Length {} >>\nstream\n",
                    samples.len()
                )
                .as_bytes(),
            );
            buf.extend_from_slice(samples);
            buf.extend_from_slice(b"\nendstream\nendobj\n");
        },
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
fn type0_sampled_tint_renders_alternate_colour_not_gray() {
    let [r, g, b, _a] = centre_pixel(spot_pdf(None));
    // tint 1.0 through the ramp -> pure red. The old fallback painted
    // gray = 1 - tint = black. Allow generous tolerance for colour management.
    assert!(
        r > 200 && g < 80 && b < 80,
        "expected red-ish spot fill, got rgb({r},{g},{b}) - Type 0 tint not evaluated?"
    );
}

#[test]
fn type3_stitching_tint_renders_second_slice_colour() {
    // Stitch two Type 2 dicts: [0, 0.5) white->blue, [0.5, 1] constant green.
    // Filling at tint 1.0 lands in the second slice -> green.
    let func = "<< /FunctionType 3 /Domain [0 1] /Bounds [0.5] /Encode [0 1 0 1] \
                /Functions [ \
                << /FunctionType 2 /Domain [0 1] /N 1 /C0 [1 1 1] /C1 [0 0 1] >> \
                << /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 1 0] /C1 [0 1 0] >> ] >>";
    let [r, g, b, _a] = centre_pixel(spot_pdf(Some(func)));
    assert!(
        g > 200 && r < 80 && b < 80,
        "expected green spot fill from the second slice, got rgb({r},{g},{b})"
    );
}
