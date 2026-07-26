//! Inline images (`BI` / `ID` / `EI`) must actually PAINT onto the page,
//! not just be extractable via `PdfDocument::extract_images` (already
//! covered by `tests/inline_image_decode.rs`). `Operator::InlineImage` was
//! parsed and even classified as a paint operator for knockout-group
//! segmentation, but the operator-execution `match` in
//! `execute_operators` had no arm for it at all — inline images were
//! silently dropped from the rendered page.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// One-page PDF whose whole page is covered by a single inline image with
/// the given dictionary text and raw sample bytes.
fn pdf_with_full_page_inline_image(dict: &str, samples: &[u8]) -> Vec<u8> {
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(b"q 200 0 0 200 0 0 cm\n");
    content.extend_from_slice(format!("BI {dict} ID ").as_bytes());
    content.extend_from_slice(samples);
    content.extend_from_slice(b"\nEI\nQ\n");

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 5];
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(&content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for id in 1..=4 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n");
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
fn inline_rgb_image_paints_onto_the_page() {
    // 1x1 pure-green RGB inline image covering the whole page. Before the
    // fix the page stayed the default white background — the operator was
    // parsed and then dropped, not an error, so nothing indicated failure.
    let green = [0x00u8, 0xFF, 0x00];
    let pdf = pdf_with_full_page_inline_image("/W 1 /H 1 /CS /RGB /BPC 8", &green);
    let [r, g, b, a] = centre_pixel(pdf);
    assert!(
        r < 40 && g > 200 && b < 40 && a > 200,
        "expected the inline image's green to be painted, got rgba({r},{g},{b},{a}) \
         — inline image not rendered?"
    );
}

#[test]
fn inline_image_mask_paints_with_current_fill_colour() {
    // A 1x1 stencil mask, sample bit 0 (default /Decode [0 1], the mark
    // colour) covering the whole page, filled with the current (blue) fill
    // colour — mirrors how a `Do`-invoked /ImageMask XObject already
    // behaves, which inline images must match.
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(b"0 0 1 rg\n"); // blue fill
    content.extend_from_slice(b"q 200 0 0 200 0 0 cm\n");
    content.extend_from_slice(b"BI /W 1 /H 1 /CS /G /BPC 1 /IM true ID ");
    content.extend_from_slice(&[0x00u8]); // single bit 0 -> "paints" by default
    content.extend_from_slice(b"\nEI\nQ\n");

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 5];
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(&content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for id in 1..=4 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());

    let [r, g, b, a] = centre_pixel(buf);
    assert!(
        r < 40 && g < 40 && b > 200 && a > 200,
        "expected the inline image mask to paint the current blue fill, got rgba({r},{g},{b},{a}) \
         — inline image mask not rendered?"
    );
}
