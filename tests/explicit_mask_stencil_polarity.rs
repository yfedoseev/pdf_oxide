//! An explicit `/Mask` stencil must hide the samples the specification says
//! it hides.
//!
//! ISO 32000-1:2008 §8.9.6.2 (`docs/spec/pdf.md:14903`), verbatim:
//!
//! > …(the default for an image mask), **a sample value of 0 shall mark the
//! > page with the current colour, and a 1 shall leave the previous contents
//! > unchanged.** If the **Decode** array is [ 1 0 ], these meanings shall be
//! > reversed.
//!
//! For an explicit `/Mask` on a base image, "mark the page" means the base
//! image shows through. So with the default `/Decode [0 1]`, **0 is opaque**
//! and 1 masks out. The renderer implemented the complement, under a comment
//! citing this clause — so a reader checking the citation against the code saw
//! two things agree and both be wrong.
//!
//! Polarity and `/Decode` are tested together because they compose: a
//! `/Decode [1 0]` mask rendered correctly *by accident* before the fix, two
//! errors cancelling, so correcting one alone would have broken those files.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
use pdf_oxide::PdfDocument;

/// A page painting a solid red 16x1 image over a blue background, masked by a
/// 16x1 stencil whose left half is sample 0 and right half sample 1.
///
/// With the default `/Decode`, the left half must show the red image and the
/// right half must show the blue backdrop. Sixteen texels rather than two so
/// the resampling ramp stays near the boundary and each half has an interior
/// that is unambiguously one value or the other.
fn masked_image_pdf(decode: &str) -> Vec<u8> {
    // Base: 16x1 DeviceRGB, every pixel red.
    let base: Vec<u8> = (0..16).flat_map(|_| [255u8, 0, 0]).collect();
    // Stencil: 16 one-bit samples, MSB first — eight 0s then eight 1s.
    let stencil: [u8; 2] = [0x00, 0xFF];

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
         /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
    );
    // Blue backdrop, then the masked image over the whole page.
    let content = b"0 0 1 rg 0 0 200 100 re f\nq 200 0 0 100 0 0 cm /Im0 Do Q\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    stream(
        &mut buf,
        &mut off,
        5,
        "/Type /XObject /Subtype /Image /Width 16 /Height 1 /ColorSpace /DeviceRGB \
         /BitsPerComponent 8 /Mask 6 0 R",
        &base,
    );
    stream(
        &mut buf,
        &mut off,
        6,
        &format!(
            "/Type /XObject /Subtype /Image /Width 16 /Height 1 /ImageMask true \
             /BitsPerComponent 1 {decode}"
        ),
        &stencil,
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// `(left_half, right_half)` centre pixels as RGB.
fn halves(pdf: Vec<u8>) -> ([u8; 3], [u8; 3]) {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let mut opts = RenderOptions::default();
    opts.dpi = 72;
    opts.format = ImageFormat::RawRgba8;
    let img = render_page(&doc, 0, &opts).expect("render");
    let (w, h) = (img.width as usize, img.height as usize);
    let at = |x: usize| {
        let i = (h / 2 * w + x) * 4;
        [img.data[i], img.data[i + 1], img.data[i + 2]]
    };
    (at(w / 8), at(w * 7 / 8))
}

fn is_red(px: [u8; 3]) -> bool {
    px[0] > 200 && px[1] < 80 && px[2] < 80
}

fn is_blue(px: [u8; 3]) -> bool {
    px[2] > 200 && px[0] < 80 && px[1] < 80
}

/// Default `/Decode`: sample 0 paints, sample 1 masks out.
#[test]
fn default_decode_paints_where_the_sample_is_zero() {
    let (left, right) = halves(masked_image_pdf(""));
    assert!(
        is_red(left),
        "sample 0 must mark the page, so the base image shows; got {left:?}"
    );
    assert!(
        is_blue(right),
        "sample 1 must leave the previous contents unchanged; got {right:?}"
    );
}

/// An explicit `/Decode [0 1]` is the default, spelled out.
#[test]
fn explicit_default_decode_behaves_as_the_default() {
    let (left, right) = halves(masked_image_pdf("/Decode [0 1]"));
    assert!(is_red(left), "got {left:?}");
    assert!(is_blue(right), "got {right:?}");
}

/// `/Decode [1 0]` reverses the meanings. This case rendered correctly before
/// the fix by accident — two errors cancelling — so it must stay correct.
#[test]
fn reversed_decode_reverses_the_meanings() {
    let (left, right) = halves(masked_image_pdf("/Decode [1 0]"));
    assert!(is_blue(left), "with /Decode [1 0] a 0 sample masks out; got {left:?}");
    assert!(is_red(right), "with /Decode [1 0] a 1 sample paints; got {right:?}");
}

/// The two decodes must be each other's complement — the property that makes
/// the polarity rule a rule rather than a pair of special cases.
#[test]
fn the_two_decodes_are_complementary() {
    let (default_left, default_right) = halves(masked_image_pdf(""));
    let (reversed_left, reversed_right) = halves(masked_image_pdf("/Decode [1 0]"));
    assert_eq!(default_left, reversed_right);
    assert_eq!(default_right, reversed_left);
}
