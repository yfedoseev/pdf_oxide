//! End-to-end: an inline image (`BI` / `ID` / `EI`) must decode to the EXACT
//! pixels the content stream carries.
//!
//! The unit tests pin `expand_inline_image_dict` in isolation. This one goes
//! through the public API a real caller uses - `PdfDocument::extract_images` -
//! and asserts the decoded bytes, so the two defects that made inline images
//! silently vanish stay fixed end to end:
//!
//!   1. the implied `/Subtype /Image` (s8.9.7) the decoder requires;
//!   2. the ABBREVIATED VALUES (Table 93/94): `/CS /RGB`, `/F /Fl`.
//!
//! Both were dropped by `if let Ok(..)` at the call sites, so the failure mode
//! was not an error - it was an image that was simply not there.

use pdf_oxide::extractors::{ImageData, PixelFormat};
use pdf_oxide::PdfDocument;

/// One-page PDF whose only content is a 2x2 inline image with the given
/// dictionary text and raw sample bytes.
fn pdf_with_inline_image(dict: &str, samples: &[u8]) -> Vec<u8> {
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(b"q 100 0 0 100 10 10 cm\n");
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

/// The four pixels of the 2x2 test image: red, green, blue, white.
const RGB_2X2: [u8; 12] = [
    0xFF, 0x00, 0x00, // red
    0x00, 0xFF, 0x00, // green
    0x00, 0x00, 0xFF, // blue
    0xFF, 0xFF, 0xFF, // white
];

#[test]
fn inline_image_with_abbreviated_colour_space_decodes_to_exact_pixels() {
    // `/CS /RGB` is the ABBREVIATION (Table 93). Before the fix the decoder saw a
    // colour space literally named "RGB", failed, and the image was dropped.
    let pdf = pdf_with_inline_image("/W 2 /H 2 /CS /RGB /BPC 8", &RGB_2X2);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");

    let images = doc.extract_images(0).expect("extract");
    assert_eq!(images.len(), 1, "the inline image must be extracted, not silently dropped");

    let img = &images[0];
    assert_eq!((img.width(), img.height()), (2, 2));
    match img.data() {
        ImageData::Raw { pixels, format } => {
            assert_eq!(*format, PixelFormat::RGB, "/CS /RGB must expand to DeviceRGB");
            assert_eq!(
                pixels.as_slice(),
                &RGB_2X2,
                "decoded samples must be the exact bytes the stream carried"
            );
        },
        other => panic!("expected raw RGB pixels, got {other:?}"),
    }
}

#[test]
fn inline_image_survives_a_flate_filter_abbreviation() {
    // `/F /Fl` is the FILTER abbreviation. Same class of bug: the decoder saw a
    // filter named "Fl" rather than FlateDecode.
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(&RGB_2X2).expect("deflate");
    let compressed = enc.finish().expect("deflate");

    let pdf = pdf_with_inline_image("/W 2 /H 2 /CS /RGB /BPC 8 /F /Fl", &compressed);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");

    let images = doc.extract_images(0).expect("extract");
    assert_eq!(images.len(), 1, "a Flate-compressed inline image must decode");
    match images[0].data() {
        ImageData::Raw { pixels, .. } => assert_eq!(
            pixels.as_slice(),
            &RGB_2X2,
            "FlateDecode must round-trip to the original samples"
        ),
        other => panic!("expected raw pixels, got {other:?}"),
    }
}

#[test]
fn inline_image_with_unabbreviated_names_still_works() {
    // A producer may spell the names out in full; Table 93 permits both. The
    // expansion must not corrupt an already-full name.
    let pdf = pdf_with_inline_image("/Width 2 /Height 2 /ColorSpace /DeviceRGB /BPC 8", &RGB_2X2);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");

    let images = doc.extract_images(0).expect("extract");
    assert_eq!(images.len(), 1);
    match images[0].data() {
        ImageData::Raw { pixels, .. } => assert_eq!(pixels.as_slice(), &RGB_2X2),
        other => panic!("expected raw pixels, got {other:?}"),
    }
}
