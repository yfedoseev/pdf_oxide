//! Extracted raster pixels must match the image's decoded samples:
//!
//! * A sub-byte image (1/2/4 bits per component) must be unpacked to one
//!    8-bit sample per pixel — `PixelFormat::Grayscale` promises
//!   `width × height × bytes_per_pixel` bytes, and a packed buffer breaks
//!   every consumer that builds an image from it.
//! * A non-default `/Decode` array must be applied to the samples
//!   (ISO 32000-1 §8.9.5.2) at every bit depth, not only for 1-bit CCITT.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::{ImageData, PixelFormat};

fn build_pdf(image_dict: &str, image_data: &[u8]) -> Vec<u8> {
    let content: &[u8] = b"q 100 0 0 100 0 0 cm /Im0 Do Q";
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 6]; // ids 1..=5
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    bodies[3] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        .to_vec();
    let mut img = format!("{image_dict}\nstream\n").into_bytes();
    img.extend_from_slice(image_data);
    img.extend_from_slice(b"\nendstream");
    bodies[4] = img;
    let mut cs = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    cs.extend_from_slice(content);
    cs.extend_from_slice(b"\nendstream");
    bodies[5] = cs;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 6];
    for id in 1..=5 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

fn extract_first_raw(pdf: Vec<u8>) -> (u32, u32, u8, PixelFormat, Vec<u8>) {
    let doc = PdfDocument::from_bytes(pdf).expect("open pdf");
    let imgs = doc.extract_images(0).expect("extract_images");
    assert!(!imgs.is_empty(), "no images extracted");
    let img = &imgs[0];
    let (fmt, pixels) = match img.data() {
        ImageData::Raw { pixels, format } => (*format, pixels.clone()),
        ImageData::Jpeg(_) => panic!("expected Raw image data"),
    };
    (img.width(), img.height(), img.bits_per_component(), fmt, pixels)
}

/// 8×8 1-bit DeviceGray, every row byte 0x55 (alternating 0/1 bits).
/// Correct extraction: 64 unpacked samples alternating 0, 255.
#[test]
fn one_bit_devicegray_unpacks_to_full_sample_plane() {
    let (w, h, bpc, fmt, pixels) = extract_first_raw(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 1 /Length 8 >>",
        &[0x55u8; 8],
    ));

    assert_eq!((w, h), (8, 8));
    assert_eq!(fmt, PixelFormat::Grayscale);
    assert_eq!(
        pixels.len(),
        (w as usize) * (h as usize) * fmt.bytes_per_pixel(),
        "buffer must hold one byte per sample, got {} bytes for {w}x{h} (bpc={bpc})",
        pixels.len()
    );
    let expected_row = [0u8, 255, 0, 255, 0, 255, 0, 255];
    for (i, row) in pixels.chunks(8).enumerate() {
        assert_eq!(row, expected_row, "row {i} mismatch");
    }
}

/// 8×8 8-bit DeviceGray with samples 0..63 and `/Decode [1 0]`: the samples
/// must come out inverted (255, 254, …, 192), not raw.
#[test]
fn decode_array_applied_to_eight_bit_gray() {
    let data: Vec<u8> = (0..64u8).collect();
    let (w, h, bpc, fmt, pixels) = extract_first_raw(build_pdf(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0] /Length 64 >>",
        &data,
    ));

    assert_eq!((w, h), (8, 8));
    assert_eq!(fmt, PixelFormat::Grayscale);
    assert_eq!(bpc, 8);
    let expected: Vec<u8> = (0..64u16).map(|s| (255 - s) as u8).collect();
    assert_eq!(pixels, expected, "/Decode [1 0] ignored: samples extracted un-inverted");
}
