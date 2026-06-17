//! Integration coverage for #750: extracting an image whose
//! `/BitsPerComponent` is 16 must not panic.
//!
//! PDF stores 16-bit colour samples as big-endian pairs (ISO 32000-1
//! §8.9.5.2); the raw decode path collapses each sample to its high byte so
//! the pixel buffer is the length the 8-bit image pipeline expects. Before the
//! fix the doubled-length buffer reached the PNG encoder and tripped its
//! internal `assert_eq!` ("Invalid buffer length"), a panic that crossed the
//! Python/WASM FFI boundary uncatchably. This fixture is a hand-built minimal
//! PDF (no third-party files) carrying one 16-bit DeviceRGB image.

use pdf_oxide::PdfDocument;

/// Minimal one-page PDF with a single `width`×`height` DeviceRGB image at
/// `/BitsPerComponent 16` (two big-endian bytes per colour sample).
fn pdf_with_16bit_image(width: u32, height: u32) -> Vec<u8> {
    // 16-bit RGB gradient: 3 channels × 2 bytes per pixel.
    let mut img = Vec::with_capacity((width * height * 6) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 257) & 0xFFFF) as u16;
            let g = ((y * 690) & 0xFFFF) as u16;
            let b = 0x8000u16;
            for v in [r, g, b] {
                img.push((v >> 8) as u8);
                img.push((v & 0xFF) as u8);
            }
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 6];
    buf.extend_from_slice(b"%PDF-1.7\n");
    let mut obj = |buf: &mut Vec<u8>, id: usize, head: String, stream: Option<&[u8]>| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{head}").as_bytes());
        if let Some(s) = stream {
            buf.extend_from_slice(b"\nstream\n");
            buf.extend_from_slice(s);
            buf.extend_from_slice(b"\nendstream");
        }
        buf.extend_from_slice(b"\nendobj\n");
    };
    obj(&mut buf, 1, "<< /Type /Catalog /Pages 2 0 R >>".into(), None);
    obj(&mut buf, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into(), None);
    obj(
        &mut buf,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] \
         /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
            .into(),
        None,
    );
    obj(
        &mut buf,
        4,
        format!(
            "<< /Type /XObject /Subtype /Image /Width {width} /Height {height} \
             /ColorSpace /DeviceRGB /BitsPerComponent 16 /Length {} >>",
            img.len()
        ),
        Some(&img),
    );
    let content = format!("q {width} 0 0 {height} 10 10 cm /Im0 Do Q");
    obj(
        &mut buf,
        5,
        format!("<< /Length {} >>", content.len()),
        Some(content.as_bytes()),
    );
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn sixteen_bit_image_extracts_to_png_without_panicking() {
    // 330×95 mirrors the dimensions in the original report.
    let doc = PdfDocument::from_bytes(pdf_with_16bit_image(330, 95)).expect("fixture parses");
    let images = doc.extract_images(0).expect("extract_images");
    assert_eq!(images.len(), 1, "the 16-bit image must be found");
    let img = &images[0];
    assert_eq!((img.width(), img.height()), (330, 95));
    // The crux of #750: encoding must succeed (recoverable Result), not panic.
    let png = img
        .to_png_bytes()
        .expect("16-bit image must encode to PNG, not panic");
    assert!(png.starts_with(&[0x89, b'P', b'N', b'G']), "valid PNG signature");
}
