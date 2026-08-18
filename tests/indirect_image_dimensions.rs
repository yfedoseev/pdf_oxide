//! An image XObject's `/Width` and `/Height` entries may be indirect
//! references — ISO 32000-1:2008 §7.3.10 permits any dictionary/stream
//! entry to be an indirect reference unless the spec explicitly forbids it
//! for that key, and neither `/Width` nor `/Height` is on that exclusion
//! list. `extract_image_from_xobject` read both via a raw `.as_integer()`
//! call with no reference resolution, so an indirect `/Width` (a pattern
//! some PDF producers use, e.g. to share a single "standard photo width"
//! object across many images) made extraction fail with "Image missing
//! /Width" even though the value was present, just one hop away.
//!
//! The fixture is a hand-built minimal PDF (no third-party files): a single
//! 10×8 8-bit DeviceGray image whose `/Width` is `6 0 R`, an indirect
//! reference to a bare integer object, while `/Height` stays inline. (10x8,
//! not smaller, to clear `extract_images`'s unrelated default 8x8
//! decorative-artifact size filter.)

use pdf_oxide::PdfDocument;

fn pdf_with_indirect_width_image() -> Vec<u8> {
    // `extract_images`'s default `ImageExtractFilter` skips anything under
    // 8x8 (decorative-artifact heuristic, unrelated to this fix) — stay at
    // or above that floor so the fixture isn't filtered out for a reason
    // that has nothing to do with indirect /Width resolution.
    let width = 10u32;
    let height = 8u32;
    let img: Vec<u8> = (0..width * height).map(|i| (i * 3) as u8).collect();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 7];
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
            .into(),
        None,
    );
    obj(
        &mut buf,
        4,
        format!(
            "<< /Type /XObject /Subtype /Image /Width 6 0 R /Height {height} \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Length {} >>",
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
    // The indirect /Width value: a bare integer object.
    obj(&mut buf, 6, format!("{width}"), None);

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn indirect_width_reference_resolves_to_correct_dimensions() {
    let doc = PdfDocument::from_bytes(pdf_with_indirect_width_image()).expect("fixture parses");
    let images = doc.extract_images(0).expect("extract_images");
    assert_eq!(images.len(), 1, "the image must be found despite indirect /Width");
    assert_eq!((images[0].width(), images[0].height()), (10, 8));
}
