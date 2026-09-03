//! A library diagnostic is reported out-of-band, never written into content.
//!
//! When an image is too large to inline, the converter used to write an HTML
//! comment into the markdown explaining why:
//!
//! ```text
//! <!-- ![Image 1 from page 1] suppressed: ~412 KB decoded image … -->
//! ```
//!
//! The converted document is what the page draws. An explanation of why this
//! library declined to inline something is a statement *about the library*, and
//! a consumer indexing or diffing the markdown has no way to tell it apart from
//! text the author wrote. It now raises an `ImageSuppressed` warning instead,
//! which the caller can surface, localise, or ignore.
//!
//! The companion assertion is that the diagnostic is not merely deleted — the
//! information still has to reach the caller, just through the right channel.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;

/// A one-page PDF carrying a single image far above the 200 KB inline cap.
///
/// The samples are pseudo-random so Flate cannot compress them below the cap;
/// a flat colour would deflate to a few hundred bytes and never trip it.
fn pdf_with_an_oversized_image() -> Vec<u8> {
    const W: usize = 420;
    const H: usize = 420;
    let mut raw = Vec::with_capacity(W * H * 3);
    let mut seed: u32 = 0x1234_5678;
    for _ in 0..(W * H * 3) {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        raw.push((seed >> 16) as u8);
    }

    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    macro_rules! obj {
        ($b:expr) => {{
            offsets.push(pdf.len());
            pdf.extend_from_slice($b);
        }};
    }
    pdf.extend_from_slice(b"%PDF-1.7\n");
    obj!(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    obj!(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    obj!(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
           /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>\nendobj\n");

    let content = b"q 200 0 0 200 0 0 cm /Im0 Do Q";
    offsets.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {W} /Height {H} \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>\nstream\n",
            raw.len()
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(&raw);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_offset = pdf.len();
    let n = offsets.len() + 1;
    let mut xref = format!("xref\n0 {n}\n0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.extend_from_slice(xref.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
            .as_bytes(),
    );
    pdf
}

fn markdown_with_images(doc: &PdfDocument) -> String {
    let opts = ConversionOptions {
        include_images: true,
        embed_images: true,
        ..Default::default()
    };
    doc.to_markdown(0, &opts).expect("markdown")
}

#[test]
fn an_oversized_image_is_reported_not_narrated() {
    let doc = PdfDocument::from_bytes(pdf_with_an_oversized_image()).expect("open");
    let md = markdown_with_images(&doc);

    assert!(
        !md.contains("suppressed"),
        "no diagnostic prose may appear in the markdown; got:\n{md}"
    );
    assert!(
        !md.contains("<!--"),
        "no HTML comment may be injected into the markdown; got:\n{md}"
    );

    // The information must still reach the caller — deleting it would trade one
    // defect for another.
    let warnings = doc.structured_warnings();
    assert!(
        warnings
            .iter()
            .any(|w| w.category == WarningCategory::ImageSuppressed && w.page == Some(0)),
        "the suppression must be reported out-of-band against page 0; got {:?}",
        warnings.iter().map(|w| (w.category, w.page)).collect::<Vec<_>>()
    );
}

#[test]
fn a_page_whose_only_image_is_suppressed_gets_no_stray_separator() {
    let doc = PdfDocument::from_bytes(pdf_with_an_oversized_image()).expect("open");
    let md = markdown_with_images(&doc);

    // The `---` rule introduces the image block. With the comment gone it would
    // otherwise be emitted with nothing following it.
    assert!(
        !md.contains("---"),
        "the image-block separator must not be emitted when no image follows it; got:\n{md}"
    );
}
