//! An image using `JPXDecode` must render even when the dictionary omits
//! `/ColorSpace`.
//!
//! ISO 32000-1:2008 Table 89 makes `/ColorSpace` "Required for images, except
//! those that use the JPXDecode filter", and says plainly: "If ColorSpace is
//! absent, the colour space specifications in the JPEG2000 data shall be
//! used." The extractor required it unconditionally and returned
//! `Image missing /ColorSpace`, so a page whose only content was such an image
//! rendered completely blank — where MuPDF, pdfium, poppler and Ghostscript
//! all paint it and agree on its tone to within 0.81 of a grey level.

#![cfg(all(feature = "rendering", feature = "jpeg2000"))]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

/// An 8x8 RGB JPEG 2000, red/blue checkerboard, lossless. Produced with
/// Pillow: `Image.new("RGB", (8, 8))`, alternating `(200, 40, 40)` and
/// `(20, 20, 220)` by `(x + y) % 2`, saved as `format="JPEG2000",
/// irreversible=False`. Embedded rather than committed as a file because a
/// codestream cannot be written by hand and the repository does not carry
/// binary image fixtures.
const JP2: &[u8] = &[
    0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A, 0x00, 0x00, 0x00, 0x14,
    0x66, 0x74, 0x79, 0x70, 0x6A, 0x70, 0x32, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6A, 0x70, 0x32, 0x20,
    0x00, 0x00, 0x00, 0x2D, 0x6A, 0x70, 0x32, 0x68, 0x00, 0x00, 0x00, 0x16, 0x69, 0x68, 0x64, 0x72,
    0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x07, 0x07, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x0F, 0x63, 0x6F, 0x6C, 0x72, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0xCE, 0x6A, 0x70, 0x32, 0x63, 0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x2F, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x07,
    0x01, 0x01, 0x07, 0x01, 0x01, 0x07, 0x01, 0x01, 0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x03, 0x04, 0x04, 0x00, 0x01, 0xFF, 0x5C, 0x00, 0x0D, 0x40, 0x40, 0x48, 0x48, 0x50, 0x48,
    0x48, 0x50, 0x48, 0x48, 0x50, 0xFF, 0x64, 0x00, 0x25, 0x00, 0x01, 0x43, 0x72, 0x65, 0x61, 0x74,
    0x65, 0x64, 0x20, 0x62, 0x79, 0x20, 0x4F, 0x70, 0x65, 0x6E, 0x4A, 0x50, 0x45, 0x47, 0x20, 0x76,
    0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x32, 0x2E, 0x35, 0x2E, 0x34, 0xFF, 0x90, 0x00, 0x0A,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x4D, 0x00, 0x01, 0xFF, 0x93, 0xC3, 0xE7, 0x02, 0x06, 0xCF, 0xB4,
    0x08, 0x08, 0x9F, 0xC0, 0x74, 0x10, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x93, 0xF3, 0x0D,
    0x13, 0xFF, 0x47, 0x93, 0x46, 0x38, 0x06, 0x04, 0x14, 0xF6, 0x94, 0x90, 0x7F, 0x90, 0x7D, 0x41,
    0x20, 0x13, 0xFF, 0x47, 0x90, 0xB5, 0xB8, 0x06, 0x12, 0x0F, 0x93, 0xF3, 0x0C, 0x11, 0x4F, 0x87,
    0x26, 0x8C, 0x70, 0x0C, 0x08, 0x29, 0xED, 0x29, 0x20, 0xFF, 0xD9,
];

/// A page filled by one JPX image. `color_space` is spliced into the image
/// dictionary verbatim, so the caller can pass `""` to omit it entirely.
fn jpx_page(color_space: &str) -> Vec<u8> {
    let content = b"q 100 0 0 100 0 0 cm /Im Do Q";
    let img_dict = format!(
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 \
         /Filter /JPXDecode {color_space} /Length {} >>",
        JP2.len()
    );

    let mut pdf: Vec<u8> = Vec::new();
    let mut off = [0usize; 6];
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.5\n");
    off[1] = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    push!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Contents 4 0 R /Resources << /XObject << /Im 5 0 R >> >> >>\nendobj\n"
    );
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(content);
    push!("\nendstream\nendobj\n");
    off[5] = pdf.len();
    push!(format!("5 0 obj\n{img_dict}\nstream\n"));
    pdf.extend_from_slice(JP2);
    push!("\nendstream\nendobj\n");

    let xref = pdf.len();
    push!("xref\n0 6\n0000000000 65535 f \r\n");
    for id in 1..=5 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

/// Fraction of the page carrying ink.
fn coverage(pdf: Vec<u8>) -> f64 {
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, &RenderOptions::default()).expect("page renders");
    let px = image::load_from_memory(&img.data)
        .expect("PNG decodes")
        .to_rgba8();
    let (w, h) = px.dimensions();
    let inked = px
        .pixels()
        .filter(|p| p[3] > 0 && (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3 < 250)
        .count();
    inked as f64 / (w as f64 * h as f64)
}

#[test]
fn a_jpx_image_without_a_colour_space_still_paints() {
    let cov = coverage(jpx_page(""));
    assert!(
        cov > 0.95,
        "Table 89 makes /ColorSpace optional for JPXDecode; the image must \
         still paint. Covered {cov:.5} of the page"
    );
}

/// Control: the same image *with* a `/ColorSpace` must be unaffected, so the
/// change cannot be read as "ignore the entry when it is present".
#[test]
fn a_jpx_image_with_a_colour_space_is_unchanged() {
    let with_cs = coverage(jpx_page("/ColorSpace /DeviceRGB"));
    let without = coverage(jpx_page(""));
    assert!(with_cs > 0.95, "the control must paint too; covered {with_cs:.5}");
    assert!(
        (with_cs - without).abs() < 0.01,
        "declaring the colour space must not change the result: \
         {with_cs:.5} with, {without:.5} without"
    );
}
