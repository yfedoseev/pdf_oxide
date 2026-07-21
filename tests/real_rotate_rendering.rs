//! A real-valued `/Rotate` (e.g. `/Rotate 90.0`, which some producers emit -
//! the lexer parses any number with a decimal point as `Object::Real`) must
//! rotate the rendered page exactly like its integer twin. `get_page_info`'s
//! `/Rotate` extraction accepted ONLY `Object::Integer`, so a real /Rotate
//! silently became 0 and the three renderer sites that read
//! `page_info.rotation` rendered the page UNROTATED - while the sibling
//! `get_page_rotation` already accepted both Integer and Real.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// Minimal one-page PDF: portrait 612x792 MediaBox with the given /Rotate
/// literal (caller controls whether it renders as an integer or a real).
fn rotated_pdf(rotate_literal: &str) -> Vec<u8> {
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
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate {rotate_literal} \
             /Contents 4 0 R >>"
        ),
    );
    let content = b"0 0 0 rg 100 100 50 50 re f\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
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

fn rendered_dims(rotate_literal: &str) -> (u32, u32) {
    let doc = PdfDocument::from_bytes(rotated_pdf(rotate_literal)).expect("parse");
    let opts = RenderOptions::default();
    let img = render_page(&doc, 0, &opts).expect("render");
    (img.width, img.height)
}

#[test]
fn real_rotate_matches_its_integer_twin() {
    // /Rotate 90.0 (Object::Real) must rotate exactly like /Rotate 90
    // (Object::Integer): the portrait page must render LANDSCAPE.
    let (w_int, h_int) = rendered_dims("90");
    let (w_real, h_real) = rendered_dims("90.0");
    assert!(w_int > h_int, "control failed: /Rotate 90 must render landscape");
    assert_eq!(
        (w_real, h_real),
        (w_int, h_int),
        "/Rotate 90.0 (real) must render exactly like /Rotate 90 (integer)"
    );
}
