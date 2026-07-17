//! A legal negative `/Rotate` (e.g. -90, equivalent to 270 per ISO 32000-1
//! s7.7.3.3 Table 30) must rotate the rendered page exactly like its positive
//! twin. The renderer's `page_info.rotation % 360` kept the sign (Rust `%` is a
//! remainder), so `-90` matched neither `90` nor `270` in the axis-swap check
//! and the page rendered UNROTATED - while `get_page_rotation` itself already
//! normalizes with `((raw % 360) + 360) % 360`.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

/// Minimal one-page PDF: portrait 612x792 MediaBox with the given /Rotate.
fn rotated_pdf(rotate: i32) -> Vec<u8> {
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
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate {rotate} \
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

fn rendered_dims(rotate: i32) -> (u32, u32) {
    let doc = PdfDocument::from_bytes(rotated_pdf(rotate)).expect("parse");
    let opts = RenderOptions::default();
    let img = render_page(&doc, 0, &opts).expect("render");
    (img.width, img.height)
}

#[test]
fn negative_rotate_matches_its_positive_twin() {
    // -90 == 270: the portrait page must render LANDSCAPE (axes swapped).
    let (w_pos, h_pos) = rendered_dims(270);
    let (w_neg, h_neg) = rendered_dims(-90);
    assert!(w_pos > h_pos, "control failed: /Rotate 270 must render landscape");
    assert_eq!(
        (w_neg, h_neg),
        (w_pos, h_pos),
        "/Rotate -90 must render exactly like /Rotate 270"
    );
}

#[test]
fn negative_full_turns_stay_portrait() {
    // -360 == 0: no rotation.
    let (w, h) = rendered_dims(-360);
    assert!(h > w, "/Rotate -360 must render portrait (equivalent to 0)");
}
