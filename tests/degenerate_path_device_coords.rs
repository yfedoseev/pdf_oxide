//! A content-stream path whose device-space coordinates exceed f32 pixel
//! precision must be skipped, not handed to the rasterizer. tiny-skia's
//! antialiased run accounting desynchronizes on such geometry and panics
//! (an `unwrap` on a run past the accumulated buffer), which can escalate
//! to a process abort under `catch_unwind`-based isolation.
//!
//! The triangle coordinates mirror the shape of the reproducing document:
//! one vertex inside the visible page, the others out at ~1e10–1e18 user
//! space, so the path straddles the pixmap and exercises the antialiased
//! clip edge.
#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

/// One-page PDF whose content stream is `content`.
fn pdf_with_content(content: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut off = vec![0usize; 6];
    buf.extend_from_slice(b"%PDF-1.7\n");

    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 467 807] /Contents 4 0 R >>",
    );
    obj(
        &mut buf,
        &mut off,
        4,
        &format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for o in off.iter().take(5).skip(1) {
        buf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    buf
}

fn renders_ok(content: &str) {
    let doc = PdfDocument::from_bytes(pdf_with_content(content)).expect("fixture parses");
    let opts = RenderOptions::default();
    let img = render_page(&doc, 0, &opts).expect("renders without panicking");
    assert!(img.data.iter().len() > 0);
}

#[test]
fn fill_with_beyond_precision_cubics_renders_without_panic() {
    // The exact filled path (verbs and coordinates) that a damaged
    // content stream produced in the wild: cubics mixing sub-pixel
    // values with coordinates up to ~4.5e18. At 150 DPI this drives
    // tiny-skia 0.12's antialiased run accounting past its buffer and
    // aborts the process when unguarded.
    renders_ok(FATAL_FILL);
}

#[test]
fn stroke_with_beyond_precision_coords_renders_without_panic() {
    // Same defect class on the stroke path.
    renders_ok("0 0 m 47918050000 4528583000000000000 l 100 100 l S");
}

const FATAL_FILL: &str = "484 173 m 484 89.45282 4831008 45285830000 0.4528583 0 c 480.275 4528583000000000000 0.4791805 0 0 0 c 47918050000 4791916 447 377.48 0.312 446.757 c 480.783 0.446 0.024 4818506 445 799.482 c 482.941 0.445 799.48315 446.052 0.4831686 446.438 c 484 136 446.92593 316 447 414.484 c 484 316 4498669 48310560000 482.248 0.45 c 481.34595 0.187 4808885 4498736.5 44983780 0 c 480.659 0.4498397 l 480.659 0.45 0.093 480.931 0.451 0.84 c 482.941 0.451 0.84 0.483 0.261 0.451 c 484 173 l 0.7729448 0.787 481.213 4498444.5 0.127 4498444 c 483.149 4498444.5 0.636 0.483846 0.053 0 c 483.45993 161.482 0.953 0.446 0.541 0.482 c 481.195 0.446 0.541 0.48 0.7729447 302 c f";

#[test]
fn ordinary_geometry_still_renders() {
    // Sanity: the guard must not swallow normal content on the same page.
    renders_ok("0 0 m 600 780 l 100 100 l h f 10 10 m 500 500 l S");
}
