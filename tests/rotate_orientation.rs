//! `/Rotate` must ROTATE the page, not mirror it.
//!
//! The 270 case mapped PDF `(x,y)` to `screen_y = x*s`, which put the page's
//! TOP-LEFT corner at the top-left of the raster; under a 270-degree turn it
//! belongs at the BOTTOM-left. That is not a wrong angle but a REFLECTION - the
//! old matrix had a POSITIVE determinant while 0/90/180 all have a negative one
//! (they carry the PDF y-up -> raster y-down flip) - so text rendered mirrored.
//!
//! Each rotation is pinned by WHERE A CORNER MARKER LANDS. That is deliberate: a
//! mirrored page has exactly the RIGHT dimensions, so a width/height check cannot
//! catch this class of bug - only the corner's quadrant can.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
use pdf_oxide::PdfDocument;

/// One-page 400x600 portrait PDF with a black square in the PDF's TOP-LEFT
/// corner (PDF space is y-up, so that is x in 0..40, y in 560..600).
fn pdf_with_top_left_marker(rotate: i32) -> Vec<u8> {
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
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] /Rotate {rotate} \
             /Contents 4 0 R >>"
        ),
    );
    // Black 40x40 square at the PDF's top-left.
    let content = b"0 0 0 rg 0 560 40 40 re f\n";
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

/// Which quadrant holds the marker's centroid: (left?, top?).
fn marker_quadrant(rotate: i32) -> (bool, bool) {
    let doc = PdfDocument::from_bytes(pdf_with_top_left_marker(rotate)).expect("parse");
    // RAW pixels, not the default PNG-encoded bytes.
    let mut opts = RenderOptions::default();
    opts.format = ImageFormat::RawRgba8;
    let img = render_page(&doc, 0, &opts).expect("render");
    let (w, h) = (img.width as usize, img.height as usize);
    assert_eq!(img.data.len(), w * h * 4, "expected raw RGBA8");

    let (mut sx, mut sy, mut n) = (0usize, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let (r, g, b) = (img.data[i], img.data[i + 1], img.data[i + 2]);
            if r < 80 && g < 80 && b < 80 {
                sx += x;
                sy += y;
                n += 1;
            }
        }
    }
    assert!(n > 100, "rotate {rotate}: marker not rendered ({n} dark px)");
    (sx / n < w / 2, sy / n < h / 2)
}

#[test]
fn rotate_places_the_corner_marker_by_turning_not_mirroring() {
    // The marker starts at the PDF's TOP-LEFT. /Rotate turns the page CLOCKWISE
    // (ISO 32000-1 s7.7.3.3), so the corner walks: TL -> TR -> BR -> BL.
    assert_eq!(marker_quadrant(0), (true, true), "0: top-left");
    assert_eq!(marker_quadrant(90), (false, true), "90: top-RIGHT");
    assert_eq!(marker_quadrant(180), (false, false), "180: bottom-RIGHT");
    // The regression: 270 used to land TOP-left (a mirror of the truth).
    assert_eq!(marker_quadrant(270), (true, false), "270: BOTTOM-left");
}

#[test]
fn rotate_270_is_not_a_mirror_of_rotate_90() {
    // A mirrored 270 and a correct 90 share the same x-column; only the y differs.
    // Pinning both guards the specific reflection that was shipping.
    let (l90, t90) = marker_quadrant(90);
    let (l270, t270) = marker_quadrant(270);
    assert_ne!((l90, t90), (l270, t270), "90 and 270 must not land in the same quadrant");
    assert!(t90 && !t270, "90 is top, 270 is bottom");
}
