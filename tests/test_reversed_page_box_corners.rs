//! A page box written on the other diagonal must describe the same page.
//!
//! ISO 32000-1:2008 §7.9.5 (`docs/spec/pdf.md:6443`):
//!
//! > Although rectangles are conventionally specified by their lower-left and
//! > upper-right corners, it is acceptable to specify any two diagonally
//! > opposite corners. Applications that process PDF should be prepared to
//! > normalize such rectangles in situations where specific corners are
//! > required.
//!
//! `Rect::from_points` built the struct literally while `Rect::new` — the
//! same type's other constructor — normalised. So `/MediaBox [612 792 0 0]`
//! produced a negative width and height, the page dimensions came out
//! negative, and pixmap allocation failed: the page did not render at all.

use pdf_oxide::geometry::Rect;
use pdf_oxide::PdfDocument;

/// One-page PDF with the given `/MediaBox` array text.
fn pdf_with_media_box(media_box: &str) -> Vec<u8> {
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
        &format!("<< /Type /Page /Parent 2 0 R /MediaBox {media_box} /Contents 4 0 R >>"),
    );
    let content = b"0 0 1 rg 10 10 100 100 re f\n";
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

/// The two constructors of one type must agree about what a rectangle is.
#[test]
fn rect_constructors_agree_on_reversed_corners() {
    let upright = Rect::from_points(0.0, 0.0, 612.0, 792.0);
    assert_eq!(upright.width, 612.0);
    assert_eq!(upright.height, 792.0);

    // The other diagonal describes the same rectangle.
    let reversed = Rect::from_points(612.0, 792.0, 0.0, 0.0);
    assert_eq!(reversed, upright, "reversed corners must normalise");

    // Mixed diagonals too.
    assert_eq!(Rect::from_points(0.0, 792.0, 612.0, 0.0), upright);
    assert_eq!(Rect::from_points(612.0, 0.0, 0.0, 792.0), upright);

    // And it must still agree with the width/height constructor.
    assert_eq!(Rect::new(0.0, 0.0, 612.0, 792.0), upright);
}

/// The page-box reader normalises, so every consumer that needs specific
/// corners gets them.
#[test]
fn reversed_media_box_reports_positive_extents() {
    let doc = PdfDocument::from_bytes(pdf_with_media_box("[612 792 0 0]")).expect("parse");
    let (llx, lly, urx, ury) = doc.get_page_media_box(0).expect("media box");
    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 612.0, 792.0),
        "the reader must return lower-left and upper-right, whichever diagonal the file used"
    );
}

/// An upright box is unchanged — the normalisation must be a no-op on the
/// overwhelmingly common case.
#[test]
fn upright_media_box_is_untouched() {
    let doc = PdfDocument::from_bytes(pdf_with_media_box("[0 0 612 792]")).expect("parse");
    assert_eq!(doc.get_page_media_box(0).expect("media box"), (0.0, 0.0, 612.0, 792.0));
}

/// A box that is neither diagonal-ordered nor origin-anchored.
#[test]
fn offset_reversed_media_box_normalises() {
    let doc = PdfDocument::from_bytes(pdf_with_media_box("[300 400 100 200]")).expect("parse");
    assert_eq!(doc.get_page_media_box(0).expect("media box"), (100.0, 200.0, 300.0, 400.0));
}

#[cfg(feature = "rendering")]
mod rendering {
    use super::pdf_with_media_box;
    use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
    use pdf_oxide::PdfDocument;

    /// The reported symptom: the page did not render at all, because the
    /// negative extents produced a zero-sized pixmap.
    #[test]
    fn reversed_media_box_page_still_renders() {
        let doc = PdfDocument::from_bytes(pdf_with_media_box("[612 792 0 0]")).expect("parse");
        let mut opts = RenderOptions::default();
        opts.dpi = 72;
        opts.format = ImageFormat::RawRgba8;
        let img = render_page(&doc, 0, &opts).expect("render must succeed, not fail allocation");
        assert_eq!(
            (img.width, img.height),
            (612, 792),
            "the page renders at the size its box describes"
        );
    }

    /// And it renders the same page an upright box would.
    #[test]
    fn reversed_and_upright_media_boxes_render_identically() {
        let render = |mb: &str| {
            let doc = PdfDocument::from_bytes(pdf_with_media_box(mb)).expect("parse");
            let mut opts = RenderOptions::default();
            opts.dpi = 72;
            opts.format = ImageFormat::RawRgba8;
            render_page(&doc, 0, &opts).expect("render").data
        };
        assert_eq!(
            render("[612 792 0 0]"),
            render("[0 0 612 792]"),
            "the same rectangle written either way must paint the same page"
        );
    }
}
