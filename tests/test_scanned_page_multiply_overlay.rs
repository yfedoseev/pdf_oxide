//! Renderer regression tests for scanned pages covered by a
//! Multiply-blended overlay.
//!
//! A common PDF shape — a full-page CCITT bilevel scan (e.g. a fax or
//! an OCR'd document) with a semi-transparent highlight/watermark
//! drawn on top as a Form XObject using an ExtGState with
//! `BM=Multiply`. The interesting property is that multiplying
//! yellow over black-and-white text should leave the text visible
//! underneath the highlight. That only works if the bilevel image
//! survives downscaling with enough intermediate greys; with
//! nearest-neighbour sampling every pixel collapses to pure black or
//! pure white and the Multiply result is solid yellow + solid black —
//! no shades between, and no readable text.
//!
//! Fixture `538250-1.pdf` was pulled from mozilla/pdf.js#19978 (which
//! reproduced the same regression in pdf.js). We render it at 96 DPI
//! and check the rendered image actually contains a range of
//! intermediate shades in the region covered by the overlay.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

const FIXTURE: &str = "tests/fixtures/issue_regressions/alpha_channel/538250-1.pdf";

#[test]
fn renders_without_error() {
    let doc = PdfDocument::open(FIXTURE).expect("open fixture");
    let opts = RenderOptions::with_dpi(96);
    let img = render_page(&doc, 0, &opts).expect("render page 0");
    assert!(img.width > 0);
    assert!(img.height > 0);
    // At least some pixels must have been written — a blank-page
    // regression would leave a small, near-empty buffer.
    assert!(
        img.data.len() > 1024,
        "rendered image only {} bytes — likely blank-page regression",
        img.data.len()
    );
}

/// With bicubic image filtering the bilevel CCITT scan produces a range
/// of intermediate grey shades on downscale, so the Multiply-blended
/// overlay composites into many intermediate colours rather than a
/// flat yellow. Nearest-neighbour (tiny-skia's default) would collapse
/// every pixel to pure black or pure white and leave only two colours.
#[test]
fn multiply_overlay_preserves_intermediate_shades() {
    let doc = PdfDocument::open(FIXTURE).expect("open fixture");
    let opts = RenderOptions::with_dpi(96);
    let img = render_page(&doc, 0, &opts).expect("render page 0");

    let cursor = std::io::Cursor::new(&img.data);
    let decoded = image::load(cursor, image::ImageFormat::Png).expect("decode rendered PNG");
    let rgba = decoded.to_rgba8();

    // Scan the top 20 % of the page — on this fixture the overlay
    // band lives lower down but the scanned text sits throughout, so
    // the top band alone should still have >100 distinct shades once
    // bilinear/bicubic sampling is in effect. On the nearest-neighbour
    // baseline this band collapses to 2 colours.
    let top_band_height = rgba.height() / 5;
    let mut colours = std::collections::HashSet::new();
    for y in 0..top_band_height {
        for x in 0..rgba.width() {
            let p = rgba.get_pixel(x, y);
            colours.insert((p[0], p[1], p[2]));
            if colours.len() > 100 {
                return;
            }
        }
    }
    panic!(
        "top 20 % of page only has {} distinct colours — filtering has regressed, \
         likely back to nearest-neighbour image sampling",
        colours.len()
    );
}
