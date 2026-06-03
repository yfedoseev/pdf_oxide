//! Regression test for Separation / DeviceN tint-transform rendering.
//!
//! A Separation (or DeviceN) fill selected with `scn` must be resolved through
//! its tint transform, not the naive `grey = 1 - tint` fallback. The bug
//! rendered a full-tint (`1 scn`) spot colour — common on InDesign-exported
//! PDFs for tinted callout boxes and headings — as solid black.
//!
//! The fixtures are the committed reproducer PDFs under
//! `examples/separation-blackout/`. Both fill the whole page with a full-tint
//! Separation colour whose tint transform maps tint 1.0 -> CMYK(0.1, 0, 0.15, 0),
//! i.e. a light green (RGB ≈ 230, 255, 216). Resolved correctly the page is
//! near-white/green; with the old fallback it collapsed to black. The two
//! fixtures exercise the two supported tint-transform function types:
//! FunctionType 2 (exponential) and FunctionType 0 (sampled).

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page_fit, ImageFormat, RenderOptions};

/// Render the centre pixel of a single-page reproducer PDF.
fn centre_pixel(pdf: &[u8]) -> (u8, u8, u8) {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open reproducer pdf");
    let options = RenderOptions::with_dpi(72).as_raw();
    let image = render_page_fit(&doc, 0, 200, 200, &options).expect("render");
    assert_eq!(image.format, ImageFormat::RawRgba8);
    assert_eq!(image.data.len() as u32, image.width * image.height * 4);

    // The whole page is the Separation fill; sample its centre.
    let (cx, cy) = (image.width / 2, image.height / 2);
    let i = ((cy * image.width + cx) * 4) as usize;
    (image.data[i], image.data[i + 1], image.data[i + 2])
}

/// Tint 1.0 → CMYK(0.1,0,0.15,0) → RGB ≈ (230, 255, 216): a light green. The
/// pre-fix fallback produced (0,0,0). Assert it's light (the transform was
/// evaluated) and green-dominant (lowest CMYK component).
fn assert_light_green(name: &str, (r, g, b): (u8, u8, u8)) {
    assert!(
        r > 150 && g > 150 && b > 150,
        "{name}: Separation fill rendered dark (r={r}, g={g}, b={b}) — tint transform not applied?",
    );
    assert!(
        g >= r && g >= b,
        "{name}: expected green-dominant tint, got ({r},{g},{b})",
    );
}

#[test]
fn separation_scn_fill_type2_is_not_black() {
    let pdf = include_bytes!("../examples/separation-blackout/separation-type2.pdf");
    assert_light_green("FunctionType 2", centre_pixel(pdf));
}

#[test]
fn separation_scn_fill_type0_is_not_black() {
    let pdf = include_bytes!("../examples/separation-blackout/separation-type0.pdf");
    assert_light_green("FunctionType 0", centre_pixel(pdf));
}