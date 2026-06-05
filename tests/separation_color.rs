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
#![cfg(feature = "rendering")]

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
    assert!(g >= r && g >= b, "{name}: expected green-dominant tint, got ({r},{g},{b})",);
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

/// FunctionType 4 (PostScript calculator) tint transform on a single-colorant
/// Separation. `{ dup 0.1 mul 0 3 -1 roll 0.15 mul 0 }` maps tint 1.0 to the
/// same CMYK(0.1,0,0.15,0) light green as the Type 0/2 fixtures.
#[test]
fn separation_scn_fill_type4_is_not_black() {
    let pdf = include_bytes!("../examples/separation-blackout/separation-type4.pdf");
    assert_light_green("FunctionType 4", centre_pixel(pdf));
}

/// Multi-colorant DeviceN (2 colorants) with a 2-in/4-out Type 4 tint transform
/// `{ exch 0.8 mul 0 3 -1 roll 0.6 mul 0 }`: tints [1,1] -> CMYK(0.8,0,0.6,0) ->
/// RGB ≈ (51,255,102), a green. Before the multi-input fix this either rendered
/// the wrong colour (only the first tint fed in) or fell back to grey.
#[test]
fn devicen_scn_fill_type4_multicolorant() {
    let pdf = include_bytes!("../examples/separation-blackout/devicen-type4.pdf");
    let (r, g, b) = centre_pixel(pdf);
    assert!(
        g > 150 && g >= r && g >= b,
        "DeviceN Type 4: expected green-dominant, got ({r},{g},{b})",
    );
    assert!(r < 130, "DeviceN Type 4: expected low red from C=0.8, got r={r} ({r},{g},{b})",);
}

/// Separation whose alternate space is `[/Lab …]`. The tint transform maps
/// tint 1.0 to Lab(53.24, 80.09, 67.2) — sRGB red. The alternate space must be
/// inspected and converted via CIELAB→sRGB, not treated as DeviceRGB (which
/// would read L=53.24 as a >100% red channel) nor as the old L*/100 grey.
#[test]
fn separation_scn_fill_lab_alternate() {
    let pdf = include_bytes!("../examples/separation-blackout/separation-lab.pdf");
    let (r, g, b) = centre_pixel(pdf);
    assert!(
        r > 150 && r > g && r > b,
        "Lab alternate: expected red-dominant from Lab(53,80,67), got ({r},{g},{b})",
    );
}
