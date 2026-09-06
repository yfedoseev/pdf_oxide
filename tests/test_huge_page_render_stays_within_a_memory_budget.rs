//! A page's raster is bounded by a budget, not by the file.
//!
//! Nothing in a PDF bounds `page_box x scale`. `docs/spec/pdf.md` leaves the
//! MediaBox extent entirely in the file's hands — Table 30 (pdf.md:5760)
//! defines it as the rectangle that "shall define the boundaries of the
//! physical medium on which the page shall be displayed or printed", with no
//! ceiling — so a conforming file may declare a page whose raster is hundreds
//! of megapixels.
//!
//! Annex C.1 (pdf.md:41166) names the constraint this runs into: "The amount
//! of memory available to a conforming reader limits the number of
//! memory-consuming objects that can be held simultaneously", and the note at
//! C.2 (pdf.md:41183) adds that "Memory limits are often exceeded before
//! architectural limits ... are reached". The spec therefore treats staying
//! inside available memory as the reader's own responsibility; it gives no
//! page size a reader may assume it will never see.
//!
//! One real 6 MB document declares a 12608 x 16806 pt page carrying a 211.9
//! megapixel image. At 72 dpi that is an 847 MB pixmap before any working
//! copy; rendering it reached 11.6 GB and the process was killed. An OOM kill
//! is a signal, so the caller never gets a `Result` and the host process dies
//! outright — a thumbnailer or web service handling untrusted input has no
//! defence, and the WASM and mobile targets have far less headroom than the
//! machine that died.
//!
//! So the renderer holds the output to `RenderOptions::max_output_pixels`,
//! reducing the scale to fit rather than failing: the caller wants an image of
//! the page, and a smaller one beats a dead process.

#![cfg(feature = "rendering")]

use pdf_oxide::rendering::{render_page, RenderOptions, DEFAULT_MAX_OUTPUT_PIXELS};
use pdf_oxide::PdfDocument;

/// A one-page document whose MediaBox is `w` x `h` points, with a black
/// square filling the middle third of the page.
///
/// The mark is placed proportionally so that a *scaled* render still contains
/// it while a render that merely allocated a smaller buffer — cropping the
/// page to its top-left corner — would not.
fn page_sized(w: u32, h: u32) -> Vec<u8> {
    let content = format!("0 0 0 rg {} {} {} {} re f\n", w / 3, h / 3, w / 3, h / 3).into_bytes();
    let mut pdf = Vec::new();
    let mut off = [0usize; 6];
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.7\n");
    off[1] = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    push!(format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
         /Contents 4 0 R /Resources << >> >>\nendobj\n"
    ));
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(&content);
    push!("endstream\nendobj\n");
    let xref = pdf.len();
    push!("xref\n0 5\n0000000000 65535 f \r\n".to_string());
    for id in 1..=4 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

fn rendered(pdf: Vec<u8>, opts: &RenderOptions) -> image::DynamicImage {
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");
    let img = render_page(&doc, 0, opts).expect("page renders");
    image::load_from_memory(&img.data).expect("PNG decodes")
}

fn rendered_dimensions(pdf: Vec<u8>, opts: &RenderOptions) -> (u32, u32) {
    let px = rendered(pdf, opts);
    (px.width(), px.height())
}

/// Fraction of pixels that are substantially darker than white.
fn ink(px: &image::DynamicImage) -> f64 {
    let rgba = px.to_rgba8();
    let dark = rgba
        .pixels()
        .filter(|p| p[3] > 0 && (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3 < 128)
        .count();
    dark as f64 / rgba.pixels().len() as f64
}

#[test]
fn test_enormous_page_is_scaled_down_to_fit_the_budget() {
    let mut opts = RenderOptions::default();
    opts.max_output_pixels = 1_000_000;

    // 5000 x 5000 pt is far over this budget at any resolution.
    let (w, h) = rendered_dimensions(page_sized(5000, 5000), &opts);

    let pixels = u64::from(w) * u64::from(h);
    assert!(
        pixels <= opts.max_output_pixels,
        "raster {w}x{h} = {pixels} px exceeds the {} px budget; without the \
         clamp the page allocates at its full natural size, and a genuinely \
         large one takes the process down by OOM",
        opts.max_output_pixels
    );

    // It should still fill the budget rather than collapsing to a stub, and
    // keep the page's square aspect ratio.
    assert!(pixels > opts.max_output_pixels / 2, "raster {w}x{h} is far below the budget");
    assert!(w.abs_diff(h) <= 1, "square page rendered as {w}x{h}");
}

#[test]
fn test_over_budget_page_is_scaled_down_rather_than_cropped() {
    // The whole page must still be in the picture. Shrinking only the output
    // buffer, while leaving the page transform at the requested scale, would
    // produce a raster of the right size holding the page's top-left corner —
    // the right dimensions and the wrong image.
    //
    // The fixture's mark covers the middle ninth of the page, so a correctly
    // scaled render inks about 1/9 of the raster wherever the budget lands,
    // and a cropped one inks either none of it or all of it.
    let mut opts = RenderOptions::default();
    opts.max_output_pixels = 1_000_000;

    let coverage = ink(&rendered(page_sized(5000, 5000), &opts));

    assert!(
        (0.06..0.17).contains(&coverage),
        "expected the centre mark to cover about 1/9 of the raster, got {coverage:.4} — \
         the page was cropped to a corner rather than scaled to fit"
    );
}

#[test]
fn test_ordinary_page_is_left_at_its_natural_size() {
    // The counter-case: US Letter is a couple of megapixels at the default
    // resolution, well inside the default budget, so nothing may be scaled
    // away. Compare against an effectively unbudgeted render rather
    // than hard-coding a size, so this keeps its meaning if the default dpi
    // changes.
    let natural = rendered_dimensions(page_sized(612, 792), &{
        let mut o = RenderOptions::default();
        o.max_output_pixels = u64::MAX;
        o
    });
    let budgeted = rendered_dimensions(page_sized(612, 792), &RenderOptions::default());

    assert_eq!(
        budgeted, natural,
        "a page well inside the budget must render at its natural size"
    );
    assert!(
        u64::from(natural.0) * u64::from(natural.1) < DEFAULT_MAX_OUTPUT_PIXELS,
        "test premise: US Letter {natural:?} should be inside the default budget"
    );
}
