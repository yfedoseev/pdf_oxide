//! Regression test: JBIG2-compressed scanner PDFs must render non-blank.
//!
//! Issue #332: scanned documents using JBIG2Decode produced blank pages
//! because the pass-through decoder returned compressed bytes as "raw" pixels.

#[cfg(feature = "rendering")]
mod tests {
    use pdf_oxide::document::PdfDocument;
    use pdf_oxide::rendering::{render_page, RenderOptions};

    const LINN_PDF: &str = "/home/yfedoseev/projects/pdf_oxide_tests/fixtures_ocr/linn.pdf";

    #[test]
    #[ignore = "requires local fixture at LINN_PDF; run with -- --ignored"]
    fn jbig2_scanner_pdf_renders_non_blank() {
        if !std::path::Path::new(LINN_PDF).exists() {
            eprintln!("skipping: fixture not found at {LINN_PDF}");
            return;
        }

        let doc = PdfDocument::open(LINN_PDF).expect("open linn.pdf");

        // as_raw() returns premultiplied RGBA8888 — the only reliable way to
        // check for blank (PNG bytes cannot be scanned for blank detection).
        let opts = RenderOptions::with_dpi(72).as_raw();
        let rendered = render_page(&doc, 0, &opts).expect("render page 0");

        // White background is (255, 255, 255, 255) in premultiplied RGBA.
        let non_bg = rendered
            .data
            .chunks(4)
            .filter(|px| px[0] != 255 || px[1] != 255 || px[2] != 255)
            .count();

        assert!(
            non_bg > 100,
            "Expected non-blank JBIG2 render, got {non_bg} non-white pixels (page is blank)"
        );
    }
}
