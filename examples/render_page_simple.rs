//! Quick page→image render tool used for triaging issue #325.
//!
//! Usage: `cargo run --release --features rendering --example render_page_simple <pdf> [page]`
//!
//! Writes the rendered JPEG to `/tmp/render_out_<page>.jpg`. Prints a one-line
//! timing report so we can measure how the text rasterizer path compares
//! against pdfium-render.

#[cfg(feature = "rendering")]
use pdf_oxide::rendering::{render_page, RenderOptions};
#[cfg(feature = "rendering")]
use pdf_oxide::PdfDocument;

#[cfg(feature = "rendering")]
fn main() -> pdf_oxide::Result<()> {
    let pdf = std::env::args()
        .nth(1)
        .expect("usage: render_page_simple <pdf> [page]");
    let page: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let doc = PdfDocument::open(&pdf)?;
    let opts = RenderOptions::with_dpi(96).as_jpeg(100);
    let start = std::time::Instant::now();
    let img = render_page(&doc, page, &opts)?;
    let elapsed = start.elapsed();
    println!(
        "page {}: {} ms, {}x{}, {} bytes",
        page,
        elapsed.as_millis(),
        img.width,
        img.height,
        img.data.len()
    );
    let out = format!("/tmp/render_out_{}.jpg", page);
    img.save(&out)?;
    println!("wrote {}", out);
    Ok(())
}

#[cfg(not(feature = "rendering"))]
fn main() {
    eprintln!("Rebuild with --features rendering");
    std::process::exit(1);
}
