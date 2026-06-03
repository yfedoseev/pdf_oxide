//! Render the Separation/DeviceN reproducer PDFs to PNG so the tint-transform
//! fix can be seen by eye.
//!
//! Run: cargo run --example render_separation_demo --features rendering [label]
//!
//! Writes `examples/separation-blackout/<name>-<label>.png` (label defaults to
//! "rendered") for each reproducer PDF and prints its centre pixel. A correct
//! renderer produces a light green (≈ 230,255,216); the pre-fix renderer
//! ignored the tint transform and painted solid black (0,0,0).

use std::path::Path;

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page_fit, ImageFormat, RenderOptions};

fn main() {
    let label = std::env::args().nth(1).unwrap_or_else(|| "rendered".to_string());
    let dir = Path::new("examples/separation-blackout");

    for name in ["separation-type2", "separation-type0"] {
        let pdf_path = dir.join(format!("{name}.pdf"));
        let bytes = std::fs::read(&pdf_path).expect("read reproducer pdf");
        let doc = PdfDocument::from_bytes(bytes).expect("open reproducer pdf");

        // Centre pixel (raw RGBA) for a quick console read-out.
        let raw = render_page_fit(&doc, 0, 200, 200, &RenderOptions::with_dpi(72).as_raw())
            .expect("render raw");
        debug_assert_eq!(raw.format, ImageFormat::RawRgba8);
        let (cx, cy) = (raw.width / 2, raw.height / 2);
        let i = ((cy * raw.width + cx) * 4) as usize;
        println!(
            "{name}: centre pixel = ({}, {}, {})",
            raw.data[i],
            raw.data[i + 1],
            raw.data[i + 2],
        );

        // PNG for the README.
        let png = render_page_fit(&doc, 0, 200, 200, &RenderOptions::with_dpi(72))
            .expect("render png");
        let png_path = dir.join(format!("{name}-{label}.png"));
        png.save(&png_path).expect("save png");
        println!("wrote {}", png_path.display());
    }
}
