// Dashed stroke lines and rectangles (Rust-idiomatic; uses `LineStyle::with_dash`).
//
// Run: cargo run --example showcase_dashed_stroke

use pdf_oxide::error::Result;
use pdf_oxide::writer::{DocumentBuilder, LineStyle};
use std::path::PathBuf;

fn main() -> Result<()> {
    let out_dir = PathBuf::from("target/examples_output/dashed_stroke");
    std::fs::create_dir_all(&out_dir)?;

    let mut builder = DocumentBuilder::new();
    builder
        .letter_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .heading(1, "Dashed Stroke Demo")
        .at(72.0, 690.0)
        .paragraph("Rectangles and lines drawn with configurable dash patterns.")
        // Dashed rectangle: [5 on, 3 off], 2pt blue border.
        .stroke_rect(
            72.0,
            560.0,
            300.0,
            80.0,
            LineStyle::new(2.0, 0.0, 0.2, 0.8).with_dash(&[5.0, 3.0], 0.0),
        )
        // Dashed line: [8 on, 4 off], 1.5pt red.
        .stroke_line(
            72.0,
            520.0,
            372.0,
            520.0,
            LineStyle::new(1.5, 0.8, 0.0, 0.0).with_dash(&[8.0, 4.0], 0.0),
        )
        .done();

    let bytes = builder.build()?;
    let path = out_dir.join("dashed.pdf");
    std::fs::write(&path, &bytes)?;
    println!("Wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(())
}
