//! Open PDF, remove headers/footers, write markdown to a file alongside the PDF.
//!
//! Usage: cargo run --example extract_markdown_clean -- <file.pdf>
use std::path::Path;

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::{Error, PdfDocument};

fn generate_markdown(filename: &str) -> Result<usize, Error> {
    match PdfDocument::open(filename) {
        Err(e) => {
            println!("Error opening {}", filename);
            Err(e)
        },
        Ok(doc) => {
            let count = doc.remove_artifacts(0.8)?;
            let options = ConversionOptions {
                detect_headings: true,
                include_artifacts: false,
                ..Default::default()
            };
            let pages = doc.page_count()?;
            let mut markdown = String::new();
            for page_index in 0..pages {
                let text = doc.to_markdown(page_index, &options)?;
                markdown.push_str(&text);
                markdown.push_str("\n\n");
            }

            let output_path = Path::new(filename).with_extension("md");
            std::fs::write(&output_path, markdown)?;
            println!("Wrote markdown to {}", output_path.display());

            Ok(count)
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pdf_file>", args[0]);
        std::process::exit(1);
    }

    let pdf_path = &args[1];
    let removed_count = generate_markdown(pdf_path)?;
    println!("Removed {} artifacts", removed_count);
    Ok(())
}
