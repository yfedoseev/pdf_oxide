//! Open PDF, remove headers/footers, print markdown to stdout.
//!
//! Usage: cargo run --example extract_markdown_clean -- <file.pdf>
use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::{Error, PdfDocument};

fn print_markdown(filename: &str) -> Result<usize, Error> {
    match PdfDocument::open(filename) {
        Err(e) => {
            println!("Error opening {}", filename);
            Err(e)
        },
        Ok(doc) => {
            doc.remove_artifacts(0.8)?;
            let options = ConversionOptions {
                detect_headings: true,
                ..Default::default()
            };
            let pages = doc.page_count()?;
            for page_index in 0..pages {
                let text = doc.to_markdown(page_index, &options)?;
                println!("{}\n", text);
            }
            Ok(pages)
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
    print_markdown(pdf_path)?;
    Ok(())
}
