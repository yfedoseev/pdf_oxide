use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <pdf_path> <output_path>", args[0]);
        std::process::exit(1);
    }

    let pdf_path = &args[1];
    let output_path = &args[2];

    match PdfDocument::open(pdf_path) {
        Ok(mut doc) => {
            let options = ConversionOptions::default();
            match doc.to_markdown_all(&options) {
                Ok(markdown) => {
                    if let Err(e) = fs::write(output_path, &markdown) {
                        eprintln!("Failed to write output: {}", e);
                        std::process::exit(1);
                    }
                    println!("Success: {} bytes", markdown.len());
                },
                Err(e) => {
                    eprintln!("Failed to convert to markdown: {}", e);
                    std::process::exit(1);
                },
            }
        },
        Err(e) => {
            eprintln!("Failed to open PDF: {}", e);
            std::process::exit(1);
        },
    }
}
