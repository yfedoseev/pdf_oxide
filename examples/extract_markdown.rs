//! Print a page's Markdown conversion to stdout.
//!
//! Usage: cargo run --example extract_markdown -- <file.pdf> [page]

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

fn main() {
    let path = std::env::args().nth(1).expect("usage: <file.pdf> [page]");
    let page: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let doc = PdfDocument::open(&path).expect("open pdf");
    let md = doc
        .to_markdown(page, &ConversionOptions::default())
        .expect("markdown");
    println!("{md}");
}
