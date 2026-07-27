// Search for a term across all pages of a PDF and print matches.
// Run: cargo run --example tutorial_search_text -- tests/fixtures/simple.pdf "the"

use pdf_oxide::{
    search::{SearchOptions, TextSearcher},
    PdfDocument,
};
use std::{env, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: tutorial_search_text <file.pdf> <query>");
        process::exit(1);
    }
    let path = &args[1];
    let query = &args[2];

    let doc = PdfDocument::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {}", path, e);
        process::exit(1);
    });

    let pages = doc.page_count().unwrap_or(0);
    println!("Searching for {:?} in {} ({} pages)...\n", query, path, pages);

    let re = regex::Regex::new(query).unwrap_or_else(|e| {
        eprintln!("Invalid regex: {}", e);
        process::exit(1);
    });

    // Build the per-page search index for every page up front, instead of
    // the lazy per-page build TextSearcher::search_page() otherwise does on
    // first use. Worth it here since we're about to search every page anyway.
    doc.prepare_search().unwrap_or_else(|e| {
        eprintln!("Failed to prepare search index: {}", e);
        process::exit(1);
    });

    let opts = SearchOptions::default();
    let mut total = 0;
    for i in 0..pages {
        let results = TextSearcher::search_page(&doc, i, &re, &opts).unwrap_or_default();
        if results.is_empty() {
            continue;
        }
        println!("Page {}: {} match(es)", i + 1, results.len());
        for r in &results {
            println!("  - {:?}", r.text);
            total += 1;
        }
    }
    println!("\nFound {} total matches.", total);

    // Free the cached search index now that we're done searching — useful
    // before heavy extraction work on the same document object.
    doc.clear_search_index();
}
