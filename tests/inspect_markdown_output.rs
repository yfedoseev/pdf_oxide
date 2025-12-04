use pdf_oxide::converters::{ConversionOptions, MarkdownConverter};
use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::SpanMergingConfig;
use std::fs;
use std::path::Path;

/// Manual inspection test - extracts markdown and saves it for manual review
/// This helps validate whether detected "spurious spaces" actually exist in output
#[test]
#[ignore] // Run with --ignored flag
fn inspect_academic_pdf_markdown() {
    let pdf_path = "tests/fixtures/regression/academic/arxiv_2510.21165v1.pdf";
    assert!(Path::new(pdf_path).exists(), "Test PDF not found: {}", pdf_path);

    // Extract markdown
    let mut doc = PdfDocument::open(pdf_path).expect("Failed to open PDF");

    let converter = MarkdownConverter::new();
    let options = ConversionOptions::default();
    let config = SpanMergingConfig::adaptive();

    let mut markdown = String::new();
    let page_count = doc.page_count().expect("Failed to get page count");

    // Extract first 5 pages
    for page_num in 0..std::cmp::min(5, page_count) {
        let spans = doc
            .extract_spans_with_config(page_num, config.clone())
            .expect("Failed to extract spans");
        let page_md = converter
            .convert_page_from_spans(&spans, &options)
            .expect("Failed to convert spans");
        markdown.push_str(&page_md);
        markdown.push_str("\n\n");
    }

    // Save to file for manual inspection
    let output_path = "/tmp/academic_pdf_inspection.md";
    fs::write(output_path, &markdown).expect("Failed to write markdown file");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  MARKDOWN INSPECTION TEST - Academic PDF                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Extracted markdown saved to: {}", output_path);
    println!("Total characters: {}", markdown.len());
    println!("Total lines: {}", markdown.lines().count());
    println!();

    // Look for patterns of actual multiple spaces
    let double_space_count = markdown.matches("  ").count();
    let triple_space_count = markdown.matches("   ").count();
    let quad_space_count = markdown.matches("    ").count();

    println!("Space Pattern Analysis:");
    println!("  Double spaces (  ): {}", double_space_count);
    println!("  Triple spaces (   ): {}", triple_space_count);
    println!("  Quad+ spaces (    ): {}", quad_space_count);
    println!();

    // Show first 10 lines with multiple spaces
    println!("Lines with multiple consecutive spaces (first 20):");
    let mut count = 0;
    for (line_num, line) in markdown.lines().enumerate() {
        if line.contains("  ") {
            if count < 20 {
                println!(
                    "  Line {}: {:?}",
                    line_num + 1,
                    line.chars().take(100).collect::<String>()
                );
                count += 1;
            }
        }
    }
    println!();

    // Extract specific patterns that might be "spurious spaces"
    println!("Searching for suspicious space patterns:");

    // Pattern: single letter followed by spaces and then letters (e.g., "a  b")
    let mut suspicious = 0;
    for (line_num, line) in markdown.lines().enumerate() {
        // Look for pattern: word, then 2+ spaces, then word
        let parts: Vec<&str> = line.split("  ").collect();
        if parts.len() > 1 {
            for i in 0..parts.len() - 1 {
                let left_end = parts[i].trim_end();
                let right_start = parts[i + 1].trim_start();

                // Check if we have word boundary
                if !left_end.is_empty() && !right_start.is_empty() {
                    if left_end
                        .chars()
                        .last()
                        .map(|c| c.is_alphabetic())
                        .unwrap_or(false)
                        && right_start
                            .chars()
                            .next()
                            .map(|c| c.is_alphabetic())
                            .unwrap_or(false)
                    {
                        suspicious += 1;
                        if suspicious <= 10 {
                            println!(
                                "  Line {}: \"{}  {}\"",
                                line_num + 1,
                                left_end.chars().rev().take(20).collect::<String>(),
                                right_start.chars().take(20).collect::<String>()
                            );
                        }
                    }
                }
            }
        }
    }
    println!("  Total suspicious boundaries with 2+ spaces: {}", suspicious);
    println!();

    println!("Note: This is a manual inspection to validate whether the detected");
    println!("'spurious spaces' actually exist as multiple-space patterns in the markdown.");
    println!();
    println!("If double_space_count is much lower than the 136 reported spurious spaces,");
    println!("then the quality detection regex is counting something different.");

    panic!("INSPECTION COMPLETE - Check output above and compare with quality metrics");
}
