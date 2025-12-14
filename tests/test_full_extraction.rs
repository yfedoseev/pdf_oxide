#![allow(
    clippy::unnecessary_cast,
    clippy::single_match,
    clippy::len_zero,
    clippy::redundant_pattern_matching
)]
#[cfg(feature = "ocr")]
#[test]
fn test_full_document_extraction() {
    use pdf_oxide::converters::ConversionOptions;
    use pdf_oxide::document::PdfDocument;

    let pdf_path = "scanned_samples/pride_prejudice.pdf";
    if !std::path::Path::new(pdf_path).exists() {
        println!("PDF not found!");
        return;
    }

    println!("\n=== FULL DOCUMENT EXTRACTION TEST ===\n");

    let mut doc = match PdfDocument::open(pdf_path) {
        Ok(d) => d,
        Err(e) => {
            println!("❌ Failed to open: {}", e);
            return;
        },
    };

    let page_count = match doc.page_count() {
        Ok(count) => count,
        Err(e) => {
            println!("Failed to get page count: {}", e);
            return;
        },
    };

    let mut total_chars = 0;
    let mut pages_with_text = 0;
    let mut total_images = 0;

    println!("Processing {} pages...", page_count);
    for page_num in 0..page_count as usize {
        match doc.extract_text(page_num) {
            Ok(text) => {
                if text.len() > 0 {
                    pages_with_text += 1;
                    total_chars += text.len();
                }
            },
            Err(_) => {},
        }

        match doc.extract_images(page_num) {
            Ok(images) => {
                total_images += images.len();
            },
            Err(_) => {},
        }

        if (page_num + 1) % 100 == 0 {
            println!("  Progress: {}/{} pages", page_num + 1, page_count);
        }
    }

    println!("\n=== EXTRACTION SUMMARY ===");
    println!("Total pages: {}", page_count);
    println!("Pages with extractable text: {}", pages_with_text);
    println!("Total characters: {}", total_chars);
    println!("Total images: {}", total_images);
    println!(
        "Average per text page: {} chars",
        if pages_with_text > 0 {
            total_chars / pages_with_text
        } else {
            0
        }
    );

    // Now convert full document to markdown
    println!("\n=== MARKDOWN CONVERSION (Full Document) ===");
    let options = ConversionOptions::default();

    // Try converting each page and collecting
    let mut all_markdown = String::new();
    for page_num in 0..page_count as usize {
        match doc.to_markdown(page_num, &options) {
            Ok(markdown) => {
                if !markdown.is_empty() {
                    all_markdown.push_str(&markdown);
                    all_markdown.push('\n');
                }
            },
            Err(e) => {
                // Some pages may fail, that's ok
                log::debug!("Page {} markdown conversion failed: {}", page_num, e);
            },
        }
    }

    println!("✅ Generated Markdown: {} chars", all_markdown.len());

    // Save it
    if let Ok(_) = std::fs::write("/tmp/pride_prejudice_full.md", &all_markdown) {
        println!("✅ Saved full document to /tmp/pride_prejudice_full.md");

        // Show stats
        let lines = all_markdown.lines().count();
        let words: Vec<&str> = all_markdown.split_whitespace().collect();
        println!("\nMarkdown Statistics:");
        println!("  Lines: {}", lines);
        println!("  Words: {}", words.len());
        println!("  Bytes: {}", all_markdown.len());
        println!("  First 200 chars:\n{}\n", all_markdown.chars().take(200).collect::<String>());
    }
}
