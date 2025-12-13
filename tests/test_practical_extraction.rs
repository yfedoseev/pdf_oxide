#[cfg(feature = "ocr")]
#[test]
fn test_practical_pdf_to_markdown() {
    use pdf_oxide::document::PdfDocument;
    use pdf_oxide::converters::ConversionOptions;

    let pdf_path = "scanned_samples/pride_prejudice.pdf";
    if !std::path::Path::new(pdf_path).exists() {
        println!("PDF not found!");
        return;
    }

    println!("\n╔════════════════════════════════════════╗");
    println!("║  PRACTICAL PDF → MARKDOWN CONVERSION  ║");
    println!("╚════════════════════════════════════════╝\n");

    let mut doc = match PdfDocument::open(pdf_path) {
        Ok(d) => d,
        Err(e) => {
            println!("❌ Failed to open: {}", e);
            return;
        }
    };

    let page_count = match doc.page_count() {
        Ok(count) => count,
        Err(e) => {
            println!("Failed to get page count: {}", e);
            return;
        }
    };

    println!("✓ PDF opened: {} pages total", page_count);

    // Extract first 50 pages (reasonable batch)
    let pages_to_extract = 50.min(page_count as usize);
    println!("✓ Extracting first {} pages...\n", pages_to_extract);

    let options = ConversionOptions::default();
    let mut markdown_content = String::new();
    let mut pages_processed = 0;
    let mut chars_extracted = 0;

    for page_num in 0..pages_to_extract {
        match doc.extract_text(page_num) {
            Ok(text) => {
                chars_extracted += text.len();
            }
            Err(_) => {}
        }

        match doc.to_markdown(page_num, &options) {
            Ok(md) => {
                if !md.is_empty() {
                    markdown_content.push_str(&md);
                    markdown_content.push_str("\n\n---\n\n");
                    pages_processed += 1;
                }
            }
            Err(_) => {}
        }
    }

    println!("═══════════════════════════════════════════════");
    println!("RESULTS:");
    println!("═══════════════════════════════════════════════");
    println!("✓ Pages processed: {}", pages_processed);
    println!("✓ Text extracted: {} characters", chars_extracted);
    println!("✓ Markdown generated: {} characters", markdown_content.len());
    println!("✓ Average per page: {} chars", 
             if pages_processed > 0 { markdown_content.len() / pages_processed } else { 0 });

    if markdown_content.len() > 0 {
        // Save the markdown
        match std::fs::write("/tmp/pride_prejudice_50pages.md", &markdown_content) {
            Ok(_) => {
                println!("✓ Saved to: /tmp/pride_prejudice_50pages.md");
                
                let lines = markdown_content.lines().count();
                let word_count = markdown_content.split_whitespace().count();
                println!("\nMARKDOWN STATISTICS:");
                println!("  - Lines: {}", lines);
                println!("  - Words: {}", word_count);
                println!("  - File size: {} KB", markdown_content.len() / 1024);
                
                println!("\nFIRST 400 CHARACTERS:");
                println!("────────────────────────────────────────");
                println!("{}", markdown_content.chars().take(400).collect::<String>());
                println!("────────────────────────────────────────");
                println!("\n✅ PDF TO MARKDOWN CONVERSION: SUCCESS");
            }
            Err(e) => println!("❌ Failed to save: {}", e),
        }
    } else {
        println!("⚠️  No markdown content generated");
    }
}
