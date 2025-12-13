#[cfg(feature = "ocr")]
#[test]
fn test_extract_and_output_markdown() {
    use pdf_oxide::document::PdfDocument;
    use pdf_oxide::converters::ConversionOptions;

    let pdf_path = "scanned_samples/pride_prejudice.pdf";
    if !std::path::Path::new(pdf_path).exists() {
        println!("PDF not found!");
        return;
    }

    println!("\n=== ACTUAL OUTPUT TEST ===\n");

    let mut doc = match PdfDocument::open(pdf_path) {
        Ok(d) => d,
        Err(e) => {
            println!("❌ Failed to open: {}", e);
            return;
        }
    };

    // Get page count
    let page_count = match doc.page_count() {
        Ok(count) => count,
        Err(e) => {
            println!("Failed to get page count: {}", e);
            return;
        }
    };
    println!("Total pages: {}", page_count);

    // Try to extract from first few pages
    for page_num in 0..3.min(page_count as usize) {
        println!("\nPage {}:", page_num);
        match doc.extract_text(page_num) {
            Ok(text) => {
                println!("  ✓ Text: {} chars", text.len());
                if text.len() > 0 {
                    let preview = text.chars().take(100).collect::<String>();
                    println!("  Preview: {}", preview);
                }
            }
            Err(e) => println!("  ✗ Text extraction failed: {}", e),
        }

        match doc.extract_images(page_num) {
            Ok(images) => println!("  ✓ Images: {}", images.len()),
            Err(e) => println!("  ✗ Image extraction failed: {}", e),
        }
    }

    // Try to convert to markdown
    println!("\n=== MARKDOWN CONVERSION ===");
    let options = ConversionOptions::default();
    match doc.to_markdown(0, &options) {
        Ok(markdown) => {
            println!("✅ SUCCESS! Generated {} chars of Markdown", markdown.len());
            if markdown.len() > 0 {
                println!("\nFirst 300 chars:");
                let preview = markdown.chars().take(300).collect::<String>();
                println!("{}", preview);
            }
            
            // Save it
            if let Ok(_) = std::fs::write("/tmp/pride_prejudice_output.md", &markdown) {
                println!("\n✅ Saved to /tmp/pride_prejudice_output.md");
                println!("File size: {} bytes", markdown.len());
            }
        }
        Err(e) => {
            println!("❌ FAILED: {}", e);
        }
    }
}
