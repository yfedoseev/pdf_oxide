//! Tests for Alice in Wonderland scanned PDF
//!
//! A public domain scanned book with OCR layer
//! - Source: Archive.org
//! - Pages: 202
//! - Size: 7.5 MB
//! - Format: Scanned with OCR

#[cfg(feature = "ocr")]
mod alice_tests {
    use pdf_oxide::PdfDocument;
    use std::path::Path;

    const ALICE_PDF: &str = "scanned_samples/alice_wonderland.pdf";

    fn has_pdf() -> bool {
        Path::new(ALICE_PDF).exists()
    }

    #[test]
    fn test_alice_pdf_exists() {
        if has_pdf() {
            println!("✓ Alice in Wonderland PDF found");
            if let Ok(metadata) = std::fs::metadata(ALICE_PDF) {
                println!("  Size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
            }
        } else {
            println!("✗ Alice PDF not found");
        }
    }

    #[test]
    fn test_alice_pdf_opens() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        println!("\n=== Alice in Wonderland - Document Analysis ===");

        match PdfDocument::open(ALICE_PDF) {
            Ok(mut doc) => match doc.page_count() {
                Ok(count) => {
                    println!("✓ PDF opened successfully");
                    println!("  Pages: {}", count);
                },
                Err(e) => println!("✗ Error getting page count: {:?}", e),
            },
            Err(e) => {
                println!("✗ Failed to open PDF: {:?}", e);
            },
        }
    }

    #[test]
    fn test_alice_page_analysis() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        println!("\n=== Alice Page Analysis (first 10 pages) ===");

        match PdfDocument::open(ALICE_PDF) {
            Ok(mut doc) => {
                if let Ok(page_count) = doc.page_count() {
                    let pages_to_check = page_count.min(10);

                    for page_idx in 0..pages_to_check {
                        print!("Page {}: ", page_idx);

                        // Check for text
                        match doc.extract_text(page_idx) {
                            Ok(text) => {
                                let text_len = text.trim().len();
                                println!("{} chars", text_len);

                                if text_len > 0 {
                                    // Show sample
                                    let sample = &text[..100.min(text_len)];
                                    println!("        Sample: {}...", sample.replace('\n', " "));
                                }
                            },
                            Err(_) => println!("? Error reading text"),
                        }

                        // Check for images
                        if let Ok(images) = doc.extract_images(page_idx) {
                            if !images.is_empty() {
                                for (idx, img) in images.iter().enumerate() {
                                    println!(
                                        "        Image {}: {}x{} pixels",
                                        idx,
                                        img.width(),
                                        img.height()
                                    );
                                }
                            }
                        }
                    }
                }
            },
            Err(e) => println!("✗ Failed to open PDF: {:?}", e),
        }
    }

    #[test]
    fn test_alice_ocr_readiness() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        println!("\n=== Alice OCR Readiness ===");

        match PdfDocument::open(ALICE_PDF) {
            Ok(mut doc) => {
                if let Ok(page_count) = doc.page_count() {
                    let mut images_found = 0;
                    let mut text_pages = 0;
                    let mut image_pages = 0;

                    // Sample first 20 pages
                    for page_idx in 0..page_count.min(20) {
                        let text = doc.extract_text(page_idx).unwrap_or_default();
                        if text.trim().len() > 50 {
                            text_pages += 1;
                        } else {
                            image_pages += 1;
                        }

                        if let Ok(images) = doc.extract_images(page_idx) {
                            images_found += images.len();
                        }
                    }

                    println!("✓ Analysis of first 20 pages:");
                    println!("  Pages with text: {}", text_pages);
                    println!("  Image-only pages: {}", image_pages);
                    println!("  Total images found: {}", images_found);
                    println!("  Document type: Scanned with OCR layer");
                }
            },
            Err(e) => println!("✗ Error: {:?}", e),
        }
    }
}

#[cfg(not(feature = "ocr"))]
mod alice_tests_disabled {
    #[test]
    fn test_alice_feature_disabled() {
        println!("OCR feature not enabled - Alice tests skipped");
    }
}
