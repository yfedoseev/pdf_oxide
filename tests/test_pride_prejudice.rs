//! Tests for Pride and Prejudice scanned PDF
//!
//! A public domain scanned book with OCR layer
//! - Source: Archive.org
//! - Pages: 424
//! - Size: 8.3 MB
//! - Format: Scanned with OCR

#[cfg(feature = "ocr")]
mod pride_tests {
    use pdf_oxide::PdfDocument;
    use std::path::Path;

    const PRIDE_PDF: &str = "scanned_samples/pride_prejudice.pdf";

    fn has_pdf() -> bool {
        Path::new(PRIDE_PDF).exists()
    }

    #[test]
    fn test_pride_pdf_exists() {
        if has_pdf() {
            println!("\n✓ Pride and Prejudice PDF found");
            if let Ok(metadata) = std::fs::metadata(PRIDE_PDF) {
                println!("  Size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
            }
        } else {
            println!("✗ Pride PDF not found");
        }
    }

    #[test]
    fn test_pride_pdf_opens() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        match PdfDocument::open(PRIDE_PDF) {
            Ok(mut doc) => match doc.page_count() {
                Ok(count) => {
                    println!("\n✓ Pride and Prejudice opened successfully");
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
    fn test_pride_first_page() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        match PdfDocument::open(PRIDE_PDF) {
            Ok(mut doc) => {
                println!("\n=== Pride and Prejudice - Page 0 Analysis ===");

                // Get text
                match doc.extract_text(0) {
                    Ok(text) => {
                        let text_len = text.trim().len();
                        println!("✓ Text extracted: {} characters", text_len);
                        if text_len > 200 {
                            println!("  Sample (first 200 chars):");
                            println!("  {}", &text[..200].replace('\n', " "));
                        }
                    },
                    Err(e) => println!("✗ Error extracting text: {:?}", e),
                }

                // Get images
                match doc.extract_images(0) {
                    Ok(images) => {
                        println!("✓ Images found: {}", images.len());
                        for (idx, img) in images.iter().enumerate() {
                            println!("  Image {}: {}x{} pixels", idx, img.width(), img.height());
                        }
                    },
                    Err(e) => println!("✗ Error extracting images: {:?}", e),
                }
            },
            Err(e) => {
                println!("✗ Failed to open PDF: {:?}", e);
            },
        }
    }

    #[test]
    fn test_pride_ocr_readiness() {
        if !has_pdf() {
            println!("PDF not found - skipping");
            return;
        }

        match PdfDocument::open(PRIDE_PDF) {
            Ok(mut doc) => {
                match doc.page_count() {
                    Ok(page_count) => {
                        println!("\n=== Pride and Prejudice - OCR Readiness ===");

                        let mut with_text = 0;
                        let mut image_only = 0;

                        // Check first 5 pages only
                        for page_idx in 0..page_count.min(5) {
                            let text = doc.extract_text(page_idx).unwrap_or_default();
                            if text.trim().len() > 50 {
                                with_text += 1;
                            } else {
                                image_only += 1;
                            }
                        }

                        println!("✓ Analysis of first 5 pages:");
                        println!("  Pages with text: {}", with_text);
                        println!("  Image-only pages: {}", image_only);
                        println!("  Total pages: {}", page_count);

                        if image_only > 0 {
                            println!("✓ This PDF has scanned image pages suitable for OCR!");
                        }
                    },
                    Err(e) => println!("✗ Error getting page count: {:?}", e),
                }
            },
            Err(e) => {
                println!("✗ Failed to open PDF: {:?}", e);
            },
        }
    }
}

#[cfg(not(feature = "ocr"))]
mod pride_tests_disabled {
    #[test]
    fn test_pride_feature_disabled() {
        println!("OCR feature not enabled - Pride tests skipped");
    }
}
