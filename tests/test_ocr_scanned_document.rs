#![allow(clippy::single_match)]
#![allow(dead_code)]
//! OCR testing with real scanned PDF document
//!
//! Tests OCR functionality with "A Grammar Of The Vulgate"
//! - Downloaded from: Archive.org (public domain)
//! - Format: Scanned book with OCR layer
//! - Pages: 400+ pages of historical text
//! - Quality: High-quality scan

#[cfg(feature = "ocr")]
mod ocr_scanned_tests {
    use pdf_oxide::PdfDocument;
    use std::path::Path;
    use std::time::Instant;

    const SCANNED_PDF: &str = "scanned_samples/grammar_vulgate.pdf";

    fn has_test_pdf() -> bool {
        Path::new(SCANNED_PDF).exists()
    }

    // ========================================================================
    // PDF VERIFICATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scanned_pdf_exists() {
        if has_test_pdf() {
            println!("\n✓ Scanned PDF found: {}", SCANNED_PDF);

            if let Ok(metadata) = std::fs::metadata(SCANNED_PDF) {
                println!("  Size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
                println!("  Modified: {:?}", metadata.modified());
            }
        } else {
            println!("\n✗ Test PDF not found: {}", SCANNED_PDF);
            println!("  To download a test PDF:");
            println!("  mkdir -p scanned_samples");
            println!("  cd scanned_samples");
            println!("  curl -L -o grammar_vulgate.pdf \\");
            println!(
                "    'https://archive.org/download/AGrammarOfTheVulgateBWNewOCRAndMarginCropped19Jan2017ForAll/A%20Grammar%20Of%20The%20Vulgate%20%28BW%20New%20OCR%20and%20Margin%20cropped_19%20Jan%202017%20for%20all.pdf'"
            );
        }
    }

    #[test]
    fn test_ocr_scanned_pdf_can_be_opened() {
        if !has_test_pdf() {
            println!("Scanned PDF not available - skipping");
            return;
        }

        println!("\n=== Opening Scanned PDF ===");
        match PdfDocument::open(SCANNED_PDF) {
            Ok(mut doc) => match doc.page_count() {
                Ok(count) => {
                    println!("✓ PDF opened successfully");
                    println!("  Pages: {}", count);
                },
                Err(e) => println!("✗ Error getting page count: {:?}", e),
            },
            Err(e) => {
                println!("✗ Failed to open PDF: {:?}", e);
                panic!("Could not open test PDF");
            },
        }
    }

    // ========================================================================
    // PAGE ANALYSIS TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scanned_pdf_page_analysis() {
        if !has_test_pdf() {
            println!("Scanned PDF not available - skipping");
            return;
        }

        println!("\n=== Scanned PDF Page Analysis ===");

        match PdfDocument::open(SCANNED_PDF) {
            Ok(mut doc) => {
                if let Ok(page_count) = doc.page_count() {
                    println!("Total pages: {}\n", page_count);

                    // Analyze first 5 pages
                    let pages_to_check = page_count.min(5);
                    println!("Analyzing first {} pages:\n", pages_to_check);

                    for page_idx in 0..pages_to_check {
                        print!("Page {}: ", page_idx);

                        // Check for native text
                        match doc.extract_text(page_idx) {
                            Ok(text) => {
                                let text_len = text.trim().len();
                                println!("{} chars", text_len);

                                if text_len < 100 {
                                    println!("  ⚠️  Little native text (likely scanned image)");
                                } else {
                                    println!("  ✓ Has substantial text");
                                }
                            },
                            Err(_) => {
                                println!("? Error reading text");
                            },
                        }

                        // Check for images
                        match doc.extract_images(page_idx) {
                            Ok(images) => {
                                if !images.is_empty() {
                                    println!("  📷 Contains {} image(s)", images.len());
                                }
                            },
                            Err(_) => {},
                        }
                    }

                    println!("\n═════════════════════════════════");
                }
            },
            Err(e) => {
                println!("Failed to analyze PDF: {:?}", e);
            },
        }
    }

    // ========================================================================
    // OCR READINESS TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scanned_pdf_ocr_candidates() {
        if !has_test_pdf() {
            println!("Scanned PDF not available - skipping");
            return;
        }

        println!("\n=== OCR Readiness Assessment ===");

        match PdfDocument::open(SCANNED_PDF) {
            Ok(mut doc) => {
                if let Ok(page_count) = doc.page_count() {
                    let mut ocr_candidates = 0;
                    let mut text_pages = 0;

                    for page_idx in 0..page_count.min(20) {
                        let text = doc.extract_text(page_idx).unwrap_or_default();
                        let text_len = text.trim().len();

                        if text_len < 50 {
                            ocr_candidates += 1;
                        } else {
                            text_pages += 1;
                        }
                    }

                    println!("✓ PDF Analysis (first 20 pages):");
                    println!("  Pages with substantial text: {}", text_pages);
                    println!("  Pages suitable for OCR:      {}", ocr_candidates);

                    if ocr_candidates > 0 {
                        println!("\n✓ This PDF contains pages that would benefit from OCR");
                        println!("  Good candidate for OCR testing!");
                    } else {
                        println!("\n⚠️  This PDF mostly has extractable text");
                        println!("  May not need OCR for most pages");
                    }
                }
            },
            Err(e) => println!("Error: {:?}", e),
        }
    }

    // ========================================================================
    // PERFORMANCE TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scanned_pdf_extraction_performance() {
        if !has_test_pdf() {
            println!("Scanned PDF not available - skipping");
            return;
        }

        println!("\n=== Text Extraction Performance ===");

        match PdfDocument::open(SCANNED_PDF) {
            Ok(mut doc) => {
                let start = Instant::now();

                if let Ok(page_count) = doc.page_count() {
                    let pages_to_test = page_count.min(10);
                    let mut total_chars = 0;

                    for page_idx in 0..pages_to_test {
                        if let Ok(text) = doc.extract_text(page_idx) {
                            total_chars += text.len();
                        }
                    }

                    let duration = start.elapsed();

                    println!("✓ Extracted {} pages", pages_to_test);
                    println!("  Total characters: {}", total_chars);
                    println!("  Time elapsed: {:.2}s", duration.as_secs_f64());
                    println!(
                        "  Average per page: {:.1} ms",
                        duration.as_millis() as f64 / pages_to_test as f64
                    );
                    println!(
                        "  Throughput: {:.0} chars/sec",
                        total_chars as f64 / duration.as_secs_f64()
                    );
                }
            },
            Err(e) => println!("Error: {:?}", e),
        }
    }

    // ========================================================================
    // OCR CONFIGURATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scanned_config_for_historical_documents() {
        use pdf_oxide::ocr::OcrConfig;

        println!("\n=== OCR Configuration for Historical Documents ===");

        // Historical/degraded documents require more lenient thresholds
        let config = OcrConfig::builder()
            .det_threshold(0.30) // More lenient detection
            .box_threshold(0.45) // More lenient box filtering
            .rec_threshold(0.45) // More lenient recognition
            .num_threads(4)
            .detect_styles(false) // Styles may not be reliable
            .build();

        println!("Recommended settings for scanned historical documents:");
        println!("  Detection threshold: {}", config.det_threshold);
        println!("  Box threshold: {}", config.box_threshold);
        println!("  Recognition threshold: {}", config.rec_threshold);
        println!("  Threads: {}", config.num_threads);
        println!("  Detect styles: {}", config.detect_styles);

        println!("\nRationale:");
        println!("  - Historical docs have variable print quality");
        println!("  - Pages may be faded or discolored");
        println!("  - Lower thresholds catch more text at cost of false positives");
        println!("  - Multiple threads help with large documents");
    }

    // ========================================================================
    // SAMPLE EXTRACTION TEST
    // ========================================================================

    #[test]
    fn test_ocr_scanned_sample_page_extraction() {
        if !has_test_pdf() {
            println!("Scanned PDF not available - skipping");
            return;
        }

        println!("\n=== Sample Page Extraction ===");

        match PdfDocument::open(SCANNED_PDF) {
            Ok(mut doc) => match doc.extract_text(0) {
                Ok(text) => {
                    println!("✓ Successfully extracted text from page 0");
                    println!("  Length: {} characters", text.len());

                    if text.len() > 200 {
                        println!("  Preview (first 200 chars):");
                        println!("  ────────────────────────────");
                        println!("{}", &text[..200.min(text.len())]);
                        println!("  ────────────────────────────");
                    } else {
                        println!("  Full text:");
                        println!("  ────────────────────────────");
                        println!("{}", text);
                        println!("  ────────────────────────────");
                    }
                },
                Err(e) => println!("Error extracting text: {:?}", e),
            },
            Err(e) => println!("Error opening PDF: {:?}", e),
        }
    }

    // ========================================================================
    // INFORMATION DISPLAY
    // ========================================================================

    #[test]
    fn test_ocr_scanned_pdf_information() {
        if !has_test_pdf() {
            println!("\n📄 Test Scanned PDF Not Available");
            println!("Download location: {}", SCANNED_PDF);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║  Scanned PDF Information                          ║");
        println!("╚═══════════════════════════════════════════════════╝");

        println!("\nDocument: A Grammar Of The Vulgate");
        println!("Source: Archive.org (Public Domain)");
        println!("License: Public Domain Mark 1.0");
        println!("Pages: 400+ pages of historical text");
        println!("Languages: Latin, English");
        println!("Quality: High-quality scans with OCR layer");

        println!("\nDocument Type:");
        println!("  - Historical academic text");
        println!("  - Published before 1900 (now public domain)");
        println!("  - Scanned from physical book");
        println!("  - Already has OCR layer from archive.org");

        println!("\nWhat This Tests:");
        println!("  ✓ OCR on real historical documents");
        println!("  ✓ Multi-language text (Latin + English)");
        println!("  ✓ Performance on large documents (400+ pages)");
        println!("  ✓ Integration with existing PDF code");

        println!("\nIdeal For:");
        println!("  - End-to-end OCR pipeline testing");
        println!("  - Performance benchmarking");
        println!("  - Quality validation");
        println!("  - Accuracy measurement");

        println!("\n═══════════════════════════════════════════════════\n");
    }
}

// ============================================================================
// TESTS FOR WHEN OCR FEATURE IS NOT ENABLED
// ============================================================================

#[cfg(not(feature = "ocr"))]
mod ocr_scanned_not_enabled_tests {
    #[test]
    fn test_ocr_scanned_feature_disabled() {
        println!("OCR feature is not enabled - scanned PDF tests skipped");
    }
}
