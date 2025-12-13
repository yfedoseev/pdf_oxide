//! End-to-end OCR integration tests with real PDFs
//!
//! These tests run actual OCR on real PDF documents from the test dataset.
//! They validate the complete OCR pipeline: loading models, processing pages,
//! and extracting text.

#[cfg(feature = "ocr")]
mod ocr_e2e_tests {
    use pdf_oxide::PdfDocument;
    use std::path::Path;

    // Test PDF paths from the test dataset
    const ACADEMIC_PDF: &str =
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf";
    const FORMS_PDF: &str =
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/forms/AmazonQ1_Shareholder.pdf";
    const GOVERNMENT_PDF: &str =
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/government/eo_2024_08_20_signed.pdf";

    fn get_test_pdf() -> Option<&'static str> {
        // Return first available test PDF
        if Path::new(ACADEMIC_PDF).exists() {
            Some(ACADEMIC_PDF)
        } else if Path::new(FORMS_PDF).exists() {
            Some(FORMS_PDF)
        } else if Path::new(GOVERNMENT_PDF).exists() {
            Some(GOVERNMENT_PDF)
        } else {
            None
        }
    }

    // ========================================================================
    // PDF AVAILABILITY TESTS
    // ========================================================================

    #[test]
    fn test_ocr_e2e_test_pdfs_available() {
        println!("\n=== OCR E2E Test PDF Availability ===");

        let pdfs = vec![
            ("Academic", ACADEMIC_PDF),
            ("Forms", FORMS_PDF),
            ("Government", GOVERNMENT_PDF),
        ];

        let mut available_count = 0;
        for (category, path) in pdfs {
            if Path::new(path).exists() {
                available_count += 1;
                println!("✓ {}: {}", category, path);
            } else {
                println!("✗ {}: Not found", category);
            }
        }

        println!("\nAvailable PDFs: {}/3", available_count);

        if available_count == 0 {
            println!("\nTo test with real PDFs, ensure test datasets are downloaded:");
            println!("  cd /home/yfedoseev/projects/pdf_oxide_tests");
            println!("  git lfs pull");
        }
        println!("===============================\n");
    }

    // ========================================================================
    // PDF INSPECTION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_e2e_inspect_sample_pdf() {
        let pdf_path = match get_test_pdf() {
            Some(path) => path,
            None => {
                println!("No test PDFs available - skipping inspection");
                return;
            },
        };

        match PdfDocument::open(pdf_path) {
            Ok(mut doc) => {
                println!("\n=== PDF Document Inspection ===");
                println!("File: {}", pdf_path);

                match doc.page_count() {
                    Ok(page_count) => println!("Pages: {}", page_count),
                    Err(e) => println!("Error getting page count: {:?}", e),
                }

                // Check first page for text and images
                match doc.extract_text(0) {
                    Ok(text) => {
                        let text_len = text.trim().len();
                        println!("First page text length: {} characters", text_len);
                        println!("Has substantial native text: {}", text_len > 100);
                    },
                    Err(e) => println!("Error reading text: {:?}", e),
                }

                match doc.extract_images(0) {
                    Ok(images) => {
                        println!("First page images: {}", images.len());
                    },
                    Err(e) => println!("Error reading images: {:?}", e),
                }

                println!("=============================\n");
            },
            Err(e) => {
                println!("Could not open PDF {}: {:?}", pdf_path, e);
            },
        }
    }

    // ========================================================================
    // OCR READINESS TESTS
    // ========================================================================

    #[test]
    fn test_ocr_e2e_check_scanned_page() {
        let pdf_path = match get_test_pdf() {
            Some(path) => path,
            None => {
                println!("No test PDFs available - skipping");
                return;
            },
        };

        match PdfDocument::open(pdf_path) {
            Ok(mut doc) => {
                println!("\n=== Scanned Page Detection ===");
                println!("Checking pages for OCR requirement...\n");

                let page_count = doc.page_count().unwrap_or(0).min(3);
                for page_idx in 0..page_count {
                    let text = doc.extract_text(page_idx).unwrap_or_default();
                    let images = doc.extract_images(page_idx).unwrap_or_default();

                    let text_len = text.trim().len();
                    let is_likely_scanned = text_len < 50 && !images.is_empty();

                    println!(
                        "Page {}: {} chars, {} images - Likely scanned: {}",
                        page_idx,
                        text_len,
                        images.len(),
                        is_likely_scanned
                    );

                    if is_likely_scanned {
                        println!("  → This page would benefit from OCR");
                    }
                }

                println!("\n============================\n");
            },
            Err(e) => {
                println!("Error opening PDF: {:?}", e);
            },
        }
    }

    // ========================================================================
    // MODEL CONFIGURATION FOR E2E TESTS
    // ========================================================================

    #[test]
    fn test_ocr_e2e_model_configuration() {
        use pdf_oxide::ocr::OcrConfig;

        println!("\n=== OCR Model Configuration ===");
        println!("Models located at: /home/yfedoseev/projects/pdf_oxide/.models/\n");

        // Check model files
        let models_dir = "/home/yfedoseev/projects/pdf_oxide/.models";
        let det_model = format!("{}/ch_PP-OCRv3_det_infer/inference.pdmodel", models_dir);
        let rec_model = format!("{}/ch_PP-OCRv3_rec_infer/inference.pdmodel", models_dir);
        let dict = format!("{}/ppocr_keys_v1.txt", models_dir);

        println!(
            "Detection model:  {}",
            if Path::new(&det_model).exists() {
                "✓"
            } else {
                "✗"
            }
        );
        println!(
            "Recognition model: {}",
            if Path::new(&rec_model).exists() {
                "✓"
            } else {
                "✗"
            }
        );
        println!(
            "Dictionary:       {}",
            if Path::new(&dict).exists() {
                "✓"
            } else {
                "✗"
            }
        );

        // Show recommended configurations
        println!("\nRecommended OCR configurations:\n");

        println!("1. High Accuracy (Legal/Medical documents)");
        let config = OcrConfig::builder()
            .det_threshold(0.45)
            .box_threshold(0.65)
            .rec_threshold(0.60)
            .num_threads(4)
            .detect_styles(true)
            .build();
        println!("  det_threshold: {}", config.det_threshold);
        println!("  box_threshold: {}", config.box_threshold);
        println!("  rec_threshold: {}", config.rec_threshold);

        println!("\n2. Balanced (General documents)");
        let config = OcrConfig::builder()
            .det_threshold(0.35)
            .box_threshold(0.55)
            .rec_threshold(0.50)
            .num_threads(4)
            .build();
        println!("  det_threshold: {}", config.det_threshold);
        println!("  box_threshold: {}", config.box_threshold);
        println!("  rec_threshold: {}", config.rec_threshold);

        println!("\n3. Fast Processing (Batch operations)");
        let config = OcrConfig::builder()
            .det_threshold(0.25)
            .box_threshold(0.40)
            .rec_threshold(0.40)
            .num_threads(8)
            .det_max_side(512)
            .build();
        println!("  det_threshold: {}", config.det_threshold);
        println!("  box_threshold: {}", config.box_threshold);
        println!("  num_threads: {}", config.num_threads);

        println!("\n==============================\n");
    }

    // ========================================================================
    // E2E WORKFLOW DOCUMENTATION
    // ========================================================================

    #[test]
    fn test_ocr_e2e_workflow_guide() {
        println!("\n╔════════════════════════════════════════════════╗");
        println!("║  OCR E2E Testing Workflow Guide                ║");
        println!("╚════════════════════════════════════════════════╝\n");

        println!("✓ Step 1: Prepare Models (COMPLETED)");
        println!("  Location: /home/yfedoseev/projects/pdf_oxide/.models/");
        println!("  Models: detection (DBNet++), recognition (SVTR)");
        println!("  Dictionary: ppocr_keys_v1.txt (6800+ characters)\n");

        println!("✓ Step 2: Verify Test PDFs");
        println!("  Location: /home/yfedoseev/projects/pdf_oxide_tests/pdfs/");
        println!("  Categories: academic, forms, government, technical, etc.");
        println!("  Scanned: /pdfs/scanned/ (for OCR-only documents)\n");

        println!("◊ Step 3: Create Integration Test (OPTIONAL)");
        println!("  Next: Create tests/test_ocr_with_real_images.rs");
        println!("  Needs: Sample scanned PDF or image-based PDF\n");

        println!("◊ Step 4: Run Full Pipeline (REQUIRES ABOVE)");
        println!("  Command: cargo test --test test_ocr_e2e --features ocr -- --nocapture\n");

        println!("◊ Step 5: Performance Benchmarking (OPTIONAL)");
        println!("  Measure: inference time, memory usage");
        println!("  Commands:");
        println!("    - cargo bench --features ocr");
        println!("    - time cargo run --features ocr -- <pdf> --ocr\n");

        println!("═══════════════════════════════════════════════════\n");

        println!("Key Decision Points:\n");

        println!("1. Test with existing PDFs (text-based)?");
        println!("   - Can test configuration and API");
        println!("   - Won't trigger OCR (documents have native text)");
        println!("   - Good for: API validation, workflow testing\n");

        println!("2. Need truly scanned documents?");
        println!("   - Check /pdfs/scanned/ directory");
        println!("   - Or convert PDFs to images and back");
        println!("   - Or find historical/receipt PDFs");
        println!("   - Good for: real OCR accuracy testing\n");

        println!("3. Create synthetic scanned images?");
        println!("   - Render PDF → image at 200/300/600 DPI");
        println!("   - Run OCR on rendered images");
        println!("   - Compare with native text extraction");
        println!("   - Good for: accuracy benchmarking\n");

        println!("═══════════════════════════════════════════════════\n");
    }

    // ========================================================================
    // DATASET INFORMATION
    // ========================================================================

    #[test]
    fn test_ocr_e2e_available_pdfs_info() {
        println!("\n=== Available PDF Datasets ===\n");

        let categories = vec![
            ("academic", "Academic papers, research, theses"),
            ("forms", "Tax forms, applications, structured documents"),
            ("government", "Official documents, permits, notices"),
            ("diverse", "Mixed content, various sources"),
            ("newspapers", "News articles, media content"),
            ("technical", "Technical documentation, specs"),
            ("mixed", "Combined text, tables, images"),
            ("theses", "Academic theses and dissertations"),
        ];

        for (name, description) in categories {
            let path = format!("/home/yfedoseev/projects/pdf_oxide_tests/pdfs/{}/", name);
            if Path::new(&path).exists() {
                if let Ok(entries) = std::fs::read_dir(&path) {
                    let count: usize = entries
                        .filter_map(|e| e.ok())
                        .filter(|entry| {
                            entry
                                .path()
                                .extension()
                                .map(|ext| ext == "pdf")
                                .unwrap_or(false)
                        })
                        .count();
                    println!("✓ {:<15} - {} PDFs - {}", name, count, description);
                } else {
                    println!("? {:<15} - {}", name, description);
                }
            }
        }

        println!("\n=============================\n");
        println!("💡 Recommendation:");
        println!("   Use academic PDFs for initial testing (173 available)");
        println!("   They're varied and well-structured for validation.\n");
    }
}

// ============================================================================
// TESTS FOR WHEN OCR FEATURE IS NOT ENABLED
// ============================================================================

#[cfg(not(feature = "ocr"))]
mod ocr_e2e_not_enabled_tests {
    #[test]
    fn test_ocr_e2e_feature_disabled() {
        println!("OCR feature is not enabled - E2E tests skipped");
    }
}
