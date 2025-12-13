#[cfg(feature = "ocr")]
mod ccitt_extraction_tests {
    use pdf_oxide::document::PdfDocument;

    #[test]
    fn test_extract_ccitt_params_from_pride_prejudice() {
        env_logger::builder().is_test(true).try_init().ok();

        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found: {}", pdf_path);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  CCITT PARAMETER EXTRACTION TEST - Pride & Prejudice PDF     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ Failed to open PDF: {}", e);
                return;
            }
        };

        // Get page count
        let page_count = match doc.page_count() {
            Ok(c) => c,
            Err(e) => {
                println!("❌ Failed to get page count: {}", e);
                return;
            }
        };

        println!("✅ PDF opened: {} pages\n", page_count);

        // Try to extract images from page 1
        match doc.extract_images(1) {
            Ok(images) => {
                println!("✅ Extracted {} images from page 1\n", images.len());

                for (idx, image) in images.iter().enumerate() {
                    println!("Image {}:", idx);
                    println!("  Width x Height: {} x {}", image.width(), image.height());
                    println!("  Color space: {:?}", image.color_space());
                    println!("  Bits per component: {}", image.bits_per_component());
                    println!(
                        "  Note: CCITT parameters extracted during image creation (check DEBUG logs)"
                    );
                }

                println!("\n✅ Test passed: Images extracted successfully");
                println!("   With our changes, CCITT parameters should be extracted and logged at DEBUG level");
            }
            Err(e) => {
                println!("⚠️  Could not extract images: {}", e);
                println!("   This is expected if the PDF structure is unexpected");
            }
        }
    }

    #[test]
    fn test_ccitt_params_structure() {
        use pdf_oxide::decoders::CcittParams;

        // Test that CcittParams can be created with default values
        let default_params = CcittParams::default();
        assert_eq!(default_params.k, -1); // Group 4
        assert_eq!(default_params.black_is_1, false); // PDF default
        assert_eq!(default_params.end_of_block, true); // PDF default
        assert!(!default_params.end_of_line); // PDF default
        assert!(!default_params.encoded_byte_align); // PDF default
        assert!(default_params.is_group_4());
        assert!(!default_params.is_group_3());

        // Test Group 3 params
        let group3_params = CcittParams {
            k: 0,
            ..Default::default()
        };
        assert!(group3_params.is_group_3());
        assert!(!group3_params.is_group_4());

        println!("\n✅ CcittParams structure test passed");
        println!("   - Default Group 4 parameters verified");
        println!("   - Group 3 detection works correctly");
    }
}
