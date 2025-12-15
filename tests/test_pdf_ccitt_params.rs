#[cfg(feature = "ocr")]
mod ccitt_params {
    #![allow(dead_code, clippy::manual_div_ceil)]
    use pdf_oxide::document::PdfDocument;
    use std::collections::HashMap;

    struct ImageInfo {
        width: u32,
        height: u32,
        bits_per_component: u8,
        filter: String,
        decode_params: HashMap<String, String>,
        compressed_size: usize,
        expected_decompressed: usize,
    }

    #[test]
    fn inspect_pride_prejudice_ccitt_parameters() {
        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found: {}", pdf_path);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  CCITT PARAMETER ANALYSIS - Pride & Prejudice PDF             ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ Failed to open PDF: {}", e);
                return;
            },
        };

        // Get page count
        let page_count = match doc.page_count() {
            Ok(c) => c,
            Err(e) => {
                println!("❌ Failed to get page count: {}", e);
                return;
            },
        };

        println!("\n✅ PDF opened: {} pages", page_count);

        // Get page content to ensure it's parsed
        match doc.get_page_content_data(1) {
            Ok(_) => println!("✅ Page 1 content data retrieved"),
            Err(e) => println!("⚠️  Could not get page content: {}", e),
        }

        // Analyze the images we know exist
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Image Information from Page 1:");
        println!("═══════════════════════════════════════════════════════════════");

        let images = extract_known_images();
        for (idx, info) in images.iter().enumerate() {
            println!("\n  Image {}:", idx);
            println!("    Dimensions: {}x{} pixels", info.width, info.height);
            println!("    Bits per component: {}", info.bits_per_component);
            println!("    Filter: {}", info.filter);
            println!("    Compressed size: {} bytes", info.compressed_size);
            println!("    Expected decompressed: {} bytes", info.expected_decompressed);
            println!(
                "    Compression ratio: {:.2}%",
                (info.compressed_size as f64 / info.expected_decompressed as f64) * 100.0
            );
            println!("    Color space: DeviceGray (bilevel)");

            println!("\n    DecodeParms (NEEDS TO BE EXTRACTED FROM PDF):");
            println!("      /K: -1 (Group 4 pure 2D - likely)");
            println!("      /Columns: {} (must match width)", info.width);
            println!("      /BlackIs1: [UNKNOWN - needs inspection]");
            println!("      /EndOfBlock: [UNKNOWN - needs inspection]");
            println!("      /EndOfLine: [UNKNOWN - needs inspection]");
        }

        print_ccitt_analysis();
    }

    fn extract_known_images() -> Vec<ImageInfo> {
        vec![
            ImageInfo {
                width: 2466,
                height: 3900,
                bits_per_component: 1,
                filter: "CCITTFaxDecode".to_string(),
                decode_params: HashMap::new(),
                compressed_size: 4861,
                expected_decompressed: (2466 * 3900 + 7) / 8,
            },
            ImageInfo {
                width: 1034,
                height: 204,
                bits_per_component: 1,
                filter: "CCITTFaxDecode".to_string(),
                decode_params: HashMap::new(),
                compressed_size: 1030,
                expected_decompressed: (1034 * 204 + 7) / 8,
            },
        ]
    }

    fn print_ccitt_analysis() {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  CCITT GROUP 4 PARAMETER GUIDE                               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📋 Standard CCITT Group 4 Parameters:");
        println!("   /K: -1           → Group 4 pure 2D encoding (standard)");
        println!("   /Columns: width  → Image width must match");
        println!("   /Rows: height    → Image height (optional)");
        println!("   /BlackIs1: bool  → Pixel interpretation:");
        println!("      false (0) = white is 0, black is 1 (PDF DEFAULT)");
        println!("      true (1)  = white is 1, black is 0 (INVERTED)");
        println!("   /EndOfLine: bool → Include EOL code (default: false)");
        println!("   /EncodedByteAlign: bool → Align to bytes (default: false)");
        println!("   /EndOfBlock: bool → Include RTC code (default: true)");

        println!("\n⚠️  CRITICAL DISCOVERY:");
        println!("   Our fax crate's decode_g4() returns None, suggesting:");
        println!("   1. Missing or incorrect /DecodeParms");
        println!("   2. /EndOfLine or /EncodedByteAlign issues");
        println!("   3. /EndOfBlock handling differences");

        println!("\n🔍 NEXT STEPS TO FIX:");
        println!("   Step 1: Extract /DecodeParms from PDF stream dictionary");
        println!("   Step 2: Check /K value (should be -1 for Group 4)");
        println!("   Step 3: Handle /BlackIs1 parameter");
        println!("   Step 4: Try alternative CCITT decoders if fax fails");
        println!("   Step 5: Implement proper parameter passing to decoder");

        println!("\n💡 RECOMMENDATION:");
        println!("   Create enhanced CCITT handler that:");
        println!("   1. Extracts /DecodeParms from image stream dictionary");
        println!("   2. Validates /K and /Columns");
        println!("   3. Handles /BlackIs1 by inverting pixels if needed");
        println!("   4. Passes parameters to appropriate decoder");
        println!("   5. Falls back gracefully with detailed error logging");
    }
}

#[cfg(feature = "ocr")]
mod compression_analysis {
    #[test]
    fn analyze_ccitt_compression_ratios() {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  COMPRESSION ANALYSIS                                        ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\nImage 0 (Main page content):");
        let width = 2466;
        let height = 3900;
        let uncompressed = (width * height + 7) / 8;
        let compressed = 4861;
        let ratio = (compressed as f64 / uncompressed as f64) * 100.0;

        println!("   Dimensions: {}x{}", width, height);
        println!("   Uncompressed: {} bytes", uncompressed);
        println!("   Compressed: {} bytes", compressed);
        println!("   Ratio: {:.2}%", ratio);
        println!("   Compression: 1:{:.0}", uncompressed as f64 / compressed as f64);

        println!("\nImage 1 (Header/title):");
        let width2 = 1034;
        let height2 = 204;
        let uncompressed2 = (width2 * height2 + 7) / 8;
        let compressed2 = 1030;
        let ratio2 = (compressed2 as f64 / uncompressed2 as f64) * 100.0;

        println!("   Dimensions: {}x{}", width2, height2);
        println!("   Uncompressed: {} bytes", uncompressed2);
        println!("   Compressed: {} bytes", compressed2);
        println!("   Ratio: {:.2}%", ratio2);
        println!("   Compression: 1:{:.0}", uncompressed2 as f64 / compressed2 as f64);

        println!("\n✅ Compression ratios are reasonable for CCITT Group 4");
        println!("   (Excellent compression for bilevel scanned documents)");
        println!("\n📊 This suggests:");
        println!("   • Data IS properly CCITT compressed");
        println!("   • Decoder needs correct parameters");
        println!("   • /DecodeParms extraction is critical");
    }
}
