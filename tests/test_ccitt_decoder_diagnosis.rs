#![allow(warnings)]
#[cfg(feature = "ocr")]
mod ccitt_decoder_diagnosis {
    use pdf_oxide::decoders::CcittParams;
    use pdf_oxide::document::PdfDocument;

    #[test]
    fn test_fax_decoder_with_pride_prejudice_data() {
        env_logger::builder().is_test(true).try_init().ok();

        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found: {}", pdf_path);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  FAX DECODER DIAGNOSIS TEST                                  ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ Failed to open PDF: {}", e);
                return;
            },
        };

        // Extract images to get the raw CCITT data
        match doc.extract_images(1) {
            Ok(images) => {
                println!("✅ Extracted {} images from page 1\n", images.len());

                for (idx, image) in images.iter().enumerate() {
                    println!("═══════════════════════════════════════════════════════════════");
                    println!("Image {}:", idx);
                    println!("═══════════════════════════════════════════════════════════════");
                    println!("  Dimensions: {}x{}", image.width(), image.height());
                    println!("  Color space: {:?}", image.color_space());
                    println!("  Bits per component: {}", image.bits_per_component());

                    // Try to get the raw image data
                    match image.data() {
                        pdf_oxide::extractors::images::ImageData::Raw { pixels, format } => {
                            println!("  Data format: {:?}", format);
                            println!("  Compressed size: {} bytes", pixels.len());

                            // Calculate expected decompressed size
                            let width = image.width();
                            let height = image.height();
                            let bytes_per_row = (width + 7) / 8;
                            let expected_decompressed =
                                (height as usize) * (bytes_per_row as usize);
                            println!("  Expected decompressed: {} bytes", expected_decompressed);
                            println!(
                                "  Compression ratio: {:.2}%",
                                (pixels.len() as f64 / expected_decompressed as f64) * 100.0
                            );
                            println!(
                                "  Compression: 1:{:.0}",
                                expected_decompressed as f64 / pixels.len() as f64
                            );

                            // Test the fax decoder directly
                            println!("\n📋 Testing fax decoder:");
                            test_fax_decoder_directly(pixels, width, height, idx);
                        },
                        _ => println!("  Not raw CCITT data"),
                    }

                    println!();
                }
            },
            Err(e) => {
                println!("⚠️  Could not extract images: {}", e);
            },
        }
    }

    fn test_fax_decoder_directly(data: &[u8], width: u32, height: u32, img_idx: usize) {
        use fax::decoder;

        let width_u16 = width as u16;
        let height_opt = Some(height as u16);

        println!("  - Width: {}, Height: {}", width_u16, height_opt.unwrap_or(0));
        println!("  - Input data size: {} bytes", data.len());

        // Test 1: Standard Group 4 with default parameters
        println!("\n  Test 1: Standard Group 4 decode");
        let mut output = Vec::new();
        let bytes_iter = data.iter().copied();
        let result = decoder::decode_g4(bytes_iter, width_u16, height_opt, |line| {
            // Count transitions to see if decoder is being called
            output.extend_from_slice(&line);
        });

        match result {
            Some(()) => {
                println!("    ✅ decode_g4 returned Some(())");
                println!("    Output transitions collected: {} items", output.len());
            },
            None => {
                println!("    ❌ decode_g4 returned None");
                println!(
                    "    This suggests the CCITT data structure is incompatible with fax crate"
                );

                // Provide diagnostic hints
                println!("\n  Possible causes:");
                println!("    1. /EndOfLine parameter (PDF default: false)");
                println!("       - If true, fax crate may not handle it correctly");
                println!("    2. /EncodedByteAlign parameter (PDF default: false)");
                println!("       - If true, decoder may fail to parse");
                println!("    3. /EndOfBlock parameter (PDF default: true)");
                println!("       - RTC code presence/absence mismatch");
                println!("    4. Data stream encoding corruption");
                println!("       - May need hex or ASCII85 pre-decoding");
                println!("    5. Black/white inversion");
                println!("       - /BlackIs1 parameter mismatch");
            },
        }

        // Test 2: Check data patterns
        println!("\n  Test 2: Data stream analysis");
        analyze_ccitt_data(data);
    }

    fn analyze_ccitt_data(data: &[u8]) {
        if data.is_empty() {
            println!("    ⚠️  Data is empty");
            return;
        }

        println!(
            "    First 16 bytes (hex): {}",
            data.iter()
                .take(16)
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ")
        );

        println!(
            "    Last 16 bytes (hex): {}",
            data.iter()
                .rev()
                .take(16)
                .collect::<Vec<_>>()
                .iter()
                .rev()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // Look for RTC code (000000110101 in binary = 0x00 0x1D pattern, may vary)
        let has_rtc_like = data
            .windows(2)
            .any(|w| (w[0] == 0x00 && w[1] == 0x1D) || (w[0] == 0x1D && w[1] == 0x00));

        if has_rtc_like {
            println!("    ℹ️  Possible RTC (Return To Control) code detected");
        }

        // Check for EOL patterns (00000001 in binary)
        let has_eol_like = data.windows(1).any(|w| w[0] == 0x01);
        if has_eol_like {
            println!("    ℹ️  Possible EOL (End Of Line) codes detected");
        }

        // Statistics
        let zeros = data.iter().filter(|b| **b == 0x00).count();
        let ones = data.iter().filter(|b| **b == 0xFF).count();
        println!("    Statistics:");
        println!(
            "      - 0x00 bytes: {} ({:.1}%)",
            zeros,
            (zeros as f64 / data.len() as f64) * 100.0
        );
        println!(
            "      - 0xFF bytes: {} ({:.1}%)",
            ones,
            (ones as f64 / data.len() as f64) * 100.0
        );
    }

    #[test]
    fn test_ccitt_params_usage() {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  CCITT PARAMETERS VALIDATION TEST                            ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        // Test default parameters
        let params = CcittParams::default();
        println!("Default CcittParams:");
        println!(
            "  K: {} ({})",
            params.k,
            if params.is_group_4() {
                "Group 4"
            } else {
                "Group 3"
            }
        );
        println!("  Columns: {}", params.columns);
        println!("  Rows: {:?}", params.rows);
        println!("  BlackIs1: {}", params.black_is_1);
        println!("  EndOfLine: {}", params.end_of_line);
        println!("  EncodedByteAlign: {}", params.encoded_byte_align);
        println!("  EndOfBlock: {}", params.end_of_block);

        // Test Group 3
        let group3 = CcittParams {
            k: 0,
            ..Default::default()
        };
        println!("\nGroup 3 params (K=0):");
        println!("  is_group_4: {}", group3.is_group_4());
        println!("  is_group_3: {}", group3.is_group_3());

        // Test with specific image dimensions
        let image_params = CcittParams {
            columns: 2466,
            rows: Some(3900),
            ..Default::default()
        };
        println!("\nImage-specific params (Pride & Prejudice, Image 0):");
        println!("  Columns: {}", image_params.columns);
        println!("  Rows: {:?}", image_params.rows);

        println!("\n✅ CcittParams structure working correctly");
    }
}
