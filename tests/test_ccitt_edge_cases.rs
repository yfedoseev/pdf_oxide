#[cfg(feature = "ocr")]
mod ccitt_edge_cases {
    use pdf_oxide::document::PdfDocument;

    #[test]
    fn test_ccitt_decoding_with_leading_zeros() {
        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found: {}", pdf_path);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  CCITT EDGE CASES TEST - Leading Zeros Handling              ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ Failed to open PDF: {}", e);
                return;
            },
        };

        match doc.extract_images(1) {
            Ok(images) => {
                for (idx, image) in images.iter().enumerate() {
                    if let pdf_oxide::extractors::images::ImageData::Raw { pixels, .. } =
                        image.data()
                    {
                        println!("Image {}:", idx);
                        println!("  Total bytes: {}", pixels.len());

                        // Check for leading zeros
                        let leading_zeros = pixels.iter().take_while(|b| **b == 0x00).count();
                        println!("  Leading zero bytes: {}", leading_zeros);

                        if leading_zeros > 0 {
                            println!(
                                "  First {} bytes: {}",
                                leading_zeros,
                                pixels
                                    .iter()
                                    .take(leading_zeros)
                                    .map(|b| format!("{:02x}", b))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            );
                            println!(
                                "  After leading zeros: {}",
                                pixels
                                    .iter()
                                    .skip(leading_zeros)
                                    .take(16)
                                    .map(|b| format!("{:02x}", b))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            );

                            // Try decoding with leading zeros stripped
                            println!("\n  Testing fax decoder with leading zeros stripped:");
                            test_fax_decoder_on_data(
                                &pixels[leading_zeros..],
                                image.width() as u16,
                                Some(image.height() as u16),
                                &format!("Image {} (zeros stripped)", idx),
                            );
                        } else {
                            println!("  No leading zeros detected");
                        }

                        println!();
                    }
                }
            },
            Err(e) => println!("Error: {}", e),
        }
    }

    fn test_fax_decoder_on_data(data: &[u8], width: u16, height: Option<u16>, _label: &str) {
        use fax::decoder;

        let mut output_count = 0;
        let mut transition_count = 0;
        let bytes_iter = data.iter().copied();
        let result = decoder::decode_g4(bytes_iter, width, height, |line| {
            output_count += 1;
            transition_count += line.len();
        });

        match result {
            Some(()) => {
                println!("    ✅ Decoder returned Some(())");
                println!("       Output count: {}", output_count);
                println!("       Total transitions: {}", transition_count);

                if output_count == 0 {
                    println!("       ⚠️  But no output was produced - data may be valid but empty");
                }
            },
            None => println!("    ❌ Decoder returned None"),
        }
    }

    #[test]
    fn test_ccitt_params_mismatch_count() {
        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found: {}", pdf_path);
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  JBIG2/CCITT MISMATCH INVENTORY                             ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("❌ Failed to open PDF: {}", e);
                return;
            },
        };

        match doc.extract_images(1) {
            Ok(images) => {
                let ccitt_override_count = images
                    .iter()
                    .filter(|img| img.bits_per_component() == 1)
                    .count();

                println!("✅ Found {} 1-bit bilevel images", ccitt_override_count);
                println!("   These should all have CCITT parameters extracted");
                println!("   and overrides applied for JBIG2Decode mislabeling\n");

                for img in &images {
                    if img.bits_per_component() == 1 {
                        println!("  Image: {}x{}", img.width(), img.height());
                        // The CCITT params would be stored internally
                        println!("    (CCITT parameters extracted during creation)");
                    }
                }
            },
            Err(e) => println!("Error: {}", e),
        }
    }
}
