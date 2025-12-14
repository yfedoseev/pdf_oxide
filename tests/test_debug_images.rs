//! Debug test to investigate image extraction issues

#[cfg(feature = "ocr")]
mod debug_tests {
    use pdf_oxide::PdfDocument;

    #[test]
    fn test_debug_image_data() {
        let pdf_path = "scanned_samples/pride_prejudice.pdf";

        match PdfDocument::open(pdf_path) {
            Ok(mut doc) => {
                println!("\n=== Debug Image Extraction ===");

                match doc.extract_images(1) {
                    Ok(images) => {
                        println!("✓ Extracted {} images from page 1", images.len());

                        for (idx, img) in images.iter().enumerate() {
                            println!("\nImage {}:", idx);
                            println!("  Dimensions: {}x{} pixels", img.width(), img.height());
                            println!("  Color space: {:?}", img.color_space());
                            println!("  Bits per component: {}", img.bits_per_component());

                            match img.data() {
                                pdf_oxide::extractors::images::ImageData::Jpeg(jpeg_data) => {
                                    println!("  Format: JPEG-encoded");
                                    println!("  Size: {} bytes", jpeg_data.len());
                                    println!("  ✓ JPEG images can be decoded directly!");
                                },
                                pdf_oxide::extractors::images::ImageData::Raw {
                                    pixels,
                                    format,
                                } => {
                                    println!("  Format: Raw pixels");
                                    println!("  Pixel format: {:?}", format);
                                    println!("  Pixel data size: {} bytes", pixels.len());

                                    // Calculate expected size
                                    let width = img.width();
                                    let height = img.height();
                                    let color_space = img.color_space();

                                    let components = match color_space {
                                        pdf_oxide::extractors::images::ColorSpace::DeviceRGB => 3,
                                        pdf_oxide::extractors::images::ColorSpace::DeviceGray => 1,
                                        pdf_oxide::extractors::images::ColorSpace::DeviceCMYK => 4,
                                        _ => 1,
                                    };

                                    let expected_size =
                                        width as usize * height as usize * components;
                                    println!("  Expected size: {} bytes", expected_size);
                                    println!(
                                        "  Match: {}",
                                        if pixels.len() == expected_size {
                                            "✓ YES"
                                        } else {
                                            "✗ NO"
                                        }
                                    );
                                },
                            }
                        }
                    },
                    Err(e) => println!("✗ Error extracting images: {:?}", e),
                }
            },
            Err(e) => println!("✗ Failed to open PDF: {:?}", e),
        }
    }
}

#[cfg(not(feature = "ocr"))]
mod debug_tests_disabled {
    #[test]
    fn test_debug_disabled() {
        println!("OCR feature not enabled");
    }
}
