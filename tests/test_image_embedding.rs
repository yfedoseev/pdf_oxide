//! Integration tests for image embedding in PDF generation.
//!
//! Tests the full image embedding workflow from raw image data
//! to PDF content stream generation.

#![allow(clippy::same_item_push, clippy::unnecessary_get_then_check)]

use pdf_oxide::writer::{
    ColorSpace, ContentStreamBuilder, ImageData, ImageFormat, ImageManager, ImagePlacement,
    PdfWriter, PdfWriterConfig,
};

// Minimal valid JPEG data (1x1 white pixel)
const MINIMAL_JPEG: &[u8] = &[
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
    0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
    0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
    0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
    0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
    0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
    0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
    0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
    0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94,
    0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2,
    0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
    0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA,
    0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x47, 0xFF,
    0xD9,
];

/// Create a minimal valid RGBA PNG image in memory (alpha channel present)
fn create_test_png_rgba(width: u32, height: u32) -> Vec<u8> {
    use std::io::Write;

    let mut data = Vec::new();
    data.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut raw_pixels = Vec::new();
    for _ in 0..height {
        raw_pixels.push(0); // None-filter byte per row
        for _ in 0..width {
            raw_pixels.extend_from_slice(&[255, 0, 0, 128]); // half-transparent red
        }
    }

    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw_pixels).unwrap();
    let compressed = encoder.finish().unwrap();

    fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], chunk_data: &[u8]) {
        out.extend_from_slice(&(chunk_data.len() as u32).to_be_bytes());
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(chunk_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(chunk_type);
        crc_data.extend_from_slice(chunk_data);
        out.extend_from_slice(&crc32fast::hash(&crc_data).to_be_bytes());
    }

    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(6); // color type RGBA
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    write_chunk(&mut data, b"IHDR", &ihdr_data);
    write_chunk(&mut data, b"IDAT", &compressed);
    write_chunk(&mut data, b"IEND", &[]);

    data
}

/// Create a minimal valid PNG image in memory
fn create_test_png(width: u32, height: u32) -> Vec<u8> {
    use std::io::Write;

    let mut data = Vec::new();

    // PNG signature
    data.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    // Create raw pixel data (RGB)
    let mut raw_pixels = Vec::new();
    for _ in 0..height {
        raw_pixels.push(0); // Filter byte (None)
        for _ in 0..width {
            raw_pixels.extend_from_slice(&[255, 0, 0]); // Red pixel
        }
    }

    // Compress with zlib
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw_pixels).unwrap();
    let compressed = encoder.finish().unwrap();

    // Helper to write a chunk
    fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], chunk_data: &[u8]) {
        // Length (big-endian)
        out.extend_from_slice(&(chunk_data.len() as u32).to_be_bytes());
        // Type
        out.extend_from_slice(chunk_type);
        // Data
        out.extend_from_slice(chunk_data);
        // CRC32 (over type + data)
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(chunk_type);
        crc_data.extend_from_slice(chunk_data);
        let crc = crc32fast::hash(&crc_data);
        out.extend_from_slice(&crc.to_be_bytes());
    }

    // IHDR chunk
    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(2); // color type (RGB)
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    write_chunk(&mut data, b"IHDR", &ihdr_data);

    // IDAT chunk (compressed pixel data)
    write_chunk(&mut data, b"IDAT", &compressed);

    // IEND chunk
    write_chunk(&mut data, b"IEND", &[]);

    data
}

mod image_data_tests {
    use super::*;

    #[test]
    fn test_image_data_creation() {
        let pixels = vec![255u8; 100 * 100 * 3]; // 100x100 RGB
        let image = ImageData::new(100, 100, ColorSpace::DeviceRGB, pixels);

        assert_eq!(image.width, 100);
        assert_eq!(image.height, 100);
        assert_eq!(image.bits_per_component, 8);
        assert_eq!(image.color_space, ColorSpace::DeviceRGB);
        assert_eq!(image.format, ImageFormat::Raw);
    }

    #[test]
    fn test_image_aspect_ratio_landscape() {
        let image = ImageData::new(200, 100, ColorSpace::DeviceRGB, vec![]);
        assert!((image.aspect_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_image_aspect_ratio_portrait() {
        let image = ImageData::new(100, 200, ColorSpace::DeviceRGB, vec![]);
        assert!((image.aspect_ratio() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_image_aspect_ratio_square() {
        let image = ImageData::new(100, 100, ColorSpace::DeviceRGB, vec![]);
        assert!((image.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fit_wide_image_to_square_box() {
        let image = ImageData::new(200, 100, ColorSpace::DeviceRGB, vec![]);
        let (w, h) = image.fit_to_box(100.0, 100.0);

        // Wide image should be constrained by width
        assert!((w - 100.0).abs() < 0.001);
        assert!((h - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_fit_tall_image_to_square_box() {
        let image = ImageData::new(100, 200, ColorSpace::DeviceRGB, vec![]);
        let (w, h) = image.fit_to_box(100.0, 100.0);

        // Tall image should be constrained by height
        assert!((w - 50.0).abs() < 0.001);
        assert!((h - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_xobject_dict_raw() {
        let image = ImageData::new(100, 50, ColorSpace::DeviceGray, vec![0; 5000]);
        let dict = image.build_xobject_dict();

        assert_eq!(dict.get("Type"), Some(&pdf_oxide::object::Object::Name("XObject".to_string())));
        assert_eq!(
            dict.get("Subtype"),
            Some(&pdf_oxide::object::Object::Name("Image".to_string()))
        );
        assert_eq!(dict.get("Width"), Some(&pdf_oxide::object::Object::Integer(100)));
        assert_eq!(dict.get("Height"), Some(&pdf_oxide::object::Object::Integer(50)));
        assert_eq!(
            dict.get("ColorSpace"),
            Some(&pdf_oxide::object::Object::Name("DeviceGray".to_string()))
        );
        // Raw format has no filter
        assert!(dict.get("Filter").is_none());
    }

    #[test]
    fn test_xobject_dict_jpeg() {
        let mut image = ImageData::new(100, 50, ColorSpace::DeviceRGB, vec![0xFF, 0xD8]);
        image.format = ImageFormat::Jpeg;

        let dict = image.build_xobject_dict();
        assert_eq!(
            dict.get("Filter"),
            Some(&pdf_oxide::object::Object::Name("DCTDecode".to_string()))
        );
    }

    #[test]
    fn test_xobject_dict_png() {
        let mut image = ImageData::new(100, 50, ColorSpace::DeviceRGB, vec![0; 100]);
        image.format = ImageFormat::Png;

        let dict = image.build_xobject_dict();
        assert_eq!(
            dict.get("Filter"),
            Some(&pdf_oxide::object::Object::Name("FlateDecode".to_string()))
        );
        // PNG should have decode params with predictor
        assert!(dict.get("DecodeParms").is_some());
    }
}

mod color_space_tests {
    use super::*;

    #[test]
    fn test_grayscale_components() {
        assert_eq!(ColorSpace::DeviceGray.components(), 1);
    }

    #[test]
    fn test_rgb_components() {
        assert_eq!(ColorSpace::DeviceRGB.components(), 3);
    }

    #[test]
    fn test_cmyk_components() {
        assert_eq!(ColorSpace::DeviceCMYK.components(), 4);
    }

    #[test]
    fn test_color_space_pdf_names() {
        assert_eq!(ColorSpace::DeviceGray.pdf_name(), "DeviceGray");
        assert_eq!(ColorSpace::DeviceRGB.pdf_name(), "DeviceRGB");
        assert_eq!(ColorSpace::DeviceCMYK.pdf_name(), "DeviceCMYK");
    }
}

mod image_placement_tests {
    use super::*;

    #[test]
    fn test_placement_creation() {
        let placement = ImagePlacement::new(100.0, 200.0, 50.0, 75.0);

        assert_eq!(placement.x, 100.0);
        assert_eq!(placement.y, 200.0);
        assert_eq!(placement.width, 50.0);
        assert_eq!(placement.height, 75.0);
    }

    #[test]
    fn test_placement_at_origin() {
        let placement = ImagePlacement::at_origin(50.0, 75.0);

        assert_eq!(placement.x, 0.0);
        assert_eq!(placement.y, 0.0);
        assert_eq!(placement.width, 50.0);
        assert_eq!(placement.height, 75.0);
    }

    #[test]
    fn test_transform_matrix() {
        let placement = ImagePlacement::new(100.0, 200.0, 50.0, 75.0);
        let (a, b, c, d, e, f) = placement.transform_matrix();

        // Matrix should be: [width, 0, 0, height, x, y]
        assert!((a - 50.0).abs() < 0.001); // scale x = width
        assert!((b - 0.0).abs() < 0.001);
        assert!((c - 0.0).abs() < 0.001);
        assert!((d - 75.0).abs() < 0.001); // scale y = height
        assert!((e - 100.0).abs() < 0.001); // translate x
        assert!((f - 200.0).abs() < 0.001); // translate y
    }
}

mod image_manager_tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = ImageManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_register_image() {
        let mut manager = ImageManager::new();
        let image = ImageData::new(100, 100, ColorSpace::DeviceRGB, vec![0; 30000]);

        let id = manager.register("test_image", image);

        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
        assert_eq!(id, "Im1");
    }

    #[test]
    fn test_register_multiple_images() {
        let mut manager = ImageManager::new();

        let id1 = manager.register("image1", ImageData::new(10, 10, ColorSpace::DeviceRGB, vec![]));
        let id2 =
            manager.register("image2", ImageData::new(20, 20, ColorSpace::DeviceGray, vec![]));
        let id3 =
            manager.register("image3", ImageData::new(30, 30, ColorSpace::DeviceCMYK, vec![]));

        assert_eq!(manager.len(), 3);
        assert_eq!(id1, "Im1");
        assert_eq!(id2, "Im2");
        assert_eq!(id3, "Im3");
    }

    #[test]
    fn test_get_image() {
        let mut manager = ImageManager::new();
        let image = ImageData::new(50, 75, ColorSpace::DeviceGray, vec![128; 3750]);
        manager.register("grayscale", image);

        let retrieved = manager.get("grayscale").expect("Image should exist");
        assert_eq!(retrieved.width, 50);
        assert_eq!(retrieved.height, 75);
        assert_eq!(retrieved.color_space, ColorSpace::DeviceGray);
    }

    #[test]
    fn test_get_nonexistent_image() {
        let manager = ImageManager::new();
        assert!(manager.get("nonexistent").is_none());
    }

    #[test]
    fn test_resource_id() {
        let mut manager = ImageManager::new();
        let id =
            manager.register("my_image", ImageData::new(10, 10, ColorSpace::DeviceRGB, vec![]));

        assert_eq!(manager.resource_id("my_image"), Some(id.as_str()));
        assert!(manager.resource_id("unknown").is_none());
    }

    #[test]
    fn test_iterate_images() {
        let mut manager = ImageManager::new();
        manager.register("a", ImageData::new(10, 10, ColorSpace::DeviceRGB, vec![]));
        manager.register("b", ImageData::new(20, 20, ColorSpace::DeviceGray, vec![]));

        let images: Vec<_> = manager.images().collect();
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_iterate_images_with_ids() {
        let mut manager = ImageManager::new();
        manager.register("photo", ImageData::new(100, 100, ColorSpace::DeviceRGB, vec![]));

        let items: Vec<_> = manager.images_with_ids().collect();
        assert_eq!(items.len(), 1);

        let (name, id, image) = items[0];
        assert_eq!(name, "photo");
        assert_eq!(id, "Im1");
        assert_eq!(image.width, 100);
    }
}

mod content_stream_tests {
    use super::*;

    #[test]
    fn test_draw_image_generates_correct_operators() {
        let mut builder = ContentStreamBuilder::new();
        builder.draw_image("Im1", 100.0, 200.0, 50.0, 75.0);

        let content = builder.build().expect("Build should succeed");
        let content_str = String::from_utf8_lossy(&content);

        // Should contain save state (q)
        assert!(content_str.contains("q"));
        // Should contain transformation matrix (cm)
        assert!(content_str.contains("cm"));
        // Should contain Do operator for XObject
        assert!(content_str.contains("/Im1 Do"));
        // Should contain restore state (Q)
        assert!(content_str.contains("Q"));
    }

    #[test]
    fn test_draw_image_at_with_placement() {
        let mut builder = ContentStreamBuilder::new();
        let placement = ImagePlacement::new(50.0, 100.0, 200.0, 150.0);
        builder.draw_image_at("Im2", &placement);

        let content = builder.build().expect("Build should succeed");
        let content_str = String::from_utf8_lossy(&content);

        assert!(content_str.contains("/Im2 Do"));
    }

    #[test]
    fn test_multiple_images_in_content_stream() {
        let mut builder = ContentStreamBuilder::new();
        builder.draw_image("Im1", 0.0, 0.0, 100.0, 100.0);
        builder.draw_image("Im2", 150.0, 0.0, 100.0, 100.0);
        builder.draw_image("Im3", 300.0, 0.0, 100.0, 100.0);

        let content = builder.build().expect("Build should succeed");
        let content_str = String::from_utf8_lossy(&content);

        assert!(content_str.contains("/Im1 Do"));
        assert!(content_str.contains("/Im2 Do"));
        assert!(content_str.contains("/Im3 Do"));
    }
}

mod compression_tests {
    use super::*;

    #[test]
    fn test_compression_enabled() {
        let config = PdfWriterConfig::default().with_compress(true);
        let mut writer = PdfWriter::with_config(config);

        {
            let mut page = writer.add_letter_page();
            page.add_text("Test compression", 72.0, 720.0, "Helvetica", 12.0);
            page.finish();
        }

        let bytes = writer.finish().expect("PDF generation should succeed");
        let content = String::from_utf8_lossy(&bytes);

        // Compressed PDF should contain FlateDecode filter
        assert!(content.contains("FlateDecode"));
    }

    #[test]
    fn test_compression_disabled() {
        let config = PdfWriterConfig::default().with_compress(false);
        let mut writer = PdfWriter::with_config(config);

        {
            let mut page = writer.add_letter_page();
            page.add_text("Test no compression", 72.0, 720.0, "Helvetica", 12.0);
            page.finish();
        }

        let bytes = writer.finish().expect("PDF generation should succeed");
        let content = String::from_utf8_lossy(&bytes);

        // Content should be readable (not compressed)
        assert!(content.contains("BT")); // Begin text operator
        assert!(content.contains("ET")); // End text operator
    }

    #[test]
    fn test_compressed_pdf_is_smaller() {
        // Create uncompressed PDF
        let config_uncompressed = PdfWriterConfig::default().with_compress(false);
        let mut writer1 = PdfWriter::with_config(config_uncompressed);
        {
            let mut page = writer1.add_letter_page();
            // Add enough text to make compression worthwhile
            for i in 0..20 {
                page.add_text(
                    &format!("Line {} with some repetitive content that compresses well", i),
                    72.0,
                    720.0 - (i as f32 * 14.0),
                    "Helvetica",
                    12.0,
                );
            }
            page.finish();
        }
        let uncompressed = writer1.finish().expect("PDF should generate");

        // Create compressed PDF
        let config_compressed = PdfWriterConfig::default().with_compress(true);
        let mut writer2 = PdfWriter::with_config(config_compressed);
        {
            let mut page = writer2.add_letter_page();
            for i in 0..20 {
                page.add_text(
                    &format!("Line {} with some repetitive content that compresses well", i),
                    72.0,
                    720.0 - (i as f32 * 14.0),
                    "Helvetica",
                    12.0,
                );
            }
            page.finish();
        }
        let compressed = writer2.finish().expect("PDF should generate");

        // Compressed should be smaller (or at least not significantly larger)
        // Note: For very small content, compression overhead might make it larger
        assert!(
            compressed.len() <= uncompressed.len() + 100,
            "Compressed ({}) should not be much larger than uncompressed ({})",
            compressed.len(),
            uncompressed.len()
        );
    }
}

mod jpeg_parsing_tests {
    use super::*;

    #[test]
    fn test_parse_minimal_jpeg() {
        let image = ImageData::from_jpeg(MINIMAL_JPEG.to_vec());
        assert!(image.is_ok(), "Should parse minimal JPEG: {:?}", image.err());

        let img = image.unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.format, ImageFormat::Jpeg);
        // Grayscale JPEG (1 component)
        assert_eq!(img.color_space, ColorSpace::DeviceGray);
    }

    #[test]
    fn test_invalid_jpeg_magic_bytes() {
        let invalid = vec![0x00, 0x00, 0x00, 0x00];
        let result = ImageData::from_jpeg(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_jpeg() {
        let truncated = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00];
        let result = ImageData::from_jpeg(truncated);
        // Should fail gracefully
        assert!(result.is_err());
    }

    #[test]
    fn test_jpeg_xobject_dict() {
        let image = ImageData::from_jpeg(MINIMAL_JPEG.to_vec()).expect("Should parse");
        let dict = image.build_xobject_dict();

        assert_eq!(
            dict.get("Filter"),
            Some(&pdf_oxide::object::Object::Name("DCTDecode".to_string()))
        );
    }
}

mod png_parsing_tests {
    use super::*;

    #[test]
    fn test_parse_minimal_png() {
        let png_data = create_test_png(1, 1);
        let image = ImageData::from_png(&png_data);
        assert!(image.is_ok(), "Should parse minimal PNG: {:?}", image.err());

        let img = image.unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.format, ImageFormat::Png);
        assert_eq!(img.color_space, ColorSpace::DeviceRGB);
    }

    #[test]
    fn test_invalid_png_magic_bytes() {
        let invalid = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = ImageData::from_png(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_png_data_is_compressed() {
        let png_data = create_test_png(2, 2);
        let image = ImageData::from_png(&png_data).expect("Should parse");
        // PNG data should be compressed with Flate
        assert_eq!(image.format, ImageFormat::Png);
    }

    #[test]
    fn test_parse_larger_png() {
        let png_data = create_test_png(100, 50);
        let image = ImageData::from_png(&png_data).expect("Should parse");

        assert_eq!(image.width, 100);
        assert_eq!(image.height, 50);
        assert_eq!(image.format, ImageFormat::Png);
        assert_eq!(image.color_space, ColorSpace::DeviceRGB);
    }
}

mod format_detection_tests {
    use super::*;

    #[test]
    fn test_detect_jpeg_format() {
        let image = ImageData::from_bytes(MINIMAL_JPEG);
        assert!(image.is_ok());
        assert_eq!(image.unwrap().format, ImageFormat::Jpeg);
    }

    #[test]
    fn test_detect_png_format() {
        let png_data = create_test_png(1, 1);
        let image = ImageData::from_bytes(&png_data);
        assert!(image.is_ok());
        assert_eq!(image.unwrap().format, ImageFormat::Png);
    }

    #[test]
    fn test_unsupported_format() {
        let unknown = vec![0x00, 0x01, 0x02, 0x03];
        let result = ImageData::from_bytes(&unknown);
        assert!(result.is_err());
    }
}

mod image_dedup_tests {
    use pdf_oxide::geometry::Rect;
    use pdf_oxide::writer::DocumentBuilder;

    /// Build a small PDF that places the same PNG bytes twice (issue #443).
    /// With deduplication the two placements share one XObject stream;
    /// without it, the PDF embeds the image data twice and is roughly
    /// `image_size` bytes larger than it needs to be.
    #[test]
    fn test_identical_images_deduplicated() {
        let png = super::create_test_png(50, 50);
        let png_size = png.len();

        let mut builder = DocumentBuilder::new();
        builder
            .a4_page()
            .image_from_bytes(&png, Rect::new(50.0, 450.0, 100.0, 100.0))
            .expect("first image")
            .image_from_bytes(&png, Rect::new(300.0, 450.0, 100.0, 100.0))
            .expect("second image (same bytes)")
            .done();

        let pdf_bytes = builder.build().expect("build");

        // The XObject data should appear only once in the output.
        // A simple proxy: the total PDF size should be less than
        // (2 × png_size + 2.5 KB overhead), meaning the image was not
        // doubled. The 2.5 KB ceiling covers the always-on Standard-14
        // Latin font dicts (twelve faces × ~33 B = ~400 B) plus
        // catalog/pages/xref/trailer structure. A doubled PNG would
        // push the total well over `2 × png_size`, so this still
        // catches the regression the test was written for — see
        // `src/writer/pdf_writer.rs::PdfWriter::finish` for why the
        // Standard-14 set is unconditional.
        let threshold = png_size + 2560; // one copy + Std-14 + generous overhead
        assert!(
            pdf_bytes.len() < threshold,
            "PDF ({} B) looks like it contains two copies of the image ({} B each); \
             expected deduplication to keep it under {} B",
            pdf_bytes.len(),
            png_size,
            threshold,
        );
    }

    /// Different images must still produce separate XObjects.
    #[test]
    fn test_different_images_not_merged() {
        let png_a = super::create_test_png(10, 10);
        let png_b = super::create_test_png(20, 20);
        assert_ne!(png_a, png_b);

        let mut builder = DocumentBuilder::new();
        builder
            .a4_page()
            .image_from_bytes(&png_a, Rect::new(50.0, 450.0, 100.0, 100.0))
            .expect("image A")
            .image_from_bytes(&png_b, Rect::new(300.0, 450.0, 100.0, 100.0))
            .expect("image B")
            .done();

        let pdf_bytes = builder.build().expect("build");
        // Both images together: must be larger than a single image + small overhead
        assert!(
            pdf_bytes.len() > png_a.len() + png_b.len(),
            "PDF should contain both distinct images"
        );
    }
}

/// Regression tests for #450: diagonal-line artifact in images with transparency.
///
/// Root cause: `build_soft_mask_dict()` omitted `DecodeParms`, so viewers
/// mistook the PNG None-filter bytes for alpha pixels, shifting every row by
/// one byte and producing a visible diagonal stripe.
mod soft_mask_decode_parms_tests {
    use pdf_oxide::object::Object;
    use pdf_oxide::writer::ImageData;

    #[test]
    fn test_soft_mask_dict_has_decode_parms() {
        let png = super::create_test_png_rgba(8, 8);
        let image = ImageData::from_png(&png).expect("should parse RGBA PNG");

        assert!(image.soft_mask.is_some(), "RGBA PNG must produce a soft mask");

        let dict = image
            .build_soft_mask_dict()
            .expect("soft_mask is Some so dict must be Some");

        let parms = match dict.get("DecodeParms") {
            Some(Object::Dictionary(d)) => d,
            other => panic!("SMask XObject must have DecodeParms dict, got: {other:?}"),
        };

        assert_eq!(
            parms.get("Predictor"),
            Some(&Object::Integer(15)),
            "Predictor must be 15 (PNG adaptive)"
        );
        assert_eq!(
            parms.get("Colors"),
            Some(&Object::Integer(1)),
            "Colors must be 1 (grayscale alpha)"
        );
        assert_eq!(parms.get("BitsPerComponent"), Some(&Object::Integer(8)),);
        assert_eq!(
            parms.get("Columns"),
            Some(&Object::Integer(8)),
            "Columns must match image width"
        );
    }

    #[test]
    fn test_opaque_png_has_no_soft_mask() {
        let png = super::create_test_png(8, 8);
        let image = ImageData::from_png(&png).expect("should parse RGB PNG");
        assert!(image.soft_mask.is_none(), "RGB PNG without alpha must not produce a soft mask");
        assert!(
            image.build_soft_mask_dict().is_none(),
            "build_soft_mask_dict must return None for opaque images"
        );
    }
}
