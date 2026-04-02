use pdf_oxide::extractors::images::{
    extract_image_from_xobject, parse_color_space, ColorSpace, ImageData, PdfImage, PixelFormat,
};
use pdf_oxide::object::Object;
use std::collections::HashMap;
use tempfile::TempDir;

#[test]
fn test_create_pdf_image_rgb() {
    // Create a simple RGB image
    let pixels = vec![
        255, 0, 0, // Red pixel
        0, 255, 0, // Green pixel
        0, 0, 255, // Blue pixel
    ];

    let image = PdfImage::new(
        3,
        1,
        ColorSpace::DeviceRGB,
        8,
        ImageData::Raw {
            pixels: pixels.clone(),
            format: PixelFormat::RGB,
        },
    );

    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 1);
    assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);
    assert_eq!(image.bits_per_component(), 8);

    match image.data() {
        ImageData::Raw { pixels: p, format } => {
            assert_eq!(p, &pixels);
            assert_eq!(*format, PixelFormat::RGB);
        },
        _ => panic!("Expected raw RGB data"),
    }
}

#[test]
fn test_create_pdf_image_grayscale() {
    // Create a grayscale image
    let pixels = vec![0, 64, 128, 192, 255];

    let image = PdfImage::new(
        5,
        1,
        ColorSpace::DeviceGray,
        8,
        ImageData::Raw {
            pixels: pixels.clone(),
            format: PixelFormat::Grayscale,
        },
    );

    assert_eq!(image.width(), 5);
    assert_eq!(image.height(), 1);
    assert_eq!(*image.color_space(), ColorSpace::DeviceGray);

    match image.data() {
        ImageData::Raw { pixels: p, format } => {
            assert_eq!(p, &pixels);
            assert_eq!(*format, PixelFormat::Grayscale);
        },
        _ => panic!("Expected raw grayscale data"),
    }
}

#[test]
fn test_create_pdf_image_cmyk() {
    // Create a CMYK image (single pixel)
    let pixels = vec![
        255, 0, 0, 0, // Cyan
    ];

    let image = PdfImage::new(
        1,
        1,
        ColorSpace::DeviceCMYK,
        8,
        ImageData::Raw {
            pixels: pixels.clone(),
            format: PixelFormat::CMYK,
        },
    );

    assert_eq!(*image.color_space(), ColorSpace::DeviceCMYK);

    match image.data() {
        ImageData::Raw { pixels: p, format } => {
            assert_eq!(p, &pixels);
            assert_eq!(*format, PixelFormat::CMYK);
        },
        _ => panic!("Expected raw CMYK data"),
    }
}

#[test]
fn test_save_rgb_as_png() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("rgb_image.png");

    // Create a 2x2 checkerboard pattern (red and blue)
    let pixels = vec![
        255, 0, 0, // Red
        0, 0, 255, // Blue
        0, 0, 255, // Blue
        255, 0, 0, // Red
    ];

    let image = PdfImage::new(
        2,
        2,
        ColorSpace::DeviceRGB,
        8,
        ImageData::Raw {
            pixels,
            format: PixelFormat::RGB,
        },
    );

    let result = image.save_as_png(&output_path);
    assert!(result.is_ok(), "Failed to save PNG: {:?}", result.err());
    assert!(output_path.exists(), "PNG file was not created");

    // Verify file is non-empty
    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0, "PNG file is empty");
}

#[test]
fn test_save_grayscale_as_png() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("gray_image.png");

    // Create a 4x1 grayscale gradient
    let pixels = vec![0, 85, 170, 255];

    let image = PdfImage::new(
        4,
        1,
        ColorSpace::DeviceGray,
        8,
        ImageData::Raw {
            pixels,
            format: PixelFormat::Grayscale,
        },
    );

    let result = image.save_as_png(&output_path);
    assert!(result.is_ok());
    assert!(output_path.exists());

    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_save_cmyk_as_png() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("cmyk_image.png");

    // Create a 1x4 CMYK image (pure cyan, magenta, yellow, black)
    let pixels = vec![
        255, 0, 0, 0, // Cyan
        0, 255, 0, 0, // Magenta
        0, 0, 255, 0, // Yellow
        0, 0, 0, 255, // Black
    ];

    let image = PdfImage::new(
        4,
        1,
        ColorSpace::DeviceCMYK,
        8,
        ImageData::Raw {
            pixels,
            format: PixelFormat::CMYK,
        },
    );

    let result = image.save_as_png(&output_path);
    assert!(result.is_ok());
    assert!(output_path.exists());

    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_save_rgb_as_jpeg() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("rgb_image.jpg");

    // Create a 3x3 green square
    let mut pixels = Vec::new();
    for _ in 0..9 {
        pixels.extend_from_slice(&[0, 255, 0]); // Green
    }

    let image = PdfImage::new(
        3,
        3,
        ColorSpace::DeviceRGB,
        8,
        ImageData::Raw {
            pixels,
            format: PixelFormat::RGB,
        },
    );

    let result = image.save_as_jpeg(&output_path);
    assert!(result.is_ok());
    assert!(output_path.exists());

    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_parse_color_space_all_variants() {
    // Test DeviceRGB
    let rgb = Object::Name("DeviceRGB".to_string());
    assert_eq!(parse_color_space(&rgb).unwrap(), ColorSpace::DeviceRGB);

    // Test DeviceGray
    let gray = Object::Name("DeviceGray".to_string());
    assert_eq!(parse_color_space(&gray).unwrap(), ColorSpace::DeviceGray);

    // Test DeviceCMYK
    let cmyk = Object::Name("DeviceCMYK".to_string());
    assert_eq!(parse_color_space(&cmyk).unwrap(), ColorSpace::DeviceCMYK);

    // Test Indexed (array form)
    let indexed = Object::Array(vec![Object::Name("Indexed".to_string())]);
    assert_eq!(parse_color_space(&indexed).unwrap(), ColorSpace::Indexed);
}

#[test]
fn test_extract_jpeg_image_from_xobject() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(200));
    dict.insert("Height".to_string(), Object::Integer(100));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    dict.insert("Filter".to_string(), Object::Name("DCTDecode".to_string()));

    // Mock JPEG data (just header bytes for testing)
    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(jpeg_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    assert_eq!(image.width(), 200);
    assert_eq!(image.height(), 100);
    assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);
    assert_eq!(image.bits_per_component(), 8);

    // Verify JPEG data is preserved
    match image.data() {
        ImageData::Jpeg(data) => {
            assert_eq!(data, &jpeg_data);
        },
        _ => panic!("Expected JPEG data"),
    }
}

#[test]
fn test_extract_raw_rgb_image_from_xobject() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(2));
    dict.insert("Height".to_string(), Object::Integer(1));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // No filter - raw uncompressed data

    let pixel_data = vec![
        255, 0, 0, // Red pixel
        0, 0, 255, // Blue pixel
    ];
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(pixel_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 1);
    assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);

    match image.data() {
        ImageData::Raw { pixels, format } => {
            assert_eq!(pixels, &pixel_data);
            assert_eq!(*format, PixelFormat::RGB);
        },
        _ => panic!("Expected raw RGB data"),
    }
}

#[test]
fn test_extract_raw_grayscale_image_from_xobject() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(4));
    dict.insert("Height".to_string(), Object::Integer(1));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceGray".to_string()));

    let pixel_data = vec![0, 85, 170, 255];
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(pixel_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    assert_eq!(image.width(), 4);
    assert_eq!(*image.color_space(), ColorSpace::DeviceGray);

    match image.data() {
        ImageData::Raw { pixels, format } => {
            assert_eq!(pixels, &pixel_data);
            assert_eq!(*format, PixelFormat::Grayscale);
        },
        _ => panic!("Expected raw grayscale data"),
    }
}

#[test]
fn test_extract_raw_cmyk_image_from_xobject() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(1));
    dict.insert("Height".to_string(), Object::Integer(1));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceCMYK".to_string()));

    let pixel_data = vec![255, 0, 0, 0]; // Pure cyan
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(pixel_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    assert_eq!(*image.color_space(), ColorSpace::DeviceCMYK);

    match image.data() {
        ImageData::Raw { pixels, format } => {
            assert_eq!(pixels, &pixel_data);
            assert_eq!(*format, PixelFormat::CMYK);
        },
        _ => panic!("Expected raw CMYK data"),
    }
}

#[test]
fn test_extract_image_error_cases() {
    // Test missing Subtype
    {
        let dict = HashMap::new();
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };
        assert!(extract_image_from_xobject(None, &xobject, None, None).is_err());
    }

    // Test wrong Subtype
    {
        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Form".to_string()));
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };
        assert!(extract_image_from_xobject(None, &xobject, None, None).is_err());
    }

    // Test missing Width
    {
        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Height".to_string(), Object::Integer(100));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };
        assert!(extract_image_from_xobject(None, &xobject, None, None).is_err());
    }

    // Test missing Height
    {
        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(100));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };
        assert!(extract_image_from_xobject(None, &xobject, None, None).is_err());
    }

    // Test missing ColorSpace
    {
        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(100));
        dict.insert("Height".to_string(), Object::Integer(100));
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };
        assert!(extract_image_from_xobject(None, &xobject, None, None).is_err());
    }
}

#[test]
fn test_jpeg_filter_array_detection() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(100));
    dict.insert("Height".to_string(), Object::Integer(100));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // Filter as array
    dict.insert("Filter".to_string(), Object::Array(vec![Object::Name("DCTDecode".to_string())]));

    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0];
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(jpeg_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    // Should recognize DCTDecode in array and treat as JPEG
    match image.data() {
        ImageData::Jpeg(data) => {
            assert_eq!(data, &jpeg_data);
        },
        _ => panic!("Expected JPEG data"),
    }
}

#[test]
fn test_bits_per_component_default() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(10));
    dict.insert("Height".to_string(), Object::Integer(10));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // No BitsPerComponent specified - should default to 8

    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(vec![0; 300]), // 10x10 RGB
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();
    assert_eq!(image.bits_per_component(), 8); // Default value
}

#[test]
fn test_color_space_components() {
    assert_eq!(ColorSpace::DeviceGray.components(), 1);
    assert_eq!(ColorSpace::DeviceRGB.components(), 3);
    assert_eq!(ColorSpace::DeviceCMYK.components(), 4);
    assert_eq!(ColorSpace::Indexed.components(), 1);
}

#[test]
fn test_pixel_format_bytes_per_pixel() {
    assert_eq!(PixelFormat::Grayscale.bytes_per_pixel(), 1);
    assert_eq!(PixelFormat::RGB.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::CMYK.bytes_per_pixel(), 4);
}

#[test]
fn test_large_image_dimensions() {
    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(4096));
    dict.insert("Height".to_string(), Object::Integer(2048));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    dict.insert("Filter".to_string(), Object::Name("DCTDecode".to_string()));

    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0];
    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(jpeg_data),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();
    assert_eq!(image.width(), 4096);
    assert_eq!(image.height(), 2048);
}

/// Regression test: DCTDecode with preceding FlateDecode filter.
///
/// Some PDFs use filter chains like [FlateDecode, DCTDecode] where the raw
/// stream data is deflate-compressed JPEG. The extractor must decode the full
/// chain (inflate first, then pass-through DCT) and return valid JPEG data.
///
/// Before this fix, the extractor would return raw deflate-compressed bytes
/// tagged as JPEG, causing `to_dynamic_image()` to fail (e.g., graph_ocred.pdf).
#[test]
fn test_jpeg_with_flatedecode_chain() {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    // Minimal JPEG data (header bytes)
    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];

    // Deflate-compress the JPEG data (simulates FlateDecode)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&jpeg_data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Verify compressed data starts with zlib header, not JPEG magic
    assert_eq!(compressed[0], 0x78, "Expected zlib header byte");
    assert_ne!(compressed[0], 0xFF, "Compressed data should NOT start with FF");

    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(100));
    dict.insert("Height".to_string(), Object::Integer(100));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // Filter chain: FlateDecode first, then DCTDecode
    dict.insert(
        "Filter".to_string(),
        Object::Array(vec![
            Object::Name("FlateDecode".to_string()),
            Object::Name("DCTDecode".to_string()),
        ]),
    );

    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(compressed),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    // Must return JPEG data (not raw pixels)
    match image.data() {
        ImageData::Jpeg(data) => {
            // The decoded data should be the original JPEG bytes
            assert_eq!(data, &jpeg_data, "Decoded JPEG data should match original");
            assert_eq!(data[0], 0xFF, "JPEG data should start with FF");
            assert_eq!(data[1], 0xD8, "JPEG data should have D8 as second byte");
        },
        ImageData::Raw { .. } => {
            panic!("Expected JPEG data, got Raw — DCTDecode filter chain not handled correctly")
        },
    }
}

/// Verify that single DCTDecode still does raw pass-through (no decoding).
#[test]
fn test_jpeg_single_dctdecode_passthrough() {
    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];

    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(50));
    dict.insert("Height".to_string(), Object::Integer(50));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // Single DCTDecode filter — raw pass-through
    dict.insert("Filter".to_string(), Object::Name("DCTDecode".to_string()));

    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(jpeg_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    match image.data() {
        ImageData::Jpeg(data) => {
            // Raw pass-through: data should be identical to stream bytes
            assert_eq!(data, &jpeg_data, "Single DCTDecode should pass through raw bytes");
        },
        _ => panic!("Expected JPEG data for single DCTDecode filter"),
    }
}

/// Verify that DCTDecode in an array (single-element) still does raw pass-through.
#[test]
fn test_jpeg_single_dctdecode_in_array_passthrough() {
    let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0];

    let mut dict = HashMap::new();
    dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
    dict.insert("Width".to_string(), Object::Integer(50));
    dict.insert("Height".to_string(), Object::Integer(50));
    dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
    dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
    // DCTDecode in a single-element array
    dict.insert("Filter".to_string(), Object::Array(vec![Object::Name("DCTDecode".to_string())]));

    let xobject = Object::Stream {
        dict,
        data: bytes::Bytes::from(jpeg_data.clone()),
    };

    let image = extract_image_from_xobject(None, &xobject, None, None).unwrap();

    match image.data() {
        ImageData::Jpeg(data) => {
            assert_eq!(data, &jpeg_data, "Single DCTDecode in array should pass through raw bytes");
        },
        _ => panic!("Expected JPEG data for DCTDecode in single-element array"),
    }
}
