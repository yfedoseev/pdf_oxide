//! Tests for page rendering functionality.
//!
//! These tests require the "rendering" feature to be enabled.

#![cfg(feature = "rendering")]

use pdf_oxide::api::{ImageFormat, Pdf, RenderOptions};

/// Helper function to create a simple test PDF.
fn create_test_pdf() -> Vec<u8> {
    let pdf = Pdf::from_text("Test document for rendering.\n\nPage 1 content.")
        .expect("Failed to create PDF");
    pdf.into_bytes()
}

mod render_options {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = RenderOptions::default();
        assert_eq!(opts.dpi, 150);
        assert_eq!(opts.format, ImageFormat::Png);
        assert!(opts.background.is_some());
        assert!(opts.render_annotations);
        assert_eq!(opts.jpeg_quality, 85);
    }

    #[test]
    fn test_with_dpi() {
        let opts = RenderOptions::with_dpi(300);
        assert_eq!(opts.dpi, 300);
        assert_eq!(opts.format, ImageFormat::Png);
    }

    #[test]
    fn test_transparent_background() {
        let opts = RenderOptions::default().with_transparent_background();
        assert!(opts.background.is_none());
    }

    #[test]
    fn test_as_jpeg() {
        let opts = RenderOptions::default().as_jpeg(90);
        assert_eq!(opts.format, ImageFormat::Jpeg);
        assert_eq!(opts.jpeg_quality, 90);
    }

    #[test]
    fn test_jpeg_quality_bounds() {
        // Quality should be clamped to 1-100
        let opts = RenderOptions::default().as_jpeg(0);
        assert_eq!(opts.jpeg_quality, 1);

        let opts = RenderOptions::default().as_jpeg(150);
        assert_eq!(opts.jpeg_quality, 100);
    }
}

mod page_rendering {
    use super::*;

    #[test]
    fn test_render_page_basic() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_input.pdf");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Render page to image
        let image = pdf.render_page(0, None).expect("Failed to render page");

        // Verify image properties
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert!(!image.data.is_empty());
        assert_eq!(image.format, ImageFormat::Png);

        // PNG magic bytes
        assert_eq!(&image.data[0..4], &[0x89, 0x50, 0x4E, 0x47]);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
    }

    #[test]
    fn test_render_page_with_options() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_opts_input.pdf");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Render with custom DPI
        let options = RenderOptions::with_dpi(72);
        let image = pdf
            .render_page_with_options(0, &options)
            .expect("Failed to render page");

        // Lower DPI should produce smaller image
        assert!(image.width > 0);
        assert!(image.height > 0);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
    }

    #[test]
    fn test_render_page_to_file_png() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_file_input.pdf");
        let output_path = temp_dir.join("test_render_output.png");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Render to PNG file
        pdf.render_page_to_file(0, &output_path)
            .expect("Failed to render page to file");

        // Verify file exists and has content
        assert!(output_path.exists());
        let file_bytes = std::fs::read(&output_path).expect("Failed to read output file");
        assert!(!file_bytes.is_empty());
        // PNG magic bytes
        assert_eq!(&file_bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_render_page_to_file_jpeg() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_jpeg_input.pdf");
        let output_path = temp_dir.join("test_render_output.jpg");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Render to JPEG file
        pdf.render_page_to_file(0, &output_path)
            .expect("Failed to render page to file");

        // Verify file exists and has content
        assert!(output_path.exists());
        let file_bytes = std::fs::read(&output_path).expect("Failed to read output file");
        assert!(!file_bytes.is_empty());
        // JPEG magic bytes
        assert_eq!(&file_bytes[0..2], &[0xFF, 0xD8]);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_render_page_with_custom_dpi() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_dpi_input.pdf");
        let output_path = temp_dir.join("test_render_dpi_output.png");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Render with custom DPI
        pdf.render_page_to_file_with_dpi(0, &output_path, 300)
            .expect("Failed to render page to file");

        // Verify file exists
        assert!(output_path.exists());

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_rendered_image_save() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_image_save_input.pdf");
        let output_path = temp_dir.join("test_image_save_output.png");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        let image = pdf.render_page(0, None).expect("Failed to render page");

        // Use RenderedImage::save method
        image.save(&output_path).expect("Failed to save image");

        // Verify file exists
        assert!(output_path.exists());

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_rendered_image_as_bytes() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_image_bytes_input.pdf");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        let image = pdf.render_page(0, None).expect("Failed to render page");

        // Get bytes directly
        let data = image.as_bytes();
        assert!(!data.is_empty());
        // Should match the data field
        assert_eq!(data, &image.data);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
    }
}

mod region_rendering {
    use super::*;
    use pdf_oxide::geometry::Rect;

    #[test]
    fn test_render_region_basic() {
        let mut pdf = Pdf::from_text("Test document for rendering.").expect("Failed to create PDF");

        // Render a region from the top-left quadrant of the page
        let region = Rect::new(0.0, 0.0, 300.0, 400.0);
        let image = pdf
            .render_region(0, &region, None)
            .expect("Failed to render region");

        // Verify image properties
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert!(!image.data.is_empty());
        assert_eq!(image.format, ImageFormat::Png);

        // PNG magic bytes
        assert_eq!(&image.data[0..4], &[0x89, 0x50, 0x4E, 0x47]);

        // Region should be smaller than full page
        let full = pdf.render_page(0, None).expect("Failed to render page");
        assert!(image.width <= full.width);
        assert!(image.height <= full.height);
    }

    #[test]
    fn test_render_region_with_options() {
        let mut pdf = Pdf::from_text("Test document for rendering.").expect("Failed to create PDF");

        // Render region as JPEG with custom DPI
        let options = RenderOptions::with_dpi(300).as_jpeg(90);
        let region = Rect::new(100.0, 100.0, 200.0, 200.0);
        let image = pdf
            .render_region(0, &region, Some(&options))
            .expect("Failed to render region");

        assert_eq!(image.format, ImageFormat::Jpeg);
        assert!(!image.data.is_empty());
        // JPEG magic bytes
        assert_eq!(&image.data[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_render_region_dimensions_scale_with_dpi() {
        let mut pdf = Pdf::from_text("Test document for rendering.").expect("Failed to create PDF");

        let region = Rect::new(0.0, 0.0, 72.0, 72.0); // 1 inch square

        // Render at 72 DPI — should be ~72x72 pixels
        let opts_72 = RenderOptions::with_dpi(72);
        let img_72 = pdf
            .render_region(0, &region, Some(&opts_72))
            .expect("Failed to render at 72 DPI");

        // Render at 144 DPI — should be ~144x144 pixels (2x)
        let opts_144 = RenderOptions::with_dpi(144);
        let img_144 = pdf
            .render_region(0, &region, Some(&opts_144))
            .expect("Failed to render at 144 DPI");

        // Double DPI should roughly double pixel dimensions
        assert!(img_144.width >= img_72.width * 2 - 2);
        assert!(img_144.width <= img_72.width * 2 + 2);
        assert!(img_144.height >= img_72.height * 2 - 2);
        assert!(img_144.height <= img_72.height * 2 + 2);
    }

    #[test]
    fn test_render_region_invalid_page() {
        let mut pdf = Pdf::from_text("Test document for rendering.").expect("Failed to create PDF");

        let region = Rect::new(0.0, 0.0, 100.0, 100.0);
        let result = pdf.render_region(999, &region, None);
        assert!(result.is_err());
    }
}

mod error_handling {
    use super::*;

    #[test]
    fn test_render_invalid_page() {
        let bytes = create_test_pdf();

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_render_invalid_input.pdf");
        std::fs::write(&input_path, &bytes).expect("Failed to write temp PDF");

        let mut pdf = Pdf::open(&input_path).expect("Failed to open PDF");

        // Try to render a page that doesn't exist
        let result = pdf.render_page(999, None);
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
    }
}
