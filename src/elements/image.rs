//! Image content element types.
//!
//! This module provides the `ImageContent` type for representing
//! images in PDFs.

use crate::geometry::Rect;

/// Image content that can be extracted from or written to a PDF.
///
/// This represents an embedded image with its positioning information.
#[derive(Debug, Clone)]
pub struct ImageContent {
    /// Bounding box where the image is placed
    pub bbox: Rect,
    /// Image format
    pub format: ImageFormat,
    /// Raw image data (decoded)
    pub data: Vec<u8>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Bits per component (typically 8)
    pub bits_per_component: u8,
    /// Color space
    pub color_space: ColorSpace,
    /// Reading order index
    pub reading_order: Option<usize>,
    /// Alternative text for accessibility
    pub alt_text: Option<String>,
}

impl ImageContent {
    /// Create a new image content element.
    pub fn new(bbox: Rect, format: ImageFormat, data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            bbox,
            format,
            data,
            width,
            height,
            bits_per_component: 8,
            color_space: ColorSpace::RGB,
            reading_order: None,
            alt_text: None,
        }
    }

    /// Set the reading order.
    pub fn with_reading_order(mut self, order: usize) -> Self {
        self.reading_order = Some(order);
        self
    }

    /// Set alternative text for accessibility.
    pub fn with_alt_text(mut self, text: impl Into<String>) -> Self {
        self.alt_text = Some(text.into());
        self
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Check if this is a grayscale image.
    pub fn is_grayscale(&self) -> bool {
        matches!(self.color_space, ColorSpace::Gray)
    }
}

impl Default for ImageContent {
    fn default() -> Self {
        Self {
            bbox: Rect::new(0.0, 0.0, 0.0, 0.0),
            format: ImageFormat::Unknown,
            data: Vec::new(),
            width: 0,
            height: 0,
            bits_per_component: 8,
            color_space: ColorSpace::RGB,
            reading_order: None,
            alt_text: None,
        }
    }
}

/// Supported image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// JPEG 2000 format (JPX)
    Jpeg2000,
    /// JBIG2 format (typically for scanned documents)
    Jbig2,
    /// Raw uncompressed image data
    Raw,
    /// Unknown or unsupported format
    Unknown,
}

impl ImageFormat {
    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg2000 => "image/jp2",
            ImageFormat::Jbig2 => "image/jbig2",
            ImageFormat::Raw => "application/octet-stream",
            ImageFormat::Unknown => "application/octet-stream",
        }
    }

    /// Get the typical file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
            ImageFormat::Jpeg2000 => "jp2",
            ImageFormat::Jbig2 => "jbig2",
            ImageFormat::Raw => "raw",
            ImageFormat::Unknown => "bin",
        }
    }
}

/// Color space for images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Grayscale (1 component)
    Gray,
    /// RGB color (3 components)
    #[default]
    RGB,
    /// CMYK color (4 components)
    CMYK,
    /// Indexed color (palette-based)
    Indexed,
    /// Lab color space
    Lab,
}

impl ColorSpace {
    /// Get the number of components for this color space.
    pub fn components(&self) -> u8 {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::RGB => 3,
            ColorSpace::CMYK => 4,
            ColorSpace::Indexed => 1,
            ColorSpace::Lab => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_content_creation() {
        let image = ImageContent::new(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            ImageFormat::Jpeg,
            vec![0u8; 1000],
            800,
            600,
        );

        assert_eq!(image.width, 800);
        assert_eq!(image.height, 600);
        assert_eq!(image.format, ImageFormat::Jpeg);
    }

    #[test]
    fn test_aspect_ratio() {
        let image = ImageContent::new(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            ImageFormat::Png,
            vec![],
            1920,
            1080,
        );

        let ratio = image.aspect_ratio();
        assert!((ratio - (1920.0 / 1080.0)).abs() < 0.001);
    }

    #[test]
    fn test_color_space_components() {
        assert_eq!(ColorSpace::Gray.components(), 1);
        assert_eq!(ColorSpace::RGB.components(), 3);
        assert_eq!(ColorSpace::CMYK.components(), 4);
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Jpeg2000.extension(), "jp2");
    }
}
