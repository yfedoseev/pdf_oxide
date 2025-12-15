//! Image extraction from PDF XObject resources.
//!
//! This module provides functionality to extract images from PDF documents,
//! including JPEG pass-through for DCT-encoded images and raw pixel decoding
//! for other image types.
//!
//! Phase 5

use crate::error::{Error, Result};
use crate::extractors::ccitt_bilevel;
use crate::geometry::Rect;
use std::path::Path;

/// A PDF image with metadata and pixel data.
///
/// Represents an image extracted from a PDF, including dimensions,
/// color space information, and the actual image data (either JPEG
/// or raw pixels).
///
/// # Examples
///
/// ```no_run
/// use pdf_oxide::extractors::images::PdfImage;
/// # fn example(image: PdfImage) -> Result<(), Box<dyn std::error::Error>> {
/// println!("Image size: {}x{}", image.width(), image.height());
/// image.save_as_png("output.png")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PdfImage {
    /// Image width in pixels
    width: u32,
    /// Image height in pixels
    height: u32,
    /// Color space of the image
    color_space: ColorSpace,
    /// Bits per color component (typically 8)
    bits_per_component: u8,
    /// Image data (JPEG or raw pixels)
    data: ImageData,
    /// Optional bounding box in PDF user space
    bbox: Option<Rect>,
    /// CCITT decompression parameters (for 1-bit bilevel images)
    ccitt_params: Option<crate::decoders::CcittParams>,
}

impl PdfImage {
    /// Create a new PDF image.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `color_space` - Color space of the image
    /// * `bits_per_component` - Bits per color component
    /// * `data` - Image data (JPEG or raw pixels)
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::extractors::images::{PdfImage, ColorSpace, ImageData, PixelFormat};
    ///
    /// let image = PdfImage::new(
    ///     100,
    ///     100,
    ///     ColorSpace::DeviceRGB,
    ///     8,
    ///     ImageData::Raw {
    ///         pixels: vec![0; 100 * 100 * 3],
    ///         format: PixelFormat::RGB,
    ///     },
    /// );
    /// ```
    pub fn new(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        bits_per_component: u8,
        data: ImageData,
    ) -> Self {
        Self {
            width,
            height,
            color_space,
            bits_per_component,
            data,
            bbox: None,
            ccitt_params: None,
        }
    }

    /// Create a new PDF image with bounding box.
    pub fn with_bbox(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        bits_per_component: u8,
        data: ImageData,
        bbox: Rect,
    ) -> Self {
        Self {
            width,
            height,
            color_space,
            bits_per_component,
            data,
            bbox: Some(bbox),
            ccitt_params: None,
        }
    }

    /// Create a new PDF image with CCITT parameters.
    pub fn with_ccitt_params(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        bits_per_component: u8,
        data: ImageData,
        ccitt_params: crate::decoders::CcittParams,
    ) -> Self {
        Self {
            width,
            height,
            color_space,
            bits_per_component,
            data,
            bbox: None,
            ccitt_params: Some(ccitt_params),
        }
    }

    /// Get the image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the image color space.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Get bits per component.
    pub fn bits_per_component(&self) -> u8 {
        self.bits_per_component
    }

    /// Get the image data.
    pub fn data(&self) -> &ImageData {
        &self.data
    }

    /// Get the bounding box if available.
    pub fn bbox(&self) -> Option<&Rect> {
        self.bbox.as_ref()
    }

    /// Set CCITT decompression parameters for this image.
    pub fn set_ccitt_params(&mut self, params: crate::decoders::CcittParams) {
        self.ccitt_params = Some(params);
    }

    /// Get CCITT decompression parameters if available.
    pub fn ccitt_params(&self) -> Option<&crate::decoders::CcittParams> {
        self.ccitt_params.as_ref()
    }

    /// Save the image as PNG format.
    ///
    /// For JPEG images, this will decode and re-encode as PNG.
    /// For raw images, this will encode the pixels as PNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be encoded or written to disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::extractors::images::PdfImage;
    /// # fn example(image: PdfImage) -> Result<(), Box<dyn std::error::Error>> {
    /// image.save_as_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_as_png(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                // Decode JPEG and re-encode as PNG
                save_jpeg_as_png(jpeg_data, path)
            },
            ImageData::Raw { pixels, format } => {
                // Encode raw pixels as PNG
                save_raw_as_png(pixels, self.width, self.height, *format, path)
            },
        }
    }

    /// Save the image as JPEG format.
    ///
    /// For images already in JPEG format, this writes the data directly.
    /// For raw images, this encodes the pixels as JPEG.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be encoded or written to disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::extractors::images::PdfImage;
    /// # fn example(image: PdfImage) -> Result<(), Box<dyn std::error::Error>> {
    /// image.save_as_jpeg("output.jpg")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_as_jpeg(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                // Write JPEG data directly
                std::fs::write(path, jpeg_data).map_err(Error::from)
            },
            ImageData::Raw { pixels, format } => {
                // Encode raw pixels as JPEG
                save_raw_as_jpeg(pixels, self.width, self.height, *format, path)
            },
        }
    }

    /// Convert this PDF image to a `DynamicImage` for processing by image crate.
    ///
    /// This enables integration with image processing libraries like OCR engines.
    /// JPEG data is decoded if necessary, and raw pixels are converted to the appropriate format.
    /// Special handling for 1-bit bilevel images (CCITT compressed).
    pub fn to_dynamic_image(&self) -> Result<image::DynamicImage> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                // Decode JPEG data
                image::load_from_memory(jpeg_data)
                    .map_err(|e| Error::Decode(format!("Failed to decode JPEG: {}", e)))
            },
            ImageData::Raw { pixels, format } => {
                // Special handling for 1-bit bilevel images (typically CCITT compressed)
                if self.bits_per_component == 1
                    && matches!(self.color_space, ColorSpace::DeviceGray)
                {
                    // Use CCITT parameters if available, otherwise use defaults
                    let params =
                        self.ccitt_params
                            .clone()
                            .unwrap_or_else(|| crate::decoders::CcittParams {
                                columns: self.width,
                                rows: Some(self.height),
                                ..Default::default()
                            });

                    // Decompress CCITT data using extracted parameters
                    let decompressed = ccitt_bilevel::decompress_ccitt(pixels, &params)?;

                    // Convert 1-bit bilevel to 8-bit grayscale
                    let grayscale =
                        ccitt_bilevel::bilevel_to_grayscale(&decompressed, self.width, self.height);

                    // Create Luma8 image
                    image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(
                        self.width,
                        self.height,
                        grayscale,
                    )
                    .ok_or_else(|| Error::Decode("Invalid image dimensions".to_string()))
                    .map(image::DynamicImage::ImageLuma8)
                } else {
                    // Standard pixel format conversion
                    match (format, self.color_space) {
                        (PixelFormat::RGB, ColorSpace::DeviceRGB) => {
                            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                                self.width,
                                self.height,
                                pixels.clone(),
                            )
                            .ok_or_else(|| Error::Decode("Invalid image dimensions".to_string()))
                            .map(image::DynamicImage::ImageRgb8)
                        },
                        (PixelFormat::Grayscale, ColorSpace::DeviceGray) => {
                            image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(
                                self.width,
                                self.height,
                                pixels.clone(),
                            )
                            .ok_or_else(|| Error::Decode("Invalid image dimensions".to_string()))
                            .map(image::DynamicImage::ImageLuma8)
                        },
                        // For other combinations, convert to RGB
                        _ => {
                            let rgb_pixels = match format {
                                PixelFormat::Grayscale => {
                                    // Expand grayscale to RGB
                                    pixels.iter().flat_map(|&g| vec![g, g, g]).collect()
                                },
                                PixelFormat::CMYK => {
                                    // Convert CMYK to RGB
                                    cmyk_to_rgb(pixels)
                                },
                                PixelFormat::RGB => pixels.clone(),
                            };
                            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                                self.width,
                                self.height,
                                rgb_pixels,
                            )
                            .ok_or_else(|| Error::Decode("Invalid image dimensions".to_string()))
                            .map(image::DynamicImage::ImageRgb8)
                        },
                    }
                }
            },
        }
    }
}

/// Image data representation.
///
/// Images can be either JPEG-encoded (pass-through from PDF) or
/// raw pixel data that needs encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageData {
    /// JPEG-encoded image data (can be saved directly)
    Jpeg(Vec<u8>),
    /// Raw pixel data that needs encoding
    Raw {
        /// Pixel data (decompressed)
        pixels: Vec<u8>,
        /// Pixel format
        format: PixelFormat,
    },
}

/// PDF color space types.
///
/// Represents the color space used by an image in a PDF document.
///
/// PDF Spec: ISO 32000-1:2008, Section 8.6 - Color Spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Device RGB color space (3 components)
    DeviceRGB,
    /// Device Grayscale color space (1 component)
    DeviceGray,
    /// Device CMYK color space (4 components)
    DeviceCMYK,
    /// Indexed color space (palette-based, 1 component)
    Indexed,
    /// Calibrated grayscale color space (1 component)
    /// PDF Spec: Section 8.6.5.2 - CalGray Color Spaces
    CalGray,
    /// Calibrated RGB color space (3 components)
    /// PDF Spec: Section 8.6.5.3 - CalRGB Color Spaces
    CalRGB,
    /// CIE L*a*b* color space (3 components)
    /// PDF Spec: Section 8.6.5.4 - Lab Color Spaces
    Lab,
    /// ICC profile-based color space (1, 3, or 4 components)
    /// PDF Spec: Section 8.6.5.5 - ICCBased Color Spaces
    ///
    /// The usize parameter specifies the number of color components (from /N entry)
    /// Common values: 1 (Gray), 3 (RGB), 4 (CMYK)
    ICCBased(usize),
    /// Separation color space (1 component - spot color)
    /// PDF Spec: Section 8.6.6.4 - Separation Color Spaces
    Separation,
    /// DeviceN color space (N components - multiple colorants)
    /// PDF Spec: Section 8.6.6.5 - DeviceN Color Spaces
    DeviceN,
    /// Pattern color space (tiling or shading patterns)
    /// PDF Spec: Section 8.7 - Patterns
    Pattern,
}

impl ColorSpace {
    /// Get the number of color components for this color space.
    pub fn components(&self) -> usize {
        match self {
            ColorSpace::DeviceGray => 1,
            ColorSpace::DeviceRGB => 3,
            ColorSpace::DeviceCMYK => 4,
            ColorSpace::Indexed => 1, // Index into palette
            ColorSpace::CalGray => 1,
            ColorSpace::CalRGB => 3,
            ColorSpace::Lab => 3,          // L*, a*, b*
            ColorSpace::ICCBased(n) => *n, // Number of components from ICC profile /N entry
            ColorSpace::Separation => 1,   // Spot color tint
            ColorSpace::DeviceN => 4,      // Variable; default to 4 (CMYK-like)
            ColorSpace::Pattern => 0,      // Pattern doesn't have color components directly
        }
    }
}

/// Pixel format for raw image data.
///
/// Represents the arrangement of color components in raw pixel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PixelFormat {
    /// RGB format (3 bytes per pixel: R, G, B)
    RGB,
    /// Grayscale format (1 byte per pixel)
    Grayscale,
    /// CMYK format (4 bytes per pixel: C, M, Y, K)
    CMYK,
}

impl PixelFormat {
    /// Get the number of bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Grayscale => 1,
            PixelFormat::RGB => 3,
            PixelFormat::CMYK => 4,
        }
    }
}

/// Convert a ColorSpace to a PixelFormat.
///
/// This is used when decoding raw image data to determine the pixel format.
fn color_space_to_pixel_format(color_space: &ColorSpace) -> PixelFormat {
    match color_space {
        ColorSpace::DeviceGray => PixelFormat::Grayscale,
        ColorSpace::DeviceRGB => PixelFormat::RGB,
        ColorSpace::DeviceCMYK => PixelFormat::CMYK,
        ColorSpace::Indexed => PixelFormat::RGB, // Indexed images are converted to RGB
        // Advanced color spaces - map to appropriate pixel formats
        ColorSpace::CalGray => PixelFormat::Grayscale, // Calibrated grayscale
        ColorSpace::CalRGB => PixelFormat::RGB,        // Calibrated RGB
        ColorSpace::Lab => PixelFormat::RGB,           // CIE L*a*b* - convert to RGB
        ColorSpace::ICCBased(n) => match n {
            // ICC profile-based - map by component count
            1 => PixelFormat::Grayscale, // 1 component: Grayscale
            3 => PixelFormat::RGB,       // 3 components: RGB
            4 => PixelFormat::CMYK,      // 4 components: CMYK
            _ => PixelFormat::RGB,       // Fallback to RGB
        },
        ColorSpace::Separation => PixelFormat::Grayscale, // Spot color - single component
        ColorSpace::DeviceN => PixelFormat::CMYK,         // Multiple colorants - treat as CMYK
        ColorSpace::Pattern => PixelFormat::RGB,          // Pattern - rasterize to RGB
    }
}

/// Parse a ColorSpace name from a PDF object.
///
/// Handles both direct name objects (e.g., /DeviceRGB) and
/// array-based color spaces (e.g., [/Indexed ...]).
///
/// # Arguments
///
/// * `obj` - The color space object from the image XObject dictionary
///
/// # Returns
///
/// The parsed ColorSpace, or an error if the color space is unsupported.
pub fn parse_color_space(obj: &crate::object::Object) -> Result<ColorSpace> {
    use crate::object::Object;

    match obj {
        Object::Name(name) => match name.as_str() {
            "DeviceRGB" => Ok(ColorSpace::DeviceRGB),
            "DeviceGray" => Ok(ColorSpace::DeviceGray),
            "DeviceCMYK" => Ok(ColorSpace::DeviceCMYK),
            "Pattern" => Ok(ColorSpace::Pattern),
            other => Err(Error::Image(format!("Unsupported color space: {}", other))),
        },
        Object::Array(arr) if !arr.is_empty() => {
            // Array-based color space (e.g., [/Indexed ..., /CalRGB ..., /ICCBased ...])
            if let Some(name) = arr[0].as_name() {
                match name {
                    "Indexed" => Ok(ColorSpace::Indexed),
                    "CalGray" => Ok(ColorSpace::CalGray),
                    "CalRGB" => Ok(ColorSpace::CalRGB),
                    "Lab" => Ok(ColorSpace::Lab),
                    "ICCBased" => {
                        // ICCBased format: [/ICCBased stream]
                        // The stream dictionary contains /N (number of components)
                        let num_components = if arr.len() > 1 {
                            // Try to extract /N from the stream dictionary
                            if let Some(stream_dict) = arr[1].as_dict() {
                                stream_dict
                                    .get("N")
                                    .and_then(|obj| match obj {
                                        Object::Integer(n) => Some(*n as usize),
                                        _ => None,
                                    })
                                    .unwrap_or_else(|| {
                                        log::debug!("ICCBased stream missing /N entry, defaulting to 3 components");
                                        3
                                    })
                            } else {
                                log::debug!(
                                    "ICCBased array doesn't contain stream dictionary, defaulting to 3 components"
                                );
                                3
                            }
                        } else {
                            log::debug!("ICCBased array too short, defaulting to 3 components");
                            3
                        };
                        log::debug!("ICCBased color space with {} components", num_components);
                        Ok(ColorSpace::ICCBased(num_components))
                    },
                    "Separation" => Ok(ColorSpace::Separation),
                    "DeviceN" => Ok(ColorSpace::DeviceN),
                    "Pattern" => Ok(ColorSpace::Pattern),
                    other => {
                        // Log unsupported color spaces for debugging
                        log::debug!("Unsupported array color space: {}", other);
                        Err(Error::Image(format!("Unsupported array color space: {}", other)))
                    },
                }
            } else {
                Err(Error::Image("Color space array must start with a name".to_string()))
            }
        },
        _ => Err(Error::Image(format!("Invalid color space object: {:?}", obj))),
    }
}

/// Extract an image from an XObject stream.
///
/// This function handles both JPEG-encoded images (DCTDecode filter)
/// and raw pixel images (other filters or no filter).
///
/// For encrypted PDFs, provide the document reference and object reference
/// to enable proper stream decryption before decompression.
///
/// # Arguments
///
/// * `doc` - Optional reference to the PdfDocument (needed for encrypted PDFs)
/// * `xobject` - The XObject stream object
/// * `obj_ref` - Optional object reference (needed for encrypted PDFs)
///
/// # Returns
///
/// A PdfImage with the extracted image data, or an error if extraction fails.
///
/// # Examples
///
/// ```no_run
/// # use pdf_oxide::extractors::images::extract_image_from_xobject;
/// # use pdf_oxide::object::Object;
/// # fn example(xobj: Object) -> Result<(), Box<dyn std::error::Error>> {
/// let image = extract_image_from_xobject(None, &xobj, None)?;
/// println!("Extracted {}x{} image", image.width(), image.height());
/// # Ok(())
/// # }
/// ```
pub fn extract_image_from_xobject(
    doc: Option<&crate::document::PdfDocument>,
    xobject: &crate::object::Object,
    obj_ref: Option<crate::object::ObjectRef>,
) -> Result<PdfImage> {
    use crate::object::Object;

    // XObject must be a stream
    let dict = xobject
        .as_dict()
        .ok_or_else(|| Error::Image("XObject is not a stream".to_string()))?;

    // Verify it's an Image XObject
    let subtype = dict
        .get("Subtype")
        .and_then(|obj| obj.as_name())
        .ok_or_else(|| Error::Image("XObject missing /Subtype".to_string()))?;

    if subtype != "Image" {
        return Err(Error::Image(format!("XObject subtype is not Image: {}", subtype)));
    }

    // Extract image dimensions
    let width = dict
        .get("Width")
        .and_then(|obj| obj.as_integer())
        .ok_or_else(|| Error::Image("Image missing /Width".to_string()))? as u32;

    let height = dict
        .get("Height")
        .and_then(|obj| obj.as_integer())
        .ok_or_else(|| Error::Image("Image missing /Height".to_string()))? as u32;

    // Extract bits per component (default: 8)
    let bits_per_component = dict
        .get("BitsPerComponent")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(8) as u8;

    // Extract color space
    let color_space_obj = dict
        .get("ColorSpace")
        .ok_or_else(|| Error::Image("Image missing /ColorSpace".to_string()))?;
    let color_space = parse_color_space(color_space_obj)?;

    // Check if this is a JPEG image (DCTDecode filter)
    let filter_names = if let Some(filter_obj) = dict.get("Filter") {
        match filter_obj {
            Object::Name(name) => vec![name.clone()],
            Object::Array(filters) => filters
                .iter()
                .filter_map(|f| f.as_name().map(String::from))
                .collect(),
            _ => vec![],
        }
    } else {
        vec![]
    };

    log::debug!("Image filters detected: {:?}", filter_names);

    let is_jpeg = filter_names.iter().any(|name| name == "DCTDecode");

    // Check for CCITT parameter mismatch (incorrectly labeled as JBIG2Decode)
    let mut ccitt_params_override: Option<crate::decoders::CcittParams> = None;
    if (filter_names.contains(&"JBIG2Decode".to_string())
        || filter_names.contains(&"Jbig2Decode".to_string()))
        && bits_per_component == 1
    {
        // Check if DecodeParms looks like CCITT parameters
        let mut ccitt_params =
            crate::object::extract_ccitt_params_with_width(dict.get("DecodeParms"), Some(width));

        // If we extracted CCITT parameters but rows is missing, use image height
        if let Some(ref mut params) = ccitt_params {
            if params.rows.is_none() {
                params.rows = Some(height);
                log::debug!(
                    "Added image height {} to CCITT parameters (was missing from /DecodeParms)",
                    height
                );
            }
        }

        if let Some(ref params) = ccitt_params {
            log::warn!(
                "PDF incorrectly labeled 1-bit image with JBIG2Decode filter but has CCITT parameters (K={})",
                params.k
            );
            ccitt_params_override = ccitt_params;
        }
    }

    // Extract image data
    let data = if is_jpeg {
        // JPEG pass-through - extract raw stream data without decoding
        match xobject {
            Object::Stream { data, .. } => ImageData::Jpeg(data.to_vec()),
            _ => return Err(Error::Image("XObject is not a stream".to_string())),
        }
    } else if ccitt_params_override.is_some() {
        // Special handling: If we detected CCITT parameters override, extract the raw stream
        // without applying the (incorrect) JBIG2Decode filter
        match xobject {
            Object::Stream { data, .. } => {
                log::debug!("Using raw CCITT data (skipping incorrect JBIG2Decode filter)");
                ImageData::Raw {
                    pixels: data.to_vec(),
                    format: PixelFormat::Grayscale,
                }
            },
            _ => return Err(Error::Image("XObject is not a stream".to_string())),
        }
    } else {
        // Decode stream data normally
        let decoded_data = if let (Some(doc), Some(ref_id)) = (doc, obj_ref) {
            doc.decode_stream_with_encryption(xobject, ref_id)?
        } else {
            xobject.decode_stream_data()?
        };
        let pixel_format = color_space_to_pixel_format(&color_space);
        ImageData::Raw {
            pixels: decoded_data,
            format: pixel_format,
        }
    };

    // Extract CCITT parameters if this is a 1-bit bilevel image
    let mut image = PdfImage::new(width, height, color_space, bits_per_component, data);

    // Use override parameters if we detected a mismatch
    if let Some(ccitt_params) = ccitt_params_override {
        log::debug!(
            "Using CCITT override parameters: K={}, BlackIs1={}, EndOfLine={}, EncodedByteAlign={}, EndOfBlock={}",
            ccitt_params.k,
            ccitt_params.black_is_1,
            ccitt_params.end_of_line,
            ccitt_params.encoded_byte_align,
            ccitt_params.end_of_block,
        );
        image.set_ccitt_params(ccitt_params);
    } else if bits_per_component == 1 && image.color_space == ColorSpace::DeviceGray {
        // Try to extract CCITT decompression parameters normally
        if let Some(mut ccitt_params) =
            crate::object::extract_ccitt_params_with_width(dict.get("DecodeParms"), Some(width))
        {
            // If rows is missing from /DecodeParms, use image height
            if ccitt_params.rows.is_none() {
                ccitt_params.rows = Some(height);
                log::debug!(
                    "Added image height {} to CCITT parameters (was missing from /DecodeParms)",
                    height
                );
            }

            log::debug!(
                "Extracted CCITT parameters: K={}, BlackIs1={}, EndOfLine={}, EncodedByteAlign={}, EndOfBlock={}, columns={}, rows={:?}",
                ccitt_params.k,
                ccitt_params.black_is_1,
                ccitt_params.end_of_line,
                ccitt_params.encoded_byte_align,
                ccitt_params.end_of_block,
                ccitt_params.columns,
                ccitt_params.rows,
            );
            image.set_ccitt_params(ccitt_params);
        }
    }

    Ok(image)
}

/// Convert CMYK pixel values to RGB.
///
/// Uses the standard CMYK to RGB conversion formula:
/// - R = (1 - C) * (1 - K) * 255
/// - G = (1 - M) * (1 - K) * 255
/// - B = (1 - Y) * (1 - K) * 255
///
/// # Arguments
///
/// * `cmyk` - CMYK pixel data (4 bytes per pixel, values 0-255)
///
/// # Returns
///
/// RGB pixel data (3 bytes per pixel, values 0-255)
///
/// # Examples
///
/// ```
/// use pdf_oxide::extractors::images::cmyk_to_rgb;
///
/// let cmyk = vec![0, 255, 255, 0]; // Cyan=0, Magenta=255, Yellow=255, Key=0
/// let rgb = cmyk_to_rgb(&cmyk);
/// assert_eq!(rgb.len(), 3);
/// ```
pub fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((cmyk.len() / 4) * 3);

    for chunk in cmyk.chunks_exact(4) {
        let c = chunk[0] as f32 / 255.0;
        let m = chunk[1] as f32 / 255.0;
        let y = chunk[2] as f32 / 255.0;
        let k = chunk[3] as f32 / 255.0;

        let r = ((1.0 - c) * (1.0 - k) * 255.0) as u8;
        let g = ((1.0 - m) * (1.0 - k) * 255.0) as u8;
        let b = ((1.0 - y) * (1.0 - k) * 255.0) as u8;

        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }

    rgb
}

/// Save JPEG data as PNG by decoding and re-encoding.
fn save_jpeg_as_png(jpeg_data: &[u8], path: impl AsRef<Path>) -> Result<()> {
    use image::ImageFormat;

    // Decode JPEG
    let img = image::load_from_memory_with_format(jpeg_data, ImageFormat::Jpeg)
        .map_err(|e| Error::Image(format!("Failed to decode JPEG: {}", e)))?;

    // Save as PNG
    img.save_with_format(path, ImageFormat::Png)
        .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
}

/// Save raw pixel data as PNG.
fn save_raw_as_png(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    path: impl AsRef<Path>,
) -> Result<()> {
    use image::{ImageBuffer, ImageFormat, Luma, Rgb};

    match format {
        PixelFormat::RGB => {
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels.to_vec())
                .ok_or_else(|| Error::Image("Invalid RGB image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Png)
                .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
        },
        PixelFormat::Grayscale => {
            let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels.to_vec())
                .ok_or_else(|| Error::Image("Invalid grayscale image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Png)
                .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
        },
        PixelFormat::CMYK => {
            // Convert CMYK to RGB first
            let rgb = cmyk_to_rgb(pixels);
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Png)
                .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
        },
    }
}

/// Save raw pixel data as JPEG.
fn save_raw_as_jpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    path: impl AsRef<Path>,
) -> Result<()> {
    use image::{ImageBuffer, ImageFormat, Luma, Rgb};

    match format {
        PixelFormat::RGB => {
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels.to_vec())
                .ok_or_else(|| Error::Image("Invalid RGB image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
        PixelFormat::Grayscale => {
            let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels.to_vec())
                .ok_or_else(|| Error::Image("Invalid grayscale image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
        PixelFormat::CMYK => {
            // Convert CMYK to RGB first
            let rgb = cmyk_to_rgb(pixels);
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;

            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
    fn test_color_space_to_pixel_format() {
        assert_eq!(color_space_to_pixel_format(&ColorSpace::DeviceGray), PixelFormat::Grayscale);
        assert_eq!(color_space_to_pixel_format(&ColorSpace::DeviceRGB), PixelFormat::RGB);
        assert_eq!(color_space_to_pixel_format(&ColorSpace::DeviceCMYK), PixelFormat::CMYK);
        assert_eq!(color_space_to_pixel_format(&ColorSpace::Indexed), PixelFormat::RGB);
    }

    #[test]
    fn test_cmyk_to_rgb_pure_cyan() {
        // Pure cyan: C=255, M=0, Y=0, K=0
        let cmyk = vec![255, 0, 0, 0];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 3);
        assert_eq!(rgb[0], 0); // R = 0
        assert_eq!(rgb[1], 255); // G = 255
        assert_eq!(rgb[2], 255); // B = 255
    }

    #[test]
    fn test_cmyk_to_rgb_pure_magenta() {
        // Pure magenta: C=0, M=255, Y=0, K=0
        let cmyk = vec![0, 255, 0, 0];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 3);
        assert_eq!(rgb[0], 255); // R = 255
        assert_eq!(rgb[1], 0); // G = 0
        assert_eq!(rgb[2], 255); // B = 255
    }

    #[test]
    fn test_cmyk_to_rgb_pure_yellow() {
        // Pure yellow: C=0, M=0, Y=255, K=0
        let cmyk = vec![0, 0, 255, 0];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 3);
        assert_eq!(rgb[0], 255); // R = 255
        assert_eq!(rgb[1], 255); // G = 255
        assert_eq!(rgb[2], 0); // B = 0
    }

    #[test]
    fn test_cmyk_to_rgb_black() {
        // Black: C=0, M=0, Y=0, K=255
        let cmyk = vec![0, 0, 0, 255];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 3);
        assert_eq!(rgb[0], 0); // R = 0
        assert_eq!(rgb[1], 0); // G = 0
        assert_eq!(rgb[2], 0); // B = 0
    }

    #[test]
    fn test_cmyk_to_rgb_white() {
        // White: C=0, M=0, Y=0, K=0
        let cmyk = vec![0, 0, 0, 0];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 3);
        assert_eq!(rgb[0], 255); // R = 255
        assert_eq!(rgb[1], 255); // G = 255
        assert_eq!(rgb[2], 255); // B = 255
    }

    #[test]
    fn test_cmyk_to_rgb_multiple_pixels() {
        // Two pixels: cyan and magenta
        let cmyk = vec![255, 0, 0, 0, 0, 255, 0, 0];
        let rgb = cmyk_to_rgb(&cmyk);
        assert_eq!(rgb.len(), 6);
        // First pixel (cyan)
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 255);
        assert_eq!(rgb[2], 255);
        // Second pixel (magenta)
        assert_eq!(rgb[3], 255);
        assert_eq!(rgb[4], 0);
        assert_eq!(rgb[5], 255);
    }

    #[test]
    fn test_pdf_image_new() {
        let image = PdfImage::new(
            100,
            200,
            ColorSpace::DeviceRGB,
            8,
            ImageData::Raw {
                pixels: vec![0; 100 * 200 * 3],
                format: PixelFormat::RGB,
            },
        );

        assert_eq!(image.width(), 100);
        assert_eq!(image.height(), 200);
        assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);
        assert_eq!(image.bits_per_component(), 8);
        assert!(image.bbox().is_none());
    }

    #[test]
    fn test_pdf_image_with_bbox() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 200.0);
        let image = PdfImage::with_bbox(
            100,
            200,
            ColorSpace::DeviceRGB,
            8,
            ImageData::Raw {
                pixels: vec![0; 100 * 200 * 3],
                format: PixelFormat::RGB,
            },
            bbox,
        );

        assert!(image.bbox().is_some());
        assert_eq!(*image.bbox().unwrap(), bbox);
    }

    #[test]
    fn test_pdf_image_jpeg_data() {
        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG header
        let image =
            PdfImage::new(100, 200, ColorSpace::DeviceRGB, 8, ImageData::Jpeg(jpeg_data.clone()));

        match image.data() {
            ImageData::Jpeg(data) => assert_eq!(data, &jpeg_data),
            _ => panic!("Expected JPEG data"),
        }
    }

    #[test]
    fn test_save_raw_rgb_as_png() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.png");

        // Create a 2x2 red image
        let pixels = vec![
            255, 0, 0, // Pixel 1: Red
            255, 0, 0, // Pixel 2: Red
            255, 0, 0, // Pixel 3: Red
            255, 0, 0, // Pixel 4: Red
        ];

        let result = save_raw_as_png(&pixels, 2, 2, PixelFormat::RGB, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_save_raw_grayscale_as_png() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_gray.png");

        // Create a 2x2 grayscale image
        let pixels = vec![0, 128, 192, 255];

        let result = save_raw_as_png(&pixels, 2, 2, PixelFormat::Grayscale, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_save_raw_cmyk_as_png() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_cmyk.png");

        // Create a 1x1 CMYK image (cyan)
        let pixels = vec![255, 0, 0, 0];

        let result = save_raw_as_png(&pixels, 1, 1, PixelFormat::CMYK, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_save_raw_rgb_as_jpeg() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.jpg");

        // Create a 2x2 blue image
        let pixels = vec![
            0, 0, 255, // Pixel 1: Blue
            0, 0, 255, // Pixel 2: Blue
            0, 0, 255, // Pixel 3: Blue
            0, 0, 255, // Pixel 4: Blue
        ];

        let result = save_raw_as_jpeg(&pixels, 2, 2, PixelFormat::RGB, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_pdf_image_save_raw_as_png() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("image.png");

        // Create a 2x2 green image
        let pixels = vec![
            0, 255, 0, // Green
            0, 255, 0, // Green
            0, 255, 0, // Green
            0, 255, 0, // Green
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
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_pdf_image_save_raw_as_jpeg() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("image.jpg");

        // Create a 2x2 red image
        let pixels = vec![
            255, 0, 0, // Red
            255, 0, 0, // Red
            255, 0, 0, // Red
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

        let result = image.save_as_jpeg(&output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_image_data_clone() {
        let data = ImageData::Jpeg(vec![1, 2, 3]);
        let cloned = data.clone();
        assert_eq!(data, cloned);
    }

    #[test]
    fn test_color_space_clone() {
        let cs = ColorSpace::DeviceRGB;
        let cloned = cs;
        assert_eq!(cs, cloned);
    }

    #[test]
    fn test_parse_color_space_device_rgb() {
        use crate::object::Object;
        let obj = Object::Name("DeviceRGB".to_string());
        let cs = parse_color_space(&obj).unwrap();
        assert_eq!(cs, ColorSpace::DeviceRGB);
    }

    #[test]
    fn test_parse_color_space_device_gray() {
        use crate::object::Object;
        let obj = Object::Name("DeviceGray".to_string());
        let cs = parse_color_space(&obj).unwrap();
        assert_eq!(cs, ColorSpace::DeviceGray);
    }

    #[test]
    fn test_parse_color_space_device_cmyk() {
        use crate::object::Object;
        let obj = Object::Name("DeviceCMYK".to_string());
        let cs = parse_color_space(&obj).unwrap();
        assert_eq!(cs, ColorSpace::DeviceCMYK);
    }

    #[test]
    fn test_parse_color_space_indexed() {
        use crate::object::Object;
        let obj = Object::Array(vec![Object::Name("Indexed".to_string())]);
        let cs = parse_color_space(&obj).unwrap();
        assert_eq!(cs, ColorSpace::Indexed);
    }

    #[test]
    fn test_parse_color_space_unsupported() {
        use crate::object::Object;
        let obj = Object::Name("UnsupportedColorSpace".to_string());
        let result = parse_color_space(&obj);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_color_space_invalid() {
        use crate::object::Object;
        let obj = Object::Integer(42);
        let result = parse_color_space(&obj);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_image_from_xobject_jpeg() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(100));
        dict.insert("Height".to_string(), Object::Integer(200));
        dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
        dict.insert("Filter".to_string(), Object::Name("DCTDecode".to_string()));

        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG header
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(jpeg_data.clone()),
        };

        let image = extract_image_from_xobject(None, &xobject, None).unwrap();
        assert_eq!(image.width(), 100);
        assert_eq!(image.height(), 200);
        assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);
        assert_eq!(image.bits_per_component(), 8);

        match image.data() {
            ImageData::Jpeg(data) => assert_eq!(data, &jpeg_data),
            _ => panic!("Expected JPEG data"),
        }
    }

    #[test]
    fn test_extract_image_from_xobject_raw() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(2));
        dict.insert("Height".to_string(), Object::Integer(2));
        dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
        // No filter - raw data

        let raw_data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]; // 4 pixels RGB
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(raw_data.clone()),
        };

        let image = extract_image_from_xobject(None, &xobject, None).unwrap();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(*image.color_space(), ColorSpace::DeviceRGB);

        match image.data() {
            ImageData::Raw { pixels, format } => {
                assert_eq!(pixels, &raw_data);
                assert_eq!(*format, PixelFormat::RGB);
            },
            _ => panic!("Expected raw data"),
        }
    }

    #[test]
    fn test_extract_image_from_xobject_grayscale() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(2));
        dict.insert("Height".to_string(), Object::Integer(2));
        dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceGray".to_string()));

        let raw_data = vec![0, 128, 192, 255]; // 4 grayscale pixels
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(raw_data.clone()),
        };

        let image = extract_image_from_xobject(None, &xobject, None).unwrap();
        assert_eq!(*image.color_space(), ColorSpace::DeviceGray);

        match image.data() {
            ImageData::Raw { format, .. } => {
                assert_eq!(*format, PixelFormat::Grayscale);
            },
            _ => panic!("Expected raw data"),
        }
    }

    #[test]
    fn test_extract_image_from_xobject_missing_subtype() {
        use crate::object::Object;
        use std::collections::HashMap;

        let dict = HashMap::new();
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };

        let result = extract_image_from_xobject(None, &xobject, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_image_from_xobject_wrong_subtype() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Form".to_string()));

        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };

        let result = extract_image_from_xobject(None, &xobject, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_image_from_xobject_missing_width() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Height".to_string(), Object::Integer(100));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));

        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(vec![]),
        };

        let result = extract_image_from_xobject(None, &xobject, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_image_from_xobject_jpeg_filter_array() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(50));
        dict.insert("Height".to_string(), Object::Integer(50));
        dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
        dict.insert("ColorSpace".to_string(), Object::Name("DeviceRGB".to_string()));
        dict.insert(
            "Filter".to_string(),
            Object::Array(vec![Object::Name("DCTDecode".to_string())]),
        );

        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from(jpeg_data.clone()),
        };

        let image = extract_image_from_xobject(None, &xobject, None).unwrap();

        match image.data() {
            ImageData::Jpeg(data) => assert_eq!(data, &jpeg_data),
            _ => panic!("Expected JPEG data"),
        }
    }
}
