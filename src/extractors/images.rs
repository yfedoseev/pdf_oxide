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
use crate::object::ObjectRef;
use std::cmp::min;
use std::path::Path;

/// A PDF image with metadata and pixel data.
///
/// Represents an image extracted from a PDF, including dimensions,
/// color space information, and the actual image data (either JPEG
/// or raw pixels).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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
    #[serde(skip_serializing_if = "ImageData::is_empty")]
    data: ImageData,
    /// Optional bounding box in PDF user space (v0.3.14)
    bbox: Option<Rect>,
    /// Rotation in degrees (v0.3.14)
    rotation_degrees: i32,
    /// Transformation matrix (v0.3.14)
    matrix: [f32; 6],
    /// CCITT decompression parameters (for 1-bit bilevel images)
    #[serde(skip)]
    ccitt_params: Option<crate::decoders::CcittParams>,
    /// Embedded ICC profile associated with the image's colour space,
    /// if any. For a plain `/ICCBased` image this is the profile from
    /// the array; for an `Indexed` image with an `ICCBased` base this
    /// is the base profile. `None` when the document only used
    /// device-dependent colour. Consumed by `save_as_*` to drive the
    /// CMYK→sRGB conversion through the CMM instead of the §10.3.5
    /// additive-clamp fallback.
    #[serde(skip)]
    icc_profile: Option<std::sync::Arc<crate::color::IccProfile>>,
    /// Rendering intent from the image dictionary's `/Intent`, or the
    /// graphics-state default per ISO 32000-1:2008 §8.6.5.8.
    rendering_intent: crate::color::RenderingIntent,
}

impl PdfImage {
    /// Create a new PDF image.
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
            rotation_degrees: 0,
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ccitt_params: None,
            icc_profile: None,
            rendering_intent: crate::color::RenderingIntent::default(),
        }
    }

    /// Create a new PDF image with spatial metadata (v0.3.14).
    pub fn with_spatial(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        bits_per_component: u8,
        data: ImageData,
        bbox: Rect,
        rotation: i32,
        matrix: [f32; 6],
    ) -> Self {
        Self {
            width,
            height,
            color_space,
            bits_per_component,
            data,
            bbox: Some(bbox),
            rotation_degrees: rotation,
            matrix,
            ccitt_params: None,
            icc_profile: None,
            rendering_intent: crate::color::RenderingIntent::default(),
        }
    }

    /// Create a new PDF image with a bounding box (v0.3.12, convenience wrapper).
    pub fn with_bbox(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        bits_per_component: u8,
        data: ImageData,
        bbox: Rect,
    ) -> Self {
        Self::with_spatial(
            width,
            height,
            color_space,
            bits_per_component,
            data,
            bbox,
            0,
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
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
            rotation_degrees: 0,
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ccitt_params: Some(ccitt_params),
            icc_profile: None,
            rendering_intent: crate::color::RenderingIntent::default(),
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

    /// Set the bounding box for this image.
    pub fn set_bbox(&mut self, bbox: Rect) {
        self.bbox = Some(bbox);
    }

    /// Get rotation in degrees.
    pub fn rotation_degrees(&self) -> i32 {
        self.rotation_degrees
    }

    /// Set rotation in degrees.
    pub fn set_rotation_degrees(&mut self, rotation: i32) {
        self.rotation_degrees = rotation;
    }

    /// Get transformation matrix.
    pub fn matrix(&self) -> [f32; 6] {
        self.matrix
    }

    /// Set transformation matrix.
    pub fn set_matrix(&mut self, matrix: [f32; 6]) {
        self.matrix = matrix;
    }

    /// Set CCITT decompression parameters for this image.
    pub fn set_ccitt_params(&mut self, params: crate::decoders::CcittParams) {
        self.ccitt_params = Some(params);
    }

    /// Get CCITT decompression parameters if available.
    pub fn ccitt_params(&self) -> Option<&crate::decoders::CcittParams> {
        self.ccitt_params.as_ref()
    }

    /// Embedded ICC profile associated with the image, if any.
    pub fn icc_profile(&self) -> Option<&std::sync::Arc<crate::color::IccProfile>> {
        self.icc_profile.as_ref()
    }

    /// Attach an ICC profile (used by extractors; colour conversion
    /// picks it up automatically when present).
    pub fn set_icc_profile(&mut self, profile: std::sync::Arc<crate::color::IccProfile>) {
        self.icc_profile = Some(profile);
    }

    /// Rendering intent — ISO 32000-1:2008 §8.6.5.8, defaults to
    /// `RelativeColorimetric`.
    pub fn rendering_intent(&self) -> crate::color::RenderingIntent {
        self.rendering_intent
    }

    /// Set the rendering intent (used by extractors when they see an
    /// explicit `/Intent` entry on the image dictionary).
    pub fn set_rendering_intent(&mut self, intent: crate::color::RenderingIntent) {
        self.rendering_intent = intent;
    }

    /// Build the source→sRGB transform from this image's embedded ICC
    /// profile (if any). Returns `None` when the image uses purely
    /// device-dependent colour, or when no profile was resolved at
    /// extraction time.
    ///
    /// The resulting transform is component-agnostic: callers pick the
    /// matching `Transform::convert_{cmyk,rgb,gray}_*` method based on
    /// the source pixel format. Used by the `decode_cmyk_jpeg_to_rgb_…`,
    /// `cmyk_to_rgb_with_transform`, and `save_raw_as_*` paths.
    fn build_icc_transform(&self) -> Option<crate::color::Transform> {
        self.icc_profile
            .as_ref()
            .map(|p| crate::color::Transform::new_srgb_target(p.clone(), self.rendering_intent))
    }

    /// Save the image as PNG format.
    pub fn save_as_png(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                if self.color_space.components() == 4 {
                    let transform = self.build_icc_transform();
                    let rgb = decode_cmyk_jpeg_to_rgb_with_profile(jpeg_data, transform.as_ref())?;
                    let buf = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                        self.width,
                        self.height,
                        rgb,
                    )
                    .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;
                    buf.save_with_format(path, image::ImageFormat::Png)
                        .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
                } else {
                    save_jpeg_as_png(jpeg_data, path)
                }
            },
            ImageData::Raw { pixels, format } => {
                // Always build the transform if a profile is present; the
                // save helper picks the right convert_* method for the
                // pixel format. RGB/Gray ICCBased samples would otherwise
                // be written as-is, which is wrong when the profile is
                // wide-gamut (Adobe RGB, ProPhoto, …) or a calibrated
                // grayscale other than sRGB gamma.
                let transform = self.build_icc_transform();
                save_raw_as_png(pixels, self.width, self.height, *format, transform.as_ref(), path)
            },
        }
    }

    /// Save the image as JPEG format.
    pub fn save_as_jpeg(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            // Pass-through for RGB / grayscale JPEGs — viewers handle those
            // uniformly. CMYK JPEGs (4-channel ColorSpace such as DeviceCMYK
            // or ICCBased N=4) must be decoded and re-encoded as RGB because
            // most viewers either fail to open CMYK JPEGs or display them
            // with inverted or washed-out colors. `decode_cmyk_jpeg_to_rgb`
            // pulls CMYK samples via `jpeg-decoder`, inspects the APP14
            // Adobe marker to detect the inverted-channel convention
            // Photoshop / InDesign / WPS write, inverts when present, then
            // does a naive CMYK→RGB conversion (full ICC profile handling
            // is a follow-up).
            ImageData::Jpeg(jpeg_data) => {
                if self.color_space.components() == 4 {
                    let transform = self.build_icc_transform();
                    let rgb = decode_cmyk_jpeg_to_rgb_with_profile(jpeg_data, transform.as_ref())?;
                    let buf = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                        self.width,
                        self.height,
                        rgb,
                    )
                    .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;
                    buf.save_with_format(path, image::ImageFormat::Jpeg)
                        .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
                } else {
                    std::fs::write(path, jpeg_data).map_err(Error::from)
                }
            },
            ImageData::Raw { pixels, format } => {
                let transform = self.build_icc_transform();
                save_raw_as_jpeg(pixels, self.width, self.height, *format, transform.as_ref(), path)
            },
        }
    }

    /// Convert image to PNG bytes in memory.
    pub fn to_png_bytes(&self) -> Result<Vec<u8>> {
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        use image::ImageEncoder;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        let encoder =
            PngEncoder::new_with_quality(&mut buffer, CompressionType::Fast, FilterType::NoFilter);

        match &self.data {
            ImageData::Raw { pixels, format } => {
                let expected_gray = (self.width * self.height) as usize;
                let expected_rgb = expected_gray * 3;

                if *format == PixelFormat::Grayscale
                    && matches!(self.color_space, ColorSpace::DeviceGray | ColorSpace::CalGray)
                    && pixels.len() == expected_gray
                {
                    // image 0.25 changed `write_image` to take
                    // `ExtendedColorType` — `ColorType::*` now converts
                    // through `Into`. API-only change, same semantics.
                    encoder
                        .write_image(pixels, self.width, self.height, image::ColorType::L8.into())
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                } else if *format == PixelFormat::RGB && pixels.len() == expected_rgb {
                    encoder
                        .write_image(pixels, self.width, self.height, image::ColorType::Rgb8.into())
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                } else {
                    let dynamic_image = self.to_dynamic_image()?;
                    let rgb = dynamic_image.to_rgb8();
                    encoder
                        .write_image(
                            rgb.as_raw(),
                            self.width,
                            self.height,
                            image::ColorType::Rgb8.into(),
                        )
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                }
            },
            ImageData::Jpeg(_) => {
                let dynamic_image = self.to_dynamic_image()?;
                let rgb = dynamic_image.to_rgb8();
                encoder
                    .write_image(
                        rgb.as_raw(),
                        self.width,
                        self.height,
                        image::ColorType::Rgb8.into(),
                    )
                    .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
            },
        }

        Ok(buffer.into_inner())
    }

    /// Convert image to a base64 data URI for embedding in HTML.
    pub fn to_base64_data_uri(&self) -> Result<String> {
        use base64::{engine::general_purpose::STANDARD, Engine};

        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                let base64_str = STANDARD.encode(jpeg_data);
                Ok(format!("data:image/jpeg;base64,{}", base64_str))
            },
            ImageData::Raw { .. } => {
                let png_bytes = self.to_png_bytes()?;
                let base64_str = STANDARD.encode(&png_bytes);
                Ok(format!("data:image/png;base64,{}", base64_str))
            },
        }
    }

    /// Convert this PDF image to a `DynamicImage`.
    pub fn to_dynamic_image(&self) -> Result<image::DynamicImage> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => {
                log::debug!(
                    "Decoding JPEG data ({} bytes), starts with: {:02X?}",
                    jpeg_data.len(),
                    &jpeg_data[..min(jpeg_data.len(), 16)]
                );
                image::load_from_memory(jpeg_data)
                    .map_err(|e| Error::Decode(format!("Failed to decode JPEG: {}", e)))
            },
            ImageData::Raw { pixels, format } => {
                if self.bits_per_component == 1
                    && matches!(self.color_space, ColorSpace::DeviceGray)
                {
                    let params =
                        self.ccitt_params
                            .clone()
                            .unwrap_or_else(|| crate::decoders::CcittParams {
                                columns: self.width,
                                rows: Some(self.height),
                                ..Default::default()
                            });

                    let decompressed = ccitt_bilevel::decompress_ccitt(pixels, &params)?;
                    let grayscale =
                        ccitt_bilevel::bilevel_to_grayscale(&decompressed, self.width, self.height);

                    image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(
                        self.width,
                        self.height,
                        grayscale,
                    )
                    .ok_or_else(|| Error::Decode("Invalid image dimensions".to_string()))
                    .map(image::DynamicImage::ImageLuma8)
                } else {
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
                        _ => {
                            let rgb_pixels = match format {
                                PixelFormat::Grayscale => {
                                    pixels.iter().flat_map(|&g| vec![g, g, g]).collect()
                                },
                                PixelFormat::CMYK => cmyk_to_rgb_with_transform(
                                    pixels,
                                    self.build_icc_transform().as_ref(),
                                ),
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
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum ImageData {
    /// JPEG-encoded image data.
    Jpeg(Vec<u8>),
    /// Raw pixel data with a specified format.
    Raw {
        /// Raw pixel bytes.
        pixels: Vec<u8>,
        /// Pixel format (RGB, Grayscale, CMYK).
        format: PixelFormat,
    },
}

impl ImageData {
    /// Returns true if the image data is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            ImageData::Jpeg(data) => data.is_empty(),
            ImageData::Raw { pixels, .. } => pixels.is_empty(),
        }
    }
}

/// PDF color space types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ColorSpace {
    /// RGB color space (3 components).
    DeviceRGB,
    /// Grayscale color space (1 component).
    DeviceGray,
    /// CMYK color space (4 components).
    DeviceCMYK,
    /// Indexed (palette-based) color space.
    Indexed,
    /// Calibrated grayscale.
    CalGray,
    /// Calibrated RGB.
    CalRGB,
    /// CIE L*a*b* color space.
    Lab,
    /// ICC profile-based color space with N components.
    ICCBased(usize),
    /// Separation (spot color) space.
    Separation,
    /// DeviceN (multi-ink) color space.
    DeviceN,
    /// Pattern color space.
    Pattern,
}

impl ColorSpace {
    /// Returns the number of color components for this color space.
    pub fn components(&self) -> usize {
        match self {
            ColorSpace::DeviceGray => 1,
            ColorSpace::DeviceRGB => 3,
            ColorSpace::DeviceCMYK => 4,
            ColorSpace::Indexed => 1,
            ColorSpace::CalGray => 1,
            ColorSpace::CalRGB => 3,
            ColorSpace::Lab => 3,
            ColorSpace::ICCBased(n) => *n,
            ColorSpace::Separation => 1,
            ColorSpace::DeviceN => 4,
            ColorSpace::Pattern => 0,
        }
    }
}

/// Pixel format for raw image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum PixelFormat {
    /// RGB format (3 bytes per pixel).
    RGB,
    /// Grayscale format (1 byte per pixel).
    Grayscale,
    /// CMYK format (4 bytes per pixel).
    CMYK,
}

impl PixelFormat {
    /// Returns the number of bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Grayscale => 1,
            PixelFormat::RGB => 3,
            PixelFormat::CMYK => 4,
        }
    }
}

fn color_space_to_pixel_format(color_space: &ColorSpace) -> PixelFormat {
    match color_space {
        ColorSpace::DeviceGray => PixelFormat::Grayscale,
        ColorSpace::DeviceRGB => PixelFormat::RGB,
        ColorSpace::DeviceCMYK => PixelFormat::CMYK,
        ColorSpace::Indexed => PixelFormat::RGB,
        ColorSpace::CalGray => PixelFormat::Grayscale,
        ColorSpace::CalRGB => PixelFormat::RGB,
        ColorSpace::Lab => PixelFormat::RGB,
        ColorSpace::ICCBased(n) => match n {
            1 => PixelFormat::Grayscale,
            3 => PixelFormat::RGB,
            4 => PixelFormat::CMYK,
            _ => PixelFormat::RGB,
        },
        ColorSpace::Separation => PixelFormat::Grayscale,
        ColorSpace::DeviceN => PixelFormat::CMYK,
        ColorSpace::Pattern => PixelFormat::RGB,
    }
}

/// Parse a ColorSpace name from a PDF object.
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
            if let Some(name) = arr[0].as_name() {
                match name {
                    "Indexed" => Ok(ColorSpace::Indexed),
                    "CalGray" => Ok(ColorSpace::CalGray),
                    "CalRGB" => Ok(ColorSpace::CalRGB),
                    "Lab" => Ok(ColorSpace::Lab),
                    "ICCBased" => {
                        let num_components = if arr.len() > 1 {
                            if let Some(stream_dict) = arr[1].as_dict() {
                                stream_dict
                                    .get("N")
                                    .and_then(|obj| match obj {
                                        Object::Integer(n) => Some(*n as usize),
                                        _ => None,
                                    })
                                    .unwrap_or(3)
                            } else {
                                3
                            }
                        } else {
                            3
                        };
                        Ok(ColorSpace::ICCBased(num_components))
                    },
                    "Separation" => Ok(ColorSpace::Separation),
                    "DeviceN" => Ok(ColorSpace::DeviceN),
                    "Pattern" => Ok(ColorSpace::Pattern),
                    other => Err(Error::Image(format!("Unsupported array color space: {}", other))),
                }
            } else {
                Err(Error::Image("Color space array must start with a name".to_string()))
            }
        },
        _ => Err(Error::Image(format!("Invalid color space object: {:?}", obj))),
    }
}

/// Extract an image from an XObject stream.
pub fn extract_image_from_xobject(
    doc: Option<&crate::document::PdfDocument>,
    xobject: &crate::object::Object,
    obj_ref: Option<ObjectRef>,
    color_space_map: Option<&std::collections::HashMap<String, crate::object::Object>>,
) -> Result<PdfImage> {
    use crate::object::Object;

    let dict = xobject
        .as_dict()
        .ok_or_else(|| Error::Image("XObject is not a stream".to_string()))?;

    let subtype = dict
        .get("Subtype")
        .and_then(|obj| obj.as_name())
        .ok_or_else(|| Error::Image("XObject missing /Subtype".to_string()))?;

    if subtype != "Image" {
        return Err(Error::Image(format!("XObject subtype is not Image: {}", subtype)));
    }

    let width = dict
        .get("Width")
        .and_then(|obj| obj.as_integer())
        .ok_or_else(|| Error::Image("Image missing /Width".to_string()))? as u32;

    let height = dict
        .get("Height")
        .and_then(|obj| obj.as_integer())
        .ok_or_else(|| Error::Image("Image missing /Height".to_string()))? as u32;

    let bits_per_component = dict
        .get("BitsPerComponent")
        .and_then(|obj| obj.as_integer())
        .unwrap_or(8) as u8;

    let color_space_obj = dict
        .get("ColorSpace")
        .ok_or_else(|| Error::Image("Image missing /ColorSpace".to_string()))?;

    let resolved_color_space = if let Some(d) = doc {
        let res = if let Some(obj_ref) = color_space_obj.as_reference() {
            d.load_object(obj_ref)?
        } else {
            color_space_obj.clone()
        };
        if let Object::Name(ref name) = res {
            if let Some(map) = color_space_map {
                map.get(name).cloned().unwrap_or(res)
            } else {
                res
            }
        } else {
            res
        }
    } else {
        color_space_obj.clone()
    };

    // For array-form color spaces (e.g. [/ICCBased <ref>], [/Indexed <base> <hi> <palette_ref>])
    // the second element is commonly an indirect reference to the ICC profile
    // stream / palette. `parse_color_space` only inspects the immediate
    // `Object::Stream` dict, so an unresolved reference silently falls back to
    // `N = 3` and a CMYK (N = 4) image is labelled as RGB. Resolve the stream
    // reference here so the component count reflects the real profile.
    let resolved_color_space =
        if let (Some(doc_mut), Object::Array(arr)) = (doc, &resolved_color_space) {
            if arr.len() > 1 {
                if let Some(second_ref) = arr[1].as_reference() {
                    if let Ok(resolved_second) = doc_mut.load_object(second_ref) {
                        let mut new_arr = arr.clone();
                        new_arr[1] = resolved_second;
                        Object::Array(new_arr)
                    } else {
                        resolved_color_space
                    }
                } else {
                    resolved_color_space
                }
            } else {
                resolved_color_space
            }
        } else {
            resolved_color_space
        };

    let color_space = parse_color_space(&resolved_color_space)?;
    // For Indexed color spaces, resolve the base color space and palette now so we
    // can expand indices to RGB after decoding the stream. Without this, raw
    // Indexed pixel data (1 byte per pixel) is mislabelled as RGB (3 bytes per
    // pixel) and ImageBuffer::from_raw rejects the wrong length. Fail fast if
    // the palette cannot be resolved so the error points at the real root cause
    // instead of the downstream "Invalid RGB image dimensions" symptom.
    let indexed_resolution: Option<IndexedResolution> = if color_space == ColorSpace::Indexed {
        let resolved = resolve_indexed_palette(doc, &resolved_color_space)?;
        if resolved.is_none() {
            return Err(Error::Image("Unable to resolve Indexed color space palette".to_string()));
        }
        resolved
    } else {
        None
    };

    // For a plain (non-Indexed) `[/ICCBased <stream>]` colour space,
    // capture the profile bytes so the CMM can convert through the
    // document's actual source characterisation instead of the
    // §10.3.5 additive-clamp fallback.
    //
    // When the image uses plain `/DeviceCMYK` with no ICC profile of
    // its own, fall back to the document's `/OutputIntents` CMYK
    // profile if one exists — the standard PDF/X assumption per
    // ISO 32000-1:2008 §14.11.5.
    let direct_icc_profile = if matches!(color_space, ColorSpace::ICCBased(_)) {
        resolve_icc_profile_from_obj(doc, &resolved_color_space)
    } else if color_space == ColorSpace::DeviceCMYK {
        doc.and_then(|d| d.output_intent_cmyk_profile())
    } else {
        None
    };

    // Per §8.6.5.8, an image dictionary may override the graphics-state
    // rendering intent via `/Intent`. Unrecognised names fall through
    // to `RelativeColorimetric`.
    let rendering_intent = dict
        .get("Intent")
        .and_then(|obj| obj.as_name())
        .map(crate::color::RenderingIntent::from_pdf_name)
        .unwrap_or_default();

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

    let has_dct = filter_names.iter().any(|name| name == "DCTDecode");
    let is_jpeg_only = has_dct && filter_names.len() == 1;
    let is_jpeg_chain = has_dct && filter_names.len() > 1;

    let is_jbig2 = filter_names
        .iter()
        .any(|n| n.eq_ignore_ascii_case("JBIG2Decode"));

    let data = if is_jbig2 {
        decode_jbig2_image(xobject, obj_ref, dict, doc, width, height)?
    } else if is_jpeg_only || is_jpeg_chain {
        let decoded = if let (Some(d), Some(ref_id)) = (doc.as_ref(), obj_ref) {
            d.decode_stream_with_encryption(xobject, ref_id)?
        } else {
            xobject.decode_stream_data()?
        };
        ImageData::Jpeg(decoded)
    } else {
        let decoded_data = if let (Some(d), Some(ref_id)) = (doc.as_ref(), obj_ref) {
            d.decode_stream_with_encryption(xobject, ref_id)?
        } else {
            xobject.decode_stream_data()?
        };
        if let Some(ir) = indexed_resolution.as_ref() {
            // Build a Transform if the Indexed base has a profile so
            // palette entries render through the real CMM (when linked).
            let transform = ir
                .base_profile
                .clone()
                .map(|p| crate::color::Transform::new_srgb_target(p, rendering_intent));
            let expanded = expand_indexed_to_rgb_with_transform(
                &decoded_data,
                &ir.palette,
                ir.base_fmt,
                width,
                height,
                bits_per_component,
                transform.as_ref(),
            )?;
            ImageData::Raw {
                pixels: expanded,
                format: PixelFormat::RGB,
            }
        } else {
            let pixel_format = color_space_to_pixel_format(&color_space);
            ImageData::Raw {
                pixels: decoded_data,
                format: pixel_format,
            }
        }
    };

    // JBIG2 decode produces 8-bit-per-channel pixels regardless of the
    // XObject's BitsPerComponent (which is 1).  Override to 8 so that
    // to_dynamic_image() does not try to CCITT-decompress the output.
    let effective_bpc = if is_jbig2 { 8 } else { bits_per_component };
    let mut image = PdfImage::new(width, height, color_space, effective_bpc, data);

    // Attach the ICC profile if we found one — prefer the direct ICCBased
    // profile, then fall back to an Indexed base's profile so the CMM has
    // something to work with for palette-backed CMYK/Lab images too.
    if let Some(p) = direct_icc_profile {
        image.set_icc_profile(p);
    } else if let Some(ir) = indexed_resolution.as_ref() {
        if let Some(p) = ir.base_profile.clone() {
            image.set_icc_profile(p);
        }
    }
    image.set_rendering_intent(rendering_intent);

    if bits_per_component == 1 && image.color_space == ColorSpace::DeviceGray {
        if let Some(mut ccitt_params) =
            crate::object::extract_ccitt_params_with_width(dict.get("DecodeParms"), Some(width))
        {
            if ccitt_params.rows.is_none() {
                ccitt_params.rows = Some(height);
            }
            image.set_ccitt_params(ccitt_params);
        }
    }

    Ok(image)
}

/// Extract and parse an `ICCBased` colour-space's profile stream.
///
/// Accepts either a fully-resolved `[/ICCBased <Stream>]` array (the
/// stream is an `Object::Stream` directly), or a `[/ICCBased <Ref>]`
/// array where the second element is a live reference — in that case
/// `doc` must be supplied so we can dereference.
///
/// Returns `None` if:
///   - `cs_obj` isn't an ICCBased array,
///   - the profile stream can't be decoded,
///   - the profile bytes fail ICC header validation, or
///   - the declared `/N` disagrees with the profile header's
///     colourSpace signature (PDF §8.6.5.5 mandates they match).
///
/// No error is returned — callers treat "no profile" as "fall back to
/// device colour space" per §8.6.5.5's /Alternate clause.
pub(crate) fn resolve_icc_profile_from_obj(
    doc: Option<&crate::document::PdfDocument>,
    cs_obj: &crate::object::Object,
) -> Option<std::sync::Arc<crate::color::IccProfile>> {
    use crate::object::Object;

    let Object::Array(arr) = cs_obj else {
        return None;
    };
    if arr.len() < 2 || arr[0].as_name() != Some("ICCBased") {
        return None;
    }

    // Second element should be a stream (already resolved by the caller
    // in the common path) or a reference we still need to dereference.
    let profile_obj = match (&arr[1], doc) {
        (Object::Stream { .. }, _) => arr[1].clone(),
        (Object::Reference(r), Some(d)) => match d.load_object(*r) {
            Ok(obj) => obj,
            Err(_) => return None,
        },
        _ => return None,
    };

    let Object::Stream { dict, .. } = &profile_obj else {
        return None;
    };
    // `N` is mandatory per PDF 32000-1 §8.6.5.5 Table 66.
    let n = dict
        .get("N")
        .and_then(|obj| obj.as_integer())
        .filter(|n| matches!(*n, 1 | 3 | 4))? as u8;

    let bytes = profile_obj.decode_stream_data().ok()?;
    let profile = crate::color::IccProfile::parse(bytes, n)?;
    Some(std::sync::Arc::new(profile))
}

/// Outcome of resolving an `[/Indexed base hival lookup]` colour space:
/// the palette in the base's pixel format, plus the base's ICC profile
/// when the base is `ICCBased`.
pub(crate) struct IndexedResolution {
    pub base_fmt: PixelFormat,
    pub palette: Vec<u8>,
    /// `None` for device-dependent bases or bases we already folded
    /// colourimetrically (e.g. Lab, whose palette is rewritten to RGB
    /// before being returned).
    pub base_profile: Option<std::sync::Arc<crate::color::IccProfile>>,
}

/// Resolve an Indexed color space's base color space and palette lookup bytes.
///
/// PDF Indexed color spaces are `[/Indexed base hival lookup]` where `lookup`
/// is either a byte string or a stream of `(hival + 1) * N` bytes (N = number
/// of components in the base color space).
fn resolve_indexed_palette(
    doc: Option<&crate::document::PdfDocument>,
    cs_obj: &crate::object::Object,
) -> Result<Option<IndexedResolution>> {
    use crate::object::Object;

    let Object::Array(arr) = cs_obj else {
        return Ok(None);
    };
    if arr.len() < 4 {
        return Ok(None);
    }

    // Resolve the base color-space object. When it's an array like
    // [/ICCBased <stream_ref>], resolve inner references so
    // parse_color_space can read /N from the ICC stream dict.
    let base_obj = if let Some(d) = doc {
        let outer = if let Some(r) = arr[1].as_reference() {
            d.load_object(r)?
        } else {
            arr[1].clone()
        };
        if let Object::Array(mut inner) = outer {
            for item in inner.iter_mut() {
                if let Some(r) = item.as_reference() {
                    if let Ok(resolved) = d.load_object(r) {
                        *item = resolved;
                    }
                }
            }
            Object::Array(inner)
        } else {
            outer
        }
    } else {
        arr[1].clone()
    };
    let base_cs = parse_color_space(&base_obj)?;
    let base_fmt = color_space_to_pixel_format(&base_cs);
    let n = base_fmt.bytes_per_pixel();

    // When the base is `/ICCBased`, capture the profile bytes so the
    // extractor can later hand them to a CMM. Parse failures reduce to
    // `None` — the decoder then falls back to §10.3.5 CMYK→RGB math as
    // if no profile were present.
    let base_profile = if matches!(base_cs, ColorSpace::ICCBased(_)) {
        resolve_icc_profile_from_obj(doc, &base_obj)
    } else {
        None
    };

    // hival bounds the valid index range. Resolve via indirect reference if
    // needed; treat invalid / missing values as "unknown" and skip truncation.
    let hival_obj = if let Some(d) = doc {
        if let Some(r) = arr[2].as_reference() {
            d.load_object(r)?
        } else {
            arr[2].clone()
        }
    } else {
        arr[2].clone()
    };
    let hival: Option<usize> = hival_obj.as_integer().and_then(|i| {
        if (0..=255).contains(&i) {
            Some(i as usize)
        } else {
            None
        }
    });

    let lookup_obj = if let Some(d) = doc {
        if let Some(r) = arr[3].as_reference() {
            d.load_object(r)?
        } else {
            arr[3].clone()
        }
    } else {
        arr[3].clone()
    };
    let mut palette_bytes = match &lookup_obj {
        Object::String(s) => s.clone(),
        Object::Stream { .. } => lookup_obj.decode_stream_data()?,
        _ => return Ok(None),
    };
    if palette_bytes.is_empty() {
        return Ok(None);
    }

    // Truncate palette to the logical length implied by hival so that indices
    // greater than hival fall into the out-of-range branch of the expander.
    // Per PDF 32000-1:2008 §8.6.6.3 the lookup is exactly (hival + 1) * N bytes;
    // anything beyond that is stray data that must not be mapped to pixels.
    if let Some(h) = hival {
        let expected = (h + 1).saturating_mul(n);
        if expected > 0 && palette_bytes.len() > expected {
            palette_bytes.truncate(expected);
        }
    }

    // Device-independent colour-space palettes must be converted to
    // RGB before being handed to the expander, which assumes palette
    // bytes are already in the output colour space. Without this step
    // Lab triples are mis-interpreted as raw RGB and render with
    // perceptually wrong colours.
    if matches!(base_cs, ColorSpace::Lab) {
        let white = extract_lab_whitepoint(&base_obj);
        let rgb_palette = lab_palette_to_rgb(&palette_bytes, white);
        // Lab palettes are now RGB; no base ICC profile to carry through.
        return Ok(Some(IndexedResolution {
            base_fmt: PixelFormat::RGB,
            palette: rgb_palette,
            base_profile: None,
        }));
    }

    Ok(Some(IndexedResolution {
        base_fmt,
        palette: palette_bytes,
        base_profile,
    }))
}

/// Expand packed Indexed image indices into RGB bytes using the palette.
///
/// Supports 1, 2, 4, and 8 bit-per-component index streams. Rows are padded
/// to byte boundaries per the PDF spec.
///
/// Returns `Err(Error::Image)` when the requested dimensions would require
/// more than `MAX_INDEXED_OUTPUT_BYTES` to decode, or when the `usize`
/// arithmetic on `width * height * channels` / `width * bpc` overflows,
/// or when the input `raw` buffer is too short to supply every row of the
/// requested height. This is an input-amplification guard for maliciously
/// crafted PDFs that pair tiny streams with extreme Indexed image
/// dimensions — see issue #324.
#[cfg(test)]
fn expand_indexed_to_rgb(
    raw: &[u8],
    palette: &[u8],
    base_fmt: PixelFormat,
    width: u32,
    height: u32,
    bpc: u8,
) -> Result<Vec<u8>> {
    expand_indexed_to_rgb_with_transform(raw, palette, base_fmt, width, height, bpc, None)
}

/// Like [`expand_indexed_to_rgb`] but routes CMYK palette entries
/// through an ICC transform when one is supplied. Used during image
/// extraction when the base colour space is `/ICCBased` with N=4.
fn expand_indexed_to_rgb_with_transform(
    raw: &[u8],
    palette: &[u8],
    base_fmt: PixelFormat,
    width: u32,
    height: u32,
    bpc: u8,
    transform: Option<&crate::color::Transform>,
) -> Result<Vec<u8>> {
    /// Hard cap on the decoded output buffer size (256 MiB). Legitimate
    /// Indexed images in real PDFs are several orders of magnitude below
    /// this — the cap only fires on pathological / adversarial inputs
    /// where `width * height` is billions of pixels.
    const MAX_INDEXED_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

    let w = width as usize;
    let h = height as usize;
    let n = base_fmt.bytes_per_pixel();

    // ISO 32000-2 §8.9.5.1 mandates bpc ∈ {1, 2, 4, 8} for Indexed color
    // spaces. Anything else (0, 3, 5, 6, 7, 9, 12, 16, …) used to be
    // accepted silently — bpc=0 was coerced to 1 and any other value fell
    // through the `read_index` `_ => 0` arm, producing a solid palette-
    // entry-0 image with no error. Reject up front so malformed input is
    // surfaced instead of decoded into nonsense pixels.
    if !matches!(bpc, 1 | 2 | 4 | 8) {
        return Err(Error::Image(format!(
            "Indexed image has invalid /BitsPerComponent {bpc} \
             (PDF spec requires 1, 2, 4, or 8)"
        )));
    }

    // Checked arithmetic for `bytes_per_row = ceil(w * bpc / 8)`.
    let bytes_per_row = w
        .checked_mul(bpc as usize)
        .map(|v| v.div_ceil(8))
        .ok_or_else(|| {
            Error::Image(format!("Indexed image row width overflow: {w} × {bpc} bpc exceeds usize"))
        })?;

    // Checked arithmetic for `w * h * 3` (output always written as RGB).
    let output_bytes = w
        .checked_mul(h)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| {
            Error::Image(format!("Indexed image output size overflow: {w} × {h} × 3 exceeds usize"))
        })?;

    if output_bytes > MAX_INDEXED_OUTPUT_BYTES {
        return Err(Error::Image(format!(
            "Indexed image decode would produce {output_bytes} bytes, \
             exceeds guard limit of {MAX_INDEXED_OUTPUT_BYTES} bytes \
             (width={w}, height={h})"
        )));
    }

    // The decoded index stream must cover every row of the image.
    // Truncated streams used to get silently zero-padded, which lets a
    // malicious PDF pair a 10-byte stream with a 10 000 × 10 000 image
    // and force a ~300 MiB allocation filled with default palette entry
    // 0. Reject that shape up front.
    let required_bytes = bytes_per_row.checked_mul(h).ok_or_else(|| {
        Error::Image(format!(
            "Indexed image required-input size overflow: {bytes_per_row} × {h} exceeds usize"
        ))
    })?;
    if raw.len() < required_bytes {
        return Err(Error::Image(format!(
            "Indexed image index stream truncated: {} bytes available, \
             {} required ({} bytes/row × {} rows)",
            raw.len(),
            required_bytes,
            bytes_per_row,
            h
        )));
    }

    let mut out = Vec::with_capacity(output_bytes);

    let read_index = |row: &[u8], x: usize| -> usize {
        match bpc {
            8 => row.get(x).copied().unwrap_or(0) as usize,
            4 => {
                let byte_idx = x / 2;
                let b = row.get(byte_idx).copied().unwrap_or(0);
                if x.is_multiple_of(2) {
                    (b >> 4) as usize
                } else {
                    (b & 0x0F) as usize
                }
            },
            2 => {
                let byte_idx = x / 4;
                let b = row.get(byte_idx).copied().unwrap_or(0);
                let shift = 6 - (x % 4) * 2;
                ((b >> shift) & 0x03) as usize
            },
            1 => {
                let byte_idx = x / 8;
                let b = row.get(byte_idx).copied().unwrap_or(0);
                let shift = 7 - (x % 8);
                ((b >> shift) & 0x01) as usize
            },
            // Unreachable: bpc is validated to be in {1, 2, 4, 8} above
            // before the closure is called, so this arm only exists to
            // satisfy exhaustiveness on `u8`.
            _ => unreachable!("bpc validated to {{1,2,4,8}} before read_index"),
        }
    };

    for y in 0..h {
        let row_start = y * bytes_per_row;
        let row_end = (row_start + bytes_per_row).min(raw.len());
        let row: &[u8] = if row_start < raw.len() {
            &raw[row_start..row_end]
        } else {
            &[]
        };
        for x in 0..w {
            let idx = read_index(row, x);
            let off = idx * n;
            if off + n > palette.len() {
                out.extend_from_slice(&[0, 0, 0]);
                continue;
            }
            match base_fmt {
                PixelFormat::RGB => out.extend_from_slice(&palette[off..off + 3]),
                PixelFormat::Grayscale => {
                    let g = palette[off];
                    out.push(g);
                    out.push(g);
                    out.push(g);
                },
                PixelFormat::CMYK => {
                    let c = palette[off];
                    let m = palette[off + 1];
                    let y_c = palette[off + 2];
                    let k = palette[off + 3];
                    let [r, g, b] = if let Some(t) = transform {
                        t.convert_cmyk_pixel(c, m, y_c, k)
                    } else {
                        cmyk_pixel_to_rgb(c, m, y_c, k)
                    };
                    out.push(r);
                    out.push(g);
                    out.push(b);
                },
            }
        }
    }
    Ok(out)
}

/// Convert a single CMYK pixel to RGB.
///
/// Shared conversion math used by both bulk CMYK→RGB and Indexed palette
/// expansion so the two paths cannot drift apart.
/// Convert one CMYK pixel to RGB using the PDF 32000-1:2008 §10.3.5 formula:
///
///   R = 1 − min(1, C + K)
///   G = 1 − min(1, M + K)
///   B = 1 − min(1, Y + K)
///
/// This is the spec-mandated fallback used whenever no ICC profile drives the
/// conversion. For pixels inside an `/ICCBased` colour space a real CMM
/// (qcms / lcms2) would replace this — tracked separately. Note the spec
/// formula is strictly additive-then-clamp; a multiplicative `(1-C)(1-K)`
/// variant is common in imaging stacks but does not match §10.3.5 on
/// heavily-inked samples.
pub(crate) fn cmyk_pixel_to_rgb(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
    let c = c as f32 / 255.0;
    let m = m as f32 / 255.0;
    let y = y as f32 / 255.0;
    let k = k as f32 / 255.0;

    let r = ((1.0 - (c + k).min(1.0)) * 255.0).round() as u8;
    let g = ((1.0 - (m + k).min(1.0)) * 255.0).round() as u8;
    let b = ((1.0 - (y + k).min(1.0)) * 255.0).round() as u8;

    [r, g, b]
}

/// Extract `/WhitePoint` from a Lab colour-space PDF object.
///
/// The object is `[/Lab << /WhitePoint [Xw Yw Zw] >>]`. Returns the
/// whitepoint as `[Xw, Yw, Zw]`, falling back to D65 if absent.
fn extract_lab_whitepoint(cs_obj: &crate::object::Object) -> [f64; 3] {
    const D65: [f64; 3] = [0.9505, 1.0, 1.0890];
    let arr = match cs_obj {
        crate::object::Object::Array(a) => a,
        _ => return D65,
    };
    if arr.len() < 2 {
        return D65;
    }
    let dict = match &arr[1] {
        crate::object::Object::Dictionary(d) => d,
        _ => return D65,
    };
    let wp = match dict.get("WhitePoint") {
        Some(crate::object::Object::Array(a)) if a.len() >= 3 => a,
        _ => return D65,
    };
    let f = |obj: &crate::object::Object| -> Option<f64> {
        match obj {
            crate::object::Object::Real(v) => Some(*v),
            crate::object::Object::Integer(v) => Some(*v as f64),
            _ => None,
        }
    };
    match (f(&wp[0]), f(&wp[1]), f(&wp[2])) {
        (Some(x), Some(y), Some(z)) => [x, y, z],
        _ => D65,
    }
}

/// Convert a Lab-encoded palette to sRGB.
///
/// Each entry is 3 bytes: L* (byte 0), a* (byte 1), b* (byte 2).
/// Decoding per PDF 32000-1:2008 §8.6.5.4:
///   L* = byte_0 / 255.0 × 100.0
///   a* = byte_1 − 128.0   (default /Range [−128 127])
///   b* = byte_2 − 128.0
///
/// Then Lab → XYZ (whitepoint-relative) → sRGB with standard gamma.
pub(crate) fn lab_palette_to_rgb(palette: &[u8], white: [f64; 3]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(palette.len());
    for chunk in palette.chunks(3) {
        if chunk.len() < 3 {
            rgb.extend_from_slice(&[0, 0, 0]);
            continue;
        }
        let [r, g, b] = lab_pixel_to_rgb(chunk[0], chunk[1], chunk[2], white);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    rgb
}

// NOTE: The XYZ→linear-sRGB matrix below assumes a D65 whitepoint. Lab CIEs
// whose `/WhitePoint` is non-D65 (D50 is common in print workflows) would
// strictly need chromatic adaptation (e.g., Bradford) from the source
// whitepoint to D65 before the sRGB matrix. We intentionally omit that for
// now — the vast majority of PDF `/Lab` spaces we encounter are D65 — but
// the caller's `white` is still used to scale `xw, yw, zw` so D65 and
// near-D65 whitepoints produce correct output. Non-D65 spaces will have a
// minor chromatic-adaptation error until this is revisited.
fn lab_pixel_to_rgb(l_byte: u8, a_byte: u8, b_byte: u8, white: [f64; 3]) -> [u8; 3] {
    let l_star = l_byte as f64 / 255.0 * 100.0;
    let a_star = a_byte as f64 - 128.0;
    let b_star = b_byte as f64 - 128.0;

    let fy = (l_star + 16.0) / 116.0;
    let fx = a_star / 500.0 + fy;
    let fz = fy - b_star / 200.0;

    let [xw, yw, zw] = white;
    let x = xw * f_inv(fx);
    let y = yw * f_inv(fy);
    let z = zw * f_inv(fz);

    // XYZ → linear sRGB (D65 matrix, IEC 61966-2-1:1999)
    let r_lin = 3.2406254773 * x - 1.5372079722 * y - 0.4986285987 * z;
    let g_lin = -0.9689307147 * x + 1.8757560609 * y + 0.0415175580 * z;
    let b_lin = 0.0557101204 * x - 0.2040210506 * y + 1.0569959423 * z;

    [srgb_gamma(r_lin), srgb_gamma(g_lin), srgb_gamma(b_lin)]
}

fn f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

fn srgb_gamma(lin: f64) -> u8 {
    let v = if lin <= 0.0031308 {
        12.92 * lin
    } else {
        1.055 * lin.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Convert a raw CMYK byte stream (4 bytes per pixel) to straight RGB bytes
/// (3 bytes per pixel) using the naive per-pixel conversion.
///
/// This is a non-ICC conversion and does not handle Adobe-inverted JPEG CMYK;
/// for JPEG-encoded CMYK streams use `decode_adobe_cmyk_jpeg` instead.
pub fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    cmyk_to_rgb_with_transform(cmyk, None)
}

/// Like [`cmyk_to_rgb`] but routes through an ICC transform when given,
/// and falls through to §10.3.5 otherwise. Used by save_raw_as_* when
/// the source image carries an ICC profile.
pub fn cmyk_to_rgb_with_transform(
    cmyk: &[u8],
    transform: Option<&crate::color::Transform>,
) -> Vec<u8> {
    if let Some(t) = transform {
        return t.convert_cmyk_buffer(cmyk);
    }
    let mut rgb = Vec::with_capacity((cmyk.len() / 4) * 3);
    for chunk in cmyk.chunks_exact(4) {
        let [r, g, b] = cmyk_pixel_to_rgb(chunk[0], chunk[1], chunk[2], chunk[3]);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    rgb
}

/// Decode a CMYK-colourspace JPEG to straight RGB bytes, applying Adobe's
/// inverted-CMYK convention when the APP14 marker requests it.
///
/// Adobe-authored CMYK / YCCK JPEGs (which most real-world producers emit
/// for print-targeted PDFs) store channel values inverted: 0 means "full
/// ink" and 255 means "no ink". Naive CMYK→RGB conversion on those raw
/// bytes yields near-black output — exactly the symptom of the issue this
/// handles. Detecting the APP14 color-transform and inverting per channel
/// before applying the standard CMYK→RGB math produces bright, correct
/// images for Adobe JPEGs while still producing correct output for
/// non-Adobe CMYK JPEGs that store values directly.
///
/// This is still a naive (non-ICC) conversion — it ignores any embedded
/// ICC profile and therefore cannot produce print-accurate colour. Proper
/// ICC handling (qcms / lcms) is a follow-up; this path is purely about
/// emitting sRGB that viewers can display without mis-interpreting the
/// channel polarity.
/// Thin wrapper that falls back to the intent-less, profile-less
/// variant — kept as the public, backwards-compatible entry point.
pub fn decode_cmyk_jpeg_to_rgb(jpeg_data: &[u8]) -> Result<Vec<u8>> {
    decode_cmyk_jpeg_to_rgb_with_profile(jpeg_data, None)
}

/// Like [`decode_cmyk_jpeg_to_rgb`] but applies the given ICC transform
/// when provided, falling back to §10.3.5 otherwise. Used internally by
/// `PdfImage::save_as_*` when the source image carries an ICCBased
/// colour space (or when the document's `OutputIntents` supplied a
/// default CMYK profile).
pub fn decode_cmyk_jpeg_to_rgb_with_profile(
    jpeg_data: &[u8],
    transform: Option<&crate::color::Transform>,
) -> Result<Vec<u8>> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(jpeg_data));
    let cmyk = decoder
        .decode()
        .map_err(|e| Error::Decode(format!("Failed to decode CMYK JPEG: {}", e)))?;
    let info = decoder
        .info()
        .ok_or_else(|| Error::Decode("JPEG info unavailable".to_string()))?;

    // Adobe APP14 marker contains a `color_transform` byte that tells
    // decoders how the channels are laid out. Value 0 on a 4-channel image
    // means "CMYK stored inverted" (the Photoshop convention); value 2
    // means "YCCK", which decoders convert to CMYK but the resulting values
    // are still inverted. Value 1 (YCbCr) only appears on 3-channel images.
    // When no APP14 is present we assume non-inverted CMYK, matching what
    // Poppler / pdfium do for bare CMYK JPEGs.
    let adobe_inverted = scan_adobe_inverted(jpeg_data);

    let pixel_count = (info.width as usize) * (info.height as usize);
    let expected = pixel_count * 4;
    if cmyk.len() < expected {
        return Err(Error::Decode(format!(
            "CMYK JPEG decoded {} bytes, expected {}",
            cmyk.len(),
            expected
        )));
    }

    // Normalize Adobe-inverted CMYK into straight CMYK first; the CMM
    // (or §10.3.5 fallback) always expects non-inverted input.
    let straight_cmyk: Vec<u8> = if adobe_inverted {
        let mut buf = Vec::with_capacity(pixel_count * 4);
        for chunk in cmyk.chunks_exact(4).take(pixel_count) {
            buf.extend_from_slice(&[
                255 - chunk[0],
                255 - chunk[1],
                255 - chunk[2],
                255 - chunk[3],
            ]);
        }
        buf
    } else {
        cmyk[..pixel_count * 4].to_vec()
    };

    if let Some(t) = transform {
        return Ok(t.convert_cmyk_buffer(&straight_cmyk));
    }

    // §10.3.5 additive-clamp fallback.
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in straight_cmyk.chunks_exact(4) {
        let [r, g, b] = cmyk_pixel_to_rgb(chunk[0], chunk[1], chunk[2], chunk[3]);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    Ok(rgb)
}

/// Walk the JPEG marker stream looking for an APP14 "Adobe" segment, and
/// return true if its `color_transform` byte indicates inverted CMYK
/// (values 0 on 4-channel, or 2 = YCCK). Returns false if no APP14 marker
/// is present or if it reports a non-inverted layout.
fn scan_adobe_inverted(jpeg_data: &[u8]) -> bool {
    let mut i = 0;
    while i + 1 < jpeg_data.len() {
        if jpeg_data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = jpeg_data[i + 1];
        i += 2;
        // Standalone markers (SOI, EOI, RSTn, TEM, fill bytes) have no length.
        if marker == 0x00 || marker == 0xFF {
            continue;
        }
        if matches!(marker, 0xD0..=0xD9) || marker == 0x01 {
            continue;
        }
        if i + 1 >= jpeg_data.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([jpeg_data[i], jpeg_data[i + 1]]) as usize;
        if seg_len < 2 || i + seg_len > jpeg_data.len() {
            break;
        }
        if marker == 0xEE && seg_len >= 14 {
            let payload = &jpeg_data[i + 2..i + seg_len];
            if payload.len() >= 12 && payload.starts_with(b"Adobe") {
                let transform = payload[11];
                return transform == 0 || transform == 2;
            }
        }
        if marker == 0xDA {
            // Start of Scan — image data follows; no more APP markers.
            break;
        }
        i += seg_len;
    }
    false
}

fn save_jpeg_as_png(jpeg_data: &[u8], path: impl AsRef<Path>) -> Result<()> {
    use image::ImageFormat;
    let img = image::load_from_memory_with_format(jpeg_data, ImageFormat::Jpeg)
        .map_err(|e| Error::Image(format!("Failed to decode JPEG: {}", e)))?;
    img.save_with_format(path, ImageFormat::Png)
        .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
}

/// Decide whether a given ICC transform should actually be applied to a
/// buffer of the given pixel format. The transform was compiled for
/// whatever component count the profile advertised; applying it to a
/// mismatched buffer (e.g. a 4-component CMYK transform to a 3-channel
/// RGB buffer) would produce garbage. A `None` transform, or a
/// transform whose profile components disagree with `format`, is
/// suppressed so the caller falls through to identity / fallback math.
fn icc_matches_format(
    transform: Option<&crate::color::Transform>,
    format: PixelFormat,
) -> Option<&crate::color::Transform> {
    let t = transform?;
    let needed = match format {
        PixelFormat::RGB => 3,
        PixelFormat::Grayscale => 1,
        PixelFormat::CMYK => 4,
    };
    if t.source_n_components() == needed {
        Some(t)
    } else {
        None
    }
}

fn save_raw_as_png(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    transform: Option<&crate::color::Transform>,
    path: impl AsRef<Path>,
) -> Result<()> {
    use image::{ImageBuffer, ImageFormat, Luma, Rgb};

    match format {
        PixelFormat::RGB => {
            // RGB source through an ICC profile (Adobe RGB, ProPhoto, wide-
            // gamut cameras) → convert to sRGB before writing. With no
            // profile the bytes are assumed sRGB already and passed through.
            let rgb = match icc_matches_format(transform, format) {
                Some(t) => t.convert_rgb_buffer(pixels),
                None => pixels.to_vec(),
            };
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid RGB image dimensions".to_string()))?;
            img.save_with_format(path, ImageFormat::Png)
                .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
        },
        PixelFormat::Grayscale => {
            // A Gray ICC profile promotes to sRGB RGB; without one the
            // single channel is written as an L8 PNG.
            if let Some(t) = icc_matches_format(transform, format) {
                let rgb = t.convert_gray_buffer(pixels);
                let img =
                    ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb).ok_or_else(|| {
                        Error::Image("Invalid grayscale image dimensions".to_string())
                    })?;
                img.save_with_format(path, ImageFormat::Png)
                    .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
            } else {
                let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels.to_vec())
                    .ok_or_else(|| {
                        Error::Image("Invalid grayscale image dimensions".to_string())
                    })?;
                img.save_with_format(path, ImageFormat::Png)
                    .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
            }
        },
        PixelFormat::CMYK => {
            let rgb = cmyk_to_rgb_with_transform(pixels, icc_matches_format(transform, format));
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;
            img.save_with_format(path, ImageFormat::Png)
                .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
        },
    }
}

fn save_raw_as_jpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    transform: Option<&crate::color::Transform>,
    path: impl AsRef<Path>,
) -> Result<()> {
    use image::{ImageBuffer, ImageFormat, Luma, Rgb};

    match format {
        PixelFormat::RGB => {
            let rgb = match icc_matches_format(transform, format) {
                Some(t) => t.convert_rgb_buffer(pixels),
                None => pixels.to_vec(),
            };
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid RGB image dimensions".to_string()))?;
            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
        PixelFormat::Grayscale => {
            if let Some(t) = icc_matches_format(transform, format) {
                let rgb = t.convert_gray_buffer(pixels);
                let img =
                    ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb).ok_or_else(|| {
                        Error::Image("Invalid grayscale image dimensions".to_string())
                    })?;
                img.save_with_format(path, ImageFormat::Jpeg)
                    .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
            } else {
                let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels.to_vec())
                    .ok_or_else(|| {
                        Error::Image("Invalid grayscale image dimensions".to_string())
                    })?;
                img.save_with_format(path, ImageFormat::Jpeg)
                    .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
            }
        },
        PixelFormat::CMYK => {
            let rgb = cmyk_to_rgb_with_transform(pixels, icc_matches_format(transform, format));
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;
            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
    }
}

/// Decode a JBIG2-compressed PDF image stream into raw grayscale pixels.
#[cfg(feature = "rendering")]
fn decode_jbig2_image(
    xobject: &crate::object::Object,
    obj_ref: Option<ObjectRef>,
    dict: &std::collections::HashMap<String, crate::object::Object>,
    doc: Option<&crate::document::PdfDocument>,
    width: u32,
    height: u32,
) -> Result<ImageData> {
    // The Jbig2Decoder in src/decoders/jbig2.rs is a pass-through: it returns
    // the raw compressed bitstream unchanged, which is exactly what hayro-jbig2
    // needs as input.
    let jbig2_bytes: Vec<u8> = if let (Some(d), Some(ref_id)) = (doc.as_ref(), obj_ref) {
        d.decode_stream_with_encryption(xobject, ref_id)?
    } else {
        xobject.decode_stream_data()?
    };

    // Load optional JBIG2Globals (shared symbol dictionaries referenced by multiple
    // embedded JBIG2 streams in the same PDF).
    let globals: Option<Vec<u8>> = (|| -> Option<Vec<u8>> {
        let dp = dict.get("DecodeParms")?.as_dict()?;
        let globals_ref = dp.get("JBIG2Globals")?.as_reference()?;
        let d = doc.as_ref()?;
        let globals_obj = d.load_object(globals_ref).ok()?;
        d.decode_stream_with_encryption(&globals_obj, globals_ref)
            .ok()
    })();

    let image = hayro_jbig2::Image::new_embedded(&jbig2_bytes, globals.as_deref())
        .map_err(|e| Error::Image(format!("JBIG2 decode error: {e}")))?;

    struct PixelCollector {
        pixels: Vec<u8>,
        row_buf: Vec<u8>,
    }

    impl hayro_jbig2::Decoder for PixelCollector {
        fn push_pixel(&mut self, black: bool) {
            self.row_buf.push(if black { 0 } else { 255 });
        }

        // chunk_count is the number of 8-pixel groups, not individual pixels.
        fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
            let v = if black { 0u8 } else { 255u8 };
            let n = chunk_count as usize * 8;
            self.row_buf.extend(std::iter::repeat_n(v, n));
        }

        fn next_line(&mut self) {
            self.pixels.append(&mut self.row_buf);
        }
    }

    let mut collector = PixelCollector {
        pixels: Vec::with_capacity((width * height) as usize),
        row_buf: Vec::with_capacity(width as usize),
    };

    image
        .decode(&mut collector)
        .map_err(|e| Error::Image(format!("JBIG2 pixel decode error: {e}")))?;

    Ok(ImageData::Raw {
        pixels: collector.pixels,
        format: PixelFormat::Grayscale,
    })
}

#[cfg(not(feature = "rendering"))]
fn decode_jbig2_image(
    _xobject: &crate::object::Object,
    _obj_ref: Option<ObjectRef>,
    _dict: &std::collections::HashMap<String, crate::object::Object>,
    _doc: Option<&crate::document::PdfDocument>,
    _width: u32,
    _height: u32,
) -> Result<ImageData> {
    Err(Error::UnsupportedFilter("JBIG2Decode".to_string()))
}

/// Expand abbreviated inline image dictionary keys to full names.
pub fn expand_inline_image_dict(
    dict: std::collections::HashMap<String, crate::object::Object>,
) -> std::collections::HashMap<String, crate::object::Object> {
    use std::collections::HashMap;
    let mut expanded = HashMap::new();
    for (key, value) in dict {
        let expanded_key = match key.as_str() {
            "W" => "Width",
            "H" => "Height",
            "CS" => "ColorSpace",
            "BPC" => "BitsPerComponent",
            "F" => "Filter",
            "DP" => "DecodeParms",
            "IM" => "ImageMask",
            "I" => "Interpolate",
            "D" => "Decode",
            "EF" => "EFontFile",
            "Intent" => "Intent",
            _ => &key,
        };
        expanded.insert(expanded_key.to_string(), value);
    }
    expanded
}

#[cfg(test)]
mod indexed_tests {
    use super::*;

    #[test]
    fn expand_indexed_rgb_8bpc() {
        // 2x2 image, 4 palette entries, each RGB
        let palette = vec![
            0, 0, 0, // index 0 black
            255, 0, 0, // index 1 red
            0, 255, 0, // index 2 green
            0, 0, 255, // index 3 blue
        ];
        let raw = vec![0, 1, 2, 3];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 2, 2, 8).unwrap();
        assert_eq!(out, vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn expand_indexed_gray_base_to_rgb() {
        // Base color space is Grayscale, palette is 1 byte per entry
        let palette = vec![10, 128, 255];
        let raw = vec![0, 1, 2];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::Grayscale, 3, 1, 8).unwrap();
        assert_eq!(out, vec![10, 10, 10, 128, 128, 128, 255, 255, 255]);
    }

    #[test]
    fn expand_indexed_out_of_range_index() {
        // Palette only has 2 entries but raw has index 5 → zeroed
        let palette = vec![10, 20, 30, 40, 50, 60];
        let raw = vec![0, 5];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 2, 1, 8).unwrap();
        assert_eq!(out, vec![10, 20, 30, 0, 0, 0]);
    }

    #[test]
    fn resolve_indexed_palette_truncates_to_hival() {
        use crate::object::Object;
        // [/Indexed /DeviceRGB 1 <inline palette>] — hival = 1, so 2 entries * 3 = 6 bytes.
        // Provide an oversized 12-byte palette; the extra 6 bytes must be dropped so
        // that indices > hival cannot pick up stray lookup data.
        let cs = Object::Array(vec![
            Object::Name("Indexed".to_string()),
            Object::Name("DeviceRGB".to_string()),
            Object::Integer(1),
            Object::String(vec![
                10, 20, 30, // entry 0
                40, 50, 60, // entry 1
                70, 80, 90, // stray — beyond hival
                100, 110, 120,
            ]),
        ]);
        let ir = resolve_indexed_palette(None, &cs).unwrap().unwrap();
        assert_eq!(ir.base_fmt, PixelFormat::RGB);
        assert_eq!(ir.palette, vec![10, 20, 30, 40, 50, 60]);
        assert!(ir.base_profile.is_none(), "DeviceRGB base has no ICC profile");
        let (fmt, palette) = (ir.base_fmt, ir.palette);

        // Index 2 (> hival) must now be treated as out-of-range → black pixel.
        let raw = vec![0, 1, 2];
        let out = expand_indexed_to_rgb(&raw, &palette, fmt, 3, 1, 8).unwrap();
        assert_eq!(out, vec![10, 20, 30, 40, 50, 60, 0, 0, 0]);
    }

    #[test]
    fn expand_indexed_cmyk_base_matches_cmyk_to_rgb() {
        // Palette has a single CMYK entry; expansion must match the shared helper.
        let palette = vec![64, 128, 192, 32];
        let raw = vec![0];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::CMYK, 1, 1, 8).unwrap();
        let expected = cmyk_pixel_to_rgb(64, 128, 192, 32);
        assert_eq!(out, expected.to_vec());
    }

    #[test]
    fn expand_indexed_1bpc_with_row_padding() {
        // 2-entry palette, 5x2 image at 1 bpc. 5 bits → 1 byte per row (3 bits padding).
        // Row 0 indices: 0,1,0,1,0 → top nibble 01010xxx = 0x50
        // Row 1 indices: 1,1,0,0,1 → top nibble 11001xxx = 0xC8
        let palette = vec![10, 20, 30, 200, 210, 220];
        let raw = vec![0x50, 0xC8];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 5, 2, 1).unwrap();
        assert_eq!(
            out,
            vec![
                10, 20, 30, 200, 210, 220, 10, 20, 30, 200, 210, 220, 10, 20, 30, // row 0
                200, 210, 220, 200, 210, 220, 10, 20, 30, 10, 20, 30, 200, 210, 220, // row 1
            ]
        );
    }

    #[test]
    fn expand_indexed_2bpc_with_row_padding() {
        // 4-entry palette, 3x1 image at 2 bpc. 6 bits → 1 byte per row (2 bits padding).
        // indices 0,1,2 → 00 01 10 xx → 0x18
        let palette = vec![
            0, 0, 0, // 0
            10, 20, 30, // 1
            40, 50, 60, // 2
            70, 80, 90, // 3
        ];
        let raw = vec![0x18];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 3, 1, 2).unwrap();
        assert_eq!(out, vec![0, 0, 0, 10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn expand_indexed_4bpc_packs_two_per_byte() {
        // 4x1 image, 4bpc: 2 indices per byte, high nibble first
        let palette = vec![
            0, 0, 0, // 0
            10, 20, 30, // 1
            40, 50, 60, // 2
            70, 80, 90, // 3
        ];
        // indices: 0,1,2,3 → packed: 0x01, 0x23
        let raw = vec![0x01, 0x23];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 4, 1, 4).unwrap();
        assert_eq!(out, vec![0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    // ---- DoS / hardening guards for #324 ----

    #[test]
    fn expand_indexed_rejects_overflow_dimensions() {
        // Dimensions that overflow usize when computing w * h * 3. Previously
        // Vec::with_capacity(w*h*3) would panic or reserve absurd amounts.
        let palette = vec![0, 0, 0, 255, 0, 0];
        let raw = vec![0, 1];
        let huge = u32::MAX / 2;
        let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, huge, huge, 8);
        assert!(result.is_err(), "overflow dimensions must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflow") || err.contains("exceeds"),
            "expected overflow/limit error, got: {err}"
        );
    }

    #[test]
    fn expand_indexed_rejects_truncated_stream() {
        // 10x10 8bpc image requires 100 index bytes. Supplying 10 used to
        // silently zero-pad the remaining rows; now it's an error.
        let palette = vec![10, 20, 30, 40, 50, 60];
        let raw = vec![0; 10];
        let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 10, 10, 8);
        assert!(result.is_err(), "truncated stream must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"), "expected truncated error, got: {err}");
    }

    #[test]
    fn expand_indexed_rejects_output_over_cap() {
        // 12 000 × 12 000 × 3 = 432 MB > 256 MB guard. The MAX_INDEXED_OUTPUT_BYTES
        // check fires before we inspect `raw.len()`, so the test doesn't need to
        // allocate a 144 MB stream — an empty buffer is enough to prove the cap
        // rejects the request.
        let palette = vec![0, 0, 0];
        let raw: Vec<u8> = Vec::new();
        let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 12_000, 12_000, 8);
        assert!(result.is_err(), "oversized output must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("guard limit") || err.contains("exceeds"),
            "expected output-size guard error, got: {err}"
        );
    }

    // ---- #338: bpc validation per ISO 32000-2 §8.9.5.1 ----

    #[test]
    fn expand_indexed_rejects_bpc_zero() {
        // bpc = 0 used to be coerced to 1 by `bpc.max(1)`, silently
        // accepting a malformed PDF. Now it must be rejected.
        let palette = vec![0, 0, 0, 255, 0, 0];
        let raw = vec![0xFF];
        let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 1, 1, 0);
        assert!(result.is_err(), "bpc=0 must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("BitsPerComponent") || err.contains("bpc"),
            "expected bpc error, got: {err}"
        );
    }

    #[test]
    fn expand_indexed_rejects_unsupported_bpc() {
        // 3, 5, 6, 7, 9, 12, 16, … are all invalid for Indexed. Previously
        // the `_ => 0` arm in `read_index` silently mapped every pixel to
        // palette entry 0, returning a solid-color image. Now they're
        // rejected up front.
        let palette = vec![0, 0, 0, 255, 0, 0];
        let raw = vec![0xFF];
        for bpc in [3u8, 5, 6, 7, 9, 12, 16] {
            let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 1, 1, bpc);
            assert!(result.is_err(), "bpc={bpc} must be rejected");
        }
    }

    #[test]
    fn expand_indexed_accepts_all_spec_bpc_values() {
        // Sanity: 1, 2, 4, 8 must still all work.
        let palette = vec![0, 0, 0, 255, 0, 0, 10, 20, 30, 40, 50, 60];
        let raw = vec![0xFF];
        for bpc in [1u8, 2, 4, 8] {
            let result = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 1, 1, bpc);
            assert!(result.is_ok(), "bpc={bpc} must be accepted, got {result:?}");
        }
    }

    // Regression test for #336. Per ISO 32000-1 §8.6.6.3, the lookup element of
    // `[/Indexed base hival lookup]` must be either a byte string or a stream.
    // Historical behaviour when it was neither: `resolve_indexed_palette` returned
    // `Ok(None)` and `extract_image_from_xobject` silently fell back to treating
    // the raw 1-byte/pixel index stream as 3-byte/pixel RGB, producing the
    // misleading "Invalid RGB image dimensions" error. The fix returns an
    // explicit `Error::Image("Unable to resolve Indexed color space palette")`.
    #[test]
    fn resolve_indexed_palette_array_lookup_returns_none() {
        use crate::object::Object;
        let cs = Object::Array(vec![
            Object::Name("Indexed".to_string()),
            Object::Name("DeviceRGB".to_string()),
            Object::Integer(1),
            // Lookup as Array-of-Array (not String or Stream) — unresolvable.
            Object::Array(vec![
                Object::Array(vec![Object::Integer(0), Object::Integer(0), Object::Integer(0)]),
                Object::Array(vec![
                    Object::Integer(255),
                    Object::Integer(255),
                    Object::Integer(255),
                ]),
            ]),
        ]);
        assert!(resolve_indexed_palette(None, &cs).unwrap().is_none());
    }

    #[test]
    fn extract_image_errors_when_indexed_lookup_is_array() {
        use crate::object::Object;
        use std::collections::HashMap;

        let mut dict = HashMap::new();
        dict.insert("Subtype".to_string(), Object::Name("Image".to_string()));
        dict.insert("Width".to_string(), Object::Integer(2));
        dict.insert("Height".to_string(), Object::Integer(1));
        dict.insert("BitsPerComponent".to_string(), Object::Integer(8));
        dict.insert(
            "ColorSpace".to_string(),
            Object::Array(vec![
                Object::Name("Indexed".to_string()),
                Object::Name("DeviceRGB".to_string()),
                Object::Integer(1),
                Object::Array(vec![Object::Integer(0), Object::Integer(0), Object::Integer(0)]),
            ]),
        );
        let xobject = Object::Stream {
            dict,
            data: bytes::Bytes::from_static(&[0, 1]),
        };

        let err = extract_image_from_xobject(None, &xobject, None, None)
            .expect_err("Indexed with Array lookup must error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Unable to resolve Indexed color space palette"),
            "error message should identify palette-resolution failure, got: {msg}"
        );
        assert!(
            !msg.contains("Invalid RGB image dimensions"),
            "must not fall through to misleading RGB-dimension error, got: {msg}"
        );
    }

    // #337 Lab→XYZ→sRGB conversion tests

    #[test]
    fn lab_pixel_mid_gray() {
        // Lab(50, 0, 0) = perceptual mid-gray → sRGB ~(119, 119, 119).
        // Byte encoding: L=128, a=128, b=128.
        let d65: [f64; 3] = [0.9505, 1.0, 1.0890];
        let [r, g, b] = super::lab_pixel_to_rgb(128, 128, 128, d65);
        for (label, v, expected) in [("R", r, 119), ("G", g, 119), ("B", b, 119)] {
            let diff = (v as i32 - expected).abs();
            assert!(diff <= 3, "Lab(50,0,0) {label}: expected ~{expected}, got {v} (Δ={diff})");
        }
    }

    #[test]
    fn lab_pixel_white() {
        // Lab(100, 0, 0) = white → sRGB ~(255, 255, 255).
        // Byte encoding: L=255, a=128, b=128.
        let d65: [f64; 3] = [0.9505, 1.0, 1.0890];
        let [r, g, b] = super::lab_pixel_to_rgb(255, 128, 128, d65);
        for (label, v) in [("R", r), ("G", g), ("B", b)] {
            assert!(v >= 250, "Lab(100,0,0) {label}: expected ~255, got {v}");
        }
    }

    #[test]
    fn lab_pixel_black() {
        // Lab(0, 0, 0) = black → sRGB ~(0, 0, 0).
        // Byte encoding: L=0, a=128, b=128.
        let d65: [f64; 3] = [0.9505, 1.0, 1.0890];
        let [r, g, b] = super::lab_pixel_to_rgb(0, 128, 128, d65);
        for (label, v) in [("R", r), ("G", g), ("B", b)] {
            assert!(v <= 5, "Lab(0,0,0) {label}: expected ~0, got {v}");
        }
    }

    #[test]
    fn lab_pixel_red_tint() {
        // Lab(50, 80, 0) has a strong red-magenta tint.
        // Byte encoding: L=128, a=208 (128+80), b=128.
        let d65: [f64; 3] = [0.9505, 1.0, 1.0890];
        let [r, g, b] = super::lab_pixel_to_rgb(128, 208, 128, d65);
        assert!(r > g + 50, "Lab(50,80,0) should have R >> G: R={r}, G={g}");
        assert!(r > b, "Lab(50,80,0) should have R > B: R={r}, B={b}");
    }

    #[test]
    fn lab_palette_round_trip() {
        // 3-entry Lab palette → RGB palette should have 9 bytes.
        let d65: [f64; 3] = [0.9505, 1.0, 1.0890];
        let palette: Vec<u8> = vec![
            0, 128, 128, // black
            128, 128, 128, // mid-gray
            255, 128, 128, // white
        ];
        let rgb = super::lab_palette_to_rgb(&palette, d65);
        assert_eq!(rgb.len(), 9, "3 Lab entries → 9 RGB bytes");
        // Black entry: all near 0
        assert!(rgb[0] <= 5 && rgb[1] <= 5 && rgb[2] <= 5);
        // White entry: all near 255
        assert!(rgb[6] >= 250 && rgb[7] >= 250 && rgb[8] >= 250);
    }

    #[test]
    fn extract_lab_whitepoint_d65() {
        use crate::object::Object;
        let cs = Object::Array(vec![
            Object::Name("Lab".to_string()),
            Object::Dictionary({
                let mut d = std::collections::HashMap::new();
                d.insert(
                    "WhitePoint".to_string(),
                    Object::Array(vec![
                        Object::Real(0.9505),
                        Object::Real(1.0),
                        Object::Real(1.0890),
                    ]),
                );
                d
            }),
        ]);
        let wp = super::extract_lab_whitepoint(&cs);
        assert!((wp[0] - 0.9505).abs() < 1e-6);
        assert!((wp[1] - 1.0).abs() < 1e-6);
        assert!((wp[2] - 1.0890).abs() < 1e-6);
    }

    #[test]
    fn extract_lab_whitepoint_missing_falls_back_to_d65() {
        use crate::object::Object;
        let cs = Object::Name("Lab".to_string());
        let wp = super::extract_lab_whitepoint(&cs);
        assert!((wp[0] - 0.9505).abs() < 1e-6);
    }
}
