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

    /// Save the image as PNG format.
    pub fn save_as_png(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => save_jpeg_as_png(jpeg_data, path),
            ImageData::Raw { pixels, format } => {
                save_raw_as_png(pixels, self.width, self.height, *format, path)
            },
        }
    }

    /// Save the image as JPEG format.
    pub fn save_as_jpeg(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.data {
            ImageData::Jpeg(jpeg_data) => std::fs::write(path, jpeg_data).map_err(Error::from),
            ImageData::Raw { pixels, format } => {
                save_raw_as_jpeg(pixels, self.width, self.height, *format, path)
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
                    encoder
                        .write_image(pixels, self.width, self.height, image::ColorType::L8)
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                } else if *format == PixelFormat::RGB && pixels.len() == expected_rgb {
                    encoder
                        .write_image(pixels, self.width, self.height, image::ColorType::Rgb8)
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                } else {
                    let dynamic_image = self.to_dynamic_image()?;
                    let rgb = dynamic_image.to_rgb8();
                    encoder
                        .write_image(rgb.as_raw(), self.width, self.height, image::ColorType::Rgb8)
                        .map_err(|e| Error::Encode(format!("Failed to encode PNG: {}", e)))?;
                }
            },
            ImageData::Jpeg(_) => {
                let dynamic_image = self.to_dynamic_image()?;
                let rgb = dynamic_image.to_rgb8();
                encoder
                    .write_image(rgb.as_raw(), self.width, self.height, image::ColorType::Rgb8)
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
                                PixelFormat::CMYK => cmyk_to_rgb(pixels),
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
    mut doc: Option<&mut crate::document::PdfDocument>,
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

    let resolved_color_space = if let Some(ref mut d) = doc {
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

    let color_space = parse_color_space(&resolved_color_space)?;

    // For Indexed color spaces, resolve the base color space and palette now so we
    // can expand indices to RGB after decoding the stream. Without this, raw
    // Indexed pixel data (1 byte per pixel) is mislabelled as RGB (3 bytes per
    // pixel) and ImageBuffer::from_raw rejects the wrong length.
    let indexed_palette: Option<(PixelFormat, Vec<u8>)> =
        if color_space == ColorSpace::Indexed {
            resolve_indexed_palette(doc.as_deref_mut(), &resolved_color_space)?
        } else {
            None
        };

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

    let mut ccitt_params_override: Option<crate::decoders::CcittParams> = None;
    if (filter_names.contains(&"JBIG2Decode".to_string())
        || filter_names.contains(&"Jbig2Decode".to_string()))
        && bits_per_component == 1
    {
        let mut ccitt_params =
            crate::object::extract_ccitt_params_with_width(dict.get("DecodeParms"), Some(width));

        if let Some(ref mut params) = ccitt_params {
            if params.rows.is_none() {
                params.rows = Some(height);
            }
            ccitt_params_override = ccitt_params;
        }
    }

    let data = if is_jpeg_only || is_jpeg_chain {
        let decoded = if let (Some(d), Some(ref_id)) = (doc.as_mut(), obj_ref) {
            d.decode_stream_with_encryption(xobject, ref_id)?
        } else {
            xobject.decode_stream_data()?
        };
        ImageData::Jpeg(decoded)
    } else if ccitt_params_override.is_some() {
        match xobject {
            Object::Stream { data, .. } => ImageData::Raw {
                pixels: data.to_vec(),
                format: PixelFormat::Grayscale,
            },
            _ => return Err(Error::Image("XObject is not a stream".to_string())),
        }
    } else {
        let decoded_data = if let (Some(d), Some(ref_id)) = (doc.as_mut(), obj_ref) {
            d.decode_stream_with_encryption(xobject, ref_id)?
        } else {
            xobject.decode_stream_data()?
        };
        if let Some((base_fmt, palette)) = indexed_palette.as_ref() {
            let expanded = expand_indexed_to_rgb(
                &decoded_data,
                palette,
                *base_fmt,
                width,
                height,
                bits_per_component,
            );
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

    let mut image = PdfImage::new(width, height, color_space, bits_per_component, data);

    if let Some(ccitt_params) = ccitt_params_override {
        image.set_ccitt_params(ccitt_params);
    } else if bits_per_component == 1 && image.color_space == ColorSpace::DeviceGray {
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

/// Resolve an Indexed color space's base color space and palette lookup bytes.
///
/// PDF Indexed color spaces are `[/Indexed base hival lookup]` where `lookup`
/// is either a byte string or a stream of `(hival + 1) * N` bytes (N = number
/// of components in the base color space).
fn resolve_indexed_palette(
    mut doc: Option<&mut crate::document::PdfDocument>,
    cs_obj: &crate::object::Object,
) -> Result<Option<(PixelFormat, Vec<u8>)>> {
    use crate::object::Object;

    let Object::Array(arr) = cs_obj else {
        return Ok(None);
    };
    if arr.len() < 4 {
        return Ok(None);
    }

    let base_obj = if let Some(ref mut d) = doc {
        if let Some(r) = arr[1].as_reference() {
            d.load_object(r)?
        } else {
            arr[1].clone()
        }
    } else {
        arr[1].clone()
    };
    let base_cs = parse_color_space(&base_obj)?;
    let base_fmt = color_space_to_pixel_format(&base_cs);

    let lookup_obj = if let Some(ref mut d) = doc {
        if let Some(r) = arr[3].as_reference() {
            d.load_object(r)?
        } else {
            arr[3].clone()
        }
    } else {
        arr[3].clone()
    };
    let palette_bytes = match &lookup_obj {
        Object::String(s) => s.clone(),
        Object::Stream { .. } => lookup_obj.decode_stream_data()?,
        _ => return Ok(None),
    };
    if palette_bytes.is_empty() {
        return Ok(None);
    }
    Ok(Some((base_fmt, palette_bytes)))
}

/// Expand packed Indexed image indices into RGB bytes using the palette.
///
/// Supports 1, 2, 4, and 8 bit-per-component index streams. Rows are padded
/// to byte boundaries per the PDF spec.
fn expand_indexed_to_rgb(
    raw: &[u8],
    palette: &[u8],
    base_fmt: PixelFormat,
    width: u32,
    height: u32,
    bpc: u8,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = base_fmt.bytes_per_pixel();
    let bpc = bpc.max(1);
    let bytes_per_row = (w * bpc as usize + 7) / 8;
    let mut out = Vec::with_capacity(w * h * 3);

    let read_index = |row: &[u8], x: usize| -> usize {
        match bpc {
            8 => row.get(x).copied().unwrap_or(0) as usize,
            4 => {
                let byte_idx = x / 2;
                let b = row.get(byte_idx).copied().unwrap_or(0);
                if x % 2 == 0 { (b >> 4) as usize } else { (b & 0x0F) as usize }
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
            _ => 0,
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
                    let c = palette[off] as u32;
                    let m = palette[off + 1] as u32;
                    let y_c = palette[off + 2] as u32;
                    let k = palette[off + 3] as u32;
                    out.push(((255 - c) * (255 - k) / 255) as u8);
                    out.push(((255 - m) * (255 - k) / 255) as u8);
                    out.push(((255 - y_c) * (255 - k) / 255) as u8);
                },
            }
        }
    }
    out
}

/// Convert CMYK pixel bytes to RGB.
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

fn save_jpeg_as_png(jpeg_data: &[u8], path: impl AsRef<Path>) -> Result<()> {
    use image::ImageFormat;
    let img = image::load_from_memory_with_format(jpeg_data, ImageFormat::Jpeg)
        .map_err(|e| Error::Image(format!("Failed to decode JPEG: {}", e)))?;
    img.save_with_format(path, ImageFormat::Png)
        .map_err(|e| Error::Image(format!("Failed to save PNG: {}", e)))
}

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
            let rgb = cmyk_to_rgb(pixels);
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
            let rgb = cmyk_to_rgb(pixels);
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                .ok_or_else(|| Error::Image("Invalid CMYK image dimensions".to_string()))?;
            img.save_with_format(path, ImageFormat::Jpeg)
                .map_err(|e| Error::Image(format!("Failed to save JPEG: {}", e)))
        },
    }
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
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 2, 2, 8);
        assert_eq!(out, vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn expand_indexed_gray_base_to_rgb() {
        // Base color space is Grayscale, palette is 1 byte per entry
        let palette = vec![10, 128, 255];
        let raw = vec![0, 1, 2];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::Grayscale, 3, 1, 8);
        assert_eq!(out, vec![10, 10, 10, 128, 128, 128, 255, 255, 255]);
    }

    #[test]
    fn expand_indexed_out_of_range_index() {
        // Palette only has 2 entries but raw has index 5 → zeroed
        let palette = vec![10, 20, 30, 40, 50, 60];
        let raw = vec![0, 5];
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 2, 1, 8);
        assert_eq!(out, vec![10, 20, 30, 0, 0, 0]);
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
        let out = expand_indexed_to_rgb(&raw, &palette, PixelFormat::RGB, 4, 1, 4);
        assert_eq!(out, vec![0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }
}
