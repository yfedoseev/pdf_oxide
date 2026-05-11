//! Page rendering module for converting PDF pages to images.
//!
//! This module provides functionality to render PDF pages to raster images
//! using the pure-Rust `tiny-skia` library.
//!
//! ## Features
//!
//! - Render pages to PNG/JPEG images
//! - Configurable DPI and image quality
//! - Support for text, paths, and images
//! - Transparency and blend modes
//!
//! ## Example
//!
//! ```ignore
//! use pdf_oxide::api::Pdf;
//! use pdf_oxide::rendering::{RenderOptions, ImageFormat};
//!
//! let mut pdf = Pdf::open("document.pdf")?;
//! let image = pdf.render_page(0, &RenderOptions::default())?;
//! image.save("page1.png")?;
//! ```
//!
//! ## Architecture
//!
//! The rendering pipeline:
//!
//! 1. Parse page content stream into operators
//! 2. Execute operators against graphics state machine
//! 3. Rasterize paths, text, and images to tiny-skia pixmap
//! 4. Convert to output format (PNG/JPEG)

mod page_renderer;
mod path_rasterizer;
mod text_rasterizer;

pub use page_renderer::{ImageFormat, PageRenderer, RenderOptions, RenderedImage};

use crate::content::GraphicsState;
use crate::error::Result;
use tiny_skia::{Color, Paint};

/// Create a Paint configured for fill operations from graphics state.
pub(crate) fn create_fill_paint(gs: &GraphicsState, blend_mode: &str) -> Paint<'static> {
    let (r, g, b) = gs.fill_color_rgb;
    let mut paint = Paint::default();

    // Note: render_mode == 3 (invisible text) is handled in the text rendering path,
    // not here, since this paint is also used for non-text fills (paths, shapes).
    paint.set_color(Color::from_rgba(r, g, b, gs.fill_alpha).unwrap_or(Color::BLACK));

    paint.anti_alias = true;

    if blend_mode != "Normal" {
        paint.blend_mode = pdf_blend_mode_to_skia(blend_mode);
    }

    paint
}

/// Create a Paint configured for stroke operations from graphics state.
pub(crate) fn create_stroke_paint(gs: &GraphicsState, blend_mode: &str) -> Paint<'static> {
    let (r, g, b) = gs.stroke_color_rgb;
    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba(r, g, b, gs.stroke_alpha).unwrap_or(Color::BLACK));
    paint.anti_alias = true;

    if blend_mode != "Normal" {
        paint.blend_mode = pdf_blend_mode_to_skia(blend_mode);
    }

    paint
}

/// Convert PDF blend mode to tiny-skia.
pub(crate) fn pdf_blend_mode_to_skia(mode: &str) -> tiny_skia::BlendMode {
    match mode {
        "Normal" => tiny_skia::BlendMode::SourceOver,
        "Multiply" => tiny_skia::BlendMode::Multiply,
        "Screen" => tiny_skia::BlendMode::Screen,
        "Overlay" => tiny_skia::BlendMode::Overlay,
        "Darken" => tiny_skia::BlendMode::Darken,
        "Lighten" => tiny_skia::BlendMode::Lighten,
        "ColorDodge" => tiny_skia::BlendMode::ColorDodge,
        "ColorBurn" => tiny_skia::BlendMode::ColorBurn,
        "HardLight" => tiny_skia::BlendMode::HardLight,
        "SoftLight" => tiny_skia::BlendMode::SoftLight,
        "Difference" => tiny_skia::BlendMode::Difference,
        "Exclusion" => tiny_skia::BlendMode::Exclusion,
        _ => tiny_skia::BlendMode::SourceOver,
    }
}

/// Render a PDF page to an image.
///
/// This is a convenience function that creates a PageRenderer and renders
/// a single page.
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page_num` - Zero-based page number
/// * `options` - Rendering options (DPI, format, etc.)
///
/// # Returns
///
/// The rendered image as bytes in the specified format.
pub fn render_page(
    doc: &crate::document::PdfDocument,
    page_num: usize,
    options: &RenderOptions,
) -> Result<RenderedImage> {
    let mut renderer = PageRenderer::new(options.clone());
    renderer.render_page(doc, page_num)
}

/// Render a rectangular region of a page. `crop_rect_pt` is in PDF
/// user-space points (origin bottom-left of the page). The crop is
/// applied to the fully-rendered image at the requested DPI.
pub fn render_page_region(
    doc: &crate::document::PdfDocument,
    page_num: usize,
    crop_rect_pt: (f32, f32, f32, f32),
    options: &RenderOptions,
) -> Result<RenderedImage> {
    // Full-page render first — the crop is a post-process on the
    // resulting raster. Wasteful if the crop is tiny, but matches
    // the semantics of every PDF viewer and avoids a parallel
    // clipped-raster code path in tiny-skia.
    let full = render_page(doc, page_num, options)?;

    let (crop_x_pt, crop_y_pt, crop_w_pt, crop_h_pt) = crop_rect_pt;
    if crop_w_pt <= 0.0 || crop_h_pt <= 0.0 {
        return Err(crate::Error::InvalidPdf(format!("invalid crop rect: {crop_rect_pt:?}")));
    }

    let media = doc.get_page_media_box(page_num)?;
    let page_h_pt = media.3 - media.1;

    // Points → pixels at the render DPI.
    let scale = options.dpi as f32 / 72.0;
    let crop_x_px = (crop_x_pt * scale).round().max(0.0) as u32;
    // Image Y is top-left origin; PDF Y is bottom-left. Flip.
    let top_y_pt = page_h_pt - (crop_y_pt + crop_h_pt);
    let crop_y_px = (top_y_pt * scale).round().max(0.0) as u32;
    let crop_w_px = (crop_w_pt * scale).round().max(1.0) as u32;
    let crop_h_px = (crop_h_pt * scale).round().max(1.0) as u32;

    // Decode, crop, re-encode using the `image` crate (already a dep).
    let full_img = image::load_from_memory(&full.data)
        .map_err(|e| crate::Error::InvalidPdf(format!("render output decode: {e}")))?;
    let x = crop_x_px.min(full_img.width().saturating_sub(1));
    let y = crop_y_px.min(full_img.height().saturating_sub(1));
    let w = crop_w_px.min(full_img.width() - x);
    let h = crop_h_px.min(full_img.height() - y);
    let cropped = full_img.crop_imm(x, y, w, h);

    let mut buf = Vec::new();
    match options.format {
        ImageFormat::Jpeg => {
            use image::ImageEncoder;
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut buf,
                options.jpeg_quality.clamp(1, 100),
            );
            encoder
                .write_image(cropped.as_bytes(), w, h, cropped.color().into())
                .map_err(|e| crate::Error::InvalidPdf(format!("jpeg encode: {e}")))?;
        },
        _ => {
            use image::codecs::png::{CompressionType, FilterType, PngEncoder};
            use image::ImageEncoder;
            PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, FilterType::Sub)
                .write_image(cropped.as_bytes(), w, h, cropped.color().into())
                .map_err(|e| crate::Error::InvalidPdf(format!("png encode: {e}")))?;
        },
    }
    Ok(RenderedImage {
        data: buf,
        width: w,
        height: h,
        format: full.format,
    })
}

/// Render a page to fit inside a target bounding box (in pixels),
/// preserving aspect ratio. Picks the DPI that makes the larger of
/// the two page dimensions match the smaller bounding-box side.
pub fn render_page_fit(
    doc: &crate::document::PdfDocument,
    page_num: usize,
    fit_w_px: u32,
    fit_h_px: u32,
    options: &RenderOptions,
) -> Result<RenderedImage> {
    if fit_w_px == 0 || fit_h_px == 0 {
        return Err(crate::Error::InvalidPdf("fit width/height must be positive".into()));
    }
    let page_info = doc.get_page_info(page_num)?;
    let rotation = page_info.rotation % 360;
    let (page_w_pt, page_h_pt) = if rotation == 90 || rotation == 270 {
        (page_info.media_box.height.max(1.0), page_info.media_box.width.max(1.0))
    } else {
        (page_info.media_box.width.max(1.0), page_info.media_box.height.max(1.0))
    };

    // Compute scale as a float ratio to avoid integer-DPI quantization (issue #480).
    let scale = (fit_w_px as f32 / page_w_pt).min(fit_h_px as f32 / page_h_pt);
    let mut opts = options.clone();
    opts.scale_override = Some(scale);
    render_page(doc, page_num, &opts)
}

/// Create a flattened PDF where each page is rendered as an image.
///
/// This "burns in" all annotations, form fields, overlays, and text into
/// a flat raster representation. Useful for redaction, archival, or
/// ensuring consistent visual output across viewers.
///
/// Returns the flattened PDF as bytes.
pub fn flatten_to_images(doc: &crate::document::PdfDocument, dpi: u32) -> Result<Vec<u8>> {
    let page_count = doc.page_count()?;
    let options = RenderOptions::with_dpi(dpi);

    // Render each page to PNG
    let tmp_dir = std::env::temp_dir().join(format!("pdf_oxide_flatten_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;

    let mut paths: Vec<String> = Vec::new();
    for page_idx in 0..page_count {
        let mut renderer = PageRenderer::new(options.clone());
        let rendered = renderer.render_page(doc, page_idx)?;
        let path = tmp_dir.join(format!("page_{}.png", page_idx));
        std::fs::write(&path, &rendered.data)?;
        paths.push(path.to_string_lossy().to_string());
    }

    // Build a new PDF from the rendered images
    let pdf = crate::api::Pdf::from_images(&paths)?;
    let bytes = pdf.into_bytes();

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(bytes)
}
