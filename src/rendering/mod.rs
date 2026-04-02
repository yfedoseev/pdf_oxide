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
    doc: &mut crate::document::PdfDocument,
    page_num: usize,
    options: &RenderOptions,
) -> Result<RenderedImage> {
    let mut renderer = PageRenderer::new(options.clone());
    renderer.render_page(doc, page_num)
}

/// Create a flattened PDF where each page is rendered as an image.
///
/// This "burns in" all annotations, form fields, overlays, and text into
/// a flat raster representation. Useful for redaction, archival, or
/// ensuring consistent visual output across viewers.
///
/// Returns the flattened PDF as bytes.
pub fn flatten_to_images(doc: &mut crate::document::PdfDocument, dpi: u32) -> Result<Vec<u8>> {
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
