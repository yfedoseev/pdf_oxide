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
fn pdf_blend_mode_to_skia(mode: &str) -> tiny_skia::BlendMode {
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

/// Render a rectangular region of a PDF page to an image.
///
/// This is a convenience function that creates a PageRenderer and renders
/// a region of a single page. The region is specified in PDF coordinate
/// space (points, origin at bottom-left).
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page_num` - Zero-based page number
/// * `region` - Rectangle in PDF coordinates to render
/// * `options` - Rendering options (DPI, format, etc.)
///
/// # Returns
///
/// The rendered image as bytes in the specified format.
pub fn render_region(
    doc: &mut crate::document::PdfDocument,
    page_num: usize,
    region: &crate::geometry::Rect,
    options: &RenderOptions,
) -> Result<RenderedImage> {
    let mut renderer = PageRenderer::new(options.clone());
    renderer.render_region(doc, page_num, region)
}
