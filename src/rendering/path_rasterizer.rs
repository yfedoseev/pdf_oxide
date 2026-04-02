//! Path rasterizer - renders PDF paths using tiny-skia.

use super::{create_fill_paint, create_stroke_paint};
use crate::content::GraphicsState;
use tiny_skia::{FillRule, LineCap, LineJoin, Path, Pixmap, Stroke, Transform};

/// Rasterizer for PDF path operations.
pub struct PathRasterizer {
    // Could hold caches, state, etc.
}

impl PathRasterizer {
    /// Create a new path rasterizer.
    pub fn new() -> Self {
        Self {}
    }

    /// Fill a path with the current fill color.
    #[allow(dead_code)]
    pub fn fill_path(
        &self,
        pixmap: &mut Pixmap,
        path: &Path,
        transform: Transform,
        gs: &GraphicsState,
        fill_rule: FillRule,
    ) {
        let paint = create_fill_paint(gs, &gs.blend_mode);
        pixmap.fill_path(path, &paint, fill_rule, transform, None);
    }

    /// Stroke a path with the current stroke color and line style.
    #[allow(dead_code)]
    pub fn stroke_path(
        &self,
        pixmap: &mut Pixmap,
        path: &Path,
        transform: Transform,
        gs: &GraphicsState,
    ) {
        let paint = create_stroke_paint(gs, &gs.blend_mode);

        let dash = if !gs.dash_pattern.0.is_empty() {
            tiny_skia::StrokeDash::new(gs.dash_pattern.0.clone(), gs.dash_pattern.1)
        } else {
            None
        };

        let stroke = Stroke {
            width: gs.line_width,
            line_cap: self.pdf_line_cap_to_skia(gs.line_cap),
            line_join: self.pdf_line_join_to_skia(gs.line_join),
            miter_limit: gs.miter_limit,
            dash,
        };

        pixmap.stroke_path(path, &paint, &stroke, transform, None);
    }

    /// Fill a path with optional clip mask.
    pub fn fill_path_clipped(
        &self,
        pixmap: &mut Pixmap,
        path: &tiny_skia::Path,
        transform: Transform,
        gs: &GraphicsState,
        fill_rule: FillRule,
        clip_mask: Option<&tiny_skia::Mask>,
    ) {
        let paint = create_fill_paint(gs, &gs.blend_mode);
        let bounds = path.bounds();
        let pixel_bounds = path.clone().transform(transform).map(|p| p.bounds());

        log::debug!(
            "PathRasterizer::fill_path: color={:?}, alpha={}, bounds={:?}, pixel_bounds={:?}",
            gs.fill_color_rgb,
            gs.fill_alpha,
            bounds,
            pixel_bounds
        );
        pixmap.fill_path(path, &paint, fill_rule, transform, clip_mask);
    }

    /// Stroke a path with optional clip mask.
    pub fn stroke_path_clipped(
        &self,
        pixmap: &mut Pixmap,
        path: &Path,
        transform: Transform,
        gs: &GraphicsState,
        clip_mask: Option<&tiny_skia::Mask>,
    ) {
        let paint = create_stroke_paint(gs, &gs.blend_mode);
        let bounds = path.bounds();
        let pixel_bounds = path.clone().transform(transform).map(|p| p.bounds());
        log::debug!(
            "PathRasterizer::stroke_path: color={:?}, alpha={}, bounds={:?}, pixel_bounds={:?}",
            gs.stroke_color_rgb,
            gs.stroke_alpha,
            bounds,
            pixel_bounds
        );

        let dash = if !gs.dash_pattern.0.is_empty() {
            tiny_skia::StrokeDash::new(gs.dash_pattern.0.clone(), gs.dash_pattern.1)
        } else {
            None
        };

        let stroke = Stroke {
            width: gs.line_width,
            line_cap: self.pdf_line_cap_to_skia(gs.line_cap),
            line_join: self.pdf_line_join_to_skia(gs.line_join),
            miter_limit: gs.miter_limit,
            dash,
        };

        pixmap.stroke_path(path, &paint, &stroke, transform, clip_mask);
    }

    /// Convert PDF line cap style to tiny-skia.
    fn pdf_line_cap_to_skia(&self, cap: u8) -> LineCap {
        match cap {
            0 => LineCap::Butt,
            1 => LineCap::Round,
            2 => LineCap::Square,
            _ => LineCap::Butt,
        }
    }

    /// Convert PDF line join style to tiny-skia.
    fn pdf_line_join_to_skia(&self, join: u8) -> LineJoin {
        match join {
            0 => LineJoin::Miter,
            1 => LineJoin::Round,
            2 => LineJoin::Bevel,
            _ => LineJoin::Miter,
        }
    }
}

impl Default for PathRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_rasterizer_new() {
        let rasterizer = PathRasterizer::new();
        // Just verify it can be created
        assert_eq!(rasterizer.pdf_line_cap_to_skia(0), LineCap::Butt);
    }

    #[test]
    fn test_line_cap_conversion() {
        let rasterizer = PathRasterizer::new();
        assert_eq!(rasterizer.pdf_line_cap_to_skia(0), LineCap::Butt);
        assert_eq!(rasterizer.pdf_line_cap_to_skia(1), LineCap::Round);
        assert_eq!(rasterizer.pdf_line_cap_to_skia(2), LineCap::Square);
        assert_eq!(rasterizer.pdf_line_cap_to_skia(99), LineCap::Butt); // Unknown defaults to Butt
    }

    #[test]
    fn test_line_join_conversion() {
        let rasterizer = PathRasterizer::new();
        assert_eq!(rasterizer.pdf_line_join_to_skia(0), LineJoin::Miter);
        assert_eq!(rasterizer.pdf_line_join_to_skia(1), LineJoin::Round);
        assert_eq!(rasterizer.pdf_line_join_to_skia(2), LineJoin::Bevel);
    }
}
