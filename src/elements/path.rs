//! Path/vector graphics content element types.
//!
//! This module provides the `PathContent` type for representing
//! vector graphics in PDFs.

use crate::extractors::text::ArtifactType;
use crate::geometry::Rect;
use crate::layout::Color;

/// Vector path content that can be extracted from or written to a PDF.
///
/// This represents vector graphics such as lines, curves, and shapes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PathContent {
    /// Bounding box of the path
    pub bbox: Rect,
    /// Path operations
    pub operations: Vec<PathOperation>,
    /// Stroke color (None for no stroke)
    pub stroke_color: Option<Color>,
    /// Fill color (None for no fill)
    pub fill_color: Option<Color>,
    /// Stroke width in points
    pub stroke_width: f32,
    /// Line cap style
    pub line_cap: LineCap,
    /// Line join style
    pub line_join: LineJoin,
    /// Optional dash pattern: `Some((dashes, phase))` emits a
    /// `[dashes...] phase d` operator before stroking. `dashes` is on/
    /// off lengths in points (e.g. `[3.0, 2.0]` = dash 3 pt, gap 2 pt,
    /// repeating); `phase` is the starting offset. `None` leaves the
    /// line solid.
    #[serde(default)]
    pub dash_pattern: Option<(Vec<f32>, f32)>,
    /// Optional 2D affine transform in PDF row order `[a b c d e f]`.
    /// When set, the path is wrapped in `q ... cm ... Q` on emission
    /// so graphics-state stays scoped. Populated by
    /// `FluentPageBuilder::{rotated, scaled, translated, with_transform}`
    /// closures — v0.3.39 (text-only) extended here to cover paths.
    /// #393 Bundle A-2 follow-up.
    #[serde(default)]
    pub matrix: Option<[f32; 6]>,
    /// Reading order index
    pub reading_order: Option<usize>,
    /// When set, this path is wrapped in `/Artifact <</Type /T>>  BDC … EMC`
    /// markers so accessibility tools can skip it. Useful for decorative
    /// separator lines (footnote rules, header/footer rules).
    #[serde(skip)]
    pub artifact_type: Option<ArtifactType>,
    /// Optional Content Group (PDF "layer") name resolved from the
    /// surrounding `BDC /OC … EMC` markers in the content stream. Set
    /// during path extraction by `PathExtractor` when an OCG is active;
    /// `None` for paths emitted outside any `/OC`-tagged marked-content
    /// region or in PDFs that do not declare optional content.
    ///
    /// The string is the human-readable `/Name` entry of the referenced
    /// `OptionalContentGroup`, e.g. `"A-GRID"`, `"S-COLS"`, `"A-WALL-DIM"`
    /// for PDFs exported from Revit/AutoCAD with layer metadata intact.
    /// Reference: ISO 32000-1:2008 §8.11 (Optional Content) + §14.6
    /// (Marked Content).
    #[serde(default)]
    pub layer: Option<String>,
}

impl PathContent {
    /// Create a new empty path content element.
    pub fn new(bbox: Rect) -> Self {
        Self {
            bbox,
            operations: Vec::new(),
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            reading_order: None,
            artifact_type: None,
            layer: None,
        }
    }

    /// Create a path from operations.
    pub fn from_operations(operations: Vec<PathOperation>) -> Self {
        let bbox = Self::compute_bbox(&operations);
        Self {
            bbox,
            operations,
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            reading_order: None,
            artifact_type: None,
            layer: None,
        }
    }

    /// Set stroke color.
    pub fn with_stroke(mut self, color: Color) -> Self {
        self.stroke_color = Some(color);
        self
    }

    /// Set fill color.
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill_color = Some(color);
        self
    }

    /// Set stroke width.
    pub fn with_stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set reading order.
    pub fn with_reading_order(mut self, order: usize) -> Self {
        self.reading_order = Some(order);
        self
    }

    /// Set the Optional Content Group (PDF "layer") name. Used by
    /// `PathExtractor` while walking the content stream to attach the
    /// active OCG name to each extracted path.
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = Some(layer.into());
        self
    }

    /// Add a path operation.
    pub fn push(&mut self, op: PathOperation) {
        self.operations.push(op);
    }

    /// Check if this path has a stroke.
    pub fn has_stroke(&self) -> bool {
        self.stroke_color.is_some() && self.stroke_width > 0.0
    }

    /// Check if this path has a fill.
    pub fn has_fill(&self) -> bool {
        self.fill_color.is_some()
    }

    /// Check if this path represents a single straight line (v0.3.14).
    ///
    /// A path is a straight line if it has exactly 2 operations:
    /// MoveTo followed by LineTo.
    pub fn is_straight_line(&self) -> bool {
        (self.operations.len() == 2
            && matches!(self.operations[0], PathOperation::MoveTo(_, _))
            && matches!(self.operations[1], PathOperation::LineTo(_, _)))
            || (self.operations.len() == 3
                && matches!(self.operations[0], PathOperation::MoveTo(_, _))
                && matches!(self.operations[1], PathOperation::LineTo(_, _))
                && matches!(self.operations[2], PathOperation::ClosePath))
    }

    /// Check if this path is a horizontal line within a tolerance (v0.3.16).
    pub fn is_horizontal_line(&self, tolerance: f32) -> bool {
        (self.is_straight_line() && self.bbox.height.abs() < tolerance)
            || (self.is_rectangle() && self.bbox.height.abs() < tolerance)
    }

    /// Check if this path is a vertical line within a tolerance (v0.3.16).
    pub fn is_vertical_line(&self, tolerance: f32) -> bool {
        (self.is_straight_line() && self.bbox.width.abs() < tolerance)
            || (self.is_rectangle() && self.bbox.width.abs() < tolerance)
    }

    /// Check if this path's bounding box is nearly touching another (v0.3.16).
    pub fn is_nearly_touching(&self, other: &Rect, tolerance: f32) -> bool {
        let expanded = Rect::new(
            self.bbox.x - tolerance,
            self.bbox.y - tolerance,
            self.bbox.width + 2.0 * tolerance,
            self.bbox.height + 2.0 * tolerance,
        );
        expanded.intersects(other)
    }

    /// Check if this path represents a single rectangle (v0.3.14).
    ///
    /// A path is a rectangle if it has exactly 1 operation: Rectangle,
    /// or if it has 5 operations: MoveTo, 3x LineTo, ClosePath that form a rectangle.
    pub fn is_rectangle(&self) -> bool {
        // Case 1: Simple Rectangle operator (re)
        if self.operations.len() == 1
            && matches!(self.operations[0], PathOperation::Rectangle(_, _, _, _))
        {
            return true;
        }

        // Case 2: MoveTo + 3x LineTo + (Optional ClosePath)
        // Must be axis-aligned. We check that consecutive points share X or Y.
        if (self.operations.len() == 5 && matches!(self.operations[4], PathOperation::ClosePath))
            || (self.operations.len() == 4)
        {
            if let (
                PathOperation::MoveTo(x0, y0),
                PathOperation::LineTo(x1, y1),
                PathOperation::LineTo(x2, y2),
                PathOperation::LineTo(x3, y3),
            ) = (
                &self.operations[0],
                &self.operations[1],
                &self.operations[2],
                &self.operations[3],
            ) {
                let tol = 0.1;
                // Check if p0..p3 form 3 sides of an axis-aligned rect
                let side1 = ((x0 - x1).abs() < tol) || ((y0 - y1).abs() < tol);
                let side2 = ((x1 - x2).abs() < tol) || ((y1 - y2).abs() < tol);
                let side3 = ((x2 - x3).abs() < tol) || ((y2 - y3).abs() < tol);

                return side1 && side2 && side3;
            }
        }

        false
    }

    /// Check if this path is "box-like" or "line-like" based on its dimensions (v0.3.16).
    /// This is a fuzzy heuristic for table detection.
    pub fn is_table_primitive(&self) -> bool {
        let w = self.bbox.width.abs();
        let h = self.bbox.height.abs();

        // Very thin horizontal or vertical line
        if (w > 5.0 && h < 2.0) || (h > 5.0 && w < 2.0) {
            return true;
        }

        // Rectangular-ish box (not too small, not too large)
        if w > 5.0 && h > 5.0 && w < 1000.0 && h < 1000.0 {
            return true;
        }

        false
    }

    // === Convenience Constructors ===

    /// Create a line path from (x1, y1) to (x2, y2).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let line = PathContent::line(10.0, 10.0, 100.0, 100.0);
    /// ```
    pub fn line(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let ops = vec![PathOperation::MoveTo(x1, y1), PathOperation::LineTo(x2, y2)];
        Self::from_operations(ops)
    }

    /// Create a rectangle path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rect = PathContent::rect(10.0, 10.0, 100.0, 50.0);
    /// ```
    pub fn rect(x: f32, y: f32, width: f32, height: f32) -> Self {
        let ops = vec![PathOperation::Rectangle(x, y, width, height)];
        Self::from_operations(ops)
    }

    /// Create an approximate circle path using Bezier curves.
    ///
    /// Uses 4 cubic Bezier curves to approximate a circle.
    /// The approximation uses the constant k = 0.5522847498 for control points.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let circle = PathContent::circle(100.0, 100.0, 50.0);
    /// ```
    pub fn circle(cx: f32, cy: f32, radius: f32) -> Self {
        // Magic constant for approximating a quarter circle with a cubic Bezier
        // k = 4 * (sqrt(2) - 1) / 3 ≈ 0.5522847498
        const K: f32 = 0.552_284_8;
        let k = radius * K;

        let ops = vec![
            // Start at top
            PathOperation::MoveTo(cx, cy + radius),
            // Top-right quadrant
            PathOperation::CurveTo(cx + k, cy + radius, cx + radius, cy + k, cx + radius, cy),
            // Bottom-right quadrant
            PathOperation::CurveTo(cx + radius, cy - k, cx + k, cy - radius, cx, cy - radius),
            // Bottom-left quadrant
            PathOperation::CurveTo(cx - k, cy - radius, cx - radius, cy - k, cx - radius, cy),
            // Top-left quadrant
            PathOperation::CurveTo(cx - radius, cy + k, cx - k, cy + radius, cx, cy + radius),
            PathOperation::ClosePath,
        ];
        Self::from_operations(ops)
    }

    /// Create a rounded rectangle path.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of the bottom-left corner
    /// * `y` - Y coordinate of the bottom-left corner
    /// * `width` - Width of the rectangle
    /// * `height` - Height of the rectangle
    /// * `radius` - Corner radius (clamped to min(width, height) / 2)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rounded = PathContent::rounded_rect(10.0, 10.0, 100.0, 50.0, 5.0);
    /// ```
    pub fn rounded_rect(x: f32, y: f32, width: f32, height: f32, radius: f32) -> Self {
        // Clamp radius to maximum valid value
        let max_radius = width.min(height) / 2.0;
        let r = radius.min(max_radius).max(0.0);

        if r <= 0.0 {
            return Self::rect(x, y, width, height);
        }

        // Magic constant for approximating a quarter circle with a cubic Bezier
        const K: f32 = 0.552_284_8;
        let k = r * K;

        let x_right = x + width;
        let y_top = y + height;

        let ops = vec![
            // Start at bottom-left corner, right of curve
            PathOperation::MoveTo(x + r, y),
            // Bottom edge
            PathOperation::LineTo(x_right - r, y),
            // Bottom-right corner curve
            PathOperation::CurveTo(x_right - r + k, y, x_right, y + r - k, x_right, y + r),
            // Right edge
            PathOperation::LineTo(x_right, y_top - r),
            // Top-right corner curve
            PathOperation::CurveTo(
                x_right,
                y_top - r + k,
                x_right - r + k,
                y_top,
                x_right - r,
                y_top,
            ),
            // Top edge
            PathOperation::LineTo(x + r, y_top),
            // Top-left corner curve
            PathOperation::CurveTo(x + r - k, y_top, x, y_top - r + k, x, y_top - r),
            // Left edge
            PathOperation::LineTo(x, y + r),
            // Bottom-left corner curve
            PathOperation::CurveTo(x, y + r - k, x + r - k, y, x + r, y),
            PathOperation::ClosePath,
        ];
        Self::from_operations(ops)
    }

    /// Compute bounding box from path operations.
    fn compute_bbox(operations: &[PathOperation]) -> Rect {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for op in operations {
            match op {
                PathOperation::MoveTo(x, y) | PathOperation::LineTo(x, y) => {
                    min_x = min_x.min(*x);
                    min_y = min_y.min(*y);
                    max_x = max_x.max(*x);
                    max_y = max_y.max(*y);
                },
                PathOperation::CurveTo(x1, y1, x2, y2, x3, y3) => {
                    for (x, y) in [(*x1, *y1), (*x2, *y2), (*x3, *y3)] {
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }
                },
                PathOperation::Rectangle(x, y, w, h) => {
                    min_x = min_x.min(*x);
                    min_y = min_y.min(*y);
                    max_x = max_x.max(*x + *w);
                    max_y = max_y.max(*y + *h);
                },
                PathOperation::ClosePath => {},
            }
        }

        if min_x == f32::MAX {
            Rect::new(0.0, 0.0, 0.0, 0.0)
        } else {
            Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
        }
    }
}

impl Default for PathContent {
    fn default() -> Self {
        Self {
            bbox: Rect::new(0.0, 0.0, 0.0, 0.0),
            operations: Vec::new(),
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            reading_order: None,
            artifact_type: None,
            layer: None,
        }
    }
}

/// A single path operation.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub enum PathOperation {
    /// Move to a point (m operator)
    MoveTo(f32, f32),
    /// Line to a point (l operator)
    LineTo(f32, f32),
    /// Bezier curve to a point (c operator)
    /// (control1_x, control1_y, control2_x, control2_y, end_x, end_y)
    CurveTo(f32, f32, f32, f32, f32, f32),
    /// Rectangle (re operator)
    /// (x, y, width, height)
    Rectangle(f32, f32, f32, f32),
    /// Close the current path (h operator)
    ClosePath,
}

/// Line cap style for strokes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum LineCap {
    /// Butt cap - line ends exactly at endpoint
    #[default]
    Butt,
    /// Round cap - semicircle at endpoint
    Round,
    /// Square cap - half square at endpoint
    Square,
}

/// Line join style for strokes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum LineJoin {
    /// Miter join - sharp corner
    #[default]
    Miter,
    /// Round join - circular arc
    Round,
    /// Bevel join - diagonal corner
    Bevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_content_creation() {
        let path = PathContent::new(Rect::new(0.0, 0.0, 100.0, 100.0))
            .with_stroke(Color::black())
            .with_stroke_width(2.0);

        assert!(path.has_stroke());
        assert!(!path.has_fill());
        assert_eq!(path.stroke_width, 2.0);
    }

    #[test]
    fn test_path_from_operations() {
        let ops = vec![
            PathOperation::MoveTo(10.0, 10.0),
            PathOperation::LineTo(50.0, 10.0),
            PathOperation::LineTo(50.0, 50.0),
            PathOperation::LineTo(10.0, 50.0),
            PathOperation::ClosePath,
        ];

        let path = PathContent::from_operations(ops);

        assert_eq!(path.bbox.x, 10.0);
        assert_eq!(path.bbox.y, 10.0);
        assert_eq!(path.bbox.width, 40.0);
        assert_eq!(path.bbox.height, 40.0);
    }

    #[test]
    fn test_path_with_fill() {
        let path = PathContent::new(Rect::new(0.0, 0.0, 100.0, 100.0))
            .with_fill(Color::new(1.0, 0.0, 0.0));

        assert!(path.has_fill());
        assert!(path.has_stroke()); // Default has stroke
    }

    #[test]
    fn test_compute_bbox_from_rectangle() {
        let ops = vec![PathOperation::Rectangle(20.0, 30.0, 100.0, 50.0)];
        let path = PathContent::from_operations(ops);

        assert_eq!(path.bbox.x, 20.0);
        assert_eq!(path.bbox.y, 30.0);
        assert_eq!(path.bbox.width, 100.0);
        assert_eq!(path.bbox.height, 50.0);
    }
}
