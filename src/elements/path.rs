//! Path/vector graphics content element types.
//!
//! This module provides the `PathContent` type for representing
//! vector graphics in PDFs.

use crate::geometry::Rect;
use crate::layout::Color;

/// Vector path content that can be extracted from or written to a PDF.
///
/// This represents vector graphics such as lines, curves, and shapes.
#[derive(Debug, Clone)]
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
    /// Reading order index
    pub reading_order: Option<usize>,
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
            reading_order: None,
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
            reading_order: None,
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
            reading_order: None,
        }
    }
}

/// A single path operation.
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
