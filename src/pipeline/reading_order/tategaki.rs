//! Tategaki (vertical writing) reading-order strategy.
//!
//! Right-to-left across columns, top-to-bottom within each column —
//! the layout convention for vertical Chinese, Japanese, and Korean
//! text. Spans whose horizontal X centers cluster together belong to
//! the same column.
//!
//! This strategy is dispatched by [`crate::pipeline::TextPipeline::process`]
//! when the per-span `wmode` tag indicates a vertical-majority page. The four
//! horizontal LTR strategies (Simple, Geometric, XYCut, StructureTree)
//! are left unchanged; tategaki always wins when the page is vertical.

use crate::error::Result;
use crate::layout::TextSpan;
use crate::pipeline::{OrderedTextSpan, ReadingOrderInfo};

use super::{ReadingOrderContext, ReadingOrderStrategy};

/// Right-to-left, top-to-bottom reading order for vertical writing
/// (CJK tategaki).
///
/// Algorithm: cluster spans into columns by X-center proximity (using
/// the median span width as the tolerance — wide enough to keep glyphs
/// of one body column together, narrow enough to separate adjacent
/// columns). Sort primarily by descending X-center (right column first)
/// and secondarily by descending Y (PDF user space y increases upward,
/// so descending y means top-first within a column).
pub struct TategakiStrategy;

impl ReadingOrderStrategy for TategakiStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        _context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // Median span width as the per-column-clustering tolerance.
        // Robust to outliers from rotated annotations / single-glyph
        // ruby annotations.
        let mut widths: Vec<f32> = spans.iter().map(|s| s.bbox.width.max(1.0)).collect();
        widths.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));
        let median_w = widths[widths.len() / 2].max(1.0);
        let tol = median_w;

        let mut indexed: Vec<(usize, TextSpan)> = spans.into_iter().enumerate().collect();
        let x_center = |s: &TextSpan| -> f32 { s.bbox.x + s.bbox.width * 0.5 };

        indexed.sort_by(|(_, a), (_, b)| {
            let ax = x_center(a);
            let bx = x_center(b);
            if (ax - bx).abs() <= tol {
                // Same column: top first (PDF user space — descending y).
                crate::utils::safe_float_cmp(b.bbox.y, a.bbox.y)
            } else {
                // Different columns: rightmost first (descending x_center).
                crate::utils::safe_float_cmp(bx, ax)
            }
        });

        Ok(indexed
            .into_iter()
            .enumerate()
            .map(|(order, (_, span))| {
                OrderedTextSpan::with_info(span, order, ReadingOrderInfo::simple())
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "TategakiStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn mk(text: &str, x: f32, y: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 12.0, 12.0),
            font_size: 12.0,
            wmode: 1,
            ..TextSpan::default()
        }
    }

    /// Two columns: A,B,C at x=500 (right), D,E,F at x=300 (left).
    /// Reading order must be the right column top-down first, then
    /// the left column top-down.
    #[test]
    fn tategaki_two_columns_right_to_left_top_to_bottom() {
        let spans = vec![
            mk("D", 300.0, 700.0),
            mk("F", 300.0, 676.0),
            mk("B", 500.0, 688.0),
            mk("C", 500.0, 676.0),
            mk("A", 500.0, 700.0),
            mk("E", 300.0, 688.0),
        ];
        let strategy = TategakiStrategy;
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();
        let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
        assert_eq!(combined, "ABCDEF");
    }

    /// A single column produces a top-down sequence.
    #[test]
    fn tategaki_single_column_top_to_bottom() {
        let spans = vec![
            mk("C", 300.0, 676.0),
            mk("A", 300.0, 700.0),
            mk("B", 300.0, 688.0),
        ];
        let strategy = TategakiStrategy;
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();
        let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
        assert_eq!(combined, "ABC");
    }
}
