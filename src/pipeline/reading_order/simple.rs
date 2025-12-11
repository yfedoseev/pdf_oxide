//! Simple top-to-bottom, left-to-right reading order strategy.

use crate::error::Result;
use crate::layout::TextSpan;
use crate::pipeline::OrderedTextSpan;

use super::{ReadingOrderContext, ReadingOrderStrategy};

/// Simple top-to-bottom, left-to-right reading order.
///
/// This strategy sorts spans by Y coordinate (descending, so top comes first)
/// then by X coordinate (ascending, so left comes first).
///
/// This is the simplest strategy and works well for single-column documents.
pub struct SimpleStrategy;

impl ReadingOrderStrategy for SimpleStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        _context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        let mut spans_with_index: Vec<_> = spans.into_iter().enumerate().collect();

        // Sort by Y descending (top first), then X ascending (left first)
        spans_with_index.sort_by(|(_, a), (_, b)| {
            let y_cmp = b
                .bbox
                .y
                .partial_cmp(&a.bbox.y)
                .unwrap_or(std::cmp::Ordering::Equal);

            match y_cmp {
                std::cmp::Ordering::Equal => a
                    .bbox
                    .x
                    .partial_cmp(&b.bbox.x)
                    .unwrap_or(std::cmp::Ordering::Equal),
                other => other,
            }
        });

        Ok(spans_with_index
            .into_iter()
            .enumerate()
            .map(|(order, (_, span))| OrderedTextSpan::new(span, order))
            .collect())
    }

    fn name(&self) -> &'static str {
        "SimpleStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight};

    fn make_span(text: &str, x: f32, y: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 50.0, 12.0),
            font_name: "Test".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            offset_semantic: false,
            split_boundary_before: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        }
    }

    #[test]
    fn test_simple_ordering() {
        let spans = vec![
            make_span("Bottom", 0.0, 50.0), // y=50 is lower on page
            make_span("Top", 0.0, 100.0),   // y=100 is higher on page
            make_span("Middle", 0.0, 75.0), // y=75 is in between
        ];

        let strategy = SimpleStrategy;
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        assert_eq!(ordered[0].span.text, "Top");
        assert_eq!(ordered[1].span.text, "Middle");
        assert_eq!(ordered[2].span.text, "Bottom");
    }

    #[test]
    fn test_left_to_right_on_same_line() {
        let spans = vec![
            make_span("Right", 100.0, 100.0),
            make_span("Left", 0.0, 100.0),
            make_span("Center", 50.0, 100.0),
        ];

        let strategy = SimpleStrategy;
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        assert_eq!(ordered[0].span.text, "Left");
        assert_eq!(ordered[1].span.text, "Center");
        assert_eq!(ordered[2].span.text, "Right");
    }
}
