//! Ordered text spans for output conversion.
//!
//! This module provides the OrderedTextSpan type which wraps TextSpan
//! with reading order information.

use crate::layout::TextSpan;

/// A text span with an assigned reading order index.
///
/// This wrapper adds ordering information to TextSpan without modifying
/// the original span data. The reading_order field represents the position
/// in the final document output (0 = first to be read).
#[derive(Debug, Clone)]
pub struct OrderedTextSpan {
    /// The underlying text span.
    pub span: TextSpan,

    /// Index in reading order (0 = first to be read).
    pub reading_order: usize,

    /// Group ID for paragraph/section grouping (optional).
    pub group_id: Option<usize>,
}

impl OrderedTextSpan {
    /// Create a new ordered span with the given reading order.
    pub fn new(span: TextSpan, reading_order: usize) -> Self {
        Self {
            span,
            reading_order,
            group_id: None,
        }
    }

    /// Set the group ID for paragraph grouping.
    pub fn with_group(mut self, group_id: usize) -> Self {
        self.group_id = Some(group_id);
        self
    }
}

/// A collection of ordered spans with helper methods.
pub struct OrderedSpans {
    spans: Vec<OrderedTextSpan>,
}

impl OrderedSpans {
    /// Create a new collection from a vector of ordered spans.
    pub fn new(spans: Vec<OrderedTextSpan>) -> Self {
        Self { spans }
    }

    /// Get the number of spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Get spans sorted by reading order.
    pub fn in_reading_order(&self) -> Vec<&OrderedTextSpan> {
        let mut sorted: Vec<_> = self.spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);
        sorted
    }

    /// Get the underlying spans.
    pub fn spans(&self) -> &[OrderedTextSpan] {
        &self.spans
    }

    /// Convert to a vector of ordered spans.
    pub fn into_vec(self) -> Vec<OrderedTextSpan> {
        self.spans
    }

    /// Group spans into lines based on Y-coordinate proximity.
    ///
    /// Returns groups of spans that appear on the same line.
    pub fn group_into_lines(&self, tolerance: f32) -> Vec<Vec<&OrderedTextSpan>> {
        if self.spans.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<_> = self.spans.iter().collect();
        sorted.sort_by(|a, b| {
            b.span
                .bbox
                .y
                .partial_cmp(&a.span.bbox.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut lines: Vec<Vec<&OrderedTextSpan>> = Vec::new();
        let mut current_line: Vec<&OrderedTextSpan> = vec![sorted[0]];
        let mut current_y = sorted[0].span.bbox.y;

        for span in sorted.into_iter().skip(1) {
            if (current_y - span.span.bbox.y).abs() <= tolerance {
                current_line.push(span);
            } else {
                lines.push(std::mem::take(&mut current_line));
                current_line = vec![span];
                current_y = span.span.bbox.y;
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }
}

impl From<Vec<OrderedTextSpan>> for OrderedSpans {
    fn from(spans: Vec<OrderedTextSpan>) -> Self {
        Self::new(spans)
    }
}

impl IntoIterator for OrderedSpans {
    type Item = OrderedTextSpan;
    type IntoIter = std::vec::IntoIter<OrderedTextSpan>;

    fn into_iter(self) -> Self::IntoIter {
        self.spans.into_iter()
    }
}
