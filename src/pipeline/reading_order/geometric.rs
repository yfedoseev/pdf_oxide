//! Column-aware geometric reading order strategy.

use crate::error::Result;
use crate::layout::TextSpan;
use crate::pipeline::OrderedTextSpan;

use super::{ReadingOrderContext, ReadingOrderStrategy};

/// Column-aware geometric reading order strategy.
///
/// This strategy detects columns based on horizontal gaps and processes
/// each column from top to bottom before moving to the next column.
///
/// This is useful for multi-column documents like academic papers,
/// newspapers, and magazines.
pub struct GeometricStrategy {
    /// Minimum gap between columns (in points).
    column_gap_threshold: f32,
}

impl GeometricStrategy {
    /// Create a new geometric strategy with default settings.
    pub fn new() -> Self {
        Self {
            column_gap_threshold: 20.0,
        }
    }

    /// Create a geometric strategy with custom column gap threshold.
    pub fn with_column_gap(threshold: f32) -> Self {
        Self {
            column_gap_threshold: threshold,
        }
    }

    /// Detect columns based on horizontal gaps.
    ///
    /// Returns column boundaries as X coordinates.
    ///
    /// # Phase 8 Enhancement: Adaptive Column Detection
    ///
    /// Instead of using a fixed threshold, this method now analyzes the gap
    /// distribution to find natural column boundaries:
    /// 1. Collects all horizontal gaps between span right edges and next span left edges
    /// 2. Calculates median gap to understand typical word spacing
    /// 3. Uses a multiplier to detect column gaps (significantly larger than word gaps)
    fn detect_columns(&self, spans: &[TextSpan]) -> Vec<f32> {
        if spans.is_empty() {
            return Vec::new();
        }

        // Phase 8: Adaptive threshold based on gap distribution
        let effective_threshold = self.calculate_adaptive_threshold(spans);

        // Collect all X coordinates (left edges)
        let mut x_coords: Vec<f32> = spans.iter().map(|s| s.bbox.x).collect();
        x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        x_coords.dedup();

        if x_coords.len() < 2 {
            return vec![x_coords.first().copied().unwrap_or(0.0)];
        }

        // Find significant gaps that indicate column boundaries
        let mut boundaries = vec![x_coords[0]];

        for i in 1..x_coords.len() {
            let gap = x_coords[i] - x_coords[i - 1];
            if gap > effective_threshold {
                boundaries.push(x_coords[i]);
            }
        }

        boundaries
    }

    /// Calculate adaptive column gap threshold based on document characteristics.
    ///
    /// Phase 8: Uses statistical analysis of horizontal gaps to detect
    /// column boundaries more accurately for documents with varying layouts.
    ///
    /// Uses left-edge-to-left-edge gaps (same as column detection) for consistency.
    fn calculate_adaptive_threshold(&self, spans: &[TextSpan]) -> f32 {
        if spans.len() < 2 {
            return self.column_gap_threshold;
        }

        // Collect all X coordinates (left edges) - same as detect_columns
        let mut x_coords: Vec<f32> = spans.iter().map(|s| s.bbox.x).collect();
        x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        x_coords.dedup();

        if x_coords.len() < 2 {
            return self.column_gap_threshold;
        }

        // Collect all gaps between left edges
        let mut gaps: Vec<f32> = Vec::new();
        for i in 1..x_coords.len() {
            let gap = x_coords[i] - x_coords[i - 1];
            if gap > 0.0 {
                gaps.push(gap);
            }
        }

        if gaps.is_empty() {
            return self.column_gap_threshold;
        }

        // Need multiple gaps to compute meaningful statistics
        // If only one or two gaps, use the configured threshold
        if gaps.len() < 3 {
            return self.column_gap_threshold;
        }

        // Sort gaps to find percentiles
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use the 25th percentile as "typical" word spacing
        // This is more robust than median for documents with varying layouts
        let p25_idx = gaps.len() / 4;
        let typical_gap = gaps[p25_idx];

        // Column gaps should be significantly larger than typical word gaps
        // Use 4x typical as the threshold (columns are much wider than word spacing)
        let adaptive_threshold = typical_gap * 4.0;

        // Ensure threshold is at least the minimum configured threshold
        let final_threshold = adaptive_threshold.max(self.column_gap_threshold);

        log::debug!(
            "Adaptive column detection: typical_gap={:.1}, adaptive_threshold={:.1}, final={:.1}",
            typical_gap,
            adaptive_threshold,
            final_threshold
        );

        final_threshold
    }
}

impl Default for GeometricStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadingOrderStrategy for GeometricStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        _context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // Detect column boundaries
        let boundaries = self.detect_columns(&spans);

        // Assign spans to columns (using indices instead of references)
        let mut column_indices: Vec<Vec<usize>> = vec![Vec::new(); boundaries.len().max(1)];
        for (idx, span) in spans.iter().enumerate() {
            let column_idx = boundaries
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &boundary)| span.bbox.x >= boundary)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            column_indices[column_idx].push(idx);
        }

        // Process each column from left to right, top to bottom within each column
        let mut ordered = Vec::new();
        let mut order = 0;

        for column in column_indices {
            // Sort spans within column by Y (top to bottom, descending Y = top first)
            let mut column_sorted = column;
            column_sorted.sort_by(|&a, &b| {
                spans[b]
                    .bbox
                    .y
                    .partial_cmp(&spans[a].bbox.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for idx in column_sorted {
                ordered.push(OrderedTextSpan::new(spans[idx].clone(), order));
                order += 1;
            }
        }

        Ok(ordered)
    }

    fn name(&self) -> &'static str {
        "GeometricStrategy"
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
    fn test_single_column() {
        let spans = vec![
            make_span("Line 3", 50.0, 50.0),
            make_span("Line 1", 50.0, 100.0),
            make_span("Line 2", 50.0, 75.0),
        ];

        let strategy = GeometricStrategy::new();
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        assert_eq!(ordered[0].span.text, "Line 1");
        assert_eq!(ordered[1].span.text, "Line 2");
        assert_eq!(ordered[2].span.text, "Line 3");
    }

    #[test]
    fn test_two_columns() {
        // Phase 8: Updated test to work with adaptive threshold
        // Using explicit column gap threshold to ensure deterministic behavior
        let spans = vec![
            // Left column
            make_span("Left 1", 50.0, 100.0),
            make_span("Left 2", 50.0, 50.0),
            // Right column (gap > 20pt)
            make_span("Right 1", 200.0, 100.0),
            make_span("Right 2", 200.0, 50.0),
        ];

        // Use explicit threshold since test data doesn't have realistic word gaps
        let strategy = GeometricStrategy::with_column_gap(30.0);
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        // Left column first, then right column
        assert_eq!(ordered[0].span.text, "Left 1");
        assert_eq!(ordered[1].span.text, "Left 2");
        assert_eq!(ordered[2].span.text, "Right 1");
        assert_eq!(ordered[3].span.text, "Right 2");
    }

    #[test]
    fn test_adaptive_column_detection() {
        // Test that adaptive threshold correctly detects columns
        // when there are many word-level gaps and one large column gap
        let spans = vec![
            // Left column - multiple words with small gaps
            make_span("Word1", 50.0, 100.0),
            make_span("Word2", 55.0, 100.0), // 5pt gap (word spacing)
            make_span("Word3", 60.0, 100.0), // 5pt gap (word spacing)
            make_span("Word4", 50.0, 50.0),
            make_span("Word5", 55.0, 50.0), // 5pt gap (word spacing)
            // Right column - large gap (>3x median)
            make_span("RightWord1", 200.0, 100.0), // 140pt gap (column)
            make_span("RightWord2", 200.0, 50.0),
        ];

        let strategy = GeometricStrategy::new();
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        // Should detect two columns and process left first
        // All left column words should come before right column
        let left_indices: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter(|(_, s)| s.span.text.starts_with("Word"))
            .map(|(i, _)| i)
            .collect();
        let right_indices: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter(|(_, s)| s.span.text.starts_with("Right"))
            .map(|(i, _)| i)
            .collect();

        // All left column indices should be less than all right column indices
        assert!(
            left_indices
                .iter()
                .all(|&l| right_indices.iter().all(|&r| l < r)),
            "Left column should be processed before right column"
        );
    }
}
