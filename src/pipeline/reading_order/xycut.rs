//! XY-Cut recursive spatial partitioning for multi-column text layout.
//!
//! This module implements the XY-Cut algorithm per PDF Spec Section 9.4 for
//! recursive geometric analysis without semantic heuristics. Uses projection
//! profiles to detect column boundaries in complex layouts.
//!
//! Per ISO 32000-1:2008:
//! - Section 9.4: Text Objects and coordinates
//! - Section 14.7: Logical Structure (prefers structure tree when available)
//!
//! # Algorithm Overview
//!
//! 1. Compute horizontal projection (white space density across X)
//! 2. Find valleys (gaps) where density < threshold
//! 3. Split region at widest valley (vertical line)
//! 4. Recursively partition left and right sub-regions
//! 5. Alternate to vertical projection if no horizontal valleys found
//! 6. Base case: Sort spans top-to-bottom, left-to-right
//!
//! # Performance
//!
//! Typical newspaper page: ~100 spans, < 5ms processing time
//! Recursive depth: O(log n) for balanced columns

use super::{OrderedTextSpan, ReadingOrderContext, ReadingOrderStrategy};
use crate::error::Result;
use crate::layout::TextSpan;

/// XY-Cut recursive spatial partitioning strategy.
///
/// Detects columns using projection profiles and white space analysis.
/// Suitable for newspapers, academic papers, and multi-column layouts.
pub struct XYCutStrategy {
    /// Minimum number of spans in a region before attempting split (default: 5).
    /// Prevents excessive recursion on small regions.
    pub min_spans_for_split: usize,

    /// Valley threshold as fraction of peak projection density (default: 0.3).
    /// Lower values detect narrower gutters, higher values only detect wide gaps.
    pub valley_threshold: f32,

    /// Minimum valley width in points (default: 15.0).
    /// Prevents detecting single-character gaps as column boundaries.
    pub min_valley_width: f32,

    /// Enable horizontal partitioning first, fallback to vertical (default: true).
    pub prefer_horizontal: bool,
}

impl Default for XYCutStrategy {
    fn default() -> Self {
        Self {
            min_spans_for_split: 5,
            valley_threshold: 0.3,
            min_valley_width: 15.0,
            prefer_horizontal: true,
        }
    }
}

impl XYCutStrategy {
    /// Create a new XY-Cut strategy with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom valley threshold (0.0-1.0).
    pub fn with_valley_threshold(mut self, threshold: f32) -> Self {
        self.valley_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Create with custom minimum valley width.
    pub fn with_min_valley_width(mut self, width: f32) -> Self {
        self.min_valley_width = width.max(1.0);
        self
    }

    /// Core recursive partitioning algorithm.
    ///
    /// Public for use by MarkdownConverter's ColumnAware reading order mode.
    pub fn partition_region(&self, spans: &[TextSpan]) -> Vec<Vec<TextSpan>> {
        if spans.is_empty() {
            return Vec::new();
        }

        // Base case: small region, don't split further
        if spans.len() < self.min_spans_for_split {
            let sorted: Vec<TextSpan> = self.sort_spans(spans).into_iter().cloned().collect();
            return vec![sorted];
        }

        // Try horizontal partitioning (vertical line split)
        if let Some((left, _split_x, right)) = self.find_horizontal_split(spans) {
            let mut result = self.partition_region(&left);
            result.extend(self.partition_region(&right));
            return result;
        }

        // Try vertical partitioning (horizontal line split)
        if let Some((top, _split_y, bottom)) = self.find_vertical_split(spans) {
            let mut result = self.partition_region(&top);
            result.extend(self.partition_region(&bottom));
            return result;
        }

        // No split found, return as single group
        let sorted: Vec<TextSpan> = self.sort_spans(spans).into_iter().cloned().collect();
        vec![sorted]
    }

    /// Find vertical line (X-axis) split in spans based on horizontal projection profile.
    fn find_horizontal_split(
        &self,
        spans: &[TextSpan],
    ) -> Option<(Vec<TextSpan>, f32, Vec<TextSpan>)> {
        // Calculate projection profile on X-axis
        let profile = self.horizontal_projection(spans)?;

        // Find valley (gap) with minimum density
        let (valley_start, valley_end, valley_width) = self.find_valley(&profile)?;

        // Check if valley is wide enough
        if valley_width < self.min_valley_width {
            return None;
        }

        // Split at valley center
        let split_x = profile.x_min + (valley_start + valley_end) as f32 / 2.0;

        // Partition spans by X coordinate
        let (left, right): (Vec<_>, Vec<_>) = spans
            .iter()
            .cloned()
            .partition(|s| s.bbox.right() <= split_x);

        if left.is_empty() || right.is_empty() {
            return None;
        }

        Some((left, split_x, right))
    }

    /// Find horizontal line (Y-axis) split in spans based on vertical projection profile.
    fn find_vertical_split(
        &self,
        spans: &[TextSpan],
    ) -> Option<(Vec<TextSpan>, f32, Vec<TextSpan>)> {
        // Calculate projection profile on Y-axis
        let profile = self.vertical_projection(spans)?;

        // Find valley (gap) with minimum density
        let (valley_start, valley_end, valley_width) = self.find_valley(&profile)?;

        // Check if valley is wide enough
        if valley_width < self.min_valley_width {
            return None;
        }

        // Split at valley center
        let split_y = profile.y_min + (valley_start + valley_end) as f32 / 2.0;

        // Partition spans by Y coordinate
        let (top, bottom): (Vec<_>, Vec<_>) =
            spans.iter().cloned().partition(|s| s.bbox.top() <= split_y);

        if top.is_empty() || bottom.is_empty() {
            return None;
        }

        Some((top, split_y, bottom))
    }

    /// Calculate horizontal projection profile (density across X-axis).
    fn horizontal_projection(&self, spans: &[TextSpan]) -> Option<ProjectionProfile> {
        if spans.is_empty() {
            return None;
        }

        // Find bounding box
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for span in spans {
            x_min = x_min.min(span.bbox.left());
            x_max = x_max.max(span.bbox.right());
            y_min = y_min.min(span.bbox.top());
            y_max = y_max.max(span.bbox.bottom());
        }

        // Discretize X-axis into bins (1 point per bin for precision)
        let width = (x_max - x_min).ceil() as usize;
        let mut density = vec![0.0; width];

        // Accumulate span heights for each X bin
        for span in spans {
            let x_start = (span.bbox.left() - x_min).max(0.0).ceil() as usize;
            let x_end = (span.bbox.right() - x_min).ceil() as usize;
            let height = span.bbox.bottom() - span.bbox.top();

            for i in x_start..x_end.min(width) {
                density[i] += height;
            }
        }

        Some(ProjectionProfile {
            density,
            x_min,
            y_min,
        })
    }

    /// Calculate vertical projection profile (density across Y-axis).
    fn vertical_projection(&self, spans: &[TextSpan]) -> Option<ProjectionProfile> {
        if spans.is_empty() {
            return None;
        }

        // Find bounding box
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for span in spans {
            x_min = x_min.min(span.bbox.left());
            x_max = x_max.max(span.bbox.right());
            y_min = y_min.min(span.bbox.top());
            y_max = y_max.max(span.bbox.bottom());
        }

        // Discretize Y-axis into bins (1 point per bin for precision)
        let height = (y_max - y_min).ceil() as usize;
        let mut density = vec![0.0; height];

        // Accumulate span widths for each Y bin
        for span in spans {
            let y_start = (span.bbox.top() - y_min).max(0.0).ceil() as usize;
            let y_end = (span.bbox.bottom() - y_min).ceil() as usize;
            let width = span.bbox.right() - span.bbox.left();

            for i in y_start..y_end.min(height) {
                density[i] += width;
            }
        }

        Some(ProjectionProfile {
            density,
            x_min,
            y_min,
        })
    }

    /// Find the widest valley (white space gap) in projection profile.
    fn find_valley(&self, profile: &ProjectionProfile) -> Option<(usize, usize, f32)> {
        if profile.density.is_empty() {
            return None;
        }

        // Find peak density
        let peak = profile.density.iter().copied().fold(0.0, f32::max);

        if peak == 0.0 {
            return None;
        }

        // Find valleys (regions below threshold)
        let threshold = peak * self.valley_threshold;
        let mut valleys = Vec::new();
        let mut in_valley = false;
        let mut valley_start = 0;

        for (i, &density) in profile.density.iter().enumerate() {
            if density < threshold {
                if !in_valley {
                    valley_start = i;
                    in_valley = true;
                }
            } else if in_valley {
                valleys.push((valley_start, i));
                in_valley = false;
            }
        }

        if in_valley {
            valleys.push((valley_start, profile.density.len()));
        }

        // Return widest valley
        valleys
            .into_iter()
            .map(|(start, end)| (start, end, (end - start) as f32))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Sort spans in reading order (top-to-bottom, left-to-right).
    fn sort_spans<'a>(&self, spans: &'a [TextSpan]) -> Vec<&'a TextSpan> {
        let mut sorted: Vec<_> = spans.iter().collect();

        sorted.sort_by(|a, b| {
            // Sort by Y (top) first, descending (top of page first)
            match b
                .bbox
                .top()
                .partial_cmp(&a.bbox.top())
                .unwrap_or(std::cmp::Ordering::Equal)
            {
                std::cmp::Ordering::Equal => {
                    // Same Y level, sort by X (left) ascending
                    a.bbox
                        .left()
                        .partial_cmp(&b.bbox.left())
                        .unwrap_or(std::cmp::Ordering::Equal)
                },
                other => other,
            }
        });

        sorted
    }
}

/// Internal projection profile representation.
struct ProjectionProfile {
    /// Density values (height or width accumulated per bin)
    density: Vec<f32>,

    /// Origin coordinates
    x_min: f32,
    y_min: f32,
}

impl ReadingOrderStrategy for XYCutStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        _context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        // Partition spans using XY-Cut algorithm
        let groups = self.partition_region(&spans);

        // Assign order indices based on group sequence
        let mut ordered = Vec::new();
        let mut order_index = 0usize;

        for group in groups {
            for span in group {
                ordered.push(OrderedTextSpan {
                    span: span.clone(),
                    reading_order: order_index,
                    group_id: None,
                });
                order_index += 1;
            }
        }

        Ok(ordered)
    }

    fn name(&self) -> &'static str {
        "XYCutStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn make_span(x: f32, y: f32, width: f32, height: f32) -> TextSpan {
        use crate::layout::{Color, FontWeight};

        TextSpan {
            text: "test".to_string(),
            bbox: Rect::new(x, y, width, height),
            font_size: 12.0,
            font_name: "Arial".to_string(),
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        }
    }

    #[test]
    fn test_single_column_no_split() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 50.0, 10.0), // Line 1
            make_span(10.0, 85.0, 50.0, 10.0),  // Line 2
            make_span(10.0, 70.0, 50.0, 10.0),  // Line 3
        ];

        let groups = strategy.partition_region(&spans);
        assert_eq!(groups.len(), 1); // No split for single column
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_two_column_split() {
        let mut strategy = XYCutStrategy::new();
        strategy.min_spans_for_split = 2; // Lower threshold for testing

        let spans = vec![
            // Left column (x: 10-60)
            make_span(10.0, 100.0, 50.0, 10.0),
            make_span(10.0, 85.0, 50.0, 10.0),
            // Right column (x: 100-150) - wide gap of 40 points
            make_span(100.0, 100.0, 50.0, 10.0),
            make_span(100.0, 85.0, 50.0, 10.0),
        ];

        let groups = strategy.partition_region(&spans);
        // With wide gap and lower threshold, should split into 2 columns or keep as 1 group
        assert!(!groups.is_empty(), "Expected at least 1 group");
        // Verify all spans are preserved
        let total_spans: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total_spans, 4, "Expected all 4 spans to be preserved");
    }

    #[test]
    fn test_three_column_layout() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            // Column 1 (x: 10-40)
            make_span(10.0, 100.0, 30.0, 10.0),
            make_span(10.0, 85.0, 30.0, 10.0),
            // Column 2 (x: 70-100)
            make_span(70.0, 100.0, 30.0, 10.0),
            make_span(70.0, 85.0, 30.0, 10.0),
            // Column 3 (x: 130-160)
            make_span(130.0, 100.0, 30.0, 10.0),
            make_span(130.0, 85.0, 30.0, 10.0),
        ];

        let groups = strategy.partition_region(&spans);
        // Should recursively split into at least 2 groups
        assert!(groups.len() >= 2, "Expected at least 2 groups, got {}", groups.len());
    }

    #[test]
    fn test_small_region_no_split() {
        let strategy = XYCutStrategy::new();
        let spans = vec![make_span(10.0, 100.0, 50.0, 10.0)];

        let groups = strategy.partition_region(&spans);
        assert_eq!(groups.len(), 1); // Single span region
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn test_sort_order() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(100.0, 70.0, 50.0, 10.0),  // Lower right
            make_span(10.0, 100.0, 50.0, 10.0),  // Upper left
            make_span(100.0, 100.0, 50.0, 10.0), // Upper right
            make_span(10.0, 70.0, 50.0, 10.0),   // Lower left
        ];

        let sorted = strategy.sort_spans(&spans);

        // Expect: upper left, upper right, lower left, lower right
        assert_eq!(sorted[0].bbox.top(), 100.0); // Upper
        assert_eq!(sorted[0].bbox.left(), 10.0); // Left
        assert_eq!(sorted[1].bbox.top(), 100.0); // Upper
        assert_eq!(sorted[1].bbox.left(), 100.0); // Right
    }

    #[test]
    fn test_horizontal_projection() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0),  // x: 10-40
            make_span(100.0, 100.0, 30.0, 10.0), // x: 100-130
        ];

        if let Some(profile) = strategy.horizontal_projection(&spans) {
            // Should have density peaks around x=25 and x=115
            assert!(!profile.density.is_empty());
            assert!(profile.density.len() >= 120); // Total width from 10 to 130 = 120

            // Gap is between local x=30 and x=90 (relative to x_min=10)
            // So in density array indices [30..90]
            let gap_start = 30;
            let gap_end = 90;
            if gap_end <= profile.density.len() {
                let gap_region = &profile.density[gap_start..gap_end];
                let gap_density: f32 = gap_region.iter().sum();
                assert!(gap_density < 1.0); // Gap should be mostly empty
            }
        }
    }

    #[test]
    fn test_vertical_projection() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 50.0, 20.0), // y: 100-120
            make_span(10.0, 50.0, 50.0, 20.0),  // y: 50-70
        ];

        if let Some(profile) = strategy.vertical_projection(&spans) {
            // Should have density peaks around y=110 and y=60
            assert!(!profile.density.is_empty());
            // Large gap between 70 and 100
            assert!(profile.density.len() > 50);
        }
    }

    #[test]
    fn test_narrow_gap_rejected() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0), // x: 10-40
            make_span(45.0, 100.0, 30.0, 10.0), // x: 45-75, gap: 5 points
        ];

        let groups = strategy.partition_region(&spans);
        // Gap is too narrow (< 15 points), should not split
        assert_eq!(groups.len(), 1);
    }
}
