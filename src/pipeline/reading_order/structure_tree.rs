//! Structure tree reading order strategy.
//!
//! Uses PDF Tagged structure tree (ISO 32000-1:2008 Section 14.7) to determine
//! the correct reading order for text spans based on their MCID values.

use crate::error::Result;
use crate::layout::TextSpan;
use crate::pipeline::OrderedTextSpan;

use super::{GeometricStrategy, ReadingOrderContext, ReadingOrderStrategy};

/// Structure tree-based reading order strategy.
///
/// This is the PDF-spec-compliant approach for Tagged PDFs (ISO 32000-1:2008
/// Section 14.7). It uses the structure tree's pre-order traversal to determine
/// the logical reading order of marked content.
///
/// For spans without MCIDs or when no structure tree is available, it falls
/// back to the GeometricStrategy (column-aware geometric analysis).
pub struct StructureTreeStrategy {
    /// Fallback strategy for spans without MCIDs.
    /// Uses GeometricStrategy for better multi-column document handling.
    fallback: GeometricStrategy,
}

impl StructureTreeStrategy {
    /// Create a new structure tree strategy.
    pub fn new() -> Self {
        Self {
            fallback: GeometricStrategy::new(),
        }
    }
}

impl Default for StructureTreeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadingOrderStrategy for StructureTreeStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        // If no structure tree or MCID order, fall back to simple strategy
        let mcid_order = match &context.mcid_order {
            Some(order) if !order.is_empty() => order,
            _ => return self.fallback.apply(spans, context),
        };

        // Create MCID -> reading order mapping
        let mcid_to_order: std::collections::HashMap<u32, usize> = mcid_order
            .iter()
            .enumerate()
            .map(|(order, &mcid)| (mcid, order))
            .collect();

        // Separate spans with and without MCIDs
        let mut with_mcid: Vec<(TextSpan, usize)> = Vec::new();
        let mut without_mcid: Vec<TextSpan> = Vec::new();

        for span in spans {
            if let Some(mcid) = span.mcid {
                if let Some(&order) = mcid_to_order.get(&mcid) {
                    with_mcid.push((span, order));
                } else {
                    // MCID not in structure tree - treat as untagged
                    without_mcid.push(span);
                }
            } else {
                without_mcid.push(span);
            }
        }

        // Sort spans by their structure tree order
        with_mcid.sort_by_key(|(_, order)| *order);

        // Build result: tagged spans first (in structure order), then untagged
        let mut result = Vec::new();
        let mut reading_order = 0;

        // Add tagged spans
        for (span, _) in with_mcid {
            result.push(OrderedTextSpan::new(span, reading_order));
            reading_order += 1;
        }

        // Add untagged spans using simple ordering
        if !without_mcid.is_empty() {
            let untagged_ordered = self.fallback.apply(without_mcid, context)?;
            for mut ordered_span in untagged_ordered {
                ordered_span.reading_order = reading_order;
                result.push(ordered_span);
                reading_order += 1;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "StructureTreeStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight};

    fn make_span(text: &str, x: f32, y: f32, mcid: Option<u32>) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 50.0, 12.0),
            font_name: "Test".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid,
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
    fn test_structure_tree_ordering() {
        // Spans with MCIDs in "wrong" visual order
        let spans = vec![
            make_span("Third", 0.0, 100.0, Some(2)),
            make_span("First", 0.0, 50.0, Some(0)),
            make_span("Second", 0.0, 75.0, Some(1)),
        ];

        let strategy = StructureTreeStrategy::new();
        let context = ReadingOrderContext::new().with_mcid_order(vec![0, 1, 2]);
        let ordered = strategy.apply(spans, &context).unwrap();

        assert_eq!(ordered[0].span.text, "First");
        assert_eq!(ordered[1].span.text, "Second");
        assert_eq!(ordered[2].span.text, "Third");
    }

    #[test]
    fn test_fallback_for_untagged() {
        // Mix of tagged and untagged spans
        let spans = vec![
            make_span("Tagged", 0.0, 100.0, Some(0)),
            make_span("Untagged", 0.0, 50.0, None),
        ];

        let strategy = StructureTreeStrategy::new();
        let context = ReadingOrderContext::new().with_mcid_order(vec![0]);
        let ordered = strategy.apply(spans, &context).unwrap();

        // Tagged comes first, then untagged
        assert_eq!(ordered[0].span.text, "Tagged");
        assert_eq!(ordered[1].span.text, "Untagged");
    }

    #[test]
    fn test_no_structure_tree_fallback() {
        let spans = vec![
            make_span("Bottom", 0.0, 50.0, None),
            make_span("Top", 0.0, 100.0, None),
        ];

        let strategy = StructureTreeStrategy::new();
        let context = ReadingOrderContext::new(); // No MCID order
        let ordered = strategy.apply(spans, &context).unwrap();

        // Should use geometric strategy fallback: top to bottom within column
        // Both spans are in same column (x=0), so ordered by Y (top first)
        assert_eq!(ordered[0].span.text, "Top");
        assert_eq!(ordered[1].span.text, "Bottom");
    }

    #[test]
    fn test_geometric_fallback_multi_column() {
        // Phase 8: Updated test to work with adaptive column detection
        // Test that multi-column documents are handled correctly via GeometricStrategy
        // Added word-level spans to provide realistic gap distribution for adaptive threshold
        let spans = vec![
            // Left column - multiple words with small gaps
            make_span("Left Top Word1", 0.0, 100.0, None),
            make_span("Left Top Word2", 5.0, 100.0, None), // 5pt word gap
            make_span("Left Top Word3", 10.0, 100.0, None), // 5pt word gap
            make_span("Left Bottom", 0.0, 50.0, None),
            // Right column (gap >> word gaps)
            make_span("Right Top", 200.0, 100.0, None),
            make_span("Right Bottom", 200.0, 50.0, None),
        ];

        let strategy = StructureTreeStrategy::new();
        let context = ReadingOrderContext::new(); // No MCID order
        let ordered = strategy.apply(spans, &context).unwrap();

        // Should process left column first, then right column
        // Verify all left spans come before right spans in order
        let left_indices: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter(|(_, s)| s.span.text.starts_with("Left"))
            .map(|(i, _)| i)
            .collect();
        let right_indices: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter(|(_, s)| s.span.text.starts_with("Right"))
            .map(|(i, _)| i)
            .collect();

        assert!(
            left_indices
                .iter()
                .all(|&l| right_indices.iter().all(|&r| l < r)),
            "Left column should be processed before right column"
        );
    }
}
