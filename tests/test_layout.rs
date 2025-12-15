#![allow(dead_code)]
//! Integration tests for layout analysis algorithms.
//!
//! These tests verify the complete layout analysis pipeline with mock data
//! simulating realistic PDF document structures.

use pdf_oxide::geometry::{Point, Rect};
use pdf_oxide::layout::{
    clustering::{cluster_chars_into_words, cluster_words_into_lines},
    reading_order::graph_based_reading_order,
    Color, FontWeight, TextBlock, TextChar,
};

// ============================================================================
// Helper Functions for Creating Mock Data
// ============================================================================

/// Create a mock character with minimal required data.
fn mock_char(c: char, x: f32, y: f32, size: f32) -> TextChar {
    TextChar {
        char: c,
        bbox: Rect::new(x, y, size * 0.6, size),
        font_name: "Times".to_string(),
        font_size: size,
        font_weight: FontWeight::Normal,
        is_italic: false,
        color: Color::black(),
        mcid: None,
    }
}

/// Create a mock character with bold weight.
fn mock_bold_char(c: char, x: f32, y: f32, size: f32) -> TextChar {
    TextChar {
        char: c,
        bbox: Rect::new(x, y, size * 0.6, size),
        font_name: "Times-Bold".to_string(),
        font_size: size,
        font_weight: FontWeight::Bold,
        is_italic: false,
        color: Color::black(),
        mcid: None,
    }
}

/// Create a text block from a string at a specific position.
fn mock_block(text: &str, x: f32, y: f32, size: f32, bold: bool) -> TextBlock {
    let chars: Vec<TextChar> = text
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if bold {
                mock_bold_char(c, x + i as f32 * size * 0.6, y, size)
            } else {
                mock_char(c, x + i as f32 * size * 0.6, y, size)
            }
        })
        .collect();

    TextBlock::from_chars(chars)
}

/// Create a two-column layout with multiple lines per column.
fn create_two_column_layout() -> Vec<TextBlock> {
    vec![
        // Left column
        mock_block("First", 0.0, 0.0, 12.0, false),
        mock_block("line", 50.0, 0.0, 12.0, false),
        mock_block("Second", 0.0, 20.0, 12.0, false),
        mock_block("line", 50.0, 20.0, 12.0, false),
        // Right column
        mock_block("Third", 300.0, 0.0, 12.0, false),
        mock_block("line", 350.0, 0.0, 12.0, false),
        mock_block("Fourth", 300.0, 20.0, 12.0, false),
        mock_block("line", 350.0, 20.0, 12.0, false),
    ]
}

// ============================================================================
// Geometry Tests
// ============================================================================

#[test]
fn test_geometry_point() {
    let p = Point::new(10.0, 20.0);
    assert_eq!(p.x, 10.0);
    assert_eq!(p.y, 20.0);
}

#[test]
fn test_geometry_rect_operations() {
    let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
    let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);

    // Intersection
    assert!(r1.intersects(&r2));

    // Union
    let union = r1.union(&r2);
    assert_eq!(union.left(), 0.0);
    assert_eq!(union.right(), 150.0);

    // Contains point
    let p1 = Point::new(50.0, 50.0);
    assert!(r1.contains_point(&p1));
}

// ============================================================================
// DBSCAN Clustering Tests
// ============================================================================

#[test]
fn test_cluster_chars_into_words_simple() {
    let chars = vec![
        mock_char('H', 0.0, 0.0, 12.0),
        mock_char('i', 8.0, 0.0, 12.0),
        // Gap
        mock_char('B', 50.0, 0.0, 12.0),
        mock_char('y', 58.0, 0.0, 12.0),
        mock_char('e', 66.0, 0.0, 12.0),
    ];

    let clusters = cluster_chars_into_words(&chars, 15.0);

    // Should produce 2 words: "Hi" and "Bye"
    assert_eq!(clusters.len(), 2);

    // Verify "Hi" cluster
    let hi_cluster = clusters.iter().find(|c| c.contains(&0)).unwrap();
    assert!(hi_cluster.contains(&0));
    assert!(hi_cluster.contains(&1));

    // Verify "Bye" cluster
    let bye_cluster = clusters.iter().find(|c| c.contains(&2)).unwrap();
    assert!(bye_cluster.contains(&2));
    assert!(bye_cluster.contains(&3));
    assert!(bye_cluster.contains(&4));
}

#[test]
fn test_cluster_words_into_lines_simple() {
    let word1 = mock_block("Hello", 0.0, 0.0, 12.0, false);
    let word2 = mock_block("World", 50.0, 1.0, 12.0, false);
    let word3 = mock_block("Next", 0.0, 30.0, 12.0, false);
    let word4 = mock_block("Line", 50.0, 31.0, 12.0, false);

    let words = vec![word1, word2, word3, word4];
    let lines = cluster_words_into_lines(&words, 5.0);

    // Should produce 2 lines
    assert_eq!(lines.len(), 2);

    // Line 1: "Hello World"
    let line1 = lines.iter().find(|l| l.contains(&0)).unwrap();
    assert!(line1.contains(&0));
    assert!(line1.contains(&1));

    // Line 2: "Next Line"
    let line2 = lines.iter().find(|l| l.contains(&2)).unwrap();
    assert!(line2.contains(&2));
    assert!(line2.contains(&3));
}

// ============================================================================
// NOTE: XY-Cut Column Detection Tests Removed
// These tests were removed in Phase 2.2 of CLEANUP_ROADMAP.md
// as the column_detector module and XY-Cut algorithm were deleted
// for PDF spec compliance (they are non-spec-compliant heuristics)
// ============================================================================

// ============================================================================
// Reading Order Tests
// ============================================================================

// NOTE: test_reading_order_tree_based removed - relied on LayoutTree and determine_reading_order (both deleted)

#[test]
#[ignore] // TODO: Layout test needs tuning for analysis parameters
fn test_reading_order_graph_based_simple() {
    let blocks = vec![
        mock_block("TopLeft", 0.0, 0.0, 12.0, false),
        mock_block("TopRight", 100.0, 0.0, 12.0, false),
        mock_block("BottomLeft", 0.0, 50.0, 12.0, false),
        mock_block("BottomRight", 100.0, 50.0, 12.0, false),
    ];

    let order = graph_based_reading_order(&blocks);

    // Reading order: left-to-right, top-to-bottom
    // Should be: 0, 1, 2, 3
    assert_eq!(order[0], 0); // TopLeft first
    assert_eq!(order[1], 1); // TopRight second
    assert_eq!(order[2], 2); // BottomLeft third
    assert_eq!(order[3], 3); // BottomRight fourth
}

#[test]
#[ignore] // TODO: Layout test needs tuning for analysis parameters
fn test_reading_order_two_columns() {
    let blocks = vec![
        mock_block("Col1Line1", 0.0, 0.0, 12.0, false),
        mock_block("Col1Line2", 0.0, 20.0, 12.0, false),
        mock_block("Col2Line1", 300.0, 0.0, 12.0, false),
        mock_block("Col2Line2", 300.0, 20.0, 12.0, false),
    ];

    let order = graph_based_reading_order(&blocks);

    // Reading order should maintain top-to-bottom within columns
    // First block should be a top block (0 or 2)
    assert!(order[0] == 0 || order[0] == 2);

    // All blocks should be included
    assert_eq!(order.len(), 4);
    assert!(order.contains(&0));
    assert!(order.contains(&1));
    assert!(order.contains(&2));
    assert!(order.contains(&3));
}

// ============================================================================
// NOTE: Heading and Table Detection Tests Removed
// These tests were removed in Phase 1 of CLEANUP_ROADMAP.md
// as heading_detector and table_detector modules were deleted
// for PDF spec compliance (they are non-spec-compliant heuristics)

// ============================================================================
// Integration Tests - Full Pipeline
// ============================================================================

// NOTE: test_full_pipeline_two_column_document removed - relied on xy_cut, determine_reading_order, and detect_headings (all deleted)

#[test]
fn test_empty_inputs() {
    // Test all functions with empty inputs
    let empty_blocks: Vec<TextBlock> = vec![];
    let empty_chars: Vec<TextChar> = vec![];

    // Clustering
    assert_eq!(cluster_chars_into_words(&empty_chars, 10.0).len(), 0);
    assert_eq!(cluster_words_into_lines(&empty_blocks, 5.0).len(), 0);

    // Reading order
    assert_eq!(graph_based_reading_order(&empty_blocks).len(), 0);
}

#[test]
fn test_single_element_inputs() {
    // Test all functions with single elements
    let single_char = vec![mock_char('A', 0.0, 0.0, 12.0)];
    let single_block = vec![mock_block("Single", 0.0, 0.0, 12.0, false)];

    // Clustering
    assert_eq!(cluster_chars_into_words(&single_char, 10.0).len(), 1);
    assert_eq!(cluster_words_into_lines(&single_block, 5.0).len(), 1);

    // Reading order
    assert_eq!(graph_based_reading_order(&single_block), vec![0]);
}
