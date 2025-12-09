//! Reading order determination for layout analysis.
//!
//! This module provides algorithms for determining the correct reading order
//! of text blocks in a document. The primary approach is graph-based topological
//! sorting which doesn't rely on layout trees.

use crate::layout::text_block::TextBlock;
use std::collections::{HashSet, VecDeque};

/// Determine reading order using graph-based topological sorting.
///
/// This is more sophisticated than tree-based ordering and can handle
/// complex layouts where blocks don't fit neatly into a tree structure.
///
/// Uses Kahn's algorithm for topological sorting of a directed acyclic graph
/// where edges represent "precedes" relationships.
///
/// # Arguments
///
/// * `blocks` - The text blocks to order
///
/// # Returns
///
/// A vector of block indices in reading order.
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::geometry::Rect;
/// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color};
/// use pdf_oxide::layout::reading_order::graph_based_reading_order;
///
/// let chars1 = vec![
///     TextChar {
///         char: 'A',
///         bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         is_italic: false,
///         color: Color::black(),
///     },
/// ];
/// let block1 = TextBlock::from_chars(chars1);
///
/// let chars2 = vec![
///     TextChar {
///         char: 'B',
///         bbox: Rect::new(50.0, 0.0, 10.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         is_italic: false,
///         color: Color::black(),
///     },
/// ];
/// let block2 = TextBlock::from_chars(chars2);
///
/// let blocks = vec![block1, block2];
/// let order = graph_based_reading_order(&blocks);
/// // Block at x=0 comes before block at x=50
/// assert_eq!(order, vec![0, 1]);
/// ```ignore
pub fn graph_based_reading_order(blocks: &[TextBlock]) -> Vec<usize> {
    if blocks.is_empty() {
        return vec![];
    }

    if blocks.len() == 1 {
        return vec![0];
    }

    let n = blocks.len();
    let mut graph = vec![HashSet::new(); n];

    // Build precedence graph: edge i->j means block i precedes block j
    for i in 0..n {
        for j in 0..n {
            if i != j && precedes(&blocks[i], &blocks[j]) {
                graph[i].insert(j);
            }
        }
    }

    // Topological sort using Kahn's algorithm
    kahn_sort(&graph)
}

/// Determine if block `a` precedes block `b` in reading order.
///
/// Reading order rules:
/// 1. If blocks are on the same line (similar Y), left precedes right
/// 2. Otherwise, top precedes bottom
///
/// Note: PDF coordinates have origin at bottom-left, Y increases upward.
/// So "top precedes bottom" means larger Y precedes smaller Y.
fn precedes(a: &TextBlock, b: &TextBlock) -> bool {
    // Tolerance for "same line" detection (5 units)
    let y_tolerance = 5.0;

    if (a.bbox.top() - b.bbox.top()).abs() < y_tolerance {
        // Same line: left precedes right
        a.bbox.left() < b.bbox.left()
    } else {
        // Different lines: top precedes bottom
        // PDF coordinates: origin at bottom-left, Y increases upward
        // So larger Y (top of page) comes before smaller Y (bottom of page)
        a.bbox.top() > b.bbox.top()
    }
}

/// Perform topological sort using Kahn's algorithm.
///
/// # Arguments
///
/// * `graph` - Adjacency list representation of a DAG
///
/// # Returns
///
/// A topologically sorted list of node indices.
fn kahn_sort(graph: &[HashSet<usize>]) -> Vec<usize> {
    let n = graph.len();

    // Compute in-degree for each node
    let mut in_degree = vec![0; n];
    for edges in graph {
        for &node in edges {
            in_degree[node] += 1;
        }
    }

    // Start with nodes that have no incoming edges
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    let mut result = vec![];

    while let Some(node) = queue.pop_front() {
        result.push(node);

        // Remove edges from this node
        for &next in &graph[node] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    // If result doesn't contain all nodes, there was a cycle (shouldn't happen with proper input)
    if result.len() != n {
        // Fallback: return all nodes in original order
        (0..n).collect()
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight, TextChar};

    fn mock_block(text: &str, x: f32, y: f32) -> TextBlock {
        let chars: Vec<TextChar> = text
            .chars()
            .enumerate()
            .map(|(i, c)| TextChar {
                char: c,
                bbox: Rect::new(x + i as f32 * 10.0, y, 10.0, 12.0),
                font_name: "Times".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                is_italic: false,
                color: Color::black(),
                mcid: None,
            })
            .collect();

        TextBlock::from_chars(chars)
    }

    #[test]
    fn test_precedes_same_line() {
        let left = mock_block("Left", 0.0, 0.0);
        let right = mock_block("Right", 100.0, 1.0);

        assert!(precedes(&left, &right));
        assert!(!precedes(&right, &left));
    }

    #[test]
    fn test_precedes_different_lines() {
        // PDF coordinates: Y increases upward, so top has LARGER Y
        let top = mock_block("Top", 0.0, 100.0); // Y=100 (top)
        let bottom = mock_block("Bottom", 0.0, 50.0); // Y=50 (bottom)

        assert!(precedes(&top, &bottom));
        assert!(!precedes(&bottom, &top));
    }

    #[test]
    fn test_graph_based_simple() {
        // PDF coordinates: Y increases upward
        let blocks = vec![
            mock_block("A", 0.0, 100.0),   // Top-left (Y=100)
            mock_block("B", 100.0, 100.0), // Top-right (Y=100)
            mock_block("C", 0.0, 50.0),    // Bottom-left (Y=50)
            mock_block("D", 100.0, 50.0),  // Bottom-right (Y=50)
        ];

        let order = graph_based_reading_order(&blocks);

        // Should read: A, B, C, D (left-to-right, top-to-bottom)
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_graph_based_two_columns() {
        // PDF coordinates: Y increases upward
        let blocks = vec![
            mock_block("Col1-Line1", 0.0, 100.0), // Left column, top (Y=100)
            mock_block("Col1-Line2", 0.0, 50.0),  // Left column, bottom (Y=50)
            mock_block("Col2-Line1", 300.0, 100.0), // Right column, top (Y=100)
            mock_block("Col2-Line2", 300.0, 50.0), // Right column, bottom (Y=50)
        ];

        let order = graph_based_reading_order(&blocks);

        // Reading order should maintain relative positions
        // First block should be from the top (0 or 2, both at Y=100)
        assert!(order[0] == 0 || order[0] == 2);

        // All blocks should be included
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_kahn_sort_simple() {
        // Graph: 0 -> 1 -> 2
        let graph = vec![
            vec![1].into_iter().collect(),
            vec![2].into_iter().collect(),
            HashSet::new(),
        ];

        let sorted = kahn_sort(&graph);
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_kahn_sort_branching() {
        // Graph: 0 -> 1
        //        0 -> 2
        //        1 -> 3
        //        2 -> 3
        let graph = vec![
            vec![1, 2].into_iter().collect(),
            vec![3].into_iter().collect(),
            vec![3].into_iter().collect(),
            HashSet::new(),
        ];

        let sorted = kahn_sort(&graph);

        // 0 must come first, 3 must come last
        assert_eq!(sorted[0], 0);
        assert_eq!(sorted[3], 3);
        // 1 and 2 can be in either order
        assert!(sorted[1] == 1 || sorted[1] == 2);
        assert!(sorted[2] == 1 || sorted[2] == 2);
    }

    #[test]
    fn test_graph_based_empty() {
        let blocks: Vec<TextBlock> = vec![];
        let order = graph_based_reading_order(&blocks);
        assert_eq!(order.len(), 0);
    }

    #[test]
    fn test_graph_based_single() {
        let blocks = vec![mock_block("Single", 0.0, 0.0)];
        let order = graph_based_reading_order(&blocks);
        assert_eq!(order, vec![0]);
    }
}
