//! DBSCAN clustering for text layout analysis.
//!
//! This module implements DBSCAN (Density-Based Spatial Clustering of Applications
//! with Noise) for grouping characters into words and words into lines.
//!
//! Note: This module is currently feature-gated but linfa-clustering API has changed.
//! For Phase 8 MVP, we use simplified distance-based clustering instead.

use crate::layout::text_block::{TextBlock, TextChar};

/// Cluster characters into words using DBSCAN.
///
/// This uses the spatial positions of characters to group them into words.
/// Characters that are close together (within `epsilon` distance) are grouped
/// into the same word.
///
/// # Arguments
///
/// * `chars` - The characters to cluster
/// * `epsilon` - The maximum distance between characters in the same word
///
/// # Returns
///
/// A vector of clusters, where each cluster is a vector of character indices.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "ml")]
/// # {
/// use pdf_oxide::geometry::Rect;
/// use pdf_oxide::layout::{TextChar, FontWeight, Color, clustering::cluster_chars_into_words};
///
/// let chars = vec![
///     TextChar {
///         char: 'H',
///         bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         color: Color::black(),
///         mcid: None,
///     },
///     TextChar {
///         char: 'i',
///         bbox: Rect::new(11.0, 0.0, 5.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         color: Color::black(),
///         mcid: None,
///     },
/// ];
///
/// let clusters = cluster_chars_into_words(&chars, 3.0);
/// // Characters within 3.0 units are grouped together
/// # }
/// ```
#[cfg(feature = "ml")]
pub fn cluster_chars_into_words(chars: &[TextChar], epsilon: f32) -> Vec<Vec<usize>> {
    if chars.is_empty() {
        return vec![];
    }

    if chars.len() == 1 {
        return vec![vec![0]];
    }

    // Simplified distance-based clustering
    // Group characters that are within epsilon distance
    let mut visited = vec![false; chars.len()];
    let mut clusters: Vec<Vec<usize>> = vec![];

    for i in 0..chars.len() {
        if visited[i] {
            continue;
        }

        let mut cluster = vec![i];
        visited[i] = true;

        // Find all chars within epsilon distance
        let mut j = 0;
        while j < cluster.len() {
            let current_idx = cluster[j];
            let current_center = chars[current_idx].bbox.center();

            // Check all unvisited characters
            for k in 0..chars.len() {
                if visited[k] {
                    continue;
                }

                let other_center = chars[k].bbox.center();
                let distance = ((current_center.x - other_center.x).powi(2)
                    + (current_center.y - other_center.y).powi(2))
                .sqrt();

                if distance <= epsilon {
                    cluster.push(k);
                    visited[k] = true;
                }
            }

            j += 1;
        }

        // Sort cluster by x-coordinate
        cluster.sort_by(|&a, &b| chars[a].bbox.x.partial_cmp(&chars[b].bbox.x).unwrap());
        clusters.push(cluster);
    }

    clusters
}

/// Cluster words into lines using DBSCAN based on Y-coordinate.
///
/// This groups words that have similar vertical positions into lines.
///
/// # Arguments
///
/// * `words` - The word blocks to cluster
/// * `epsilon_y` - The maximum vertical distance between words in the same line
///
/// # Returns
///
/// A vector of line clusters, where each cluster is a vector of word indices.
/// Words within each line are sorted left-to-right.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "ml")]
/// # {
/// use pdf_oxide::geometry::Rect;
/// use pdf_oxide::layout::{TextChar, TextBlock, FontWeight, Color, clustering::cluster_words_into_lines};
///
/// let chars1 = vec![
///     TextChar {
///         char: 'H',
///         bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         color: Color::black(),
///         mcid: None,
///     },
/// ];
/// let word1 = TextBlock::from_chars(chars1);
///
/// let chars2 = vec![
///     TextChar {
///         char: 'W',
///         bbox: Rect::new(50.0, 1.0, 10.0, 12.0),
///         font_name: "Times".to_string(),
///         font_size: 12.0,
///         font_weight: FontWeight::Normal,
///         color: Color::black(),
///         mcid: None,
///     },
/// ];
/// let word2 = TextBlock::from_chars(chars2);
///
/// let words = vec![word1, word2];
/// let lines = cluster_words_into_lines(&words, 5.0);
/// // Words within 5.0 units vertically are grouped into the same line
/// # }
/// ```
#[cfg(feature = "ml")]
pub fn cluster_words_into_lines(words: &[TextBlock], epsilon_y: f32) -> Vec<Vec<usize>> {
    if words.is_empty() {
        return vec![];
    }

    if words.len() == 1 {
        return vec![vec![0]];
    }

    // Simplified Y-coordinate clustering
    // Group words with similar Y positions
    let mut visited = vec![false; words.len()];
    let mut clusters: Vec<Vec<usize>> = vec![];

    for i in 0..words.len() {
        if visited[i] {
            continue;
        }

        let mut cluster = vec![i];
        visited[i] = true;

        // Find all words with similar Y position
        let mut j = 0;
        while j < cluster.len() {
            let current_idx = cluster[j];
            let current_y = words[current_idx].bbox.y;

            // Check all unvisited words
            for k in 0..words.len() {
                if visited[k] {
                    continue;
                }

                let other_y = words[k].bbox.y;
                let distance = (current_y - other_y).abs();

                if distance <= epsilon_y {
                    cluster.push(k);
                    visited[k] = true;
                }
            }

            j += 1;
        }

        // Sort cluster by X position (left-to-right)
        cluster.sort_by(|&a, &b| words[a].bbox.x.partial_cmp(&words[b].bbox.x).unwrap());
        clusters.push(cluster);
    }

    clusters
}

// Fallback implementations when ML feature is not enabled

/// Cluster characters into words using spatial DBSCAN (fallback).
///
/// This is the fallback implementation used when the `ml` feature is not enabled.
/// It uses true spatial DBSCAN that checks ALL characters within epsilon distance,
/// not just consecutive ones. This fixes word segmentation issues where characters
/// may be out of order in the input array.
#[cfg(not(feature = "ml"))]
pub fn cluster_chars_into_words(chars: &[TextChar], epsilon: f32) -> Vec<Vec<usize>> {
    if chars.is_empty() {
        return vec![];
    }

    if chars.len() == 1 {
        return vec![vec![0]];
    }

    // True spatial DBSCAN: check ALL characters within epsilon distance
    let mut visited = vec![false; chars.len()];
    let mut clusters: Vec<Vec<usize>> = vec![];

    // Debug: Check if we have characters near Y=1535 (the problematic line)
    let debug_y = 1535.0;
    let has_debug_chars = chars.iter().any(|c| (c.bbox.y - debug_y).abs() < 10.0);

    if has_debug_chars {
        log::warn!("🔍 DEBUG: Processing line near Y={}, epsilon={:.1}", debug_y, epsilon);
        log::warn!("Characters in this region:");
        for (idx, ch) in chars.iter().enumerate() {
            if (ch.bbox.y - debug_y).abs() < 10.0 {
                log::warn!("  [{}] '{}' at X={:.1}, Y={:.1}", idx, ch.char, ch.bbox.x, ch.bbox.y);
            }
        }
    }

    for i in 0..chars.len() {
        if visited[i] {
            continue;
        }

        let mut cluster = vec![i];
        visited[i] = true;

        // Debug: Log if this is a character in the problematic region
        let is_debug_char = (chars[i].bbox.y - debug_y).abs() < 10.0;
        if is_debug_char && has_debug_chars {
            log::warn!(
                "🔍 Starting cluster with char[{}] '{}' at X={:.1}",
                i,
                chars[i].char,
                chars[i].bbox.x
            );
        }

        // BFS to find all connected characters
        let mut j = 0;
        while j < cluster.len() {
            let current_idx = cluster[j];
            let current = &chars[current_idx];
            let current_center = current.bbox.center();

            // Check ALL unvisited characters (not just consecutive ones!)
            for k in 0..chars.len() {
                if visited[k] {
                    continue;
                }

                let other = &chars[k];
                let other_center = other.bbox.center();

                // Compute spatial distance
                let dx = (current_center.x - other_center.x).abs();
                let dy = (current_center.y - other_center.y).abs();

                // Word boundary heuristic: same line + close horizontally
                // Use font size for vertical tolerance (more robust than fixed epsilon)
                let same_line = dy < current.font_size * 0.5;
                let close_horiz = dx <= epsilon;

                // Debug: Log distance checks for problematic chars
                if is_debug_char && has_debug_chars && (other.bbox.y - debug_y).abs() < 10.0 {
                    if same_line && close_horiz {
                        log::warn!(
                            "  ✅ Adding '{}' (dx={:.1}, dy={:.1}) - CONNECTED",
                            other.char,
                            dx,
                            dy
                        );
                    } else if same_line {
                        log::warn!(
                            "  ❌ Rejecting '{}' (dx={:.1}, dy={:.1}) - too far horizontally",
                            other.char,
                            dx,
                            dy
                        );
                    }
                }

                if same_line && close_horiz {
                    cluster.push(k);
                    visited[k] = true;
                }
            }

            j += 1;
        }

        // Sort cluster by X position (left-to-right)
        cluster.sort_by(|&a, &b| chars[a].bbox.x.partial_cmp(&chars[b].bbox.x).unwrap());

        // Debug: Show the final cluster if it contains debug chars
        if is_debug_char && has_debug_chars {
            let cluster_text: String = cluster.iter().map(|&idx| chars[idx].char).collect();
            log::warn!("🔍 Final cluster: \"{}\"", cluster_text);
        }

        clusters.push(cluster);
    }

    clusters
}

/// Cluster words into lines using column-aware Y-coordinate grouping (fallback).
///
/// This is a simplified implementation used when the `ml` feature is not enabled.
/// It groups words that have similar Y coordinates AND are horizontally connected,
/// avoiding mixing words from different columns.
#[cfg(not(feature = "ml"))]
pub fn cluster_words_into_lines(words: &[TextBlock], epsilon_y: f32) -> Vec<Vec<usize>> {
    if words.is_empty() {
        return vec![];
    }

    let mut clusters: Vec<Vec<usize>> = vec![];
    let mut assigned = vec![false; words.len()];

    // Estimate column gap threshold: if two words are more than 50pt apart horizontally,
    // they're likely in different columns
    let column_gap_threshold = 50.0;

    for i in 0..words.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![i];
        assigned[i] = true;

        // Use BFS to find horizontally connected words at the same Y
        let mut j = 0;
        while j < cluster.len() {
            let current_idx = cluster[j];
            let current_word = &words[current_idx];

            // Check all unassigned words
            for k in 0..words.len() {
                if assigned[k] {
                    continue;
                }

                let other_word = &words[k];

                // Check if on same line (Y coordinate)
                let y_dist = (current_word.bbox.y - other_word.bbox.y).abs();
                if y_dist > epsilon_y {
                    continue;
                }

                // Check if horizontally connected (not across column gap)
                let x_dist = (current_word.bbox.right() - other_word.bbox.left())
                    .abs()
                    .min((other_word.bbox.right() - current_word.bbox.left()).abs());

                // Words are in the same line if they're close horizontally
                // (within column gap threshold)
                if x_dist < column_gap_threshold {
                    cluster.push(k);
                    assigned[k] = true;
                }
            }

            j += 1;
        }

        // Sort by x-coordinate
        cluster.sort_by(|&a, &b| words[a].bbox.x.partial_cmp(&words[b].bbox.x).unwrap());

        clusters.push(cluster);
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight};

    fn mock_char(c: char, x: f32, y: f32) -> TextChar {
        TextChar {
            char: c,
            bbox: Rect::new(x, y, 10.0, 12.0),
            font_name: "Times".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            color: Color::black(),
            mcid: None,
            is_italic: false,
        }
    }

    #[test]
    fn test_cluster_chars_empty() {
        let chars = vec![];
        let clusters = cluster_chars_into_words(&chars, 8.0);
        assert_eq!(clusters.len(), 0);
    }

    #[test]
    fn test_cluster_chars_single() {
        let chars = vec![mock_char('A', 0.0, 0.0)];
        let clusters = cluster_chars_into_words(&chars, 8.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_cluster_chars_into_words() {
        // "Hello World" - two words
        let chars = vec![
            mock_char('H', 0.0, 0.0),
            mock_char('e', 11.0, 0.0),
            mock_char('l', 22.0, 0.0),
            mock_char('l', 33.0, 0.0),
            mock_char('o', 44.0, 0.0),
            // Big gap
            mock_char('W', 100.0, 0.0),
            mock_char('o', 111.0, 0.0),
            mock_char('r', 122.0, 0.0),
            mock_char('l', 133.0, 0.0),
            mock_char('d', 144.0, 0.0),
        ];

        let clusters = cluster_chars_into_words(&chars, 20.0);

        // Should have 2 clusters
        assert_eq!(clusters.len(), 2);

        // First cluster: "Hello" (indices 0-4)
        assert!(clusters[0].contains(&0));
        assert!(clusters[0].contains(&1));
        assert!(clusters[0].contains(&2));
        assert!(clusters[0].contains(&3));
        assert!(clusters[0].contains(&4));

        // Second cluster: "World" (indices 5-9)
        assert!(clusters[1].contains(&5));
        assert!(clusters[1].contains(&6));
        assert!(clusters[1].contains(&7));
        assert!(clusters[1].contains(&8));
        assert!(clusters[1].contains(&9));
    }

    #[test]
    fn test_cluster_words_empty() {
        let words: Vec<TextBlock> = vec![];
        let clusters = cluster_words_into_lines(&words, 5.0);
        assert_eq!(clusters.len(), 0);
    }

    #[test]
    fn test_cluster_words_single() {
        let chars = vec![mock_char('A', 0.0, 0.0)];
        let word = TextBlock::from_chars(chars);
        let words = vec![word];

        let clusters = cluster_words_into_lines(&words, 5.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_cluster_words_into_lines() {
        // Two lines: "Hello World" on line 1, "Foo Bar" on line 2
        let word1 = TextBlock::from_chars(vec![mock_char('H', 0.0, 0.0)]);
        let word2 = TextBlock::from_chars(vec![mock_char('W', 50.0, 1.0)]); // Same line
        let word3 = TextBlock::from_chars(vec![mock_char('F', 0.0, 30.0)]); // Different line
        let word4 = TextBlock::from_chars(vec![mock_char('B', 50.0, 31.0)]); // Same as word3

        let words = vec![word1, word2, word3, word4];
        let lines = cluster_words_into_lines(&words, 5.0);

        // Should have 2 lines
        assert_eq!(lines.len(), 2);

        // Verify clustering
        // Line 1: words 0 and 1
        assert!(lines[0].contains(&0));
        assert!(lines[0].contains(&1));

        // Line 2: words 2 and 3
        assert!(lines[1].contains(&2));
        assert!(lines[1].contains(&3));
    }

    #[test]
    fn test_words_sorted_by_x_in_line() {
        // Create words in reverse order (right to left) on same line
        // Using realistic word spacing (< 50pt column gap threshold)
        let word1 = TextBlock::from_chars(vec![mock_char('W', 40.0, 0.0)]); // "World" at x=40
        let word2 = TextBlock::from_chars(vec![mock_char('H', 0.0, 1.0)]); // "Hello" at x=0

        let words = vec![word1, word2];
        let lines = cluster_words_into_lines(&words, 5.0);

        assert_eq!(lines.len(), 1);
        // Should be sorted: index 1 (x=0) before index 0 (x=40)
        assert_eq!(lines[0], vec![1, 0]);
    }
}
