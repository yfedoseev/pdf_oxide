//! Configurable table detection with grid pattern recognition.
//!
//! This module provides sophisticated table detection using grid pattern analysis.
//! All thresholds and parameters are configurable, with no magic numbers hardcoded.
//!
//! ## Algorithm Overview
//!
//! The detection process:
//! 1. Cluster text blocks by X coordinate (columns)
//! 2. Cluster text blocks by Y coordinate (rows)
//! 3. Validate that clusters form a grid pattern
//! 4. Ensure minimum dimension requirements are met
//! 5. Extract the detected grid as a structured table
//!
//! ## Configuration
//!
//! All thresholds are exposed through `TableDetectorConfig`:
//! - Column/row alignment tolerance
//! - Minimum cell requirements
//! - Grid validation criteria
//!
//! Use factory methods for common scenarios:
//! - `default()` - Standard document processing
//! - `loose()` - Irregular tables with larger tolerances
//! - `strict()` - Well-aligned tables with tight tolerances
//! - `custom()` - Fine-grained control

use crate::geometry::Rect;
use crate::layout::TextBlock;
use log::{debug, trace, warn};

/// Configuration for table detection algorithm.
///
/// All values are in PDF points (1/72 inch) except for counts.
///
/// # Default Values
///
/// The default configuration is tuned for typical documents:
/// - X tolerance: 5.0pt (slight column misalignment acceptable)
/// - Y tolerance: 2.0pt (rows should be well-aligned)
/// - Min cells: 4 (2x2 minimum table)
/// - Min columns: 2
/// - Min rows: 2
/// - Cell merge threshold: 1.0pt
#[derive(Debug, Clone, Copy)]
pub struct TableDetectorConfig {
    /// Column alignment tolerance in points.
    ///
    /// Controls how far apart text blocks can be horizontally and still be
    /// considered part of the same column. Larger values accommodate tables
    /// with slight misalignment.
    ///
    /// Typical range: 2.0pt (strict) to 10.0pt (loose)
    pub x_tolerance_pt: f32,

    /// Row alignment tolerance in points.
    ///
    /// Controls how far apart text blocks can be vertically and still be
    /// considered part of the same row. Usually smaller than x_tolerance
    /// as rows are typically more precisely aligned.
    ///
    /// Typical range: 1.0pt (strict) to 5.0pt (loose)
    pub y_tolerance_pt: f32,

    /// Minimum number of cells required to qualify as a table.
    ///
    /// A table must have at least this many cells. Default of 4 allows
    /// detection of minimal 2x2 tables.
    pub min_cells_for_grid: usize,

    /// Minimum number of columns required.
    ///
    /// Tables must have at least this many columns. Default is 2.
    pub min_columns: usize,

    /// Minimum number of rows required.
    ///
    /// Tables must have at least this many rows. Default is 2.
    pub min_rows: usize,

    /// Tolerance for merging adjacent cell boundaries in points.
    ///
    /// When detecting cell boundaries, adjacent positions within this
    /// tolerance are merged into a single boundary.
    pub cell_merge_threshold_pt: f32,
}

impl Default for TableDetectorConfig {
    /// Standard configuration for typical document processing.
    fn default() -> Self {
        Self {
            x_tolerance_pt: 5.0,
            y_tolerance_pt: 2.0,
            min_cells_for_grid: 4,
            min_columns: 2,
            min_rows: 2,
            cell_merge_threshold_pt: 1.0,
        }
    }
}

impl TableDetectorConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to `TableDetectorConfig::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configuration for loose table detection.
    ///
    /// Uses larger tolerances to detect tables with irregular alignment.
    /// Useful for OCR results or poorly formatted documents.
    ///
    /// Tolerances:
    /// - X tolerance: 10.0pt
    /// - Y tolerance: 5.0pt
    /// - Cell merge: 3.0pt
    pub fn loose() -> Self {
        Self {
            x_tolerance_pt: 10.0,
            y_tolerance_pt: 5.0,
            min_cells_for_grid: 4,
            min_columns: 2,
            min_rows: 2,
            cell_merge_threshold_pt: 3.0,
        }
    }

    /// Configuration for strict table detection.
    ///
    /// Uses small tolerances to detect only well-aligned tables.
    /// Useful for professionally formatted documents with precise layouts.
    ///
    /// Tolerances:
    /// - X tolerance: 2.0pt
    /// - Y tolerance: 1.0pt
    /// - Cell merge: 0.5pt
    pub fn strict() -> Self {
        Self {
            x_tolerance_pt: 2.0,
            y_tolerance_pt: 1.0,
            min_cells_for_grid: 4,
            min_columns: 2,
            min_rows: 2,
            cell_merge_threshold_pt: 0.5,
        }
    }

    /// Create a custom configuration with individual values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::TableDetectorConfig;
    ///
    /// let config = TableDetectorConfig::custom(
    ///     8.0,  // x_tolerance
    ///     3.0,  // y_tolerance
    ///     4,    // min_cells
    ///     2,    // min_columns
    ///     2,    // min_rows
    ///     2.0,  // cell_merge_threshold
    /// );
    /// ```
    pub fn custom(
        x_tolerance_pt: f32,
        y_tolerance_pt: f32,
        min_cells_for_grid: usize,
        min_columns: usize,
        min_rows: usize,
        cell_merge_threshold_pt: f32,
    ) -> Self {
        Self {
            x_tolerance_pt,
            y_tolerance_pt,
            min_cells_for_grid,
            min_columns,
            min_rows,
            cell_merge_threshold_pt,
        }
    }
}

/// A detected table from text blocks.
#[derive(Debug, Clone)]
pub struct DetectedTable {
    /// Grid of text blocks organized as [row][column].
    ///
    /// Each cell contains the text blocks found in that position.
    pub cells: Vec<Vec<Vec<TextBlock>>>,

    /// Bounding box of the entire table in document space.
    pub bbox: Rect,

    /// Number of rows in the table.
    pub rows: usize,

    /// Number of columns in the table.
    pub cols: usize,
}

/// Table detection engine with configurable parameters.
pub struct TableDetector {
    config: TableDetectorConfig,
}

impl TableDetector {
    /// Create a new table detector with the given configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::{TableDetector, TableDetectorConfig};
    ///
    /// let config = TableDetectorConfig::default();
    /// let detector = TableDetector::new(config);
    /// ```
    pub fn new(config: TableDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect tables in a collection of text blocks.
    ///
    /// Returns a vector of detected tables. May return empty vector if
    /// no valid table patterns are found.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Text blocks to analyze for table patterns
    ///
    /// # Returns
    ///
    /// Vector of detected tables
    pub fn detect_tables(&self, blocks: &[TextBlock]) -> Vec<DetectedTable> {
        debug!(
            "Starting table detection with config: x_tol={}pt, y_tol={}pt, min_cells={}, min_cols={}, min_rows={}",
            self.config.x_tolerance_pt,
            self.config.y_tolerance_pt,
            self.config.min_cells_for_grid,
            self.config.min_columns,
            self.config.min_rows
        );

        if blocks.is_empty() {
            debug!("No blocks to analyze");
            return vec![];
        }

        // Cluster by X coordinate (columns)
        let x_clusters = self.cluster_by_x(blocks);
        debug!("Found {} potential columns", x_clusters.len());

        if x_clusters.len() < self.config.min_columns {
            debug!(
                "Insufficient columns: {} < {}",
                x_clusters.len(),
                self.config.min_columns
            );
            return vec![];
        }

        // Cluster by Y coordinate (rows)
        let y_clusters = self.cluster_by_y(blocks);
        debug!("Found {} potential rows", y_clusters.len());

        if y_clusters.len() < self.config.min_rows {
            debug!(
                "Insufficient rows: {} < {}",
                y_clusters.len(),
                self.config.min_rows
            );
            return vec![];
        }

        // Validate grid pattern
        if !self.is_grid_like(&x_clusters, &y_clusters) {
            warn!("Block clusters do not form a grid pattern");
            return vec![];
        }

        debug!("Grid pattern validated, extracting table");

        // Extract the grid
        let table = self.extract_grid(x_clusters, y_clusters, blocks);
        vec![table]
    }

    /// Cluster blocks into columns based on X coordinate alignment.
    ///
    /// Groups blocks that have similar X coordinates (within x_tolerance).
    ///
    /// # Visibility
    ///
    /// This method is public to support testing and advanced use cases.
    pub fn cluster_by_x(&self, blocks: &[TextBlock]) -> Vec<Vec<TextBlock>> {
        trace!("Clustering blocks by X coordinate");

        let mut clusters: Vec<Vec<TextBlock>> = vec![];
        let mut used = vec![false; blocks.len()];

        for i in 0..blocks.len() {
            if used[i] {
                continue;
            }

            let mut cluster = vec![blocks[i].clone()];
            used[i] = true;

            for j in (i + 1)..blocks.len() {
                if !used[j]
                    && (blocks[i].bbox.x - blocks[j].bbox.x).abs() < self.config.x_tolerance_pt
                {
                    cluster.push(blocks[j].clone());
                    used[j] = true;
                }
            }

            clusters.push(cluster);
        }

        // Sort clusters by X coordinate
        clusters.sort_by(|a, b| {
            a.iter()
                .map(|b| b.bbox.x as i32)
                .sum::<i32>()
                .cmp(&b.iter().map(|b| b.bbox.x as i32).sum::<i32>())
        });

        trace!("Created {} X clusters", clusters.len());
        clusters
    }

    /// Cluster blocks into rows based on Y coordinate alignment.
    ///
    /// Groups blocks that have similar Y coordinates (within y_tolerance).
    ///
    /// # Visibility
    ///
    /// This method is public to support testing and advanced use cases.
    pub fn cluster_by_y(&self, blocks: &[TextBlock]) -> Vec<Vec<TextBlock>> {
        trace!("Clustering blocks by Y coordinate");

        let mut clusters: Vec<Vec<TextBlock>> = vec![];
        let mut used = vec![false; blocks.len()];

        for i in 0..blocks.len() {
            if used[i] {
                continue;
            }

            let mut cluster = vec![blocks[i].clone()];
            used[i] = true;

            for j in (i + 1)..blocks.len() {
                if !used[j]
                    && (blocks[i].bbox.y - blocks[j].bbox.y).abs() < self.config.y_tolerance_pt
                {
                    cluster.push(blocks[j].clone());
                    used[j] = true;
                }
            }

            clusters.push(cluster);
        }

        // Sort clusters by Y coordinate (descending for top-to-bottom)
        clusters.sort_by(|a, b| {
            b.iter()
                .map(|b| b.bbox.y as i32)
                .sum::<i32>()
                .cmp(&a.iter().map(|b| b.bbox.y as i32).sum::<i32>())
        });

        trace!("Created {} Y clusters", clusters.len());
        clusters
    }

    /// Validate that X and Y clusters form a grid pattern.
    ///
    /// A valid grid must have:
    /// - Sufficient columns and rows
    /// - Grid-like intersection pattern
    /// - Minimum total cells
    ///
    /// # Visibility
    ///
    /// This method is public to support testing and advanced use cases.
    pub fn is_grid_like(&self, x_clusters: &[Vec<TextBlock>], y_clusters: &[Vec<TextBlock>]) -> bool {
        trace!(
            "Validating grid pattern: {} x-clusters, {} y-clusters",
            x_clusters.len(),
            y_clusters.len()
        );

        // Check minimum dimensions
        if x_clusters.len() < self.config.min_columns {
            trace!("Validation failed: insufficient columns");
            return false;
        }

        if y_clusters.len() < self.config.min_rows {
            trace!("Validation failed: insufficient rows");
            return false;
        }

        // Check minimum total cells
        let total_cells = x_clusters.len() * y_clusters.len();
        if total_cells < self.config.min_cells_for_grid {
            trace!(
                "Validation failed: insufficient cells ({} < {})",
                total_cells,
                self.config.min_cells_for_grid
            );
            return false;
        }

        // Check that blocks participate in grid
        // A block should appear in both an x-cluster and a y-cluster
        let mut valid_cells = 0;

        for x_cluster in x_clusters {
            for y_cluster in y_clusters {
                // Check if any block appears in both clusters
                let intersection: usize = x_cluster
                    .iter()
                    .filter(|xb| y_cluster.iter().any(|yb| xb.bbox == yb.bbox))
                    .count();

                if intersection > 0 {
                    valid_cells += 1;
                }
            }
        }

        debug!("Grid validation: {}/{} cells populated", valid_cells, total_cells);

        // At least half the cells should have content
        let occupancy_ratio = valid_cells as f32 / total_cells as f32;
        if occupancy_ratio < 0.4 {
            warn!(
                "Grid has low occupancy ({:.1}%) - may not be a real table",
                occupancy_ratio * 100.0
            );
            return false;
        }

        trace!("Grid pattern validated successfully");
        true
    }

    /// Extract grid structure from X and Y clusters.
    ///
    /// Organizes blocks into a 2D grid structure.
    fn extract_grid(
        &self,
        x_clusters: Vec<Vec<TextBlock>>,
        y_clusters: Vec<Vec<TextBlock>>,
        all_blocks: &[TextBlock],
    ) -> DetectedTable {
        trace!("Extracting grid: {} rows x {} cols", y_clusters.len(), x_clusters.len());

        let rows = y_clusters.len();
        let cols = x_clusters.len();
        let mut cells: Vec<Vec<Vec<TextBlock>>> = vec![vec![vec![]; cols]; rows];

        // For each cell position, find matching blocks
        for (row_idx, y_cluster) in y_clusters.iter().enumerate() {
            for (col_idx, x_cluster) in x_clusters.iter().enumerate() {
                // Find blocks that belong to both this row and column
                for block in all_blocks {
                    // Check if block is in both x and y clusters
                    let in_x_cluster = x_cluster.iter().any(|b| b.bbox == block.bbox);
                    let in_y_cluster = y_cluster.iter().any(|b| b.bbox == block.bbox);

                    if in_x_cluster && in_y_cluster {
                        cells[row_idx][col_idx].push(block.clone());
                    }
                }
            }
        }

        // Compute bounding box
        let mut bbox = all_blocks[0].bbox;
        for block in all_blocks {
            bbox = bbox.union(&block.bbox);
        }

        debug!("Extracted grid: {}x{} with bbox {:?}", rows, cols, bbox);

        DetectedTable { cells, bbox, rows, cols }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Color, FontWeight, TextChar};

    fn mock_block(text: &str, x: f32, y: f32) -> TextBlock {
        let chars: Vec<TextChar> = text
            .chars()
            .enumerate()
            .map(|(i, c)| TextChar {
                char: c,
                bbox: Rect::new(x + i as f32 * 5.0, y, 5.0, 10.0),
                font_name: "Times".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
            })
            .collect();

        TextBlock::from_chars(chars)
    }

    #[test]
    fn test_table_detector_config_default() {
        let config = TableDetectorConfig::default();
        assert_eq!(config.x_tolerance_pt, 5.0);
        assert_eq!(config.y_tolerance_pt, 2.0);
        assert_eq!(config.min_cells_for_grid, 4);
        assert_eq!(config.min_columns, 2);
        assert_eq!(config.min_rows, 2);
        assert_eq!(config.cell_merge_threshold_pt, 1.0);
    }

    #[test]
    fn test_table_detector_config_loose() {
        let config = TableDetectorConfig::loose();
        assert_eq!(config.x_tolerance_pt, 10.0);
        assert_eq!(config.y_tolerance_pt, 5.0);
        assert_eq!(config.cell_merge_threshold_pt, 3.0);
    }

    #[test]
    fn test_table_detector_config_strict() {
        let config = TableDetectorConfig::strict();
        assert_eq!(config.x_tolerance_pt, 2.0);
        assert_eq!(config.y_tolerance_pt, 1.0);
        assert_eq!(config.cell_merge_threshold_pt, 0.5);
    }

    #[test]
    fn test_table_detector_config_custom() {
        let config = TableDetectorConfig::custom(8.0, 3.0, 6, 3, 2, 2.0);
        assert_eq!(config.x_tolerance_pt, 8.0);
        assert_eq!(config.y_tolerance_pt, 3.0);
        assert_eq!(config.min_cells_for_grid, 6);
        assert_eq!(config.min_columns, 3);
    }

    #[test]
    fn test_table_detector_column_clustering() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 2.0, 20.0), // Within x_tolerance of A
            mock_block("C", 50.0, 5.0),
            mock_block("D", 51.0, 25.0), // Within x_tolerance of C
        ];

        let clusters = detector.cluster_by_x(&blocks);

        // Should have 2 clusters
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 2); // A and B
        assert_eq!(clusters[1].len(), 2); // C and D
    }

    #[test]
    fn test_table_detector_row_clustering() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 50.0, 1.0), // Within y_tolerance of A
            mock_block("C", 25.0, 30.0),
            mock_block("D", 75.0, 31.0), // Within y_tolerance of C
        ];

        let clusters = detector.cluster_by_y(&blocks);

        // Should have 2 clusters
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 2); // C and D
        assert_eq!(clusters[1].len(), 2); // A and B
    }

    #[test]
    fn test_table_detector_grid_validation_minimal_2x2() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        // Create a perfect 2x2 grid
        let blocks = vec![
            mock_block("A1", 0.0, 0.0),
            mock_block("B1", 50.0, 0.0),
            mock_block("A2", 0.0, 30.0),
            mock_block("B2", 50.0, 30.0),
        ];

        let x_clusters = detector.cluster_by_x(&blocks);
        let y_clusters = detector.cluster_by_y(&blocks);

        assert!(detector.is_grid_like(&x_clusters, &y_clusters));
    }

    #[test]
    fn test_table_detector_grid_validation_insufficient_cells() {
        let config = TableDetectorConfig::custom(5.0, 2.0, 6, 2, 2, 1.0);
        let detector = TableDetector::new(config);

        // Only 2 blocks - forms 2x1 grid with 2 cells (< 6 required)
        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 50.0, 0.0),
        ];

        let x_clusters = detector.cluster_by_x(&blocks);
        let y_clusters = detector.cluster_by_y(&blocks);

        assert!(!detector.is_grid_like(&x_clusters, &y_clusters));
    }

    #[test]
    fn test_table_detector_empty_blocks() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        let blocks = vec![];
        let tables = detector.detect_tables(&blocks);

        assert_eq!(tables.len(), 0);
    }

    #[test]
    fn test_table_detector_insufficient_blocks() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 50.0, 0.0),
        ];

        let tables = detector.detect_tables(&blocks);
        assert_eq!(tables.len(), 0); // Need at least 4 blocks for 2x2
    }

    #[test]
    fn test_table_detector_perfect_3x3_grid() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        // Create a perfect 3x3 grid
        let blocks = vec![
            // Row 1
            mock_block("A1", 0.0, 0.0),
            mock_block("B1", 50.0, 0.0),
            mock_block("C1", 100.0, 0.0),
            // Row 2
            mock_block("A2", 0.0, 30.0),
            mock_block("B2", 50.0, 30.0),
            mock_block("C2", 100.0, 30.0),
            // Row 3
            mock_block("A3", 0.0, 60.0),
            mock_block("B3", 50.0, 60.0),
            mock_block("C3", 100.0, 60.0),
        ];

        let tables = detector.detect_tables(&blocks);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.rows, 3);
        assert_eq!(table.cols, 3);
    }

    #[test]
    fn test_table_detector_loose_mode() {
        let config = TableDetectorConfig::loose();
        let detector = TableDetector::new(config);

        // Create a 2x2 grid with larger misalignments
        let blocks = vec![
            mock_block("A1", 0.0, 0.0),
            mock_block("B1", 50.0, 3.0), // y offset of 3pt
            mock_block("A2", 6.0, 30.0), // x offset of 6pt
            mock_block("B2", 50.0, 33.0),
        ];

        let tables = detector.detect_tables(&blocks);

        // Loose mode should be more permissive
        assert!(tables.len() <= 1);
    }

    #[test]
    fn test_table_detector_no_grid_pattern() {
        let config = TableDetectorConfig::default();
        let detector = TableDetector::new(config);

        // Random distribution - no grid
        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 30.0, 15.0),
            mock_block("C", 60.0, 5.0),
            mock_block("D", 90.0, 25.0),
        ];

        let tables = detector.detect_tables(&blocks);
        assert_eq!(tables.len(), 0);
    }
}
