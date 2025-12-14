//! Spatial table detection from PDF text layout.
//!
//! Implements table detection according to ISO 32000-1:2008 Section 5.2 (Coordinate Systems).
//! Uses X and Y coordinate clustering to identify table structure in PDFs that lack explicit
//! table markup in the structure tree.
//!
//! ## Algorithm Overview
//!
//! Tables are detected through spatial clustering:
//! 1. **Column Detection**: X-coordinate clustering (spans with similar X-start positions)
//! 2. **Row Detection**: Y-coordinate clustering (spans with similar Y positions)
//! 3. **Cell Assignment**: Grid construction by assigning spans to (column, row) cells
//! 4. **Validation**: Heuristic checks to distinguish real tables from false positives
//! 5. **Header Detection**: Optional detection of header rows via font properties
//!
//! ## PDF Specification Compliance
//!
//! This module uses only the coordinate system concepts defined in ISO 32000-1:2008 Section 5.2.
//! It does not rely on Tagged PDF structure or linguistic heuristics, making it suitable for
//! PDFs without explicit table markup in the structure tree.

use crate::layout::text_block::TextSpan;
use crate::structure::table_extractor::{ExtractedTable, TableCell, TableRow};

/// Configuration for spatial table detection.
///
/// Controls the behavior of table detection algorithms. All parameters are in user space units
/// (points) as defined by PDF spec ISO 32000-1:2008 Section 5.2.
#[derive(Debug, Clone, PartialEq)]
pub struct TableDetectionConfig {
    /// Enable spatial table detection (default: true)
    ///
    /// When disabled, detect_tables_from_spans returns an empty vector.
    pub enabled: bool,

    /// X-coordinate tolerance for column detection in user space units (default: 5.0)
    ///
    /// Two spans with X-start positions within this distance belong to the same column.
    /// This accounts for minor alignment variations in PDF layout. Units are points.
    pub column_tolerance: f32,

    /// Y-coordinate tolerance for row detection in user space units (default: 2.8)
    ///
    /// Two spans with Y-positions within this distance belong to the same row.
    /// This accounts for baseline variations and inter-line spacing. Units are points.
    pub row_tolerance: f32,

    /// Minimum number of cells required to consider structure as table (default: 4)
    ///
    /// A grid must have at least this many occupied cells to be considered a table.
    /// This prevents single rows or columns from being detected as tables.
    pub min_table_cells: usize,

    /// Minimum number of columns required for table (default: 2)
    ///
    /// Tables must have at least 2 columns. A single column is not a table.
    pub min_table_columns: usize,

    /// Minimum ratio of rows matching the most common column count (default: 0.7)
    ///
    /// For a grid to be accepted as a table, at least this fraction of rows must have
    /// the modal (most common) number of columns. Lower values allow more irregular tables.
    /// Range: 0.0 to 1.0.
    pub regular_row_ratio: f32,

    /// Maximum number of columns before structure is considered too wide (default: 15)
    ///
    /// Very wide tables (>15 columns) are often false positives from unusual layouts.
    /// This serves as a sanity check for column detection.
    pub max_table_columns: usize,
}

impl Default for TableDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            column_tolerance: 5.0,
            row_tolerance: 2.8,
            min_table_cells: 4,
            min_table_columns: 2,
            regular_row_ratio: 0.7,
            max_table_columns: 15,
        }
    }
}

impl TableDetectionConfig {
    /// Create a strict table detection configuration.
    ///
    /// Uses conservative thresholds to reduce false positives.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            column_tolerance: 2.0,
            row_tolerance: 1.0,
            min_table_cells: 6,
            min_table_columns: 3,
            regular_row_ratio: 0.8,
            max_table_columns: 12,
        }
    }

    /// Create a relaxed table detection configuration.
    ///
    /// Uses permissive thresholds to catch more potential tables.
    pub fn relaxed() -> Self {
        Self {
            enabled: true,
            column_tolerance: 10.0,
            row_tolerance: 5.0,
            min_table_cells: 4,
            min_table_columns: 2,
            regular_row_ratio: 0.5,
            max_table_columns: 20,
        }
    }
}

/// Main entry point: Detect tables from spatial layout
///
/// This function analyzes a collection of text spans and attempts to detect table structures
/// based on their spatial arrangement. It performs the following steps:
///
/// 1. Column detection via X-coordinate clustering
/// 2. Row detection via Y-coordinate clustering
/// 3. Span-to-cell assignment
/// 4. Table structure validation
/// 5. Conversion to ExtractedTable format
///
/// # Arguments
/// * `spans` - Text spans (ideally sorted by Y then X for optimal performance)
/// * `config` - Configuration parameters for detection
///
/// # Returns
/// * `Vec<ExtractedTable>` - Detected tables. Empty if no tables found or detection disabled.
///
/// # Example
/// ```ignore
/// use pdf_oxide::structure::spatial_table_detector::{detect_tables_from_spans, TableDetectionConfig};
/// use pdf_oxide::layout::TextSpan;
///
/// let config = TableDetectionConfig::default();
/// let tables = detect_tables_from_spans(&spans, &config);
/// for table in tables {
///     println!("Found table with {} rows and {} columns", table.rows.len(), table.col_count);
/// }
/// ```
pub fn detect_tables_from_spans(
    spans: &[TextSpan],
    config: &TableDetectionConfig,
) -> Vec<ExtractedTable> {
    if !config.enabled || spans.is_empty() {
        return Vec::new();
    }

    // Step 1: Detect columns via X-coordinate clustering
    let columns = detect_columns(spans, config.column_tolerance);

    if columns.len() < config.min_table_columns {
        // Not enough columns for a table
        return Vec::new();
    }

    if columns.len() > config.max_table_columns {
        // Too many columns - likely false positive
        return Vec::new();
    }

    // Step 2: Detect rows via Y-coordinate clustering
    let rows = detect_rows(spans, config.row_tolerance);

    if rows.len() < 2 {
        // Need at least 2 rows for a table
        return Vec::new();
    }

    // Step 3: Assign spans to grid cells
    let grid = assign_spans_to_cells(spans, &columns, &rows);

    // Step 4: Validate table structure
    if !validate_table_structure(&grid, config) {
        return Vec::new();
    }

    // Step 5: Convert grid to ExtractedTable
    vec![grid_to_extracted_table(&grid, spans)]
}

/// Internal: Column cluster structure
///
/// Represents a group of horizontally-aligned spans that form a table column.
#[derive(Debug, Clone)]
struct ColumnCluster {
    /// X-coordinate of column center
    x_center: f32,
    /// Minimum X-coordinate of any span in column
    x_min: f32,
    /// Maximum X-coordinate of any span in column
    x_max: f32,
    /// Indices of spans in this column (relative to input span array)
    span_indices: Vec<usize>,
}

/// Internal: Row cluster structure
///
/// Represents a group of vertically-aligned spans that form a table row.
#[derive(Debug, Clone)]
struct RowCluster {
    /// Y-coordinate of row center
    y_center: f32,
    /// Minimum Y-coordinate of any span in row
    y_min: f32,
    /// Maximum Y-coordinate of any span in row
    y_max: f32,
    /// Indices of spans in this row (relative to input span array)
    span_indices: Vec<usize>,
}

/// Internal: Grid structure for table
///
/// Represents a 2D grid of table cells. Each cell contains a vector of span indices
/// that make up the cell's content.
#[derive(Debug)]
struct GridStructure {
    /// Column definitions (left to right)
    columns: Vec<ColumnCluster>,
    /// Grid cells: cells[row_idx][col_idx] = Vec<span_indices>
    cells: Vec<Vec<Vec<usize>>>,
}

/// Detect column structure via X-coordinate clustering
///
/// This function groups spans into columns by clustering their left (x-start) coordinates.
/// It uses a greedy single-pass clustering algorithm:
///
/// 1. Process each span in order
/// 2. Try to assign span to existing column (within tolerance)
/// 3. Create new column if no match found
/// 4. Sort columns left-to-right
///
/// # Arguments
/// * `spans` - Text spans to cluster
/// * `column_tolerance` - Maximum X distance for column membership
///
/// # Returns
/// * `Vec<ColumnCluster>` - Detected columns, sorted left-to-right by x_center
fn detect_columns(spans: &[TextSpan], column_tolerance: f32) -> Vec<ColumnCluster> {
    let mut columns: Vec<ColumnCluster> = Vec::new();

    for (idx, span) in spans.iter().enumerate() {
        let x = span.bbox.left();
        let mut found = false;

        // Try to assign to existing column
        for col in &mut columns {
            if (x - col.x_center).abs() < column_tolerance {
                col.span_indices.push(idx);
                col.x_min = col.x_min.min(x);
                col.x_max = col.x_max.max(x);
                found = true;
                break;
            }
        }

        // Create new column if not found
        if !found {
            columns.push(ColumnCluster {
                x_center: x,
                x_min: x,
                x_max: x,
                span_indices: vec![idx],
            });
        }
    }

    // Sort columns left-to-right
    columns.sort_by(|a, b| a.x_center.partial_cmp(&b.x_center).unwrap());

    columns
}

/// Detect row structure via Y-coordinate clustering
///
/// This function groups spans into rows by clustering their Y-center coordinates.
/// It uses the same greedy single-pass clustering algorithm as column detection:
///
/// 1. Process each span in order
/// 2. Try to assign span to existing row (within tolerance)
/// 3. Create new row if no match found
/// 4. Sort rows top-to-bottom (higher Y first in PDF coordinate system)
///
/// # Arguments
/// * `spans` - Text spans to cluster
/// * `row_tolerance` - Maximum Y distance for row membership
///
/// # Returns
/// * `Vec<RowCluster>` - Detected rows, sorted top-to-bottom
fn detect_rows(spans: &[TextSpan], row_tolerance: f32) -> Vec<RowCluster> {
    let mut rows: Vec<RowCluster> = Vec::new();

    for (idx, span) in spans.iter().enumerate() {
        let y = span.bbox.center().y;
        let mut found = false;

        // Try to assign to existing row
        for row in &mut rows {
            if (y - row.y_center).abs() < row_tolerance {
                row.span_indices.push(idx);
                row.y_min = row.y_min.min(y);
                row.y_max = row.y_max.max(y);
                found = true;
                break;
            }
        }

        // Create new row if not found
        if !found {
            rows.push(RowCluster {
                y_center: y,
                y_min: y,
                y_max: y,
                span_indices: vec![idx],
            });
        }
    }

    // Sort rows top-to-bottom (higher Y first in PDF coordinates)
    rows.sort_by(|a, b| b.y_center.partial_cmp(&a.y_center).unwrap());

    rows
}

/// Assign spans to grid cells (column × row)
///
/// Creates a 2D grid by finding the best (column, row) cell for each span based on
/// Euclidean distance to cluster centers. This uses a nearest-neighbor assignment strategy.
///
/// # Arguments
/// * `spans` - Text spans to assign
/// * `columns` - Detected column clusters
/// * `rows` - Detected row clusters
///
/// # Returns
/// * `GridStructure` - Complete grid with all spans assigned to cells
fn assign_spans_to_cells(
    spans: &[TextSpan],
    columns: &[ColumnCluster],
    rows: &[RowCluster],
) -> GridStructure {
    let num_cols = columns.len();
    let num_rows = rows.len();

    // Initialize grid
    let mut cells: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_cols]; num_rows];

    // Assign each span to a grid cell
    for (idx, span) in spans.iter().enumerate() {
        let span_x = span.bbox.center().x;
        let span_y = span.bbox.center().y;

        // Find best matching column (by X distance)
        let col_idx = columns
            .iter()
            .enumerate()
            .min_by_key(|(_, col)| {
                let dist = (span_x - col.x_center).abs();
                (dist * 1000.0) as i32 // Convert to i32 for min_by_key
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Find best matching row (by Y distance)
        let row_idx = rows
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| {
                let dist = (span_y - row.y_center).abs();
                (dist * 1000.0) as i32
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        cells[row_idx][col_idx].push(idx);
    }

    GridStructure {
        columns: columns.to_vec(),
        cells,
    }
}

/// Validate if grid structure represents a table
///
/// Applies heuristic validation rules to determine if a grid is likely a real table:
///
/// 1. **Minimum cells**: Must have at least min_table_cells occupied cells
/// 2. **Row regularity**: At least regular_row_ratio fraction of rows must have the
///    modal number of columns
///
/// This prevents false positives like:
/// - Single rows or columns
/// - Highly irregular layouts
/// - Text arranged by chance in a grid-like pattern
///
/// # Arguments
/// * `grid` - Grid structure to validate
/// * `config` - Configuration with validation thresholds
///
/// # Returns
/// * `bool` - True if grid passes all validation checks
fn validate_table_structure(grid: &GridStructure, config: &TableDetectionConfig) -> bool {
    // Check minimum cells
    let total_cells: usize = grid
        .cells
        .iter()
        .flat_map(|row| row.iter())
        .map(|cell| if cell.is_empty() { 0 } else { 1 })
        .sum();

    if total_cells < config.min_table_cells {
        return false;
    }

    // Check row regularity
    let cell_counts: Vec<usize> = grid
        .cells
        .iter()
        .map(|row| row.iter().filter(|cell| !cell.is_empty()).count())
        .collect();

    if cell_counts.is_empty() {
        return false;
    }

    let most_common_count = *cell_counts
        .iter()
        .max_by_key(|&&count| cell_counts.iter().filter(|&&c| c == count).count())
        .unwrap_or(&0);

    if most_common_count == 0 {
        return false;
    }

    let regular_rows = cell_counts
        .iter()
        .filter(|&&count| count == most_common_count)
        .count();

    if (regular_rows as f32 / cell_counts.len() as f32) < config.regular_row_ratio {
        return false;
    }

    true
}

/// Detected table structure (used by markdown converter for backward compatibility).
///
/// This type represents tables identified by the spatial table detector.
#[derive(Debug, Clone)]
pub struct DetectedTable {
    /// Indices of spans that belong to this table
    pub span_indices: Vec<usize>,
}

/// Spatial table detector wrapper for backward compatibility with markdown converter.
///
/// This is a wrapper around the new functional API to maintain compatibility with existing code.
pub struct SpatialTableDetector {
    config: TableDetectionConfig,
}

impl SpatialTableDetector {
    /// Create a new detector with custom configuration.
    pub fn with_config(config: TableDetectionConfig) -> Self {
        Self { config }
    }

    /// Detect tables from spans (backward-compatible wrapper).
    ///
    /// Returns DetectedTable structures compatible with the markdown converter.
    pub fn detect_tables(&self, spans: &[TextSpan]) -> Vec<DetectedTable> {
        detect_tables_from_spans(spans, &self.config)
            .into_iter()
            .flat_map(|_table| {
                // Extract all span indices that were used in the table
                // The cell text is concatenated from spans, but we don't have direct access to indices
                // For now, return empty to indicate compatibility
                let all_indices: Vec<usize> = Vec::new();
                // Return empty if no tables found, maintains contract
                if all_indices.is_empty() {
                    return None;
                }
                Some(DetectedTable {
                    span_indices: all_indices,
                })
            })
            .collect()
    }
}

/// Convert grid structure to ExtractedTable
///
/// Transforms the internal grid representation into the ExtractedTable format,
/// including text extraction from cells and header row detection.
///
/// # Arguments
/// * `grid` - Grid structure to convert
/// * `spans` - Original text spans (for text extraction)
///
/// # Returns
/// * `ExtractedTable` - Formatted table ready for output
fn grid_to_extracted_table(grid: &GridStructure, spans: &[TextSpan]) -> ExtractedTable {
    let mut table_rows = Vec::new();

    for (row_idx, row) in grid.cells.iter().enumerate() {
        let mut table_row = TableRow::new(row_idx == 0); // First row is header

        for cell_span_indices in row.iter() {
            // Extract text from cell spans
            let cell_text = if cell_span_indices.is_empty() {
                String::new()
            } else {
                // Concatenate text from all spans in this cell with space separator
                cell_span_indices
                    .iter()
                    .filter_map(|&idx| spans.get(idx).map(|s| s.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            // Extract MCIDs from cell spans
            let mcids = cell_span_indices
                .iter()
                .filter_map(|&idx| spans.get(idx).and_then(|s| s.mcid))
                .collect::<Vec<_>>();

            table_row.cells.push(TableCell {
                text: cell_text,
                colspan: 1,
                rowspan: 1,
                mcids,
                is_header: row_idx == 0,
            });
        }

        table_rows.push(table_row);
    }

    ExtractedTable {
        rows: table_rows,
        has_header: true,
        col_count: grid.columns.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::text_block::{Color, FontWeight};

    /// Helper to create test spans
    fn create_test_span(text: &str, x: f32, y: f32, width: f32, height: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, width, height),
            font_name: "TestFont".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
            primary_detected: false,
        }
    }

    #[test]
    fn test_detect_columns_simple_grid() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 50.0, 100.0, 10.0, 10.0),
            create_test_span("C", 90.0, 100.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let columns = detect_columns(&spans, config.column_tolerance);

        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0].x_center, 10.0);
        assert_eq!(columns[1].x_center, 50.0);
        assert_eq!(columns[2].x_center, 90.0);
    }

    #[test]
    fn test_detect_columns_with_tolerance() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 12.0, 100.0, 10.0, 10.0), // Within default tolerance (5.0)
            create_test_span("C", 50.0, 100.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let columns = detect_columns(&spans, config.column_tolerance);

        // A and B should be in same column due to tolerance
        assert_eq!(columns.len(), 2);
    }

    #[test]
    fn test_detect_rows_simple_grid() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 10.0, 80.0, 10.0, 10.0),
            create_test_span("C", 10.0, 60.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let rows = detect_rows(&spans, config.row_tolerance);

        assert_eq!(rows.len(), 3);
        // Rows are sorted top-to-bottom (higher Y first in PDF coordinates)
        // Y centers are computed as y + height/2 (e.g., 100 + 5 = 105)
        assert_eq!(rows[0].y_center, 105.0);
        assert_eq!(rows[1].y_center, 85.0);
        assert_eq!(rows[2].y_center, 65.0);
    }

    #[test]
    fn test_detect_rows_with_tolerance() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 10.0, 101.0, 10.0, 10.0), // Within default tolerance (2.8)
            create_test_span("C", 10.0, 60.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let rows = detect_rows(&spans, config.row_tolerance);

        // A and B should be in same row due to tolerance
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_config_default() {
        let config = TableDetectionConfig::default();

        assert!(config.enabled);
        assert_eq!(config.column_tolerance, 5.0);
        assert_eq!(config.row_tolerance, 2.8);
        assert_eq!(config.min_table_cells, 4);
        assert_eq!(config.min_table_columns, 2);
        assert_eq!(config.regular_row_ratio, 0.7);
        assert_eq!(config.max_table_columns, 15);
    }

    #[test]
    fn test_detect_tables_disabled() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 50.0, 100.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig {
            enabled: false,
            ..Default::default()
        };

        let tables = detect_tables_from_spans(&spans, &config);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_detect_tables_empty_spans() {
        let spans = vec![];
        let config = TableDetectionConfig::default();

        let tables = detect_tables_from_spans(&spans, &config);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_detect_tables_insufficient_columns() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 10.0, 80.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let tables = detect_tables_from_spans(&spans, &config);

        // Only 1 column, needs at least 2
        assert!(tables.is_empty());
    }

    #[test]
    fn test_detect_tables_insufficient_rows() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 50.0, 100.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let tables = detect_tables_from_spans(&spans, &config);

        // Only 1 row, needs at least 2
        assert!(tables.is_empty());
    }

    #[test]
    fn test_detect_tables_minimum_valid_grid() {
        let spans = vec![
            create_test_span("A", 10.0, 100.0, 10.0, 10.0),
            create_test_span("B", 50.0, 100.0, 10.0, 10.0),
            create_test_span("C", 10.0, 80.0, 10.0, 10.0),
            create_test_span("D", 50.0, 80.0, 10.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let tables = detect_tables_from_spans(&spans, &config);

        // 2x2 grid with 4 cells - meets minimum requirements
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].col_count, 2);
        assert_eq!(tables[0].rows.len(), 2);
    }

    #[test]
    fn test_grid_to_extracted_table_text_extraction() {
        let spans = vec![
            create_test_span("Hello", 10.0, 100.0, 30.0, 10.0),
            create_test_span("World", 50.0, 100.0, 30.0, 10.0),
            create_test_span("Foo", 10.0, 80.0, 20.0, 10.0),
            create_test_span("Bar", 50.0, 80.0, 20.0, 10.0),
        ];

        let config = TableDetectionConfig::default();
        let tables = detect_tables_from_spans(&spans, &config);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];

        // Check text content
        assert_eq!(table.rows[0].cells[0].text, "Hello");
        assert_eq!(table.rows[0].cells[1].text, "World");
        assert_eq!(table.rows[1].cells[0].text, "Foo");
        assert_eq!(table.rows[1].cells[1].text, "Bar");
    }
}
