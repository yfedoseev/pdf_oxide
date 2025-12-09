//! Table extraction from PDF structure tree.
//!
//! Implements table detection and reconstruction according to ISO 32000-1:2008 Section 14.8.4.3.4
//! (Table Elements).
//!
//! Table structure hierarchy:
//! - Table: Top-level container
//!   - THead: Optional header row group
//!   - TBody: One or more body row groups
//!   - TFoot: Optional footer row group
//! - TR: Table row (contains TH and/or TD cells)
//!   - TH: Table header cell
//!   - TD: Table data cell

use crate::error::Error;
use crate::layout::TextBlock;
use crate::structure::types::{StructChild, StructElem, StructType};

/// A complete extracted table with rows and optional header information.
#[derive(Debug, Clone)]
pub struct ExtractedTable {
    /// Rows of the table (alternating between header and body rows)
    pub rows: Vec<TableRow>,

    /// Whether the table has an explicit header section
    pub has_header: bool,

    /// Number of columns (inferred from first row)
    pub col_count: usize,
}

/// A single row in a table.
#[derive(Debug, Clone)]
pub struct TableRow {
    /// Cells in this row
    pub cells: Vec<TableCell>,

    /// Whether this is a header row
    pub is_header: bool,
}

/// A single cell in a table.
#[derive(Debug, Clone)]
pub struct TableCell {
    /// Text content of the cell
    pub text: String,

    /// Number of columns this cell spans (default 1)
    pub colspan: u32,

    /// Number of rows this cell spans (default 1)
    pub rowspan: u32,

    /// MCID values that make up this cell's content
    pub mcids: Vec<u32>,

    /// Whether this is a header cell
    pub is_header: bool,
}

impl Default for ExtractedTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractedTable {
    /// Create a new extracted table
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            has_header: false,
            col_count: 0,
        }
    }

    /// Add a row to the table
    pub fn add_row(&mut self, row: TableRow) {
        if self.col_count == 0 && !row.cells.is_empty() {
            self.col_count = row.cells.len();
        }
        self.rows.push(row);
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl TableRow {
    /// Create a new table row
    pub fn new(is_header: bool) -> Self {
        Self {
            cells: Vec::new(),
            is_header,
        }
    }

    /// Add a cell to the row
    pub fn add_cell(&mut self, cell: TableCell) {
        self.cells.push(cell);
    }
}

impl TableCell {
    /// Create a new table cell
    pub fn new(text: String, is_header: bool) -> Self {
        Self {
            text,
            colspan: 1,
            rowspan: 1,
            mcids: Vec::new(),
            is_header,
        }
    }

    /// Set colspan
    pub fn with_colspan(mut self, colspan: u32) -> Self {
        self.colspan = colspan;
        self
    }

    /// Set rowspan
    pub fn with_rowspan(mut self, rowspan: u32) -> Self {
        self.rowspan = rowspan;
        self
    }

    /// Add an MCID
    pub fn add_mcid(&mut self, mcid: u32) {
        self.mcids.push(mcid);
    }
}

/// Extract a table from a structure element tree.
///
/// According to PDF spec Section 14.8.4.3.4, a Table element may contain:
/// - Direct TR (table row) children, OR
/// - THead (optional) + TBody (one or more) + TFoot (optional)
///
/// # Arguments
/// * `table_elem` - The Table structure element
/// * `text_blocks` - All text blocks in the document (for MCID matching)
///
/// # Returns
/// * `ExtractedTable` containing all rows and cells
pub fn extract_table(
    table_elem: &StructElem,
    text_blocks: &[TextBlock],
) -> Result<ExtractedTable, Error> {
    let mut table = ExtractedTable::new();

    // Check table structure
    let has_thead = table_elem
        .children
        .iter()
        .any(|child| matches!(child, StructChild::StructElem(elem) if elem.struct_type == StructType::THead));

    if has_thead {
        table.has_header = true;
    }

    // Process all children
    for child in &table_elem.children {
        match child {
            StructChild::StructElem(elem) => match elem.struct_type {
                StructType::TR => {
                    // Direct row in table
                    let row = extract_row(elem, text_blocks, false)?;
                    table.add_row(row);
                },
                StructType::THead => {
                    // Header row group
                    extract_row_group(elem, text_blocks, true, &mut table)?;
                },
                StructType::TBody => {
                    // Body row group
                    extract_row_group(elem, text_blocks, false, &mut table)?;
                },
                StructType::TFoot => {
                    // Footer row group
                    extract_row_group(elem, text_blocks, false, &mut table)?;
                },
                _ => {
                    // Skip other elements (caption, etc.)
                },
            },
            StructChild::MarkedContentRef { .. } => {
                // Skip direct content references
            },
            StructChild::ObjectRef(_, _) => {
                // Skip object references
            },
        }
    }

    Ok(table)
}

/// Extract rows from a row group (THead, TBody, TFoot).
fn extract_row_group(
    group_elem: &StructElem,
    text_blocks: &[TextBlock],
    is_header: bool,
    table: &mut ExtractedTable,
) -> Result<(), Error> {
    for child in &group_elem.children {
        match child {
            StructChild::StructElem(elem) if elem.struct_type == StructType::TR => {
                let row = extract_row(elem, text_blocks, is_header)?;
                table.add_row(row);
            },
            _ => {
                // Skip non-row elements
            },
        }
    }
    Ok(())
}

/// Extract a single row (TR element).
fn extract_row(
    tr_elem: &StructElem,
    text_blocks: &[TextBlock],
    force_header: bool,
) -> Result<TableRow, Error> {
    let mut row = TableRow::new(force_header);

    for child in &tr_elem.children {
        match child {
            StructChild::StructElem(elem) => match elem.struct_type {
                StructType::TH => {
                    // Header cell
                    let cell = extract_cell(elem, text_blocks, true)?;
                    row.add_cell(cell);
                },
                StructType::TD => {
                    // Data cell
                    let cell = extract_cell(elem, text_blocks, false)?;
                    row.add_cell(cell);
                },
                _ => {
                    // Skip other elements
                },
            },
            StructChild::MarkedContentRef { .. } => {
                // Skip direct content references
            },
            StructChild::ObjectRef(_, _) => {
                // Skip object references
            },
        }
    }

    Ok(row)
}

/// Extract a single cell (TH or TD element).
fn extract_cell(
    cell_elem: &StructElem,
    text_blocks: &[TextBlock],
    is_header: bool,
) -> Result<TableCell, Error> {
    // Collect all MCIDs from this cell
    let mut mcids = Vec::new();
    collect_mcids(cell_elem, &mut mcids);

    // Find all text blocks that match these MCIDs
    let mut cell_text = String::new();
    for mcid in &mcids {
        for block in text_blocks {
            if let Some(block_mcid) = block.mcid {
                if block_mcid == *mcid {
                    if !cell_text.is_empty() && !cell_text.ends_with(' ') {
                        cell_text.push(' ');
                    }
                    cell_text.push_str(&block.text);
                    break;
                }
            }
        }
    }

    let mut cell = TableCell::new(cell_text.trim().to_string(), is_header);
    cell.mcids = mcids;

    Ok(cell)
}

/// Recursively collect all MCIDs from a structure element and its children.
fn collect_mcids(elem: &StructElem, mcids: &mut Vec<u32>) {
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { mcid, .. } => {
                mcids.push(*mcid);
            },
            StructChild::StructElem(child_elem) => {
                // Recursively collect from child elements
                collect_mcids(child_elem, mcids);
            },
            StructChild::ObjectRef(_, _) => {
                // Skip object references
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_table_new() {
        let table = ExtractedTable::new();
        assert!(table.is_empty());
        assert_eq!(table.col_count, 0);
        assert!(!table.has_header);
    }

    #[test]
    fn test_table_row_new() {
        let header_row = TableRow::new(true);
        assert!(header_row.is_header);
        assert!(header_row.cells.is_empty());

        let body_row = TableRow::new(false);
        assert!(!body_row.is_header);
    }

    #[test]
    fn test_table_cell_new() {
        let cell = TableCell::new("Hello".to_string(), false);
        assert_eq!(cell.text, "Hello");
        assert!(!cell.is_header);
        assert_eq!(cell.colspan, 1);
        assert_eq!(cell.rowspan, 1);
        assert!(cell.mcids.is_empty());
    }

    #[test]
    fn test_table_cell_with_spans() {
        let cell = TableCell::new("Data".to_string(), false)
            .with_colspan(2)
            .with_rowspan(3);

        assert_eq!(cell.colspan, 2);
        assert_eq!(cell.rowspan, 3);
    }

    #[test]
    fn test_table_cell_header() {
        let cell = TableCell::new("Header".to_string(), true);
        assert!(cell.is_header);
    }

    #[test]
    fn test_table_row_add_cells() {
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Cell1".to_string(), false));
        row.add_cell(TableCell::new("Cell2".to_string(), false));

        assert_eq!(row.cells.len(), 2);
        assert_eq!(row.cells[0].text, "Cell1");
        assert_eq!(row.cells[1].text, "Cell2");
    }

    #[test]
    fn test_extracted_table_add_rows() {
        let mut table = ExtractedTable::new();
        let mut row1 = TableRow::new(false);
        row1.add_cell(TableCell::new("A".to_string(), false));
        row1.add_cell(TableCell::new("B".to_string(), false));

        table.add_row(row1);
        assert_eq!(table.col_count, 2);
        assert_eq!(table.rows.len(), 1);
    }

    #[test]
    fn test_extracted_table_has_header() {
        let mut table = ExtractedTable::new();
        assert!(!table.has_header);

        table.has_header = true;
        assert!(table.has_header);
    }
}
