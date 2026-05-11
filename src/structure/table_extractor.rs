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
use crate::geometry::Rect;
use crate::layout::TextBlock;
use crate::structure::types::{StructChild, StructElem, StructType};

/// A complete extracted table with rows and optional header information.
#[derive(Debug, Clone)]
pub struct Table {
    /// Rows of the table (alternating between header and body rows)
    pub rows: Vec<TableRow>,

    /// Whether the table has an explicit header section
    pub has_header: bool,

    /// Number of columns (inferred from first row)
    pub col_count: usize,

    /// Bounding box of the table region (used to exclude table spans from normal rendering)
    pub bbox: Option<Rect>,
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

    /// Original text spans that make up this cell's content, with
    /// font/style metadata preserved for format-aware rendering.
    pub spans: Vec<crate::layout::TextSpan>,

    /// Number of columns this cell spans (default 1)
    pub colspan: u32,

    /// Number of rows this cell spans (default 1)
    pub rowspan: u32,

    /// MCID values that make up this cell's content
    pub mcids: Vec<u32>,

    /// Bounding box of the cell (v0.3.14)
    pub bbox: Option<Rect>,

    /// Whether this is a header cell
    pub is_header: bool,
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}

impl Table {
    /// Create a new extracted table
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            has_header: false,
            col_count: 0,
            bbox: None,
        }
    }

    /// Check whether this table looks like a real data grid as opposed to
    /// spurious spatial output (form layouts, label-colon-value pairs).
    ///
    /// A real grid (per #457 Step 4) has:
    /// - ≥2 rows
    /// - ≥2 columns
    /// - Consistent column population: at least 50% of rows must have
    ///   at least 2 non-empty cells. Filters out the common
    ///   form-as-table false positive where rows look like
    ///   `| Single label, all other slots empty | | | |`.
    pub fn is_real_grid(&self) -> bool {
        if self.col_count < 2 || self.rows.len() < 2 {
            return false;
        }
        let rows_with_two_or_more_filled_cells = self
            .rows
            .iter()
            .filter(|r| r.cells.iter().filter(|c| !c.text.trim().is_empty()).count() >= 2)
            .count();
        let ratio = rows_with_two_or_more_filled_cells as f32 / self.rows.len() as f32;

        // Wide tables (≥ 8 columns) are high-risk false positives: prose sentences
        // can be split into many single-phrase cells by decorative rule lines.
        // Real wide data tables have most rows densely filled (≥ 60% of columns);
        // prose-split false tables have highly variable row fill counts (some rows
        // have 1-2 filled cells, others have 10+), so the fraction of "dense" rows
        // is well below 70%.
        if self.col_count >= 8 {
            let min_dense = ((self.col_count as f32 * 0.6) as usize).max(2);
            let dense_rows = self
                .rows
                .iter()
                .filter(|r| {
                    r.cells.iter().filter(|c| !c.text.trim().is_empty()).count() >= min_dense
                })
                .count();
            let dense_row_ratio = dense_rows as f32 / self.rows.len() as f32;
            return self.rows.len() >= 3 && ratio >= 0.7 && dense_row_ratio >= 0.70;
        }

        ratio >= 0.5
    }

    /// Render the table as clean, space-padded plain text.
    pub fn render_text(&self) -> String {
        let col_count = self.col_count;
        if col_count == 0 || self.rows.is_empty() {
            return String::new();
        }

        // Calculate column widths from cell content
        let mut col_widths = vec![0usize; col_count];
        for row in &self.rows {
            let mut col_idx = 0;
            for cell in &row.cells {
                if cell.colspan == 1 && col_idx < col_count {
                    let w = cell.text.trim().chars().count();
                    col_widths[col_idx] = col_widths[col_idx].max(w);
                }
                col_idx += cell.colspan as usize;
            }
        }

        // Ensure minimum width of 2 per column
        for w in &mut col_widths {
            if *w < 2 {
                *w = 2;
            }
        }

        // Trim trailing empty columns (no non-empty cell contributes content
        // to that column, including cells with colspan > 1 that cover it).
        let effective_cols = {
            let mut eff = col_widths.len();
            while eff > 0 {
                let col = eff - 1;
                let all_empty = self.rows.iter().all(|row| {
                    let mut ci = 0;
                    for cell in &row.cells {
                        let span = cell.colspan as usize;
                        let covers_col = ci <= col && col < ci + span;
                        if covers_col {
                            return cell.text.trim().is_empty();
                        }
                        ci += span;
                    }
                    true
                });
                if all_empty {
                    eff -= 1;
                } else {
                    break;
                }
            }
            eff
        };
        if effective_cols == 0 {
            return String::new();
        }
        let col_widths = &col_widths[..effective_cols];

        // Detect right-aligned columns (all non-empty cells look like numbers/currency)
        let is_right_aligned: Vec<bool> = (0..effective_cols)
            .map(|c| {
                let mut has_content = false;
                for row in &self.rows {
                    let mut ci = 0;
                    for cell in &row.cells {
                        if ci == c && cell.colspan == 1 {
                            let t = cell.text.trim();
                            if !t.is_empty() {
                                has_content = true;
                                // Check if it looks like a number or currency value
                                let stripped: String = t
                                    .chars()
                                    .filter(|ch| {
                                        !matches!(
                                            ch,
                                            '$' | '€'
                                                | '£'
                                                | ','
                                                | ' '
                                                | '%'
                                                | '+'
                                                | '-'
                                                | '('
                                                | ')'
                                        )
                                    })
                                    .collect();
                                if stripped.is_empty() || stripped.parse::<f64>().is_err() {
                                    return false;
                                }
                            }
                        }
                        ci += cell.colspan as usize;
                    }
                }
                has_content
            })
            .collect();

        let mut output = String::new();

        for row in &self.rows {
            let mut col_idx = 0;
            let mut cells_text = Vec::new();
            for cell in &row.cells {
                let text = cell.text.trim();
                if col_idx < effective_cols {
                    // For colspan > 1, calculate merged width
                    let width = if cell.colspan > 1 {
                        let end = (col_idx + cell.colspan as usize).min(effective_cols);
                        let base: usize = col_widths[col_idx..end].iter().sum();
                        // Add 2 spaces per gap between merged columns
                        base + (end - col_idx).saturating_sub(1) * 2
                    } else {
                        col_widths[col_idx]
                    };
                    let formatted = if cell.colspan == 1 && is_right_aligned[col_idx] {
                        format!("{:>width$}", text, width = width)
                    } else {
                        format!("{:<width$}", text, width = width)
                    };
                    cells_text.push(formatted);
                } else {
                    cells_text.push(text.to_string());
                }
                col_idx += cell.colspan as usize;
            }
            output.push_str(cells_text.join("  ").trim_end());
            output.push('\n');
        }

        output
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

    /// Check if any cell in this row has a colspan > 1 (v0.3.16).
    pub fn has_colspan(&self) -> bool {
        self.cells.iter().any(|c| c.colspan > 1)
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
            spans: Vec::new(),
            colspan: 1,
            rowspan: 1,
            mcids: Vec::new(),
            bbox: None,
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

/// Find all Table structure elements in the structure tree for a given page.
///
/// Recursively walks the structure tree to collect StructElem nodes where
/// `struct_type == StructType::Table` and the element (or any descendant)
/// has marked content on the specified page.
///
/// # Arguments
/// * `struct_tree` - The structure tree root
/// * `page_num` - Page number to match (0-based)
///
/// # Returns
/// * `Vec<&StructElem>` - Table elements found for the page
pub fn find_table_elements(
    struct_tree: &crate::structure::types::StructTreeRoot,
    page_num: u32,
) -> Vec<&StructElem> {
    let mut tables = Vec::new();
    for elem in &struct_tree.root_elements {
        collect_table_elements(elem, page_num, &mut tables);
    }
    tables
}

/// Recursively collect Table elements that have content on the given page.
fn collect_table_elements<'a>(
    elem: &'a StructElem,
    page_num: u32,
    tables: &mut Vec<&'a StructElem>,
) {
    if elem.struct_type == StructType::Table {
        if element_has_page_content(elem, page_num) {
            tables.push(elem);
        }
        return; // Don't recurse into table children looking for nested tables
    }

    for child in &elem.children {
        if let StructChild::StructElem(child_elem) = child {
            collect_table_elements(child_elem, page_num, tables);
        }
    }
}

/// Check if a structure element or any descendant has marked content on the given page.
fn element_has_page_content(elem: &StructElem, page_num: u32) -> bool {
    // Check the element's own page attribute
    if elem.page == Some(page_num) {
        return true;
    }

    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { page, .. } => {
                if *page == page_num {
                    return true;
                }
            },
            StructChild::StructElem(child_elem) => {
                if element_has_page_content(child_elem, page_num) {
                    return true;
                }
            },
            StructChild::ObjectRef(_, _) => {},
        }
    }

    false
}

/// Extract a table from a structure element tree using TextSpans (MCID matching).
///
/// Converts TextSpans to a format suitable for MCID-based cell text extraction,
/// then delegates to the standard `extract_table` function.
///
/// # Arguments
/// * `table_elem` - The Table structure element
/// * `spans` - Text spans from the page (with MCID values)
///
/// # Returns
/// * `Table` containing all rows and cells
pub fn extract_table_from_spans(
    table_elem: &StructElem,
    spans: &[crate::layout::TextSpan],
) -> Result<Table, Error> {
    // Convert spans to TextBlocks for MCID matching, applying column-spanning
    // decimal split so that "12.11" (sailing score columns) becomes "12 11".
    let text_blocks: Vec<TextBlock> = spans
        .iter()
        .filter(|s| s.mcid.is_some())
        .map(|s| {
            let text = span_text_for_cell(s);
            TextBlock {
                chars: Vec::new(),
                bbox: s.bbox,
                text,
                avg_font_size: s.font_size,
                dominant_font: s.font_name.clone(),
                is_bold: s.font_weight.is_bold(),
                is_italic: s.is_italic,
                mcid: s.mcid,
            }
        })
        .collect();
    extract_table(table_elem, &text_blocks)
}

/// Return the display text for a span when used as a table cell token.
/// Mirrors `PdfDocument::push_span_text`: splits column-spanning decimals
/// (e.g. "12.11" across adjacent score columns) at the decimal point.
pub(super) fn span_text_for_cell(span: &crate::layout::TextSpan) -> String {
    let text = &span.text;
    // Must be an "N.M" pattern with all-digit parts and a single dot.
    let dot_pos = match text.find('.') {
        Some(p) if p > 0 && p < text.len() - 1 => p,
        _ => return text.clone(),
    };
    if text[dot_pos + 1..].contains('.') {
        return text.clone();
    }
    if !text[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
        return text.clone();
    }
    if !text[dot_pos + 1..].chars().all(|c| c.is_ascii_digit()) {
        return text.clone();
    }
    let char_count = text.chars().count();
    let expected_width = if !span.char_widths.is_empty() {
        let cw_sum: f32 = span.char_widths.iter().sum();
        cw_sum * (char_count as f32 / span.char_widths.len() as f32)
    } else if span.font_size > 0.0 {
        // 0.50em per char: digits are narrower than average; keeps the
        // fallback from producing false negatives on word_spans (char_widths=[]).
        span.font_size * 0.50 * char_count as f32
    } else {
        return text.clone();
    };
    let gap = span.bbox.width - expected_width;
    if span.font_size > 0.0 && gap > span.font_size * 1.0 {
        format!("{} {}", &text[..dot_pos], &text[dot_pos + 1..])
    } else {
        text.clone()
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
/// * `Table` containing all rows and cells
pub fn extract_table(table_elem: &StructElem, text_blocks: &[TextBlock]) -> Result<Table, Error> {
    let mut table = Table::new();

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
    table: &mut Table,
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

    // Find all text blocks that match these MCIDs, joining them with position-aware
    // spacing: insert a space only when there is a genuine horizontal gap between
    // adjacent spans on the same line, or when spans are on different lines.
    // This prevents spurious spaces inside CJK expressions like "Q（peu/d）" whose
    // glyphs are stored as separate marked-content runs that abut each other.
    let mut cell_text = String::new();
    let mut prev_block: Option<&TextBlock> = None;
    for mcid in &mcids {
        for block in text_blocks {
            if let Some(block_mcid) = block.mcid {
                if block_mcid == *mcid {
                    if !cell_text.is_empty() {
                        let need_space = if let Some(prev) = prev_block {
                            let y_diff = (block.bbox.y - prev.bbox.y).abs();
                            let line_h = prev.bbox.height.max(block.bbox.height);
                            if y_diff > line_h * 0.5 {
                                // Different lines — always insert a space.
                                true
                            } else {
                                // Same line — only insert a space when there is an actual
                                // horizontal gap (> 15% of font size, matching document.rs).
                                let gap = block.bbox.x - (prev.bbox.x + prev.bbox.width);
                                let font_size =
                                    prev.avg_font_size.max(block.avg_font_size).max(1.0);
                                gap > font_size * 0.15
                            }
                        } else {
                            !cell_text.ends_with(' ')
                        };
                        if need_space {
                            cell_text.push(' ');
                        }
                    }
                    cell_text.push_str(&block.text);
                    prev_block = Some(block);
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
    use crate::structure::types::StructTreeRoot;

    #[test]
    fn test_table_new() {
        let table = Table::new();
        assert!(table.is_empty());
        assert_eq!(table.col_count, 0);
        assert!(!table.has_header);
        assert!(table.bbox.is_none());
    }

    #[test]
    fn test_table_bbox() {
        let mut table = Table::new();
        assert!(table.bbox.is_none());

        table.bbox = Some(Rect::new(10.0, 20.0, 100.0, 50.0));
        assert!(table.bbox.is_some());
        let bbox = table.bbox.unwrap();
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 50.0);
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
    fn test_table_add_rows() {
        let mut table = Table::new();
        let mut row1 = TableRow::new(false);
        row1.add_cell(TableCell::new("A".to_string(), false));
        row1.add_cell(TableCell::new("B".to_string(), false));

        table.add_row(row1);
        assert_eq!(table.col_count, 2);
        assert_eq!(table.rows.len(), 1);
    }

    #[test]
    fn test_table_has_header() {
        let mut table = Table::new();
        assert!(!table.has_header);

        table.has_header = true;
        assert!(table.has_header);
    }

    // ============================================================================
    // find_table_elements() tests
    // ============================================================================

    /// Helper: create a minimal Table StructElem with MarkedContentRefs on a given page
    fn make_table_elem(page: u32, mcids: &[u32]) -> StructElem {
        let mut table = StructElem::new(StructType::Table);
        let mut tr = StructElem::new(StructType::TR);
        for &mcid in mcids {
            let mut td = StructElem::new(StructType::TD);
            td.add_child(StructChild::MarkedContentRef { mcid, page });
            tr.add_child(StructChild::StructElem(Box::new(td)));
        }
        table.add_child(StructChild::StructElem(Box::new(tr)));
        table
    }

    #[test]
    fn test_find_table_elements_finds_table_on_matching_page() {
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(make_table_elem(0, &[1, 2]));

        let tables = find_table_elements(&tree, 0);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].struct_type, StructType::Table);
    }

    #[test]
    fn test_find_table_elements_skips_table_on_different_page() {
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(make_table_elem(1, &[1, 2]));

        let tables = find_table_elements(&tree, 0);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_find_table_elements_empty_tree() {
        let tree = StructTreeRoot::new();
        let tables = find_table_elements(&tree, 0);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_find_table_elements_multiple_tables() {
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(make_table_elem(0, &[1, 2]));
        tree.add_root_element(make_table_elem(0, &[3, 4]));

        let tables = find_table_elements(&tree, 0);
        assert_eq!(tables.len(), 2);
    }

    #[test]
    fn test_find_table_elements_nested_in_section() {
        let mut tree = StructTreeRoot::new();
        let mut sect = StructElem::new(StructType::Sect);
        sect.add_child(StructChild::StructElem(Box::new(make_table_elem(0, &[1]))));
        tree.add_root_element(sect);

        let tables = find_table_elements(&tree, 0);
        assert_eq!(tables.len(), 1);
    }

    #[test]
    fn test_find_table_elements_table_with_page_attribute() {
        let mut tree = StructTreeRoot::new();
        let mut table = StructElem::new(StructType::Table);
        table.page = Some(2);
        // No MarkedContentRef children, but page attribute matches
        tree.add_root_element(table);

        let tables = find_table_elements(&tree, 2);
        assert_eq!(tables.len(), 1);
    }

    #[test]
    fn test_find_table_elements_mixed_pages() {
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(make_table_elem(0, &[1]));
        tree.add_root_element(make_table_elem(1, &[2]));
        tree.add_root_element(make_table_elem(0, &[3]));

        let page0_tables = find_table_elements(&tree, 0);
        assert_eq!(page0_tables.len(), 2);

        let page1_tables = find_table_elements(&tree, 1);
        assert_eq!(page1_tables.len(), 1);
    }

    // ============================================================================
    // element_has_page_content() tests
    // ============================================================================

    #[test]
    fn test_element_has_page_content_via_mcid() {
        let mut elem = StructElem::new(StructType::P);
        elem.add_child(StructChild::MarkedContentRef { mcid: 1, page: 3 });

        assert!(element_has_page_content(&elem, 3));
        assert!(!element_has_page_content(&elem, 0));
    }

    #[test]
    fn test_element_has_page_content_via_page_attribute() {
        let mut elem = StructElem::new(StructType::P);
        elem.page = Some(5);

        assert!(element_has_page_content(&elem, 5));
        assert!(!element_has_page_content(&elem, 0));
    }

    #[test]
    fn test_element_has_page_content_recursive() {
        let mut parent = StructElem::new(StructType::Sect);
        let mut child = StructElem::new(StructType::P);
        child.add_child(StructChild::MarkedContentRef { mcid: 1, page: 2 });
        parent.add_child(StructChild::StructElem(Box::new(child)));

        assert!(element_has_page_content(&parent, 2));
        assert!(!element_has_page_content(&parent, 0));
    }

    #[test]
    fn test_element_has_page_content_empty() {
        let elem = StructElem::new(StructType::P);
        assert!(!element_has_page_content(&elem, 0));
    }

    #[test]
    fn test_element_has_page_content_object_ref_ignored() {
        let mut elem = StructElem::new(StructType::P);
        elem.add_child(StructChild::ObjectRef(1, 0));
        assert!(!element_has_page_content(&elem, 0));
    }

    // ============================================================================
    // extract_table_from_spans() tests
    // ============================================================================

    fn make_text_span(text: &str, mcid: Option<u32>) -> crate::layout::TextSpan {
        use crate::layout::text_block::{Color, FontWeight};

        crate::layout::TextSpan {
            artifact_type: None,
            text: text.to_string(),
            bbox: Rect::new(0.0, 0.0, 50.0, 12.0),
            font_name: "Test".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
            primary_detected: false,
            char_widths: vec![],
        }
    }

    #[test]
    fn test_extract_table_from_spans_basic() {
        // Build a simple Table > TR > [TD, TD] structure
        let mut table_elem = StructElem::new(StructType::Table);
        let mut tr = StructElem::new(StructType::TR);
        let mut td1 = StructElem::new(StructType::TD);
        td1.add_child(StructChild::MarkedContentRef { mcid: 10, page: 0 });
        let mut td2 = StructElem::new(StructType::TD);
        td2.add_child(StructChild::MarkedContentRef { mcid: 11, page: 0 });
        tr.add_child(StructChild::StructElem(Box::new(td1)));
        tr.add_child(StructChild::StructElem(Box::new(td2)));
        table_elem.add_child(StructChild::StructElem(Box::new(tr)));

        let spans = vec![
            make_text_span("Hello", Some(10)),
            make_text_span("World", Some(11)),
            make_text_span("Unrelated", Some(99)),
        ];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].cells.len(), 2);
        assert_eq!(result.rows[0].cells[0].text, "Hello");
        assert_eq!(result.rows[0].cells[1].text, "World");
    }

    #[test]
    fn test_extract_table_from_spans_no_matching_mcids() {
        let mut table_elem = StructElem::new(StructType::Table);
        let mut tr = StructElem::new(StructType::TR);
        let mut td = StructElem::new(StructType::TD);
        td.add_child(StructChild::MarkedContentRef { mcid: 10, page: 0 });
        tr.add_child(StructChild::StructElem(Box::new(td)));
        table_elem.add_child(StructChild::StructElem(Box::new(tr)));

        // Spans have different MCIDs
        let spans = vec![make_text_span("Other", Some(99))];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].cells[0].text, ""); // No matching content
    }

    #[test]
    fn test_extract_table_from_spans_filters_no_mcid_spans() {
        let mut table_elem = StructElem::new(StructType::Table);
        let mut tr = StructElem::new(StructType::TR);
        let mut td = StructElem::new(StructType::TD);
        td.add_child(StructChild::MarkedContentRef { mcid: 5, page: 0 });
        tr.add_child(StructChild::StructElem(Box::new(td)));
        table_elem.add_child(StructChild::StructElem(Box::new(tr)));

        // Mix of spans with and without MCIDs
        let spans = vec![
            make_text_span("No MCID", None),
            make_text_span("Has MCID", Some(5)),
        ];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert_eq!(result.rows[0].cells[0].text, "Has MCID");
    }

    #[test]
    fn test_extract_table_from_spans_with_thead() {
        let mut table_elem = StructElem::new(StructType::Table);

        // THead > TR > TH
        let mut thead = StructElem::new(StructType::THead);
        let mut hdr_tr = StructElem::new(StructType::TR);
        let mut th = StructElem::new(StructType::TH);
        th.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        hdr_tr.add_child(StructChild::StructElem(Box::new(th)));
        thead.add_child(StructChild::StructElem(Box::new(hdr_tr)));
        table_elem.add_child(StructChild::StructElem(Box::new(thead)));

        // TBody > TR > TD
        let mut tbody = StructElem::new(StructType::TBody);
        let mut body_tr = StructElem::new(StructType::TR);
        let mut td = StructElem::new(StructType::TD);
        td.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });
        body_tr.add_child(StructChild::StructElem(Box::new(td)));
        tbody.add_child(StructChild::StructElem(Box::new(body_tr)));
        table_elem.add_child(StructChild::StructElem(Box::new(tbody)));

        let spans = vec![
            make_text_span("Header", Some(1)),
            make_text_span("Data", Some(2)),
        ];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert!(result.has_header);
        assert_eq!(result.rows.len(), 2);
        assert!(result.rows[0].is_header);
        assert!(!result.rows[1].is_header);
        assert_eq!(result.rows[0].cells[0].text, "Header");
        assert_eq!(result.rows[1].cells[0].text, "Data");
    }

    #[test]
    fn test_extract_table_from_spans_empty_table() {
        let table_elem = StructElem::new(StructType::Table);
        let spans: Vec<crate::layout::TextSpan> = vec![];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert!(result.is_empty());
    }

    /// Regression test for issue-336-example: adjacent MCID spans (gap ≤ 0) must NOT
    /// have a space inserted between them.  The PDF stores e.g. "Q" (MCID 1) and "（"
    /// (MCID 2) as separate marked-content runs that abut each other on the same line.
    /// Before the fix, extract_cell always inserted a space between any two MCID blocks,
    /// producing "Q （peu/d）" instead of the correct "Q（peu/d）".
    #[test]
    fn test_extract_cell_adjacent_mcid_spans_no_space() {
        use crate::layout::text_block::{Color, FontWeight};

        // Build TD > [MCID 1, MCID 2, MCID 3]  (three adjacent spans on the same line)
        let mut td = StructElem::new(StructType::TD);
        td.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        td.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });
        td.add_child(StructChild::MarkedContentRef { mcid: 3, page: 0 });
        let mut tr = StructElem::new(StructType::TR);
        tr.add_child(StructChild::StructElem(Box::new(td)));
        let mut table_elem = StructElem::new(StructType::Table);
        table_elem.add_child(StructChild::StructElem(Box::new(tr)));

        // Exact coordinates from issue-336 page 0 (Q（peu/d） column header):
        //   "Q"     x=345.79 w=8.22  end=354.01
        //   "（"    x=353.83 w=10.56 end=364.39   gap=-0.18 (overlap → no space)
        //   "peu/d" x=364.39 w=25.24             gap=0.00  (touching → no space)
        let base = crate::layout::TextSpan {
            artifact_type: None,
            text: String::new(),
            bbox: Rect::new(0.0, 678.0, 0.0, 10.56),
            font_name: "Test".to_string(),
            font_size: 10.56,
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
            primary_detected: false,
            char_widths: vec![],
        };
        let spans = vec![
            crate::layout::TextSpan {
                text: "Q".into(),
                bbox: Rect::new(345.79, 678.0, 8.22, 10.56),
                mcid: Some(1),
                ..base.clone()
            },
            crate::layout::TextSpan {
                text: "（".into(),
                bbox: Rect::new(353.83, 678.0, 10.56, 10.56),
                mcid: Some(2),
                ..base.clone()
            },
            crate::layout::TextSpan {
                text: "peu/d".into(),
                bbox: Rect::new(364.39, 678.0, 25.24, 10.56),
                mcid: Some(3),
                ..base.clone()
            },
        ];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert_eq!(
            result.rows[0].cells[0].text, "Q（peu/d",
            "adjacent MCID spans must not get a space inserted between them"
        );
    }

    /// Companion test: MCID spans on different lines (multi-line cell) DO get a space.
    #[test]
    fn test_extract_cell_multiline_mcid_spans_have_space() {
        let mut td = StructElem::new(StructType::TD);
        td.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        td.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });
        let mut tr = StructElem::new(StructType::TR);
        tr.add_child(StructChild::StructElem(Box::new(td)));
        let mut table_elem = StructElem::new(StructType::Table);
        table_elem.add_child(StructChild::StructElem(Box::new(tr)));

        let base = crate::layout::TextSpan {
            artifact_type: None,
            text: String::new(),
            bbox: Rect::new(0.0, 0.0, 0.0, 12.0),
            font_name: "Test".to_string(),
            font_size: 12.0,
            font_weight: crate::layout::text_block::FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: crate::layout::text_block::Color::black(),
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
            primary_detected: false,
            char_widths: vec![],
        };
        // Line 1: "Hello" ends at x=100, y=200.  Line 2: "World" starts at x=10, y=188.
        // y_diff = 12 > line_h * 0.5 = 6 → different lines → space inserted.
        let spans = vec![
            crate::layout::TextSpan {
                text: "Hello".into(),
                bbox: Rect::new(10.0, 200.0, 90.0, 12.0),
                mcid: Some(1),
                ..base.clone()
            },
            crate::layout::TextSpan {
                text: "World".into(),
                bbox: Rect::new(10.0, 188.0, 90.0, 12.0),
                mcid: Some(2),
                ..base.clone()
            },
        ];

        let result = extract_table_from_spans(&table_elem, &spans).unwrap();
        assert_eq!(result.rows[0].cells[0].text, "Hello World");
    }
}
