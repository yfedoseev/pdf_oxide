//! PDF Logical Structure (Tagged PDF) support.
//!
//! This module implements parsing and traversal of PDF logical structure trees
//! according to ISO 32000-1:2008 Section 14.7.
//!
//! ## Overview
//!
//! Tagged PDFs contain explicit document structure that defines reading order,
//! semantic meaning, and accessibility information. This is the PDF-spec-compliant
//! way to determine reading order, as opposed to heuristic layout analysis.
//!
//! ## Structure Tree
//!
//! A structure tree consists of:
//! - **StructTreeRoot**: The root of the structure hierarchy
//! - **StructElem**: Structure elements (paragraphs, headings, sections, etc.)
//! - **ParentTree**: Maps marked content IDs to structure elements
//! - **Marked Content**: Tagged content in page streams (BMC/BDC/EMC operators)
//!
//! ## Reading Order
//!
//! Reading order is determined by pre-order traversal of the structure tree:
//! 1. Visit structure element
//! 2. Extract associated marked content
//! 3. Recursively visit children in order
//!
//! ## Example
//!
//! ```ignore
//! use pdf_oxide::structure::StructureTreeRoot;
//!
//! // Parse structure tree from document
//! if let Some(struct_tree) = document.structure_tree()? {
//!     // Traverse in document order
//!     for page_num in 0..document.page_count() {
//!         let ordered_content = struct_tree.extract_ordered_content(page_num)?;
//!         // Content is now in correct reading order!
//!     }
//! }
//! ```

mod parser;
pub mod spatial_table_detector;
pub mod table_extractor;
pub mod traversal;
pub mod types;

pub use parser::parse_structure_tree;
pub use spatial_table_detector::{
    detect_tables_from_spans, DetectedTable, SpatialTableDetector, TableDetectionConfig,
};
pub use table_extractor::{extract_table, ExtractedTable, TableCell, TableRow};
pub use traversal::{extract_reading_order, traverse_structure_tree, OrderedContent};
pub use types::{ParentTree, StructChild, StructElem, StructTreeRoot, StructType};
