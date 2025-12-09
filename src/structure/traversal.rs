//! Structure tree traversal for extracting reading order.
//!
//! Implements pre-order traversal of structure trees to determine correct reading order.

use super::types::{StructChild, StructElem, StructTreeRoot, StructType};
use crate::error::Error;

/// Represents an ordered content item extracted from structure tree.
#[derive(Debug, Clone)]
pub struct OrderedContent {
    /// Page number
    pub page: u32,

    /// Marked Content ID
    pub mcid: u32,

    /// Structure type (for semantic information)
    pub struct_type: String,

    /// Pre-parsed structure type for efficient access
    pub parsed_type: StructType,

    /// Is this a heading?
    pub is_heading: bool,

    /// Is this a block-level element?
    pub is_block: bool,
}

/// Traverse the structure tree and extract ordered content for a specific page.
///
/// This performs a pre-order traversal of the structure tree, extracting
/// marked content references in document order.
///
/// # Arguments
/// * `struct_tree` - The structure tree root
/// * `page_num` - The page number to extract content for
///
/// # Returns
/// * Vector of ordered content items for the specified page
pub fn traverse_structure_tree(
    struct_tree: &StructTreeRoot,
    page_num: u32,
) -> Result<Vec<OrderedContent>, Error> {
    let mut result = Vec::new();

    // Traverse each root element
    for root_elem in &struct_tree.root_elements {
        traverse_element(root_elem, page_num, &mut result)?;
    }

    Ok(result)
}

/// Recursively traverse a structure element.
///
/// Performs pre-order traversal:
/// 1. Process current element's marked content (if on target page)
/// 2. Recursively process children in order
fn traverse_element(
    elem: &StructElem,
    target_page: u32,
    result: &mut Vec<OrderedContent>,
) -> Result<(), Error> {
    let struct_type_str = format!("{:?}", elem.struct_type);
    let parsed_type = elem.struct_type.clone();
    let is_heading = elem.struct_type.is_heading();
    let is_block = elem.struct_type.is_block();

    // Process children in order
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { mcid, page } => {
                // If this marked content is on the target page, add it
                if *page == target_page {
                    result.push(OrderedContent {
                        page: *page,
                        mcid: *mcid,
                        struct_type: struct_type_str.clone(),
                        parsed_type: parsed_type.clone(),
                        is_heading,
                        is_block,
                    });
                }
            },

            StructChild::StructElem(child_elem) => {
                // Recursively traverse child element
                traverse_element(child_elem, target_page, result)?;
            },

            StructChild::ObjectRef(_obj_num, _gen) => {
                // TODO: Resolve object reference and traverse
                // For MVP, skip object references
            },
        }
    }

    Ok(())
}

/// Extract all marked content IDs in reading order for a page.
///
/// This is a simpler interface that just returns the MCIDs in order,
/// which can be used to reorder extracted text blocks.
///
/// # Arguments
/// * `struct_tree` - The structure tree root
/// * `page_num` - The page number
///
/// # Returns
/// * Vector of MCIDs in reading order
pub fn extract_reading_order(
    struct_tree: &StructTreeRoot,
    page_num: u32,
) -> Result<Vec<u32>, Error> {
    let ordered_content = traverse_structure_tree(struct_tree, page_num)?;
    Ok(ordered_content.into_iter().map(|c| c.mcid).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::types::{StructChild, StructElem, StructType};

    #[test]
    fn test_simple_traversal() {
        // Create a simple structure tree:
        // Document
        //   ├─ P (MCID=0, page=0)
        //   └─ P (MCID=1, page=0)
        let mut root = StructElem::new(StructType::Document);

        let mut p1 = StructElem::new(StructType::P);
        p1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut p2 = StructElem::new(StructType::P);
        p2.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });

        root.add_child(StructChild::StructElem(Box::new(p1)));
        root.add_child(StructChild::StructElem(Box::new(p2)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // Extract reading order
        let order = extract_reading_order(&struct_tree, 0).unwrap();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_page_filtering() {
        // Create structure with content on different pages
        let mut root = StructElem::new(StructType::Document);

        let mut p1 = StructElem::new(StructType::P);
        p1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut p2 = StructElem::new(StructType::P);
        p2.add_child(StructChild::MarkedContentRef { mcid: 1, page: 1 });

        root.add_child(StructChild::StructElem(Box::new(p1)));
        root.add_child(StructChild::StructElem(Box::new(p2)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // Extract page 0 - should only get MCID 0
        let order_page_0 = extract_reading_order(&struct_tree, 0).unwrap();
        assert_eq!(order_page_0, vec![0]);

        // Extract page 1 - should only get MCID 1
        let order_page_1 = extract_reading_order(&struct_tree, 1).unwrap();
        assert_eq!(order_page_1, vec![1]);
    }

    #[test]
    fn test_nested_structure() {
        // Create nested structure:
        // Document
        //   └─ Sect
        //       ├─ H1 (MCID=0)
        //       └─ P (MCID=1)
        let mut root = StructElem::new(StructType::Document);

        let mut sect = StructElem::new(StructType::Sect);

        let mut h1 = StructElem::new(StructType::H1);
        h1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut p = StructElem::new(StructType::P);
        p.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });

        sect.add_child(StructChild::StructElem(Box::new(h1)));
        sect.add_child(StructChild::StructElem(Box::new(p)));

        root.add_child(StructChild::StructElem(Box::new(sect)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // Should traverse in order: H1 (MCID 0), then P (MCID 1)
        let order = extract_reading_order(&struct_tree, 0).unwrap();
        assert_eq!(order, vec![0, 1]);
    }
}
