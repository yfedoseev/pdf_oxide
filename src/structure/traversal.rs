//! Structure tree traversal for extracting reading order.
//!
//! Implements pre-order traversal of structure trees to determine correct reading order.

use super::types::{StructChild, StructElem, StructTreeRoot, StructType};
use crate::error::Error;

/// Role this content plays inside a List (PDF spec §14.8.4.3).
///
/// MCRs nested under list-context ancestors carry their role so the
/// markdown converter can emit `- item` / `1. item` correctly even when
/// the immediate parent of the MCR is a Span or P (the common Word /
/// Acrobat output shape `LI → LBody → Span → MCR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListRole {
    /// Inside an LI (list item) but not under Lbl/LBody yet (or LI
    /// itself holds the MCR directly).
    LI,
    /// Inside the Lbl (label) sub-element of an LI — the bullet/number.
    Lbl,
    /// Inside the LBody (body) sub-element of an LI — the item text.
    LBody,
}

/// Represents an ordered content item extracted from structure tree.
#[derive(Debug, Clone)]
pub struct OrderedContent {
    /// Page number
    pub page: u32,

    /// Marked Content ID (None for word break markers)
    pub mcid: Option<u32>,

    /// Structure type (for semantic information)
    pub struct_type: String,

    /// Pre-parsed structure type for efficient access
    pub parsed_type: StructType,

    /// Is this a heading?
    ///
    /// True when the MCR is nested under any heading ancestor (H, H1..H6),
    /// not just when the immediate parent is a heading. Word-generated
    /// tagged PDFs commonly wrap heading text in `H1 → Span → MCR`, where
    /// the heading semantic must still be recovered.
    pub is_heading: bool,

    /// If the MCR is nested under any heading ancestor, the level of that
    /// ancestor (H1 → 1, …, H6 → 6, generic H → 1). None otherwise.
    pub heading_level: Option<u8>,

    /// Role inside a list, when nested under any L/LI ancestor. None if
    /// this MCR has no list ancestor.
    pub list_role: Option<ListRole>,

    /// Is this a block-level element?
    pub is_block: bool,

    /// Is this a word break marker (WB element)?
    ///
    /// When true, a space should be inserted at this position during
    /// text assembly. This supports CJK text that uses WB elements
    /// to mark word boundaries.
    pub is_word_break: bool,

    /// Identifier of the nearest block-level ancestor (P, H*, LI, Sect,
    /// Div, Art, …) — increments each time the traversal enters a new
    /// block element. Two MCRs that share a `block_id` belong to the
    /// same logical paragraph; a change in `block_id` between adjacent
    /// MCRs is the structure-tree-authoritative paragraph boundary
    /// (PDF spec ISO 32000-1:2008 §14.8.4). The markdown / HTML
    /// converters rely on this to split paragraphs when a tagged PDF's
    /// inter-paragraph gap is too small for the geometric heuristic.
    /// 0 means "no enclosing block element seen" (root-level Span).
    pub block_id: u32,

    /// Actual text replacement from /ActualText (optional)
    /// Per PDF spec Section 14.9.4, when present this replaces all
    /// descendant content with the specified text.
    pub actual_text: Option<String>,
}

/// Inheritable context propagated down the structure tree during traversal.
///
/// Tracks the nearest heading and list ancestors so deeply nested MCRs
/// (`H1 → Span → MCR`, `LI → LBody → Span → MCR`) carry the correct
/// semantic role on the resulting `OrderedContent`. Without this, the
/// markdown converter saw the immediate parent (Span / P) and lost the
/// heading / list-item information altogether.
#[derive(Debug, Clone, Copy, Default)]
struct InheritedContext {
    heading_level: Option<u8>,
    list_role: Option<ListRole>,
    /// Identifier of the nearest block-level ancestor — see
    /// `OrderedContent::block_id`.
    block_id: u32,
}

impl InheritedContext {
    /// Returns true when `t` is a block-level element that should bump
    /// the paragraph counter on entry. Spans, links, and similar inline
    /// elements do not.
    fn is_paragraph_block(t: &StructType) -> bool {
        matches!(
            t,
            StructType::P
                | StructType::H
                | StructType::H1
                | StructType::H2
                | StructType::H3
                | StructType::H4
                | StructType::H5
                | StructType::H6
                | StructType::LI
                | StructType::Lbl
                | StructType::LBody
                | StructType::Sect
                | StructType::Div
                | StructType::Art
                | StructType::Note
                | StructType::Reference
                | StructType::BibEntry
                | StructType::Code
                | StructType::TR
                | StructType::TH
                | StructType::TD
        )
    }

    fn descend(self, child: &StructType, counter: &mut u32) -> Self {
        let heading_level = match child {
            StructType::H1 => Some(1),
            StructType::H2 => Some(2),
            StructType::H3 => Some(3),
            StructType::H4 => Some(4),
            StructType::H5 => Some(5),
            StructType::H6 => Some(6),
            // Generic /H carries no level on its own.
            StructType::H => Some(self.heading_level.unwrap_or(1)),
            _ => self.heading_level,
        };
        let list_role = match child {
            StructType::Lbl => Some(ListRole::Lbl),
            StructType::LBody => Some(ListRole::LBody),
            StructType::LI => Some(self.list_role.unwrap_or(ListRole::LI)),
            // L starts list context but doesn't itself hold MCRs as items;
            // its LI children promote to ListRole::LI on descent.
            StructType::L => self.list_role,
            _ => self.list_role,
        };
        let block_id = if Self::is_paragraph_block(child) {
            *counter += 1;
            *counter
        } else {
            self.block_id
        };
        Self {
            heading_level,
            list_role,
            block_id,
        }
    }
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
    let mut block_counter = 0u32;

    // Traverse each root element
    for root_elem in &struct_tree.root_elements {
        traverse_element(
            root_elem,
            page_num,
            InheritedContext::default(),
            &mut block_counter,
            &mut result,
        )?;
    }

    Ok(result)
}

/// Traverse the structure tree once and build content for ALL pages.
///
/// This is much more efficient than calling `traverse_structure_tree` once per page,
/// which would walk the entire tree N times. Instead, we walk the tree once and
/// collect content items into per-page buckets.
///
/// Returns a HashMap mapping page numbers to their ordered content items.
pub fn traverse_structure_tree_all_pages(
    struct_tree: &StructTreeRoot,
) -> std::collections::HashMap<u32, Vec<OrderedContent>> {
    let mut result: std::collections::HashMap<u32, Vec<OrderedContent>> =
        std::collections::HashMap::new();

    let mut block_counter = 0u32;
    for root_elem in &struct_tree.root_elements {
        traverse_element_all_pages(
            root_elem,
            InheritedContext::default(),
            &mut block_counter,
            &mut result,
        );
    }

    result
}

/// Recursively traverse a structure element, collecting content for all pages.
///
/// `ctx` carries inherited semantics from heading and list ancestors so deeply
/// nested MCRs (e.g. `H1 → Span → MCR`, `LI → LBody → Span → MCR`) emit
/// content tagged with the right role, not just the immediate parent's role.
fn traverse_element_all_pages(
    elem: &StructElem,
    ctx: InheritedContext,
    block_counter: &mut u32,
    result: &mut std::collections::HashMap<u32, Vec<OrderedContent>>,
) {
    let struct_type_str = format!("{:?}", elem.struct_type);
    let parsed_type = elem.struct_type.clone();
    let descended = ctx.descend(&parsed_type, block_counter);
    let is_heading_inherited = descended.heading_level.is_some();
    let is_block = elem.struct_type.is_block();
    let is_word_break = elem.struct_type.is_word_break();

    // If /ActualText is present, it replaces all descendant content (PDF spec 14.9.4)
    if let Some(ref actual_text) = elem.actual_text {
        // Collect all pages this element has content on
        let pages = collect_pages(elem);
        for page in pages {
            result.entry(page).or_default().push(OrderedContent {
                page,
                mcid: None,
                struct_type: struct_type_str.clone(),
                parsed_type: parsed_type.clone(),
                is_heading: is_heading_inherited,
                heading_level: descended.heading_level,
                list_role: descended.list_role,
                is_block,
                is_word_break: false,
                block_id: descended.block_id,
                actual_text: Some(actual_text.clone()),
            });
        }
        return;
    }

    // Process children in order
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { mcid, page } => {
                result.entry(*page).or_default().push(OrderedContent {
                    page: *page,
                    mcid: Some(*mcid),
                    struct_type: struct_type_str.clone(),
                    parsed_type: parsed_type.clone(),
                    is_heading: is_heading_inherited,
                    heading_level: descended.heading_level,
                    list_role: descended.list_role,
                    is_block,
                    is_word_break: false,
                    block_id: descended.block_id,
                    actual_text: None,
                });
            },

            StructChild::StructElem(child_elem) => {
                // If parent is WB, emit word break markers before processing child
                if is_word_break {
                    let child_pages = collect_pages(child_elem);
                    for page in child_pages {
                        result.entry(page).or_default().push(OrderedContent {
                            page,
                            mcid: None,
                            struct_type: struct_type_str.clone(),
                            parsed_type: parsed_type.clone(),
                            is_heading: false,
                            heading_level: None,
                            list_role: descended.list_role,
                            is_block: false,
                            is_word_break: true,
                            block_id: descended.block_id,
                            actual_text: None,
                        });
                    }
                }
                traverse_element_all_pages(child_elem, descended, block_counter, result);
            },

            StructChild::ObjectRef(_obj_num, _gen) => {
                log::debug!("Skipping unresolved ObjectRef({}, {})", _obj_num, _gen);
            },
        }
    }
}

/// Collect all page numbers that a structure element has content on.
fn collect_pages(elem: &StructElem) -> Vec<u32> {
    let mut pages = Vec::new();
    collect_pages_recursive(elem, &mut pages);
    pages.sort_unstable();
    pages.dedup();
    pages
}

fn collect_pages_recursive(elem: &StructElem, pages: &mut Vec<u32>) {
    if let Some(page) = elem.page {
        pages.push(page);
    }
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { page, .. } => {
                pages.push(*page);
            },
            StructChild::StructElem(child_elem) => {
                collect_pages_recursive(child_elem, pages);
            },
            _ => {},
        }
    }
}

/// Recursively traverse a structure element.
///
/// Performs pre-order traversal:
/// 1. Process current element's marked content (if on target page)
/// 2. Recursively process children in order
/// 3. Handle WB (word break) elements by emitting markers
fn traverse_element(
    elem: &StructElem,
    target_page: u32,
    ctx: InheritedContext,
    block_counter: &mut u32,
    result: &mut Vec<OrderedContent>,
) -> Result<(), Error> {
    let struct_type_str = format!("{:?}", elem.struct_type);
    let parsed_type = elem.struct_type.clone();
    let descended = ctx.descend(&parsed_type, block_counter);
    let is_heading_inherited = descended.heading_level.is_some();
    let is_block = elem.struct_type.is_block();
    let is_word_break = elem.struct_type.is_word_break();

    // If /ActualText is present, it replaces all descendant content (PDF spec 14.9.4)
    if let Some(ref actual_text) = elem.actual_text {
        if has_content_on_page(elem, target_page) {
            result.push(OrderedContent {
                page: target_page,
                mcid: None,
                struct_type: struct_type_str,
                parsed_type,
                is_heading: is_heading_inherited,
                heading_level: descended.heading_level,
                list_role: descended.list_role,
                is_block,
                is_word_break: false,
                block_id: descended.block_id,
                actual_text: Some(actual_text.clone()),
            });
            return Ok(());
        }
    }

    // If this is a WB (word break) element, emit a word break marker
    if is_word_break {
        result.push(OrderedContent {
            page: target_page,
            mcid: None,
            struct_type: struct_type_str.clone(),
            parsed_type: parsed_type.clone(),
            is_heading: false,
            heading_level: None,
            list_role: descended.list_role,
            is_block: false,
            is_word_break: true,
            block_id: descended.block_id,
            actual_text: None,
        });
        // WB elements typically have no children, but process any just in case
    }

    // Process children in order
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { mcid, page } => {
                // If this marked content is on the target page, add it
                if *page == target_page {
                    result.push(OrderedContent {
                        page: *page,
                        mcid: Some(*mcid),
                        struct_type: struct_type_str.clone(),
                        parsed_type: parsed_type.clone(),
                        is_heading: is_heading_inherited,
                        heading_level: descended.heading_level,
                        list_role: descended.list_role,
                        is_block,
                        is_word_break: false,
                        block_id: descended.block_id,
                        actual_text: None,
                    });
                }
            },

            StructChild::StructElem(child_elem) => {
                // Recursively traverse child element
                traverse_element(child_elem, target_page, descended, block_counter, result)?;
            },

            StructChild::ObjectRef(_obj_num, _gen) => {
                // ObjectRef should be resolved at parse time (structure/parser.rs).
                // If we encounter one here, it means the reference couldn't be resolved.
                log::debug!("Skipping unresolved ObjectRef({}, {})", _obj_num, _gen);
            },
        }
    }

    Ok(())
}

/// Check if a structure element has any content on the target page.
fn has_content_on_page(elem: &StructElem, target_page: u32) -> bool {
    if elem.page == Some(target_page) {
        return true;
    }
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { page, .. } => {
                if *page == target_page {
                    return true;
                }
            },
            StructChild::StructElem(child_elem) => {
                if has_content_on_page(child_elem, target_page) {
                    return true;
                }
            },
            _ => {},
        }
    }
    false
}

/// Extract all marked content IDs in reading order for a page.
///
/// This is a simpler interface that just returns the MCIDs in order,
/// which can be used to reorder extracted text blocks.
///
/// Note: Word break (WB) markers are filtered out since they don't have MCIDs.
/// Use `traverse_structure_tree` directly if you need word break information.
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
    Ok(ordered_content
        .into_iter()
        .filter_map(|c| c.mcid) // Filter out word break markers (mcid=None)
        .collect())
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

    #[test]
    fn test_word_break_elements() {
        // Create structure with WB (word break) elements for CJK text:
        // P
        //   ├─ Span (MCID=0) - "你好"
        //   ├─ WB             - word boundary marker
        //   └─ Span (MCID=1) - "世界"
        let mut root = StructElem::new(StructType::P);

        let mut span1 = StructElem::new(StructType::Span);
        span1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let wb = StructElem::new(StructType::WB);

        let mut span2 = StructElem::new(StructType::Span);
        span2.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });

        root.add_child(StructChild::StructElem(Box::new(span1)));
        root.add_child(StructChild::StructElem(Box::new(wb)));
        root.add_child(StructChild::StructElem(Box::new(span2)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // traverse_structure_tree should include the word break marker
        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        assert_eq!(ordered.len(), 3); // MCID 0, WB, MCID 1
        assert_eq!(ordered[0].mcid, Some(0));
        assert!(!ordered[0].is_word_break);
        assert_eq!(ordered[1].mcid, None); // WB has no MCID
        assert!(ordered[1].is_word_break);
        assert_eq!(ordered[2].mcid, Some(1));
        assert!(!ordered[2].is_word_break);

        // extract_reading_order should filter out WB markers
        let mcids = extract_reading_order(&struct_tree, 0).unwrap();
        assert_eq!(mcids, vec![0, 1]); // Only MCIDs, no WB
    }

    #[test]
    fn test_empty_tree() {
        let struct_tree = StructTreeRoot::new();
        let order = extract_reading_order(&struct_tree, 0).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_empty_page() {
        let mut root = StructElem::new(StructType::Document);
        let mut p = StructElem::new(StructType::P);
        p.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        root.add_child(StructChild::StructElem(Box::new(p)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // Page 5 has no content
        let order = extract_reading_order(&struct_tree, 5).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_nested_heading_propagates_is_heading_to_inner_mcr() {
        // Word365 / docling pattern: H1 wraps Span which holds the actual MCR.
        // The MCR must inherit is_heading from its H1 ancestor, not from
        // the immediate Span parent (Span.is_heading() == false).
        // Reproduces issue #377 word365_structure regression.
        let mut h1 = StructElem::new(StructType::H1);
        let mut span = StructElem::new(StructType::Span);
        span.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        h1.add_child(StructChild::StructElem(Box::new(span)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(h1);

        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        let heading_mcrs: Vec<_> = ordered.iter().filter(|c| c.is_heading).collect();
        assert_eq!(
            heading_mcrs.len(),
            1,
            "H1 → Span → MCR must propagate is_heading=true to the inner MCR"
        );
        assert_eq!(heading_mcrs[0].mcid, Some(0));
        // Same expectation from the all-pages traversal used by markdown.
        let by_page = traverse_structure_tree_all_pages(&struct_tree);
        let heading_mcrs_all: Vec<_> = by_page
            .get(&0)
            .unwrap()
            .iter()
            .filter(|c| c.is_heading)
            .collect();
        assert_eq!(heading_mcrs_all.len(), 1);
    }

    #[test]
    fn test_nested_li_lbody_keeps_list_context() {
        // word365 / pdfa pattern: LI → LBody → MCR. LBody is the list-item
        // body and must be tagged as such; LI ancestry must be discoverable
        // when emitting markdown bullets.
        let mut li = StructElem::new(StructType::LI);
        let mut lbody = StructElem::new(StructType::LBody);
        lbody.add_child(StructChild::MarkedContentRef { mcid: 7, page: 0 });
        li.add_child(StructChild::StructElem(Box::new(lbody)));
        let mut l = StructElem::new(StructType::L);
        l.add_child(StructChild::StructElem(Box::new(li)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(l);

        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        let li_mcrs: Vec<_> = ordered
            .iter()
            .filter(|c| matches!(c.list_role, Some(crate::structure::ListRole::LBody)))
            .collect();
        assert_eq!(
            li_mcrs.len(),
            1,
            "LI → LBody → MCR must carry list_role=LBody on the inner MCR"
        );
    }

    /// D8b coverage — every standard heading level (H1..H6) propagates
    /// to a deeply nested MCR. Parametrised over all six levels in the
    /// same test to keep the lock-in compact.
    #[test]
    fn test_nested_heading_propagates_for_h1_through_h6() {
        let levels = [
            (StructType::H1, 1u8),
            (StructType::H2, 2),
            (StructType::H3, 3),
            (StructType::H4, 4),
            (StructType::H5, 5),
            (StructType::H6, 6),
        ];
        for (h_type, expected_level) in levels {
            // H? → Sect → Span → MCR (3-level nesting, reflects the
            // worst-case shape seen in word365_structure-class fixtures).
            let mut head = StructElem::new(h_type.clone());
            let mut sect = StructElem::new(StructType::Sect);
            let mut span = StructElem::new(StructType::Span);
            span.add_child(StructChild::MarkedContentRef { mcid: 42, page: 0 });
            sect.add_child(StructChild::StructElem(Box::new(span)));
            head.add_child(StructChild::StructElem(Box::new(sect)));
            let mut tree = StructTreeRoot::new();
            tree.add_root_element(head);

            let ordered = traverse_structure_tree(&tree, 0).unwrap();
            let item = ordered.iter().find(|c| c.mcid == Some(42)).unwrap();
            assert!(
                item.is_heading,
                "H{} → Sect → Span → MCR must carry is_heading=true",
                expected_level
            );
            assert_eq!(
                item.heading_level,
                Some(expected_level),
                "H{} ancestor must propagate heading_level={}",
                expected_level,
                expected_level
            );
        }
    }

    /// D8b coverage — generic /H without an explicit level reports
    /// heading_level=Some(1) (the only sensible default per spec
    /// §14.8.4.2 when no surrounding heading exists).
    #[test]
    fn test_generic_h_without_level_defaults_to_h1() {
        let mut h = StructElem::new(StructType::H);
        let mut span = StructElem::new(StructType::Span);
        span.add_child(StructChild::MarkedContentRef { mcid: 9, page: 0 });
        h.add_child(StructChild::StructElem(Box::new(span)));
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(h);
        let ordered = traverse_structure_tree(&tree, 0).unwrap();
        let item = ordered.iter().find(|c| c.mcid == Some(9)).unwrap();
        assert!(item.is_heading);
        assert_eq!(item.heading_level, Some(1));
    }

    /// D8b negative case — adjacent heading and body MCRs at the same
    /// nesting level must keep their respective roles. A bug that
    /// "leaked" heading flag from a prior sibling into the next would
    /// flip every body paragraph after a heading into a heading.
    #[test]
    fn test_heading_role_does_not_bleed_into_following_paragraph() {
        let mut doc = StructElem::new(StructType::Document);
        let mut h1 = StructElem::new(StructType::H1);
        h1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        let mut p = StructElem::new(StructType::P);
        p.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        doc.add_child(StructChild::StructElem(Box::new(h1)));
        doc.add_child(StructChild::StructElem(Box::new(p)));
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(doc);

        let ordered = traverse_structure_tree(&tree, 0).unwrap();
        let h_item = ordered.iter().find(|c| c.mcid == Some(0)).unwrap();
        let p_item = ordered.iter().find(|c| c.mcid == Some(1)).unwrap();
        assert!(h_item.is_heading);
        assert!(!p_item.is_heading, "sibling P must not inherit H1's flag");
        assert_eq!(p_item.heading_level, None);
    }

    /// D8b coverage — list role variants on direct MCRs (LI carrying
    /// its own MCR without LBody/Lbl wrappers) and LBody siblings
    /// inside one LI.
    #[test]
    fn test_list_role_variants() {
        // Tree:
        // L
        //   ├─ LI (mcid=0, direct)         → role = LI
        //   └─ LI
        //        ├─ Lbl  (mcid=1)          → role = Lbl
        //        └─ LBody (mcid=2)         → role = LBody
        let mut l = StructElem::new(StructType::L);
        let mut li_a = StructElem::new(StructType::LI);
        li_a.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        let mut li_b = StructElem::new(StructType::LI);
        let mut lbl = StructElem::new(StructType::Lbl);
        lbl.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        let mut lbody = StructElem::new(StructType::LBody);
        lbody.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });
        li_b.add_child(StructChild::StructElem(Box::new(lbl)));
        li_b.add_child(StructChild::StructElem(Box::new(lbody)));
        l.add_child(StructChild::StructElem(Box::new(li_a)));
        l.add_child(StructChild::StructElem(Box::new(li_b)));
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(l);

        let ordered = traverse_structure_tree(&tree, 0).unwrap();
        let m0 = ordered.iter().find(|c| c.mcid == Some(0)).unwrap();
        let m1 = ordered.iter().find(|c| c.mcid == Some(1)).unwrap();
        let m2 = ordered.iter().find(|c| c.mcid == Some(2)).unwrap();
        assert!(matches!(m0.list_role, Some(ListRole::LI)));
        assert!(matches!(m1.list_role, Some(ListRole::Lbl)));
        assert!(matches!(m2.list_role, Some(ListRole::LBody)));
        // None of the list MCRs are headings.
        assert!(!m0.is_heading && !m1.is_heading && !m2.is_heading);
    }

    /// D5 coverage at the traversal layer — block_id must increment
    /// across sibling block elements but stay constant inside one
    /// block, even when the block contains multiple Span children.
    #[test]
    fn test_block_id_groups_within_block_and_changes_across() {
        let mut doc = StructElem::new(StructType::Document);
        let mut p1 = StructElem::new(StructType::P);
        let mut span_a = StructElem::new(StructType::Span);
        span_a.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        let mut span_b = StructElem::new(StructType::Span);
        span_b.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });
        p1.add_child(StructChild::StructElem(Box::new(span_a)));
        p1.add_child(StructChild::StructElem(Box::new(span_b)));
        let mut p2 = StructElem::new(StructType::P);
        p2.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });
        doc.add_child(StructChild::StructElem(Box::new(p1)));
        doc.add_child(StructChild::StructElem(Box::new(p2)));
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(doc);

        let ordered = traverse_structure_tree(&tree, 0).unwrap();
        let m0 = ordered.iter().find(|c| c.mcid == Some(0)).unwrap();
        let m1 = ordered.iter().find(|c| c.mcid == Some(1)).unwrap();
        let m2 = ordered.iter().find(|c| c.mcid == Some(2)).unwrap();
        assert_eq!(m0.block_id, m1.block_id, "two MCRs inside the same /P must share block_id");
        assert_ne!(
            m0.block_id, m2.block_id,
            "MCRs in different /P elements must have different block_id"
        );
        assert!(m0.block_id > 0, "block_id should be positive once any block is entered");
    }

    /// D5 coverage — Span elements at the root (no enclosing block)
    /// keep block_id=0 so the converter's "Some, Some, equal" check
    /// stays well-defined.
    #[test]
    fn test_root_span_has_block_id_zero() {
        let mut span = StructElem::new(StructType::Span);
        span.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        let mut tree = StructTreeRoot::new();
        tree.add_root_element(span);
        let ordered = traverse_structure_tree(&tree, 0).unwrap();
        assert_eq!(ordered[0].block_id, 0);
    }

    #[test]
    fn test_object_ref_skipped() {
        let mut root = StructElem::new(StructType::Document);
        root.add_child(StructChild::ObjectRef(42, 0));
        root.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let order = extract_reading_order(&struct_tree, 0).unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_traverse_all_pages() {
        let mut root = StructElem::new(StructType::Document);

        let mut p1 = StructElem::new(StructType::P);
        p1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut p2 = StructElem::new(StructType::P);
        p2.add_child(StructChild::MarkedContentRef { mcid: 1, page: 1 });

        let mut p3 = StructElem::new(StructType::P);
        p3.add_child(StructChild::MarkedContentRef { mcid: 2, page: 0 });

        root.add_child(StructChild::StructElem(Box::new(p1)));
        root.add_child(StructChild::StructElem(Box::new(p2)));
        root.add_child(StructChild::StructElem(Box::new(p3)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let all_pages = traverse_structure_tree_all_pages(&struct_tree);
        assert_eq!(all_pages.len(), 2); // pages 0 and 1
        assert_eq!(all_pages[&0].len(), 2); // MCIDs 0 and 2
        assert_eq!(all_pages[&1].len(), 1); // MCID 1
    }

    #[test]
    fn test_actual_text_replaces_descendants() {
        let mut root = StructElem::new(StructType::Document);

        let mut elem = StructElem::new(StructType::Span);
        elem.actual_text = Some("Replacement text".to_string());
        elem.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        root.add_child(StructChild::StructElem(Box::new(elem)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].actual_text, Some("Replacement text".to_string()));
        assert_eq!(ordered[0].mcid, None); // No MCID when actual_text is used
    }

    #[test]
    fn test_actual_text_wrong_page() {
        let mut root = StructElem::new(StructType::Document);

        let mut elem = StructElem::new(StructType::Span);
        elem.actual_text = Some("Replacement".to_string());
        elem.add_child(StructChild::MarkedContentRef { mcid: 0, page: 1 });

        root.add_child(StructChild::StructElem(Box::new(elem)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        // Page 0 has no content (actual_text elem is on page 1)
        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        assert!(ordered.is_empty());
    }

    #[test]
    fn test_heading_and_block_flags() {
        let mut root = StructElem::new(StructType::Document);

        let mut h1 = StructElem::new(StructType::H1);
        h1.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut span = StructElem::new(StructType::Span);
        span.add_child(StructChild::MarkedContentRef { mcid: 1, page: 0 });

        root.add_child(StructChild::StructElem(Box::new(h1)));
        root.add_child(StructChild::StructElem(Box::new(span)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let ordered = traverse_structure_tree(&struct_tree, 0).unwrap();
        assert_eq!(ordered.len(), 2);
        assert!(ordered[0].is_heading);
        assert!(ordered[0].is_block);
        assert!(!ordered[1].is_heading);
        assert!(!ordered[1].is_block);
    }

    #[test]
    fn test_collect_pages() {
        let mut elem = StructElem::new(StructType::Document);
        elem.page = Some(0);

        let mut child = StructElem::new(StructType::P);
        child.add_child(StructChild::MarkedContentRef { mcid: 0, page: 1 });
        child.add_child(StructChild::MarkedContentRef { mcid: 1, page: 2 });

        elem.add_child(StructChild::StructElem(Box::new(child)));

        let pages = collect_pages(&elem);
        assert_eq!(pages, vec![0, 1, 2]);
    }

    #[test]
    fn test_traverse_all_pages_with_actual_text() {
        let mut root = StructElem::new(StructType::Document);

        let mut elem = StructElem::new(StructType::Span);
        elem.actual_text = Some("Hello".to_string());
        elem.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        elem.add_child(StructChild::MarkedContentRef { mcid: 1, page: 1 });

        root.add_child(StructChild::StructElem(Box::new(elem)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let all_pages = traverse_structure_tree_all_pages(&struct_tree);
        // Actual text should appear on both pages
        assert!(all_pages.contains_key(&0));
        assert!(all_pages.contains_key(&1));
        assert_eq!(all_pages[&0][0].actual_text, Some("Hello".to_string()));
    }

    #[test]
    fn test_traverse_all_pages_word_break_with_children() {
        let mut root = StructElem::new(StructType::P);

        let mut wb = StructElem::new(StructType::WB);
        let mut child = StructElem::new(StructType::Span);
        child.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });
        wb.add_child(StructChild::StructElem(Box::new(child)));

        root.add_child(StructChild::StructElem(Box::new(wb)));

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let all_pages = traverse_structure_tree_all_pages(&struct_tree);
        let page0 = &all_pages[&0];
        // Should have word break marker and the child's MCID
        assert!(page0.iter().any(|c| c.is_word_break));
        assert!(page0.iter().any(|c| c.mcid == Some(0)));
    }

    #[test]
    fn test_traverse_all_pages_object_ref() {
        let mut root = StructElem::new(StructType::Document);
        root.add_child(StructChild::ObjectRef(99, 0));
        root.add_child(StructChild::MarkedContentRef { mcid: 0, page: 0 });

        let mut struct_tree = StructTreeRoot::new();
        struct_tree.add_root_element(root);

        let all_pages = traverse_structure_tree_all_pages(&struct_tree);
        assert_eq!(all_pages[&0].len(), 1);
        assert_eq!(all_pages[&0][0].mcid, Some(0));
    }

    #[test]
    fn test_has_content_on_page_deep() {
        let mut root = StructElem::new(StructType::Document);
        let mut sect = StructElem::new(StructType::Sect);
        let mut p = StructElem::new(StructType::P);
        p.add_child(StructChild::MarkedContentRef { mcid: 0, page: 3 });
        sect.add_child(StructChild::StructElem(Box::new(p)));
        root.add_child(StructChild::StructElem(Box::new(sect)));

        assert!(has_content_on_page(&root, 3));
        assert!(!has_content_on_page(&root, 0));
    }
}
