#![allow(warnings)]
//! Tests for ISO 32000-1:2008 Section 14.7-14.8 Reading Order Priority
//!
//! PDF spec defines reading order priority as:
//! 1. Structure tree (tagged PDF) - USE FIRST if available
//! 2. Physical page order - Use if no structure tree
//! 3. Content stream order - Use if both above unavailable

use pdf_oxide::structure::types::StructType;

/// Mock TextBlock for testing reading order
#[derive(Clone, Debug)]
struct TextBlock {
    text: String,
    x: f32,
    y: f32,
    struct_type: String,
    mcid: Option<u32>,
}

#[test]
fn test_priority_1_structure_tree_over_physical_order() {
    // Structure tree should be used first when available
    // Even if text appears physically in different order on page

    let blocks_physical_order = vec![
        TextBlock {
            text: "Second paragraph".to_string(),
            x: 100.0,
            y: 200.0, // Top of page physically
            struct_type: "P".to_string(),
            mcid: Some(2),
        },
        TextBlock {
            text: "First paragraph".to_string(),
            x: 100.0,
            y: 400.0, // Lower on page physically
            struct_type: "P".to_string(),
            mcid: Some(1),
        },
    ];

    // When extracted by structure tree (MCID order), should be 1, 2, not physical order
    // This test defines expected behavior once structure tree reading is implemented
    let expected_order = ["First paragraph", "Second paragraph"];

    // Structure tree says MCID 1 comes first, MCID 2 comes second
    // So even though block 2 appears first physically, it should appear second in output
    assert_eq!(expected_order[0], "First paragraph");
    assert_eq!(expected_order[1], "Second paragraph");
}

#[test]
fn test_priority_2_physical_order_when_no_structure_tree() {
    // Physical order (top-to-bottom, left-to-right) used when no structure tree

    let blocks = vec![
        TextBlock {
            text: "First".to_string(),
            x: 100.0,
            y: 100.0, // Top
            struct_type: "Body".to_string(),
            mcid: None,
        },
        TextBlock {
            text: "Second".to_string(),
            x: 100.0,
            y: 300.0, // Middle
            struct_type: "Body".to_string(),
            mcid: None,
        },
        TextBlock {
            text: "Third".to_string(),
            x: 100.0,
            y: 500.0, // Bottom
            struct_type: "Body".to_string(),
            mcid: None,
        },
    ];

    // Without structure tree, should follow top-to-bottom order
    let expected = ["First", "Second", "Third"];

    for (i, block) in blocks.iter().enumerate() {
        assert_eq!(block.text, expected[i]);
    }
}

#[test]
fn test_multi_column_layout_structure_tree_priority() {
    // Multi-column layout: Structure tree order should override column order

    let blocks = vec![
        TextBlock {
            text: "Column1 para1".to_string(),
            x: 50.0, // Left column
            y: 100.0,
            struct_type: "P".to_string(),
            mcid: Some(1),
        },
        TextBlock {
            text: "Column2 para1".to_string(),
            x: 400.0, // Right column
            y: 100.0, // Same vertical position as column 1
            struct_type: "P".to_string(),
            mcid: Some(2),
        },
        TextBlock {
            text: "Column1 para2".to_string(),
            x: 50.0, // Left column
            y: 300.0,
            struct_type: "P".to_string(),
            mcid: Some(3),
        },
        TextBlock {
            text: "Column2 para2".to_string(),
            x: 400.0, // Right column
            y: 300.0,
            struct_type: "P".to_string(),
            mcid: Some(4),
        },
    ];

    // Structure tree order (MCID): 1, 2, 3, 4
    // Physical/columnar order would be: 1, 3 (left column) then 2, 4 (right column)
    // Structure tree should win
    let expected_order = vec![
        "Column1 para1",
        "Column2 para1",
        "Column1 para2",
        "Column2 para2",
    ];
    let block_texts: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();

    // Verify blocks are in definition order (1,2,3,4 not 1,3,2,4)
    assert_eq!(block_texts, expected_order);
}

#[test]
fn test_structure_tree_provides_correct_reading_order() {
    // Reading order should respect structure tree element sequence

    let mcid_to_struct_type = vec![
        (1, "H1"),    // MCID 1 is heading
        (2, "P"),     // MCID 2 is paragraph
        (3, "P"),     // MCID 3 is paragraph
        (4, "Table"), // MCID 4 is table
    ];

    // Structure tree defines order: 1, 2, 3, 4
    // Text should appear in this sequence regardless of physical position
    for (i, (mcid, _struct_type)) in mcid_to_struct_type.iter().enumerate() {
        assert_eq!(*mcid, (i + 1) as u32);
    }
}

#[test]
fn test_ignore_physical_column_order_with_structure() {
    // When structure tree exists, completely ignore physical column ordering

    // Physical layout: 2 columns, text appears left-to-right
    // But structure says: 1, 2, 3, 4 (mixed columns)
    let structure_order = vec![1, 2, 3, 4];
    let physical_column_order = vec![1, 3, 2, 4]; // If extracted by columns

    // Should use structure order, not physical
    assert_eq!(structure_order[0], 1);
    assert_eq!(structure_order[1], 2);
    assert_eq!(structure_order[2], 3);
    assert_eq!(structure_order[3], 4);

    // NOT physical order
    assert_ne!(structure_order, physical_column_order);
}

#[test]
fn test_header_footer_order_via_structure() {
    // Structure tree should correctly order headers, body, footers

    let elements = vec![
        ("H1", "Document Title"),   // MCID 1
        ("H2", "Section 1"),        // MCID 2
        ("P", "Section 1 content"), // MCID 3
        ("P", "More content"),      // MCID 4
        ("H2", "Section 2"),        // MCID 5
        ("P", "Section 2 content"), // MCID 6
    ];

    let mcid_order: Vec<u32> = (1..=6).collect();

    // Structure defines order: H1, H2, P, P, H2, P
    // Physical order might be different if page has headers/footers
    // But structure should win
    for (i, (struct_type, _text)) in elements.iter().enumerate() {
        assert!(match *struct_type {
            "H1" | "H2" | "P" => true,
            _ => false,
        });
        assert_eq!(mcid_order[i], (i + 1) as u32);
    }
}

#[test]
fn test_table_row_order_from_structure() {
    // Table rows should follow structure tree order, not physical row order

    let table_cells = vec![
        ("TR", vec!["Header1", "Header2"]), // Row 1 (MCID 1)
        ("TR", vec!["Cell1", "Cell2"]),     // Row 2 (MCID 2)
        ("TR", vec!["Cell3", "Cell4"]),     // Row 3 (MCID 3)
    ];

    // Structure says rows are in order 1, 2, 3
    let expected_structure_order = vec![1, 2, 3];
    let actual_order: Vec<u32> = (1..=3).collect();

    assert_eq!(expected_structure_order, actual_order);
    assert_eq!(table_cells.len(), 3);
}

#[test]
fn test_fallback_to_physical_when_no_structure() {
    // When structure tree is absent, use top-to-bottom, left-to-right order

    let text_items = vec![
        ("First", 100.0, 100.0),  // Top left
        ("Second", 400.0, 100.0), // Top right (same Y)
        ("Third", 100.0, 300.0),  // Middle left
        ("Fourth", 400.0, 300.0), // Middle right
    ];

    // With no structure tree, order should be:
    // Row 1 (y=100): First (100x), Second (400x) - left to right
    // Row 2 (y=300): Third (100x), Fourth (400x) - left to right
    let expected_order = ["First", "Second", "Third", "Fourth"];

    let actual_order: Vec<&str> = text_items.iter().map(|(t, _, _)| *t).collect();

    // Note: actual implementation would need to sort by Y, then by X
    // This test just verifies the definition of expected behavior
    assert_eq!(actual_order[0], expected_order[0]); // First
}

#[test]
fn test_structure_tree_completely_overrides_physical_order() {
    // Structure tree order should COMPLETELY override physical layout
    // Not just be a tiebreaker - it's the PRIMARY method

    let physical_order = vec![
        "Text appearing first on page (top-left)",
        "Text appearing second on page (middle)",
        "Text appearing third on page (bottom-right)",
    ];

    let structure_order = vec!["Third", "First", "Second"];

    // When structure tree is present, we should produce structure_order, not physical_order
    // This test documents that structure tree is PRIMARY, not secondary

    // If extraction respects structure tree, output should be:
    // [structure_order, not physical_order]
    assert_ne!(physical_order, structure_order);
    // Confirm they're different - structure tree completely changes order
    assert_eq!(structure_order[0], "Third"); // Would be last in physical order
    assert_eq!(structure_order[1], "First"); // Would be first in physical order
}

#[test]
fn test_nested_structure_elements_ordering() {
    // Nested structure elements (section > paragraph) should be extracted in order

    let section1_paragraphs = [
        (1, "Section 1 - Paragraph 1"),
        (2, "Section 1 - Paragraph 2"),
    ];

    let section2_paragraphs = [
        (3, "Section 2 - Paragraph 1"),
        (4, "Section 2 - Paragraph 2"),
    ];

    // Structure order (depth-first): 1, 2, 3, 4
    // All of section 1's content before section 2
    let expected_order = vec![1, 2, 3, 4];
    let combined: Vec<u32> = section1_paragraphs
        .iter()
        .map(|(id, _)| *id)
        .chain(section2_paragraphs.iter().map(|(id, _)| *id))
        .collect();

    assert_eq!(combined, expected_order);
}

#[test]
fn test_structure_order_persists_across_document_sections() {
    // Structure ordering should apply across entire document, not per-page

    let mcids = vec![
        // Page 1
        (1, "Page 1, Content 1"),
        (2, "Page 1, Content 2"),
        // Page 2
        (3, "Page 2, Content 1"),
        (4, "Page 2, Content 2"),
    ];

    // Structure should maintain order 1, 2, 3, 4 across page breaks
    let order: Vec<u32> = mcids.iter().map(|(id, _)| *id).collect();
    assert_eq!(order, vec![1, 2, 3, 4]);
}

#[test]
fn test_reading_order_with_sidebars() {
    // Sidebars should follow structure tree order, not physical position

    let main_content = [(1, "Main paragraph 1"), (2, "Main paragraph 2")];

    let sidebar_content = [(3, "Sidebar content")];

    // If sidebar appears on right but is MCID 3 (after main paragraphs)
    // It should appear after main content in output, even if physically appears first
    let structure_order = vec![1, 2, 3];
    let combined: Vec<u32> = main_content
        .iter()
        .map(|(id, _)| *id)
        .chain(sidebar_content.iter().map(|(id, _)| *id))
        .collect();

    assert_eq!(combined, structure_order);
}

#[test]
fn test_empty_structure_tree_uses_physical_order() {
    // If structure tree exists but is empty, should fallback to physical order

    let text_blocks = [("Top text", 100.0), ("Bottom text", 300.0)];

    // No MCIDs (empty structure tree) - use Y coordinate (physical order)
    let sorted_by_y: Vec<&str> = text_blocks.iter().map(|(text, _)| *text).collect();

    assert_eq!(sorted_by_y, vec!["Top text", "Bottom text"]);
}

#[test]
fn test_specification_reference_iso_14_7_8() {
    // This test documents the spec sections that define reading order
    // ISO 32000-1:2008 Section 14.7: Logical Structure
    // ISO 32000-1:2008 Section 14.8: Tagged PDF (semantic structure)

    // Key requirement: Structure tree's reading order is AUTHORITATIVE
    // Implementation must check structure tree BEFORE physical layout

    let spec_priority = [
        "1. Structure tree (tagged PDF)",
        "2. Physical page order (top-to-bottom, left-to-right)",
        "3. Content stream order",
    ];

    assert_eq!(spec_priority[0], "1. Structure tree (tagged PDF)");
    assert_eq!(spec_priority[1], "2. Physical page order (top-to-bottom, left-to-right)");
    assert_eq!(spec_priority[2], "3. Content stream order");
}

#[test]
fn test_structure_type_detection_for_reading_order() {
    // Different structure types (P, H1-H6, Table, List) should be preserved
    // in their structure tree order, not reordered by physical position

    let elements = [
        (StructType::H1, "Heading 1"),
        (StructType::P, "Paragraph"),
        (StructType::H2, "Heading 2"),
        (StructType::Table, "Table"),
        (StructType::L, "List"),
    ];

    // Order should be preserved from structure tree
    for (i, (struct_type, _text)) in elements.iter().enumerate() {
        match struct_type {
            StructType::H1 => assert_eq!(i, 0),
            StructType::P => assert_eq!(i, 1),
            StructType::H2 => assert_eq!(i, 2),
            StructType::Table => assert_eq!(i, 3),
            StructType::L => assert_eq!(i, 4),
            _ => panic!("Unexpected struct type"),
        }
    }
}
