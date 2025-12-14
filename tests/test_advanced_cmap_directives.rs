//! Advanced CMap Directives Tests for Phase 4.1
//!
//! Tests for supporting advanced CMap features:
//! - beginnotdefrange sections (undefined character handling)
//! - Advanced escape sequences (space, tab, newline, hex)
//! - Flexible whitespace in CMap syntax
//!
//! Spec: PDF 32000-1:2008 Section 9.10.3 (ToUnicode CMaps)

#[test]
fn test_cmap_beginnotdefrange_section() {
    //! Test 1: beginnotdefrange sections should map undefined characters
    //!
    //! When a PDF contains undefined character ranges in a ToUnicode CMap,
    //! they should be parsed and used as fallback mappings for characters
    //! not covered by bfchar/bfrange sections.
    //!
    //! Example: beginnotdefrange
    //!   <0000> <00FF> <FFFD>
    //! endbfrange
    //! Maps all unmapped codes 0x0000-0x00FF to U+FFFD (replacement char)

    let cmap_with_notdefrange = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (Identity)
/Supplement 0
>> def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfchar
<0041> <0041>
endbfchar
1 beginnotdefrange
<0000> <0040> <FFFD>
endnotdefrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    // Parse the CMap (this would be embedded in a PDF font)
    let result = pdf_oxide::fonts::cmap::parse_tounicode_cmap(cmap_with_notdefrange.as_bytes());

    assert!(result.is_ok(), "Should parse CMap with beginnotdefrange");

    let cmap = result.unwrap();

    // Should have mapping for 0x0041
    assert_eq!(cmap.get(&0x0041), Some(&"A".to_string()), "Mapped character should be 'A'");

    // Should have notdef mapping for undefined range
    assert_eq!(cmap.get(&0x0000), Some(&"\u{FFFD}".to_string()), "Notdef should map to U+FFFD");
    assert_eq!(
        cmap.get(&0x0020),
        Some(&"\u{FFFD}".to_string()),
        "Notdef range should map to U+FFFD"
    );
}

#[test]
fn test_cmap_escape_sequences_space_tab() {
    //! Test 2: Escape sequences for whitespace (space, tab, newline)
    //!
    //! CMaps should support symbolic escape sequences:
    //! - <space> for space character (U+0020)
    //! - <tab> for tab (U+0009)
    //! - <newline> for newline (U+000A)
    //! - <carriage return> for CR (U+000D)

    let cmap_with_escapes = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (Identity)
/Supplement 0
>> def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
4 beginbfchar
<0001> <space>
<0002> <tab>
<0003> <newline>
<0020> <0020>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let result = pdf_oxide::fonts::cmap::parse_tounicode_cmap(cmap_with_escapes.as_bytes());

    assert!(result.is_ok(), "Should parse CMap with escape sequences");

    let cmap = result.unwrap();

    // Space escape should map to actual space character
    assert_eq!(cmap.get(&0x0001), Some(&" ".to_string()), "space escape should map to space");

    // Tab escape should map to actual tab
    assert_eq!(cmap.get(&0x0002), Some(&"\t".to_string()), "tab escape should map to tab");

    // Newline escape should map to actual newline
    assert_eq!(
        cmap.get(&0x0003),
        Some(&"\n".to_string()),
        "newline escape should map to newline"
    );

    // Regular hex should still work
    assert_eq!(cmap.get(&0x0020), Some(&" ".to_string()), "hex 0020 should map to space");
}

#[test]
fn test_cmap_flexible_whitespace() {
    //! Test 3: Flexible whitespace in CMap syntax
    //!
    //! CMap parsers should be lenient with whitespace:
    //! - Extra spaces between tokens
    //! - Tabs and newlines
    //! - Variable spacing in angle brackets

    let cmap_flexible_whitespace = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (Identity)
/Supplement 0
>> def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000>   <FFFF>
endcodespacerange
3 beginbfchar
<0041>    <0041>
< 0042 >  < 0042 >
<0043><0043>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let result = pdf_oxide::fonts::cmap::parse_tounicode_cmap(cmap_flexible_whitespace.as_bytes());

    assert!(result.is_ok(), "Should parse CMap with flexible whitespace");

    let cmap = result.unwrap();

    // All three should parse correctly despite whitespace variations
    assert_eq!(cmap.get(&0x0041), Some(&"A".to_string()), "Normal spacing should work");
    assert_eq!(cmap.get(&0x0042), Some(&"B".to_string()), "Extra spaces should work");
    assert_eq!(cmap.get(&0x0043), Some(&"C".to_string()), "No spaces should work");
}

#[test]
fn test_cmap_hex_escape_sequences() {
    //! Test 4: Hex escape sequences for special characters
    //!
    //! CMap should support escape sequences for:
    //! - \\n (newline, 0x0A)
    //! - \\t (tab, 0x09)
    //! - \\r (carriage return, 0x0D)
    //! - \\( and \\) (literal parens)
    //! - \\\\ (literal backslash)

    let cmap_hex_escapes = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (Identity)
/Supplement 0
>> def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0001> <000A>
<0002> <0009>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let result = pdf_oxide::fonts::cmap::parse_tounicode_cmap(cmap_hex_escapes.as_bytes());

    assert!(result.is_ok(), "Should parse CMap with hex escape sequences");

    let cmap = result.unwrap();

    // Hex 000A should map to newline
    assert_eq!(cmap.get(&0x0001), Some(&"\n".to_string()), "Hex 000A should map to newline");

    // Hex 0009 should map to tab
    assert_eq!(cmap.get(&0x0002), Some(&"\t".to_string()), "Hex 0009 should map to tab");
}

#[test]
fn test_cmap_edge_case_empty_notdefrange() {
    //! Test 5: Edge case - empty or sparse notdefrange sections
    //!
    //! Should handle:
    //! - Empty notdefrange sections gracefully
    //! - Multiple notdefrange sections
    //! - Notdefrange after bfrange

    let cmap_sparse_notdef = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (Identity)
/Supplement 0
>> def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfchar
<0041> <0041>
endbfchar
2 beginbfrange
<0050> <0059> <0050>
<0061> <007A> <0061>
endbfrange
1 beginnotdefrange
<0000> <0040> <FFFD>
endnotdefrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let result = pdf_oxide::fonts::cmap::parse_tounicode_cmap(cmap_sparse_notdef.as_bytes());

    assert!(result.is_ok(), "Should parse CMap with complex sections");

    let cmap = result.unwrap();

    // Explicit bfchar should work
    assert_eq!(cmap.get(&0x0041), Some(&"A".to_string()), "bfchar mapping should work");

    // bfrange should work
    assert_eq!(cmap.get(&0x0050), Some(&"P".to_string()), "bfrange sequential should work");
    assert_eq!(cmap.get(&0x0061), Some(&"a".to_string()), "bfrange sequential should work");

    // notdefrange should provide fallback
    assert_eq!(
        cmap.get(&0x0001),
        Some(&"\u{FFFD}".to_string()),
        "notdefrange should provide fallback"
    );
}
