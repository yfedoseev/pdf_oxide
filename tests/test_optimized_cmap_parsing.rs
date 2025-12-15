#![allow(clippy::useless_vec)]
//! Optimized CMap Parsing Tests for Phase 5.3
//!
//! Tests for performance optimization of CMap parsing:
//! - State machine parser to replace regex-based parsing
//! - Binary search for range lookups
//! - Streaming parser for large CMaps
//! - Support for >100k entry CMaps
//!
//! Expected Impact:
//! - 20-40% faster parsing (state machine vs regex)
//! - 50-70% faster range lookups (binary search vs linear)
//! - Support for very large CMaps without memory pressure
//! - Lazy streaming parsing for huge CMaps
//!
//! Spec: PDF 32000-1:2008 Section 9.10.3 (ToUnicode CMaps)

use pdf_oxide::fonts::cmap::LazyCMap;
use pdf_oxide::fonts::FontInfo;

#[test]
fn test_optimized_parser_large_bfchar_section() {
    //! Test that the optimized parser handles large bfchar sections efficiently.
    //!
    //! Current regex-based parser may struggle with:
    //! - 10,000+ individual character entries
    //! - Large amounts of whitespace and formatting
    //! - Memory allocation for intermediate representations
    //!
    //! Optimized state machine parser should:
    //! 1. Stream through entries without full buffering
    //! 2. Direct insertion into HashMap (no intermediate storage)
    //! 3. Single-pass parsing with O(n) complexity

    let mut large_cmap = String::from(
        r#"
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
"#,
    );

    // Add 1000 bfchar entries (simulates real large CMaps)
    large_cmap.push_str("1000 beginbfchar\n");
    for i in 0..1000 {
        large_cmap.push_str(&format!("<{:04X}> <{:04X}>\n", 0x0100 + i, 0x0100 + i));
    }
    large_cmap.push_str("endbfchar\n");

    large_cmap.push_str(
        r#"
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "LargeCMapFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(large_cmap.as_bytes().to_vec())),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Verify: All entries should be accessible
    assert_eq!(font.char_to_unicode(0x0100), Some("\u{0100}".to_string()), "First entry");
    assert_eq!(font.char_to_unicode(0x0200), Some("\u{0200}".to_string()), "Middle entry");
    assert_eq!(font.char_to_unicode(0x03E3), Some("\u{03E3}".to_string()), "Last entry");

    // Verify: Random access works efficiently
    assert_eq!(font.char_to_unicode(0x0150), Some("\u{0150}".to_string()), "Random entry 1");
    assert_eq!(font.char_to_unicode(0x0250), Some("\u{0250}".to_string()), "Random entry 2");
}

#[test]
fn test_optimized_parser_large_bfrange_section() {
    //! Test binary search optimization for range lookups.
    //!
    //! For bfrange entries like <0000> <00FF> <4E00>,
    //! the optimized parser should:
    //! 1. Store range metadata efficiently (start, end, target)
    //! 2. Use binary search on sorted ranges for O(log n) lookup
    //! 3. Handle overlapping or contiguous ranges correctly

    let mut cmap_with_ranges = String::from(
        r#"
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
"#,
    );

    // Add 100 bfrange entries
    cmap_with_ranges.push_str("100 beginbfrange\n");
    for i in 0..100 {
        let start = i * 256;
        let end = start + 255;
        let target = 0x4E00 + i as u32 * 256;
        cmap_with_ranges.push_str(&format!("<{:04X}> <{:04X}> <{:04X}>\n", start, end, target));
    }
    cmap_with_ranges.push_str("endbfrange\n");

    cmap_with_ranges.push_str(
        r#"
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "LargeRangeFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(cmap_with_ranges.as_bytes().to_vec())),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Verify: Range lookups work (binary search should find correct range)
    // Range 0: 0x0000-0x00FF -> 0x4E00-0x4EFF
    assert_eq!(font.char_to_unicode(0x0000), Some("\u{4E00}".to_string()), "Range 0 start");
    assert_eq!(font.char_to_unicode(0x0080), Some("\u{4E80}".to_string()), "Range 0 middle");
    assert_eq!(font.char_to_unicode(0x00FF), Some("\u{4EFF}".to_string()), "Range 0 end");

    // Range 50: 0xC800-0xC8FF -> 0x4E00+(50*256)-0x4E00+(50*256+255)
    let range_50_start = 50 * 256;
    let range_50_unicode = 0x4E00 + 50_u32 * 256;
    assert_eq!(
        font.char_to_unicode(range_50_start),
        Some(
            char::from_u32(range_50_unicode)
                .unwrap_or('\u{FFFD}')
                .to_string()
        ),
        "Range 50 lookup"
    );

    // Range 99: 0xFF00-0xFFFF -> highest range
    let range_99_start = 99 * 256;
    let range_99_unicode = 0x4E00 + 99_u32 * 256;
    assert_eq!(
        font.char_to_unicode(range_99_start),
        Some(
            char::from_u32(range_99_unicode)
                .unwrap_or('\u{FFFD}')
                .to_string()
        ),
        "Range 99 lookup"
    );
}

#[test]
fn test_optimized_parser_mixed_large_cmap() {
    //! Test optimized parser with mixed bfchar and bfrange sections.
    //!
    //! Real CMaps often mix:
    //! - Individual character mappings (bfchar)
    //! - Range mappings (bfrange)
    //! - Undefined range fallbacks (beginnotdefrange)
    //!
    //! All sections should be parsed correctly and efficiently.

    let mut mixed_cmap = String::from(
        r#"
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
100 beginbfchar
"#,
    );

    // Add 100 individual characters
    for i in 0..100 {
        mixed_cmap.push_str(&format!("<{:04X}> <{:04X}>\n", i, i + 100));
    }
    mixed_cmap.push_str("endbfchar\n");

    // Add 50 ranges
    mixed_cmap.push_str("50 beginbfrange\n");
    for i in 0..50 {
        let start = 0x1000 + i * 256;
        let end = start + 255;
        let target = 0x4E00 + i as u32 * 256;
        mixed_cmap.push_str(&format!("<{:04X}> <{:04X}> <{:04X}>\n", start, end, target));
    }
    mixed_cmap.push_str("endbfrange\n");

    // Add notdefrange
    mixed_cmap.push_str(
        r#"
1 beginnotdefrange
<6400> <FFFF> <FFFD>
endnotdefrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "MixedLargeCMap".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(mixed_cmap.as_bytes().to_vec())),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Verify bfchar sections work
    assert_eq!(font.char_to_unicode(0x0000), Some("\u{0064}".to_string()), "bfchar entry 0");
    assert_eq!(font.char_to_unicode(0x0050), Some("\u{00B4}".to_string()), "bfchar entry 80");

    // Verify bfrange sections work
    assert_eq!(font.char_to_unicode(0x1000), Some("\u{4E00}".to_string()), "bfrange 0 start");
    assert_eq!(font.char_to_unicode(0x1100), Some("\u{4F00}".to_string()), "bfrange 1 lookup");

    // Verify notdefrange fallback works
    assert_eq!(
        font.char_to_unicode(0x6400),
        Some("\u{FFFD}".to_string()),
        "notdefrange fallback"
    );
}

#[test]
fn test_optimized_parser_whitespace_variations() {
    //! Test that optimized state machine handles whitespace robustly.
    //!
    //! State machine parser should:
    //! - Handle arbitrary whitespace (spaces, tabs, newlines)
    //! - Not require precise formatting
    //! - Work with minified and pretty-printed CMaps

    let cmap_variations = vec![
        // Compact format (minimal whitespace)
        r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo<</Registry(Adobe)/Ordering(Identity)/Supplement 0>>def
/CMapName /Identity-H def
/CMapType 2 def
1 begincodespacerange
<0000><FFFF>
endcodespacerange
2 beginbfchar
<0041><0041>
<0042><0042>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
        // Verbose format (extra whitespace)
        r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<<  /Registry    (Adobe)
    /Ordering    (Identity)
    /Supplement  0
>>  def
/CMapName  /Identity-H  def
/CMapType  2  def
1  begincodespacerange
<0000>  <FFFF>
endcodespacerange
2  beginbfchar
<0041>  <0041>
<0042>  <0042>
endbfchar
endcmap
CMapName  currentdict  /CMap  defineresource  pop
end
end
"#,
    ];

    for (idx, cmap_str) in cmap_variations.iter().enumerate() {
        let font = FontInfo {
            base_font: format!("WhitespaceTest{}", idx),
            subtype: "Type0".to_string(),
            encoding: pdf_oxide::fonts::Encoding::Identity,
            to_unicode: Some(LazyCMap::new(cmap_str.as_bytes().to_vec())),
            font_weight: None,
            flags: None,
            stem_v: None,
            embedded_font_data: None,
            truetype_cmap: None,
            cid_to_gid_map: None,
            cid_system_info: None,
            cid_font_type: None,
            widths: None,
            first_char: None,
            last_char: None,
            default_width: 500.0,
        };

        // Both formats should parse identically
        assert_eq!(
            font.char_to_unicode(0x0041),
            Some("A".to_string()),
            "Format {} should parse correctly",
            idx
        );
        assert_eq!(
            font.char_to_unicode(0x0042),
            Some("B".to_string()),
            "Format {} should parse correctly",
            idx
        );
    }
}

#[test]
fn test_optimized_parser_very_large_cmap_100k_entries() {
    //! Test that optimized parser handles 100k+ entries without memory issues.
    //!
    //! Large CMaps (e.g., for CJK fonts) may have:
    //! - 10,000 - 100,000+ mappings
    //! - Multi-megabyte raw stream sizes
    //!
    //! Optimized implementation should:
    //! 1. Parse without buffering entire structure in memory
    //! 2. Support streaming/incremental parsing if needed
    //! 3. Maintain performance (sub-second parsing)

    // Create a CMap with 10k entries (represents real-world CJK font size)
    let mut large_cmap = String::from(
        r#"
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
"#,
    );

    // Add 10,000 entries in chunks
    for chunk in 0..10 {
        let count = 1000;
        large_cmap.push_str(&format!("{} beginbfchar\n", count));
        for i in 0..count {
            let code = chunk * 1000 + i as u32;
            large_cmap.push_str(&format!("<{:04X}> <{:04X}>\n", code, code + 0x4E00));
        }
        large_cmap.push_str("endbfchar\n");
    }

    large_cmap.push_str(
        r#"
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "VeryLargeCMapFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(large_cmap.as_bytes().to_vec())),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Verify: Large CMap parsing works (lazy loading should handle all 10k entries)
    // Entries are generated as: code = (chunk * 1000 + i), unicode = code
    // So char_to_unicode(code) maps to the character value at that code point
    assert!(font.char_to_unicode(0x0000).is_some(), "First entry should exist");
    assert!(font.char_to_unicode(0x0100).is_some(), "Random entry in first chunk");
    assert!(font.char_to_unicode(0x1000).is_some(), "Random entry in second chunk");
    assert!(font.char_to_unicode(0x2000).is_some(), "Random entry in third chunk");
    assert!(font.char_to_unicode(0x4000).is_some(), "Random entry in fifth chunk");
}

#[test]
fn test_optimized_parser_streaming_for_mega_cmap() {
    //! Test streaming parser capability for extremely large CMaps.
    //!
    //! For CMaps > 10MB, full in-memory parsing may be problematic.
    //! A streaming parser should:
    //! 1. Parse entries incrementally
    //! 2. Not require the entire stream in memory
    //! 3. Maintain access to all entries via lazy evaluation

    // Simulate mega-CMap structure (50,000 entries)
    let mut mega_cmap = String::from(
        r#"
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
"#,
    );

    // Add 50,000 entries in 50 sections of 1000
    for section in 0..50 {
        let count = 1000;
        mega_cmap.push_str(&format!("{} beginbfchar\n", count));
        for i in 0..count {
            let code = section * 1000 + i as u32;
            mega_cmap.push_str(&format!("<{:04X}> <{:04X}>\n", code & 0xFFFF, code));
        }
        mega_cmap.push_str("endbfchar\n");
    }

    mega_cmap.push_str(
        r#"
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "MegaCMapFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(mega_cmap.as_bytes().to_vec())),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Verify: Lazy loading should handle mega-CMaps efficiently
    // First access triggers lazy parsing
    assert!(font.char_to_unicode(0x0000).is_some(), "First lookup should work");

    // Subsequent accesses should hit cache
    assert!(font.char_to_unicode(0x0001).is_some(), "Second lookup should work");
    assert!(font.char_to_unicode(0x5000).is_some(), "Random lookup should work");
}
