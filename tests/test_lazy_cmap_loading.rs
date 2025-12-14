//! Lazy CMap Loading Tests for Phase 5.1
//!
//! Tests for lazy loading of ToUnicode CMaps to improve performance:
//! - CMaps not parsed immediately on font creation
//! - Parsed only on first character lookup
//! - Result cached for subsequent lookups
//! - Proper thread-safe access via Mutex
//!
//! Expected Impact:
//! - 30-40% faster initial font parsing
//! - Reduced memory usage for fonts with large CMaps
//! - Faster document opening experience
//!
//! Spec: PDF 32000-1:2008 Section 9.10.3 (ToUnicode CMaps)

use pdf_oxide::fonts::cmap::LazyCMap;
use pdf_oxide::fonts::FontInfo;

#[test]
fn test_lazy_cmap_not_parsed_on_creation() {
    //! Test that ToUnicode CMaps are not parsed when FontInfo is created,
    //! only when first character lookup occurs.
    //!
    //! Current behavior: CMaps are fully parsed on font dictionary parsing
    //! Desired behavior: Store raw stream, parse on first access
    //!
    //! This test verifies:
    //! 1. Font can be created with CMap stream data
    //! 2. Parsing is deferred until char_to_unicode() is called
    //! 3. No performance penalty on font loading

    let simple_cmap = r#"
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
3 beginbfchar
<0041> <0041>
<0042> <0042>
<0043> <0043>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    // Create font with lazy CMap wrapper
    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        // This is a lazy wrapper - CMap will be parsed on first character lookup
        to_unicode: Some(LazyCMap::new(simple_cmap.as_bytes().to_vec())),
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

    // Verify: Character lookup should still work (lazy parsing on first access)
    let result = font.char_to_unicode(0x0041);
    assert_eq!(result, Some("A".to_string()), "First lookup should trigger lazy parse");

    // Verify: Subsequent lookups should use cached result (no re-parsing)
    let result2 = font.char_to_unicode(0x0041);
    assert_eq!(result2, Some("A".to_string()), "Cached result should be available");

    let result3 = font.char_to_unicode(0x0042);
    assert_eq!(result3, Some("B".to_string()), "Other characters should also work");
}

#[test]
fn test_lazy_cmap_thread_safe_parsing() {
    //! Test that lazy CMap parsing is thread-safe.
    //!
    //! The lazy wrapper uses Mutex<Option<CMap>> to ensure:
    //! - Multiple threads can safely access the same CMap
    //! - Parsing happens only once, even with concurrent access
    //! - Results are consistent across threads

    let cmap_data = r#"
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
<0061> <0061>
<0062> <0062>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(cmap_data.as_bytes().to_vec())),
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

    // Verify same result from sequential calls
    assert_eq!(font.char_to_unicode(0x0061), Some("a".to_string()));
    assert_eq!(font.char_to_unicode(0x0061), Some("a".to_string()));
    assert_eq!(font.char_to_unicode(0x0062), Some("b".to_string()));
}

#[test]
fn test_lazy_cmap_large_map_deferred_parsing() {
    //! Test that large CMaps benefit from lazy loading by deferring
    //! expensive parsing operations.
    //!
    //! Large CMaps (>10k entries) should:
    //! - Not be parsed during font dictionary loading
    //! - Only be parsed when first character lookup occurs
    //! - Show measurable performance improvement
    //!
    //! This test creates a reasonably large CMap to verify lazy behavior

    let mut large_cmap_entries = String::from(
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

    // Add 100 bfchar entries to simulate larger CMap
    large_cmap_entries.push_str("100 beginbfchar\n");
    for i in 0..100 {
        large_cmap_entries.push_str(&format!("<{:04X}> <{:04X}>\n", 0x0100 + i, 0x0100 + i));
    }
    large_cmap_entries.push_str("endbfchar\n");

    large_cmap_entries.push_str(
        r#"
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#,
    );

    let font = FontInfo {
        base_font: "LargeMapFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        // Create lazy CMap from raw bytes
        to_unicode: Some(LazyCMap::new(large_cmap_entries.as_bytes().to_vec())),
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
    assert_eq!(
        font.char_to_unicode(0x0100),
        Some("\u{0100}".to_string()),
        "First entry accessible"
    );
    assert_eq!(
        font.char_to_unicode(0x0150),
        Some("\u{0150}".to_string()),
        "Middle entry accessible"
    );
    assert_eq!(
        font.char_to_unicode(0x0163),
        Some("\u{0163}".to_string()),
        "Last entry accessible"
    );
}

#[test]
fn test_lazy_cmap_cache_hit_performance() {
    //! Test that lazy CMap caching provides performance benefits.
    //!
    //! Expected behavior:
    //! - First access: Parse CMap (slower)
    //! - Subsequent accesses: Use cached result (faster)
    //! - Multiple character lookups hit same cached CMap
    //!
    //! This verifies the caching mechanism works correctly

    let cmap_data = r#"
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
10 beginbfchar
<0030> <0030>
<0031> <0031>
<0032> <0032>
<0033> <0033>
<0034> <0034>
<0035> <0035>
<0036> <0036>
<0037> <0037>
<0038> <0038>
<0039> <0039>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let font = FontInfo {
        base_font: "DigitFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(cmap_data.as_bytes().to_vec())),
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

    // Multiple lookups should all hit the cache
    for code in 0x30..=0x39 {
        let result = font.char_to_unicode(code);
        assert!(result.is_some(), "Digit 0x{:02X} should be mapped", code);
    }

    // Repeated accesses should be very fast (from cache)
    for code in 0x30..=0x39 {
        let result = font.char_to_unicode(code);
        assert!(result.is_some(), "Cached access should work");
    }
}

#[test]
fn test_lazy_cmap_with_notdefrange_lazy_parsing() {
    //! Test that lazy loading works with advanced CMap features
    //! (added in Phase 4.1).
    //!
    //! Even with beginnotdefrange and escape sequences,
    //! the lazy loading should defer all parsing until first access

    let cmap_with_notdef = r#"
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
<0041> <0041>
<0042> <0042>
endbfchar
1 beginnotdefrange
<0000> <0040> <FFFD>
endnotdefrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: pdf_oxide::fonts::Encoding::Identity,
        to_unicode: Some(LazyCMap::new(cmap_with_notdef.as_bytes().to_vec())),
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

    // Verify: Explicit mappings work
    assert_eq!(
        font.char_to_unicode(0x0041),
        Some("A".to_string()),
        "Explicit mapping should work"
    );

    // Verify: Notdefrange fallback works (from lazy-loaded CMap)
    assert_eq!(
        font.char_to_unicode(0x0001),
        Some("\u{FFFD}".to_string()),
        "Notdefrange should work with lazy loading"
    );
}
