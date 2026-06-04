//! Tests for ToUnicode CMap document-order processing.
//!
//! pdf.js, MuPDF, and Poppler all parse bfchar and bfrange sections in document
//! order with last-wins semantics.  When the same code appears in multiple
//! sections, whichever section appears last in the stream provides the
//! authoritative mapping.

use pdf_oxide::fonts::cmap::parse_tounicode_cmap;

/// bfchar entries that overlap with a bfrange must take precedence.
///
/// Setup:
///   - bfrange: codes 0x41..0x5A → 'A'..'Z'  (ASCII uppercase letters)
///   - bfchar:  code  0x41       → '!'         (specific override for 'A')
///
/// Expected: code 0x41 resolves to '!', not 'A'.
#[test]
fn bfchar_overrides_bfrange_for_same_code() {
    let cmap_data = b"\
/CIDInit /ProcSet findresource begin\n\
12 dict begin\n\
begincmap\n\
/CMapName /Test def\n\
1 begincodespacerange\n\
<00> <FF>\n\
endcodespacerange\n\
1 beginbfrange\n\
<41> <5A> <0041>\n\
endbfrange\n\
1 beginbfchar\n\
<41> <0021>\n\
endbfchar\n\
endcmap\n\
end\n\
end\n";

    let cmap = parse_tounicode_cmap(cmap_data).expect("parse CMap");

    // Code 0x41 must use the bfchar override ('!' = U+0021), not the bfrange value ('A').
    let val = cmap.get(&0x41).expect("code 0x41 must be mapped");
    assert_eq!(
        val, "!",
        "bfchar <41>=<0021> must override bfrange <41>-<5A>=<0041>; got {:?}",
        val
    );

    // Other codes in the bfrange still use the bfrange mapping.
    let b_val = cmap.get(&0x42).expect("code 0x42 must be mapped");
    assert_eq!(b_val, "B", "code 0x42 should still resolve to 'B' from the bfrange");
}

/// A bfrange without an overlapping bfchar is unaffected.
#[test]
fn bfrange_codes_without_bfchar_override_use_range_value() {
    let cmap_data = b"\
/CIDInit /ProcSet findresource begin\n\
12 dict begin\n\
begincmap\n\
/CMapName /Test def\n\
1 begincodespacerange\n\
<00> <FF>\n\
endcodespacerange\n\
1 beginbfrange\n\
<41> <45> <0041>\n\
endbfrange\n\
1 beginbfchar\n\
<41> <0021>\n\
endbfchar\n\
endcmap\n\
end\n\
end\n";

    let cmap = parse_tounicode_cmap(cmap_data).expect("parse CMap");

    // Codes 0x42..0x45 have no bfchar override — bfrange value prevails.
    assert_eq!(cmap.get(&0x42).as_deref(), Some("B"));
    assert_eq!(cmap.get(&0x43).as_deref(), Some("C"));
    assert_eq!(cmap.get(&0x44).as_deref(), Some("D"));
    assert_eq!(cmap.get(&0x45).as_deref(), Some("E"));
}

/// bfchar entries that do not overlap with any bfrange are still mapped.
#[test]
fn bfchar_without_bfrange_overlap_is_mapped() {
    let cmap_data = b"\
/CIDInit /ProcSet findresource begin\n\
12 dict begin\n\
begincmap\n\
/CMapName /Test def\n\
1 begincodespacerange\n\
<00> <FF>\n\
endcodespacerange\n\
1 beginbfrange\n\
<41> <45> <0041>\n\
endbfrange\n\
1 beginbfchar\n\
<61> <0078>\n\
endbfchar\n\
endcmap\n\
end\n\
end\n";

    let cmap = parse_tounicode_cmap(cmap_data).expect("parse CMap");

    // Code 0x61 ('a' area) has only a bfchar entry → must resolve to 'x' (U+0078).
    assert_eq!(cmap.get(&0x61).as_deref(), Some("x"), "bfchar-only code 0x61 should map to 'x'");
}

/// When bfchar appears *before* bfrange in the stream and both map the same
/// code, the bfrange entry wins because it comes later (document order).
///
/// This is the mirror of `bfchar_overrides_bfrange_for_same_code` and
/// confirms that the implementation is truly document-order rather than
/// always favouring one section type over the other.
#[test]
fn bfrange_overrides_bfchar_when_bfrange_comes_last() {
    let cmap_data = b"\
/CIDInit /ProcSet findresource begin\n\
12 dict begin\n\
begincmap\n\
/CMapName /Test def\n\
1 begincodespacerange\n\
<00> <FF>\n\
endcodespacerange\n\
1 beginbfchar\n\
<41> <0021>\n\
endbfchar\n\
1 beginbfrange\n\
<41> <5A> <0041>\n\
endbfrange\n\
endcmap\n\
end\n\
end\n";

    let cmap = parse_tounicode_cmap(cmap_data).expect("parse CMap");

    // bfrange comes last in the stream, so it wins: code 0x41 → 'A', not '!'.
    let val = cmap.get(&0x41).expect("code 0x41 must be mapped");
    assert_eq!(
        val, "A",
        "bfrange <41>-<5A>=<0041> must win over earlier bfchar <41>=<0021>; got {:?}",
        val
    );
}
