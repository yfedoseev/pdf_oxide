//! Native (born-digital) multi-script extraction through the auto
//! surface (#519: confirm non-Latin works, not just English).
//!
//! Coverage rationale (honest): broad CJK / Arabic / Hebrew / Cyrillic
//! born-digital extraction is already exercised by the repo's
//! **running, non-ignored** suites — `test_cjk_script_support` (55),
//! `test_complex_script_support` (44, incl. RTL Arabic/Hebrew),
//! `test_cyrillic_encoding_and_utf8_sniff`, plus the 4-byte / CMap /
//! CFF font suites — all against the canonical `extract_text`.
//! `tests/test_auto_semantics_519.rs` proves
//! `AutoExtractor::extract_text` is **byte-identical** to canonical
//! `extract_text` per page, so the auto surface inherits that
//! multi-script coverage by construction.
//!
//! This test adds a direct, visible auto-surface assertion on a real
//! committed CJK document so the guarantee is concretely demonstrated,
//! not only transitive.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::auto::AutoExtractor;

/// `tests/fixtures/1.pdf` is a real Chinese financial research report
/// (~17k chars of born-digital CJK). These ideographs are present
/// (股=stock 票=ticket 基金=fund 研究=research 专题=topic).
const CJK_MARKERS: &[char] = &['股', '票', '基', '金', '研', '究', '专', '题'];

#[test]
fn auto_extracts_native_cjk_text_519() {
    let doc = PdfDocument::open("tests/fixtures/1.pdf").expect("open 1.pdf (CJK)");
    let ae = AutoExtractor::new();

    let text = ae.extract_text(&doc, 0).expect("ae.extract_text");
    let present: Vec<char> = CJK_MARKERS
        .iter()
        .copied()
        .filter(|c| text.contains(*c))
        .collect();
    assert!(
        present.len() >= 6,
        "AutoExtractor must extract native CJK text — expected ≥6 of \
         {CJK_MARKERS:?}, found {present:?}"
    );
    assert!(
        !text.contains('\u{FFFD}'),
        "CJK extraction must not be garbled (U+FFFD present)"
    );

    // The auto surface must be byte-faithful to the canonical extractor
    // on this multi-byte/CJK document (no script-specific divergence).
    assert_eq!(
        text,
        doc.extract_text(0).expect("canonical extract_text"),
        "auto native path must equal canonical extract_text for CJK"
    );

    // Markdown of the CJK page must also carry the script, faithfully.
    let md = ae.extract_markdown(&doc, 0).expect("ae.extract_markdown");
    assert_eq!(
        md,
        doc.to_markdown(0, &ConversionOptions::default())
            .expect("to_markdown"),
        "auto markdown must equal canonical to_markdown for CJK"
    );
    assert!(
        CJK_MARKERS.iter().filter(|c| md.contains(**c)).count() >= 6,
        "markdown must retain the CJK content"
    );
}
