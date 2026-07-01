//! Regression test for issue #794's digit-less-parity gate
//! (`ensure_running_artifact_signatures`): a constant, digit-less phrase
//! that appears on BOTH page parities — not confined to (or complementary
//! on) one side — must not be treated as recto/verso alternating chrome,
//! even if one parity alone happens to clear the 80% bar.
//!
//! Mirrors the real over-strip found on IRS Form 709's "(If more space is
//! needed, attach additional statements.)" instructional note: it recurs
//! on 4 of 5 even-indexed body pages (80%, clears the gate) AND 2 of 5
//! odd-indexed body pages (40%, real but below the gate) — a shape the
//! current gate can't distinguish from genuine one-sided alternation
//! (e.g. "com" in a split "ERLC.com" footer, which is [0, N] — absent
//! from the losing parity entirely).

mod common;
use common::build_pdf_with_page_extras;
use pdf_oxide::PdfDocument;

fn footer_line(text: &str) -> String {
    format!("BT /F1 10 Tf 1 0 0 1 72 30 Tm ({text}) Tj ET\n")
}

#[test]
fn remove_footers_preserves_phrase_present_on_both_parities() {
    // Even-indexed pages 2,4,6,8 (4 of 5 = 80%) and odd-indexed pages 3,5
    // (2 of 5 = 40%) — present on both parities, not confined to one.
    let phrase_pages = [2usize, 3, 4, 5, 6, 8];
    let bytes = build_pdf_with_page_extras(10, |i| {
        if phrase_pages.contains(&i) {
            footer_line("If more space is needed, attach additional statements.")
        } else {
            String::new()
        }
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    // 0.9: high enough that pass 2's own exact-text heuristic
    // (`ceil(10 * 0.9) = 9`) can't independently catch the phrase, which
    // recurs on only 6 of 10 pages — this test is about the digit-less-
    // parity signature path specifically.
    doc.remove_footers(0.9).unwrap();

    for &page in &phrase_pages {
        let text = doc.extract_text(page).unwrap();
        assert!(
            text.contains("If more space is needed"),
            "page {page}: real instructional phrase wrongly removed as \
             footer chrome — it appears on both page parities (80% even, \
             40% odd), not confined to one side, so it isn't genuine \
             recto/verso alternating chrome: {text:?}"
        );
    }
}
