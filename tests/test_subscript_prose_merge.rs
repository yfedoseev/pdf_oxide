//! In-prose chemistry subscripts (`NH3` inside a sentence) and the limit of the
//! current span-level sub/superscript merge.
//!
//! **Status of `subscript_after_long_base_does_not_float`: `#[ignore]`.** It is a
//! faithful reproducer of a real benchmark defect that the current architecture
//! cannot fix with a localized change, kept here to pin the root cause for a
//! future reading-order pass.
//!
//! Root cause: `merge_sub_superscript_spans` runs *after* line assembly and
//! works at span granularity, appending the subscript to the end of its base
//! span. In prose the base letter `H` is in the middle of an assembled line span
//! (`"the NH inversion transitions"`) because the lowered subscript glyph does
//! not break the horizontal run. The whole line is therefore the base candidate
//! (its last token `transitions` is not a valid host), and even if it were, the
//! append-to-end model would misplace the digit. The correct fix binds the
//! subscript to its base *character* before the line span is assembled — a
//! reading-order change that must be validated against the full corpus, not a
//! gate tweak (relaxing the host/x-adjacency gates regresses the corpus).
//!
//! `water_formula_still_merges` stays active: the short-base case the current
//! merge DOES handle must not regress.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::writer::{PageBuilder, PdfWriter};

fn put(page: &mut PageBuilder<'_>, text: &str, x: f32, y: f32, font: &str, size: f32) {
    page.add_text(text, x, y, font, size);
    // Force a BT/ET boundary so adjacent runs stay separate text objects.
    page.draw_rect(0.0, 0.0, 0.0, 0.0);
}

fn build_and_extract(build_fn: impl FnOnce(&mut PdfWriter)) -> String {
    let mut writer = PdfWriter::new();
    build_fn(&mut writer);
    let bytes = writer.finish().expect("build PDF");
    let doc = PdfDocument::from_bytes(bytes).expect("open PDF");
    doc.extract_text(0).expect("extract page 0")
}

#[test]
#[ignore = "known limitation: span-level merge cannot place a mid-line subscript \
            whose base letter is interior to an assembled line span — needs a \
            pre-assembly reading-order binding validated against the full corpus"]
fn subscript_after_long_base_does_not_float() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "the NH", 100.0, 200.0, "Helvetica", 12.0);
        put(&mut page, "3", 134.0, 197.0, "Helvetica", 8.0);
        put(&mut page, "inversion transitions", 142.0, 200.0, "Helvetica", 12.0);
        put(&mut page, "second line of body text here", 100.0, 186.0, "Helvetica", 12.0);
    });

    assert!(
        out.contains("NH3") || out.contains("NH\u{2083}"),
        "subscript did not attach to its base: {out:?}"
    );
    assert!(
        !out.contains("transitions 3") && !out.contains("transitions3"),
        "subscript floated to line end: {out:?}"
    );
}

#[test]
fn water_formula_still_merges() {
    // The short-base case the current merge handles must not regress.
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "H", 100.0, 200.0, "Helvetica", 14.0);
        put(&mut page, "2", 112.0, 197.0, "Helvetica", 9.0);
        put(&mut page, "O", 122.0, 200.0, "Helvetica", 14.0);
    });
    let collapsed: String = out.split_whitespace().collect();
    assert_eq!(collapsed, "H\u{2082}O", "H2O regressed: {collapsed:?}");
}
