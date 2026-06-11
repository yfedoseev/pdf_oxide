//! A signed unit exponent (`s−1`, `m−2`) must not be rewritten into a Unicode
//! sub/superscript digit.
//!
//! The document-level pass substitutes ASCII digits in lowered/raised
//! smaller-font spans with their Unicode sub/superscript equivalents — correct
//! for chemistry (`H₂O`) and ordinals (`8ᵗʰ`). But a scientific unit exponent
//! such as `km s−1` is, by the plaintext convention every reference extractor
//! follows, kept as ASCII `s−1`. The geometric classifier fires inconsistently
//! on these (some occurrences become `s−₁`, others stay `s−1`), so the result is
//! both wrong and non-deterministic. A digit whose nearest preceding glyph is a
//! minus/hyphen sign is a signed exponent and must be left as ASCII.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::writer::{PageBuilder, PdfWriter};

fn put(page: &mut PageBuilder<'_>, text: &str, x: f32, y: f32, font: &str, size: f32) {
    page.add_text(text, x, y, font, size);
    // Force a BT/ET boundary so each glyph is a separate text object/span.
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
fn signed_unit_exponent_stays_ascii() {
    // The exponent `-1` is set as one smaller (9 pt), lowered (−3 pt) run — the
    // shape a typesetter emits for `s⁻¹` — flanked by base-size letters so it
    // satisfies the substitution's token-internal gate. Without the guard the
    // whole run is rewritten to the Unicode subscript `₋₁`; with it the run
    // stays ASCII `-1`, matching the plaintext convention.
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "s", 100.0, 200.0, "Helvetica", 14.0);
        put(&mut page, "-1", 110.0, 197.0, "Helvetica", 9.0); // smaller + lowered signed run
        put(&mut page, "s", 124.0, 200.0, "Helvetica", 14.0);
    });

    let collapsed: String = out.split_whitespace().collect();
    assert!(
        !collapsed.contains('\u{2081}') && !collapsed.contains('\u{208B}'),
        "signed unit exponent wrongly rewritten to subscript: {collapsed:?}"
    );
    assert!(collapsed.contains("-1"), "expected ASCII '-1' to survive, got: {collapsed:?}");
}

#[test]
fn chemistry_subscript_still_substitutes() {
    // Guard must NOT regress the real subscript case: `H2O` → `H₂O`. The digit's
    // preceding glyph is the letter `H`, not a sign.
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "H", 100.0, 200.0, "Helvetica", 14.0);
        put(&mut page, "2", 112.0, 197.0, "Helvetica", 9.0);
        put(&mut page, "O", 122.0, 200.0, "Helvetica", 14.0);
    });

    let collapsed: String = out.split_whitespace().collect();
    assert_eq!(collapsed, "H\u{2082}O", "chemistry subscript regressed: {collapsed:?}");
}
