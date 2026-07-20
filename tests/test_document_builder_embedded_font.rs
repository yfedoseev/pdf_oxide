//! Regression tests for issue #382 — CJK / Unicode text through the
//! high-level `DocumentBuilder` API.
//!
//! Before the fix, `FluentPageBuilder::font(user_name, size).text(...)`
//! stored the user-supplied font name as a plain string, and the
//! content-stream emitter silently fell back to Helvetica for any
//! name outside the base-14 set. CJK / Cyrillic / Greek characters
//! rendered as empty or tofu.
//!
//! After the fix, `DocumentBuilder::register_embedded_font(name, font)`
//! routes matching text through the Type-0 / CIDFontType2 path so the
//! embedded subset is actually used.

use pdf_oxide::writer::{DocumentBuilder, EmbeddedFont};
use pdf_oxide::PdfDocument;
use std::path::Path;

/// DejaVu Sans covers Latin + Cyrillic + Greek — enough to prove the
/// Type-0 path is live without shipping a CJK font in the test fixtures.
fn load_dejavu_sans() -> EmbeddedFont {
    let font_path = Path::new("tests/fixtures/fonts/DejaVuSans.ttf");
    EmbeddedFont::from_file(font_path)
        .expect("tests/fixtures/fonts/DejaVuSans.ttf should exist and parse")
}

#[test]
fn document_builder_registered_embedded_font_round_trips_cyrillic() {
    let font = load_dejavu_sans();

    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVuCustom", font);
    builder
        .a4_page()
        .font("DejaVuCustom", 12.0)
        .at(72.0, 700.0)
        .text("Привет, мир!")
        .at(72.0, 680.0)
        .text("Καλημέρα κόσμε")
        .done();

    let bytes = builder.build().expect("build should succeed");

    let doc = PdfDocument::from_bytes(bytes).expect("parse produced pdf");
    let text = doc.extract_text(0).expect("extract_text should succeed");

    assert!(text.contains("Привет, мир!"), "Cyrillic round-trip failed — got: {text:?}");
    assert!(text.contains("Καλημέρα κόσμε"), "Greek round-trip failed — got: {text:?}");
}

#[test]
fn document_builder_registered_embedded_font_emits_hex_tj_not_literal() {
    let font = load_dejavu_sans();

    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVuCustom", font);
    builder
        .a4_page()
        .font("DejaVuCustom", 12.0)
        .at(72.0, 700.0)
        .text("Привет")
        .done();

    let bytes = builder.build().expect("build should succeed");
    let content = String::from_utf8_lossy(&bytes);

    // Embedded-font path emits hex strings: `<HHHH...> Tj`.
    // Base-14 fallback would emit a literal: `(Привет) Tj`.
    assert!(
        content.contains("/EF1"),
        "content stream should reference the /EFn embedded resource — \
         missing /EF1 in output"
    );
    assert!(
        !content.contains("(Привет)"),
        "raw Cyrillic literal leaked through — embedded-font path not taken"
    );
}

#[test]
fn document_builder_unknown_font_name_falls_back_to_base14_no_panic() {
    // Sanity: behaviour for unregistered names is unchanged — the
    // content still renders (via Helvetica) and `build()` succeeds.
    let mut builder = DocumentBuilder::new();
    builder
        .letter_page()
        .font("TotallyMadeUpFont", 12.0)
        .at(72.0, 700.0)
        .text("hello world")
        .done();

    let bytes = builder.build().expect("build should succeed");
    let content = String::from_utf8_lossy(&bytes);

    assert!(content.starts_with("%PDF-1.7"));
    // Base-14 ASCII text is emitted as a literal Tj.
    assert!(content.contains("(hello world)"));
}

#[test]
fn document_builder_mixed_base14_and_embedded_on_same_page() {
    let font = load_dejavu_sans();

    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVuCustom", font);
    builder
        .a4_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .text("English via base-14")
        .font("DejaVuCustom", 12.0)
        .at(72.0, 700.0)
        .text("Привет via embedded")
        .done();

    let bytes = builder.build().expect("build should succeed");

    let doc = PdfDocument::from_bytes(bytes).expect("parse produced pdf");
    let text = doc.extract_text(0).expect("extract_text should succeed");

    assert!(text.contains("English via base-14"), "base-14 text missing: {text:?}");
    assert!(text.contains("Привет via embedded"), "embedded text missing: {text:?}");
}

/// Regression for the embedded-font `inline_color` colour drop: the
/// glyph-run dispatch in `PdfWriter::add_element` used to route embedded
/// text through the colour-agnostic `add_embedded_text` without ever
/// emitting the element's fill colour, so `inline_color`/coloured text on
/// an embedded font rendered black (no `rg` operator in the stream). The
/// base-14 path always emits `rg`; the embedded path must match it.
#[test]
fn document_builder_embedded_font_inline_color_emits_rg() {
    let font = load_dejavu_sans();

    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVuCustom", font);
    builder
        .a4_page()
        .font("DejaVuCustom", 20.0)
        .at(10.0, 12.0)
        .inline_color(0.5, 0.5, 0.5, "hello")
        .done();

    let bytes = builder.build().expect("build should succeed");
    let content = String::from_utf8_lossy(&bytes);

    // Embedded path must be the one taken (control: the /EFn resource).
    assert!(content.contains("/EF1"), "embedded-font path not taken — missing /EF1");
    // The requested fill colour must reach the content stream.
    assert!(
        content.contains("0.5 0.5 0.5 rg"),
        "fill-colour operator missing for embedded-font inline_color — \
         embedded text would render black"
    );
}
