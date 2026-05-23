//! Round-trip tests for embedded TrueType fonts (FONT-3).
//!
//! Build a PDF that registers a custom font, writes some Unicode text
//! through the embedded path, then re-opens the PDF via `Pdf::from_bytes`
//! and asserts that `extract_text` returns the input. This is the
//! v0.3.35 plan's acceptance criterion for Phase FONT — if the
//! Type 0 / CIDFontType2 / FontDescriptor / FontFile2 / ToUnicode
//! object graph is correct, this round-trip works; if any of them is
//! malformed, extract_text returns garbage glyph IDs instead of the
//! source string.

use pdf_oxide::writer::{EmbeddedFont, PdfWriter};
use pdf_oxide::PdfDocument;

const DEJAVU_SANS: &[u8] = include_bytes!("fixtures/fonts/DejaVuSans.ttf");

fn build_pdf_with_text(text: &str, font_size: f32) -> Vec<u8> {
    let mut writer = PdfWriter::new();
    let font = EmbeddedFont::from_data(Some("DejaVuSans".to_string()), DEJAVU_SANS.to_vec())
        .expect("DejaVuSans must parse");
    let resource_name = writer.register_embedded_font(font);

    let mut page = writer.add_letter_page();
    // Letter is 612×792 pt. Place text comfortably above the bottom margin.
    page.add_embedded_text(text, 72.0, 720.0, &resource_name, font_size);

    writer.finish().expect("writer must finish")
}

fn extract_text_from_bytes(bytes: Vec<u8>) -> String {
    let doc = PdfDocument::from_bytes(bytes).expect("PDF must re-open");
    // Single page in every fixture; concatenate just in case future fixtures
    // grow.
    let pages = doc.page_count().expect("page_count must succeed");
    let mut out = String::new();
    for i in 0..pages {
        out.push_str(&doc.extract_text(i).expect("extract_text must succeed"));
        out.push('\n');
    }
    out
}

#[test]
fn round_trip_ascii() {
    let bytes = build_pdf_with_text("Hello, World!", 12.0);
    let extracted = extract_text_from_bytes(bytes);
    assert!(
        extracted.contains("Hello, World!"),
        "expected 'Hello, World!' in extracted text, got: {extracted:?}",
    );
}

#[test]
fn round_trip_latin_extended() {
    // Accented Latin: tests that ToUnicode CMap covers codepoints above
    // 0x7f. DejaVuSans has full Latin Extended coverage.
    let input = "café déjà vu";
    let bytes = build_pdf_with_text(input, 14.0);
    let extracted = extract_text_from_bytes(bytes);
    assert!(
        extracted.contains(input),
        "expected {input:?} in extracted text, got: {extracted:?}",
    );
}

#[test]
fn round_trip_cyrillic() {
    // Cyrillic exercises the same Identity-H + ToUnicode path with a
    // codepoint range entirely above U+0400. DejaVuSans covers Cyrillic.
    let input = "Привет мир";
    let bytes = build_pdf_with_text(input, 14.0);
    let extracted = extract_text_from_bytes(bytes);
    assert!(
        extracted.contains(input),
        "expected Cyrillic {input:?} in extracted text, got: {extracted:?}",
    );
}

#[test]
fn round_trip_greek() {
    let input = "Καλημέρα κόσμε";
    let bytes = build_pdf_with_text(input, 14.0);
    let extracted = extract_text_from_bytes(bytes);
    assert!(
        extracted.contains(input),
        "expected Greek {input:?} in extracted text, got: {extracted:?}",
    );
}

#[test]
fn round_trip_hebrew() {
    // Hebrew is RTL but at this layer (no shaping) we just round-trip
    // the codepoints as-is. Phase LAYOUT will add proper BiDi.
    //
    // v0.3.54 (#537): the extractor now runs a geometric visual-vs-
    // logical detector over RTL runs. Our writer above draws Hebrew
    // glyphs in *input-stream order* at increasing x (no shaping),
    // which the detector correctly classifies as visual-order and
    // reverses to produce logical-order codepoints. The test
    // therefore accepts EITHER:
    //   * the original input (writer eventually implements proper
    //     RTL shaping → glyphs drawn right-to-left → extractor sees
    //     descending x → no reversal → byte-for-byte round trip), OR
    //   * the reversed input (current writer's no-shaping behaviour
    //     → extractor sees ascending x → reverses to logical order
    //     → round-trip produces the reversed codepoint sequence).
    // Both are correct given the writer's documented state.
    let input = "שלום עולם";
    let reversed: String = input.chars().rev().collect();
    let bytes = build_pdf_with_text(input, 14.0);
    let extracted = extract_text_from_bytes(bytes);
    assert!(
        extracted.contains(input) || extracted.contains(&reversed),
        "expected Hebrew {input:?} or its reverse {reversed:?} in extracted text, got: {extracted:?}",
    );
}

#[test]
fn pdf_validates_as_pdf_1_7() {
    // Sanity: produced bytes start with the PDF header and end with the
    // EOF marker.
    let bytes = build_pdf_with_text("Hi", 12.0);
    assert!(
        bytes.starts_with(b"%PDF-1.7"),
        "PDF must start with %PDF-1.7 header, got: {:?}",
        std::str::from_utf8(&bytes[..16]).unwrap_or("(non-utf8)"),
    );
    assert!(
        bytes.ends_with(b"%%EOF\n") || bytes.ends_with(b"%%EOF"),
        "PDF must end with %%EOF",
    );
    // The embedded font dict graph must be present.
    let utf8_lossy = String::from_utf8_lossy(&bytes);
    assert!(utf8_lossy.contains("/Subtype /Type0"));
    assert!(utf8_lossy.contains("/Subtype /CIDFontType2"));
    assert!(utf8_lossy.contains("/Encoding /Identity-H"));
    assert!(utf8_lossy.contains("/CIDToGIDMap /Identity"));
    assert!(utf8_lossy.contains("/Registry (Adobe)"));
    assert!(utf8_lossy.contains("/Ordering (Identity)"));
}

#[test]
fn unknown_font_resource_silently_noops() {
    // Caller passes a resource name we never registered. Should not
    // panic; the page just has no text.
    let mut writer = PdfWriter::new();
    let mut page = writer.add_letter_page();
    page.add_embedded_text("ignored", 100.0, 700.0, "EF999", 12.0);
    let bytes = writer.finish().expect("must finish despite unknown font");
    assert!(bytes.starts_with(b"%PDF-1.7"));
}
