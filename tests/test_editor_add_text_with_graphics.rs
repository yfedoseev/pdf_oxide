//! `add_text` on an existing PDF must preserve existing graphics content.
//!
//! Verifies that editing a page with both text and graphics operators
//! (filled rectangles, etc.) correctly survives a round-trip through the
//! `DocumentEditor`, and that `select_pages` does not silently discard edits.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::elements::{FontSpec, TextContent, TextStyle};
use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::{DocumentBuilder, PageSize};

// ---------------------------------------------------------------------------
// Helper: build a source PDF that contains BOTH text and a filled rectangle.
// The rectangle is critical: it is a graphics operator that the
// HierarchicalExtractor cannot round-trip, so any path that replaces the
// original content stream will visibly lose it.
// ---------------------------------------------------------------------------
fn build_source_pdf_with_text_and_graphics() -> Vec<u8> {
    // Build a raw minimal PDF that has:
    //   - a filled grey rectangle (graphics)
    //   - a "Original text" label (text)
    // so we can verify both survive after add_text.
    let mut pdf = b"%PDF-1.7\n".to_vec();

    let off_catalog = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off_pages = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off_page = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n\
        << /Type /Page /Parent 2 0 R \
           /MediaBox [0 0 612 792] \
           /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R >> >> >>\n\
        endobj\n",
    );

    // Content stream: grey filled box + "Original text" label
    let content =
        b"0.8 g\n100 600 200 100 re f\n0 g\nBT /F1 14 Tf 110 640 Td (Original text) Tj ET";
    let off_content = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off_font = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n\
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
           /Encoding /WinAnsiEncoding >>\n\
        endobj\n",
    );

    // xref + trailer
    let xref_pos = pdf.len();
    let offsets = [
        0usize,
        off_catalog,
        off_pages,
        off_page,
        off_content,
        off_font,
    ];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

// ---------------------------------------------------------------------------
// Scenario 1: add_text then save – the exact case from the bug report.
// Expected: output PDF has BOTH original content AND the overlay text.
// ---------------------------------------------------------------------------
#[test]
fn test_add_text_on_existing_pdf_preserves_original_content() {
    let source = build_source_pdf_with_text_and_graphics();

    let mut editor = DocumentEditor::from_bytes(source).expect("open source PDF");

    let page_index: usize = 0;
    let mut page = editor.get_page(page_index).expect("get_page");

    let cx = page.width / 2.0;
    let cy = page.height / 2.0;
    let font_size = 24.0;
    let text = "hello world";
    let approx_width = text.len() as f32 * font_size * 0.5;
    let text_bbox =
        Rect::new(cx - approx_width / 2.0, cy - font_size / 2.0, approx_width, font_size);

    let content =
        TextContent::new(text, text_bbox, FontSpec::helvetica(font_size), TextStyle::new());
    page.add_text(content);

    editor.save_page(page).expect("save_page");
    assert!(editor.is_modified(), "editor must be marked modified after save_page");

    let output_bytes = editor.save_to_bytes().expect("save_to_bytes");

    // ── Re-open and inspect ────────────────────────────────────────────────
    let doc = PdfDocument::from_bytes(output_bytes.clone()).expect("re-open output");

    // Text extraction must find both the original label and the new text.
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all_text: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("Original text"),
        "original text must survive add_text; got: {:?}",
        all_text,
    );
    assert!(
        all_text.contains("hello world") || output_bytes.windows(11).any(|w| w == b"hello world"),
        "added text must be present in output; extracted: {:?}",
        all_text,
    );

    // The raw output bytes must contain the grey-box path operator from the
    // original content stream, confirming graphics were not discarded.
    assert!(
        output_bytes.windows(4).any(|w| w == b"re f"),
        "original rectangle fill operator 're f' must survive in output bytes",
    );
}

// ---------------------------------------------------------------------------
// Scenario 2: select_pages after save_page must not discard the overlay.
// The bug was that select_pages shifted page indices so modified_content
// lookups used the wrong key.
// ---------------------------------------------------------------------------
#[test]
fn test_add_text_then_select_pages_preserves_overlay() {
    // Build a 3-page document so we can select page 1 (middle) specifically.
    let mut builder = DocumentBuilder::new();
    {
        let p = builder.page(PageSize::Letter);
        p.at(72.0, 720.0).text("Page zero").done();
    }
    {
        let p = builder.page(PageSize::Letter);
        p.at(72.0, 720.0).text("Page one original").done();
    }
    {
        let p = builder.page(PageSize::Letter);
        p.at(72.0, 720.0).text("Page two").done();
    }
    let source = builder.build().expect("build source");

    let mut editor = DocumentEditor::from_bytes(source).expect("open source");

    let page_index: usize = 1;
    let mut page = editor.get_page(page_index).expect("get_page 1");

    let text_bbox = Rect::new(100.0, 400.0, 200.0, 30.0);
    let content =
        TextContent::new("overlay text", text_bbox, FontSpec::helvetica(14.0), TextStyle::new());
    page.add_text(content);
    editor.save_page(page).expect("save_page");

    // select_pages used to break the index mapping so modified content was lost
    editor.select_pages(&[page_index]).expect("select_pages");

    let output_bytes = editor.save_to_bytes().expect("save_to_bytes");

    // The single remaining page must contain the original text.
    let doc = PdfDocument::from_bytes(output_bytes.clone()).expect("re-open output");
    let spans = doc.extract_spans(0).expect("extract_spans page 0");
    let all_text: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("Page one original"),
        "original text on selected page must be preserved; got: {:?}",
        all_text,
    );

    // The overlay must also be present (raw byte check for the ASCII text or
    // extraction — both are acceptable evidence).
    let overlay_found =
        all_text.contains("overlay text") || output_bytes.windows(12).any(|w| w == b"overlay text");
    assert!(
        overlay_found,
        "overlay text must appear in output after select_pages; extracted: {:?}",
        all_text,
    );
}

// ---------------------------------------------------------------------------
// Scenario 3: saving a loaded page without adding anything must not
// corrupt the page (is_modified=true but content unchanged).
// ---------------------------------------------------------------------------
#[test]
fn test_save_page_without_adding_content_is_noop() {
    let source = build_source_pdf_with_text_and_graphics();
    let mut editor = DocumentEditor::from_bytes(source).expect("open source PDF");

    let page = editor.get_page(0).expect("get_page");
    editor.save_page(page).expect("save_page noop");
    assert!(editor.is_modified());

    let output_bytes = editor.save_to_bytes().expect("save_to_bytes");

    // Original text must still be readable.
    let doc = PdfDocument::from_bytes(output_bytes.clone()).expect("re-open");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all_text: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("Original text"),
        "original text must survive noop save; got: {:?}",
        all_text,
    );
    // Graphics operator must also survive.
    assert!(
        output_bytes.windows(4).any(|w| w == b"re f"),
        "original rectangle fill operator 're f' must survive noop save",
    );
}
