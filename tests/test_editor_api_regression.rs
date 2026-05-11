//! Comprehensive regression tests for the DocumentEditor API.
//!
//! Tests are organized into groups:
//!   1. #483 overlay: add_text / add_path preserves original content
//!   2. select_pages-first ordering: modification after select_pages uses correct page
//!   3. erase_region combined with select_pages
//!   4. set_page_rotation combined with select_pages
//!   5. set_page_media_box / set_page_crop_box combined with select_pages
//!   6. Multiple pages edited in one document
//!   7. add_image / add_path overlay on existing PDFs
//!   8. Real-PDF round-trips using files from ~/projects/pdf_oxide_tests/irs/

use pdf_oxide::document::PdfDocument;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::elements::{FontSpec, TextContent, TextStyle};
use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::{DocumentBuilder, PageSize};

// ── Test PDF builders ────────────────────────────────────────────────────────

/// Build a minimal single-page PDF containing text and a filled grey rectangle.
fn single_page_pdf_with_content() -> Vec<u8> {
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

/// Build an N-page document using DocumentBuilder.  Each page has a unique text label.
fn multi_page_pdf(page_labels: &[&str]) -> Vec<u8> {
    let mut builder = DocumentBuilder::new();
    for label in page_labels {
        let p = builder.page(PageSize::Letter);
        p.at(72.0, 720.0).text(label).done();
    }
    builder.build().expect("build multi-page PDF")
}

fn add_text_overlay(page: &mut pdf_oxide::editor::dom::PdfPage, text: &str) {
    let cx = page.width / 2.0;
    let cy = page.height / 2.0;
    let font_size = 18.0;
    let approx_width = text.len() as f32 * font_size * 0.5;
    let bbox = Rect::new(cx - approx_width / 2.0, cy - font_size / 2.0, approx_width, font_size);
    page.add_text(TextContent::new(text, bbox, FontSpec::helvetica(font_size), TextStyle::new()));
}

// ── Group 1: overlay preserves original content ──────────────────────────────

#[test]
fn overlay_add_text_preserves_original_text_and_graphics() {
    let source = single_page_pdf_with_content();
    let mut editor = DocumentEditor::from_bytes(source).expect("open PDF");
    let mut page = editor.get_page(0).expect("get_page");
    add_text_overlay(&mut page, "overlay text");
    editor.save_page(page).expect("save_page");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    // Original rectangle operator must survive
    assert!(
        bytes.windows(4).any(|w| w == b"re f"),
        "graphics operator 're f' lost after add_text"
    );

    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("Original text"), "original text lost after add_text; got: {all:?}");
    let overlay_present =
        all.contains("overlay text") || bytes.windows(12).any(|w| w == b"overlay text");
    assert!(overlay_present, "overlay text not found in output; got: {all:?}");
}

#[test]
fn overlay_add_path_preserves_original_content() {
    use pdf_oxide::elements::{PathContent, PathOperation};

    let source = single_page_pdf_with_content();
    let mut editor = DocumentEditor::from_bytes(source).expect("open PDF");
    let mut page = editor.get_page(0).expect("get_page");

    // Add a simple line path as overlay
    let path = PathContent::from_operations(vec![
        PathOperation::MoveTo(50.0, 50.0),
        PathOperation::LineTo(150.0, 150.0),
    ]);
    page.add_path(path);
    editor.save_page(page).expect("save_page");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    assert!(
        bytes.windows(4).any(|w| w == b"re f"),
        "original rectangle fill lost after add_path"
    );

    let doc = PdfDocument::from_bytes(bytes).expect("reopen");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("Original text"), "original text lost after add_path; got: {all:?}");
}

// ── Group 2: select_pages-first ordering ─────────────────────────────────────

/// add_text overlay correctly applied when select_pages is called BEFORE get_page/save_page.
#[test]
fn overlay_add_text_after_select_pages() {
    // 3-page doc; select middle page; then add overlay
    let source = multi_page_pdf(&["Page zero", "Page one", "Page two"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // select_pages FIRST — previously this broke the index mapping
    editor.select_pages(&[1]).expect("select_pages");

    let mut page = editor.get_page(0).expect("get_page after select_pages");
    add_text_overlay(&mut page, "post-select overlay");
    editor.save_page(page).expect("save_page");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(all.contains("Page one"), "selected page original text lost; got: {all:?}");
    let overlay_present = all.contains("post-select overlay")
        || bytes
            .windows(20)
            .any(|w| w == b"post-select overlay\n".get(..w.len()).unwrap_or(&[]));
    let overlay_raw = bytes.windows(19).any(|w| w == b"post-select overlay");
    assert!(
        overlay_present || overlay_raw,
        "overlay not found after select_pages-first add_text; got: {all:?}"
    );
}

/// Variant: select_pages keeps page 2 (index 2), add_text should land on it.
#[test]
fn overlay_add_text_after_select_pages_last_page() {
    let source = multi_page_pdf(&["First", "Second", "Third"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    editor.select_pages(&[2]).expect("select_pages");

    let mut page = editor.get_page(0).expect("get_page");
    add_text_overlay(&mut page, "third-page-overlay");
    editor.save_page(page).expect("save_page");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(all.contains("Third"), "original text of selected page not found; got: {all:?}");
    let has_overlay =
        all.contains("third-page-overlay") || bytes.windows(18).any(|w| w == b"third-page-overlay");
    assert!(has_overlay, "overlay not found; got: {all:?}");
}

// ── Group 3: erase_region with select_pages ──────────────────────────────────

/// erase_region called AFTER select_pages should affect the correct (selected) page.
#[test]
fn erase_region_after_select_pages() {
    let source = multi_page_pdf(&["First page text", "Second page text", "Third page text"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // Keep page 1 only, then erase on the now-only output page 0
    editor.select_pages(&[1]).expect("select_pages");
    editor
        .erase_region(0, [0.0, 0.0, 612.0, 792.0])
        .expect("erase_region");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    // The erase overlay should produce a white rectangle — "1 1 1 rg" is its fill operator
    assert!(
        bytes.windows(9).any(|w| w == b"1 1 1 rg\n"),
        "erase overlay not applied after select_pages-first erase_region"
    );
}

/// erase_region called BEFORE select_pages (traditional order) must still work.
#[test]
fn erase_region_before_select_pages() {
    let source = multi_page_pdf(&["Alpha", "Beta", "Gamma"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // Erase on source page 1 (output index 1)
    editor
        .erase_region(1, [0.0, 0.0, 612.0, 792.0])
        .expect("erase_region");
    // Then keep only page 1
    editor.select_pages(&[1]).expect("select_pages");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    assert!(
        bytes.windows(9).any(|w| w == b"1 1 1 rg\n"),
        "erase overlay lost when select_pages called after erase_region"
    );
}

// ── Group 4: set_page_rotation with select_pages ─────────────────────────────

/// Rotation set AFTER select_pages must be applied to the correct page.
#[test]
fn set_page_rotation_after_select_pages() {
    let source = multi_page_pdf(&["P0", "P1", "P2"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    editor.select_pages(&[1]).expect("select_pages");
    editor.set_page_rotation(0, 90).expect("set_page_rotation");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    // /Rotate 90 must appear in the output PDF bytes
    assert!(
        bytes.windows(10).any(|w| w == b"/Rotate 90"),
        "/Rotate 90 not found in output after select_pages-first rotation"
    );
}

/// Rotation set BEFORE select_pages (traditional order) must survive.
#[test]
fn set_page_rotation_before_select_pages() {
    let source = multi_page_pdf(&["P0", "P1", "P2"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // Set rotation on output page 1 (which is source page 1)
    editor.set_page_rotation(1, 180).expect("set_page_rotation");
    editor.select_pages(&[1]).expect("select_pages");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    assert!(
        bytes.windows(11).any(|w| w == b"/Rotate 180"),
        "/Rotate 180 lost when select_pages called after set_page_rotation"
    );
}

/// get_page_rotation must reflect set_page_rotation after select_pages.
#[test]
fn get_page_rotation_consistent_after_select_pages() {
    let source = multi_page_pdf(&["P0", "P1", "P2"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    editor.select_pages(&[2]).expect("select_pages");
    editor.set_page_rotation(0, 270).expect("set_page_rotation");

    let rotation = editor.get_page_rotation(0).expect("get_page_rotation");
    assert_eq!(
        rotation, 270,
        "get_page_rotation did not reflect set_page_rotation after select_pages"
    );
}

// ── Group 5: set_page_media_box / set_page_crop_box with select_pages ─────────

#[test]
fn set_page_media_box_after_select_pages() {
    let source = multi_page_pdf(&["P0", "P1", "P2"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    editor.select_pages(&[1]).expect("select_pages");
    editor
        .set_page_media_box(0, [0.0, 0.0, 400.0, 600.0])
        .expect("set_page_media_box");

    let mb = editor.get_page_media_box(0).expect("get_page_media_box");
    assert_eq!(
        mb,
        [0.0, 0.0, 400.0, 600.0],
        "media_box not reflected by getter after select_pages"
    );

    let bytes = editor.save_to_bytes().expect("save_to_bytes");
    // /MediaBox [0 0 400 600] should appear
    assert!(bytes.windows(8).any(|w| w == b"MediaBox"), "/MediaBox not written to output");
}

#[test]
fn set_page_crop_box_after_select_pages() {
    let source = multi_page_pdf(&["P0", "P1", "P2"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    editor.select_pages(&[0]).expect("select_pages");
    editor
        .set_page_crop_box(0, [10.0, 10.0, 590.0, 780.0])
        .expect("set_page_crop_box");

    let cb = editor.get_page_crop_box(0).expect("get_page_crop_box");
    assert_eq!(
        cb,
        Some([10.0, 10.0, 590.0, 780.0]),
        "crop_box not reflected by getter after select_pages"
    );
}

// ── Group 6: Multiple pages edited in one document ───────────────────────────

/// Edit three pages independently in one editor session.
#[test]
fn multiple_pages_with_overlays() {
    let source = multi_page_pdf(&["Alpha", "Beta", "Gamma"]);
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    for i in 0..3 {
        let mut page = editor.get_page(i).expect("get_page");
        add_text_overlay(&mut page, &format!("overlay-{i}"));
        editor.save_page(page).expect("save_page");
    }

    let bytes = editor.save_to_bytes().expect("save_to_bytes");
    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");

    for i in 0..3 {
        let spans = doc.extract_spans(i).expect("extract_spans");
        let all: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let label = ["Alpha", "Beta", "Gamma"][i];
        assert!(all.contains(label), "original text '{label}' lost on page {i}; got: {all:?}");
    }

    // All three overlays must be present somewhere in the output bytes
    for i in 0..3 {
        let tag = format!("overlay-{i}");
        assert!(
            bytes.windows(tag.len()).any(|w| w == tag.as_bytes()),
            "overlay '{tag}' not found in output bytes"
        );
    }
}

/// Same page saved twice: second save must not corrupt the page.
#[test]
fn save_page_twice_second_wins() {
    let source = single_page_pdf_with_content();
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // First save: add "first overlay"
    {
        let mut page = editor.get_page(0).expect("get_page 1");
        add_text_overlay(&mut page, "first overlay");
        editor.save_page(page).expect("save_page 1");
    }
    // Second save: add "second overlay" (re-loads the page)
    {
        let mut page = editor.get_page(0).expect("get_page 2");
        add_text_overlay(&mut page, "second overlay");
        editor.save_page(page).expect("save_page 2");
    }

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    // Original content must still be present
    assert!(
        bytes.windows(4).any(|w| w == b"re f"),
        "original rectangle lost after two consecutive save_page calls"
    );

    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let all: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all.contains("Original text"),
        "original text lost after two save_page; got: {all:?}"
    );
}

// ── Group 7: erase_region + add_text combo on same page ──────────────────────

#[test]
fn erase_then_add_text_on_existing_page() {
    let source = single_page_pdf_with_content();
    let mut editor = DocumentEditor::from_bytes(source).expect("open");

    // Erase the area where the original grey box is
    editor
        .erase_region(0, [100.0, 600.0, 200.0, 100.0])
        .expect("erase_region");

    // Also add new text
    let mut page = editor.get_page(0).expect("get_page");
    add_text_overlay(&mut page, "replacement text");
    editor.save_page(page).expect("save_page");

    let bytes = editor.save_to_bytes().expect("save_to_bytes");

    // Erase overlay (white rect) must be present
    assert!(
        bytes.windows(9).any(|w| w == b"1 1 1 rg\n"),
        "erase overlay not found in output"
    );
    // New text must be present
    let has_text = bytes.windows(16).any(|w| w == b"replacement text");
    assert!(has_text, "replacement text not found in output");
}

// ── Group 8: Real-PDF round-trips ────────────────────────────────────────────

/// Round-trip an IRS form PDF without modifications — it must remain openable.
#[test]
fn real_pdf_fw2_noop_roundtrip() {
    let path = std::path::Path::new("/home/yfedoseev/projects/pdf_oxide_tests/irs/fw2_2024.pdf");
    if !path.exists() {
        eprintln!("skipping real-PDF test: {path:?} not found");
        return;
    }

    let bytes = std::fs::read(path).expect("read fw2_2024.pdf");
    let mut editor = DocumentEditor::from_bytes(bytes).expect("open fw2_2024");

    let output = editor.save_to_bytes().expect("save fw2_2024");

    // Must be a valid PDF that can be re-opened
    PdfDocument::from_bytes(output).expect("re-open fw2_2024 output");
}

/// Add text overlay to IRS form page 0 — original form content must survive.
#[test]
fn real_pdf_fw2_add_text_overlay() {
    let path = std::path::Path::new("/home/yfedoseev/projects/pdf_oxide_tests/irs/fw2_2024.pdf");
    if !path.exists() {
        eprintln!("skipping real-PDF test: {path:?} not found");
        return;
    }

    let bytes = std::fs::read(path).expect("read fw2_2024.pdf");
    let mut editor = DocumentEditor::from_bytes(bytes).expect("open fw2_2024");

    let mut page = editor.get_page(0).expect("get_page 0");
    add_text_overlay(&mut page, "test annotation");
    editor.save_page(page).expect("save_page");

    let output = editor.save_to_bytes().expect("save fw2_2024 with overlay");

    // Must be a valid PDF
    let doc = PdfDocument::from_bytes(output.clone()).expect("re-open with overlay");

    // Annotation text must be present in raw bytes
    let has_annotation = output.windows(15).any(|w| w == b"test annotation");
    assert!(has_annotation, "overlay text 'test annotation' not found in output bytes");

    // The document must still have page content (non-empty spans or at least a page)
    assert!(doc.page_count().unwrap_or(0) >= 1, "output document has no pages");
}

/// select_pages on a real multi-page PDF preserves the selected page.
#[test]
fn real_pdf_select_pages_preserves_content() {
    let path = std::path::Path::new("/home/yfedoseev/projects/pdf_oxide_tests/irs/fw2_2024.pdf");
    if !path.exists() {
        eprintln!("skipping real-PDF test: {path:?} not found");
        return;
    }

    let bytes = std::fs::read(path).expect("read fw2_2024.pdf");
    let orig_doc = PdfDocument::from_bytes(bytes.clone()).expect("open original");
    let page_count = orig_doc.page_count().unwrap_or(0);
    drop(orig_doc);

    if page_count < 2 {
        eprintln!("skipping: fw2_2024.pdf has only {page_count} page(s)");
        return;
    }

    let mut editor = DocumentEditor::from_bytes(bytes).expect("open fw2_2024");
    editor.select_pages(&[0]).expect("select_pages");

    let output = editor.save_to_bytes().expect("save after select_pages");
    let doc = PdfDocument::from_bytes(output).expect("re-open after select_pages");

    assert_eq!(doc.page_count().unwrap_or(0), 1, "expected exactly 1 page after select_pages");
}

/// select_pages-then-add_text on a real PDF: overlay must land on the right page.
#[test]
fn real_pdf_select_pages_then_add_text() {
    let path = std::path::Path::new("/home/yfedoseev/projects/pdf_oxide_tests/irs/fw2_2024.pdf");
    if !path.exists() {
        eprintln!("skipping real-PDF test: {path:?} not found");
        return;
    }

    let bytes = std::fs::read(path).expect("read fw2_2024.pdf");
    let mut editor = DocumentEditor::from_bytes(bytes).expect("open fw2_2024");

    editor.select_pages(&[0]).expect("select_pages");

    let mut page = editor.get_page(0).expect("get_page after select_pages");
    add_text_overlay(&mut page, "post-select-fw2-overlay");
    editor.save_page(page).expect("save_page");

    let output = editor.save_to_bytes().expect("save_to_bytes");

    let has_overlay = output.windows(23).any(|w| w == b"post-select-fw2-overlay");
    assert!(has_overlay, "overlay text not found in real-PDF output after select_pages");
}
