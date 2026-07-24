//! Regression tests for #940: `PdfDocument.save()`/`to_bytes()` silently
//! dropped DOM edits (`PdfPage::set_text`, `remove_element`) made to
//! pre-existing (source-loaded) content — the overlay-save path used for
//! such pages only serialized newly *added* elements, never edits/removals
//! of the original ones. See `PdfPage::pending_erasures` /
//! `DocumentEditor::save_page`.

use pdf_oxide::api::PdfBuilder;
use pdf_oxide::document::PdfDocument;
use pdf_oxide::editor::DocumentEditor;

fn contains(haystack: &[u8], needle: &str) -> bool {
    let needle = needle.as_bytes();
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn write_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn set_text_erasure_is_persisted_to_saved_bytes() {
    let mut pdf = PdfBuilder::new()
        .from_text("UNIQUE_MARKER_TOKEN_12345\n\nSecond line stays.")
        .unwrap();
    let input_bytes = pdf.to_bytes().unwrap();

    let mut editor = DocumentEditor::from_bytes(input_bytes).unwrap();
    let mut page = editor.get_page(0).unwrap();
    assert!(page.is_loaded_from_source());
    let target = page.children()[0].as_text().unwrap().id();

    page.set_text(target, "").unwrap();
    editor.save_page(page).unwrap();

    let out = editor.save_to_bytes().unwrap();
    assert!(
        !contains(&out, "UNIQUE_MARKER_TOKEN_12345"),
        "erased text must not survive in the saved bytes"
    );
    assert!(contains(&out, "Second line stays"), "unrelated content must be unaffected");

    let path = write_temp(&out, "issue_940_erase.pdf");
    let doc = PdfDocument::open(&path).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(!text.contains("UNIQUE_MARKER_TOKEN_12345"));
    assert!(text.contains("Second line stays"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn remove_element_is_persisted_to_saved_bytes() {
    let mut pdf = PdfBuilder::new()
        .from_text("REMOVE_ME_MARKER\n\nKeep this line.")
        .unwrap();
    let input_bytes = pdf.to_bytes().unwrap();

    let mut editor = DocumentEditor::from_bytes(input_bytes).unwrap();
    let mut page = editor.get_page(0).unwrap();
    let target = page.children()[0].as_text().unwrap().id();

    assert!(page.remove_element(target));
    editor.save_page(page).unwrap();

    let out = editor.save_to_bytes().unwrap();
    assert!(
        !contains(&out, "REMOVE_ME_MARKER"),
        "removed element must not survive in the saved bytes"
    );
    assert!(contains(&out, "Keep this line"));
}

#[test]
fn set_text_replacement_is_persisted_and_readable() {
    let mut pdf = PdfBuilder::new().from_text("OLD_TEXT_MARKER").unwrap();
    let input_bytes = pdf.to_bytes().unwrap();

    let mut editor = DocumentEditor::from_bytes(input_bytes).unwrap();
    let mut page = editor.get_page(0).unwrap();
    let target = page.children()[0].as_text().unwrap().id();

    page.set_text(target, "NEW_TEXT_MARKER").unwrap();
    editor.save_page(page).unwrap();

    let out = editor.save_to_bytes().unwrap();
    assert!(!contains(&out, "OLD_TEXT_MARKER"));
    assert!(contains(&out, "NEW_TEXT_MARKER"));

    let path = write_temp(&out, "issue_940_replace.pdf");
    let doc = PdfDocument::open(&path).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(!text.contains("OLD_TEXT_MARKER"));
    assert!(text.contains("NEW_TEXT_MARKER"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn in_memory_children_reflect_edit_immediately_after_set_text() {
    // The in-memory DOM view was never the broken part (#940) — only the
    // saved bytes were. Guard against a fix that regresses this.
    let mut pdf = PdfBuilder::new().from_text("SOME_MARKER").unwrap();
    let input_bytes = pdf.to_bytes().unwrap();

    let mut editor = DocumentEditor::from_bytes(input_bytes).unwrap();
    let mut page = editor.get_page(0).unwrap();
    let target = page.children()[0].as_text().unwrap().id();

    page.set_text(target, "").unwrap();
    let text = page.get_element(target).unwrap();
    assert_eq!(text.as_text().unwrap().text(), "");
}
