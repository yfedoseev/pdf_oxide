//! Regression tests for #940: `PdfDocument.save()`/`to_bytes()` silently
//! dropped DOM edits (`PdfPage::set_text`, `remove_element`) made to
//! pre-existing (source-loaded) content — the overlay-save path used for
//! such pages only serialized newly *added* elements, never edits/removals
//! of the original ones. See `PdfPage::pending_erasures` /
//! `DocumentEditor::save_page`.
//!
//! Also covers a `/Contents` array-merge bug found while fixing #940: when a
//! page's *original* `/Contents` was already a multi-entry array (ISO
//! 32000-1 §7.7.3.3, common in real-world PDFs) and `save_page` had to both
//! append an overlay-additions stream and apply a destructive redaction on
//! the same save, rewriting only the array's first entry left the other
//! original entries in place — each of which is unconditionally dropped as
//! an orphan by the redaction write-skip, so they end up as dangling
//! references. Lenient readers (including pdf_oxide's own) tolerate this
//! silently; strict validators reject it. Same defect class independently
//! diagnosed against `add_text()` + `apply_redactions_destructive()` in #799
//! / tracked in #941.

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

fn finalize_pdf(pdf: &mut Vec<u8>, obj_offsets: &[usize]) {
    let xref_offset = pdf.len();
    let count = obj_offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &obj_offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        count, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
}

/// A single page whose `/Contents` is already a 2-entry array `[4 0 R 6 0 R]`
/// — legal per ISO 32000-1 §7.7.3.3 and common in real-world producers.
fn build_multi_stream_contents_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents [4 0 R 6 0 R] /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    );
    let off4 = pdf.len();
    let content1 = b"BT /F1 12 Tf 72 720 Td (STREAM_ONE_MARKER) Tj ET";
    pdf.extend_from_slice(
        format!("4 0 obj\n<< /Length {} >>\nstream\n", content1.len()).as_bytes(),
    );
    pdf.extend_from_slice(content1);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );
    let off6 = pdf.len();
    let content2 = b"BT /F1 12 Tf 72 700 Td (STREAM_TWO_MARKER) Tj ET";
    pdf.extend_from_slice(
        format!("6 0 obj\n<< /Length {} >>\nstream\n", content2.len()).as_bytes(),
    );
    pdf.extend_from_slice(content2);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(&mut pdf, &[0, off1, off2, off3, off4, off5, off6]);
    pdf
}

/// Every object id referenced by `/Contents [.. N 0 R ..]` (or a bare
/// `/Contents N 0 R`) must exist as an `N 0 obj` somewhere in the file —
/// otherwise the array holds a dangling reference.
fn assert_no_dangling_contents_refs(pdf_bytes: &[u8]) {
    let text = String::from_utf8_lossy(pdf_bytes);
    let start = text.find("/Contents").expect("no /Contents entry found");
    let after = &text[start + "/Contents".len()..];
    let end = after.find(['/', '>']).unwrap_or(after.len());
    let refs_section = &after[..end];
    let mut ids = Vec::new();
    let tokens: Vec<&str> = refs_section.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        if tokens.get(i + 1) == Some(&"0") && tokens.get(i + 2) == Some(&"R") {
            if let Ok(id) = tokens[i].trim_start_matches('[').parse::<u32>() {
                ids.push(id);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    assert!(!ids.is_empty(), "expected at least one object reference in /Contents");
    for id in ids {
        let marker = format!("{} 0 obj", id);
        assert!(
            contains(pdf_bytes, &marker),
            "/Contents references object {id} 0 R, but no `{id} 0 obj` exists — dangling reference"
        );
    }
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
fn set_text_replacement_on_multi_stream_page_leaves_no_dangling_contents_ref() {
    let input = build_multi_stream_contents_pdf();
    let mut editor = DocumentEditor::from_bytes(input).unwrap();
    let mut page = editor.get_page(0).unwrap();
    let children = page.children();
    assert_eq!(children.len(), 2);
    let target = children[0].as_text().unwrap().id();

    // Non-empty replacement text on an original element triggers both the
    // overlay-additions path (for the new text) and destructive redaction
    // (to remove the old text) in the same save — the exact combination
    // that exposed the /Contents merge bug.
    page.set_text(target, "REPLACEMENT_TEXT").unwrap();
    editor.save_page(page).unwrap();

    let out = editor.save_to_bytes().unwrap();
    assert_no_dangling_contents_refs(&out);

    let path = write_temp(&out, "multi_stream_contents_replace.pdf");
    let doc = PdfDocument::open(&path).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(!text.contains("STREAM_ONE_MARKER"));
    assert_eq!(
        text.matches("STREAM_TWO_MARKER").count(),
        1,
        "the untouched original stream must survive exactly once, not be duplicated or dropped"
    );
    assert!(text.contains("REPLACEMENT_TEXT"));
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
