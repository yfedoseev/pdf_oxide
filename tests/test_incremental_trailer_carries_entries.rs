//! An incremental update's trailer must carry the previous trailer's entries.
//!
//! ISO 32000-1:2008 §7.5.6 (`docs/spec/pdf.md:3639`):
//!
//! > The added trailer shall contain **all the entries except the Prev entry
//! > (if present) from the previous trailer**, whether modified or not. In
//! > addition, the added trailer dictionary shall contain a **Prev** entry
//! > giving the location of the previous cross-reference section.
//!
//! Only `/Size`, `/Prev`, `/Root` and `/Info` were written, so every other
//! entry was dropped. `/ID` is the one that matters: Table 15 NOTE 2 warns that
//! its absence "might prevent the file from functioning in some workflows that
//! depend on files being uniquely identified", and it is required outright once
//! an `/Encrypt` entry is present.
//!
//! (An encrypted source is refused by the incremental path entirely, since
//! appending plaintext objects under a document key would corrupt them — so
//! `/Encrypt` itself cannot reach this trailer. `/ID` can and did.)

#![cfg(not(target_arch = "wasm32"))]

use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};

/// A one-page PDF whose trailer carries `/ID` and a custom entry, written to a
/// temp file so the incremental path can append to it.
///
/// Hand-built rather than produced by `DocumentBuilder`, because that writer
/// emits no `/ID` — so a fixture built with it would have nothing to carry and
/// would pass vacuously.
fn source_file() -> (tempfile::TempDir, std::path::PathBuf) {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    let content = b"BT /F1 12 Tf 72 720 Td (Original body) Tj ET\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    // /ID is the entry §7.5.6 requires the added trailer to carry across.
    buf.extend_from_slice(
        b"trailer\n<< /Size 6 /Root 1 0 R \
          /ID [<0123456789ABCDEF0123456789ABCDEF> <0123456789ABCDEF0123456789ABCDEF>] >>\n\
          startxref\n",
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("source.pdf");
    std::fs::write(&path, &buf).expect("write source");
    (dir, path)
}

/// The bytes appended after `original_len` — i.e. the incremental update.
fn appended_region(all: &[u8], original_len: usize) -> String {
    String::from_utf8_lossy(&all[original_len.min(all.len())..]).into_owned()
}

/// Save incrementally and return `(original_len, whole file)`.
fn save_incrementally() -> (usize, Vec<u8>) {
    let (dir, path) = source_file();
    let original_len = std::fs::metadata(&path).expect("stat").len() as usize;

    let mut editor = DocumentEditor::open(&path).expect("open");
    editor.set_title("Edited");
    editor
        .save_with_options(&path, SaveOptions::incremental())
        .expect("incremental save");

    let all = std::fs::read(&path).expect("read back");
    drop(dir);
    (original_len, all)
}

/// The source must actually carry an `/ID`, or this test proves nothing.
#[test]
fn test_source_trailer_has_an_id_to_carry() {
    let (_dir, path) = source_file();
    let bytes = std::fs::read(&path).expect("read");
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("/ID"),
        "fixture precondition: the source trailer should carry an /ID"
    );
}

/// The added trailer carries `/ID` across.
#[test]
fn test_added_trailer_carries_id() {
    let (original_len, all) = save_incrementally();
    let appended = appended_region(&all, original_len);
    assert!(appended.contains("trailer"), "expected an appended trailer:\n{appended}");
    assert!(
        appended.contains("/ID"),
        "the added trailer dropped /ID, which §7.5.6 says it shall carry:\n{appended}"
    );
}

/// The entries the update recomputes are still present and correct.
#[test]
fn test_added_trailer_still_has_size_prev_and_root() {
    let (original_len, all) = save_incrementally();
    let appended = appended_region(&all, original_len);
    for key in ["/Size", "/Prev", "/Root"] {
        assert!(appended.contains(key), "the added trailer is missing {key}:\n{appended}");
    }
}

/// `/Prev` must appear exactly once — carrying the previous trailer's entries
/// must not also copy its `/Prev`, which §7.5.6 excludes by name.
#[test]
fn prev_is_not_duplicated_from_the_previous_trailer() {
    let (original_len, all) = save_incrementally();
    let appended = appended_region(&all, original_len);
    assert_eq!(
        appended.matches("/Prev").count(),
        1,
        "/Prev should appear once in the added trailer:\n{appended}"
    );
}

/// The result must still parse and keep its content.
#[test]
fn test_incrementally_updated_file_still_reads() {
    let (_original_len, all) = save_incrementally();
    let doc = pdf_oxide::PdfDocument::from_bytes(all).expect("reopen the updated file");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("Original body"), "content lost: {text:?}");
}
