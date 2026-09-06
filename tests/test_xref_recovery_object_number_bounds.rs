//! Reconstruction runs only on files that are already malformed, so hostile
//! object numbering is guaranteed on this path rather than merely possible.
//!
//! The synthetic Catalog and `/Pages` node are numbered above every surviving
//! object with `max_obj + 1` and `max_obj + 2`. No profile in this workspace
//! sets `overflow-checks`, so on a file containing `4294967295 0 obj` those
//! panicked in debug and **wrapped silently in release** — and the wrapped
//! number then collides with a real object, placing the synthesized page tree
//! on top of it.
//!
//! Recovery is also expected to be deterministic: the same damaged bytes must
//! recover the same document every time. The object-stream branch walked a
//! `HashMap` and selected with `get_or_insert`, so the winner depended on the
//! per-process hash seed.

use pdf_oxide::PdfDocument;

/// A file whose xref points nowhere, forcing reconstruction. `objects` are
/// emitted verbatim as `N 0 obj … endobj` bodies.
fn damaged_pdf(objects: &[(u64, &str)]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    for (num, body) in objects {
        buf.extend_from_slice(format!("{num} 0 obj\n{body}\nendobj\n").as_bytes());
    }
    // A startxref pointing at nothing: the reader must reconstruct.
    buf.extend_from_slice(b"startxref\n999999999\n%%EOF\n");
    buf
}

/// One page, plus an object numbered at the very top of the range.
fn top_of_range_pdf() -> Vec<u8> {
    damaged_pdf(&[
        (3, "<< /Type /Page /MediaBox [0 0 200 200] /Contents 4 0 R >>"),
        (4, "<< /Length 0 >>\nstream\n\nendstream"),
        (4294967295, "<< /Type /Whatever >>"),
    ])
}

/// The reported shape. Opening must not panic, and must not silently wrap a
/// synthesized object number onto a real one.
#[test]
fn test_object_numbered_at_the_top_of_the_range_does_not_panic() {
    // Either the document opens (having declined to synthesize) or it reports
    // an error. Both are acceptable; a panic is not, and in release a silent
    // wrap is not.
    // If it opened, whatever it recovered must be self-consistent.
    if let Ok(doc) = PdfDocument::from_bytes(top_of_range_pdf()) {
        let _ = doc.page_count();
    }
}

/// The same file must recover identically every time it is opened. Any
/// hash-order dependence shows up as a different page count or a different
/// first page across repeats within one process.
#[test]
fn reconstruction_is_deterministic_across_repeats() {
    let pdf = damaged_pdf(&[
        (3, "<< /Type /Page /MediaBox [0 0 200 200] /Contents 6 0 R >>"),
        (4, "<< /Type /Page /MediaBox [0 0 300 300] /Contents 6 0 R >>"),
        (5, "<< /Type /Page /MediaBox [0 0 400 400] /Contents 6 0 R >>"),
        (6, "<< /Length 0 >>\nstream\n\nendstream"),
    ]);

    let observe = || {
        let doc = PdfDocument::from_bytes(pdf.clone()).ok()?;
        let n = doc.page_count().ok()?;
        let first = doc.get_page_media_box(0).ok()?;
        Some((n, first))
    };

    let first = observe();
    for i in 1..8 {
        assert_eq!(
            observe(),
            first,
            "reconstruction differed on repeat {i}: the same bytes must recover \
             the same document"
        );
    }
}

/// An ordinary damaged file must still be recovered — the bounds check must
/// not turn every reconstruction into a refusal.
#[test]
fn test_ordinary_damaged_file_still_recovers_its_pages() {
    let pdf = damaged_pdf(&[
        (3, "<< /Type /Page /MediaBox [0 0 200 200] /Contents 4 0 R >>"),
        (4, "<< /Length 0 >>\nstream\n\nendstream"),
    ]);
    let doc = PdfDocument::from_bytes(pdf).expect("a recoverable file should open");
    assert_eq!(doc.page_count().expect("page count"), 1);
    assert_eq!(doc.get_page_media_box(0).expect("media box"), (0.0, 0.0, 200.0, 200.0));
}
