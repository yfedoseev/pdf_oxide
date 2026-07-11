//! Tests for `remove_headers` page-number
//! detector, which uses two purely structural signals — position (top or
//! bottom margin band) and shape (a short number, or a number that
//! varies page to page) — with no concept of "page number" beyond that.

use pdf_oxide::PdfDocument;

// ---------------- test helper: build_pdf_with_page_extras -------------------
//
// two fn used only by build_pdf_with_page_extras write one object each,
// recording its offset as they go:
// - `buf`      the buffer we're writing into
// - `off[id]`  start of object definition

// -- write a plain dictionary object --
fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

// -- write a `stream` object - used here for page content --
fn stream(buf: &mut Vec<u8>, off: &mut [usize], id: usize, data: &[u8]) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

/// Minimal single-page-content PDF builder: N pages, each with a body
/// paragraph plus arbitrary extra content-stream text supplied per page.
fn build_pdf_with_page_extras(
    page_count: usize,
    extra_per_page: impl Fn(usize) -> String,
) -> Vec<u8> {
    // buffer for the PDF we're building
    let mut buf: Vec<u8> = Vec::new();

    // `off[N]` = byte offset where object N's bytes start, filled in as
    // each object is written below. `xref_off` (further down) separately
    // records where the xref table itself starts.
    let mut off = vec![0usize; 4 + page_count * 2];

    // PDF File header
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    // Catalog
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");

    // Pages tree root
    // Build the /Kids array value ahead of time: "5 0 R 7 0 R 9 0 R ..."
    // — one indirect reference per page object we're about to create.
    // (Page objects are 5, 7, 9, ... because each page also needs a
    // content-stream object right before it: 4, 6, 8, ... — see the loop
    // below.)
    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );

    // Font resource
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    // --- One content stream + one page object, per page ---
    for i in 0..page_count {
        let content_id = 4 + i * 2; // 4, 6, 8, ...
        let page_id = 5 + i * 2; // 5, 7, 9, ...

        // text object per page + whatever the test wants to insert
        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n{}",
            extra_per_page(i)
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());

        // Page object:
        // physical size - `/MediaBox`, in points `[0 0 612 792]` is US Letter
        // resources it can reference by name - `/Resources`
        // -  just our one font as `/F1`
        // object w/ drawing instructions `/Contents` with content-stream object
        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    // Cross-reference table - `xref_off`
    // flat table mapping object number -> byte offset
    // records where THIS table itself starts (needed for the trailer).
    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());

    // fixed, required first entry marking object 0 as "free"
    buf.extend_from_slice(b"0000000000 65535 f \n");

    // one `NNNNNNNNNN 00000 n` line per real object, `n` meaning "in use",
    // giving its 10-digit zero-padded byte offset
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    // Trailer
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// The simplest, most basic case: a bare page number ("1", "2", "3", ...)
/// alone on its own line in the header band, changing every page — the
/// textbook definition of a page number. `is_bare_page_number_text` marks
/// this unconditionally per page (isolated + short + digits-only), with
/// no cross-page signature lookup and no first-occurrence exemption
/// involved, so this is core, expected behavior that should already work
/// correctly on `main`.
#[test]
fn remove_headers_removes_simple_page_number() {
    let bytes = build_pdf_with_page_extras(5, |i| {
        format!("BT /F1 10 Tf 1 0 0 1 300 760 Tm ({}) Tj ET\n", i + 1)
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_headers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            text.contains("Body text placeholder"),
            "page {page}: body wrongly removed: {text:?}"
        );
        let page_number = format!("{}", page + 1);
        assert!(
            !text.contains(&page_number),
            "page {page}: page number {page_number:?} should have been removed: {text:?}"
        );
    }
}

/// One step up from a bare digit: "page 1", "page 2", ... — not digits-only
/// (it has letters too), so `is_bare_page_number_text` doesn't apply at
/// all. This has to go through the cross-page signature detector instead:
/// `normalize_artifact_signature` turns "page 1" into "page #", sees that
/// shape recur across pages with the digit changing, and flags it as a
/// page number that way.
#[test]
fn remove_headers_removes_word_plus_page_number() {
    let bytes = build_pdf_with_page_extras(5, |i| {
        format!("BT /F1 10 Tf 1 0 0 1 280 760 Tm (page {}) Tj ET\n", i + 1)
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_headers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            text.contains("Body text placeholder"),
            "page {page}: body wrongly removed: {text:?}"
        );
    }

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        let marker = format!("page {}", page + 1);
        assert!(
            !text.contains(&marker),
            "page {page}: marker {marker:?} should have been removed: {text:?}"
        );
    }
}

#[test]
fn remove_headers_with_text_and_pagenum_keeps_top_line_of_text() {
    let sentences = [
        "here—how, bowed down by the weight of the subject which you have laid upon my",
        "than gravel, no very great harm was done. The only charge I could bring against the",
        "white wings, a deprecating, silvery, kindly gentleman, who regretted in a low voice as he",
        "skittles presumably of an evening. An unending stream of gold and silver, I thought,",
        "flushed crimson; had been emptied; had been filled. And thus by degrees was lit",
    ];

    let bytes = build_pdf_with_page_extras(5, |i| {
        format!("BT /F1 10 Tf 1 0 0 1 300 760 Tm (Page {}) Tj ET\nBT /F1 10 Tf 1 0 0 1 300 725 Tm ({}) Tj ET\n", i + 1, sentences[i])
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_headers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        assert!(text.contains(sentences[page]), "page {page}: body wrongly removed: {text:?}");
        let page_number = format!("{}", page + 1);
        assert!(
            !text.contains(&page_number),
            "page {page}: page number {page_number:?} should have been removed: {text:?}"
        );
    }
}
