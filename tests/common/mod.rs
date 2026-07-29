//! Shared PDF-building helpers for integration tests.

// two fn for building common PDF objects -- they write one object each,
// recording its offset as they go:
// - `buf`      the buffer we're writing into
// - `off[id]`  start of object definition

// -- write a plain dictionary object --
pub fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

// -- write a `stream` object - used here for page content --
pub fn stream(buf: &mut Vec<u8>, off: &mut [usize], id: usize, data: &[u8]) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

/// Minimal single-page-content PDF builder: N pages, each with a body
/// paragraph plus arbitrary extra content-stream text supplied per page.
pub fn build_pdf_with_page_extras(
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
