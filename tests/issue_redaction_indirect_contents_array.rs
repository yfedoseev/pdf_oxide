//! A page's `/Contents` may validly be an indirect reference to an object
//! that is itself an array of content-stream references (`/Contents 5 0 R`
//! where object 5 is `[6 0 R 7 0 R]`), not just a direct reference to a
//! single stream or a direct array of stream references. Redacting such a
//! page should work the same as any other.

use pdf_oxide::editor::DocumentEditor;

/// Build a minimal single-page PDF where `/Contents` is an indirect
/// reference (object 4) to an array of two content-stream references,
/// rather than a direct array or a direct stream reference.
fn pdf_with_indirect_contents_array() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 8];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");

    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 6 0 R >> >> /Contents 4 0 R >>",
    );
    // Object 4 is itself an array, reached only via an indirect reference.
    obj(&mut buf, &mut off, 4, "[5 0 R 7 0 R]");
    stream(&mut buf, &mut off, 5, b"BT /F1 24 Tf 72 700 Td (Top half) Tj ET");
    obj(&mut buf, &mut off, 6, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    stream(&mut buf, &mut off, 7, b"BT /F1 24 Tf 72 650 Td (Bottom half) Tj ET");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for id in 1..=7 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());

    buf
}

/// Open a PDF, redact a region on a page, apply the redaction — should
/// succeed regardless of how `/Contents` is structured internally.
#[test]
fn redaction_on_indirect_contents_array_does_not_error() {
    let bytes = pdf_with_indirect_contents_array();
    let mut editor = DocumentEditor::from_bytes(bytes).expect("should open");

    editor
        .add_redaction(0, [72.0, 690.0, 300.0, 710.0], None)
        .expect("should queue redaction");

    let result = editor.apply_redactions_destructive(Default::default());

    assert!(
        result.is_ok(),
        "apply_redactions_destructive failed on a page whose /Contents is an \
         indirect reference to an array of stream references: {:?}",
        result.err()
    );
}
