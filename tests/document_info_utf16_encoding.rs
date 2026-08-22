//! `/Info` dictionary text strings (`/Title`, `/Author`, ...) are PDF text
//! strings per ISO 32000-1:2008 §7.9.2.2 — UTF-16BE with a `FE FF` byte-order
//! mark, or PDFDocEncoding when unprefixed. They are never raw UTF-8.
//! `DocumentInfo::from_object` used `String::from_utf8_lossy` directly on
//! the string bytes, which mangles every non-ASCII UTF-16BE-encoded value
//! into replacement characters (each source character is 2 bytes, almost
//! none of which form valid UTF-8 sequences).
//!
//! The fixture is a synthetic PDF whose `/Info /Title` is the Cyrillic word
//! "Привет" (Russian for "hello"), UTF-16BE-encoded with a BOM — a value
//! that has no valid interpretation as UTF-8 at all, so any UTF-8-based
//! decode reliably corrupts it.

use pdf_oxide::editor::{DocumentEditor, EditableDocument};

const TITLE: &str = "Привет";

fn utf16be_bom(s: &str) -> Vec<u8> {
    let mut bytes = vec![0xFE, 0xFF];
    for unit in s.encode_utf16() {
        bytes.extend_from_slice(&unit.to_be_bytes());
    }
    bytes
}

/// Escape raw bytes into a PDF string literal `(...)` body — only `(`, `)`,
/// and `\` need a backslash per ISO 32000-1:2008 §7.3.4.2.
fn escape_pdf_string_literal(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 2);
    out.push(b'(');
    for &b in bytes {
        if b == b'(' || b == b')' || b == b'\\' {
            out.push(b'\\');
        }
        out.push(b);
    }
    out.push(b')');
    out
}

fn build_pdf_with_utf16_title() -> Vec<u8> {
    let title_literal = escape_pdf_string_literal(&utf16be_bom(TITLE));

    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = vec![0usize];

    let off1 = pdf.len();
    offsets.push(off1);
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    offsets.push(off2);
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    offsets.push(off3);
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>\nendobj\n",
    );

    let off4 = pdf.len();
    offsets.push(off4);
    let content = b"";
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    offsets.push(off5);
    pdf.extend_from_slice(b"5 0 obj\n<< /Title ");
    pdf.extend_from_slice(&title_literal);
    pdf.extend_from_slice(b" >>\nendobj\n");

    let xref_offset = pdf.len();
    let count = offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{}\n%%EOF\n",
            count, xref_offset
        )
        .as_bytes(),
    );
    pdf
}

#[test]
fn info_title_decodes_utf16be_bom_correctly() {
    let mut editor = DocumentEditor::from_bytes(build_pdf_with_utf16_title()).expect("open pdf");
    let info = editor.get_info().expect("get_info");

    assert_eq!(
        info.title.as_deref(),
        Some(TITLE),
        "UTF-16BE /Title with BOM must decode to the original Unicode string, got {:?}",
        info.title
    );
}
