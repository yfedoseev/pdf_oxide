//! A font diagnostic is attributed to the document that raised it.
//!
//! The free-function warning sink is keyed by **thread**, not by document, and
//! the drain is first-caller-wins. So two documents opened in sequence on one
//! thread could take each other's warnings: parse A without draining, parse B,
//! and B's accessor reported A's diagnostics as its own.
//!
//! `FontInfo::from_dict` already receives `&PdfDocument`, so its two producers
//! — the Type 3 notice and the missing-`/ToUnicode` notice — can be attributed
//! directly without changing any signature. That matters because `from_dict`
//! is a `pub fn` and this crate ships bindings for fourteen languages, so
//! threading a sink parameter through it would be an API break.
//!
//! This does not close the leak for the three producers that hold no document
//! (`parse_stream_data`, the operator-cap notice, and the rasterizer's
//! dropped-glyph report); those still need the sink scoped to the open
//! document. It does close it for the two raised most often.

use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;

/// A Type0 font with no `/ToUnicode`, which raises `ToUnicodeMissing`.
fn pdf_with_a_tounicode_gap() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    macro_rules! obj {
        ($b:expr) => {{
            offsets.push(pdf.len());
            pdf.extend_from_slice($b);
        }};
    }
    pdf.extend_from_slice(b"%PDF-1.7\n");
    obj!(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    obj!(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    obj!(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    );

    let content = b"BT /F1 12 Tf 10 100 Td <0041> Tj ET";
    offsets.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    obj!(
        b"5 0 obj\n<< /Type /Font /Subtype /Type0 /BaseFont /Test-Identity \
           /Encoding /Identity-H /DescendantFonts [6 0 R] >>\nendobj\n"
    );
    obj!(
        b"6 0 obj\n<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test \
           /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
           /FontDescriptor 7 0 R /DW 1000 >>\nendobj\n"
    );
    obj!(
        b"7 0 obj\n<< /Type /FontDescriptor /FontName /Test /Flags 4 \
           /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200 \
           /CapHeight 700 /StemV 80 >>\nendobj\n"
    );

    let xref_offset = pdf.len();
    let n = offsets.len() + 1;
    let mut xref = format!("xref\n0 {n}\n0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.extend_from_slice(xref.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
            .as_bytes(),
    );
    pdf
}

/// The same page with an ordinary Standard-14 font, which raises nothing.
fn pdf_with_a_plain_font() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    macro_rules! obj {
        ($b:expr) => {{
            offsets.push(pdf.len());
            pdf.extend_from_slice($b);
        }};
    }
    pdf.extend_from_slice(b"%PDF-1.7\n");
    obj!(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    obj!(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    obj!(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    );

    let content = b"BT /F1 12 Tf 10 100 Td (hi) Tj ET";
    offsets.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    obj!(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");

    let xref_offset = pdf.len();
    let n = offsets.len() + 1;
    let mut xref = format!("xref\n0 {n}\n0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.extend_from_slice(xref.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
            .as_bytes(),
    );
    pdf
}

#[test]
fn test_font_warning_stays_with_the_document_that_raised_it() {
    // A is opened and read but never drained — the shape that leaked.
    let a = PdfDocument::from_bytes(pdf_with_a_tounicode_gap()).expect("open a");
    let _ = a.extract_text(0);

    let b = PdfDocument::from_bytes(pdf_with_a_plain_font()).expect("open b");
    let _ = b.extract_text(0);

    let a_cats: Vec<_> = a.structured_warnings().iter().map(|w| w.category).collect();
    assert!(
        a_cats.contains(&WarningCategory::ToUnicodeMissing),
        "precondition: the Type0 document must raise the diagnostic at all, \
         otherwise the assertion below passes vacuously; got {a_cats:?}"
    );

    let b_cats: Vec<_> = b.structured_warnings().iter().map(|w| w.category).collect();
    assert!(
        !b_cats.contains(&WarningCategory::ToUnicodeMissing),
        "the second document must not inherit the first document's font \
         diagnostic; got {b_cats:?}"
    );
}
