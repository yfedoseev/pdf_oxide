//! Both structured-warning accessors report the same warnings.
//!
//! `structured_warnings()` and `take_structured_warnings()` differ only in
//! whether the sink is left populated afterwards. They must not differ in
//! *which* warnings they can see.
//!
//! Five of the nine producers write only to the free-function (thread-local)
//! sink rather than to the per-document one: every parser `SpecViolation`, the
//! operator-cap truncation, the Type 3 and missing-`/ToUnicode` font
//! diagnostics, and the dropped-glyph report. The draining accessor did not
//! drain that sink, so a caller using it observed none of them — and that is
//! the accessor `pdf_document_take_structured_warnings` exposes, so the gap
//! reached every language binding.
//!
//! The existing round-trip test passed over this because it exercises a
//! `NoTextLayer` warning, which is raised on the per-document sink.

use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;

/// A PDF whose stream keyword is followed by a bare carriage return.
///
/// ISO 32000-1:2008 §7.3.8.1 requires the `stream` keyword to be followed by
/// CRLF or a single LF, and explicitly not by CR alone — so this raises a
/// `SpecViolation` from the parser, which is a free-function-sink producer.
fn pdf_with_a_bare_cr_after_stream() -> Vec<u8> {
    let content = b"BT /F1 12 Tf 10 10 Td (hi) Tj ET";
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

    offsets.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream", content.len()).as_bytes());
    pdf.extend_from_slice(b"\r"); // bare CR — the violation
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
fn test_draining_accessor_sees_a_parser_diagnostic() {
    let doc = PdfDocument::from_bytes(pdf_with_a_bare_cr_after_stream()).expect("open");
    let _ = doc.extract_text(0);

    // Drain FIRST — that is the whole point. Calling `structured_warnings()`
    // beforehand would move the free-function entries into the per-document
    // sink and mask the defect.
    let taken = doc.take_structured_warnings();

    assert!(
        taken
            .iter()
            .any(|w| w.category == WarningCategory::SpecViolation),
        "the draining accessor must report the parser's spec violation; got {:?}",
        taken.iter().map(|w| w.category).collect::<Vec<_>>()
    );
}

#[test]
fn both_accessors_agree_on_what_they_can_see() {
    let a = PdfDocument::from_bytes(pdf_with_a_bare_cr_after_stream()).expect("open");
    let _ = a.extract_text(0);
    let via_take: Vec<_> = a
        .take_structured_warnings()
        .iter()
        .map(|w| w.category)
        .collect();

    let b = PdfDocument::from_bytes(pdf_with_a_bare_cr_after_stream()).expect("open");
    let _ = b.extract_text(0);
    let via_snapshot: Vec<_> = b.structured_warnings().iter().map(|w| w.category).collect();

    assert!(
        via_snapshot.contains(&WarningCategory::SpecViolation),
        "precondition: the non-draining accessor already reported the violation, \
         so this fixture does raise one; got {via_snapshot:?}"
    );
    assert_eq!(
        via_take.contains(&WarningCategory::SpecViolation),
        via_snapshot.contains(&WarningCategory::SpecViolation),
        "the two accessors must not differ in which warnings they can observe: \
         take saw {via_take:?}, snapshot saw {via_snapshot:?}"
    );
}
