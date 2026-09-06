//! Two documents read in sequence on one thread do not take each other's
//! diagnostics.
//!
//! Five producers hold no `PdfDocument` — the stream parser, the operator-cap
//! notice, the Type 3 and missing-`/ToUnicode` font notices, and the
//! rasterizer's dropped-glyph report — so they write to a thread-local sink.
//! That sink is keyed by **thread**, not by document, and the drain is
//! first-caller-wins. Parse A without draining it, then parse B, and B's
//! accessor reported A's diagnostics as its own.
//!
//! The two font producers were fixed by attributing them directly (they do
//! receive the document). The rest cannot be, so each operation now borrows the
//! thread-local sink for its duration: it sets aside whatever was pending on
//! entry, absorbs what it raised on exit, and puts the other document's entries
//! back untouched.
//!
//! This exercises the parser's `SpecViolation`, which is one of the producers
//! that holds no document — so it covers the path the font fix could not.

use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;

/// ISO 32000-1:2008 §7.3.8.1 requires the `stream` keyword to be followed by
/// CRLF or a single LF, never by CR alone. A bare CR raises `SpecViolation`
/// from the parser — a producer with no document in scope.
fn pdf_with_a_bare_cr_after_stream() -> Vec<u8> {
    build(true)
}

/// The same document with a well-formed stream keyword, raising nothing.
fn clean_pdf() -> Vec<u8> {
    build(false)
}

fn build(bare_cr: bool) -> Vec<u8> {
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
    pdf.extend_from_slice(if bare_cr { b"\r" } else { b"\n" });
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

fn cats(doc: &PdfDocument) -> Vec<WarningCategory> {
    doc.structured_warnings()
        .iter()
        .map(|w| w.category)
        .collect()
}

#[test]
fn test_second_document_does_not_inherit_the_first_documents_violation() {
    // A is read and never drained — the shape that leaked.
    let a = PdfDocument::from_bytes(pdf_with_a_bare_cr_after_stream()).expect("open a");
    let _ = a.extract_text(0);

    let b = PdfDocument::from_bytes(clean_pdf()).expect("open b");
    let _ = b.extract_text(0);

    let a_cats = cats(&a);
    assert!(
        a_cats.contains(&WarningCategory::SpecViolation),
        "precondition: the malformed document must raise the violation at all, \
         otherwise the assertion below passes vacuously; got {a_cats:?}"
    );

    let b_cats = cats(&b);
    assert!(
        !b_cats.contains(&WarningCategory::SpecViolation),
        "the clean document must not inherit the malformed one's parser \
         diagnostic; got {b_cats:?}"
    );
}

/// The other half of the contract: borrowing the sink must not *lose* the
/// first document's diagnostics either. A is still able to report its own
/// violation after B has come and gone.
#[test]
fn test_first_documents_diagnostics_survive_the_second_document() {
    let a = PdfDocument::from_bytes(pdf_with_a_bare_cr_after_stream()).expect("open a");
    let _ = a.extract_text(0);

    {
        let b = PdfDocument::from_bytes(clean_pdf()).expect("open b");
        let _ = b.extract_text(0);
        let _ = b.structured_warnings();
    }

    let a_cats = cats(&a);
    assert!(
        a_cats.contains(&WarningCategory::SpecViolation),
        "the first document's diagnostic must survive an intervening document — \
         scoping the sink must set entries aside, not discard them; got {a_cats:?}"
    );
}
