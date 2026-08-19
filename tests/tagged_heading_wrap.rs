//! A heading that wraps across lines is one heading.
//!
//! ISO 32000-1 §14.8.3 makes one BLSE a single block that "can be split
//! between lines of text". That applies to an `<Hn>` as much as to a `<P>`,
//! so a title drawn on two baselines inside one heading element must emit as
//! one markdown heading rather than one per line.
//!
//! The continuation signals that serve body text cannot decide this: they
//! require the next line to open lowercase, and a title-cased heading
//! capitalises every line, so `Tax and` / `Credits` reads as two headings.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Tagged one-page PDF: an `<H2>` whose marked content is drawn on two
/// baselines, then a `<P>` of body text so the heading has something to end
/// against.
fn wrapped_heading_pdf() -> Vec<u8> {
    let content: &[u8] = b"BT
/H2 <</MCID 0>> BDC
/F2 18 Tf
1 0 0 1 72 700 Tm (Tax and) Tj
1 0 0 1 72 678 Tm (Credits) Tj
EMC
/P <</MCID 1>> BDC
/F1 10 Tf
1 0 0 1 72 640 Tm (Body text follows the heading.) Tj
EMC
ET";
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");

    let bodies: Vec<(usize, Vec<u8>)> = vec![
        (
            1,
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 10 0 R /MarkInfo << /Marked true >> >>"
                .to_vec(),
        ),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec()),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
/Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> /Contents 5 0 R /StructParents 0 >>"
                .to_vec(),
        ),
        (4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec()),
        (5, stream),
        (6, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>".to_vec()),
        (10, b"<< /Type /StructTreeRoot /K [11 0 R] >>".to_vec()),
        (
            11,
            b"<< /Type /StructElem /S /Document /P 10 0 R /K [12 0 R 13 0 R] >>".to_vec(),
        ),
        (
            12,
            b"<< /Type /StructElem /S /H2 /P 11 0 R /Pg 3 0 R /K 0 >>".to_vec(),
        ),
        (
            13,
            b"<< /Type /StructElem /S /P /P 11 0 R /Pg 3 0 R /K 1 >>".to_vec(),
        ),
    ];

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let max_id = 13;
    let mut offsets = vec![0usize; max_id + 1];
    for (id, body) in &bodies {
        offsets[*id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", max_id + 1).as_bytes());
    for id in 1..=max_id {
        if offsets[id] != 0 {
            out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
        } else {
            out.extend_from_slice(b"0000000000 65535 f \n");
        }
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", max_id + 1)
            .as_bytes(),
    );
    out
}

#[test]
fn wrapped_heading_emits_one_heading() {
    let doc = PdfDocument::from_bytes(wrapped_heading_pdf()).expect("parse");
    let md = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("to_markdown");

    let headings: Vec<&str> = md.lines().filter(|l| l.starts_with('#')).collect();
    assert_eq!(
        headings.len(),
        1,
        "a heading split across two baselines inside one <H2> must emit once, got {headings:?}\n\n{md}"
    );
    let heading = headings[0];
    assert!(
        heading.contains("Tax and") && heading.contains("Credits"),
        "both lines of the wrapped heading must survive in it, got {heading:?}\n\n{md}"
    );
}
