//! `PdfDocument::extract_structured` end-to-end (issue #536).
//!
//! Builds a synthetic two-column page in memory and asserts the structured
//! surface returns body regions tagged with the correct column order. No real
//! or MPL-licensed PDFs are used.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::structured::RegionRole;

/// One-page PDF with two text columns: left at x=70, right at x=360.
fn two_column_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = vec![0usize; 6];

    let obj = |pdf: &mut Vec<u8>, offsets: &mut Vec<usize>, n: usize, body: &str| {
        offsets[n] = pdf.len();
        pdf.extend_from_slice(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    obj(&mut pdf, &mut offsets, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut pdf, &mut offsets, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut pdf,
        &mut offsets,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    );
    // Two lines per column, left column (x=70) and right column (x=360).
    let content = "BT /F1 11 Tf 70 700 Td (Left line one) Tj ET\n\
                   BT /F1 11 Tf 70 680 Td (Left line two) Tj ET\n\
                   BT /F1 11 Tf 360 700 Td (Right line one) Tj ET\n\
                   BT /F1 11 Tf 360 680 Td (Right line two) Tj ET";
    obj(
        &mut pdf,
        &mut offsets,
        4,
        &format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content),
    );
    obj(
        &mut pdf,
        &mut offsets,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    );

    let xref = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{off:010} 00000 n \r\n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref
        )
        .as_bytes(),
    );
    pdf
}

#[test]
fn extract_structured_returns_body_regions() {
    let doc = PdfDocument::from_bytes(two_column_pdf()).unwrap();
    let page = doc.extract_structured(0).unwrap();

    assert_eq!(page.page_index, 0);
    assert!((page.page_width - 612.0).abs() < 1.0);
    assert!(!page.regions.is_empty(), "structured page must have regions");

    // All four lines are body text.
    assert!(
        page.regions.iter().all(|r| r.kind == RegionRole::BodyBlock),
        "all regions should be BodyBlock; got: {:?}",
        page.regions.iter().map(|r| &r.kind).collect::<Vec<_>>()
    );

    // The combined text must contain all four fragments.
    let all: String = page
        .regions
        .iter()
        .map(|r| r.text.clone())
        .collect::<Vec<_>>()
        .join(" ");
    for frag in [
        "Left line one",
        "Left line two",
        "Right line one",
        "Right line two",
    ] {
        assert!(all.contains(frag), "missing {frag:?} in {all:?}");
    }

    // Two-column geometry: both column indices 0 and 1 must appear.
    let cols: std::collections::BTreeSet<Option<usize>> =
        page.regions.iter().map(|r| r.column_index).collect();
    assert!(
        cols.contains(&Some(0)) && cols.contains(&Some(1)),
        "both columns must be detected; got {cols:?}"
    );
}
