//! Regression test for B3: running-artifact detector was removing a
//! document's cover-page title when that title also happened to repeat
//! as the per-page running header.
//!
//! Pre-fix behaviour: any text normalised-signature that appeared in the
//! top/bottom 12% band on ≥50% of pages was classified as a pagination
//! artifact and dropped from the extracted text on every page.
//!
//! Post-fix: the first page on which a signature appears keeps the
//! span; subsequent pages drop it.
//!
//! This test synthesises a 3-page PDF where the string "Universal Title"
//! appears at the top of every page (i.e. classifies as a running
//! header) and is the only distinguishing content on page 1. After the
//! fix the title must still appear in page 1's extracted text.

use pdf_oxide::PdfDocument;

/// Build a 3-page PDF where every page prints "Universal Title" at a
/// header position plus a unique body label.
fn running_header_pdf() -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];

    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let push = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &str| {
        offsets.push(out.len());
        let id = offsets.len() - 1;
        out.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    // 1 Catalog, 2 Pages, 3/4/5 Page objects, 6/7/8 content streams, 9 Font
    push(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>");
    push(&mut out, &mut offsets, "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>");
    let page_common = "/Type /Page /Parent 2 0 R /MediaBox [0 0 600 900] \
                       /Resources << /Font << /F0 9 0 R >> >>";
    push(&mut out, &mut offsets, &format!("<< {page_common} /Contents 6 0 R >>"));
    push(&mut out, &mut offsets, &format!("<< {page_common} /Contents 7 0 R >>"));
    push(&mut out, &mut offsets, &format!("<< {page_common} /Contents 8 0 R >>"));

    // Each page: header "Universal Title" at Y=860 (in the top 12% band
    // of a 900pt page: band is > 792) + unique body text at Y=400.
    let header_y = 860;
    let body_y = 400;
    for body in ["PageOneBody", "PageTwoBody", "PageThreeBody"] {
        let stream = format!(
            "BT /F0 14 Tf 50 {header_y} Td (Universal Title) Tj ET \
             BT /F0 12 Tf 50 {body_y} Td ({body}) Tj ET\n"
        );
        push(
            &mut out,
            &mut offsets,
            &format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len() + 1),
        );
    }

    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

    let xref_offset = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_offset
        )
        .as_bytes(),
    );
    out
}

#[test]
fn first_occurrence_of_running_header_kept_on_page_one() {
    let pdf = running_header_pdf();
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let mut doc = PdfDocument::open(tmp.path()).expect("open");
    assert_eq!(doc.page_count().unwrap(), 3);

    let p0 = doc.extract_text(0).expect("page 0");
    let p1 = doc.extract_text(1).expect("page 1");
    let p2 = doc.extract_text(2).expect("page 2");

    // Bodies always extracted.
    assert!(p0.contains("PageOneBody"), "page 0 body missing: {p0:?}");
    assert!(p1.contains("PageTwoBody"), "page 1 body missing: {p1:?}");
    assert!(p2.contains("PageThreeBody"), "page 2 body missing: {p2:?}");

    // Page 0 — the first occurrence — MUST keep the running-header
    // title. Pre-fix behaviour would strip it.
    assert!(
        p0.contains("Universal Title"),
        "B3 regression: first page should keep the running-header text \
         as it also serves as the cover-page title; got {p0:?}"
    );

    // Pages 1 and 2 drop it as a pagination artifact.
    assert!(
        !p1.contains("Universal Title"),
        "page 1 should suppress the running header on second+ occurrences: {p1:?}"
    );
    assert!(
        !p2.contains("Universal Title"),
        "page 2 should suppress the running header on second+ occurrences: {p2:?}"
    );
}
