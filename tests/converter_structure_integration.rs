//! End-to-end converter structure tests: a hand-built PDF exercising a
//! font-size heading hierarchy, a bullet list, an ordered list, and a `/Link`
//! annotation flows through `to_markdown` / `to_html`, and the output must carry
//! the right heading levels, `<ul>/<ol>/<li>`, and resolved hyperlinks.
//!
//! Covers the markdown/HTML structure fixes for heading/list boundaries and
//! hyperlink emission (no external fixtures — the PDF is built inline).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Append `id 0 obj\n<body>\nendobj\n`, recording the object's byte offset.
fn obj(buf: &mut Vec<u8>, offsets: &mut [usize], id: usize, body: &str) {
    offsets[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
    buf.extend_from_slice(body.as_bytes());
    buf.extend_from_slice(b"\nendobj\n");
}

/// A one-page PDF: a 22pt title (→H1), two 16pt section headings (→H2), a
/// bullet list, an ordered list, and a Link annotation over "the full report".
/// The bullet glyph is WinAnsi 0x95 (•).
fn structured_pdf() -> Vec<u8> {
    // Byte string so the WinAnsi bullet 0x95 can be embedded literally.
    let content: &[u8] = b"\
BT /F1 22 Tf 72 750 Td (Quarterly Report) Tj ET\n\
BT /F1 11 Tf 72 720 Td (This document summarizes results.) Tj ET\n\
BT /F1 16 Tf 72 690 Td (Highlights) Tj ET\n\
BT /F1 11 Tf 72 665 Td (\x95 Revenue grew steadily.) Tj ET\n\
BT /F1 11 Tf 72 650 Td (\x95 Costs remained flat.) Tj ET\n\
BT /F1 16 Tf 72 615 Td (Next Steps) Tj ET\n\
BT /F1 11 Tf 72 590 Td (1. Finalize the budget.) Tj ET\n\
BT /F1 11 Tf 72 575 Td (2. Hire two engineers.) Tj ET\n\
BT /F1 11 Tf 72 545 Td (See ) Tj ET\n\
BT /F1 11 Tf 95 545 Td (the full report) Tj ET\n\
BT /F1 11 Tf 168 545 Td (.) Tj ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 8]; // ids 1..=7

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    // 1: catalog
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    // 2: page tree
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    // 3: page — references the content stream, font, and the Link annotation
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /Annots [6 0 R] >>",
    );
    // 4: content stream
    off[4] = buf.len();
    buf.extend_from_slice(b"4 0 obj\n");
    buf.extend_from_slice(format!("<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"endstream\nendobj\n");
    // 5: WinAnsi Helvetica (so 0x95 decodes to the bullet •)
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // 6: Link annotation over "the full report" → URI action
    obj(
        &mut buf,
        &mut off,
        6,
        "<< /Type /Annot /Subtype /Link /Rect [95 543 168 558] /Border [0 0 0] \
         /A << /S /URI /URI (https://example.com/report) >> >>",
    );

    // xref
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    buf
}

fn opts() -> ConversionOptions {
    ConversionOptions::default()
}

#[test]
fn markdown_structure_and_links() {
    let doc = PdfDocument::from_bytes(structured_pdf()).unwrap();
    let md = doc.to_markdown(0, &opts()).unwrap();

    // Heading hierarchy from font-size ratios (base ≈ 11pt body).
    assert!(md.contains("# Quarterly Report"), "missing H1:\n{md}");
    assert!(md.contains("## Highlights"), "missing H2 Highlights:\n{md}");
    assert!(md.contains("## Next Steps"), "missing H2 Next Steps:\n{md}");

    // Heading must not be glued to the following list item.
    assert!(!md.contains("Highlights -"), "heading glued to bullet:\n{md}");
    assert!(!md.contains("Highlights •"), "heading glued to bullet glyph:\n{md}");
    assert!(!md.contains("Next Steps 1."), "heading glued to ordered item:\n{md}");

    // Bullet list rendered with markdown markers; ordered list keeps its own
    // numbering with no doubled "- 1." marker.
    assert!(md.contains("- Revenue grew steadily."), "bullet item missing:\n{md}");
    assert!(md.contains("1. Finalize the budget."), "ordered item missing:\n{md}");
    assert!(!md.contains("- 1."), "ordered item double-marked:\n{md}");

    // Hyperlink resolved from the /Link annotation. The anchor may
    // cover the whole merged line; what matters is the URL is not lost.
    assert!(
        md.contains("the full report](https://example.com/report)"),
        "markdown link missing:\n{md}"
    );
}

#[test]
fn html_structure_and_links() {
    let doc = PdfDocument::from_bytes(structured_pdf()).unwrap();
    let html = doc.to_html(0, &opts()).unwrap();

    // Heading levels (no <strong> wrap).
    assert!(html.contains("<h1>Quarterly Report</h1>"), "missing H1:\n{html}");
    assert!(html.contains("<h2>Highlights</h2>"), "missing H2:\n{html}");
    assert!(!html.contains("<h1><strong>"), "heading wrapped in <strong>:\n{html}");

    // Lists as <ul>/<ol>/<li>, not <p>.
    assert!(
        html.contains("<ul>") && html.contains("<li>Revenue grew steadily.</li>"),
        "bullet list:\n{html}"
    );
    assert!(
        html.contains("<ol>") && html.contains("<li>Finalize the budget.</li>"),
        "ordered list:\n{html}"
    );

    // Hyperlink as an anchor. The anchor may cover the whole merged
    // line; what matters is the href resolves to the target.
    assert!(
        html.contains("href=\"https://example.com/report\"")
            && html.contains("the full report</a>"),
        "html link missing:\n{html}"
    );
}
