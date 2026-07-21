//! `extract_spans_with_reading_order` must drop text lying entirely outside the
//! page MediaBox, the same way `extract_spans` does via `postprocess_spans`.
//!
//! A doc that reuses one big Form XObject across pages relies on a `W n` clip to
//! hide the off-page portion; the raw extractor does not honour `W n`, so without
//! this cull the reading-order path emits every page's worth of off-page spans.

use pdf_oxide::document::{PdfDocument, ReadingOrder};

/// A one-page PDF (MediaBox 0 0 200 200) that draws "VISIBLE" on the page and
/// "OFFPAGE" far above it (y=5000, outside the MediaBox).
fn pdf_with_offpage_text() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut off: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n");
    off.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]\n\
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n",
    );
    let content = b"BT /F1 12 Tf 50 100 Td (VISIBLE) Tj ET\n\
                    BT /F1 12 Tf 50 5000 Td (OFFPAGE) Tj ET\n";
    off.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    let xref = pdf.len();
    let mut x = String::from("xref\n0 6\n0000000000 65535 f \n");
    for o in &off {
        x.push_str(&format!("{o:010} 00000 n \n"));
    }
    pdf.extend_from_slice(x.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

/// A one-page PDF whose MediaBox is written with SWAPPED corners
/// (`[0 200 200 0]`, ury < lly). Corner normalisation must still read it as a
/// 200x200 page, so the on-page text survives instead of the inverted bounds
/// dropping the whole page.
fn pdf_with_swapped_mediabox() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut off: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n");
    off.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 200 200 0]\n\
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n",
    );
    let content = b"BT /F1 12 Tf 50 100 Td (ONPAGE) Tj ET\n";
    off.push(pdf.len());
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    off.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    let xref = pdf.len();
    let mut x = String::from("xref\n0 6\n0000000000 65535 f \n");
    for o in &off {
        x.push_str(&format!("{o:010} 00000 n \n"));
    }
    pdf.extend_from_slice(x.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

fn ro_texts(pdf: Vec<u8>) -> Vec<String> {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    doc.extract_spans_with_reading_order(0, ReadingOrder::TopToBottom)
        .expect("spans")
        .into_iter()
        .map(|s| s.text.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

#[test]
fn reading_order_drops_offpage_spans() {
    let texts = ro_texts(pdf_with_offpage_text());
    assert!(
        texts.iter().any(|t| t.contains("VISIBLE")),
        "on-page text must survive: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("OFFPAGE")),
        "off-MediaBox text must be dropped: {texts:?}"
    );
}

#[test]
fn swapped_mediabox_corners_do_not_drop_the_page() {
    let texts = ro_texts(pdf_with_swapped_mediabox());
    assert!(
        texts.iter().any(|t| t.contains("ONPAGE")),
        "a swapped-corner MediaBox must not drop on-page text: {texts:?}"
    );
}
