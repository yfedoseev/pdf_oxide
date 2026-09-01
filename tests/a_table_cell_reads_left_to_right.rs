//! A table cell's members are read left to right, and every surface agrees.
//!
//! `extract_cell_text` ordered a cell's members by `bbox.center().y`
//! descending, with no x tiebreak. A centre moves with font size, so a raised
//! marker in a *smaller* font sorts to the front of the cell while staying
//! inside the line-grouping tolerance — and the backward gap to the member now
//! behind it suppresses the separator, so the two concatenate.
//!
//! `142.56 ± 59.19*^` came out as `59.19*^142.56 ±`, on `extract_text` only:
//! the markdown and HTML renderers walk `cell.spans` in stored order and never
//! saw it. Two surfaces disagreeing about one cell's token order is the real
//! defect; ordering by baseline the way the row comparator already does makes
//! them agree by construction.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// A ruled two-column table whose right cell holds a number, then a raised
/// marker in a smaller font, then a run to the marker's right — so the
/// backward step is what would suppress the separator.
fn cell_with_a_raised_marker() -> Vec<u8> {
    let content: &[u8] = b"0.5 w\n\
        50 40 m 350 40 l S\n50 90 m 350 90 l S\n50 140 m 350 140 l S\n\
        50 40 m 50 140 l S\n200 40 m 200 140 l S\n350 40 m 350 140 l S\n\
        BT\n\
        /F1 8 Tf 1 0 0 1 60 110 Tm (Fasting glucose) Tj\n\
        /F1 8 Tf 1 0 0 1 210 110 Tm (142.56) Tj\n\
        /F2 5 Tf 1 0 0 1 244 113.5 Tm (*) Tj\n\
        /F1 8 Tf 1 0 0 1 248 110 Tm (^) Tj\n\
        ET\n";

    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R /F2 5 0 R >> >> >>"
            .to_vec(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content.to_vec(),
            b"\nendstream".to_vec(),
        ]
        .concat(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    ];

    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    let n = objects.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

/// The number must come before the marker that annotates it.
#[test]
fn the_number_precedes_its_raised_marker() {
    let doc = PdfDocument::from_bytes(cell_with_a_raised_marker()).expect("fixture parses");
    let text = doc.extract_text(0).expect("text");
    let num = text.find("142.56");
    let marker = text.find('*');
    assert!(num.is_some(), "the cell's number is missing entirely:\n{text}");
    if let (Some(n), Some(m)) = (num, marker) {
        assert!(
            n < m,
            "the raised marker sorted ahead of the number it annotates — a \
             smaller font moves bbox.center().y, which is why the cell must be \
             ordered by baseline:\n{text}"
        );
    }
}

/// And the surfaces must not disagree about the cell's token order. This is
/// the invariant worth pinning: `extract_text` reads `cell.text`, while the
/// markdown and HTML renderers walk `cell.spans`, so the two can silently
/// diverge.
#[test]
fn every_surface_orders_the_cell_the_same_way() {
    let doc = PdfDocument::from_bytes(cell_with_a_raised_marker()).expect("fixture parses");
    let opts = ConversionOptions::default();
    let text = doc.extract_text(0).expect("text");
    let md = doc.to_markdown(0, &opts).expect("markdown");

    let order = |s: &str| -> Vec<&'static str> {
        let mut v: Vec<(usize, &'static str)> = Vec::new();
        for tok in ["142.56", "*", "^"] {
            if let Some(i) = s.find(tok) {
                v.push((i, tok));
            }
        }
        v.sort();
        v.into_iter().map(|(_, t)| t).collect()
    };
    assert_eq!(
        order(&text),
        order(&md),
        "extract_text and to_markdown disagree about the cell's token order\n\
         text: {text}\nmd: {md}"
    );
}
