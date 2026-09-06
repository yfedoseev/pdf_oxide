//! A reference marker is separated from the word it annotates; a maths
//! subscript is not.
//!
//! ISO 32000-1:2008 §9.4.4 makes the glyph advance the only thing that moves
//! the text position, so a marker typeset at the base word's advance edge is
//! not separated by geometry — the converter's gap rule correctly reports no
//! gap, and the two run together as `phosphorylation55`.
//!
//! Gap size cannot settle it, and the real measurements are inverted from the
//! intuition: footnote markers sit about 0.10 em from their word, while genuine
//! subscripts (`W2`, `CP3`, `H1`) sit at 0.14 em and beyond. Any threshold that
//! splits the first fuses the second.
//!
//! What separates them is the distinction the text path already draws when it
//! merges sub/superscripts: a host is a *symbol* (`H`, `x`, `ADP`, `SO`), not a
//! prose word. This asserts both directions on one page, because a fix that
//! only splits is as wrong as one that only joins.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// A page with a body font and a smaller marker font, drawing
/// `<base><marker>` at `x`/`y` with the marker at the base's advance edge.
fn two_font_page(base: &str, base_w: f32, marker: &str) -> Vec<u8> {
    // Body at 10pt, marker at 7pt starting 0.99pt past the base's advance —
    // the measured geometry from a real paper.
    let content = format!(
        "BT /F1 10 Tf 72 700 Td ({base}) Tj ET\n\
         BT /F2 7 Tf {} 700 Td ({marker}) Tj ET",
        72.0 + base_w + 0.99
    )
    .into_bytes();

    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>"
            .to_vec(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content,
            b"\nendstream".to_vec(),
        ]
        .concat(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>".to_vec(),
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

fn html_of(pdf: &[u8]) -> String {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open");
    doc.to_html(0, &ConversionOptions::default()).expect("html")
}

fn markdown_of(pdf: &[u8]) -> String {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("open");
    doc.to_markdown(0, &ConversionOptions::default())
        .expect("md")
}

/// Helvetica `phosphorylation` at 10pt is 70.59pt wide by Adobe's metrics
/// (7059/1000 em). Getting this wrong puts the marker *before* the word ends,
/// which is a negative gap and a different case entirely.
const PHOSPHORYLATION_W: f32 = 70.59;

#[test]
fn test_footnote_marker_does_not_glue_to_its_word() {
    let pdf = two_font_page("phosphorylation", PHOSPHORYLATION_W, "55.");
    for (surface, out) in [("html", html_of(&pdf)), ("markdown", markdown_of(&pdf))] {
        assert!(
            !out.contains("phosphorylation55"),
            "{surface}: the reference marker must not glue to the word it annotates; got:\n{out}"
        );
        assert!(
            out.contains("phosphorylation"),
            "{surface}: the word itself must survive; got:\n{out}"
        );
    }
}

/// The counter-case. `H` is a subscript host, so `H2` is one token and must
/// stay joined — at a gap *larger* than the marker's, which is why no
/// gap threshold can do this job.
#[test]
fn test_subscript_on_a_symbol_host_stays_joined() {
    // Helvetica `H` at 10pt is 7.22pt.
    let pdf = two_font_page("H", 7.22, "2");
    for (surface, out) in [("html", html_of(&pdf)), ("markdown", markdown_of(&pdf))] {
        assert!(
            !out.contains("H 2"),
            "{surface}: a subscript on a symbol host must stay joined; got:\n{out}"
        );
    }
}
