//! Integration test: a full-width line above a two-column body must not be
//! sliced by the column (vertical) cut.
//!
//! ISO 32000-1:2008 §14.8.3 ("Basic Layout Model") treats a column area as a
//! *reference area*; a line that spans the full page width (a title, an author
//! line, a section banner) belongs to a single-column band, not to either
//! column. A producer may emit such a line as several show-strings whose break
//! happens to straddle the body's column gutter. The geometric reading-order
//! pass must keep that line whole and ordered before the columns — never split
//! it at the gutter and relocate the tail into the second column.
//!
//! Hand-built untagged PDF (no third-party fixture). The header is two Tj
//! fragments ("...and Carol " then "Williams ...") whose seam falls inside the
//! body's column gutter, mirroring the academic-two-column corpus case where
//! "M. Tanaka" was split into "M. T" / "anaka — Department of Astronomy".

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

/// Build a one-page untagged PDF: a full-width header line (two Tj fragments
/// straddling the gutter) above a six-line two-column body.
fn full_width_header_two_column_pdf() -> Vec<u8> {
    // Content stream: absolute Tm positioning per show-string.
    // Page is 612 wide. Left column x≈80..180, gutter ≈180..330, right
    // column x≈330..520. The header's two fragments span 80..~284 and
    // 288..~500, so the header's ink bridges the gutter (its seam ≈285 sits
    // inside the 180..330 valley the body creates).
    let mut content = String::from("BT /F1 12 Tf\n");
    // Full-width header line at y=750, emitted as two show-strings.
    content.push_str("1 0 0 1 80 750 Tm (Alice Smith, Bob Jones, and Carol ) Tj\n");
    content.push_str("1 0 0 1 288 750 Tm (Williams of the Physics Department) Tj\n");
    // Six left-column lines and six right-column lines. The two columns are
    // independent text flows, so their baselines do not coincide — the right
    // column is offset by a few points, as in any real two-column document.
    let mut y = 710;
    for n in ["one", "two", "three", "four", "five", "six"] {
        content.push_str(&format!("1 0 0 1 80 {y} Tm (Left col line {n}) Tj\n"));
        content.push_str(&format!("1 0 0 1 330 {} Tm (Right col line {n}) Tj\n", y - 7));
        y -= 20;
    }
    content.push_str("ET\n");
    let content = content.into_bytes();

    let mut pdf: Vec<u8> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let mut off = [0usize; 6];
    off[1] = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\
         /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
    );
    off[4] = pdf.len();
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\
         /Encoding /WinAnsiEncoding >>\nendobj\n",
    );
    off[5] = pdf.len();
    pdf.extend_from_slice(format!("5 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(&content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for o in &off[1..6] {
        pdf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    pdf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    pdf
}

fn pos(haystack: &str, needle: &str) -> usize {
    haystack
        .find(needle)
        .unwrap_or_else(|| panic!("markdown is missing {needle:?}; got:\n{haystack}"))
}

// Fixed (#734). `to_markdown` detects the two-column gutter and converts the
// columns independently (`convert_columns_split`). The header is emitted as two
// show-strings straddling the gutter; the band-aware split keeps any line that
// shares a Y-band with a gutter-crossing span whole (a full-width band), so the
// header stays intact and ahead of the body instead of being split across the
// column boundary.
#[test]
fn full_width_header_is_not_split_by_column_cut() {
    let pdf = full_width_header_two_column_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF must parse");
    let md = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("markdown");

    // The header's second fragment must stay with its first fragment — i.e.
    // the whole header precedes the body. With the column-cut bug, "Williams"
    // is relocated into the right column and lands AFTER every left-column
    // line, so "Left col line one" would appear before "Williams".
    assert!(
        pos(&md, "Williams") < pos(&md, "Left col line one"),
        "full-width header tail was relocated into a column; md:\n{md}"
    );

    // And the columns themselves must not interleave: the entire left column
    // precedes the right column.
    assert!(
        pos(&md, "Left col line six") < pos(&md, "Right col line one"),
        "columns interleaved; md:\n{md}"
    );
}
