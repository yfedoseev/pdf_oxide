//! A column cut may not pass through a run that crosses it.
//!
//! A newspaper front page carries a nameplate at the top right, a section
//! banner at the top left, a dateline reached by a wide repositioning jump,
//! a headline set to the full measure, and a single body column on the left.
//!
//! The column detector builds a density profile of the page and looks for an
//! empty vertical corridor. It deliberately leaves full-measure runs out of
//! that profile so a banner headline cannot hide the columns beneath it — but
//! a run left out of the profile still occupies the page. Between the body
//! column's right edge and the nameplate's left edge there is a wide gap, and
//! with the headline's ink missing from the profile that gap reads as a column
//! gutter. The cut is taken, and the nameplate and dateline are emitted after
//! the entire body column instead of at the top of the page where they are
//! printed.
//!
//! ISO 32000-1:2008 §9.4.4 computes the glyph displacement along the writing
//! axis and sets the component for the other axis to 0, so a horizontal run
//! occupies one unbroken interval on the X axis. The headline's interval
//! covers the candidate corridor on both sides, which means the corridor is
//! not empty and there is no column boundary there.
//!
//! poppler, PyMuPDF and pdfium all emit the nameplate first.

use pdf_oxide::PdfDocument;

/// Horizontal scaling (`Tz`) is what sets each run's width here, so the
/// numbers are transcribed rather than derived: the density profile measures
/// character count against font size, and a run rebuilt to the same bounding
/// box with a different character count does not exercise the same predicate.
const CONTENT: &str = concat!(
    // Nameplate, top right — the run that must be read first.
    "BT 3 Tr 1 0 0 1 235.20 753.36 Tm /F1 21 Tf 119.45 Tz (Gazette Herald) Tj 100 Tz ET\n",
    // Section banner, top left.
    "BT 3 Tr 1 0 0 1 126.72 711.60 Tm /F1 38 Tf 68.28 Tz (l) Tj 100 Tz ET\n",
    // Masthead line: two asterisks, then the dateline reached by a 101pt
    // repositioning jump inside one text object.
    "BT 3 Tr /F1 8 Tf\n",
    "1 0 0 1 132.48 697.20 Tm 128.53 Tz (*) Tj\n",
    "1 0 0 1 139.44 697.20 Tm 208.87 Tz (*) Tj\n",
    "1 0 0 1 246.96 696.96 Tm 115.28 Tz (WEDNESDAY,) Tj\n",
    "1 0 0 1 312.96 696.96 Tm 119.69 Tz (JUNE) Tj\n",
    "1 0 0 1 342.00 696.96 Tm 134.89 Tz (3,) Tj\n",
    "1 0 0 1 355.44 696.96 Tm 144.45 Tz (2020) Tj\n",
    "ET\n",
    // The full-measure headline: 55.68 -> 534.77, crossing the whole page.
    "BT 3 Tr 1 0 0 1 55.68 628.32 Tm /F1 29 Tf 96.83 Tz \
     (Wide Headline Runs Across The Page) Tj 100 Tz ET\n",
    "BT 3 Tr 1 0 0 1 75.60 606.72 Tm /F1 10 Tf 107.29 Tz (By Some Person) Tj 100 Tz ET\n",
    "BT 3 Tr 1 0 0 1 78.72 597.84 Tm /F1 6 Tf 142.24 Tz (Wire Service Name) Tj 100 Tz ET\n",
);

/// Eleven body lines in a single left column, 54.48 -> 175.48.
fn body_column() -> String {
    let mut s = String::new();
    let mut y = 588.48_f32;
    while y > 495.0 {
        s.push_str(&format!(
            "BT 3 Tr 1 0 0 1 54.48 {y:.2} Tm /F1 8 Tf 112.89 Tz \
             (aaaa bbbb cccc dddd eeee ffff) Tj 100 Tz ET\n"
        ));
        y -= 8.8;
    }
    s
}

fn front_page() -> Vec<u8> {
    let content = format!("{CONTENT}{}", body_column()).into_bytes();
    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_vec(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content,
            b"\nendstream".to_vec(),
        ]
        .concat(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
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

#[test]
fn a_nameplate_is_read_at_the_top_of_the_page_it_is_printed_on() {
    let doc = PdfDocument::from_bytes(front_page()).expect("open");
    let text = doc.extract_text(0).expect("text");

    let nameplate = text
        .find("Gazette")
        .unwrap_or_else(|| panic!("nameplate missing:\n{text}"));
    let body = text
        .find("aaaa")
        .unwrap_or_else(|| panic!("body missing:\n{text}"));
    assert!(
        nameplate < body,
        "the nameplate is printed at the top of the page and must be read \
         before the body column, not after it; got nameplate@{nameplate} \
         body@{body} in:\n{text}"
    );
}

/// The dateline travels with the nameplate: both sit in the band the spurious
/// column cut carved off, so a fix that rescued only the nameplate would leave
/// the dateline stranded below the body.
#[test]
fn a_dateline_is_read_at_the_top_of_the_page_it_is_printed_on() {
    let doc = PdfDocument::from_bytes(front_page()).expect("open");
    let text = doc.extract_text(0).expect("text");
    let dateline = text
        .find("WEDNESDAY")
        .unwrap_or_else(|| panic!("dateline missing:\n{text}"));
    let body = text
        .find("aaaa")
        .unwrap_or_else(|| panic!("body missing:\n{text}"));
    assert!(
        dateline < body,
        "the dateline must be read with the masthead, not after the body \
         column; got dateline@{dateline} body@{body} in:\n{text}"
    );
}
