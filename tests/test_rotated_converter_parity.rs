//! Every text-producing surface must read a rotated page the same way.
//!
//! The rotated reading frame is applied at one call site, inside
//! `extract_text`. `to_markdown` and `to_html` assemble the same page in raw
//! page space, so the same document extracts correctly through one surface and
//! incorrectly through the other.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

/// A page whose text is predominantly rotated: three 90° lines, so the
/// dominant-rotation vote fires and `extract_text` assembles in the frame.
fn rotated_page_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    for (x, text) in [
        (200, "Engine oil capacity"),
        (214, "six point zero quarts"),
        (228, "Motorcraft synthetic blend"),
    ] {
        content.extend_from_slice(format!("0 1 -1 0 {x} 150 Tm ({text}) Tj\n").as_bytes());
    }
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// Remove HTML tags before tokenizing. `reading_order` trims non-alphanumerics
/// from a token's EDGES, which cannot clear a tag: `<p>Engine` trims the `<`,
/// then stops at the alphanumeric `p`, leaving `p>Engine`. Tags have to go
/// before tokenizing, not after.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut depth = 0usize;
    for c in html.chars() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth = depth.saturating_sub(1);
                out.push(' ');
            },
            _ if depth == 0 => out.push(c),
            _ => {},
        }
    }
    out
}

/// Strip markup and whitespace so only the reading order is compared.
fn reading_order(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

#[test]
fn converters_read_a_rotated_page_in_the_same_order_as_extract_text() {
    let doc = PdfDocument::from_bytes(rotated_page_pdf()).expect("parse fixture");
    let options = ConversionOptions::default();

    let from_text = reading_order(&doc.extract_text(0).expect("extract text"));
    let from_markdown = reading_order(&doc.to_markdown(0, &options).expect("to_markdown"));

    // The fixture must reach the framed path, or the comparison is vacuous.
    assert!(
        from_text.windows(3).any(|w| w == ["Engine", "oil", "capacity"]),
        "extract_text did not assemble the rotated page: {from_text:?}"
    );

    assert_eq!(
        from_text, from_markdown,
        "extract_text and to_markdown disagree on a rotated page"
    );
}

#[test]
fn html_reads_a_rotated_page_in_the_same_order_as_extract_text() {
    let doc = PdfDocument::from_bytes(rotated_page_pdf()).expect("parse fixture");
    let options = ConversionOptions::default();

    let from_text = reading_order(&doc.extract_text(0).expect("extract text"));
    let from_html = reading_order(&strip_tags(&doc.to_html(0, &options).expect("to_html")));

    assert!(
        from_text.windows(3).any(|w| w == ["Engine", "oil", "capacity"]),
        "extract_text did not assemble the rotated page: {from_text:?}"
    );
    assert_eq!(
        from_text, from_html,
        "extract_text and to_html disagree on a rotated page"
    );
}

#[test]
fn plain_text_reads_a_rotated_page_in_the_same_order_as_extract_text() {
    let doc = PdfDocument::from_bytes(rotated_page_pdf()).expect("parse fixture");
    let options = ConversionOptions::default();

    let from_text = reading_order(&doc.extract_text(0).expect("extract text"));
    let from_plain = reading_order(&doc.to_plain_text(0, &options).expect("to_plain_text"));

    assert!(
        from_text.windows(3).any(|w| w == ["Engine", "oil", "capacity"]),
        "extract_text did not assemble the rotated page: {from_text:?}"
    );
    assert_eq!(
        from_text, from_plain,
        "extract_text and to_plain_text disagree on a rotated page"
    );
}

/// The layer/ink-filtered surface assembles from its own span source, so it
/// needs the frame applied there too — with no filters it must equal
/// `extract_text`, and with a filter that excludes nothing it still must.
#[test]
fn filtered_text_reads_a_rotated_page_in_the_same_order_as_extract_text() {
    use std::collections::HashSet;

    let doc = PdfDocument::from_bytes(rotated_page_pdf()).expect("parse fixture");

    let from_text = reading_order(&doc.extract_text(0).expect("extract text"));
    let mut inks = HashSet::new();
    inks.insert("NoSuchInkOnThisPage".to_string());
    let filtered = doc
        .extract_text_filtered(0, HashSet::new(), inks)
        .expect("extract_text_filtered");
    let from_filtered = reading_order(&filtered);

    assert!(
        from_text.windows(3).any(|w| w == ["Engine", "oil", "capacity"]),
        "extract_text did not assemble the rotated page: {from_text:?}"
    );
    assert_eq!(
        from_text, from_filtered,
        "extract_text and extract_text_filtered disagree on a rotated page"
    );
}

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

/// A subscript inside a rotated run sits off the run's writing axis by the
/// subscript drop, not by a line height. It must stay attached to the formula
/// it belongs to: `N`, a smaller `2`, and `O` drawn as three runs of one
/// rotated label are one chemical formula, not three fragments separated by
/// unrelated page content.
fn rotated_formula_with_subscript_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // A 90-degree label "N2O" whose middle glyph is dropped below the baseline,
    // the shape a rotated chart axis uses. Under a 90-degree matrix the drop is
    // a displacement along -x, i.e. PERPENDICULAR to the +y writing axis — the
    // exact quantity the continuation test measures.
    //
    // One BT/ET and one Tf size throughout, deliberately: `ET` flushes the run
    // buffer and so does a Tf size change, either of which would leave the
    // buffer empty at the next Tm and make the continuation test unreachable —
    // the assertion below would then hold no matter what that test decided.
    content.extend_from_slice(b"BT /F1 18 Tf\n");
    content.extend_from_slice(b"0 1 -1 0 246 244 Tm (N) Tj\n");
    content.extend_from_slice(b"0 1 -1 0 242 258 Tm (2) Tj\n");
    content.extend_from_slice(b"0 1 -1 0 246 266 Tm (O) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// A baseline drop inside a rotated run is a sub-glyph perpendicular offset, not
/// a line break: the formula must not be split apart by the continuation test.
#[test]
fn rotated_subscript_formula_stays_contiguous() {
    let doc = PdfDocument::from_bytes(rotated_formula_with_subscript_pdf()).expect("parse fixture");
    let text = doc.extract_text(0).expect("extract text");
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join("");
    assert!(flat.contains("N2O"), "rotated subscripted formula came apart: {text:?}");
}
