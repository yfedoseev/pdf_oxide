//! Markdown post-processing must not fuse distinct blocks: a heading and the
//! paragraph after it, consecutive genuine headings, or the words inside a
//! bold run. Each PDF here is hand-built with known-correct content; the
//! assertions pin the un-fused output.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

/// One-page untagged PDF from a raw content stream, with two Type1 fonts
/// F1 (Helvetica) and F2 (Helvetica-Bold).
fn build_pdf(content: &str) -> Vec<u8> {
    let content = content.as_bytes();
    let mut pdf: Vec<u8> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let mut off = [0usize; 7];
    off[1] = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\
         /Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> /Contents 5 0 R >>\nendobj\n",
    );
    off[4] = pdf.len();
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\
         /Encoding /WinAnsiEncoding >>\nendobj\n",
    );
    off[5] = pdf.len();
    pdf.extend_from_slice(format!("5 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    off[6] = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold\
         /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for o in &off[1..7] {
        pdf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    pdf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    pdf
}

fn markdown(content: &str) -> String {
    let doc = PdfDocument::from_bytes(build_pdf(content)).expect("parse");
    doc.to_markdown(0, &ConversionOptions::default())
        .expect("to_markdown")
}

/// A 20pt heading followed by a 12pt paragraph that begins with a digit.
/// The paragraph must stay a separate block, not be glued onto the heading.
#[test]
fn heading_keeps_following_numeric_paragraph_separate() {
    let md = markdown(
        "BT /F1 20 Tf 1 0 0 1 72 720 Tm (Revenue) Tj \
         /F1 12 Tf 1 0 0 1 72 690 Tm (2024 was a record year.) Tj ET",
    );
    assert!(
        !md.lines()
            .any(|l| l.contains("Revenue") && l.contains("2024")),
        "paragraph merged into the heading line:\n{md}"
    );
    assert!(md.contains("2024 was a record year."), "paragraph text lost:\n{md}");
    assert!(md.contains("Revenue"), "heading text lost:\n{md}");
}

/// Three distinct short headings must stay three headings — not fuse into
/// `## Sales Marketing Engineering`.
#[test]
fn consecutive_short_headings_stay_separate() {
    let md = markdown(
        "BT /F1 20 Tf 1 0 0 1 72 720 Tm (Sales) Tj \
         1 0 0 1 72 690 Tm (Marketing) Tj \
         1 0 0 1 72 660 Tm (Engineering) Tj \
         /F1 12 Tf 1 0 0 1 72 620 Tm (The three teams reported strong results this quarter overall.) Tj ET",
    );
    let names = ["Sales", "Marketing", "Engineering"];
    for line in md.lines() {
        let hits = names.iter().filter(|n| line.contains(*n)).count();
        assert!(hits <= 1, "distinct headings fused into one line {line:?}:\n{md}");
    }
    for n in names {
        assert!(md.contains(n), "heading {n:?} lost:\n{md}");
    }
}

/// A heading too long for one line stays ONE heading. The continuation here
/// starts with a capitalized word ("North"), which is how Title Case wraps
/// almost always break — so this must not depend on the continuation's
/// spelling, only on the first line having run to the column margin.
#[test]
fn wrapped_heading_stays_one_heading() {
    // The 20pt heading's first line runs to the same right margin as the body
    // text below it, i.e. it wrapped; the second line continues it.
    let md = markdown(
        "BT /F1 20 Tf 1 0 0 1 72 720 Tm (Quarterly Report for the Western) Tj \
         1 0 0 1 72 696 Tm (North America Region) Tj \
         /F1 12 Tf 1 0 0 1 72 660 Tm (Revenue grew across every product line this quarter.) Tj ET",
    );
    let heading_lines: Vec<&str> = md
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .collect();
    assert_eq!(
        heading_lines.len(),
        1,
        "a wrapped heading must stay one heading, got {heading_lines:?}\n{md}"
    );
    assert!(
        heading_lines[0].contains("Quarterly Report for the Western")
            && heading_lines[0].contains("North America Region"),
        "both wrapped lines must land in the one heading: {heading_lines:?}\n{md}"
    );
}

/// Bold prose `A gitHub repo` must keep its spaces; the CamelCase inside one
/// word is not license to delete the word boundaries around it.
#[test]
fn bold_prose_with_camelcase_word_keeps_spaces() {
    let md = markdown(
        "BT /F1 12 Tf 1 0 0 1 72 720 Tm (This project uses a special tool.) Tj \
         1 0 0 1 72 696 Tm (See the notes below for details.) Tj \
         /F2 12 Tf 1 0 0 1 72 672 Tm (A gitHub repo) Tj ET",
    );
    assert!(!md.contains("AgitHubrepo"), "spaces deleted inside the bold run:\n{md}");
    assert!(md.contains("A gitHub repo"), "bold prose altered:\n{md}");
}
