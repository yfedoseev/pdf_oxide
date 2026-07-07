//! Pages whose text is rotated 90° via the text matrix
//! (landscape tables typeset on portrait pages, no `/Rotate` key) must
//! extract in the rotated reading frame, and words must carry the rotation.
//!
//! Before the fix, the canonical reading-order pipeline re-sorted the spans
//! with portrait-frame comparators (interleaving every rotated line into
//! word salad), the plain-text assembler grouped lines in the portrait
//! frame, and `rotation_degrees` was dropped at both `TextSpan::to_chars`
//! and word assembly — so consumers could neither read the text in order
//! nor detect that they should handle the page differently.

use pdf_oxide::document::PdfDocument;

/// Portrait page, no `/Rotate`, with a 3-line "landscape table": each line
/// is drawn with `Tm = [0 1 -1 0 x y]` (90° counter-clockwise), so lines
/// stack along +x and read along +y — plus two upright footer words, so the
/// dominant-rotation vote must fire on a mixed page.
///
/// Read with the page turned clockwise, the text is:
///
/// ```text
/// Scheduled maintenance chart     (line at x=100)
/// second inspection row           (line at x=120)
/// third service row               (line at x=140)
/// ```
fn fixture_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    let lines: [(i32, [&str; 3]); 3] = [
        (100, ["Scheduled", "maintenance", "chart"]),
        (120, ["second", "inspection", "row"]),
        (140, ["third", "service", "row"]),
    ];
    for (x, words) in lines {
        let mut y = 200;
        for w in words {
            content.extend_from_slice(format!("0 1 -1 0 {x} {y} Tm ({w}) Tj\n").as_bytes());
            y += 80;
        }
    }
    // Upright minority content (a footer), far from the rotated block.
    content.extend_from_slice(b"1 0 0 1 100 50 Tm (page) Tj\n");
    content.extend_from_slice(b"1 0 0 1 140 50 Tm (312) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

#[test]
fn words_on_dominant_rotation_page_read_in_rotated_frame() {
    let doc = PdfDocument::from_bytes(fixture_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    let rotated: Vec<&str> = words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !matches!(*w, "page" | "312"))
        .collect();
    assert_eq!(
        rotated,
        vec![
            "Scheduled",
            "maintenance",
            "chart",
            "second",
            "inspection",
            "row",
            "third",
            "service",
            "row"
        ],
        "rotated lines must read line-by-line in the rotated frame, got {:?}",
        words
    );
}

#[test]
fn words_carry_rotation_degrees() {
    let doc = PdfDocument::from_bytes(fixture_pdf()).expect("parse fixture");
    let words = doc.extract_words(0).expect("extract words");
    assert!(!words.is_empty());
    for w in &words {
        let expected = if matches!(w.text.as_str(), "page" | "312") {
            0.0
        } else {
            90.0
        };
        assert_eq!(
            w.rotation_degrees, expected,
            "word {:?} must expose its text-matrix rotation",
            w.text
        );
    }
}

#[test]
fn chars_carry_rotation_degrees_through_to_chars() {
    let doc = PdfDocument::from_bytes(fixture_pdf()).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");
    let rotated = spans
        .iter()
        .find(|s| s.text.contains("Scheduled"))
        .expect("rotated span present");
    assert_eq!(rotated.rotation_degrees, 90.0);
    for c in rotated.to_chars() {
        assert_eq!(
            c.rotation_degrees, 90.0,
            "to_chars must propagate the span rotation to {:?}",
            c.char
        );
    }
}

#[test]
fn text_on_dominant_rotation_page_reads_in_rotated_frame() {
    let doc = PdfDocument::from_bytes(fixture_pdf()).expect("parse fixture");
    let text = doc.extract_text(0).expect("extract text");
    let pos = |needle: &str| {
        text.find(needle)
            .unwrap_or_else(|| panic!("{needle:?} missing from {text:?}"))
    };
    assert!(pos("Scheduled") < pos("maintenance"));
    assert!(pos("maintenance") < pos("chart"));
    assert!(pos("chart") < pos("second"));
    assert!(pos("second") < pos("inspection"));
    assert!(pos("inspection") < pos("row"));
    assert!(pos("inspection") < pos("third"));
    assert!(pos("third") < pos("service"));
    let last_row = text.rfind("row").expect("last row");
    assert!(pos("service") < last_row);
}

/// A single rotated margin stamp must NOT trigger the whole-page rotated
/// frame: the horizontal body keeps its portrait reading order and the
/// stamp is appended after it (the pre-existing firewall behavior).
#[test]
fn minority_rotation_does_not_hijack_page_order() {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    for (y, line) in [(700, "First body line"), (680, "Second body line")] {
        for (i, w) in line.split(' ').enumerate() {
            let x = 100 + i * 60;
            content.extend_from_slice(format!("1 0 0 1 {x} {y} Tm ({w}) Tj\n").as_bytes());
        }
    }
    content.extend_from_slice(b"0 1 -1 0 30 300 Tm (stamp) Tj\n");
    content.extend_from_slice(b"ET");
    let doc = PdfDocument::from_bytes(build_minimal_pdf_raw(
        &content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]",
    ))
    .expect("parse");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("words")
        .into_iter()
        .map(|w| w.text)
        .collect();
    let first = words.iter().position(|w| w == "First").expect("First");
    let second = words.iter().position(|w| w == "Second").expect("Second");
    let stamp = words.iter().position(|w| w == "stamp").expect("stamp");
    assert!(first < second, "body order preserved, got {:?}", words);
    assert!(
        stamp > second,
        "minority rotated stamp appended after the body, got {:?}",
        words
    );
}

// ---------------------------------------------------------------------------
// Minimal raw PDF builder (same pattern as test_extraction_robustness.rs)
// ---------------------------------------------------------------------------

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
