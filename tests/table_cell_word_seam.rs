//! A table cell must not re-decide word joins the span-level merger already
//! made from per-glyph advance evidence.
//!
//! A producer that draws one word as several show operations, repositioning
//! between them with sub-em gaps (declared glyph widths understating the true
//! pen steps), puts the cell text builder in a position no gap threshold can
//! win from. The fixture below reproduces that inversion exactly:
//!
//!   intra-word seam, must NOT become a space :  1.75 pt
//!   real word space, must     become a space :  1.53 pt
//!
//! The seam is WIDER than the legitimate space, so no constant separates
//! them: a threshold high enough to keep "Credit" whole (>= 1.75) also
//! swallows the space before "<", and one low enough to keep that space
//! (< 1.53) also splits "Credit". Only the source-order evidence tells them
//! apart — the space is a space GLYPH occupying its own character position,
//! while the seam is pure repositioning between consecutive glyphs. The
//! span-level merger reads that evidence and joins correctly, so table
//! extraction must consume its verdict rather than re-derive the join from
//! raw bbox gaps.

use pdf_oxide::document::PdfDocument;

/// A fully ruled 3-row, 2-column grid whose middle value cell draws
/// "Credit < 21 500 euros" as four show ops: `(Cre) (d) (it)` with 1.75 pt
/// seam gaps, then `( < 21 500 euros)` continuing at the exact pen position.
///
/// Glyph widths are 453/1000 em (4.077 pt at 9 pt) except the space, which is
/// declared 170/1000 em — a 1.53 pt advance, NARROWER than the 1.75 pt seams
/// yet still above the cell builder's own `font_size * 0.15` (1.35 pt) floor,
/// so a correct implementation keeps it. That inversion is the whole point of
/// the fixture: a threshold high enough to keep "Credit" whole also swallows
/// the space before "<", and one low enough to keep that space also splits
/// "Credit".
fn fragmented_cell_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // Grid rules: 3 rows x 2 cols, fully stroked.
    for y in [710.0f32, 690.0, 670.0, 650.0] {
        content.extend_from_slice(format!("0.7 w 95 {y} m 420 {y} l S\n").as_bytes());
    }
    for x in [95.0f32, 300.0, 420.0] {
        content.extend_from_slice(format!("0.7 w {x} 650 m {x} 710 l S\n").as_bytes());
    }
    content.extend_from_slice(b"BT /F1 9 Tf\n");
    // Header row + a plain last row so the detector sees a real grid.
    content.extend_from_slice(b"1 0 0 1 100 695 Tm (Garanties) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 695 Tm (Montant) Tj\n");
    content.extend_from_slice(b"1 0 0 1 100 655 Tm (Duree) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 655 Tm (25 ans) Tj\n");
    // Value cell drawn as fragments. Glyph advance = 0.453 * 9 = 4.077 pt.
    // (Cre) spans 100 .. 112.231; each following fragment starts 1.75 pt
    // past the declared end of the previous one.
    content.extend_from_slice(b"1 0 0 1 100 675 Tm (Cre) Tj\n");
    content.extend_from_slice(b"1 0 0 1 113.981 675 Tm (d) Tj\n");
    content.extend_from_slice(b"1 0 0 1 119.808 675 Tm (it) Tj\n");
    // Continuation at the exact pen position: 119.808 + 2 * 4.077 = 127.962.
    content.extend_from_slice(b"1 0 0 1 127.962 675 Tm ( < 21 500 euros) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 675 Tm (Oui) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// The fixture is only meaningful while no constant can separate an
/// intra-word seam from a legitimate word space. Measure both from the
/// extracted geometry and assert the inversion, so that a future change to
/// the fixture (or to how widths become advances) cannot quietly turn these
/// tests into something a threshold tweak would satisfy.
#[test]
fn seam_is_wider_than_the_word_space_it_must_be_told_apart_from() {
    let doc = PdfDocument::from_bytes(fragmented_cell_pdf()).expect("open");
    let words = doc.extract_words(0).expect("words");
    let gap_after = |text: &str| -> f32 {
        let i = words
            .iter()
            .position(|w| w.text == text)
            .unwrap_or_else(|| {
                panic!(
                    "no word {text:?} in {:?}",
                    words.iter().map(|w| &w.text).collect::<Vec<_>>()
                )
            });
        words[i + 1].bbox.x - (words[i].bbox.x + words[i].bbox.width)
    };

    // "Cre" -> "d": pure repositioning between consecutive glyphs.
    let seam = gap_after("Cre");
    // "it" -> "<": a real space glyph sits between them in the source.
    let space = gap_after("it");

    assert!(
        (seam - 1.75).abs() < 0.05,
        "fixture drift: intra-word seam is {seam:.3} pt, expected 1.75"
    );
    assert!(
        (space - 1.53).abs() < 0.05,
        "fixture drift: word space is {space:.3} pt, expected 1.53"
    );
    assert!(
        seam > space,
        "fixture no longer pins the defect: the intra-word seam ({seam:.3} pt) must be \
         WIDER than the legitimate word space ({space:.3} pt), or a gap threshold could \
         separate them and the fix under test would be unnecessary"
    );
    // The space must also stay above the cell builder's own floor, or the
    // second test would be demanding a space this change does not restore
    // (it re-glues split words; it does not re-split fused ones).
    assert!(
        space > 9.0 * 0.15,
        "fixture drift: word space {space:.3} pt fell below the cell builder's \
         font_size * 0.15 floor, so it is dropped for reasons this change does not address"
    );
}

#[test]
fn sub_em_fragment_seams_do_not_split_cell_words() {
    let doc = PdfDocument::from_bytes(fragmented_cell_pdf()).expect("open");

    // The span-level merger must see the fragments as one unspaced word —
    // this pins the fixture's geometry; if it drifts (a seam wide enough
    // that the merger itself inserts a space), fail here, loudly.
    let spans = doc.extract_spans(0).expect("spans");
    let flow = spans
        .iter()
        .find(|s| s.text.contains("euros"))
        .expect("flow span with the fragmented line");
    assert!(
        flow.text.contains("Credit < 21 500 euros"),
        "fixture drift: span-level merger no longer joins the seams: {:?}",
        flow.text
    );

    let tables = doc.extract_tables(0).expect("tables");
    let cell = tables
        .iter()
        .flat_map(|t| &t.rows)
        .flat_map(|r| &r.cells)
        .find(|c| c.text.contains("euros"))
        .expect("cell with the fragmented line");
    assert_eq!(
        cell.text, "Credit < 21 500 euros",
        "cell builder re-split a word the span-level merger had joined"
    );
}

/// Same fixture through the text-assembly table path (`extract_text`), which
/// feeds the detector from its own word-derived spans.
#[test]
fn sub_em_fragment_seams_do_not_split_words_in_extracted_text() {
    let doc = PdfDocument::from_bytes(fragmented_cell_pdf()).expect("open");
    // Pin that this page really does exercise the table path: without a
    // detected table the assertions below are satisfied by ordinary flow
    // text and would pass with the fix reverted.
    assert!(
        !doc.extract_tables(0).expect("tables").is_empty(),
        "fixture no longer detects a table, so this test would not cover the table path"
    );
    let text = doc.extract_text(0).expect("text");
    assert!(text.contains("Credit"), "extracted text lost the fragmented word: {text:?}");
    assert!(!text.contains("Cre d it"), "extracted text shows the re-split word: {text:?}");
}

/// A word the merger drew consecutively but which then jumps BACKWARD is a
/// displayed-math denominator or a wrapped line, not a kerned seam. Source
/// adjacency alone would fuse it; the gap band's lower bound is what refuses
/// it. Without that bound the cell reads `= dt` fused as one token.
#[test]
fn backward_jump_within_a_run_is_not_fused_into_the_previous_word() {
    let mut content = Vec::new();
    for y in [710.0f32, 690.0, 670.0, 650.0] {
        content.extend_from_slice(format!("0.7 w 95 {y} m 420 {y} l S\n").as_bytes());
    }
    for x in [95.0f32, 300.0, 420.0] {
        content.extend_from_slice(format!("0.7 w {x} 650 m {x} 710 l S\n").as_bytes());
    }
    content.extend_from_slice(b"BT /F1 9 Tf\n");
    content.extend_from_slice(b"1 0 0 1 100 695 Tm (Garanties) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 695 Tm (Montant) Tj\n");
    content.extend_from_slice(b"1 0 0 1 100 655 Tm (Duree) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 655 Tm (25 ans) Tj\n");
    // `=` drawn at the fraction's mid-height, then the denominator `dt`
    // repositioned ~3 em back to the left — consecutive in the source, far
    // apart on the page.
    content.extend_from_slice(b"1 0 0 1 140 675 Tm (=) Tj\n");
    content.extend_from_slice(b"1 0 0 1 113 675 Tm (dt) Tj\n");
    content.extend_from_slice(b"1 0 0 1 305 675 Tm (Oui) Tj\n");
    content.extend_from_slice(b"ET");
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");

    let doc = PdfDocument::from_bytes(pdf).expect("open");
    for table in doc.extract_tables(0).expect("tables") {
        for row in &table.rows {
            for cell in &row.cells {
                assert!(
                    !cell.text.contains("=dt") && !cell.text.contains("dt="),
                    "a backward jump was fused into one token: {:?}",
                    cell.text
                );
            }
        }
    }
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

    // 453/1000 em for every WinAnsi code except the space (code 32, the first
    // entry), which is 170/1000. At 9 pt that is a 1.53 pt word space — see
    // the module docs: NARROWER than the 1.75 pt intra-word seams so no
    // constant separates the two, yet still above the cell builder's
    // font_size * 0.15 floor so a correct implementation keeps it.
    let off5 = pdf.len();
    let mut widths_v = vec!["453"; 95];
    widths_v[0] = "170";
    let widths = widths_v.join(" ");
    pdf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
             /Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 126 /Widths [{widths}] >>\nendobj\n"
        )
        .as_bytes(),
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
