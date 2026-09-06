//! A contents entry's number and its title are one row, not two columns.
//!
//! ISO 32000-1:2008 §9.4.4 gives a horizontal show-string one unbroken
//! interval on the X axis, so a run that starts left of a candidate corridor
//! and ends right of it proves the corridor is not empty there. A table of
//! contents makes that concrete: the deeper entries indent their number, which
//! opens a clean vertical channel between the number rail and the title
//! column, while the shallower entries — indented less, and typeset as one run
//! from the title through the dot leader to the page number — reach straight
//! across that channel.
//!
//! Read as a gutter, the page comes out as a rail of bare section numbers
//! followed, a column-run later, by titles with no numbers on them.
//!
//! The counter-case is the shape the guard must stay blind to: a full-measure
//! banner over two columns of prose. The banner also blankets the right
//! column, but it is the first and only thing printed on its row, so it is
//! furniture and the columns beneath it still read column-major.

use pdf_oxide::PdfDocument;

const RAIL_X_2: f32 = 82.0;
const RAIL_X_3: f32 = 92.0;
const TITLE_X_2: f32 = 122.0;
const TITLE_X_3: f32 = 142.0;
const MEASURE_RIGHT: f32 = 504.0;
const TOP: f32 = 710.0;
const LEADING: f32 = 11.76;
/// Three-level groups: one shallow entry, then several deep ones under it.
const GROUPS: usize = 6;
const DEEP_PER_GROUP: usize = 6;

fn show(c: &mut String, x: f32, y: f32, t: &str) {
    show_at(c, x, y, 10.0, t);
}

fn show_at(c: &mut String, x: f32, y: f32, size: f32, t: &str) {
    c.push_str(&format!("BT /F1 {size} Tf {x} {y} Td ({t}) Tj ET\n"));
}

fn shallow_number(g: usize) -> String {
    format!("10.{} ", g + 1)
}

fn deep_number(g: usize, i: usize) -> String {
    format!("10.{}.{} ", g + 1, i + 1)
}

fn shallow_title(g: usize) -> String {
    format!("Redirection {}xx{} 4{}", g + 3, ".".repeat(109), g)
}

fn deep_title(g: usize, i: usize) -> String {
    format!("Multiple Choices {}{} 4{}", ".".repeat(100), g, i)
}

fn pdf_with(content: String) -> Vec<u8> {
    let content = content.into_bytes();
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

fn contents_page() -> Vec<u8> {
    let mut c = String::new();
    // Running head: the page furniture a contents page carries above the list.
    show(&mut c, 72.0, 747.0, "RFC 2616 ");
    show(&mut c, 267.6, 747.0, "HTTP/1.1 ");
    show(&mut c, 460.6, 747.0, "June, 1999");
    let mut row = 0usize;
    for g in 0..GROUPS {
        let y = TOP - LEADING * row as f32;
        show(&mut c, RAIL_X_2, y, &shallow_number(g));
        show(&mut c, TITLE_X_2, y, &shallow_title(g));
        row += 1;
        for i in 0..DEEP_PER_GROUP {
            let y = TOP - LEADING * row as f32;
            show(&mut c, RAIL_X_3, y, &deep_number(g, i));
            show(&mut c, TITLE_X_3, y, &deep_title(g, i));
            row += 1;
        }
    }
    pdf_with(c)
}

/// Every deep entry keeps its number and its title on one line.
#[test]
fn test_contents_entry_keeps_its_number_and_its_title_on_one_line() {
    let doc = PdfDocument::from_bytes(contents_page()).expect("open");
    let out = doc.extract_text(0).expect("text");
    for g in 0..GROUPS {
        for i in 0..DEEP_PER_GROUP {
            let num = deep_number(g, i);
            let num = num.trim_end();
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with(num))
                .unwrap_or_else(|| panic!("`{num}` missing from output:\n{out}"));
            assert!(
                line.contains("Multiple Choices"),
                "the number rail was cut away from the titles: `{num}` reads as \
                 `{line}`, with its title emitted elsewhere. A run that reaches \
                 across a candidate corridor proves the corridor is not empty \
                 (ISO 32000-1 §9.4.4). Got:\n{out}"
            );
        }
    }
}

/// The page only tests the corridor if the corridor is where this file says it
/// is: the deep numbers must stop short of the deep titles, and the shallow
/// entries' runs must reach across the resulting channel.
#[test]
fn test_fixture_has_a_channel_the_shallow_entries_reach_across() {
    let doc = PdfDocument::from_bytes(contents_page()).expect("open");
    let spans = doc.extract_spans(0).expect("spans");
    let deep_rail_right = spans
        .iter()
        .filter(|s| s.text.starts_with("10.") && s.text.matches('.').count() >= 2)
        .map(|s| s.bbox.x + s.bbox.width)
        .fold(f32::MIN, f32::max);
    assert!(
        deep_rail_right < TITLE_X_3 - 10.0,
        "the deep number rail ends at {deep_rail_right}, leaving no channel \
         before the titles at {TITLE_X_3}"
    );
    let mut crossing = 0usize;
    let mut blanket = 0usize;
    for s in &spans {
        let (x0, x1) = (s.bbox.x, s.bbox.x + s.bbox.width);
        if x0 < deep_rail_right + 12.0 && x1 > TITLE_X_3 {
            crossing += 1;
        }
        if x1 >= MEASURE_RIGHT - 20.0 {
            blanket += 1;
        }
    }
    assert!(
        crossing >= 3,
        "only {crossing} runs reach across the channel; the fixture needs the \
         shallow entries to cross it"
    );
    assert!(
        blanket >= GROUPS * (DEEP_PER_GROUP + 1),
        "only {blanket} runs reach the measure; the dot leaders must run to the \
         page-number column"
    );
}

// --- counter-case: two columns of prose still read column-major ---

const BODY: &str = "an ordinary sentence of running body text here";
const COL_LEFT: f32 = 72.0;
/// A body line measures 210.11pt at 9pt Helvetica; the gutter is 8pt.
const COL_RIGHT: f32 = 290.11;
const PROSE_ROWS: usize = 14;

fn two_column_body() -> Vec<u8> {
    let mut c = String::new();
    for r in 0..PROSE_ROWS {
        let y = 700.0 - 14.0 * r as f32;
        show_at(&mut c, COL_LEFT, y, 9.0, &format!("LR{:02} {BODY}", r + 1));
        show_at(&mut c, COL_RIGHT, y, 9.0, &format!("RL{:02} {BODY}", r + 1));
    }
    pdf_with(c)
}

/// The guard that keeps a labelled row whole must not refuse an ordinary
/// column cut: two columns of prose, nothing reaching across the gutter, still
/// read one column at a time.
#[test]
fn two_columns_of_prose_still_read_column_major() {
    let doc = PdfDocument::from_bytes(two_column_body()).expect("open");
    let out = doc.extract_text(0).expect("text");
    let at = |tag: &str| {
        out.find(tag)
            .unwrap_or_else(|| panic!("`{tag}` missing from output:\n{out}"))
    };
    assert!(
        at(&format!("LR{PROSE_ROWS:02}")) < at("RL01"),
        "the right column is spliced into the left column — an empty gutter is \
         still a gutter. Got:\n{out}"
    );
}
