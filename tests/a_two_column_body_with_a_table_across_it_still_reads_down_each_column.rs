//! A two-column body with a full-measure ruled table across it still reads
//! down each column.
//!
//! A regulation page sets its text in two narrow columns and interrupts them
//! with a table whose rules and caption run the full measure. The table is
//! genuine and is detected. But the page then met a predicate that decides
//! whether a page's multi-column signal comes from its table alone by keeping
//! the MINIMUM left edge of each row band, and on a two-column page that
//! minimum is the left column's margin on every band — the right column is
//! never seen. It declared the signal tabular and the page was sorted
//! row-aware, reading the two columns straight across:
//!
//! ```text
//! must be designed and constructed to  | warning devices must be config-
//! prevent contamination of the ...     | ured so that they may not be ...
//! ```
//!
//! came out as `must be config-` / `prevent contamination of the` / `ured so
//! that`, so the wrapped word can never be rejoined and `configured` is gone
//! from the page.
//!
//! The column branches decline such a page for a good reason — the table's
//! rows and its centred caption straddle the gutter, so there is no clean
//! corridor — and the untagged fallback is the content stream's own order
//! (ISO 32000-1:2008 §14.8.2.3, page content order), which a two-column page
//! writes column-major. The row-aware override is withheld whenever the prose
//! outside the tables starts at exactly two column positions.
//!
//! The counter-case is a single-column page with the same table: its cells
//! still trip the multi-column detector, and the row-aware sort is still the
//! right answer there, so the override must survive for it.

use pdf_oxide::PdfDocument;

/// Helvetica advance widths at 1000 units/em for the glyphs used here.
fn advance(c: char) -> f32 {
    match c {
        'a' | 'c' | 'e' | 'g' | 'n' | 'o' | 'p' | 'q' | 's' | 'u' | 'v' | 'x' | 'y' | 'z' => 556.0,
        'b' | 'd' | 'h' | 'k' => 556.0,
        'f' | 'r' | 't' => 278.0,
        'i' | 'j' | 'l' => 222.0,
        'm' => 833.0,
        'w' => 722.0,
        'A' | 'B' | 'E' | 'P' | 'R' | 'S' | 'T' | 'V' => 667.0,
        'C' | 'D' | 'H' | 'N' | 'U' | 'M' | 'O' => 722.0,
        'I' | 'F' | 'L' => 556.0,
        '0'..='9' => 556.0,
        '(' | ')' | '-' | '.' | ',' | ' ' | ':' | '/' => 278.0,
        _ => 556.0,
    }
}

fn width_of(s: &str, size: f32) -> f32 {
    s.chars().map(|c| advance(c) / 1000.0 * size).sum()
}

const SIZE: f32 = 8.0;
const LEAD: f32 = 9.0;
/// A paragraph break on the page this models is two tenths of a point deeper
/// than a line, so each column's baselines drift off the other's grid a little
/// at every paragraph, as the measured page's do.
const PARA_LEAD: f32 = 9.2;
/// The column measure as the ink fills it, in the proportions the measured
/// page has at its 8 pt size: a wrap-hyphen line ends at the measure exactly,
/// a line that ends in a word carries the justifier's trailing space, widened
/// by `Tw`, a few points past it, and the right column starts 12 pt after the
/// measure — on the page 132 + 168 = 300 against 312, with the trailing
/// spaces reaching 303–307. Helvetica sets the same words about a fifth
/// narrower than the page's face, so the measure is 140 here and the right
/// column starts at 284, keeping every gap the predicates measure.
const MEASURE: f32 = 140.0;
const LEFT_X: f32 = 132.0;
const RIGHT_X: f32 = 284.0;

/// One justified line. Word spacing is spread so the ink is exactly `MEASURE`
/// wide; a line that does not end at a wrap hyphen also carries the
/// justifier's trailing space, stretched by the same `Tw`. Every line below is
/// worded so that `Tw` stays under 3.2 pt, as a justifier keeps it.
fn line(x: f32, y: f32, text: &str) -> String {
    let spaces = text.matches(' ').count() as f32;
    let tw = ((MEASURE - width_of(text, SIZE)) / spaces).max(0.0);
    let text = if text.ends_with('-') { text.to_string() } else { format!("{text} ") };
    format!("BT /F1 {SIZE} Tf {tw:.3} Tw 1 0 0 1 {x:.2} {y:.2} Tm ({text}) Tj ET\n")
}

/// A column: paragraphs of lines, set on the leading with a paragraph break
/// between them. Wrap hyphens end their lines, as they do on the page.
const LEFT_ABOVE: &[&[&str]] = &[
    &[
        "(2) Except as otherwise provided in",
        "paragraph (b) of this section, each one",
        "respirator must be fitted with a sub-",
        "stantial, durable container bearing the",
        "markings which show the applicant's",
        "name, and the type of respirator it con-",
        "tains, and all appropriate approval la-",
        "bels for the class of respirator in it.",
    ],
    &[
        "(3) Containers for respirators may also",
        "provide for storage of more than one",
        "respirator; however, such containers",
        "must be designed and constructed to",
        "prevent contamination of the respira-",
        "tors which are not removed, and to pre-",
        "vent damage to respirators in transit.",
    ],
];
const RIGHT_ABOVE: &[&[&str]] = &[
    &[
        "(b) Blowers must be so designed to",
        "achieve the air flow rates required at",
        "the pressure the facepiece needs, and",
        "its warning devices must be config-",
        "ured so that they may not be de-ener-",
        "gized by the wearer while the blower",
        "runs at the rated flow of the facepiece.",
        "The design must indicate the flow rate.",
        "The blower assembly may not be ener-",
        "gized while the harness is removed.",
    ],
    &[
        "(c) Each blower must also include a",
        "low-flow warning. It must be actively",
        "and readily indicate when the flow rate",
        "has fallen below the rated minimum.",
        "The warning must be audible or visible",
    ],
];
const LEFT_BELOW: &[&[&str]] = &[&[
    "(a) Dry exhalation valves and the valve",
    "seats will be subjected to a suction of",
    "25 mm water-column height while held",
    "in a normal operating position, and the",
    "leakage measured must not then ex-",
    "ceed the limits set out in paragraph (c).",
]];
const RIGHT_BELOW: &[&[&str]] = &[&[
    "(b) Leakage between the valve and its",
    "seat may not exceed 30 mL per minute",
    "when held in a normal operating posi-",
    "tion and closed, and the test must then",
    "be repeated after the valve has been",
    "cycled ten times under the same load.",
]];

/// The full-measure ruled table between the paragraphs: a centred caption on
/// the gutter, three columns of rules from the left margin to the right
/// column's edge, dot-leader stubs that cross the gutter, and numeric cells.
fn table(top: f32) -> String {
    let (x0, x1, x2, x3) = (LEFT_X, 312.0, 372.0, RIGHT_X + MEASURE);
    let row = 12.0;
    let y0 = top;
    let y1 = top - row;
    let y2 = top - 2.0 * row;
    let y3 = top - 3.0 * row;
    let mut s = String::new();
    // The caption and its unit line are centred on the page, so both sit on
    // the gutter; the stubs' dot leaders run to the first column rule, well
    // past it. Four runs straddle the gutter, as the regulation's table has.
    for (dy, size, caption) in [(15.0, SIZE, "MAXIMUM RESISTANCE"), (6.0, 6.0, "[mm water-column height]")] {
        let cx = (x0 + x3) * 0.5 - width_of(caption, size) * 0.5;
        s += &format!("BT /F1 {size} Tf 1 0 0 1 {cx:.2} {:.2} Tm ({caption}) Tj ET\n", top + dy);
    }
    s += "0.5 w\n";
    for x in [x0, x1, x2, x3] {
        s += &format!("{x:.2} {y3:.2} m {x:.2} {y0:.2} l S\n");
    }
    for y in [y0, y1, y2, y3] {
        s += &format!("{x0:.2} {y:.2} m {x3:.2} {y:.2} l S\n");
    }
    let cell = |x: f32, y: f32, t: &str| {
        format!("BT /F1 {SIZE} Tf 1 0 0 1 {:.2} {:.2} Tm ({t}) Tj ET\n", x + 4.0, y + 3.0)
    };
    s += &cell(x0, y1, "Respirator type");
    s += &cell(x1, y1, "Initial");
    s += &cell(x2, y1, "Final");
    let stub = |label: &str| {
        let dots = ((x1 - x0 - 8.0 - width_of(label, SIZE)) / width_of(".", SIZE)) as usize;
        format!("{label} {}", ".".repeat(dots))
    };
    s += &cell(x0, y2, &stub("Non-Powered (N, R, and P)"));
    s += &cell(x1, y2, "35");
    s += &cell(x2, y2, "N/A");
    s += &cell(x0, y3, &stub("Powered (tight fitting)"));
    s += &cell(x1, y3, "50");
    s += &cell(x2, y3, "70");
    s
}

fn column(x: f32, top: f32, paragraphs: &[&[&str]]) -> String {
    let mut y = top;
    let mut s = String::new();
    for (i, para) in paragraphs.iter().enumerate() {
        if i > 0 {
            y -= PARA_LEAD - LEAD;
        }
        for t in para.iter() {
            s += &line(x, y, t);
            y -= LEAD;
        }
    }
    s
}

/// The page as the regulation sets it: both columns above the table, the
/// table, both columns below it — content-stream order is column-major within
/// each band, as a two-column page's stream is.
fn two_column_page() -> Vec<u8> {
    let mut content = String::new();
    content += &column(LEFT_X, 700.0, LEFT_ABOVE);
    content += &column(RIGHT_X, 700.0, RIGHT_ABOVE);
    content += &table(540.0);
    content += &column(LEFT_X, 480.0, LEFT_BELOW);
    content += &column(RIGHT_X, 480.0, RIGHT_BELOW);
    build_page(&content)
}

/// The counter-case: the same table on a single-column page. The table's
/// column-aligned cells are the page's only multi-column signal.
fn single_column_page() -> Vec<u8> {
    let mut content = String::new();
    let wide = |y: f32, t: &str| {
        format!("BT /F1 {SIZE} Tf 1 0 0 1 {LEFT_X:.2} {y:.2} Tm ({t}) Tj ET\n")
    };
    let prose = [
        "The blower assembly is tested against the resistance limits in the table below,",
        "measured in millimetres of water column at the rated flow of the respirator,",
        "with the facepiece sealed to the test head and the exhalation valve closed.",
        "Each reading is taken after the blower has run for at least two minutes,",
        "and the higher of two consecutive readings is the value that is recorded.",
        "A respirator that exceeds either limit fails the test and is not approved.",
        "The table gives the limit for each class at the start and the end of a run,",
        "and applies to every model submitted under this subpart for approval.",
    ];
    for (i, t) in prose.iter().enumerate() {
        content += &wide(600.0 - i as f32 * LEAD, t);
    }
    content += &table(500.0);
    let after = [
        "Where a respirator is fitted with more than one blower, each blower is",
        "tested on its own and the table applies to each of them separately.",
        "The applicant records every reading on the form provided for the purpose.",
        "Readings are kept with the approval file for the life of the approval.",
    ];
    for (i, t) in after.iter().enumerate() {
        content += &wide(440.0 - i as f32 * LEAD, t);
    }
    build_page(&content)
}

/// The page's text with the justifier's trailing spaces trimmed from each
/// line, so line-boundary assertions read as the lines do.
fn text_of(pdf: Vec<u8>) -> String {
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let text = doc.extract_text(0).expect("text");
    text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n")
}

#[test]
fn a_word_wrapped_in_the_right_column_is_whole_when_a_table_crosses_the_page() {
    let text = text_of(two_column_page());
    // Wrap hyphens rejoined and lines run together, so a phrase reads across
    // a line break the same whether the extractor dehyphenated it or not.
    let rejoined = text.replace("-\n", "").replace('\n', " ");
    assert!(
        rejoined.contains("must be configured so that they"),
        "the right column's wrapped word is read down its own column, not across \
         the gutter into the left column's next line. Got:\n{text}"
    );
    assert!(
        rejoined.contains("constructed to prevent contamination of the respirators"),
        "the left column reads down as well. Got:\n{text}"
    );
}

#[test]
fn every_word_of_both_columns_is_still_on_the_page() {
    let text = text_of(two_column_page());
    for word in ["Containers", "transit", "wearer", "Leakage", "operating", "cycled", "Initial", "Final"] {
        assert!(text.contains(word), "{word:?} is missing from:\n{text}");
    }
}

/// The counter-direction: a single-column page whose only multi-column
/// signal is the table keeps its row-aware order — the prose above the table
/// reads first, then the table, then the prose below it.
#[test]
fn a_single_column_page_with_the_same_table_still_reads_top_to_bottom() {
    let text = text_of(single_column_page());
    let at = |s: &str| text.find(s).unwrap_or_else(|| panic!("{s:?} missing from:\n{text}"));
    assert!(at("not approved") < at("Respirator type"), "prose above the table first:\n{text}");
    assert!(at("Final") < at("more than one blower"), "prose below the table last:\n{text}");
    assert!(
        text.contains("measured in millimetres of water column at the rated flow"),
        "a full-measure line is one line:\n{text}"
    );
}

/// Minimal single-content-stream page writer.
fn build_page(content: &str) -> Vec<u8> {
    let content = content.as_bytes().to_vec();
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
