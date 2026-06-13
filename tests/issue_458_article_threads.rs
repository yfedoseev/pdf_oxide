//! Issue #458 — article-thread (`/Threads`) parsing (ISO 32000-1:2008 §12.4.3).
//!
//! Article threads chain logically-connected content ("beads") across columns
//! and pages in author-supplied reading order — the canonical signal for
//! untagged legacy magazine layouts. This test hand-builds a two-page document
//! with a single three-bead thread that flows
//!   page-1 right column → page-2 left column → page-2 right column
//! and asserts the parser recovers the beads in `/N` order with correct
//! page indices and rectangles. PDF is hand-built — no third-party fixture.

use pdf_oxide::structure::parse_article_threads;
use pdf_oxide::PdfDocument;

/// Two pages (612×792). One thread, three beads chained via `/N` (circular
/// doubly-linked list per §12.4.3): bead A on page 1 (right column), bead B on
/// page 2 (left column), bead C on page 2 (right column).
fn threaded_magazine_pdf() -> Vec<u8> {
    let page_content = b"BT /F1 11 Tf 1 0 0 1 60 700 Tm (column text) Tj ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 12];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    // 1 Catalog (with /Threads), 2 Pages, 3/5 Pages, 4/6 contents, 7 Font,
    // 8 Thread dict, 9/10/11 Beads.
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R /Threads [8 0 R] >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R /B [9 0 R] >>",
    );
    stream(&mut buf, &mut off, 4, page_content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R /B [10 0 R 11 0 R] >>",
    );
    stream(&mut buf, &mut off, 6, page_content);
    obj(&mut buf, &mut off, 7, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    // Thread dict: /F → first bead (9).
    obj(&mut buf, &mut off, 8, "<< /Type /Thread /F 9 0 R /I << /Title (Feature) >> >>");
    // Bead A — page 1 right column.
    obj(
        &mut buf,
        &mut off,
        9,
        "<< /Type /Bead /T 8 0 R /N 10 0 R /V 11 0 R /P 3 0 R /R [310 60 560 740] >>",
    );
    // Bead B — page 2 left column.
    obj(
        &mut buf,
        &mut off,
        10,
        "<< /Type /Bead /T 8 0 R /N 11 0 R /V 9 0 R /P 5 0 R /R [40 60 290 740] >>",
    );
    // Bead C — page 2 right column.
    obj(
        &mut buf,
        &mut off,
        11,
        "<< /Type /Bead /T 8 0 R /N 9 0 R /V 10 0 R /P 5 0 R /R [310 60 560 740] >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 12\n0000000000 65535 f \n");
    for id in 1..=11 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 12 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// One page, two columns, whose thread deliberately reads the **right** column
/// before the **left** — divergent from naive geometric (left-to-right) order —
/// so the activation gate fires and `extract_words` follows the thread. Bead A =
/// right column, bead B = left column (in `/N` order).
fn thread_right_before_left_pdf() -> Vec<u8> {
    // Left column word at x≈60/y=700; right column word at x≈340/y=680 (a
    // different baseline so the two stay distinct words). Geometric order reads
    // LEFTWORD (higher) first; the thread reads the right column (RIGHTWORD) first.
    let content = b"BT /F1 11 Tf 1 0 0 1 60 700 Tm (LEFTWORD) Tj ET\n\
        BT /F1 11 Tf 1 0 0 1 340 680 Tm (RIGHTWORD) Tj ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 9];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R /Threads [6 0 R] >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /B [7 0 R 8 0 R] >>",
    );
    stream(&mut buf, &mut off, 4, content);
    obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    // Thread: /F → bead A (the RIGHT column, read first).
    obj(&mut buf, &mut off, 6, "<< /Type /Thread /F 7 0 R >>");
    // Bead A — right column (read first per the thread).
    obj(
        &mut buf,
        &mut off,
        7,
        "<< /Type /Bead /T 6 0 R /N 8 0 R /V 8 0 R /P 3 0 R /R [320 60 560 740] >>",
    );
    // Bead B — left column (read second).
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /Bead /T 6 0 R /N 7 0 R /V 7 0 R /P 3 0 R /R [40 60 300 740] >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn thread_overrides_geometric_order_in_extract_words() {
    let doc = PdfDocument::from_bytes(thread_right_before_left_pdf()).unwrap();
    let words = doc.extract_words(0).unwrap();
    let order: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    let right = order.iter().position(|w| w.contains("RIGHTWORD"));
    let left = order.iter().position(|w| w.contains("LEFTWORD"));
    let (right, left) = (right.expect("right word"), left.expect("left word"));
    assert!(
        right < left,
        "article thread must read the right column before the left: {order:?}"
    );
}

#[test]
fn parses_cross_page_three_bead_thread() {
    let doc = PdfDocument::from_bytes(threaded_magazine_pdf()).unwrap();
    let threads = parse_article_threads(&doc);

    assert_eq!(threads.len(), 1, "exactly one thread declared");
    let thread = &threads[0];
    assert_eq!(thread.title.as_deref(), Some("Feature"));
    assert_eq!(thread.beads.len(), 3, "three beads in the chain");

    // Beads in /N order: A (page 0), B (page 1), C (page 1).
    assert_eq!(thread.beads[0].page_index, 0, "bead A on page 1");
    assert_eq!(thread.beads[1].page_index, 1, "bead B on page 2");
    assert_eq!(thread.beads[2].page_index, 1, "bead C on page 2");

    // Bead A is the right column (llx ≈ 310); bead B is the left column (llx ≈ 40).
    assert!(thread.beads[0].rect.x >= 300.0, "bead A is the right column");
    assert!(thread.beads[1].rect.x < 300.0, "bead B is the left column");
}
