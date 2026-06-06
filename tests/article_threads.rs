//! #458: article-thread (`/Threads`) parsing — ISO 32000-1:2008 §12.4.3.
//!
//! Hand-builds a minimal PDF with one thread of three beads (a circular
//! doubly-linked list) spanning two pages, and asserts the parser walks the
//! chain in `/N` order, resolves each bead's `/P` page and `/R` rectangle, and
//! terminates on the circular wrap. No external fixtures.

use pdf_oxide::structure::parse_article_threads;
use pdf_oxide::PdfDocument;

fn obj(buf: &mut Vec<u8>, offsets: &mut [usize], id: usize, body: &str) {
    offsets[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
    buf.extend_from_slice(body.as_bytes());
    buf.extend_from_slice(b"\nendobj\n");
}

/// One thread, three beads (A→B→C→A), beads A/B on page 0 and C on page 1.
fn threaded_pdf() -> Vec<u8> {
    let mut buf = Vec::new();
    let mut off = vec![0usize; 9]; // ids 1..=8
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R /Threads [5 0 R] >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>");
    obj(&mut buf, &mut off, 3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>");
    obj(&mut buf, &mut off, 4, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Thread /F 6 0 R /I << /Title (Cover Story) >> >>",
    );
    // Bead A (page 0)
    obj(
        &mut buf,
        &mut off,
        6,
        "<< /Type /Bead /T 5 0 R /N 7 0 R /V 8 0 R /P 3 0 R /R [50 600 300 700] >>",
    );
    // Bead B (page 0)
    obj(
        &mut buf,
        &mut off,
        7,
        "<< /Type /Bead /N 8 0 R /V 6 0 R /P 3 0 R /R [320 600 560 700] >>",
    );
    // Bead C (page 1), /N wraps back to A
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /Bead /N 6 0 R /V 7 0 R /P 4 0 R /R [50 600 560 700] >>",
    );

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn parses_thread_chain_in_order_and_terminates() {
    let doc = PdfDocument::from_bytes(threaded_pdf()).unwrap();
    let threads = parse_article_threads(&doc);

    assert_eq!(threads.len(), 1, "exactly one thread");
    let t = &threads[0];
    assert_eq!(t.title.as_deref(), Some("Cover Story"));
    assert_eq!(t.beads.len(), 3, "three beads; circular /N must terminate");

    // Chain order A, B, C.
    assert_eq!(t.beads[0].page_index, 0);
    assert_eq!(t.beads[1].page_index, 0);
    assert_eq!(t.beads[2].page_index, 1);

    // Bead A rect [50 600 300 700] -> x=50 y=600 w=250 h=100.
    let a = &t.beads[0].rect;
    assert!((a.x - 50.0).abs() < 1e-3 && (a.y - 600.0).abs() < 1e-3);
    assert!((a.width - 250.0).abs() < 1e-3 && (a.height - 100.0).abs() < 1e-3);

    // Bead C spans the full text column on page 1.
    let c = &t.beads[2].rect;
    assert!((c.width - 510.0).abs() < 1e-3);
}

#[test]
fn document_without_threads_yields_none() {
    // The form fixture builder elsewhere proves threadless docs parse to empty;
    // here a trivially-threadless catalog must give zero threads.
    let mut buf = Vec::new();
    let mut off = vec![0usize; 4];
    buf.extend_from_slice(b"%PDF-1.7\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(&mut buf, &mut off, 3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 4\n0000000000 65535 f \n");
    for id in 1..=3 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());

    let doc = PdfDocument::from_bytes(buf).unwrap();
    assert!(parse_article_threads(&doc).is_empty());
}
