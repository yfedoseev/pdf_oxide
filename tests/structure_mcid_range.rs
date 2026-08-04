//! `/K` marked-content ids that do not fit a `u32` must be rejected, not wrapped.
//!
//! A structure element's `/K` may carry integer children (bare MCIDs) and
//! marked-content reference dictionaries carrying `/MCID`. Both are `u32` in
//! the reading-order model. Truncating a wider or negative integer into that
//! `u32` makes a malformed id alias a real one, which silently corrupts the
//! page's reading order rather than dropping the bad entry.

use pdf_oxide::PdfDocument;

fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

fn stream(buf: &mut Vec<u8>, off: &mut [usize], id: usize, data: &[u8]) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

fn finish(buf: &mut Vec<u8>, off: &[usize], max_id: usize) {
    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", max_id + 1).as_bytes());
    for id in 1..=max_id {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", max_id + 1).as_bytes(),
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
}

/// A one-page tagged PDF whose single paragraph carries the given `/K` body.
/// The page's content stream declares one real marked-content id, MCID 5.
fn tagged_pdf_with_k(k_body: &str) -> Vec<u8> {
    let content = b"BT /F1 12 Tf\n\
        /P <</MCID 5>> BDC 1 0 0 1 60 700 Tm (Real content with MCID five) Tj EMC\n\
        ET\n";
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 9];
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(
        &mut buf,
        &mut off,
        1,
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R >>",
    );
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 6 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, content);
    obj(&mut buf, &mut off, 5, "<< >>");
    obj(&mut buf, &mut off, 6, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        8,
        &format!("<< /Type /StructElem /S /P /P 7 0 R /Pg 3 0 R /K {k_body} >>"),
    );
    finish(&mut buf, &off, 8);
    buf
}

fn reading_order(k_body: &str) -> Vec<u32> {
    let doc = PdfDocument::from_bytes(tagged_pdf_with_k(k_body)).expect("fixture parses");
    let tree = doc
        .structure_tree()
        .expect("structure tree read")
        .expect("structure tree present");
    pdf_oxide::structure::extract_reading_order(&tree, 0).expect("reading order")
}

#[test]
fn k_mcid_above_u32_max_is_skipped_not_wrapped() {
    // 4294967301 is 2^32 + 5. Truncated to u32 it becomes 5 and aliases the
    // page's real MCID 5, so the page reports MCID 5 twice.
    let order = reading_order("[4294967301 5]");
    assert_eq!(order, vec![5], "out-of-range /K child aliased onto the real MCID 5");
}

#[test]
fn negative_k_mcid_is_skipped_not_wrapped() {
    let order = reading_order("-1");
    assert!(
        order.is_empty(),
        "negative /K child was kept as {order:?} instead of being skipped"
    );
}

#[test]
fn negative_k_mcid_in_an_array_is_skipped_not_wrapped() {
    let order = reading_order("[-1 5]");
    assert_eq!(order, vec![5], "negative /K child was not skipped: {order:?}");
}

#[test]
fn out_of_range_mcr_mcid_is_skipped_not_wrapped() {
    // The marked-content reference dictionary form of the same defect.
    let order = reading_order("[<< /Type /MCR /Pg 3 0 R /MCID 4294967301 >> 5]");
    assert_eq!(
        order,
        vec![5],
        "out-of-range /MCID in an /MCR dict aliased onto the real MCID 5"
    );
}

#[test]
fn in_range_k_mcids_are_unaffected() {
    let order = reading_order("[5]");
    assert_eq!(order, vec![5]);
}
