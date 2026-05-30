//! Trustworthy-structure-tree reading-order gate (issue #608, fix-issues item 6).
//!
//! A `/StructTreeRoot` encodes the producer's *logical structure order* — a
//! depth-first traversal of the tag hierarchy — which ISO 32000-1:2008
//! §14.8.2.3.1 makes authoritative for reading order, *unless* the document
//! advertises `/MarkInfo /Suspects true` (the `/TagSuspect /Ordering` signal),
//! in which case the page content (geometric) order must be used instead.
//!
//! These fixtures build a one-page PDF in which the structure tree's DFS order
//! (`Bravo`, `Alpha`) deliberately **differs** from the on-page geometric order
//! (`Alpha` at y=700 above `Bravo` at y=600). Toggling `/MarkInfo` lets us
//! assert the predicate `prefers_structure_reading_order()` and the routing of
//! `extract_text` / `to_plain_text` / `to_markdown` across the precedence table.
//!
//! All fixtures are synthesised in memory (`PdfDocument::from_bytes`); no real
//! or MPL-licensed PDFs are used.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

/// Minimal PDF assembler that tracks byte offsets for a clean xref table.
struct PdfBuilder {
    buf: Vec<u8>,
    offsets: Vec<usize>, // index = object number; [0] is the free entry
}

impl PdfBuilder {
    fn new() -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");
        PdfBuilder {
            buf,
            offsets: vec![0],
        }
    }

    fn obj(&mut self, n: usize, body: &str) {
        while self.offsets.len() <= n {
            self.offsets.push(0);
        }
        self.offsets[n] = self.buf.len();
        self.buf
            .extend_from_slice(format!("{} 0 obj\n", n).as_bytes());
        self.buf.extend_from_slice(body.as_bytes());
        self.buf.extend_from_slice(b"\nendobj\n");
    }

    fn stream_obj(&mut self, n: usize, data: &str) {
        let body = format!("<< /Length {} >>\nstream\n{}\nendstream", data.len(), data);
        self.obj(n, &body);
    }

    fn finish(mut self, root: usize) -> Vec<u8> {
        let xref_offset = self.buf.len();
        let count = self.offsets.len();
        self.buf
            .extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
        self.buf.extend_from_slice(b"0000000000 65535 f \r\n");
        for &off in &self.offsets[1..] {
            self.buf
                .extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
        }
        self.buf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root {} 0 R >>\nstartxref\n{}\n%%EOF\n",
                count, root, xref_offset
            )
            .as_bytes(),
        );
        self.buf
    }
}

/// Build a one-page PDF whose structure-tree DFS order is `Bravo, Alpha` while
/// the geometric order is `Alpha, Bravo`.
///
/// * `tree_ref`  — catalog references `/StructTreeRoot` (objects 6–8, 10).
/// * `mark_info` — when `Some(body)`, catalog references `/MarkInfo` (obj 9)
///   with that dictionary body; when `None`, no `/MarkInfo` is present.
fn build_pdf(tree_ref: bool, mark_info: Option<&str>) -> Vec<u8> {
    let mut b = PdfBuilder::new();

    // 1: Catalog
    let mut catalog = String::from("<< /Type /Catalog /Pages 2 0 R");
    if tree_ref {
        catalog.push_str(" /StructTreeRoot 6 0 R");
    }
    if mark_info.is_some() {
        catalog.push_str(" /MarkInfo 9 0 R");
    }
    catalog.push_str(" >>");
    b.obj(1, &catalog);

    // 2: Pages
    b.obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");

    // 3: Page (StructParents 0 keys the parent tree)
    b.obj(
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> /StructParents 0 >>",
    );

    // 4: Content stream — Alpha (MCID 0) at y=700, Bravo (MCID 1) at y=600.
    let content = "/P << /MCID 0 >> BDC BT /F1 12 Tf 100 700 Td (Alpha) Tj ET EMC\n\
                   /P << /MCID 1 >> BDC BT /F1 12 Tf 100 600 Td (Bravo) Tj ET EMC";
    b.stream_obj(4, content);

    // 5: Font
    b.obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

    // 6: StructTreeRoot — /K lists Bravo (obj 7) BEFORE Alpha (obj 8), so DFS
    //    order is MCID 1 then MCID 0, the reverse of geometric order.
    b.obj(6, "<< /Type /StructTreeRoot /K [7 0 R 8 0 R] /ParentTree 10 0 R >>");
    // 7: P element for MCID 1 (Bravo)
    b.obj(7, "<< /Type /StructElem /S /P /P 6 0 R /Pg 3 0 R /K 1 >>");
    // 8: P element for MCID 0 (Alpha)
    b.obj(8, "<< /Type /StructElem /S /P /P 6 0 R /Pg 3 0 R /K 0 >>");

    // 9: MarkInfo (body varies per fixture; emitted only as referenced)
    b.obj(9, mark_info.unwrap_or("<< /Marked false >>"));

    // 10: ParentTree — Nums[0] indexed by MCID: 0 -> Alpha(8), 1 -> Bravo(7).
    b.obj(10, "<< /Nums [0 [8 0 R 7 0 R]] >>");

    b.finish(1)
}

/// Returns true when `first` appears before `second` in `text` (both required).
fn precedes(text: &str, first: &str, second: &str) -> bool {
    match (text.find(first), text.find(second)) {
        (Some(a), Some(b)) => a < b,
        _ => false,
    }
}

fn assert_both_present(text: &str) {
    assert!(
        text.contains("Alpha") && text.contains("Bravo"),
        "both Alpha and Bravo must extract; got: {text:?}"
    );
}

// ── F1: trustworthy marked (Marked true, Suspects false) → structure order ──
//
// All four text accessors must honour the structure tree (Bravo before Alpha):
// `extract_text` and `to_plain_text` assemble directly from MCID order;
// `to_markdown` and `to_html` route through MCID-driven reading order.
#[test]
fn f1_trustworthy_marked_uses_structure_order() {
    let pdf = build_pdf(true, Some("<< /Marked true /Suspects false >>"));
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let opts = ConversionOptions::default();

    assert!(
        doc.prefers_structure_reading_order(),
        "trustworthy marked+non-suspect tree must be preferred"
    );

    for (name, out) in [
        ("extract_text", doc.extract_text(0).unwrap()),
        ("to_plain_text", doc.to_plain_text(0, &opts).unwrap()),
        ("to_markdown", doc.to_markdown(0, &opts).unwrap()),
        ("to_html", doc.to_html(0, &opts).unwrap()),
    ] {
        assert_both_present(&out);
        assert!(
            precedes(&out, "Bravo", "Alpha"),
            "{name} must use structure order (Bravo before Alpha); got: {out:?}"
        );
    }
}

// ── F2: suspect (Marked true, Suspects true) → geometric order ──
#[test]
fn f2_suspect_falls_back_to_geometric_order() {
    let pdf = build_pdf(true, Some("<< /Marked true /Suspects true >>"));
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let opts = ConversionOptions::default();

    assert!(
        !doc.prefers_structure_reading_order(),
        "a /Suspects true tree must NOT be preferred (§14.8.2.3.1)"
    );

    let et = doc.extract_text(0).unwrap();
    assert_both_present(&et);
    assert!(
        precedes(&et, "Alpha", "Bravo"),
        "suspect doc: extract_text must use geometric order (Alpha before Bravo); got: {et:?}"
    );

    let pt = doc.to_plain_text(0, &opts).unwrap();
    assert!(
        precedes(&pt, "Alpha", "Bravo"),
        "suspect doc: to_plain_text must be geometric (Alpha before Bravo); got: {pt:?}"
    );

    let md = doc.to_markdown(0, &opts).unwrap();
    assert!(
        precedes(&md, "Alpha", "Bravo"),
        "suspect doc: to_markdown must be geometric (Alpha before Bravo); got: {md:?}"
    );
}

// ── F3: PDF-1.4-style, /StructTreeRoot present but NO /MarkInfo → trustworthy ──
#[test]
fn f3_catalog_struct_tree_without_mark_info_is_trustworthy() {
    let pdf = build_pdf(true, None);
    let doc = PdfDocument::from_bytes(pdf).unwrap();

    assert!(
        doc.prefers_structure_reading_order(),
        "catalog /StructTreeRoot with no /MarkInfo is a valid tagged PDF (§7.7.2)"
    );

    let et = doc.extract_text(0).unwrap();
    assert_both_present(&et);
    assert!(
        precedes(&et, "Bravo", "Alpha"),
        "no-MarkInfo tagged PDF must use structure order; got: {et:?}"
    );
}

// ── F4: Marked false + tree present → preserves legacy behaviour (trustworthy) ──
//
// The legacy inline gate loaded the tree whenever `marked || catalog-has-root`,
// so a `/Marked false` document with a `/StructTreeRoot` was already struct-
// ordered. The predicate preserves that exactly (only `/Suspects true` changes
// behaviour), guaranteeing byte-identity for existing corpora.
#[test]
fn f4_marked_false_with_tree_preserves_legacy_structure_order() {
    let pdf = build_pdf(true, Some("<< /Marked false >>"));
    let doc = PdfDocument::from_bytes(pdf).unwrap();

    assert!(
        doc.prefers_structure_reading_order(),
        "Marked false + /StructTreeRoot stayed struct-ordered under the legacy gate"
    );

    let et = doc.extract_text(0).unwrap();
    assert_both_present(&et);
    assert!(
        precedes(&et, "Bravo", "Alpha"),
        "Marked false + tree must keep structure order (legacy parity); got: {et:?}"
    );
}

// ── F5: untagged (no /StructTreeRoot) → geometric, predicate false ──
#[test]
fn f5_untagged_uses_geometric_order() {
    let pdf = build_pdf(false, None);
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let opts = ConversionOptions::default();

    assert!(
        !doc.prefers_structure_reading_order(),
        "a document with no /StructTreeRoot must not prefer structure order"
    );

    let et = doc.extract_text(0).unwrap();
    assert_both_present(&et);
    assert!(
        precedes(&et, "Alpha", "Bravo"),
        "untagged doc: extract_text must be geometric (Alpha before Bravo); got: {et:?}"
    );

    let pt = doc.to_plain_text(0, &opts).unwrap();
    assert!(
        precedes(&pt, "Alpha", "Bravo"),
        "untagged doc: to_plain_text must be geometric (Alpha before Bravo); got: {pt:?}"
    );
}
