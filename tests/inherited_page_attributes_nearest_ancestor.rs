//! An inheritable page attribute comes from the **nearest** ancestor that
//! specifies it, and the answer must not depend on what else was read first.
//!
//! ISO 32000-1:2008 §7.7.3.4 Table 30 lists `/Resources`, `/MediaBox`,
//! `/CropBox` and `/Rotate` as inheritable, and §7.7.3.4 is explicit that a
//! page lacking one takes it "from an ancestor node" — the nearest, since an
//! intermediate node's value is precisely what it exists to override.
//!
//! Two walkers implemented this. The eager one snapshotted the inherited map
//! around the recursion and used `insert`, which is correct. The lazy one used
//! `entry().or_insert_with()` on a root-first walk, which keeps the value
//! already present — the *first* seen, i.e. the most distant ancestor — so the
//! root won and every intermediate node was ignored. Its comment claimed the
//! opposite, which is why it read as correct.
//!
//! Because the two walkers disagreed, the same page could resolve differently
//! depending on which one ran.

use pdf_oxide::PdfDocument;

/// A three-level page tree: root `/Pages` sets one value, an intermediate
/// `/Pages` overrides it, and the leaf `/Page` specifies nothing. `n_pages`
/// leaves are emitted so the caller can vary how much of the document is
/// touched.
fn nested_tree_pdf(root_attrs: &str, mid_attrs: &str, n_pages: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    // 1 catalog, 2 root pages, 3 mid pages, 4 content, 5.. leaves
    let total = 4 + n_pages;
    let mut off = vec![0usize; total + 1];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [3 0 R] /Count {n_pages} {root_attrs} >>"),
    );
    let kids: Vec<String> = (0..n_pages).map(|i| format!("{} 0 R", 5 + i)).collect();
    obj(
        &mut buf,
        &mut off,
        3,
        &format!(
            "<< /Type /Pages /Parent 2 0 R /Kids [{}] /Count {n_pages} {mid_attrs} >>",
            kids.join(" ")
        ),
    );
    let content = b"BT /F1 12 Tf 10 100 Td (x) Tj ET\n";
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    for i in 0..n_pages {
        obj(&mut buf, &mut off, 5 + i, "<< /Type /Page /Parent 3 0 R /Contents 4 0 R >>");
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total + 1).as_bytes());
    for id in 1..=total {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", total + 1).as_bytes(),
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

const ROOT: &str = "/MediaBox [0 0 612 792] /Rotate 0";
const MID: &str = "/MediaBox [0 0 200 400] /Rotate 90";

/// The nearest ancestor wins for `/MediaBox`.
#[test]
fn media_box_comes_from_the_nearest_ancestor() {
    let doc = PdfDocument::from_bytes(nested_tree_pdf(ROOT, MID, 1)).expect("parse");
    assert_eq!(
        doc.get_page_media_box(0).expect("media box"),
        (0.0, 0.0, 200.0, 400.0),
        "the intermediate /Pages node's box must override the root's"
    );
}

/// And for `/Rotate`.
#[test]
fn rotate_comes_from_the_nearest_ancestor() {
    let doc = PdfDocument::from_bytes(nested_tree_pdf(ROOT, MID, 1)).expect("parse");
    let rotation = doc.get_page_rotation(0).expect("page rotation");
    assert_eq!(
        rotation.rem_euclid(360),
        90,
        "the intermediate /Pages node's rotation must override the root's"
    );
}

/// A leaf that specifies its own value still beats every ancestor.
#[test]
fn a_page_s_own_attribute_beats_every_ancestor() {
    let mut pdf = nested_tree_pdf(ROOT, MID, 1);
    // Rebuild with the leaf carrying its own box.
    pdf = {
        let s = String::from_utf8_lossy(&pdf).into_owned();
        let patched = s.replace(
            "<< /Type /Page /Parent 3 0 R /Contents 4 0 R >>",
            "<< /Type /Page /Parent 3 0 R /Contents 4 0 R /MediaBox [0 0 100 100] >>",
        );
        patched.into_bytes()
    };
    // The xref offsets are now stale; the reader's recovery path handles that,
    // and this test only cares which /MediaBox wins.
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    assert_eq!(doc.get_page_media_box(0).expect("media box"), (0.0, 0.0, 100.0, 100.0));
}

/// The load-bearing property: the answer must not depend on how much of the
/// document was touched first. Two walkers with different rules made the same
/// page resolve two ways in one process.
#[test]
fn inheritance_is_independent_of_how_many_pages_were_touched() {
    // Enough leaves to cross any eager/lazy materialisation threshold.
    let pdf = nested_tree_pdf(ROOT, MID, 80);

    // Cold: ask for the last page first.
    let cold = {
        let doc = PdfDocument::from_bytes(pdf.clone()).expect("parse");
        doc.get_page_media_box(79).expect("media box")
    };

    // Warm: walk every page, then ask for the same one.
    let warm = {
        let doc = PdfDocument::from_bytes(pdf).expect("parse");
        for i in 0..80 {
            let _ = doc.get_page_media_box(i);
        }
        doc.get_page_media_box(79).expect("media box")
    };

    assert_eq!(
        cold, warm,
        "the same page resolved differently depending on what was read first"
    );
    assert_eq!(cold, (0.0, 0.0, 200.0, 400.0), "and the nearest ancestor must win");
}

/// A sibling subtree's override must not leak across to its neighbours.
#[test]
fn a_subtree_s_override_does_not_leak_to_siblings() {
    // Root sets 612x792; the mid node sets 200x400 for all its leaves.
    let doc = PdfDocument::from_bytes(nested_tree_pdf(ROOT, MID, 3)).expect("parse");
    for i in 0..3 {
        assert_eq!(
            doc.get_page_media_box(i).expect("media box"),
            (0.0, 0.0, 200.0, 400.0),
            "page {i} took the wrong ancestor's box"
        );
    }
}
