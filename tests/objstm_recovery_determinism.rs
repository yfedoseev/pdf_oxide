//! Recovering a damaged file must pick the same page tree every time.
//!
//! `recover_from_objstms` keeps the first `/Type /Pages` it meets while walking
//! what an object stream yields. That walk has to be ordered by object number:
//! `parse_object_stream` returns a `HashMap`, and Rust seeds each map instance
//! separately, so an unordered walk lets the same bytes recover a different
//! page tree between reconstructions inside one process.

use pdf_oxide::object::Object;
use pdf_oxide::xref_reconstruction::reconstruct_xref;
use std::io::Cursor;

/// The object number the recovered Catalog points its `/Pages` at.
fn synthesized_pages_target(synthetic: &[(pdf_oxide::object::ObjectRef, Object)]) -> Option<u32> {
    synthetic.iter().find_map(|(_, obj)| {
        let dict = obj.as_dict()?;
        if dict.get("Type").and_then(|t| t.as_name()) != Some("Catalog") {
            return None;
        }
        match dict.get("Pages")? {
            Object::Reference(r) => Some(r.id),
            _ => None,
        }
    })
}

/// One object stream holding two `/Type /Pages` objects, numbered 3 and 4, in a
/// file with no usable xref so reconstruction has to run.
fn pdf_with_two_page_trees_in_an_objstm() -> Vec<u8> {
    let page_tree = b"<</Type/Pages/Kids[]/Count 0>>";
    let mut objects = Vec::new();
    objects.extend_from_slice(page_tree);
    objects.push(b' ');
    objects.extend_from_slice(page_tree);

    let pairs = format!("3 0 4 {} ", page_tree.len() + 1);
    let first = pairs.len();
    let mut data = pairs.into_bytes();
    data.extend_from_slice(&objects);

    let dict = format!("<< /Type /ObjStm /N 2 /First {} /Length {} >>", first, data.len());
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n1 0 obj\n");
    pdf.extend_from_slice(dict.as_bytes());
    pdf.extend_from_slice(b"\nstream\n");
    pdf.extend_from_slice(&data);
    pdf.extend_from_slice(b"\nendstream\nendobj\n%%EOF\n");
    pdf
}

#[test]
fn recovery_picks_the_same_page_tree_every_run() {
    let pdf = pdf_with_two_page_trees_in_an_objstm();

    // Repeated in ONE process: each `HashMap` instance is seeded separately, so
    // this alternates without needing separate runs.
    let mut seen = Vec::new();
    for _ in 0..64 {
        let mut cursor = Cursor::new(pdf.clone());
        let (_xref, _trailer, synthetic) =
            reconstruct_xref(&mut cursor).expect("recovery finds the packed page trees");
        seen.push(synthesized_pages_target(&synthetic));
    }

    let first = seen[0];
    assert!(
        seen.iter().all(|&r| r == first),
        "recovered page tree varies between reconstructions: {seen:?}"
    );
    assert_eq!(first, Some(3), "recovery must anchor on the lowest-numbered page tree");
}
