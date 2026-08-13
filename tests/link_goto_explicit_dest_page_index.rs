//! An explicit destination `[pageRef /XYZ …]` names a page *object*;
//! `LinkDestination::Explicit.page` is documented as a 0-based page index, so
//! the reference must be resolved to the page's position in the page tree
//! (ISO 32000-1 §12.3.2.2). The fixture makes the two diverge: page 2 of the
//! document is object number 6. Both routes to a destination are covered —
//! a `/A` GoTo action and a direct `/Dest` entry.

use pdf_oxide::annotations::{LinkAction, LinkDestination};
use pdf_oxide::PdfDocument;

fn two_page_link_pdf() -> Vec<u8> {
    let content: &[u8] = b"BT /F1 12 Tf 20 100 Td (Page One) Tj ET";
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 9]; // ids 1..=8
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>".to_vec();
    bodies[3] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 7 0 R >> >> /Contents 5 0 R /Annots [4 0 R 8 0 R] >>"
        .to_vec();
    // GoTo action pointing at the page-2 object (6 0 R).
    bodies[4] = b"<< /Type /Annot /Subtype /Link /Rect [10 10 100 30] /Border [0 0 0] \
/A << /S /GoTo /D [6 0 R /XYZ 0 200 0] >> >>"
        .to_vec();
    let mut cs = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    cs.extend_from_slice(content);
    cs.extend_from_slice(b"\nendstream");
    bodies[5] = cs;
    bodies[6] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec();
    bodies[7] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec();
    // Direct /Dest destination to the same page object.
    bodies[8] = b"<< /Type /Annot /Subtype /Link /Rect [10 40 100 60] /Border [0 0 0] \
/Dest [6 0 R /XYZ 0 200 0] >>"
        .to_vec();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 9];
    for id in 1..=8 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

fn assert_page_index(dest: &LinkDestination, route: &str) {
    match dest {
        LinkDestination::Explicit { page, fit_type, .. } => {
            assert_eq!(fit_type, "XYZ", "{route}");
            assert_eq!(
                *page, 1,
                "{route}: destination must be the 0-based page-tree index \
                 (page object 6 is page 1), not the raw object number"
            );
        },
        other => panic!("{route}: expected explicit destination, got {other:?}"),
    }
}

/// A balanced page tree: the target sits in the SECOND subtree, so the index
/// only comes out right if pages in earlier subtrees are all counted. A flat
/// fixture cannot tell a correct walk from one that loses a subtree's count.
fn nested_tree_link_pdf() -> Vec<u8> {
    let content: &[u8] = b"BT /F1 12 Tf 20 100 Td (One) Tj ET";
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 12]; // ids 1..=11
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    // Root -> two intermediate Pages nodes, two leaves each.
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 4 >>".to_vec();
    bodies[3] = b"<< /Type /Pages /Parent 2 0 R /Kids [5 0 R 6 0 R] /Count 2 >>".to_vec();
    bodies[4] = b"<< /Type /Pages /Parent 2 0 R /Kids [7 0 R 8 0 R] /Count 2 >>".to_vec();
    bodies[5] = b"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 11 0 R >> >> /Contents 9 0 R /Annots [10 0 R] >>"
        .to_vec();
    bodies[6] = b"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 200 200] >>".to_vec();
    bodies[7] = b"<< /Type /Page /Parent 4 0 R /MediaBox [0 0 200 200] >>".to_vec();
    // The link target: 4th page overall, so index 3.
    bodies[8] = b"<< /Type /Page /Parent 4 0 R /MediaBox [0 0 200 200] >>".to_vec();
    let mut cs = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    cs.extend_from_slice(content);
    cs.extend_from_slice(b"\nendstream");
    bodies[9] = cs;
    bodies[10] = b"<< /Type /Annot /Subtype /Link /Rect [10 10 100 30] /Border [0 0 0] \
/A << /S /GoTo /D [8 0 R /XYZ 0 200 0] >> >>"
        .to_vec();
    bodies[11] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 12];
    for id in 1..=11 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 12\n0000000000 65535 f \n");
    for id in 1..=11 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 12 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

#[test]
fn destination_in_a_nested_subtree_counts_earlier_subtrees() {
    let doc = PdfDocument::from_bytes(nested_tree_link_pdf()).expect("parse");
    let annots = doc.get_annotations(0).expect("annotations");
    let dest = annots
        .iter()
        .find_map(|a| match &a.action {
            Some(LinkAction::GoTo(d)) => Some(d.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no GoTo action in {annots:#?}"));
    match dest {
        LinkDestination::Explicit { page, .. } => assert_eq!(
            page, 3,
            "object 8 is the 4th page overall; both pages of the first subtree must be counted"
        ),
        other => panic!("expected explicit destination, got {other:?}"),
    }
}

#[test]
fn goto_action_resolves_page_ref_to_page_index() {
    let doc = PdfDocument::from_bytes(two_page_link_pdf()).expect("parse");
    let annots = doc.get_annotations(0).expect("annotations");
    let action_dest = annots
        .iter()
        .find_map(|a| match &a.action {
            Some(LinkAction::GoTo(dest)) => Some(dest.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no GoTo action in {annots:#?}"));
    assert_page_index(&action_dest, "/A GoTo action");
}

#[test]
fn direct_dest_resolves_page_ref_to_page_index() {
    let doc = PdfDocument::from_bytes(two_page_link_pdf()).expect("parse");
    let annots = doc.get_annotations(0).expect("annotations");
    let dest = annots
        .iter()
        .find_map(|a| a.destination.clone())
        .unwrap_or_else(|| panic!("no /Dest destination in {annots:#?}"));
    assert_page_index(&dest, "/Dest entry");
}
