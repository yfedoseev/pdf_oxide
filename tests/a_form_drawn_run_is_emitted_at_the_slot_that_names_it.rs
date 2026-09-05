//! A structure slot emits the marked content it names, even when a Form
//! XObject drew it.
//!
//! ISO 32000-1:2008 §14.7.4.2 scopes an `/MCID` to its content stream, and
//! §14.7.4.3 gives a marked-content reference an `/Stm` entry naming that
//! stream. A structure element whose kid is a bare integer carries no `/Stm`,
//! so it resolves against the page's own stream — yet a producer that draws
//! part of a page through a Form XObject routinely numbers one continuous id
//! space across the page stream and the form, and references all of it that
//! way. The reference's scope and the glyphs' scope then disagree although
//! nothing collides: the id belongs to exactly one stream.
//!
//! Matched on the scope alone, such a reference finds nothing, its slot emits
//! nothing, and the run is swept into the tail of unreferenced marked content
//! appended after the page — a callout label printed between two paragraphs
//! comes out fifteen lines later, beside whatever else the tail collected.
//!
//! The counter-case is a genuine collision: a page and a form that both number
//! from 0. There the scope is the only thing that tells the two apart, and a
//! cross-scope match would put the form's text in the page's slot.
//!
//! Hand-built synthetic PDFs; no third-party fixture.

use pdf_oxide::PdfDocument;

/// A tagged page whose middle paragraph is drawn inside a Form XObject, with
/// the page and the form sharing one continuous MCID numbering (0, then the
/// form's 1, then 2) and every element referencing it as a bare integer.
///
/// The label is set a point larger than the body and indented to the left of
/// it, as a callout label is, and sits on the row between the two body lines.
fn continuous_numbering_across_a_form() -> Vec<u8> {
    let page_content = b"BT /F1 10 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 76 490 Tm (Alphabody first paragraph) Tj EMC\n\
        ET\n\
        q 1 0 0 1 0 0 cm /Fm0 Do Q\n\
        BT /F1 10 Tf\n\
        /P <</MCID 2>> BDC 1 0 0 1 76 466 Tm (Omegabody second paragraph) Tj EMC\n\
        ET\n";

    // The form restarts nothing: it continues the page's numbering at 1.
    let form_content = b"BT /F1 11 Tf\n\
        /P <</MCID 1>> BDC 1 0 0 1 47 478 Tm (TIPLABEL) Tj EMC\n\
        ET\n";

    let struct_objects: Vec<(usize, String)> = vec![
        (7, "<< /Type /StructTreeRoot /K [8 0 R] >>".to_string()),
        (
            8,
            "<< /Type /StructElem /S /Document /P 7 0 R /K [9 0 R 10 0 R 11 0 R] >>".to_string(),
        ),
        (9, "<< /Type /StructElem /S /P /P 8 0 R /Pg 3 0 R /K 0 >>".to_string()),
        // The form's run, referenced with a bare integer: no /Stm, so the
        // reference resolves against the page's stream while the glyphs carry
        // the form's scope.
        (10, "<< /Type /StructElem /S /P /P 8 0 R /Pg 3 0 R /K 1 >>".to_string()),
        (11, "<< /Type /StructElem /S /P /P 8 0 R /Pg 3 0 R /K 2 >>".to_string()),
    ];

    build(page_content, form_content, struct_objects)
}

/// The counter-case: the page numbers 0 and 1, the form restarts at 0, and the
/// form's element names its stream with `/Stm` as §14.7.4.3 provides. The
/// bare id 0 is claimed by two streams, so nothing may be matched across
/// scopes.
fn colliding_numbering_across_a_form() -> Vec<u8> {
    let page_content = b"BT /F1 10 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 76 490 Tm (Alphabody first paragraph) Tj EMC\n\
        /P <</MCID 1>> BDC 1 0 0 1 76 466 Tm (Omegabody second paragraph) Tj EMC\n\
        ET\n\
        q 1 0 0 1 0 0 cm /Fm0 Do Q\n";

    let form_content = b"BT /F1 11 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 47 400 Tm (TIPLABEL) Tj EMC\n\
        ET\n";

    let struct_objects: Vec<(usize, String)> = vec![
        (7, "<< /Type /StructTreeRoot /K [8 0 R] >>".to_string()),
        (
            8,
            "<< /Type /StructElem /S /Document /P 7 0 R /K [9 0 R 10 0 R 11 0 R] >>".to_string(),
        ),
        (9, "<< /Type /StructElem /S /P /P 8 0 R /Pg 3 0 R /K 0 >>".to_string()),
        (10, "<< /Type /StructElem /S /P /P 8 0 R /Pg 3 0 R /K 1 >>".to_string()),
        (
            11,
            "<< /Type /StructElem /S /P /P 8 0 R /K \
             << /Type /MCR /Pg 3 0 R /Stm 6 0 R /MCID 0 >> >>"
                .to_string(),
        ),
    ];

    build(page_content, form_content, struct_objects)
}

/// Objects 1-6 are fixed (catalog, pages, page, page stream, font, form); the
/// caller supplies the structure tree as objects 7 and up.
fn build(
    page_content: &[u8],
    form_content: &[u8],
    struct_objects: Vec<(usize, String)>,
) -> Vec<u8> {
    let last = struct_objects.iter().map(|(id, _)| *id).max().unwrap_or(6);
    let mut buf: Vec<u8> = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut off = vec![0usize; last + 1];

    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

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
         /Resources << /Font << /F1 5 0 R >> /XObject << /Fm0 6 0 R >> >> \
         /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, "", page_content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    stream(
        &mut buf,
        &mut off,
        6,
        "/Type /XObject /Subtype /Form /BBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >>",
        form_content,
    );
    for (id, body) in &struct_objects {
        obj(&mut buf, &mut off, *id, body);
    }

    let xref = buf.len();
    let size = last + 1;
    buf.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for id in 1..=last {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    buf
}

fn text_of(pdf: Vec<u8>) -> String {
    PdfDocument::from_bytes(pdf)
        .expect("open")
        .extract_text(0)
        .expect("text")
}

/// The label must be emitted at the slot that names it — between the two body
/// paragraphs — not swept into the tail of unreferenced marked content.
#[test]
fn a_form_drawn_run_stays_in_the_slot_that_names_it() {
    let out = text_of(continuous_numbering_across_a_form());

    let alpha = out
        .find("Alphabody")
        .unwrap_or_else(|| panic!("first paragraph missing in:\n{out}"));
    let label = out
        .find("TIPLABEL")
        .unwrap_or_else(|| panic!("the form's label is missing entirely in:\n{out}"));
    let omega = out
        .find("Omegabody")
        .unwrap_or_else(|| panic!("second paragraph missing in:\n{out}"));

    assert!(
        alpha < label && label < omega,
        "the label is drawn between the two paragraphs and its structure slot \
         sits between theirs, so it must be emitted between them; got:\n{out}"
    );
}

/// Counter-case: two streams that both number from 0 must stay apart. The
/// form's run is named through `/Stm` and drawn last, so it must come last —
/// it must not be pulled into the page's own MCID 0 slot ahead of the body.
#[test]
fn a_colliding_id_is_never_matched_across_scopes() {
    let out = text_of(colliding_numbering_across_a_form());

    let alpha = out
        .find("Alphabody")
        .unwrap_or_else(|| panic!("first paragraph missing in:\n{out}"));
    let omega = out
        .find("Omegabody")
        .unwrap_or_else(|| panic!("second paragraph missing in:\n{out}"));
    let label = out
        .find("TIPLABEL")
        .unwrap_or_else(|| panic!("the form's label is missing entirely in:\n{out}"));

    assert!(
        alpha < omega && omega < label,
        "the form's MCID 0 is not the page's MCID 0; it must be emitted at its \
         own slot, after both page paragraphs; got:\n{out}"
    );
}
