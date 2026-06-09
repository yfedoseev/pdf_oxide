//! Integration test for structure-authoritative paragraph reflow in Markdown
//! (ISO 32000-1 §14.8.3: one `<P>` BLSE is a single paragraph that "can be
//! split between lines of text").
//!
//! A wrapped prose line — the previous line runs to the column's right margin
//! and the next line continues lowercase — must reflow into one paragraph. A
//! deliberately short / capitalised line (a form field, a record row) must keep
//! its line break even inside the same `<P>` block.
//!
//! Both PDFs are hand-built tagged documents (no third-party fixture).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Build a one-page tagged PDF. `paras` is a list of paragraphs; each paragraph
/// is a list of (text, x, y) lines that become one `<P>` element with one MCID
/// per line.
fn tagged_paragraphs_pdf(paras: &[Vec<(&str, i32, i32)>]) -> Vec<u8> {
    // Content stream: one MCID-marked Tj per line, absolutely positioned.
    let mut content = String::from("BT /F1 12 Tf\n");
    let mut mcid = 0u32;
    for para in paras {
        for (txt, x, y) in para {
            content
                .push_str(&format!("/P <</MCID {mcid}>> BDC 1 0 0 1 {x} {y} Tm ({txt}) Tj EMC\n"));
            mcid += 1;
        }
    }
    content.push_str("ET\n");
    let content = content.into_bytes();

    // Object numbering: 1 Catalog, 2 Pages, 3 Page, 4 Contents, 5 Font,
    // 6 StructTreeRoot, then one StructElem per <P>.
    let p_obj_base = 7u32;
    let mut buf: Vec<u8> = Vec::new();
    let total_objs = 6 + paras.len();
    let mut off = vec![0usize; total_objs + 1];
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
    obj(
        &mut buf,
        &mut off,
        1,
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 6 0 R >>",
    );
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, &content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // StructTreeRoot /K refs every <P>.
    let kids: Vec<String> = (0..paras.len())
        .map(|i| format!("{} 0 R", p_obj_base as usize + i))
        .collect();
    obj(
        &mut buf,
        &mut off,
        6,
        &format!("<< /Type /StructTreeRoot /K [{}] >>", kids.join(" ")),
    );
    // One <P> per paragraph, each /K listing its line MCIDs.
    let mut next_mcid = 0u32;
    for (i, para) in paras.iter().enumerate() {
        let mcids: Vec<String> = para
            .iter()
            .map(|_| {
                let m = next_mcid;
                next_mcid += 1;
                m.to_string()
            })
            .collect();
        obj(
            &mut buf,
            &mut off,
            p_obj_base as usize + i,
            &format!("<< /Type /StructElem /S /P /P 6 0 R /Pg 3 0 R /K [{}] >>", mcids.join(" ")),
        );
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for id in 1..=total_objs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", total_objs + 1).as_bytes(),
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn wrapped_prose_line_reflows_into_one_paragraph() {
    // One <P>: line 1 runs to the right margin (x≈430) and ends without a
    // terminator; line 2 starts lowercase. → one reflowed paragraph.
    let pdf = tagged_paragraphs_pdf(&[vec![
        ("the quick brown fox jumps over the lazy dog and", 72, 700),
        ("then it continues onto the next line", 72, 678),
    ]]);
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();
    assert!(
        md.contains("lazy dog and then it continues"),
        "wrapped prose did not reflow into one paragraph: {md:?}"
    );
}

#[test]
fn capitalised_field_lines_keep_their_breaks() {
    // One <P> holding two record rows; line 2 starts with a capital letter, so
    // it is a deliberate break, not a wrap — the line break must survive.
    let pdf = tagged_paragraphs_pdf(&[vec![
        ("Full name: Maria Chen and her details here", 72, 700),
        ("Date of birth is nineteen ninety", 72, 678),
    ]]);
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();
    assert!(
        !md.contains("here Date of birth"),
        "capitalised record rows were wrongly merged: {md:?}"
    );
}
