//! Recovery of a TRUNCATED PDF whose Catalog, page-tree root, xref and trailer
//! were all chopped off the end of the file.
//!
//! Real-world shape (a 5 MiB-capped web crawl): the file keeps a valid `%PDF`
//! header and the bulk of its objects - including `/Type /Page` content - but the
//! end-of-file structures are gone. Existing xref reconstruction scans the
//! surviving `N G obj` markers, then FAILS because it cannot find a `/Type
//! /Catalog` to anchor the page tree ("Could not find catalog"). The document is
//! a total loss even though its pages are right there.
//!
//! Recovery: synthesize a Catalog from the surviving page tree - a `/Type /Pages`
//! root if one survived, else a flat `/Pages` node listing the orphan `/Type
//! /Page` objects - and inject it so the document opens and extracts.

use pdf_oxide::PdfDocument;

/// Build a PDF whose objects are laid out Page-first, Catalog/Pages LAST (as
/// Linearized and many web PDFs are), then optionally lop off the tail so the
/// Catalog, page-tree root, xref and trailer vanish.
///
/// `keep_pages_root`: if true, the `/Type /Pages` node survives the cut (only the
/// Catalog + xref + trailer are lost); if false, the pages are fully orphaned.
fn truncated_pdf(keep_pages_root: bool) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 16];
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

    buf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");

    // Font and the two page-content streams come FIRST - this is the bulk that
    // survives a tail truncation. Each page carries its OWN /MediaBox and
    // /Resources so nothing depends on inheritance from the (doomed) parent.
    obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    stream(&mut buf, &mut off, 6, b"BT /F1 24 Tf 72 700 Td (ALPHA) Tj ET");
    stream(&mut buf, &mut off, 7, b"BT /F1 24 Tf 72 700 Td (BRAVO) Tj ET");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
    );
    obj(
        &mut buf,
        &mut off,
        4,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>",
    );

    // Everything from here on is what a tail truncation destroys.
    let truncate_at = buf.len();

    // Page-tree root (obj 2) and Catalog (obj 1), last.
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for id in 1..=7 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());

    if keep_pages_root {
        // Cut off only the Catalog + xref + trailer; the /Type /Pages root at
        // obj 2 survives.
        buf.truncate(off[1]);
    } else {
        // Cut off the Pages root too - fully orphaned pages.
        buf.truncate(truncate_at);
    }
    buf
}

fn extracted_text(pdf: Vec<u8>) -> String {
    let doc = PdfDocument::from_bytes(pdf).expect("truncated PDF must still OPEN via recovery");
    let n = doc.page_count().expect("page count");
    (0..n)
        .map(|p| doc.extract_text(p).unwrap_or_default())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Catalog gone, `/Type /Pages` root survives: synthesize a Catalog pointing at
/// the surviving root.
#[test]
fn recovers_when_pages_root_survives() {
    let text = extracted_text(truncated_pdf(true));
    assert!(text.contains("ALPHA"), "page 1 text must survive; got {text:?}");
    assert!(text.contains("BRAVO"), "page 2 text must survive; got {text:?}");
}

/// Catalog AND page-tree root gone: synthesize a flat `/Pages` from the orphan
/// `/Type /Page` objects.
#[test]
fn recovers_orphan_pages_with_no_pages_root() {
    let text = extracted_text(truncated_pdf(false));
    assert!(text.contains("ALPHA"), "page 1 text must survive; got {text:?}");
    assert!(text.contains("BRAVO"), "page 2 text must survive; got {text:?}");
}

/// A truncated PDF whose pages live INSIDE an object stream (PDF 1.5+ pack the
/// Catalog and page dictionaries into `/Type /ObjStm`, which the `N G obj` offset
/// scan cannot see). Recovery must decompress the surviving ObjStm, find the
/// packed pages, and rebuild a Catalog over them.
///
/// The ObjStm here carries NO `/Filter` - the spec permits an uncompressed object
/// stream, which keeps the fixture free of a Flate dependency while exercising the
/// exact same parse_object_stream + inject path.
fn truncated_pdf_pages_in_objstm() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 16];

    buf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");

    // Uncompressed survivors: the font and the two content streams (streams can
    // never live inside an ObjStm, so these are always at real offsets).
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
    obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    stream(&mut buf, &mut off, 6, b"BT /F1 24 Tf 72 700 Td (GAMMA) Tj ET");
    stream(&mut buf, &mut off, 7, b"BT /F1 24 Tf 72 700 Td (DELTA) Tj ET");

    // An object stream packing the two page dictionaries (objects 3 and 4). Its
    // body is "3 0 4 <off> " then the two dicts back to back; /First is where the
    // first dict begins, /N is the count. No /Filter -> the bytes are literal.
    let page3 = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>";
    let page4 = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>";
    let header = format!("3 0 4 {} ", page3.len() + 1);
    let first = header.len();
    let objstm_body = format!("{header}{page3} {page4}");
    off[8] = buf.len();
    buf.extend_from_slice(
        format!(
            "8 0 obj\n<< /Type /ObjStm /N 2 /First {first} /Length {} >>\nstream\n",
            objstm_body.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(objstm_body.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Everything below is destroyed by the truncation: the page-tree root, the
    // Catalog, the xref stream and the trailer.
    let truncate_at = buf.len();
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());

    buf.truncate(truncate_at);
    buf
}

#[test]
fn recovers_pages_packed_inside_an_object_stream() {
    let text = extracted_text(truncated_pdf_pages_in_objstm());
    assert!(text.contains("GAMMA"), "ObjStm page 1 must survive; got {text:?}");
    assert!(text.contains("DELTA"), "ObjStm page 2 must survive; got {text:?}");
}
