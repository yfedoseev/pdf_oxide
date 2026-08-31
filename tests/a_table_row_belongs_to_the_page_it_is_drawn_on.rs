//! A table that spans pages contributes only its own rows to each page, and a
//! cell resolves against the page that numbered it.
//!
//! ISO 32000-1:2008 §14.7.4.2 says a marked-content identifier is unique only
//! "within its content stream", and Table 324 gives the marked-content
//! reference a `/Pg` entry naming the page whose stream it belongs to —
//! required whenever the structure element has none. §14.8.4.3.4 makes one
//! `Table` element the whole table however many pages it is laid out across.
//!
//! Together those say the identifier alone names nothing. Resolving a cell by
//! bare number against a single page's spans lets every page's rows fill
//! themselves from whatever that page happens to have numbered the same, and
//! the output grows as pages x rows.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Two pages, one `/Table`, two `/TR` on each — and marked-content numbering
/// that restarts at 0 on the second page, exactly as a real paginated table
/// does.
fn paginated_table_pdf() -> Vec<u8> {
    let page1 = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 700 Tm (North) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 200 700 Tm (120) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 676 Tm (South) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 200 676 Tm (90) Tj EMC\n\
        ET\n";
    let page2 = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 700 Tm (East) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 200 700 Tm (70) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 676 Tm (West) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 200 676 Tm (45) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let last = 24usize;
    let mut off = vec![0usize; last + 1];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(
        &mut buf,
        &mut off,
        1,
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 10 0 R >>",
    );
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, page1);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    obj(
        &mut buf,
        &mut off,
        6,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R /StructParents 1 >>",
    );
    stream(&mut buf, &mut off, 7, page2);

    // One Table, four TR: two drawn on page 3 0 R, two on page 6 0 R.
    obj(&mut buf, &mut off, 10, "<< /Type /StructTreeRoot /K [11 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        11,
        "<< /Type /StructElem /S /Table /P 10 0 R /K [12 0 R 13 0 R 14 0 R 15 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        12,
        "<< /Type /StructElem /S /TR /P 11 0 R /Pg 3 0 R /K [16 0 R 17 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        13,
        "<< /Type /StructElem /S /TR /P 11 0 R /Pg 3 0 R /K [18 0 R 19 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        14,
        "<< /Type /StructElem /S /TR /P 11 0 R /Pg 6 0 R /K [20 0 R 21 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        15,
        "<< /Type /StructElem /S /TR /P 11 0 R /Pg 6 0 R /K [22 0 R 23 0 R] >>",
    );
    for (id, parent, pg, mcid) in [
        (16usize, 12usize, "3 0 R", 0u32),
        (17, 12, "3 0 R", 1),
        (18, 13, "3 0 R", 2),
        (19, 13, "3 0 R", 3),
        (20, 14, "6 0 R", 0),
        (21, 14, "6 0 R", 1),
        (22, 15, "6 0 R", 2),
        (23, 15, "6 0 R", 3),
    ] {
        obj(
            &mut buf,
            &mut off,
            id,
            &format!("<< /Type /StructElem /S /TD /P {parent} 0 R /Pg {pg} /K {mcid} >>"),
        );
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", last + 1).as_bytes());
    for id in 1..=last {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", last + 1).as_bytes());
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

fn html_of(page: usize) -> String {
    let doc = PdfDocument::from_bytes(paginated_table_pdf()).expect("parse");
    doc.to_html(page, &ConversionOptions::default()).expect("html")
}

/// A page renders its own rows and no others. Without the page qualifier each
/// page emitted all four, the two foreign ones refilled from this page's
/// numbering — so page 0 said `North` twice.
#[test]
fn a_page_renders_only_the_rows_drawn_on_it() {
    let page0 = html_of(0);
    assert_eq!(
        page0.matches("North").count(),
        1,
        "page 0 repeated a row it owns; a foreign TR was refilled from this page's MCIDs:\n{page0}"
    );
    assert_eq!(
        page0.matches("<tr").count(),
        2,
        "page 0 should carry its own two rows:\n{page0}"
    );

    let page1 = html_of(1);
    assert_eq!(
        page1.matches("East").count(),
        1,
        "page 1 repeated a row it owns:\n{page1}"
    );
    assert_eq!(
        page1.matches("<tr").count(),
        2,
        "page 1 should carry its own two rows:\n{page1}"
    );
}

/// The other half: a page must not show text drawn on a different page.
#[test]
fn a_cell_does_not_borrow_another_pages_numbering() {
    let page0 = html_of(0);
    for foreign in ["East", "West", "70", "45"] {
        assert!(
            !page0.contains(foreign),
            "page 0 shows {foreign:?}, which is drawn on page 1:\n{page0}"
        );
    }
    let page1 = html_of(1);
    for foreign in ["North", "South", "120", "90"] {
        assert!(
            !page1.contains(foreign),
            "page 1 shows {foreign:?}, which is drawn on page 0:\n{page1}"
        );
    }
}

/// And the rows it does render still carry their own values.
#[test]
fn the_rows_a_page_owns_are_still_populated() {
    let page0 = html_of(0);
    assert!(page0.contains("North") && page0.contains("120"), "{page0}");
    assert!(page0.contains("South") && page0.contains("90"), "{page0}");
}
