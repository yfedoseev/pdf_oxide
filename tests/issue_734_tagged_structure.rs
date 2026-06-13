//! Issue #734 §4/§5/§6 — spec-correct structure-tree surfacing in
//! `extract_structured` (ISO 32000-1:2008 §14.8.4).
//!
//! For a TAGGED PDF the structure tree is the authoritative, page-independent
//! source for labels and sections (§14.7.1):
//!   * `Lbl` (§14.8.4.3.3) — a label/numeral → `MarginalLabel` region (#4);
//!   * `Sect`/`Art`/`Part` (§14.8.4.2) — a section → `section_id` on each
//!     region, stable across pages, so a chapter that continues onto the next
//!     page keeps one id (#5) and content that spills into a column belonging
//!     to the previous chapter is grouped with it, not with the page's header
//!     chapter (#6).
//!
//! Hand-built tagged PDF (no third-party fixture). Chapter 1 (`Sect`) spans
//! page 1 AND page 2; chapter 2 (`Sect`) is on page 2 only.

use pdf_oxide::PdfDocument;

fn tagged_two_chapter_pdf() -> Vec<u8> {
    // Page 1: Lbl "1" + body. Page 2: chapter-1 continuation, then chapter 2.
    let content1 = b"BT /F1 12 Tf\n\
        /Lbl <</MCID 0>> BDC 1 0 0 1 60 700 Tm (1.) Tj EMC\n\
        /P <</MCID 1>> BDC 1 0 0 1 80 700 Tm (Au commencement) Tj EMC\n\
        ET\n";
    let content2 = b"BT /F1 12 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 80 700 Tm (suite du chapitre) Tj EMC\n\
        /Lbl <</MCID 1>> BDC 1 0 0 1 60 660 Tm (1.) Tj EMC\n\
        /P <</MCID 2>> BDC 1 0 0 1 80 660 Tm (Nouveau chapitre) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 18];
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
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 8 0 R >>",
    );
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, content1);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R /StructParents 1 >>",
    );
    stream(&mut buf, &mut off, 6, content2);
    obj(
        &mut buf,
        &mut off,
        7,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // Structure tree: Document → Sect1 (pages 1+2) , Sect2 (page 2).
    // /ParentTree (§14.7.4.4) maps each page's StructParents key to its MCIDs'
    // parent structure elements (in MCID order), so the structure-order
    // extractor can resolve page content back to the tree.
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /StructTreeRoot /K [9 0 R] /ParentTree 17 0 R >>",
    );
    obj(&mut buf, &mut off, 9, "<< /Type /StructElem /S /Document /K [10 0 R 14 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        10,
        "<< /Type /StructElem /S /Sect /P 9 0 R /K [11 0 R 12 0 R 13 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        11,
        "<< /Type /StructElem /S /Lbl /P 10 0 R /Pg 3 0 R /K [0] >>",
    );
    obj(
        &mut buf,
        &mut off,
        12,
        "<< /Type /StructElem /S /P /P 10 0 R /Pg 3 0 R /K [1] >>",
    );
    // Chapter-1 continuation on page 2 (mcid 0 of page 2).
    obj(
        &mut buf,
        &mut off,
        13,
        "<< /Type /StructElem /S /P /P 10 0 R /Pg 5 0 R /K [0] >>",
    );
    obj(
        &mut buf,
        &mut off,
        14,
        "<< /Type /StructElem /S /Sect /P 9 0 R /K [15 0 R 16 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        15,
        "<< /Type /StructElem /S /Lbl /P 14 0 R /Pg 5 0 R /K [1] >>",
    );
    obj(
        &mut buf,
        &mut off,
        16,
        "<< /Type /StructElem /S /P /P 14 0 R /Pg 5 0 R /K [2] >>",
    );
    // ParentTree: page-1 (key 0) MCIDs 0,1 → [Lbl, P]; page-2 (key 1) MCIDs
    // 0,1,2 → [P(ch1-cont), Lbl, P].
    obj(
        &mut buf,
        &mut off,
        17,
        "<< /Nums [0 [11 0 R 12 0 R] 1 [13 0 R 15 0 R 16 0 R]] >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 18\n0000000000 65535 f \n");
    for id in 1..=17 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 18 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// #734 §4: an `Lbl` structure element becomes a `MarginalLabel` region.
#[test]
fn lbl_structure_elements_become_marginal_labels() {
    use pdf_oxide::RegionRole;
    let doc = PdfDocument::from_bytes(tagged_two_chapter_pdf()).unwrap();
    let page1 = doc.extract_structured(0).unwrap();

    let label = page1
        .regions
        .iter()
        .find(|r| r.text.trim() == "1.")
        .expect("the '1.' label region must exist");
    assert_eq!(
        label.kind,
        RegionRole::MarginalLabel,
        "an Lbl-tagged numeral must be a MarginalLabel, got {:?}",
        label.kind
    );
}

/// #734 §5/§6: every region carries its `Sect` section index; chapter 1 keeps
/// ONE section id across pages 1 and 2, and the page-2 chapter-1 continuation
/// is grouped with chapter 1 — NOT with the chapter-2 content on the same page.
#[test]
fn sections_are_stable_across_pages_and_separate_chapters() {
    let doc = PdfDocument::from_bytes(tagged_two_chapter_pdf()).unwrap();
    let page1 = doc.extract_structured(0).unwrap();
    let page2 = doc.extract_structured(1).unwrap();

    let sect_of = |page: &pdf_oxide::StructuredPage, needle: &str| -> Option<usize> {
        page.regions
            .iter()
            .find(|r| r.text.contains(needle))
            .and_then(|r| r.section_id)
    };

    let ch1_p1 = sect_of(&page1, "Au commencement").expect("ch1 page1 section");
    let ch1_p2 = sect_of(&page2, "suite du chapitre").expect("ch1 page2 section");
    let ch2_p2 = sect_of(&page2, "Nouveau chapitre").expect("ch2 page2 section");

    // Cross-page continuity (#5): chapter 1 has the same id on both pages.
    assert_eq!(ch1_p1, ch1_p2, "chapter 1 must keep one section id across pages");
    // Spillover (#6): the page-2 chapter-1 continuation is a different section
    // from the chapter-2 content that shares the page.
    assert_ne!(ch1_p2, ch2_p2, "chapter 1 continuation must not merge with chapter 2");
}
