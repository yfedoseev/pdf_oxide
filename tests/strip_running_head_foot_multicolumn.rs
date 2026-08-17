//! `ConversionOptions::strip_running_headers_footers` must strip only
//! whole, verbatim-repeated running header/footer **lines**, never a
//! fragment that merely recurs across pages while its surrounding line
//! differs (#1022).
//!
//! `repeated_running_head_foot` previously collected repetition
//! signatures from individual **spans**, not assembled lines. A span is
//! often just a fragment of a visual line (a font/emphasis-run boundary
//! splits one line into several spans — common in academic-paper body
//! text with italicized terms). If that fragment happens to recur
//! verbatim across enough pages — while the rest of the line it's part of
//! differs every time, because it's ordinary body text, not page furniture
//! — the old code treated the fragment itself as a running-header/footer
//! signature and deleted it everywhere it occurred, including mid-sentence
//! in unrelated paragraphs.
//!
//! The fixture is a synthetic 3-page PDF (no third-party files): every
//! page has a genuine, identical, whole-line footer ("CONFIDENTIAL DRAFT")
//! that must still be stripped, and a body paragraph whose first line
//! starts with an italic-font fragment ("individual genes") — landing it
//! in its own span — followed by page-specific continuation text in the
//! regular font, so the *complete* first line differs on every page even
//! though the leading fragment repeats.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

const FOOTER: &str = "CONFIDENTIAL DRAFT";
const FRAGMENT: &str = "individual genes";

/// Page-specific continuation text following the recurring `FRAGMENT`, so
/// the complete first line is unique per page.
const SUFFIXES: [&str; 3] = [
    " contribute to observed traits.",
    " affect disease onset here.",
    " shape the outcome always.",
];

fn build_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets: Vec<usize> = vec![0];

    let mut obj = |pdf: &mut Vec<u8>, id: usize, head: String, stream: Option<&[u8]>| {
        while offsets.len() <= id {
            offsets.push(0);
        }
        offsets[id] = pdf.len();
        pdf.extend_from_slice(format!("{id} 0 obj\n{head}").as_bytes());
        if let Some(s) = stream {
            pdf.extend_from_slice(b"\nstream\n");
            pdf.extend_from_slice(s);
            pdf.extend_from_slice(b"\nendstream");
        }
        pdf.extend_from_slice(b"\nendobj\n");
    };

    // 1: Catalog, 2: Pages
    obj(&mut pdf, 1, "<< /Type /Catalog /Pages 2 0 R >>".to_string(), None);
    obj(
        &mut pdf,
        2,
        "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_string(),
        None,
    );

    // MediaBox height 600: top-15% band is y > 510, bottom-15% band is y < 90.
    // 3 Page objects: 3, 4, 5. Content streams: 6, 7, 8. Fonts: 9 (regular), 10 (italic).
    for i in 0..3usize {
        obj(
            &mut pdf,
            3 + i,
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] \
                 /Resources << /Font << /F1 9 0 R /F2 10 0 R >> >> /Contents {} 0 R >>",
                6 + i
            ),
            None,
        );
    }
    for i in 0..3usize {
        let content = format!(
            "BT\n/F2 12 Tf\n1 0 0 1 40 560 Tm\n({FRAGMENT}) Tj\n\
             /F1 12 Tf\n({}) Tj\nET\n\
             BT\n/F1 10 Tf\n1 0 0 1 40 30 Tm\n({FOOTER}) Tj\nET\n",
            SUFFIXES[i]
        );
        let content = content.into_bytes();
        obj(&mut pdf, 6 + i, format!("<< /Length {} >>", content.len()), Some(&content));
    }

    obj(
        &mut pdf,
        9,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_string(),
        None,
    );
    obj(
        &mut pdf,
        10,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>"
            .to_string(),
        None,
    );

    let xref_pos = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

#[test]
fn coincidentally_repeating_fragment_survives_while_footer_is_stripped() {
    let doc = PdfDocument::from_bytes(build_pdf()).expect("fixture parses");
    let opts = ConversionOptions {
        strip_running_headers_footers: true,
        ..Default::default()
    };

    for page in 0..3usize {
        let md = doc.to_markdown(page, &opts).expect("to_markdown");
        assert!(
            !md.contains(FOOTER),
            "page {page}: the genuine repeated footer must be stripped, got: {md:?}"
        );
        assert!(
            md.contains(FRAGMENT),
            "page {page}: the coincidentally-repeating fragment must survive — \
             its complete line differs per page, so it is not real page furniture, \
             got: {md:?}"
        );
        assert!(
            md.contains(SUFFIXES[page].trim()),
            "page {page}: the page-specific continuation must survive alongside \
             the fragment, got: {md:?}"
        );
    }
}
