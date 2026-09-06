//! Extraction returns what the page contains, and reports the rest as data.
//!
//! A scanned page with no text layer used to yield an English sentence in the
//! markdown saying so. A consumer cannot tell that sentence from text the page
//! actually contains, so it is indexed, embedded and searched as though the
//! document said it — and on one page it said something false, describing a
//! vector drawing as a rasterised image OCR would recover.
//!
//! No reference extractor does this: MuPDF, poppler, pypdf and pdfminer all
//! return nothing for such a page. `to_html_all` in this crate already returns
//! an empty page div. The reason belongs in the diagnostic channel, where a
//! caller decides whether to surface it, where, and in what language.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;

/// A one-page PDF whose only content is a full-page image — a scan with no
/// text layer. 16x16 rather than a token 2x2: images below a few pixels are
/// filtered as spacers, and a fixture under that floor would pass by never
/// being seen rather than by being judged.
fn scanned_page_pdf() -> Vec<u8> {
    let content = b"q 200 0 0 120 0 0 cm /Im0 Do Q\n";
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| 90 + (i % 60) as u8).collect();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 120] \
         /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    off[5] = buf.len();
    buf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 16 /Height 16 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>\nstream\n",
            pixels.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&pixels);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// The page draws no text, so extraction returns none.
#[test]
fn test_scanned_page_extracts_as_nothing() {
    let doc = PdfDocument::from_bytes(scanned_page_pdf()).expect("parse");
    let md = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("markdown");
    assert!(
        md.trim().is_empty(),
        "the page contains no text; extraction must not invent any:\n{md}"
    );
}

/// Specifically: none of the sentence that used to be injected.
#[test]
fn no_sentence_about_the_page_appears_in_the_content() {
    let doc = PdfDocument::from_bytes(scanned_page_pdf()).expect("parse");
    let md = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("markdown");
    for phrase in ["OCR", "scanned", "rasterised", "text layer"] {
        assert!(
            !md.to_lowercase().contains(&phrase.to_lowercase()),
            "{phrase:?} is the library talking about the page, not the page's \
             own content:\n{md}"
        );
    }
}

/// And the reason is still available — out of band, as data.
///
/// Without this the fix would be indistinguishable from simply losing the
/// signal, which is the one thing the injected marker did get right.
#[test]
fn test_reason_is_reported_as_a_diagnostic() {
    let doc = PdfDocument::from_bytes(scanned_page_pdf()).expect("parse");
    let _ = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("markdown");
    let warnings = doc.structured_warnings();
    let hit = warnings
        .iter()
        .find(|w| w.category == WarningCategory::NoTextLayer);
    let w = hit.unwrap_or_else(|| {
        panic!(
            "the page yielded no text and no diagnostic said why; got {:?}",
            warnings.iter().map(|w| w.category).collect::<Vec<_>>()
        )
    });
    assert_eq!(w.page, Some(0), "the diagnostic must name the page");
}

/// The category's wire token is stable — bindings match on the string.
#[test]
fn test_category_serialises_to_a_stable_token() {
    assert_eq!(WarningCategory::NoTextLayer.as_str(), "no_text_layer");
}
