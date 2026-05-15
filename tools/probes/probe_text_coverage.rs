/// Quick text-preservation check after the inflation fixes.
/// Compares extracted text from the source PDF vs. the PDF→PPTX→PDF
/// round-trip, by word-set coverage. A regression here would mean we
/// dropped content along with the bloat.

use office_oxide::{Document, DocumentFormat};
use pdf_oxide::{
    api::Pdf, converters::office::{ir_to_pdf_bytes_pub, OfficeConfig}, document::PdfDocument,
};
use std::collections::HashSet;
use std::env;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/academic/arxiv_2510.21165v1.pdf".to_string()
    });

    let src_text = extract_pdf_text(&path);
    let src_words = word_set(&src_text);
    println!("source words: {}", src_words.len());

    let doc = PdfDocument::open(&path).expect("open");
    let pptx = doc.to_pptx_bytes().expect("to_pptx_bytes");
    let cur = std::io::Cursor::new(pptx);
    let pptx_doc = Document::from_reader(cur, DocumentFormat::Pptx).expect("parse");
    let ir = pptx_doc.to_ir();
    let bytes = ir_to_pdf_bytes_pub(&ir, &OfficeConfig::default()).expect("ir->pdf");

    let tmp = std::env::temp_dir().join("probe_rt.pdf");
    std::fs::write(&tmp, &bytes).expect("write");
    let rt_text = extract_pdf_text(tmp.to_str().unwrap());
    let rt_words = word_set(&rt_text);
    println!("round-trip words: {}", rt_words.len());

    let kept = src_words.iter().filter(|w| rt_words.contains(*w)).count();
    let coverage = kept as f64 / src_words.len().max(1) as f64;
    println!("coverage source→rt: {:.1}%", coverage * 100.0);
}

fn extract_pdf_text(path: &str) -> String {
    let mut p = Pdf::open(path).expect("open pdf");
    let n = p.page_count().unwrap_or(0);
    (0..n)
        .filter_map(|i| p.extract_text_lines(i).ok())
        .flat_map(|v| v.into_iter().map(|l| l.text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn word_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .collect()
}
