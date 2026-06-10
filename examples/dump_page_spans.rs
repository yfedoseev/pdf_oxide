//! Debug aid: dump a page's extracted text spans with positions.
//!
//! Prints each span's text and bounding box so layout/reading-order issues
//! (column assignment, line grouping, span splitting) can be inspected.
//!
//! Usage: cargo run --example dump_page_spans -- <file.pdf> [page] [substring]
//! With an optional substring, only spans whose text contains it are shown.

use pdf_oxide::document::PdfDocument;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: <file.pdf> [page] [substring]");
    let page: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let needle = std::env::args().nth(3);

    let doc = PdfDocument::open(&path).expect("open pdf");
    let spans = doc.extract_spans(page).expect("extract spans");
    println!("page {page}: {} spans", spans.len());
    for s in &spans {
        if let Some(n) = &needle {
            if !s.text.contains(n.as_str()) {
                continue;
            }
        }
        println!(
            "x={:7.2} y={:7.2} w={:7.2} h={:6.2}  {:?}",
            s.bbox.x, s.bbox.y, s.bbox.width, s.bbox.height, s.text
        );
    }
}
