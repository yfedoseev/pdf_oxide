//! Dump plain extracted text for one page. `cargo run --example dump_text -- <pdf> <page>`.
use pdf_oxide::document::PdfDocument;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let doc = PdfDocument::open(&args[1]).expect("open");
    let page: usize = args[2].parse().expect("page");
    if std::env::var("DUMP_SPANS").is_ok() {
        let mut spans = doc.extract_spans(page).expect("spans");
        spans.sort_by(|a, b| {
            b.bbox
                .y
                .partial_cmp(&a.bbox.y)
                .unwrap()
                .then(a.bbox.x.partial_cmp(&b.bbox.x).unwrap())
        });
        let lim: usize = std::env::var("N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        for s in spans.iter().take(lim) {
            println!(
                "x={:6.1} r={:6.1} y={:6.1} w={:5.1} | {}",
                s.bbox.x,
                s.bbox.x + s.bbox.width,
                s.bbox.y,
                s.bbox.width,
                s.text.chars().take(40).collect::<String>()
            );
        }
        return;
    }
    println!("{}", doc.extract_text(page).expect("extract"));
}
