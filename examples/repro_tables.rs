use pdf_oxide::document::PdfDocument;

fn main() {
    let path = std::env::args().nth(1).expect("pdf path");
    let page: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(24);
    let doc = PdfDocument::open(&path).expect("open");
    let words = doc.extract_words(page).map(|w| w.len()).unwrap_or(0);
    let lines = doc.extract_lines(page).map(|l| l.len()).unwrap_or(0);
    let tables = doc.extract_tables(page).expect("tables");
    println!("page {page}: words={words} lines={lines} tables={}", tables.len());
    for (i, t) in tables.iter().enumerate() {
        println!("  table {i}: rows={} real_grid={}", t.rows.len(), t.is_real_grid());
    }
}
