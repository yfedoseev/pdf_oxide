use pdf_oxide::converters::ConversionOptions;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump_all_formats <pdf> <format>");
        std::process::exit(1)
    });
    let format = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "text".to_string());

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        },
    };
    let doc = match pdf_oxide::document::PdfDocument::from_bytes(bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error parsing {path}: {e}");
            std::process::exit(1);
        },
    };
    let _ = doc.authenticate(b"");

    let opts = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let page_count = doc.page_count().unwrap_or(0);

    match format.as_str() {
        "text" => {
            for i in 0..page_count {
                if let Ok(t) = doc.extract_text(i) {
                    println!("{t}");
                }
            }
        },
        "markdown" => match doc.to_markdown_all(&opts) {
            Ok(md) => print!("{md}"),
            Err(e) => {
                eprintln!("markdown error: {e}");
                std::process::exit(1);
            },
        },
        "html" => match doc.to_html_all(&opts) {
            Ok(html) => print!("{html}"),
            Err(e) => {
                eprintln!("html error: {e}");
                std::process::exit(1);
            },
        },
        other => {
            eprintln!("unknown format: {other} (use text/markdown/html)");
            std::process::exit(1);
        },
    }
}
