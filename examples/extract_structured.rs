use pdf_oxide::{extractors::StructuredExtractor, PdfDocument};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pdf_file> [page_num]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} document.pdf", args[0]);
        eprintln!("  {} document.pdf 0", args[0]);
        eprintln!("  {} document.pdf 1", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let page_num: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    println!("==> Extracting structured content from: {}", path);
    println!("==> Page: {}", page_num);

    let mut doc = PdfDocument::open(path)?;
    let mut extractor = StructuredExtractor::new();

    let structured = extractor.extract_page(&mut doc, page_num)?;

    println!("\n==> Metadata:");
    println!("  Total Elements: {}", structured.metadata.element_count);
    println!("  Headers: {}", structured.metadata.header_count);
    println!("  Paragraphs: {}", structured.metadata.paragraph_count);
    println!("  Lists: {}", structured.metadata.list_count);
    println!("  Tables: {}", structured.metadata.table_count);
    println!("  Page Size: {:.1} x {:.1}", structured.page_size.0, structured.page_size.1);

    println!("\n==> Content:");
    println!("{}", "=".repeat(80));

    for element in structured.elements.iter() {
        match element {
            pdf_oxide::extractors::DocumentElement::Header {
                level,
                text,
                style,
                bbox,
            } => {
                println!("\n[H{}] {}", level, text);
                println!(
                    "     Style: {:.1}pt, {}{} {}",
                    style.font_size,
                    if style.bold { "bold, " } else { "" },
                    if style.italic { "italic, " } else { "" },
                    style.font_family
                );
                println!(
                    "     Position: ({:.1}, {:.1}), Size: {:.1} x {:.1}",
                    bbox.0, bbox.1, bbox.2, bbox.3
                );
            },
            pdf_oxide::extractors::DocumentElement::Paragraph {
                text,
                style,
                bbox,
                alignment,
            } => {
                // Truncate very long paragraphs for display
                let display_text = if text.len() > 200 {
                    format!("{}...", &text[..200])
                } else {
                    text.clone()
                };

                println!("\n[P] {}", display_text);
                println!(
                    "     Style: {}pt, {}{} {}",
                    style.font_size,
                    if style.bold { "bold, " } else { "" },
                    if style.italic { "italic, " } else { "" },
                    style.font_family
                );
                println!(
                    "     Alignment: {:?}, Position: ({:.1}, {:.1})",
                    alignment, bbox.0, bbox.1
                );
            },
            pdf_oxide::extractors::DocumentElement::List {
                items,
                ordered,
                bbox,
            } => {
                println!(
                    "\n[{}] {} items",
                    if *ordered {
                        "Ordered List"
                    } else {
                        "Unordered List"
                    },
                    items.len()
                );
                println!("     Position: ({:.1}, {:.1})", bbox.0, bbox.1);

                for (j, item) in items.iter().enumerate().take(10) {
                    let display_text = if item.text.len() > 100 {
                        format!("{}...", &item.text[..100])
                    } else {
                        item.text.clone()
                    };

                    println!("     {}. {}", j + 1, display_text);
                }

                if items.len() > 10 {
                    println!("     ... and {} more items", items.len() - 10);
                }
            },
            pdf_oxide::extractors::DocumentElement::Table {
                rows,
                cols,
                cells,
                bbox,
            } => {
                println!("\n[Table] {} rows × {} columns", rows, cols);
                println!("     Position: ({:.1}, {:.1})", bbox.0, bbox.1);

                // Show first few rows
                for (i, row) in cells.iter().take(3).enumerate() {
                    println!("     Row {}: {:?}", i + 1, row);
                }

                if *rows > 3 {
                    println!("     ... and {} more rows", rows - 3);
                }
            },
        }
    }

    println!("\n{}", "=".repeat(80));

    // Export JSON
    println!("\n==> JSON Export:");
    println!("{}", "=".repeat(80));
    let json = structured.to_json()?;

    // Pretty-print JSON with indentation
    let parsed: serde_json::Value = serde_json::from_str(&json)?;
    println!("{}", serde_json::to_string_pretty(&parsed)?);

    println!("\n{}", "=".repeat(80));

    // Export plain text
    println!("\n==> Plain Text Export:");
    println!("{}", "=".repeat(80));
    let plain_text = structured.to_plain_text();
    println!("{}", plain_text);

    println!("\n{}", "=".repeat(80));
    println!("\n==> Extraction complete!");

    Ok(())
}
