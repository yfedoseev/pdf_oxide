//! Inspect raw character codes from PDF to debug Unicode mapping issues.
//!
//! Usage: cargo run --example inspect_char_codes <pdf_path> [page_num]
use pdf_oxide::{error::Result, PdfDocument};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <pdf_path> [page_num]", args[0]);
        std::process::exit(1);
    }

    let pdf_path = &args[1];
    let page_num = args.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);

    println!("=== Inspecting Character Codes ===");
    println!("PDF: {}", pdf_path);
    println!("Page: {}", page_num);
    println!();

    let mut doc = PdfDocument::open(pdf_path)?;
    let chars = doc.extract_spans(page_num as usize)?;

    // Focus on the problematic region around Y=1535, X=680-730
    println!("Characters near Y=1535, X=680-730 (problematic region):");
    println!("{}", "=".repeat(100));

    for (idx, ch) in chars.iter().enumerate() {
        if (ch.bbox.y - 1535.9).abs() < 5.0 && ch.bbox.x >= 680.0 && ch.bbox.x <= 730.0 {
            let unicode_str = ch
                .text
                .chars()
                .map(|c| format!("U+{:04X}", c as u32))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "[{:3}] text='{}' ({}) at X={:.1}, Y={:.1} font_size={:.1}",
                idx, ch.text, unicode_str, ch.bbox.x, ch.bbox.y, ch.font_size
            );
        }
    }

    println!();
    println!("Expected at this location: 'methods. The fin'");
    println!("What we're getting: see above");

    Ok(())
}
