//! Dump span context around a known concatenation pattern (issue #377 D5d
//! research). Locates spans whose text matches one of the joined tokens
//! and prints the surrounding spans with full geometry so we can see
//! the cross-column reading-order pattern.

use pdf_oxide::PdfDocument;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: dump_spans_around <pdf> <substr>")?;
    let needle = env::args().nth(2).ok_or("usage: dump_spans_around <pdf> <substr>")?;
    let mut doc = PdfDocument::open(&path)?;
    let n_pages = doc.page_count()?;
    for p in 0..n_pages {
        let spans = match doc.extract_spans(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut hits: Vec<usize> = Vec::new();
        for (i, s) in spans.iter().enumerate() {
            if s.text.contains(needle.as_str()) {
                hits.push(i);
            }
        }
        if hits.is_empty() {
            continue;
        }
        println!("=== page {} — {} hits ===", p, hits.len());
        for h in hits.iter().take(3) {
            let lo = h.saturating_sub(4);
            let hi = (h + 4).min(spans.len());
            println!("--- context around span #{} ---", h);
            for (i, s) in spans.iter().enumerate().skip(lo).take(hi - lo) {
                let mark = if i == *h { ">>" } else { "  " };
                println!(
                    "{} #{:5} y={:6.1} x={:6.1} w={:5.1} sz={:4.1} mcid={:?} text={:?}",
                    mark,
                    i,
                    s.bbox.y,
                    s.bbox.x,
                    s.bbox.width,
                    s.font_size,
                    s.mcid,
                    &s.text.chars().take(60).collect::<String>()
                );
            }
        }
        return Ok(());
    }
    println!("not found");
    Ok(())
}
