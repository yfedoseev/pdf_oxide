//! 0.3.75 advance-accuracy audit: dump pdf_oxide's per-glyph cumulative x for
//! spans containing a target substring, to compare against a reference
//! extractor's per-char origins (pymupdf/poppler) and localise where the
//! stored advance diverges from the true text-matrix position.
//!
//! Usage: cargo run --example audit_advance -- <pdf> <page> <substring>
use pdf_oxide::document::PdfDocument;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let doc = PdfDocument::open(&args[1]).expect("open");
    let page: usize = args[2].parse().expect("page");
    let needle = &args[3];
    let spans = doc.extract_spans(page).expect("spans");
    // Inter-span gap view: for any span whose text contains the needle (or the
    // one right before it), print [text | x | end | gap-to-next] so we can see
    // how a word/variable boundary is partitioned and what geometric gap the
    // span-merge heuristic sees.
    println!("--- span/gap view (needle {:?}) ---", needle);
    for (i, s) in spans.iter().enumerate() {
        let hit = s.text.contains(needle.as_str())
            || spans
                .get(i + 1)
                .is_some_and(|n| n.text.contains(needle.as_str()));
        if !hit {
            continue;
        }
        let end = s.bbox.x + s.bbox.width;
        let gap = spans.get(i + 1).map(|n| n.bbox.x - end).unwrap_or(f32::NAN);
        let em = if s.font_size > 0.0 {
            gap / s.font_size
        } else {
            0.0
        };
        println!(
            "  [{:<24}] x={:8.3} end={:8.3} fs={:.2} font={} -> gap={:6.3} ({:.3}em)",
            format!("{:?}", s.text),
            s.bbox.x,
            end,
            s.font_size,
            s.font_name,
            gap,
            em
        );
    }
    println!("--- per-glyph view ---");
    for s in &spans {
        if !s.text.contains(needle.as_str()) {
            continue;
        }
        let cw_sum: f32 = s.char_widths.iter().sum();
        println!(
            "SPAN {:?}\n  bbox.x={:.3} bbox.width={:.3} cw_sum={:.3} chars={} cw_entries={}",
            s.text,
            s.bbox.x,
            s.bbox.width,
            cw_sum,
            s.text.chars().count(),
            s.char_widths.len()
        );
        // Per-char cumulative x (start-of-glyph position in pdf_oxide's model).
        let mut x = s.bbox.x;
        for (ch, w) in s.text.chars().zip(s.char_widths.iter()) {
            let mark = if ch == ' ' { "  <- SPACE" } else { "" };
            println!("    {:?}  x={:8.3}  adv={:6.3}{}", ch, x, w, mark);
            x += w;
        }
        println!();
    }
}
