//! De-risk harness for the v0.3.68 region classifier (plan Phase A4).
//!
//! Proves the {Prose,Reference} vs {Table,Form} discrimination on the five PDFs
//! that reverted every prior column-reorder attempt. For each page it splits
//! spans at the content midline (a clean-2-col gutter proxy), classifies the
//! left half, right half, and whole page, and flags pages where BOTH halves are
//! reorderable (Prose/Reference). The reorder gate would fire exactly there.
//!
//! GO/NO-GO: 2-col prose/reference PDFs (CFR, PMC) should report many
//! `BOTH_REORDERABLE` pages; table/form PDFs (IRS, google_doc) should report
//! ~none. Run: `cargo run --example classify_probe -- <pdf> [page_limit]`.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::layout::{classify_region, RegionClass, TextSpan};

fn classify_half(spans: &[TextSpan], keep_left: bool, mid: f32) -> RegionClass {
    let idx: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let center = (s.bbox.left() + s.bbox.right()) * 0.5;
            if keep_left {
                center < mid
            } else {
                center >= mid
            }
        })
        .map(|(i, _)| i)
        .collect();
    classify_region(spans, &idx)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: classify_probe <pdf> [page_limit]");
        std::process::exit(2);
    }
    let path = &args[1];
    let page_limit: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let doc = match PdfDocument::open(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open failed: {e}");
            std::process::exit(1);
        },
    };
    let pages = doc.page_count().unwrap_or(0).min(page_limit);
    let mut both_reorderable = 0usize;
    let mut scanned = 0usize;

    println!(
        "{:<6} {:>6} {:<10} {:<10} {:<10} flag",
        "page", "spans", "whole", "left", "right"
    );
    for p in 0..pages {
        let spans = match doc.extract_spans(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if spans.len() < 12 {
            continue;
        }
        scanned += 1;
        let all_idx: Vec<usize> = (0..spans.len()).collect();
        let min_left = spans.iter().map(|s| s.bbox.left()).fold(f32::MAX, f32::min);
        let max_right = spans
            .iter()
            .map(|s| s.bbox.right())
            .fold(f32::MIN, f32::max);
        let mid = (min_left + max_right) * 0.5;

        let whole = classify_region(&spans, &all_idx);
        let left = classify_half(&spans, true, mid);
        let right = classify_half(&spans, false, mid);
        let flag = if left.is_reorderable_column() && right.is_reorderable_column() {
            both_reorderable += 1;
            "BOTH_REORDERABLE"
        } else {
            ""
        };
        println!(
            "{:<6} {:>6} {:<10?} {:<10?} {:<10?} {}",
            p,
            spans.len(),
            whole,
            left,
            right,
            flag
        );
    }
    println!("\nSUMMARY {}: {}/{} pages BOTH_REORDERABLE", path, both_reorderable, scanned);
}
