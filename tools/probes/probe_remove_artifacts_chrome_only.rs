//! Reviewer's suggested corpus check for issue #794 (PR #795): prints every
//! span `remove_artifacts(threshold)` actually erased, for manual review —
//! it does NOT judge chrome vs. body content itself. Every printed entry
//! should read as a page number or running header/footer fragment; if any
//! look like ordinary prose/headings, that's a real over-strip bug.
//!
//! Diffs RAW SPANS per page (`extract_spans`) between a plain-opened
//! baseline and a copy with `remove_artifacts` applied — this is the layer
//! that actually carries signal. `extract_text`/`to_markdown` are NOT
//! useful for this check: both already auto-exclude any
//! `artifact_type`-tagged span unconditionally (`assemble_text_from_spans`
//! at src/document.rs, `to_markdown`'s renderer at
//! src/pipeline/converters/markdown.rs), independent of whether
//! `remove_artifacts` was ever called — since `mark_running_artifact_spans`
//! tags bare page numbers automatically on every extraction. So a
//! word/paragraph-level diff of extracted text or markdown output is 0 by
//! construction for exactly this class of artifact and proves nothing.
//! `extract_spans` is the one API that does NOT drop artifact-tagged spans
//! on its own — only `erase_region` (triggered by `remove_artifacts`)
//! removes them from its returned list, so a span-count/text diff there is
//! the only place the real effect is observable.
//!
//! Usage: cargo run --release --example probe_remove_artifacts_chrome_only -- <path.pdf> [threshold]
//! (copy back into examples/ first — see tools/probes/README.md)

use pdf_oxide::PdfDocument;
use std::collections::HashMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        "/home/yfedoseev/projects/pdf_oxide_tests/pdfs/diverse/warandpeace030164mbp.pdf".to_string()
    });
    let threshold: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.8);

    eprintln!("Opening {path} (baseline)...");
    let baseline_doc = PdfDocument::open(&path).expect("open pdf (baseline)");

    eprintln!("Opening {path} (remove_artifacts({threshold}))...");
    let cleaned_doc = PdfDocument::open(&path).expect("open pdf (cleaned)");
    let removed = cleaned_doc.remove_artifacts(threshold).expect("remove_artifacts");
    eprintln!("remove_artifacts reported {removed} spans erased");

    println!("--- per-page span diff (what `remove_artifacts` actually erased) ---");
    let page_count = baseline_doc.page_count().expect("page_count");
    let mut total_erased = 0usize;
    for page in 0..page_count {
        let base_spans = baseline_doc.extract_spans(page).unwrap_or_default();
        let clean_spans = cleaned_doc.extract_spans(page).unwrap_or_default();
        if base_spans.len() == clean_spans.len() {
            continue;
        }
        println!(
            "page {page}: {} spans -> {} spans",
            base_spans.len(),
            clean_spans.len()
        );
        // Multiset keyed on (text, exact bbox bits) — a HashSet of text
        // alone would falsely flag every duplicate-text span on the page
        // as "erased" if even one true duplicate was actually removed.
        fn span_key(s: &pdf_oxide::layout::TextSpan) -> (String, u32, u32, u32, u32) {
            (
                s.text.clone(),
                s.bbox.x.to_bits(),
                s.bbox.y.to_bits(),
                s.bbox.width.to_bits(),
                s.bbox.height.to_bits(),
            )
        }
        let mut clean_counts: HashMap<(String, u32, u32, u32, u32), usize> = HashMap::new();
        for s in &clean_spans {
            *clean_counts.entry(span_key(s)).or_insert(0) += 1;
        }
        for s in &base_spans {
            let key = span_key(s);
            match clean_counts.get_mut(&key) {
                Some(count) if *count > 0 => *count -= 1,
                _ => {
                    total_erased += 1;
                    println!(
                        "  ERASED: {:?}  bbox=(x={:.1}, y={:.1}, w={:.1}, h={:.1})",
                        s.text, s.bbox.x, s.bbox.y, s.bbox.width, s.bbox.height
                    );
                },
            }
        }
    }
    println!();
    println!("total spans erased: {total_erased}");
    println!(
        "Review each ERASED entry above: every one should read as a page number or running \
         header/footer fragment. Any that look like ordinary prose/headings are a real bug."
    );
}
