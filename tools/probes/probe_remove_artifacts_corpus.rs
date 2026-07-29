//! Corpus probe for the running-artifact heuristic on this branch: for
//! every PDF in a corpus directory, compares `extract_text` (keeps
//! everything, `include_artifacts: true`) against `extract_text_with_options`
//! with `include_artifacts: false` (drops every span `mark_running_artifact_spans`
//! tagged), so the result can be diffed between `main` and this branch's HEAD.
//!
//! Optional per-page removed-words dump via `--verbose <substring>`), so
//! the result can be diffed between branches
//!
//! Usage: cargo run --release --example probe_remove_artifacts_corpus -- <corpus_dir> [--verbose <substring>]

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

fn find_pdfs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_pdfs(&path, out);
        } else if path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("pdf"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

fn hash_text(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Words present in `before` but not in `after`, in order, keeping
/// duplicates (a word removed twice shows up twice) — a coarse but
/// readable summary of what a removal call actually erased.
fn removed_words(before: &str, after: &str) -> Vec<String> {
    let mut after_words: Vec<&str> = after.split_whitespace().collect();
    let mut removed = Vec::new();
    for word in before.split_whitespace() {
        if let Some(pos) = after_words.iter().position(|w| *w == word) {
            after_words.remove(pos);
        } else {
            removed.push(word.to_string());
        }
    }
    removed
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "test_datasets".to_string());
    let verbose_substring = args
        .iter()
        .position(|a| a == "--verbose")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let mut pdfs = Vec::new();
    find_pdfs(Path::new(&corpus_dir), &mut pdfs);
    pdfs.sort();

    for path in &pdfs {
        let rel = path.strip_prefix(&corpus_dir).unwrap_or(path);
        let doc = match PdfDocument::open(path) {
            Ok(d) => d,
            Err(e) => {
                println!("{}\tERROR opening: {e}", rel.display());
                continue;
            },
        };
        let pages = doc.page_count().unwrap_or(0);

        let stripped_options =
            ConversionOptions { include_artifacts: false, ..Default::default() };

        let mut before = Vec::with_capacity(pages);
        for p in 0..pages {
            before.push(doc.extract_text(p).unwrap_or_default());
        }

        let mut after = Vec::with_capacity(pages);
        for p in 0..pages {
            after.push(
                doc.extract_text_with_options(p, &stripped_options).unwrap_or_default(),
            );
        }

        let chars_before: i64 = before.iter().map(|t| t.chars().count() as i64).sum();
        let chars_after: i64 = after.iter().map(|t| t.chars().count() as i64).sum();

        println!(
            "{}\tpages={}\tchars_removed={}\tafter_hash={:016x}",
            rel.display(),
            pages,
            chars_before - chars_after,
            hash_text(&after.concat())
        );

        let is_verbose_target = verbose_substring
            .as_ref()
            .is_some_and(|needle| rel.to_string_lossy().contains(needle.as_str()));
        if is_verbose_target {
            for p in 0..pages {
                let removed = removed_words(&before[p], &after[p]);
                if !removed.is_empty() {
                    println!("  page {p}: removed words = {removed:?}");
                }
            }
        }
    }
}
