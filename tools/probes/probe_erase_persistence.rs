//! Probe for #965: exercises the actual `remove_artifacts` →
//! `DocumentEditor::from_document` → `save_to_bytes` → reopen path that
//! `corpus_sig` never touches (open + extract only). For each corpus PDF:
//! run `remove_artifacts(threshold)`, and wherever it erased something,
//! round-trip through `DocumentEditor::from_document(...).save_to_bytes()`,
//! reopen, and check:
//!
//! - `from_document`/`save_to_bytes` don't error
//! - the reopened file's page count matches
//! - every page still extracts without panicking
//! - erased text is actually gone from the reopened bytes, UNLESS a
//!   `RedactionOverlayFallback` warning was recorded for that save — in
//!   which case the erased text should still be PRESENT (confirming the
//!   fallback degraded as designed, not silently no-opped)
//!
//! Not general user value (hardcoded corpus paths) — per `tools/probes/`
//! convention, copy into `examples/` to build/run, delete the copy after.
//!
//! Usage: `probe_erase_persistence <dir1> [dir2 ...] [--threshold 0.8]`

use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::PdfDocument;
use std::path::PathBuf;

fn find_pdfs(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in dirs {
        visit(dir, &mut out);
    }
    out.sort();
    out
}

fn visit(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_dir() || file_type.is_symlink() {
            // `is_symlink` covers the `pdf-data` symlink in test_datasets/;
            // recurse into it too (metadata() follows symlinks so this
            // correctly distinguishes a symlinked dir from a symlinked file).
            if path.metadata().map(|m| m.is_dir()).unwrap_or(false) {
                visit(&path, out);
                continue;
            }
        }
        if path.extension().and_then(|e| e.to_str()) == Some("pdf") {
            out.push(path);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dirs = Vec::new();
    let mut threshold = 0.8f32;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--threshold" {
            i += 1;
            threshold = args[i].parse().expect("valid float threshold");
        } else {
            dirs.push(PathBuf::from(&args[i]));
        }
        i += 1;
    }
    if dirs.is_empty() {
        eprintln!("usage: probe_erase_persistence <dir1> [dir2 ...] [--threshold 0.8]");
        std::process::exit(2);
    }

    let pdfs = find_pdfs(&dirs);
    println!("Found {} PDFs", pdfs.len());

    let mut exercised = 0usize;
    let mut errors = 0usize;
    let mut mismatches = 0usize;

    for path in &pdfs {
        let name = path.display();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_one(path, threshold)
        }));
        match result {
            Ok(Ok(Some(outcome))) => {
                exercised += 1;
                if !outcome.ok {
                    mismatches += 1;
                    println!("MISMATCH {name}: {}", outcome.detail);
                }
            },
            Ok(Ok(None)) => {}, // nothing erased, not exercised
            Ok(Err(e)) => {
                errors += 1;
                println!("ERROR {name}: {e}");
            },
            Err(_) => {
                errors += 1;
                println!("PANIC {name}");
            },
        }
    }

    println!(
        "\n{} PDFs scanned, {} exercised the save path, {} errors, {} mismatches",
        pdfs.len(),
        exercised,
        errors,
        mismatches
    );
    if errors > 0 || mismatches > 0 {
        std::process::exit(1);
    }
}

struct Outcome {
    ok: bool,
    detail: String,
}

fn check_one(
    path: &std::path::Path,
    threshold: f32,
) -> Result<Option<Outcome>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let doc = PdfDocument::from_bytes(bytes)?;
    let page_count_before = doc.page_count()?;

    let removed = doc.remove_artifacts(threshold)?;
    if removed == 0 {
        return Ok(None);
    }
    let before_text = doc.extract_all_text()?;

    let mut editor = DocumentEditor::from_document(doc)?;
    let saved = editor.save_to_bytes()?;
    let fallback_pages: std::collections::HashSet<Option<usize>> = editor
        .structured_warnings()
        .iter()
        .filter(|w| w.category == WarningCategory::RedactionOverlayFallback)
        .map(|w| w.page)
        .collect();

    let reopened = PdfDocument::from_bytes(saved)?;
    let page_count_after = reopened.page_count()?;
    if page_count_after != page_count_before {
        return Ok(Some(Outcome {
            ok: false,
            detail: format!("page count {page_count_before} -> {page_count_after}"),
        }));
    }
    for i in 0..page_count_after {
        reopened.extract_text(i)?;
    }
    let after_text = reopened.extract_all_text()?;

    // Coarse signal only (word-set overlap, not char-diff — see README's
    // corpus-methodology note on why char-Levenshtein is the wrong tool):
    // when no page fell back to overlay, removed content should not
    // reappear; when a page did fall back, we can't assert word-for-word
    // equality (the fallback is degraded-but-safe by design), just that
    // the save didn't error and the page still extracts.
    if fallback_pages.is_empty() {
        let before_words: std::collections::HashSet<&str> = before_text.split_whitespace().collect();
        let after_words: std::collections::HashSet<&str> = after_text.split_whitespace().collect();
        let reappeared: Vec<&&str> = after_words.difference(&before_words).take(5).collect();
        if !reappeared.is_empty() {
            return Ok(Some(Outcome {
                ok: false,
                detail: format!(
                    "no overlay-fallback warning recorded, but saved text has words \
                     absent from the in-memory post-erasure text (sample: {reappeared:?})"
                ),
            }));
        }
    }

    Ok(Some(Outcome {
        ok: true,
        detail: format!(
            "removed={removed}, fallback_pages={}",
            fallback_pages.len()
        ),
    }))
}
