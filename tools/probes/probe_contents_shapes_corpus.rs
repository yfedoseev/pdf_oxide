//! Probe for #1107: for each corpus PDF, open via `DocumentEditor::from_bytes`,
//! queue a redaction region on page 0, and call `apply_redactions_destructive`.
//! Queuing any region is enough to exercise `get_page_content_bytes`'s
//! `/Contents`-shape parsing (via `resolve_page_content_elements`)
//! regardless of whether the region overlaps any text, so this checks the
//! actual code this fix touches without needing to know each page's layout.
//!
//! Not general user value (hardcoded corpus paths) — per `tools/probes/`
//! convention, copy into `examples/` to build/run, delete the copy after.
//!
//! Usage: `probe_contents_shapes_corpus <dir1> [dir2 ...]`

use pdf_oxide::editor::DocumentEditor;
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
        if path.metadata().map(|m| m.is_dir()).unwrap_or(false) {
            visit(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("pdf") {
            out.push(path);
        }
    }
}

fn main() {
    let dirs: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if dirs.is_empty() {
        eprintln!("usage: probe_contents_shapes_corpus <dir1> [dir2 ...]");
        std::process::exit(2);
    }

    let pdfs = find_pdfs(&dirs);
    println!("Found {} PDFs", pdfs.len());

    let mut ok = 0usize;
    let mut errors = 0usize;

    for path in &pdfs {
        let name = path.display();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_one(path)));
        match result {
            Ok(Ok(())) => ok += 1,
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

    println!("\n{} PDFs scanned, {ok} ok, {errors} errors", pdfs.len());
    if errors > 0 {
        std::process::exit(1);
    }
}

fn check_one(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let mut editor = DocumentEditor::from_bytes(bytes)?;
    // Region doesn't need to overlap text — queuing any region is enough to
    // route apply_redactions_destructive through get_page_content_bytes.
    editor.add_redaction(0, [0.0, 0.0, 50.0, 50.0], None)?;
    editor.apply_redactions_destructive(Default::default())?;
    Ok(())
}
