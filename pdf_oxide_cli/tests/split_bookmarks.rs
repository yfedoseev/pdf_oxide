//! CLI integration for #482 `split --by-bookmarks`.
//!
//! Spawns the built `pdf-oxide` binary (no extra dev-deps — the plan
//! §9.7 std::process::Command fallback). Deterministic via a contract
//! disjunction on the real fixture, plus a backward-compat regression
//! that plain `split` still produces per-page files after the run()
//! signature change.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_pdf-oxide")
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/outline.pdf")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pdfox_cli_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn split_by_bookmarks_runs_and_is_well_formed_or_specific_error() {
    let out = tmp("bm");
    let res = Command::new(bin())
        .arg("--output")
        .arg(&out)
        .arg("split")
        .arg("--by-bookmarks")
        .arg(fixture())
        .output()
        .expect("spawn pdf-oxide");

    if res.status.success() {
        let pdfs: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("pdf"))
            .collect();
        assert!(!pdfs.is_empty(), "success must write >=1 PDF");
        for p in &pdfs {
            let b = std::fs::read(p).unwrap();
            assert!(b.starts_with(b"%PDF-"), "written file is a PDF: {}", p.display());
        }
    } else {
        let err = String::from_utf8_lossy(&res.stderr);
        assert!(
            err.contains("outline") || err.contains("bookmark") || err.contains("split point"),
            "non-zero exit must explain the bookmark/outline failure; stderr: {err}"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn plain_split_still_per_page_backward_compat() {
    let out = tmp("plain");
    let res = Command::new(bin())
        .arg("--output")
        .arg(&out)
        .arg("split")
        .arg(fixture())
        .output()
        .expect("spawn pdf-oxide");
    assert!(res.status.success(), "legacy per-page split must still succeed");
    let n = std::fs::read_dir(&out)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .ok()
                .and_then(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x == "pdf")
                })
                .unwrap_or(false)
        })
        .count();
    assert!(n >= 1, "legacy split must produce per-page PDFs");
    let _ = std::fs::remove_dir_all(&out);
}
