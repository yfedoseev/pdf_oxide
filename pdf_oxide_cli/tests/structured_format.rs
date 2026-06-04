//! The `text --format structured` CLI path exposes the library's
//! `extract_structured` API — emitting `StructuredPage` JSON with typed regions
//! and per-region `column_index`, so two-column PDFs are not line-interleaved.
//!
//! Spawns the built `pdf-oxide` binary (no extra dev-deps; string assertions on
//! the emitted JSON, mirroring the split-bookmarks integration test).

use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_pdf-oxide")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures")
        .join(name)
}

#[test]
fn text_structured_emits_structured_page_json() {
    let out = Command::new(bin())
        .args(["text", "--format", "structured"])
        .arg(fixture("multi_column_table.pdf"))
        .output()
        .expect("run pdf-oxide");

    assert!(
        out.status.success(),
        "command failed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // StructuredPage JSON shape (snake_case): a `structured` block per page with
    // typed regions carrying `kind` (RegionRole) and `column_index`.
    assert!(stdout.contains("\"format\": \"structured\""), "got: {stdout}");
    assert!(stdout.contains("\"regions\""), "must contain regions array: {stdout}");
    assert!(stdout.contains("\"kind\""), "regions must carry a RegionRole kind: {stdout}");
    assert!(stdout.contains("\"column_index\""), "regions must carry column_index: {stdout}");
}

#[test]
fn text_structured_is_listed_as_a_valid_format() {
    // Guard the clap value_parser: `structured` must be an accepted --format.
    let out = Command::new(bin())
        .args(["text", "--format", "structured"])
        .arg(fixture("hello_structure.pdf"))
        .output()
        .expect("run pdf-oxide");
    assert!(
        out.status.success(),
        "structured must be an accepted format; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
