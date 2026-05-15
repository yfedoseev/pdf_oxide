# Diagnostic probes

One-off instrumentation scripts written during specific debugging sessions
(PDF→Office round-trip page-count investigations, May 2026). Each file
hardcodes corpus paths under `/home/yfedoseev/projects/pdf_oxide_tests/`
and has no general user value — they exist to make the original bugs
reproducible if regression hunting is needed later.

To run one, copy it back into `examples/` and `cargo run --release
--example <name> --features rendering`. Keeping them out of `examples/`
is deliberate: that directory ships via `cargo publish` and is API
documentation surface, which these are not.

## Files

- `probe_inflation.rs` — PPTX round-trip inflation probe (commit `ec074d52`)
- `probe_xlsx_inflation.rs` — XLSX round-trip inflation probe (commit `af2505e8`)
- `probe_docx_loss.rs` — DOCX content-loss probe (commit `47321523`)
- `probe_text_coverage.rs` — content-preservation sanity check
- `analyze_office_conversion.rs` — broader corpus-wide conversion stats
- `time_one.rs` — page-count summary helper
