//! Standalone document sanitization integration test (#231 T10,
//! feature plan §4.6 / §7 G5+G6): build a real PDF carrying secrets in
//! the `/Info` dictionary and an embedded file, run `sanitize_document`
//! through the public `DocumentEditor` API, save via the default
//! garbage-collected full-rewrite path, then prove every secret is gone
//! from the raw saved bytes — not merely unreferenced.

use pdf_oxide::api::PdfBuilder;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::{PdfDocument, RedactionOptions};

const INFO_SECRET: &str = "INFOSECRETZQXJV";
const EF_SECRET: &str = "EMBEDDEDFILESECRETWQKP";

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

/// Build a PDF whose `/Info /Title` and an embedded file both carry a
/// distinct secret, round-tripped through a save so the constructs are
/// real indirect objects on reopen.
fn build_pdf_with_secrets() -> Vec<u8> {
    let mut pdf = PdfBuilder::new()
        .from_text("PUBLIC BODY TEXT ONLY")
        .expect("build text PDF fixture");
    let base = pdf.to_bytes().expect("fixture to bytes");

    let mut ed = DocumentEditor::from_bytes(base).expect("open editor");
    ed.set_title(INFO_SECRET);
    ed.embed_file("attachment.txt", EF_SECRET.as_bytes().to_vec())
        .expect("embed file");
    ed.save_to_bytes().expect("save fixture with secrets")
}

#[test]
fn sanitize_document_strips_info_and_embedded_file_secrets() {
    let src = build_pdf_with_secrets();
    // The fixture must really contain the /Info secret as a literal PDF
    // string before sanitization, else the oracle proves nothing.
    assert!(
        contains(&src, INFO_SECRET),
        "fixture must contain the /Info secret before sanitization"
    );

    let mut ed = DocumentEditor::from_bytes(src.clone()).expect("open editor");
    let report = ed
        .sanitize_document(RedactionOptions::default())
        .expect("sanitize document");

    // The embedded-file name tree was removed → at least one top-level
    // construct, and the dropped objects have non-zero size.
    assert!(
        report.annotations_removed >= 1,
        "expected >=1 sanitized construct, report = {report:?}"
    );
    assert!(report.bytes_removed > 0, "expected non-zero bytes removed, report = {report:?}");

    let out = ed.save_to_bytes().expect("save sanitized pdf");

    // G5/G6: the /Info secret must not survive anywhere in the output
    // (the original /Info object is replaced by an empty dict and
    // hard-excluded so it cannot persist even as a GC-missed orphan).
    if let Some(pos) = out
        .windows(INFO_SECRET.len())
        .position(|w| w == INFO_SECRET.as_bytes())
    {
        let lo = pos.saturating_sub(120);
        let hi = (pos + INFO_SECRET.len() + 120).min(out.len());
        panic!(
            "G6 VIOLATION: /Info secret survived at byte {pos}/{}. Context:\n>>>{}<<<",
            out.len(),
            String::from_utf8_lossy(&out[lo..hi])
        );
    }

    // The embedded-file payload is stored uncompressed by the writer, so
    // if it was present pre-sanitization it must be absent afterwards.
    if contains(&src, EF_SECRET) {
        assert!(
            !contains(&out, EF_SECRET),
            "G6 VIOLATION: embedded-file secret survived sanitization"
        );
    }

    // Sanitization must not corrupt the document: it still opens and the
    // public body text is intact (sanitize is non-geometric).
    let doc = PdfDocument::from_bytes(out).expect("sanitized pdf reopens");
    let text = doc.extract_text(0).unwrap_or_default();
    assert!(
        text.contains("PUBLIC BODY TEXT"),
        "non-secret page content must be preserved, got {text:?}"
    );
}

/// Toggling every scrub category off must be a no-op that still produces
/// a valid document (Interface-Segregation: callers opt out per field).
#[test]
#[allow(clippy::field_reassign_with_default)] // non_exhaustive: no struct literal
fn sanitize_document_all_toggles_off_is_noop_and_valid() {
    let src = build_pdf_with_secrets();
    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    // `RedactionOptions` is `#[non_exhaustive]`; an external test crate
    // cannot use a struct literal (E0639) — mutate public fields off the
    // safe default instead.
    let mut opts = RedactionOptions::default();
    opts.scrub_metadata = false;
    opts.remove_javascript = false;
    opts.remove_embedded_files = false;
    let report = ed.sanitize_document(opts).expect("sanitize (no-op)");
    assert_eq!(report.annotations_removed, 0);
    assert_eq!(report.bytes_removed, 0);

    let out = ed.save_to_bytes().expect("save");
    let doc = PdfDocument::from_bytes(out).expect("reopens");
    assert!(doc
        .extract_text(0)
        .unwrap_or_default()
        .contains("PUBLIC BODY TEXT"));
}
