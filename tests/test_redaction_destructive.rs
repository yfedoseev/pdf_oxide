//! Destructive-redaction integration test (#231) — the [BLOCK] security
//! gate from `00-common-foundation.md` §6.3 / feature plan §7: build a
//! real PDF containing a secret, redact it through the public
//! `DocumentEditor` API, save via the default garbage-collected
//! full-rewrite path, then prove the secret is **gone** — both from
//! re-extracted text (G1) and from the raw saved bytes (G6).
//!
//! This is the end-to-end proof that the redaction is destructive, not a
//! cosmetic overlay over surviving content.

use pdf_oxide::api::PdfBuilder;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::{PdfDocument, RedactionOptions};

/// Extract all text from page 0 of a PDF byte buffer.
fn page0_text(bytes: &[u8]) -> String {
    let doc = PdfDocument::from_bytes(bytes.to_vec()).expect("open pdf for extraction");
    doc.extract_text(0).unwrap_or_default()
}

const SECRET: &str = "TOPSECRETPASSWORDXYZZY";

fn build_secret_pdf() -> Vec<u8> {
    let body = format!("PUBLIC HEADER LINE\n{SECRET}\nPUBLIC FOOTER LINE");
    let mut pdf = PdfBuilder::new()
        .from_text(&body)
        .expect("build text PDF fixture");
    pdf.to_bytes().expect("fixture to bytes")
}

/// Whole-page redaction must physically remove every glyph and leave no
/// recoverable trace of the secret anywhere.
#[test]
fn destructive_redaction_removes_secret_text_and_bytes() {
    let src = build_secret_pdf();
    // Sanity: the secret really is in the source (the fixture is valid).
    assert!(
        page0_text(&src).contains(SECRET),
        "fixture must contain the secret before redaction"
    );

    // Save *uncompressed* (still GC'd) so the raw-byte G6 scan is a
    // valid oracle: with the default compressed `full_rewrite()`,
    // FlateDecode would erase the literal `SECRET` byte sequence even
    // if redaction did nothing, making the assertion vacuous (Copilot
    // review, PR #512). `garbage_collect` is kept so the orphaned
    // original content object is still dropped.
    let raw_opts = pdf_oxide::editor::SaveOptions {
        compress: false,
        ..pdf_oxide::editor::SaveOptions::full_rewrite()
    };

    // Control: the *unredacted* document, saved through the exact same
    // uncompressed path, MUST still contain the literal secret — this
    // proves the byte scan can actually see it, so its later absence is
    // caused by redaction and not by the serializer.
    {
        let mut ctrl = DocumentEditor::from_bytes(build_secret_pdf()).expect("open control");
        let ctrl_bytes = ctrl
            .save_to_bytes_with_options(raw_opts.clone())
            .expect("save control pdf");
        assert!(
            ctrl_bytes
                .windows(SECRET.len())
                .any(|w| w == SECRET.as_bytes()),
            "control: uncompressed save must preserve the literal secret \
             (otherwise the G6 byte scan below is not a valid oracle)"
        );
    }

    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    // Cover the whole page (over-redaction is acceptable; the point is
    // that nothing survives).
    ed.add_redaction(0, [0.0, 0.0, 5000.0, 5000.0], None)
        .expect("queue redaction");
    let report = ed
        .apply_redactions_destructive(RedactionOptions::default())
        .expect("apply destructive redaction");
    assert!(
        report.glyphs_removed > 0,
        "expected glyphs to be physically removed, report = {report:?}"
    );
    assert!(report.bytes_removed > 0, "expected non-zero bytes removed");

    let out = ed
        .save_to_bytes_with_options(raw_opts)
        .expect("save redacted pdf");

    // G6: the secret literal must not survive in the raw saved bytes
    // (redacted content is written uncompressed; the original content
    // object is orphaned and dropped by the GC full rewrite).
    if let Some(pos) = out
        .windows(SECRET.len())
        .position(|w| w == SECRET.as_bytes())
    {
        let lo = pos.saturating_sub(120);
        let hi = (pos + SECRET.len() + 120).min(out.len());
        let ctx = String::from_utf8_lossy(&out[lo..hi]);
        panic!("G6 VIOLATION: secret at byte {pos}/{}. Context:\n>>>{}<<<", out.len(), ctx);
    }

    // G1: re-extracting text from the saved PDF must not yield the secret.
    let text = page0_text(&out);
    assert!(
        !text.contains(SECRET),
        "secret still recoverable via text extraction after redaction: {text:?}"
    );
}

/// Re-redacting an already-redacted document is a no-op and never panics
/// (G8 idempotence at the document level).
#[test]
fn destructive_redaction_is_idempotent() {
    let src = build_secret_pdf();
    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    ed.add_redaction(0, [0.0, 0.0, 5000.0, 5000.0], None)
        .unwrap();
    ed.apply_redactions_destructive(RedactionOptions::default())
        .expect("first pass");
    let once = ed.save_to_bytes().expect("save once");

    let mut ed2 = DocumentEditor::from_bytes(once).expect("reopen");
    ed2.add_redaction(0, [0.0, 0.0, 5000.0, 5000.0], None)
        .unwrap();
    // Second pass over already-clean content: must not error or panic.
    let _ = ed2
        .apply_redactions_destructive(RedactionOptions::default())
        .expect("second pass is safe");
    let twice = ed2.save_to_bytes().expect("save twice");

    assert!(!page0_text(&twice).contains(SECRET), "secret reappeared after re-redaction");
}

/// `redaction_count` reflects queued programmatic regions.
#[test]
fn redaction_count_tracks_queued_regions() {
    let src = build_secret_pdf();
    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    assert_eq!(ed.redaction_count(0).unwrap(), 0);
    ed.add_redaction(0, [10.0, 10.0, 50.0, 50.0], None).unwrap();
    ed.add_redaction(0, [60.0, 60.0, 90.0, 90.0], Some([1.0, 0.0, 0.0]))
        .unwrap();
    assert_eq!(ed.redaction_count(0).unwrap(), 2);
    // Out-of-range page is a clean error, not a panic.
    assert!(ed.redaction_count(9999).is_err());
}
