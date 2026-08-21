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

/// #1107: a page dict may have `/Contents` as an indirect reference to an
/// object that is itself an array of stream references — a third legal
/// shape beyond a direct single-stream reference or a direct array written
/// inline in the page dict (ISO 32000-1:2008 Table 30 types `/Contents` as
/// "stream or array" with no direct/indirect qualifier, unlike e.g.
/// `/Parent` in the same table which explicitly requires "an indirect
/// reference").
/// This asserts the real, secret-relevant behavior — same
/// control/G6-byte-scan/G1-re-extraction oracle as
/// `destructive_redaction_removes_secret_text_and_bytes` above
fn indirect_contents_array_pdf() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 8];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
    );
    // Object 4 is itself an array, reached only via an indirect reference —
    // the shape `resolve_page_content_elements` now handles.
    obj(&mut buf, &mut off, 4, "[5 0 R 6 0 R]");
    let header = b"BT /F1 24 Tf 72 700 Td (PUBLIC HEADER LINE) Tj ET".to_vec();
    stream(&mut buf, &mut off, 5, &header);
    let secret_content = format!("BT /F1 24 Tf 72 650 Td ({SECRET}) Tj ET").into_bytes();
    stream(&mut buf, &mut off, 6, &secret_content);
    obj(&mut buf, &mut off, 7, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for id in 1..=7 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn destructive_redaction_indirect_contents_array_removes_secret_text_and_bytes() {
    let src = indirect_contents_array_pdf();
    assert!(
        page0_text(&src).contains(SECRET),
        "fixture must contain the secret before redaction"
    );

    let raw_opts = pdf_oxide::editor::SaveOptions {
        compress: false,
        ..pdf_oxide::editor::SaveOptions::full_rewrite()
    };

    // Control: unredacted, saved through the exact same uncompressed path,
    // must still contain the literal secret bytes — proves the byte scan
    // below is a valid oracle.
    {
        let mut ctrl =
            DocumentEditor::from_bytes(indirect_contents_array_pdf()).expect("open control");
        let ctrl_bytes = ctrl
            .save_to_bytes_with_options(raw_opts.clone())
            .expect("save control pdf");
        assert!(
            ctrl_bytes
                .windows(SECRET.len())
                .any(|w| w == SECRET.as_bytes()),
            "control: uncompressed save must preserve the literal secret"
        );
    }

    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    // Region over the secret's baseline only (y ~650) — leaves the header
    // line (y ~700) untouched, so success here can't be explained by
    // redacting the whole page.
    ed.add_redaction(0, [0.0, 630.0, 612.0, 680.0], None)
        .expect("queue redaction");
    let report = ed
        .apply_redactions_destructive(RedactionOptions::default())
        .expect(
            "apply_redactions_destructive must not error on an indirect \
                 reference to an array of stream references",
        );
    assert!(report.glyphs_removed > 0, "expected glyphs removed, report={report:?}");

    let out = ed
        .save_to_bytes_with_options(raw_opts)
        .expect("save redacted pdf");

    // G6: the secret literal must not survive anywhere in the raw saved
    // bytes — not just absent from the *live* /Contents, but genuinely
    // dropped, including as an orphaned object. This is what the parallel
    // orphan-id fix in `apply_redactions_destructive_for_page` actually
    // buys: fixing only the crash without it would still pass a
    // text-re-extraction-only check while leaving this assertion failing.
    if let Some(pos) = out
        .windows(SECRET.len())
        .position(|w| w == SECRET.as_bytes())
    {
        let lo = pos.saturating_sub(120);
        let hi = (pos + SECRET.len() + 120).min(out.len());
        let ctx = String::from_utf8_lossy(&out[lo..hi]);
        panic!(
            "G6 VIOLATION: secret at byte {pos}/{}, indirect-reference-to-array \
             page. Context:\n>>>{}<<<",
            out.len(),
            ctx
        );
    }

    // re-extraction must not yield the secret either.
    let text = page0_text(&out);
    assert!(!text.contains(SECRET), "secret still recoverable via text extraction: {text:?}");
    // Unrelated header text on the other stream in the same array must
    // survive — proves this isn't a whole-page wipe.
    assert!(
        text.contains("PUBLIC HEADER LINE"),
        "unredacted header text from the other stream must survive: {text:?}"
    );
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
