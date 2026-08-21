//! #965: erasure staged on a `PdfDocument` (via `erase_region`,
//! `remove_headers`/`remove_footers`/`remove_artifacts`) must survive a
//! `DocumentEditor` save. Checking the in-memory editor state after save
//! proves nothing about the bytes actually written, so every assertion
//! here re-parses the saved output before checking it.
//!
//! Design invariants under test:
//! - `DocumentEditor::from_document` only carries staged regions across; it
//!   never performs content-stream rewriting itself, so it cannot fail in
//!   any new way just because regions are staged — even on a page whose
//!   redaction will later be refused.
//! - Actual removal happens during save. Where it can be done safely, the
//!   saved file has the region genuinely gone (not just visually covered).
//! - Where removal can't be done safely, save still succeeds: the region
//!   falls back to a visual overlay and a structured warning is recorded
//!   rather than the save silently doing nothing or hard-erroring.

use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::extractors::warnings::WarningCategory;
use pdf_oxide::geometry::Rect;
use pdf_oxide::PdfDocument;

struct Run {
    x: f32,
    y: f32,
    text: &'static str,
    codes: &'static [u16],
}

/// Minimal Type0 PDF using the given predefined CMap name (`Identity-H` or
/// `Identity-V`). Each glyph advances 12pt (W=1000 at 12 Tf); `/ToUnicode`
/// maps every CID to its scalar so the run is extractable either way.
fn type0_pdf(encoding: &str, runs: &[Run]) -> Vec<u8> {
    let mut content = String::new();
    for r in runs {
        let hex: String = r.codes.iter().map(|c| format!("{c:04X}")).collect();
        content.push_str(&format!("BT /F1 12 Tf 1 0 0 1 {:.1} {:.1} Tm <{hex}> Tj ET\n", r.x, r.y));
    }
    let mut pairs: Vec<(u16, char)> = Vec::new();
    for r in runs {
        for (code, ch) in r.codes.iter().zip(r.text.chars()) {
            pairs.push((*code, ch));
        }
    }
    let mut bf = String::new();
    for (code, ch) in &pairs {
        bf.push_str(&format!("<{code:04X}> <{:04X}>\n", *ch as u32));
    }
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend",
        pairs.len(),
        bf
    );
    let mut w = String::new();
    for (code, _) in &pairs {
        w.push_str(&format!("{code} [1000] "));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 9];
    buf.extend_from_slice(b"%PDF-1.7\n");
    let mut obj = |buf: &mut Vec<u8>, id: usize, body: String| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    obj(&mut buf, 1, "<< /Type /Catalog /Pages 2 0 R >>".into());
    obj(&mut buf, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into());
    obj(
        &mut buf,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        4,
        format!("<< /Length {} >>\nstream\n{content}endstream", content.len()),
    );
    obj(
        &mut buf,
        5,
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /IDFix /Encoding /{encoding} \
             /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
        ),
    );
    obj(
        &mut buf,
        6,
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /IDFix \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 8 0 R /DW 1000 /W [ {w}] /CIDToGIDMap /Identity >>"
        ),
    );
    obj(
        &mut buf,
        7,
        format!("<< /Length {} >>\nstream\n{tounicode}\nendstream", tounicode.len() + 1),
    );
    obj(
        &mut buf,
        8,
        "<< /Type /FontDescriptor /FontName /IDFix /Flags 4 \
         /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
         /CapHeight 700 /StemV 80 >>"
            .into(),
    );
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

fn body_and_footer_runs() -> Vec<Run> {
    vec![
        Run {
            x: 100.0,
            y: 700.0,
            text: "BODY",
            codes: &[1, 2, 3, 4],
        },
        Run {
            x: 100.0,
            y: 40.0,
            text: "FOOTER",
            codes: &[5, 6, 7, 8, 9, 10],
        },
    ]
}

/// `from_document` must stay infallible with staged erasures even on a page
/// whose redaction will later be refused (Identity-V — vertical writing,
/// not decodable by the box-based redaction engine, ISO 32000-1 §9.7.5.2).
/// Construction only copies region data; it never runs the fallible
/// content-stream rewrite, so nothing about the region's later fate can
/// make this call fail.
#[test]
fn from_document_stays_infallible_when_page_redaction_will_be_refused() {
    let runs = body_and_footer_runs();
    let src = type0_pdf("Identity-V", &runs);
    let doc = PdfDocument::from_bytes(src).expect("open source document");
    doc.erase_region(
        0,
        Rect {
            x: 90.0,
            y: 30.0,
            width: 200.0,
            height: 20.0,
        },
    )
    .expect("stage erasure");

    let editor = DocumentEditor::from_document(doc);
    assert!(
        editor.is_ok(),
        "from_document must not fail just because a staged region will later \
         be refused at save time: {:?}",
        editor.err()
    );
}

/// Staged erasure that redaction *can* handle safely must be genuinely gone
/// from the saved bytes — not merely absent from the in-memory editor.
/// Re-parses the saved output rather than trusting anything checked before
/// save. Also asserts the untouched body text is not collateral damage.
#[test]
fn erasure_on_redactable_page_is_gone_after_save_and_reparse() {
    let runs = body_and_footer_runs();
    let src = type0_pdf("Identity-H", &runs);
    let doc = PdfDocument::from_bytes(src).expect("open source document");
    // Footer run sits on baseline y=40; box covers just that run.
    doc.erase_region(
        0,
        Rect {
            x: 90.0,
            y: 30.0,
            width: 200.0,
            height: 20.0,
        },
    )
    .expect("stage erasure");

    let mut editor = DocumentEditor::from_document(doc).expect("from_document");
    let saved = editor.save_to_bytes().expect("save_to_bytes");

    let reopened = PdfDocument::from_bytes(saved).expect("reopen saved bytes");
    let text = reopened
        .extract_text(0)
        .expect("extract_text on reopened doc");

    assert!(!text.contains("FOOTER"), "footer must be gone from saved bytes, got: {text:?}");
    assert!(text.contains("BODY"), "body text must survive, got: {text:?}");
}

/// Where safe removal is refused (Identity-V), save must still succeed —
/// falling back to a visual overlay rather than erroring the whole save —
/// and must record a structured warning so the caller can detect the
/// degraded case instead of silently getting a file that still extracts
/// the "removed" text.
#[test]
fn save_falls_back_to_overlay_and_warns_when_page_redaction_is_refused() {
    let runs = body_and_footer_runs();
    let src = type0_pdf("Identity-V", &runs);
    let doc = PdfDocument::from_bytes(src).expect("open source document");
    doc.erase_region(
        0,
        Rect {
            x: 90.0,
            y: 30.0,
            width: 200.0,
            height: 20.0,
        },
    )
    .expect("stage erasure");

    let mut editor = DocumentEditor::from_document(doc).expect("from_document");
    let saved = editor.save_to_bytes();
    assert!(
        saved.is_ok(),
        "save must succeed even when a page's redaction is refused \
         (degrade to overlay, never hard-fail the whole save): {:?}",
        saved.err()
    );

    let warnings = editor.structured_warnings();
    assert!(
        warnings
            .iter()
            .any(|w| w.category == WarningCategory::RedactionOverlayFallback),
        "expected a RedactionOverlayFallback warning when redaction is \
         refused and the page falls back to a cosmetic overlay, got: {warnings:?}"
    );
}

/// A region inherited via `from_document` and a region added directly by a
/// caller via the public `erase_region` API, on the same page, must not be
/// conflated in either direction: the inherited region is eligible for
/// destructive removal at save time, but a direct `erase_region` call is
/// documented as cosmetic-only and must stay that way even when the same
/// page also has an inherited region that DOES get destructively removed.
#[test]
fn inherited_and_direct_erase_regions_on_same_page_are_handled_independently() {
    let runs = vec![
        Run {
            x: 100.0,
            y: 700.0,
            text: "BODY",
            codes: &[1, 2, 3, 4],
        },
        Run {
            x: 100.0,
            y: 40.0,
            text: "FOOTER",
            codes: &[5, 6, 7, 8, 9, 10],
        },
        Run {
            x: 300.0,
            y: 40.0,
            text: "DIRECT",
            codes: &[11, 12, 13, 14, 15, 16],
        },
    ];
    let src = type0_pdf("Identity-H", &runs);
    let doc = PdfDocument::from_bytes(src).expect("open source document");
    // Stage the FOOTER region on the PdfDocument itself, as remove_footers
    // would — this is what `from_document` inherits.
    doc.erase_region(
        0,
        Rect {
            x: 90.0,
            y: 30.0,
            width: 200.0,
            height: 20.0,
        },
    )
    .expect("stage erasure");

    let mut editor = DocumentEditor::from_document(doc).expect("from_document");
    // DIRECT region added AFTER from_document, via the public cosmetic-only
    // API — distinct rect from the inherited one, same page.
    editor
        .erase_region(0, [290.0, 30.0, 420.0, 50.0])
        .expect("direct erase_region");

    let saved = editor.save_to_bytes().expect("save_to_bytes");

    // The direct region must still render as a cosmetic overlay box.
    assert!(
        saved.windows(9).any(|w| w == b"1 1 1 rg\n"),
        "direct erase_region call must still produce a cosmetic overlay"
    );

    let reopened = PdfDocument::from_bytes(saved).expect("reopen saved bytes");
    let text = reopened
        .extract_text(0)
        .expect("extract_text on reopened doc");

    assert!(
        !text.contains("FOOTER"),
        "inherited region must be genuinely gone, got: {text:?}"
    );
    assert!(text.contains("BODY"), "untouched body text must survive, got: {text:?}");
    assert!(
        text.contains("DIRECT"),
        "a direct erase_region call must stay cosmetic-only — its text must \
         still be extractable underneath the overlay, got: {text:?}"
    );
}
