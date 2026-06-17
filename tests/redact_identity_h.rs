//! Integration coverage for #748: destructive redaction of Type0 text encoded
//! with the horizontal identity CMap (Identity-H).
//!
//! Before the fix the engine refused every composite/Type0 font, so a CJK (or
//! any CID-keyed) document could not be redacted at all. This builds a minimal
//! hand-written Type0/Identity-H PDF (no third-party files): two runs on
//! separate baselines, text carried by `/ToUnicode` and advances by `/W`. A
//! region over the first run must physically remove only that run's glyphs and
//! leave the second run intact — proving redaction is supported AND correctly
//! targeted (no under- or over-redaction across the baseline gap).

use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::{PdfDocument, RedactionOptions};

struct Run {
    x: f32,
    y: f32,
    text: &'static str,
    codes: &'static [u16],
}

/// Minimal Type0/Identity-H PDF. Each glyph advances 12pt (W=1000 at 12 Tf);
/// each run is its own `BT…ET`. `/ToUnicode` maps every CID to its scalar.
fn identity_h_pdf(runs: &[Run]) -> Vec<u8> {
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
        "<< /Type /Font /Subtype /Type0 /BaseFont /IDFix /Encoding /Identity-H \
         /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
            .into(),
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

#[test]
fn identity_h_destructive_redaction_targets_only_the_region() {
    // "SECRET" (CIDs 1..6) on the upper baseline; "PUBLIC" (CIDs 7..12) 40pt
    // below. ToUnicode makes both extractable.
    let runs = [
        Run {
            x: 100.0,
            y: 700.0,
            text: "SECRET",
            codes: &[1, 2, 3, 4, 5, 6],
        },
        Run {
            x: 100.0,
            y: 660.0,
            text: "PUBLIC",
            codes: &[7, 8, 9, 10, 11, 12],
        },
    ];
    let src = identity_h_pdf(&runs);

    // Sanity: both runs extract before redaction.
    let before = PdfDocument::from_bytes(src.clone())
        .unwrap()
        .extract_text(0)
        .unwrap();
    assert!(
        before.contains("SECRET") && before.contains("PUBLIC"),
        "fixture text: {before:?}"
    );

    let mut ed = DocumentEditor::from_bytes(src).expect("open editor");
    // Region over the upper baseline only (glyph boxes ≈ y 696..712 for SECRET,
    // y 656..672 for PUBLIC) — covers SECRET, clears PUBLIC.
    ed.add_redaction(0, [90.0, 690.0, 220.0, 720.0], None)
        .expect("queue redaction");
    let report = ed
        .apply_redactions_destructive(RedactionOptions::default())
        .expect("Identity-H redaction must be SUPPORTED, not refused");
    assert!(report.glyphs_removed > 0, "expected glyphs removed, report={report:?}");

    let out = ed
        .save_to_bytes_with_options(pdf_oxide::editor::SaveOptions::full_rewrite())
        .expect("save redacted pdf");
    let after = PdfDocument::from_bytes(out)
        .unwrap()
        .extract_text(0)
        .unwrap();
    assert!(!after.contains("SECRET"), "target run must be gone, got: {after:?}");
    assert!(after.contains("PUBLIC"), "non-target run must survive, got: {after:?}");
}
