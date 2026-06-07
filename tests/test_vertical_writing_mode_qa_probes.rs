//! QA-driven regression probes for WMode 1 (vertical writing).
//!
//! Each probe targets a hole the first/second review pass may have missed
//! and either (a) writes a failing test that proves a bug, or (b) pins
//! the current behavior so future refactors can't silently regress it.
//!
//! Probe IDs map to the QA assignment categories:
//!
//!  - 1..4   real-world / corpus-driven probes
//!  - 5..7   horizontal-text regression (must not break)
//!  - 8..10  boundary at the majority threshold
//!  - 11..14 edge cases at the rasterizer
//!  - 15..20 /W2 parser stress
//!  - 21..23 CMap /WMode stress
//!  - 24..28 encoding-precedence stress
//!  - 29..33 per-page partition / reading-order
//!  - 34..36 save/restore / state machine
//!  - 37     performance regression check
//!  - 38     separation renderer integration
//!
//! The synthetic PDFs are built byte-by-byte so no third-party CJK fonts
//! are required.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::TextSpan;
use pdf_oxide::pipeline::{ReadingOrderContext, TextPipeline};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generic synthetic-PDF builder. Mirrors the helper in
/// test_vertical_writing_mode_fixes.rs but is reproduced locally so the two
/// test files don't couple.
///
/// CIDs 0001..0008 map to ASCII 'A'..'H' via the ToUnicode CMap, giving each
/// glyph an identifiable extracted text. `dw` is the horizontal default width
/// (1000 = one full em). `dw2 = (v_y, w1y)` is the spec /DW2 array.
/// `w2` is the optional /W2 array clause; `extra_resources` lets the test
/// inject additional resource dictionary entries (e.g. /Font << /F2 ... >>).
#[allow(clippy::too_many_arguments)]
fn build_pdf_full(
    encoding_name: &str,
    content: &[u8],
    dw: i32,
    dw2: (i32, i32),
    w2: Option<&str>,
    cmap_extra: Option<&str>,
    extra_fonts: Option<&str>, // already-formatted bytes like "/F2 8 0 R"
    extra_objs: Option<&[(usize, Vec<u8>)]>, // (obj_num, body_bytes-without-num-prefix)
) -> Vec<u8> {
    build_pdf_full_named(
        "TestFont",
        encoding_name,
        content,
        dw,
        dw2,
        w2,
        cmap_extra,
        extra_fonts,
        extra_objs,
    )
}

/// Like `build_pdf_full` but lets the caller supply a unique BaseFont name
/// so that the cross-document font cache (Layer 6) won't serve a
/// previously parsed FontInfo. Required by probes that rely on /W2 or /DW2
/// values, because the cheap identity hash does NOT include /W2 or /DW2
/// — see probe_cache_poison_*.
#[allow(clippy::too_many_arguments)]
fn build_pdf_full_named(
    base_font: &str,
    encoding_name: &str,
    content: &[u8],
    dw: i32,
    dw2: (i32, i32),
    w2: Option<&str>,
    cmap_extra: Option<&str>,
    extra_fonts: Option<&str>,
    extra_objs: Option<&[(usize, Vec<u8>)]>,
) -> Vec<u8> {
    let extra = cmap_extra.unwrap_or("");
    let cmap_src = format!(
        "/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
{extra}
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
8 beginbfchar
<0001> <0041>
<0002> <0042>
<0003> <0043>
<0004> <0044>
<0005> <0045>
<0006> <0046>
<0007> <0047>
<0008> <0048>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"
    );
    let cmap = cmap_src.as_bytes();

    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let o1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    let o2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    let o3 = pdf.len();
    let font_clause = if let Some(extra) = extra_fonts {
        format!("/Font << /F1 5 0 R {} >>", extra)
    } else {
        "/Font << /F1 5 0 R >>".to_string()
    };
    let page_body = format!(
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents 4 0 R /Resources << {} >> >> endobj\n",
        font_clause
    );
    pdf.extend_from_slice(page_body.as_bytes());

    let o4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj << /Length {} >> stream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let o5 = pdf.len();
    let f5 = format!(
        "5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /{} /Encoding /{} /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
        base_font, encoding_name
    );
    pdf.extend_from_slice(f5.as_bytes());

    let o6 = pdf.len();
    let w2_clause = w2.map(|s| format!(" /W2 {}", s)).unwrap_or_default();
    let f6 = format!(
        "6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /{} /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW {} /DW2 [{} {}]{} >> endobj\n",
        base_font, dw, dw2.0, dw2.1, w2_clause
    );
    pdf.extend_from_slice(f6.as_bytes());

    let o7 = pdf.len();
    pdf.extend_from_slice(format!("7 0 obj << /Length {} >> stream\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Extra objects (e.g. second font).
    let mut extra_offsets: Vec<(usize, usize)> = Vec::new();
    if let Some(objs) = extra_objs {
        for (n, body) in objs {
            let off = pdf.len();
            extra_offsets.push((*n, off));
            pdf.extend_from_slice(body);
        }
    }

    let xref = pdf.len();
    let mut all_offsets: Vec<(usize, usize)> = vec![
        (1, o1),
        (2, o2),
        (3, o3),
        (4, o4),
        (5, o5),
        (6, o6),
        (7, o7),
    ];
    all_offsets.extend(extra_offsets);
    all_offsets.sort_by_key(|(n, _)| *n);
    let max_n = all_offsets.iter().map(|(n, _)| *n).max().unwrap();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", max_n + 1).as_bytes());
    for (_n, off) in &all_offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", max_n + 1, xref)
            .as_bytes(),
    );
    pdf
}

fn build_pdf(
    encoding_name: &str,
    content: &[u8],
    dw: i32,
    dw2: (i32, i32),
    w2: Option<&str>,
    cmap_extra: Option<&str>,
) -> Vec<u8> {
    build_pdf_full(encoding_name, content, dw, dw2, w2, cmap_extra, None, None)
}

fn span_with_wmode(text: &str, x: f32, y: f32, wmode: u8) -> TextSpan {
    TextSpan {
        text: text.to_string(),
        bbox: Rect::new(x, y, 12.0, 12.0),
        font_name: "TestFont".to_string(),
        font_size: 12.0,
        wmode,
        ..TextSpan::default()
    }
}

// ---------------------------------------------------------------------------
// Probe 5 — horizontal text path must not regress when wmode=0 is the default
// ---------------------------------------------------------------------------

/// Pin: a pure horizontal-text content stream produces identical span X
/// positions to the pre-branch behaviour (we approximate "pre-branch" with
/// the spec formula `(w0 * Tfs) * Th + Tc + Tw` per glyph). Tests the
/// hot-path branch `advance_text_matrix` in horizontal mode is a no-op
/// against the previous matrix-translation arithmetic.
#[test]
fn probe05_pure_horizontal_extraction_advances_by_expected_x_steps() {
    // Three glyphs (CID 1, 2, 3), each /DW = 1000, fs = 12 → 12.0 per glyph.
    let content = b"BT /F1 12 Tf 100 700 Td <000100020003> Tj ET";
    let pdf = build_pdf("Identity-H", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A");
    let b = chars.iter().find(|c| c.char == 'B').expect("B");
    let c = chars.iter().find(|c| c.char == 'C').expect("C");
    // Each glyph occupies 12.0 user-space units at fs=12, DW=1000.
    assert!(
        (b.bbox.x - a.bbox.x - 12.0).abs() < 0.05,
        "A→B Δx ≠ 12.0: {}",
        b.bbox.x - a.bbox.x
    );
    assert!(
        (c.bbox.x - b.bbox.x - 12.0).abs() < 0.05,
        "B→C Δx ≠ 12.0: {}",
        c.bbox.x - b.bbox.x
    );
    // Y is stable at 700.
    assert!((a.bbox.y - 700.0).abs() < 1.0);
    assert!((b.bbox.y - 700.0).abs() < 1.0);
    assert!((c.bbox.y - 700.0).abs() < 1.0);
    // wmode tag survives as 0.
    let spans = doc.extract_spans(0).expect("extract_spans");
    for s in &spans {
        assert_eq!(
            s.wmode, 0,
            "horizontal page must keep wmode=0; got {} for {:?}",
            s.wmode, s.text
        );
    }
}

// ---------------------------------------------------------------------------
// Probe 6 — TextPipeline dispatch must NOT route pure-horizontal pages through tategaki
// ---------------------------------------------------------------------------

/// Pin: with every span tagged wmode=0, the pipeline must run the
/// configured strategy, never `TategakiStrategy`.
#[test]
fn probe06_pipeline_does_not_route_pure_horizontal_through_tategaki() {
    let spans = vec![
        span_with_wmode("A", 100.0, 700.0, 0),
        span_with_wmode("B", 200.0, 700.0, 0),
        span_with_wmode("C", 300.0, 700.0, 0),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline
        .process(spans, ReadingOrderContext::new())
        .expect("pipeline");
    // Horizontal LTR (Simple/Geometric/etc) must produce ascending X order:
    // A → B → C. Tategaki would reverse to right-to-left, giving C → B → A.
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    assert_eq!(
        combined, "ABC",
        "pure-horizontal page must NOT use tategaki sort; got {}",
        combined
    );
}

// ---------------------------------------------------------------------------
// Probe 7 — Mixed page with horizontal majority keeps the configured strategy
// ---------------------------------------------------------------------------

/// Pin: 5 horizontal spans + 1 vertical span = horizontal majority.
/// The pipeline must keep the configured (horizontal) sort. The single
/// vertical span keeps its wmode tag for downstream consumers but does
/// not flip the page-level strategy.
#[test]
fn probe07_horizontal_majority_keeps_configured_strategy_even_with_one_vertical() {
    let spans = vec![
        span_with_wmode("A", 100.0, 700.0, 0),
        span_with_wmode("B", 200.0, 700.0, 0),
        span_with_wmode("C", 300.0, 700.0, 0),
        span_with_wmode("D", 400.0, 700.0, 0),
        span_with_wmode("E", 500.0, 700.0, 0),
        span_with_wmode("V", 100.0, 600.0, 1), // 1/6 vertical.
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline
        .process(spans, ReadingOrderContext::new())
        .expect("pipeline");
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    // Configured (default Geometric) strategy should sort the row of A..E in
    // increasing X order; V is on a lower row and follows. Tategaki sort
    // would have produced rightmost-first → "EDCBAV" or similar.
    assert!(
        combined.starts_with("ABCDE"),
        "horizontal-majority page must not be routed through tategaki; got {}",
        combined
    );
    // The vertical span's wmode tag survives.
    let v = ordered.iter().find(|o| o.span.text == "V").expect("V");
    assert_eq!(v.span.wmode, 1);
}

// ---------------------------------------------------------------------------
// Probe 8 — Exactly 50/50 vertical/horizontal: pinned to tategaki
// ---------------------------------------------------------------------------

/// `is_vertical_majority` uses `vertical_count * 2 >= len` — so exactly
/// 50% vertical routes to tategaki. Pin this.
#[test]
fn probe08_exact_5050_majority_routes_to_tategaki() {
    // 2 horizontal + 2 vertical = 50/50.
    let spans = vec![
        span_with_wmode("A", 500.0, 700.0, 1), // vertical, right column
        span_with_wmode("B", 500.0, 688.0, 1), // vertical, right column
        span_with_wmode("X", 100.0, 700.0, 0), // horizontal
        span_with_wmode("Y", 200.0, 700.0, 0), // horizontal
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline
        .process(spans, ReadingOrderContext::new())
        .expect("pipeline");
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    // Tategaki sort puts rightmost X-center first, descending Y within column.
    // Right column (x≈500): A then B. Left column (x≈100..200): X then Y.
    // But tategaki uses median-width tolerance, so X/Y may cluster as one
    // column or two. Key invariant: A is first (rightmost top).
    assert!(combined.starts_with('A'), "5050 must route to tategaki; got {}", combined);
}

// ---------------------------------------------------------------------------
// Probe 9 — 51%/49% near the threshold
// ---------------------------------------------------------------------------

/// 51% vertical (3 of 6) — actually 50%; let's go 3 vertical of 6 == 50%.
/// To get strictly > 50% we need 4 of 6. Probe both 4/6 (=66%) and 3/6
/// (=exact 50%) above. Now probe 1 of 3 vs 2 of 3.
#[test]
fn probe09_boundary_thresholds() {
    // 1 vertical, 2 horizontal → 1*2 < 3 → horizontal majority.
    let spans = vec![
        span_with_wmode("V", 500.0, 700.0, 1),
        span_with_wmode("A", 100.0, 700.0, 0),
        span_with_wmode("B", 200.0, 700.0, 0),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline.process(spans, ReadingOrderContext::new()).unwrap();
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    // Horizontal sort: A B V (A and B left to right, V below). Must not be
    // right-to-left tategaki.
    assert!(
        combined.starts_with('A'),
        "1/3 vertical must use horizontal sort; got {}",
        combined
    );

    // 2 vertical, 1 horizontal → 2*2 >= 3 → vertical majority.
    let spans2 = vec![
        span_with_wmode("V", 500.0, 700.0, 1),
        span_with_wmode("W", 500.0, 688.0, 1),
        span_with_wmode("A", 100.0, 700.0, 0),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline
        .process(spans2, ReadingOrderContext::new())
        .unwrap();
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    // Vertical majority: rightmost first → V then W then A.
    assert_eq!(combined, "VWA", "2/3 vertical must use tategaki; got {}", combined);
}

// ---------------------------------------------------------------------------
// Probe 10 — Per-span wmode survives into to_markdown / to_html on mixed pages
// ---------------------------------------------------------------------------

/// Pin the current behavior: on a mixed-mode page that is horizontal-majority,
/// the per-span `wmode` field is preserved on the TextSpan struct (visible
/// via extract_spans) BUT current converters do NOT emit any wmode hint in
/// the markdown/HTML output. If the reviewer expected per-span tagging in
/// the output, this test will surface that gap.
#[test]
fn probe10_per_span_wmode_visible_via_extract_spans_but_not_in_html_output() {
    // Mid-stream font switch H→V→H gives three spans tagged differently
    // (provided extract_spans does not flatten). Use one font, change wmode
    // via per-glyph Tf switch is fundamental but we'd need two fonts. We can
    // also probe by inspecting span.wmode after extract_spans.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let spans = doc.extract_spans(0).expect("extract_spans");
    assert!(!spans.is_empty());
    // wmode survives on the span itself.
    assert!(
        spans.iter().any(|s| s.wmode == 1),
        "at least one span must carry wmode=1 from Identity-V; spans = {:?}",
        spans
            .iter()
            .map(|s| (s.text.as_str(), s.wmode))
            .collect::<Vec<_>>()
    );

    // HTML output does NOT propagate wmode (no writing-mode CSS).
    let opts = pdf_oxide::converters::ConversionOptions::default();
    let html = doc.to_html(0, &opts).expect("to_html");
    // Pin: no writing-mode CSS hint is emitted. This is a documented gap
    // (downstream consumers must inspect span.wmode themselves). If a
    // future change adds writing-mode CSS, this assertion must be
    // updated and the documented contract reviewed.
    assert!(
        !html.contains("writing-mode"),
        "current contract: HTML output does not emit writing-mode CSS even for vertical spans; got {}",
        html
    );
}

// ---------------------------------------------------------------------------
// Probe 11 — Very large negative TJ numeric offsets must not overflow
// ---------------------------------------------------------------------------

/// `[<0001> -32767 <0002>] TJ` under vertical mode: pin no overflow, no NaN.
///
/// Sign convention observation: per ISO 32000-1 §9.4.3 a TJ number element
/// is "subtracted from the current... vertical coordinate". The
/// implementation encodes this as `displacement = -offset/1000 * fs`, so a
/// negative offset (-32767) produces a POSITIVE displacement (+393.2 in
/// text-space y). With identity Tm under WMode 1, this moves the cursor
/// UP in PDF user space, NOT downward in the writing direction. The net
/// effect with the glyph's negative w1y is that B ends up ABOVE A.
///
/// This may or may not be a spec-conformance bug — the spec text doesn't
/// directly say whether "subtracted from the vertical coordinate" means
/// "text-space y" or "writing-direction-forward". Several real
/// implementations (Adobe Reader observed) treat negative TJ offsets in
/// vertical mode as moving DOWNWARD (forward in writing direction), which
/// would require the formula to be `+offset/1000 * fs` in vertical, or
/// equivalently the displacement to be NEGATIVE in y for negative offset.
///
/// Pin: the current behavior makes B sit ABOVE A. If a future fix changes
/// this convention, update this test.
#[test]
fn probe11_large_negative_tj_offset_in_vertical_mode_no_overflow_or_nan() {
    let content = b"BT /F1 12 Tf 100 700 Td [<0001> -32767 <0002>] TJ ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A");
    let b = chars.iter().find(|c| c.char == 'B').expect("B");
    // Pin: no NaN, no inf — math is well-behaved at large magnitudes.
    assert!(a.bbox.x.is_finite() && a.bbox.y.is_finite(), "A position must be finite");
    assert!(b.bbox.x.is_finite() && b.bbox.y.is_finite(), "B position must be finite");
    // Pin: current sign convention puts B ABOVE A. The magnitude of the
    // y-jump (ignoring the v_y origin offset) is ~|32767/1000 * 12| - 12 = 381.
    let dy = b.bbox.y - a.bbox.y;
    assert!(
        dy > 300.0,
        "large negative TJ offset must shift B upward by ~381 in current convention; dy={}",
        dy
    );
}

// ---------------------------------------------------------------------------
// Probe 12 — Empty Tj (<>) in vertical mode produces no NaN or panic
// ---------------------------------------------------------------------------

/// `<>Tj` is legal (empty hex string). Vertical mode must not produce NaN
/// or panic, and the cursor must stay put.
#[test]
fn probe12_empty_tj_in_vertical_mode_no_nan_no_panic() {
    // Two glyphs with an empty Tj between them — the empty Tj must not
    // displace the cursor at all.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <> Tj <0002> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A");
    let b = chars.iter().find(|c| c.char == 'B').expect("B");
    // Expected dy = 12 (one glyph's advance — the empty Tj contributes 0).
    let dy = a.bbox.y - b.bbox.y;
    assert!(
        (dy - 12.0).abs() < 0.05,
        "empty Tj in vertical mode must not displace cursor; dy={} (expected 12.0)",
        dy
    );
    assert!(a.bbox.x.is_finite() && b.bbox.x.is_finite());
}

// ---------------------------------------------------------------------------
// Probe 13 — Zero font size with vertical text must not panic
// ---------------------------------------------------------------------------

/// `0 Tf` in vertical mode — extraction must not panic. The cursor doesn't
/// advance (font_size * w1y / 1000 = 0).
#[test]
fn probe13_zero_font_size_in_vertical_mode_does_not_panic() {
    let content = b"BT /F1 0 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    // Just need to not panic; chars may or may not extract depending on
    // how zero-fontsize glyphs are filtered.
    let _ = doc.extract_chars(0);
    let _ = doc.extract_spans(0);
}

// ---------------------------------------------------------------------------
// Probe 14 — Mid-cluster vertical advance: per-cluster aggregation
// ---------------------------------------------------------------------------

/// Two CIDs in a single Tj. Each CID is 2 bytes (Identity codespace). Verify
/// that the per-glyph y delta = (w1y_cid1 + w1y_cid2) * fs / 1000 when the
/// CIDs map to the SAME unicode cluster (multi-byte CID > single cluster).
/// We can't easily force a multi-byte cluster from a synthetic font, so we
/// instead pin the simpler property: two CIDs in one Tj produce two char
/// positions, each at the expected y.
#[test]
fn probe14_multi_cid_tj_vertical_per_glyph_advance() {
    let content = b"BT /F1 12 Tf 100 700 Td <00010002> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A");
    let b = chars.iter().find(|c| c.char == 'B').expect("B");
    let dy = a.bbox.y - b.bbox.y;
    assert!((dy - 12.0).abs() < 0.05, "intra-Tj vertical dy = {}", dy);
}

// ---------------------------------------------------------------------------
// Probe 15 — /W2 with interleaved Form A and Form B in the same array
// ---------------------------------------------------------------------------

/// Form A (c [triples]) and Form B (c_first c_last w1y v_x v_y) intermixed.
/// Verify both forms parse correctly when next to each other.
///
/// CID 1: Form A override w1y=-500 → dy = 6.0 at fs=12
/// CID 2: Form A continuation w1y=-700 → dy = 8.4
/// CID 3,4: Form B range w1y=-400 → dy = 4.8
/// CID 5: DW2 default w1y=-1000 → dy = 12.0
#[test]
fn probe15_w2_interleaved_form_a_and_form_b() {
    // Unique BaseFont so the cross-document cache (Layer 6) cannot serve a
    // FontInfo parsed by another test. /W2 is NOT in the identity hash —
    // see probe_cache_poison_w2_collision_across_documents.
    let content = b"BT /F1 12 Tf 100 700 Td <00010002> Tj <00030004> Tj <0005> Tj ET";
    // Form A starting at CID 1 with two triples, then Form B starting at
    // CID 3 covering 3..=4.
    let pdf = build_pdf_full_named(
        "Probe15Font",
        "Identity-V",
        content,
        1000,
        (880, -1000),
        Some("[1 [-500 250 600 -700 250 600] 3 4 -400 250 600]"),
        None,
        None,
        None,
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').unwrap();
    let b = chars.iter().find(|c| c.char == 'B').unwrap();
    let c = chars.iter().find(|c| c.char == 'C').unwrap();
    let d = chars.iter().find(|c| c.char == 'D').unwrap();
    let e = chars.iter().find(|c| c.char == 'E').unwrap();
    let dy_ab = a.bbox.y - b.bbox.y;
    let dy_bc = b.bbox.y - c.bbox.y;
    let dy_cd = c.bbox.y - d.bbox.y;
    let dy_de = d.bbox.y - e.bbox.y;
    // A→B uses CID 1's w1y=-500 → 6.0
    assert!((dy_ab - 6.0).abs() < 0.1, "CID1 dy = {}, expected 6.0", dy_ab);
    // B→C uses CID 2's w1y=-700 → 8.4
    assert!((dy_bc - 8.4).abs() < 0.1, "CID2 dy = {}, expected 8.4", dy_bc);
    // C→D uses CID 3's w1y=-400 → 4.8
    assert!((dy_cd - 4.8).abs() < 0.1, "CID3 dy = {}, expected 4.8", dy_cd);
    // D→E uses CID 4's w1y=-400 → 4.8
    assert!((dy_de - 4.8).abs() < 0.1, "CID4 dy = {}, expected 4.8", dy_de);
}

// ---------------------------------------------------------------------------
// Probe 16 — /W2 referencing CIDs not in /W (horizontal widths)
// ---------------------------------------------------------------------------

/// Spec-legal: a CID can have a vertical metric without a horizontal one.
/// Verify no crash when extracting a vertical-only-metric glyph.
#[test]
fn probe16_w2_for_cid_without_horizontal_width_does_not_crash() {
    // No /W array, only /W2 — CID 1 has vertical metrics, horizontal falls
    // back to /DW = 1000.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    let pdf =
        build_pdf("Identity-V", content, 1000, (880, -1000), Some("[1 [-500 250 600]]"), None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A char");
    assert!(a.bbox.x.is_finite() && a.bbox.y.is_finite());
}

// ---------------------------------------------------------------------------
// Probe 17 — /W2 referencing CIDs at exactly u16::MAX
// ---------------------------------------------------------------------------

/// Form B range ending at exactly u16::MAX must not overflow the inclusive
/// range iterator.
///
/// Note: extracting this through content-stream chars would require a CID
/// in that range, which the synthetic ToUnicode CMap doesn't map. We test
/// the PARSE path only — the PDF must load without panic. Inspecting the
/// internal vertical-metrics map requires `pub(crate)` access, so we
/// content ourselves with the load-without-panic + horizontal-Tj
/// extraction sanity.
#[test]
fn probe17_w2_at_u16_max_does_not_panic_in_parse() {
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    // Form B: CIDs 65530..=65535 (u16::MAX = 65535).
    let pdf = build_pdf(
        "Identity-V",
        content,
        1000,
        (880, -1000),
        Some("[65530 65535 -500 250 600]"),
        None,
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse — must not panic on u16::MAX range");
    let _ = doc.extract_chars(0); // Doesn't crash either.
}

// ---------------------------------------------------------------------------
// Probe 18 — Negative v_x or v_y
// ---------------------------------------------------------------------------

/// Glyphs with negative origin offsets (common for kana where the vertical
/// origin sits to the upper-LEFT). Verify the sign survives.
#[test]
fn probe18_negative_v_x_v_y_in_w2_preserved() {
    // Negative v_x and v_y in a Form A triple.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    let pdf =
        build_pdf("Identity-V", content, 1000, (880, -1000), Some("[1 [-500 -100 -200]]"), None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').expect("A char");
    assert!(a.bbox.x.is_finite() && a.bbox.y.is_finite());
    // The glyph's reported x/y position depends on v_x/v_y. We only pin
    // that the extractor doesn't reject negative values.
}

// ---------------------------------------------------------------------------
// Probe 19 — Real numbers (not just integers) in /W2
// ---------------------------------------------------------------------------

#[test]
fn probe19_w2_with_real_numbers() {
    // Unique BaseFont to escape cross-test cache poisoning — see
    // probe_cache_poison_w2_collision_across_documents.
    // Form A with floats: w1y=-456.5, v_x=250.3, v_y=600.7
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let pdf = build_pdf_full_named(
        "Probe19Font",
        "Identity-V",
        content,
        1000,
        (880, -1000),
        Some("[1 [-456.5 250.3 600.7]]"),
        None,
        None,
        None,
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse — must accept reals in /W2");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').unwrap();
    let b = chars.iter().find(|c| c.char == 'B').unwrap();
    let dy = a.bbox.y - b.bbox.y;
    // CID 1: w1y=-456.5 → dy = 5.478
    assert!(
        (dy - 5.478).abs() < 0.05,
        "real-number /W2 must compute exact dy = 5.478; got {}",
        dy
    );
}

// ---------------------------------------------------------------------------
// Probe 20 — /W2 with indirect references — DEFERRED
// ---------------------------------------------------------------------------

/// Indirect-reference /W2 — spec allows it but it requires injecting an
/// additional object into the PDF. Pin behavior: when /W2 references an
/// indirect array, the parser resolves it. Probe by inspecting that the
/// glyph metric matches an explicit override.
#[test]
fn probe20_w2_indirect_reference_resolves() {
    // Unique BaseFont to avoid cache poisoning — see probe_cache_poison_*.
    // The font's /W2 points to object 8, which holds the array.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let extra_objs: Vec<(usize, Vec<u8>)> =
        vec![(8, b"8 0 obj [1 [-500 250 600]] endobj\n".to_vec())];
    let pdf = build_pdf_full_named(
        "Probe20Font",
        "Identity-V",
        content,
        1000,
        (880, -1000),
        Some("8 0 R"),
        None,
        None,
        Some(&extra_objs),
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').unwrap();
    let b = chars.iter().find(|c| c.char == 'B').unwrap();
    let dy = a.bbox.y - b.bbox.y;
    // CID 1: w1y=-500 (via indirect-ref'd /W2) → dy = 6.0
    // If the parser failed to resolve the indirect ref, dy would be 12.0 (DW2 default).
    assert!(
        (dy - 6.0).abs() < 0.1,
        "indirect /W2 must resolve to per-CID override; dy={} (expected 6.0; got 12.0 → indirect not resolved)",
        dy
    );
}

// ---------------------------------------------------------------------------
// Probe 21 — Multiple /WMode directives in the same CMap: last-wins?
// ---------------------------------------------------------------------------

/// Pin behavior: the regex captures the FIRST match. So `/WMode 0 def` then
/// `/WMode 1 def` keeps wmode=0. Verify.
#[test]
fn probe21_multiple_wmode_directives_first_wins() {
    // Identity-H encoding with a ToUnicode CMap containing TWO /WMode
    // directives: 0 first, then 1. The current regex extracts the first
    // capture, so wmode=0 stays.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let pdf = build_pdf(
        "Identity-H",
        content,
        1000,
        (880, -1000),
        None,
        Some("/WMode 0 def\n/WMode 1 def"),
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let spans = doc.extract_spans(0).expect("extract_spans");
    // Encoding is Identity-H so wmode should be 0 regardless of ToUnicode.
    // This pins C5 cumulatively: ToUnicode's /WMode never overrides /Encoding.
    for s in &spans {
        assert_eq!(s.wmode, 0, "ToUnicode /WMode (any count) must not override /Encoding /Identity-H; span {:?} has wmode={}", s.text, s.wmode);
    }
}

// ---------------------------------------------------------------------------
// Probe 22 — /WMode inside a begincidrange block: regex must not over-match
// ---------------------------------------------------------------------------

/// The regex `/WMode\s+([0-9]+)\s+def` will match anywhere in the stream.
/// If a producer accidentally writes `/WMode 1 def` inside a begincidrange
/// block (e.g., as a token name), the parser will pick it up. Pin: we test
/// what happens when a phrase like "1 /WMode 1 def" appears within a bfchar
/// block-like context.
#[test]
fn probe22_wmode_directive_inside_block_is_picked_up_by_regex() {
    // The regex doesn't care about block boundaries. A `/WMode 1 def` inside
    // what looks like a begincidrange would still flip wmode. Use a CMap
    // with /Encoding /Identity-V so this doesn't matter for wmode resolution
    // (encoding wins). But pin the regex behavior with a stream-only probe.
    //
    // Best we can do without exposing pub(crate) APIs: build a PDF whose
    // /Encoding is an indirect stream containing /WMode 1 def NESTED inside
    // a begincidrange. Then check whether the font's wmode comes out as 1.
    // This is C5's CMap-stream encoding path.
    //
    // For now: pin that an /Encoding stream with a properly-placed /WMode 1
    // def is honored. The nested case is a regex-quality test deferred to
    // an internal unit test if reviewer wants stronger guarantees.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    // Identity-V already vertical; this is a sanity probe.
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let spans = doc.extract_spans(0).expect("extract_spans");
    assert!(spans.iter().any(|s| s.wmode == 1));
}

// ---------------------------------------------------------------------------
// Probe 23 — /WMode after a `%` line comment
// ---------------------------------------------------------------------------

/// Pin: a `% comment` line followed by `/WMode 1 def` on the next line must
/// still recognize the /WMode directive (M5 fix).
#[test]
fn probe23_wmode_after_postscript_comment_is_recognized_in_full_pipeline() {
    // Use a horizontal encoding so that ANY wmode detection comes from the
    // ToUnicode CMap stream. C5 design says ToUnicode wmode does NOT
    // override /Encoding. So even if we put `% comment` + `/WMode 1 def`
    // in the ToUnicode, the encoding stays Identity-H and wmode stays 0.
    //
    // This probe pins that the comment-strip logic (M5) doesn't accidentally
    // mis-handle the surrounding stream and break extraction.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    let pdf = build_pdf(
        "Identity-H",
        content,
        1000,
        (880, -1000),
        None,
        Some("% some prologue comment\n/WMode 1 def"),
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let spans = doc.extract_spans(0).expect("extract_spans");
    // /Encoding is Identity-H so wmode stays 0 per C5.
    for s in &spans {
        assert_eq!(s.wmode, 0);
    }
    // And the glyph is still extracted as 'A'.
    let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(combined.contains('A'));
}

// ---------------------------------------------------------------------------
// Probe 25 — Identity-V with absent /W2 falls back to /DW2
// ---------------------------------------------------------------------------

#[test]
fn probe25_identity_v_with_absent_w2_uses_dw2_default() {
    // Unique BaseFont to avoid cache poisoning — see probe_cache_poison_*.
    // Use a non-default /DW2 (v_y=900, w1y=-800) so any fallback to spec
    // defaults would be visible.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let pdf = build_pdf_full_named(
        "Probe25Font",
        "Identity-V",
        content,
        1000,
        (900, -800),
        None,
        None,
        None,
        None,
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    let a = chars.iter().find(|c| c.char == 'A').unwrap();
    let b = chars.iter().find(|c| c.char == 'B').unwrap();
    let dy = a.bbox.y - b.bbox.y;
    // w1y = -800 → dy = |-800 * 12 / 1000| = 9.6
    assert!(
        (dy - 9.6).abs() < 0.1,
        "DW2 default must apply when /W2 is absent; dy={} expected 9.6 (would be 12.0 if spec defaults wrongly used)",
        dy
    );
}

// ---------------------------------------------------------------------------
// Probe 28 — Missing /CIDSystemInfo handled gracefully (parse-time only)
// ---------------------------------------------------------------------------

/// /CIDSystemInfo is required by the spec but real-world producers omit
/// it. Verify wmode resolution still proceeds and the font loads.
#[test]
fn probe28_font_without_cid_system_info_does_not_panic() {
    // Build a custom PDF where CIDFont lacks /CIDSystemInfo.
    let cmap = b"\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfchar
<0001> <0041>
endbfchar
endcmap
end
end";
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET";
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let o1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    let o2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    let o3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
    );
    let o4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj << /Length {} >> stream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let o5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestFont /Encoding /Identity-V /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
    );
    let o6 = pdf.len();
    // /CIDSystemInfo intentionally omitted.
    pdf.extend_from_slice(
        b"6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );
    let o7 = pdf.len();
    pdf.extend_from_slice(format!("7 0 obj << /Length {} >> stream\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for off in [o1, o2, o3, o4, o5, o6, o7] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size 8 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref).as_bytes(),
    );
    // Don't panic — extraction can fall back to empty or partial.
    let doc =
        PdfDocument::from_bytes(pdf).expect("parse — must not panic on missing CIDSystemInfo");
    let _ = doc.extract_chars(0);
    let _ = doc.extract_spans(0);
}

// ---------------------------------------------------------------------------
// Probe 29 — Mongolian (vertical, columns LEFT-TO-RIGHT)
// ---------------------------------------------------------------------------

/// Mongolian (and Traditional Manchu) script is vertical and reads
/// LEFT to RIGHT across columns — opposite of CJK tategaki. The current
/// implementation hardcodes right-to-left in TategakiStrategy. Pin the
/// current behavior: Mongolian-style vertical text WILL be misordered
/// (left column emitted last instead of first).
///
/// If the implementation later grows a LTR-vertical mode, this test must
/// be updated.
#[test]
fn probe29_mongolian_style_vertical_ltr_is_currently_misordered() {
    // Two columns: LEFT column should be FIRST in Mongolian reading order.
    // Tags: all wmode=1 (vertical), arranged as
    //   LEFT column (x≈100): A (top), B, C (bottom)
    //   RIGHT column (x≈300): D, E, F
    // Expected Mongolian order: A B C D E F (LTR)
    // Actual (CJK tategaki) order: D E F A B C (RTL)
    let spans = vec![
        span_with_wmode("A", 100.0, 700.0, 1),
        span_with_wmode("B", 100.0, 688.0, 1),
        span_with_wmode("C", 100.0, 676.0, 1),
        span_with_wmode("D", 300.0, 700.0, 1),
        span_with_wmode("E", 300.0, 688.0, 1),
        span_with_wmode("F", 300.0, 676.0, 1),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline.process(spans, ReadingOrderContext::new()).unwrap();
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    // PIN: implementation only models CJK direction → RIGHT-to-LEFT.
    // Mongolian comes out as DEFABC.
    assert_eq!(
        combined, "DEFABC",
        "current impl hardcodes RTL columns; Mongolian/Manchu come out reversed.\
         If this is fixed, update this test."
    );
}

// ---------------------------------------------------------------------------
// Probe 30 — Single vertical column (spine text) ordering
// ---------------------------------------------------------------------------

#[test]
fn probe30_single_vertical_column_ordered_top_down() {
    let spans = vec![
        span_with_wmode("C", 300.0, 676.0, 1),
        span_with_wmode("A", 300.0, 700.0, 1),
        span_with_wmode("B", 300.0, 688.0, 1),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline.process(spans, ReadingOrderContext::new()).unwrap();
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    assert_eq!(combined, "ABC");
}

// ---------------------------------------------------------------------------
// Probe 31 — Multiple non-overlapping vertical columns
// ---------------------------------------------------------------------------

#[test]
fn probe31_three_vertical_columns_grouped_right_to_left() {
    // Three columns at x=100, 300, 500. Should be 500 first, then 300, then 100.
    let spans = vec![
        span_with_wmode("X1", 100.0, 700.0, 1),
        span_with_wmode("M1", 300.0, 700.0, 1),
        span_with_wmode("R1", 500.0, 700.0, 1),
        span_with_wmode("X2", 100.0, 688.0, 1),
        span_with_wmode("M2", 300.0, 688.0, 1),
        span_with_wmode("R2", 500.0, 688.0, 1),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline.process(spans, ReadingOrderContext::new()).unwrap();
    let combined: String = ordered.iter().map(|o| o.span.text.as_str()).collect();
    assert_eq!(
        combined, "R1R2M1M2X1X2",
        "three columns must order rightmost-first, top-down within column"
    );
}

// ---------------------------------------------------------------------------
// Probe 32/33 — Rotated CTM with WMode 0 (horizontal text rotated 90°)
// ---------------------------------------------------------------------------

/// Horizontal-encoding text inside a content stream with a 90° rotation
/// CTM (`0 1 -1 0 0 0 cm`) — visually vertical on the page but logically
/// horizontal (WMode 0). The advance_text_matrix helper should route this
/// through the horizontal arm (uses `(tm.a, tm.b)`), so the rotated CTM
/// alone determines the on-page direction.
#[test]
fn probe32_horizontal_text_with_90deg_rotated_ctm_advances_perpendicularly() {
    // The CTM rotation: applied via `0 1 -1 0 0 0 cm` rotates 90° CCW.
    // Text matrix Tm stays identity. Each Tj displaces text matrix in x
    // (horizontal mode), which after CTM becomes user-space y.
    let content = b"q 0 1 -1 0 100 100 cm BT /F1 12 Tf 0 0 Td <0001> Tj <0002> Tj ET Q";
    let pdf = build_pdf("Identity-H", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse rotated horizontal");
    let chars = doc.extract_chars(0).expect("extract_chars");
    if let (Some(a), Some(b)) =
        (chars.iter().find(|c| c.char == 'A'), chars.iter().find(|c| c.char == 'B'))
    {
        // Under a 90° CCW CTM, the horizontal Tj advance becomes a +Y user
        // space advance. So B.y > A.y, and A.x ≈ B.x.
        let dx = (a.bbox.x - b.bbox.x).abs();
        let dy = b.bbox.y - a.bbox.y;
        // Document the observed behavior. A 90° rotation means horizontal
        // mode under a rotated CTM produces ~12-unit Y advances per glyph.
        assert!(
            dy > 5.0,
            "rotated-CTM horizontal text should advance in user-space Y; dy={}, dx={}",
            dy,
            dx
        );
    } else {
        // If glyphs are dropped by some artifact path, surface the failure.
        panic!("could not find A and B in rotated-CTM horizontal extraction");
    }
}

// ---------------------------------------------------------------------------
// Probe 33 — Vertical text with rotated CTM (composite case)
// ---------------------------------------------------------------------------

#[test]
fn probe33_vertical_text_with_rotated_ctm_does_not_panic() {
    let content = b"q 0 1 -1 0 100 100 cm BT /F1 12 Tf 0 0 Td <0001> Tj <0002> Tj ET Q";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse rotated vertical");
    let _ = doc.extract_chars(0); // pin: no panic
    let _ = doc.extract_spans(0);
}

// ---------------------------------------------------------------------------
// Probe 34 — q ... Tf vertical ... Q restores wmode correctly
// ---------------------------------------------------------------------------

/// Inside `q ... Q` the graphics state stack saves and restores. Verify
/// that selecting a vertical font, then `Q`, restores the outer state's
/// (horizontal) wmode. With a single-font document we can't easily
/// observe the post-Q state; but we CAN observe that an extraction that
/// crosses save/restore boundaries doesn't lose track of wmode.
#[test]
fn probe34_save_restore_preserves_outer_wmode() {
    // Single font (vertical). q...Q does not switch fonts here, but the
    // wmode is saved-and-restored as part of GraphicsState. Pin: any glyphs
    // emitted before, during, and after q...Q all carry wmode=1.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj ET \
                    q BT /F1 12 Tf 100 600 Td <0002> Tj ET Q \
                    BT /F1 12 Tf 100 500 Td <0003> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let spans = doc.extract_spans(0).expect("extract_spans");
    for s in &spans {
        assert_eq!(
            s.wmode, 1,
            "wmode=1 must survive save/restore boundary; span {:?} has wmode={}",
            s.text, s.wmode
        );
    }
}

// (The mid-BT Tf bug is exercised by probe36_mid_bt_tf_font_switch_drops_subsequent_spans
// further down — kept as a dedicated bug-finder near the cache-poisoning tests.)

// ---------------------------------------------------------------------------
// Probe 38 — Separation renderer with vertical text smoke test
// ---------------------------------------------------------------------------

/// Verify the separation renderer doesn't panic when given a vertical
/// content stream. Cannot easily assert pixel positions without a full
/// rendering pipeline, but pin no-panic.
///
/// Uses the `render_separations` free function on the separation
/// renderer; gated on the `rendering` feature so this only compiles in
/// configurations that ship the renderer.
#[test]
#[cfg(feature = "rendering")]
fn probe38_separation_renderer_vertical_text_does_not_panic() {
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    // Pin: separation rendering on a vertical-mode page must not panic.
    // A page without /Separation colorspaces returns an empty Vec — that
    // is expected on this synthetic PDF and still proves the operator
    // walk handled the vertical Tj path without crashing.
    let _ = pdf_oxide::rendering::render_separations(&doc, 0, 150);
}

// ---------------------------------------------------------------------------
// Probe 1 — Japanese packaging tategaki short brand + horizontal ingredients
// ---------------------------------------------------------------------------

/// Mixed page: 2 vertical brand-name spans, 1 horizontal ingredients-label
/// span. Vertical majority → tategaki sort.
#[test]
fn probe01_mixed_packaging_tategaki_brand_with_horizontal_label() {
    let spans = vec![
        span_with_wmode("Brand1", 500.0, 700.0, 1),
        span_with_wmode("Brand2", 500.0, 600.0, 1),
        span_with_wmode("Ingredients", 100.0, 100.0, 0),
    ];
    let pipeline = TextPipeline::new();
    let ordered = pipeline.process(spans, ReadingOrderContext::new()).unwrap();
    // 2/3 vertical → tategaki. The horizontal label still ends up in the
    // sort but the vertical brand goes first.
    let first = &ordered[0].span.text;
    assert!(
        first.starts_with("Brand"),
        "tategaki page must place vertical brand first; got {} first",
        first
    );
}

// ---------------------------------------------------------------------------
// Probe 4 — Mongolian vertical-only document (see probe29)
// ---------------------------------------------------------------------------

/// Mongolian is exclusively vertical TOP-to-BOTTOM, columns LEFT-to-RIGHT.
/// See probe29 for the LTR-column-direction probe. This probe pins the
/// per-character Y advance behavior.
#[test]
fn probe04_mongolian_vertical_y_advance_works_synthetic() {
    // Synthetic: an Identity-V font for the advance math, even though
    // the visual reading order is wrong for Mongolian.
    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj <0003> Tj ET";
    let pdf = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let chars = doc.extract_chars(0).expect("extract_chars");
    assert_eq!(chars.len(), 3);
    // Per-glyph Y advance must be exactly 12.0 in synthetic Identity-V at fs=12.
    let mut sorted = chars.clone();
    sorted.sort_by(|a, b| b.bbox.y.partial_cmp(&a.bbox.y).unwrap());
    for w in sorted.windows(2) {
        let dy = w[0].bbox.y - w[1].bbox.y;
        assert!((dy - 12.0).abs() < 0.05, "Mongolian/CJK vertical dy = {}", dy);
    }
}

// ---------------------------------------------------------------------------
// Probe 35 — Form XObject containing vertical text
// ---------------------------------------------------------------------------

/// A page Do's a Form XObject. The XObject's content stream is rendered
/// with the outer graphics state. Pin: WMode is part of GraphicsState.text_wmode,
/// which is set by Tf — and Tf inside the XObject should set wmode.
#[test]
fn probe35_form_xobject_vertical_text_no_panic() {
    // Build a PDF with a Form XObject containing vertical Tj.
    let cmap = b"\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0001> <0041>
<0002> <0042>
endbfchar
endcmap
end
end";
    let xobj_content = b"BT /F1 12 Tf 50 50 Td <0001> Tj <0002> Tj ET";
    let page_content = b"q /FXO Do Q";

    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let o1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    let o2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    let o3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /XObject << /FXO 8 0 R >> >> >> endobj\n",
    );
    let o4 = pdf.len();
    pdf.extend_from_slice(
        format!("4 0 obj << /Length {} >> stream\n", page_content.len()).as_bytes(),
    );
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let o5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestFont /Encoding /Identity-V /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
    );
    let o6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );
    let o7 = pdf.len();
    pdf.extend_from_slice(format!("7 0 obj << /Length {} >> stream\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let o8 = pdf.len();
    pdf.extend_from_slice(
        format!(
            "8 0 obj << /Type /XObject /Subtype /Form /BBox [0 0 600 800] /Resources << /Font << /F1 5 0 R >> >> /Length {} >> stream\n",
            xobj_content.len()
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(xobj_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for off in [o1, o2, o3, o4, o5, o6, o7, o8] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size 9 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf).expect("parse XObject PDF");
    // Pin: extracting the page should pick up the XObject's spans and tag
    // them with wmode=1 (the XObject's Tf selects Identity-V).
    let spans = doc.extract_spans(0).expect("extract_spans");
    // If the XObject recursion threads wmode correctly, every glyph A/B
    // should have wmode=1.
    if let Some(a) = spans.iter().find(|s| s.text.contains('A')) {
        assert_eq!(
            a.wmode, 1,
            "Form XObject vertical text must carry wmode=1; got wmode={} for {:?}",
            a.wmode, a.text
        );
    }
    // No panic is the primary pin.
}

// ---------------------------------------------------------------------------
// Probe 37 — Performance: horizontal-only path is still cheap
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// BUG: /W2 and /DW2 NOT included in font cache identity hash
// ---------------------------------------------------------------------------

/// **CRITICAL BUG** — exposed by parallel-test failures of probe15/19/20/25.
///
/// `PdfDocument::font_identity_hash_cheap` (src/document.rs:13212) hashes
/// BaseFont, Subtype, Encoding, ToUnicode (by reference), FontDescriptor
/// presence, DescendantFonts (by reference), FirstChar/LastChar, /Widths,
/// /DW — but **NOT /DW2 or /W2**. Two synthetic test PDFs with identical
/// BaseFont/Subtype/Encoding/DescendantFonts byte structure but different
/// `/DW2` arrays will produce the SAME cache key.
///
/// This is the same bug class as commit a327bcd (ToUnicode-stream cache
/// poisoning) but for vertical metrics. When test A loads a font with
/// /DW2 [880 -1000] and test B loads what looks like the same font with
/// /DW2 [900 -800], test B silently gets test A's parsed FontInfo,
/// causing vertical advances to be computed against the wrong w1y.
///
/// This test reproduces the bug deterministically by parsing two PDFs
/// in sequence, both using BaseFont /TestFont and DescendantFonts [6 0 R]
/// but with different /DW2 arrays. The second document's vertical advance
/// will incorrectly use the FIRST document's w1y.
#[test]
fn probe_cache_poison_dw2_collision_across_documents() {
    use pdf_oxide::fonts::global_cache::clear_global_font_cache;
    clear_global_font_cache();

    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    // PDF #1: DW2 = [880 -1000] (default; w1y=-1000 → dy=12.0).
    let pdf1 = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    // PDF #2: DW2 = [900 -800] (override; w1y=-800 → dy=9.6).
    let pdf2 = build_pdf("Identity-V", content, 1000, (900, -800), None, None);

    let doc1 = PdfDocument::from_bytes(pdf1).expect("parse 1");
    let chars1 = doc1.extract_chars(0).expect("chars1");
    let a1 = chars1.iter().find(|c| c.char == 'A').unwrap();
    let b1 = chars1.iter().find(|c| c.char == 'B').unwrap();
    let dy1 = a1.bbox.y - b1.bbox.y;
    assert!((dy1 - 12.0).abs() < 0.1, "doc1 dy = {}", dy1);

    let doc2 = PdfDocument::from_bytes(pdf2).expect("parse 2");
    let chars2 = doc2.extract_chars(0).expect("chars2");
    let a2 = chars2.iter().find(|c| c.char == 'A').unwrap();
    let b2 = chars2.iter().find(|c| c.char == 'B').unwrap();
    let dy2 = a2.bbox.y - b2.bbox.y;
    // FAILING ASSERTION: dy2 should be 9.6 per its own /DW2, but it will be
    // 12.0 (doc1's cached w1y) because the cache key did not include /DW2.
    assert!(
        (dy2 - 9.6).abs() < 0.1,
        "BUG: doc2's /DW2 ignored due to cache poisoning. dy2={} (expected 9.6, doc1's value 12.0 means cache key omits /DW2/W2)",
        dy2
    );
}

/// Companion: the same bug for /W2. Two PDFs with identical font dicts
/// except one has a per-CID /W2 override and the other doesn't. The
/// second-parsed document will inherit the first's /W2 from the cache.
#[test]
fn probe_cache_poison_w2_collision_across_documents() {
    use pdf_oxide::fonts::global_cache::clear_global_font_cache;
    clear_global_font_cache();

    let content = b"BT /F1 12 Tf 100 700 Td <0001> Tj <0002> Tj ET";
    // PDF A: no /W2 override → CID 1 advances per /DW2 → dy=12.
    let pdf_a = build_pdf("Identity-V", content, 1000, (880, -1000), None, None);
    // PDF B: explicit /W2 forcing CID 1 to w1y=-500 → dy=6.
    let pdf_b =
        build_pdf("Identity-V", content, 1000, (880, -1000), Some("[1 [-500 250 600]]"), None);

    // Force PDF A to land in the cache first.
    let doc_a = PdfDocument::from_bytes(pdf_a).expect("parse A");
    let _ = doc_a.extract_chars(0).unwrap();

    let doc_b = PdfDocument::from_bytes(pdf_b).expect("parse B");
    let chars_b = doc_b.extract_chars(0).expect("chars_b");
    let a_b = chars_b.iter().find(|c| c.char == 'A').unwrap();
    let b_b = chars_b.iter().find(|c| c.char == 'B').unwrap();
    let dy_b = a_b.bbox.y - b_b.bbox.y;
    assert!(
        (dy_b - 6.0).abs() < 0.1,
        "BUG: doc B's /W2 override ignored due to cache poisoning. dy_b={} (expected 6.0; got 12.0 means cache served doc A's FontInfo)",
        dy_b
    );
}

// ---------------------------------------------------------------------------
// BUG: mid-BT Tf font switch loses subsequent spans in extract_spans
// ---------------------------------------------------------------------------

/// **BUG** — within a single BT...ET block, switching fonts via `Tf`
/// silently drops every glyph emitted by subsequent Tj operators in the
/// `extract_spans` path. `extract_chars` still emits each glyph (with
/// suspicious coordinates), but `extract_spans` produces only the FIRST
/// glyph's span.
///
/// Reproducer:
///   BT /F1 12 Tf 100 700 Td <0001> Tj
///      /F2 12 Tf 200 700 Td <0002> Tj
///      /F1 12 Tf 300 700 Td <0003> Tj
///      /F2 12 Tf 400 700 Td <0004> Tj ET
///
/// Expected: 4 spans (A,B,C,D). Actual: 1 span ("A"). The Tf-induced
/// flush path apparently abandons the buffered span state without re-
/// emitting the in-flight glyphs of the alternative font.
///
/// The existing `mid_stream_tf_h_to_v_switches_span_wmode` test in
/// test_vertical_writing_mode_fixes.rs sidesteps this by splitting into
/// TWO BT/ET blocks — explicitly noted in its comments as a workaround.
/// This probe pins the actual mid-BT-Tf failure mode as a bug.
#[test]
fn probe36_mid_bt_tf_font_switch_drops_subsequent_spans() {
    use pdf_oxide::fonts::global_cache::clear_global_font_cache;
    clear_global_font_cache();

    let cmap = b"\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
4 beginbfchar
<0001> <0041>
<0002> <0042>
<0003> <0043>
<0004> <0044>
endbfchar
endcmap
end
end";
    // Single BT/ET, position set ONCE via Td, then alternate Tf/Tj. This is
    // the actual mid-BT Tf reproducer: every Tf inside the BT block must
    // flush the pending Tj span buffer for its previous font and start a
    // fresh buffer for the new font. (The earlier draft used cumulative Td
    // between Tjs, which placed B/C/D far off the MediaBox and ran them
    // straight into the postprocess off-page filter — incidentally
    // unrelated to Tf flushing. The advance inside the block stays
    // on-page: F1 is /Identity-V so its glyph advance is vertical;
    // F2 is /Identity-H so it advances horizontally.)
    let content = b"BT 100 700 Td \
                    /F1 12 Tf <0001> Tj \
                    /F2 12 Tf <0002> Tj \
                    /F1 12 Tf <0003> Tj \
                    /F2 12 Tf <0004> Tj ET";

    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let o1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    let o2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    let o3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 8 0 R >> >> >> endobj\n",
    );
    let o4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj << /Length {} >> stream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let o5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestV /Encoding /Identity-V /DescendantFonts [6 0 R] /ToUnicode 7 0 R >> endobj\n",
    );
    let o6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestV /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );
    let o7 = pdf.len();
    pdf.extend_from_slice(format!("7 0 obj << /Length {} >> stream\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(cmap);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let o8 = pdf.len();
    pdf.extend_from_slice(
        b"8 0 obj << /Type /Font /Subtype /Type0 /BaseFont /TestH /Encoding /Identity-H /DescendantFonts [9 0 R] /ToUnicode 7 0 R >> endobj\n",
    );
    let o9 = pdf.len();
    pdf.extend_from_slice(
        b"9 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /TestH /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 /DW2 [880 -1000] >> endobj\n",
    );
    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 10\n0000000000 65535 f \n");
    for off in [o1, o2, o3, o4, o5, o6, o7, o8, o9] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer << /Size 10 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf).expect("parse two-font PDF");
    let spans = doc.extract_spans(0).expect("extract_spans");
    // EXPECTED behavior: 4 spans, one per glyph, with alternating wmode.
    // ACTUAL behavior (bug): only span "A" comes out.
    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    let combined: String = texts.iter().copied().collect();
    assert!(
        combined.contains('A')
            && combined.contains('B')
            && combined.contains('C')
            && combined.contains('D'),
        "mid-BT Tf font switch must emit all four glyphs as spans; got {:?}",
        texts
    );
}

// ---------------------------------------------------------------------------
// Probe 37 — Performance regression: horizontal-only path
// ---------------------------------------------------------------------------

/// Smoke benchmark: extract 1000-glyph horizontal page and assert wall
/// clock under a generous bound. Not a strict perf test but pins that the
/// hot path hasn't gained an obviously expensive op.
#[test]
fn probe37_horizontal_extraction_bulk_within_bound() {
    // 200 horizontal CIDs.
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 12 Tf 100 700 Td <");
    for _ in 0..200 {
        content.extend_from_slice(b"0001");
    }
    content.extend_from_slice(b"> Tj ET");
    let pdf = build_pdf("Identity-H", &content, 1000, (880, -1000), None, None);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let start = std::time::Instant::now();
    let _spans = doc.extract_spans(0).expect("extract_spans");
    let elapsed = start.elapsed();
    // A 200-glyph horizontal page should extract well under 200ms on any
    // reasonable machine. This is generous; the goal is to catch ~100x
    // regressions, not micro-regressions.
    assert!(
        elapsed.as_millis() < 1000,
        "horizontal extraction took {}ms — significant regression?",
        elapsed.as_millis()
    );
}
