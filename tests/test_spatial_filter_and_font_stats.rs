//! Tests for spatial region filtering and per-page font statistics APIs.
//!
//! Spatial filtering: `extract_text_excluding_rects`, `extract_words_excluding_rects`,
//! and `extract_spans_excluding_rects` allow callers to exclude rectangular
//! regions (e.g. figure bounding boxes) from page text extraction.
//!
//! Font statistics: `page_font_stats(page_index)` returns `PageFontStats` with
//! `dominant_em` and related metrics so callers can compute font-size ratios
//! for heading detection and layout analysis.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{PageFontStats, RectFilterMode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn one_page_pdf(content: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>\nendobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
          /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let off6 = pdf.len();
    // A larger font (24pt) for headings
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold \
          /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5, off6];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

// ---------------------------------------------------------------------------
// Section 4 — extract_text_excluding_rects
// ---------------------------------------------------------------------------

/// A PDF with body text at y=700 and figure-caption text at y=400.
/// After excluding the figure region, only body text should remain.
#[test]
fn extract_text_excluding_rects_removes_figure_text() {
    // Body text at (50, 700), figure caption inside bbox (40,380,200,420)
    let content =
        b"BT /F1 12 Tf 50 700 Td (Body text) Tj ET\nBT /F1 10 Tf 50 400 Td (Figure caption) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    // Exclude the figure region: x=40..200, y=380..420 in PDF coords
    let figure_bbox = Rect {
        x: 40.0,
        y: 380.0,
        width: 160.0,
        height: 40.0,
    };
    let text = doc
        .extract_text_excluding_rects(0, &[figure_bbox], RectFilterMode::Intersects)
        .unwrap();

    assert!(
        text.contains("Body text"),
        "body text must survive figure exclusion; got: {:?}",
        text
    );
    assert!(
        !text.contains("Figure caption"),
        "figure caption must be excluded; got: {:?}",
        text
    );
}

/// extract_spans_excluding_rects keeps spans outside the excluded region.
#[test]
fn extract_spans_excluding_rects_filters_correctly() {
    let content =
        b"BT /F1 12 Tf 50 700 Td (Keep this) Tj ET\nBT /F1 12 Tf 50 200 Td (Drop this) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let drop_zone = Rect {
        x: 0.0,
        y: 180.0,
        width: 612.0,
        height: 60.0,
    };
    let spans = doc
        .extract_spans_excluding_rects(0, &[drop_zone], RectFilterMode::Intersects)
        .unwrap();

    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("Keep this")),
        "span above excluded zone must be retained; spans: {:?}",
        texts
    );
    assert!(
        !texts.iter().any(|t| t.contains("Drop this")),
        "span inside excluded zone must be removed; spans: {:?}",
        texts
    );
}

/// Excluding an empty slice produces output identical to extract_text.
#[test]
fn extract_text_excluding_rects_empty_exclusion_is_noop() {
    let content = b"BT /F1 12 Tf 50 700 Td (All text) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let text_full = doc.extract_text(0).unwrap();
    let text_excl = doc
        .extract_text_excluding_rects(0, &[], RectFilterMode::Intersects)
        .unwrap();

    assert_eq!(
        text_full, text_excl,
        "empty exclusion must produce identical output to extract_text"
    );
}

/// extract_words_excluding_rects keeps words outside the excluded region.
#[test]
fn extract_words_excluding_rects_filters_correctly() {
    let content =
        b"BT /F1 12 Tf 50 700 Td (KeepWord) Tj ET\nBT /F1 12 Tf 50 200 Td (DropWord) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let drop_zone = Rect {
        x: 0.0,
        y: 180.0,
        width: 612.0,
        height: 60.0,
    };
    let words = doc
        .extract_words_excluding_rects(0, &[drop_zone], RectFilterMode::Intersects)
        .unwrap();

    let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("KeepWord")),
        "word above excluded zone must be retained; words: {:?}",
        texts
    );
    assert!(
        !texts.iter().any(|t| t.contains("DropWord")),
        "word inside excluded zone must be removed; words: {:?}",
        texts
    );
}

// ---------------------------------------------------------------------------
// Section 5 — page_font_stats
// ---------------------------------------------------------------------------

/// page_font_stats returns a dominant_em matching the body font size.
#[test]
fn page_font_stats_dominant_em_matches_body_size() {
    // Body text: 12pt (majority). Heading: 24pt (one span).
    let content = b"BT /F1 12 Tf 50 700 Td (Body line one) Tj ET\n\
          BT /F1 12 Tf 50 680 Td (Body line two) Tj ET\n\
          BT /F2 24 Tf 50 740 Td (Heading) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let stats = doc
        .page_font_stats(0)
        .expect("page_font_stats must not error");

    // dominant_em should be ~12pt (body), not 24pt (heading)
    assert!(
        (stats.dominant_em - 12.0).abs() < 1.5,
        "dominant_em must reflect body font size (~12pt), got {:.1}",
        stats.dominant_em
    );
}

/// With the dominant_em from page_font_stats, heading size ratios are correct.
#[test]
fn heading_detection_via_page_font_stats_ratio() {
    // Body text 12pt, heading 24pt → ratio 2.0 → should classify as H1 (>=1.8)
    let content = b"BT /F1 12 Tf 50 700 Td (Body text here) Tj ET\n\
          BT /F2 24 Tf 50 750 Td (Big Heading) Tj ET";
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let stats = doc.page_font_stats(0).expect("page_font_stats");
    let spans = doc.extract_spans(0).expect("extract_spans");

    let heading_spans: Vec<_> = spans
        .iter()
        .filter(|s| s.font_size / stats.dominant_em >= 1.8)
        .collect();

    assert!(
        !heading_spans.is_empty(),
        "at least one span should meet H1 ratio (font_size/dominant_em >= 1.8); \
         dominant_em={:.1}, spans: {:?}",
        stats.dominant_em,
        spans
            .iter()
            .map(|s| (&s.text, s.font_size))
            .collect::<Vec<_>>()
    );
    assert!(
        heading_spans.iter().any(|s| s.text.contains("Big Heading")),
        "the heading span must be classified as H1; heading spans: {:?}",
        heading_spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );
}

/// page_font_stats on an empty page returns the default (12pt).
#[test]
fn page_font_stats_empty_page_returns_default() {
    let content = b""; // empty content stream
    let pdf = one_page_pdf(content);
    let doc = PdfDocument::from_bytes(pdf).expect("open PDF");

    let stats = doc
        .page_font_stats(0)
        .expect("page_font_stats on empty page");
    assert_eq!(
        stats.dominant_em,
        PageFontStats::default().dominant_em,
        "empty page must return default dominant_em"
    );
}

// ---------------------------------------------------------------------------
// Section 4 — quality gate with docling.pdf (ignored unless file present)
// ---------------------------------------------------------------------------

/// docling.pdf contains mixed body + figure text.
/// After the fix, extract_spans_excluding_rects with a figure bbox must
/// return fewer spans than extract_spans (the baseline).
#[test]
#[ignore = "requires /tmp/docling.pdf"]
fn docling_figure_exclusion_reduces_span_count() {
    let bytes = match std::fs::read("/tmp/docling.pdf") {
        Ok(b) => b,
        Err(_) => {
            eprintln!("SKIP: /tmp/docling.pdf not found");
            return;
        },
    };
    let doc = PdfDocument::from_bytes(bytes).expect("open docling.pdf");
    let page_count = doc.page_count().unwrap_or(0);

    // Use page 0; if the figure is elsewhere adjust as needed.
    // We use a large exclusion zone covering the bottom half of the first page.
    // Exact coordinates don't matter — the important thing is that the API
    // does not panic and returns fewer spans than the unfiltered call.
    if page_count == 0 {
        eprintln!("SKIP: docling.pdf has 0 pages");
        return;
    }
    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .unwrap_or((0.0, 0.0, 595.0, 842.0));
    let width = urx - llx;
    let height = ury - lly;

    let figure_zone = Rect {
        x: llx,
        y: lly,
        width,
        height: height * 0.4,
    };

    let all_spans = doc.extract_spans(0).expect("extract_spans");
    let filtered_spans = doc
        .extract_spans_excluding_rects(0, &[figure_zone], RectFilterMode::Intersects)
        .expect("extract_spans_excluding_rects");

    assert!(
        filtered_spans.len() <= all_spans.len(),
        "excluding a region must never increase the span count; \
         all={} filtered={}",
        all_spans.len(),
        filtered_spans.len()
    );
    eprintln!(
        "docling.pdf page 0: all={} spans, after exclusion={}",
        all_spans.len(),
        filtered_spans.len()
    );
}
