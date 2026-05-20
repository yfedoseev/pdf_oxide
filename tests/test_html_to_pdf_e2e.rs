//! End-to-end tests for `Pdf::from_html_css` — the v0.3.35
//! HTML+CSS→PDF pipeline (issue #248).
//!
//! Walks the full path: HTML parse → cascade → box tree → Taffy
//! layout → paginate → paint → PdfWriter → re-open via PdfDocument →
//! extract_text round-trip.

use pdf_oxide::api::Pdf;
use pdf_oxide::PdfDocument;

const DEJAVU: &[u8] = include_bytes!("fixtures/fonts/DejaVuSans.ttf");
const DEJAVU_MONO: &[u8] = include_bytes!("fixtures/fonts/DejaVuSansMono.ttf");

fn build_and_extract(html: &str, css: &str) -> String {
    let pdf = Pdf::from_html_css(html, css, DEJAVU.to_vec()).expect("from_html_css");
    let bytes = pdf.into_bytes();
    let doc = PdfDocument::from_bytes(bytes).expect("re-open PDF");
    let pages = doc.page_count().expect("page count");
    let mut out = String::new();
    for i in 0..pages {
        out.push_str(&doc.extract_text(i).expect("extract_text"));
        out.push('\n');
    }
    out
}

#[test]
fn simple_paragraph_round_trips() {
    let extracted = build_and_extract("<p>Hello, world!</p>", "");
    assert!(
        extracted.contains("Hello, world!"),
        "expected 'Hello, world!' in: {extracted:?}"
    );
}

#[test]
fn multi_paragraph_round_trips() {
    let extracted = build_and_extract("<p>First paragraph.</p><p>Second paragraph.</p>", "");
    assert!(extracted.contains("First paragraph."));
    assert!(extracted.contains("Second paragraph."));
}

#[test]
fn nested_html_round_trips() {
    let extracted = build_and_extract("<div><h1>Title</h1><p>Body text here.</p></div>", "");
    assert!(extracted.contains("Title"));
    assert!(extracted.contains("Body text here."));
}

#[test]
fn css_styling_does_not_lose_text() {
    let extracted = build_and_extract(
        "<h1>Header</h1><p>Body.</p>",
        "h1 { color: blue; font-size: 24pt } p { color: gray }",
    );
    assert!(extracted.contains("Header"));
    assert!(extracted.contains("Body"));
}

#[test]
fn unicode_round_trips() {
    let extracted = build_and_extract("<p>café Привет ❤</p>", "");
    assert!(extracted.contains("café"));
    assert!(extracted.contains("Привет"));
}

/// B1 RED — three sibling `<p>` elements must each emit ALL their
/// words, not just the first one. The 10-doc cross-render corpus
/// demonstrated only ~20% of words survive; this test catches that
/// regression at unit-test granularity by counting words after
/// extraction.
#[test]
fn three_paragraphs_emit_all_words_in_order() {
    let extracted = build_and_extract(
        "<p>Alpha beta gamma delta epsilon zeta.</p>\
         <p>One two three four five six seven eight.</p>\
         <p>Red orange yellow green blue indigo violet.</p>",
        "",
    );
    let normalized: String = extracted.split_whitespace().collect::<Vec<_>>().join(" ");
    // Each paragraph's full content must survive.
    let p1_words = ["Alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
    let p2_words = [
        "One", "two", "three", "four", "five", "six", "seven", "eight",
    ];
    let p3_words = [
        "Red", "orange", "yellow", "green", "blue", "indigo", "violet",
    ];
    for w in p1_words
        .iter()
        .chain(p2_words.iter())
        .chain(p3_words.iter())
    {
        assert!(normalized.contains(w), "missing word `{w}` from extracted text:\n{extracted}");
    }
    // Order: the three paragraphs' anchor words must appear in order.
    let alpha = normalized.find("Alpha").unwrap_or(usize::MAX);
    let one = normalized.find("One").unwrap_or(usize::MAX);
    let red = normalized.find("Red").unwrap_or(usize::MAX);
    assert!(alpha < one, "Alpha must precede One in paragraph order");
    assert!(one < red, "One must precede Red in paragraph order");
}

/// B2 RED — a single `<p>` with ~50 words must round-trip ≥ 90 % of
/// its words. The cross-render harness showed ~20 % retention on
/// multi-line paragraphs (bodies after the first word lost their
/// position). This test exercises the inline formatter at moderate
/// length without involving sibling paragraphs.
#[test]
fn long_single_paragraph_keeps_all_words() {
    let words: Vec<String> = (1..=60).map(|i| format!("word{i}")).collect();
    let body = words.join(" ");
    let html = format!("<p>{body}</p>");
    let extracted = build_and_extract(&html, "");
    let normalized: String = extracted.split_whitespace().collect::<Vec<_>>().join(" ");
    let present = words
        .iter()
        .filter(|w| normalized.contains(w.as_str()))
        .count();
    let ratio = present as f32 / words.len() as f32;
    assert!(
        ratio >= 0.90,
        "long paragraph retained only {present}/{} words ({:.0}%): {extracted}",
        words.len(),
        ratio * 100.0
    );
}

/// B1 visual-positioning RED — beyond text content, the SPAN
/// Y-coordinates must reflect the document's logical paragraph order.
/// `extract_text` reads in stream order (so it can pass even when the
/// PDF is visually broken); a stronger test inspects bbox.y on
/// extracted spans and asserts each paragraph's first-word span sits
/// at a strictly lower y than the previous (PDF coordinates: y=0 is
/// page bottom, y grows up).
#[test]
fn three_paragraphs_have_decreasing_y_baselines() {
    let pdf = Pdf::from_html_css(
        "<p>Alpha first paragraph.</p>\
         <p>Beta second paragraph.</p>\
         <p>Gamma third paragraph.</p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("re-open PDF");
    let spans = doc.extract_spans(0).expect("extract_spans");
    let pos = |needle: &str| -> Option<f32> {
        spans
            .iter()
            .find(|s| s.text.contains(needle))
            .map(|s| s.bbox.y)
    };
    let alpha_y = pos("Alpha").expect("Alpha span missing");
    let beta_y = pos("Beta").expect("Beta span missing");
    let gamma_y = pos("Gamma").expect("Gamma span missing");
    assert!(
        alpha_y > beta_y,
        "Alpha (y={alpha_y}) must sit ABOVE Beta (y={beta_y}) on the page (PDF y grows up)"
    );
    assert!(beta_y > gamma_y, "Beta (y={beta_y}) must sit ABOVE Gamma (y={gamma_y})");
    // And: each paragraph's body words must share that paragraph's
    // baseline (within one line height), not be scattered across the
    // page at increasing x.
    let alpha_body_y = pos("first").expect("`first` span missing");
    assert!(
        (alpha_body_y - alpha_y).abs() < 30.0,
        "`first` (y={alpha_body_y}) must sit on Alpha's line (y={alpha_y})"
    );
}

/// FU7 — `<a href>` must emit a PDF link annotation. Smoke-test at the
/// byte level (no public link-extraction API yet) by looking for the
/// `/URI (…)` action in the PDF stream.
#[test]
fn anchor_link_emits_uri_annotation() {
    let pdf = Pdf::from_html_css(
        "<p>Visit <a href=\"https://example.com\">example.com</a> today.</p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let bytes = pdf.into_bytes();
    let s = String::from_utf8_lossy(&bytes);
    assert!(
        s.contains("/Subtype /Link") || s.contains("/Subtype/Link"),
        "expected a /Link annotation in the PDF stream"
    );
    assert!(
        s.contains("example.com"),
        "expected the href `example.com` to appear in the /URI action"
    );
}

/// FU1 — `page-break-before: always` and `page-break-after: always`
/// move content onto a new page.
#[test]
fn page_break_before_forces_new_page() {
    let pdf = Pdf::from_html_css(
        "<p>First page content.</p>\
         <h1 class=\"pb\">Second Page</h1>\
         <p>Body on page two.</p>",
        ".pb { page-break-before: always }",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("re-open");
    let pages = doc.page_count().expect("page_count");
    assert!(pages >= 2, "page-break-before should yield ≥2 pages; got {pages}");
    let p0 = doc.extract_text(0).expect("p0");
    let p1 = doc.extract_text(1).expect("p1");
    assert!(p0.contains("First"), "page 0 should have `First`; got {p0:?}");
    assert!(
        p1.contains("Second Page") || p1.contains("Second"),
        "page 1 should have `Second Page`; got {p1:?}"
    );
}

/// FU1 (b) — inline `style="page-break-before: always"` also works.
#[test]
fn inline_style_page_break_before_forces_new_page() {
    let pdf = Pdf::from_html_css(
        "<p>First page.</p>\
         <h1 style=\"page-break-before: always\">Second</h1>\
         <p>Two.</p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("re-open");
    assert!(doc.page_count().unwrap() >= 2, "inline page-break-before should yield ≥2 pages");
}

/// B4/FU8 — `font-family` picks a registered font; unknown families
/// fall back to the default (first-registered) font.
#[test]
fn multi_font_cascade_selects_registered_family() {
    let pdf = Pdf::from_html_css_with_fonts(
        "<p>body</p><p class=\"m\">mono</p>",
        ".m { font-family: 'DejaVu Sans Mono' }",
        vec![
            ("DejaVu Sans".to_string(), DEJAVU.to_vec()),
            ("DejaVu Sans Mono".to_string(), DEJAVU_MONO.to_vec()),
        ],
    )
    .expect("from_html_css_with_fonts");
    let bytes = pdf.into_bytes();
    let font_file_count = String::from_utf8_lossy(&bytes)
        .matches("/FontFile2")
        .count();
    let doc = PdfDocument::from_bytes(bytes).expect("reopen");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("body"));
    assert!(text.contains("mono"));
    assert!(font_file_count >= 2, "expected ≥2 embedded fonts in PDF; got {font_file_count}");
}

/// FU6 — `<ul>` gets bullet markers, `<ol>` gets numeric markers.
#[test]
fn list_markers_are_emitted() {
    let extracted = build_and_extract(
        "<ul><li>Apple</li><li>Banana</li></ul><ol><li>First</li><li>Second</li></ol>",
        "",
    );
    // Bullet char (U+2022)
    assert!(
        extracted.contains('\u{2022}') || extracted.contains("•"),
        "expected a bullet marker; got {extracted:?}"
    );
    assert!(
        extracted.contains("1.") && extracted.contains("2."),
        "expected numeric markers `1.` and `2.`; got {extracted:?}"
    );
}

#[test]
fn produces_valid_pdf_header() {
    let pdf = Pdf::from_html_css("<p>x</p>", "", DEJAVU.to_vec()).unwrap();
    let bytes = pdf.into_bytes();
    assert!(bytes.starts_with(b"%PDF-1.7"));
}

// ─────────────────────────────────────────────────────────────────────
// CR1 — tokenizer char-boundary safety on multi-byte content. The CSS
// parser ignore_case lookahead panicked on non-ASCII byte boundaries
// before the fix; an end-to-end call with multi-byte CSS exercises the
// hot path behind the public API (unit test already covers the direct
// function).
// ─────────────────────────────────────────────────────────────────────
#[test]
fn multibyte_css_selectors_do_not_panic() {
    let extracted = build_and_extract("<p class=\"é\">café content</p>", ".é { color: red }");
    assert!(extracted.contains("café"));
}

// ─────────────────────────────────────────────────────────────────────
// B3 — Arabic/RTL paragraph is shaped via rustybuzz and the glyph
// stream survives a PdfDocument reload.
// ─────────────────────────────────────────────────────────────────────
#[cfg(feature = "system-fonts")]
#[test]
fn arabic_rtl_paragraph_shapes_and_renders() {
    let pdf = Pdf::from_html_css("<p>هذا نص عربي</p>", "", DEJAVU.to_vec()).expect("from_html_css");
    let bytes = pdf.into_bytes();
    // The rustybuzz path emits a hex-encoded TJ stream; just assert
    // the PDF opens and has one page. Visual correctness is covered by
    // the unit-level font_shaping tests.
    let doc = PdfDocument::from_bytes(bytes).expect("reopen");
    assert_eq!(doc.page_count().expect("pages"), 1);
}

// ─────────────────────────────────────────────────────────────────────
// FU1 — page-break-after is honoured (mirror of the existing
// page-break-before test). Also verifies multiple successive breaks
// accumulate pages rather than overwriting each other.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn page_break_after_opens_fresh_page() {
    let pdf = Pdf::from_html_css(
        "<p class=\"pba\">Cover.</p>\
         <p>Main body.</p>",
        ".pba { page-break-after: always }",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("reopen");
    assert!(doc.page_count().unwrap() >= 2);
    let p0 = doc.extract_text(0).unwrap();
    let p1 = doc.extract_text(1).unwrap();
    assert!(p0.contains("Cover"), "page 0 got {p0:?}");
    assert!(p1.contains("Main body"), "page 1 got {p1:?}");
}

#[test]
fn multiple_page_breaks_accumulate_pages() {
    let pdf = Pdf::from_html_css(
        "<p>One.</p>\
         <h1 class=\"pb\">Two</h1>\
         <p>Two body.</p>\
         <h1 class=\"pb\">Three</h1>\
         <p>Three body.</p>",
        ".pb { page-break-before: always }",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("reopen");
    assert!(doc.page_count().unwrap() >= 3, "two page-breaks should yield ≥3 pages");
}

// ─────────────────────────────────────────────────────────────────────
// FU2 — ::before / ::after generated content. Covers literal strings,
// attr(), quotes, and both-at-once on the same host element.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn pseudo_before_literal_string_emits_text() {
    let extracted =
        build_and_extract("<p class=\"note\">Remember</p>", ".note::before { content: \"§ \" }");
    assert!(
        extracted.contains("§") || extracted.contains("\u{00a7}"),
        "expected `§` from ::before; got {extracted:?}"
    );
    assert!(extracted.contains("Remember"));
}

#[test]
fn pseudo_after_literal_string_emits_text() {
    let extracted =
        build_and_extract("<p class=\"note\">Header</p>", ".note::after { content: \" ✓\" }");
    assert!(
        extracted.contains("✓") || extracted.contains("\u{2713}"),
        "expected `✓` from ::after; got {extracted:?}"
    );
}

#[test]
fn pseudo_before_and_after_both_emit() {
    let extracted = build_and_extract(
        "<p>middle</p>",
        "p::before { content: \"PRE \" } p::after { content: \" POST\" }",
    );
    assert!(extracted.contains("PRE"));
    assert!(extracted.contains("POST"));
    assert!(extracted.contains("middle"));
}

#[test]
fn pseudo_attr_function_resolves_attribute() {
    let extracted = build_and_extract(
        "<p data-label=\"HINT\">body</p>",
        "p::before { content: attr(data-label) }",
    );
    assert!(extracted.contains("HINT"), "attr(data-label) should resolve; got {extracted:?}");
}

#[test]
fn pseudo_content_none_emits_nothing() {
    let extracted = build_and_extract("<p>body</p>", "p::before { content: none }");
    assert!(extracted.contains("body"));
    // No crash / no stray markers is the main assertion — the
    // literal string from content: none should NOT appear in output.
    assert!(!extracted.contains("none"));
}

// ─────────────────────────────────────────────────────────────────────
// FU3 — opacity <=0.01 hides the element *and* its text descendants;
// transform: translate shifts the baseline.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn opacity_zero_hides_paragraph_and_text() {
    let extracted = build_and_extract(
        "<p>visible-para</p>\
         <p style=\"opacity: 0\">invisible-para</p>",
        "",
    );
    assert!(extracted.contains("visible-para"));
    assert!(
        !extracted.contains("invisible-para"),
        "opacity:0 text should not render; got {extracted:?}"
    );
}

#[test]
fn opacity_zero_on_ancestor_hides_descendants() {
    let extracted = build_and_extract(
        "<div style=\"opacity: 0\"><p>child-text</p></div>\
         <p>after-text</p>",
        "",
    );
    assert!(extracted.contains("after-text"));
    assert!(
        !extracted.contains("child-text"),
        "ancestor opacity:0 should hide descendants; got {extracted:?}"
    );
}

#[test]
fn opacity_above_threshold_still_renders() {
    let extracted = build_and_extract("<p style=\"opacity: 0.5\">halfway</p>", "");
    assert!(extracted.contains("halfway"));
}

#[test]
fn translate_shifts_text_baseline_in_x() {
    // Apply translateX(100px) to one paragraph; its `shifted` span
    // should sit further right than the baseline paragraph.
    let pdf = Pdf::from_html_css(
        "<p>baseline</p>\
         <p style=\"transform: translateX(100px)\">shifted</p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("reopen");
    let spans = doc.extract_spans(0).expect("spans");
    let baseline_x = spans
        .iter()
        .find(|s| s.text.contains("baseline"))
        .expect("baseline span")
        .bbox
        .x;
    let shifted_x = spans
        .iter()
        .find(|s| s.text.contains("shifted"))
        .expect("shifted span")
        .bbox
        .x;
    assert!(
        shifted_x > baseline_x + 50.0,
        "shifted (x={shifted_x}) should sit ≥50pt right of baseline (x={baseline_x})"
    );
}

// ─────────────────────────────────────────────────────────────────────
// FU4 — <img> data-URI embedding yields an /XObject Image in the PDF.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn data_uri_png_image_becomes_xobject() {
    // 1×1 transparent PNG via base64.
    let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=";
    let html =
        format!("<p>before</p><img src=\"{data_uri}\" width=\"32\" height=\"32\"><p>after</p>");
    let pdf = Pdf::from_html_css(&html, "", DEJAVU.to_vec()).expect("from_html_css");
    let bytes = pdf.into_bytes();
    let s = String::from_utf8_lossy(&bytes);
    assert!(
        s.contains("/Subtype /Image") || s.contains("/Subtype/Image"),
        "expected an Image XObject in PDF"
    );
    // Text around the image must still round-trip.
    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("before"));
    assert!(text.contains("after"));
}

#[test]
fn missing_image_src_does_not_panic() {
    // Broken / empty / non-data URIs should produce a PDF without
    // panicking — the image is silently dropped.
    let pdf = Pdf::from_html_css(
        "<p>a</p><img src=\"https://example.com/cat.png\"><p>b</p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let doc = PdfDocument::from_bytes(pdf.into_bytes()).expect("reopen");
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("a"));
    assert!(text.contains("b"));
}

// ─────────────────────────────────────────────────────────────────────
// FU6 — list markers for nested lists.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn nested_unordered_list_has_bullets_on_both_levels() {
    let extracted = build_and_extract("<ul><li>Outer<ul><li>Inner</li></ul></li></ul>", "");
    let bullets = extracted.matches('\u{2022}').count();
    assert!(bullets >= 2, "expected ≥2 bullets in nested ul; got {bullets} in {extracted:?}");
    assert!(extracted.contains("Outer"));
    assert!(extracted.contains("Inner"));
}

#[test]
fn ordered_list_numbers_sequentially_across_many_items() {
    let extracted =
        build_and_extract("<ol><li>A</li><li>B</li><li>C</li><li>D</li><li>E</li></ol>", "");
    for marker in ["1.", "2.", "3.", "4.", "5."] {
        assert!(extracted.contains(marker), "missing `{marker}` from {extracted:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// FU7 — anchor link annotations, multiple hrefs, and the href text
// body round-trips alongside.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn multiple_anchor_links_all_emit_annotations() {
    let pdf = Pdf::from_html_css(
        "<p>one <a href=\"https://a.example\">A</a></p>\
         <p>two <a href=\"https://b.example\">B</a></p>\
         <p>three <a href=\"https://c.example\">C</a></p>",
        "",
        DEJAVU.to_vec(),
    )
    .expect("from_html_css");
    let bytes = pdf.into_bytes();
    let s = String::from_utf8_lossy(&bytes);
    let link_count = s.matches("/Subtype /Link").count() + s.matches("/Subtype/Link").count();
    assert!(link_count >= 3, "expected ≥3 /Link annotations; got {link_count}");
    for host in ["a.example", "b.example", "c.example"] {
        assert!(s.contains(host), "expected `{host}` href in /URI action");
    }
}

#[test]
fn anchor_without_href_emits_no_link_annotation() {
    let pdf =
        Pdf::from_html_css("<p><a>plain text</a></p>", "", DEJAVU.to_vec()).expect("from_html_css");
    let bytes = pdf.into_bytes();
    let s = String::from_utf8_lossy(&bytes);
    let links = s.matches("/Subtype /Link").count() + s.matches("/Subtype/Link").count();
    assert_eq!(links, 0, "href-less <a> must not emit a /Link");
    let doc = PdfDocument::from_bytes(bytes).expect("reopen");
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("plain text"));
}

// ─────────────────────────────────────────────────────────────────────
// B4/FU8 — multi-font fallback: unknown family falls back to default,
// matched family picks the right resource, quoted and bare names both
// resolve.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn unknown_font_family_falls_back_to_default() {
    let extracted = {
        let pdf = Pdf::from_html_css_with_fonts(
            "<p style=\"font-family: 'Not A Real Font'\">fallback</p>",
            "",
            vec![("DejaVu Sans".to_string(), DEJAVU.to_vec())],
        )
        .expect("from_html_css_with_fonts");
        let bytes = pdf.into_bytes();
        let doc = PdfDocument::from_bytes(bytes).expect("reopen");
        doc.extract_text(0).expect("extract")
    };
    assert!(extracted.contains("fallback"));
}

#[test]
fn bare_and_quoted_font_family_both_resolve() {
    // Quoted family on one paragraph, bare identifier on another —
    // both should match the registered family.
    let pdf = Pdf::from_html_css_with_fonts(
        "<p style=\"font-family: 'DejaVu Sans Mono'\">quoted</p>\
         <p style=\"font-family: DejaVu Sans\">bare</p>",
        "",
        vec![
            ("DejaVu Sans".to_string(), DEJAVU.to_vec()),
            ("DejaVu Sans Mono".to_string(), DEJAVU_MONO.to_vec()),
        ],
    )
    .expect("from_html_css_with_fonts");
    let bytes = pdf.into_bytes();
    let doc = PdfDocument::from_bytes(bytes).expect("reopen");
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("quoted"));
    assert!(text.contains("bare"));
}

#[test]
fn three_font_cascade_embeds_each_font_file() {
    let pdf = Pdf::from_html_css_with_fonts(
        "<p>regular</p>\
         <p class=\"b\">bold</p>\
         <p class=\"m\">mono</p>",
        ".b { font-family: 'DejaVu Sans Bold' } \
         .m { font-family: 'DejaVu Sans Mono' }",
        vec![
            ("DejaVu Sans".to_string(), DEJAVU.to_vec()),
            (
                "DejaVu Sans Bold".to_string(),
                std::fs::read("tests/fixtures/fonts/DejaVuSans-Bold.ttf").unwrap(),
            ),
            ("DejaVu Sans Mono".to_string(), DEJAVU_MONO.to_vec()),
        ],
    )
    .expect("from_html_css_with_fonts");
    let bytes = pdf.into_bytes();
    let s = String::from_utf8_lossy(&bytes);
    let embed_count = s.matches("/FontFile2").count();
    assert!(
        embed_count >= 3,
        "expected ≥3 embedded fonts (one per registered family); got {embed_count}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Kitchen-sink — every feature in this branch wired into one document
// so regressions in interaction surface here.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn kitchen_sink_document_round_trips_all_features() {
    let html = "\
        <h1 id=\"t\">Title</h1>\
        <p>Intro paragraph with <a href=\"https://example.com\">a link</a>.</p>\
        <ul><li>Apple</li><li>Banana</li></ul>\
        <ol><li>First</li><li>Second</li></ol>\
        <p class=\"note\">Section body.</p>\
        <h1 class=\"pb\">Chapter Two</h1>\
        <p style=\"opacity: 0\">secret</p>\
        <p style=\"transform: translateX(50px)\">shifted</p>\
        <p>End.</p>\
    ";
    let css = "\
        .pb { page-break-before: always } \
        .note::before { content: \"NOTE: \" } \
        p { color: black }\
    ";
    let pdf = Pdf::from_html_css(html, css, DEJAVU.to_vec()).expect("from_html_css");
    let bytes = pdf.into_bytes();

    // Multi-page via page-break.
    let doc = PdfDocument::from_bytes(bytes.clone()).expect("reopen");
    let pages = doc.page_count().unwrap();
    assert!(pages >= 2, "expected ≥2 pages, got {pages}");

    // Page 0 — intro content, bullets, numbers, ::before text.
    let p0 = doc.extract_text(0).unwrap();
    assert!(p0.contains("Title"));
    assert!(p0.contains("Intro"));
    assert!(p0.contains('\u{2022}') || p0.contains("•"), "bullet missing from {p0:?}");
    assert!(p0.contains("1.") && p0.contains("2."), "numbers missing from {p0:?}");
    assert!(p0.contains("NOTE:"), "::before missing from {p0:?}");
    assert!(p0.contains("Apple") && p0.contains("Banana"));

    // Page 1 — after page-break, with opacity:0 and translate effects.
    let p1 = doc.extract_text(1).unwrap();
    assert!(p1.contains("Chapter Two"));
    assert!(!p1.contains("secret"), "opacity:0 text leaked: {p1:?}");
    assert!(p1.contains("shifted"));
    assert!(p1.contains("End"));

    // Link annotation survived.
    let s = String::from_utf8_lossy(&bytes);
    assert!(
        s.contains("/Subtype /Link") || s.contains("/Subtype/Link"),
        "no /Link annotation in kitchen-sink PDF"
    );
    assert!(s.contains("example.com"));
}

/// Regression guard for the Python feature-guard tests that #523
/// previously broke: `Pdf::from_html_css(html, css_a, font).to_bytes()`
/// must produce different bytes from `from_html_css(html, css_b, font)`
/// when the only CSS difference is a property the renderer takes through
/// the pipeline. These tests *don't* prove the CSS property renders
/// correctly — they just prove the two outputs differ, which is what
/// `python/tests/test_api_coverage.py::TestHtmlCssCreation::*_changes_output`
/// asserts. The byte-differential comes from Standard-14 font dict
/// allocation order shifting between the two CSS variants; see
/// `src/writer/pdf_writer.rs::PdfWriter::finish` for why all twelve
/// Standard-14 Latin fonts are unconditionally registered.
#[test]
fn css_font_weight_bold_produces_different_bytes() {
    let normal = Pdf::from_html_css("<p>text</p>", "", DEJAVU.to_vec())
        .unwrap()
        .to_bytes()
        .unwrap();
    let bold = Pdf::from_html_css("<p>text</p>", "p { font-weight: bold; }", DEJAVU.to_vec())
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_ne!(normal, bold, "CSS font-weight had no effect");
}

#[test]
fn css_background_color_produces_different_bytes() {
    let no_bg = Pdf::from_html_css("<p>text</p>", "", DEJAVU.to_vec())
        .unwrap()
        .to_bytes()
        .unwrap();
    let with_bg =
        Pdf::from_html_css("<p>text</p>", "body { background-color: yellow; }", DEJAVU.to_vec())
            .unwrap()
            .to_bytes()
            .unwrap();
    assert_ne!(no_bg, with_bg, "CSS background-color had no effect");
}
