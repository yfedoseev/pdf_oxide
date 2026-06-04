//! `to_markdown` / `to_html` must honour `ConversionOptions::exclude_regions`
//! and `include_region`, identically to the plain-text path. These region
//! filters were previously applied only in `extract_text` / `to_plain_text`,
//! so markdown and HTML emitted the full page regardless of the requested
//! exclusions. These tests lock in the converter-agnostic behaviour.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::RectFilterMode;
use pdf_oxide::PdfDocument;

const HELLO: &str = "tests/fixtures/hello_structure.pdf";
const PAPER: &str = "tests/fixtures/1008.3918v2.pdf";

fn open(path: &str) -> PdfDocument {
    PdfDocument::open(path).expect("open fixture")
}

/// Excluding a region covering the whole media box must leave no text in the
/// markdown.
#[test]
fn whole_page_exclude_empties_markdown() {
    let doc = open(HELLO);
    let (x0, y0, x1, y1) = doc.get_page_media_box(0).expect("media box");
    let opts = ConversionOptions {
        exclude_regions: vec![Rect::new(x0, y0, x1 - x0, y1 - y0)],
        exclude_regions_mode: RectFilterMode::FullyContained,
        ..Default::default()
    };
    let md = doc.to_markdown(0, &opts).expect("markdown");
    assert!(
        !md.contains("Hello") && !md.contains("World"),
        "whole-page exclusion must drop all text, got: {md:?}"
    );
}

/// Same guarantee for the HTML surface.
#[test]
fn whole_page_exclude_empties_html() {
    let doc = open(HELLO);
    let (x0, y0, x1, y1) = doc.get_page_media_box(0).expect("media box");
    let opts = ConversionOptions {
        exclude_regions: vec![Rect::new(x0, y0, x1 - x0, y1 - y0)],
        exclude_regions_mode: RectFilterMode::FullyContained,
        ..Default::default()
    };
    let html = doc.to_html(0, &opts).expect("html");
    assert!(
        !html.contains("Hello") && !html.contains("World"),
        "whole-page exclusion must drop all text from HTML, got: {html:?}"
    );
}

/// Sanity: with no region filters the content is still emitted (the filter is a
/// no-op when unset, so we are not silently dropping everything).
#[test]
fn no_filter_keeps_content() {
    let doc = open(HELLO);
    let md = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("markdown");
    assert!(md.contains("Hello World"), "default options must keep content, got: {md:?}");
}

/// `include_region` keeps only spans inside the region — present when the region
/// covers the text, absent when it covers an empty band.
#[test]
fn include_region_scopes_markdown() {
    let doc = open(HELLO);
    // "Hello World" sits at bbox (100, 700, 124, 24) in y-up page space.
    let covering = ConversionOptions {
        include_region: Some((Rect::new(90.0, 690.0, 160.0, 50.0), RectFilterMode::FullyContained)),
        ..Default::default()
    };
    assert!(
        doc.to_markdown(0, &covering)
            .expect("md")
            .contains("Hello World"),
        "include_region covering the text must keep it"
    );

    let elsewhere = ConversionOptions {
        include_region: Some((Rect::new(0.0, 0.0, 612.0, 100.0), RectFilterMode::FullyContained)),
        ..Default::default()
    };
    let md = doc.to_markdown(0, &elsewhere).expect("md");
    assert!(
        !md.contains("Hello") && !md.contains("World"),
        "include_region over an empty band must drop the text, got: {md:?}"
    );
}

/// A sub-region exclusion (not the whole page) drops the content it covers
/// while the rest of the page survives. Excludes the top band of the page and
/// checks that a distinctive word from the top of the rendered output (the
/// title) disappears, while the page is not emptied. The target word is taken
/// from the rendered markdown so the assertion is independent of how spans are
/// segmented, and the band is expressed relative to the media box so the test
/// carries no hard-coded coordinates.
#[test]
fn sub_region_exclude_drops_top_band() {
    let doc = open(PAPER);
    let (x0, y0, x1, y1) = doc.get_page_media_box(0).expect("media box");
    let height = y1 - y0;

    let full = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("md");
    // A long word that occurs exactly once → unambiguous to match. In reading
    // order the first such word is near the top of the page (the title).
    let target = full
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
        })
        .find(|w| w.len() >= 8 && full.matches(w.as_str()).count() == 1)
        .expect("a distinctive top-of-page word");

    // Exclude the top 45% of the page (y-up: the high-y band).
    let top_band = Rect::new(x0, y0 + height * 0.55, x1 - x0, height * 0.45);
    let opts = ConversionOptions {
        exclude_regions: vec![top_band],
        exclude_regions_mode: RectFilterMode::Intersects,
        ..Default::default()
    };
    let excluded = doc.to_markdown(0, &opts).expect("md");
    assert!(
        !excluded.contains(&target),
        "top-band exclusion must drop the title word {target:?}"
    );
    assert!(
        !excluded.trim().is_empty() && excluded.len() < full.len(),
        "a sub-region exclusion must keep the rest of the page, not empty it"
    );
}
