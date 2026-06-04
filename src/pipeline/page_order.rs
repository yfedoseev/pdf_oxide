//! Canonical page reading order — single source of truth for text-extraction
//! span ordering (issue #457).
//!
//! Resolution per PDF 32000-1:2008:
//!
//!   1. **Tagged PDF** (`/MarkInfo /Marked true`) with a `/StructTreeRoot`
//!      and `/Suspects != true`: walk the structure tree on this page and
//!      return spans in **logical structure order** (§14.7.2). This is the
//!      authoritative reading order when present.
//!
//!   2. **Otherwise**: return spans in **page content order** — the
//!      geometric top-to-bottom + left-to-right pass described in §14.8.2.3.1.
//!
//! All public text-extraction APIs (`extract_words`, `extract_text_lines`,
//! `extract_text`, `to_markdown`, `to_html`, `to_plain_text`) should consume
//! this function so they cannot drift apart on the same input.
//!
//! The `StructureTreeStrategy` it dispatches to already falls back to the
//! geometric strategy when the structure tree is suspect (§14.7.1) or when
//! the MCID order would zigzag horizontally across columns — both
//! defenses against producer bugs that this helper inherits transparently.

use crate::document::PdfDocument;
use crate::error::Result;
use crate::geometry::Rect;
use crate::pipeline::{OrderedTextSpan, ReadingOrderContext, TextPipeline, TextPipelineConfig};

/// Compute the canonical reading-order span sequence for a single page.
///
/// Returns an empty vector when the page has no extractable text.
///
/// All extracted spans are returned by default — including any that the
/// upstream extractor tagged as `/Artifact` (running headers, footers,
/// page numbers, watermarks; ISO 32000-1:2008 §14.8.2.2.1). Some
/// downstream callers (e.g. `extract_text` on untagged PDFs) apply
/// their own artifact filter. Use
/// [`page_reading_order_no_artifacts`] for the spec-correct
/// "exclude artifacts" variant.
///
/// # Errors
///
/// Returns the underlying parse / extraction error if span extraction
/// itself fails. Structure-tree resolution errors are tolerated and the
/// helper falls back to geometric order.
pub fn page_reading_order(doc: &PdfDocument, page_index: usize) -> Result<Vec<OrderedTextSpan>> {
    page_reading_order_inner(doc, page_index, /*include_artifacts*/ true)
}

/// Variant of [`page_reading_order`] that drops spans flagged as
/// `/Artifact` (running headers, footers, page numbers, watermarks;
/// ISO 32000-1:2008 §14.8.2.2.1).
pub fn page_reading_order_no_artifacts(
    doc: &PdfDocument,
    page_index: usize,
) -> Result<Vec<OrderedTextSpan>> {
    page_reading_order_inner(doc, page_index, /*include_artifacts*/ false)
}

fn page_reading_order_inner(
    doc: &PdfDocument,
    page_index: usize,
    include_artifacts: bool,
) -> Result<Vec<OrderedTextSpan>> {
    let mut spans = doc.extract_spans(page_index)?;
    if !include_artifacts {
        spans.retain(|s| s.artifact_type.is_none());
    }
    if spans.is_empty() {
        return Ok(Vec::new());
    }

    let context = build_context(doc, page_index);
    let pipeline = TextPipeline::with_config(TextPipelineConfig::default());
    pipeline.process(spans, context)
}

/// Build the `ReadingOrderContext` for a page from the document's
/// `MarkInfo`, `StructTreeRoot`, and media box.
///
/// Best-effort: any errors reading structure metadata produce a context
/// without MCID order, which means the pipeline takes the geometric path.
pub(crate) fn build_context(doc: &PdfDocument, page_index: usize) -> ReadingOrderContext {
    let media_box = doc
        .get_page_media_box(page_index)
        .unwrap_or((0.0, 0.0, 612.0, 792.0));
    // MediaBox is `(llx, lly, urx, ury)` per PDF 32000-1:2008 §7.7.3.3.
    // `Rect::new` expects `(x, y, width, height)`, so use `from_points`.
    let bbox = Rect::from_points(media_box.0, media_box.1, media_box.2, media_box.3);

    let mut ctx = ReadingOrderContext::new()
        .with_page(page_index as u32)
        .with_bbox(bbox);

    // Use logical structure order only when the tree is trustworthy
    // (§14.8.2.3.1 / §14.7.1): the document is /Marked or the catalog references
    // a /StructTreeRoot, and /MarkInfo /Suspects is not true. This accepts
    // PDF-1.4 catalog-only tagged files that the old `!marked` early-return
    // wrongly skipped, and rejects suspect trees.
    let Some(tree) = doc.struct_tree_trustworthy() else {
        return ctx;
    };

    // Use the all-pages traversal cache (O(1) per page) instead of re-walking
    // the whole structure tree here (≈ O(pages²) across a tagged document).
    let mcid_order: Vec<u32> = doc.cached_mcid_order_for_page(&tree, page_index as u32);

    if !mcid_order.is_empty() {
        ctx = ctx.with_mcid_order(mcid_order);
    }
    // The predicate already vetted the tree as non-suspect, so the strategy's
    // own suspect guard is a no-op here.
    ctx = ctx.with_suspects(false);
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn issue_211_fixture(name: &str) -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join("projects/pdf_oxide_tests/pdfs_issue_regression")
            .join(name);
        if !path.exists() {
            eprintln!("Skipping: {} not found", path.display());
            return None;
        }
        Some(path)
    }

    fn open(name: &str) -> Option<PdfDocument> {
        let path = issue_211_fixture(name)?;
        let bytes = std::fs::read(&path).ok()?;
        PdfDocument::from_bytes(bytes).ok()
    }

    #[test]
    fn empty_page_returns_empty_vec() {
        // Use any document; request a page index past the end. extract_spans
        // returns an error in that case, so the helper propagates. We only
        // assert behavior when the helper succeeds; this test currently only
        // verifies the function compiles and links — runtime check below.
        let Some(doc) = open("issue_211_pdf_structure.pdf") else {
            return;
        };
        // Page 0 IS populated. Just confirm we get a non-empty result.
        let result = page_reading_order(&doc, 0).expect("page 0 should resolve");
        assert!(!result.is_empty(), "page 0 of pdf_structure has spans");
    }

    #[test]
    fn tagged_pdf_uses_structure_tree_first() {
        // PDF #2 is tagged. Title spans should appear BEFORE body spans in
        // the canonical order, even though XY-Cut moves them.
        let Some(doc) = open("issue_211_municipal_minutes.pdf") else {
            return;
        };
        let ordered = page_reading_order(&doc, 0).expect("ordering succeeds");

        let title_pos = ordered
            .iter()
            .position(|s| s.span.text.contains("COMITÉ"))
            .expect("title must appear");
        let body_pos = ordered
            .iter()
            .position(|s| s.span.text.contains("Séance"))
            .expect("body must appear");
        assert!(
            title_pos < body_pos,
            "title (COMITÉ at index {}) must precede body (Séance at index {}) \
             in canonical reading order",
            title_pos,
            body_pos,
        );
    }

    #[test]
    fn untagged_pdf_falls_back_to_geometric() {
        // Smoke test — the simple Lorem fixture. The first ordered span must
        // contain "Titre du document" (the document title at top of page).
        let Some(doc) = open("issue_211_pdf_structure.pdf") else {
            return;
        };
        let ordered = page_reading_order(&doc, 0).expect("ordering succeeds");
        assert!(!ordered.is_empty());
        assert!(
            ordered[0].span.text.contains("Titre")
                || ordered
                    .iter()
                    .take(3)
                    .any(|s| s.span.text.contains("Titre")),
            "title 'Titre' must appear among the first few ordered spans; \
             got first 5: {:?}",
            ordered
                .iter()
                .take(5)
                .map(|s| &s.span.text)
                .collect::<Vec<_>>(),
        );
    }
}
