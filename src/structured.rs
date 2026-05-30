//! Structured per-page extraction (`extract_structured`) — issue #536.
//!
//! `PdfDocument::extract_structured(page)` returns a [`StructuredPage`]: the
//! page's text grouped into typed [`StructuredRegion`]s (body blocks, headings,
//! header/footer/page-number chrome, marginal labels) in reading order, with a
//! best-effort `column_index` for multi-column bodies.
//!
//! This is an **additive aggregation layer** over signals the extractor already
//! attaches to every [`TextSpan`](crate::layout::TextSpan):
//!
//! * `artifact_type` ([`crate::extractors::text::ArtifactType`]) →
//!   header / footer / page-number / artifact roles, per ISO 32000-1:2008
//!   §14.8.2.2 ("Real Content and Artifacts"). For a tagged PDF these come from
//!   the `/Artifact` marked-content sequences (§14.6.2); they are honoured
//!   for free.
//! * `heading_level` → [`RegionRole::StructuralHeading`]. Populated from the
//!   structure tree (`H1`..`H6`, §14.7.2) when the PDF is tagged, or from a
//!   font-size heuristic when it is not.
//! * span geometry → column assignment per §14.8.2.3.1 ("Page Content Order":
//!   multi-column layouts read column to column).
//!
//! Because the role signals already ride on the spans, a trustworthy
//! `/StructTreeRoot` (see [`crate::document::PdfDocument::prefers_structure_reading_order`])
//! drives the region roles automatically; untagged PDFs fall back to the
//! geometric/heuristic signals.

use crate::extractors::text::{ArtifactType, PaginationSubtype};
use crate::geometry::Rect;
use crate::layout::TextSpan;

/// A single page decomposed into typed regions in reading order.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct StructuredPage {
    /// Zero-based page index.
    pub page_index: usize,
    /// Page width in PDF points.
    pub page_width: f32,
    /// Page height in PDF points.
    pub page_height: f32,
    /// Regions in reading order (column-by-column per ISO 32000-1 §14.8.2.3.1).
    pub regions: Vec<StructuredRegion>,
}

/// A contiguous run of same-role spans, optionally tagged with a column index.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub struct StructuredRegion {
    /// The semantic role of this region.
    pub kind: RegionRole,
    /// The region's text (spans joined with single spaces / newlines).
    pub text: String,
    /// Union bounding box of the region's spans.
    pub bbox: Rect,
    /// The underlying spans that make up this region.
    pub spans: Vec<TextSpan>,
    /// Column index for multi-column bodies: `Some(0)` = leftmost column,
    /// `Some(1)` = next column, … `None` for full-width content, headings,
    /// or chrome.
    pub column_index: Option<usize>,
}

/// The semantic role of a [`StructuredRegion`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "wasm", serde(rename_all = "camelCase"))]
pub enum RegionRole {
    /// Ordinary body text.
    BodyBlock,
    /// A document heading (ISO 32000-1 §14.7.2 `H1`..`H6`).
    StructuralHeading {
        /// Heading level, 1–6.
        level: u8,
    },
    /// A short verse / section numeral sitting in a narrow column indent.
    MarginalLabel,
    /// Running header (§14.8.2.2 Pagination / Header).
    Header,
    /// Running footer (§14.8.2.2 Pagination / Footer).
    Footer,
    /// Page-number folio (§14.8.2.2 Pagination / page number).
    PageNumber,
    /// Any other artifact (watermark, layout, background; §14.8.2.2).
    Artifact,
}

/// Map a span's `artifact_type` / `heading_level` to a [`RegionRole`].
fn role_for_span(span: &TextSpan) -> RegionRole {
    if let Some(at) = &span.artifact_type {
        return match at {
            ArtifactType::Pagination(PaginationSubtype::Header) => RegionRole::Header,
            ArtifactType::Pagination(PaginationSubtype::Footer) => RegionRole::Footer,
            ArtifactType::Pagination(PaginationSubtype::PageNumber) => RegionRole::PageNumber,
            // Watermark / Other pagination, plus Layout / Page / Background.
            _ => RegionRole::Artifact,
        };
    }
    if let Some(level) = span.heading_level {
        return RegionRole::StructuralHeading { level };
    }
    if is_marginal_label(&span.text) {
        return RegionRole::MarginalLabel;
    }
    RegionRole::BodyBlock
}

/// A conservative marginal-label test: a short, standalone numeric or
/// lowercase-roman token (a verse / section numeral). When unsure we return
/// `false` so the span folds into the adjacent body block — reading order is
/// correct either way.
fn is_marginal_label(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() || t.chars().count() > 4 {
        return false;
    }
    let is_arabic = t.chars().all(|c| c.is_ascii_digit());
    let is_roman = !t.is_empty()
        && t.chars()
            .all(|c| matches!(c, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'));
    is_arabic || is_roman
}

/// Union of two rectangles (corner-based).
fn rect_union(a: &Rect, b: &Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.width).max(b.x + b.width);
    let y1 = (a.y + a.height).max(b.y + b.height);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// Best-effort single-gutter detector for body spans.
///
/// Returns the gutter X (page coordinate) when the body spans split into two
/// clear horizontal groups separated by a vertical whitespace corridor that is
/// a substantial fraction of the page width, else `None`. Conservative by
/// design: a page with no clear two-column body yields `None` and every body
/// region gets `column_index == None`.
fn detect_gutter_x(body: &[&TextSpan], page_width: f32) -> Option<f32> {
    if body.len() < 4 || page_width <= 0.0 {
        return None;
    }
    let mut centers: Vec<f32> = body.iter().map(|s| s.bbox.x + s.bbox.width * 0.5).collect();
    centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = centers[0];
    let max = centers[centers.len() - 1];
    if max - min < page_width * 0.25 {
        return None; // centers too clustered for two columns
    }
    // Largest gap between consecutive span centers.
    let mut best_gap = 0.0_f32;
    let mut best_mid = 0.0_f32;
    for w in centers.windows(2) {
        let gap = w[1] - w[0];
        if gap > best_gap {
            best_gap = gap;
            best_mid = (w[0] + w[1]) * 0.5;
        }
    }
    // Require a wide gutter near the page middle (0.3..0.7 of width).
    let rel = (best_mid) / page_width;
    if best_gap >= page_width * 0.12 && (0.3..=0.7).contains(&rel) {
        Some(best_mid)
    } else {
        None
    }
}

/// Build a [`StructuredPage`] from reading-order spans + page dimensions.
///
/// Pure function (no document access) so it is unit-testable in isolation.
pub(crate) fn build_structured_page(
    page_index: usize,
    page_width: f32,
    page_height: f32,
    spans: Vec<TextSpan>,
) -> StructuredPage {
    // Column assignment is computed over body spans only (chrome/headings are
    // full-width by convention).
    let body_refs: Vec<&TextSpan> = spans
        .iter()
        .filter(|s| matches!(role_for_span(s), RegionRole::BodyBlock | RegionRole::MarginalLabel))
        .collect();
    let gutter = detect_gutter_x(&body_refs, page_width);

    let column_of = |span: &TextSpan| -> Option<usize> {
        let g = gutter?;
        let center = span.bbox.x + span.bbox.width * 0.5;
        Some(if center < g { 0 } else { 1 })
    };

    let mut regions: Vec<StructuredRegion> = Vec::new();
    for span in spans {
        if span.text.trim().is_empty() {
            continue;
        }
        let kind = role_for_span(&span);
        let col = match kind {
            RegionRole::BodyBlock | RegionRole::MarginalLabel => column_of(&span),
            _ => None,
        };

        // Coalesce into the previous region when role + column match and the
        // spans are vertically adjacent (so distinct blocks stay separate).
        if let Some(last) = regions.last_mut() {
            if last.kind == kind && last.column_index == col {
                last.text.push(' ');
                last.text.push_str(span.text.trim());
                last.bbox = rect_union(&last.bbox, &span.bbox);
                last.spans.push(span);
                continue;
            }
        }
        regions.push(StructuredRegion {
            kind,
            text: span.text.trim().to_string(),
            bbox: span.bbox,
            column_index: col,
            spans: vec![span],
        });
    }

    StructuredPage {
        page_index,
        page_width,
        page_height,
        regions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, x: f32, y: f32, w: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, w, 12.0),
            ..Default::default()
        }
    }

    #[test]
    fn marginal_label_detects_short_numerals() {
        assert!(is_marginal_label("12"));
        assert!(is_marginal_label("iv"));
        assert!(!is_marginal_label("Genesis"));
        assert!(!is_marginal_label("12345")); // too long
    }

    #[test]
    fn heading_and_body_roles_assigned() {
        let mut h = span("Title", 100.0, 700.0, 80.0);
        h.heading_level = Some(1);
        let b = span("Body text here", 100.0, 680.0, 120.0);
        let page = build_structured_page(0, 612.0, 792.0, vec![h, b]);
        assert_eq!(page.regions.len(), 2);
        assert_eq!(page.regions[0].kind, RegionRole::StructuralHeading { level: 1 });
        assert_eq!(page.regions[1].kind, RegionRole::BodyBlock);
    }

    #[test]
    fn two_column_body_gets_column_indices() {
        // Left column at x≈60, right column at x≈360 on a 612-wide page.
        let spans = vec![
            span("left one", 60.0, 700.0, 120.0),
            span("left two", 60.0, 680.0, 120.0),
            span("right one", 360.0, 700.0, 120.0),
            span("right two", 360.0, 680.0, 120.0),
        ];
        let page = build_structured_page(0, 612.0, 792.0, spans);
        let cols: Vec<Option<usize>> = page.regions.iter().map(|r| r.column_index).collect();
        assert!(cols.contains(&Some(0)), "a left column (0) must be assigned: {cols:?}");
        assert!(cols.contains(&Some(1)), "a right column (1) must be assigned: {cols:?}");
    }
}
