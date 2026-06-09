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
/// columns separated by a vertical whitespace corridor that **no span crosses**,
/// else `None`. Conservative by design: a page with no clear two-column body
/// yields `None` and every body region gets `column_index == None`.
///
/// Detection is **edge-based** (a valley in the horizontal projection of span
/// extents), not center-of-mass based. A span-center histogram collapses on the
/// layouts that need column routing most — short, ragged lines (Bible verses,
/// reference editions) and word-level spans pack centres densely across each
/// column, leaving no single wide centre-gap even though the columns are
/// visually obvious. The empty corridor between the left column's right edge and
/// the right column's left edge survives all of that: inter-word gaps inside a
/// line are crossed by other lines at different y, so only a true column gutter
/// forms a page-spanning empty band. Because a real gutter can be narrow
/// (≈8–20pt) the width gate is an absolute minimum, not a fraction of the page —
/// the "no span crosses it" + "substantial body on both sides" + "near the page
/// middle" conditions are what guard against false positives.
fn detect_gutter_x(body: &[&TextSpan], page_width: f32) -> Option<f32> {
    /// A column gutter is a true empty channel; ordinary inter-word/-glyph gaps
    /// are both narrower and crossed by other lines, so they never survive.
    const MIN_GUTTER_PT: f32 = 8.0;
    /// Each side must hold at least a small line or two — a single off-margin
    /// token is not a column. The empty-corridor + near-middle gates do the real
    /// false-positive rejection.
    const MIN_SIDE_SPANS: usize = 2;

    if body.len() < 4 || page_width <= 0.0 {
        return None;
    }
    // Horizontal extents (left, right) of every finite, non-empty body span.
    let mut boxes: Vec<(f32, f32)> = body
        .iter()
        .filter(|s| {
            s.bbox.width > 0.0
                && s.bbox.x.is_finite()
                && s.bbox.width.is_finite()
                && !s.text.trim().is_empty()
        })
        .map(|s| (s.bbox.x, s.bbox.x + s.bbox.width))
        .collect();
    if boxes.len() < 4 {
        return None;
    }
    boxes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let content_min = boxes.iter().map(|b| b.0).fold(f32::INFINITY, f32::min);
    let content_max = boxes.iter().map(|b| b.1).fold(f32::NEG_INFINITY, f32::max);
    if content_max - content_min < page_width * 0.25 {
        return None; // body too narrow to hold two columns
    }

    // Sweep-merge the extents left-to-right; the widest forward jump between the
    // running right edge and the next span's left edge is the widest empty
    // corridor no span crosses. Sorted by left edge, every span entirely left of
    // a clean corridor precedes every span entirely right of it, so the span
    // index at the jump is the left-column count.
    let mut cover_right = boxes[0].1;
    let mut best_gap = 0.0_f32;
    let mut best_mid = 0.0_f32;
    let mut left_count = 0usize;
    for i in 1..boxes.len() {
        let gap = boxes[i].0 - cover_right;
        if gap > best_gap {
            best_gap = gap;
            best_mid = (cover_right + boxes[i].0) * 0.5;
            left_count = i;
        }
        cover_right = cover_right.max(boxes[i].1);
    }

    let rel = best_mid / page_width;
    let right_count = boxes.len() - left_count;
    if best_gap >= MIN_GUTTER_PT
        && (0.3..=0.7).contains(&rel)
        && left_count >= MIN_SIDE_SPANS
        && right_count >= MIN_SIDE_SPANS
    {
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

    /// A reference-edition layout (Bible verses): two narrow columns with a
    /// narrow gutter, short ragged verse lines, word-level spans, and marginal
    /// verse numerals at each column's left edge. The old center-of-mass gap
    /// detector returned `None` here (the gutter is far under 12 % of the page
    /// width and word centres pack each column densely); the edge-based corridor
    /// detector must still split the columns.
    #[test]
    fn narrow_gutter_short_line_columns_get_indices() {
        // Page 432pt wide. Left column x∈[36,206], right column x∈[226,396],
        // gutter ≈ [206,226] (20pt, ~4.6 % of width). Word-level spans.
        let pw = 432.0;
        let mut spans = Vec::new();
        let mut y = 700.0;
        for row in 0..6 {
            // Left column: a marginal verse numeral then two short words.
            spans.push(span(&format!("{}", row + 1), 36.0, y, 8.0));
            spans.push(span("Au", 52.0, y, 26.0));
            spans.push(span("commencement", 84.0, y, 110.0));
            // Right column: marginal numeral then two short words.
            spans.push(span(&format!("{}", row + 14), 226.0, y, 12.0));
            spans.push(span("Et", 244.0, y, 22.0));
            spans.push(span("Dieu", 272.0, y, 40.0));
            y -= 14.0;
        }
        let page = build_structured_page(0, pw, 792.0, spans);
        let cols: Vec<Option<usize>> = page.regions.iter().map(|r| r.column_index).collect();
        assert!(cols.contains(&Some(0)), "left column (0) not assigned: {cols:?}");
        assert!(cols.contains(&Some(1)), "right column (1) not assigned: {cols:?}");
    }

    /// A single-column page of ordinary prose (lines spanning most of the body
    /// width at varying right edges) must NOT be split into columns — there is
    /// no empty corridor for any line to leave clear.
    #[test]
    fn single_column_prose_has_no_gutter() {
        let pw = 612.0;
        let mut spans = Vec::new();
        let mut y = 700.0;
        let widths = [430.0, 460.0, 410.0, 470.0, 440.0, 455.0, 425.0, 465.0];
        for w in widths {
            spans.push(span("a single column prose line of body text", 80.0, y, w));
            y -= 14.0;
        }
        let page = build_structured_page(0, pw, 792.0, spans);
        let cols: Vec<Option<usize>> = page.regions.iter().map(|r| r.column_index).collect();
        assert!(
            cols.iter().all(|c| c.is_none()),
            "single-column prose wrongly split into columns: {cols:?}"
        );
    }
}
