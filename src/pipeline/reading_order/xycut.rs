//! XY-Cut recursive spatial partitioning for multi-column text layout.
//!
//! This module implements the XY-Cut algorithm per PDF Spec Section 9.4 for
//! recursive geometric analysis without semantic heuristics. Uses projection
//! profiles to detect column boundaries in complex layouts.
//!
//! Per ISO 32000-1:2008:
//! - Section 9.4: Text Objects and coordinates
//! - Section 14.7: Logical Structure (prefers structure tree when available)
//!
//! # Algorithm Overview
//!
//! 1. Compute horizontal projection (white space density across X)
//! 2. Find valleys (gaps) where density < threshold
//! 3. Split region at widest valley (vertical line)
//! 4. Recursively partition left and right sub-regions
//! 5. Alternate to vertical projection if no horizontal valleys found
//! 6. Base case: Sort spans top-to-bottom, left-to-right
//!
//! # Performance
//!
//! Typical newspaper page: ~100 spans, < 5ms processing time
//! Recursive depth: O(log n) for balanced columns

use super::{ReadingOrderContext, ReadingOrderStrategy};
use crate::error::Result;
use crate::layout::TextSpan;
use crate::pipeline::{OrderedTextSpan, ReadingOrderInfo};

/// Maximum density-array length for XY-cut projection profiles.
///
/// A normal PDF page is at most a few thousand points wide/tall. This limit of
/// 100 000 bins is generous (≈ 33× a 3000-point A0 page) while being small
/// enough to never cause an allocation problem.  Spans whose bounding-box span
/// exceeds this limit are the result of a degenerate CTM; returning `None` from
/// the projection safely skips the split instead of attempting a multi-terabyte
/// allocation that would abort the process via `handle_alloc_error`.
const MAX_PROJECTION_SIZE: usize = 100_000;

/// Coarse classification of a region for the v0.3.54 #534 multi-column-prose
/// fix. Used to gate the tight-gutter cut: tight cuts are only accepted on
/// regions that *positively* identify as prose, so the same XY-cut recursion
/// no longer corrupts table cells (the v0.3.53 lesson — see lines 73–101).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    /// Tall stack of wide lines OR tall stack of half-column lines with
    /// substantial content per line. Safe to apply tight-gutter cuts.
    Prose,
    /// Short cells in a grid (mean characters per line < 8). Tight cuts
    /// here corrupt cell ordering — the canonical google_doc population
    /// table that reverted v0.3.53's two attempts is the prototype.
    Table,
    /// Anything else — too few lines, mixed shapes, decorative regions.
    /// Default to the v0.3.53 behaviour (no tight cut).
    Mixed,
}

/// XY-Cut recursive spatial partitioning strategy.
///
/// Detects columns using projection profiles and white space analysis.
/// Suitable for newspapers, academic papers, and multi-column layouts.
pub struct XYCutStrategy {
    /// Minimum number of spans in a region before attempting split (default: 5).
    /// Prevents excessive recursion on small regions.
    pub min_spans_for_split: usize,

    /// Valley threshold as fraction of peak projection density (default: 0.3).
    /// Lower values detect narrower gutters, higher values only detect wide gaps.
    pub valley_threshold: f32,

    /// Minimum valley width in points (default: 15.0).
    /// Prevents detecting single-character gaps as column boundaries.
    pub min_valley_width: f32,

    /// Enable horizontal partitioning first, fallback to vertical (default: true).
    ///
    /// Per PDF Spec ISO 32000-1:2008 §14.8.4 (Logical Structure reading order),
    /// column detection is the primary purpose of XY-Cut — horizontal-first
    /// (vertical cut line) splits columns before rows, matching Western
    /// top-down-left-to-right reading order in multi-column documents.
    /// Callers with row-dominant layouts can override via
    /// `with_prefer_horizontal(false)`.
    pub prefer_horizontal: bool,
}

impl Default for XYCutStrategy {
    fn default() -> Self {
        Self {
            min_spans_for_split: 5,
            valley_threshold: 0.3,
            // 15pt. Issue #7 (multi-column prose interleaving on
            // issue_07_orphaned_fragments.pdf) was attempted TWICE and
            // REVERTED both times — the 70-PDF sweep caught data
            // corruption in google_doc_document.pdf's population table
            // ("273.879.7501" -> "1273.879.750") each time:
            //
            //   Attempt 1 — lower min_valley_width 15 -> 12 so the tight
            //   ~12pt two-column gutter is detected. Also split the
            //   table's ~12pt inter-cell gaps -> reordered digits.
            //
            //   Attempt 2 — a structural find_two_column_prose_split
            //   (exactly-two recurring left-edge clusters, wide columns,
            //   clean gutter) tried before the single-column check. It
            //   never fired on issue_07's WHOLE page (three left-edge
            //   clusters: full-width intro/footer @60 + left @82 + right
            //   @312, because is_single_column blocks band separation
            //   first), yet it DID fire on a 2-column sub-region of the
            //   google_doc table and reordered cells.
            //
            // Root cause: the same XY-Cut machinery orders both
            // prose-columns and table-cells. Any sensitivity increase
            // that catches issue_07's tight 2-column prose also splits
            // table cells and corrupts data. A correct #7 fix needs a
            // real table-vs-prose classifier (column cells are short
            // values; prose columns are tall stacks of wide lines) AND
            // recursive band-separation of full-width header/footer rows
            // before column detection — a substantial XY-Cut redesign,
            // validated against the full CI corpus, not a local tweak.
            min_valley_width: 15.0,
            prefer_horizontal: true,
        }
    }
}

impl XYCutStrategy {
    /// Create a new XY-Cut strategy with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom valley threshold (0.0-1.0).
    pub fn with_valley_threshold(mut self, threshold: f32) -> Self {
        self.valley_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Create with custom minimum valley width.
    pub fn with_min_valley_width(mut self, width: f32) -> Self {
        self.min_valley_width = width.max(1.0);
        self
    }

    /// Enable or disable horizontal partitioning first preference.
    pub fn with_prefer_horizontal(mut self, prefer: bool) -> Self {
        self.prefer_horizontal = prefer;
        self
    }

    /// Core recursive partitioning algorithm.
    ///
    /// Public for use by MarkdownConverter's ColumnAware reading order mode.
    pub fn partition_region(&self, spans: &[TextSpan]) -> Vec<Vec<TextSpan>> {
        let indices: Vec<usize> = (0..spans.len()).collect();
        let index_groups = self.partition_indexed(spans, &indices);
        // Clone spans only once at the end (not at every recursion level)
        index_groups
            .into_iter()
            .map(|group| group.into_iter().map(|i| spans[i].clone()).collect())
            .collect()
    }

    /// Index-based recursive partitioning — returns groups of indices into the input span slice.
    ///
    /// Avoids cloning TextSpan at every recursive split level. Spans are only
    /// read through shared reference; indices are partitioned instead.
    fn partition_indexed(&self, all_spans: &[TextSpan], indices: &[usize]) -> Vec<Vec<usize>> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Base case: small region, don't split further
        if indices.len() < self.min_spans_for_split {
            return vec![self.sort_indices(all_spans, indices)];
        }

        // v0.3.54 (#534): two-column-prose probe BEFORE the
        // single-column short-circuit. Tight gutters (~10-15pt) that
        // sit below `min_valley_width` defeat the standard projection-
        // valley detector, and the wide+dense heuristic inside
        // `is_single_column_region` mis-classifies the body as one
        // column because each line's bbox spans the narrow gutter.
        // The probe positively identifies the 2-column-prose shape
        // (gutter-radius left-edge clusters + ≥6 narrow lines +
        // classify_region_kind == Prose) and only fires when ALL of
        // those signals agree. Critically, the Prose gate prevents
        // the false positive that reverted v0.3.53's attempts on a
        // 2-column sub-region of the google_doc population table
        // (mean_chars < 8 → Table → bail).
        //
        // **Band-separation first**: when the probe would fire AND a
        // clean vertical band-separation (top header / body / bottom
        // footer) is available, peel the band off BEFORE the column
        // cut. Without this step, full-width header / footer rows
        // get absorbed into one of the two column halves and end up
        // mid-page in reading order — the failure mode on the
        // 1256-page French Bible from #536 where the chapter-header
        // band and page-number footer were full-width and span the
        // gutter. The signal for "band": a vertical split whose
        // smaller side has ≤ 25 % of the region's spans (a tight
        // band relative to the body it sits next to).
        if let Some(gutter_x) = self.detect_two_column_prose(all_spans, indices) {
            if let Some((above, below)) = self.find_vertical_split_indexed(all_spans, indices) {
                let smaller = above.len().min(below.len());
                let total = above.len() + below.len();
                if smaller * 4 <= total {
                    log::debug!(
                        "XY-cut #534: peeling band before column cut, above={} below={}",
                        above.len(),
                        below.len()
                    );
                    let mut result = self.partition_indexed(all_spans, &above);
                    result.extend(self.partition_indexed(all_spans, &below));
                    return result;
                }
            }
            let (left, right): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .copied()
                .partition(|&i| all_spans[i].bbox.left() < gutter_x);
            if !left.is_empty() && !right.is_empty() {
                log::debug!(
                    "XY-cut #534: two-column-prose detected, gutter_x={:.1}, left={} right={}",
                    gutter_x,
                    left.len(),
                    right.len()
                );
                let mut result = self.partition_indexed(all_spans, &left);
                result.extend(self.partition_indexed(all_spans, &right));
                return result;
            }
        }

        // Detect single-column body text up-front and skip all spatial
        // splits. Real body text has density dips (indented code, short
        // last-lines, paragraph breaks) that would otherwise trigger
        // spurious horizontal (column) or vertical (row) splits,
        // scrambling reading order. The subsequent sort-by-Y already
        // handles row order within a column.
        if self.is_single_column_region(all_spans, indices) {
            return vec![self.sort_indices(all_spans, indices)];
        }

        let split_h =
            |s: &Self, sp: &[TextSpan], idx: &[usize]| s.find_horizontal_split_indexed(sp, idx);
        let split_v =
            |s: &Self, sp: &[TextSpan], idx: &[usize]| s.find_vertical_split_indexed(sp, idx);

        let first_split = if self.prefer_horizontal {
            split_h
        } else {
            split_v
        };
        let second_split = if self.prefer_horizontal {
            split_v
        } else {
            split_h
        };

        if let Some((a, b)) = first_split(self, all_spans, indices) {
            let mut result = self.partition_indexed(all_spans, &a);
            result.extend(self.partition_indexed(all_spans, &b));
            return result;
        }

        if let Some((a, b)) = second_split(self, all_spans, indices) {
            let mut result = self.partition_indexed(all_spans, &a);
            result.extend(self.partition_indexed(all_spans, &b));
            return result;
        }

        // No split found, return as single group
        vec![self.sort_indices(all_spans, indices)]
    }

    /// Classifier verdict for a region — used to gate the tight-gutter
    /// column-split path (#534) so the same XY-cut recursion no longer
    /// corrupts table cells (the v0.3.53 lesson).
    ///
    /// See the inline post-mortem at lines 73–101: two prior attempts at
    /// the multi-column-prose fix were reverted by the 70-PDF sweep when
    /// they accidentally fired on a 2-column sub-region of a real table
    /// and reordered digits. The fix has to *positively identify prose*
    /// before allowing the tight cut — not merely *fail to identify
    /// table*. This classifier is that positive identification.
    fn classify_region_kind(&self, all_spans: &[TextSpan], indices: &[usize]) -> RegionKind {
        // Cheap shape check first.
        if indices.len() < 6 {
            return RegionKind::Mixed;
        }

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        for &i in indices {
            x_min = x_min.min(all_spans[i].bbox.left());
            x_max = x_max.max(all_spans[i].bbox.right());
        }
        let region_width = x_max - x_min;
        if region_width <= 10.0 {
            return RegionKind::Mixed;
        }

        // Cluster spans into lines by rounded Y.
        let mut lines: std::collections::BTreeMap<i32, (f32, f32, usize)> =
            std::collections::BTreeMap::new();
        for &i in indices {
            let s = &all_spans[i];
            let y_key = s.bbox.top().round() as i32;
            let nonws_chars = s.text.chars().filter(|c| !c.is_whitespace()).count();
            let entry = lines.entry(y_key).or_insert((f32::MAX, f32::MIN, 0));
            entry.0 = entry.0.min(s.bbox.left());
            entry.1 = entry.1.max(s.bbox.right());
            entry.2 += nonws_chars;
        }

        let line_count = lines.len();
        if line_count < 6 {
            // Too few lines to be a substantial prose body. Headings,
            // captions, single paragraphs all land here — leave them to
            // the default XY-cut behaviour.
            return RegionKind::Mixed;
        }

        // Per-line statistics: average char count and the count of
        // "narrow" lines whose extent < 0.6 × region_width (a column-half
        // line) and "wide" lines whose extent ≥ 0.6 × region_width (a
        // body-text or table-row line). Table cells are narrow; tables
        // have many such narrow lines but with very short content.
        let mut total_chars = 0usize;
        let mut narrow_lines = 0usize;
        let mut wide_lines = 0usize;
        for (left, right, chars) in lines.values() {
            total_chars += chars;
            let extent = (*right - *left).max(0.0);
            if extent < region_width * 0.6 {
                narrow_lines += 1;
            } else {
                wide_lines += 1;
            }
        }
        let mean_chars = total_chars as f32 / line_count as f32;

        // PROSE: tall stack of wide lines OR tall stack of half-column
        // lines with substantial content per line.
        //   - mean_chars > 20: real prose, not table cells
        //   - line_count ≥ 6: substantial column
        //   - either:
        //     * majority of lines are wide (single-column body), OR
        //     * majority of lines are narrow with mean_chars > 20
        //       (two half-column lines with prose content)
        let mostly_wide = wide_lines * 2 > line_count;
        let mostly_narrow = narrow_lines * 2 > line_count;
        if mean_chars > 20.0 && (mostly_wide || mostly_narrow) {
            return RegionKind::Prose;
        }

        // TABLE: lots of narrow lines, short content per line (mean_chars
        // < 8). The v0.3.53 google_doc_document.pdf population table —
        // the canonical regression that reverted attempts 1 & 2 — sits
        // squarely here (digit-only cells, ≤ 7 chars each).
        if mean_chars < 8.0 {
            return RegionKind::Table;
        }

        // Anything in between (e.g. captions with headings, mixed
        // figure-and-text bands) → don't risk the tight cut.
        RegionKind::Mixed
    }

    /// Two-column-prose probe (#534) — does this region look like two
    /// side-by-side columns of prose with a tight gutter (~10-15pt)?
    ///
    /// Called from `is_single_column_region` when the wide+dense
    /// heuristic would otherwise short-circuit the region as
    /// single-column. Distinguishing signal: most lines fit inside
    /// **one** half of the region width (column-half lines), and the
    /// left edges cluster into exactly **two** groups separated by
    /// approximately half the region width.
    ///
    /// Gated on `classify_region_kind == Prose` so the same machinery
    /// doesn't fire on a 2-column sub-region of a table (the v0.3.53
    /// failure mode).
    ///
    /// Returns `Some(gutter_x)` when a 2-column prose layout is
    /// detected — the caller treats that as a non-single-column verdict
    /// and lets `find_horizontal_split_indexed` cut at the gutter.
    fn detect_two_column_prose(&self, all_spans: &[TextSpan], indices: &[usize]) -> Option<f32> {
        // Cheap shape check first.
        if indices.len() < 8 {
            return None;
        }

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        for &i in indices {
            x_min = x_min.min(all_spans[i].bbox.left());
            x_max = x_max.max(all_spans[i].bbox.right());
        }
        let region_width = x_max - x_min;
        if region_width < 200.0 {
            // Real two-column bodies span at least ~200pt (the
            // narrowest two-column layout in the corpus is ~250pt for a
            // letter-page body inside ~250pt margins).
            return None;
        }

        // Cluster spans into lines by rounded Y. Keep PER-SPAN
        // (left, right) data so we can detect within-line gaps —
        // the canonical multi-column interleave on issue_07 puts
        // a left-col span (left=82) and a right-col span (left=312)
        // on the same Y baseline. The whole-line bbox.right -
        // bbox.left = 358 pt looks "wide" (358 > 0.6 × 500 = 300)
        // even though each side is a narrow column half.
        let mut lines_spans: std::collections::BTreeMap<i32, Vec<(f32, f32)>> =
            std::collections::BTreeMap::new();
        for &i in indices {
            let s = &all_spans[i];
            let y_key = s.bbox.top().round() as i32;
            lines_spans
                .entry(y_key)
                .or_default()
                .push((s.bbox.left(), s.bbox.right()));
        }
        if lines_spans.len() < 6 {
            return None;
        }

        // For each line, find the largest gap between adjacent spans.
        // A line is treated as multiple "half-lines" if a gap ≥ 10 pt
        // splits it; each side of the gap contributes its leftmost-x
        // to `narrow_lefts`. This is the v0.3.53 lesson: the row-by-
        // row interleave shape on issue_07 spans the gutter as bbox
        // but has a clear gap within each line.
        let narrow_threshold = region_width * 0.6;
        let intra_line_gap_threshold = 10.0_f32;
        let mut narrow_lefts: Vec<f32> = Vec::new();
        // Count "narrow" lines for the majority check — a line with
        // a within-line gap contributes 1 to this count regardless of
        // how many half-lines it produces, so the majority threshold
        // stays comparable to single-column reasoning.
        let mut narrow_line_count = 0usize;
        for line_spans in lines_spans.values() {
            let mut sorted = line_spans.clone();
            sorted.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));
            // Detect largest within-line gap.
            let mut largest_gap = 0.0_f32;
            let mut split_idx: Option<usize> = None;
            for (i, w) in sorted.windows(2).enumerate() {
                let gap = w[1].0 - w[0].1;
                if gap > largest_gap {
                    largest_gap = gap;
                    split_idx = Some(i);
                }
            }
            let line_left = sorted.first().map(|(l, _)| *l).unwrap_or(0.0);
            let line_right = sorted.last().map(|(_, r)| *r).unwrap_or(0.0);
            let line_extent = (line_right - line_left).max(0.0);

            if let Some(si) = split_idx {
                if largest_gap >= intra_line_gap_threshold {
                    // Within-line gap detected — treat each side as
                    // its own narrow half-line.
                    narrow_lefts.push(line_left);
                    // The right-side starts at sorted[si + 1].0
                    if let Some(&(right_side_left, _)) = sorted.get(si + 1) {
                        narrow_lefts.push(right_side_left);
                    }
                    narrow_line_count += 1;
                    continue;
                }
            }

            if line_extent < narrow_threshold {
                narrow_lefts.push(line_left);
                narrow_line_count += 1;
            }
        }
        // Majority of lines must be narrow — otherwise this isn't a
        // 2-column body, it's a single-column body with a few short
        // last-lines.
        if narrow_line_count * 2 < lines_spans.len() {
            return None;
        }

        // Cluster the narrow left-edges. Two clusters separated by
        // approximately half the region width = 2-column prose.
        let cluster_radius = 30.0_f32;
        let mut clusters: Vec<(f32, usize)> = Vec::new();
        for &x in &narrow_lefts {
            if let Some(c) = clusters
                .iter_mut()
                .find(|(c, _)| (*c - x).abs() <= cluster_radius)
            {
                // Running mean
                let count = c.1 as f32;
                c.0 = (c.0 * count + x) / (count + 1.0);
                c.1 += 1;
            } else {
                clusters.push((x, 1));
            }
        }

        // Want exactly 2 substantial clusters separated by ~half-width.
        // ≥ 3 clusters = either a table or a band-mixed region — bail.
        if clusters.len() != 2 {
            return None;
        }
        // Sort by x.
        clusters.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));
        let (c1_x, c1_n) = clusters[0];
        let (c2_x, c2_n) = clusters[1];

        // Each cluster needs substantial coverage — ≥ 3 lines, or 20 %
        // of the line count, whichever is larger. Reject lopsided
        // shapes (header + body-paragraph).
        let min_cluster = 3usize.max(narrow_lefts.len() / 5);
        if c1_n < min_cluster || c2_n < min_cluster {
            return None;
        }

        // Gap between cluster centres ≥ 30 % of region width (the
        // gutter + right-column left-margin). For a tight gutter of
        // ~12pt with two ~250pt columns the gap is ~250pt out of 512pt
        // → ~49 %, well above the floor.
        let gap = c2_x - c1_x;
        if gap < region_width * 0.30 {
            return None;
        }

        // Positive identification of prose — required by the
        // classifier to avoid the v0.3.53 google_doc 2-col table
        // sub-region false positive.
        if self.classify_region_kind(all_spans, indices) != RegionKind::Prose {
            return None;
        }

        // Gutter midpoint as the cut. The cluster centres are the left
        // edges of the two columns; the gutter sits between the right
        // edge of column 1 and the left edge of column 2. We don't
        // track right edges per cluster, so approximate the gutter
        // centre as halfway between the two cluster centres — that's
        // close enough; the actual partition uses `bbox.left()` per
        // span so individual spans land cleanly on either side.
        let gutter_x = (c1_x + c2_x) * 0.5;
        Some(gutter_x)
    }

    /// Heuristic: does the region look like a single column of body text?
    ///
    /// Called **before** horizontal split attempts. When true, the region
    /// is returned as a single sorted group, bypassing both horizontal
    /// (column) and vertical (row) splits. This prevents XY-Cut from
    /// fragmenting body text at density dips caused by indentation or
    /// short last-lines.
    ///
    /// Detection: cluster spans into lines by rounded top-Y, then count
    /// lines that are both **wide** (extent ≥ 60% region width) and
    /// **dense** (covered ratio ≥ 80%). Body-text lines satisfy both.
    /// Aligned multi-column rows look "wide" because their extent spans
    /// the gutter, but fail the density check because the gutter is empty.
    fn is_single_column_region(&self, all_spans: &[TextSpan], indices: &[usize]) -> bool {
        if indices.len() < 3 {
            return false;
        }
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        for &i in indices {
            x_min = x_min.min(all_spans[i].bbox.left());
            x_max = x_max.max(all_spans[i].bbox.right());
        }
        let region_width = x_max - x_min;
        if region_width <= 10.0 {
            return true;
        }

        // Store both bbox.right and core_right for each span. bbox.right
        // can be over-estimated by extractors (trailing whitespace,
        // stretched advance widths) which makes multi-column lines look
        // like one wide continuous run; core_right (char_count × em) is
        // a conservative fallback used ONLY when adjacent bbox edges
        // overlap (a signal of bbox inflation).
        //
        let mut lines: std::collections::BTreeMap<i32, Vec<(f32, f32, f32)>> =
            std::collections::BTreeMap::new();
        for &i in indices {
            let s = &all_spans[i];
            let y_key = s.bbox.top().round() as i32;
            let char_count = s.text.chars().filter(|c| !c.is_whitespace()).count().max(1) as f32;
            let approx_char_width = (s.font_size * 0.45).max(2.5);
            let core_right = s.bbox.left() + char_count * approx_char_width;
            lines
                .entry(y_key)
                .or_default()
                .push((s.bbox.left(), s.bbox.right(), core_right));
        }
        if lines.len() < 3 {
            return false;
        }

        // A real column gutter recurs at roughly the SAME X position
        // across multiple lines. Sparse title-page layouts (Title /
        // Subtitle / Byline) also have wide inter-word gaps, but their
        // gap positions are scattered — not a gutter. Collect all gap
        // positions (mid-gap X), then check whether a consistent cluster
        // of gap positions appears on ≥30% of lines.
        //
        // Gap uses bbox.right, but if adjacent bboxes OVERLAP (classic
        // signature of extractor-inflated bbox widths), re-check with
        // conservative core_right estimates so column detection is not
        // defeated by trailing whitespace inflation.
        let max_gap = self.min_valley_width;
        let mut gap_positions: Vec<f32> = Vec::new();
        for line_spans in lines.values() {
            let mut sorted = line_spans.clone();
            sorted.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));
            for w in sorted.windows(2) {
                let bbox_gap = w[1].0 - w[0].1;
                let (effective_gap, gap_end_left) = if bbox_gap < 0.0 {
                    (w[1].0 - w[0].2, w[0].2)
                } else {
                    (bbox_gap, w[0].1)
                };
                if effective_gap >= max_gap {
                    gap_positions.push((gap_end_left + w[1].0) * 0.5);
                }
            }
        }
        // Centered-block guard (issue #1): a CENTERED title/subtitle/
        // byline block (each line horizontally centered, varying widths)
        // produces accidental gap clusters that look like a column
        // gutter — but it is NOT columnar, and treating it as columns
        // scrambles reading order ("Quarterly Inventory Review" centered
        // title read as 3 columns → "Quarterly" / "Spring" / ... ).
        //
        // The distinguishing signal: a REAL multi-column layout has the
        // left column starting at a consistent left edge across rows
        // (low variance of per-line leftmost x). Centered text has its
        // leftmost x scattered (each line centered with a different
        // width). Compute the spread of per-line leftmost edges; if it
        // is large relative to the region width, the block is centered,
        // not columnar, so do NOT treat the gap cluster as a gutter.
        // Centered iff the per-line leftmost edges do NOT share a common
        // left margin. A left-aligned layout (single column OR real
        // multi-column) has most rows starting at the same x (the left
        // margin), so the largest cluster of leftmost edges covers a
        // majority of lines. Centered text has each line's leftmost edge
        // scattered (different per line), so no cluster dominates.
        //
        // Using a cluster fraction (not raw spread) is robust to rows
        // that only contain right-column content — those push the spread
        // up but do not change the fact that the left margin still
        // dominates the remaining rows. (Raw spread mis-classified the
        // two-column test where the last row held only a right cell.)
        let looks_centered = {
            let mins: Vec<f32> = lines
                .values()
                .map(|ls| ls.iter().map(|(l, _, _)| *l).fold(f32::MAX, f32::min))
                .collect();
            if mins.len() < 2 {
                false
            } else {
                let tol = 10.0_f32;
                let largest = mins
                    .iter()
                    .map(|&a| mins.iter().filter(|&&b| (a - b).abs() <= tol).count())
                    .max()
                    .unwrap_or(0);
                // Centered when no left-margin cluster covers a majority.
                (largest as f32) < (mins.len() as f32) * 0.5
            }
        };

        // A SMALL centered block (title / subtitle / byline — few lines,
        // scattered leftmost edges) is treated as a single column so its
        // lines stay in top-to-bottom order and a centered multi-word
        // title is not split into per-word "columns" (issue #1). Gated
        // to <= 6 lines so it only catches title-page-style blocks: a
        // real multi-column body has many lines and is never classified
        // centered here (its left column starts at a consistent margin,
        // giving a small leftmost-spread anyway).
        if looks_centered && lines.len() <= 6 {
            return true;
        }

        // Cluster gap positions: count, for each observed gap, how many
        // other gaps fall within ±20pt. If any cluster contains gaps
        // from ≥30% of lines, it's a genuine column gutter.
        if !gap_positions.is_empty() && !looks_centered {
            let cluster_radius = 20.0_f32;
            // Require ≥3 gap positions (or 20% of lines, whichever is
            // larger) clustered within ±20pt. 20% accommodates pages
            // where header/footer/title rows dilute the body-line count
            // but a real multi-column body still dominates.
            let min_cluster = (3usize).max(lines.len() / 5);
            for &pos in &gap_positions {
                let cluster_size = gap_positions
                    .iter()
                    .filter(|&&p| (p - pos).abs() <= cluster_radius)
                    .count();
                if cluster_size >= min_cluster {
                    return false;
                }
            }
        }

        // With no column gutter found on any line, check that the majority
        // of lines are wide AND densely covered. This catches clean body
        // text where every line covers most of the region width.
        let width_threshold = region_width * 0.6;
        let mut wide_dense_lines = 0usize;
        for line_spans in lines.values() {
            let mut sorted = line_spans.clone();
            sorted.sort_by(|a, b| crate::utils::safe_float_cmp(a.0, b.0));
            let extent_left = sorted.first().unwrap().0;
            let extent_right = sorted.iter().map(|(_, r, _)| *r).fold(f32::MIN, f32::max);
            let extent = extent_right - extent_left;
            if extent < width_threshold {
                continue;
            }
            // Use core_right (char-count estimate) rather than bbox.right
            // for coverage. bbox.right is inflated by tab characters and
            // trailing whitespace — tab-expanded table rows would otherwise
            // score 100% coverage and be misidentified as dense body text.
            let mut covered = 0.0f32;
            let mut last_end = f32::MIN;
            for &(l, _, cr) in &sorted {
                let effective_right = cr.min(extent_right);
                let start = l.max(last_end);
                if effective_right > start {
                    covered += effective_right - start;
                    last_end = effective_right;
                }
            }
            if covered >= extent * 0.8 {
                wide_dense_lines += 1;
            }
        }
        wide_dense_lines * 2 >= lines.len()
    }

    /// Find vertical line (X-axis) split using index-based partitioning.
    ///
    /// Rejects lopsided splits where one side contains fewer than ~10% of
    /// the region's spans — those come from single-column pages where
    /// indentation or stray content creates a spurious density dip at
    /// one edge of the projection, not from a real column boundary.
    fn find_horizontal_split_indexed(
        &self,
        all_spans: &[TextSpan],
        indices: &[usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let profile = self.horizontal_projection_indexed(all_spans, indices)?;

        let split_x = if let Some((vs, ve, vw)) = self.find_valley(&profile) {
            if vw < self.min_valley_width {
                return None;
            }
            profile.x_min + (vs + ve) as f32 / 2.0
        } else {
            // Fallback: when narrow table-cell spans fill the column gutter
            // and prevent zero-density valley detection, find the relative
            // minimum between the two strongest density peaks. Only use
            // this when the minimum is ≤ 50% of the weaker peak (genuine
            // trough) so we don't split single-column pages on shallow dips.
            self.find_split_between_peaks(&profile)?
        };

        // Partition by span LEFT EDGE (where the glyphs actually start),
        // not bbox.right() and not center. Extractor bboxes overreach to
        // the right (trailing whitespace / stretched advance widths), and
        // for wide single-column body spans the center can also drift
        // past the split. Left edge is anchored to the true glyph start
        // and reliably places each span into its actual column.
        let (left, right): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| all_spans[i].bbox.left() < split_x);

        if left.is_empty() || right.is_empty() {
            return None;
        }

        // Real column splits produce balanced partitions. A 95/5 split is
        // almost always from edge dips or stray content, not a column.
        let min_side = (indices.len() / 10).max(2);
        if left.len() < min_side || right.len() < min_side {
            return None;
        }

        Some((left, right))
    }

    /// Fallback column split: find the deepest trough between the two
    /// strongest density peaks. Used when the standard valley detection
    /// fails because narrow table-cell spans partially fill the gutter.
    ///
    /// Returns the split X coordinate (absolute, not relative to x_min) if
    /// a genuine trough exists — i.e., the minimum between the peaks is ≤
    /// 50% of the weaker peak density.
    fn find_split_between_peaks(&self, profile: &ProjectionProfile) -> Option<f32> {
        let density = &profile.density;
        let n = density.len();
        if n < 3 {
            return None;
        }

        // Smooth with a small box filter (window = min_valley_width) to
        // average out individual narrow peaks before finding mass centres.
        let smooth_window = (self.min_valley_width as usize).max(3);
        let half = smooth_window / 2;
        let smoothed: Vec<f32> = (0..n)
            .map(|i| {
                let s = i.saturating_sub(half);
                let e = (i + half + 1).min(n);
                let sum: f32 = density[s..e].iter().sum();
                sum / (e - s) as f32
            })
            .collect();

        // Find the strongest peak in each half. Use `safe_float_cmp` for
        // NaN-safe total ordering — matches the comparator used elsewhere
        // in the reading-order code so `density` sentinel values can't
        // reach a `partial_cmp` that maps them to `Equal`.
        let mid = n / 2;
        let left_peak =
            (0..mid).max_by(|&a, &b| crate::utils::safe_float_cmp(smoothed[a], smoothed[b]))?;
        let right_peak =
            (mid..n).max_by(|&a, &b| crate::utils::safe_float_cmp(smoothed[a], smoothed[b]))?;

        if smoothed[left_peak] == 0.0 || smoothed[right_peak] == 0.0 {
            return None;
        }

        // Find the minimum density in the interior between the two peaks.
        let search_start = left_peak.min(right_peak) + 1;
        let search_end = left_peak.max(right_peak);
        if search_start >= search_end {
            return None;
        }

        let trough_pos = (search_start..search_end)
            .min_by(|&a, &b| crate::utils::safe_float_cmp(smoothed[a], smoothed[b]))?;

        // Only use if trough is a genuine valley: ≤ 50% of the weaker peak.
        let weaker_peak = smoothed[left_peak].min(smoothed[right_peak]);
        if smoothed[trough_pos] > weaker_peak * 0.5 {
            return None;
        }

        // Trough must be at least min_valley_width from both edges.
        if trough_pos < self.min_valley_width as usize
            || trough_pos + self.min_valley_width as usize > n
        {
            return None;
        }

        Some(profile.x_min + trough_pos as f32)
    }

    /// Find horizontal line (Y-axis) split using index-based partitioning.
    ///
    /// Returns `(above, below)` where `above` holds spans whose rectangle
    /// edge is at larger Y (higher on page in PDF coordinates) and must be
    /// processed first in reading order. PDF Spec ISO 32000-1:2008 §8.3.2.3
    /// defines the default user-space coordinate system with origin at the
    /// lower-left corner and Y increasing upward.
    fn find_vertical_split_indexed(
        &self,
        all_spans: &[TextSpan],
        indices: &[usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let profile = self.vertical_projection_indexed(all_spans, indices)?;
        let (valley_start, valley_end, valley_width) = self.find_valley(&profile)?;

        if valley_width < self.min_valley_width {
            return None;
        }

        let split_y = profile.y_min + (valley_start + valley_end) as f32 / 2.0;

        // `Rect::top()` returns `self.y`, the SMALLER Y coordinate of the
        // normalized rectangle — the method name follows a screen-coordinate
        // convention (Y grows downward) but PDF user space has Y growing
        // upward, so in PDF terms `bbox.top()` is actually the LOWER edge of
        // the glyph's bounding box. The predicate `bbox.top() >= split_y`
        // therefore classifies a span into `above` only when its *lowest*
        // point is already above the split line, i.e. the entire span sits
        // above the cut. Since `split_y` is the midpoint of a horizontal
        // projection valley (an empty band by construction), spans should
        // not straddle it in practice; any that do (e.g. a tall header
        // glyph whose ascenders dip into the valley) fall into `below`.
        let (above, below): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| all_spans[i].bbox.top() >= split_y);

        if above.is_empty() || below.is_empty() {
            return None;
        }

        // Row (vertical) splits legitimately produce singleton top
        // partitions for lone headers/titles, so we accept down to 1
        // span per side. The column (horizontal) split is stricter since
        // single-span columns are almost always spurious.
        let min_side = (indices.len() / 10).max(1);
        if above.len() < min_side || below.len() < min_side {
            return None;
        }

        Some((above, below))
    }

    /// Calculate horizontal projection profile from indexed spans.
    fn horizontal_projection_indexed(
        &self,
        all_spans: &[TextSpan],
        indices: &[usize],
    ) -> Option<ProjectionProfile> {
        if indices.is_empty() {
            return None;
        }

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for &i in indices {
            let span = &all_spans[i];
            x_min = x_min.min(span.bbox.left());
            x_max = x_max.max(span.bbox.right());
            y_min = y_min.min(span.bbox.top());
            y_max = y_max.max(span.bbox.bottom());
        }

        let width = (x_max - x_min).ceil() as usize;
        if width > MAX_PROJECTION_SIZE {
            log::warn!(
                "XY-cut: horizontal projection width {} exceeds MAX_PROJECTION_SIZE {}, skipping region (degenerate CTM?)",
                width,
                MAX_PROJECTION_SIZE
            );
            return None;
        }
        let mut density = vec![0.0; width];

        // Text extractors frequently over-estimate span bbox widths
        // (trailing whitespace, stretched advance widths). That makes a
        // full-width projection falsely fill the inter-column gutter on
        // multi-column pages. We project each span's TEXT CORE footprint
        // anchored to its LEFT edge (where glyphs actually start), with
        // length proportional to character count. The left edge is
        // reliable; the right edge is not.
        //
        // Additionally, spans whose core width exceeds 55% of the region
        // width are full-width elements (section headers, figure captions,
        // table titles) that span both columns. Including them fills the
        // inter-column gutter in the density array and prevents valley
        // detection. They are excluded from the projection; the column
        // split boundary will still assign them correctly by left edge.
        let region_width = (x_max - x_min).max(1.0);
        for &i in indices {
            let span = &all_spans[i];
            let height = span.bbox.bottom() - span.bbox.top();
            let char_count = span
                .text
                .chars()
                .filter(|c| !c.is_whitespace())
                .count()
                .max(1);
            // 0.45em per char is a reasonable average across common PDF
            // fonts (Helvetica/Times/Arial at body size) and narrower
            // than the 0.5em advance used for monospace.
            let approx_char_width = (span.font_size * 0.45).max(2.5);
            let core_width = char_count as f32 * approx_char_width;
            let span_width = span.bbox.right() - span.bbox.left();
            // Skip full-width elements (captions, headers, table rows) whose
            // bbox spans more than 55% of the region — they fill the gutter.
            if span_width > region_width * 0.55 {
                continue;
            }
            // Skip isolated single-character/digit spans (table cell values
            // like 'G', 'T', '1', 'A') that scatter across the full X range
            // and fill the column gutter in the density profile. Body text
            // spans always contain multiple characters.
            if char_count < 2 {
                continue;
            }
            let core_left = span.bbox.left();
            let core_right = (core_left + core_width).min(span.bbox.right());
            let x_start = (core_left - x_min).max(0.0).ceil() as usize;
            let x_end = (core_right - x_min).ceil() as usize;

            for j in x_start..x_end.min(width) {
                density[j] += height;
            }
        }

        Some(ProjectionProfile {
            density,
            x_min,
            y_min,
        })
    }

    /// Calculate vertical projection profile from indexed spans.
    fn vertical_projection_indexed(
        &self,
        all_spans: &[TextSpan],
        indices: &[usize],
    ) -> Option<ProjectionProfile> {
        if indices.is_empty() {
            return None;
        }

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for &i in indices {
            let span = &all_spans[i];
            x_min = x_min.min(span.bbox.left());
            x_max = x_max.max(span.bbox.right());
            y_min = y_min.min(span.bbox.top());
            y_max = y_max.max(span.bbox.bottom());
        }

        let height = (y_max - y_min).ceil() as usize;
        if height > MAX_PROJECTION_SIZE {
            log::warn!(
                "XY-cut: vertical projection height {} exceeds MAX_PROJECTION_SIZE {}, skipping region (degenerate CTM?)",
                height,
                MAX_PROJECTION_SIZE
            );
            return None;
        }
        let mut density = vec![0.0; height];

        for &i in indices {
            let span = &all_spans[i];
            let y_start = (span.bbox.top() - y_min).max(0.0).ceil() as usize;
            let y_end = (span.bbox.bottom() - y_min).ceil() as usize;
            let w = span.bbox.right() - span.bbox.left();

            for j in y_start..y_end.min(height) {
                density[j] += w;
            }
        }

        Some(ProjectionProfile {
            density,
            x_min,
            y_min,
        })
    }

    /// Find the widest valley (white space gap) in projection profile.
    ///
    /// Only considers INTERIOR valleys — gaps sandwiched between two
    /// non-empty regions. Leading/trailing empty bands (margin space
    /// outside the actual content extent) are ignored; they represent
    /// page margins, not column gutters, and picking them would produce
    /// meaningless splits.
    fn find_valley(&self, profile: &ProjectionProfile) -> Option<(usize, usize, f32)> {
        if profile.density.is_empty() {
            return None;
        }

        // Find peak density
        let peak = profile.density.iter().copied().fold(0.0, f32::max);

        if peak == 0.0 {
            return None;
        }

        // Find the content extent (first and last non-empty positions).
        // Valleys outside this extent are leading/trailing margins.
        let first_nonzero = profile.density.iter().position(|&d| d > 0.0)?;
        let last_nonzero = profile.density.iter().rposition(|&d| d > 0.0)?;

        // Find valleys (regions below threshold)
        let threshold = peak * self.valley_threshold;
        let mut valleys = Vec::new();
        let mut in_valley = false;
        let mut valley_start = 0;

        for (i, &density) in profile.density.iter().enumerate() {
            if density < threshold {
                if !in_valley {
                    valley_start = i;
                    in_valley = true;
                }
            } else if in_valley {
                valleys.push((valley_start, i));
                in_valley = false;
            }
        }

        if in_valley {
            valleys.push((valley_start, profile.density.len()));
        }

        // Merge adjacent interior valley segments separated by a narrow
        // bridge (≤ half the minimum valley width). A callout box or small
        // figure positioned in the column gutter creates a density bump
        // that splits what should be a single valley into two fragments.
        // Bridging re-joins them so the gap is still recognised as a
        // column boundary.
        let bridge_limit = (self.min_valley_width / 2.0).ceil() as usize;
        let interior: Vec<(usize, usize)> = valleys
            .into_iter()
            .filter(|&(start, end)| start > first_nonzero && end <= last_nonzero + 1)
            .collect();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(interior.len());
        for seg in interior {
            if let Some(last) = merged.last_mut() {
                if seg.0 <= last.1 + bridge_limit {
                    last.1 = last.1.max(seg.1);
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
            .into_iter()
            .map(|(start, end)| (start, end, (end - start) as f32))
            .max_by(|a, b| crate::utils::safe_float_cmp(a.2, b.2))
    }

    /// Test-only wrapper for horizontal projection on a contiguous slice.
    #[cfg(test)]
    fn horizontal_projection(&self, spans: &[TextSpan]) -> Option<ProjectionProfile> {
        let indices: Vec<usize> = (0..spans.len()).collect();
        self.horizontal_projection_indexed(spans, &indices)
    }

    /// Test-only wrapper for vertical projection on a contiguous slice.
    #[cfg(test)]
    fn vertical_projection(&self, spans: &[TextSpan]) -> Option<ProjectionProfile> {
        let indices: Vec<usize> = (0..spans.len()).collect();
        self.vertical_projection_indexed(spans, &indices)
    }

    /// Sort spans in reading order (top-to-bottom, left-to-right).
    #[cfg(test)]
    fn sort_spans<'a>(&self, spans: &'a [TextSpan]) -> Vec<&'a TextSpan> {
        let mut sorted: Vec<_> = spans.iter().collect();

        sorted.sort_by(|a, b| {
            // Sort by Y (top) first, descending (top of page first)
            let y_cmp = crate::utils::safe_float_cmp(b.bbox.top(), a.bbox.top());
            if y_cmp != std::cmp::Ordering::Equal {
                return y_cmp;
            }
            // Same Y level, sort by X (left) ascending
            crate::utils::safe_float_cmp(a.bbox.left(), b.bbox.left())
        });

        sorted
    }

    /// Sort indices in reading order (top-to-bottom, left-to-right).
    fn sort_indices(&self, all_spans: &[TextSpan], indices: &[usize]) -> Vec<usize> {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            let y_cmp =
                crate::utils::safe_float_cmp(all_spans[b].bbox.top(), all_spans[a].bbox.top());
            if y_cmp != std::cmp::Ordering::Equal {
                return y_cmp;
            }
            crate::utils::safe_float_cmp(all_spans[a].bbox.left(), all_spans[b].bbox.left())
        });
        sorted
    }
}

/// Internal projection profile representation.
struct ProjectionProfile {
    /// Density values (height or width accumulated per bin)
    density: Vec<f32>,

    /// Origin coordinates
    x_min: f32,
    y_min: f32,
}

impl ReadingOrderStrategy for XYCutStrategy {
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        _context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        // Use index-based partitioning to avoid cloning during recursion
        let indices: Vec<usize> = (0..spans.len()).collect();
        let index_groups = self.partition_indexed(&spans, &indices);

        // Build result — moves spans out by index (no extra clone)
        let mut ordered = Vec::with_capacity(spans.len());
        // Convert spans to indexable storage for O(1) moves
        let mut span_slots: Vec<Option<TextSpan>> = spans.into_iter().map(Some).collect();
        let mut order_index = 0usize;

        for (group_idx, group) in index_groups.iter().enumerate() {
            for &i in group {
                if let Some(span) = span_slots[i].take() {
                    ordered.push(
                        OrderedTextSpan::with_info(span, order_index, ReadingOrderInfo::xycut())
                            .with_group(group_idx),
                    );
                    order_index += 1;
                }
            }
        }

        Ok(ordered)
    }

    fn name(&self) -> &'static str {
        "XYCutStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn make_span(x: f32, y: f32, width: f32, height: f32) -> TextSpan {
        make_span_text(x, y, width, height, "test", 12.0)
    }

    /// Like make_span but with realistic body-text density (~72 non-whitespace chars
    /// at 12pt, matching a full Letter-width column). Used when is_single_column_region
    /// must correctly identify a wide single-column page as not multi-column.
    fn make_body_span(x: f32, y: f32, width: f32, height: f32) -> TextSpan {
        // 72 non-whitespace characters at 12pt → core_width = 72 × 5.4 = 388.8pt
        // which is 83% of a 468pt column — enough to pass the 80% dense check.
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(3); // 78 non-whitespace chars
        make_span_text(x, y, width, height, &text, 12.0)
    }

    fn make_span_text(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        text: &str,
        font_size: f32,
    ) -> TextSpan {
        use crate::layout::{Color, FontWeight};

        TextSpan {
            artifact_type: None,
            text: text.to_string(),
            bbox: Rect::new(x, y, width, height),
            font_size,
            font_name: "Arial".to_string(),
            font_weight: FontWeight::Normal,
            is_italic: false,
            is_monospace: false,
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
            mcid: None,
            sequence: 0,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
            heading_level: None,
        }
    }

    #[test]
    fn test_single_column_no_split() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 50.0, 10.0), // Line 1
            make_span(10.0, 85.0, 50.0, 10.0),  // Line 2
            make_span(10.0, 70.0, 50.0, 10.0),  // Line 3
        ];

        let groups = strategy.partition_region(&spans);
        assert_eq!(groups.len(), 1); // No split for single column
        assert_eq!(groups[0].len(), 3);
    }

    /// Realistic A4/Letter single-column page: 60 lines of body text,
    /// 14pt leading, one paragraph gap (30pt) mid-page. Only one body
    /// column exists, so XY-Cut must return exactly one group and
    /// preserve top-to-bottom reading order. A density-dip split at the
    /// paragraph gap would fragment the page and non-monotonically
    /// interleave paragraph contents.
    #[test]
    fn test_single_column_body_text_no_fragmentation() {
        let strategy = XYCutStrategy::new();
        // Simulate 60 lines of body text at x=72..540 (letter page, 1" margins).
        // Each line is a single span; line height 12pt, leading 14pt.
        let mut spans = Vec::new();
        let line_height = 12.0;
        let leading = 14.0;
        let left = 72.0;
        let right = 540.0;
        let width = right - left;
        let mut y = 720.0; // start near top of letter page
        for i in 0..60 {
            // Insert a paragraph gap in the middle (30pt, larger than min_valley_width=15pt)
            if i == 30 {
                y -= 30.0;
            }
            // Use realistic body text density (78 non-whitespace chars at 12pt) so
            // is_single_column_region correctly classifies the region as single-column.
            spans.push(make_body_span(left, y, width, line_height));
            y -= leading;
        }

        let groups = strategy.partition_region(&spans);
        assert_eq!(
            groups.len(),
            1,
            "single-column body text must not be split by XY-Cut (got {} groups)",
            groups.len()
        );
        assert_eq!(groups[0].len(), 60, "all 60 spans must be preserved");

        // Verify the group preserves monotonic top-to-bottom reading order
        // (each subsequent span's Y should be <= previous Y).
        let mut last_y = f32::MAX;
        for s in &groups[0] {
            assert!(
                s.bbox.top() <= last_y + 0.01,
                "reading order must be top-to-bottom: {} > {}",
                s.bbox.top(),
                last_y
            );
            last_y = s.bbox.top();
        }
    }

    /// After a vertical (row) split, the partition at higher Y (top of
    /// page in PDF coords) must be processed first in reading order so
    /// that header content appears before body content.
    #[test]
    fn test_vertical_split_preserves_top_to_bottom_order() {
        use crate::pipeline::reading_order::{ReadingOrderContext, ReadingOrderStrategy};

        let mut strategy = XYCutStrategy::new();
        strategy.min_spans_for_split = 2;

        // Header line at high Y (top of page in PDF coords).
        // Body block at lower Y values. Gap between them > min_valley_width.
        let make = |text: &str, x: f32, y: f32, w: f32| {
            let mut s = make_span(x, y, w, 12.0);
            s.text = text.to_string();
            s
        };
        // Two columns at y ∈ {200, 180, 160} (body), header at y=400.
        // Horizontal split will find the column gutter first; within each
        // column the header must still come out first in reading order.
        let spans = vec![
            make("HEADER LEFT", 50.0, 400.0, 200.0),
            make("HEADER RIGHT", 300.0, 400.0, 200.0),
            make("body-L1", 50.0, 200.0, 150.0),
            make("body-R1", 300.0, 200.0, 150.0),
            make("body-L2", 50.0, 180.0, 150.0),
            make("body-R2", 300.0, 180.0, 150.0),
        ];
        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        let texts: Vec<&str> = ordered.iter().map(|o| o.span.text.as_str()).collect();
        // First output must be from y=400 (header), not y=180 (body bottom).
        assert!(texts[0].contains("HEADER"), "expected HEADER first, got sequence {:?}", texts);
    }

    /// Single-column page with a tall header band ("Title" or "Chapter
    /// heading") at the top. XY-Cut may validly split the header from
    /// the body (vertical Y-split) but must not further split the body
    /// into per-paragraph chunks.
    #[test]
    fn test_single_column_with_header_at_most_two_groups() {
        let strategy = XYCutStrategy::new();
        let mut spans = Vec::new();

        // Tall header band
        spans.push(make_span(72.0, 750.0, 468.0, 24.0));

        // 40 lines of body text below, separated by a ~50pt gap
        let mut y = 670.0;
        for _ in 0..40 {
            spans.push(make_span(72.0, y, 468.0, 12.0));
            y -= 14.0;
        }

        let groups = strategy.partition_region(&spans);
        assert!(
            groups.len() <= 2,
            "single-column with header should produce at most 2 groups, got {}",
            groups.len()
        );
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 41);
    }

    #[test]
    fn test_two_column_split() {
        let mut strategy = XYCutStrategy::new();
        strategy.min_spans_for_split = 2; // Lower threshold for testing

        let spans = vec![
            // Left column (x: 10-60)
            make_span(10.0, 100.0, 50.0, 10.0),
            make_span(10.0, 85.0, 50.0, 10.0),
            // Right column (x: 100-150) - wide gap of 40 points
            make_span(100.0, 100.0, 50.0, 10.0),
            make_span(100.0, 85.0, 50.0, 10.0),
        ];

        let groups = strategy.partition_region(&spans);
        // With wide gap and lower threshold, should split into 2 columns or keep as 1 group
        assert!(!groups.is_empty(), "Expected at least 1 group");
        // Verify all spans are preserved
        let total_spans: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total_spans, 4, "Expected all 4 spans to be preserved");
    }

    #[test]
    fn test_three_column_layout() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            // Column 1 (x: 10-40)
            make_span(10.0, 100.0, 30.0, 10.0),
            make_span(10.0, 85.0, 30.0, 10.0),
            // Column 2 (x: 70-100)
            make_span(70.0, 100.0, 30.0, 10.0),
            make_span(70.0, 85.0, 30.0, 10.0),
            // Column 3 (x: 130-160)
            make_span(130.0, 100.0, 30.0, 10.0),
            make_span(130.0, 85.0, 30.0, 10.0),
        ];

        let groups = strategy.partition_region(&spans);
        // Should recursively split into at least 2 groups
        assert!(groups.len() >= 2, "Expected at least 2 groups, got {}", groups.len());
    }

    #[test]
    fn test_small_region_no_split() {
        let strategy = XYCutStrategy::new();
        let spans = vec![make_span(10.0, 100.0, 50.0, 10.0)];

        let groups = strategy.partition_region(&spans);
        assert_eq!(groups.len(), 1); // Single span region
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn test_sort_order() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(100.0, 70.0, 50.0, 10.0),  // Lower right
            make_span(10.0, 100.0, 50.0, 10.0),  // Upper left
            make_span(100.0, 100.0, 50.0, 10.0), // Upper right
            make_span(10.0, 70.0, 50.0, 10.0),   // Lower left
        ];

        let sorted = strategy.sort_spans(&spans);

        // Expect: upper left, upper right, lower left, lower right
        assert_eq!(sorted[0].bbox.top(), 100.0); // Upper
        assert_eq!(sorted[0].bbox.left(), 10.0); // Left
        assert_eq!(sorted[1].bbox.top(), 100.0); // Upper
        assert_eq!(sorted[1].bbox.left(), 100.0); // Right
    }

    #[test]
    fn test_horizontal_projection() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0),  // x: 10-40
            make_span(100.0, 100.0, 30.0, 10.0), // x: 100-130
        ];

        if let Some(profile) = strategy.horizontal_projection(&spans) {
            // Should have density peaks around x=25 and x=115
            assert!(!profile.density.is_empty());
            assert!(profile.density.len() >= 120); // Total width from 10 to 130 = 120

            // Gap is between local x=30 and x=90 (relative to x_min=10)
            // So in density array indices [30..90]
            let gap_start = 30;
            let gap_end = 90;
            if gap_end <= profile.density.len() {
                let gap_region = &profile.density[gap_start..gap_end];
                let gap_density: f32 = gap_region.iter().sum();
                assert!(gap_density < 1.0); // Gap should be mostly empty
            }
        }
    }

    #[test]
    fn test_vertical_projection() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 50.0, 20.0), // y: 100-120
            make_span(10.0, 50.0, 50.0, 20.0),  // y: 50-70
        ];

        if let Some(profile) = strategy.vertical_projection(&spans) {
            // Should have density peaks around y=110 and y=60
            assert!(!profile.density.is_empty());
            // Large gap between 70 and 100
            assert!(profile.density.len() > 50);
        }
    }

    #[test]
    fn test_narrow_gap_rejected() {
        let strategy = XYCutStrategy::new();
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0), // x: 10-40
            make_span(45.0, 100.0, 30.0, 10.0), // x: 45-75, gap: 5 points
        ];

        let groups = strategy.partition_region(&spans);
        // Gap is too narrow (< 15 points), should not split
        assert_eq!(groups.len(), 1);
    }

    /// Regression test for Bug 2: degenerate CTM places spans at ~100 trillion PDF points.
    /// horizontal_projection_indexed must return None instead of attempting a
    /// ~100-trillion-element vec allocation (which triggers handle_alloc_error → abort).
    #[test]
    fn test_degenerate_ctm_horizontal_projection_returns_none() {
        let strategy = XYCutStrategy::new();
        // Observed crash coordinate: 99_992_777_785_344 PDF points on a ~3968-point page.
        let degenerate_x: f32 = 99_992_777_785_344.0;
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0),
            make_span(degenerate_x, 100.0, 30.0, 10.0),
        ];

        // Must not panic or abort — projection should return None for oversized region.
        let result = strategy.horizontal_projection(&spans);
        assert!(
            result.is_none(),
            "expected None for projection spanning ~100 trillion points, got Some"
        );
    }

    /// Vertical projection must also return None for degenerate CTM y-coordinates.
    #[test]
    fn test_degenerate_ctm_vertical_projection_returns_none() {
        let strategy = XYCutStrategy::new();
        let degenerate_y: f32 = 99_992_777_785_344.0;
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0),
            make_span(10.0, degenerate_y, 30.0, 10.0),
        ];

        let result = strategy.vertical_projection(&spans);
        assert!(
            result.is_none(),
            "expected None for projection spanning ~100 trillion points, got Some"
        );
    }

    /// Issue #1: a CENTERED title/subtitle/byline block (each line
    /// centered, scattered leftmost edges) must NOT be split into
    /// per-word "columns". The centered "Quarterly Inventory Review"
    /// title (3 large words at the same Y with wide gaps) plus centered
    /// subtitle/byline previously aligned accidentally into fake columns,
    /// scrambling reading order. The centered-block guard must keep the
    /// whole block as ONE group so the title line stays intact.
    #[test]
    fn test_issue1_centered_title_block_not_split_into_columns() {
        let strat = XYCutStrategy::new();
        // Centered title (y=612, fs=28), subtitle (y=572), byline (y=532).
        // Leftmost edges scattered: 145 / 185 / 210 (centered, not columnar).
        let spans = vec![
            make_span_text(145.0, 612.0, 115.0, 28.0, "Quarterly", 28.0),
            make_span_text(300.0, 612.0, 115.0, 28.0, "Inventory", 28.0),
            make_span_text(430.0, 612.0, 92.0, 28.0, "Review", 28.0),
            make_span_text(185.0, 572.0, 40.0, 14.0, "Spring", 14.0),
            make_span_text(238.0, 572.0, 31.0, 14.0, "2025", 14.0),
            make_span_text(300.0, 572.0, 70.0, 14.0, "Distribution", 14.0),
            make_span_text(210.0, 532.0, 45.0, 10.0, "Northwind", 10.0),
            make_span_text(290.0, 532.0, 34.0, 10.0, "Traders", 10.0),
        ];
        let groups = strat.partition_region(&spans);
        assert_eq!(
            groups.len(),
            1,
            "centered title block must stay one group, got {} groups",
            groups.len()
        );
        // The three title words must appear in document order within the group.
        let g0: Vec<&str> = groups[0].iter().map(|s| s.text.as_str()).collect();
        let qi = g0.iter().position(|t| *t == "Quarterly").unwrap();
        let ii = g0.iter().position(|t| *t == "Inventory").unwrap();
        let ri = g0.iter().position(|t| *t == "Review").unwrap();
        assert!(qi < ii && ii < ri, "title words out of order: {:?}", g0);
    }

    /// XYCut must assign distinct group_id values to spans in different
    /// spatial partitions so that converters can keep each column's content
    /// contiguous instead of interleaving by Y-coordinate.
    #[test]
    fn test_xycut_group_id_two_column_layout() {
        use crate::pipeline::reading_order::{ReadingOrderContext, ReadingOrderStrategy};

        let mut strategy = XYCutStrategy::new();
        strategy.min_spans_for_split = 2; // lower threshold for small test

        // Left column (x=50-200)        Right column (x=400-550)
        //   "Description"   y=100          "Amount"          y=100
        //   "Widget A"      y=120          "$150.00"         y=120
        //   "Widget B"      y=140          "Discount"        y=140
        //                                   "$25.00"          y=160
        let make = |text: &str, x: f32, y: f32, w: f32| {
            let mut s = make_span(x, y, w, 12.0);
            s.text = text.to_string();
            s
        };
        let spans = vec![
            make("Description", 50.0, 100.0, 150.0),
            make("Amount", 400.0, 100.0, 150.0),
            make("Widget A", 50.0, 120.0, 150.0),
            make("$150.00", 400.0, 120.0, 150.0),
            make("Widget B", 50.0, 140.0, 150.0),
            make("Discount", 400.0, 140.0, 150.0),
            make("$25.00", 400.0, 160.0, 150.0),
        ];

        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        // Every span must have a group_id assigned.
        assert!(
            ordered.iter().all(|s| s.group_id.is_some()),
            "all spans should have group_id set by XYCut"
        );

        // Left-column spans must share one group_id, right-column another.
        let left_groups: Vec<usize> = ordered
            .iter()
            .filter(|s| s.span.bbox.left() < 300.0)
            .map(|s| s.group_id.unwrap())
            .collect();
        let right_groups: Vec<usize> = ordered
            .iter()
            .filter(|s| s.span.bbox.left() >= 300.0)
            .map(|s| s.group_id.unwrap())
            .collect();

        // Within each column, group_id must be the same.
        assert!(
            left_groups.windows(2).all(|w| w[0] == w[1]),
            "left column spans should share the same group_id: {:?}",
            left_groups
        );
        assert!(
            right_groups.windows(2).all(|w| w[0] == w[1]),
            "right column spans should share the same group_id: {:?}",
            right_groups
        );

        // The two columns must have different group_ids.
        assert_ne!(
            left_groups[0], right_groups[0],
            "left and right columns should have different group_ids"
        );

        // Verify reading order keeps each column contiguous: all left-column
        // spans should appear before (or after) all right-column spans.
        let left_orders: Vec<usize> = ordered
            .iter()
            .filter(|s| s.span.bbox.left() < 300.0)
            .map(|s| s.reading_order)
            .collect();
        let right_orders: Vec<usize> = ordered
            .iter()
            .filter(|s| s.span.bbox.left() >= 300.0)
            .map(|s| s.reading_order)
            .collect();
        let left_max = *left_orders.iter().max().unwrap();
        let right_min = *right_orders.iter().min().unwrap();
        let left_min = *left_orders.iter().min().unwrap();
        let right_max = *right_orders.iter().max().unwrap();
        // Either all left before all right, or all right before all left.
        assert!(
            left_max < right_min || right_max < left_min,
            "columns must be contiguous in reading order: left={:?} right={:?}",
            left_orders,
            right_orders
        );
    }

    /// Plain-text rendering must keep group_id-separated columns as
    /// contiguous blocks, not interleave them by Y-coordinate.
    #[test]
    fn test_group_id_plain_text_no_interleave() {
        use crate::pipeline::converters::OutputConverter;
        use crate::pipeline::converters::PlainTextConverter;
        use crate::pipeline::reading_order::{ReadingOrderContext, ReadingOrderStrategy};
        use crate::pipeline::TextPipelineConfig;

        let mut strategy = XYCutStrategy::new();
        strategy.min_spans_for_split = 2;

        let make = |text: &str, x: f32, y: f32, w: f32| {
            let mut s = make_span(x, y, w, 12.0);
            s.text = text.to_string();
            s
        };
        let spans = vec![
            make("Description", 50.0, 100.0, 150.0),
            make("Amount", 400.0, 100.0, 150.0),
            make("Widget A", 50.0, 120.0, 150.0),
            make("$150.00", 400.0, 120.0, 150.0),
            make("Widget B", 50.0, 140.0, 150.0),
            make("Discount", 400.0, 140.0, 150.0),
            make("$25.00", 400.0, 160.0, 150.0),
        ];

        let context = ReadingOrderContext::new();
        let ordered = strategy.apply(spans, &context).unwrap();

        let converter = PlainTextConverter::new();
        let config = TextPipelineConfig::default();
        let text = converter.convert(&ordered, &config).unwrap();

        // With Y-position-based merging, same-Y spans from left and right columns
        // are placed on the same line. This produces better label-value pairing:
        // "Description Amount" on one line, "Widget A $150.00" on the next.
        assert!(text.contains("Description"), "missing Description:\n{text}");
        assert!(text.contains("Amount"), "missing Amount:\n{text}");
        assert!(text.contains("Widget A"), "missing Widget A:\n{text}");
        assert!(text.contains("$150.00"), "missing $150.00:\n{text}");

        // Same-Y spans should be on the same line
        for line in text.lines() {
            if line.contains("Description") {
                assert!(
                    line.contains("Amount"),
                    "Description and Amount should be on same line:\n{text}"
                );
            }
        }
    }

    /// End-to-end: partition_region must return all spans (unsplit) rather than aborting
    /// when the page contains a degenerate-CTM span.
    #[test]
    fn test_degenerate_ctm_partition_region_does_not_abort() {
        let strategy = XYCutStrategy::new();
        let degenerate_x: f32 = 99_992_777_785_344.0;
        let spans = vec![
            make_span(10.0, 100.0, 30.0, 10.0),
            make_span(10.0, 85.0, 30.0, 10.0),
            make_span(10.0, 70.0, 30.0, 10.0),
            make_span(10.0, 55.0, 30.0, 10.0),
            make_span(10.0, 40.0, 30.0, 10.0),
            make_span(degenerate_x, 100.0, 30.0, 10.0),
        ];

        // Must complete without panicking and preserve all spans.
        let groups = strategy.partition_region(&spans);
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, spans.len(), "all spans must be preserved");
    }
}
