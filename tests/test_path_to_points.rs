//! Integration coverage for `PathContent::to_points` (issue #147) against the
//! committed curve-heavy fixtures. The unit tests in `src/elements/path.rs`
//! prove correctness on synthetic shapes; these assert that flattening real,
//! extracted paths upholds global invariants:
//!
//!   * every emitted point is finite,
//!   * every point lies within the path's bounding box (curve points stay in
//!     the control-point convex hull, hence in the bbox),
//!   * the point count per path is bounded (no runaway subdivision),
//!   * curves are actually exercised and densified (the test isn't a no-op).

use pdf_oxide::document::PdfDocument;
use pdf_oxide::elements::PathOperation;

/// Per-path sanity ceiling: even a very wiggly path should not flatten to more
/// than this many points at a 0.5 pt tolerance. Catches subdivision blowups.
const MAX_POINTS_PER_PATH: usize = 200_000;

fn check_fixture(path: &str, tolerance: f32) {
    let doc = PdfDocument::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let pages = doc
        .page_count()
        .unwrap_or_else(|e| panic!("page_count {path}: {e}"));

    let mut paths_seen = 0usize;
    let mut curves_seen = 0usize;
    let mut densified_curve_paths = 0usize;

    for page in 0..pages {
        let extracted = match doc.extract_paths(page) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for path_content in &extracted {
            paths_seen += 1;
            let curve_ops = path_content
                .operations
                .iter()
                .filter(|o| matches!(o, PathOperation::CurveTo(..)))
                .count();
            curves_seen += curve_ops;

            let subpaths = path_content.to_points(tolerance);
            let total: usize = subpaths.iter().map(Vec::len).sum();
            assert!(
                total <= MAX_POINTS_PER_PATH,
                "{path} page {page}: {total} points exceeds ceiling"
            );

            let bbox = &path_content.bbox;
            // Float slack for accumulated f32 error in subdivision arithmetic.
            let eps = 1e-2_f32 + 1e-4_f32 * bbox.width.abs().max(bbox.height.abs());
            for sub in &subpaths {
                for &(x, y) in sub {
                    assert!(
                        x.is_finite() && y.is_finite(),
                        "{path} page {page}: non-finite point ({x},{y})"
                    );
                    assert!(
                        x >= bbox.x - eps
                            && x <= bbox.x + bbox.width + eps
                            && y >= bbox.y - eps
                            && y <= bbox.y + bbox.height + eps,
                        "{path} page {page}: point ({x},{y}) outside bbox {bbox:?}",
                    );
                }
            }

            // A path containing curves must flatten to strictly more vertices
            // than it has operations — evidence the Bézier was densified.
            if curve_ops > 0 && total > path_content.operations.len() {
                densified_curve_paths += 1;
            }
        }
    }

    assert!(paths_seen > 0, "{path}: no paths extracted — fixture changed?");
    assert!(
        curves_seen > 0,
        "{path}: no curves found — fixture no longer exercises flattening"
    );
    assert!(
        densified_curve_paths > 0,
        "{path}: no curve path was densified — flattening not exercised"
    );
}

#[test]
fn to_points_invariants_arxiv_paper() {
    check_fixture("tests/fixtures/1008.3918v2.pdf", 0.5);
}

#[test]
fn to_points_invariants_fixture_1() {
    check_fixture("tests/fixtures/1.pdf", 0.5);
}
