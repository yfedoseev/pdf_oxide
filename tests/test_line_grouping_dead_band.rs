//! Regression tests for the widened same-line threshold fix:
//! forward-gap guard, `should_insert_space` harmonization, and
//! threshold-boundary behavior.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::writer::{PageBuilder, PdfWriter};

fn put(page: &mut PageBuilder<'_>, text: &str, x: f32, y: f32, font: &str, size: f32) {
    page.add_text(text, x, y, font, size);
    // Force a BT/ET boundary so adjacent add_text calls are emitted as
    // separate text objects. Without this, the extractor may merge them
    // into a single span, making these regression assertions ineffective.
    page.draw_rect(0.0, 0.0, 0.0, 0.0);
}

fn build_and_extract(build_fn: impl FnOnce(&mut PdfWriter)) -> String {
    let mut writer = PdfWriter::new();
    build_fn(&mut writer);
    let bytes = writer.finish().expect("build PDF");
    let doc = PdfDocument::from_bytes(bytes).expect("open PDF");
    doc.extract_text(0).expect("extract page 0")
}

fn newline_between(out: &str, before: &str, after: &str) -> bool {
    let a = out
        .find(before)
        .unwrap_or_else(|| panic!("missing {:?}: {:?}", before, out));
    let b = out
        .find(after)
        .unwrap_or_else(|| panic!("missing {:?}: {:?}", after, out));
    out[a + before.len()..b].contains('\n')
}

// A. Title (20pt) + small right-aligned marker (10pt), y_diff=4.5pt.
#[test]
fn title_plus_right_aligned_marker_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "Form 1040", 72.0, 700.0, "Helvetica", 20.0);
        put(&mut page, "(Rev. Jan 2024)", 260.0, 695.5, "Helvetica", 10.0);
    });

    assert!(newline_between(&out, "Form 1040", "(Rev. Jan 2024)"), "got {:?}", out);
}

// B. Header (16pt) + small instruction (9pt), y_diff=3.5pt.
#[test]
fn header_plus_small_instruction_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "Section 3", 72.0, 700.0, "Helvetica", 16.0);
        put(&mut page, "(see instructions)", 240.0, 696.5, "Helvetica", 9.0);
    });

    assert!(newline_between(&out, "Section 3", "(see instructions)"), "got {:?}", out);
}

// C. Body (11pt) + small annotation (8pt) in dead-band, y_diff=3.5pt.
#[test]
fn body_plus_small_annotation_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "See reference 12", 72.0, 700.0, "Helvetica", 11.0);
        put(&mut page, "[updated 2024]", 220.0, 696.5, "Helvetica", 8.0);
    });

    assert!(newline_between(&out, "See reference 12", "[updated 2024]"), "got {:?}", out);
}

// D. Two-row small-gutter dead-band layout. K=1.5 accepts narrow intra-row
// gaps as residual — pin row-boundary integrity only.
#[test]
fn small_gutter_dead_band_rows_preserved() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "AA1", 72.0, 700.0, "Helvetica", 10.0);
        put(&mut page, "BB1", 92.0, 700.0, "Helvetica", 10.0);
        put(&mut page, "AA2", 72.0, 685.6, "Helvetica", 10.0);
        put(&mut page, "BB2", 92.0, 682.1, "Helvetica", 10.0);
    });

    assert!(newline_between(&out, "BB1", "AA2"), "got {:?}", out);
}

// E1. 12pt pair at y_diff=5.99 < 14.4 = 1.2*min_fs: stays same-line.
#[test]
fn threshold_boundary_inside_stays_same_line() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "LLL", 72.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "RRR", 95.0, 694.01, "Helvetica", 12.0);
    });

    let out = out.trim_end();
    assert!(!newline_between(out, "LLL", "RRR"), "got {:?}", out);
}

// E2. 12pt pair at y_diff=14.51 > 14.4 = 1.2*min_fs: splits into two lines.
#[test]
fn threshold_boundary_outside_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "LLL", 72.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "RRR", 95.0, 685.49, "Helvetica", 12.0);
    });

    assert!(newline_between(&out, "LLL", "RRR"), "got {:?}", out);
}

// 12pt pair at y_diff=1.5 (below old 2.0 threshold): the forward-gap
// guard's y_diff gate must not fire even with a wide word-spacing gap.
#[test]
fn pair_below_old_threshold_space_merges() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "Alpha", 100.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "Beta", 180.0, 698.5, "Helvetica", 12.0);
    });

    let out = out.trim_end();
    assert!(out.contains("Alpha Beta"), "got {:?}", out);
}

// 12pt pair in dead-band (y_diff=4.0) with a narrow gap ~5pt
// (gap/fs ≈ 0.4): documented residual — space-merges rather than splits.
#[test]
fn pair_dead_band_narrow_gap_space_merges() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "First", 100.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "Second", 128.0, 696.0, "Helvetica", 12.0);
    });

    let out = out.trim_end();
    assert!(out.contains("First Second"), "got {:?}", out);
    assert!(!newline_between(out, "First", "Second"), "got {:?}", out);
}

// 12pt pair at y_diff=15.0 (above the 14.4 = 1.2*fs same-line threshold): splits.
#[test]
fn pair_above_new_threshold_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "High", 100.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "Low", 100.0, 685.0, "Helvetica", 12.0);
    });

    assert!(newline_between(&out, "High", "Low"), "got {:?}", out);
}

// Wide-gutter two-column, fs=10, intra-row y_diff=4.0 and gap >> 1.5*fs.
// Forward-gap guard fires regardless of the dead-band Y-jitter.
#[test]
fn wide_gutter_dead_band_column_splits() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "Left", 80.0, 700.0, "Helvetica", 10.0);
        put(&mut page, "Right", 400.0, 696.0, "Helvetica", 10.0);
    });

    assert!(newline_between(&out, "Left", "Right"), "got {:?}", out);
}

// F. Aligned two-column negative control — fix must not change extraction
// when every row has identical baselines.
#[test]
fn aligned_two_column_extracts_unchanged() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "HdrLeft", 72.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "HdrRight", 300.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "BodyLeft", 72.0, 685.6, "Helvetica", 12.0);
        put(&mut page, "BodyRight", 300.0, 685.6, "Helvetica", 12.0);
        put(&mut page, "FootLeft", 72.0, 671.2, "Helvetica", 12.0);
        put(&mut page, "FootRight", 300.0, 671.2, "Helvetica", 12.0);
    });

    for cell in [
        "HdrLeft",
        "HdrRight",
        "BodyLeft",
        "BodyRight",
        "FootLeft",
        "FootRight",
    ] {
        assert!(out.contains(cell), "missing {:?}: {:?}", cell, out);
    }
    assert!(newline_between(&out, "HdrRight", "BodyLeft"), "got {:?}", out);
    assert!(newline_between(&out, "BodyRight", "FootLeft"), "got {:?}", out);
}
