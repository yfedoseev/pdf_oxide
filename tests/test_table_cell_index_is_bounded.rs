//! A ruled table's cell index costs memory bounded by the page's content, not by
//! its coordinates.
//!
//! Peak heap is measured with a counting global allocator. This file is its own
//! test binary, and the tests share a lock so their measurements do not overlap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use pdf_oxide::PdfDocument;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// A one-page untagged PDF with a ruled two-row table of `cols` columns.
///
/// The rows are bounded by rules at y = 100, 200 and `top`; with `top` far
/// above 200 the upper row's cells become that tall. `extra` is appended to the
/// content stream, and the MediaBox is grown to contain the table.
fn ruled_table(cols: usize, top: f64, extra: &str) -> Vec<u8> {
    let xs: Vec<f64> = (0..=cols).map(|i| 50.0 + 125.0 * i as f64).collect();
    let (left, right) = (xs[0], xs[cols]);
    let mut content = String::from("0.5 w\n");
    for y in [100.0, 200.0, top] {
        content.push_str(&format!("{left} {y} m {right} {y} l S\n"));
    }
    for x in &xs {
        content.push_str(&format!("{x} 100 m {x} {top} l S\n"));
    }
    for (i, x) in xs[..cols].iter().enumerate() {
        let x = x + 10.0;
        content.push_str(&format!("BT /F1 10 Tf {x} 150 Td (A{i}) Tj ET\n"));
        content.push_str(&format!("BT /F1 10 Tf {x} 250 Td (B{i}) Tj ET\n"));
    }
    content.push_str(extra);
    let media_top = (top + 100.0).max(792.0);
    let media_right = (right + 50.0).max(612.0);
    let bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {media_right} {media_top}] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        format!("<< /Length {} >>\nstream\n{content}endstream", content.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    let n = bodies.len() + 1;
    out.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

/// Peak heap growth, in bytes, while extracting page 0, and its text.
fn extract(bytes: Vec<u8>) -> (usize, String) {
    let doc = PdfDocument::from_bytes(bytes).expect("fixture parses");
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let text = doc.extract_text(0).expect("page extracts");
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    drop(doc);
    (peak, text)
}

fn assert_cells_present(text: &str, cols: usize) {
    for i in 0..cols {
        assert!(text.contains(&format!("A{i}")), "A{i} missing from {text:?}");
        assert!(text.contains(&format!("B{i}")), "B{i} missing from {text:?}");
    }
}

/// A ruling 1e8 pt up makes the upper cells millions of bands tall.
#[test]
fn test_a_far_ruling_does_not_grow_the_cell_index() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (near_peak, near) = extract(ruled_table(2, 300.0, ""));
    let (far_peak, far) = extract(ruled_table(2, 1e8, ""));
    assert_cells_present(&far, 2);
    assert_eq!(far, near, "a taller top row changed the extracted text");
    let mib = far_peak as f64 / (1024.0 * 1024.0);
    eprintln!("peak heap: {near_peak} bytes normal, {far_peak} bytes with a far ruling");
    assert!(far_peak < 32 << 20, "peak heap with a far ruling was {mib:.1} MiB");
}

/// A span beside the far ruling makes the probed window cover the whole cell.
#[test]
fn test_a_span_beside_a_far_ruling_does_not_grow_the_cell_index() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let top = 2e8;
    let far_span = format!("BT /F1 10 Tf 60 {} Td (far) Tj ET\n", top - 50.0);
    let (peak, text) = extract(ruled_table(4, top, &far_span));
    assert_cells_present(&text, 4);
    assert!(text.contains("far"), "far span missing from {text:?}");
    let mib = peak as f64 / (1024.0 * 1024.0);
    eprintln!("peak heap with a span beside a far ruling: {mib:.1} MiB");
    assert!(peak < 64 << 20, "peak heap with a span beside a far ruling was {mib:.1} MiB");
}
