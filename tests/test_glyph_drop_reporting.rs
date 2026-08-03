//! A dropped glyph must leave a trace in the log.
//!
//! Four points in the text rasterizer swallow a glyph that produced no outline
//! while the cursor keeps advancing, so the page renders with an invisible gap.
//! A consumer cannot tell that gap from genuine whitespace: OCR over such a page
//! transcribes the gap faithfully and the loss reaches whatever consumes the
//! text next.
//!
//! This pins the reporting, not the rendering: the raster output is expected to
//! be byte-identical, and only the log gains a line.

#![cfg(feature = "rendering")]

use std::sync::{Mutex, OnceLock};

use log::{Level, Metadata, Record};

/// Warnings captured during a render.
static CAPTURED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

struct CaptureWarnings;

impl log::Log for CaptureWarnings {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            captured().lock().expect("capture lock").push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

fn captured() -> &'static Mutex<Vec<String>> {
    CAPTURED.get_or_init(|| Mutex::new(Vec::new()))
}

/// Install the capturing logger once for this test binary.
fn install_capture() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let _ = log::set_boxed_logger(Box::new(CaptureWarnings));
        log::set_max_level(log::LevelFilter::Warn);
    });
}

/// A page asking for a glyph the font cannot supply: an embedded TrueType font
/// whose `cmap` maps nothing, so every code resolves to GID 0 and, being
/// non-whitespace, is never painted.
fn unmapped_glyph_pdf() -> Vec<u8> {
    // TODO: the fixture that provably reaches the drop is still being built.
    // Rendering needs a real font program, so this is not a two-line content
    // stream like the extraction fixtures.
    Vec::new()
}

#[test]
#[ignore = "fixture that drives a glyph drop is still being built; see PR body"]
fn a_dropped_glyph_is_reported() {
    install_capture();
    let pdf = unmapped_glyph_pdf();
    let doc = pdf_oxide::document::PdfDocument::from_bytes(pdf).expect("parse fixture");
    let _ = doc.render_page(0, 1.0);

    let warnings = captured().lock().expect("capture lock");
    assert!(
        warnings.iter().any(|w| w.contains("glyph")),
        "a glyph was dropped with no warning: {warnings:?}"
    );
}
