//! Closes the #519 gap: AutoExtractor must actually extract text from
//! an IMAGE inside a PDF (OCR), not silently fall back to native.
//!
//! Root cause that was fixed: `route()` passed `None` for the OCR
//! engine, so `extract_text_with_ocr` never OCR'd — the Auto surface
//! always degraded to native text even with models + the `ocr`
//! feature. `route()` now loads an engine from
//! `AutoExtractor::model_cache_dir()` (the `prefetch_models` /
//! `scripts/setup_ocr_models.sh` contract).
//!
//! Model-gated: skips cleanly when models are not provisioned (so the
//! no-model default lane stays green); the CI OCR lane provisions them
//! via `setup_ocr_models.sh` + `PDF_OXIDE_MODEL_DIR`, where this runs
//! for real. Requires the `ocr` feature.
#![cfg(feature = "ocr")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::auto::{AutoExtractor, ExtractSource, PageKind};

fn models_present() -> bool {
    let d = AutoExtractor::model_cache_dir();
    d.join("det.onnx").is_file() && d.join("rec.onnx").is_file() && d.join("en_dict.txt").is_file()
}

const FIX: &str = "tests/fixtures/ocr/auto_image_text_en.pdf";
// The committed fixture is an image-only PDF of the rendered line
// "OCR fidelity test hello world 2024". OCR is inherently fuzzy and
// the freely-downloadable community models are mediocre at
// recognition (e.g. "fidelity"→"tdenfy", "hello"→"neno"), so the
// honest invariant is: the auto surface *genuinely OCR'd the image*
// (source=Ocr, ocr_used) and recovered substantive, reliably-mapped
// text — NOT a perfect transcription.
const WANT: &[&str] = &["OCR", "TEST"];

#[test]
fn auto_extracts_text_from_image_via_ocr_519() {
    if !models_present() {
        eprintln!(
            "SKIP: OCR models absent at {} — run scripts/setup_ocr_models.sh \
             or set PDF_OXIDE_MODEL_DIR (CI OCR lane provisions these).",
            AutoExtractor::model_cache_dir().display()
        );
        return;
    }
    let doc = PdfDocument::open(FIX).expect("open image-of-text pdf");

    // It is an image-only page → the classifier must route it to OCR.
    let cls = doc.classify_page(0).expect("classify_page");
    assert_eq!(
        cls.kind,
        PageKind::Scanned,
        "image-only page must classify Scanned, got {:?}",
        cls.kind
    );

    let pe = AutoExtractor::new()
        .extract_page(&doc, 0)
        .expect("extract_page");
    // Structural: OCR genuinely ran (this is the gap that was fixed —
    // route() used to pass a None engine so OCR never executed).
    assert_eq!(
        pe.regions[0].source,
        ExtractSource::Ocr,
        "source must be Ocr (OCR genuinely ran), got {:?} text={:?}",
        pe.regions[0].source,
        pe.text
    );
    assert!(pe.ocr_used, "ocr_used must be true when OCR produced the text");
    assert!(!pe.text.trim().is_empty(), "OCR must recover non-empty text from the image");
    // Content: at least one reliably-recovered token is present.
    let up = pe.text.to_uppercase();
    assert!(
        WANT.iter().any(|w| up.contains(*w)),
        "OCR-recovered text must contain a reliable token {WANT:?}; got {:?}",
        pe.text
    );

    // The one-shot path must also OCR-recover substantive text.
    let auto = doc.extract_text_auto(0).expect("extract_text_auto");
    assert!(
        !auto.trim().is_empty() && WANT.iter().any(|w| auto.to_uppercase().contains(*w)),
        "extract_text_auto must also OCR-recover the image text: {auto:?}"
    );
}
