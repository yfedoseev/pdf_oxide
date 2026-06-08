//! Hybrid page = a real native text layer **and** a raster image that
//! itself contains text. #517's `PageKind::ImageText` is documented as
//! a hybrid (native text plus region OCR of image-borne text), and
//! `extract_text_with_ocr`'s HybridPage branch literally says "merge
//! both sources".
//!
//! Defect this pins (found by an external downstream consumer running
//! the full v0.3.50/v0.3.51 surface): the HybridPage branch did
//! `if ocr_len > native_len * 2 { ocr } else { native }` — an
//! either/or that **silently discarded the in-image text** whenever
//! the native layer was longer. On a PDF with a text paragraph plus a
//! screenshot/figure containing a caption you got the paragraph and
//! lost the caption — you could NOT "extract both". And
//! `AutoExtractor::extract_page` emitted a single region whose
//! `source` was derived from `classify_page().kind` (so native text
//! came back labelled `source = Ocr`), violating the per-region
//! provenance contract.
//!
//! The fixture `auto_hybrid_text_image_en.pdf` is a born-digital page:
//! a native sentence ("... Confidential Quarterly Memo 2026 ...") plus
//! an embedded raster image of the line "OCR fidelity test hello world
//! 2024". The honest invariant (OCR is fuzzy): the assembled text
//! carries the native sentence AND at least one reliably-recovered
//! token from the image, and the per-region provenance is truthful.
//!
//! Model-gated: skips cleanly when models are not provisioned (the
//! no-model default lane stays green); the CI OCR lane provisions them
//! and runs it for real. Requires the `ocr` feature.
#![cfg(feature = "ocr")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::auto::{AutoExtractor, ExtractSource, PageKind};

fn models_present() -> bool {
    let d = AutoExtractor::model_cache_dir();
    d.join("det.onnx").is_file() && d.join("rec.onnx").is_file() && d.join("en_dict.txt").is_file()
}

const FIX: &str = "tests/fixtures/ocr/auto_hybrid_text_image_en.pdf";
const NATIVE: &str = "CONFIDENTIAL QUARTERLY MEMO 2026";
const IMG_TOKENS: &[&str] = &["OCR", "TEST", "HELLO", "WORLD", "2024", "FIDELITY"];

#[test]
fn auto_hybrid_recovers_native_and_in_image_text_520() {
    if !models_present() {
        eprintln!(
            "SKIP: OCR models absent at {} — run scripts/setup_ocr_models.sh \
             or set PDF_OXIDE_MODEL_DIR (CI OCR lane provisions these).",
            AutoExtractor::model_cache_dir().display()
        );
        return;
    }
    let doc = PdfDocument::open(FIX).expect("open hybrid text+image pdf");

    // Native text alone must still be extractable directly.
    let native_only = doc.extract_text(0).unwrap_or_default().to_uppercase();
    assert!(native_only.contains(NATIVE), "native text layer must extract: {native_only:?}");

    // Hybrid classification.
    let cls = doc.classify_page(0).expect("classify_page");
    assert!(
        matches!(cls.kind, PageKind::ImageText | PageKind::Mixed),
        "native text + image-with-text must classify ImageText/Mixed, got {:?}",
        cls.kind
    );

    let pe = AutoExtractor::new()
        .extract_page(&doc, 0)
        .expect("extract_page");
    let up = pe.text.to_uppercase();

    // (1) Both sources present in the assembled page text.
    assert!(
        up.contains(NATIVE),
        "assembled text must keep the native sentence; got {:?}",
        pe.text
    );
    let recovered: Vec<&str> = IMG_TOKENS
        .iter()
        .copied()
        .filter(|t| up.contains(t))
        .collect();
    assert!(
        !recovered.is_empty(),
        "assembled text must include OCR-recovered in-image text \
         (one of {IMG_TOKENS:?}); the HybridPage merge dropped it. got {:?}",
        pe.text
    );
    assert!(pe.ocr_used, "ocr_used must be true on a hybrid page that was OCR'd");

    // (2) Truthful per-region provenance: a NativeText region carrying
    // the native sentence AND an Ocr region carrying recovered image
    // text — never native text mislabelled `source = Ocr`.
    let native_region = pe
        .regions
        .iter()
        .any(|r| r.source == ExtractSource::NativeText && r.text.to_uppercase().contains(NATIVE));
    let ocr_region = pe.regions.iter().any(|r| {
        r.source == ExtractSource::Ocr
            && IMG_TOKENS.iter().any(|t| r.text.to_uppercase().contains(t))
    });
    assert!(
        native_region,
        "expected a NativeText-source region with the native sentence; regions={:?}",
        pe.regions
            .iter()
            .map(|r| (r.source, r.text.chars().take(40).collect::<String>()))
            .collect::<Vec<_>>()
    );
    assert!(
        ocr_region,
        "expected an Ocr-source region with recovered image text; regions={:?}",
        pe.regions
            .iter()
            .map(|r| (r.source, r.text.chars().take(40).collect::<String>()))
            .collect::<Vec<_>>()
    );
    let no_native_as_ocr = !pe.regions.iter().any(|r| {
        r.source == ExtractSource::Ocr
            && r.text.to_uppercase().contains(NATIVE)
            && !IMG_TOKENS.iter().any(|t| r.text.to_uppercase().contains(t))
    });
    assert!(
        no_native_as_ocr,
        "native text must NOT be mislabelled source=Ocr; regions={:?}",
        pe.regions
            .iter()
            .map(|r| (r.source, r.text.len()))
            .collect::<Vec<_>>()
    );

    // (3) The string one-shots must also surface BOTH.
    let one_shot = doc
        .extract_text_auto(0)
        .expect("extract_text_auto")
        .to_uppercase();
    assert!(
        one_shot.contains(NATIVE) && IMG_TOKENS.iter().any(|t| one_shot.contains(t)),
        "extract_text_auto must surface native AND image text; got {one_shot:?}"
    );

    // The direct OCR escape hatch must merge too (not either/or).
    let merged = pdf_oxide::ocr::extract_text_with_ocr(
        &doc,
        0,
        Some(
            &pdf_oxide::ocr::OcrEngine::new(
                AutoExtractor::model_cache_dir().join("det.onnx"),
                AutoExtractor::model_cache_dir().join("rec.onnx"),
                AutoExtractor::model_cache_dir().join("en_dict.txt"),
                pdf_oxide::ocr::OcrConfig::default(),
            )
            .expect("engine"),
        ),
        pdf_oxide::ocr::OcrExtractOptions::default(),
    )
    .expect("extract_text_with_ocr")
    .to_uppercase();
    assert!(
        merged.contains(NATIVE) && IMG_TOKENS.iter().any(|t| merged.contains(t)),
        "extract_text_with_ocr HybridPage must MERGE native+image, not pick one; got {merged:?}"
    );
}
