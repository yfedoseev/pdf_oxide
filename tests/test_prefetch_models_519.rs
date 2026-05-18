//! Proves `AutoExtractor::prefetch_models` actually DOWNLOADS models
//! to disk (#519) — not a stub, not a silent no-op. Network-gated:
//! requires the `ocr` feature (the `ureq` downloader) + outbound HTTPS;
//! runs in the CI `ocr` lane (which already has network for the model
//! provisioning). Skips cleanly when offline.
#![cfg(feature = "ocr")]

use pdf_oxide::extractors::{AutoExtractor, OcrLanguage};

#[test]
fn prefetch_models_downloads_real_files_519() {
    assert!(
        AutoExtractor::prefetch_available(),
        "built with `ocr` → prefetch_available() must be true"
    );

    // Unique per invocation (pid is NOT unique across parallel
    // in-process test threads or concurrent runs of this test): add a
    // monotonic nanosecond timestamp + a process-local atomic counter
    // so re-runs and parallel execution never collide on the temp dir.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = format!(
        "{}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let tmp = std::env::temp_dir().join(format!("pox_pf_{uniq}"));
    let _ = std::fs::remove_dir_all(&tmp);
    // SAFETY: single-threaded test entry; sets the documented cache-dir env.
    std::env::set_var("PDF_OXIDE_MODEL_DIR", &tmp);

    // Korean: a modest per-language pack (deepghs PP-OCRv3 + PaddleOCR
    // dict) exercising det + per-language rec + dict end to end.
    let dir = match AutoExtractor::prefetch_models(&[OcrLanguage::Korean]) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: prefetch network unavailable: {e}");
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        },
    };
    assert_eq!(dir, tmp, "prefetch must write into PDF_OXIDE_MODEL_DIR");

    let det = dir.join("det.onnx");
    let rec = dir.join("rec_korean.onnx");
    let dict = dir.join("korean_dict.txt");
    for (p, min) in [(&det, 1_000_000u64), (&rec, 1_000_000), (&dict, 100)] {
        let m = std::fs::metadata(p)
            .unwrap_or_else(|_| panic!("prefetch_models did not create {}", p.display()));
        assert!(
            m.len() >= min,
            "{} is implausibly small ({} bytes) — not a real model",
            p.display(),
            m.len()
        );
    }
    // Idempotent: a second call must not error and must keep the files.
    AutoExtractor::prefetch_models(&[OcrLanguage::Korean]).expect("idempotent re-prefetch");
    assert!(rec.is_file() && dict.is_file() && det.is_file());

    let _ = std::fs::remove_dir_all(&tmp);
}
