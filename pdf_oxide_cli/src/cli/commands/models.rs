//! `pdf-oxide models <prefetch|manifest>` — build-time OCR/layout
//! model provisioning (#513/#517). `prefetch` is the documented
//! Dockerfile `RUN`; `manifest` prints the air-gapped JSON manifest.

use crate::cli::args::ModelsAction;
use pdf_oxide::extractors::{AutoExtractor, OcrLanguage};

pub fn run(action: &ModelsAction) -> pdf_oxide::Result<()> {
    match action {
        ModelsAction::Prefetch { languages, all } => {
            // `--all` (or `-l all`) = every supported language — the
            // Docker/CI "bake everything" build case.
            let want_all = *all || languages.iter().any(|l| l.eq_ignore_ascii_case("all"));
            let langs: Vec<OcrLanguage> = if want_all {
                OcrLanguage::ALL.to_vec()
            } else if languages.is_empty() {
                vec![OcrLanguage::English]
            } else {
                let mut v = Vec::new();
                for l in languages {
                    match OcrLanguage::from_code(l) {
                        Some(lang) => v.push(lang),
                        None => {
                            eprintln!(
                                "warning: unknown OCR language '{l}' — skipped \
                                 (see `pdf-oxide models manifest` for supported codes)"
                            );
                        },
                    }
                }
                if v.is_empty() {
                    v.push(OcrLanguage::English);
                }
                v
            };
            if !AutoExtractor::prefetch_available() {
                eprintln!(
                    "warning: this `pdf-oxide` was built WITHOUT the `ocr` \
                     feature — no models will be downloaded (only the cache \
                     dir is created). Rebuild/install with `--features ocr` \
                     to actually fetch models."
                );
            }
            let dir = AutoExtractor::prefetch_models(&langs)?;
            if AutoExtractor::prefetch_available() {
                println!(
                    "models cache: {} ({} language pack(s) provisioned)",
                    dir.display(),
                    langs.len()
                );
            } else {
                println!(
                    "models cache dir ensured: {} (no fetch — see warning above)",
                    dir.display()
                );
            }
            Ok(())
        },
        ModelsAction::Manifest => {
            println!("{}", AutoExtractor::model_manifest());
            Ok(())
        },
    }
}
