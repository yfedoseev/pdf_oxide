//! Per-call extraction signal — what fired during a single text-extraction
//! invocation. v0.3.56 additive surface for #559, #563, #571, #574, #562.
//!
//! Returned alongside the existing return values by the new `*_status`
//! companion accessors (`extract_text_status`, `to_plain_text_status`,
//! `extract_words_status`, etc.). The existing accessors keep their
//! original return shapes — this type is purely additive.
//!
//! **Naming note (v0.3.56)**: the v0.3.56 plan called this type
//! `ExtractionStatus`, but v0.3.51 already exposes an
//! `extractors::auto::ExtractionStatus` (page-level Complete /
//! PartialSuccess / NoTextRecovered for the AutoExtractor surface from
//! #517). Renaming this to `ExtractionSignal` keeps both types public and
//! preserves additive back-compat.
//!
//! See `docs/releases/plans/v0.3.56/api-design.md` §1 for the design.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Per-call extraction signal returned by the `*_status` companion
/// accessors. `Ok` means the operation completed without any caveat;
/// every other variant carries enough structured detail for a caller
/// to either route to OCR, raise a typed exception, or warn the user.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExtractionSignal {
    /// Operation completed cleanly; the returned text is authoritative.
    #[default]
    Ok,

    /// The content-stream parser hit `MAX_OPERATORS` (default 1,000,000)
    /// and stopped emitting glyphs. The returned text is everything up to
    /// that point. `at_op` is the operator-index at which truncation fired.
    /// Fix: `ParserOptions { max_ops_per_stream: None }`. Closes #559.
    Truncated {
        /// Operator index at which truncation fired.
        at_op: usize,
    },

    /// The page has no text layer (no `/Font` resources used, no `Tj` /
    /// `TJ` / `'` / `"` operators observed). The returned text is empty.
    /// Route to OCR via `extract_text_ocr(page, engine)`. Closes #563.
    NoTextLayer,

    /// `count` glyphs on the page mapped to `U+FFFD` (REPLACEMENT
    /// CHARACTER) because the font has neither ToUnicode nor a usable
    /// AGL fallback. The returned text includes those U+FFFD chars
    /// (v0.3.56 stops filtering them silently). Caller can decide whether
    /// to keep, drop, or OCR. Closes #571.
    UnmappedGlyphs {
        /// Number of `U+FFFD` chars in the returned text.
        count: usize,
    },

    /// OCR was requested but the backend is unavailable.
    /// Closes #569, #573, #574.
    OcrUnavailable {
        /// The reason OCR is unavailable.
        reason: OcrUnavailableReason,
    },

    /// Document was encrypted and the caller has not yet authenticated.
    /// Body operations on the document return this signal (with empty
    /// text) until `doc.authenticate(password)` succeeds. Closes #562.
    PasswordRequired,

    /// A composite signal: multiple non-Ok signals fired on the same
    /// call. The vector preserves order of occurrence. Used when, for
    /// example, a page is both truncated AND has unmapped glyphs.
    Multiple(Vec<ExtractionSignal>),
}

/// Reason that OCR is unavailable on the current call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OcrUnavailableReason {
    /// `libonnxruntime.so` / `.dylib` / `.dll` failed to load via
    /// `dlopen` / `LoadLibrary`.
    DylibMissing,
    /// OCR feature is compile-time disabled in this build.
    FeatureDisabled,
    /// No `OcrEngine` was supplied and the caller invoked
    /// `extract_text_ocr_only` (which requires an explicit engine).
    /// `extract_text_auto` does NOT raise this — it silently degrades.
    EngineNotProvided,
    /// `ort::Session::run` or `Session::builder().commit()` returned
    /// an error.
    ModelLoadFailed {
        /// Underlying error string from the ORT crate.
        detail: String,
    },
    /// ORT init panicked (e.g. corrupted Mutex from a prior failed init).
    /// Captured by `std::panic::catch_unwind`.
    InitPanicked {
        /// Panic payload as a string.
        detail: String,
    },
}

impl ExtractionSignal {
    /// True when no caveat fired. Convenience for `== Self::Ok`.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// True when the signal indicates the page has no recoverable text
    /// from the embedded text layer (and so OCR is the recourse). Covers
    /// `NoTextLayer` only — `UnmappedGlyphs` is NOT a should-OCR case
    /// because OCR is unlikely to do better than the existing glyph
    /// names.
    pub fn should_ocr(&self) -> bool {
        match self {
            Self::NoTextLayer => true,
            Self::Multiple(children) => children.iter().any(|c| c.should_ocr()),
            _ => false,
        }
    }

    /// Push another signal into a `Multiple` or upgrade a scalar to a
    /// `Multiple`. Internal helper used when more than one signal fires
    /// on the same call.
    pub fn push(&mut self, other: ExtractionSignal) {
        if matches!(other, Self::Ok) {
            return;
        }
        match self {
            Self::Ok => *self = other,
            Self::Multiple(v) => v.push(other),
            _ => {
                let first = std::mem::replace(self, Self::Ok);
                *self = Self::Multiple(vec![first, other]);
            },
        }
    }
}

impl OcrUnavailableReason {
    /// Stable string identifier (matches the Python-binding string form).
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::DylibMissing => "dylib_missing",
            Self::FeatureDisabled => "feature_disabled",
            Self::EngineNotProvided => "engine_not_provided",
            Self::ModelLoadFailed { .. } => "model_load_failed",
            Self::InitPanicked { .. } => "init_panicked",
        }
    }

    /// Detail string (empty for variants without detail).
    pub fn detail(&self) -> String {
        match self {
            Self::ModelLoadFailed { detail } | Self::InitPanicked { detail } => detail.clone(),
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_is_ok() {
        assert!(ExtractionSignal::Ok.is_ok());
        assert!(!ExtractionSignal::NoTextLayer.is_ok());
    }

    #[test]
    fn no_text_layer_should_ocr() {
        assert!(ExtractionSignal::NoTextLayer.should_ocr());
        assert!(!ExtractionSignal::Ok.should_ocr());
        assert!(!ExtractionSignal::Truncated { at_op: 1000 }.should_ocr());
        assert!(!ExtractionSignal::UnmappedGlyphs { count: 3 }.should_ocr());
    }

    #[test]
    fn push_upgrades_ok_to_other() {
        let mut s = ExtractionSignal::Ok;
        s.push(ExtractionSignal::Truncated { at_op: 1000 });
        assert_eq!(s, ExtractionSignal::Truncated { at_op: 1000 });
    }

    #[test]
    fn push_skips_ok() {
        let mut s = ExtractionSignal::Truncated { at_op: 1000 };
        s.push(ExtractionSignal::Ok);
        assert_eq!(s, ExtractionSignal::Truncated { at_op: 1000 });
    }

    #[test]
    fn push_two_scalars_creates_multiple() {
        let mut s = ExtractionSignal::Truncated { at_op: 1000 };
        s.push(ExtractionSignal::UnmappedGlyphs { count: 3 });
        match s {
            ExtractionSignal::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], ExtractionSignal::Truncated { at_op: 1000 });
                assert_eq!(v[1], ExtractionSignal::UnmappedGlyphs { count: 3 });
            },
            _ => panic!("expected Multiple"),
        }
    }

    #[test]
    fn push_into_multiple_appends() {
        let mut s = ExtractionSignal::Multiple(vec![ExtractionSignal::NoTextLayer]);
        s.push(ExtractionSignal::Truncated { at_op: 500 });
        match s {
            ExtractionSignal::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[1], ExtractionSignal::Truncated { at_op: 500 });
            },
            _ => panic!("expected Multiple"),
        }
    }

    #[test]
    fn multiple_with_no_text_layer_should_ocr() {
        let s = ExtractionSignal::Multiple(vec![
            ExtractionSignal::NoTextLayer,
            ExtractionSignal::UnmappedGlyphs { count: 3 },
        ]);
        assert!(s.should_ocr());
    }

    #[test]
    fn ocr_unavailable_kind_str_stable() {
        assert_eq!(OcrUnavailableReason::DylibMissing.kind_str(), "dylib_missing");
        assert_eq!(OcrUnavailableReason::EngineNotProvided.kind_str(), "engine_not_provided");
        assert_eq!(
            OcrUnavailableReason::ModelLoadFailed { detail: "x".into() }.kind_str(),
            "model_load_failed"
        );
    }

    #[test]
    fn ocr_unavailable_detail_passthrough() {
        let r = OcrUnavailableReason::ModelLoadFailed {
            detail: "missing.onnx".into(),
        };
        assert_eq!(r.detail(), "missing.onnx");
        assert_eq!(OcrUnavailableReason::DylibMissing.detail(), "");
    }

    #[test]
    fn signal_serializes_to_snake_case_json() {
        let s = ExtractionSignal::Truncated { at_op: 1_000_000 };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"truncated\""));
        assert!(json.contains("\"at_op\":1000000"));
    }
}
