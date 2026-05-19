//! #524 — cross-target (real wasm runtime) tests for the WASM OCR
//! surface. Model-free by design: they assert the *contract* the
//! browser/Deno/edge JS recipe depends on (manifest shape, clean error
//! paths across the wasm boundary) without fetching tens of MB of
//! models, so they run fast in CI on every wasm target.
//!
//! Run (Node):
//!   RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
//!     wasm-pack test --node -- --no-default-features --features wasm-ocr
//!
//! Gated to the `wasm-ocr` feature directly (which is the documented
//! build flag) so it can only compile when both `wasm` and
//! `ocr-tract` are on — the prior `feature = "ocr-tract"` cfg alone
//! would let this file try to compile under
//! `--features ocr-tract --target wasm32` *without* `wasm`, where
//! `pdf_oxide::wasm` doesn't exist and the imports below fail.
//! (#523 Copilot review.)
#![cfg(all(target_arch = "wasm32", feature = "wasm-ocr"))]

use wasm_bindgen_test::*;

use pdf_oxide::wasm::{model_manifest, prefetch_available, WasmOcrEngine};

// Runs in Node (wasm-pack test --node); no `run_in_browser`.

/// The host drives model delivery off `modelManifest()`; its shape is a
/// load-bearing contract for the documented JS recipe.
#[wasm_bindgen_test]
fn model_manifest_has_detector_and_languages() {
    let m = model_manifest();
    assert!(m.contains("\"detector\""), "manifest missing detector: {m}");
    assert!(m.contains("\"languages\""), "manifest missing languages: {m}");
    assert!(m.contains("http"), "manifest missing model URLs: {m}");
}

/// WASM cannot download to a cache; provisioning is host-side. This
/// must stay `false` so callers know to supply bytes themselves.
#[wasm_bindgen_test]
fn prefetch_is_host_side_only() {
    assert!(!prefetch_available());
}

/// Invalid model bytes must surface as a clean JS error across the
/// wasm boundary — never a panic/`unreachable` (which aborts the whole
/// module in a browser). Proves the error path actually works in a
/// real wasm runtime, not just on native.
#[wasm_bindgen_test]
fn ocr_engine_rejects_garbage_models_without_panicking() {
    let not_onnx = b"this is definitely not an ONNX protobuf";
    let res = WasmOcrEngine::new(not_onnx, not_onnx, "a\nb\n", None);
    assert!(res.is_err(), "expected a clean Err for non-ONNX bytes, got Ok");
}
