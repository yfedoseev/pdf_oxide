//! `compliance` — stub for v0.3.53. To be filled in across Phases 2–5 per the
//! task plan in `docs/releases/plans/v0.3.53/feature-NNN-java-binding.md`.
//!
//! Real implementation will hold `#[no_mangle] pub extern "system" fn
//! Java_fyi_oxide_pdf_<Class>_*` entries calling through to the
//! existing pdf_oxide C ABI in `src/ffi.rs`. Every entry goes through
//! the jni-rs 0.22 panic-barrier per `00-common-foundation.md` §2.
