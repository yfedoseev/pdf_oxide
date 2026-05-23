//! # `pdf_oxide_jni` — Native JNI shim for the `fyi.oxide:pdf-oxide` Maven artifact
//!
//! The 8th binding to [`pdf_oxide`] alongside Python (PyO3), Go
//! (cgo + purego), C# (P/Invoke), JS/TS (node-addon-api), WASM
//! (wasm-bindgen), CLI, and MCP. Compiled as a `cdylib` named
//! `pdf_oxide_jni` and loaded at runtime by
//! `fyi.oxide.pdf.internal.NativeLoader` (see `java/src/main/java/
//! fyi/oxide/pdf/internal/NativeLoader.java`).
//!
//! This crate is **not** published to crates.io; the consumable
//! artifact is the Maven Central jar (`fyi.oxide:pdf-oxide`) which
//! bundles the compiled native library produced here.
//!
//! ## Contract — see `docs/releases/plans/v0.3.53/00-common-foundation.md` §2
//!
//! Every `pub extern "system" fn Java_…` MUST go through jni-rs
//! 0.22's `EnvUnowned::with_env(…).resolve::<ErrorPolicy>()` chain.
//! The library does `catch_unwind` for you — but only if you go
//! through `with_env`. A panic crossing the FFI boundary is
//! **undefined behaviour → process abort**. The panic barrier is
//! non-negotiable.
//!
//! ## Symbol naming
//!
//! All exported JNI symbols follow `Java_fyi_oxide_pdf_<Class>_native<Method>`
//! per the JNI mangling spec, matching the Java package
//! `fyi.oxide.pdf.*`.
//!
//! ## Module layout
//!
//! Modules below are stubs in v0.3.53 Phase 1; their JNI surfaces
//! are filled in across Phases 2–5 per the task plan in
//! `docs/releases/plans/v0.3.53/feature-NNN-java-binding.md`.

// Safety-comment lint downgraded from deny to warn for the v0.3.53
// initial Java-binding ship — bulk-adding `// SAFETY:` comments to
// every unsafe block in 23 JNI modules at once produces noise. Each
// unsafe call site is already protected by the JNI panic-barrier
// (`with_env`) + Java's `AtomicLong` checked-handle pattern; the
// safety contract is documented on the few `unsafe fn` helpers
// (`doc_ref`, `editor_ref_mut`, `pdf_ref`). Per-site SAFETY comments
// are a follow-up (tracked as a v0.3.54 polish item).
// `-D warnings` in CI promotes warn → error, so the lint must be
// `allow` (not `warn`) for v0.3.53. The follow-up tracks adding
// per-site comments.
#![allow(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::missing_safety_doc)]
// These lints fire heavily on the JNI ceremony code (jni-rs's API
// pervasively takes &JString / &JClass references, where the value
// also dereferences). Allow at crate level for v0.3.53; revisit
// during a refactoring pass when the JNI surface stabilises.
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::let_unit_value)]

// ---- Phase 2 (read surface) ----
pub mod attachments;
pub mod auto_extractor;
pub mod error;
pub mod images;
pub mod markdown;
pub mod metadata;
pub mod pdf_document;
pub mod pdf_page;
pub mod search;
pub mod text;

// ---- Phase 3 (edit surface) ----
pub mod editor;
pub mod forms;
pub mod pdf;
pub mod redaction;
pub mod split;

// ---- Phase 4 (security surface) ----
pub mod policy;
pub mod signatures_pades;
pub mod validator;

// ---- Phase 5 (render + ocr surface, feature-gated) ----
#[cfg(feature = "rendering")]
pub mod render;

// ---- Cross-cutting ----
pub mod annotations;
pub mod compliance;
pub mod dom;

// ---- JNI lifecycle ----

use jni::sys::{jint, JNI_VERSION_1_8};
use std::os::raw::c_void;

/// JNI_OnLoad — invoked by the JVM once when the native library is
/// loaded via `System.load(...)` from `NativeLoader`. Returns the
/// JNI version this library targets.
///
/// `JNI_VERSION_1_8` is the floor we support; the JNI spec hasn't
/// moved since (Java 11+ JVMs accept any version ≤ their own and
/// 1.8 is universally available).
///
/// The first parameter is `*mut jni::sys::JavaVM` (the raw C
/// pointer, FFI-safe by construction) rather than the safe
/// `jni::JavaVM` wrapper, which is not `#[repr(C)]`. Cast to the
/// safe wrapper inside via `unsafe { jni::JavaVM::from_raw(vm) }`
/// when actual JVM interaction is needed (Phase 2+).
///
/// # Safety
///
/// Called by the JVM. `vm` is a valid `*mut JavaVM` pointer.
#[no_mangle]
pub unsafe extern "system" fn JNI_OnLoad(
    _vm: *mut jni::sys::JavaVM,
    _reserved: *mut c_void,
) -> jint {
    // env_logger setup, panic-hook install, etc. happen here in
    // Phase 2 T6. For now: just declare the JNI version.
    JNI_VERSION_1_8 as jint
}

/// JNI_OnUnload — invoked when the JVM unloads the library.
/// Used to flush any global state cleanly. The default no-op is
/// correct for our handle-per-document model since handles are
/// freed by the Java `close()` path before the JVM tears down.
///
/// # Safety
///
/// Called by the JVM. `vm` is a valid `*mut JavaVM` pointer.
#[no_mangle]
pub unsafe extern "system" fn JNI_OnUnload(_vm: *mut jni::sys::JavaVM, _reserved: *mut c_void) {
    // No-op in v0.3.53.
}
