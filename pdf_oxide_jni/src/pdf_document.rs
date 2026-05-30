//! JNI surface for `fyi.oxide.pdf.PdfDocument`.
//!
//! Implements the read-side entry points: open / close / pageCount /
//! extractText. Bindings against [`pdf_oxide::PdfDocument`] directly
//! (no C-ABI middleman — Python/WASM bindings use the same pattern;
//! Go/C# go through the C ABI because their FFI mechanisms require
//! `extern "C"`).
//!
//! ## Handle lifecycle
//!
//! - `nativeOpenPath` / `nativeOpenBytes` allocate a `Box<PdfDocument>`,
//!   leak it via `Box::into_raw`, and return the raw pointer cast to
//!   `jlong`. The Java side stores this in a `volatile long handle`
//!   field.
//! - `nativeClose` reclaims the `Box` via `Box::from_raw` and drops
//!   it. The Java side then zeroes its handle field — subsequent
//!   accesses go through `checkHandle()` and throw
//!   `PdfInvalidStateException`. Idempotent close on the Java side
//!   prevents double-free.
//!
//! ## Panic barrier
//!
//! Every entry-point wraps its body in [`EnvUnowned::with_env`] so
//! panics never cross the FFI boundary. Per
//! `docs/releases/plans/v0.3.53/00-common-foundation.md` §2 this is
//! non-negotiable.

use std::path::PathBuf;

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jint, jlong};
use jni::EnvUnowned;
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// Reify a handle (jlong) as a borrowed `&PdfDocument`. The Java side
/// guarantees the handle is non-zero (it calls `checkHandle()` before
/// every native call); we still assert.
///
/// # Safety
///
/// `handle` must be a valid pointer returned by `nativeOpen*` and not
/// yet freed. The Java side's `volatile long handle` + idempotent
/// `close()` enforces this; null handles are caught here as a defense.
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: PdfDocument handle was 0");
    // SAFETY: caller guarantees `handle` points to a leaked Box<PdfDocument>.
    unsafe { &*(handle as *const PdfDocument) }
}

// ──────────────────────────── open(path) ───────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeOpenPath` — open from filesystem path.
///
/// # Safety
///
/// JVM-invoked. Receives an FFI-safe `EnvUnowned` (jni 0.22) which
/// `with_env` upgrades to a safe `Env`. Returns the leaked
/// `Box<PdfDocument>` pointer as `jlong`, or 0 on error (with a Java
/// exception thrown).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeOpenPath<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    path: JString<'local>,
) -> jlong {
    env.with_env(|env| -> Result<jlong, JniError> {
        // jni 0.22: `Env::get_string` is deprecated in favour of
        // `JString::try_to_string(env)` (decodes modified UTF-8 →
        // standard UTF-8 String).
        let path_str: String = path.try_to_string(env)?;
        let path_buf = PathBuf::from(path_str);
        match PdfDocument::open(&path_buf) {
            Ok(doc) => Ok(Box::into_raw(Box::new(doc)) as jlong),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0) // unreachable — throw_pdf returns Err
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────── open(bytes) ───────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeOpenBytes` — open from in-memory bytes.
///
/// # Safety
///
/// JVM-invoked. The byte[] is copied into a Rust `Vec<u8>` via
/// `convert_byte_array` (the JNI region access is bounded; no critical
/// section held across allocations).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeOpenBytes<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    bytes: JByteArray<'local>,
) -> jlong {
    env.with_env(|env| -> Result<jlong, JniError> {
        // convert_byte_array copies the array region; no critical
        // pin. Acceptable for v0.3.53 — direct ByteBuffer zero-copy
        // is a future enhancement (api-design.md §12).
        let vec: Vec<u8> = env.convert_byte_array(&bytes)?;
        match PdfDocument::from_bytes(vec) {
            Ok(doc) => Ok(Box::into_raw(Box::new(doc)) as jlong),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────────── close ─────────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeClose` — free the native handle.
///
/// The Java side guarantees this is called at most once per handle
/// (via the `volatile long handle` field + idempotent close + cleaner
/// disarm). Null/zero handles are a no-op (defensive).
///
/// # Safety
///
/// JVM-invoked. `handle` must be a valid pointer returned by
/// `nativeOpenPath` / `nativeOpenBytes` that has not yet been freed.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeClose<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    let _ = env
        .with_env(|_env| -> Result<(), JniError> {
            if handle != 0 {
                // SAFETY: handle was returned by nativeOpen* and not yet freed.
                unsafe {
                    drop(Box::from_raw(handle as *mut PdfDocument));
                }
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}

// ─────────────────────────── authenticate ─────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeAuthenticate` — provide a
/// password for an encrypted PDF.
///
/// Returns `true` if authentication succeeded (or the PDF is not
/// encrypted), `false` on wrong password. Wraps
/// [`pdf_oxide::PdfDocument::authenticate`] — see its docs for the
/// invalidate-cache behaviour after a successful auth.
///
/// # Safety
///
/// JVM-invoked. `handle` is a valid PdfDocument pointer; `password`
/// is a Java byte[].
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeAuthenticate<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    password: JByteArray<'local>,
) -> jni::sys::jboolean {
    env.with_env(|env| -> Result<jni::sys::jboolean, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let pw: Vec<u8> = env.convert_byte_array(&password)?;
        match doc.authenticate(&pw) {
            Ok(true) => Ok(jni::sys::JNI_TRUE),
            Ok(false) => Ok(jni::sys::JNI_FALSE),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(jni::sys::JNI_FALSE)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ──────────────────────────── pageCount ────────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativePageCount` — return page count as jint.
///
/// # Safety
///
/// JVM-invoked. `handle` must be a valid (non-zero) PdfDocument pointer.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativePageCount<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        // SAFETY: Java side asserted handle != 0 before calling.
        let doc = unsafe { doc_ref(handle) };
        match doc.page_count() {
            Ok(n) => Ok(n as jint),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(-1)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ──────────────────────── extractTextAuto ─────────────────────────────────

/// `nativeExtractTextAuto` — v0.3.51 #517 graceful auto extraction.
/// Wraps [`pdf_oxide::PdfDocument::extract_text_auto`] which routes
/// text-vs-OCR per-page with graceful fallback when OCR is unavailable.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeExtractTextAuto<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        if page_index < 0 {
            let class = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
            let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
            let _ = env.throw_new(&class, &msg);
            return Err(JniError::JavaException);
        }
        match doc.extract_text_auto(page_index as usize) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(jni::objects::JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────── producer / creator ──────────────────────────────

/// `nativeProducer` — Document Info `/Producer` (returns null if absent).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeProducer<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.document_producer() {
            Some(s) => Ok(env.new_string(s)?),
            None => Ok(jni::objects::JString::default()),
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeCreator` — Document Info `/Creator` (returns null if absent).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeCreator<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.document_creator() {
            Some(s) => Ok(env.new_string(s)?),
            None => Ok(jni::objects::JString::default()),
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────── extractText ───────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeExtractText` — extract text from a page.
///
/// # Safety
///
/// JVM-invoked. `handle` must be valid; `page_index` may be out of
/// range and we surface that as a `PdfParseException` (per the v0.3.52
/// Rust Error::ParseError mapping).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeExtractText<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: Java side asserted handle != 0 before calling.
        let doc = unsafe { doc_ref(handle) };
        if page_index < 0 {
            // Match Java's IndexOutOfBoundsException convention for
            // negative page indices. The Rust core would also error,
            // but with a less specific message.
            let class = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
            let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
            let _ = env.throw_new(&class, &msg);
            return Err(JniError::JavaException);
        }
        match doc.extract_text(page_index as usize) {
            Ok(text) => Ok(env.new_string(text)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                // Unreachable but type-required:
                Ok(jni::objects::JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────── extractStructured ─────────────────────────────────

/// `Java_fyi_oxide_pdf_PdfDocument_nativeExtractStructured` — extract a page as
/// structured typed regions (issue #536), returned as a JSON string (a
/// serialized `StructuredPage`).
///
/// # Safety
///
/// JVM-invoked. `handle` must be valid; a negative `page_index` is surfaced as
/// `IndexOutOfBoundsException`.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeExtractStructured<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: Java side asserted handle != 0 before calling.
        let doc = unsafe { doc_ref(handle) };
        if page_index < 0 {
            let class = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
            let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
            let _ = env.throw_new(&class, &msg);
            return Err(JniError::JavaException);
        }
        match doc.extract_structured(page_index as usize) {
            Ok(structured) => match serde_json::to_string(&structured) {
                Ok(json) => Ok(env.new_string(json)?),
                Err(e) => {
                    let class = jni::strings::JNIString::from("java/lang/RuntimeException");
                    let msg = jni::strings::JNIString::from(format!(
                        "structured serialization failed: {e}"
                    ));
                    let _ = env.throw_new(&class, &msg);
                    Err(JniError::JavaException)
                },
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                // Unreachable but type-required:
                Ok(jni::objects::JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}
