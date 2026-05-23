//! JNI surface for the v0.3.51 AutoExtractor — partial v0.3.53
//! coverage.
//!
//! Wires the simplest path: `classifyPage(pageIndex) -> int` returning
//! the ordinal of a Java `PageClass` enum value. Future follow-ups:
//! `extractPage` / `extractDocument` with the full AutoResult tree
//! (typed reasons + regions + confidence), needing the JSON-envelope
//! wire format from the v0.3.51 C ABI.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong};
use jni::EnvUnowned;
use pdf_oxide::extractors::auto::{AutoExtractor as RsAutoExtractor, PageKind};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: AutoExtractor handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// Map Rust `PageKind` → Java `PageClass` ordinal:
/// 0=TEXT_LAYER, 1=SCANNED, 2=MIXED, 3=EMPTY.
/// Locked to the Java enum declaration order in
/// `fyi/oxide/pdf/auto/PageClass.java`.
fn page_class_ordinal(kind: PageKind) -> jint {
    match kind {
        PageKind::TextLayer => 0,
        PageKind::Scanned => 1,
        PageKind::ImageText | PageKind::Mixed => 2,
        PageKind::Empty => 3,
        // Future PageKind variants (the enum is #[non_exhaustive])
        // fall through to MIXED to preserve forward-compatibility.
        _ => 2,
    }
}

/// `Java_fyi_oxide_pdf_AutoExtractor_nativeClassifyPageOrdinal` —
/// classify a single page; returns the ordinal of a Java
/// `fyi.oxide.pdf.auto.PageClass` enum value.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_AutoExtractor_nativeClassifyPageOrdinal<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.classify_page(page_index as usize) {
            Ok(c) => Ok(page_class_ordinal(c.kind)),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeExtractPageJson` — full v0.3.51 rich PageExtraction
/// serialized to JSON. Java callers parse with their preferred
/// JSON library (org.json / jackson / gson / etc.) — the binding
/// doesn't impose one. JSON carries text + regions[] + confidence
/// + reason + ocrUsed + per-region bbox/reason/confidence.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_AutoExtractor_nativeExtractPageJson<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let extractor = RsAutoExtractor::new();
        match extractor.extract_page(doc, page_index as usize) {
            Ok(page) => {
                let json = serde_json::to_string(&page).unwrap_or_else(|e| {
                    // Build the fallback via serde_json so the error
                    // message is JSON-escaped — a raw format! would emit
                    // invalid JSON if `e` contained quotes/backslashes.
                    serde_json::json!({ "_serde_error": e.to_string() }).to_string()
                });
                Ok(env.new_string(json)?)
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeExtractDocumentJson` — full v0.3.51 rich DocumentExtraction
/// serialized to JSON.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_AutoExtractor_nativeExtractDocumentJson<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let extractor = RsAutoExtractor::new();
        match extractor.extract_document(doc) {
            Ok(d) => {
                let json = serde_json::to_string(&d).unwrap_or_else(|e| {
                    // Build the fallback via serde_json so the error
                    // message is JSON-escaped — a raw format! would emit
                    // invalid JSON if `e` contained quotes/backslashes.
                    serde_json::json!({ "_serde_error": e.to_string() }).to_string()
                });
                Ok(env.new_string(json)?)
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeClassifyDocumentOrdinals` — classify every page; returns
/// `int[]` of `PageClass` ordinals (length == pageCount).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_AutoExtractor_nativeClassifyDocumentOrdinals<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jni::sys::jintArray {
    env.with_env(|env| -> Result<jni::sys::jintArray, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.classify_document() {
            Ok(c) => {
                let ords: Vec<jint> = c.pages.iter().map(|k| page_class_ordinal(*k)).collect();
                let arr = env.new_int_array(ords.len())?;
                arr.set_region(env, 0, &ords)?;
                Ok(arr.into_raw())
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(std::ptr::null_mut())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}
