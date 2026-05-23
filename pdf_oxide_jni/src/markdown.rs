//! JNI surface for `fyi.oxide.pdf.MarkdownConverter`.
//!
//! Static converters from a [`pdf_oxide::PdfDocument`] to Markdown or
//! HTML. The Java side passes the handle pointer (jlong) and we
//! delegate to the borrowed document. Uses
//! [`pdf_oxide::converters::ConversionOptions::default()`] for now;
//! tunable options follow per `api-design.md` §7.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong};
use jni::EnvUnowned;
use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: MarkdownConverter handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_MarkdownConverter_nativeToMarkdownPage<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let opts = ConversionOptions::default();
        match doc.to_markdown(page_index as usize, &opts) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_MarkdownConverter_nativeToMarkdownAll<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let opts = ConversionOptions::default();
        match doc.to_markdown_all(&opts) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_MarkdownConverter_nativeToHtmlPage<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let opts = ConversionOptions::default();
        match doc.to_html(page_index as usize, &opts) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_MarkdownConverter_nativeToHtmlAll<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JString<'local> {
    env.with_env(|env| -> Result<JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let opts = ConversionOptions::default();
        match doc.to_html_all(&opts) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}
