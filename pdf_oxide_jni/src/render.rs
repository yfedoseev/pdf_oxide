//! JNI surface for {@code fyi.oxide.pdf.PdfDocument.render*} —
//! page rasterisation to PNG / raw bytes (the `rendering` feature
//! gate).
//!
//! v0.3.53 ships the simple `render(pageIndex) -> byte[]` path that
//! returns 150 DPI PNG bytes (pdf_oxide's default `RenderOptions`).
//! A future {@link fyi.oxide.pdf.render.RenderOptions} surface will
//! expose DPI / format / background customisation.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JByteArray, JClass};
use jni::sys::{jbyteArray, jint, jlong};
use jni::EnvUnowned;
use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: render handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// `nativeRenderPng` — render a page to PNG bytes at the supplied
/// DPI (150 if {@code dpi <= 0}).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeRenderPng<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
    dpi: jint,
) -> jbyteArray {
    env.with_env(|env| -> Result<jbyteArray, JniError> {
        if page_index < 0 {
            let cls = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
            let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
            let _ = env.throw_new(&cls, &msg);
            return Err(JniError::JavaException);
        }
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let mut opts = RenderOptions::default();
        if dpi > 0 {
            opts.dpi = dpi as u32;
        }
        match render_page(doc, page_index as usize, &opts) {
            Ok(img) => {
                let arr: JByteArray = env.byte_array_from_slice(img.as_bytes())?;
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
