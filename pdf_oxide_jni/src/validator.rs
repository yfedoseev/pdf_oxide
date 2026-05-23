//! JNI surface for {@code fyi.oxide.pdf.PdfValidator} — PDF/A and
//! PDF/UA compliance validators (v0.3.50).
//!
//! v0.3.53 ships **simplified boolean variants**:
//! `isPdfA(doc, level)` and `isPdfUa(doc, level)` returning just the
//! verdict. Full {@link fyi.oxide.pdf.compliance.ValidationResult}
//! marshalling (with the violations list + detected level) lands in
//! a follow-up.
//!
//! Level encoding across the JNI boundary uses the Java enum ordinal.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::JClass;
use jni::sys::{jboolean, jint, jlong, JNI_FALSE, JNI_TRUE};
use jni::EnvUnowned;
use pdf_oxide::compliance::{validate_pdf_a, validate_pdf_ua, PdfALevel, PdfUaLevel};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: caller (Java side) guarantees single-threaded access per
/// `00-common-foundation.md` §2.7 (PdfDocument is not thread-safe).
/// `handle` is a valid pointer to a leaked Box<PdfDocument>.
#[inline]
unsafe fn doc_mut<'h>(handle: jlong) -> &'h mut PdfDocument {
    debug_assert!(handle != 0, "JNI: PdfValidator handle was 0");
    unsafe { &mut *(handle as *mut PdfDocument) }
}

fn map_pdfa_ordinal<'local>(env: &mut jni::Env<'local>, ord: jint) -> Result<PdfALevel, JniError> {
    match ord {
        0 => Ok(PdfALevel::A1a),
        1 => Ok(PdfALevel::A1b),
        2 => Ok(PdfALevel::A2a),
        3 => Ok(PdfALevel::A2b),
        4 => Ok(PdfALevel::A2u),
        5 => Ok(PdfALevel::A3a),
        6 => Ok(PdfALevel::A3b),
        7 => Ok(PdfALevel::A3u),
        8..=10 => {
            let cls =
                jni::strings::JNIString::from("fyi/oxide/pdf/exception/PdfUnsupportedException");
            let msg =
                jni::strings::JNIString::from("PDF/A-4 levels not yet supported by pdf_oxide");
            env.throw_new(&cls, &msg)?;
            Err(JniError::JavaException)
        },
        _ => {
            let cls = jni::strings::JNIString::from("java/lang/IllegalArgumentException");
            let msg = jni::strings::JNIString::from(format!("unknown PdfALevel ordinal {}", ord));
            env.throw_new(&cls, &msg)?;
            Err(JniError::JavaException)
        },
    }
}

fn map_pdfua_ordinal<'local>(
    env: &mut jni::Env<'local>,
    ord: jint,
) -> Result<PdfUaLevel, JniError> {
    match ord {
        0 => Ok(PdfUaLevel::Ua1),
        1 => {
            let cls =
                jni::strings::JNIString::from("fyi/oxide/pdf/exception/PdfUnsupportedException");
            let msg = jni::strings::JNIString::from("PDF/UA-2 not yet supported by pdf_oxide");
            env.throw_new(&cls, &msg)?;
            Err(JniError::JavaException)
        },
        _ => {
            let cls = jni::strings::JNIString::from("java/lang/IllegalArgumentException");
            let msg = jni::strings::JNIString::from(format!("unknown PdfUaLevel ordinal {}", ord));
            env.throw_new(&cls, &msg)?;
            Err(JniError::JavaException)
        },
    }
}

/// `Java_fyi_oxide_pdf_PdfValidator_nativeIsPdfA` — quick verdict.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfValidator_nativeIsPdfA<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    level_ordinal: jint,
) -> jboolean {
    env.with_env(|env| -> Result<jboolean, JniError> {
        let level = map_pdfa_ordinal(env, level_ordinal)?;
        let doc = unsafe { doc_mut(handle) };
        match validate_pdf_a(doc, level) {
            Ok(r) => Ok(if r.is_compliant { JNI_TRUE } else { JNI_FALSE }),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JNI_FALSE)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `Java_fyi_oxide_pdf_PdfValidator_nativeIsPdfUa` — quick verdict.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfValidator_nativeIsPdfUa<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    level_ordinal: jint,
) -> jboolean {
    env.with_env(|env| -> Result<jboolean, JniError> {
        let level = map_pdfua_ordinal(env, level_ordinal)?;
        let doc = unsafe { doc_mut(handle) };
        match validate_pdf_ua(doc, level) {
            Ok(r) => Ok(if r.is_compliant { JNI_TRUE } else { JNI_FALSE }),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JNI_FALSE)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}
