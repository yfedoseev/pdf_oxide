//! JNI surface for {@code fyi.oxide.pdf.PdfValidator} — PDF/A and
//! PDF/UA compliance validators (v0.3.50).
//!
//! v0.3.53 ships **simplified boolean variants**:
//! `isPdfA(doc, level)` and `isPdfUa(doc, level)` returning just the
//! verdict. Full {@link fyi.oxide.pdf.compliance.ValidationResult}
//! marshalling (with the violations list + detected level) lands in
//! a follow-up.
//!
//! Level encoding across the JNI boundary mirrors the cdylib's C ABI
//! wire format (v0.3.55 #547): {@code PdfALevel.ordinal()} matches
//! {@code src/ffi.rs:1225} (B before A within each level), and
//! {@code PdfUaLevel.code()} matches {@code src/ffi.rs:5538}
//! (1-indexed: UA-1=1, UA-2=2). Every pdf_oxide binding (Java, C#,
//! Ruby, PHP, Go) sends the same integer for the same level — keeping
//! the JNI mapping aligned with that wire format is what makes
//! cross-binding parity tests pass.

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

/// Translate a Java {@code PdfALevel.ordinal()} into the Rust enum.
///
/// The JNI wire format mirrors the cdylib C ABI documented at
/// `src/ffi.rs:1225` (`0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a
/// 7=A3u`). Java's `PdfALevel` is reordered (B before A within each
/// level — see v0.3.55 #547) so `.ordinal()` matches this mapping
/// directly. C# / Ruby / PHP / Go all use the same numeric encoding;
/// the JNI shim aligning here is what makes "feature sets identical
/// across languages" actually hold.
///
/// Higher ordinals (8..=10) are the Java-side `A_4`, `A_4E`, `A_4F`
/// placeholders — not yet implemented in the cdylib.
fn map_pdfa_ordinal<'local>(env: &mut jni::Env<'local>, ord: jint) -> Result<PdfALevel, JniError> {
    match ord {
        0 => Ok(PdfALevel::A1b),
        1 => Ok(PdfALevel::A1a),
        2 => Ok(PdfALevel::A2b),
        3 => Ok(PdfALevel::A2a),
        4 => Ok(PdfALevel::A2u),
        5 => Ok(PdfALevel::A3b),
        6 => Ok(PdfALevel::A3a),
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

/// Translate a Java {@code PdfUaLevel.code()} into the Rust enum.
///
/// Wire format matches `src/ffi.rs:5538` — the cdylib treats `level
/// == 2` as PDF/UA-2 and anything else as PDF/UA-1. Java's
/// `PdfUaLevel` is explicit-coded (UA_1=1, UA_2=2) and the public
/// API now calls `.code()` rather than `.ordinal()` when crossing
/// the JNI boundary. This brings Java in lock-step with the C#
/// (`Ua1=1, Ua2=2`), Ruby (`{ua1: 1, ua2: 2}`), and PHP (`PDFUA_1=1,
/// PDFUA_2=2`) bindings.
fn map_pdfua_ordinal<'local>(
    env: &mut jni::Env<'local>,
    ord: jint,
) -> Result<PdfUaLevel, JniError> {
    match ord {
        1 => Ok(PdfUaLevel::Ua1),
        2 => Ok(PdfUaLevel::Ua2),
        _ => {
            let cls = jni::strings::JNIString::from("java/lang/IllegalArgumentException");
            let msg = jni::strings::JNIString::from(format!("unknown PdfUaLevel code {}", ord));
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
