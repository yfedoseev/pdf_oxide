//! JNI surface for `fyi.oxide.pdf.DocumentEditor` — the write-side
//! counterpart to {@link fyi.oxide.pdf.PdfDocument}. Wraps
//! [`pdf_oxide::editor::DocumentEditor`].
//!
//! v0.3.53 ships: open, close, setFormField (Text + Boolean variants),
//! saveToBytes. Follow-ups: addRedaction + applyRedactionsDestructive
//! (with the [BLOCK] oracle from v0.3.50 #231), scrubMetadata, and
//! Choice/MultiChoice form fields.

use std::path::PathBuf;
use std::sync::Mutex;

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jboolean, jbyteArray, jint, jlong, JNI_TRUE};
use jni::EnvUnowned;
use pdf_oxide::editor::{DocumentEditor, FormFieldValue};

use crate::error::throw_pdf;

/// Mutex-wrapped editor — DocumentEditor APIs take `&mut self`, so
/// the JNI side needs exclusive access on every call. The Java side
/// already documents non-thread-safety; the Mutex is a defense
/// against accidental concurrent calls.
type SharedEditor = Mutex<DocumentEditor>;

#[inline]
unsafe fn editor_ref<'h>(handle: jlong) -> &'h SharedEditor {
    debug_assert!(handle != 0, "JNI: DocumentEditor handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const SharedEditor) }
}

// ─────────────────────────── open(path) ────────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeOpenPath<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    path: JString<'local>,
) -> jlong {
    env.with_env(|env| -> Result<jlong, JniError> {
        let path_str: String = path.try_to_string(env)?;
        let path_buf = PathBuf::from(path_str);
        match DocumentEditor::open(&path_buf) {
            Ok(ed) => {
                let boxed = Box::new(Mutex::new(ed));
                Ok(Box::into_raw(boxed) as jlong)
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────── open(bytes) ───────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeOpenBytes<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    bytes: JByteArray<'local>,
) -> jlong {
    env.with_env(|env| -> Result<jlong, JniError> {
        let vec: Vec<u8> = env.convert_byte_array(&bytes)?;
        match DocumentEditor::from_bytes(vec) {
            Ok(ed) => {
                let boxed = Box::new(Mutex::new(ed));
                Ok(Box::into_raw(boxed) as jlong)
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────── setFormField (Text) ───────────────────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeSetFormFieldText<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    name: JString<'local>,
    value: JString<'local>,
) {
    let _ = env
        .with_env(|env| -> Result<(), JniError> {
            let name_str: String = name.try_to_string(env)?;
            let value_str: String = value.try_to_string(env)?;
            // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
            let editor = unsafe { editor_ref(handle) };
            let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
            if let Err(e) = guard.set_form_field_value(&name_str, FormFieldValue::Text(value_str)) {
                throw_pdf(env, &e)?;
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}

// ───────────────────── setFormField (Boolean / checkbox) ───────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeSetFormFieldBoolean<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    name: JString<'local>,
    checked: jboolean,
) {
    let _ = env
        .with_env(|env| -> Result<(), JniError> {
            let name_str: String = name.try_to_string(env)?;
            // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
            let editor = unsafe { editor_ref(handle) };
            let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
            if let Err(e) =
                guard.set_form_field_value(&name_str, FormFieldValue::Boolean(checked == JNI_TRUE))
            {
                throw_pdf(env, &e)?;
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}

// ──────────────────────────── addRedaction ────────────────────────────────

/// `nativeAddRedaction` — queue a redaction region for a page.
/// Rectangle is in PDF user-space `(x0, y0, x1, y1)`. Fill color is
/// the configured default for v0.3.53. Does NOT apply destructively
/// — call `nativeApplyRedactionsDestructive` (Phase 3 T11 — gated
/// on the v0.3.50 [BLOCK] oracle) to actually remove content.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeAddRedaction<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
) {
    let _ = env
        .with_env(|env| -> Result<(), JniError> {
            if page_index < 0 {
                let cls = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
                let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
                let _ = env.throw_new(&cls, &msg);
                return Err(JniError::JavaException);
            }
            // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
            let editor = unsafe { editor_ref(handle) };
            let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
            let rect = [x0 as f32, y0 as f32, x1 as f32, y1 as f32];
            if let Err(e) = guard.add_redaction(page_index as usize, rect, None) {
                throw_pdf(env, &e)?;
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}

/// `nativeRedactionCount` — total redactions queued for the page
/// (programmatic + source `/Redact` annotations).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeRedactionCount<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        if page_index < 0 {
            let cls = jni::strings::JNIString::from("java/lang/IndexOutOfBoundsException");
            let msg = jni::strings::JNIString::from(format!("page index {} < 0", page_index));
            let _ = env.throw_new(&cls, &msg);
            return Err(JniError::JavaException);
        }
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let editor = unsafe { editor_ref(handle) };
        let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
        match guard.redaction_count(page_index as usize) {
            Ok(n) => Ok(n as jint),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(-1)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ──────────────────── applyRedactionsDestructive ──────────────────────────

/// `nativeApplyRedactionsDestructive` — execute all queued
/// redactions, returning the number of regions actually applied.
/// The Rust core fail-closes on composite/Type0/unknown-font pages
/// (refused via `Error::Unsupported` rather than risking silent
/// under-redaction). Uses default `RedactionOptions` which scrub
/// document metadata + remove embedded files + drop JavaScript +
/// strip hidden OCGs — the v0.3.50 #231 safety contract.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeApplyRedactionsDestructive<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let editor = unsafe { editor_ref(handle) };
        let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
        let opts = pdf_oxide::redaction::RedactionOptions::default();
        match guard.apply_redactions_destructive(opts) {
            Ok(report) => Ok(report.regions as jint),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(-1)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────── saveToBytes ───────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeSaveToBytes<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jbyteArray {
    env.with_env(|env| -> Result<jbyteArray, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let editor = unsafe { editor_ref(handle) };
        let mut guard = editor.lock().expect("DocumentEditor mutex poisoned");
        match guard.save_to_bytes() {
            Ok(bytes) => Ok(env.byte_array_from_slice(&bytes)?.into_raw()),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(std::ptr::null_mut())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

// ─────────────────────────────── close ─────────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_DocumentEditor_nativeClose<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    let _ = env
        .with_env(|_env| -> Result<(), JniError> {
            if handle != 0 {
                unsafe {
                    drop(Box::from_raw(handle as *mut SharedEditor));
                }
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}
