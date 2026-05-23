//! JNI surface for `fyi.oxide.pdf.PdfPolicy` — the v0.3.50 #230
//! crypto-governance policy.
//!
//! Process-global state on the Rust side
//! ([`pdf_oxide::crypto::active`]). Java {@link
//! fyi.oxide.pdf.PdfPolicy} exposes `current()` / `set(PolicyMode)`
//! / presets.
//!
//! Encoding for `PolicyMode` across the JNI boundary: a small
//! `jint` discriminant matching the {@link
//! fyi.oxide.pdf.policy.PolicyMode} ordinal:
//!
//! - `0` = COMPAT
//! - `1` = STRICT
//! - `2` = FIPS_STRICT

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::JClass;
use jni::sys::jint;
use jni::EnvUnowned;
use pdf_oxide::crypto::{active_policy, set_policy, PolicyMode, SecurityPolicy};

use crate::error::PdfErrorKind;

const POLICY_COMPAT: jint = 0;
const POLICY_STRICT: jint = 1;
const POLICY_FIPS_STRICT: jint = 2;

/// `Java_fyi_oxide_pdf_PdfPolicy_nativeCurrentOrdinal` — return the
/// ordinal of the active {@link PolicyMode}.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPolicy_nativeCurrentOrdinal<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> jint {
    env.with_env(|_env| -> Result<jint, JniError> {
        let p = active_policy();
        Ok(match p.mode() {
            PolicyMode::Compat => POLICY_COMPAT,
            PolicyMode::Strict => POLICY_STRICT,
            PolicyMode::FipsStrict => POLICY_FIPS_STRICT,
            // Future variants (CnsaStrict etc., introduced in #230 Phase D/E):
            // bucket as STRICT for the Java surface until we expose a richer
            // enum. Documented in api-design.md §15.
            _ => POLICY_STRICT,
        })
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `Java_fyi_oxide_pdf_PdfPolicy_nativeSetByOrdinal` — set the
/// process-global policy from an ordinal. Throws a Java
/// {@link IllegalArgumentException} for unknown ordinals.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPolicy_nativeSetByOrdinal<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    ordinal: jint,
) {
    let _ = env
        .with_env(|env| -> Result<(), JniError> {
            let policy = match ordinal {
                POLICY_COMPAT => SecurityPolicy::compat(),
                POLICY_STRICT => SecurityPolicy::strict(),
                POLICY_FIPS_STRICT => SecurityPolicy::fips_strict(),
                _ => {
                    let cls = jni::strings::JNIString::from("java/lang/IllegalArgumentException");
                    let msg = jni::strings::JNIString::from(format!(
                        "unknown PolicyMode ordinal {}",
                        ordinal
                    ));
                    env.throw_new(&cls, &msg)?;
                    return Err(JniError::JavaException);
                },
            };
            if let Err(e) = set_policy(policy) {
                // SetPolicyError is its own type — surface as a generic
                // PdfException(kind=Other) with the underlying message.
                let msg = jni::strings::JNIString::from(format!("set_policy failed: {}", e));
                let cls = jni::strings::JNIString::from(PdfErrorKind::Other.java_class());
                env.throw_new(&cls, &msg)?;
                return Err(JniError::JavaException);
            }
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>();
}
