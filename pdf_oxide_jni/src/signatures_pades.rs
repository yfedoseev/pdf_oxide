//! JNI surface for `fyi.oxide.pdf.PdfSigner` — PAdES signatures
//! (v0.3.50 #235). v0.3.53 ships the **read-only verify path**:
//! `classifyLevel(byte[])` enumerates a PDF's signatures and returns
//! the highest PAdES level present (B_B / B_T / B_LT). The full
//! `sign(...)` / `verify(...)` write-path requires PKCS#12 key
//! material + TSA HTTP plumbing + ETSI EN 319 142-1 conformance work
//! — multi-week, follow-up.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jbyteArray, jint};
use jni::EnvUnowned;
#[cfg(feature = "signatures")]
use pdf_oxide::signatures::{
    classify_pades_level, enumerate_signatures, read_dss, sign_pdf_bytes_pades, PadesLevel,
    RevocationMaterial, SignOptions, SigningCredentials,
};
#[cfg(all(feature = "signatures", feature = "tsa-client"))]
use pdf_oxide::signatures::{TsaClient, TsaClientConfig};
#[cfg(feature = "signatures")]
use pdf_oxide::PdfDocument;

#[cfg(feature = "signatures")]
use crate::error::throw_pdf;

#[cfg(feature = "signatures")]
fn level_ordinal(l: PadesLevel) -> jint {
    match l {
        PadesLevel::BB => 0,
        PadesLevel::BT => 1,
        PadesLevel::BLt => 2,
        // Future PadesLevel::BLta etc. (the enum is #[non_exhaustive])
        // collapses to B_LT for the v0.3.53 Java surface (the Java
        // SignatureLevel enum is B_B/B_T/B_LT only).
        _ => 2,
    }
}

/// `Java_fyi_oxide_pdf_PdfSigner_nativeSignBB` — basic PAdES B-B
/// signing. Loads credentials from a PKCS#12 / PFX byte[] + password,
/// signs the PDF, returns the signed bytes.
///
/// v0.3.53 limitation: ONLY produces PAdES-B-B (no timestamp).
/// B-T / B-LT require an RFC 3161 TSA HTTP client; deferred follow-up.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfSigner_nativeSignBB<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    pdf_bytes: JByteArray<'local>,
    pkcs12_bytes: JByteArray<'local>,
    password: JString<'local>,
) -> jbyteArray {
    #[cfg(not(feature = "signatures"))]
    {
        let _ = (pdf_bytes, pkcs12_bytes, password);
        let _ = env
            .with_env(|env| -> Result<jbyteArray, JniError> {
                let cls = jni::strings::JNIString::from(
                    "fyi/oxide/pdf/exception/PdfUnsupportedException",
                );
                let msg = jni::strings::JNIString::from(
                "PdfSigner.sign requires pdf_oxide_jni built with --features signatures (or full)");
                env.throw_new(&cls, &msg)?;
                Err(JniError::JavaException)
            })
            .resolve::<ThrowRuntimeExAndDefault>();
        std::ptr::null_mut()
    }
    #[cfg(feature = "signatures")]
    {
        env.with_env(|env| -> Result<jbyteArray, JniError> {
            let pdf: Vec<u8> = env.convert_byte_array(&pdf_bytes)?;
            let p12: Vec<u8> = env.convert_byte_array(&pkcs12_bytes)?;
            let pw: String = password.try_to_string(env)?;
            let credentials = match SigningCredentials::from_pkcs12(&p12, &pw) {
                Ok(c) => c,
                Err(e) => {
                    throw_pdf(env, &e)?;
                    return Ok(std::ptr::null_mut());
                },
            };
            let opts = SignOptions::default();
            let material = RevocationMaterial::default();
            match sign_pdf_bytes_pades(&pdf, &credentials, opts, PadesLevel::BB, None, &material) {
                Ok(signed) => Ok(env.byte_array_from_slice(&signed)?.into_raw()),
                Err(e) => {
                    throw_pdf(env, &e)?;
                    Ok(std::ptr::null_mut())
                },
            }
        })
        .resolve::<ThrowRuntimeExAndDefault>()
    }
}

/// `Java_fyi_oxide_pdf_PdfSigner_nativeSign` — full PAdES signing
/// path supporting B-B / B-T / B-LT levels. B-T and B-LT require
/// a non-null `tsaUrl` (a public TSA endpoint that speaks RFC 3161
/// over HTTP). The Rust core's existing TSA client makes the
/// outbound HTTP POST and constructs the timestamp token; the
/// signing pipeline then embeds it as the `signature-time-stamp`
/// CMS unsigned attribute (B-T) and optionally writes the DSS
/// incremental update (B-LT).
///
/// Level ordinals: 0=B_B, 1=B_T, 2=B_LT.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfSigner_nativeSign<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    pdf_bytes: JByteArray<'local>,
    pkcs12_bytes: JByteArray<'local>,
    password: JString<'local>,
    level_ordinal: jint,
    tsa_url: JString<'local>,
) -> jbyteArray {
    #[cfg(not(feature = "signatures"))]
    {
        let _ = (pdf_bytes, pkcs12_bytes, password, level_ordinal, tsa_url);
        let _ = env
            .with_env(|env| -> Result<jbyteArray, JniError> {
                let cls = jni::strings::JNIString::from(
                    "fyi/oxide/pdf/exception/PdfUnsupportedException",
                );
                let msg = jni::strings::JNIString::from(
                    "PdfSigner.sign requires pdf_oxide_jni built with --features signatures",
                );
                env.throw_new(&cls, &msg)?;
                Err(JniError::JavaException)
            })
            .resolve::<ThrowRuntimeExAndDefault>();
        std::ptr::null_mut()
    }
    #[cfg(feature = "signatures")]
    {
        env.with_env(|env| -> Result<jbyteArray, JniError> {
            let pdf: Vec<u8> = env.convert_byte_array(&pdf_bytes)?;
            let p12: Vec<u8> = env.convert_byte_array(&pkcs12_bytes)?;
            let pw: String = password.try_to_string(env)?;

            let level = match level_ordinal {
                0 => PadesLevel::BB,
                1 => PadesLevel::BT,
                2 => PadesLevel::BLt,
                _ => {
                    let cls = jni::strings::JNIString::from("java/lang/IllegalArgumentException");
                    let msg = jni::strings::JNIString::from(format!(
                        "unknown SignatureLevel ordinal {}",
                        level_ordinal
                    ));
                    env.throw_new(&cls, &msg)?;
                    return Err(JniError::JavaException);
                },
            };

            // tsa_url is empty / null → None; otherwise build TsaClient.
            // Only used when `tsa-client` feature is enabled.
            #[cfg(feature = "tsa-client")]
            let tsa_url_str: String = if tsa_url.is_null() {
                String::new()
            } else {
                tsa_url.try_to_string(env).unwrap_or_default()
            };
            #[cfg(not(feature = "tsa-client"))]
            let _ = tsa_url;

            let credentials = match SigningCredentials::from_pkcs12(&p12, &pw) {
                Ok(c) => c,
                Err(e) => {
                    throw_pdf(env, &e)?;
                    return Ok(std::ptr::null_mut());
                },
            };
            let opts = SignOptions::default();
            let material = RevocationMaterial::default();

            // For B-T / B-LT, build the timestamper closure.
            #[cfg(feature = "tsa-client")]
            {
                if !tsa_url_str.is_empty() {
                    let tsa = TsaClient::new(TsaClientConfig::new(tsa_url_str.clone()));
                    let timestamper = |data: &[u8]| -> pdf_oxide::Result<Vec<u8>> {
                        tsa.request_timestamp(data)
                            .map(|t| t.token_bytes().to_vec())
                    };
                    return match sign_pdf_bytes_pades(
                        &pdf,
                        &credentials,
                        opts,
                        level,
                        Some(&timestamper),
                        &material,
                    ) {
                        Ok(signed) => Ok(env.byte_array_from_slice(&signed)?.into_raw()),
                        Err(e) => {
                            throw_pdf(env, &e)?;
                            Ok(std::ptr::null_mut())
                        },
                    };
                }
            }

            // No TSA — only B-B is permitted; B-T/B-LT will error.
            match sign_pdf_bytes_pades(&pdf, &credentials, opts, level, None, &material) {
                Ok(signed) => Ok(env.byte_array_from_slice(&signed)?.into_raw()),
                Err(e) => {
                    throw_pdf(env, &e)?;
                    Ok(std::ptr::null_mut())
                },
            }
        })
        .resolve::<ThrowRuntimeExAndDefault>()
    }
}

/// `Java_fyi_oxide_pdf_PdfSigner_nativeClassifyPdfLevel` — open the
/// PDF bytes, enumerate signatures, return the ordinal of the
/// HIGHEST PAdES level present. Returns `-1` when there are no
/// signatures (Java side surfaces this as a thrown
/// {@link IllegalStateException}, since classifying a non-signed PDF
/// has no meaningful answer).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfSigner_nativeClassifyPdfLevel<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    pdf_bytes: JByteArray<'local>,
) -> jint {
    #[cfg(not(feature = "signatures"))]
    {
        // Build without `signatures` feature: surface as Unsupported.
        let _ = pdf_bytes;
        let _ = env.with_env(|env| -> Result<jint, JniError> {
            let cls = jni::strings::JNIString::from(
                "fyi/oxide/pdf/exception/PdfUnsupportedException");
            let msg = jni::strings::JNIString::from(
                "PdfSigner.classifyLevel requires pdf_oxide_jni built with --features signatures (or full)");
            env.throw_new(&cls, &msg)?;
            Err(JniError::JavaException)
        })
        .resolve::<ThrowRuntimeExAndDefault>();
        -1
    }
    #[cfg(feature = "signatures")]
    {
        env.with_env(|env| -> Result<jint, JniError> {
            let bytes: Vec<u8> = env.convert_byte_array(&pdf_bytes)?;
            let mut doc = match PdfDocument::from_bytes(bytes) {
                Ok(d) => d,
                Err(e) => {
                    throw_pdf(env, &e)?;
                    return Ok(-1);
                },
            };
            let sigs = match enumerate_signatures(&mut doc) {
                Ok(s) => s,
                Err(e) => {
                    throw_pdf(env, &e)?;
                    return Ok(-1);
                },
            };
            if sigs.is_empty() {
                return Ok(-1);
            }
            let dss = read_dss(&doc).ok().flatten();
            let max_level = sigs
                .iter()
                .map(|s| classify_pades_level(s, dss.as_ref()))
                .max()
                .unwrap_or(PadesLevel::BB);
            Ok(level_ordinal(max_level))
        })
        .resolve::<ThrowRuntimeExAndDefault>()
    }
}
