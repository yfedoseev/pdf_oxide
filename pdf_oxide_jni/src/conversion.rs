//! JNI surface for `fyi.oxide.pdf.PdfAConverter` — static bytes-in/
//! bytes-out PDF/A conversion, wrapping
//! `pdf_oxide::compliance::convert_to_pdf_a`.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::jni_sig;
use jni::objects::{JByteArray, JClass, JObject};
use jni::strings::JNIString;
use jni::sys::jint;
use jni::EnvUnowned;
use pdf_oxide::compliance::{convert_to_pdf_a, ActionType, ConversionResult, PdfALevel};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

fn level_from_ordinal(ordinal: jint) -> Option<PdfALevel> {
    match ordinal {
        0 => Some(PdfALevel::A1b),
        1 => Some(PdfALevel::A1a),
        2 => Some(PdfALevel::A2b),
        3 => Some(PdfALevel::A2a),
        4 => Some(PdfALevel::A2u),
        5 => Some(PdfALevel::A3b),
        6 => Some(PdfALevel::A3a),
        7 => Some(PdfALevel::A3u),
        _ => None,
    }
}

/// `Java_fyi_oxide_pdf_PdfAConverter_nativeConvert` — convert `pdf` to
/// the PDF/A level identified by `level_ordinal`, returning a fully
/// populated `fyi.oxide.pdf.compliance.ConversionResult`. `level` is
/// the Java-side `PdfALevel` enum constant the caller already holds
/// (the Java caller filters out PDF/A-4 before reaching here); it is
/// embedded into the result unchanged rather than reconstructed from
/// the ordinal.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfAConverter_nativeConvert<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    pdf_bytes: JByteArray<'local>,
    level_ordinal: jint,
    level: JObject<'local>,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        let Some(pdf_level) = level_from_ordinal(level_ordinal) else {
            let cls = JNIString::from("java/lang/IllegalArgumentException");
            let msg = JNIString::from(format!("unknown PdfALevel ordinal {}", level_ordinal));
            env.throw_new(&cls, &msg)?;
            return Ok(JObject::null());
        };

        let pdf: Vec<u8> = env.convert_byte_array(&pdf_bytes)?;
        let mut doc = match PdfDocument::from_bytes(pdf) {
            Ok(d) => d,
            Err(e) => {
                throw_pdf(env, &e)?;
                return Ok(JObject::null());
            },
        };

        match convert_to_pdf_a(&mut doc, pdf_level) {
            Ok(result) => build_conversion_result(env, &doc.source_bytes, &result, &level),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn action_type_field<'local>(
    env: &mut jni::Env<'local>,
    class: &JClass<'local>,
    name: &str,
) -> Result<JObject<'local>, JniError> {
    env.get_static_field(
        class,
        &JNIString::from(name),
        jni_sig!("Lfyi/oxide/pdf/compliance/ActionType;"),
    )?
    .l()
}

fn build_conversion_result<'local>(
    env: &mut jni::Env<'local>,
    converted_pdf: &[u8],
    result: &ConversionResult,
    level: &JObject<'local>,
) -> Result<JObject<'local>, JniError> {
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;

    let action_type_class =
        env.find_class(&JNIString::from("fyi/oxide/pdf/compliance/ActionType"))?;
    let at_added_xmp_metadata = action_type_field(env, &action_type_class, "ADDED_XMP_METADATA")?;
    let at_added_pdfa_identification =
        action_type_field(env, &action_type_class, "ADDED_PDFA_IDENTIFICATION")?;
    let at_embedded_font = action_type_field(env, &action_type_class, "EMBEDDED_FONT")?;
    let at_added_output_intent = action_type_field(env, &action_type_class, "ADDED_OUTPUT_INTENT")?;
    let at_removed_javascript = action_type_field(env, &action_type_class, "REMOVED_JAVASCRIPT")?;
    let at_removed_encryption = action_type_field(env, &action_type_class, "REMOVED_ENCRYPTION")?;
    let at_flattened_transparency =
        action_type_field(env, &action_type_class, "FLATTENED_TRANSPARENCY")?;
    let at_removed_embedded_files =
        action_type_field(env, &action_type_class, "REMOVED_EMBEDDED_FILES")?;
    let at_added_structure = action_type_field(env, &action_type_class, "ADDED_STRUCTURE")?;
    let at_fixed_annotation = action_type_field(env, &action_type_class, "FIXED_ANNOTATION")?;
    let at_added_language = action_type_field(env, &action_type_class, "ADDED_LANGUAGE")?;

    let action_class =
        env.find_class(&JNIString::from("fyi/oxide/pdf/compliance/ConversionAction"))?;
    let action_ctor = env.get_method_id(
        &action_class,
        &JNIString::from("<init>"),
        jni_sig!("(Lfyi/oxide/pdf/compliance/ActionType;Ljava/lang/String;Ljava/lang/String;)V"),
    )?;

    let error_class =
        env.find_class(&JNIString::from("fyi/oxide/pdf/compliance/ConversionError"))?;
    let error_ctor = env.get_method_id(
        &error_class,
        &JNIString::from("<init>"),
        jni_sig!("(Ljava/lang/String;Ljava/lang/String;)V"),
    )?;

    let actions_list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: result.actions.len() as i32,
            }],
        )?
    };
    for action in &result.actions {
        let action_type_obj = match action.action_type {
            ActionType::AddedXmpMetadata => &at_added_xmp_metadata,
            ActionType::AddedPdfaIdentification => &at_added_pdfa_identification,
            ActionType::EmbeddedFont => &at_embedded_font,
            ActionType::AddedOutputIntent => &at_added_output_intent,
            ActionType::RemovedJavaScript => &at_removed_javascript,
            ActionType::RemovedEncryption => &at_removed_encryption,
            ActionType::FlattenedTransparency => &at_flattened_transparency,
            ActionType::RemovedEmbeddedFiles => &at_removed_embedded_files,
            ActionType::AddedStructure => &at_added_structure,
            ActionType::FixedAnnotation => &at_fixed_annotation,
            ActionType::AddedLanguage => &at_added_language,
        };
        let description = env.new_string(&action.description)?;
        let fixed_error_code: JObject = match action.fixed_error {
            Some(code) => env.new_string(code.to_string())?.into(),
            None => JObject::null(),
        };
        let action_obj = unsafe {
            env.new_object_unchecked(
                &action_class,
                action_ctor,
                &[
                    jni::sys::jvalue {
                        l: action_type_obj.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: description.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: fixed_error_code.as_raw(),
                    },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &actions_list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue {
                    l: action_obj.as_raw(),
                }],
            )?;
        }
    }

    let errors_list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: result.errors.len() as i32,
            }],
        )?
    };
    for error in &result.errors {
        let error_code = env.new_string(error.error_code.to_string())?;
        let reason = env.new_string(&error.reason)?;
        let error_obj = unsafe {
            env.new_object_unchecked(
                &error_class,
                error_ctor,
                &[
                    jni::sys::jvalue {
                        l: error_code.as_raw(),
                    },
                    jni::sys::jvalue { l: reason.as_raw() },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &errors_list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue {
                    l: error_obj.as_raw(),
                }],
            )?;
        }
    }

    let converted_bytes = env.byte_array_from_slice(converted_pdf)?;

    let result_class =
        env.find_class(&JNIString::from("fyi/oxide/pdf/compliance/ConversionResult"))?;
    let result_ctor = env.get_method_id(
        &result_class,
        &JNIString::from("<init>"),
        jni_sig!("(ZLfyi/oxide/pdf/compliance/PdfALevel;[BLjava/util/List;Ljava/util/List;)V"),
    )?;
    let result_obj = unsafe {
        env.new_object_unchecked(
            &result_class,
            result_ctor,
            &[
                jni::sys::jvalue {
                    z: result.success as jni::sys::jboolean,
                },
                jni::sys::jvalue { l: level.as_raw() },
                jni::sys::jvalue {
                    l: converted_bytes.as_raw(),
                },
                jni::sys::jvalue {
                    l: actions_list.as_raw(),
                },
                jni::sys::jvalue {
                    l: errors_list.as_raw(),
                },
            ],
        )?
    };
    Ok(result_obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinal_maps_only_pdf_a_1_2_3_levels() {
        assert_eq!(level_from_ordinal(0), Some(PdfALevel::A1b));
        assert_eq!(level_from_ordinal(7), Some(PdfALevel::A3u));
        assert_eq!(level_from_ordinal(8), None);
        assert_eq!(level_from_ordinal(-1), None);
    }
}
