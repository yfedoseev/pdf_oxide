//! JNI surface for `fyi.oxide.pdf.PdfDocument.formFields()` — read
//! the document's AcroForm fields as `List<FormField>`.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JObject};
use jni::sys::jlong;
use jni::EnvUnowned;
use pdf_oxide::extractors::forms::{FieldType, FieldValue, FormExtractor};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: forms handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// `Java_fyi_oxide_pdf_PdfDocument_nativeFormFields` — extract all
/// AcroForm fields. Returns `ArrayList<FormField>`. v0.3.53
/// limitation: pdf_oxide's form extractor doesn't expose per-field
/// page index, so each FormField's `pageIndex` is -1 (unknown).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeFormFields<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match FormExtractor::extract_fields(doc) {
            Ok(fields) => build_form_field_list(env, &fields),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn build_form_field_list<'local>(
    env: &mut jni::Env<'local>,
    fields: &[pdf_oxide::extractors::forms::FormField],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let ff_class = env.find_class(&JNIString::from("fyi/oxide/pdf/form/FormField"))?;
    let ff_ctor = env.get_method_id(
        &ff_class,
        &JNIString::from("<init>"),
        jni_sig!("(Ljava/lang/String;Lfyi/oxide/pdf/form/FormFieldType;Ljava/lang/String;Lfyi/oxide/pdf/geometry/BBox;I)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let ft_class = env.find_class(&JNIString::from("fyi/oxide/pdf/form/FormFieldType"))?;
    let ft_text = env
        .get_static_field(
            &ft_class,
            &JNIString::from("TEXT"),
            jni_sig!("Lfyi/oxide/pdf/form/FormFieldType;"),
        )?
        .l()?;
    let ft_checkbox = env
        .get_static_field(
            &ft_class,
            &JNIString::from("CHECKBOX"),
            jni_sig!("Lfyi/oxide/pdf/form/FormFieldType;"),
        )?
        .l()?;
    let ft_choice = env
        .get_static_field(
            &ft_class,
            &JNIString::from("CHOICE"),
            jni_sig!("Lfyi/oxide/pdf/form/FormFieldType;"),
        )?
        .l()?;
    let ft_signature = env
        .get_static_field(
            &ft_class,
            &JNIString::from("SIGNATURE"),
            jni_sig!("Lfyi/oxide/pdf/form/FormFieldType;"),
        )?
        .l()?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: fields.len() as i32,
            }],
        )?
    };

    for f in fields {
        // Map Rust FieldType → Java FormFieldType. Button → CHECKBOX
        // for v0.3.53 (richer button/checkbox/radio split needs /Ff
        // bit-2 inspection — follow-up).
        let ft_ref = match &f.field_type {
            FieldType::Button => &ft_checkbox,
            FieldType::Text => &ft_text,
            FieldType::Choice => &ft_choice,
            FieldType::Signature => &ft_signature,
            FieldType::Unknown(_) => &ft_text,
        };

        // Map Rust FieldValue → Optional<String> (null on Java side).
        let val_opt: Option<String> = match &f.value {
            FieldValue::Text(s) | FieldValue::Name(s) => Some(s.clone()),
            FieldValue::Boolean(b) => Some(b.to_string()),
            FieldValue::Array(v) => Some(v.join(",")),
            FieldValue::None => None,
        };
        let val_ref: JObject = match &val_opt {
            Some(s) => env.new_string(s)?.into(),
            None => JObject::null(),
        };

        let bbox_obj: JObject = match f.bounds {
            Some([x0, y0, x1, y1]) => unsafe {
                env.new_object_unchecked(
                    &bbox_class,
                    bbox_ctor,
                    &[
                        jni::sys::jvalue { d: x0 },
                        jni::sys::jvalue { d: y0 },
                        jni::sys::jvalue { d: x1 },
                        jni::sys::jvalue { d: y1 },
                    ],
                )?
            },
            None => JObject::null(),
        };

        let name = env.new_string(&f.full_name)?;
        let ff_obj = unsafe {
            env.new_object_unchecked(
                &ff_class,
                ff_ctor,
                &[
                    jni::sys::jvalue { l: name.as_raw() },
                    jni::sys::jvalue { l: ft_ref.as_raw() },
                    jni::sys::jvalue {
                        l: val_ref.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: bbox_obj.as_raw(),
                    },
                    jni::sys::jvalue { i: -1 }, // page index unknown
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: ff_obj.as_raw() }],
            )?;
        }
    }
    Ok(list)
}
