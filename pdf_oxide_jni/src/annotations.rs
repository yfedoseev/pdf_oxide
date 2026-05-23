//! JNI surface for `fyi.oxide.pdf.PdfPage.annotations()` — read
//! annotations for a page as `List<Annotation>`.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JObject};
use jni::sys::{jint, jlong};
use jni::EnvUnowned;
use pdf_oxide::annotation_types::AnnotationSubtype;
use pdf_oxide::annotations::LinkAction;
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: annotations handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// `Java_fyi_oxide_pdf_PdfPage_nativeAnnotations` — extract page
/// annotations as `ArrayList<Annotation>`.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeAnnotations<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.get_annotations(page_index as usize) {
            Ok(annots) => build_annotation_list(env, &annots, page_index),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Map pdf_oxide AnnotationSubtype to the Java AnnotationType enum
/// constant name (ordinal-by-name lookup via GetStaticField).
fn java_type_name(subtype: AnnotationSubtype) -> &'static str {
    match subtype {
        AnnotationSubtype::Text => "TEXT",
        AnnotationSubtype::Link => "LINK",
        AnnotationSubtype::FreeText => "FREE_TEXT",
        AnnotationSubtype::Line => "LINE",
        AnnotationSubtype::Square => "SQUARE",
        AnnotationSubtype::Circle => "CIRCLE",
        AnnotationSubtype::Highlight => "HIGHLIGHT",
        AnnotationSubtype::Underline => "UNDERLINE",
        AnnotationSubtype::Squiggly => "SQUIGGLY",
        AnnotationSubtype::StrikeOut => "STRIKEOUT",
        AnnotationSubtype::Stamp => "STAMP",
        AnnotationSubtype::FileAttachment => "FILE_ATTACHMENT",
        _ => "OTHER",
    }
}

fn build_annotation_list<'local>(
    env: &mut jni::Env<'local>,
    annots: &[pdf_oxide::annotations::Annotation],
    page_index: jint,
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let an_class = env.find_class(&JNIString::from("fyi/oxide/pdf/annotation/Annotation"))?;
    let an_ctor = env.get_method_id(
        &an_class,
        &JNIString::from("<init>"),
        jni_sig!("(Lfyi/oxide/pdf/annotation/AnnotationType;ILfyi/oxide/pdf/geometry/BBox;Ljava/lang/String;Ljava/lang/String;)V"),
    )?;
    let at_class = env.find_class(&JNIString::from("fyi/oxide/pdf/annotation/AnnotationType"))?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: annots.len() as i32,
            }],
        )?
    };

    for a in annots {
        // Annotation type enum constant via reflection-like GetStaticField.
        let name = JNIString::from(java_type_name(a.subtype_enum));
        let type_obj = env
            .get_static_field(
                &at_class,
                &name,
                jni_sig!("Lfyi/oxide/pdf/annotation/AnnotationType;"),
            )?
            .l()?;

        // BBox (zero-rect when /Rect is missing).
        let r = a.rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let bbox = unsafe {
            env.new_object_unchecked(
                &bbox_class,
                bbox_ctor,
                &[
                    jni::sys::jvalue { d: r[0] },
                    jni::sys::jvalue { d: r[1] },
                    jni::sys::jvalue { d: r[2] },
                    jni::sys::jvalue { d: r[3] },
                ],
            )?
        };

        let contents_obj: JObject = match &a.contents {
            Some(s) => env.new_string(s)?.into(),
            None => JObject::null(),
        };

        // URI from LinkAction::Uri if present.
        let uri_str: Option<String> = match &a.action {
            Some(LinkAction::Uri(u)) => Some(u.clone()),
            _ => None,
        };
        let uri_obj: JObject = match &uri_str {
            Some(s) => env.new_string(s)?.into(),
            None => JObject::null(),
        };

        let an_obj = unsafe {
            env.new_object_unchecked(
                &an_class,
                an_ctor,
                &[
                    jni::sys::jvalue {
                        l: type_obj.as_raw(),
                    },
                    jni::sys::jvalue { i: page_index },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue {
                        l: contents_obj.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: uri_obj.as_raw(),
                    },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: an_obj.as_raw() }],
            )?;
        }
    }
    Ok(list)
}
