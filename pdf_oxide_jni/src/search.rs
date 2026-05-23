//! JNI surface for `fyi.oxide.pdf.PdfDocument.search` — text search
//! across the document. Returns `List<SearchMatch>` with the page
//! index, bbox, and matched text for each hit.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jint, jlong, JNI_TRUE};
use jni::EnvUnowned;
use pdf_oxide::search::{SearchOptions, TextSearcher};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: search handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// `nativeSearch` — search for a pattern across the document; returns
/// `ArrayList<SearchMatch>`. Each match is (pageIndex, bbox, text).
///
/// `literal=true` treats the pattern as literal text (escapes regex
/// metacharacters); `literal=false` uses the pattern as a regex.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfDocument_nativeSearch<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    pattern: JString<'local>,
    case_insensitive: jboolean,
    literal: jboolean,
    max_results: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        let pat: String = pattern.try_to_string(env)?;
        let opts = SearchOptions {
            case_insensitive: case_insensitive == JNI_TRUE,
            literal: literal == JNI_TRUE,
            whole_word: false,
            max_results: if max_results <= 0 {
                0
            } else {
                max_results as usize
            },
            page_range: None,
        };
        match TextSearcher::search(doc, &pat, &opts) {
            Ok(results) => build_search_match_list(env, &results),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn build_search_match_list<'local>(
    env: &mut jni::Env<'local>,
    results: &[pdf_oxide::search::SearchResult],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let sm_class = env.find_class(&JNIString::from("fyi/oxide/pdf/search/SearchMatch"))?;
    let sm_ctor = env.get_method_id(
        &sm_class,
        &JNIString::from("<init>"),
        jni_sig!("(ILfyi/oxide/pdf/geometry/BBox;Ljava/lang/String;)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: results.len() as i32,
            }],
        )?
    };

    for r in results {
        let bbox = unsafe {
            env.new_object_unchecked(
                &bbox_class,
                bbox_ctor,
                &[
                    jni::sys::jvalue { d: r.bbox.x as f64 },
                    jni::sys::jvalue { d: r.bbox.y as f64 },
                    jni::sys::jvalue {
                        d: (r.bbox.x + r.bbox.width) as f64,
                    },
                    jni::sys::jvalue {
                        d: (r.bbox.y + r.bbox.height) as f64,
                    },
                ],
            )?
        };
        let text = env.new_string(&r.text)?;
        let sm = unsafe {
            env.new_object_unchecked(
                &sm_class,
                sm_ctor,
                &[
                    jni::sys::jvalue { i: r.page as i32 },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue { l: text.as_raw() },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: sm.as_raw() }],
            )?;
        }
    }
    Ok(list)
}
