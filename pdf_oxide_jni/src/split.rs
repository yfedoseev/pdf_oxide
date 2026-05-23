//! JNI surface for {@code fyi.oxide.pdf.Pdf.splitByBookmarks*} —
//! the v0.3.50 #482 feature.
//!
//! Returns a Java `byte[][]` (array-of-byte-arrays) where each
//! element is one segment's PDF bytes, in document order. The
//! companion `nativeSplitSegmentCount` returns just the count for
//! quick preview.

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JByteArray, JClass, JObject};
use jni::sys::{jint, jobjectArray};
use jni::EnvUnowned;
use pdf_oxide::split_bookmarks::{
    plan_split_by_bookmarks, split_by_bookmarks_to_bytes, BookmarkLevel, SplitByBookmarksOptions,
};
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

fn opts_for(level: jint) -> SplitByBookmarksOptions {
    SplitByBookmarksOptions {
        level: BookmarkLevel::from_u32(if level < 0 { 0 } else { level as u32 }),
        ..Default::default()
    }
}

/// `Java_fyi_oxide_pdf_Pdf_nativePlanSplitCount` — return the number
/// of segments a split at `level` would produce, without actually
/// splitting. Useful for preview / progress estimation.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_Pdf_nativePlanSplitCount<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    src_bytes: JByteArray<'local>,
    level: jint,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        let bytes: Vec<u8> = env.convert_byte_array(&src_bytes)?;
        let doc = match PdfDocument::from_bytes(bytes) {
            Ok(d) => d,
            Err(e) => {
                throw_pdf(env, &e)?;
                return Ok(-1);
            },
        };
        let opts = opts_for(level);
        match plan_split_by_bookmarks(&doc, &opts) {
            Ok(segs) => Ok(segs.len() as jint),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(-1)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `Java_fyi_oxide_pdf_Pdf_nativeSplitBytes` — split the source PDF
/// at bookmark boundaries; returns a `byte[][]` with one element
/// per segment in document order.
///
/// Bookmark titles / file names are NOT returned by this entry
/// point; callers needing them should use the future
/// `nativeSplitBytesWithSegments` variant (Phase 3 follow-up — needs
/// a `SegmentInfo` value type marshaller).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_Pdf_nativeSplitBytes<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    src_bytes: JByteArray<'local>,
    level: jint,
) -> jobjectArray {
    env.with_env(|env| -> Result<jobjectArray, JniError> {
        let bytes: Vec<u8> = env.convert_byte_array(&src_bytes)?;
        let opts = opts_for(level);
        let parts = match split_by_bookmarks_to_bytes(&bytes, &opts) {
            Ok(p) => p,
            Err(e) => {
                throw_pdf(env, &e)?;
                return Ok(std::ptr::null_mut());
            },
        };
        // Build a Java byte[][] (object array of byte[]).
        let cls_name = jni::strings::JNIString::from("[B");
        let byte_array_class = env.find_class(&cls_name)?;
        let outer = env.new_object_array(parts.len() as i32, &byte_array_class, JObject::null())?;
        for (i, (_seg, bs)) in parts.iter().enumerate() {
            let inner: JByteArray = env.byte_array_from_slice(bs)?;
            // jni 0.22: set_object_array_element is deprecated;
            // use the JObjectArray method form.
            outer.set_element(env, i, &inner)?;
        }
        Ok(outer.into_raw())
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}
