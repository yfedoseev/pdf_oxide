//! JNI surface for `fyi.oxide.pdf.PdfPage`.
//!
//! Lightweight per-page accessors that delegate into the parent
//! [`pdf_oxide::PdfDocument`]. The Java side keeps no native handle
//! of its own — it borrows the parent's, so closing the parent
//! invalidates all pages (the per-call `requireHandleForCallers()`
//! check on the Java side handles that).

use jni::errors::{Error as JniError, ThrowRuntimeExAndDefault};
use jni::objects::{JClass, JObject};
use jni::sys::{jboolean, jdoubleArray, jint, jlong, JNI_TRUE};
use jni::EnvUnowned;
use pdf_oxide::PdfDocument;

use crate::error::throw_pdf;

/// SAFETY: see [`crate::pdf_document::doc_ref`].
#[inline]
unsafe fn doc_ref<'h>(handle: jlong) -> &'h PdfDocument {
    debug_assert!(handle != 0, "JNI: PdfPage handle was 0");
    // SAFETY: caller upholds the unsafe fn contract — handle was checked by the JNI panic-barrier and Java's checked-handle pattern guarantees non-null + valid lifetime.
    unsafe { &*(handle as *const PdfDocument) }
}

/// `Java_fyi_oxide_pdf_PdfPage_nativeReadBBox` — read media-box or
/// crop-box as a fresh `double[4]` of `(x0, y0, x1, y1)`.
///
/// v0.3.53 always returns the media-box; the boolean parameter is
/// reserved for the future `getCropBox` path.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeReadBBox<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
    _is_media: jboolean,
) -> jdoubleArray {
    env.with_env(|env| -> Result<jdoubleArray, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.get_page_media_box(page_index as usize) {
            Ok((x0, y0, x1, y1)) => {
                let arr = env.new_double_array(4)?;
                let buf: [f64; 4] = [x0 as f64, y0 as f64, x1 as f64, y1 as f64];
                // jni 0.22: set_double_array_region is deprecated in favour of
                // the JDoubleArray-method form.
                arr.set_region(env, 0, &buf)?;
                Ok(arr.into_raw())
            },
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(std::ptr::null_mut())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `Java_fyi_oxide_pdf_PdfPage_nativeTextInRect` — extract text
/// within a rectangle of the page (PDF user-space coordinates).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeTextInRect<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
) -> jni::objects::JString<'local> {
    env.with_env(|env| -> Result<jni::objects::JString<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        // Java BBox is (x0,y0,x1,y1); Rust Rect is (x, y, w, h).
        let rect = pdf_oxide::geometry::Rect {
            x: x0 as f32,
            y: y0 as f32,
            width: (x1 - x0) as f32,
            height: (y1 - y0) as f32,
        };
        match doc.extract_text_in_rect(
            page_index as usize,
            rect,
            pdf_oxide::layout::RectFilterMode::Intersects,
        ) {
            Ok(s) => Ok(env.new_string(s)?),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(jni::objects::JString::default())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `Java_fyi_oxide_pdf_PdfPage_nativeRotation` — page rotation in
/// degrees (0, 90, 180, 270).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeRotation<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> jint {
    env.with_env(|env| -> Result<jint, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.get_page_rotation(page_index as usize) {
            Ok(r) => Ok(r as jint),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(0)
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeWords` — extract words for a page as a Java
/// `ArrayList<TextWord>`. Each word is constructed via the Java
/// `TextWord(String, BBox, float)` constructor + `BBox(double,
/// double, double, double)` constructor.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeWords<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.extract_words(page_index as usize) {
            Ok(words) => build_text_word_list(env, &words),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Construct an `ArrayList<TextWord>` from a slice of pdf_oxide Words.
fn build_text_word_list<'local>(
    env: &mut jni::Env<'local>,
    words: &[pdf_oxide::layout::Word],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let textword_class = env.find_class(&JNIString::from("fyi/oxide/pdf/text/TextWord"))?;
    let textword_ctor = env.get_method_id(
        &textword_class,
        &JNIString::from("<init>"),
        jni_sig!("(Ljava/lang/String;Lfyi/oxide/pdf/geometry/BBox;F)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: words.len() as i32,
            }],
        )?
    };

    for w in words {
        // Rust Rect is (x, y, width, height); convert to BBox (x0, y0, x1, y1).
        let bbox = unsafe {
            env.new_object_unchecked(
                &bbox_class,
                bbox_ctor,
                &[
                    jni::sys::jvalue { d: w.bbox.x as f64 },
                    jni::sys::jvalue { d: w.bbox.y as f64 },
                    jni::sys::jvalue {
                        d: (w.bbox.x + w.bbox.width) as f64,
                    },
                    jni::sys::jvalue {
                        d: (w.bbox.y + w.bbox.height) as f64,
                    },
                ],
            )?
        };
        let text = env.new_string(&w.text)?;
        let tw = unsafe {
            env.new_object_unchecked(
                &textword_class,
                textword_ctor,
                &[
                    jni::sys::jvalue { l: text.as_raw() },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue { f: 1.0_f32 },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: tw.as_raw() }],
            )?;
        }
    }
    Ok(list)
}

/// `nativeLines` — extract text lines as `ArrayList<TextLine>`.
/// Each line carries a nested `List<TextWord>` of its constituent
/// words.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeLines<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.extract_text_lines(page_index as usize) {
            Ok(lines) => build_text_line_list(env, &lines),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Construct `ArrayList<TextLine>` with nested `List<TextWord>` per line.
fn build_text_line_list<'local>(
    env: &mut jni::Env<'local>,
    lines: &[pdf_oxide::layout::TextLine],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let tl_class = env.find_class(&JNIString::from("fyi/oxide/pdf/text/TextLine"))?;
    let tl_ctor = env.get_method_id(
        &tl_class,
        &JNIString::from("<init>"),
        jni_sig!("(Ljava/lang/String;Lfyi/oxide/pdf/geometry/BBox;Ljava/util/List;)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: lines.len() as i32,
            }],
        )?
    };
    for line in lines {
        let words_list = build_text_word_list(env, &line.words)?;
        let bbox = unsafe {
            env.new_object_unchecked(
                &bbox_class,
                bbox_ctor,
                &[
                    jni::sys::jvalue {
                        d: line.bbox.x as f64,
                    },
                    jni::sys::jvalue {
                        d: line.bbox.y as f64,
                    },
                    jni::sys::jvalue {
                        d: (line.bbox.x + line.bbox.width) as f64,
                    },
                    jni::sys::jvalue {
                        d: (line.bbox.y + line.bbox.height) as f64,
                    },
                ],
            )?
        };
        let text = env.new_string(&line.text)?;
        let tl = unsafe {
            env.new_object_unchecked(
                &tl_class,
                tl_ctor,
                &[
                    jni::sys::jvalue { l: text.as_raw() },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue {
                        l: words_list.as_raw(),
                    },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: tl.as_raw() }],
            )?;
        }
    }
    Ok(list)
}

/// `nativeChars` — extract characters for a page as a Java
/// `ArrayList<TextChar>`. Each char is (codepoint, BBox, confidence).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeChars<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.extract_chars(page_index as usize) {
            Ok(chars) => build_text_char_list(env, &chars),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `nativeTables` — extract tables for a page as `ArrayList<Table>`.
/// Each Java Table carries a flat List<TableCell> with explicit row/
/// column indices; pdf_oxide's nested rows-of-cells structure is
/// flattened here.
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeTables<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.extract_tables(page_index as usize) {
            Ok(tables) => build_table_list(env, &tables),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn build_table_list<'local>(
    env: &mut jni::Env<'local>,
    tables: &[pdf_oxide::structure::table_extractor::Table],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let t_class = env.find_class(&JNIString::from("fyi/oxide/pdf/table/Table"))?;
    let t_ctor = env.get_method_id(
        &t_class,
        &JNIString::from("<init>"),
        jni_sig!("(Lfyi/oxide/pdf/geometry/BBox;IILjava/util/List;)V"),
    )?;
    let tc_class = env.find_class(&JNIString::from("fyi/oxide/pdf/table/TableCell"))?;
    let tc_ctor = env.get_method_id(
        &tc_class,
        &JNIString::from("<init>"),
        jni_sig!("(Ljava/lang/String;Lfyi/oxide/pdf/geometry/BBox;IIII)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let outer = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: tables.len() as i32,
            }],
        )?
    };

    for table in tables {
        // Build the flat cells list with explicit row/col indices.
        let total_cells: usize = table.rows.iter().map(|r| r.cells.len()).sum();
        let cells_list = unsafe {
            env.new_object_unchecked(
                &list_class,
                list_ctor,
                &[jni::sys::jvalue {
                    i: total_cells as i32,
                }],
            )?
        };
        for (row_idx, row) in table.rows.iter().enumerate() {
            for (col_idx, cell) in row.cells.iter().enumerate() {
                let cell_bbox = match cell.bbox {
                    Some(r) => unsafe {
                        env.new_object_unchecked(
                            &bbox_class,
                            bbox_ctor,
                            &[
                                jni::sys::jvalue { d: r.x as f64 },
                                jni::sys::jvalue { d: r.y as f64 },
                                jni::sys::jvalue {
                                    d: (r.x + r.width) as f64,
                                },
                                jni::sys::jvalue {
                                    d: (r.y + r.height) as f64,
                                },
                            ],
                        )?
                    },
                    None => unsafe {
                        env.new_object_unchecked(
                            &bbox_class,
                            bbox_ctor,
                            &[
                                jni::sys::jvalue { d: 0.0 },
                                jni::sys::jvalue { d: 0.0 },
                                jni::sys::jvalue { d: 0.0 },
                                jni::sys::jvalue { d: 0.0 },
                            ],
                        )?
                    },
                };
                let text = env.new_string(&cell.text)?;
                let tc = unsafe {
                    env.new_object_unchecked(
                        &tc_class,
                        tc_ctor,
                        &[
                            jni::sys::jvalue { l: text.as_raw() },
                            jni::sys::jvalue {
                                l: cell_bbox.as_raw(),
                            },
                            jni::sys::jvalue { i: row_idx as i32 },
                            jni::sys::jvalue { i: col_idx as i32 },
                            jni::sys::jvalue {
                                i: cell.rowspan as i32,
                            },
                            jni::sys::jvalue {
                                i: cell.colspan as i32,
                            },
                        ],
                    )?
                };
                unsafe {
                    env.call_method_unchecked(
                        &cells_list,
                        list_add,
                        jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                        &[jni::sys::jvalue { l: tc.as_raw() }],
                    )?;
                }
            }
        }

        let table_bbox = match table.bbox {
            Some(r) => unsafe {
                env.new_object_unchecked(
                    &bbox_class,
                    bbox_ctor,
                    &[
                        jni::sys::jvalue { d: r.x as f64 },
                        jni::sys::jvalue { d: r.y as f64 },
                        jni::sys::jvalue {
                            d: (r.x + r.width) as f64,
                        },
                        jni::sys::jvalue {
                            d: (r.y + r.height) as f64,
                        },
                    ],
                )?
            },
            None => unsafe {
                env.new_object_unchecked(
                    &bbox_class,
                    bbox_ctor,
                    &[
                        jni::sys::jvalue { d: 0.0 },
                        jni::sys::jvalue { d: 0.0 },
                        jni::sys::jvalue { d: 0.0 },
                        jni::sys::jvalue { d: 0.0 },
                    ],
                )?
            },
        };
        let t_obj = unsafe {
            env.new_object_unchecked(
                &t_class,
                t_ctor,
                &[
                    jni::sys::jvalue {
                        l: table_bbox.as_raw(),
                    },
                    jni::sys::jvalue {
                        i: table.rows.len() as i32,
                    },
                    jni::sys::jvalue {
                        i: table.col_count as i32,
                    },
                    jni::sys::jvalue {
                        l: cells_list.as_raw(),
                    },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &outer,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: t_obj.as_raw() }],
            )?;
        }
    }
    Ok(outer)
}

/// `nativeImages` — extract raster images for a page as a Java
/// `ArrayList<ExtractedImage>`. Each image is (bytes, format, bbox,
/// width, height). Bytes are the encoded stream (JPEG) or the raw
/// pixel buffer (RAW format).
#[no_mangle]
pub extern "system" fn Java_fyi_oxide_pdf_PdfPage_nativeImages<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    page_index: jint,
) -> JObject<'local> {
    env.with_env(|env| -> Result<JObject<'local>, JniError> {
        // SAFETY: handle checked by JNI panic-barrier; Java's AtomicLong checkHandle guarantees non-null + valid pointer.
        let doc = unsafe { doc_ref(handle) };
        match doc.extract_images(page_index as usize) {
            Ok(imgs) => build_extracted_image_list(env, &imgs),
            Err(e) => {
                throw_pdf(env, &e)?;
                Ok(JObject::null())
            },
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Construct an `ArrayList<ExtractedImage>` from a slice of PdfImages.
fn build_extracted_image_list<'local>(
    env: &mut jni::Env<'local>,
    imgs: &[pdf_oxide::extractors::PdfImage],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let img_class = env.find_class(&JNIString::from("fyi/oxide/pdf/image/ExtractedImage"))?;
    let img_ctor = env.get_method_id(
        &img_class,
        &JNIString::from("<init>"),
        jni_sig!("([BLfyi/oxide/pdf/image/ImageFormat;Lfyi/oxide/pdf/geometry/BBox;II)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;
    let fmt_class = env.find_class(&JNIString::from("fyi/oxide/pdf/image/ImageFormat"))?;
    let fmt_jpeg = env
        .get_static_field(
            &fmt_class,
            &JNIString::from("JPEG"),
            jni_sig!("Lfyi/oxide/pdf/image/ImageFormat;"),
        )?
        .l()?;
    let fmt_raw = env
        .get_static_field(
            &fmt_class,
            &JNIString::from("RAW"),
            jni_sig!("Lfyi/oxide/pdf/image/ImageFormat;"),
        )?
        .l()?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: imgs.len() as i32,
            }],
        )?
    };
    for img in imgs {
        let (bytes_arr, fmt_ref) = match img.data() {
            pdf_oxide::extractors::ImageData::Jpeg(bs) => {
                (env.byte_array_from_slice(bs)?, &fmt_jpeg)
            },
            pdf_oxide::extractors::ImageData::Raw { pixels, .. } => {
                (env.byte_array_from_slice(pixels)?, &fmt_raw)
            },
        };
        let (x0, y0, x1, y1) = match img.bbox() {
            Some(r) => (r.x as f64, r.y as f64, (r.x + r.width) as f64, (r.y + r.height) as f64),
            None => (0.0, 0.0, 0.0, 0.0),
        };
        let bbox = unsafe {
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
        };
        let img_obj = unsafe {
            env.new_object_unchecked(
                &img_class,
                img_ctor,
                &[
                    jni::sys::jvalue {
                        l: bytes_arr.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: fmt_ref.as_raw(),
                    },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue {
                        i: img.width() as i32,
                    },
                    jni::sys::jvalue {
                        i: img.height() as i32,
                    },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue {
                    l: img_obj.as_raw(),
                }],
            )?;
        }
    }
    Ok(list)
}

/// Construct an `ArrayList<TextChar>` from a slice of pdf_oxide TextChars.
fn build_text_char_list<'local>(
    env: &mut jni::Env<'local>,
    chars: &[pdf_oxide::layout::TextChar],
) -> Result<JObject<'local>, JniError> {
    use jni::jni_sig;
    use jni::strings::JNIString;
    let list_class = env.find_class(&JNIString::from("java/util/ArrayList"))?;
    let list_ctor = env.get_method_id(&list_class, &JNIString::from("<init>"), jni_sig!("(I)V"))?;
    let list_add =
        env.get_method_id(&list_class, &JNIString::from("add"), jni_sig!("(Ljava/lang/Object;)Z"))?;
    let tc_class = env.find_class(&JNIString::from("fyi/oxide/pdf/text/TextChar"))?;
    let tc_ctor = env.get_method_id(
        &tc_class,
        &JNIString::from("<init>"),
        jni_sig!("(ILfyi/oxide/pdf/geometry/BBox;F)V"),
    )?;
    let bbox_class = env.find_class(&JNIString::from("fyi/oxide/pdf/geometry/BBox"))?;
    let bbox_ctor =
        env.get_method_id(&bbox_class, &JNIString::from("<init>"), jni_sig!("(DDDD)V"))?;

    let list = unsafe {
        env.new_object_unchecked(
            &list_class,
            list_ctor,
            &[jni::sys::jvalue {
                i: chars.len() as i32,
            }],
        )?
    };
    for c in chars {
        let bbox = unsafe {
            env.new_object_unchecked(
                &bbox_class,
                bbox_ctor,
                &[
                    jni::sys::jvalue { d: c.bbox.x as f64 },
                    jni::sys::jvalue { d: c.bbox.y as f64 },
                    jni::sys::jvalue {
                        d: (c.bbox.x + c.bbox.width) as f64,
                    },
                    jni::sys::jvalue {
                        d: (c.bbox.y + c.bbox.height) as f64,
                    },
                ],
            )?
        };
        let tc = unsafe {
            env.new_object_unchecked(
                &tc_class,
                tc_ctor,
                &[
                    jni::sys::jvalue { i: c.char as i32 },
                    jni::sys::jvalue { l: bbox.as_raw() },
                    jni::sys::jvalue { f: 1.0_f32 },
                ],
            )?
        };
        unsafe {
            env.call_method_unchecked(
                &list,
                list_add,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                &[jni::sys::jvalue { l: tc.as_raw() }],
            )?;
        }
    }
    Ok(list)
}

// Silence unused warning until the rotation guard is wired.
#[allow(dead_code)]
const _: jboolean = JNI_TRUE;
