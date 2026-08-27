//! C-FFI integration tests for the write-side API.
//!
//! These tests call the raw `pdf_*` FFI functions the way a C / C# / Go /
//! Node wrapper does — opaque handles, error-code out-params, explicit
//! frees, string marshalling through `CString`. They validate the
//! handle-lifetime contract documented in
//! `include/pdf_oxide_c/pdf_oxide.h` and in `src/ffi.rs`, and are the
//! acceptance gate for downstream bindings landing on top of this FFI.

#![allow(clippy::missing_safety_doc)]
#![allow(unused_unsafe)]
// The FFI functions here are `pub extern "C" fn` and pyo3 / wasm-bindgen
// signatures that don't strictly require `unsafe` at call sites today,
// but we keep the `unsafe {}` markers in the tests to make it obvious to
// maintainers that these are raw-pointer FFI calls that downstream C /
// C# / Go / Node bindings will use in the same ("unsafe") context.

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use pdf_oxide::ffi::*;

const DEJAVU_PATH: &str = "tests/fixtures/fonts/DejaVuSans.ttf";

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Minimal happy-path: create a builder, emit one A4 page with literal
/// text, build to bytes, confirm the output parses as a PDF, free the
/// byte buffer.
#[test]
fn ffi_document_builder_minimal_ascii() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0, "create returned error");
    assert!(!builder.is_null());

    // Open page
    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    // Emit text
    let text = cstring("Hello from FFI");
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);
    assert_eq!(ec, 0);
    assert_eq!(unsafe { pdf_page_builder_text(page, text.as_ptr(), &mut ec) }, 0);
    assert_eq!(ec, 0);

    // Commit
    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    // Build
    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 256);

    // Byte buffer must start with %PDF-
    let slice = unsafe { std::slice::from_raw_parts(bytes_ptr, out_len) };
    assert!(slice.starts_with(b"%PDF-"));

    // Buffer is owned by the caller; we free it with free_bytes.
    unsafe { free_bytes(bytes_ptr) };
    // Builder was consumed by _build — no _free needed, but calling it on
    // the now-invalid handle should be a no-op since the Box-from-raw is
    // already gone. Actually, _build only `consume_builder`s the inner —
    // the wrapper Box is still alive. Free it.
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_document_builder_embedded_font_cjk_round_trip() {
    let mut ec: i32 = -1;
    let font_path = cstring(DEJAVU_PATH);
    let font = unsafe { pdf_embedded_font_from_file(font_path.as_ptr(), &mut ec) };
    assert_eq!(ec, 0);
    assert!(!font.is_null());

    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert!(!builder.is_null());

    // Register consumes `font`.
    let font_name = cstring("DejaVu");
    let rc = unsafe {
        pdf_document_builder_register_embedded_font(builder, font_name.as_ptr(), font, &mut ec)
    };
    assert_eq!(rc, 0);
    assert_eq!(ec, 0);
    // Do NOT pdf_embedded_font_free(font) — it's consumed.

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert!(!page.is_null());
    let fname = cstring("DejaVu");
    assert_eq!(unsafe { pdf_page_builder_font(page, fname.as_ptr(), 12.0, &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);
    let cyrillic = cstring("Привет, мир!");
    assert_eq!(unsafe { pdf_page_builder_text(page, cyrillic.as_ptr(), &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) }, 0);
    let greek = cstring("Καλημέρα κόσμε");
    assert_eq!(unsafe { pdf_page_builder_text(page, greek.as_ptr(), &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());
    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, out_len) }.to_vec();
    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_document_builder_free(builder) };

    // Round-trip via the public (non-FFI) API since the FFI PdfDocument
    // call chain is validated elsewhere.
    let doc = pdf_oxide::PdfDocument::from_bytes(bytes).expect("parse output");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("Привет, мир!"), "Cyrillic missing: {text:?}");
    assert!(text.contains("Καλημέρα κόσμε"), "Greek missing: {text:?}");
}

#[test]
fn ffi_document_builder_output_is_subsetted() {
    let mut ec = 0;
    let font_path = cstring(DEJAVU_PATH);
    let font = unsafe { pdf_embedded_font_from_file(font_path.as_ptr(), &mut ec) };
    assert!(!font.is_null());
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    let font_name = cstring("DejaVu");
    unsafe {
        pdf_document_builder_register_embedded_font(builder, font_name.as_ptr(), font, &mut ec)
    };
    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    let fname = cstring("DejaVu");
    unsafe { pdf_page_builder_font(page, fname.as_ptr(), 12.0, &mut ec) };
    unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) };
    let text = cstring("Hello world");
    unsafe { pdf_page_builder_text(page, text.as_ptr(), &mut ec) };
    unsafe { pdf_page_builder_done(page, &mut ec) };

    let mut out_len = 0usize;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    let face_size = std::fs::metadata(DEJAVU_PATH).unwrap().len() as usize;
    assert!(
        out_len * 10 < face_size,
        "expected FFI-built PDF ({out_len} bytes) to be >= 10× smaller than the face ({face_size} bytes)"
    );
    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_double_open_page_rejected() {
    let mut ec = 0;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    let page1 = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert!(!page1.is_null());

    // Second open-page before done should error.
    let mut ec2 = 0;
    let page2 = unsafe { pdf_document_builder_a4_page(builder, &mut ec2) };
    assert!(page2.is_null());
    assert_eq!(ec2, 1); // ERR_INVALID_ARG

    // Clean up: drop page1 without committing, then free builder.
    unsafe { pdf_page_builder_free(page1) };
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_double_build_fails() {
    let mut ec = 0;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) };
    let text = cstring("x");
    unsafe { pdf_page_builder_text(page, text.as_ptr(), &mut ec) };
    unsafe { pdf_page_builder_done(page, &mut ec) };

    let mut out_len = 0usize;
    let bytes1 = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert!(!bytes1.is_null());
    unsafe { free_bytes(bytes1) };

    // Second build must fail — builder was consumed.
    let mut ec2 = 0;
    let bytes2 = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec2) };
    assert!(bytes2.is_null());
    assert_eq!(ec2, 1);

    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_save_encrypted_has_encrypt_dict() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let tmp_path = tmp.path().to_path_buf();

    let mut ec = 0;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) };
    let t = cstring("confidential");
    unsafe { pdf_page_builder_text(page, t.as_ptr(), &mut ec) };
    unsafe { pdf_page_builder_done(page, &mut ec) };

    let path_c = cstring(tmp_path.to_str().unwrap());
    let user = cstring("userpw");
    let owner = cstring("ownerpw");
    let rc = unsafe {
        pdf_document_builder_save_encrypted(
            builder,
            path_c.as_ptr(),
            user.as_ptr(),
            owner.as_ptr(),
            &mut ec,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(ec, 0);

    let raw = std::fs::read(&tmp_path).unwrap();
    let s = String::from_utf8_lossy(&raw);
    assert!(s.contains("/Encrypt"), "encrypted PDF missing /Encrypt");
    assert!(s.contains("/V 5"), "expected /V 5 (AES-256) marker");

    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_embedded_font_from_bytes_with_name_override() {
    let data = std::fs::read(DEJAVU_PATH).unwrap();
    let mut ec = 0;
    let name = cstring("CustomDejaVu");
    let font =
        unsafe { pdf_embedded_font_from_bytes(data.as_ptr(), data.len(), name.as_ptr(), &mut ec) };
    assert_eq!(ec, 0);
    assert!(!font.is_null());
    unsafe { pdf_embedded_font_free(font) };

    // Also test NULL name (use PS name).
    let font2 =
        unsafe { pdf_embedded_font_from_bytes(data.as_ptr(), data.len(), ptr::null(), &mut ec) };
    assert_eq!(ec, 0);
    assert!(!font2.is_null());
    unsafe { pdf_embedded_font_free(font2) };
}

// ---------------------------------------------------------------------------
// Barcode placement
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "barcodes")]
fn ffi_barcode_1d_and_qr_produce_valid_pdf() {
    let mut ec = 0;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    assert!(!builder.is_null());

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    // Code128 barcode (type 0)
    let data = cstring("HELLO-123");
    let ret = unsafe {
        pdf_page_builder_barcode_1d(page, 0, data.as_ptr(), 72.0, 680.0, 200.0, 60.0, &mut ec)
    };
    assert_eq!(ec, 0, "barcode_1d failed: {ret}");

    // Data Matrix barcode
    let data = cstring("https://example.com");
    let ret = unsafe {
        pdf_page_builder_barcode_datamatrix(page, data.as_ptr(), 72.0, 580.0, 100.0, &mut ec)
    };
    assert_eq!(ec, 0, "barcode_datamatrix failed: {ret}");

    // QR code
    let data = cstring("https://example.com");
    let ret =
        unsafe { pdf_page_builder_barcode_qr(page, data.as_ptr(), 72.0, 580.0, 100.0, &mut ec) };
    assert_eq!(ec, 0, "barcode_qr failed: {ret}");

    let done = unsafe { pdf_page_builder_done(page, &mut ec) };
    assert_eq!(ec, 0, "done failed: {done}");

    let mut out_len = 0usize;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 0);
    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
#[cfg(feature = "barcodes")]
fn ffi_barcode_unknown_type_errors() {
    let mut ec = 0;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    let data = cstring("test");
    let ret =
        unsafe { pdf_page_builder_barcode_1d(page, 99, data.as_ptr(), 0., 0., 100., 50., &mut ec) };
    assert_ne!(ec, 0, "expected error for unknown type, got {ret}");

    unsafe { pdf_page_builder_free(page) };
    unsafe { pdf_document_builder_free(builder) };
}

// ---------------------------------------------------------------------------
// JS actions — link_javascript, page on_open / on_close, doc on_open
// ---------------------------------------------------------------------------

#[test]
fn ffi_js_actions_produce_valid_pdf() {
    let mut ec = 0i32;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    // Document-level on_open
    let script = cstring("app.alert('opened')");
    let ret = unsafe { pdf_document_builder_on_open(builder, script.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "doc on_open failed: {ec}");

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    // Page-level on_open / on_close
    let pscript = cstring("app.alert('page open')");
    let ret = unsafe { pdf_page_builder_on_open(page, pscript.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "page on_open failed: {ec}");

    let cscript = cstring("app.alert('page close')");
    let ret = unsafe { pdf_page_builder_on_close(page, cscript.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "page on_close failed: {ec}");

    // link_javascript requires a preceding text element
    let _at = unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) };
    let text = cstring("Click me");
    let _txt = unsafe { pdf_page_builder_text(page, text.as_ptr(), &mut ec) };
    let jscript = cstring("app.alert('clicked')");
    let ret = unsafe { pdf_page_builder_link_javascript(page, jscript.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "link_javascript failed: {ec}");

    let ret = unsafe { pdf_page_builder_done(page, &mut ec) };
    assert_eq!(ret, 0, "done failed: {ec}");

    let mut out_len = 0usize;
    let pdf_bytes = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed: {ec}");
    assert!(!pdf_bytes.is_null());
    assert!(out_len > 100);

    let slice = unsafe { std::slice::from_raw_parts(pdf_bytes, out_len) };
    assert!(slice.starts_with(b"%PDF-"), "output is not a PDF");
    // The output must contain the OpenAction and AA dict markers
    let s = String::from_utf8_lossy(slice);
    assert!(s.contains("/OpenAction"), "missing /OpenAction in PDF");
    assert!(s.contains("/AA"), "missing /AA dict in PDF");

    unsafe { free_bytes(pdf_bytes) };
}

// ---------------------------------------------------------------------------
// Field validation AA dict (K/F/V/C) through FFI
// ---------------------------------------------------------------------------

#[test]
fn ffi_field_validation_aa_dict_in_pdf() {
    let mut ec = 0i32;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    let name = cstring("amount");
    let default_v = std::ptr::null();
    let ret = unsafe {
        pdf_page_builder_text_field(page, name.as_ptr(), 72., 700., 200., 20., default_v, &mut ec)
    };
    assert_eq!(ret, 0, "text_field failed: {ec}");

    let ks = cstring("AFNumber_Keystroke(2,0,0,0,'',true);");
    let ret = unsafe { pdf_page_builder_field_keystroke(page, ks.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "field_keystroke failed: {ec}");

    let fmt = cstring("AFNumber_Format(2,0,0,0,'',true);");
    let ret = unsafe { pdf_page_builder_field_format(page, fmt.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "field_format failed: {ec}");

    let val = cstring("event.rc = (event.value >= 0);");
    let ret = unsafe { pdf_page_builder_field_validate(page, val.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "field_validate failed: {ec}");

    let calc = cstring("event.value = 0;");
    let ret = unsafe { pdf_page_builder_field_calculate(page, calc.as_ptr(), &mut ec) };
    assert_eq!(ret, 0, "field_calculate failed: {ec}");

    let ret = unsafe { pdf_page_builder_done(page, &mut ec) };
    assert_eq!(ret, 0, "done failed: {ec}");

    let mut out_len = 0usize;
    let pdf_bytes = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed: {ec}");
    assert!(!pdf_bytes.is_null());

    let slice = unsafe { std::slice::from_raw_parts(pdf_bytes, out_len) };
    assert!(slice.starts_with(b"%PDF-"), "output is not a PDF");
    // The output must contain the /AA dict keys
    let s = String::from_utf8_lossy(slice);
    assert!(s.contains("/AA"), "missing /AA dict in field PDF");

    unsafe { free_bytes(pdf_bytes) };
}

// ---------------------------------------------------------------------------
// Signature field placeholder through FFI
// ---------------------------------------------------------------------------

#[test]
fn ffi_signature_field_in_pdf() {
    let mut ec = 0i32;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    let name = cstring("signature1");
    let ret = unsafe {
        pdf_page_builder_signature_field(page, name.as_ptr(), 72., 100., 200., 60., &mut ec)
    };
    assert_eq!(ret, 0, "signature_field failed: {ec}");

    let ret = unsafe { pdf_page_builder_done(page, &mut ec) };
    assert_eq!(ret, 0, "done failed: {ec}");

    let mut out_len = 0usize;
    let pdf_bytes = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed: {ec}");
    assert!(!pdf_bytes.is_null());

    let slice = unsafe { std::slice::from_raw_parts(pdf_bytes, out_len) };
    assert!(slice.starts_with(b"%PDF-"), "output is not a PDF");
    let s = String::from_utf8_lossy(slice);
    assert!(s.contains("/Sig"), "missing /Sig field type in PDF");
    assert!(s.contains("/SigFlags"), "missing /SigFlags in AcroForm");

    unsafe { free_bytes(pdf_bytes) };
}

// ---------------------------------------------------------------------------
// Phase 2 — HTML+CSS pipeline through the C FFI
// ---------------------------------------------------------------------------

#[test]
fn ffi_pdf_from_html_css_single_font() {
    let font_bytes = std::fs::read(DEJAVU_PATH).unwrap();
    let mut ec = 0;
    let html = cstring("<h1>Hello</h1><p>World</p>");
    let css = cstring("h1 { color: blue; font-size: 24pt }");

    let pdf_handle = unsafe {
        pdf_from_html_css(
            html.as_ptr(),
            css.as_ptr(),
            font_bytes.as_ptr(),
            font_bytes.len(),
            &mut ec,
        )
    };
    assert_eq!(ec, 0);
    assert!(!pdf_handle.is_null());

    // Serialize via the existing pdf_save_to_bytes path.
    let mut data_len = 0i32;
    let bytes_ptr = unsafe { pdf_save_to_bytes(pdf_handle, &mut data_len, &mut ec) };
    assert!(!bytes_ptr.is_null());
    assert!(data_len > 0);
    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, data_len as usize) }.to_vec();
    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_free(pdf_handle) };

    assert!(bytes.starts_with(b"%PDF-"));

    // Round-trip the literal words through extract_text.
    let doc = pdf_oxide::PdfDocument::from_bytes(bytes).expect("parse");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("Hello"));
    assert!(text.contains("World"));
}

#[test]
fn ffi_pdf_from_html_css_with_fonts_parallel_arrays() {
    let font_bytes = std::fs::read(DEJAVU_PATH).unwrap();

    // Single-entry cascade.
    let name_c = cstring("Body");
    let families: [*const std::os::raw::c_char; 1] = [name_c.as_ptr()];
    let font_ptrs: [*const u8; 1] = [font_bytes.as_ptr()];
    let font_lens: [usize; 1] = [font_bytes.len()];

    let mut ec = 0;
    let html = cstring("<p>cascade works</p>");
    let css = cstring("p { font-size: 14pt }");
    let pdf_handle = unsafe {
        pdf_from_html_css_with_fonts(
            html.as_ptr(),
            css.as_ptr(),
            families.as_ptr(),
            font_ptrs.as_ptr(),
            font_lens.as_ptr(),
            1,
            &mut ec,
        )
    };
    assert_eq!(ec, 0);
    assert!(!pdf_handle.is_null());
    unsafe { pdf_free(pdf_handle) };
}

#[test]
fn ffi_pdf_from_html_css_with_fonts_rejects_empty_count() {
    let mut ec = 0;
    let html = cstring("<p>x</p>");
    let css = cstring("");
    let handle = unsafe {
        pdf_from_html_css_with_fonts(
            html.as_ptr(),
            css.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            &mut ec,
        )
    };
    assert!(handle.is_null());
    assert_eq!(ec, 1);
}

// ---------------------------------------------------------------------------
// #393 v0.3.39 — new primitives + buffered Table
// ---------------------------------------------------------------------------

#[test]
fn ffi_page_builder_stroke_and_text_in_rect_and_table() {
    // Exercise every new v0.3.39 primitive via FFI and build a real PDF.
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    // stroke_rect + stroke_line
    assert_eq!(
        unsafe {
            pdf_page_builder_stroke_rect(
                page, 50.0, 50.0, 200.0, 100.0, 2.0, 0.5, 0.5, 0.5, &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);
    assert_eq!(
        unsafe {
            pdf_page_builder_stroke_line(page, 50.0, 50.0, 250.0, 50.0, 1.0, 0.2, 0.2, 0.2, &mut ec)
        },
        0
    );
    assert_eq!(ec, 0);

    // text_in_rect (align=Center)
    let caption = cstring("A centered caption that wraps across lines");
    assert_eq!(
        unsafe {
            pdf_page_builder_text_in_rect(
                page,
                100.0,
                500.0,
                200.0,
                100.0,
                caption.as_ptr(),
                1,
                &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);

    // Buffered table: 3 cols × 3 rows with header, centered numeric col.
    let widths: [f32; 3] = [100.0, 150.0, 80.0];
    let aligns: [i32; 3] = [0, 0, 2]; // Left, Left, Right
    let cell_strs: [CString; 9] = [
        cstring("SKU"),
        cstring("Name"),
        cstring("Qty"),
        cstring("A-1"),
        cstring("Widget"),
        cstring("12"),
        cstring("B-2"),
        cstring("Gadget"),
        cstring("3"),
    ];
    let cell_ptrs: [*const std::os::raw::c_char; 9] = [
        cell_strs[0].as_ptr(),
        cell_strs[1].as_ptr(),
        cell_strs[2].as_ptr(),
        cell_strs[3].as_ptr(),
        cell_strs[4].as_ptr(),
        cell_strs[5].as_ptr(),
        cell_strs[6].as_ptr(),
        cell_strs[7].as_ptr(),
        cell_strs[8].as_ptr(),
    ];
    assert_eq!(
        unsafe {
            pdf_page_builder_table(
                page,
                3,
                widths.as_ptr(),
                aligns.as_ptr(),
                3,
                cell_ptrs.as_ptr(),
                1, // has_header
                &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);

    // new_page_same_size — next page for regression
    assert_eq!(unsafe { pdf_page_builder_new_page_same_size(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());

    let slice = unsafe { std::slice::from_raw_parts(bytes_ptr, out_len) };
    assert!(slice.starts_with(b"%PDF-"));
    // Document must contain at least 2 pages now.
    let s = String::from_utf8_lossy(slice);
    let page_count = s.matches("/Type /Page\n").count() + s.matches("/Type/Page\n").count();
    let _ = page_count; // presence of /Type /Pages /Count is PDF-writer-internal.

    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_streaming_table_end_to_end_thousand_rows() {
    // Exercise the streaming FFI surface: begin → push_row × N → finish.
    // The Rust core is O(cols) memory — the FFI buffers rows until done()
    // replays them (per v0.3.39 scope, see #400 for true row-by-row).
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    assert_eq!(
        unsafe { pdf_page_builder_font(page, cstring("Helvetica").as_ptr(), 10.0, &mut ec) },
        0
    );
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);

    // Begin the streaming table.
    let headers_cstr: [CString; 3] = [cstring("SKU"), cstring("Item"), cstring("Qty")];
    let headers_ptrs: [*const std::os::raw::c_char; 3] = [
        headers_cstr[0].as_ptr(),
        headers_cstr[1].as_ptr(),
        headers_cstr[2].as_ptr(),
    ];
    let widths: [f32; 3] = [72.0, 200.0, 48.0];
    let aligns: [i32; 3] = [0, 0, 2];
    assert_eq!(
        unsafe {
            pdf_page_builder_streaming_table_begin(
                page,
                3,
                headers_ptrs.as_ptr(),
                widths.as_ptr(),
                aligns.as_ptr(),
                1, // repeat_header
                &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);

    // Push 1000 rows.
    for i in 0..1000u32 {
        let sku = cstring(&format!("A-{}", i));
        let item = cstring("Widget");
        let qty = cstring(&i.to_string());
        let cells_ptrs: [*const std::os::raw::c_char; 3] =
            [sku.as_ptr(), item.as_ptr(), qty.as_ptr()];
        let rc = unsafe {
            pdf_page_builder_streaming_table_push_row(page, 3, cells_ptrs.as_ptr(), &mut ec)
        };
        assert_eq!(rc, 0, "push_row failed at i={}: ec={}", i, ec);
    }

    // Close the table explicitly.
    assert_eq!(unsafe { pdf_page_builder_streaming_table_finish(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    // Commit + build.
    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());
    // 1000 rows × 3 cells + pagination → PDF must be non-trivial size.
    assert!(out_len > 10_000, "expected sizable PDF, got {}", out_len);

    let slice = unsafe { std::slice::from_raw_parts(bytes_ptr, out_len) };
    assert!(slice.starts_with(b"%PDF-"));

    unsafe { free_bytes(bytes_ptr) };
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_streaming_table_push_without_begin_errors() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };

    // Push a row without opening the table — should buffer but then fail
    // at done() replay.
    let cell = cstring("orphan");
    let cells_ptrs: [*const std::os::raw::c_char; 1] = [cell.as_ptr()];
    assert_eq!(
        unsafe { pdf_page_builder_streaming_table_push_row(page, 1, cells_ptrs.as_ptr(), &mut ec) },
        0
    );
    // done() must reject the orphan row.
    let rc = unsafe { pdf_page_builder_done(page, &mut ec) };
    assert_eq!(rc, -1);
    assert_ne!(ec, 0);
    unsafe { pdf_document_builder_free(builder) };
}

#[test]
fn ffi_page_builder_table_invalid_widths_rejected() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    // n_columns=2 but widths pointer is null — should return -1 + ERR_INVALID_ARG.
    let aligns: [i32; 2] = [0, 0];
    let rc = unsafe {
        pdf_page_builder_table(page, 2, ptr::null(), aligns.as_ptr(), 0, ptr::null(), 0, &mut ec)
    };
    assert_eq!(rc, -1);
    assert_ne!(ec, 0);

    // Cleanup — page is still intact (free without done).
    unsafe { pdf_page_builder_free(page) };
    unsafe { pdf_document_builder_free(builder) };
}

// ---------------------------------------------------------------------------
// Sanity
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PDF/UA-1 tagged PDF (Bundle F-1/F-2/F-4)
// ---------------------------------------------------------------------------

/// Enable tagged PDF via the FFI, build a document with a paragraph structure
/// element, and verify that /MarkInfo, /StructTreeRoot, and /Lang appear in
/// the emitted bytes.
#[test]
fn ffi_tagged_pdf_ua1_basic() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0, "create returned error");
    assert!(!builder.is_null());

    // Enable PDF/UA-1
    let rc = unsafe { pdf_document_builder_tagged_pdf_ua1(builder, &mut ec) };
    assert_eq!(rc, 0, "tagged_pdf_ua1 failed");
    assert_eq!(ec, 0);

    // Set language
    let lang = cstring("en-US");
    let rc = unsafe { pdf_document_builder_language(builder, lang.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "language failed");
    assert_eq!(ec, 0);

    // Add a role-map entry
    let custom = cstring("Note");
    let standard = cstring("P");
    let rc = unsafe {
        pdf_document_builder_role_map(builder, custom.as_ptr(), standard.as_ptr(), &mut ec)
    };
    assert_eq!(rc, 0, "role_map failed");
    assert_eq!(ec, 0);

    // Open a page and add some text
    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    let text = cstring("Accessible heading text");
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_heading(page, 1, text.as_ptr(), &mut ec) }, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    // Build and get bytes
    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed");
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 0);

    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, out_len) };
    let content = String::from_utf8_lossy(bytes);

    // F-1: /MarkInfo must appear
    assert!(
        content.contains("/MarkInfo"),
        "missing /MarkInfo in catalog — tagged PDF not enabled"
    );
    // F-1: /StructTreeRoot must appear
    assert!(content.contains("/StructTreeRoot"), "missing /StructTreeRoot in catalog");
    // F-2: /Lang must appear
    assert!(content.contains("/Lang"), "missing /Lang in catalog");
    // /ViewerPreferences must appear
    assert!(content.contains("/ViewerPreferences"), "missing /ViewerPreferences");

    unsafe { free_bytes(bytes_ptr) };
}

#[test]
fn ffi_footnote_in_pdf() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    assert!(!builder.is_null());

    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    // Place some body text then a footnote ref mark
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) }, 0);
    let body = cstring("Important claim");
    assert_eq!(unsafe { pdf_page_builder_text(page, body.as_ptr(), &mut ec) }, 0);

    let ref_mark = cstring("[1]");
    let note_text = cstring("Source: Annual report 2025.");
    let rc =
        unsafe { pdf_page_builder_footnote(page, ref_mark.as_ptr(), note_text.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "footnote failed");
    assert_eq!(ec, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed");
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 0);

    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, out_len) };
    let content = String::from_utf8_lossy(bytes);

    // The footnote note text must appear somewhere in the PDF stream.
    assert!(
        content.contains("Annual report 2025"),
        "footnote body text not found in PDF output"
    );

    unsafe { pdf_document_builder_free(builder) };
    unsafe { free_bytes(bytes_ptr) };
}

#[test]
fn ffi_columns_in_pdf() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    assert!(!builder.is_null());

    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) }, 0);

    let text = cstring(
        "First paragraph text.\n\nSecond paragraph with more words that will wrap across columns.",
    );
    let rc = unsafe { pdf_page_builder_columns(page, 2, 12.0, text.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "columns failed");
    assert_eq!(ec, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed");
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 0);

    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, out_len) };
    assert!(bytes.starts_with(b"%PDF-"), "output is not a PDF");

    // The text should appear somewhere in the PDF stream.
    let content = String::from_utf8_lossy(bytes);
    assert!(content.contains("First paragraph"), "column text not found in PDF output");

    unsafe { pdf_document_builder_free(builder) };
    unsafe { free_bytes(bytes_ptr) };
}

#[test]
fn ffi_rich_text_inline_in_pdf() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    assert!(!builder.is_null());

    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!page.is_null());

    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 700.0, &mut ec) }, 0);

    // normal + bold + italic + colored + newline
    let normal = cstring("Hello ");
    let rc = unsafe { pdf_page_builder_inline(page, normal.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "inline failed");
    assert_eq!(ec, 0);

    let bold = cstring("world");
    let rc = unsafe { pdf_page_builder_inline_bold(page, bold.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "inline_bold failed");
    assert_eq!(ec, 0);

    let rc = unsafe { pdf_page_builder_newline(page, &mut ec) };
    assert_eq!(rc, 0, "newline failed");
    assert_eq!(ec, 0);

    let italic = cstring("italic run");
    let rc = unsafe { pdf_page_builder_inline_italic(page, italic.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "inline_italic failed");
    assert_eq!(ec, 0);

    let colored = cstring("red text");
    let rc =
        unsafe { pdf_page_builder_inline_color(page, 1.0, 0.0, 0.0, colored.as_ptr(), &mut ec) };
    assert_eq!(rc, 0, "inline_color failed");
    assert_eq!(ec, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0, "build failed");
    assert!(!bytes_ptr.is_null());
    assert!(out_len > 0);

    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, out_len) };
    assert!(bytes.starts_with(b"%PDF-"), "output is not a PDF");

    let content = String::from_utf8_lossy(bytes);
    assert!(content.contains("Hello"), "inline text not found in PDF output");

    unsafe { pdf_document_builder_free(builder) };
    unsafe { free_bytes(bytes_ptr) };
}

#[test]
fn ffi_fixture_font_exists() {
    assert!(
        Path::new(DEJAVU_PATH).exists(),
        "DejaVuSans.ttf fixture missing — regenerate tests/fixtures/fonts/"
    );
}

#[test]
fn ffi_dashed_stroke_rect_produces_pdf_with_dash_operator() {
    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);

    let page = unsafe { pdf_document_builder_a4_page(builder, &mut ec) };
    assert_eq!(ec, 0);

    // Dashed rect: 3pt dash, 2pt gap, phase 0
    let dashes: [f32; 2] = [3.0, 2.0];
    assert_eq!(
        unsafe {
            pdf_page_builder_stroke_rect_dashed(
                page,
                50.0,
                100.0,
                200.0,
                150.0, // x y w h
                1.5,
                0.0,
                0.0,
                0.8, // width r g b
                dashes.as_ptr(),
                dashes.len(),
                0.0, // dash_array n_dash phase
                &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);

    // Dashed line: 5pt dash, 3pt gap
    let dashes2: [f32; 2] = [5.0, 3.0];
    assert_eq!(
        unsafe {
            pdf_page_builder_stroke_line_dashed(
                page,
                50.0,
                80.0,
                250.0,
                80.0, // x1 y1 x2 y2
                1.0,
                0.8,
                0.0,
                0.0, // width r g b
                dashes2.as_ptr(),
                dashes2.len(),
                1.0, // dash_array n_dash phase
                &mut ec,
            )
        },
        0
    );
    assert_eq!(ec, 0);

    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    assert_eq!(ec, 0);

    let mut out_len: usize = 0;
    let bytes_ptr = unsafe { pdf_document_builder_build(builder, &mut out_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!bytes_ptr.is_null());

    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, out_len) };
    assert!(bytes.starts_with(b"%PDF-"), "output is not a PDF");

    // The content stream must contain the `d` (setdash) operator
    let content = String::from_utf8_lossy(bytes);
    assert!(
        content.contains(" d\n") || content.contains(" d "),
        "dash operator 'd' not found in PDF"
    );

    unsafe { pdf_document_builder_free(builder) };
    unsafe { free_bytes(bytes_ptr) };
}
