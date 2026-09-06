//! A diagnostic raised inside the library must reach a C caller.
//!
//! The structured-warning channel existed only in Rust: `structured_warnings()`
//! on the document, mirrored in the Python binding, and nothing in the C ABI.
//! Every other binding — Go, C#, C++, Ruby, PHP, Swift, Kotlin, Scala, Clojure,
//! Dart, Elixir, Julia, R, Zig, ObjC, Node — reaches the library through
//! `src/ffi.rs` and so could not read a diagnostic of any kind.
//!
//! That is the other half of not writing prose into extracted content. A page
//! with no text now extracts as nothing, which is only safe if the reason is
//! retrievable; for eighteen of nineteen surfaces the injected sentence was the
//! only channel that had ever existed.
//!
//! These call the raw `pdf_*` functions the way a wrapper does — opaque handle,
//! error-code out-param, explicit free.

#![allow(clippy::missing_safety_doc)]
#![allow(unused_unsafe)]

use std::ffi::CStr;
use std::ptr;

use pdf_oxide::ffi::*;

/// A one-page PDF whose only content is a full-page image: a scan with no text
/// layer, which raises `no_text_layer` when the markdown is produced.
fn scanned_page_pdf() -> Vec<u8> {
    let content = b"q 200 0 0 120 0 0 cm /Im0 Do Q\n";
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| 90 + (i % 60) as u8).collect();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 120] \
         /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    off[5] = buf.len();
    buf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 16 /Height 16 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>\nstream\n",
            pixels.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&pixels);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// Read a malloc'd C string and free it the way a wrapper must.
unsafe fn take_c_string(p: *mut std::os::raw::c_char) -> String {
    assert!(!p.is_null(), "accessor returned null");
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { free_string(p) };
    s
}

/// Produce the markdown (which raises the diagnostic), then read it back
/// through the C ABI and confirm the category and page survived.
#[test]
fn test_diagnostic_crosses_the_c_abi_with_its_category_and_page() {
    let bytes = scanned_page_pdf();
    let mut err: i32 = -1;
    let doc = unsafe { pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut err) };
    assert!(!doc.is_null(), "load failed, error_code={err}");

    let md = unsafe { pdf_document_to_markdown(doc, 0, &mut err) };
    if !md.is_null() {
        let text = unsafe { take_c_string(md) };
        assert!(
            text.trim().is_empty(),
            "the page draws no text, so extraction must return none: {text:?}"
        );
    }

    let json = unsafe { take_c_string(pdf_document_structured_warnings(doc, &mut err)) };
    assert_eq!(err, 0, "accessor reported error_code={err}");
    assert!(
        json.contains("\"no_text_layer\""),
        "the reason must reach a C caller, not only a Rust one: {json}"
    );
    assert!(
        json.contains("\"page\":0"),
        "the diagnostic must name the page it belongs to: {json}"
    );

    unsafe { pdf_document_free(doc) };
}

/// The non-draining accessor is non-draining, and the draining one drains.
/// A batch pipeline depends on the difference.
#[test]
fn test_draining_accessor_drains_and_the_other_does_not() {
    let bytes = scanned_page_pdf();
    let mut err: i32 = -1;
    let doc = unsafe { pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut err) };
    assert!(!doc.is_null(), "load failed, error_code={err}");
    let md = unsafe { pdf_document_to_markdown(doc, 0, &mut err) };
    if !md.is_null() {
        unsafe { free_string(md) };
    }

    let first = unsafe { take_c_string(pdf_document_structured_warnings(doc, &mut err)) };
    let again = unsafe { take_c_string(pdf_document_structured_warnings(doc, &mut err)) };
    assert_eq!(first, again, "reading twice must not consume the entries");

    let drained = unsafe { take_c_string(pdf_document_take_structured_warnings(doc, &mut err)) };
    assert_eq!(drained, first, "the drain must return what was there");
    let after = unsafe { take_c_string(pdf_document_structured_warnings(doc, &mut err)) };
    assert_eq!(after, "[]", "after draining the sink is empty, got {after}");

    unsafe { pdf_document_free(doc) };
}

/// A null handle is rejected rather than dereferenced.
#[test]
fn test_null_handle_is_an_error_not_a_crash() {
    let mut err: i32 = 0;
    let p = pdf_document_structured_warnings(ptr::null_mut(), &mut err);
    assert!(p.is_null());
    assert_ne!(err, 0, "a null handle must set an error code");
}
