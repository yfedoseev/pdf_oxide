//! PKCS#12 sign round-trip via the `pdf_sign_bytes_pades_opts` 5-arg
//! struct-shim (#546).
//!
//! Companion to `tests/test_pkcs12_signing.rs` (which exercises the
//! original 7-arg `pdf_sign_bytes`). This file is the first end-to-end
//! validation of `pdf_sign_bytes_pades_opts` — the entry point used by
//! the purego-Go and PHP-FFI bindings, both of which can't cleanly call
//! the 18-arg `pdf_sign_bytes_pades` directly.
//!
//! If this test passes but the PHP equivalent crashes, the bug is in
//! PHP's `PadesSignOptionsC` struct marshalling. If this test ALSO
//! crashes, the bug is in the Rust shim itself.
#![cfg(feature = "signatures")]
#![allow(clippy::missing_safety_doc)]
// The `pdf_oxide::ffi::*` re-exports lose their `unsafe fn` qualifier
// in some toolchain versions; `unused_unsafe` then fires on every
// FFI call site that should-by-spec be `unsafe`. Mirrors
// `test_pkcs12_signing.rs`'s same allow.
#![allow(unused_unsafe)]

use pdf_oxide::ffi::*;
use std::ffi::CString;
use std::ptr;

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[test]
fn pkcs12_sign_pdf_bytes_via_opts_shim_round_trip() {
    let p12_data =
        std::fs::read("tests/fixtures/test_signing.p12").expect("test_signing.p12 must exist");
    let password = cstring("testpass");
    let mut ec: i32 = -1;

    let cert_handle = unsafe {
        pdf_certificate_load_from_bytes(
            p12_data.as_ptr() as *const _,
            p12_data.len() as i32,
            password.as_ptr(),
            &mut ec,
        )
    };
    assert_eq!(ec, 0, "pdf_certificate_load_from_bytes returned error {ec}");
    assert!(!cert_handle.is_null(), "certificate handle must not be null");

    // Build a minimal PDF the same way as test_pkcs12_signing.rs.
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    let text = cstring("Signed via opts shim");
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);
    assert_eq!(
        unsafe { pdf_page_builder_font(page, cstring("Helvetica").as_ptr(), 12.0, &mut ec) },
        0
    );
    assert_eq!(unsafe { pdf_page_builder_text(page, text.as_ptr(), &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    let mut pdf_len: usize = 0;
    let pdf_ptr = unsafe { pdf_document_builder_build(builder, &mut pdf_len, &mut ec) };
    assert_eq!(ec, 0);
    assert!(!pdf_ptr.is_null());
    unsafe { pdf_document_builder_free(builder) };

    // Pack PadesSignOptionsC the same way the PHP/Ruby bindings do:
    // certificate_handle + level=0 (B-B), every other pointer NULL,
    // every count zero. This is the minimum-viable invocation.
    let opts = PadesSignOptionsC {
        certificate_handle: cert_handle as *const std::ffi::c_void,
        certs: ptr::null(),
        cert_lens: ptr::null(),
        n_certs: 0,
        crls: ptr::null(),
        crl_lens: ptr::null(),
        n_crls: 0,
        ocsps: ptr::null(),
        ocsp_lens: ptr::null(),
        n_ocsps: 0,
        tsa_url: ptr::null(),
        reason: ptr::null(),
        location: ptr::null(),
        level: 0, // B-B
    };
    // 13 pointer-sized fields + i32 level + tail padding to pointer
    // alignment. On 64-bit: 13*8 + 4 + 4pad = 112B; on 32-bit: 13*4 + 4 = 56B.
    let ptr_size = std::mem::size_of::<*const std::ffi::c_void>();
    let expected = 13 * ptr_size + std::mem::size_of::<i32>();
    let expected = expected.next_multiple_of(ptr_size);
    assert_eq!(
        std::mem::size_of::<PadesSignOptionsC>(),
        expected,
        "PadesSignOptionsC layout must be 13 pointers + i32 + tail-pad-to-pointer \
         (Ruby/PHP FFI bindings replicate this struct byte-for-byte)"
    );

    let mut signed_len: usize = 0;
    let signed_ptr =
        unsafe { pdf_sign_bytes_pades_opts(pdf_ptr, pdf_len, &opts, &mut signed_len, &mut ec) };
    unsafe { free_bytes(pdf_ptr as *mut _) };

    assert_eq!(ec, 0, "pdf_sign_bytes_pades_opts returned error {ec}");
    assert!(!signed_ptr.is_null(), "signed PDF must not be null");
    assert!(signed_len > pdf_len, "signed PDF must be larger than unsigned");

    let signed_bytes = unsafe { std::slice::from_raw_parts(signed_ptr, signed_len) };
    assert!(
        signed_bytes.starts_with(b"%PDF-"),
        "output must be a PDF (got {:?})",
        &signed_bytes[..8.min(signed_bytes.len())]
    );

    let content = String::from_utf8_lossy(signed_bytes);
    assert!(
        content.contains("/Sig") || content.contains("/ByteRange"),
        "signed PDF must contain /Sig or /ByteRange"
    );

    unsafe { free_bytes(signed_ptr as *mut _) };
    unsafe { pdf_certificate_free(cert_handle) };
}
