//! Thread-safety tests: concurrent reads and renders on shared PdfDocument.
//!
//! * `concurrent_document_reads_no_panic` — original test from #398: 8 threads
//!   each open-and-extract via FFI.
//! * `concurrent_renders_no_panic` — regression for #481: 8 threads render the
//!   same page simultaneously via the high-level Rust API.  The Rust API
//!   serialises render state via an internal Mutex, so this must never crash.
#![allow(clippy::missing_safety_doc)]
#![allow(unused_unsafe)]

use pdf_oxide::ffi::*;
use std::ffi::CString;

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[test]
fn concurrent_document_reads_no_panic() {
    use std::sync::Arc;

    let mut ec: i32 = -1;
    let builder = unsafe { pdf_document_builder_create(&mut ec) };
    assert_eq!(ec, 0);
    let page = unsafe { pdf_document_builder_letter_page(builder, &mut ec) };
    assert_eq!(ec, 0);
    assert_eq!(
        unsafe { pdf_page_builder_font(page, cstring("Helvetica").as_ptr(), 12.0, &mut ec) },
        0
    );
    assert_eq!(unsafe { pdf_page_builder_at(page, 72.0, 720.0, &mut ec) }, 0);
    let t = cstring("Concurrent read test");
    assert_eq!(unsafe { pdf_page_builder_text(page, t.as_ptr(), &mut ec) }, 0);
    assert_eq!(unsafe { pdf_page_builder_done(page, &mut ec) }, 0);
    let mut pdf_len: usize = 0;
    let pdf_ptr = unsafe { pdf_document_builder_build(builder, &mut pdf_len, &mut ec) };
    assert_eq!(ec, 0);
    let pdf_bytes: Arc<Vec<u8>> =
        Arc::new(unsafe { std::slice::from_raw_parts(pdf_ptr as *const u8, pdf_len) }.to_vec());
    unsafe { free_bytes(pdf_ptr) };
    unsafe { pdf_document_builder_free(builder) };

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let bytes = Arc::clone(&pdf_bytes);
            std::thread::spawn(move || {
                let mut ec: i32 = -1;
                let doc =
                    unsafe { pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut ec) };
                assert_eq!(ec, 0, "open failed in thread");
                let text_ptr = unsafe { pdf_document_extract_text(doc, 0, &mut ec) };
                assert_eq!(ec, 0, "extract_text failed in thread");
                let text = unsafe { std::ffi::CStr::from_ptr(text_ptr) }
                    .to_string_lossy()
                    .to_string();
                unsafe { free_string(text_ptr) };
                unsafe { pdf_document_free(doc) };
                assert!(text.contains("Concurrent"), "unexpected text content: {text:.100}");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
}

/// Regression test for #481: concurrent render calls must not crash.
///
/// The C# and JS bindings had a race condition where they released a lock before
/// the native render call completed, allowing two threads to call into the same
/// native handle simultaneously (UB).  Those fixes live in
/// `csharp/PdfOxide.Tests/ThreadSafetyTests.cs` and
/// `js/tests/worker-threads-safety.test.mjs`.
///
/// This Rust-level test verifies that the rendering pipeline itself (tiny-skia,
/// font rasteriser, etc.) is safe to call from multiple threads at the same time
/// when each thread has its own `Pdf` handle opened from shared bytes.  Each
/// handle is independent, so no lock is needed and this is a true concurrency
/// test of the underlying libraries.
#[cfg(feature = "rendering")]
#[test]
fn concurrent_renders_no_panic() {
    use pdf_oxide::api::{Pdf, RenderOptions};
    use std::sync::Arc;

    // Build a simple one-page PDF to render.
    let bytes: Arc<Vec<u8>> = Arc::new(
        Pdf::from_text("Concurrent render test")
            .expect("build PDF")
            .into_bytes(),
    );

    let opts = Arc::new(RenderOptions::with_dpi(72));

    // Each thread opens its own Pdf handle from the shared bytes and renders.
    // The handles are independent (no shared mutable state), so this exercises
    // thread-safety of the underlying font/rasteriser libraries.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let b = Arc::clone(&bytes);
            let o = Arc::clone(&opts);
            std::thread::spawn(move || {
                let mut pdf = Pdf::from_bytes((*b).clone()).expect("open PDF in thread");
                let img = pdf.render_page(0, Some(&o)).expect("render must not fail");
                assert!(!img.data.is_empty(), "rendered image data must not be empty");
                assert!(img.width > 0 && img.height > 0, "rendered dimensions must be positive");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("render thread panicked");
    }
}
