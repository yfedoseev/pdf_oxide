//! Thread-safety tests: concurrent reads and renders on shared PdfDocument.
//!
//! * `concurrent_document_reads_no_panic` — original test from #398: 8 threads
//!   each open-and-extract via FFI.
//! * `concurrent_renders_no_panic` — regression for #481: 8 threads render the
//!   same page simultaneously via the high-level Rust API.  The Rust API
//!   serialises render state via an internal Mutex, so this must never crash.
//! * `concurrent_render_page_fit_one_shared_handle_no_spurious_parse` —
//!   regression for #507: many threads call `pdf_render_page_fit` on a *single
//!   shared* FFI handle, exactly as the C# binding does.  Reproduces the #398
//!   Race A split-lock bug (seek and read on the shared reader split across two
//!   lock acquisitions) that surfaced as a spurious `[1000] invalid PDF
//!   structure or content stream` parse error.  Runs on every CI run, so the
//!   guard does not depend on the sometimes-flaky extended-features C# job.
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

/// Regression test for #507: concurrent `pdf_render_page_fit` on ONE shared
/// FFI document handle must never return a spurious parse error.
///
/// The C# binding (and the extended-features C# CI job: barcodes, rendering,
/// signatures, tsa-client, system-fonts) keeps a single native
/// `*mut PdfDocument` and calls render from multiple managed threads.  Before
/// the #398 Race A fix in `PdfDocument`, the object-header probe on the render
/// path acquired the shared reader lock twice — once to `seek`, once to
/// `read` — so a second thread could re-seek the shared file between the two,
/// making the first thread read a different object's bytes.  That produced an
/// intermittent `[1000] invalid PDF structure or content stream`
/// (`ERR_PARSE`), seen only under concurrency.
///
/// Unlike `concurrent_renders_no_panic` (each thread has its *own* handle, so
/// no shared reader), this test deliberately shares ONE handle to exercise the
/// internal `lock_or_recover()` serialisation.  Every render of a valid page
/// must succeed (`ERR_SUCCESS`); any `ERR_PARSE` is the regression.
#[cfg(feature = "rendering")]
#[test]
fn concurrent_render_page_fit_one_shared_handle_no_spurious_parse() {
    use pdf_oxide::api::Pdf;
    use std::sync::Arc;

    // ERR_SUCCESS / ERR_PARSE are private to the ffi module; mirror the
    // documented C ABI values (see src/ffi.rs).
    const ERR_SUCCESS: i32 = 0;
    const ERR_PARSE: i32 = 3;

    let bytes: Vec<u8> = Pdf::from_text("Shared-handle render race regression #507")
        .expect("build PDF")
        .into_bytes();

    // One shared native handle, opened once — this is the C# binding shape.
    let mut ec: i32 = -1;
    let doc = unsafe { pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut ec) };
    assert_eq!(ec, ERR_SUCCESS, "open_from_bytes failed");
    assert!(!doc.is_null(), "open_from_bytes returned null");

    // Raw pointers are !Send; pass the address as usize and cast back. This is
    // sound precisely because the #398/#507 contract makes shared `&`-access to
    // `PdfDocument` safe (the reader is Mutex-guarded via `lock_or_recover`).
    let doc_addr = doc as usize;

    const THREADS: usize = 8;
    const ITERS: usize = 16;

    let barrier = Arc::new(std::sync::Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || -> Result<(), String> {
                let doc = doc_addr as *mut _;
                b.wait(); // maximise overlap on the shared reader
                for i in 0..ITERS {
                    let mut ec: i32 = -1;
                    let img = unsafe { pdf_render_page_fit(doc, 0, 200, 200, 0, &mut ec) };
                    if ec == ERR_PARSE {
                        return Err(format!(
                            "iter {i}: spurious ERR_PARSE ([1000] invalid PDF \
                             structure) — #507 shared-handle race regressed"
                        ));
                    }
                    if ec != ERR_SUCCESS || img.is_null() {
                        return Err(format!(
                            "iter {i}: render failed ec={ec}, null={}",
                            img.is_null()
                        ));
                    }
                    unsafe { pdf_rendered_image_free(img) };
                }
                Ok(())
            })
        })
        .collect();

    let mut failures = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {},
            Ok(Err(e)) => failures.push(e),
            Err(_) => failures.push("render thread panicked".to_string()),
        }
    }

    unsafe { pdf_document_free(doc) };

    assert!(failures.is_empty(), "shared-handle render race (#507): {failures:?}");
}

/// Regression for #505: concurrent renders of an **embedded-font** PDF must
/// never raise a spurious `[1000]` parse error.
///
/// The C# `ThreadSafetyTests.RenderPageFit_ParallelForEach_DoesNotThrow` /
/// `RenderPage_ParallelForEach_DoesNotThrow` flaked on `main` CI because the
/// embedded-font cmap classifier memoised its result in a process-wide map
/// keyed on `Arc::as_ptr(font_bytes)`. When a font `Arc<Vec<u8>>` was dropped
/// (font-cache eviction / per-page renderer reset) and the allocator recycled
/// its address for an unrelated font, a stale `(is_byte_indexed,
/// has_unicode_cmap)` flipped the render branch and surfaced as
/// `ParseException [1000]`.
///
/// The existing #507 guard above uses `Pdf::from_text` (Helvetica, no
/// embedded font) so it never reached the classifier. This test mirrors the
/// C# fixture exactly — `Pdf::from_markdown(...)`, which embeds a font — and
/// hammers a single shared FFI handle so the classifier runs on every render
/// across all threads. Any `ERR_PARSE` is the #505 regression.
#[cfg(feature = "rendering")]
#[test]
fn concurrent_render_embedded_font_no_spurious_parse_505() {
    use pdf_oxide::api::Pdf;
    use std::sync::Arc;

    const ERR_SUCCESS: i32 = 0;
    const ERR_PARSE: i32 = 3;

    // Identical to the C# CreateTestDoc(): a 3-page markdown PDF. Markdown
    // rendering embeds a font, so every render exercises the embedded-font
    // cmap classifier that #505 is about.
    let bytes: Vec<u8> =
        Pdf::from_markdown("# Thread Safety\n\nPage 1.\n\n---\n\nPage 2.\n\n---\n\nPage 3.")
            .expect("build markdown PDF")
            .into_bytes();

    let mut ec: i32 = -1;
    let doc = unsafe { pdf_document_open_from_bytes(bytes.as_ptr(), bytes.len(), &mut ec) };
    assert_eq!(ec, ERR_SUCCESS, "open_from_bytes failed");
    assert!(!doc.is_null(), "open_from_bytes returned null");
    let doc_addr = doc as usize;

    // Use the real page count (the C# test reads doc.PageCount the same way);
    // markdown pagination isn't guaranteed to be 3 pages.
    let mut ec: i32 = -1;
    let pages = unsafe { pdf_document_get_page_count(doc, &mut ec) };
    assert_eq!(ec, ERR_SUCCESS, "page_count failed");
    assert!(pages >= 1, "expected at least one page, got {pages}");
    let pages = pages as usize;

    // Keep 8 threads — concurrency is the point: the #505 race only
    // manifests with simultaneous font alloc/drop across threads. Half the
    // iterations use `pdf_render_page`, which is a full-page 150-DPI render
    // (not a small target), so ITERS is kept modest to bound CI cost: 8×16
    // = 128 renders (~64 full-page of a tiny 3-page markdown doc) is ample
    // churn for the classifier race without slowing/flaking the suite.
    const THREADS: usize = 8;
    const ITERS: usize = 16;

    let barrier = Arc::new(std::sync::Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || -> Result<(), String> {
                let doc = doc_addr as *mut _;
                b.wait();
                for i in 0..ITERS {
                    let page = (i % pages) as i32;
                    let mut ec: i32 = -1;
                    // Alternate the two render entry points the C# tests use.
                    let img = if (t + i) % 2 == 0 {
                        unsafe { pdf_render_page_fit(doc, page, 200, 260, 0, &mut ec) }
                    } else {
                        unsafe { pdf_render_page(doc, page, 0, &mut ec) }
                    };
                    if ec == ERR_PARSE {
                        return Err(format!(
                            "thread {t} iter {i} page {page}: spurious ERR_PARSE \
                             ([1000]) — embedded-font classifier race (#505) regressed"
                        ));
                    }
                    if ec != ERR_SUCCESS || img.is_null() {
                        return Err(format!(
                            "thread {t} iter {i} page {page}: render failed ec={ec}, null={}",
                            img.is_null()
                        ));
                    }
                    unsafe { pdf_rendered_image_free(img) };
                }
                Ok(())
            })
        })
        .collect();

    let mut failures = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {},
            Ok(Err(e)) => failures.push(e),
            Err(_) => failures.push("render thread panicked".to_string()),
        }
    }

    unsafe { pdf_document_free(doc) };

    assert!(failures.is_empty(), "embedded-font render race (#505): {failures:?}");
}
