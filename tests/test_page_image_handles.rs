//! Integration tests for the enumerate-then-materialize image API.
//!
//! Verifies that `page_image_handles` returns correct metadata without
//! decompressing streams, and that `decode()` / `raw_compressed_bytes()`
//! materialise the image on demand.

use pdf_oxide::elements::{ContentElement, ImageContent, ImageFormat};
use pdf_oxide::extractors::images::PdfFilter;
use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::{PdfWriter, PdfWriterConfig};
use pdf_oxide::PdfDocument;

// Minimal valid 1×1 white JPEG (SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI)
const MINIMAL_JPEG: &[u8] = &[
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
    0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
    0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
    0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
    0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xD9,
];

/// Build a minimal PDF containing a single JPEG image on page 0.
fn build_pdf_with_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut writer = PdfWriter::with_config(PdfWriterConfig::default());

    let bbox = Rect::new(0.0, 0.0, width as f32, height as f32);
    let image_content =
        ImageContent::new(bbox, ImageFormat::Jpeg, MINIMAL_JPEG.to_vec(), width, height);

    let mut page = writer.add_a4_page();
    page.add_element(&ContentElement::Image(image_content));
    page.finish();

    writer.finish().expect("PDF write failed")
}

#[test]
fn page_image_handles_returns_one_handle_for_single_jpeg() {
    let pdf_bytes = build_pdf_with_jpeg(100, 80);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");

    assert_eq!(handles.len(), 1, "expected exactly one image handle");
    let h = &handles[0];
    // MINIMAL_JPEG is intrinsically 1×1 (SOF0 marker); display size is separate
    assert_eq!(h.width, 1);
    assert_eq!(h.height, 1);
    assert!(!h.is_inline);
    assert_eq!(h.paint_order, 0);
}

#[test]
fn page_image_handles_jpeg_has_dct_filter() {
    let pdf_bytes = build_pdf_with_jpeg(50, 50);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");
    assert!(!handles.is_empty());

    let h = &handles[0];
    assert!(
        h.filter_chain.contains(&PdfFilter::DCTDecode),
        "JPEG XObject must report DCTDecode in filter_chain, got {:?}",
        h.filter_chain
    );
}

#[test]
fn page_image_handles_decode_produces_valid_image() {
    let pdf_bytes = build_pdf_with_jpeg(1, 1);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");
    assert_eq!(handles.len(), 1);

    let handle = handles.into_iter().next().unwrap();
    let image = handle.decode().expect("decode");

    assert_eq!(image.width(), 1);
    assert_eq!(image.height(), 1);
}

#[test]
fn page_image_handles_raw_compressed_bytes_non_empty() {
    let pdf_bytes = build_pdf_with_jpeg(1, 1);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");
    let handle = handles.into_iter().next().expect("handle");

    let raw = handle.raw_compressed_bytes().expect("raw bytes");
    assert!(!raw.is_empty(), "raw compressed bytes must be non-empty");
}

#[test]
fn page_image_handles_filter_then_decode_skips_small_images() {
    // MINIMAL_JPEG is intrinsically 1×1; filter for >= 100×100 should skip it
    let pdf_bytes = build_pdf_with_jpeg(200, 200);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");

    let decoded: Vec<_> = handles
        .into_iter()
        .filter(|h| h.width >= 100 && h.height >= 100)
        .map(|h| h.decode())
        .collect::<Result<_, _>>()
        .expect("decode");

    // The 1×1 JPEG is smaller than the 100×100 threshold — zero decoded
    assert_eq!(decoded.len(), 0);
}

#[test]
fn page_image_handles_empty_page_returns_empty_vec() {
    let mut writer = PdfWriter::with_config(PdfWriterConfig::default());
    writer.add_a4_page().finish();
    let pdf_bytes = writer.finish().expect("PDF write");

    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    let handles = doc.page_image_handles(0).expect("page_image_handles");

    assert!(handles.is_empty(), "empty page must yield zero handles");
}

#[test]
fn pdf_filter_from_name_roundtrip() {
    assert_eq!(PdfFilter::from_name("DCTDecode"), PdfFilter::DCTDecode);
    assert_eq!(PdfFilter::from_name("DCT"), PdfFilter::DCTDecode);
    assert_eq!(PdfFilter::from_name("FlateDecode"), PdfFilter::FlateDecode);
    assert_eq!(PdfFilter::from_name("Fl"), PdfFilter::FlateDecode);
    assert_eq!(PdfFilter::from_name("JPXDecode"), PdfFilter::JPXDecode);
    assert_eq!(PdfFilter::from_name("LZWDecode"), PdfFilter::LZWDecode);
    assert_eq!(PdfFilter::from_name("CCITTFaxDecode"), PdfFilter::CCITTFaxDecode);
    assert_eq!(
        PdfFilter::from_name("UnknownFilter"),
        PdfFilter::Other("UnknownFilter".to_string())
    );
}

// ── Test A: Form XObject does not produce a handle ────────────────────────────

/// Build a minimal PDF whose page Resources/XObject dictionary contains a
/// Form XObject (not an Image XObject).  The PDF has no actual image content
/// inside the Form; `page_image_handles` correctly returns an empty vec because
/// there are no Image XObjects reachable from the page content (recursion into
/// the Form is performed, but the Form itself contains nothing paintable).
fn build_pdf_with_form_xobject_only() -> Vec<u8> {
    // We construct the PDF bytes manually so we can place a /Subtype /Form
    // XObject in the resources without using the writer (which only writes
    // Image XObjects).
    //
    // Object layout:
    //   1 Catalog → 2
    //   2 Pages   → [3]
    //   3 Page    → Resources: { XObject: { Im0: 5 0 R } }, Contents: 4 0 R
    //   4 Content stream  (just a `Do` operator that paints the form)
    //   5 Form XObject stream  (/Type /XObject /Subtype /Form)
    let content = b"q /Im0 Do Q";
    let form_stream_data = b""; // empty form content
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>\nendobj\n",
    );

    let off4 = pdf.len();
    let content_str = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(content_str.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    let form_str = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Length {} >>\nstream\n",
        form_stream_data.len()
    );
    pdf.extend_from_slice(form_str.as_bytes());
    pdf.extend_from_slice(form_stream_data);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 6\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    pdf
}

#[test]
fn form_xobject_does_not_produce_image_handle() {
    // The Form is recursed into, but it contains no Image XObjects or inline
    // images, so the handle list is correctly empty. This test exercises the
    // recursion path (cycle detection, Resources fallback, Form Matrix, etc.)
    // without producing any handles.
    let pdf_bytes = build_pdf_with_form_xobject_only();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    let handles = doc.page_image_handles(0).expect("page_image_handles");
    assert_eq!(
        handles.len(),
        0,
        "Form with no images inside must yield 0 handles; got {}",
        handles.len()
    );
}

// ── Test B: N images → paint_order is [0, 1, 2] ──────────────────────────────

fn build_pdf_with_n_jpegs(n: usize) -> Vec<u8> {
    let mut writer = PdfWriter::with_config(PdfWriterConfig::default());
    let mut page = writer.add_a4_page();
    for i in 0..n {
        let x = (i as f32) * 50.0;
        let bbox = Rect::new(x, 0.0, 40.0, 40.0);
        let img = ImageContent::new(bbox, ImageFormat::Jpeg, MINIMAL_JPEG.to_vec(), 1, 1);
        page.add_element(&ContentElement::Image(img));
    }
    page.finish();
    writer.finish().expect("PDF write failed")
}

#[test]
fn three_images_have_paint_order_zero_one_two() {
    let pdf_bytes = build_pdf_with_n_jpegs(3);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    let handles = doc.page_image_handles(0).expect("page_image_handles");

    assert_eq!(handles.len(), 3, "expected 3 handles");
    let paint_orders: Vec<usize> = handles.iter().map(|h| h.paint_order).collect();
    assert_eq!(
        paint_orders,
        vec![0, 1, 2],
        "paint_order values must be [0, 1, 2] in content-stream order, got {:?}",
        paint_orders
    );
}

// ── Test C: JPEG raw bytes start with SOI marker [0xFF, 0xD8] ─────────────────

#[test]
fn raw_compressed_bytes_starts_with_jpeg_soi_marker() {
    let pdf_bytes = build_pdf_with_jpeg(1, 1);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    let handles = doc.page_image_handles(0).expect("page_image_handles");
    let handle = handles.into_iter().next().expect("handle");

    let raw = handle.raw_compressed_bytes().expect("raw bytes");
    assert!(
        raw.len() >= 2,
        "raw bytes must be at least 2 bytes for SOI check, got {}",
        raw.len()
    );
    assert_eq!(
        &raw[..2],
        &[0xFF, 0xD8],
        "JPEG raw bytes must start with SOI marker FF D8, got {:02X} {:02X}",
        raw[0],
        raw[1]
    );
}

// ── Test D: Out-of-bounds page index returns Err ──────────────────────────────

#[test]
fn out_of_bounds_page_index_returns_err() {
    let pdf_bytes = build_pdf_with_jpeg(1, 1);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    // The PDF has only 1 page (index 0); index 999 must be an error.
    let result = doc.page_image_handles(999);
    assert!(
        result.is_err(),
        "page_image_handles(999) on a 1-page PDF must return Err, got Ok"
    );
}

// ── Test E: Inline image decode (placeholder — no writer-level support yet) ───

// The pdf_oxide writer does not currently expose an API for inserting inline
// images (BI/ID/EI sequences) into content streams. Constructing a correct
// inline-image PDF by hand requires matching the exact byte offsets expected
// by the content-stream parser, which is fragile without writer support.
//
// This test is left as an `#[ignore]` placeholder until the writer gains
// first-class inline-image support, at which point it should be filled in
// following the pattern of the decode test above.
#[test]
#[ignore = "inline-image writer support not yet available; fill in once PdfWriter exposes BI/ID/EI"]
fn inline_image_decode_produces_valid_image() {
    // TODO: construct a PDF with an inline image and assert decode() succeeds.
}

// ── Positive Form XObject recursion tests (addresses maintainer review #588) ──

/// Build a minimal PDF where the page paints a Form XObject, and the Form
/// itself paints a real Image XObject (the 1×1 MINIMAL_JPEG).
///
/// This exercises the new `page_image_handles` Form recursion path:
///   Page content Do → Form → (Form Resources) Image Do
/// The returned handle must have correct paint_order, dimensions, filter chain,
/// and decode() must succeed.
fn build_pdf_with_image_inside_form() -> Vec<u8> {
    // Object layout (all generation 0):
    //   1 Catalog
    //   2 Pages (Kids=[3])
    //   3 Page (Resources: XObject → Fm0 6 0 R, Contents 4 0 R)
    //   4 Page content stream:  "q /Fm0 Do Q"
    //   5 Image XObject (MINIMAL_JPEG, 1×1)
    //   6 Form XObject (Resources: XObject → Im0 5 0 R, content: "q /Im0 Do Q")
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    // Page references the Form in its Resources and a content stream
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Contents 4 0 R /Resources << /XObject << /Fm0 6 0 R >> >> >>\nendobj\n",
    );

    // Page content: just invoke the Form
    let page_content = b"q /Fm0 Do Q";
    let off4 = pdf.len();
    let content_str = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(content_str.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // 5: The actual Image XObject (JPEG)
    let img_data = MINIMAL_JPEG;
    let off5 = pdf.len();
    let img_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
         /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",
        img_data.len()
    );
    pdf.extend_from_slice(img_hdr.as_bytes());
    pdf.extend_from_slice(img_data);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // 6: The Form XObject. It has its own Resources containing the Image,
    //    and its content stream paints that Image.
    let form_content = b"q /Im0 Do Q";
    let form_res = b"<< /XObject << /Im0 5 0 R >> >>";
    let off6 = pdf.len();
    let form_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources {} /Length {} >>\nstream\n",
        String::from_utf8_lossy(form_res),
        form_content.len()
    );
    pdf.extend_from_slice(form_hdr.as_bytes());
    pdf.extend_from_slice(form_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // xref + trailer
    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 7\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off4).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off5).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off6).as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    pdf
}

/// Positive test: images inside Form XObjects are now enumerated by
/// page_image_handles (the core requirement from the maintainer review).
#[test]
fn form_xobject_recursion_returns_inner_image_handles() {
    let pdf_bytes = build_pdf_with_image_inside_form();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");

    let handles = doc.page_image_handles(0).expect("page_image_handles");

    assert_eq!(
        handles.len(),
        1,
        "expected exactly one image handle from inside the Form; got {}",
        handles.len()
    );

    let h0 = handles
        .into_iter()
        .next()
        .expect("at least one handle from Form recursion");
    assert_eq!(h0.paint_order, 0, "the single image must have paint_order 0");
    assert_eq!(h0.width, 1);
    assert_eq!(h0.height, 1);
    assert!(
        h0.filter_chain.contains(&PdfFilter::DCTDecode),
        "expected DCTDecode filter for the JPEG inside the Form, got {:?}",
        h0.filter_chain
    );

    // Decode must succeed and produce a valid 1×1 image.
    let decoded = h0.decode().expect("decode image from inside Form");
    assert_eq!(decoded.width(), 1);
    assert_eq!(decoded.height(), 1);

    // (bbox/rotation already validated by construction in the handle builders)
}

/// Basic cycle-safety smoke test for Form recursion.
/// A Form that references itself must not cause infinite recursion or panic.
#[test]
fn form_xobject_self_reference_is_safe() {
    // Minimal self-referential Form (its own Resources XObject dict points at itself).
    // Page content invokes it once. We expect 0 image handles and no crash/loop.
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Contents 4 0 R /Resources << /XObject << /Fm0 5 0 R >> >> >>\nendobj\n",
    );

    let page_content = b"q /Fm0 Do Q";
    let off4 = pdf.len();
    let c4 = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(c4.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Form whose Resources/XObject contains a reference back to itself (Fm0).
    // Content stream is empty (no images anyway).
    let form_res = b"<< /XObject << /Fm0 5 0 R >> >>";
    let form_stream = b"";
    let off5 = pdf.len();
    let c5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources {} /Length {} >>\nstream\n",
        String::from_utf8_lossy(form_res),
        form_stream.len()
    );
    pdf.extend_from_slice(c5.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 6\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for off in [off1, off2, off3, off4, off5] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf).expect("open self-referential Form PDF");
    let handles = doc
        .page_image_handles(0)
        .expect("page_image_handles on cyclic Form");
    assert_eq!(handles.len(), 0, "self-referential Form with no images must yield 0 handles");
}

#[test]
fn handle_methods_take_ref_and_clone() {
    // 8c: decode() / raw_compressed_bytes() borrow &self, so a single handle
    // supports a two-phase inspect → raw → decode flow, and PdfImageHandle is
    // Clone for ergonomic filtering/retaining.
    let pdf_bytes = build_pdf_with_jpeg(64, 64);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("open PDF");
    let handles = doc.page_image_handles(0).expect("handles");
    let h = &handles[0];

    // Inspect → raw → decode, all on the same borrowed handle.
    let raw = h.raw_compressed_bytes().expect("raw bytes");
    assert!(!raw.is_empty(), "raw JPEG bytes must be present");
    assert!(h.decode().is_ok(), "decode on &handle must succeed");

    // Still usable after both calls; Clone works and leaves the original intact.
    let h2 = h.clone();
    assert_eq!(h2.width, h.width);
    assert!(h.raw_compressed_bytes().is_ok(), "original handle still usable");
}
