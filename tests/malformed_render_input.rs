//! Renderer robustness against malformed, document-controlled geometry.
//!
//! Every construct here is legal PDF syntax carrying out-of-range values.
//! The renderer must skip the construct and return a `Result` — never panic,
//! and never fabricate a plausible-but-wrong result.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, render_separations, RenderOptions};

/// Assemble a PDF with a correct xref from raw object bodies.
/// `objects[i]` is the body of object i+1 (no "N 0 obj"/"endobj" wrapper).
fn build_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            objects.len() + 1,
            xref_pos
        )
        .as_bytes(),
    );
    out
}

fn obj(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

fn stream_obj(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut v = format!("<< {} /Length {} >>\nstream\n", dict, data.len()).into_bytes();
    v.extend_from_slice(data);
    v.extend_from_slice(b"\nendstream");
    v
}

/// A Type 1 shading whose `/Function` is a Type 3 stitching function with an
/// empty `/Domain`. The stitching function has no sub-domain to select from,
/// so it contributes no colour; the page must still render.
#[test]
fn stitching_function_with_empty_domain_renders() {
    let objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] /Contents 4 0 R \
             /Resources << /Shading << /Sh0 5 0 R >> >> >>"),
        stream_obj("", b"q /Sh0 sh Q"),
        obj("<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] /Function 6 0 R >>"),
        obj("<< /FunctionType 3 /Domain [] /Functions [7 0 R] /Bounds [] /Encode [0 1] >>"),
        obj("<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0] /C1 [1 1 1] >>"),
    ];
    let doc = PdfDocument::from_bytes(build_pdf(&objects)).expect("synthetic PDF parses");

    let img = render_page(&doc, 0, &RenderOptions::default())
        .expect("page with an empty-domain stitching function renders");
    assert!(!img.data.is_empty(), "renderer produced an empty buffer");
}

/// A Type 0 sampled function declaring a `/Size` grid far larger than its
/// sample stream. The declared grid overflows the flat-index arithmetic, so
/// the function must be rejected rather than indexed with a wrapped offset.
#[test]
fn sampled_function_size_larger_than_stream_renders() {
    let samples = vec![0u8; 64];
    let objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] /Contents 4 0 R \
             /Resources << /Shading << /Sh0 5 0 R >> >> >>"),
        stream_obj("", b"q /Sh0 sh Q"),
        obj("<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] /Function 6 0 R >>"),
        // Two input dimensions with ~2^32 samples each: the stride reaches 2^64.
        stream_obj(
            "/FunctionType 0 /Domain [0 1 0 1] /Range [0 1 0 1 0 1] \
             /Size [4294967296 4294967296] /BitsPerSample 8",
            &samples,
        ),
    ];
    let doc = PdfDocument::from_bytes(build_pdf(&objects)).expect("synthetic PDF parses");

    let img = render_page(&doc, 0, &RenderOptions::default())
        .expect("page with an oversized sampled function renders");
    assert!(!img.data.is_empty(), "renderer produced an empty buffer");
}

fn separation_imagemask_pdf(w: &str, h: &str) -> Vec<u8> {
    let stencil: Vec<u8> = vec![0x00; 8];
    let objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R \
             /Resources << /XObject << /Im1 5 0 R >> /ColorSpace << /CS1 6 0 R >> >> >>"),
        stream_obj("", b"q\n/CS1 cs\n1 scn\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n"),
        stream_obj(
            &format!(
                "/Type /XObject /Subtype /Image /Width {w} /Height {h} \
                 /ImageMask true /BitsPerComponent 1"
            ),
            &stencil,
        ),
        obj("[/Separation /Pantone-185 /DeviceCMYK 7 0 R]"),
        obj("<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [0 0.85 0.45 0] >>"),
    ];
    build_pdf(&objects)
}

/// A separation `/ImageMask` with negative dimensions. Only the sign differs
/// from the control below, so a clean render proves the dimensions are
/// rejected rather than reinterpreted as a near-`usize::MAX` pixel count.
#[test]
fn separation_image_mask_negative_dimensions_render() {
    let control = PdfDocument::from_bytes(separation_imagemask_pdf("8", "8")).expect("parse");
    let control_plates =
        render_separations(&control, 0, 72).expect("8x8 control image mask renders");
    assert!(
        !control_plates.is_empty(),
        "control produced no separation plates, so the image-mask path is not reached"
    );

    let doc = PdfDocument::from_bytes(separation_imagemask_pdf("-1", "-1")).expect("parse");
    let plates =
        render_separations(&doc, 0, 72).expect("page with a negative-dimension image mask renders");
    assert_eq!(
        plates.len(),
        control_plates.len(),
        "skipping the malformed mask must not drop the separation plates"
    );
}

fn masked_image_pdf(mask_dict: &str, mask_data: &[u8]) -> Vec<u8> {
    let objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 50 50] /Contents 4 0 R \
             /Resources << /XObject << /Im0 5 0 R >> >> >>"),
        stream_obj("", b"q 20 0 0 20 10 10 cm /Im0 Do Q"),
        stream_obj(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask 6 0 R",
            &[0x10u8, 0x20, 0x30, 0x40],
        ),
        stream_obj(mask_dict, mask_data),
    ];
    build_pdf(&objects)
}

/// An image whose `/Mask` sub-image is zero pixels wide. There is no mask
/// sample to read, so the mask is skipped and the base image still paints.
#[test]
fn zero_width_mask_sub_image_renders() {
    let pdf = masked_image_pdf(
        "/Type /XObject /Subtype /Image /Width 0 /Height 1 \
         /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[],
    );
    let doc = PdfDocument::from_bytes(pdf).expect("synthetic PDF parses");

    let img = render_page(&doc, 0, &RenderOptions::default())
        .expect("page with a zero-width mask renders");
    assert!(!img.data.is_empty(), "renderer produced an empty buffer");
}
