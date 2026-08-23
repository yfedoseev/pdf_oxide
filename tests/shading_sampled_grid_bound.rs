//! Renderer robustness against malformed, document-controlled geometry.
//!
//! Every construct here is legal PDF syntax carrying out-of-range values.
//! The renderer must skip the construct and return a `Result` — never panic,
//! and never fabricate a plausible-but-wrong result.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

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
