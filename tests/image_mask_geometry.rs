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

fn separation_imagemask_pdf(w: &str, h: &str) -> Vec<u8> {
    separation_imagemask_pdf_with(w, h, "", &[0x00; 8])
}

fn separation_imagemask_pdf_with(w: &str, h: &str, extra_dict: &str, stencil: &[u8]) -> Vec<u8> {
    let objects = vec![
        obj("<< /Type /Catalog /Pages 2 0 R >>"),
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R \
             /Resources << /XObject << /Im1 5 0 R >> /ColorSpace << /CS1 6 0 R >> >> >>"),
        stream_obj("", b"q\n/CS1 cs\n1 scn\n50 0 0 50 25 25 cm\n/Im1 Do\nQ\n"),
        stream_obj(
            &format!(
                "/Type /XObject /Subtype /Image /Width {w} /Height {h} \
                 /ImageMask true /BitsPerComponent 1{extra_dict}"
            ),
            stencil,
        ),
        obj("[/Separation /Pantone-185 /DeviceCMYK 7 0 R]"),
        obj("<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [0 0.85 0.45 0] >>"),
    ];
    build_pdf(&objects)
}

fn pantone_plate(pdf: Vec<u8>) -> pdf_oxide::rendering::SeparationPlate {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    render_separations(&doc, 0, 72)
        .expect("separations render")
        .into_iter()
        .find(|p| p.ink_name == "Pantone-185")
        .expect("Pantone-185 plate emitted")
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

/// A separation `/ImageMask` whose declared geometry is positive, in range and
/// overflows nothing — and which no stream of any size could back. Rejecting
/// only unrepresentable dimensions still let this one through to an allocation
/// sized from the declaration: 2^31 x 2^31 is 2^62 bytes.
#[test]
fn separation_image_mask_larger_than_its_stream_renders() {
    let control = PdfDocument::from_bytes(separation_imagemask_pdf("8", "8")).expect("parse");
    let control_plates =
        render_separations(&control, 0, 72).expect("8x8 control image mask renders");

    let doc = PdfDocument::from_bytes(separation_imagemask_pdf("2147483648", "2147483648"))
        .expect("parse");
    let plates = render_separations(&doc, 0, 72)
        .expect("page with an image mask larger than its stream renders");
    assert_eq!(
        plates.len(),
        control_plates.len(),
        "skipping the unbacked mask must not drop the separation plates"
    );
}

/// A CCITT-compressed separation stencil. The decode chain passes
/// `/CCITTFaxDecode` through raw, so the mask path receives the compressed
/// bytes — fewer than the 8 the 8x8 geometry needs. The mask must be skipped,
/// not expanded from zero-padded compressed bytes (padded 0-bits mean "paint",
/// so padding fabricates ink the file never specified). Skipping loses the
/// mask's real ink until this path grows a CCITT decode like the page
/// renderer's stencil path has; this test pins the skip-not-fabricate half.
#[test]
fn ccitt_separation_image_mask_is_skipped_not_fabricated() {
    // Control: an uncompressed all-paint stencil must put ink on the plate,
    // proving the assertion below is reached through a painting mask path.
    let control = pantone_plate(separation_imagemask_pdf("8", "8"));
    assert!(
        control.data.iter().any(|&v| v > 0),
        "control stencil painted no ink, so the mask path is not reached"
    );

    // Payload bytes are never decoded (pass-through); only their length —
    // shorter than the 8-byte 8x8 geometry, as compressed data is — matters.
    let plate = pantone_plate(separation_imagemask_pdf_with(
        "8",
        "8",
        " /Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns 8 /Rows 8 >>",
        &[0x25, 0x88, 0x40, 0x80],
    ));
    assert!(
        plate.data.iter().all(|&v| v == 0),
        "a stencil the decoded stream cannot back must be skipped, not painted from padding"
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
