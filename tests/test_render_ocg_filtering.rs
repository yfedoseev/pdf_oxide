//! Tests for OCG layer filtering in the rendering pipeline.
//!
//! Verifies that `RenderOptions::excluded_layers` suppresses graphical content
//! (rectangles, text) drawn inside OCG-tagged BDC scopes while preserving
//! content outside those scopes.
//!
//! The whole suite exercises `pdf_oxide::rendering`, which is gated behind the
//! `rendering` feature, so the file is compiled only when that feature is on —
//! otherwise `cargo test` (default features) cannot resolve the import.
#![cfg(feature = "rendering")]

use std::collections::HashSet;

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{ImageFormat, RenderOptions};

// ============================================================================
// Helper: build a PDF with a colored rectangle inside an OCG layer + text outside
// ============================================================================

/// Build a PDF where:
/// - A red rectangle is drawn inside BDC /OC /MC0 scope (OCG "Background")
/// - Text "HELLO" is drawn outside any layer scope
///
/// The rectangle fills a 200x200 area at (50, 550) in PDF coords.
fn build_pdf_with_ocg_rect_and_text() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Obj 1: Catalog with OCProperties
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R] /D << /ON [6 0 R] >> >> >>\nendobj\n\n",
    );

    // Obj 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    // Obj 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 6 0 R >> >> >>\nendobj\n\n",
    );

    // Obj 4: Content stream
    // Red rectangle inside OCG "Background", then text outside any layer
    let content = b"/OC /MC0 BDC q 1 0 0 rg 50 550 200 200 re f Q EMC \
                    BT /F1 24 Tf 300 650 Td (HELLO) Tj ET";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Font
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );

    // Obj 6: OCG dictionary
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Background >>\nendobj\n\n");

    // Xref
    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());

    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

/// Build a PDF with an OCMD-referenced rectangle.
fn build_pdf_with_ocmd_rect() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [7 0 R] /D << /ON [7 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 6 0 R >> >> >>\nendobj\n\n",
    );

    // Blue rectangle inside OCMD scope, green rectangle outside
    let content = b"/OC /MC0 BDC 0 0 1 rg 50 550 200 200 re f EMC \
                    0 1 0 rg 300 550 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    // Obj 6: OCMD referencing OCG
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCMD /OCGs [7 0 R] /P /AllOn >>\nendobj\n\n");
    // Obj 7: OCG
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /Watermark >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

/// Build a PDF with OCG-tagged content inside a Form XObject.
fn build_pdf_with_ocg_in_form_xobject() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [7 0 R] /D << /ON [7 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 6 0 R >> /XObject << /Fm0 5 0 R >> >> >>\nendobj\n\n",
    );

    // Page content: just invoke Form XObject, then draw green rect outside
    let page_content = b"/Fm0 Do 0 1 0 rg 300 550 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject with OCG-tagged red rect
    let form_stream = b"/OC /MC0 BDC 1 0 0 rg 50 550 200 200 re f EMC";
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
            /Resources << /Font << /F1 6 0 R >> /Properties << /MC0 7 0 R >> >>\n\
            /Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_hdr.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 6: Font
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );

    // Obj 7: OCG dictionary
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /Background >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

// ============================================================================
// Helper: render a PDF and get raw RGBA pixel data
// ============================================================================

fn render_raw(doc: &PdfDocument, excluded: HashSet<String>) -> (Vec<u8>, u32, u32) {
    let mut options = RenderOptions::with_dpi(72);
    options.format = ImageFormat::RawRgba8;
    options.excluded_layers = excluded;
    let img = pdf_oxide::rendering::render_page(doc, 0, &options).expect("render");
    (img.data, img.width, img.height)
}

/// Sample the average color in a rectangular pixel region.
/// Returns (r, g, b, a) as straight-alpha floats in 0..255.
///
/// Panics on an empty region (caller bug) so out-of-frame coordinates are
/// detected rather than silently returning black.
fn sample_region(
    data: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> (f32, f32, f32, f32) {
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut a_sum = 0u64;
    let mut count = 0u64;
    for py in y..(y + h).min(height) {
        for px in x..(x + w).min(width) {
            let idx = ((py * width + px) * 4) as usize;
            if idx + 3 < data.len() {
                // tiny-skia stores premultiplied RGBA; un-premultiply for comparison
                let a = data[idx + 3] as f32;
                if a > 0.0 {
                    let scale = 255.0 / a;
                    r_sum += (data[idx] as f32 * scale).round().min(255.0) as u64;
                    g_sum += (data[idx + 1] as f32 * scale).round().min(255.0) as u64;
                    b_sum += (data[idx + 2] as f32 * scale).round().min(255.0) as u64;
                } else {
                    r_sum += data[idx] as u64;
                    g_sum += data[idx + 1] as u64;
                    b_sum += data[idx + 2] as u64;
                }
                a_sum += data[idx + 3] as u64;
                count += 1;
            }
        }
    }
    assert!(
        count > 0,
        "sample_region({x},{y},{w},{h}) selected zero pixels in a {width}x{height} image"
    );
    (
        r_sum as f32 / count as f32,
        g_sum as f32 / count as f32,
        b_sum as f32 / count as f32,
        a_sum as f32 / count as f32,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_render_ocg_rect_visible_without_filter() {
    let pdf_bytes = build_pdf_with_ocg_rect_and_text();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let (data, width, height) = render_raw(&doc, HashSet::new());

    // The red rectangle region (PDF coords 50,550 to 250,750 mapped to 72 DPI pixels)
    // At 72 DPI, 1pt = 1px. PDF y=550..750 → pixel y = 792-750..792-550 = 42..242
    let (r, g, b, a) = sample_region(&data, width, height, 100, 80, 50, 50);
    assert!(r > 200.0, "Red channel should be high in rect region, got {r}");
    assert!(g < 50.0, "Green should be low, got {g}");
    assert!(b < 50.0, "Blue should be low, got {b}");
    assert!(a > 200.0, "Alpha should be high, got {a}");
}

#[test]
fn test_render_ocg_rect_hidden_with_filter() {
    let pdf_bytes = build_pdf_with_ocg_rect_and_text();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded = HashSet::from(["Background".to_string()]);
    let (data, width, height) = render_raw(&doc, excluded);

    // The red rectangle region should now be white (background)
    let (r, g, b, a) = sample_region(&data, width, height, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "Rect region should be white background when layer excluded, got ({r}, {g}, {b})"
    );
    assert!(a > 250.0, "Alpha should still be high (white bg), got {a}");
}

#[test]
fn test_render_unrelated_layer_filter_preserves_content() {
    let pdf_bytes = build_pdf_with_ocg_rect_and_text();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Exclude a layer that doesn't exist — should not affect rendering
    let excluded = HashSet::from(["NonExistentLayer".to_string()]);
    let (data_filtered, _width, _) = render_raw(&doc, excluded);
    let (data_unfiltered, _, _) = render_raw(&doc, HashSet::new());

    // Both renders should be identical
    assert_eq!(
        data_filtered, data_unfiltered,
        "Filtering a non-existent layer must not change the rendered output"
    );
}

#[test]
fn test_render_ocmd_rect_hidden_with_filter() {
    let pdf_bytes = build_pdf_with_ocmd_rect();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Without filter: blue rect at (50,550) should be visible
    let (data_unfiltered, width, height) = render_raw(&doc, HashSet::new());
    let (r, g, b, _a) = sample_region(&data_unfiltered, width, height, 100, 80, 50, 50);
    assert!(b > 200.0, "Blue rect should be visible without filter, got b={b}");
    assert!(r < 50.0 && g < 50.0, "Should be blue, got r={r} g={g}");

    // Green rect at (300,550) should be visible
    let (_r, g, _b, _a) = sample_region(&data_unfiltered, width, height, 350, 80, 50, 50);
    assert!(g > 200.0, "Green rect should be visible, got g={g}");

    // With filter: exclude "Watermark" → blue rect gone, green rect remains
    let excluded = HashSet::from(["Watermark".to_string()]);
    let (data_filtered, width, height) = render_raw(&doc, excluded);

    // Blue rect region should now be white
    let (r, g, b, _) = sample_region(&data_filtered, width, height, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "Blue rect should be hidden, got ({r}, {g}, {b})"
    );

    // Green rect should still be visible
    let (_r, g, _b, _) = sample_region(&data_filtered, width, height, 350, 80, 50, 50);
    assert!(g > 200.0, "Green rect should survive filter, got g={g}");
}

#[test]
fn test_render_ocg_in_form_xobject_filtered() {
    let pdf_bytes = build_pdf_with_ocg_in_form_xobject();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Without filter: red rect from Form XObject should be visible
    let (data_unfiltered, width, height) = render_raw(&doc, HashSet::new());
    let (r, _g, _b, _a) = sample_region(&data_unfiltered, width, height, 100, 80, 50, 50);
    assert!(r > 200.0, "Red rect from XObject should be visible, got r={r}");

    // With filter: exclude "Background" → red rect gone, green rect remains
    let excluded = HashSet::from(["Background".to_string()]);
    let (data_filtered, width, height) = render_raw(&doc, excluded);

    // Red rect region should now be white
    let (r, g, b, _) = sample_region(&data_filtered, width, height, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "Red rect from XObject should be hidden, got ({r}, {g}, {b})"
    );

    // Green rect at (300,550) should still be visible
    let (_r, g, _b, _) = sample_region(&data_filtered, width, height, 350, 80, 50, 50);
    assert!(g > 200.0, "Green rect should survive filter, got g={g}");
}

#[test]
fn test_render_empty_excluded_layers_matches_default() {
    let pdf_bytes = build_pdf_with_ocg_rect_and_text();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let (data_default, _, _) = render_raw(&doc, HashSet::new());

    let mut options = RenderOptions::with_dpi(72);
    options.format = ImageFormat::RawRgba8;
    // excluded_layers defaults to empty
    let img = pdf_oxide::rendering::render_page(&doc, 0, &options).expect("render");

    assert_eq!(
        data_default, img.data,
        "Default excluded_layers (empty) should match explicit empty HashSet"
    );
}

// ============================================================================
// Additional realistic regression tests for the OCG render fix-up.
// ============================================================================

/// Append the standard xref + trailer block for a single-page test PDF.
fn write_xref_trailer(pdf: &mut Vec<u8>, offsets: &[usize]) {
    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
}

/// Build a PDF where a `W n` clip is issued inside an excluded BDC scope
/// (no surrounding `q/Q`). Outside the scope, a green rect tries to fill the
/// whole page. If the clip leaked from the excluded scope, the green fill
/// would be restricted to the clipped sub-region; with the fix, the green
/// fills the whole page.
fn build_pdf_with_clip_inside_excluded_bdc() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [5 0 R] /D << /ON [5 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Properties << /MC0 5 0 R >> >> >>\nendobj\n\n",
    );

    // Inside excluded layer: build a path covering only the top-left 100x100,
    // then `W n` to set it as a clip. No q/Q, no fill. Outside: paint a
    // green rect 200x200 at (200,200) — it must be visible if the clip was
    // properly gated.
    //
    // Note: the path uses moveto/lineto and clip with the non-zero rule.
    let content: &[u8] = b"/OC /MC0 BDC \
                           0 700 m 100 700 l 100 800 l 0 800 l h W n \
                           EMC \
                           0 1 0 rg 200 200 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"5 0 obj\n<< /Type /OCG /Name /ClipLeak >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_clip_does_not_leak_from_excluded_layer() {
    let pdf_bytes = build_pdf_with_clip_inside_excluded_bdc();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // With "ClipLeak" excluded, the green rect (PDF coords 200..400 x 200..400,
    // pixel y = 792-400..792-200 = 392..592 at 72 DPI) must paint freely.
    let excluded = HashSet::from(["ClipLeak".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);

    // Sample near the centre of the green rect: pixel x ~ 300, pixel y ~ 492.
    let (r, g, b, _) = sample_region(&data, w, h, 290, 480, 20, 20);
    assert!(
        g > 240.0 && r < 30.0 && b < 30.0,
        "Green rect should paint unclipped after excluded BDC closes; got ({r}, {g}, {b})"
    );
}

/// Build a PDF where an annotation (a "Square" annotation with /AP) carries
/// an /OC entry referencing an excluded OCG.
fn build_pdf_with_oc_annotation() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // 1 Catalog, 2 Pages, 3 Page, 4 Annot, 5 AP /N stream, 6 OCG.
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R] /D << /ON [6 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 7 0 R /Annots [4 0 R] >>\nendobj\n\n",
    );
    // Annotation with /OC referencing OCG 6.
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /Annot /Subtype /Square /Rect [50 550 250 750]\n\
           /AP << /N 5 0 R >> /OC 6 0 R /F 4 >>\nendobj\n\n",
    );
    // Appearance stream: a red 200x200 fill matching the /Rect.
    let ap_content: &[u8] = b"q 1 0 0 rg 0 0 200 200 re f Q";
    offsets.push(pdf.len());
    let ap_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 200 200]\n\
           /Resources << >> /Length {} >>\nstream\n",
        ap_content.len()
    );
    pdf.extend_from_slice(ap_hdr.as_bytes());
    pdf.extend_from_slice(ap_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    // OCG
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Watermark >>\nendobj\n\n");
    // Empty content stream so only the annotation paints.
    let content: &[u8] = b"";
    offsets.push(pdf.len());
    let hdr = format!("7 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_annotation_oc_is_filtered() {
    let pdf_bytes = build_pdf_with_oc_annotation();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Without filter: red appearance fills the annot rect (PDF y=550..750
    // -> pixel y=42..242 at 72 DPI), so sampling at pixel (100,80) sees red.
    let (data_unfiltered, w, h) = render_raw(&doc, HashSet::new());
    let (r, g, b, _) = sample_region(&data_unfiltered, w, h, 100, 80, 50, 50);
    assert!(
        r > 200.0 && g < 50.0 && b < 50.0,
        "Annotation appearance should be red without filter, got ({r}, {g}, {b})"
    );

    // With filter: annotation must be skipped, region is white background.
    let excluded = HashSet::from(["Watermark".to_string()]);
    let (data_filtered, w, h) = render_raw(&doc, excluded);
    let (r, g, b, _) = sample_region(&data_filtered, w, h, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "Annotation with /OC pointing at excluded OCG must be hidden; got ({r}, {g}, {b})"
    );
}

/// Build a PDF where a layer's /Name is encoded as UTF-16LE with BOM.
/// Catches the renderer's previously-lossy decoder (which only handled
/// UTF-16BE BOM).
fn build_pdf_with_utf16le_layer_name() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [5 0 R] /D << /ON [5 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Properties << /MC0 5 0 R >> >> >>\nendobj\n\n",
    );
    let content: &[u8] = b"/OC /MC0 BDC 1 0 0 rg 50 550 200 200 re f EMC";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // OCG /Name = UTF-16LE BOM + "Layer" — hex string <FFFE 4C00 6100 7900 6500 7200>.
    // Decoded that's "Layer". (No trailing NUL.)
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /OCG /Name <FFFE4C006100790065007200> >>\nendobj\n\n",
    );

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_utf16le_layer_name_is_filtered() {
    let pdf_bytes = build_pdf_with_utf16le_layer_name();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Filter on the decoded layer name. Without the UTF-16LE-BOM fix in the
    // renderer this would silently fail to match and the red rect would
    // still paint.
    let excluded = HashSet::from(["Layer".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);
    let (r, g, b, _) = sample_region(&data, w, h, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "UTF-16LE-named OCG must be filtered; rect region got ({r}, {g}, {b})"
    );
}

/// Build a PDF with two text runs in a single BT/ET, where the first run is
/// inside an excluded OCG scope. The second run must paint at the correct
/// X position — i.e. shifted by the advance of the first run.
fn build_pdf_with_text_split_by_bdc() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R] /D << /ON [6 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 6 0 R >> >> >>\nendobj\n\n",
    );

    // BT /F1 24 Tf 100 400 Td
    //   (Hello ) Tj                     ← visible, in black
    //   /OC /MC0 BDC (SECRET) Tj EMC    ← inside excluded layer
    //   (World) Tj                       ← must paint to the RIGHT of the
    //                                      hidden (SECRET) glyphs, not under
    //                                      the trailing space of "Hello "
    // ET
    let content: &[u8] = b"BT /F1 24 Tf 100 400 Td \
                           (Hello ) Tj \
                           /OC /MC0 BDC (SECRET) Tj EMC \
                           (World) Tj ET";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Hidden >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_text_advance_preserved_through_excluded_run() {
    let pdf_bytes = build_pdf_with_text_split_by_bdc();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Helvetica widths at 24pt: "Hello " advance ≈ 65pt, "SECRET" ≈ 100pt.
    // Without the fix, "World" overlaps "Hello " starting at ~x=165 (pixel
    // y around 792-400 = 392). With the fix, "World" must paint at x ≈ 265
    // (i.e. clearly to the right of pixel 220).
    let excluded = HashSet::from(["Hidden".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);

    // Sample two horizontal strips around y=400 in PDF coords (pixel y ≈ 392)
    // both before pixel 200 (where Hello sits and World must not bleed) and
    // after pixel 240 (where World should land after the suppressed SECRET).
    let baseline_y: u32 = 386; // a few pixels around the glyph baseline area
    let h_strip: u32 = 18;

    // Region under (or just to the right of) the visible "Hello " text:
    // pixel x ≈ 100..200. Sample the rightmost 30 pixels — these are right
    // before SECRET begins. There may be background or trailing whitespace.
    let (_hr, _hg, _hb, _ha) = sample_region(&data, w, h, 100, baseline_y, 60, h_strip);

    // Region where SECRET would have rasterised — must be empty/white.
    let secret_x: u32 = 180;
    let (sr, sg, sb, _) = sample_region(&data, w, h, secret_x, baseline_y, 60, h_strip);
    assert!(
        sr > 240.0 && sg > 240.0 && sb > 240.0,
        "SECRET run must not paint; region sampled at x={secret_x} got ({sr}, {sg}, {sb})"
    );

    // Region where "World" must paint AFTER the suppressed SECRET. If the
    // text advance was skipped (bug #4), "World" would have painted under
    // SECRET's slot (around x≈165..200) and this far-right region would be
    // pure background. With the fix, dark glyph ink shows up here.
    let world_x: u32 = 275;
    let (wr, wg, wb, _) = sample_region(&data, w, h, world_x, baseline_y, 50, h_strip);
    let world_brightness = (wr + wg + wb) / 3.0;
    assert!(
        world_brightness < 245.0,
        "World text should paint at x≈{world_x} after suppressed run; got brightness {world_brightness}"
    );
}

/// Build a PDF with 12 levels of nested BDC/EMC, alternating excluded and
/// non-excluded layers, with a fill at the innermost scope.
fn build_pdf_with_deeply_nested_bdc() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [5 0 R 6 0 R] /D << /ON [5 0 R 6 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Properties << /KEEP 5 0 R /DROP 6 0 R >> >> >>\nendobj\n\n",
    );

    // 12-deep nesting:
    //  /OC /KEEP BDC × 6 alternating with /OC /DROP BDC × 6.
    // Innermost: paint a red rect. Because one or more /DROP scopes are on
    // the stack, the rect must be suppressed.
    let mut content: Vec<u8> = Vec::new();
    for i in 0..12 {
        if i % 2 == 0 {
            content.extend_from_slice(b"/OC /KEEP BDC ");
        } else {
            content.extend_from_slice(b"/OC /DROP BDC ");
        }
    }
    content.extend_from_slice(b"1 0 0 rg 50 550 200 200 re f ");
    for _ in 0..12 {
        content.extend_from_slice(b"EMC ");
    }
    // After all EMCs, paint a green rect that MUST be visible.
    content.extend_from_slice(b"0 1 0 rg 300 550 200 200 re f");

    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(&content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"5 0 obj\n<< /Type /OCG /Name /Keep >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Drop >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_deeply_nested_bdc_stack_unwinds_correctly() {
    let pdf_bytes = build_pdf_with_deeply_nested_bdc();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded = HashSet::from(["Drop".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);

    // Red rect at PDF (50,550)-(250,750) → pixel x=50..250, y=42..242.
    let (r, g, b, _) = sample_region(&data, w, h, 100, 80, 50, 50);
    assert!(
        r > 250.0 && g > 250.0 && b > 250.0,
        "Inner red rect must be suppressed by /Drop scope on 12-level stack; got ({r}, {g}, {b})"
    );
    // Green rect at PDF (300,550)-(500,750) → pixel x=300..500, y=42..242.
    let (gr, gg, gb, _) = sample_region(&data, w, h, 350, 80, 50, 50);
    assert!(
        gg > 200.0 && gr < 50.0 && gb < 50.0,
        "After all 12 EMCs, depth must be 0 and green rect visible; got ({gr}, {gg}, {gb})"
    );
}

/// Build a PDF whose two pages each have a different OCG usage.
fn build_pdf_multi_page() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // 1 Catalog, 2 Pages, 3 Page1, 4 Content1, 5 Page2, 6 Content2,
    // 7 OCG-PageOne, 8 OCG-PageTwo.
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [7 0 R 8 0 R] /D << /ON [7 0 R 8 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n\n");
    // Page 1
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Properties << /L1 7 0 R >> >> >>\nendobj\n\n",
    );
    let content1: &[u8] = b"/OC /L1 BDC 1 0 0 rg 50 550 200 200 re f EMC";
    offsets.push(pdf.len());
    let hdr1 = format!("4 0 obj\n<< /Length {} >>\nstream\n", content1.len());
    pdf.extend_from_slice(hdr1.as_bytes());
    pdf.extend_from_slice(content1);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    // Page 2
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 6 0 R\n\
           /Resources << /Properties << /L2 8 0 R >> >> >>\nendobj\n\n",
    );
    let content2: &[u8] = b"/OC /L2 BDC 0 0 1 rg 50 550 200 200 re f EMC";
    offsets.push(pdf.len());
    let hdr2 = format!("6 0 obj\n<< /Length {} >>\nstream\n", content2.len());
    pdf.extend_from_slice(hdr2.as_bytes());
    pdf.extend_from_slice(content2);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /PageOne >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"8 0 obj\n<< /Type /OCG /Name /PageTwo >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

fn render_page_raw(
    doc: &PdfDocument,
    page: usize,
    excluded: HashSet<String>,
) -> (Vec<u8>, u32, u32) {
    let mut options = RenderOptions::with_dpi(72);
    options.format = ImageFormat::RawRgba8;
    options.excluded_layers = excluded;
    let img = pdf_oxide::rendering::render_page(doc, page, &options).expect("render");
    (img.data, img.width, img.height)
}

#[test]
fn test_multi_page_per_page_filtering() {
    let pdf_bytes = build_pdf_multi_page();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Exclude PageOne only; page 1's red rect must vanish, page 2's blue rect remains.
    let excluded = HashSet::from(["PageOne".to_string()]);

    let (p1, w1, h1) = render_page_raw(&doc, 0, excluded.clone());
    let (r1, g1, b1, _) = sample_region(&p1, w1, h1, 100, 80, 50, 50);
    assert!(
        r1 > 250.0 && g1 > 250.0 && b1 > 250.0,
        "Page 1 rect must be hidden when PageOne excluded; got ({r1}, {g1}, {b1})"
    );

    let (p2, w2, h2) = render_page_raw(&doc, 1, excluded);
    let (r2, g2, b2, _) = sample_region(&p2, w2, h2, 100, 80, 50, 50);
    assert!(
        b2 > 200.0 && r2 < 50.0 && g2 < 50.0,
        "Page 2 blue rect must survive a PageOne-only filter; got ({r2}, {g2}, {b2})"
    );
}

/// Build a PDF that exercises one specific OCMD /P policy.
///
/// Two OCGs ("A", "B") and one OCMD with the given policy.
/// Two rects: one inside the OCMD scope (test color), one outside (yellow,
/// always visible).
fn build_pdf_with_ocmd_policy(policy: &str) -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // 1 Catalog, 2 Pages, 3 Page, 4 Content,
    // 5 OCMD, 6 OCG /A, 7 OCG /B
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R 7 0 R] /D << /ON [6 0 R 7 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Properties << /MC0 5 0 R >> >> >>\nendobj\n\n",
    );
    let content: &[u8] = b"/OC /MC0 BDC 0 0 1 rg 50 550 200 200 re f EMC \
                           1 1 0 rg 300 550 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    // OCMD with chosen /P, referencing OCGs 6 and 7.
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!("5 0 obj\n<< /Type /OCMD /OCGs [6 0 R 7 0 R] /P /{} >>\nendobj\n\n", policy)
            .as_bytes(),
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /A >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /B >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

/// For each (policy, excluded_state) combination, assert whether the OCMD
/// rect is visible. The yellow rect outside the BDC scope is always visible
/// and serves as a control.
fn assert_ocmd_visibility(policy: &str, excluded: HashSet<String>, expect_visible: bool) {
    let pdf_bytes = build_pdf_with_ocmd_policy(policy);
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let (data, w, h) = render_raw(&doc, excluded.clone());

    // Blue rect at pixel (50..250, 42..242). Sample its centre.
    let (r, g, b, _) = sample_region(&data, w, h, 100, 80, 50, 50);
    if expect_visible {
        assert!(
            b > 200.0 && r < 50.0 && g < 50.0,
            "[{policy} excluded={:?}] expected visible blue, got ({r}, {g}, {b})",
            excluded
        );
    } else {
        assert!(
            r > 250.0 && g > 250.0 && b > 250.0,
            "[{policy} excluded={:?}] expected hidden, got ({r}, {g}, {b})",
            excluded
        );
    }

    // Yellow control rect at pixel (300..500, 42..242) must always be visible.
    let (yr, yg, yb, _) = sample_region(&data, w, h, 350, 80, 50, 50);
    assert!(
        yr > 200.0 && yg > 200.0 && yb < 50.0,
        "[{policy} excluded={:?}] yellow control rect missing: ({yr}, {yg}, {yb})",
        excluded
    );
}

// Note: the renderer short-circuits the OCMD evaluation when the user
// supplied no excluded layers (the common "render everything" path). The
// tests below therefore exercise each /P policy with at least one OCG
// excluded — the realistic prepress use case.

#[test]
fn test_ocmd_policy_anyon() {
    // AnyOn (default): visible iff any referenced OCG is on. Hide iff all off.
    // A on, B on  -> visible
    assert_ocmd_visibility("AnyOn", HashSet::from(["X".to_string()]), true);
    // A off, B on -> visible (B still on)
    assert_ocmd_visibility("AnyOn", HashSet::from(["A".to_string()]), true);
    // A off, B off -> hidden
    assert_ocmd_visibility("AnyOn", HashSet::from(["A".to_string(), "B".to_string()]), false);
}

#[test]
fn test_ocmd_policy_allon() {
    // AllOn: visible iff all on. Hide iff any off.
    assert_ocmd_visibility("AllOn", HashSet::from(["X".to_string()]), true);
    // A off -> any off -> hidden
    assert_ocmd_visibility("AllOn", HashSet::from(["A".to_string()]), false);
    assert_ocmd_visibility("AllOn", HashSet::from(["A".to_string(), "B".to_string()]), false);
}

#[test]
fn test_ocmd_policy_anyoff() {
    // AnyOff: visible iff any off. Hide iff all on.
    // A off, B on -> visible
    assert_ocmd_visibility("AnyOff", HashSet::from(["A".to_string()]), true);
    // A off, B off -> visible
    assert_ocmd_visibility("AnyOff", HashSet::from(["A".to_string(), "B".to_string()]), true);
}

#[test]
fn test_ocmd_policy_alloff() {
    // AllOff: visible iff all off. Hide iff any on.
    // A off, B on -> hidden (B still on)
    assert_ocmd_visibility("AllOff", HashSet::from(["A".to_string()]), false);
    // A off, B off -> visible
    assert_ocmd_visibility("AllOff", HashSet::from(["A".to_string(), "B".to_string()]), true);
}

/// Build a synthetic prepress-style PDF: dieline (cyan), varnish (yellow),
/// and a CMYK process magenta artwork rect, each on its own OCG layer, plus
/// a text overlay on no layer.
fn build_pdf_prepress_layers() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // 1 Catalog, 2 Pages, 3 Page, 4 Content, 5 Font,
    // 6 OCG Dieline, 7 OCG Varnish, 8 OCG Artwork.
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R 7 0 R 8 0 R]\n\
                           /D << /ON [6 0 R 7 0 R 8 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >>\n\
                         /Properties << /D 6 0 R /V 7 0 R /A 8 0 R >> >> >>\nendobj\n\n",
    );

    // Layered content:
    //   Artwork (magenta-ish, RGB 1 0 1): big rect at (50,550)-(250,750)
    //   Dieline (cyan, RGB 0 1 1): thin rect at (260,550)-(310,750)
    //   Varnish (yellow, RGB 1 1 0): rect at (320,550)-(520,750)
    //   Text: "PRINT" overlaid, no OCG
    let content: &[u8] = b"\
        /OC /A BDC 1 0 1 rg 50 550 200 200 re f EMC \
        /OC /D BDC 0 1 1 rg 260 550 50 200 re f EMC \
        /OC /V BDC 1 1 0 rg 320 550 200 200 re f EMC \
        BT /F1 24 Tf 100 400 Td (PRINT) Tj ET";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Dieline >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /Varnish >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"8 0 obj\n<< /Type /OCG /Name /Artwork >>\nendobj\n\n");

    write_xref_trailer(&mut pdf, &offsets);
    pdf
}

#[test]
fn test_prepress_fixture_combinations() {
    let pdf_bytes = build_pdf_prepress_layers();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Default render: all three layer rects must be visible.
    let (data_all, w, h) = render_raw(&doc, HashSet::new());
    let (ar, ag, ab, _) = sample_region(&data_all, w, h, 100, 80, 30, 30);
    assert!(
        ar > 200.0 && ab > 200.0 && ag < 80.0,
        "Artwork (magenta) missing: ({ar}, {ag}, {ab})"
    );
    let (dr, dg, db, _) = sample_region(&data_all, w, h, 270, 80, 30, 30);
    assert!(
        dr < 80.0 && dg > 200.0 && db > 200.0,
        "Dieline (cyan) missing: ({dr}, {dg}, {db})"
    );
    let (vr, vg, vb, _) = sample_region(&data_all, w, h, 370, 80, 30, 30);
    assert!(
        vr > 200.0 && vg > 200.0 && vb < 80.0,
        "Varnish (yellow) missing: ({vr}, {vg}, {vb})"
    );

    // Drop Dieline + Varnish (a typical "show me the artwork only" view).
    let excluded = HashSet::from(["Dieline".to_string(), "Varnish".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);
    let (ar, ag, ab, _) = sample_region(&data, w, h, 100, 80, 30, 30);
    assert!(
        ar > 200.0 && ab > 200.0 && ag < 80.0,
        "Artwork should remain when only finishing layers dropped: ({ar}, {ag}, {ab})"
    );
    let (dr, dg, db, _) = sample_region(&data, w, h, 270, 80, 30, 30);
    assert!(
        dr > 240.0 && dg > 240.0 && db > 240.0,
        "Dieline must be hidden: ({dr}, {dg}, {db})"
    );
    let (vr, vg, vb, _) = sample_region(&data, w, h, 370, 80, 30, 30);
    assert!(
        vr > 240.0 && vg > 240.0 && vb > 240.0,
        "Varnish must be hidden: ({vr}, {vg}, {vb})"
    );

    // Drop Artwork only — finishing layers remain (typical "tech-pack only" view).
    let excluded = HashSet::from(["Artwork".to_string()]);
    let (data, w, h) = render_raw(&doc, excluded);
    let (ar, ag, ab, _) = sample_region(&data, w, h, 100, 80, 30, 30);
    assert!(
        ar > 240.0 && ag > 240.0 && ab > 240.0,
        "Artwork must be hidden: ({ar}, {ag}, {ab})"
    );
    let (dr, dg, db, _) = sample_region(&data, w, h, 270, 80, 30, 30);
    assert!(dr < 80.0 && dg > 200.0 && db > 200.0, "Dieline must remain: ({dr}, {dg}, {db})");
    let (vr, vg, vb, _) = sample_region(&data, w, h, 370, 80, 30, 30);
    assert!(vr > 200.0 && vg > 200.0 && vb < 80.0, "Varnish must remain: ({vr}, {vg}, {vb})");
}

// ---------------------------------------------------------------------------
// /VE visibility expression coverage
// ---------------------------------------------------------------------------

/// Build a PDF whose OCMD uses a `/VE` visibility expression.
///
/// The expression is `[/<op> <ocg_ref> ...]`. We test:
///   /Not  — visible iff the OCG is excluded (inverted)
///   /And  — visible iff all referenced OCGs are on
///   /Or   — visible iff any referenced OCG is on
/// Two OCGs ("A", "B") are defined. A blue rect is inside the OCMD scope, a
/// green rect is always visible.
fn build_pdf_with_ve(expr: &[u8]) -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [7 0 R 8 0 R] /D << /ON [7 0 R 8 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 6 0 R >> >> >>\nendobj\n\n",
    );

    let content = b"/OC /MC0 BDC 0 0 1 rg 50 550 200 200 re f EMC \
                    0 1 0 rg 300 550 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );

    // Obj 6: OCMD with /VE expression
    offsets.push(pdf.len());
    let ocmd = format!(
        "6 0 obj\n<< /Type /OCMD /VE {} >>\nendobj\n\n",
        std::str::from_utf8(expr).unwrap()
    );
    pdf.extend_from_slice(ocmd.as_bytes());

    // Obj 7 & 8: OCGs
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /A >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"8 0 obj\n<< /Type /OCG /Name /B >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

fn ocmd_scope_is_visible(doc: &PdfDocument, excluded: HashSet<String>) -> bool {
    let (data, w, h) = render_raw(doc, excluded);
    let (r, g, b, _) = sample_region(&data, w, h, 100, 100, 30, 30);
    // Blue rect is inside the OCMD scope. Visible -> blue (b high, r/g low).
    // Hidden -> white background (all high).
    b > 200.0 && r < 80.0 && g < 80.0
}

#[test]
fn test_ve_not_operator() {
    // /VE = [/Not 7 0 R]
    // Visible iff OCG "A" is off (i.e. "A" is excluded).
    let doc = PdfDocument::from_bytes(build_pdf_with_ve(b"[/Not 7 0 R]")).unwrap();
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::new()),
        "Not(A): with A on, scope should be hidden"
    );
    assert!(
        ocmd_scope_is_visible(&doc, HashSet::from(["A".to_string()])),
        "Not(A): with A excluded (off), scope should be visible"
    );
}

#[test]
fn test_ve_and_operator() {
    // /VE = [/And 7 0 R 8 0 R]
    // Visible iff both A and B are on.
    let doc = PdfDocument::from_bytes(build_pdf_with_ve(b"[/And 7 0 R 8 0 R]")).unwrap();
    assert!(ocmd_scope_is_visible(&doc, HashSet::new()), "And(A,B): both on -> visible");
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::from(["A".to_string()])),
        "And(A,B): A excluded -> hidden"
    );
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::from(["B".to_string()])),
        "And(A,B): B excluded -> hidden"
    );
}

#[test]
fn test_ve_or_operator() {
    // /VE = [/Or 7 0 R 8 0 R]
    // Visible iff A or B is on.
    let doc = PdfDocument::from_bytes(build_pdf_with_ve(b"[/Or 7 0 R 8 0 R]")).unwrap();
    assert!(ocmd_scope_is_visible(&doc, HashSet::new()), "Or(A,B): both on -> visible");
    assert!(
        ocmd_scope_is_visible(&doc, HashSet::from(["A".to_string()])),
        "Or(A,B): A excluded but B on -> visible"
    );
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::from(["A".to_string(), "B".to_string()])),
        "Or(A,B): both excluded -> hidden"
    );
}

// ---------------------------------------------------------------------------
// /D default-off respect — ISO 32000-1 §8.11.4
// ---------------------------------------------------------------------------

/// PDF whose /OCProperties/D entry marks an OCG as off by default.
fn build_pdf_with_default_off_layer() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [6 0 R] /D << /BaseState /ON /OFF [6 0 R] >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 6 0 R >> >> >>\nendobj\n\n",
    );

    let content = b"/OC /MC0 BDC 0 0 1 rg 50 550 200 200 re f EMC \
                    0 1 0 rg 300 550 200 200 re f";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n\
           /Encoding /WinAnsiEncoding >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Watermark >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        n_obj, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

#[test]
fn test_default_off_layer_hidden_with_empty_exclusions() {
    // /D/OFF marks Watermark off — content should be hidden even when the
    // caller passes no excluded_layers.
    let doc = PdfDocument::from_bytes(build_pdf_with_default_off_layer()).unwrap();
    let (data, w, h) = render_raw(&doc, HashSet::new());
    let (r, g, b, _) = sample_region(&data, w, h, 100, 100, 30, 30);
    assert!(
        r > 240.0 && g > 240.0 && b > 240.0,
        "Default-off OCG content should be hidden: ({r}, {g}, {b})"
    );
    let (r2, g2, b2, _) = sample_region(&data, w, h, 350, 100, 30, 30);
    assert!(
        r2 < 80.0 && g2 > 200.0 && b2 < 80.0,
        "Unrelated content should remain: ({r2}, {g2}, {b2})"
    );
}

#[test]
fn test_ve_nested_expression() {
    // /VE = [/And 7 0 R [/Not 8 0 R]]   — A on AND B off
    let doc = PdfDocument::from_bytes(build_pdf_with_ve(b"[/And 7 0 R [/Not 8 0 R]]")).unwrap();
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::new()),
        "And(A, Not(B)): A on, B on -> hidden"
    );
    assert!(
        ocmd_scope_is_visible(&doc, HashSet::from(["B".to_string()])),
        "And(A, Not(B)): A on, B excluded -> visible"
    );
    assert!(
        !ocmd_scope_is_visible(&doc, HashSet::from(["A".to_string(), "B".to_string()])),
        "And(A, Not(B)): A excluded -> hidden"
    );
}
