//! Tests for OCG layer and Separation ink filtering in text extraction.
//!
//! Bug 1: XObject cached spans bypass suppression — filtered extraction
//!         replays unfiltered cached spans, leaking excluded content.
//! Bug 2: `inside_excluded_ink` not re-evaluated after Form XObject restore —
//!         ink exclusion state lost after XObject processing.
//! Bug 3: `get_layers` doc comment bleeds into `extract_chars` — verified
//!         separately via `cargo doc`.

use std::collections::HashSet;

use pdf_oxide::document::PdfDocument;

// ============================================================================
// Helper: build a PDF with an OCG layer wrapping text inside a Form XObject
// ============================================================================

/// Build a PDF where a Form XObject contains OCG-tagged text.
///
/// Page content:  /Fm0 Do
/// Form XObject:  BDC /OC /MC0  →  BT (LAYERED) Tj ET  →  EMC
///                                  BT (VISIBLE) Tj ET
///
/// The OCG "HiddenLayer" is defined in OCProperties and referenced via
/// the Form XObject's /Resources /Properties.
fn build_pdf_with_ocg_in_xobject() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Obj 1: Catalog with OCProperties
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n\
           /OCProperties << /OCGs [8 0 R] /D << /ON [8 0 R] >> >> >>\nendobj\n\n",
    );

    // Obj 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    // Obj 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 6 0 R >> /XObject << /Fm0 5 0 R >> >> >>\nendobj\n\n",
    );

    // Obj 4: Page content — just invoke the Form XObject
    let page_content = b"/Fm0 Do";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject with OCG-tagged text + untagged text
    // BDC syntax: /Tag /PropertiesName BDC ... EMC
    let form_stream =
        b"/OC /MC0 BDC BT /F1 12 Tf 50 700 Td (LAYERED) Tj ET EMC BT /F1 12 Tf 50 650 Td (VISIBLE) Tj ET";
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
            /Resources << /Font << /F1 6 0 R >> /Properties << /MC0 8 0 R >> >>\n\
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

    // Obj 7: (unused, keep numbering simple)
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\nnull\nendobj\n\n");

    // Obj 8: OCG dictionary
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"8 0 obj\n<< /Type /OCG /Name /HiddenLayer >>\nendobj\n\n");

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

// ============================================================================
// Helper: build a PDF with Separation ink color space in a Form XObject
// ============================================================================

/// Build a PDF where a Form XObject switches to a Separation ink, renders text,
/// then returns. After the XObject, the page renders more text in DeviceGray.
///
/// Page content:  BT (BEFORE) Tj ET  /Fm0 Do  BT (AFTER) Tj ET
/// Form XObject:  /CS1 cs  BT (SPOT_INK) Tj ET
///
/// /CS1 is [/Separation /SpotRed /DeviceRGB ...].
fn build_pdf_with_ink_in_xobject() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Obj 1: Catalog
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");

    // Obj 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    // Obj 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 6 0 R >> /XObject << /Fm0 5 0 R >> >> >>\nendobj\n\n",
    );

    // Obj 4: Page content — text before XObject, invoke XObject, text after
    let page_content =
        b"BT /F1 12 Tf 50 700 Td (BEFORE) Tj ET /Fm0 Do BT /F1 12 Tf 50 600 Td (AFTER) Tj ET";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject — sets Separation color space, then renders text
    let form_stream = b"/CS1 cs 1 scn BT /F1 12 Tf 50 650 Td (SPOT_INK) Tj ET";
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
            /Resources << /Font << /F1 6 0 R >>\n\
            /ColorSpace << /CS1 7 0 R >> >>\n\
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

    // Obj 7: Separation color space [/Separation /SpotRed /DeviceRGB <tint fn>]
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n[/Separation /SpotRed /DeviceRGB 8 0 R]\nendobj\n\n");

    // Obj 8: Tint transform function (Type 4 PostScript calculator)
    let tint_fn = b"{ 0 0 }";
    offsets.push(pdf.len());
    let fn_hdr = format!(
        "8 0 obj\n<< /FunctionType 4 /Domain [0.0 1.0]\n\
         /Range [0.0 1.0 0.0 1.0 0.0 1.0] /Length {} >>\nstream\n",
        tint_fn.len()
    );
    pdf.extend_from_slice(fn_hdr.as_bytes());
    pdf.extend_from_slice(tint_fn);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

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

// ============================================================================
// Bug 1: XObject cached spans bypass suppression
// ============================================================================

#[test]
fn test_xobject_cache_does_not_leak_filtered_content() {
    let pdf_bytes = build_pdf_with_ocg_in_xobject();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // First: unfiltered extraction — caches XObject spans
    let text_all = doc.extract_text(0).expect("unfiltered extract");
    assert!(
        text_all.contains("LAYERED"),
        "Unfiltered should contain LAYERED, got: {:?}",
        text_all
    );
    assert!(
        text_all.contains("VISIBLE"),
        "Unfiltered should contain VISIBLE, got: {:?}",
        text_all
    );

    // Second: filtered extraction excluding "HiddenLayer"
    // BUG: cached XObject spans from first call are replayed verbatim,
    // so LAYERED text leaks through despite being in excluded layer.
    let excluded = HashSet::from(["HiddenLayer".to_string()]);
    let text_filtered = doc
        .extract_text_filtered(0, excluded, HashSet::new())
        .expect("filtered extract");

    assert!(
        text_filtered.contains("VISIBLE"),
        "Filtered should still contain VISIBLE, got: {:?}",
        text_filtered
    );
    assert!(
        !text_filtered.contains("LAYERED"),
        "Filtered must NOT contain LAYERED (XObject cache leak), got: {:?}",
        text_filtered
    );
}

// ============================================================================
// Bug 2: inside_excluded_ink not re-evaluated after XObject restore
// ============================================================================

#[test]
fn test_ink_exclusion_restored_after_xobject() {
    let pdf_bytes = build_pdf_with_ink_in_xobject();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Extract with "SpotRed" excluded
    let excluded_inks = HashSet::from(["SpotRed".to_string()]);
    let text = doc
        .extract_text_filtered(0, HashSet::new(), excluded_inks)
        .expect("filtered extract");

    // SPOT_INK should be suppressed (it's in SpotRed color space)
    assert!(
        !text.contains("SPOT_INK"),
        "SPOT_INK should be suppressed by ink filter, got: {:?}",
        text
    );

    // BEFORE and AFTER are in DeviceGray — should NOT be suppressed.
    // BUG: the Form XObject sets inside_excluded_ink=true for SpotRed,
    // but after XObject restore, inside_excluded_ink is not re-evaluated
    // back to false. So AFTER text is incorrectly suppressed.
    assert!(
        text.contains("BEFORE"),
        "BEFORE should be visible (DeviceGray), got: {:?}",
        text
    );
    assert!(
        text.contains("AFTER"),
        "AFTER should be visible (DeviceGray, post-XObject restore), got: {:?}",
        text
    );
}

// ============================================================================
// Bug 3: text-only parser skips color operators — ink filtering is inert
// ============================================================================

/// Build a PDF with Separation ink color space directly in the page stream.
/// No XObjects — isolates the parser from caching.
///
/// Page content:
///   BT (PROCESS) Tj ET          ← default (DeviceGray) fill
///   /CS1 cs 1 scn               ← switch to Separation /SpotRed
///   BT (SPOT) Tj ET             ← text in SpotRed ink
///   0 g                         ← switch back to DeviceGray
///   BT (RESTORED) Tj ET         ← text in DeviceGray again
fn build_pdf_with_inline_separation() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Obj 1: Catalog
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");

    // Obj 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    // Obj 3: Page with ColorSpace in resources
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >>\n\
           /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n\n",
    );

    // Obj 4: Content stream
    let content = b"BT /F1 12 Tf 50 700 Td (PROCESS) Tj ET /CS1 cs 1 scn BT /F1 12 Tf 50 650 Td (SPOT) Tj ET 0 g BT /F1 12 Tf 50 600 Td (RESTORED) Tj ET";
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

    // Obj 6: Separation color space array
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n[/Separation /SpotRed /DeviceRGB 7 0 R]\nendobj\n\n");

    // Obj 7: Tint transform function
    let tint_fn = b"{ 0 0 }";
    offsets.push(pdf.len());
    let fn_hdr = format!(
        "7 0 obj\n<< /FunctionType 4 /Domain [0.0 1.0]\n\
         /Range [0.0 1.0 0.0 1.0 0.0 1.0] /Length {} >>\nstream\n",
        tint_fn.len()
    );
    pdf.extend_from_slice(fn_hdr.as_bytes());
    pdf.extend_from_slice(tint_fn);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

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

#[test]
fn test_ink_filtering_blocked_by_text_only_parser() {
    // The text-only parser (parse_and_execute_text_only) skips color operators
    // (cs, rg, g, k) for performance. This means the SetFillColorSpace handler
    // never fires, and ink filtering is completely inert.
    //
    // This test proves the bug: excluding "SpotRed" has no effect because the
    // `cs` operator is never delivered to the extractor.
    let pdf_bytes = build_pdf_with_inline_separation();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Unfiltered: all three texts present
    let text_all = doc.extract_text(0).expect("unfiltered");
    assert!(text_all.contains("PROCESS"), "got: {:?}", text_all);
    assert!(text_all.contains("SPOT"), "got: {:?}", text_all);
    assert!(text_all.contains("RESTORED"), "got: {:?}", text_all);

    // Filtered: exclude SpotRed ink
    let excluded_inks = HashSet::from(["SpotRed".to_string()]);
    let text_filtered = doc
        .extract_text_filtered(0, HashSet::new(), excluded_inks)
        .expect("filtered");

    // PROCESS and RESTORED are in DeviceGray — must survive
    assert!(
        text_filtered.contains("PROCESS"),
        "PROCESS (DeviceGray) should survive ink filter, got: {:?}",
        text_filtered
    );
    assert!(
        text_filtered.contains("RESTORED"),
        "RESTORED (DeviceGray after `0 g`) should survive ink filter, got: {:?}",
        text_filtered
    );

    // SPOT is in SpotRed Separation — must be suppressed
    assert!(
        !text_filtered.contains("SPOT"),
        "SPOT (SpotRed ink) must be suppressed by ink filter, got: {:?}",
        text_filtered
    );
}

// ============================================================================
// Basic OCG filtering (no XObject cache involvement)
// ============================================================================
// Task 1: region + filter combo
// ============================================================================

fn build_pdf_for_region_and_filter() -> Vec<u8> {
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

    let content = b"BT /F1 12 Tf 50 700 Td (TOP_VISIBLE) Tj ET \
                    /OC /MC0 BDC BT /F1 12 Tf 50 650 Td (TOP_HIDDEN) Tj ET EMC \
                    BT /F1 12 Tf 50 100 Td (BOT_VISIBLE) Tj ET";
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
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Overlay >>\nendobj\n\n");

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
fn test_extract_text_filtered_respects_region() {
    let pdf_bytes = build_pdf_for_region_and_filter();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded = HashSet::from(["Overlay".to_string()]);
    let chars = doc
        .extract_chars_filtered(0, excluded, HashSet::new())
        .expect("filtered chars");

    use pdf_oxide::geometry::Rect;
    use pdf_oxide::layout::{RectFilterMode, SpatialCollectionFiltering};
    let region = Rect::new(0.0, 600.0, 612.0, 200.0);
    let filtered: Vec<_> = chars.filter_by_rect(&region, RectFilterMode::Intersects);
    let text: String = filtered.iter().map(|c| c.char).collect();

    assert!(
        text.contains("TOP_VISIBLE"),
        "TOP_VISIBLE should survive both filters, got: {:?}",
        text
    );
    assert!(
        !text.contains("TOP_HIDDEN"),
        "TOP_HIDDEN should be excluded by layer filter, got: {:?}",
        text
    );
    assert!(
        !text.contains("BOT_VISIBLE"),
        "BOT_VISIBLE should be excluded by region filter, got: {:?}",
        text
    );
}

fn build_pdf_for_full_pipeline_region_test() -> Vec<u8> {
    // Page has 3 visible-after-filter runs arranged so that pipeline
    // assembly (sort + whitespace + line breaks) is observable:
    //   y=720 x=50:  "Hello"   (line 1, left)
    //   y=720 x=200: "World"   (line 1, right) -> space between expected
    //   y=700 x=50:  "Second"  (line 2)         -> newline before expected
    //   y=650 (excluded OCG layer): "HIDDEN"
    //   y=100 (outside region): "Footer"
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

    let content = b"BT /F1 12 Tf 50 720 Td (Hello) Tj ET \
                    BT /F1 12 Tf 200 720 Td (World) Tj ET \
                    BT /F1 12 Tf 50 700 Td (Second) Tj ET \
                    /OC /MC0 BDC BT /F1 12 Tf 50 650 Td (HIDDEN) Tj ET EMC \
                    BT /F1 12 Tf 50 100 Td (Footer) Tj ET";
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
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Overlay >>\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", n_obj);
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
fn test_extract_text_filtered_in_rect_uses_full_pipeline() {
    // extract_text_filtered_in_rect must go through the full text-assembly
    // pipeline: reading-order sort, whitespace between adjacent runs,
    // line breaks between rows. Composing layer/ink filters with a region
    // must NOT regress to char-stream concatenation.
    let pdf_bytes = build_pdf_for_full_pipeline_region_test();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded = HashSet::from(["Overlay".to_string()]);
    let region = pdf_oxide::geometry::Rect::new(0.0, 600.0, 612.0, 200.0);
    let text = doc
        .extract_text_filtered_in_rect(
            0,
            excluded,
            HashSet::new(),
            region,
            pdf_oxide::layout::RectFilterMode::Intersects,
        )
        .expect("filtered text in rect");

    assert!(text.contains("Hello"), "Hello missing: {:?}", text);
    assert!(text.contains("World"), "World missing: {:?}", text);
    assert!(text.contains("Second"), "Second missing: {:?}", text);
    assert!(!text.contains("HIDDEN"), "HIDDEN should be layer-excluded: {:?}", text);
    assert!(!text.contains("Footer"), "Footer should be region-excluded: {:?}", text);

    // Whitespace between "Hello" and "World" on the same row.
    let h = text.find("Hello").unwrap();
    let w = text.find("World").unwrap();
    let between = &text[h + "Hello".len()..w];
    assert!(
        between.chars().any(|c| c.is_whitespace()),
        "Expected whitespace between Hello and World; got {:?}",
        text
    );

    // Newline or other separator between line 1 and "Second" (line 2).
    let s = text.find("Second").unwrap();
    let line_break = &text[w + "World".len()..s];
    assert!(line_break.contains('\n'), "Expected line break before 'Second'; got {:?}", text);
}

// ============================================================================
// Basic OCG filtering (no XObject cache involvement)
// ============================================================================

/// Build a minimal PDF with inline OCG-tagged text (no XObjects).
fn build_pdf_with_inline_ocg() -> Vec<u8> {
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

    // Obj 4: Content stream with OCG-tagged and untagged text
    // BDC syntax: /Tag /PropertiesName BDC ... EMC
    let content =
        b"BT /F1 12 Tf 50 700 Td (ALWAYS) Tj ET /OC /MC0 BDC BT /F1 12 Tf 50 650 Td (HIDDEN) Tj ET EMC";
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

    // Obj 6: OCG
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /OCG /Name /Overlay >>\nendobj\n\n");

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

#[test]
fn test_ocg_layer_filtering_inline() {
    let pdf_bytes = build_pdf_with_inline_ocg();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // Unfiltered: both texts present
    let text_all = doc.extract_text(0).expect("unfiltered");
    assert!(text_all.contains("ALWAYS"), "got: {:?}", text_all);
    assert!(text_all.contains("HIDDEN"), "got: {:?}", text_all);

    // Filtered: exclude "Overlay" layer
    let excluded = HashSet::from(["Overlay".to_string()]);
    let text_filtered = doc
        .extract_text_filtered(0, excluded, HashSet::new())
        .expect("filtered");
    assert!(text_filtered.contains("ALWAYS"), "got: {:?}", text_filtered);
    assert!(
        !text_filtered.contains("HIDDEN"),
        "HIDDEN should be excluded, got: {:?}",
        text_filtered
    );
}

#[test]
fn test_get_layers_returns_ocg_names() {
    let pdf_bytes = build_pdf_with_inline_ocg();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let layers = doc.get_layers().expect("get_layers");
    assert!(
        layers.contains(&"Overlay".to_string()),
        "Should find 'Overlay' layer, got: {:?}",
        layers
    );
}

#[test]
fn test_get_page_inks_returns_separation_names() {
    let pdf_bytes = build_pdf_with_ink_in_xobject();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    // The Separation color space is in the Form XObject's resources, not the
    // page's direct resources. get_page_inks only checks page resources, so
    // it won't find SpotRed here. This is a known limitation — the test
    // documents expected behavior.
    let inks = doc.get_page_inks(0).expect("get_page_inks");
    // SpotRed is in XObject resources, not page resources — empty is correct
    assert!(
        !inks.contains(&"SpotRed".to_string()),
        "SpotRed is in XObject resources, not page, got: {:?}",
        inks
    );
}

/// Build a minimal PDF whose page-level /Resources/ColorSpace declares
/// both a /Separation and a /DeviceN entry. Verifies the happy path of
/// get_page_inks — the previous test only covered the negative case
/// (XObject-local color space).
fn build_pdf_with_page_level_inks() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100]\n\
           /Contents 4 0 R\n\
           /Resources << /ColorSpace << /CS1 5 0 R /CS2 6 0 R >> >> >>\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n[/Separation /PANTONE#20185#20C /DeviceCMYK 7 0 R]\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n[/DeviceN [/Cyan /Magenta /SpotGold] /DeviceCMYK 7 0 R]\nendobj\n\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 4 /C0 [0 0 0 0] /C1 [1 0 0 0] >>\nendobj\n\n",
    );

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", n_obj);
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
fn test_get_page_inks_happy_path() {
    let pdf_bytes = build_pdf_with_page_level_inks();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");
    let inks = doc.get_page_inks(0).expect("get_page_inks");

    // Both Separation and DeviceN ink names should be returned.
    assert!(
        inks.iter()
            .any(|i| i == "PANTONE 185 C" || i == "PANTONE#20185#20C"),
        "Separation ink missing: {:?}",
        inks
    );
    assert!(inks.contains(&"SpotGold".to_string()), "DeviceN ink missing: {:?}", inks);
    // Cyan/Magenta are process colorant names declared as DeviceN components
    // — they're enumerated as plate-able inks at this layer.
    assert!(inks.contains(&"Cyan".to_string()), "DeviceN component missing: {:?}", inks);
}

#[test]
fn test_empty_filters_fall_through_to_normal_extraction() {
    let pdf_bytes = build_pdf_with_inline_ocg();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let text_normal = doc.extract_text(0).expect("normal");
    let text_filtered = doc
        .extract_text_filtered(0, HashSet::new(), HashSet::new())
        .expect("empty filters");

    assert_eq!(
        text_normal, text_filtered,
        "Empty filters should produce identical output to unfiltered extraction"
    );
}

// ============================================================================
// Task 2: OCMD (Optional Content Membership Dictionary) support
// ============================================================================

fn build_pdf_with_ocmd() -> Vec<u8> {
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
    let content = b"BT /F1 12 Tf 50 700 Td (ALWAYS) Tj ET /OC /MC0 BDC BT /F1 12 Tf 50 650 Td (MEMBER) Tj ET EMC";
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
    pdf.extend_from_slice(b"7 0 obj\n<< /Type /OCG /Name /MemberLayer >>\nendobj\n\n");

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
fn test_ocmd_layer_filtering() {
    let pdf_bytes = build_pdf_with_ocmd();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let layers = doc.get_layers().expect("get_layers");
    assert!(layers.contains(&"MemberLayer".to_string()), "got: {:?}", layers);

    let text_all = doc.extract_text(0).expect("unfiltered");
    assert!(text_all.contains("ALWAYS"), "got: {:?}", text_all);
    assert!(text_all.contains("MEMBER"), "got: {:?}", text_all);

    let excluded = HashSet::from(["MemberLayer".to_string()]);
    let text_filtered = doc
        .extract_text_filtered(0, excluded, HashSet::new())
        .expect("filtered");
    assert!(text_filtered.contains("ALWAYS"), "got: {:?}", text_filtered);
    assert!(
        !text_filtered.contains("MEMBER"),
        "MEMBER should be excluded via OCMD resolution, got: {:?}",
        text_filtered
    );
}

// ============================================================================
// Task 3: DeviceN all-or-nothing semantics
// ============================================================================

fn build_pdf_with_devicen() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 5 0 R >>\n\
           /ColorSpace << /CS1 6 0 R >> >> >>\nendobj\n\n",
    );

    let content = b"BT /F1 12 Tf 50 700 Td (PLAIN) Tj ET \
                    /CS1 cs 1 0 scn BT /F1 12 Tf 50 650 Td (MIXED_INK) Tj ET";
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
    pdf.extend_from_slice(b"6 0 obj\n[/DeviceN [/Cyan /SpotGold] /DeviceRGB 7 0 R]\nendobj\n\n");

    let tint_fn = b"{ 0 exch }";
    offsets.push(pdf.len());
    let fn_hdr = format!(
        "7 0 obj\n<< /FunctionType 4 /Domain [0.0 1.0 0.0 1.0]\n\
         /Range [0.0 1.0 0.0 1.0 0.0 1.0] /Length {} >>\nstream\n",
        tint_fn.len()
    );
    pdf.extend_from_slice(fn_hdr.as_bytes());
    pdf.extend_from_slice(tint_fn);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

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
fn test_devicen_excludes_entire_colorspace_if_any_ink_matches() {
    // Documents the all-or-nothing DeviceN behavior:
    // Excluding "SpotGold" suppresses ALL text in the DeviceN space,
    // even though Cyan (a process color) is also part of the space.
    let pdf_bytes = build_pdf_with_devicen();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded_inks = HashSet::from(["SpotGold".to_string()]);
    let text = doc
        .extract_text_filtered(0, HashSet::new(), excluded_inks)
        .expect("filtered");

    assert!(text.contains("PLAIN"), "PLAIN (DeviceGray) should survive, got: {:?}", text);
    assert!(
        !text.contains("MIXED_INK"),
        "MIXED_INK should be suppressed (DeviceN contains excluded SpotGold), got: {:?}",
        text
    );
}

// ============================================================================
// Pipeline parity: filtering non-matching layers must not alter output
// ============================================================================

/// Build a PDF with a table-like layout and an OCG layer that doesn't
/// overlap with any content. Filtering this layer should produce output
/// identical to unfiltered extraction.
fn build_pdf_with_table_and_unrelated_layer() -> Vec<u8> {
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

    // Content: multiple text lines (simulating rows), plus an empty OCG layer
    let content = b"BT /F1 12 Tf 50 700 Td (Name) Tj 200 0 Td (Age) Tj ET \
                    BT /F1 12 Tf 50 680 Td (Alice) Tj 200 0 Td (30) Tj ET \
                    BT /F1 12 Tf 50 660 Td (Bob) Tj 200 0 Td (25) Tj ET \
                    /OC /MC0 BDC EMC";
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
fn test_filtering_unrelated_layer_produces_identical_output() {
    let pdf_bytes = build_pdf_with_table_and_unrelated_layer();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let text_normal = doc.extract_text(0).expect("unfiltered");
    let text_filtered = doc
        .extract_text_filtered(0, HashSet::from(["Dieline".to_string()]), HashSet::new())
        .expect("filtered with non-matching layer");

    assert_eq!(
        text_normal, text_filtered,
        "Excluding a layer with no content must produce identical output.\n\
         Unfiltered: {:?}\n\
         Filtered:   {:?}",
        text_normal, text_filtered
    );
}

// ============================================================================
// Bug 4: DeviceN names array as indirect reference is not resolved during
// ink-filter classification — mirrors a real-world shape
// `/CS6 [/DeviceN 4 0 R /DeviceCMYK <attrs>]` where the names list is a
// separate indirect object shared across multiple colour spaces.
// ============================================================================

/// Build a PDF where a Form XObject paints text in a DeviceN colour space
/// whose names array is an **indirect reference** to another object,
/// matching the banana-label `/CS6 [/DeviceN 4 0 R …]` shape.
///
/// Page content :  BT (BEFORE) Tj ET  /Fm0 Do  BT (AFTER) Tj ET
/// Form content :  /CS1 cs 1 0 0 0 scn BT (NESTED_SPOT_TEXT) Tj ET
/// Form resources : /CS1 → 7 0 R
/// Obj 7 : [/DeviceN 8 0 R /DeviceCMYK 9 0 R]   ← names slot is INDIRECT
/// Obj 8 : [/Cyan /Magenta /Yellow /SpotRed]    ← the names list
fn build_pdf_with_indirect_devicen_names_in_form() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Obj 1: Catalog
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n");

    // Obj 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n");

    // Obj 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
           /Contents 4 0 R\n\
           /Resources << /Font << /F1 6 0 R >> /XObject << /Fm0 5 0 R >> >> >>\nendobj\n\n",
    );

    // Obj 4: Page content
    let page_content =
        b"BT /F1 12 Tf 50 700 Td (BEFORE) Tj ET /Fm0 Do BT /F1 12 Tf 50 600 Td (AFTER) Tj ET";
    offsets.push(pdf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    pdf.extend_from_slice(hdr.as_bytes());
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    // Obj 5: Form XObject — selects /CS1 (DeviceN), sets all 4 components,
    // then paints (NESTED_SPOT_TEXT). Because /CS1's names array includes
    // /SpotRed, exclude_inks={SpotRed} must suppress this text.
    let form_stream = b"/CS1 cs 1 0 0 0 scn BT /F1 12 Tf 50 650 Td (NESTED_SPOT_TEXT) Tj ET";
    offsets.push(pdf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
            /Resources << /Font << /F1 6 0 R >>\n\
            /ColorSpace << /CS1 7 0 R >> >>\n\
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

    // Obj 7: DeviceN colour space. Names slot is INDIRECT (8 0 R).
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"7 0 obj\n[/DeviceN 8 0 R /DeviceCMYK 9 0 R]\nendobj\n\n");

    // Obj 8: the actual names array.
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"8 0 obj\n[/Cyan /Magenta /Yellow /SpotRed]\nendobj\n\n");

    // Obj 9: identity-like Type 4 tint transform (output = input on all four
    // process channels; just needs to be present so the colour space parses).
    let tint_fn = b"{ 0 0 0 0 }";
    offsets.push(pdf.len());
    let fn_hdr = format!(
        "9 0 obj\n<< /FunctionType 4 /Domain [0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0]\n\
         /Range [0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0] /Length {} >>\nstream\n",
        tint_fn.len()
    );
    pdf.extend_from_slice(fn_hdr.as_bytes());
    pdf.extend_from_slice(tint_fn);
    pdf.extend_from_slice(b"\nendstream\nendobj\n\n");

    let xref_offset = pdf.len();
    let n_obj = offsets.len() + 1;
    let mut xref = format!("xref\n0 {}\n", n_obj);
    xref.push_str("0000000000 65535 f \n");
    for off in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", off));
    }
    pdf.extend_from_slice(xref.as_bytes());
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n_obj} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
            .as_bytes(),
    );
    pdf
}

#[test]
fn test_ink_filter_resolves_indirect_devicen_names_in_nested_form() {
    let pdf_bytes = build_pdf_with_indirect_devicen_names_in_form();
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("parse PDF");

    let excluded_inks = HashSet::from(["SpotRed".to_string()]);
    let text = doc
        .extract_text_filtered(0, HashSet::new(), excluded_inks)
        .expect("filtered extract");

    // The DeviceN colour space's names array is an indirect reference. The
    // extractor's is_excluded_ink_color_space must resolve it before
    // checking colorant names; otherwise NESTED_SPOT_TEXT slips through.
    assert!(
        !text.contains("NESTED_SPOT_TEXT"),
        "NESTED_SPOT_TEXT should be suppressed — SpotRed is one of the DeviceN \
         colorants (via indirect names array). Got: {:?}",
        text
    );

    // Page-level text outside the DeviceN colour space must still be visible.
    assert!(
        text.contains("BEFORE"),
        "BEFORE should be visible (DeviceGray, before form invocation). Got: {:?}",
        text
    );
    assert!(
        text.contains("AFTER"),
        "AFTER should be visible (DeviceGray, after form returns). Got: {:?}",
        text
    );
}
