//! Spec-driven tests for deep spot-ink discovery through nested Form
//! XObject resources (ISO 32000-1 §8.6.6.2 Separation, §8.6.6.3 DeviceN,
//! §8.10 Form XObjects).
//!
//! Fixture builders mirror the pattern in `tests/test_separation_overprint.rs`:
//! hand-rolled PDF byte buffers with explicit object numbers and explicit xref.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::render_separations;

/// Build a single-page PDF whose page-level /Resources/ColorSpace is empty
/// and whose content stream invokes one Form XObject. The Form XObject's
/// /Resources/ColorSpace declares a /Separation /SpotRed space.
fn build_pdf_with_spot_in_nested_form() -> Vec<u8> {
    let page_content = b"/Fm0 Do\n";
    let form_content = b"";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /Fm0 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(page_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /ColorSpace << /CS1 6 0 R >> >> \
            /Length {} >>\nstream\n",
        form_content.len()
    );
    buf.extend_from_slice(form_hdr.as_bytes());
    buf.extend_from_slice(form_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"6 0 obj\n[/Separation /SpotRed /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [0 1 0 0] >>\nendobj\n",
    );

    finalize_pdf(buf, offsets)
}

fn finalize_pdf(mut buf: Vec<u8>, offsets: Vec<usize>) -> Vec<u8> {
    let xref_offset = buf.len();
    buf.extend_from_slice(b"xref\n");
    buf.extend_from_slice(format!("0 {}\n", offsets.len() + 1).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len() + 1,
            xref_offset
        )
        .as_bytes(),
    );
    buf
}

#[test]
fn deep_finds_spot_declared_in_nested_form() {
    let doc = PdfDocument::from_bytes(build_pdf_with_spot_in_nested_form()).expect("parse");

    // Shallow API: misses the nested declaration (documented contract).
    let shallow = doc.get_page_inks(0).expect("shallow");
    assert!(
        !shallow.contains(&"SpotRed".to_string()),
        "shallow get_page_inks must NOT find XObject-local inks; got {:?}",
        shallow
    );

    // Deep API: finds it.
    let deep = doc.get_page_inks_deep(0).expect("deep");
    assert!(
        deep.contains(&"SpotRed".to_string()),
        "deep walk must surface SpotRed declared in nested form; got {:?}",
        deep
    );
}

#[test]
fn deep_finds_declared_but_unused_spot() {
    // The form declares /Separation /SpotRed in its /Resources/ColorSpace
    // but its content stream never paints with it (form_content is empty
    // in the fixture). Discovery surfaces declared-not-used inks.
    let doc = PdfDocument::from_bytes(build_pdf_with_spot_in_nested_form()).expect("parse");
    let deep = doc.get_page_inks_deep(0).expect("deep");
    assert!(
        deep.contains(&"SpotRed".to_string()),
        "declared-but-unused spot must be discovered; got {:?}",
        deep
    );
}

/// Two-level nesting: Page → FormA → FormB, where only FormB declares the spot.
fn build_pdf_with_spot_two_levels_deep() -> Vec<u8> {
    let page_content = b"/FmA Do\n";
    let form_a_content = b"/FmB Do\n";
    let form_b_content = b"";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /FmA 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(page_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fma_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /XObject << /FmB 6 0 R >> >> \
            /Length {} >>\nstream\n",
        form_a_content.len()
    );
    buf.extend_from_slice(fma_hdr.as_bytes());
    buf.extend_from_slice(form_a_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fmb_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /ColorSpace << /CS1 7 0 R >> >> \
            /Length {} >>\nstream\n",
        form_b_content.len()
    );
    buf.extend_from_slice(fmb_hdr.as_bytes());
    buf.extend_from_slice(form_b_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"7 0 obj\n[/Separation /DeepSpot /DeviceCMYK 8 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"8 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [1 0 0 0] >>\nendobj\n",
    );
    finalize_pdf(buf, offsets)
}

#[test]
fn deep_finds_spot_two_levels_deep() {
    let doc = PdfDocument::from_bytes(build_pdf_with_spot_two_levels_deep()).expect("parse");
    let deep = doc.get_page_inks_deep(0).expect("deep");
    assert!(
        deep.contains(&"DeepSpot".to_string()),
        "two-level nested form spot must surface; got {:?}",
        deep
    );
}

/// Cycle: FmA invokes FmB which invokes FmA back. The walker's visited-set
/// must terminate the recursion without an error and without unbounded growth.
fn build_pdf_with_cyclic_forms() -> Vec<u8> {
    let page_content = b"/FmA Do\n";
    let form_a_content = b"/FmB Do\n";
    let form_b_content = b"/FmA Do\n"; // cycle back

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /FmA 5 0 R /FmB 6 0 R >> \
                       /ColorSpace << /CS1 7 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(page_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fma_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /XObject << /FmB 6 0 R >> >> \
            /Length {} >>\nstream\n",
        form_a_content.len()
    );
    buf.extend_from_slice(fma_hdr.as_bytes());
    buf.extend_from_slice(form_a_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fmb_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /XObject << /FmA 5 0 R >> >> \
            /Length {} >>\nstream\n",
        form_b_content.len()
    );
    buf.extend_from_slice(fmb_hdr.as_bytes());
    buf.extend_from_slice(form_b_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"7 0 obj\n[/Separation /CycleSpot /DeviceCMYK 8 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"8 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [1 0 0 0] >>\nendobj\n",
    );
    finalize_pdf(buf, offsets)
}

#[test]
fn deep_terminates_on_form_cycle() {
    let doc = PdfDocument::from_bytes(build_pdf_with_cyclic_forms()).expect("parse");
    let deep = doc.get_page_inks_deep(0).expect("deep");
    let count = deep.iter().filter(|s| s.as_str() == "CycleSpot").count();
    assert_eq!(count, 1, "CycleSpot must appear exactly once after dedupe; got {:?}", deep);
}

/// Two sibling forms whose /Resources/ColorSpace each point at the SAME
/// `/Separation /SharedSpot` object — dedupe must collapse them.
fn build_pdf_with_shared_spot_in_two_forms() -> Vec<u8> {
    let page_content = b"/FmA Do\n/FmB Do\n";
    let form_a_content = b"";
    let form_b_content = b"";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /XObject << /FmA 5 0 R /FmB 6 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(page_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fma_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /ColorSpace << /CSa 7 0 R >> >> \
            /Length {} >>\nstream\n",
        form_a_content.len()
    );
    buf.extend_from_slice(fma_hdr.as_bytes());
    buf.extend_from_slice(form_a_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    let fmb_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
            /Resources << /ColorSpace << /CSb 7 0 R >> >> \
            /Length {} >>\nstream\n",
        form_b_content.len()
    );
    buf.extend_from_slice(fmb_hdr.as_bytes());
    buf.extend_from_slice(form_b_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"7 0 obj\n[/Separation /SharedSpot /DeviceCMYK 8 0 R]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"8 0 obj\n<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0 0] /C1 [1 0 0 0] >>\nendobj\n",
    );
    finalize_pdf(buf, offsets)
}

#[test]
fn deep_dedupes_spot_declared_in_multiple_forms() {
    let doc = PdfDocument::from_bytes(build_pdf_with_shared_spot_in_two_forms()).expect("parse");
    let deep = doc.get_page_inks_deep(0).expect("deep");
    let count = deep.iter().filter(|s| s.as_str() == "SharedSpot").count();
    assert_eq!(count, 1, "SharedSpot must dedupe across forms; got {:?}", deep);
}

#[test]
fn render_separations_allocates_plate_for_nested_form_spot() {
    // The bug that motivated this whole plan: a spot ink declared only
    // inside a Form XObject must show up as a plate from render_separations.
    let doc = PdfDocument::from_bytes(build_pdf_with_spot_in_nested_form()).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let plate_names: Vec<&str> = plates.iter().map(|p| p.ink_name.as_str()).collect();
    assert!(
        plate_names.contains(&"SpotRed"),
        "render_separations must allocate a plate for nested-form spot; got {:?}",
        plate_names
    );
}

/// DeviceN colour space whose names array is an **indirect reference**:
///   /CS1 → [/DeviceN 4 0 R /DeviceCMYK <<attrs>>]
///   4 0 obj [/Cyan /Magenta /Yellow /yellow#20fluorescent] endobj
///
/// This is a common emission pattern for DeviceN spaces with many inks
/// — the names list is shared as a separate indirect object. The
/// extractor must resolve `4 0 R` before pulling ink names out.
fn build_pdf_with_devicen_indirect_names() -> Vec<u8> {
    let page_content = b"";

    let mut buf = Vec::new();
    let mut offsets = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Contents 4 0 R \
           /Resources << /ColorSpace << /CS1 5 0 R >> >> >>\nendobj\n",
    );
    offsets.push(buf.len());
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(page_content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    offsets.push(buf.len());
    // CS1 → DeviceN with the names list as an indirect ref (6 0 R).
    buf.extend_from_slice(b"5 0 obj\n[/DeviceN 6 0 R /DeviceCMYK 7 0 R]\nendobj\n");
    offsets.push(buf.len());
    // Indirect names list with a #20-escaped multi-word ink name.
    buf.extend_from_slice(b"6 0 obj\n[/Cyan /Magenta /Yellow /yellow#20fluorescent]\nendobj\n");
    offsets.push(buf.len());
    buf.extend_from_slice(
        b"7 0 obj\n<< /FunctionType 2 /Domain [0 1 0 1 0 1 0 1] /N 1 \
            /C0 [0 0 0 0] /C1 [0 0 0 1] >>\nendobj\n",
    );
    finalize_pdf(buf, offsets)
}

#[test]
fn deep_resolves_indirect_devicen_names_array() {
    let doc = PdfDocument::from_bytes(build_pdf_with_devicen_indirect_names()).expect("parse");
    let deep = doc.get_page_inks_deep(0).expect("deep");
    assert!(
        deep.contains(&"yellow fluorescent".to_string()),
        "deep walk must resolve indirect /DeviceN names arrays and decode #20 escapes; got {:?}",
        deep
    );
    // Same content is reachable from the shallow scan too — page-level CS.
    let shallow = doc.get_page_inks(0).expect("shallow");
    assert!(
        shallow.contains(&"yellow fluorescent".to_string()),
        "shallow get_page_inks must also resolve indirect names arrays; got {:?}",
        shallow
    );
}
