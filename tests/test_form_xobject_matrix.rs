//! Test that Form XObject /Matrix is applied during text extraction.
//!
//! Per ISO 32000-1 §8.10.1, when a Form XObject is invoked via `Do`, its
//! `/Matrix` entry is concatenated with the current CTM. This transforms
//! coordinates from the XObject's internal space to page space.
//!
//! Before the fix, text extraction ignored `/Matrix`, causing extracted
//! coordinates to be in the XObject's internal space rather than page space.

use pdf_oxide::document::PdfDocument;

/// Build a minimal PDF where a Form XObject has a `/Matrix` that scales by 0.5.
///
/// The Form XObject places text at (200, 400) in its internal coordinate space.
/// With /Matrix [0.5 0 0 0.5 0 0], the text should appear at (100, 200) in
/// page space.
fn build_form_xobject_with_matrix_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    // Header
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Object 1: Catalog
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    // Object 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 6 0 R >> /XObject << /Form1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    // Object 4: Page content stream — just invokes the Form XObject
    let content = b"/Form1 Do";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 5: Form XObject with /Matrix [0.5 0 0 0.5 0 0] (scale 50%)
    // Text is placed at (200, 400) in XObject space → (100, 200) in page space
    let form_stream = b"BT /F1 24 Tf 200 400 Td (Hello) Tj ET";
    offsets.push(pdf.len());
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 1224 1584]\n\
         /Matrix [0.5 0 0 0.5 0 0]\n\
         /Resources << /Font << /F1 6 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_obj.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 6: Font
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    // Cross-reference table
    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    // Trailer
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    pdf
}

/// Build a PDF where a Form XObject has /Matrix with translation.
///
/// The Form XObject places text at (0, 0) in its internal space.
/// With /Matrix [1 0 0 1 100 200], the text should appear at (100, 200) in page space.
fn build_form_xobject_with_translation_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 6 0 R >> /XObject << /Form1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    let content = b"/Form1 Do";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Form XObject with /Matrix [1 0 0 1 100 200] (translation only)
    let form_stream = b"BT /F1 12 Tf 50 60 Td (World) Tj ET";
    offsets.push(pdf.len());
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
         /Matrix [1 0 0 1 100 200]\n\
         /Resources << /Font << /F1 6 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_obj.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    pdf
}

/// Build a PDF where Form XObject has no /Matrix (should default to identity).
fn build_form_xobject_without_matrix_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 6 0 R >> /XObject << /Form1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    let content = b"/Form1 Do";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Form XObject WITHOUT /Matrix — should use identity
    let form_stream = b"BT /F1 12 Tf 100 200 Td (NoMatrix) Tj ET";
    offsets.push(pdf.len());
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
         /Resources << /Font << /F1 6 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_obj.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    pdf
}

#[test]
fn test_form_xobject_matrix_scaling_applied_to_spans() {
    let _ = env_logger::builder().is_test(true).try_init();
    let pdf_bytes = build_form_xobject_with_matrix_pdf();

    let mut doc = PdfDocument::from_bytes(pdf_bytes).expect("Failed to parse test PDF");

    let spans = doc.extract_spans(0).expect("Failed to extract spans");
    assert!(!spans.is_empty(), "Should extract at least one span");

    let hello_span = spans.iter().find(|s| s.text.contains("Hello"));
    assert!(
        hello_span.is_some(),
        "Should find 'Hello' span, got: {:?}",
        spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );

    let span = hello_span.unwrap();

    // The text is at (200, 400) in XObject space with /Matrix [0.5 0 0 0.5 0 0].
    // After matrix application: x = 200*0.5 = 100, y = 400*0.5 = 200.
    // Allow some tolerance for font metrics.
    assert!(
        span.bbox.x < 150.0,
        "X coordinate should be ~100 (scaled from 200), got {}",
        span.bbox.x
    );
    assert!(span.bbox.x > 50.0, "X coordinate should be ~100, got {}", span.bbox.x);

    // Font size should also be scaled: 24 * 0.5 = 12
    assert!(
        span.font_size < 16.0,
        "Font size should be ~12 (24 * 0.5), got {}",
        span.font_size
    );
    assert!(span.font_size > 8.0, "Font size should be ~12, got {}", span.font_size);
}

#[test]
fn test_form_xobject_matrix_translation_applied_to_spans() {
    let _ = env_logger::builder().is_test(true).try_init();
    let pdf_bytes = build_form_xobject_with_translation_pdf();

    let mut doc = PdfDocument::from_bytes(pdf_bytes).expect("Failed to parse test PDF");

    let spans = doc.extract_spans(0).expect("Failed to extract spans");
    assert!(!spans.is_empty(), "Should extract at least one span");

    let world_span = spans.iter().find(|s| s.text.contains("World"));
    assert!(
        world_span.is_some(),
        "Should find 'World' span, got: {:?}",
        spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );

    let span = world_span.unwrap();

    // Text at (50, 60) in XObject space + /Matrix translation (100, 200)
    // Expected page-space x = 50 + 100 = 150
    assert!(
        span.bbox.x > 120.0 && span.bbox.x < 180.0,
        "X should be ~150 (50 + 100 translation), got {}",
        span.bbox.x
    );
}

#[test]
fn test_form_xobject_without_matrix_uses_identity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let pdf_bytes = build_form_xobject_without_matrix_pdf();

    let mut doc = PdfDocument::from_bytes(pdf_bytes).expect("Failed to parse test PDF");

    let spans = doc.extract_spans(0).expect("Failed to extract spans");
    assert!(!spans.is_empty(), "Should extract at least one span");

    let span = spans.iter().find(|s| s.text.contains("NoMatrix"));
    assert!(
        span.is_some(),
        "Should find 'NoMatrix' span, got: {:?}",
        spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );

    let span = span.unwrap();

    // Text at (100, 200) with identity matrix — coordinates should be unchanged
    assert!(
        span.bbox.x > 80.0 && span.bbox.x < 120.0,
        "X should be ~100 (identity matrix), got {}",
        span.bbox.x
    );
    assert!(
        span.font_size > 10.0 && span.font_size < 14.0,
        "Font size should be ~12 (identity matrix), got {}",
        span.font_size
    );
}

#[test]
fn test_form_xobject_matrix_does_not_leak_to_parent() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Build a PDF with page-level text AFTER the Form XObject invocation
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 6 0 R >> /XObject << /Form1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    // Page content: invoke Form XObject (with 0.5 scale), then draw text at (300, 500)
    let content = b"/Form1 Do\nBT /F1 20 Tf 300 500 Td (After) Tj ET";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Form XObject with 0.5× scaling
    let form_stream = b"BT /F1 24 Tf 200 400 Td (Inside) Tj ET";
    offsets.push(pdf.len());
    let form_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 1224 1584]\n\
         /Matrix [0.5 0 0 0.5 0 0]\n\
         /Resources << /Font << /F1 6 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form_stream.len()
    );
    pdf.extend_from_slice(form_obj.as_bytes());
    pdf.extend_from_slice(form_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    let mut doc = PdfDocument::from_bytes(pdf).expect("Failed to parse test PDF");

    let spans = doc.extract_spans(0).expect("Failed to extract spans");

    let after_span = spans.iter().find(|s| s.text.contains("After"));
    assert!(
        after_span.is_some(),
        "Should find 'After' span, got: {:?}",
        spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );

    let span = after_span.unwrap();

    // The page-level text at (300, 500) should NOT be affected by the Form XObject's matrix.
    // If state leaks, x would be 300*0.5=150 instead of 300.
    assert!(
        span.bbox.x > 250.0,
        "Page-level text X should be ~300 (not scaled by XObject matrix), got {}",
        span.bbox.x
    );
    assert!(
        span.font_size > 16.0,
        "Page-level font size should be ~20 (not scaled), got {}",
        span.font_size
    );
}

#[test]
fn test_form_xobject_nested_matrix_composition() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Build a PDF with a nested Form XObject chain:
    // Page → Form1 (translate +100,+50) → Form2 (scale 0.5×)
    // Text at (200, 300) in Form2's space should end up at:
    //   Form2 scale: (200*0.5, 300*0.5) = (100, 150)
    //   Form1 translate: (100+100, 150+50) = (200, 200)
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Object 1: Catalog
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    // Object 3: Page
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 7 0 R >> /XObject << /Form1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    // Object 4: Page content — invokes outer Form XObject
    let content = b"/Form1 Do";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 5: Outer Form XObject — translate (100, 50), invokes inner Form2
    let form1_stream = b"/Form2 Do";
    offsets.push(pdf.len());
    let form1_obj = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792]\n\
         /Matrix [1 0 0 1 100 50]\n\
         /Resources << /Font << /F1 7 0 R >> /XObject << /Form2 6 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form1_stream.len()
    );
    pdf.extend_from_slice(form1_obj.as_bytes());
    pdf.extend_from_slice(form1_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 6: Inner Form XObject — scale 0.5×, contains text at (200, 300)
    let form2_stream = b"BT /F1 24 Tf 200 300 Td (Nested) Tj ET";
    offsets.push(pdf.len());
    let form2_obj = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 1224 1584]\n\
         /Matrix [0.5 0 0 0.5 0 0]\n\
         /Resources << /Font << /F1 7 0 R >> >>\n\
         /Length {} >>\nstream\n",
        form2_stream.len()
    );
    pdf.extend_from_slice(form2_obj.as_bytes());
    pdf.extend_from_slice(form2_stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 7: Font
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    let mut doc = PdfDocument::from_bytes(pdf).expect("Failed to parse test PDF");
    let spans = doc.extract_spans(0).expect("Failed to extract spans");

    let nested_span = spans.iter().find(|s| s.text.contains("Nested"));
    assert!(
        nested_span.is_some(),
        "Should find 'Nested' span, got: {:?}",
        spans.iter().map(|s| &s.text).collect::<Vec<_>>()
    );

    let span = nested_span.unwrap();

    // Expected: x = 200*0.5 + 100 = 200, y = 300*0.5 + 50 = 200
    assert!(
        span.bbox.x > 160.0 && span.bbox.x < 240.0,
        "X should be ~200 (nested transform composition), got {}",
        span.bbox.x
    );

    // Font size: 24 * 0.5 = 12 (only inner scale applies to font)
    assert!(
        span.font_size > 8.0 && span.font_size < 16.0,
        "Font size should be ~12 (24 * 0.5 from inner scale), got {}",
        span.font_size
    );
}

#[test]
fn test_rotated_text_does_not_produce_extreme_coordinates() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Build a PDF with 90°-rotated text (simulating chart Y-axis labels).
    // The text matrix has d≈0 and b=1, which previously caused a
    // divide-by-zero in the text matrix advance calculation.
    let mut pdf = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
          /Contents 4 0 R\n\
          /Resources << /Font << /F1 5 0 R >> >>\n\
          >>\nendobj\n",
    );

    // Content: 90° rotated text using Tm with [0 1 -1 0 x y] (rotation matrix)
    // Multiple text show operations to exercise the advance calculation
    let content = b"BT\n\
        /F1 12 Tf\n\
        0 1 -1 0 50 400 Tm\n\
        (Rotated Label One) Tj\n\
        0 1 -1 0 80 400 Tm\n\
        (Rotated Label Two) Tj\n\
        ET";
    offsets.push(pdf.len());
    let header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    pdf.extend_from_slice(header.as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = pdf.len();
    let total_objects = offsets.len() + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {}\n", total_objects).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects, xref_offset
        )
        .as_bytes(),
    );

    let mut doc = PdfDocument::from_bytes(pdf).expect("Failed to parse test PDF");
    let spans = doc.extract_spans(0).expect("Failed to extract spans");

    assert!(!spans.is_empty(), "Should extract rotated text spans");

    // All spans must have coordinates within the page bounds.
    // Before the fix, the divide-by-zero in text matrix advance produced
    // coordinates in the hundreds of millions.
    for span in &spans {
        assert!(
            span.bbox.x.abs() < 1000.0,
            "Rotated text X should be within page bounds, got {} for {:?}",
            span.bbox.x,
            span.text
        );
        assert!(
            span.bbox.y.abs() < 1000.0,
            "Rotated text Y should be within page bounds, got {} for {:?}",
            span.bbox.y,
            span.text
        );
        assert!(
            span.font_size < 100.0,
            "Rotated text font size should be reasonable, got {} for {:?}",
            span.font_size,
            span.text
        );
    }
}
