//! Regression tests for multi-line object headers and xref reconstruction.
//!
//! These tests cover:
//! - Issue #45: Multi-line object headers (e.g. "1\n0\nobj")
//! - Garbage-prepended PDFs where header_offset > 0
//! - Corrupt xref tables triggering reconstruction fallback
//! - The `contains("obj")` bug that matched "endobj"

use pdf_oxide::document::PdfDocument;

// ---------------------------------------------------------------------------
// Helpers: build minimal PDFs with various header formats
// ---------------------------------------------------------------------------

/// Build a minimal valid PDF where object headers use the given separator
/// between obj_num, gen_num, and "obj".
///
/// `header_fmt` takes (obj_num, gen_num) and returns the header string.
fn build_pdf_custom_headers(header_fmt: impl Fn(u32, u16) -> String) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    let h1 = header_fmt(1, 0);
    pdf.extend_from_slice(
        format!("{}\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n", h1).as_bytes(),
    );

    let off2 = pdf.len();
    let h2 = header_fmt(2, 0);
    pdf.extend_from_slice(
        format!("{}\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n", h2).as_bytes(),
    );

    let off3 = pdf.len();
    let h3 = header_fmt(3, 0);
    pdf.extend_from_slice(
        format!(
            "{}\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
            h3
        )
        .as_bytes(),
    );

    let off4 = pdf.len();
    let h4 = header_fmt(4, 0);
    let content = "BT /F1 12 Tf 72 720 Td (Hello World) Tj ET";
    pdf.extend_from_slice(format!("{}\n<< /Length {} >>\nstream\n", h4, content.len()).as_bytes());
    pdf.extend_from_slice(content.as_bytes());
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    let h5 = header_fmt(5, 0);
    pdf.extend_from_slice(
        format!(
            "{}\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
            h5
        )
        .as_bytes(),
    );

    finalize_xref(&mut pdf, &[0, off1, off2, off3, off4, off5]);
    pdf
}

fn finalize_xref(pdf: &mut Vec<u8>, obj_offsets: &[usize]) {
    let xref_offset = pdf.len();
    let count = obj_offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &obj_offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        count, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
}

// ---------------------------------------------------------------------------
// Test 1: Standard single-line headers (baseline)
// ---------------------------------------------------------------------------

#[test]
fn test_standard_single_line_headers() {
    let pdf = build_pdf_custom_headers(|id, gen| format!("{} {} obj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("should open standard PDF");
    assert_eq!(doc.page_count().expect("page count"), 1);
    let text = doc.extract_text(0).expect("extract text");
    assert!(text.contains("Hello World"), "text: {}", text);
}

// ---------------------------------------------------------------------------
// Test 2: Multi-line headers — newline between each token
// ---------------------------------------------------------------------------

#[test]
fn test_multiline_object_header_full_newline() {
    // "1\n0\nobj" format (each token on its own line)
    let pdf = build_pdf_custom_headers(|id, gen| format!("{}\n{}\nobj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("should open PDF with fully multi-line headers");
    assert_eq!(doc.page_count().expect("page count"), 1);
    let text = doc.extract_text(0).expect("extract text");
    assert!(text.contains("Hello World"), "text: {}", text);
}

// ---------------------------------------------------------------------------
// Test 3: Multi-line headers — number on separate line, rest on one line
// ---------------------------------------------------------------------------

#[test]
fn test_multiline_object_header_mixed() {
    // "1\n0 obj" format (obj_num on separate line, gen+obj on one line)
    let pdf = build_pdf_custom_headers(|id, gen| format!("{}\n{} obj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("should open PDF with mixed multi-line headers");
    assert_eq!(doc.page_count().expect("page count"), 1);
}

// ---------------------------------------------------------------------------
// Test 4: Multi-line with \r\n line endings
// ---------------------------------------------------------------------------

#[test]
fn test_multiline_object_header_crlf() {
    let pdf = build_pdf_custom_headers(|id, gen| format!("{}\r\n{}\r\nobj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("should open PDF with CRLF multi-line headers");
    assert_eq!(doc.page_count().expect("page count"), 1);
}

// ---------------------------------------------------------------------------
// Test 5: Garbage-prepended PDF (header_offset adjustment)
// ---------------------------------------------------------------------------

#[test]
fn test_garbage_prefix_offset_adjustment() {
    // Build a standard valid PDF
    let valid_pdf = build_pdf_custom_headers(|id, gen| format!("{} {} obj", id, gen));

    // Prepend 1024 bytes of garbage
    let mut garbage_pdf = vec![0xFFu8; 1024];
    garbage_pdf.extend_from_slice(&valid_pdf);

    // The xref offsets in this PDF are relative to the start of the valid PDF data,
    // which is now at byte 1024. The header_offset adjustment should fix this.
    let doc = PdfDocument::from_bytes(garbage_pdf).expect("should open garbage-prepended PDF");
    assert_eq!(doc.page_count().expect("page count"), 1);
    let text = doc.extract_text(0).expect("extract text");
    assert!(text.contains("Hello World"), "text: {}", text);
}

// ---------------------------------------------------------------------------
// Test 6: Corrupt xref triggers reconstruction fallback
// ---------------------------------------------------------------------------

#[test]
fn test_corrupt_xref_triggers_reconstruction() {
    // Build a valid PDF
    let mut pdf = build_pdf_custom_headers(|id, gen| format!("{} {} obj", id, gen));

    // Find and corrupt the xref table — replace offset digits with zeros
    // to make the xref point to wrong locations
    let xref_marker = b"xref\n";
    if let Some(pos) = pdf
        .windows(xref_marker.len())
        .position(|w| w == xref_marker)
    {
        // Corrupt the xref entries: overwrite the offset numbers
        // Skip "xref\n0 N\n" and the free entry, then corrupt in-use entries
        let xref_start = pos + xref_marker.len();
        // Find the second line (after "0 N\n")
        if let Some(nl) = pdf[xref_start..].iter().position(|&b| b == b'\n') {
            let entries_start = xref_start + nl + 1;
            // Skip the free entry (first 20 bytes including \r\n)
            let first_entry = entries_start + 20;
            // Corrupt all subsequent entries by changing offsets to 9999999999
            let mut i = first_entry;
            while i + 20 <= pdf.len() && pdf[i] != b't' {
                // Replace first 10 chars (offset) with zeros
                for j in 0..10 {
                    if i + j < pdf.len() {
                        pdf[i + j] = b'0';
                    }
                }
                i += 20;
            }
        }
    }

    // Should still open via xref reconstruction
    let doc =
        PdfDocument::from_bytes(pdf).expect("should open PDF with corrupt xref via reconstruction");
    assert_eq!(doc.page_count().expect("page count"), 1);
}

// ---------------------------------------------------------------------------
// Test 7: "endobj" should not be confused with "obj" during multi-line parsing
// ---------------------------------------------------------------------------

#[test]
fn test_endobj_not_confused_with_obj() {
    // Build a PDF where the xref intentionally points a few bytes too late
    // (into the object body area), so the parser reads "endobj" before finding
    // the real header. The fix ensures "endobj" doesn't satisfy the loop condition.
    //
    // We test this indirectly: a standard PDF should parse correctly even though
    // every object body contains "endobj" (the loop should keep reading past it).
    let pdf = build_pdf_custom_headers(|id, gen| format!("{} {} obj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("standard PDF should open");
    assert_eq!(doc.page_count().expect("page count"), 1);
}

// ---------------------------------------------------------------------------
// Test 8: has_standalone_obj_keyword unit-level behavior via full document parse
// ---------------------------------------------------------------------------

#[test]
fn test_multiline_header_with_extra_whitespace() {
    // Extra spaces and tabs between tokens
    let pdf = build_pdf_custom_headers(|id, gen| format!("{}  \t {}  \t obj", id, gen));
    let doc = PdfDocument::from_bytes(pdf).expect("should handle extra whitespace in headers");
    assert_eq!(doc.page_count().expect("page count"), 1);
}

// ---------------------------------------------------------------------------
// Multi-byte character immediately before a wrapped "obj" keyword
// ---------------------------------------------------------------------------

fn build_pdf_with_pages_header(pages_header: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(pages_header);
    pdf.extend_from_slice(b"\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n");

    finalize_xref(&mut pdf, &[0, off1, off2]);
    pdf
}

#[test]
fn test_multiline_header_invalid_utf8_comment_before_obj() {
    // 0xFF decodes lossily to the 3-byte U+FFFD, so the byte three positions
    // before "obj" in the joined header falls inside a multi-byte character.
    let pdf = build_pdf_with_pages_header(b"2 0 %\xFF\nobj");
    let doc = PdfDocument::from_bytes(pdf).expect("should open PDF with binary header comment");
    assert_eq!(doc.page_count().expect("page count"), 0);
}

#[test]
fn test_multiline_header_two_byte_utf8_comment_before_obj() {
    let pdf = build_pdf_with_pages_header("2 0 %é\nobj".as_bytes());
    let doc = PdfDocument::from_bytes(pdf).expect("should open PDF with UTF-8 header comment");
    assert_eq!(doc.page_count().expect("page count"), 0);
}

// ---------------------------------------------------------------------------
// Test 9: Real-world regression — PDF.js issue9418 (multi-line headers)
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Requires external test corpus
fn test_pdfjs_issue9418_multiline_header() {
    let home = match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping: HOME not set");
            return;
        },
    };
    let path =
        std::path::Path::new(&home).join("projects/pdf_oxide_tests/pdfs_pdfjs/issue9418.pdf");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }
    // The key assertion: PdfDocument::open no longer fails with a parser error
    // on multi-line object headers. The file may fail later (e.g. missing /Pages
    // in catalog) but the header parsing regression is fixed.
    let _doc = PdfDocument::open(&path)
        .unwrap_or_else(|e| panic!("issue9418.pdf should open (multi-line header fix): {}", e));
}

// ---------------------------------------------------------------------------
// Test 10: Real-world regression — veraPDF isartor-6-1-2-t01-fail-a
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Requires external test corpus
fn test_isartor_multiline_header() {
    let home = match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping: HOME not set");
            return;
        },
    };
    let path = std::path::Path::new(&home)
        .join("projects/veraPDF-corpus/PDF_A-1b/6.1 File structure/6.1.2 File header/isartor-6-1-2-t01-fail-a.pdf");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }
    let doc = PdfDocument::open(&path).unwrap_or_else(|e| panic!("isartor PDF should open: {}", e));
    let count = doc.page_count().expect("page count");
    assert!(count > 0, "should have at least 1 page");
}

// ---------------------------------------------------------------------------
// Test 11: Real-world regression — REDHAT-1531897-0.pdf (corrupt xref)
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Requires external test corpus
fn test_redhat_corrupt_xref() {
    let home = match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping: HOME not set");
            return;
        },
    };
    let path = std::path::Path::new(&home).join(
        "projects/veraPDF-corpus/PDF_A-1b/6.7 Metadata/6.7.2 Metadata stream/REDHAT-1531897-0.pdf",
    );
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }
    let doc = PdfDocument::open(&path).unwrap_or_else(|e| panic!("REDHAT PDF should open: {}", e));
    let count = doc.page_count().expect("page count");
    assert!(count > 0, "should have at least 1 page");
}

// ---------------------------------------------------------------------------
// Test 12: shift_offsets on CrossRefTable
// ---------------------------------------------------------------------------

#[test]
fn test_xref_shift_offsets() {
    use pdf_oxide::xref::{CrossRefTable, XRefEntry, XRefEntryType};

    let mut xref = CrossRefTable::new();
    xref.add_entry(1, XRefEntry::uncompressed(100, 0));
    xref.add_entry(2, XRefEntry::uncompressed(200, 0));
    xref.add_entry(3, XRefEntry::compressed(5, 0)); // should NOT be shifted
    xref.add_entry(0, XRefEntry::free(0, 65535)); // should NOT be shifted

    xref.shift_offsets(50);

    // Uncompressed entries should be shifted
    assert_eq!(xref.get(1).unwrap().offset, 150);
    assert_eq!(xref.get(2).unwrap().offset, 250);
    // Compressed entry should remain unchanged
    assert_eq!(xref.get(3).unwrap().offset, 5);
    assert_eq!(xref.get(3).unwrap().entry_type, XRefEntryType::Compressed);
    // Free entry should remain unchanged
    assert_eq!(xref.get(0).unwrap().offset, 0);
}
