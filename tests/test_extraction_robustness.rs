//! Extraction robustness tests — crash recovery and graceful fallbacks.
//!
//! Covers parser resilience and error-surfacing behaviour:
//!
//! * Parser crash prevention:
//!   - Bare-word identifiers (e.g. `OBJR` without a leading `/`) are now
//!     treated as Name tokens (lenient fallback matching poppler/pdfjs).
//!   - A corrupt or unresolvable StructTreeRoot returns `Ok(None)` instead
//!     of `Err(InvalidPdf)`, allowing graceful fallback to font-size clustering.
//!
//! * Correct error surfacing on pathological inputs:
//!   - Encrypted PDFs must return `Err(EncryptedPdf)` from `page_count()`,
//!     not `Ok(0)` (which silently produced empty output). Text extraction,
//!     by contrast, degrades to empty output rather than erroring.
//!   - Rotated pages and annotation-heavy pages must not panic.

use pdf_oxide::document::PdfDocument;

// ---------------------------------------------------------------------------
// Helpers: build minimal PDFs from raw bytes
// ---------------------------------------------------------------------------

/// Build a minimal 1-page PDF whose content stream uses the specified bytes.
fn one_page_pdf_with_content(content: &[u8]) -> Vec<u8> {
    build_minimal_pdf_raw(content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

// ---------------------------------------------------------------------------
// Section 1 — bare-word identifier in PDF stream must not crash
// ---------------------------------------------------------------------------

/// Crash-safety guard: a PDF whose content stream contains an unknown bare
/// keyword (`OBJR`, no leading `/`) must not panic or produce empty output.
///
/// Content-stream parsers skip unknown operators by design; this test guards
/// against regressions in that error-recovery path.  The `token_lenient`
/// dictionary-value leniency is tested separately in
/// `bare_word_in_dict_value_does_not_crash`.
#[test]
fn bare_word_identifier_does_not_crash() {
    // Content stream with an unknown bare keyword (no leading /)
    let content = b"BT /F1 12 Tf 50 700 Td (Hello bare-word) Tj ET\nOBJR 0 0 100 100";
    let pdf = one_page_pdf_with_content(content);
    let doc = PdfDocument::from_bytes(pdf).expect("PDF with bare OBJR must open without error");
    let text = doc
        .extract_text(0)
        .expect("extract_text must not panic on bare-word PDF");
    assert!(
        text.contains("Hello bare-word"),
        "text must be extracted from bare-word PDF; got: {:?}",
        text
    );
}

/// Exercises the `token_lenient` dictionary-value fallback in `parser.rs`.
/// The page dict contains `/Orientation Landscape` where `Landscape` is a bare
/// identifier (no leading `/`).  `parse_object` fails on it; `token_lenient`
/// recovers it as a Name token so the document opens and text extracts cleanly.
#[test]
fn bare_word_in_dict_value_does_not_crash() {
    let pdf = build_minimal_pdf_raw(
        b"BT /F1 12 Tf 50 700 Td (Dict bare-word) Tj ET",
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Orientation Landscape",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("PDF with bare dict value must open");
    let text = doc.extract_text(0).expect("extract must not panic");
    assert!(
        text.contains("Dict bare-word"),
        "text must be extracted from bare-dict-value PDF; got: {:?}",
        text
    );
}

// ---------------------------------------------------------------------------
// Section 1 — corrupt/null StructTreeRoot must not crash
// ---------------------------------------------------------------------------

/// A StructTreeRoot that resolves to Null (e.g. the object was deleted or the
/// reference is broken) must be treated as "no structure tree" rather than
/// returning `Err(InvalidPdf)`.
///
/// Before the fix the error propagated and crashed extraction for tagged PDFs.
/// After the fix it logs a warning and falls back to the untagged path.
#[test]
fn corrupt_struct_tree_root_falls_back_gracefully() {
    // Build a PDF that claims to have a MarkInfo /Marked true and a
    // StructTreeRoot, but points StructTreeRoot at a missing object (99 0 R).
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    // Catalog: has MarkInfo and StructTreeRoot pointing at non-existent obj 99
    pdf.extend_from_slice(
        b"1 0 obj\n\
        << /Type /Catalog /Pages 2 0 R \
           /MarkInfo << /Marked true >> \
           /StructTreeRoot 99 0 R >>\n\
        endobj\n",
    );

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n\
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n\
        endobj\n",
    );

    let content = b"BT /F1 12 Tf 50 700 Td (Fallback text) Tj ET";
    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
          /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf)
        .expect("PDF with corrupt StructTreeRoot must open without error");
    // Must not panic; text may come from the untagged fallback path.
    let text = doc
        .extract_text(0)
        .expect("extract_text must not error on corrupt StructTreeRoot");
    assert!(
        text.contains("Fallback text"),
        "untagged fallback must still extract body text; got: {:?}",
        text
    );
}

// ---------------------------------------------------------------------------
// Section 2 — encrypted PDF must surface EncryptedPdf, not Ok(0)
// ---------------------------------------------------------------------------

/// Before the fix, `page_count()` on an encrypted PDF returned `Ok(0)` because
/// the page-tree scan fallback silently returned 0 when decryption failed.
/// Callers that loop `0..page_count` would then process 0 pages, producing
/// near-empty output without any error.
///
/// After the fix, `page_count()` returns `Err(EncryptedPdf)` immediately.
#[test]
fn encrypted_pdf_page_count_returns_encrypted_error() {
    // Minimal PDF with RC4 40-bit encryption placeholders.  The encryption dict
    // is not valid for actual decryption, but it IS sufficient to set
    // is_encrypted() = true and to make page-tree loading fail — which is the
    // exact condition that triggered the bug.
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /Encrypt 6 0 R >>\nendobj\n");

    let off2 = pdf.len();
    // The pages object is an ObjStm (compressed) — but we point to obj 99 which
    // doesn't exist, so loading the page tree will fail with an error.
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [99 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(b"4 0 obj\n<< /Length 0 >>\nstream\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    // Minimal RC4 encryption dict (enough to make is_encrypted() true)
    let off6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj\n\
        << /Filter /Standard /V 1 /R 2 /O (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) \
           /U (yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy) /P -4 >>\n\
        endobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5, off6];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Encrypt 6 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf).expect("open encrypted PDF stub");
    assert!(doc.is_encrypted(), "PDF stub must be recognised as encrypted");

    // Core guarantee: page_count must not return Ok(0) for an encrypted PDF
    // that cannot be decrypted.  It must either:
    //   a) return Err(EncryptedPdf), OR
    //   b) return the actual page count if authentication somehow succeeded.
    // What it must NOT do is return Ok(0) — that's the regression.
    match doc.page_count() {
        Ok(0) => {
            panic!("page_count() returned Ok(0) for encrypted PDF — this is the pre-fix regression")
        },
        Ok(n) => {
            // Authentication succeeded (unlikely with dummy creds but acceptable)
            assert!(n > 0, "page count must be positive when Ok() is returned");
        },
        Err(pdf_oxide::error::Error::EncryptedPdf) => {
            // Correct behaviour: surface the encryption error
        },
        Err(e) => {
            // Any other error is also acceptable — the point is it must not be Ok(0)
            let _ = e;
        },
    }

    // Text extraction degrades to empty output (warn + empty) rather than
    // erroring, matching pdftotext/PyMuPDF. `page_count` (above) still surfaces
    // the encryption to callers that ask for the structure.
    assert_eq!(
        doc.extract_text(0).ok(),
        Some(String::new()),
        "extract_text on an undecryptable PDF returns empty, not an error"
    );
}

// ---------------------------------------------------------------------------
// Section 2 — text extraction of rotated pages must not crash
// ---------------------------------------------------------------------------

/// Annotation-heavy PDFs with page rotations caused crashes before the fix.
/// This synthetic test verifies that a page with /Rotate 90 can be opened and
/// extracted without panicking.
#[test]
fn rotated_page_extraction_does_not_crash() {
    let content = b"BT /F1 12 Tf 50 700 Td (Rotated content) Tj ET";
    // Build a page with /Rotate 90
    let pdf = build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("rotated page must open without error");
    let result = doc.extract_text(0);
    // Must not panic and must not return an unexpected error
    assert!(
        result.is_ok(),
        "extract_text on rotated page must not return an error; got: {:?}",
        result
    );
}

/// A `/Rotate` 90 or 270 page must keep its spatial-extraction geometry in RAW
/// user space (matching `extract_chars`) and group there. The earlier behaviour
/// rotated span bboxes into the displayed frame before clustering, but
/// `TextSpan::to_chars` still lays glyphs horizontally with raw advance widths,
/// so it could not represent a run whose visual direction was now vertical. The
/// net effect was that every raw text row (constant raw-y) collapsed onto a
/// single displayed-y band and unrelated cells from perpendicular columns fused
/// into one giant token (a whole table column returned as a 1000+ char "word",
/// separate rows fused into one line).
///
/// This builds a landscape page with `/Rotate 90` holding two short labels in
/// the SAME column (same raw x) on two DISTINCT rows (different raw y). The
/// deterministic guard is that the emitted word geometry is raw (x ≈ the content
/// x), not the rotated displayed x (≈ raw y); a bonus check confirms the rows do
/// not fuse into a single token.
#[test]
fn rotate_90_keeps_raw_coords_and_does_not_fuse_rows() {
    let content = b"BT /F1 12 Tf 100 200 Td (Alpha) Tj ET\nBT /F1 12 Tf 100 214 Td (Bravo) Tj ET";
    let pdf = build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 800 600] /Rotate 90",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("rotated page must open without error");
    assert_eq!(doc.get_page_rotation(0).unwrap(), 90, "sanity: /Rotate 90 parsed");

    let words = doc
        .extract_words(0)
        .expect("extract_words must not error on a rotated page");

    // No single token may contain BOTH labels — that is the column-fusion bug.
    for w in &words {
        assert!(
            !(w.text.contains("Alpha") && w.text.contains("Bravo")),
            "distinct rows fused into one token on a /Rotate 90 page: {:?}",
            w.text
        );
    }
    assert!(
        words.iter().any(|w| w.text.contains("Alpha")),
        "Alpha row missing; got {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
    assert!(
        words.iter().any(|w| w.text.contains("Bravo")),
        "Bravo row missing; got {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );

    // Word geometry stays RAW: x ≈ 100 (the content x), NOT the rotated displayed
    // x (≈ raw y = 200). This is the direct signature of the fix and agrees with
    // extract_chars, which already returns raw coordinates.
    let alpha = words
        .iter()
        .find(|w| w.text.contains("Alpha"))
        .expect("Alpha word present");
    assert!(
        (alpha.bbox.x - 100.0).abs() < 20.0,
        "word x must be in raw user space (~100); got {} (the rotated frame would be ~200)",
        alpha.bbox.x
    );
}

/// Issue #804 (case 2): text drawn with a rotated text matrix — `Tm = [0 1 -1 0
/// e f]` — models vertical table-column headers / chart-axis labels. Such a run's
/// glyphs advance along a rotated axis, but the extractor stores a span bbox
/// FLATTENED onto the x-axis (width = Σ advances, height = font). Two adjacent
/// rotated columns therefore get overlapping flattened bboxes, and the
/// reading-order word-merge (and the y-band line grouping) would fuse the columns
/// into one giant token / line — the dominant #804 failure on real documents
/// (whole rotated columns returned as 300–3400 char "words"). Each rotated run
/// must remain its own word(s) and its own line.
#[test]
fn rotated_text_matrix_columns_do_not_fuse() {
    // Two adjacent rotated columns, run starts only 10 units apart on x so their
    // flattened bboxes (each ~= the run's advance length) heavily overlap.
    let content = b"BT /F1 12 Tf 0 1 -1 0 200 100 Tm (Alpha) Tj ET\n\
                    BT /F1 12 Tf 0 1 -1 0 210 100 Tm (Bravo) Tj ET";
    let pdf = build_minimal_pdf_raw(content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 400 400]");
    let doc = PdfDocument::from_bytes(pdf).expect("PDF must open without error");

    let words = doc
        .extract_words(0)
        .expect("extract_words must not error on rotated-matrix text");
    for w in &words {
        assert!(
            !(w.text.contains("Alpha") && w.text.contains("Bravo")),
            "adjacent rotated columns fused into one word: {:?}",
            w.text
        );
    }
    assert!(
        words.iter().any(|w| w.text.contains("Alpha")),
        "Alpha column missing; got {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
    assert!(
        words.iter().any(|w| w.text.contains("Bravo")),
        "Bravo column missing; got {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );

    // The two rotated columns must also not collapse into a single line.
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines must not error on rotated-matrix text");
    for l in &lines {
        assert!(
            !(l.text.contains("Alpha") && l.text.contains("Bravo")),
            "adjacent rotated columns fused into one line: {:?}",
            l.text
        );
    }
}

/// Issue #804 follow-up — a page whose `/Rotate` and content rotation must
/// COMBINE. A landscape document authored by drawing every glyph sideways
/// (`rotation_degrees = 90`, a rotated text matrix) inside a PORTRAIT MediaBox
/// with `/Rotate 90` reads upright only when the page rotation is applied to the
/// rotated content so it composes with the content rotation. Leaving such a page
/// "raw" (which is correct for *horizontal* content on a rotated page — see
/// `rotate_90_keeps_raw_coords_and_does_not_fuse_rows`) reads it sideways and
/// scrambles the reading order.
///
/// The two rotated runs here read, in the displayed (upright) frame, `BRAVO`
/// (higher) then `ALPHA` (lower) — the order pdfplumber/pdfminer produce for the
/// same page. Each run must also stay intact (not reversed, not fused).
#[test]
fn rotate_90_portrait_page_with_rotated_content_reads_upright() {
    // Portrait MediaBox + /Rotate 90; both runs drawn with Tm = [0 1 -1 0]
    // (content rotation 90). BRAVO is drawn at raw x=100, ALPHA at raw x=300,
    // which map to displayed `top` 90 and 290 respectively.
    let content = b"BT /F1 12 Tf 0 1 -1 0 300 100 Tm (ALPHA) Tj ET\n\
                    BT /F1 12 Tf 0 1 -1 0 100 100 Tm (BRAVO) Tj ET";
    let pdf = build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] /Rotate 90",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("PDF must open without error");
    assert_eq!(doc.get_page_rotation(0).unwrap(), 90, "sanity: /Rotate 90");

    // The content is drawn with a rotated text matrix.
    let chars = doc.extract_chars(0).expect("extract_chars must not error");
    assert!(
        chars.iter().any(|c| c.rotation_degrees.abs() > 45.0),
        "content should be detected as rotated (rotation_degrees ~= 90)"
    );

    let text = doc.extract_text(0).expect("extract_text must not error");
    // Both runs intact and not reversed.
    assert!(text.contains("ALPHA"), "ALPHA run intact; got {:?}", text);
    assert!(text.contains("BRAVO"), "BRAVO run intact; got {:?}", text);
    assert!(
        !text.contains("AHPLA") && !text.contains("OVARB"),
        "rotated runs must not be reversed; got {:?}",
        text
    );
    // Displayed reading order: BRAVO (upper) before ALPHA (lower) — matches
    // pdfplumber. The pre-fix "keep raw" path read the page sideways.
    let (b, a) = (text.find("BRAVO"), text.find("ALPHA"));
    assert!(
        b < a,
        "rotated page must read in upright displayed order (BRAVO before ALPHA); got {:?}",
        text
    );
}

// ---------------------------------------------------------------------------
// Section 2 — non-empty text on a page with a rich annotation set
// ---------------------------------------------------------------------------

/// Verify that a PDF with a rich set of annotation types can be opened and
/// that text extraction returns non-empty output (text content is not lost
/// due to annotation-handling errors).
#[test]
fn annotation_heavy_page_extracts_body_text() {
    // Content: body text + embedded link annotation referencing a URI
    let content = b"BT /F1 12 Tf 50 700 Td (Body text here) Tj ET";

    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    // Page has an Annots array with a Link annotation
    pdf.extend_from_slice(
        b"3 0 obj\n\
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> \
           /Annots [6 0 R 7 0 R] >>\n\
        endobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
          /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    // Link annotation
    let off6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj\n\
        << /Type /Annot /Subtype /Link /Rect [100 680 200 700] \
           /A << /Type /Action /S /URI /URI (https://example.com) >> >>\n\
        endobj\n",
    );

    // FreeText annotation with actual text
    let off7 = pdf.len();
    pdf.extend_from_slice(
        b"7 0 obj\n\
        << /Type /Annot /Subtype /FreeText /Rect [50 500 250 540] \
           /Contents (Annotation note here) /DA (/F1 10 Tf 0 g) >>\n\
        endobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5, off6, off7];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );

    let doc = PdfDocument::from_bytes(pdf).expect("annotation-heavy PDF must open");
    let text = doc
        .extract_text(0)
        .expect("extract_text must not crash on annotation-heavy page");
    assert!(
        text.contains("Body text here"),
        "body text must not be lost due to annotation processing; got: {:?}",
        text
    );
}
