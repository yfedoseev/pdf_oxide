//! Regression tests for issue #794: `remove_footers` on untagged PDFs
//! where the footer's page number shares a baseline with constant brand
//! chrome — in one combined span, and in a recto/verso (alternating-page)
//! layout split across separate spans.

use pdf_oxide::PdfDocument;

// ---------------- test helper: build_pdf_with_page_extras -------------------
//
// two fn used by pdf build helpers to write one object each,
// recording its offset as they go:
// - `buf`      the buffer we're writing into
// - `off[id]`  start of object definition

// -- write a plain dictionary object --
fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

// -- write a `stream` object - used here for page content --
fn stream(buf: &mut Vec<u8>, off: &mut [usize], id: usize, data: &[u8]) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

fn build_combined_span_footer_pdf(page_count: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 4 + page_count * 2];

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    for i in 0..page_count {
        let content_id = 4 + i * 2;
        let page_id = 5 + i * 2;
        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n\
             BT /F1 10 Tf 1 0 0 1 72 30 Tm ({} com) Tj ET\n",
            i + 1
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());
        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// A footer where the page number and a constant brand fragment are ONE
/// span ("1 com", "2 com", ...) — mirroring a real single-run footer like
/// "1 ERLC.com". Confirmed to fail before this fix and pass after: the
/// exact-text pass only matches a span's full text verbatim against
/// itself, so a span whose digit varies every page never repeats and is
/// invisible to it; the digit-normalizing signature ("# com") is what
/// catches it.
#[test]
fn remove_footers_strips_combined_page_number_and_brand_span() {
    let bytes = build_combined_span_footer_pdf(6);
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    let removed = doc.remove_footers(0.2).unwrap();
    assert!(removed > 0, "expected remove_footers to erase something");

    for page in 1..6 {
        let text = doc.extract_text(page).unwrap();
        assert!(!text.contains("com"), "page {page}: brand fragment survived: {text:?}");
        assert!(
            !text.contains(&format!("{}", page + 1)),
            "page {page}: page number survived: {text:?}"
        );
        assert!(
            text.contains("Body text placeholder"),
            "page {page}: body content wrongly removed: {text:?}"
        );
    }
}

/// Build a PDF where a short (<=3 char), constant, digit-less brand
/// fragment (e.g. the "com" in "ERLC.com") appears ONLY on odd pages,
/// alongside a page number — mirroring a recto/verso footer where a
/// domain is split across spans and only printed on every other page.
fn build_parity_brand_footer_pdf(page_count: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 4 + page_count * 2];

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    for i in 0..page_count {
        let content_id = 4 + i * 2;
        let page_id = 5 + i * 2;
        let footer = if i % 2 == 1 {
            format!(
                "BT /F1 10 Tf 1 0 0 1 72 30 Tm ({}) Tj ET\n\
                 BT /F1 10 Tf 1 0 0 1 100 30 Tm (com) Tj ET\n",
                i + 1
            )
        } else {
            String::new()
        };
        let content =
            format!("BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n{footer}");
        stream(&mut buf, &mut off, content_id, content.as_bytes());
        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// A short, constant, digit-less brand fragment (e.g. "com") that only
/// ever appears on ONE parity of pages (recto/verso alternation) must
/// still be recognised as running footer chrome, even though it never
/// reaches 50% of the whole document.
#[test]
fn remove_footers_strips_parity_alternating_brand_fragment() {
    let bytes = build_parity_brand_footer_pdf(10);
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    let removed = doc.remove_footers(0.2).unwrap();
    assert!(removed > 0, "expected remove_footers to erase something");

    // Page 1 (index 1) is the first occurrence and is deliberately
    // exempted (it might be a genuine cover-page title); pages 3/5/7/9
    // are not, and must be fully cleaned.
    for page in [3usize, 5, 7, 9] {
        let text = doc.extract_text(page).unwrap();
        assert!(!text.contains("com"), "page {page}: brand fragment survived: {text:?}");
        assert!(
            text.contains("Body text placeholder"),
            "page {page}: body content wrongly removed: {text:?}"
        );
    }
}

/// Build a PDF where a legitimate page-number footer ("Page N", far
/// right) and an unrelated real footnote ("* see appendix for details",
/// far left) share the same baseline on a minority of pages, spaced well
/// past any reasonable column-gap threshold.
fn build_baseline_neighbor_footer_pdf(page_count: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 4 + page_count * 2];

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    // Footnote pages mixed across both parities (3 even, 3 odd out of 5
    // each) — 60% of either parity, deliberately under the 80%-of-one-
    // parity bar the digit-less-parity path requires, so the footnote can
    // never qualify as chrome standing alone.
    let footnote_pages = [0usize, 2, 4, 5, 7, 9];

    for i in 0..page_count {
        let content_id = 4 + i * 2;
        let page_id = 5 + i * 2;
        let footnote = if footnote_pages.contains(&i) {
            "BT /F1 10 Tf 1 0 0 1 36 30 Tm (* see appendix for details) Tj ET\n".to_string()
        } else {
            String::new()
        };
        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n\
             {footnote}\
             BT /F1 10 Tf 1 0 0 1 400 30 Tm (Page {}) Tj ET\n",
            i + 1
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());
        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// A real footnote sharing a baseline with a legitimate page-number
/// footer, far apart horizontally, on a minority of pages.
/// Column-gap-unaware line grouping fuses the footnote into the passing
/// page-number's signature and erases both, because the page number's
/// varying digit pushes the *merged* signature into the looser
/// 50%-of-whole-document varying-literal path even though the footnote
/// alone (60% of either parity) never clears the stricter 80%-of-one-
/// parity bar it would otherwise need. The footnote must survive on every
/// page it appears on; the page number must be stripped everywhere except
/// its first occurrence (kept in case it's a cover-page title).
#[test]
fn remove_footers_preserves_footnote_sharing_baseline_with_page_number() {
    let bytes = build_baseline_neighbor_footer_pdf(10);
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    // 0.8: high enough that pass 2's own exact-text heuristic
    // (`ceil(10 * 0.8) = 8`) can't independently catch the footnote, which
    // recurs on 6 of 10 pages. This test is about whether grouping spans
    // into lines by baseline lets an unrelated footnote defeat removal of
    // the legit constant footer sharing that baseline (or vice versa) —
    // not about that pass-2 heuristic, so it must be kept out of the way.
    let removed = doc.remove_footers(0.8).unwrap();
    assert!(removed > 0, "expected remove_footers to erase something");

    for page in [0usize, 2, 4, 5, 7, 9] {
        let text = doc.extract_text(page).unwrap();
        assert!(
            text.contains("* see appendix for details"),
            "page {page}: real footnote wrongly removed as footer chrome: {text:?}"
        );
    }

    for page in 1..10 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            !text.contains(&format!("Page {}", page + 1)),
            "page {page}: page-number footer survived: {text:?}"
        );
    }
}

/// Build a PDF where a varying-digit page-number footer sits alone on its
/// line on every page EXCEPT one, where an incidental, page-unique snippet
/// lands close enough (within the column-gap threshold) to merge onto the
/// same line. `chrome` renders the footer's content-stream snippet(s) for a
/// given (1-indexed) page number — a single span for `"Page N"`, or
/// multiple adjacent spans to simulate chrome split across spans (e.g.
/// OCR'd "Page" / "N"). `incidental` is the raw content-stream snippet for
/// the incidental content, emitted only on `merge_on_page`.
fn build_incidental_neighbor_footer_pdf(
    page_count: usize,
    merge_on_page: usize,
    incidental: &str,
    chrome: impl Fn(usize) -> String,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 4 + page_count * 2];

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    for i in 0..page_count {
        let content_id = 4 + i * 2;
        let page_id = 5 + i * 2;
        let incidental_snippet = if i == merge_on_page { incidental } else { "" };
        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n\
             {incidental_snippet}\
             {}",
            chrome(i + 1)
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());
        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Does incidental content merging into the SAME group as genuine chrome
/// cause that page's chrome to escape removal, since the merged per-page
/// literal ("Note Page #") no longer matches the cross-page-recurring
/// signature ("Page #") that every other (unmerged) page shares?
#[test]
fn remove_footers_strips_page_num_span_keeps_incidental_content() {
    // "Note" at x=355 (10pt font, ~20pt wide, ends ~375) sits 25pt from
    // "Page N" at x=400 — inside the (font_size * 3.0).max(30.0) = 30pt
    // column-gap threshold, so group_band_lines merges them into one group
    // on this page only.
    let incidental = "BT /F1 10 Tf 1 0 0 1 355 30 Tm (Note) Tj ET\n";
    let chrome = |n: usize| format!("BT /F1 10 Tf 1 0 0 1 400 30 Tm (Page {n}) Tj ET\n");
    let bytes = build_incidental_neighbor_footer_pdf(6, 2, incidental, chrome);
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    doc.remove_footers(0.5).unwrap();

    for page in 1..6 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            !text.contains(&format!("Page {}", page + 1)),
            "page {page}: page-number footer survived: {text:?}"
        );
    }
}

/// Similar to above test, but the chrome itself is split into two spans
/// ("Page" / "N") the way OCR text commonly comes out, with a
/// space-sized gap between them (font-relative).
#[test]
fn remove_footers_strips_split_page_num_keeps_incidental_content() {
    let incidental = "BT /F1 10 Tf 1 0 0 1 355 30 Tm (Note) Tj ET\n";
    // 10pt Helvetica: "Page" is ~4 chars wide (~20pt); a space-sized gap at
    // this font size is ~2.8pt (Helvetica space advance is 278/1000 em).
    let chrome = |n: usize| {
        format!(
            "BT /F1 10 Tf 1 0 0 1 400 30 Tm (Page) Tj ET\n\
             BT /F1 10 Tf 1 0 0 1 422.8 30 Tm ({n}) Tj ET\n"
        )
    };
    let bytes = build_incidental_neighbor_footer_pdf(6, 2, incidental, chrome);
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    doc.remove_footers(0.5).unwrap();

    for page in 1..6 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            !text.contains(&format!("Page{}", page + 1)),
            "page {page}: split-span page-number footer survived: {text:?}"
        );
        assert!(
            !text.contains(&format!("Page {}", page + 1)),
            "page {page}: split-span page-number footer survived: {text:?}"
        );
    }

    let merged_page_text = doc.extract_text(2).unwrap();
    assert!(
        merged_page_text.contains("Note"),
        "page 2: incidental content wrongly removed: {merged_page_text:?}"
    );
}
