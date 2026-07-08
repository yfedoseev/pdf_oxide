//! tests for remove_footer verifying that it does not remove
//! page content that happens to overlap the footer area

use pdf_oxide::PdfDocument;

// ---------------- test helper: build_pdf_with_page_extras -------------------
//
// two fn used only by build_pdf_with_page_extras write one object each,
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

/// Minimal single-page-content PDF builder: N pages, each with a body
/// paragraph plus arbitrary extra content-stream text supplied per page.
fn build_pdf_with_page_extras(
    page_count: usize,
    extra_per_page: impl Fn(usize) -> String,
) -> Vec<u8> {
    // buffer for the PDF we're building
    let mut buf: Vec<u8> = Vec::new();

    // `off[N]` = byte offset where object N's bytes start, filled in as
    // each object is written below. `xref_off` (further down) separately
    // records where the xref table itself starts.
    let mut off = vec![0usize; 4 + page_count * 2];

    // PDF File header
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    // Catalog
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");

    // Pages tree root
    // Build the /Kids array value ahead of time: "5 0 R 7 0 R 9 0 R ..."
    // — one indirect reference per page object we're about to create.
    // (Page objects are 5, 7, 9, ... because each page also needs a
    // content-stream object right before it: 4, 6, 8, ... — see the loop
    // below.)
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

    // Font resource
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    // --- One content stream + one page object, per page ---
    for i in 0..page_count {
        let content_id = 4 + i * 2; // 4, 6, 8, ...
        let page_id = 5 + i * 2; // 5, 7, 9, ...

        // text object per page + whatever the test wants to insert
        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n{}",
            extra_per_page(i)
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());

        // Page object:
        // physical size - `/MediaBox`, in points `[0 0 612 792]` is US Letter
        // resources it can reference by name - `/Resources`
        // -  just our one font as `/F1`
        // object w/ drawing instructions `/Contents` with content-stream object
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

    // Cross-reference table - `xref_off`
    // flat table mapping object number -> byte offset
    // records where THIS table itself starts (needed for the trailer).
    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());

    // fixed, required first entry marking object 0 as "free"
    buf.extend_from_slice(b"0000000000 65535 f \n");

    // one `NNNNNNNNNN 00000 n` line per real object, `n` meaning "in use",
    // giving its 10-digit zero-padded byte offset
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }

    // Trailer
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

/// Real (public-domain) prose — five DISTINCT excerpts from Virginia
/// Woolf's "A Room of One's Own" (gutenberg.net.au/ebooks02/0200791h.html),
/// one per page, each laid out line-by-line so the paragraph runs from
/// clearly above the footer band down through it — simulating dense body
/// text whose last few lines happen to land in the margin zone
///
/// Line height 14pt, starting at y=160 (well
/// above both the 12% line at 95.04 and the 15% line at 118.8 on a
/// 792pt-tall page) — each paragraph physically straddles the boundary
/// rather than sitting entirely inside or outside the band.
///
/// No text repeats across pages, `threshold` is bumped to 0.5
/// (`min_occurrences = ceil(5 * 0.5) = 3`) to stay clear of the
/// unrelated `min_occurrences` degenerate case (at 0.2 with 5 pages
/// that resolves to 1, meaning any single unique line would count as
/// "recurring") — this test is about whether real prose gets mistaken
/// for chrome.
#[test]
fn remove_footers_preserves_real_prose_overlapping_band() {
    let excerpts: [&[&str]; 5] = [
        &[
            "The strains of the gramophone blared out from the",
            "rooms within. It was impossible not to reflect the",
            "reflection whatever it may have been was cut short.",
            "The clock struck; it was time to find one's way to",
            "luncheon.",
        ],
        &[
            "So we talked standing at the window and looking, as",
            "so many thousands look every night, down on the",
            "domes and towers of the famous city beneath us. It",
            "was very beautiful, very mysterious in the autumn",
            "moonlight.",
        ],
        &[
            "All human beings were laid asleep prone, horizontal,",
            "dumb. Nobody seemed stirring in the streets of",
            "Oxbridge. Even the door of the hotel sprang open at",
            "the touch of an invisible hand not a boots was",
            "sitting up to light me to bed, it was so late.",
        ],
        &[
            "The usual hoarse-voiced men paraded the streets",
            "with plants on barrows. Some shouted; others sang.",
            "London was like a workshop. London was like a",
            "machine. We were all being shot backwards and",
            "forwards on this plain foundation to make some",
            "pattern.",
        ],
        &[
            "while my own notebook rioted with the wildest",
            "scribble of contradictory jottings. It was",
            "distressing, it was bewildering, it was humiliating.",
            "Truth had run through my fingers. Every drop had",
            "escaped.",
        ],
    ];

    let bytes = build_pdf_with_page_extras(5, |i| {
        let excerpt = excerpts[i];
        let mut content = String::new();
        for (line_idx, line) in excerpt.iter().enumerate() {
            let y = 160 - (line_idx as i32) * 14;
            content.push_str(&format!("BT /F1 10 Tf 1 0 0 1 72 {y} Tm ({line}) Tj ET\n"));
        }
        content
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_footers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        for line in excerpts[page] {
            assert!(
                text.contains(line),
                "page {page}: real prose line {line:?} was wrongly removed as footer \
                 chrome: {text:?}"
            );
        }
    }
}

/// Real PDFs, especially OCR output (Tesseract, Acrobat, OmniPage) and
/// programmatic/desktop-publishing generators (InDesign, PDFKit, etc),
/// very often place each WORD as its own separate text element
/// with its own coordinates, rather than a whole line as one string.
///
/// Six pages, each an unrelated one-line "sentence" of DIFFERENT words —
/// no two pages share the same sentence — except every sentence happens
/// to end with the same common word, "the", positioned at the same (x,
/// y) on every page because it's always the 5th word on the line. That
/// mirrors how, in a real book, a short common word can land at the same
/// wrapped-line position across many unrelated pages by pure coincidence
/// of layout — not because it's chrome.
#[test]
fn remove_footers_preserves_common_word_across_unique_sentences() {
    let sentences: [[&str; 5]; 6] = [
        ["He", "walked", "down", "to", "the"],
        ["She", "turned", "back", "toward", "the"],
        ["They", "wandered", "along", "beside", "the"],
        ["It", "drifted", "slowly", "past", "the"],
        ["We", "lingered", "there", "beyond", "the"],
        ["I", "hesitated", "just", "before", "the"],
    ];

    let bytes = build_pdf_with_page_extras(6, |i| {
        let words = sentences[i];
        let mut content = String::new();
        for (word_idx, word) in words.iter().enumerate() {
            let x = 72 + (word_idx as i32) * 40;
            content.push_str(&format!("BT /F1 10 Tf 1 0 0 1 {x} 30 Tm ({word}) Tj ET\n"));
        }
        content
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_footers(0.2).unwrap();

    for (page, words) in sentences.iter().enumerate() {
        let text = doc.extract_text(page).unwrap();
        for word in words {
            assert!(
                text.contains(word),
                "page {page}: word {word:?} from an otherwise-unique sentence was \
                 wrongly removed as footer chrome: {text:?}"
            );
        }
    }
}
