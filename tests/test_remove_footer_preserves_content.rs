//! tests for remove_footer verifying that it does not remove
//! page content that happens to overlap the footer area

mod common;
use common::build_pdf_with_page_extras;
use pdf_oxide::PdfDocument;

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

/// A phrase that legitimately repeats across pages of a form but at a
/// DIFFERENT horizontal position each time (e.g. a per-field instruction like
/// "see instructions" that sits beside a different control on each page) must
/// not be mistaken for chrome. Genuine running footers are position-locked;
/// this one drifts in x, so it is real content.
#[test]
fn remove_footers_preserves_repeated_phrase_that_drifts_in_x() {
    let phrase = "See the instructions";
    // Same phrase, in the footer band (y=30) on all 5 pages, but shifted well
    // beyond the x tolerance (72, 152, 232, 312, 392 → spread 320pt).
    let bytes = build_pdf_with_page_extras(5, |i| {
        let x = 72 + (i as i32) * 80;
        format!("BT /F1 10 Tf 1 0 0 1 {x} 30 Tm ({phrase}) Tj ET\n")
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_footers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            text.contains(phrase),
            "page {page}: phrase {phrase:?} repeats at a different x per page (real \
             content, not a running footer) but was removed: {text:?}"
        );
    }
}

/// The complementary guard: a genuine, position-locked running footer must
/// still be removed. Without this, "preserve everything" would silently
/// neuter the feature.
#[test]
fn remove_footers_still_removes_position_locked_footer() {
    let phrase = "Confidential Draft Only";
    // Identical (x=72, y=30) on every page — a real running footer.
    let bytes = build_pdf_with_page_extras(5, |_i| {
        format!("BT /F1 10 Tf 1 0 0 1 72 30 Tm ({phrase}) Tj ET\n")
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    let removed = doc.remove_footers(0.5).unwrap();
    assert!(removed >= 5, "expected the running footer removed on every page, got {removed}");

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        assert!(
            !text.contains(phrase),
            "page {page}: position-locked running footer {phrase:?} should have been \
             removed: {text:?}"
        );
    }
}

/// Zone-scoped erasure: when a string qualifies as footer chrome, only the
/// occurrence IN the footer band is erased — an identically worded span
/// elsewhere on the page (a real body label) must survive. The old heuristic
/// erased every span whose text matched, deleting body content.
#[test]
fn remove_footers_preserves_body_twin_of_footer_text() {
    let phrase = "Company Confidential";
    // Footer occurrence at (72, 30) [removed] and a body twin at (72, 350)
    // [must survive], on every page.
    let bytes = build_pdf_with_page_extras(5, |_i| {
        format!(
            "BT /F1 10 Tf 1 0 0 1 72 30 Tm ({phrase}) Tj ET\n\
             BT /F1 10 Tf 1 0 0 1 72 350 Tm ({phrase}) Tj ET\n"
        )
    });
    let doc = PdfDocument::from_bytes(bytes).unwrap();
    doc.remove_footers(0.5).unwrap();

    for page in 0..5 {
        let text = doc.extract_text(page).unwrap();
        let hits = text.matches(phrase).count();
        assert_eq!(
            hits, 1,
            "page {page}: footer occurrence of {phrase:?} should be removed and the body \
             twin kept (expected 1 remaining, got {hits}): {text:?}"
        );
    }
}
