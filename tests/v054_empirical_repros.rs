//! v0.3.54 empirical verification — load the captured pre-fix repros
//! and assert the fixes land. Run with:
//!
//! ```
//! cargo test --test v054_empirical_repros -- --nocapture
//! ```
//!
//! These tests load real PDFs from `/tmp/v054-repros/` (recovered from
//! issue attachments and share/test_pdfs/), so they're **opt-in**: each
//! test bails gracefully if its fixture is missing. They're not vendored
//! into the repo (third-party PDFs, unknown redistribution rights), but
//! they're the canonical fixtures from issues #534 / #535 / #536 /
//! #537 and from `~/projects/share/share/PDF_OXIDE_ISSUES.md`.
//!
//! Per `feedback_empirical_verification`: prove capabilities by running
//! them; honest about gaps; never claim untested.

use pdf_oxide::PdfDocument;
use std::path::Path;

fn read_pdf(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("[v054] fixture missing, skipping: {}", path);
        return None;
    }
    std::fs::read(path).ok()
}

/// Truncate `s` to at most `max_bytes`, rounded down to the nearest
/// UTF-8 char boundary. `&s[..max_bytes]` panics if `max_bytes` lands
/// mid-codepoint (very likely with Hebrew / diacritics).
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// #537: Hebrew RTL — `U_Magic_Palace_Eilat.pdf` (issue attachment).
///
/// Pre-fix: Hebrew codepoints emitted in visual order
///   `### חרק` = U+05D7 U+05E8 U+05E7 (chet-resh-qof — REVERSED).
/// Post-fix: should emit in logical order
///   `### קרח` = U+05E7 U+05E8 U+05D7 (qof-resh-chet — Hebrew "insect").
#[test]
fn fix_537_hebrew_magic_palace_logical_order() {
    let Some(bytes) = read_pdf("/tmp/v054-repros/hebrew_537.pdf") else {
        return;
    };
    let doc = PdfDocument::from_bytes(bytes).expect("parse hebrew PDF");
    let opts = pdf_oxide::converters::ConversionOptions::default();
    let md = doc.to_markdown_all(&opts).expect("extract markdown");
    eprintln!("[#537] head of markdown (first ~1000 bytes):");
    eprintln!("{}", truncate_at_char_boundary(&md, 1000));
    // Spot-check: Hebrew should NOT appear with the reversed codepoint
    // signature U+05D7 U+05E8 U+05E7 ("חרק" in visual / wrong order).
    // We can't assert the exact correct word without ground-truth
    // labelling, but the visual-order reversed signature is a strong
    // negative signal — if it's present, the detector didn't fire.
    let bad_visual = "\u{05D7}\u{05E8}\u{05E7}";
    if md.contains(bad_visual) {
        eprintln!(
            "[#537] WARN — output still contains reversed-Hebrew signature {:?} \
             (may be a coincidental run; check the actual content)",
            bad_visual
        );
    }
}

/// #534: tight 2-col prose — `share/test_pdfs/issue_07_orphaned_fragments.pdf`.
///
/// Pre-fix: rows interleave, producing fragments like
///   "Bulk storage zone is running at 87% capacity for with Q1 output of 8,000".
/// Post-fix: column-by-column reading order; the left-col line should
/// not be glued to the right-col line on the same y-baseline.
#[test]
fn fix_534_multicol_orphan_no_row_interleave() {
    let Some(bytes) =
        read_pdf("/home/yfedoseev/projects/share/share/test_pdfs/issue_07_orphaned_fragments.pdf")
    else {
        return;
    };
    let doc = PdfDocument::from_bytes(bytes).expect("parse issue_07 PDF");
    let opts = pdf_oxide::converters::ConversionOptions::default();
    let md = doc.to_markdown_all(&opts).expect("extract markdown");
    eprintln!("[#534] markdown:");
    eprintln!("{}", md);
    // The canonical interleave glues "87% capacity for" (left-col) to
    // "with Q1 output of 8,000" (right-col). If those two phrases are
    // back-to-back in the output, the row-interleave bug is still
    // present.
    let bad = "87% capacity for with Q1";
    assert!(
        !md.contains(bad),
        "[#534] row-by-row interleave bug still present — found {:?} in output. \
         The left-column and right-column lines are glued together.",
        bad
    );
}

/// #535: bullet `•` and `fi`/`fl` ligature decode via the new §9.10.2
/// Priority 3c fallback. Fixture: `share/test_pdfs/issue_13_unicode_ligatures.pdf`.
#[test]
fn fix_535_bullet_and_ligature_decode() {
    let Some(bytes) =
        read_pdf("/home/yfedoseev/projects/share/share/test_pdfs/issue_13_unicode_ligatures.pdf")
    else {
        return;
    };
    let doc = PdfDocument::from_bytes(bytes).expect("parse issue_13 PDF");
    let opts = pdf_oxide::converters::ConversionOptions::default();
    let md = doc.to_markdown_all(&opts).expect("extract markdown");
    eprintln!("[#535] markdown:");
    eprintln!("{}", md);
    // Bullet character should be U+2022, not U+2B59 (the wrong-glyph
    // substitution we're fixing).
    let bad_bullet = "\u{2B59}";
    assert!(
        !md.contains(bad_bullet),
        "[#535] wrong bullet U+2B59 ❍ still present in output — the §9.10.2 \
         Priority 3c (embedded post-table → AGL) fallback didn't fire."
    );
}

/// #535b, superseded. A Type0 font whose `/ToUnicode` misses the drawn codes
/// no longer has its text "recovered" by guessing.
///
/// The original fixture is deliberately broken: `/Encoding /Identity-H` makes
/// the codes two bytes (`<0048 0065 006C 006C 006F>`, "Hello" as CIDs), while
/// its `/ToUnicode` declares a **one-byte** codespace `<00> <FF>` and maps
/// only `<41>`..`<5A>`. Read literally, the single byte `0x48` falls in that
/// range and yields `H`; nothing else does.
///
/// v0.3.54 filled the gap by treating an uncovered CID as a Unicode
/// codepoint, which turned this into "Hello", and this test pinned that.
/// **v0.3.71 removed the guess on purpose**: it emitted
/// "plausible-but-wrong, content-like" characters — a `ti` ligature became
/// `:` so `notificacao` read `no:ficacao` — which is silent corruption. For
/// Identity-ordered fonts the CID-as-Unicode guess is now restricted to
/// whitespace, and any other uncovered CID decodes to `U+FFFD`.
///
/// So the assertion is inverted: what matters is that the letters are **not**
/// invented. Verified to fail identically at the v0.3.77 tag, so this was
/// never a v0.3.78 regression — the test simply outlived the decision it
/// encoded, and went unnoticed because it read its fixture from an absolute
/// path outside the repository and returned `Ok` when that path was missing.
/// It now builds its own.
#[test]
fn cmap_miss_does_not_invent_letters() {
    let doc = PdfDocument::from_bytes(cmap_miss_pdf()).expect("synthetic PDF parses");
    let text = doc.extract_text(0).expect("extract page 0");

    // The one code the /ToUnicode genuinely covers.
    assert!(text.contains('H'), "the code the CMap does cover should still decode: {text:?}");

    // The codes it does not cover must not be guessed into letters.
    for invented in ["ello", "Hello", "World", "orld"] {
        assert!(
            !text.contains(invented),
            "uncovered CIDs were guessed into {invented:?}, which v0.3.71 removed \
             deliberately as silent corruption: {text:?}"
        );
    }
}

/// The fixture from #535b, built in-code: Type0 / Identity-H over a
/// CIDFontType2, with a `/ToUnicode` whose codespace is one byte wide while
/// the text is written in two-byte codes.
fn cmap_miss_pdf() -> Vec<u8> {
    let content = b"BT\n/F1 24 Tf\n50 700 Td\n<0048 0065 006C 006C 006F> Tj\n\
                    0 -30 Td\n<0057 006F 0072 006C 0064> Tj\nET\n"
        .to_vec();
    let tounicode = b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
                      /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
                      /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
                      1 begincodespacerange\n<00> <FF>\nendcodespacerange\n\
                      1 beginbfrange\n<41> <5A> <0041>\nendbfrange\n\
                      endcmap\nCMapSpaceUsed\nend end\n"
        .to_vec();

    let mut pdf = Vec::new();
    let mut off = [0usize; 9];
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.7\n");
    off[1] = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = pdf.len();
    push!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    );
    off[4] = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()));
    pdf.extend_from_slice(&content);
    push!("endstream\nendobj\n");
    off[5] = pdf.len();
    push!(
        "5 0 obj\n<< /Type /Font /Subtype /Type0 /BaseFont /TestFont \
         /Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>\nendobj\n"
    );
    off[6] = pdf.len();
    push!(
        "6 0 obj\n<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont \
         /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
         /DW 600 >>\nendobj\n"
    );
    off[7] = pdf.len();
    push!(format!("7 0 obj\n<< /Length {} >>\nstream\n", tounicode.len()));
    pdf.extend_from_slice(&tounicode);
    push!("endstream\nendobj\n");

    let xref = pdf.len();
    push!("xref\n0 8\n0000000000 65535 f \r\n");
    for id in 1..=7 {
        push!(format!("{:010} 00000 n \r\n", off[id]));
    }
    push!(format!("trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"));
    pdf
}

/// #536: French Louis Segond Bible page 10 (Genesis 1). The pre-fix
/// failure: the multi-column body got rendered as a Markdown table
/// where each row glues a left-column verse to a right-column verse.
/// Post-fix should NOT contain a Markdown table over the verse-body
/// region.
#[test]
fn fix_536_bible_no_table_cascade() {
    let Some(bytes) = read_pdf("/tmp/v054-repros/twocol_536_v3.pdf") else {
        return;
    };
    let doc = PdfDocument::from_bytes(bytes).expect("parse Bible PDF");
    // Page 10 (Genesis 1) — the canonical bug site.
    let opts = pdf_oxide::converters::ConversionOptions::default();
    let md = doc.to_markdown(9, &opts).expect("extract page 10");
    eprintln!("[#536] page 10 markdown (first ~2000 bytes):");
    eprintln!("{}", truncate_at_char_boundary(&md, 2000));
    // The pre-fix output had `| 1 Au | commencement | Dieu | créa | ... |`
    // — a Markdown table with verse 1 spread across cells. The fix is
    // correct if the body extracts as prose paragraphs, not a Markdown
    // grid.
    let bad_grid = "| 1 Au | commencement |";
    assert!(
        !md.contains(bad_grid),
        "[#536] Bible page 10 still rendered as Markdown table — found {:?} \
         in output. The 2-col-prose classifier + tight-gutter cut didn't \
         resolve the spatial-table-detector cascade.",
        bad_grid
    );
}
