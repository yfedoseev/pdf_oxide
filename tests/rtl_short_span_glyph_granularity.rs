//! `document.rs`'s "Pass 0.75" (merge-and-reverse, added to fix per-glyph
//! scanned-OCR Hebrew text) fires on *any* run of 4+ consecutive short
//! (≤2-char) same-line RTL spans — it has no way to tell a genuine
//! one-glyph-per-show-op OCR run from four ordinary, already-multi-glyph,
//! already-logical-order spans that just happen to be short (a
//! ligature/kerning-pair break, a font-subsetting run boundary — routine in
//! any digitally-authored Hebrew or Arabic PDF). `ordinary_two_char_spans_are_not_reversed`
//! below reproduces this with four 2-letter spans; the fix is narrow —
//! tighten the gate to "exactly one base RTL letter" (matching
//! `extractors/text.rs`'s already-correct `is_rtl_glyph_piece`) instead of
//! "≤2 chars total". See the `//` markers in `document.rs`'s Pass 0.75
//! (`is_short_rtl` / `is_short`) for exactly where.
//!
//! That fix is necessary but not sufficient. `ordinary_two_char_spans_are_not_reversed`
//! still fails after it — but not the way it originally did. Before the
//! fix, `document.rs`'s Pass 0.75 merged the 4 spans itself and reversed
//! their combined text. After the fix, Pass 0.75 no longer touches them,
//! but the test still fails, with a different, very telling shape: the
//! full 8-character run comes back as one exact character-for-character
//! reversal (`"אבגדהוזח"` → `"חזוהדגבא"`, span boundaries preserved as
//! line breaks at their new positions). That's the signature of a
//! completely different, unrelated merge path: these 4 spans are tightly
//! kerned (touching, zero gap) — ordinary same-language adjacent-span
//! joining (unrelated to any OCR/glyph-piece detection, and unrelated to
//! Pass 0.75) combines them into one span on its own, and *that* combined
//! span then gets swept up and reversed by whatever this codebase's
//! general geometric visual/logical detector is doing here — the same
//! class of confidence-gated ascending-x heuristic already fixed once
//! (for a different flush site) elsewhere in this codebase, evidently not
//! yet correct for whatever path this ordinary-merge produces. This
//! matches the real Hebrew/Arabic Wikipedia article exports in our corpus
//! (ordinary digitally-authored PDFs, not scanned/OCR) that still come
//! out with entire lines individually word-mirrored even with the Pass
//! 0.75 fix applied and rebased onto the per-word OCR-sandwich fix
//! already on `main`. The exact before/after `extract_text()` output on
//! those real documents (not the PDFs themselves) is in this PR's review
//! thread. This second problem needs someone with more context on the
//! confidence-gated detector's other call sites than a first-time reader
//! of this codebase has — it isn't Pass 0.75, and it isn't fixed here.

use pdf_oxide::PdfDocument;

/// Minimal untagged one-page PDF: a simple TrueType font (`/FirstChar 0`,
/// `/Widths`, `/ToUnicode`) and a plain content stream, matching the shape
/// `tests/rtl_tj_array_word_buffer.rs` (on `main`) already uses for the
/// per-word case of this same issue.
fn build_pdf(tounicode_bfchars: &str, widths: &str, last_char: usize, content_ops: &str) -> Vec<u8> {
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n\
         1 begincodespacerange <00> <FF> endcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end",
        tounicode_bfchars.lines().filter(|l| !l.trim().is_empty()).count(),
        tounicode_bfchars,
    );

    let content_bytes = content_ops.as_bytes();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, "", content_bytes);
    obj(
        &mut buf,
        &mut off,
        5,
        &format!(
            "<< /Type /Font /Subtype /TrueType /BaseFont /Synthetic \
             /FirstChar 0 /LastChar {last_char} /Widths [{widths}] /ToUnicode 6 0 R >>"
        ),
    );
    stream(&mut buf, &mut off, 6, "", tounicode.as_bytes());

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    buf
}

// 8 distinct Hebrew letters, codes <01>..<08>, plus <20> for space.
const TOUNICODE: &str = "\
<01> <05D0>
<02> <05D1>
<03> <05D2>
<04> <05D3>
<05> <05D4>
<06> <05D5>
<07> <05D6>
<08> <05D7>
<20> <0020>
";
const WIDTHS: &str = "600 600 600 600 600 600 600 600 600";
const LAST_CHAR: usize = 8;

/// Ordinary digitally-authored shape (the regression): four consecutive
/// **2-letter** spans, each already in correct logical order on its own
/// (a routine font kerning-pair/subsetting-run split — not per-glyph OCR),
/// same y, ascending x. Matches `wiki-cat-he/source.pdf`'s real span shape
/// at the point where this PR corrupts it (spans there: "בד" then "מל",
/// among others in the same short-span run).
#[test]
fn ordinary_two_char_spans_are_not_reversed() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf\n\
         50 100 Td [<01><02>] TJ\n\
         64 0 Td [<03><04>] TJ\n\
         78 0 Td [<05><06>] TJ\n\
         92 0 Td [<07><08>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[regression] two-char-span run extracted: {text:?}");

    // Each 2-letter span is already correct; the run merge must not touch
    // per-span letter order OR the run's overall span order.
    let expected = "\u{05D0}\u{05D1}\u{05D2}\u{05D3}\u{05D4}\u{05D5}\u{05D6}\u{05D7}";
    assert!(
        text.contains(expected),
        "ordinary short (2-char) RTL spans got merged and reversed as if they \
         were a per-glyph OCR run — this is the wiki-cat-he/wiki-cat-ar \
         regression. got: {text:?}"
    );
}

/// The actual per-glyph OCR shape this branch targets: four consecutive
/// **1-letter** spans (one show op per glyph — matches the reporter's
/// scanned-Hebrew repro's real content stream, `[<04>] TJ` / `[<03>] TJ` /
/// `[<02>] TJ` / `[<01>] TJ`), same y, ascending x, stored in logical
/// order. This is the case Pass 0.75 must keep fixing — any change to fix
/// the test above must not break this one.
#[test]
fn true_per_glyph_run_is_still_reversed() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf\n\
         50 100 Td [<01>] TJ\n\
         64 0 Td [<02>] TJ\n\
         78 0 Td [<03>] TJ\n\
         92 0 Td [<04>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[target] per-glyph run extracted: {text:?}");

    // Logical order is code 04 first (rightmost = read first in RTL), down
    // to code 01 last — the reverse of storage/draw order.
    let expected = "\u{05D3}\u{05D2}\u{05D1}\u{05D0}";
    assert!(
        text.contains(expected),
        "true per-glyph OCR run (one show op per glyph) must still be \
         merged and reversed to logical order — got: {text:?}"
    );
}

