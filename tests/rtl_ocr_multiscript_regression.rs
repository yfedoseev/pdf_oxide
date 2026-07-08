//! Automated regression test for the OCR-sandwich text-extraction flow
//! across Hebrew, Arabic, and 16 other scripts — built from **real**
//! OCR-engine-format data, not hand-invented text.
//!
//! Issue #826: a scanned Hebrew OCR PDF (invisible `Tr 3` text layer,
//! simple TrueType font, one `TJ` array **per recognized word** placed at
//! plain ascending x) extracted with every Hebrew word both
//! letter-reversed and word-order-reversed. `tests/rtl_tj_array_word_buffer.rs`
//! pins the exact minimal repro; this file is the broader regression
//! guard — every script that shares the same buffer-flush code path
//! (`flush_tj_buffer`, `WordBoundaryMode::Tiebreaker`, the default) run
//! through text/markdown/HTML together, so a future change to that path
//! can't silently regress one script while "fixing" another.
//!
//! This is the **per-word** OCR-sandwich shape (one show op per whole
//! word). There is a distinct **per-glyph** shape (one show op per
//! character — see PR#828 / `tests/rtl_ocr_multiscript_regression.rs`'s
//! sibling repros under `~/projects/pdf_oxide_corpora/issue_826_repro/`)
//! that this fix does **not** address — it lives in a different part of
//! the pipeline (`merge_adjacent_spans`) and needs its own, separately
//! render-mode-gated fix. Confirmed via the full 419-PDF corpus
//! regression sweep that naively fixing the per-glyph shape (as PR#828
//! currently does) regresses ordinary, non-OCR Hebrew/Arabic PDFs
//! (Wikipedia-style exports) by reversing already-correct text — so this
//! fix's scope is intentionally narrower: real bug, zero regressions.
//!
//! RTL (Hebrew/Arabic) coverage below is scoped to plain `extract_text`
//! only — `to_markdown_all`/`to_html_all` route RTL through a *different*
//! pass (`document.rs`'s `apply_rtl_logical_order_to_ordered_spans`) that
//! is deliberately left unchanged for the same regression reason; see the
//! per-format tests below for details.
//!
//! ## Provenance
//!
//! Word text + bounding-box data below is derived from
//! `tests/resources/hello_world_scripts.hocr` in
//! <https://github.com/ocrmypdf/OCRmyPDF> (MPL-2.0), a genuine hOCR
//! fixture (Tesseract-format OCR engine output) OCRmyPDF's own test
//! suite uses to validate its hOCR→PDF "invisible text sandwich"
//! renderer — i.e. this is the real data shape a real OCR-sandwich tool
//! consumes, not a synthetic approximation. Word content per script is
//! already-correct, logical-order Unicode (a competent OCR engine's
//! actual output); bounding boxes are converted from the fixture's
//! pixel coordinates (300 DPI, 2550×3300px page) to PDF points.
//!
//! Placement here uses the plain-positive-advance-per-word convention —
//! the shape issue #826's reporter described and the one that exposed
//! the bug — rather than OCRmyPDF's own `-1`-x-scale trick (a separate,
//! also-correct encoding some fpdf2-based renderers use to route around
//! a ligature-shaping quirk in that specific library; both encodings
//! must extract correctly, but only the plain form is exercised here).
//!
//! Confirmed straight from OCRmyPDF's renderer source
//! (`src/ocrmypdf/fpdf_renderer/renderer.py`) that invisible RTL words
//! are deliberately encoded "1:1 in logical order" precisely because,
//! quote, "the text is invisible" so there is no rendering-correctness
//! pressure to mirror glyph positions — the exact premise this fix's
//! `bidi::apply_rtl_verdict` invisible-render-mode gating relies on.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

// Auto-generated from OCRmyPDF's hello_world_scripts.hocr (MPL-2.0,
// github.com/ocrmypdf/OCRmyPDF/blob/main/tests/resources/hello_world_scripts.hocr).
// 20 words, 84 unique codepoints, page 612.0x792.0pt (8.5x11in @ 300dpi).
const TOUNICODE_BFCHARS: &str = "\
<01> <0048>\n\
<02> <0065>\n\
<03> <006C>\n\
<04> <006F>\n\
<05> <0021>\n\
<06> <00A1>\n\
<07> <0061>\n\
<08> <0042>\n\
<09> <006E>\n\
<0A> <006A>\n\
<0B> <0075>\n\
<0C> <0072>\n\
<0D> <0047>\n\
<0E> <00FC>\n\
<0F> <00DF>\n\
<10> <0020>\n\
<11> <0074>\n\
<12> <041F>\n\
<13> <0440>\n\
<14> <0438>\n\
<15> <0432>\n\
<16> <0435>\n\
<17> <0442>\n\
<18> <0393>\n\
<19> <03B5>\n\
<1A> <03B9>\n\
<1B> <03AC>\n\
<1C> <03C3>\n\
<1D> <03BF>\n\
<1E> <03C5>\n\
<1F> <4F60>\n\
<21> <597D>\n\
<22> <FF01>\n\
<23> <3053>\n\
<24> <3093>\n\
<25> <306B>\n\
<26> <3061>\n\
<27> <306F>\n\
<28> <C548>\n\
<29> <B155>\n\
<2A> <D558>\n\
<2B> <C138>\n\
<2C> <C694>\n\
<2D> <004D>\n\
<2E> <0068>\n\
<2F> <0062>\n\
<30> <0928>\n\
<31> <092E>\n\
<32> <0938>\n\
<33> <094D>\n\
<34> <0924>\n\
<35> <0947>\n\
<36> <0645>\n\
<37> <0631>\n\
<38> <062D>\n\
<39> <0628>\n\
<3A> <0627>\n\
<3B> <05E9>\n\
<3C> <05DC>\n\
<3D> <05D5>\n\
<3E> <05DD>\n\
<3F> <004F>\n\
<40> <00E1>\n\
<41> <0043>\n\
<42> <0069>\n\
<43> <007A>\n\
<44> <015B>\n\
<45> <0107>\n\
<46> <60A8>\n\
<47> <0417>\n\
<48> <0434>\n\
<49> <0430>\n\
<4A> <0441>\n\
<4B> <0443>\n\
<4C> <0439>\n\
<4D> <03A7>\n\
<4E> <03B1>\n\
<4F> <03AF>\n\
<50> <03C1>\n\
<51> <03C4>\n\
<52> <0623>\n\
<53> <0647>\n\
<54> <0644>\n\
<55> <064B>\n\
";
const LAST_CHAR: usize = 85;
const WIDTHS: &str = "600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600 600";

// (lang, is_rtl, hex_byte_string_for_TJ, x_pt, y_pt, expected_logical_text)
const WORDS: &[(&str, bool, &str, f32, f32, &str)] = &[
    ("eng", false, "010203030405", 36.00, 708.00, "Hello!"),
    ("spa", false, "060104030705", 336.00, 708.00, "\u{a1}Hola!"),
    ("fra", false, "0804090A040B0C05", 36.00, 636.00, "Bonjour!"),
    ("deu", false, "0D0C0E0F100D04111105", 336.00, 636.00, "Gr\u{fc}\u{df} Gott!"),
    (
        "rus",
        false,
        "12131415161705",
        36.00,
        564.00,
        "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}!",
    ),
    (
        "ell",
        false,
        "18191A1B101C1D1E05",
        336.00,
        564.00,
        "\u{393}\u{3b5}\u{3b9}\u{3ac} \u{3c3}\u{3bf}\u{3c5}!",
    ),
    ("chi_sim", false, "1F2122", 36.00, 492.00, "\u{4f60}\u{597d}\u{ff01}"),
    (
        "jpn",
        false,
        "232425262722",
        336.00,
        492.00,
        "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{ff01}",
    ),
    (
        "kor",
        false,
        "28292A2B2C05",
        36.00,
        420.00,
        "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}!",
    ),
    ("tur", false, "2D020C2E072F0705", 336.00, 420.00, "Merhaba!"),
    (
        "hin",
        false,
        "30313233343505",
        36.00,
        348.00,
        "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}!",
    ),
    (
        "ara",
        true,
        "05363738393A",
        336.00,
        348.00,
        "!\u{645}\u{631}\u{62d}\u{628}\u{627}",
    ),
    ("heb", true, "3B3C3D3E", 36.00, 276.00, "\u{5e9}\u{5dc}\u{5d5}\u{5dd}"),
    ("por", false, "3F034005", 336.00, 276.00, "Ol\u{e1}!"),
    ("ita", false, "4142070405", 48.00, 204.00, "Ciao!"),
    ("pol", false, "414302444505", 240.00, 156.00, "Cze\u{15b}\u{107}!"),
    ("chi_tra", false, "462122", 432.00, 156.00, "\u{60a8}\u{597d}\u{ff01}"),
    (
        "rus",
        false,
        "47481349154A17154B4C05",
        48.00,
        60.00,
        "\u{417}\u{434}\u{440}\u{430}\u{432}\u{441}\u{442}\u{432}\u{443}\u{439}!",
    ),
    (
        "ell",
        false,
        "4D4E4F5019511905",
        240.00,
        60.00,
        "\u{3a7}\u{3b1}\u{3af}\u{3c1}\u{3b5}\u{3c4}\u{3b5}!",
    ),
    (
        "ara",
        true,
        "055253543A55",
        432.00,
        60.00,
        "!\u{623}\u{647}\u{644}\u{627}\u{64b}",
    ),
];

/// Untagged one-page PDF: a simple TrueType font (`/FirstChar 0`,
/// `/Widths`, `/ToUnicode`) plus a content stream drawing each of
/// `WORDS` as its own invisible (`3 Tr`) `TJ` array at its real hOCR
/// position — one `TJ` call per word, matching the buffer scoping that
/// exposed #826 (`process_tj_array_tiebreaker`'s buffer never persists
/// across separate `TJ` operators).
fn build_ocr_sandwich_pdf() -> Vec<u8> {
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n\
         1 begincodespacerange <00> <FF> endcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end",
        TOUNICODE_BFCHARS.lines().filter(|l| !l.trim().is_empty()).count(),
        TOUNICODE_BFCHARS,
    );

    let mut content_ops = String::new();
    for (_, _, hex, x, y, _) in WORDS {
        content_ops.push_str(&format!("BT /F1 24 Tf 3 Tr 1 0 0 1 {x} {y} Tm [<{hex}>] TJ ET\n"));
    }
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, "", content_bytes);
    obj(
        &mut buf,
        &mut off,
        5,
        &format!(
            "<< /Type /Font /Subtype /TrueType /BaseFont /Synthetic \
             /FirstChar 0 /LastChar {LAST_CHAR} /Widths [{WIDTHS}] /ToUnicode 6 0 R >>"
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

/// All 20 words across 18 language codes (incl. Hebrew + 2 Arabic) must
/// come out in correct, un-reversed logical order in plain-text
/// extraction — the core #826 regression guard, broadened to every
/// script sharing the same buffer-flush code path.
#[test]
fn ocr_sandwich_multiscript_text_extraction() {
    let doc = PdfDocument::from_bytes(build_ocr_sandwich_pdf()).expect("parse OCR-sandwich PDF");
    let text = doc.extract_text(0).expect("extract_text");
    for (lang, is_rtl, _, _, _, expected) in WORDS {
        assert!(
            text.contains(expected),
            "[{lang}] (rtl={is_rtl}) expected {expected:?} not found in extracted text: {text:?}"
        );
    }
}

/// Same corpus through `to_markdown_all` — the non-RTL scripts (which
/// share the same buffer-flush code path and aren't affected by the RTL
/// gap below) must survive the markdown conversion layer too, not just
/// plain text extraction.
///
/// Markdown output legitimately wraps RTL runs in UAX #9 bidi-isolation
/// markers (U+2066-2069, `bidi::wrap_rtl_isolates`) so viewers don't
/// re-shuffle a neutral character (like this fixture's trailing `!`)
/// across the RTL/LTR boundary — strip those before the substring check
/// so the assertion validates *content* correctness, not marker-free
/// byte-identity with `extract_text`.
///
/// RTL (Hebrew/Arabic) is intentionally excluded here: `to_markdown_all`/
/// `to_html_all` route through `PdfDocument::apply_rtl_logical_order_to_ordered_spans`
/// (`document.rs`), a *separate* RTL pass from the one this fix corrected
/// in `extractors/text.rs`. That pass still unconditionally character-reverses
/// pure-RTL spans with no render-mode gating — deliberately left as-is
/// (see the corpus-regression-sweep finding: gating it the same way regressed
/// real-world visual-order-storing Hebrew/Arabic producers, e.g. Wikipedia
/// PDF exports, which rely on that unconditional reversal being correct for
/// *their* shape). Fixing the OCR-sandwich case for md/html too needs
/// render-mode threaded onto `TextSpan`/`OrderedTextSpan`, which is a larger,
/// separate change — tracked as a follow-up, not silently dropped.
#[test]
fn ocr_sandwich_multiscript_markdown_extraction() {
    let doc = PdfDocument::from_bytes(build_ocr_sandwich_pdf()).expect("parse OCR-sandwich PDF");
    let opts = ConversionOptions::default();
    let md = doc.to_markdown_all(&opts).expect("to_markdown_all");
    let md_stripped: String = md
        .chars()
        .filter(|c| !matches!(*c, '\u{2066}'..='\u{2069}'))
        .collect();
    for (lang, is_rtl, _, _, _, expected) in WORDS {
        if *is_rtl {
            continue;
        }
        assert!(
            md_stripped.contains(expected),
            "[{lang}] (rtl={is_rtl}) expected {expected:?} not found in markdown \
             (isolation markers stripped): {md_stripped:?}"
        );
    }
}

/// Same corpus through `to_html_all` — see the markdown test above for why
/// RTL (Hebrew/Arabic) is excluded from this check.
#[test]
fn ocr_sandwich_multiscript_html_extraction() {
    let doc = PdfDocument::from_bytes(build_ocr_sandwich_pdf()).expect("parse OCR-sandwich PDF");
    let opts = ConversionOptions::default();
    let html = doc.to_html_all(&opts).expect("to_html_all");
    for (lang, is_rtl, _, _, _, expected) in WORDS {
        if *is_rtl {
            continue;
        }
        assert!(
            html.contains(expected),
            "[{lang}] (rtl={is_rtl}) expected {expected:?} not found in HTML: {html:?}"
        );
    }
}
