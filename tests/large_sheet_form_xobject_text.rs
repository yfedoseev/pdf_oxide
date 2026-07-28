//! End-to-end: text that lives inside Form XObjects must survive span
//! extraction on a large-format drawing sheet whose page stream is dominated
//! by vector linework.
//!
//! CAD / construction plan sheets draw their schedules, general notes and
//! title block as re-used Form XObject blocks (`Do`), with tens of thousands
//! of path segments in the page stream. Two properties of that shape combine:
//!
//!   1. the page stream is far larger than the 256 KB threshold above which
//!      span extraction pre-scans for text-bearing regions instead of parsing
//!      the whole stream, and
//!   2. `Do` invocations outnumber page-level `BT` blocks by more than 10:1.
//!
//! The pre-scan drops every `Do` position under (2) - a heuristic aimed at
//! chart/figure graphics - so on such a sheet it reports NO text regions at
//! all and `extract_spans` returns nothing, while `extract_chars` (which does
//! not pre-scan) returns the full glyph set for the same page. The identical
//! sheet under the 256 KB threshold extracts correctly, which is what makes
//! this a pre-scan defect and not a font/encoding one.

use pdf_oxide::PdfDocument;

/// A large-format sheet: `n_forms` text-bearing Form XObjects invoked from a
/// page stream padded with `pad` bytes of vector linework, plus `n_page_bt`
/// page-level text blocks.
fn drawing_sheet(n_forms: usize, n_page_bt: usize, pad: usize) -> Vec<u8> {
    const W: u32 = 2592;
    const H: u32 = 1728;
    // Object ids: 1 catalog, 2 pages, 3 page, 4 contents, 5 font, 6.. forms.
    const FIRST_FORM: usize = 6;

    let mut content = String::new();
    let mut n = 0usize;
    while content.len() < pad {
        let x = (n * 7) % W as usize;
        let y = (n * 13) % H as usize;
        content.push_str(&format!("{x} {y} m {} {} l S\n", x + 5, y + 9));
        n += 1;
    }
    for i in 0..n_page_bt {
        content.push_str(&format!(
            "BT /F1 14 Tf 100 {} Td (PAGE LEVEL TITLE {i:02}) Tj ET\n",
            1600 - i * 20
        ));
    }
    for i in 0..n_forms {
        let x = 100 + (i % 5) * 450;
        let y = 1500 - (i / 5) * 60;
        content.push_str(&format!("q 1 0 0 1 {x} {y} cm /Fm{i} Do Q\n"));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; FIRST_FORM + n_forms];
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id - 1] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream_obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, s: &str| {
        off[id - 1] = buf.len();
        buf.extend_from_slice(
            format!(
                "{id} 0 obj\n<< {dict} /Length {} >>\nstream\n{s}\nendstream\nendobj\n",
                s.len()
            )
            .as_bytes(),
        );
    };

    let xobjects: String = (0..n_forms)
        .map(|i| format!("/Fm{i} {} 0 R", FIRST_FORM + i))
        .collect::<Vec<_>>()
        .join(" ");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Font << /F1 5 0 R >> /XObject << {xobjects} >> >> \
             /Contents 4 0 R >>"
        ),
    );
    stream_obj(&mut buf, &mut off, 4, "", &content);
    obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    for i in 0..n_forms {
        stream_obj(
            &mut buf,
            &mut off,
            FIRST_FORM + i,
            "/Type /XObject /Subtype /Form /BBox [0 0 400 40] \
             /Resources << /Font << /F1 5 0 R >> >>",
            &format!("BT /F1 12 Tf 10 10 Td (NOTE {i:03} GENERAL NOTES) Tj ET"),
        );
    }

    let count = FIRST_FORM + n_forms; // ids 1..=count-1 are in use
    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {count}\n0000000000 65535 f \n").as_bytes());
    for o in off.iter().take(count - 1) {
        buf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    buf
}

/// Glyphs the page really carries, per the character layer.
fn char_count(doc: &PdfDocument) -> usize {
    doc.extract_chars(0)
        .expect("extract_chars")
        .iter()
        .filter(|c| !c.char.is_whitespace())
        .count()
}

#[test]
fn form_xobject_text_survives_on_a_linework_heavy_sheet() {
    let big = drawing_sheet(60, 0, 300_000);
    let doc = PdfDocument::from_bytes(big).expect("parse");

    let chars = char_count(&doc);
    let spans = doc.extract_spans(0).expect("extract_spans");
    let text = doc.extract_text(0).expect("extract_text");

    assert!(chars > 1_000, "fixture should carry glyphs, got {chars}");
    assert!(
        !spans.is_empty(),
        "page stream >256 KB with 60 text-bearing Do and no page-level BT \
         produced {} spans against {chars} chars",
        spans.len()
    );
    assert!(
        text.contains("NOTE 000") && text.contains("NOTE 059"),
        "text from Form XObjects missing; extracted {} chars",
        text.trim().len()
    );
}

#[test]
fn linework_padding_alone_must_not_change_extraction() {
    // Same sheet, once under and once over the pre-scan threshold. The padding
    // is pure `m`/`l`/`S` linework, so both must extract the same text.
    let small = PdfDocument::from_bytes(drawing_sheet(60, 0, 1_000)).expect("parse small");
    let big = PdfDocument::from_bytes(drawing_sheet(60, 0, 300_000)).expect("parse big");

    let small_text = small.extract_text(0).expect("small text");
    let big_text = big.extract_text(0).expect("big text");

    assert_eq!(
        small_text.split_whitespace().collect::<Vec<_>>(),
        big_text.split_whitespace().collect::<Vec<_>>(),
        "crossing the pre-scan size threshold changed the extracted text"
    );
}

#[test]
fn page_level_text_does_not_mask_lost_form_text() {
    // A handful of page-level BT blocks alongside many Do invocations: the
    // partial-loss shape, where a sheet returns its title block but drops
    // every schedule/note drawn as a block.
    let doc = PdfDocument::from_bytes(drawing_sheet(60, 5, 300_000)).expect("parse");
    let chars = char_count(&doc);
    let text = doc.extract_text(0).expect("extract_text");
    let recovered = text.chars().filter(|c| !c.is_whitespace()).count();

    assert!(
        recovered * 2 >= chars,
        "recovered {recovered} of {chars} chars ({}%)",
        recovered * 100 / chars.max(1)
    );
}
