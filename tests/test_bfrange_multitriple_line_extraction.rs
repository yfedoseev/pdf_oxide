//! Regression: `extract_chars` / `extract_text` must resolve EVERY
//! `<src> <end> <dst>` triple of a `beginbfrange` block, even when a producer
//! emits the whole block as a single newline-free line with every triple
//! concatenated (e.g. `100 beginbfrange` with all triples on one line).
//!
//! Before the fix, `parse_bfrange_line` used `captures()` (first match only), so
//! only the FIRST triple was decoded; every later CID fell through to U+FFFD.
//! This garbled Chinese (Founder/PKU subset fonts) in `extract_chars` and sent
//! pages down the OCR fallback path. See issue #1497.
//!
//! Every fixture is built in memory (repo convention: no binary fixtures). Each
//! font is a non-embedded Type0 / Identity-H whose text layer is driven entirely
//! by its `/ToUnicode` CMap, so extraction must resolve CIDs through that CMap.

use pdf_oxide::document::PdfDocument;

/// (CID, Unicode) pairs the synthetic CMap maps. Each is a 1:1 range.
const ENTRIES: &[(u32, u32)] = &[
    (0x0001, 0x4E2D), // 中
    (0x0002, 0x4E66), // 书
    (0x0003, 0x4E00), // 一
    (0x0004, 0x534E), // 华
    (0x0005, 0x4EBA), // 人
];

/// Wrap a `beginbfrange` *body* (the triples, possibly concatenated on one line)
/// in a minimal ToUnicode CMap shell. The count is always 1 because the body is
/// a single `beginbfrange` block regardless of how many triples it holds.
fn wrap_bfrange(body: &str) -> String {
    format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         1 beginbfrange\n{body}\nendbfrange\n\
         endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    )
}

/// Emit one `<CID> Tj` operator per code, in draw order, inside a BT/ET block.
fn content_for(codes: &[u32]) -> String {
    let mut c = String::from("BT\n/F1 13 Tf\n40 770 Td\n");
    for code in codes {
        c.push_str(&format!("<{code:04X}> Tj\n"));
    }
    c.push_str("ET");
    c
}

/// Build a minimal non-embedded Type0 / Identity-H PDF whose text layer is
/// driven entirely by a `/ToUnicode` CMap (no embedded glyph outlines), so text
/// extraction must resolve CIDs through that CMap.
///
/// `content` is the page content stream, `cmap` the full ToUnicode CMap stream,
/// and `registry` / `ordering` the descendant font's `CIDSystemInfo`. Use a
/// vendor `ordering` (e.g. `"PKU2"`) to reproduce issue #1497, where the CMap is
/// the only way to recover Unicode.
fn assemble_type0_pdf(content: &str, cmap: &str, registry: &str, ordering: &str) -> Vec<u8> {
    let content_len = content.len();
    let cmap_len = cmap.len();

    let objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_string(),
        format!("<< /Length {content_len} >>\nstream\n{content}\nendstream"),
        "<< /Type /Font /Subtype /Type0 /BaseFont /CIDFont+F1 /Encoding /Identity-H \
         /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
            .to_string(),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /CIDFont+F1 \
             /CIDSystemInfo << /Registry ({registry}) /Ordering ({ordering}) /Supplement 0 >> \
             /FontDescriptor 7 0 R /DW 500 /CIDToGIDMap /Identity >>"
        ),
        "<< /Type /FontDescriptor /FontName /CIDFont+F1 /Flags 4 \
         /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
         /CapHeight 700 /StemV 80 /MissingWidth 500 >>"
            .to_string(),
        format!("<< /Length {cmap_len} >>\nstream\n{cmap}\nendstream"),
    ];

    // Byte-accurate xref table.
    let mut out: Vec<u8> = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref_off = out.len();
    let size = objs.len() + 1;
    out.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF").as_bytes(),
    );
    out
}

/// Collect the resolved code points of page 0, in draw order.
fn resolved_code_points(pdf: &[u8]) -> Vec<u32> {
    let doc = PdfDocument::from_bytes(pdf.to_vec()).expect("parse synthetic PDF");
    let chars = doc.extract_chars(0).expect("extract_chars page 0");
    chars.iter().map(|c| c.char as u32).collect()
}

// ---------------------------------------------------------------------------
// Original (1:1, concatenated) regression — kept as the baseline guard.
// ---------------------------------------------------------------------------

/// Build the original synthetic PDF: five 1:1 triples on one concatenated line.
fn build_type0_pdf_concatenated_bfrange() -> Vec<u8> {
    let triples: String = ENTRIES
        .iter()
        .map(|(cid, uni)| format!("<{cid:04X}> <{cid:04X}> <{uni:04X}>"))
        .collect();
    let cmap = wrap_bfrange(&triples);
    let codes: Vec<u32> = ENTRIES.iter().map(|(c, _)| *c).collect();
    assemble_type0_pdf(&content_for(&codes), &cmap, "Adobe", "Identity")
}

/// `extract_chars` over a Type0 font whose ToUnicode `beginbfrange` lists every
/// triple on one line must recover ALL the Chinese characters, not just the
/// first, and must emit zero U+FFFD.
#[test]
fn extract_chars_resolves_every_bfrange_triple_on_one_line() {
    let pdf = build_type0_pdf_concatenated_bfrange();
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");

    let chars = doc.extract_chars(0).expect("extract_chars page 0");
    let text: String = chars.iter().map(|c| c.char).collect();

    assert!(
        !text.contains('\u{FFFD}'),
        "extracted text contained U+FFFD (replacement char) — non-first bfrange \
         triples were dropped: {:?}",
        text
    );

    for (_, uni) in ENTRIES {
        let expected = char::from_u32(*uni).expect("valid BMP code point");
        assert!(
            text.contains(expected),
            "missing {}(U+{:04X}) in extracted text: {:?}",
            expected,
            uni,
            text
        );
    }
}

/// `extract_text` shares the same resolution path; it must also return clean
/// Chinese with no replacement characters.
#[test]
fn extract_text_resolves_every_bfrange_triple_on_one_line() {
    let pdf = build_type0_pdf_concatenated_bfrange();
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");

    let text = doc.extract_text(0).expect("extract_text page 0");

    assert!(
        !text.contains('\u{FFFD}'),
        "extracted text contained U+FFFD (replacement char): {:?}",
        text
    );
    for (_, uni) in ENTRIES {
        let expected = char::from_u32(*uni).expect("valid BMP code point");
        assert!(
            text.contains(expected),
            "missing {}(U+{:04X}) in extracted text: {:?}",
            expected,
            uni,
            text
        );
    }
}

// ---------------------------------------------------------------------------
// Extended coverage for the fix's genuinely new behaviour.
// ---------------------------------------------------------------------------

/// A `beginbfrange` emitted as one concatenated line can carry real *ranges*
/// (start != end). The fix changed range expansion to increment the dst per
/// code; verify every code in each range resolves end-to-end (not just the
/// first triple, and not all collapsed onto a single dst).
#[test]
fn extract_chars_resolves_ranges_in_concatenated_bfrange() {
    // <0001> <0003> <4E00> -> 0x4E00..0x4E02 ; <0004> <0006> <4E03> -> 0x4E03..0x4E05
    let line = "<0001> <0003> <4E00><0004> <0006> <4E03>";
    let cmap = wrap_bfrange(line);
    let codes = [0x0001u32, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006];
    let pdf = assemble_type0_pdf(&content_for(&codes), &cmap, "Adobe", "Identity");

    let got = resolved_code_points(&pdf);
    assert_eq!(got, (0x4E00u32..=0x4E05).collect::<Vec<_>>());
}

/// A concatenated `beginbfrange` line can carry supplementary-plane (5-hex) and
/// UTF-16 surrogate-pair (8-hex) destinations — rare CJK and emoji. Both go
/// through `decode_bfrange_dst` / `decode_utf16_surrogate_pair` and must surface
/// as real characters, not U+FFFD.
#[test]
fn extract_chars_resolves_supplementary_and_surrogate_in_concatenated_bfrange() {
    // 0x0001 -> U+20000 (𠀀, CJK Ext-B) ; 0x0002 -> U+1F600 (😀, emoji)
    let line = "<0001> <0001> <20000><0002> <0002> <1F600>";
    let cmap = wrap_bfrange(line);
    let codes = [0x0001u32, 0x0002];
    let pdf = assemble_type0_pdf(&content_for(&codes), &cmap, "Adobe", "Identity");

    let got = resolved_code_points(&pdf);
    assert_eq!(got, vec![0x20000u32, 0x1F600u32]);
}

/// An array entry and a sequential triple on the same concatenated line must
/// both resolve end-to-end — the masking that hides the array body from the
/// sequential regex must not drop either half.
#[test]
fn extract_chars_resolves_mixed_array_and_sequential_on_one_line() {
    // array: 0x0001..0x0003 -> 一 丁 丂 ; sequential: 0x0004 -> 'A'
    let line = "<0001> <0003> [<4E00> <4E01> <4E02>]<0004> <0004> <0041>";
    let cmap = wrap_bfrange(line);
    let codes = [0x0001u32, 0x0002, 0x0003, 0x0004];
    let pdf = assemble_type0_pdf(&content_for(&codes), &cmap, "Adobe", "Identity");

    let got = resolved_code_points(&pdf);
    assert_eq!(got, vec![0x4E00u32, 0x4E01, 0x4E02, 0x0041]);
}

/// Control: the same five code points expressed with one triple per line (the
/// "normal" producer format) must also extract cleanly. This guards against the
/// fix regressing the non-concatenated path and pairs with the concatenated
/// regression guards above (only the concatenated form fails without the fix).
#[test]
fn extract_chars_resolves_normal_multiline_bfrange() {
    let body = "<0001> <0001> <4E2D>\n<0002> <0002> <4E66>\n<0003> <0003> <4E00>\n\
                <0004> <0004> <534E>\n<0005> <0005> <4EBA>";
    let cmap = wrap_bfrange(body);
    let codes = [0x0001u32, 0x0002, 0x0003, 0x0004, 0x0005];
    let pdf = assemble_type0_pdf(&content_for(&codes), &cmap, "Adobe", "Identity");

    let got = resolved_code_points(&pdf);
    assert_eq!(got, vec![0x4E2Du32, 0x4E66, 0x4E00, 0x534E, 0x4EBA]);
}

/// Faithful reproduction of issue #1497: a Type0 / Identity-H font whose
/// `CIDSystemInfo` /Ordering is a *vendor* collection (Founder `PKU2`) — where
/// the embedded ToUnicode CMap is the only way to recover Unicode — with the
/// CMap's `beginbfrange` emitted as one concatenated line. Before the fix this
/// extracted ~94% U+FFFD; it must now extract clean CJK with no replacement
/// characters.
#[test]
fn extract_chars_resolves_vendor_ordering_concatenated_bfrange() {
    // <0001> <0003> <4E00> -> 0x4E00..0x4E02 ; <0004> <0006> <4E03> -> 0x4E03..0x4E05
    let line = "<0001> <0003> <4E00><0004> <0006> <4E03>";
    let cmap = wrap_bfrange(line);
    let codes = [0x0001u32, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006];
    let pdf = assemble_type0_pdf(&content_for(&codes), &cmap, "Founder", "PKU2");

    let got = resolved_code_points(&pdf);
    assert_eq!(got, (0x4E00u32..=0x4E05).collect::<Vec<_>>());
}
