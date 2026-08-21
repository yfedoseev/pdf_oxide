//! A dropped glyph must surface a warning, on every page it happens.
//!
//! The fixture embeds a TrueType font whose only cmap subtable is
//! byte-indexed (Macintosh Roman, format 0) and maps just 'A'. Rendering
//! "AB" paints the 'A' box and drops the 'B': its code resolves to glyph 0,
//! which is non-whitespace and never painted, while the cursor still
//! advances. That gap is indistinguishable from real whitespace downstream,
//! so the renderer must push a `GlyphDropped` warning.
//!
//! The warning latch is page-scoped (the `k_zero_warning_emitted` pattern):
//! one warning per font per page, and the next page warns again. A
//! process-lifetime latch fails this test, because bulk ingestion would go
//! silent after the first page.

#![cfg(feature = "rendering")]

use pdf_oxide::extractors::warnings::{drain_global_warnings, WarningCategory};
use pdf_oxide::rendering::{PageRenderer, RenderOptions};
use pdf_oxide::PdfDocument;

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// A minimal TrueType font: glyph 0 empty, glyph 1 a box, and a single
/// byte-indexed cmap subtable (platform 1, encoding 0, format 0) that maps
/// only 'A' (0x41) to glyph 1. Every other byte resolves to glyph 0.
fn broken_subset_font() -> Vec<u8> {
    // glyf: glyph 0 is empty (zero bytes). Glyph 1 is one square contour,
    // four on-curve points, 16-bit deltas.
    let mut glyf: Vec<u8> = Vec::new();
    glyf.extend(be16(1)); // numberOfContours
    glyf.extend(be16(50)); // xMin
    glyf.extend(be16(0)); // yMin
    glyf.extend(be16(450)); // xMax
    glyf.extend(be16(700)); // yMax
    glyf.extend(be16(3)); // endPtsOfContours[0]
    glyf.extend(be16(0)); // instructionLength
    glyf.extend([0x01, 0x01, 0x01, 0x01]); // flags: on-curve, long deltas
    for dx in [50i16, 400, 0, -400] {
        glyf.extend(dx.to_be_bytes());
    }
    for dy in [0i16, 0, 700, 0] {
        glyf.extend(dy.to_be_bytes());
    }

    // loca (short format, offset/2): glyph 0 empty, glyph 1 = all of glyf.
    let mut loca: Vec<u8> = Vec::new();
    loca.extend(be16(0));
    loca.extend(be16(0));
    loca.extend(be16((glyf.len() / 2) as u16));

    let mut head: Vec<u8> = Vec::new();
    head.extend(be32(0x0001_0000)); // version
    head.extend(be32(0)); // fontRevision
    head.extend(be32(0)); // checkSumAdjustment
    head.extend(be32(0x5F0F_3CF5)); // magicNumber
    head.extend(be16(0)); // flags
    head.extend(be16(1000)); // unitsPerEm
    head.extend([0u8; 16]); // created + modified
    head.extend(be16(0)); // xMin
    head.extend(be16(0)); // yMin
    head.extend(be16(450)); // xMax
    head.extend(be16(700)); // yMax
    head.extend(be16(0)); // macStyle
    head.extend(be16(8)); // lowestRecPPEM
    head.extend(be16(2)); // fontDirectionHint
    head.extend(be16(0)); // indexToLocFormat: short
    head.extend(be16(0)); // glyphDataFormat

    let mut hhea: Vec<u8> = Vec::new();
    hhea.extend(be32(0x0001_0000)); // version
    hhea.extend(be16(800)); // ascender
    hhea.extend((-200i16).to_be_bytes()); // descender
    hhea.extend(be16(0)); // lineGap
    hhea.extend(be16(500)); // advanceWidthMax
    hhea.extend(be16(0)); // minLeftSideBearing
    hhea.extend(be16(0)); // minRightSideBearing
    hhea.extend(be16(450)); // xMaxExtent
    hhea.extend(be16(1)); // caretSlopeRise
    hhea.extend(be16(0)); // caretSlopeRun
    hhea.extend(be16(0)); // caretOffset
    hhea.extend([0u8; 8]); // reserved
    hhea.extend(be16(0)); // metricDataFormat
    hhea.extend(be16(2)); // numberOfHMetrics

    let mut hmtx: Vec<u8> = Vec::new();
    for _ in 0..2 {
        hmtx.extend(be16(500)); // advanceWidth
        hmtx.extend(be16(0)); // leftSideBearing
    }

    let mut maxp: Vec<u8> = Vec::new();
    maxp.extend(be32(0x0001_0000)); // version
    maxp.extend(be16(2)); // numGlyphs
    maxp.extend([0u8; 26]); // limits, unused by the parser

    // cmap: version 0, one encoding record → format 0 subtable mapping
    // only 0x41 → glyph 1.
    let mut cmap: Vec<u8> = Vec::new();
    cmap.extend(be16(0)); // version
    cmap.extend(be16(1)); // numTables
    cmap.extend(be16(1)); // platformID: Macintosh
    cmap.extend(be16(0)); // encodingID: Roman
    cmap.extend(be32(12)); // subtable offset
    cmap.extend(be16(0)); // format 0
    cmap.extend(be16(262)); // length
    cmap.extend(be16(0)); // language
    let mut glyph_ids = [0u8; 256];
    glyph_ids[0x41] = 1;
    cmap.extend(glyph_ids);

    // Assemble the sfnt: table records sorted by tag.
    let tables: [(&[u8; 4], &Vec<u8>); 7] = [
        (b"cmap", &cmap),
        (b"glyf", &glyf),
        (b"head", &head),
        (b"hhea", &hhea),
        (b"hmtx", &hmtx),
        (b"loca", &loca),
        (b"maxp", &maxp),
    ];
    let num_tables = tables.len() as u16;
    let mut font: Vec<u8> = Vec::new();
    font.extend(be32(0x0001_0000)); // sfnt version
    font.extend(be16(num_tables));
    let entry_selector = 15 - num_tables.leading_zeros() as u16; // floor(log2 n)
    let search_range = (1u16 << entry_selector) * 16;
    font.extend(be16(search_range));
    font.extend(be16(entry_selector));
    font.extend(be16(num_tables * 16 - search_range));

    let mut offset = 12 + 16 * tables.len();
    for (tag, data) in &tables {
        font.extend_from_slice(*tag);
        font.extend(be32(0)); // checksum: not verified by the parser
        font.extend(be32(offset as u32));
        font.extend(be32(data.len() as u32));
        offset += data.len().div_ceil(4) * 4;
    }
    for (_, data) in &tables {
        font.extend_from_slice(data);
        font.extend(std::iter::repeat_n(0u8, data.len().div_ceil(4) * 4 - data.len()));
    }
    font
}

/// A PDF with `page_count` identical pages, each painting "AB" in the
/// broken embedded font. 'A' paints; 'B' drops.
fn pdf_with_broken_font(page_count: usize) -> Vec<u8> {
    let font = broken_subset_font();
    let content = b"BT /F1 24 Tf 50 100 Td (AB) Tj ET";
    let n_objs = 5 + 2 * page_count; // catalog, pages, font, descriptor, fontfile + per page: page, contents

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; n_objs + 1];
    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: String| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    let kids: Vec<String> = (0..page_count)
        .map(|p| format!("{} 0 R", 6 + 2 * p))
        .collect();
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>".to_string());
    obj(
        &mut buf,
        &mut off,
        2,
        format!("<< /Type /Pages /Kids [{}] /Count {page_count} >>", kids.join(" ")),
    );
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /TrueType /BaseFont /BrokenSubset /FirstChar 65 /LastChar 66 \
         /Widths [500 500] /FontDescriptor 4 0 R >>"
            .to_string(),
    );
    obj(
        &mut buf,
        &mut off,
        4,
        "<< /Type /FontDescriptor /FontName /BrokenSubset /Flags 4 /FontBBox [0 0 450 700] \
         /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 /FontFile2 5 0 R >>"
            .to_string(),
    );
    off[5] = buf.len();
    buf.extend_from_slice(
        format!("5 0 obj\n<< /Length {} /Length1 {} >>\nstream\n", font.len(), font.len())
            .as_bytes(),
    );
    buf.extend_from_slice(&font);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    for p in 0..page_count {
        let page_id = 6 + 2 * p;
        let contents_id = page_id + 1;
        obj(
            &mut buf,
            &mut off,
            page_id,
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {contents_id} 0 R >>"
            ),
        );
        off[contents_id] = buf.len();
        buf.extend_from_slice(
            format!("{contents_id} 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes(),
        );
        buf.extend_from_slice(content);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", n_objs + 1).as_bytes());
    for id in 1..=n_objs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", n_objs + 1)
            .as_bytes(),
    );
    buf
}

/// Glyph-drop warnings naming our fixture font, drained from the global sink.
fn drain_broken_subset_warnings() -> Vec<String> {
    drain_global_warnings()
        .into_iter()
        .filter(|w| w.category == WarningCategory::GlyphDropped)
        .map(|w| w.message)
        .filter(|m| m.contains("BrokenSubset"))
        .collect()
}

#[test]
fn a_dropped_glyph_warns_on_every_page_it_paints() {
    let doc = PdfDocument::from_bytes(pdf_with_broken_font(2)).expect("parse fixture");
    let mut renderer = PageRenderer::new(RenderOptions::default());
    let _ = drain_broken_subset_warnings();

    renderer.render_page(&doc, 0).expect("render page 0");
    let first = drain_broken_subset_warnings();
    assert_eq!(first.len(), 1, "page 0 must warn exactly once for the broken font: {first:?}");
    assert!(
        first[0].contains("0x42"),
        "the warning must name the dropped code 0x42: {}",
        first[0]
    );

    renderer.render_page(&doc, 1).expect("render page 1");
    let second = drain_broken_subset_warnings();
    assert_eq!(
        second.len(),
        1,
        "page 1 must warn again — the latch is page-scoped, not process-scoped: {second:?}"
    );
}
