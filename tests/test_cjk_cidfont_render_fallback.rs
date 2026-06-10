//! Regression: a composite (Type 0) font that references a CJK glyph
//! collection but embeds NO glyph program must still paint glyphs when the
//! page is rasterised, rather than coming out blank.
//!
//! Many real-world Japanese/Chinese/Korean PDFs reference Adobe's predefined
//! CIDFonts (Ryumin-Light, GothicBBB-Medium, STSong-Light, …) without bundling
//! outlines, because Acrobat historically shipped those faces. ISO 32000-2:2020
//! §9.7.5.2 requires a processor to support the Adobe-CNS1/GB1/Japan1/Korea1
//! character collections; the renderer must therefore substitute a covering
//! font instead of silently dropping every glyph. With the `cjk-render-fallback`
//! feature the bundled Droid Sans Fallback is registered in the font database
//! as the guaranteed last-resort CJK face, so coverage no longer depends on
//! whichever fonts happen to be installed on the host (the original failure was
//! observed on CJK-fontless machines).
//!
//! The fixture is 100% synthetic — no third-party file, no embedded glyph
//! program. Decoding is driven by the `/ToUnicode` CMap (so extraction already
//! works); this test exercises the *paint* path specifically.
#![cfg(all(feature = "rendering", feature = "cjk-render-fallback"))]

use pdf_oxide::rendering::{PageRenderer, RenderOptions};
use pdf_oxide::PdfDocument;

/// Two CJK code points covered by the bundled fallback: 東 (U+6771, JP/ZH) and
/// 中 (U+4E2D, ZH). CIDs are arbitrary (Identity-H: code == CID == GID).
const GLYPHS: &[(u16, char)] = &[(1, '東'), (2, '中')];

/// Build a single small page that shows the `GLYPHS` with a Type0 /
/// CIDFontType2 / Identity-H font carrying a `/ToUnicode` CMap and NO embedded
/// `FontFile`. BaseFont mimics a predefined Adobe CIDFont name.
fn build_cjk_no_embed_pdf() -> Vec<u8> {
    let size = 100.0f32;
    let dw_units = 1000u32;

    let mut content = format!("BT\n/F1 {size} Tf\n1 0 0 1 20 60 Tm\n");
    for (cid, _) in GLYPHS {
        content.push_str(&format!("<{cid:04X}> Tj\n{size:.1} 0 Td\n"));
    }
    content.push_str("ET\n");
    let content_b = content.into_bytes();

    let bf: String = GLYPHS
        .iter()
        .map(|(cid, ch)| format!("<{cid:04X}> <{:04X}>", *ch as u32))
        .collect::<Vec<_>>()
        .join("\n");
    let cmap = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{bf}\nendbfchar\nendcmap\nend\nend",
        GLYPHS.len()
    );
    let cmap_b = cmap.into_bytes();

    let basefont = "STSong-Light";
    let objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 200] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_string(),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content_b.len(),
            String::from_utf8_lossy(&content_b)
        ),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{basefont} /Encoding /Identity-H \
             /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
        ),
        // CIDFontType2 descendant with a FontDescriptor that has NO FontFile* —
        // the crux of the bug: no outlines anywhere in the document.
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{basefont} \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 7 >> \
             /FontDescriptor 7 0 R /DW {dw_units} /CIDToGIDMap /Identity >>"
        ),
        format!(
            "<< /Type /FontDescriptor /FontName /{basefont} /Flags 6 \
             /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 \
             /CapHeight 700 /StemV 80 >>"
        ),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            cmap_b.len(),
            String::from_utf8_lossy(&cmap_b)
        ),
    ];

    let mut out: Vec<u8> = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF",
            objs.len() + 1
        )
        .as_bytes(),
    );
    out
}

/// Count pixels darker than near-white in a raw premultiplied-RGBA8 buffer.
fn dark_pixel_count(data: &[u8]) -> usize {
    data.chunks_exact(4)
        .filter(|px| px[0] < 200 || px[1] < 200 || px[2] < 200)
        .count()
}

#[test]
fn cjk_cidfont_without_embedded_outlines_renders_glyphs() {
    let pdf = build_cjk_no_embed_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse pdf");

    let opts = RenderOptions::with_dpi(150).as_raw();
    let mut renderer = PageRenderer::new(opts);
    let img = renderer.render_page(&doc, 0).expect("render page");

    assert_eq!(img.data.len(), (img.width * img.height * 4) as usize, "raw RGBA8 buffer");

    // Two large CJK glyphs at 150 DPI cover thousands of pixels. A blank page
    // (the bug) would be ~0. Require a generous floor well above antialiasing
    // noise but far below the real glyph coverage.
    let dark = dark_pixel_count(&img.data);
    assert!(
        dark > 500,
        "expected painted CJK glyphs (>500 dark px), got {dark} — \
         composite font with no embedded outlines rendered blank"
    );
}
