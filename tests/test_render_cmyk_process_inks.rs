//! Render-level pin for the DeviceCMYK composite path (no OutputIntent).
//!
//! Rasterises a page that paints DeviceCMYK **vector swatches** and
//! **`k`-coloured text** through the live composite renderer
//! (`run_pipeline_for_logical` -> `cmyk_to_rgb_via_intent`), with **no**
//! `/OutputIntents` profile declared, so the conversion takes the
//! process-ink fallback (`crate::color::cmyk_to_rgb`) rather than the
//! ISO 32000-1:2008 §10.3.5 additive clamp `R = 1 - min(1, C+K)`.
//!
//! This is the end-to-end gap that the extraction/image-path unification
//! left open: the vector/text composite render used to stay on the
//! additive clamp, so a `0 0 0 1 k` fill rendered pure black `(0,0,0)`
//! instead of the K-ink `#231F20`. These assertions fail loudly on the
//! additive path and pass only when the composite render is on process
//! inks.

#![cfg(feature = "rendering")]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

/// Build a 100x100 pt page with two DeviceCMYK vector swatches and a
/// `k`-coloured text run, all via the `k` (fill CMYK) operator:
///   - cyan swatch  `1 0 0 0 k` at pdf (10,60)-(40,90)   -> image centre (25,25)
///   - K swatch     `0 0 0 1 k` at pdf (60,60)-(90,90)   -> image centre (75,25)
///   - K text       `0 0 0 1 k (FW) Tj` baseline pdf (15,30) -> image rows ~53..70
fn build_cmyk_swatch_and_text_pdf() -> Vec<u8> {
    let content = "1 0 0 0 k\n\
                   10 60 30 30 re f\n\
                   0 0 0 1 k\n\
                   60 60 30 30 re f\n\
                   BT\n/F1 24 Tf\n0 0 0 1 k\n15 30 Td\n(FW) Tj\nET\n";
    let content_bytes = content.as_bytes();

    let mut buf = Vec::new();
    let mut offsets = Vec::new();

    buf.extend_from_slice(b"%PDF-1.4\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(buf.len());
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(buf.len());
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    );

    offsets.push(buf.len());
    let stream_header = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_bytes.len());
    buf.extend_from_slice(stream_header.as_bytes());
    buf.extend_from_slice(content_bytes);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(buf.len());
    buf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    let xref_offset = buf.len();
    buf.extend_from_slice(b"xref\n");
    buf.extend_from_slice(format!("0 {}\n", offsets.len() + 1).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len() + 1,
            xref_offset
        )
        .as_bytes(),
    );

    buf
}

fn render_rgba_100(doc: &PdfDocument) -> Vec<u8> {
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(doc, 0, &opts).expect("render_page");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    assert_eq!(img.data.len(), 100 * 100 * 4, "expected a 100x100 RGBA raster");
    img.data
}

fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let off = ((y * 100 + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

/// `|a - b| <= tol` on each channel.
fn near(got: (u8, u8, u8), want: (u8, u8, u8), tol: i32) -> bool {
    let d = |a: u8, b: u8| (a as i32 - b as i32).abs();
    d(got.0, want.0) <= tol && d(got.1, want.1) <= tol && d(got.2, want.2) <= tol
}

#[test]
fn device_cmyk_vector_and_text_render_via_process_inks_not_additive_clamp() {
    let pdf = build_cmyk_swatch_and_text_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse PDF");
    let rgba = render_rgba_100(&doc);

    // --- Cyan vector swatch: 100% cyan is the measured corner #00ADEF
    //     = (0, 173, 239), NOT the additive-clamp (0, 255, 255). ---
    let cyan = pixel_at(&rgba, 25, 25);
    assert_eq!(cyan.3, 255, "cyan swatch centre should be opaque");
    assert!(
        near((cyan.0, cyan.1, cyan.2), (0, 173, 239), 3),
        "cyan swatch should be process-ink #00ADEF, got {cyan:?}"
    );
    assert_ne!(
        (cyan.0, cyan.1, cyan.2),
        (0, 255, 255),
        "cyan swatch must NOT be the additive-clamp (0,255,255)"
    );

    // --- K vector swatch: 100% K is the measured corner #231F20
    //     = (35, 31, 32), NOT the additive-clamp pure black (0, 0, 0). ---
    let k = pixel_at(&rgba, 75, 25);
    assert_eq!(k.3, 255, "K swatch centre should be opaque");
    assert!(
        near((k.0, k.1, k.2), (35, 31, 32), 3),
        "K swatch should be process-ink #231F20, got {k:?}"
    );
    assert_ne!(
        (k.0, k.1, k.2),
        (0, 0, 0),
        "K swatch must NOT be additive-clamp pure black (0,0,0)"
    );

    // --- K-coloured text: the fully-inked interior of a glyph stroke must
    //     land on the K-ink #231F20, not pure black. Scan the text band
    //     for the darkest pixel and require it to match the K-ink. ---
    let mut darkest: Option<(u8, u8, u8)> = None;
    for y in 48..74 {
        for x in 12..58 {
            let (r, g, b, a) = pixel_at(&rgba, x, y);
            if a == 0 {
                continue;
            }
            let sum = r as u32 + g as u32 + b as u32;
            if darkest.is_none_or(|(dr, dg, db)| sum < dr as u32 + dg as u32 + db as u32) {
                darkest = Some((r, g, b));
            }
        }
    }
    let darkest = darkest.expect("expected inked text pixels in the text band");
    assert!(
        near(darkest, (35, 31, 32), 8),
        "darkest K-text pixel should be process-ink #231F20, got {darkest:?} \
         (additive clamp would paint it pure black)"
    );

    // --- Global discriminator: with the process-ink composite live, no
    //     mark on the page is pure black. The additive clamp would fill
    //     both the K swatch and the K text with (0,0,0) pixels. ---
    let pure_black = rgba
        .chunks_exact(4)
        .filter(|px| px[3] > 0 && px[0] == 0 && px[1] == 0 && px[2] == 0)
        .count();
    assert_eq!(
        pure_black, 0,
        "process-ink composite must not emit any pure-black (0,0,0) pixel; found {pure_black}"
    );
}
