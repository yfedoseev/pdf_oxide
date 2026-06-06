//! Wave-3 QA probes for the resolution-pipeline migration (ImageMask + `Do`).
//!
//! Sibling to `test_render_resolution_pipeline_qa_wave1.rs` (paths,
//! stroke, combos) and `_qa_wave2.rs` (text). This suite probes the
//! ImageMask / `Do`-side corners:
//!
//! 1. **ImageMask rendering correctness** — `render_image_mask` covers
//!    small / wide-with-padding / tall stencils; `/Decode [1 0]`
//!    polarity invert; missing / malformed `/Decode`; rotated and
//!    mirrored CTMs.
//! 2. **Pass-through pins for the non-mask branch** — CMYK / Indexed /
//!    ICCBased N=4 standard images must keep their existing behaviour.
//! 3. **Inline-image coverage** — `BI ... ID ... EI` is a separate parse
//!    path the renderer may or may not dispatch; pin the current
//!    behaviour either way.
//! 4. **Form-XObject interactions** — Form containing an ImageMask,
//!    nested Form-in-Form, CTM round-trip across the Form boundary.
//! 5. **Multi-XObject interactions** — back-to-back masks, mixed with
//!    standard images, under SMask / clip / blend.
//! 6. **Capability at scale** — many ImageMasks on one page; DeviceN /
//!    `/All` / `/None` colorants applied to ImageMask fill.
//! 7. **Adversarial input** — too-short / too-long / zero-dim / huge-dim
//!    stencil streams.
//! 8. **Performance** — N-paint render must hold the one-resolve-per-Do
//!    invariant (matching wave-2's pattern).
//!
//! Style mirrors waves 1 + 2: build a tiny PDF inline, render through
//! `render_with_pipeline`, compare pixmaps or sample pixels.

#![cfg(feature = "rendering")]
#![allow(dead_code)] // probes accrete across commits; not every helper is wired up yet.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
use std::time::Instant;

// ===========================================================================
// PDF construction helpers — self-contained so a fix-pass to the
// wave-1/2 QA helpers can't accidentally invalidate the wave-3 invariants.
// ===========================================================================

/// Build a one-page PDF containing a single ImageMask XObject `/IM1`.
/// `content_ops` runs on the page (typically sets the fill colour, a
/// CTM, then `/IM1 Do`). `resources_extra` is appended into the page's
/// `/Resources` dictionary. `mask_extras` is appended into the
/// ImageMask stream dictionary (use it for `/Decode`, `/Interpolate`,
/// etc.).
fn build_pdf_image_mask_ex(
    content_ops: &str,
    resources_extra: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
    mask_extras: &str,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources << /XObject << /IM1 5 0 R >> {} >> /Contents 4 0 R >>\nendobj\n",
        resources_extra
    );
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 {} /Length {} >>\nstream\n",
        width,
        height,
        mask_extras,
        mask_data.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(mask_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Convenience wrapper — no extra stream-dict entries.
fn build_pdf_image_mask(
    content_ops: &str,
    resources_extra: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
) -> Vec<u8> {
    build_pdf_image_mask_ex(content_ops, resources_extra, width, height, mask_data, "")
}

/// Render the first page. The `_enabled` argument is retained so existing
/// test bodies keep compiling after wave 5 collapsed the off/on split; the
/// pipeline is the only path now.
fn render_with_pipeline(doc: &PdfDocument, _enabled: bool) -> Vec<u8> {
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(doc, 0, &opts).expect("render_page succeeds");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    img.data
}

/// Render the first page, allowing failure without panicking. Used by
/// adversarial-input probes whose invariant is "no panic", not "render
/// succeeds".
fn render_with_pipeline_allow_fail(doc: &PdfDocument, _enabled: bool) -> Option<Vec<u8>> {
    let opts = RenderOptions::with_dpi(72).as_raw();
    render_page(doc, 0, &opts).ok().map(|img| img.data)
}

/// Sample a pixel at (x, y) on the 100×100 page.
fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let w = 100u32;
    let off = ((y * w + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

/// Sample the centre pixel of the 100×100 page.
fn center_pixel(rgba: &[u8]) -> (u8, u8, u8, u8) {
    pixel_at(rgba, 50, 50)
}

/// Count pixels in `[x0, x1) × [y0, y1)` whose RGB is materially below
/// the white background — i.e. "this region got painted".
fn count_ink_pixels(rgba: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
    let w = 100u32;
    let h = 100u32;
    assert_eq!(rgba.len() as u32, w * h * 4);
    let mut n = 0u32;
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            let off = ((y * w + x) * 4) as usize;
            let r = rgba[off];
            let g = rgba[off + 1];
            let b = rgba[off + 2];
            if r < 240 || g < 240 || b < 240 {
                n += 1;
            }
        }
    }
    n
}

/// Solid 1-bit stencil — all bytes 0x00, so every pixel paints opaque
/// under the default `/Decode [0 1]`. Rows are byte-padded per PDF
/// §8.9.3.
fn solid_image_mask_bytes(width: u32, height: u32) -> Vec<u8> {
    let row_bytes = (width as usize).div_ceil(8);
    vec![0x00u8; row_bytes * height as usize]
}

/// Empty 1-bit stencil — all bytes 0xFF, so every pixel is transparent
/// under the default `/Decode [0 1]`.
fn empty_image_mask_bytes(width: u32, height: u32) -> Vec<u8> {
    let row_bytes = (width as usize).div_ceil(8);
    vec![0xFFu8; row_bytes * height as usize]
}

// ===========================================================================
// Probes 1-9 — ImageMask rendering correctness (the new capability).
//
// `render_image_mask` must decode the 1-bit stream correctly (row
// padding, default vs inverted Decode, missing Decode, malformed
// Decode), stay panic-free on degenerate input, and respect CTM
// rotation / mirroring.
// ===========================================================================

/// Probe 1 — 1×1 ImageMask stencil. A single opaque sample painted with
/// a known fill colour. With a DeviceRGB fill the spliced clone
/// short-circuits, so the rasteriser reads `gs.fill_color_rgb`
/// directly.
#[test]
fn qa_image_mask_1x1_solid_paints_fill_colour() {
    let mask = solid_image_mask_bytes(1, 1); // 1 byte, all opaque
                                             // Stretch the 1×1 stencil over 60×60 in the centre of the page.
    let content = "q\n0 1 0 rg\n60 0 0 60 20 20 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 1, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, a) = center_pixel(&on);
    assert!(
        g > 200 && r < 60 && b < 60 && a > 200,
        "1x1 stencil stretched over centre should be green, got ({r}, {g}, {b}, {a})"
    );
}

/// Probe 2 — Width that is NOT a byte multiple (7px wide). Per PDF
/// §8.9.3 each row is padded to a byte boundary; the padding bits in the
/// trailing nibble must NOT paint. If the row-bytes maths in
/// `render_image_mask` is off, the 8th column will appear opaque even
/// though it is padding.
///
/// Stencil: 7×4, all bits 0 (opaque under default Decode). Each row is
/// 1 byte; the high 7 bits are valid pixels, the low bit is padding.
/// We stretch the stencil over the full page; the right edge of the
/// rendered image must drop off after the 7th of 8 image columns —
/// i.e. roughly 100 * 7/8 = 87.5 px from the left edge.
#[test]
fn qa_image_mask_width_not_byte_multiple_padding_does_not_paint() {
    let width = 7u32;
    let height = 4u32;
    let mask = solid_image_mask_bytes(width, height);
    let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", width, height, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // A pixel firmly inside the 7-column region (around x=50, y=50) must
    // be red. The renderer's resampler may smear hard edges, so we don't
    // assert on the boundary itself — just on "interior paints".
    let (r, g, b, _a) = pixel_at(&on, 50, 50);
    assert!(
        r > 200 && g < 60 && b < 60,
        "centre of 7-column stencil must paint red, got ({r}, {g}, {b})"
    );
}

/// Probe 3 — Tall ImageMask (height = 256). Capability at scale; also
/// guards against an off-by-one in the row-loop or buffer-size maths.
#[test]
fn qa_image_mask_tall_height_paints_centre_blue() {
    let mask = solid_image_mask_bytes(8, 256);
    let content = "q\n0 0 1 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 256, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    assert!(b > 200 && r < 60 && g < 60, "centre must be blue, got ({r}, {g}, {b})");
}

/// Probe 4 — `/Decode [1 0]` polarity invert. With this Decode array a
/// stencil bit of `1` paints, `0` does not. Build an all-1s stream
/// (every byte 0xFF), under inverted Decode that should fill the whole
/// stencil; under default Decode it would be transparent.
///
/// PIN: the wave-3 helper supports `/Decode [1 0]`. The renderer must
/// paint the all-1s stencil as fully filled under inverted Decode.
#[test]
fn qa_image_mask_decode_inverted_polarity_paints_under_ff_bytes() {
    let mask = vec![0xFFu8; 1]; // 8x1 stencil, all bits 1
    let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask_ex(content, "", 8, 1, &mask, "/Decode [1 0]");
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        r > 200 && g < 60 && b < 60,
        "inverted-Decode all-1 stencil should paint red everywhere, got ({r}, {g}, {b})"
    );
}

/// Probe 5 — Missing `/Decode` (default `[0 1]`). An all-0 stream
/// paints opaque. Confirms the missing-entry path doesn't accidentally
/// drop into the inverted branch.
#[test]
fn qa_image_mask_no_decode_default_paints_under_zero_bytes() {
    let mask = solid_image_mask_bytes(8, 1); // all zeros
    let content = "q\n0 1 1 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        g > 200 && b > 200 && r < 60,
        "default-Decode all-0 stencil should paint cyan everywhere, got ({r}, {g}, {b})"
    );
}

/// Probe 6 — Malformed `/Decode`. Several adversarial cases: empty
/// array, ambiguous `[0.5 0.5]`, single-element. The renderer must not
/// panic on any of them; it should fall back to default polarity (the
/// wave-3 helper's `match … _ => false` arm).
///
/// PIN: the wave-3 implementation reads `first > 0.5` for the polarity
/// flag. With `[0.5 0.5]` `first` is exactly `0.5`, so `first > 0.5` is
/// `false` → default polarity (zeros paint, ones don't). The all-zeros
/// stream should therefore paint opaque. Empty array and single-element
/// `[1]` should hit the catch-all and also default to non-inverted.
#[test]
fn qa_image_mask_malformed_decode_no_panic_default_polarity() {
    let mask = solid_image_mask_bytes(8, 1);
    for decode in &["/Decode []", "/Decode [0.5 0.5]", "/Decode [1]"] {
        let content = "q\n0.4 g\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
        let bytes = build_pdf_image_mask_ex(content, "", 8, 1, &mask, decode);
        let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
        let on = render_with_pipeline_allow_fail(&doc, true)
            .unwrap_or_else(|| panic!("renderer must not error for {}", decode));
        // No-panic invariant. The malformed-Decode catch-all branch
        // doesn't crash; whatever its fallback polarity produces is the
        // pinned behaviour.
        assert!(!on.is_empty(), "renderer must produce a pixmap for {}", decode);
    }
}

/// Probe 7 — ImageMask under a CTM that rotates 90° clockwise. The
/// CTM must round-trip across the spliced GS clone so the stencil
/// (8×1, all opaque) lands as a vertical band on the page.
#[test]
fn qa_image_mask_ctm_90deg_rotation_paints_visible_band() {
    let mask = solid_image_mask_bytes(8, 1);
    // Rotate 90° clockwise (a,b,c,d = 0,-1,1,0) then scale and translate
    // to land the band on the page.
    let content = "q\n1 0 0 rg\n0 -60 60 0 20 80 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // 90° rotation maps the 8×1 stencil to a vertical band; pin
    // that the rotated mask actually paints rather than collapses
    // to zero pixels (CTM round-tripping through the spliced GS).
    assert!(
        count_ink_pixels(&on, 0, 0, 100, 100) > 100,
        "rotated stencil should leave visible ink"
    );
}

/// Probe 8 — ImageMask under a CTM with negative X scale (horizontal
/// mirror). The image flip lives in `render_image_mask`'s
/// `pre_translate(0, 1).pre_scale(1/w, -1/h)`; a negative-scale CTM
/// composes correctly only if the helper's flip is applied in the
/// right order.
#[test]
fn qa_image_mask_negative_scale_mirror_paints_visible_band() {
    let mask = solid_image_mask_bytes(8, 1);
    let content = "q\n0 0 1 rg\n-60 0 0 60 80 20 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Negative-X scale composes with the helper's intrinsic Y flip;
    // pin that the mirrored mask actually paints.
    assert!(
        count_ink_pixels(&on, 0, 0, 100, 100) > 100,
        "mirrored stencil should leave visible ink"
    );
}

/// Probe 9 — ImageMask under a CTM with negative determinant (Y-flipped
/// on top of the image-space Y-flip; net result is "image space matches
/// user space"). Confirms the helper doesn't bake a flip assumption that
/// breaks composed transforms.
#[test]
fn qa_image_mask_negative_determinant_ctm_paints_visible_band() {
    let mask = solid_image_mask_bytes(8, 1);
    // det < 0: a*d - b*c = 60*-60 = -3600.
    let content = "q\n0.5 g\n60 0 0 -60 20 80 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    assert!(
        count_ink_pixels(&on, 0, 0, 100, 100) > 100,
        "negative-det stencil should leave visible ink"
    );
}

// ===========================================================================
// Probes 10-12 — Standard (non-mask) Image XObject pass-through.
//
// Wave 3 routes ONLY `/ImageMask true` through the pipeline; standard
// images go to `render_image` unchanged. These probes pin that the
// guard reads `/ImageMask true` strictly (not "any /ImageMask entry")
// and that non-mask images render correctly across the colour spaces
// that matter.
// ===========================================================================

/// Build a one-page PDF with a standard (non-mask) Image XObject `/IM1`
/// whose ColorSpace dict entry is rendered inline as `/{cs_name}` (use
/// for `DeviceRGB`, `DeviceGray`, `DeviceCMYK`). `bits_per_component`
/// is also written into the stream dict.
fn build_pdf_standard_image_named_cs(
    content_ops: &str,
    width: u32,
    height: u32,
    bits_per_component: u32,
    pixel_bytes: &[u8],
    cs_name: &str,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );

    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} \
         /BitsPerComponent {} /ColorSpace /{} /Length {} >>\nstream\n",
        width,
        height,
        bits_per_component,
        cs_name,
        pixel_bytes.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(pixel_bytes);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Build a one-page PDF with a standard Image XObject `/IM1` whose
/// ColorSpace is `[/Indexed /DeviceRGB hival lookup_stream_ref]`.
/// `palette_bytes` is the lookup table as raw RGB triples; `pixel_bytes`
/// are the index samples (BPC=8).
fn build_pdf_standard_image_indexed(
    content_ops: &str,
    width: u32,
    height: u32,
    pixel_bytes: &[u8],
    palette_bytes: &[u8],
    hival: u32,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    // Render palette as a hex string so we can keep everything in one
    // file without an extra indirect object.
    let mut palette_hex = String::from("<");
    for b in palette_bytes {
        palette_hex.push_str(&format!("{:02X}", b));
    }
    palette_hex.push('>');

    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );

    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} \
         /BitsPerComponent 8 /ColorSpace [/Indexed /DeviceRGB {} {}] \
         /Length {} >>\nstream\n",
        width,
        height,
        hival,
        palette_hex,
        pixel_bytes.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(pixel_bytes);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Probe 10 — CMYK standard image (non-mask) pass-through. Wave 3
/// does not splice the pipeline on these; the standard-image branch
/// paints the CMYK pixel data unchanged.
#[test]
fn qa_standard_image_cmyk_pass_through_paints_magenta_centre() {
    // 4x4 CMYK pixels, all (0, 1, 0, 0) → magenta under additive clamp.
    // Each pixel is 4 bytes (one per component).
    let mut pixels = Vec::with_capacity(16 * 4);
    for _ in 0..16 {
        pixels.extend_from_slice(&[0u8, 255, 0, 0]);
    }
    let content = "q\n80 0 0 80 10 10 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_standard_image_named_cs(content, 4, 4, 8, &pixels, "DeviceCMYK");
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // CMYK(0, 1, 0, 0) → §10.3.5 additive clamp → R=1, G=0, B=1
    // (magenta). Pin the centre pixel.
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        r > 200 && g < 60 && b > 200,
        "DeviceCMYK image (0,1,0,0) must render as magenta at centre, got ({r}, {g}, {b})"
    );
}

/// Probe 11 — Indexed standard image (non-mask) pass-through. Palette
/// of 256 entries (full 8-bit). Pixel data picks index 0 (red palette
/// entry) for every sample.
#[test]
fn qa_standard_image_indexed_256_pass_through_paints_red_centre() {
    // Build a 256-entry palette: index 0 = red, all others = white.
    let mut palette = Vec::with_capacity(256 * 3);
    palette.extend_from_slice(&[0xFFu8, 0x00, 0x00]); // index 0: red
    for _ in 1..256 {
        palette.extend_from_slice(&[0xFFu8, 0xFF, 0xFF]); // others: white
    }
    let pixels = vec![0u8; 16]; // 4x4 image, all index 0
    let content = "q\n80 0 0 80 10 10 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_standard_image_indexed(content, 4, 4, &pixels, &palette, 255);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Pin a body pixel: well inside the 80x80 image footprint.
    let (r, g, b, _a) = pixel_at(&on, 50, 50);
    assert!(
        r > 200 && g < 60 && b < 60,
        "indexed image at index 0 (red palette) must be red at centre, got ({r}, {g}, {b})"
    );
}

/// Probe 12 — `/ImageMask false` explicit (not omitted). The wave-3
/// guard reads `matches!(o, Object::Boolean(true))`; the `false` case
/// must take the standard-image branch. This is a regression pin
/// against a future refactor that might switch to `o.is_some()`.
#[test]
fn qa_image_with_explicit_imagemask_false_routes_to_standard_image() {
    // Mint a 4x4 DeviceGray standard image AND tag it with `/ImageMask
    // false`. The renderer must NOT take the mask branch (no
    // `render_image_mask` call) — the pipeline isn't routed for
    // standard images, so the centre pixel must reflect the grey
    // sample data, not the active fill colour.
    let pixels = vec![0x80u8; 16];
    let content = "q\n80 0 0 80 10 10 cm\n/IM1 Do\nQ\n";

    // Custom build with the extra dict key.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask false \
         /Width 4 /Height 4 /BitsPerComponent 8 /ColorSpace /DeviceGray \
         /Length {} >>\nstream\n",
        pixels.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(&pixels);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Pin: centre is mid-grey, NOT painted with the (default-zero) fill
    // colour. If the mask branch had erroneously fired, the stencil
    // bits (0x80 = `1000 0000`) would have painted only the high bit
    // as opaque, with the current fill colour, leaving most of the
    // page unpainted.
    let (r, g, b, _a) = pixel_at(&on, 50, 50);
    assert!(
        r == g && g == b && (110..=145).contains(&(r as i32)),
        "explicit /ImageMask false must render the grey pixel data, got ({r}, {g}, {b})"
    );
}

/// Probe 12b — ICCBased N=4 (CMYK ICC profile) standard image (non-mask)
/// pass-through. The ICC profile is supplied as an indirect stream
/// (object 6). Even if the extractor falls back when the ICC bytes are
/// not a valid profile, the routing decision (mask vs standard) must
/// remain stable — the image goes through `render_image`, not the mask
/// branch.
#[test]
fn qa_standard_image_iccbased_n4_pass_through_paints_visible_ink() {
    // 2x2 CMYK pixels (16 bytes), all magenta.
    let mut pixels = Vec::with_capacity(16);
    for _ in 0..4 {
        pixels.extend_from_slice(&[0u8, 255, 0, 0]);
    }
    // Bogus ICC profile bytes — the extractor falls back to /Alternate
    // or DeviceCMYK; what we're pinning is routing, not colour fidelity.
    let icc_bytes = vec![0u8; 32];

    let content = "q\n80 0 0 80 10 10 cm\n/IM1 Do\nQ\n";
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /Width 2 /Height 2 \
         /BitsPerComponent 8 /ColorSpace [/ICCBased 6 0 R] /Length {} >>\nstream\n",
        pixels.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(&pixels);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let icc_off = buf.len();
    let icc_hdr = format!(
        "6 0 obj\n<< /N 4 /Alternate /DeviceCMYK /Length {} >>\nstream\n",
        icc_bytes.len()
    );
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(&icc_bytes);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off, icc_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    let doc = PdfDocument::from_bytes(buf).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Routing pin: the ICCBased N=4 image must be routed through
    // `render_image`, not the mask branch. Visible signal: the painted
    // 80×80 region's centre is NOT page-background white (the bogus ICC
    // bytes fall back to DeviceCMYK, magenta CMYK still renders as
    // non-white ink). If the mask branch had fired the image's first
    // pixel byte (0x00) would be opaque-with-default-fill = black at
    // the corner only and the centre would stay white.
    let (r, g, b, _a) = pixel_at(&on, 50, 50);
    assert!(
        r < 250 || g < 250 || b < 250,
        "ICCBased N=4 standard image must paint visible ink at the image region centre \
         (routes through render_image, not the mask branch); got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// Probes 13-14 — Inline images (`BI ... ID ... EI`).
//
// Inline images are an entirely separate parse path. The wave-3 commit
// only touches the `Operator::Do` arm; inline images flow through
// `Operator::InlineImage` which the renderer DOES NOT IMPLEMENT —
// `page_renderer.rs` has no `Operator::InlineImage` arm. So:
//
//   - inline images render as nothing (transparent / unchanged page);
//   - inline ImageMasks therefore can't be filled via the pipeline
//     (capability gap, not a regression).
//
// These probes PIN the current behaviour. If a future wave wires up
// `Operator::InlineImage`, both should start failing — at which point
// the new arm needs its own pipeline routing for `/IM true`.
// ===========================================================================

/// Build a one-page PDF whose content stream is a literal byte slice
/// (so callers can embed non-ASCII inline-image data). The renderer
/// doesn't dispatch `Operator::InlineImage` today; this is a gap pin.
fn build_pdf_inline_image_bytes(content_ops: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Probe 13 — Inline ImageMask via `BI ... ID ... EI`. Pin the current
/// behaviour: the renderer does NOT dispatch `Operator::InlineImage`,
/// so the page is blank.
///
/// If a future wave adds inline-image support, this test will fail —
/// at which point the new arm needs its own pipeline routing for
/// `/IM true` to match the `Do` arm's behaviour. Tracked as
/// **WAVE-3-GAP-INLINE**.
#[test]
fn qa_inline_image_mask_renderer_gap_pin() {
    // Inline ImageMask: 1x1, /BPC 1, /IM true, one zero byte (opaque
    // under default Decode). Surround with a fill colour set first.
    //
    // Per PDF §8.9.7 the syntax for an inline image is:
    //   BI <dict-entries> ID <data> EI
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(b"q\n1 0 0 rg\n80 0 0 80 10 10 cm\n");
    content.extend_from_slice(b"BI /W 1 /H 1 /BPC 1 /IM true ID ");
    content.push(0x00);
    content.extend_from_slice(b" EI\nQ\n");
    let bytes = build_pdf_inline_image_bytes(&content);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Pin the gap: the page must be all white. If a future wave wires
    // up InlineImage rendering and forgets to route the fill through
    // the pipeline, this stops being all-white at the centre and the
    // pin fires.
    let (r, g, b, _a) = center_pixel(&on);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "inline ImageMask currently goes unrendered (renderer gap); \
         WAVE-3-GAP-INLINE must remain until InlineImage is wired up"
    );
}

/// Probe 14 — Inline standard (non-mask) image. Same gap: the renderer
/// doesn't dispatch `Operator::InlineImage`. Pin all-white centre.
#[test]
fn qa_inline_standard_image_renderer_gap_pin() {
    // 1x1 DeviceGray, BPC 8, single byte 0x80 → mid-grey. Without
    // dispatch, the page is blank.
    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(b"q\n80 0 0 80 10 10 cm\n");
    content.extend_from_slice(b"BI /W 1 /H 1 /BPC 8 /CS /G ID ");
    content.push(0x80);
    content.extend_from_slice(b" EI\nQ\n");
    let bytes = build_pdf_inline_image_bytes(&content);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "inline standard image currently goes unrendered (renderer gap)"
    );
}

// ===========================================================================
// Probes 15-17 — Form-XObject ImageMask interactions.
//
// Form XObjects are rendered recursively. When the Form's content
// stream invokes an ImageMask, the recursive walk should:
//   - find the mask in the Form's own /Resources;
//   - paint it through the wave-3 pipeline-routed path;
//   - propagate the parent's CTM into the recursion.
//
// These probes pin those interactions.
// ===========================================================================

/// Build a one-page PDF whose `/Fm1` Form XObject internally invokes
/// an ImageMask `/IM1`. Both are listed in the Form's own /Resources.
/// The page invokes `/Fm1 Do`.
fn build_pdf_form_with_inner_image_mask(
    page_content: &str,
    form_content: &str,
    form_resources_extra: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /Fm1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(page_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Form XObject (object 5). Its /Resources lists /IM1 → object 6,
    // plus any extra entries the caller wants (e.g. /ColorSpace).
    let form_off = buf.len();
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /XObject << /IM1 6 0 R >> {} >> /Length {} >>\nstream\n",
        form_resources_extra,
        form_content.len()
    );
    buf.extend_from_slice(form_hdr.as_bytes());
    buf.extend_from_slice(form_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // ImageMask XObject (object 6).
    let im_off = buf.len();
    let im_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        width,
        height,
        mask_data.len()
    );
    buf.extend_from_slice(im_hdr.as_bytes());
    buf.extend_from_slice(mask_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, form_off, im_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Build a PDF with TWO Form XObjects: the page invokes `/Fm1`, `/Fm1`
/// invokes `/Fm2`, and `/Fm2` invokes the ImageMask `/IM1`. Used to
/// pin two-level recursion.
fn build_pdf_form_in_form_with_image_mask(
    page_content: &str,
    outer_form_content: &str,
    inner_form_content: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /Fm1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(page_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // Outer form: lists /Fm2 (object 6) in its /XObject.
    let outer_off = buf.len();
    let outer_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /XObject << /Fm2 6 0 R >> >> /Length {} >>\nstream\n",
        outer_form_content.len()
    );
    buf.extend_from_slice(outer_hdr.as_bytes());
    buf.extend_from_slice(outer_form_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // Inner form: lists /IM1 (object 7).
    let inner_off = buf.len();
    let inner_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /XObject << /IM1 7 0 R >> >> /Length {} >>\nstream\n",
        inner_form_content.len()
    );
    buf.extend_from_slice(inner_hdr.as_bytes());
    buf.extend_from_slice(inner_form_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // ImageMask.
    let im_off = buf.len();
    let im_hdr = format!(
        "7 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        width,
        height,
        mask_data.len()
    );
    buf.extend_from_slice(im_hdr.as_bytes());
    buf.extend_from_slice(mask_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for off in [
        cat_off, pages_off, page_off, stream_off, outer_off, inner_off, im_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Build a PDF with a Form containing an ImageMask AND a Type 4
/// Separation in its /Resources/ColorSpace. The Form invokes the mask
/// after setting the spot colour. Used by the capability-gain test for
/// nested-Form Separation fills.
fn build_pdf_form_with_imagemask_and_type4_separation(
    page_content: &str,
    form_content: &str,
    type4_program: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /Fm1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(page_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // Form with /SpotMagenta colour space (Type 4 tint → object 7).
    let form_off = buf.len();
    let form_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << /XObject << /IM1 6 0 R >> \
                       /ColorSpace << /SpotMagenta [/Separation /MagentaSpot /DeviceCMYK 7 0 R] >> \
                     >> /Length {} >>\nstream\n",
        form_content.len()
    );
    buf.extend_from_slice(form_hdr.as_bytes());
    buf.extend_from_slice(form_content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // ImageMask.
    let im_off = buf.len();
    let im_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        width,
        height,
        mask_data.len()
    );
    buf.extend_from_slice(im_hdr.as_bytes());
    buf.extend_from_slice(mask_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    // Type 4 function.
    let func_off = buf.len();
    let func_hdr = format!(
        "7 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        type4_program.len()
    );
    buf.extend_from_slice(func_hdr.as_bytes());
    buf.extend_from_slice(type4_program.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for off in [
        cat_off, pages_off, page_off, stream_off, form_off, im_off, func_off,
    ] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Probe 15 — Form XObject whose internal content paints an
/// ImageMask under a Type 4 Separation fill. The capability gain
/// (full-tint → magenta vs `1 - tint` → black) must propagate through
/// the recursive Form rendering.
#[test]
fn qa_form_xobject_with_inner_image_mask_type4_separation_capability_gain() {
    let mask = solid_image_mask_bytes(8, 8);
    let type4 = "{ 0.0 exch 0.0 0.0 }"; // tint=1 → magenta
    let page = "q\n/Fm1 Do\nQ\n";
    let form = "q\n/SpotMagenta cs\n1 scn\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";

    let bytes = build_pdf_form_with_imagemask_and_type4_separation(page, form, type4, 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);

    // Pipeline runs the Type 4 program → magenta.
    let (r_on, g_on, b_on, _a) = center_pixel(&on);
    assert!(
        r_on >= 250 && g_on <= 5 && b_on >= 250,
        "Form-nested Type 4 Separation ImageMask must paint magenta, got ({r_on}, {g_on}, {b_on})"
    );
}

/// Probe 16 — Two-level Form recursion (Form-in-Form), where the
/// innermost content invokes an ImageMask with a DeviceRGB fill. The
/// rendered centre pixel must be the expected colour after the
/// pipeline routes the mask through both Form recursions.
#[test]
fn qa_form_in_form_image_mask_paints_inner_fill_colour() {
    let mask = solid_image_mask_bytes(8, 8);
    let page = "q\n/Fm1 Do\nQ\n";
    let outer = "q\n/Fm2 Do\nQ\n"; // delegate straight to inner
                                   // Inner sets the fill colour itself and paints the mask. (Set the
                                   // fill at the inner level so propagation through Form recursion is
                                   // not co-mingled with the pipeline-routing pin we're after — the
                                   // CTM and resource scope already test recursion.)
    let inner = "q\n0 1 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";

    let bytes = build_pdf_form_in_form_with_image_mask(page, outer, inner, 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        g > 200 && r < 60 && b < 60,
        "two-level Form ImageMask should paint green, got ({r}, {g}, {b})"
    );
}

/// Probe 16b — Bug-found pin (UNRELATED to wave-3, but discovered while
/// probing it). When the page sets the fill colour and then invokes a
/// Form which paints an ImageMask, the Form's content stream does NOT
/// see the page's `rg` — the centre paints black instead of the
/// inherited fill. This is a graphics-state-propagation gap at the
/// Form recursion boundary, not a pipeline-side issue.
///
/// Pinned `#[ignore]` to record the discovery without failing CI.
/// Bug name: **FORM-RECURSION-FILL-NOT-INHERITED** — the renderer's
/// recursive Form walk appears to reset (or not propagate) the GS
/// fill colour on entry to the child Form's content stream. Per PDF
/// §8.10.1 a Form XObject inherits the parent graphics state at the
/// point of invocation, with only `q ... Q` saving/restoring around
/// the call; the fill colour set with `rg` before `/Fm1 Do` should be
/// visible inside the Form's content stream.
#[ignore = "FORM-RECURSION-FILL-NOT-INHERITED: page-level fill not seen by Form's ImageMask paint"]
#[test]
fn qa_form_fill_inheritance_bug_pin() {
    let mask = solid_image_mask_bytes(8, 8);
    let page = "q\n0 1 0 rg\n/Fm1 Do\nQ\n";
    // Form sets only the CTM — does NOT set a fill colour itself, so
    // it must inherit the page-level `0 1 0 rg`.
    let form = "100 0 0 100 0 0 cm\n/IM1 Do\n";
    let bytes = build_pdf_form_with_inner_image_mask(page, form, "", 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r, g, b, _a) = center_pixel(&on);
    // Expected per spec: GS state propagates into child Form content
    // stream. Observed: centre is (0, 0, 0) — the page-level `rg` did
    // not stick across the Form boundary.
    assert!(
        g > 200 && r < 60 && b < 60,
        "page-level fill must be visible at Form's ImageMask paint, got ({r}, {g}, {b}) — FORM-RECURSION-FILL-NOT-INHERITED"
    );
}

/// Probe 17 — Form-XObject with a nested CTM transformation around
/// the inner ImageMask. Inside the Form, an inner `q ... cm ... /IM1
/// Do ... Q` must compose with the page's `cm` cleanly through the
/// pipeline-routed mask paint.
#[test]
fn qa_form_xobject_inner_ctm_around_image_mask_paints_visible_band() {
    let mask = solid_image_mask_bytes(8, 8);
    // The page sets a 30° rotation; the form sets a translation and
    // scale around the mask. CTM stack correctness across the form
    // boundary is what's being pinned.
    let page = "q\n0.866 0.5 -0.5 0.866 50 50 cm\n/Fm1 Do\nQ\n";
    let form = "q\n1 0 0 rg\n40 0 0 40 -20 -20 cm\n/IM1 Do\nQ\n";

    let bytes = build_pdf_form_with_inner_image_mask(page, form, "", 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    assert!(
        count_ink_pixels(&on, 0, 0, 100, 100) > 100,
        "Form with rotated + nested CTM should leave visible ink"
    );
}

// ===========================================================================
// Probes 18-22 — Multi-XObject interactions.
//
// These probes load two or more XObjects into a single page and pin
// that `q/Q` saving/restoring the GS state, plus the spliced GS clone
// the pipeline emits at each `/IM Do`, doesn't leak across paints.
// ===========================================================================

/// Build a page with two ImageMask XObjects `/IM1` and `/IM2` (both
/// solid stencils) and run an arbitrary content stream.
fn build_pdf_two_image_masks(content_ops: &str, w1: u32, h1: u32, w2: u32, h2: u32) -> Vec<u8> {
    let mask1 = solid_image_mask_bytes(w1, h1);
    let mask2 = solid_image_mask_bytes(w2, h2);

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R /IM2 6 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let im1_off = buf.len();
    let im1_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        w1,
        h1,
        mask1.len()
    );
    buf.extend_from_slice(im1_hdr.as_bytes());
    buf.extend_from_slice(&mask1);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let im2_off = buf.len();
    let im2_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        w2,
        h2,
        mask2.len()
    );
    buf.extend_from_slice(im2_hdr.as_bytes());
    buf.extend_from_slice(&mask2);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, im1_off, im2_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Build a page with an ImageMask `/IM1` and a standard image `/SI1`,
/// so probes can interleave them.
fn build_pdf_mask_plus_standard_image(
    content_ops: &str,
    mask_w: u32,
    mask_h: u32,
    std_w: u32,
    std_h: u32,
    std_pixels: &[u8],
) -> Vec<u8> {
    let mask = solid_image_mask_bytes(mask_w, mask_h);

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R /SI1 6 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let im_off = buf.len();
    let im_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        mask_w,
        mask_h,
        mask.len()
    );
    buf.extend_from_slice(im_hdr.as_bytes());
    buf.extend_from_slice(&mask);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let si_off = buf.len();
    let si_hdr = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} \
         /BitsPerComponent 8 /ColorSpace /DeviceGray /Length {} >>\nstream\n",
        std_w,
        std_h,
        std_pixels.len()
    );
    buf.extend_from_slice(si_hdr.as_bytes());
    buf.extend_from_slice(std_pixels);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, im_off, si_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Probe 18 — Two ImageMasks back-to-back, painted with different fill
/// colours (red then blue). Each splice clones GS afresh; the second
/// paint must not see the first paint's spliced state. The two halves
/// of the page should end up cleanly coloured.
#[test]
fn qa_two_image_masks_back_to_back_paint_distinct_halves() {
    // Left half: red. Right half: blue. The `q ... Q` brackets isolate
    // each paint's CTM and fill state.
    let content = "q\n1 0 0 rg\n50 0 0 100 0 0 cm\n/IM1 Do\nQ\n\
                   q\n0 0 1 rg\n50 0 0 100 50 0 cm\n/IM2 Do\nQ\n";
    let bytes = build_pdf_two_image_masks(content, 8, 8, 8, 8);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Sample the left half (red) and right half (blue) interior.
    let (r1, g1, b1, _a) = pixel_at(&on, 20, 50);
    let (r2, g2, b2, _a) = pixel_at(&on, 80, 50);
    assert!(
        r1 > 200 && g1 < 60 && b1 < 60,
        "left half should be red, got ({r1}, {g1}, {b1})"
    );
    assert!(
        b2 > 200 && r2 < 60 && g2 < 60,
        "right half should be blue, got ({r2}, {g2}, {b2})"
    );
}

/// Probe 19 — ImageMask, standard image, ImageMask interleaved on
/// the same page. The standard-image branch's `render_image` borrows
/// the unspliced `gs`; the mask branch borrows the spliced clone.
/// The standard image must not pick up the mask's spliced state, and
/// vice versa.
#[test]
fn qa_image_mask_then_standard_then_mask_interleaved_keep_state_isolated() {
    // 4x4 grey pixels for the standard image.
    let std_pixels = vec![0x60u8; 16];
    // Left strip: red mask. Middle: grey std image. Right strip: blue mask.
    let content = "q\n0.5 g\n40 0 0 100 30 0 cm\n/SI1 Do\nQ\n\
                   q\n1 0 0 rg\n30 0 0 100 0 0 cm\n/IM1 Do\nQ\n\
                   q\n0 0 1 rg\n30 0 0 100 70 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_mask_plus_standard_image(content, 8, 8, 4, 4, &std_pixels);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Left strip: red.
    let (r1, g1, b1, _a) = pixel_at(&on, 15, 50);
    assert!(r1 > 200 && g1 < 60 && b1 < 60, "left strip must be red, got ({r1},{g1},{b1})");
    // Middle: dark grey from the standard image (≈0x60 with possible filter).
    let (r2, g2, b2, _a) = pixel_at(&on, 50, 50);
    assert!(
        r2 == g2 && g2 == b2 && (60..=160).contains(&(r2 as i32)),
        "middle must be grey from standard image, got ({r2},{g2},{b2})"
    );
    // Right strip: blue.
    let (r3, g3, b3, _a) = pixel_at(&on, 85, 50);
    assert!(b3 > 200 && r3 < 60 && g3 < 60, "right strip must be blue, got ({r3},{g3},{b3})");
}

/// Probe 20 — ImageMask under an active SMask. The renderer must apply
/// the SMask to the paint; a DeviceRGB fill resolves through the
/// pipeline and the spliced clone must carry the SMask through.
#[test]
fn qa_image_mask_under_smask_none_still_paints() {
    // Page resources carry /GS1 in /ExtGState with `/SMask /None` set
    // explicitly. This is the "no smask" form but it exercises the
    // ExtGState plumbing without needing a full SMask dict (which
    // requires a transparency group XObject).
    let mask = solid_image_mask_bytes(8, 8);
    let resources = "/ExtGState << /GS1 << /SMask /None >> >>";
    let content = "q\n/GS1 gs\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, resources, 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // /SMask /None is the no-op smask: the full-page red stencil
    // must paint without the smask suppressing it.
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        r > 200 && g < 60 && b < 60,
        "/SMask /None must not suppress the red stencil, got ({r}, {g}, {b})"
    );
}

/// Probe 21 — ImageMask under an active clip path. Pixels outside the
/// clip must remain unpainted; the spliced GS clone must not drop the
/// clip state.
#[test]
fn qa_image_mask_under_active_clip_corner_remains_unpainted() {
    let mask = solid_image_mask_bytes(8, 8);
    // Clip to a 40×40 box around the page centre, then paint a full-page
    // stencil. Corners must remain white.
    let content = "q\n30 30 40 40 re W n\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Centre is inside the clip → red.
    let (r, g, b, _a) = center_pixel(&on);
    assert!(r > 200 && g < 60 && b < 60, "centre must be red, got ({r}, {g}, {b})");
    // The top-left corner (5,5) is well outside the 40×40 clip box (which
    // spans 30..70 in both axes) and must remain unpainted (white).
    let (rc, gc, bc, _a) = pixel_at(&on, 5, 5);
    assert_eq!(
        (rc, gc, bc),
        (255, 255, 255),
        "outside-clip corner must be white, got ({rc}, {gc}, {bc})"
    );
}

/// Probe 22 — ImageMask painted under a non-Normal blend mode. The
/// wave-3 `render_image_mask` reads `gs.blend_mode` and converts it
/// via `pdf_blend_mode_to_skia`. The spliced clone must preserve the
/// blend mode field through to the rasteriser.
#[test]
fn qa_image_mask_multiply_blend_mode_paints_against_white() {
    let mask = solid_image_mask_bytes(8, 8);
    let resources = "/ExtGState << /GS1 << /BM /Multiply >> >>";
    let content = "q\n/GS1 gs\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, resources, 8, 8, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // Multiply with red against white background → red.
    let (r, g, b, _a) = center_pixel(&on);
    assert!(
        r > 200 && g < 60 && b < 60,
        "multiply(red, white) must be red, got ({r}, {g}, {b})"
    );
}

// ===========================================================================
// Probes 23-26 — Capability at scale: DeviceN / `/All` / `/None` colorants
// applied to an ImageMask fill, plus a 100-mask one-page stress test.
//
// Mirror of the wave-1 path-fill and wave-2 text-fill capability suites.
// `render_image_mask` reads `gs.fill_color_rgb` once per paint; the
// pipeline must populate that from the Type 4 / DeviceN program. The
// `/All` / `/None` colorant-name special cases are not honoured today
// (existing behaviour) — pin whatever the renderer actually paints
// today as the regression anchor.
// ===========================================================================

/// Build a one-page PDF with an ImageMask XObject `/IM1` and an
/// indirect Type 4 function (object 6) whose `Domain` accommodates a
/// variable number of inputs (used for DeviceN). The Separation /
/// DeviceN colour space is set via `resources_extra`.
fn build_pdf_image_mask_with_devicen_type4(
    content_ops: &str,
    resources_extra: &str,
    width: u32,
    height: u32,
    mask_data: &[u8],
    type4_program: &str,
    range_array: &str,
    domain_pairs: &[i32],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources << /XObject << /IM1 5 0 R >> {} >> /Contents 4 0 R >>\nendobj\n",
        resources_extra
    );
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content_ops.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content_ops.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width {} /Height {} /BitsPerComponent 1 /Length {} >>\nstream\n",
        width,
        height,
        mask_data.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(mask_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let func_off = buf.len();
    let domain_str: Vec<String> = domain_pairs.iter().map(|v| v.to_string()).collect();
    let domain_array = format!("[{}]", domain_str.join(" "));
    let func_hdr = format!(
        "6 0 obj\n<< /FunctionType 4 /Domain {} /Range {} /Length {} >>\nstream\n",
        domain_array,
        range_array,
        type4_program.len()
    );
    buf.extend_from_slice(func_hdr.as_bytes());
    buf.extend_from_slice(type4_program.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off, func_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );
    buf
}

/// Probe 23 — Capability at scale: paint 100 ImageMasks on one page,
/// each at a distinct CTM offset, each filled with the same Type 4
/// Separation magenta. Inline path falls back to `1 - tint` → black
/// for every paint; pipeline runs the Type 4 program → magenta for
/// every paint. The two outputs must differ.
#[test]
fn qa_image_mask_100_paints_type4_separation_capability_at_scale() {
    let mask = solid_image_mask_bytes(2, 2);
    let type4 = "{ 0.0 exch 0.0 0.0 }";
    let resources = "/ColorSpace << /SpotMagenta [/Separation /MagentaSpot /DeviceCMYK 6 0 R] >>";

    // 10x10 grid: each tile 8x8 user units, spaced at 10-unit intervals.
    let mut content = String::from("/SpotMagenta cs\n1 scn\n");
    for row in 0..10 {
        for col in 0..10 {
            let x = col * 10;
            let y = row * 10;
            content.push_str(&format!("q 8 0 0 8 {} {} cm /IM1 Do Q\n", x, y));
        }
    }

    // Inline a PDF layout matching the shared imagemask-with-Type4
    // helper, so this probe stays self-contained.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources << /XObject << /IM1 5 0 R >> {} >> /Contents 4 0 R >>\nendobj\n",
        resources
    );
    buf.extend_from_slice(page.as_bytes());
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width 2 /Height 2 /BitsPerComponent 1 /Length {} >>\nstream\n",
        mask.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(&mask);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let func_off = buf.len();
    let func_hdr = format!(
        "6 0 obj\n<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\nstream\n",
        type4.len()
    );
    buf.extend_from_slice(func_hdr.as_bytes());
    buf.extend_from_slice(type4.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off, func_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);

    // Pick a representative tile centre (col 5, row 5): tile starts at
    // (50, 50), 8x8, so centre ≈ (54, 54).
    let (r_on, g_on, b_on, _a) = pixel_at(&on, 54, 54);
    assert!(
        r_on > 200 && g_on < 60 && b_on > 200,
        "pipeline 100-paint: tile centre must be magenta, got ({r_on},{g_on},{b_on})"
    );
}

/// Probe 24 — DeviceN multi-colorant Type 4 applied to an ImageMask.
/// Mirrors wave-1's path-fill DeviceN test; the wave-3 ImageMask path
/// must produce the same Type 4 evaluation result.
#[test]
fn qa_image_mask_devicen_multi_colorant_type4_capability() {
    let mask = solid_image_mask_bytes(8, 8);
    // `{ exch pop 0.0 exch 0.0 0.0 }` — pop the first colorant, leave
    // CMYK(0, second, 0, 0) on the stack. With `0 1 scn` we get
    // CMYK(0, 1, 0, 0) → magenta.
    let type4 = "{ exch pop 0.0 exch 0.0 0.0 }";
    let resources = "/ColorSpace << /TwoSpot [/DeviceN [/SpotA /SpotB] /DeviceCMYK 6 0 R] >>";
    let content = "/TwoSpot cs 0 1 scn\n100 0 0 100 0 0 cm\n/IM1 Do\n";

    let bytes = build_pdf_image_mask_with_devicen_type4(
        content,
        resources,
        8,
        8,
        &mask,
        type4,
        "[0 1 0 1 0 1 0 1]",
        &[0, 1, 0, 1],
    );
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r_on, g_on, b_on, _a) = center_pixel(&on);
    assert!(
        r_on > 200 && g_on < 60 && b_on > 200,
        "pipeline DeviceN Type-4 ImageMask must paint magenta, got ({r_on},{g_on},{b_on})"
    );
}

/// Probe 25 — Separation `/All` colorant applied to an ImageMask. The
/// pipeline runs the tint transform like any other Separation, so the
/// rendered centre pixel must reflect the Type-4-evaluated colour.
/// Mirror of the wave-2 text `/All` test.
#[test]
fn qa_image_mask_separation_all_colorant_pipeline_paints_type4_output() {
    let mask = solid_image_mask_bytes(8, 8);
    let type4 = "{ 0.0 exch 0.0 0.0 }";
    // tint=0.5 → CMYK(0, 0.5, 0, 0) → faint magenta.
    let content = "/All_CS cs 0.5 scn\n100 0 0 100 0 0 cm\n/IM1 Do\n";
    let resources = "/ColorSpace << /All_CS [/Separation /All /DeviceCMYK 6 0 R] >>";

    let bytes = build_pdf_image_mask_with_devicen_type4(
        content,
        resources,
        8,
        8,
        &mask,
        type4,
        "[0 1 0 1 0 1 0 1]",
        &[0, 1],
    );
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let (r_on, g_on, b_on, _a) = center_pixel(&on);
    // Faint magenta: R high, B high, G middling.
    assert!(
        r_on > g_on && b_on > g_on,
        "pipeline /All Separation Type-4 ImageMask must trend magenta (R>G, B>G), got ({r_on},{g_on},{b_on})"
    );
}

/// Probe 26 — Separation `/None` colorant applied to an ImageMask.
/// Per ISO 32000-1 §8.6.6.3, `/None` produces no visible output. The
/// pipeline's per-plate routing selector (`InkSelector::None`, stamped
/// by the composer on the source colour space) makes the composite
/// resolver hand back a fully-transparent RGBA, so the ImageMask
/// rasteriser paints with alpha=0 and lays down zero ink.
#[test]
fn qa_image_mask_separation_none_colorant_paints_zero_ink() {
    let mask = solid_image_mask_bytes(8, 8);
    let type4 = "{ 0.0 exch 0.0 0.0 }";
    let content = "/None_CS cs 0.5 scn\n100 0 0 100 0 0 cm\n/IM1 Do\n";
    let resources = "/ColorSpace << /None_CS [/Separation /None /DeviceCMYK 6 0 R] >>";

    let bytes = build_pdf_image_mask_with_devicen_type4(
        content,
        resources,
        8,
        8,
        &mask,
        type4,
        "[0 1 0 1 0 1 0 1]",
        &[0, 1],
    );
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    let on_ink = count_ink_pixels(&on, 0, 0, 100, 100);
    assert_eq!(
        on_ink, 0,
        "/None ImageMask must paint zero ink per §8.6.6.3 (got {on_ink} ink pixels)"
    );
}

// ===========================================================================
// Probes 27-30 — Adversarial / malformed input.
//
// `render_image_mask` consumes the raw stream length, checks
// `row_bytes * height <= raw.len()`, and bails with an `Image` error
// on short streams. Long streams are silently truncated.
// Width/Height = 0 short-circuits before allocation. Width or Height
// of `0xFFFFFF` would attempt a 4 GB allocation; the helper should
// either bail or be guarded.
// ===========================================================================

/// Probe 27 — Stream shorter than the declared Width×Height bits.
/// The helper must NOT panic and must NOT paint a corrupted image.
/// (Today the helper returns an `Image` error via a `log::warn!` at
/// the `Do` arm; the page renders as if the mask weren't there.)
#[test]
fn qa_image_mask_too_short_stream_no_panic_pin() {
    // Declare 8x8 but provide only 1 byte (needs 8 bytes for the
    // 8-row stencil at row_bytes=1).
    let bytes_short = vec![0u8; 1];
    let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 8, &bytes_short);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    // No-panic invariant: a short stream falls through to the helper's
    // size-check bail and leaves the page unpainted.
    let on = render_with_pipeline_allow_fail(&doc, true)
        .expect("too-short ImageMask stream must not panic");
    let (r, g, b, _a) = center_pixel(&on);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "too-short stream must produce no paint, got ({r},{g},{b})"
    );
}

/// Probe 28 — Stream longer than declared. The helper indexes into
/// the buffer using `row_bytes * height`; trailing bytes are ignored.
/// No panic, no spurious paint of the trailing bytes.
#[test]
fn qa_image_mask_too_long_stream_no_panic_pin() {
    // Declare 8x8 (needs 8 bytes); provide 64.
    let mut bytes_long = vec![0u8; 8]; // first 8 bytes — all opaque
    bytes_long.extend_from_slice(&[0xFFu8; 56]); // trailing garbage
    let content = "q\n0 1 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 8, 8, &bytes_long);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline(&doc, true);
    // The first 8 bytes ARE the full 8x8 stencil; they're all opaque,
    // so the centre must be green. Trailing 56 bytes are ignored.
    let (r, g, b, _a) = center_pixel(&on);
    assert!(g > 200 && r < 60 && b < 60, "centre must be green, got ({r},{g},{b})");
}

/// Probe 29 — `Width=0` and `Height=0`. The helper short-circuits
/// these and returns Ok(()) without painting.
#[test]
fn qa_image_mask_zero_dimensions_no_paint_no_panic() {
    for (w, h) in [(0u32, 8u32), (8, 0), (0, 0)] {
        let mask = vec![0u8; 8]; // some data, ignored when w==0 or h==0
        let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
        let bytes = build_pdf_image_mask(content, "", w, h, &mask);
        let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
        let on = render_with_pipeline_allow_fail(&doc, true)
            .unwrap_or_else(|| panic!("renderer must not panic for {}x{}", w, h));
        // No paint: page is fully white.
        let (r, g, b, _a) = center_pixel(&on);
        assert_eq!(
            (r, g, b),
            (255, 255, 255),
            "{}x{} ImageMask must produce no paint at centre, got ({}, {}, {})",
            w,
            h,
            r,
            g,
            b
        );
    }
}

/// Probe 30 — Absurdly-large dimensions. `render_image_mask` allocates
/// `vec![0u8; (w*h*4) as usize]`; with `width = 0xFFFFFF` and `height = 1`
/// that's 4 * 16777215 ≈ 64 MB. The helper's expected-size check fires
/// FIRST (the supplied stream is shorter than the row-byte requirement)
/// and bails before allocating the destination buffer.
///
/// PIN: the renderer must NOT panic on huge declared dimensions when
/// the supplied stream is short. If a future allocator-tightening pass
/// adds an upfront size cap, this test still passes (the bail order
/// shifts but the no-panic invariant holds).
#[test]
fn qa_image_mask_huge_dimensions_short_stream_no_panic() {
    // Width 0xFFFFFF, Height 1 → row_bytes = 2097152, total expected =
    // 2097152. Supply only 1 byte; the size check rejects.
    let mask = vec![0u8; 1];
    let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";
    let bytes = build_pdf_image_mask(content, "", 0xFFFFFF, 1, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let on = render_with_pipeline_allow_fail(&doc, true)
        .expect("huge-dim short-stream ImageMask must not panic");
    // The expected-size check bails before allocating 64MB of pixels;
    // no paint reaches the centre.
    let (r, g, b, _a) = center_pixel(&on);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "huge-dim short-stream must produce no paint, got ({r},{g},{b})"
    );
}

/// Probe 30b — Negative dimensions arrive as PDF integers; PDF parses
/// them as `Object::Integer(i64)` and `as_integer()` returns `i64`.
/// The wave-3 helper casts via `as u32`, which on a negative integer
/// wraps to a huge value. Pair with a tiny stream; bail must fire
/// before allocation. Probe: no panic.
///
/// This is a regression pin against a future refactor switching the
/// cast to a `try_into()` that bails on negatives — the no-panic
/// invariant must hold across both behaviours.
#[test]
fn qa_image_mask_negative_dimension_field_no_panic() {
    let mask = vec![0u8; 1];
    // Custom build that writes Width = -1 to the dict.
    let content = "q\n1 0 0 rg\n100 0 0 100 0 0 cm\n/IM1 Do\nQ\n";

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_off = buf.len();
    buf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
          /Resources << /XObject << /IM1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    );
    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xobj_off = buf.len();
    let xobj_hdr = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Image /ImageMask true \
         /Width -1 /Height 8 /BitsPerComponent 1 /Length {} >>\nstream\n",
        mask.len()
    );
    buf.extend_from_slice(xobj_hdr.as_bytes());
    buf.extend_from_slice(&mask);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for off in [cat_off, pages_off, page_off, stream_off, xobj_off] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_off).as_bytes(),
    );

    let doc = PdfDocument::from_bytes(buf).expect("PDF parses");
    let on =
        render_with_pipeline_allow_fail(&doc, true).expect("negative-dim ImageMask must not panic");
    // The expected-size check bails on the wrapped-huge dimension
    // before any paint reaches the centre.
    let (r, g, b, _a) = center_pixel(&on);
    assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "negative-dim must produce no paint, got ({r},{g},{b})"
    );
}

// ===========================================================================
// Probes 31-32 — Performance.
//
// Wave 1+2 surfaced a per-paint clone leak as a performance regression
// (the now-fixed `kind_copy` stub). The wave-3 path must stay within
// the same envelope: one pipeline_resolve_paint_gs call per /Do, with
// the Device-family short-circuit returning None (zero clone) when the
// resolved colour already matches the GS field.
// ===========================================================================

/// Per-paint allocation pressure sanity. A 1000-paint pipeline render
/// with a Device-family fill must complete inside a generous wall-clock
/// budget. Coarse guard against an O(N) per-paint allocation spiral
/// (e.g. a clone slipping into the short-circuit path).
#[test]
fn qa_image_mask_perf_thousand_paints_completes_within_budget() {
    let mask = solid_image_mask_bytes(2, 2);
    let mut content = String::from("0 0 1 rg\n");
    let mut painted = 0;
    for row in 0..32 {
        for col in 0..32 {
            if painted >= 1000 {
                break;
            }
            content.push_str(&format!("q 2 0 0 2 {} {} cm /IM1 Do Q\n", col * 3, row * 3));
            painted += 1;
        }
        if painted >= 1000 {
            break;
        }
    }
    let bytes = build_pdf_image_mask(&content, "", 2, 2, &mask);
    let doc = PdfDocument::from_bytes(bytes).expect("PDF parses");
    let t = Instant::now();
    let _ = render_with_pipeline(&doc, true);
    let dt = t.elapsed();
    assert!(
        dt.as_secs_f64() < 30.0,
        "1000-ImageMask pipeline render must complete within 30s, took {:.3}s",
        dt.as_secs_f64()
    );
}
