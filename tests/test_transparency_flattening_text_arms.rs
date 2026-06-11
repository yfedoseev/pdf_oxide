//! Round-3 QA — text-showing paint-arm probes.
//!
//! The round-3 implementation wired SMask + overprint + compose-first
//! correction onto Tj / TJ / ' / " (text-showing operators). The
//! round-3 agent flagged these as "wired but unverified — needs font
//! fixture infrastructure". This file closes that verification gap.
//!
//! Fixtures use `/Type /Font /Subtype /Type1 /BaseFont /Helvetica`,
//! one of the standard 14 fonts a PDF viewer resolves without an
//! embedded font program. The renderer's text rasteriser falls back
//! to bundled DejaVu Sans for actual glyph outlines.
//!
//! Each probe asserts the soft-mask / overprint effect modulates the
//! painted glyph pixels, not just the page background. Black text on
//! white, sampled at the centre of the glyph stroke.

#![cfg(all(feature = "rendering", feature = "icc"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

// ===========================================================================
// Synthetic PDF builder with a Helvetica font resource
// ===========================================================================
//
// Object layout:
//   1 /Catalog
//   2 /Pages
//   3 /Page (refs 4 content, 5 font, optional 6+ extras)
//   4 content stream
//   5 /Font /Type1 /Helvetica
//   6+ caller-supplied extras (XObject forms etc.)

fn build_text_pdf(content: &str, resources_extra: &str, extra_objs: &[&str]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let off_cat = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off_pages = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off_page = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 5 0 R >> {} >> /Contents 4 0 R >>\nendobj\n",
        resources_extra
    );
    buf.extend_from_slice(page.as_bytes());

    let off_content = buf.len();
    let hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let off_font = buf.len();
    buf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extra_objs {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj.as_bytes());
    }

    let xref_off = buf.len();
    let total_objs = 5 + extra_offs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [off_cat, off_pages, off_page, off_content, off_font] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    for off in extra_offs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs + 1,
            xref_off
        )
        .as_bytes(),
    );
    buf
}

fn render_rgba_200(pdf_bytes: Vec<u8>) -> Vec<u8> {
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("synthetic PDF parses");
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(&doc, 0, &opts).expect("render_page succeeds");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    assert_eq!(img.width, 200);
    assert_eq!(img.height, 200);
    img.data
}

fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let off = ((y * 200 + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

/// Scan the painted region and return the minimum red channel value
/// observed (lowest value ⇒ darkest pixel ⇒ centre of a glyph
/// stroke). Defensive bounds so we don't drop a panic for an empty
/// region.
fn min_r_in_region(rgba: &[u8], x_min: u32, x_max: u32, y_min: u32, y_max: u32) -> u8 {
    let mut min_r = 255u8;
    for y in y_min..y_max {
        for x in x_min..x_max {
            let (r, _, _, _) = pixel_at(rgba, x, y);
            if r < min_r {
                min_r = r;
            }
        }
    }
    min_r
}

/// Return the mean RGB of the painted (non-white) pixels in the
/// region. Painted = at least one channel below 240. If no pixel is
/// painted, returns (255, 255, 255, 0) — caller decides what that
/// means for the assertion.
fn mean_painted_rgb(
    rgba: &[u8],
    x_min: u32,
    x_max: u32,
    y_min: u32,
    y_max: u32,
) -> (f32, f32, f32, u32) {
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;
    let mut n = 0u32;
    for y in y_min..y_max {
        for x in x_min..x_max {
            let (r, g, b, _) = pixel_at(rgba, x, y);
            if r < 240 || g < 240 || b < 240 {
                r_sum += r as u32;
                g_sum += g as u32;
                b_sum += b as u32;
                n += 1;
            }
        }
    }
    if n == 0 {
        (255.0, 255.0, 255.0, 0)
    } else {
        let n_f = n as f32;
        (r_sum as f32 / n_f, g_sum as f32 / n_f, b_sum as f32 / n_f, n)
    }
}

// ===========================================================================
// Sanity: Helvetica fixture actually paints glyph pixels
// ===========================================================================
//
// Before relying on the fixture, prove the renderer actually deposits
// glyph pixels on the page. Pattern: white background, BT … Tj ET
// with black fill — assert at least one pixel in the text band is
// significantly darker than white.

#[test]
fn text_helvetica_fixture_paints_glyph_pixels() {
    let content = "1 1 1 rg\n0 0 200 200 re\nf\n\
                   0 0 0 rg\n\
                   BT /F1 48 Tf 30 80 Td (HELLO) Tj ET\n";
    let rgba = render_rgba_200(build_text_pdf(content, "", &[]));
    // Text band — PDF y=80 baseline, ascender ~48*0.75 = 36, so painted
    // glyph pixels live around image y = 200 - 80 = 120 minus ascender
    // ⇒ y ~ 80..130 in image space, x ~ 30..180.
    let darkest = min_r_in_region(&rgba, 30, 180, 80, 130);
    assert!(
        darkest < 100,
        "Helvetica fixture must paint visible glyphs — expected at least \
         one pixel with r < 100 in the text band; got darkest r = \
         {darkest}. If the renderer didn't deposit glyphs, the SMask / \
         overprint probes below cannot discriminate."
    );
}

// ===========================================================================
// Text-arm SMask probes — Tj / TJ / ' / "
// ===========================================================================
//
// Each probe lays a white background, declares an ExtGState with
// /SMask /S /Luminosity /G <50% grey form>, then runs the text-
// showing operator. With the round-3 wiring, the painted glyph
// pixels should be modulated by the 50% luminance soft-mask:
// black-on-white text becomes mid-grey.
//
// Without wiring, the glyph pixels would be fully opaque black
// (~0, 0, 0). The probe asserts the mean PAINTED-pixel RGB is
// significantly lighter than fully-opaque black AND distinct from
// fully-white (proves there ARE painted pixels and they got
// modulated).

fn fixture_smask_text(text_op: &str) -> Vec<u8> {
    let smask_form = "0.5 g\n0 0 200 200 re\nf\n";
    let obj_6 = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        smask_form.len(),
        smask_form
    );
    let content = format!(
        "1 1 1 rg\n0 0 200 200 re\nf\n\
         /Sm gs\n\
         0 0 0 rg\n\
         {}\n",
        text_op
    );
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 6 0 R >> >> >>";
    build_text_pdf(&content, resources, &[&obj_6])
}

fn assert_smask_modulates(rgba: &[u8], op_name: &str) {
    let (mean_r, mean_g, mean_b, n) = mean_painted_rgb(rgba, 30, 180, 80, 130);
    assert!(
        n >= 50,
        "{op_name} text under SMask: expected ≥ 50 painted pixels in \
         the text band; got {n}. If the text didn't render, the SMask \
         probe cannot discriminate the wiring."
    );
    // 50% luminance SMask on black-text-on-white ⇒ painted pixels
    // composite to mid-grey (~128). Without wiring, painted pixels
    // stay opaque black (~0). Assert mean is meaningfully above the
    // unwired baseline.
    let unwired_threshold = 60.0;
    assert!(
        mean_r > unwired_threshold && mean_g > unwired_threshold && mean_b > unwired_threshold,
        "{op_name} text under SMask: expected mean painted-pixel RGB \
         elevated above unwired-black baseline (>{unwired_threshold} on \
         each channel ⇒ SMask 50% luminance modulation visible); got \
         mean=({mean_r:.0}, {mean_g:.0}, {mean_b:.0}) over n={n} pixels. \
         If close to (0, 0, 0) the SMask wiring on {op_name} is broken."
    );
}

#[test]
fn qa_round3_smask_modulates_tj_text() {
    let rgba = render_rgba_200(fixture_smask_text("BT /F1 48 Tf 30 80 Td (HELLO) Tj ET"));
    assert_smask_modulates(&rgba, "Tj");
}

#[test]
fn qa_round3_smask_modulates_tj_array_text() {
    let rgba = render_rgba_200(fixture_smask_text("BT /F1 48 Tf 30 80 Td [(HE) -50 (LLO)] TJ ET"));
    assert_smask_modulates(&rgba, "TJ");
}

#[test]
fn qa_round3_smask_modulates_apostrophe_text() {
    // ' (apostrophe / NextLineShowText) — moves to next line and shows.
    // Requires a Tw / Tc set explicitly per PDF spec. Provide a leading
    // initial Tj so the apostrophe-line is the second visible band.
    let rgba =
        render_rgba_200(fixture_smask_text("BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj (BTM) ' ET"));
    assert_smask_modulates(&rgba, "'");
}

#[test]
fn qa_round3_smask_modulates_quote_text() {
    // " (quote / SetSpacingNextLineShowText) — takes Tw, Tc, string
    // operands. Same structural pattern as '.
    let rgba =
        render_rgba_200(fixture_smask_text("BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj 0 0 (BTM) \" ET"));
    assert_smask_modulates(&rgba, "\"");
}

// ===========================================================================
// Text-arm overprint probes — CMYK text under /op true
// ===========================================================================
//
// Lay a cyan 50% backdrop, then paint yellow text over it with
// /op true /OPM 1 (overprint). With the round-3 wiring, the
// painted glyph pixels should retain the cyan plate where the yellow
// glyph overlaps — overprint adds plates rather than knocking them
// out. Without wiring, the yellow knocks the cyan out completely.
//
// The probe compares the painted-pixel RGB to a no-overprint render
// of the same fixture. With overprint, painted pixels include the
// retained cyan ⇒ the green channel stays high but the blue drops
// LESS than without overprint (cyan = (0, 1, 1) RGB).

fn fixture_overprint_text(text_op: &str) -> Vec<u8> {
    let content = format!(
        "0.5 0 0 0 k\n0 0 200 200 re\nf\n\
         /OpOn gs\n\
         0 0 1 0 k\n\
         0 0 1 0 K\n\
         {}\n",
        text_op
    );
    let resources = "/ExtGState << /OpOn << /Type /ExtGState /op true /OP true /OPM 1 >> >>";
    build_text_pdf(&content, resources, &[])
}

fn fixture_no_overprint_text(text_op: &str) -> Vec<u8> {
    let content = format!(
        "0.5 0 0 0 k\n0 0 200 200 re\nf\n\
         0 0 1 0 k\n\
         0 0 1 0 K\n\
         {}\n",
        text_op
    );
    build_text_pdf(&content, "", &[])
}

fn assert_overprint_modulates(rgba_op: &[u8], rgba_no_op: &[u8], op_name: &str) {
    // Painted-pixel mean inside the text band on each render.
    let (r_op, g_op, b_op, n_op) = mean_painted_rgb(rgba_op, 30, 180, 80, 130);
    let (r_no, g_no, b_no, n_no) = mean_painted_rgb(rgba_no_op, 30, 180, 80, 130);
    assert!(
        n_op >= 50 && n_no >= 50,
        "{op_name} overprint probe: expected ≥ 50 painted pixels in \
         each render; got n_op={n_op}, n_no_op={n_no}. Text fixture \
         didn't render glyphs — can't discriminate overprint wiring."
    );
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 20.0,
        "{op_name} text overprint vs no-overprint painted-pixel mean \
         delta: expected > 20.0 (overprint retains cyan plate where \
         yellow glyph overlaps the backdrop); got delta={delta:.1} \
         between op=({r_op:.0}, {g_op:.0}, {b_op:.0}) and \
         no_op=({r_no:.0}, {g_no:.0}, {b_no:.0}). If delta ≈ 0 the \
         overprint wiring on {op_name} is broken."
    );
}

#[test]
fn qa_round3_overprint_modulates_tj_text() {
    let rgba_op = render_rgba_200(fixture_overprint_text("BT /F1 48 Tf 30 80 Td (HELLO) Tj ET"));
    let rgba_no = render_rgba_200(fixture_no_overprint_text("BT /F1 48 Tf 30 80 Td (HELLO) Tj ET"));
    assert_overprint_modulates(&rgba_op, &rgba_no, "Tj");
}

#[test]
fn qa_round3_overprint_modulates_tj_array_text() {
    let rgba_op =
        render_rgba_200(fixture_overprint_text("BT /F1 48 Tf 30 80 Td [(HE) -50 (LLO)] TJ ET"));
    let rgba_no =
        render_rgba_200(fixture_no_overprint_text("BT /F1 48 Tf 30 80 Td [(HE) -50 (LLO)] TJ ET"));
    assert_overprint_modulates(&rgba_op, &rgba_no, "TJ");
}

#[test]
fn qa_round3_overprint_modulates_apostrophe_text() {
    let rgba_op =
        render_rgba_200(fixture_overprint_text("BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj (BTM) ' ET"));
    let rgba_no = render_rgba_200(fixture_no_overprint_text(
        "BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj (BTM) ' ET",
    ));
    assert_overprint_modulates(&rgba_op, &rgba_no, "'");
}

#[test]
fn qa_round3_overprint_modulates_quote_text() {
    let rgba_op = render_rgba_200(fixture_overprint_text(
        "BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj 0 0 (BTM) \" ET",
    ));
    let rgba_no = render_rgba_200(fixture_no_overprint_text(
        "BT /F1 48 Tf 12 TL 30 80 Td (TOP) Tj 0 0 (BTM) \" ET",
    ));
    assert_overprint_modulates(&rgba_op, &rgba_no, "\"");
}
