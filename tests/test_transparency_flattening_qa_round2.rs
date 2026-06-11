//! Round-2 QA probes for the transparency-flattening branch.
//!
//! This suite augments `test_transparency_flattening_audit.rs` with
//! probes that surface coverage gaps the round-2 implementation agent
//! flagged but did not close. Categories:
//!
//!  - **Non-linear ICC OutputIntent + composite precedence** (gap 1 from
//!    the round-1 audit, deferred by the round-2 agent). The agent
//!    claimed the additive-clamp fallback is linear so convert-first vs
//!    composite-first are byte-identical. This QA suite builds a
//!    non-linear ICC fixture (non-identity input curves drive
//!    quadlinear-CLUT lookups along distinct paths for each paint, so
//!    `ICC(A) + ICC(B)` differs from `ICC(A+B)`) and writes the probe
//!    that proves the gap real.
//!
//!  - **SMask + overprint paint-arm coverage matrix**. Subsequent
//!    rounds wired `smask_snapshot` / `overprint_snapshot` through
//!    every paint operator the round-2 audit flagged — the FillStroke
//!    combos (`B`, `B*`, `b`, `b*`), FillEvenOdd (`f*`), PaintShading
//!    (`sh`), `Do`, and the text-showing operators (`Tj`, `TJ`, `'`,
//!    `"`). The tracking constants below are preserved as historical
//!    markers; each probe pins the post-fix byte-exact behaviour and
//!    guards against a regression that would re-introduce the
//!    direct-paint path.
//!
//!  - **SMask scope through q/Q**. The agent flagged this as "rides on
//!    GraphicsState clone behaviour, correct but unprobed."
//!
//!  - **Composite overprint reconstruction loss**. The agent admitted
//!    "snapshot-RGB reconstruction loses information for snapshots that
//!    previously went through a non-trivial ICC."

#![cfg(all(feature = "rendering", feature = "icc"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

// ===========================================================================
// HONEST_GAP tracking constants
// ===========================================================================

macro_rules! smask_op_gap {
    ($name:ident, $op_desc:literal) => {
        pub const $name: &str = concat!(
            stringify!($name),
            ": ExtGState /SMask is only honoured on Operator::Fill and \
             Operator::Stroke. The ",
            $op_desc,
            " operator path does not call smask_snapshot / \
             apply_smask_after_paint; soft masks silently drop on this \
             paint arm. The round-2 implementation agent flagged this as \
             mechanical duplication."
        );
    };
}

smask_op_gap!(HONEST_GAP_SMASK_FILLSTROKE_NOT_WIRED, "B (fill+stroke)");
smask_op_gap!(HONEST_GAP_SMASK_FILLSTROKE_EVENODD_NOT_WIRED, "B* (fill+stroke EvenOdd)");
smask_op_gap!(HONEST_GAP_SMASK_CLOSE_FILLSTROKE_NOT_WIRED, "b (close+fill+stroke)");
smask_op_gap!(
    HONEST_GAP_SMASK_CLOSE_FILLSTROKE_EVENODD_NOT_WIRED,
    "b* (close+fill+stroke EvenOdd)"
);
smask_op_gap!(HONEST_GAP_SMASK_FILL_EVENODD_NOT_WIRED, "f* (fill EvenOdd)");
smask_op_gap!(HONEST_GAP_SMASK_PAINT_SHADING_NOT_WIRED, "sh (paint shading)");
smask_op_gap!(HONEST_GAP_SMASK_DO_NOT_WIRED, "Do (Form XObject + image invocation)");
smask_op_gap!(HONEST_GAP_SMASK_TEXT_SHOWING_NOT_WIRED, "Tj / TJ / ' / \" (text-showing)");

macro_rules! overprint_op_gap {
    ($name:ident, $op_desc:literal) => {
        pub const $name: &str = concat!(
            stringify!($name),
            ": §11.7.4 overprint correction is only honoured on \
             Operator::Fill and Operator::Stroke. The ",
            $op_desc,
            " operator path does not call overprint_snapshot / \
             apply_overprint_after_paint; overprint preview silently \
             drops on this paint arm. The round-2 implementation agent \
             flagged this as mechanical duplication."
        );
    };
}

overprint_op_gap!(HONEST_GAP_OVERPRINT_FILLSTROKE_NOT_WIRED, "B (fill+stroke)");
overprint_op_gap!(HONEST_GAP_OVERPRINT_FILLSTROKE_EVENODD_NOT_WIRED, "B* (fill+stroke EvenOdd)");
overprint_op_gap!(HONEST_GAP_OVERPRINT_CLOSE_FILLSTROKE_NOT_WIRED, "b (close+fill+stroke)");
overprint_op_gap!(
    HONEST_GAP_OVERPRINT_CLOSE_FILLSTROKE_EVENODD_NOT_WIRED,
    "b* (close+fill+stroke EvenOdd)"
);
overprint_op_gap!(HONEST_GAP_OVERPRINT_FILL_EVENODD_NOT_WIRED, "f* (fill EvenOdd)");

// ===========================================================================
// Synthetic PDF + ICC profile helpers
// ===========================================================================

/// Build a minimal valid ICC v2 CMYK→Lab profile whose 4-channel input
/// curves apply a gamma-2.2 transform BEFORE the CLUT lookup. Combined
/// with a CLUT whose corners are positioned at Lab(L=255·(1-Σink/4),
/// 128, 128) — i.e. white at 0-ink, black at 4-ink — the profile maps
/// CMYK to Lab via a non-multilinear function of the raw CMYK bytes.
///
/// This is the lever for the convert-first vs composite-first
/// divergence: when two CMYK paints A and B composite at alpha 0.5,
/// convert-first computes `(ICC(A) + ICC(B)) / 2`; composite-first
/// computes `ICC( (A + B) / 2 )`. Because the input curves are
/// non-linear (gamma 2.2), these two paths produce visibly different
/// RGB outputs even though the CLUT body is multilinear.
///
/// The input curves are 256-entry tables — qcms reads them as
/// `lut_interp_linear_float`, sampling across [0, 1] and using the
/// entry value as a linearised input to the CLUT. A gamma-2.2 curve
/// gives `entry[i] = (i/255)^(1/2.2) * 255`.
fn build_nonlinear_cmyk_to_lab_lut8_profile() -> Vec<u8> {
    let in_chan: u8 = 4;
    let out_chan: u8 = 3;
    let grid: u8 = 2;
    let mut lut = Vec::with_capacity(2048);

    lut.extend_from_slice(&0x6d66_7431u32.to_be_bytes()); // 'mft1'
    lut.extend_from_slice(&0u32.to_be_bytes()); // reserved
    lut.push(in_chan);
    lut.push(out_chan);
    lut.push(grid);
    lut.push(0);

    // Identity matrix (CMYK input ignores matrix per qcms but we still
    // need to emit it).
    let identity: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        lut.extend_from_slice(&(v as u32).to_be_bytes());
    }

    // Input tables — gamma-2.2 forward curve per channel. This is the
    // non-linearity that makes the profile divergent under
    // convert-first vs composite-first.
    //
    // Per qcms's iccread of `mft1` (ICC.1:2004-10 §10.8), the input
    // table is 256 bytes per channel. qcms interprets each entry as a
    // u8 in 0..=255 sampled across the input domain [0, 1] via
    // `lut_interp_linear_float`. Writing entry[i] = ((i/255)^(1/2.2) *
    // 255) gives a gamma-2.2 forward curve that lifts mid-tones.
    for _ in 0..in_chan {
        for i in 0..256u16 {
            let v = ((i as f64) / 255.0).powf(1.0 / 2.2);
            let byte = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            lut.push(byte);
        }
    }

    // CLUT: 2^4 = 16 grid points × 3 output channels. Corner ordering
    // follows qcms's `CLU` function (chain.rs:300-302) where the index
    // is `x * x_stride + y * y_stride + z * z_stride + w` with strides
    // `x_stride = grid^3`, `y_stride = grid^2`, `z_stride = grid`, `w`
    // = stride 1. The first input channel (C) thus walks the
    // outermost dimension.
    //
    // We position the corners so that "no ink" (0,0,0,0) → Lab(L=255,
    // a=128, b=128) (white) and "full ink" (255,255,255,255) →
    // Lab(L=0, a=128, b=128) (black). Linear interpolation between
    // corners in the CLUT body is multilinear, but the input gamma
    // curve above makes the overall mapping non-linear.
    let grid_size = (grid as usize).pow(in_chan as u32);
    for idx in 0..grid_size {
        // idx bits give (C, M, Y, K) at the corner positions.
        // qcms's CLU stride order is (x = first channel = C outermost,
        // w = last channel = K innermost). So idx = c*8 + m*4 + y*2 + k.
        let c = (idx >> 3) & 1;
        let m = (idx >> 2) & 1;
        let y = (idx >> 1) & 1;
        let k = idx & 1;
        let total = c + m + y + k;
        // L decreases as total ink increases: 0 ink → L byte 255,
        // 4 ink → L byte 0.
        let l_byte = (255 - total * 63).min(255) as u8;
        lut.push(l_byte);
        lut.push(128); // a* = 0
        lut.push(128); // b* = 0
    }

    // Output tables — identity 0..=255.
    for _ in 0..out_chan {
        for i in 0..256u16 {
            lut.push(i as u8);
        }
    }

    let mut profile = vec![0u8; 128];
    let total_size: u32 = 128 + 4 + 12 + lut.len() as u32;
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes()); // v2
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes()); // intent perceptual
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes()); // X 0.9642
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // Y 1.0
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes()); // Z 0.8249

    profile.extend_from_slice(&1u32.to_be_bytes()); // tag count
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes()); // 'A2B0'
    profile.extend_from_slice(&144u32.to_be_bytes()); // offset
    profile.extend_from_slice(&(lut.len() as u32).to_be_bytes()); // size
    profile.extend_from_slice(&lut);

    profile
}

/// Build a one-page PDF with a content stream, optional resource-dict
/// fragment, and extra indirect objects starting at object 5. When
/// `icc_profile` is `Some`, the catalog declares an `/OutputIntents`
/// array referencing object 5 (the ICC profile stream), and extra
/// objects start at 6.
fn build_pdf_with_optional_output_intent(
    content: &str,
    resources_inner: &str,
    extra_objs: &[&str],
    icc_profile: Option<&[u8]>,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");

    let cat_off = buf.len();
    let catalog = if icc_profile.is_some() {
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic Non-Linear CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n".to_string()
    } else {
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_string()
    };
    buf.extend_from_slice(catalog.as_bytes());

    let pages_off = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let page_off = buf.len();
    let page = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << {} >> /Contents 4 0 R >>\nendobj\n",
        resources_inner
    );
    buf.extend_from_slice(page.as_bytes());

    let stream_off = buf.len();
    let stream_hdr = format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len());
    buf.extend_from_slice(stream_hdr.as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let mut extra_offs: Vec<usize> = Vec::new();

    let mut next_obj_num = 5;
    if let Some(icc) = icc_profile {
        extra_offs.push(buf.len());
        let icc_hdr = format!("{} 0 obj\n<< /N 4 /Length {} >>\nstream\n", next_obj_num, icc.len());
        buf.extend_from_slice(icc_hdr.as_bytes());
        buf.extend_from_slice(icc);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
        next_obj_num += 1;
    }

    for obj in extra_objs {
        extra_offs.push(buf.len());
        // Caller emits the object with its own leading number — we
        // assume the caller numbered them starting at `next_obj_num`.
        let _ = next_obj_num;
        buf.extend_from_slice(obj.as_bytes());
    }

    let xref_off = buf.len();
    let total_objs = 4 + extra_offs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off] {
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

fn render_rgba(pdf_bytes: Vec<u8>) -> Vec<u8> {
    let doc = PdfDocument::from_bytes(pdf_bytes).expect("synthetic PDF parses");
    let opts = RenderOptions::with_dpi(72).as_raw();
    let img = render_page(&doc, 0, &opts).expect("render_page succeeds");
    assert_eq!(img.format, ImageFormat::RawRgba8);
    assert_eq!(img.width, 100);
    assert_eq!(img.height, 100);
    img.data
}

fn pixel_at(rgba: &[u8], x: u32, y: u32) -> (u8, u8, u8, u8) {
    let off = ((y * 100 + x) * 4) as usize;
    (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3])
}

fn mean_rgb(rgba: &[u8], x_min: u32, x_max: u32, y_min: u32, y_max: u32) -> (f32, f32, f32) {
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;
    let mut n = 0u32;
    for y in y_min..y_max {
        for x in x_min..x_max {
            let (r, g, b, _) = pixel_at(rgba, x, y);
            r_sum += r as u32;
            g_sum += g as u32;
            b_sum += b as u32;
            n += 1;
        }
    }
    let n = n as f32;
    (r_sum as f32 / n, g_sum as f32 / n, b_sum as f32 / n)
}

// ===========================================================================
// Sanity: the non-linear ICC fixture is non-degenerate
// ===========================================================================
//
// Before relying on the non-linear ICC to surface convert-first vs
// composite-first divergence, prove the profile actually maps distinct
// CMYK inputs to distinct RGB outputs and is non-linear in at least one
// channel. Two single-paint renders at CMYK(0,0,0,0) and
// CMYK(0.5,0.5,0.5,0.5) must produce visibly different RGB.

fn fixture_nonlinear_icc_single_cmyk(c: f32, m: f32, y: f32, k: f32) -> Vec<u8> {
    let content = format!("{c} {m} {y} {k} k\n10 10 80 80 re\nf\n");
    let profile = build_nonlinear_cmyk_to_lab_lut8_profile();
    build_pdf_with_optional_output_intent(&content, "", &[], Some(&profile))
}

#[test]
fn nonlinear_icc_distinct_cmyk_yields_distinct_rgb() {
    let r0 = render_rgba(fixture_nonlinear_icc_single_cmyk(0.0, 0.0, 0.0, 0.0));
    let r_full = render_rgba(fixture_nonlinear_icc_single_cmyk(1.0, 1.0, 1.0, 1.0));
    let r_half = render_rgba(fixture_nonlinear_icc_single_cmyk(0.5, 0.5, 0.5, 0.5));
    let (r_a, g_a, b_a) = mean_rgb(&r0, 30, 70, 30, 70);
    let (r_b, g_b, b_b) = mean_rgb(&r_full, 30, 70, 30, 70);
    let (r_c, g_c, b_c) = mean_rgb(&r_half, 30, 70, 30, 70);
    // The three samples must be distinguishable.
    let delta_full = (r_a - r_b).abs() + (g_a - g_b).abs() + (b_a - b_b).abs();
    let delta_half_to_zero = (r_a - r_c).abs() + (g_a - g_c).abs() + (b_a - b_c).abs();
    let delta_half_to_full = (r_b - r_c).abs() + (g_b - g_c).abs() + (b_b - b_c).abs();
    assert!(
        delta_full > 50.0,
        "non-linear ICC must drive CMYK(0,0,0,0)→white vs CMYK(1,1,1,1)→dark; \
         got delta {delta_full:.1} between ({r_a:.0},{g_a:.0},{b_a:.0}) and \
         ({r_b:.0},{g_b:.0},{b_b:.0})"
    );
    assert!(
        delta_half_to_zero > 20.0 && delta_half_to_full > 20.0,
        "non-linear ICC: 50% CMYK should not equal 0% or 100% CMYK; got \
         half=({r_c:.0},{g_c:.0},{b_c:.0}), 0={r_a:.0}, full={r_b:.0}"
    );
}

// ===========================================================================
// Gap 1 — compose-before-convert under a NON-LINEAR ICC OutputIntent
// ===========================================================================
//
// The probe builds two PDFs:
//
//   A. Two CMYK paints with /ca 0.5 on the upper one, declaring the
//      non-linear ICC profile as /OutputIntents.
//   B. Same paints, no /OutputIntents (additive-clamp fallback).
//
// Convert-first ordering (current pdf_oxide behaviour):
//
//   for each paint:
//     CMYK → RGB via ICC at paint-resolution time
//     SourceOver alpha-blend in RGB pixmap
//
// Compose-first ordering (spec-correct per §11.4 + Annex G):
//
//   for each paint:
//     accumulate CMYK in source space (SourceOver in CMYK)
//   single CMYK → RGB conversion via ICC at the end
//
// Under a non-linear ICC, `ICC(α·A + (1-α)·B) ≠ α·ICC(A) +
// (1-α)·ICC(B)` because the input curves are not identity. The
// difference between the convert-first and composite-first results is
// the test signal the round-2 agent claimed didn't exist for any
// fixture they could build.
//
// The probe samples the OVERLAP region and asserts the rendered output
// matches the compose-first expected value (the spec-correct one). If
// the implementation is convert-first (as today), the rendered output
// matches the convert-first formula and DIFFERS from the expected
// compose-first value — the probe fails, surfacing the gap.

fn fixture_nonlinear_icc_two_overlapping_cmyk_paints() -> Vec<u8> {
    // Lower paint: CMYK(0, 0, 0, 0) — no ink, fully white through the
    // non-linear ICC. Upper paint at /ca 0.5: CMYK(1, 1, 1, 1) — full
    // ink, dark through the non-linear ICC.
    //
    // Overlap composite-first: source-over in CMYK at α=0.5 gives
    //   composited CMYK = 0.5·(1,1,1,1) + 0.5·(0,0,0,0) = (0.5, 0.5, 0.5, 0.5)
    // → through the non-linear ICC at the CMYK(0.5, 0.5, 0.5, 0.5)
    //   tetrahedral interpolation, where input curves apply gamma-2.2
    //   to each 0.5 byte (0.5^(1/2.2) ≈ 0.73) before the CLUT lookup.
    //
    // Overlap convert-first (current code): convert each paint
    // separately, then blend in RGB.
    //   convert(CMYK(0,0,0,0)) = RGB(white) ≈ (255, 255, 255)
    //   convert(CMYK(1,1,1,1)) = RGB(black) ≈ (0, 0, 0)
    //   blend at α=0.5 = ((0+255)/2, (0+255)/2, (0+255)/2) = (~128, ~128, ~128)
    //
    // The compose-first expected value depends on the precise
    // gamma-2.2 + multilinear CLUT computation; we capture it by
    // computing what the same ICC produces for a single-paint
    // CMYK(0.5, 0.5, 0.5, 0.5) (the composited CMYK quadruple). If
    // the implementation is composite-first, the overlap region's
    // rendered RGB equals the single-paint render's RGB at that
    // quadruple. If convert-first (current code), it equals the
    // RGB-blend value ~(128, 128, 128).
    let content = "0 0 0 0 k\n10 10 80 80 re\nf\n\
                   /Half gs\n\
                   1 1 1 1 k\n\
                   20 20 60 60 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >>";
    let profile = build_nonlinear_cmyk_to_lab_lut8_profile();
    build_pdf_with_optional_output_intent(content, resources, &[], Some(&profile))
}

/// IGNORED — pins the compose-first vs convert-first divergence under
/// a non-linear ICC OutputIntent. As-shipped (convert-first), the
/// overlap region shows the RGB-blend of pre-converted paints. Spec-
/// correct (compose-first) would show the ICC-converted value of the
/// composited CMYK.
///
/// **TEST SIGNAL**: this probe FAILS at HEAD precisely when the
/// implementation is convert-first; it PASSES when composite-first is
/// landed. The agent's claim "no observable test signal" is rebutted by
/// this fixture if and only if the fixture's CMYK(0.5,0.5,0.5,0.5)
/// single-paint render produces a value distinct from the overlap-blend
/// value.
#[test]
fn qa_round2_compose_before_convert_under_nonlinear_icc() {
    let rgba_two = render_rgba(fixture_nonlinear_icc_two_overlapping_cmyk_paints());
    let rgba_composited = render_rgba(fixture_nonlinear_icc_single_cmyk(0.5, 0.5, 0.5, 0.5));

    // Overlap region centre — PDF (40, 40) → image (40, 60) (PDF y=40,
    // image y=100-40 = 60). Sample a 20×20 mean to swamp AA noise.
    let (or_mean_r, or_mean_g, or_mean_b) = mean_rgb(&rgba_two, 35, 65, 35, 65);
    let (cs_mean_r, cs_mean_g, cs_mean_b) = mean_rgb(&rgba_composited, 35, 65, 35, 65);

    // The compose-first expected value is the single-paint render of
    // CMYK(0.5, 0.5, 0.5, 0.5). The convert-first actual value blends
    // RGB(white) with RGB(black) in RGB, giving ~(128, 128, 128).
    //
    // Under a non-linear ICC, these MUST differ — otherwise the
    // round-2 agent's deferral claim ("compose-first vs convert-first
    // are byte-identical") would be correct.
    // BYTE-EXACT reference. The round-2 QA agent originally pinned this
    // with a triple-channel L1 sum < 15.0 tolerance; round-3 QA
    // hand-derived the byte-exact value by reading the agent's failure
    // output at parent SHA 5585ce4 — at convert-first HEAD, the overlap
    // measured (129, 129, 129) and the single-paint reference (66, 66,
    // 66), a 189-byte L1 delta. The "≈ 66" value is what the non-linear
    // ICC produces for CMYK(0.5, 0.5, 0.5, 0.5): gamma-2.2 input curves
    // raise each 0.5 byte to ≈ 0.728 (the qcms 256-entry table sample),
    // multilinear interp over the 2⁴ CLUT corners L = 255 − 63·(c+m+y+k)
    // gives ≈ 255 − 252·x; the qcms tetrahedral path lands every pixel
    // in the 30×30 sample on byte 66 exactly. The 30×30 mean is exactly
    // 66.0 on every channel — no AA noise inside the overlap region.
    //
    // The previous `compose_first_delta < 15.0` tolerance is replaced
    // by an exact-equality assertion on the integer mean. The
    // convert-first reference at (128, 128, 128) is preserved as a
    // discrimination check — should the implementation regress, we want
    // to know whether it landed on convert-first or some third value.
    let or_int_r = or_mean_r.round() as i32;
    let or_int_g = or_mean_g.round() as i32;
    let or_int_b = or_mean_b.round() as i32;
    let cs_int_r = cs_mean_r.round() as i32;
    let cs_int_g = cs_mean_g.round() as i32;
    let cs_int_b = cs_mean_b.round() as i32;

    // Single-paint reference is byte-exact 66/66/66 because the
    // non-linear ICC fixture is deterministic and the 30×30 sample is
    // entirely inside the painted rect.
    assert_eq!(
        (cs_int_r, cs_int_g, cs_int_b),
        (66, 66, 66),
        "single-paint reference under non-linear ICC must be byte-exact \
         RGB(66, 66, 66); got ({cs_int_r}, {cs_int_g}, {cs_int_b}). \
         Fixture drift — re-derive the reference from the curve+CLUT \
         tables."
    );

    // Compose-first impl must hit the same byte-exact reference. The
    // CMYK source-space alpha blend of (0,0,0,0) and (1,1,1,1) at α=0.5
    // is exactly CMYK(0.5, 0.5, 0.5, 0.5), and the ICC conversion is
    // deterministic — so byte-exact equality holds.
    assert_eq!(
        (or_int_r, or_int_g, or_int_b),
        (66, 66, 66),
        "ISO 32000-1 §11.4 compose-first overlap under non-linear ICC: \
         expected byte-exact RGB(66, 66, 66) (single-paint reference). Got \
         ({or_int_r}, {or_int_g}, {or_int_b}); single-paint reference \
         ({cs_int_r}, {cs_int_g}, {cs_int_b}); convert-first reference \
         (128, 128, 128)."
    );
}

// ===========================================================================
// SMask + overprint paint-arm coverage matrix
// ===========================================================================
//
// The round-2 impl wires soft-mask + overprint correction ONLY on
// Operator::Fill (`f`) and Operator::Stroke (`S`). Every other paint
// operator continues to take the direct path that the round-1 audit
// proved drops SMask + overprint state. We pin each uncovered arm with
// a probe that exercises that operator under an active /SMask or
// /op-true ExtGState. Each probe is `#[ignore]`-marked with the
// matching HONEST_GAP constant; the round-3 fix lifts the ignore.
//
// Each fixture follows the same template:
//
//   1. White background fill (Operator::Fill, the path that IS wired).
//   2. Push ExtGState declaring /SMask or /op true.
//   3. Run the target paint operator that should be modulated.
//
// The assertion checks that the destination pixel reflects the SMask
// or overprint effect, which it will not as-shipped.

fn fixture_smask_for_op(op_ops: &str) -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    let content = format!(
        "1 1 1 rg\n0 0 100 100 re\nf\n\
         /Sm gs\n\
         1 0 0 rg\n\
         1 0 0 RG\n5 w\n\
         {}\n",
        op_ops
    );
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >>";
    build_pdf_with_optional_output_intent(&content, resources, &[&obj_5], None)
}

/// IGNORED — SMask on `B` (fill+stroke). The Fill arm IS wired but
/// `B` takes the FillStroke branch which is unwired.
#[test]
fn qa_round2_smask_modulates_fill_stroke_combo() {
    let pdf = fixture_smask_for_op("20 20 60 60 re\nB\n");
    let rgba = render_rgba(pdf);
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // SMask /S /Luminosity with 50% grey form. BT.601 luminance of
    // (0.5, 0.5, 0.5) = 0.30·0.5 + 0.59·0.5 + 0.11·0.5 = 0.5.
    // Modulated alpha m = 127/255 (after byte-round of 0.5·255).
    // dest = m·painted + (1-m)·snapshot = (127/255)·(255,0,0) +
    // (128/255)·(255,255,255) → channel-by-channel byte rounds to
    // (255, 127, 127). The byte-exact reference is what the
    // apply_smask_after_paint loop emits; any drift in the modulation
    // path surfaces as a value change.
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "B (FillStroke) under SMask /Luminosity 50% grey form: \
         expected byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_FILLSTROKE_NOT_WIRED
    );
}

/// IGNORED — SMask on `B*` (fill+stroke EvenOdd).
#[test]
fn qa_round2_smask_modulates_fill_stroke_evenodd_combo() {
    let pdf = fixture_smask_for_op("20 20 60 60 re\nB*\n");
    let rgba = render_rgba(pdf);
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "B* (FillStrokeEvenOdd) under SMask /Luminosity 50%: \
         expected byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_FILLSTROKE_EVENODD_NOT_WIRED
    );
}

/// IGNORED — SMask on `b` (close+fill+stroke).
#[test]
fn qa_round2_smask_modulates_close_fill_stroke_combo() {
    // Use a path that needs closing — moveto + lineto + lineto + b.
    let pdf = fixture_smask_for_op("20 20 m\n80 20 l\n80 80 l\n20 80 l\nb\n");
    let rgba = render_rgba(pdf);
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "b (CloseFillStroke) under SMask /Luminosity 50%: \
         expected byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_CLOSE_FILLSTROKE_NOT_WIRED
    );
}

/// IGNORED — SMask on `b*` (close+fill+stroke EvenOdd).
#[test]
fn qa_round2_smask_modulates_close_fill_stroke_evenodd_combo() {
    let pdf = fixture_smask_for_op("20 20 m\n80 20 l\n80 80 l\n20 80 l\nb*\n");
    let rgba = render_rgba(pdf);
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "b* (CloseFillStrokeEvenOdd) under SMask /Luminosity 50%: \
         expected byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_CLOSE_FILLSTROKE_EVENODD_NOT_WIRED
    );
}

/// IGNORED — SMask on `f*` (fill EvenOdd).
#[test]
fn qa_round2_smask_modulates_fill_evenodd() {
    let pdf = fixture_smask_for_op("20 20 60 60 re\nf*\n");
    let rgba = render_rgba(pdf);
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "f* (FillEvenOdd) under SMask /Luminosity 50%: \
         expected byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_FILL_EVENODD_NOT_WIRED
    );
}

fn fixture_overprint_for_op(op_ops: &str) -> Vec<u8> {
    // CMYK backdrop fill (cyan 50%) then the target operator paints
    // yellow with overprint on. With overprint, the overlap should
    // retain the cyan plate. Without (as-shipped on uncovered arms),
    // the yellow knocks the cyan out completely.
    let content = format!(
        "0.5 0 0 0 k\n10 10 80 80 re\nf\n\
         /OpOn gs\n\
         0 0 1 0 k\n\
         0 0 1 0 K\n5 w\n\
         {}\n",
        op_ops
    );
    let resources = "/ExtGState << /OpOn << /Type /ExtGState /op true /OP true /OPM 1 >> >>";
    build_pdf_with_optional_output_intent(&content, resources, &[], None)
}

fn fixture_no_overprint_for_op(op_ops: &str) -> Vec<u8> {
    let content = format!(
        "0.5 0 0 0 k\n10 10 80 80 re\nf\n\
         0 0 1 0 k\n\
         0 0 1 0 K\n5 w\n\
         {}\n",
        op_ops
    );
    build_pdf_with_optional_output_intent(&content, "", &[], None)
}

#[test]
fn qa_round2_overprint_modulates_fill_stroke_combo() {
    let with_op = render_rgba(fixture_overprint_for_op("30 30 50 50 re\nB\n"));
    let no_op = render_rgba(fixture_no_overprint_for_op("30 30 50 50 re\nB\n"));
    let (r_op, g_op, b_op) = mean_rgb(&with_op, 40, 60, 40, 60);
    let (r_no, g_no, b_no) = mean_rgb(&no_op, 40, 60, 40, 60);
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 30.0,
        "B (FillStroke) overprint vs no-overprint delta: expected > 30, got \
         {delta:.1} between ({r_op:.0},{g_op:.0},{b_op:.0}) and \
         ({r_no:.0},{g_no:.0},{b_no:.0}). {}",
        HONEST_GAP_OVERPRINT_FILLSTROKE_NOT_WIRED
    );
}

#[test]
fn qa_round2_overprint_modulates_fill_stroke_evenodd_combo() {
    let with_op = render_rgba(fixture_overprint_for_op("30 30 50 50 re\nB*\n"));
    let no_op = render_rgba(fixture_no_overprint_for_op("30 30 50 50 re\nB*\n"));
    let (r_op, g_op, b_op) = mean_rgb(&with_op, 40, 60, 40, 60);
    let (r_no, g_no, b_no) = mean_rgb(&no_op, 40, 60, 40, 60);
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 30.0,
        "B* overprint vs no-overprint delta: expected > 30, got {delta:.1}. {}",
        HONEST_GAP_OVERPRINT_FILLSTROKE_EVENODD_NOT_WIRED
    );
}

#[test]
fn qa_round2_overprint_modulates_close_fill_stroke_combo() {
    let with_op = render_rgba(fixture_overprint_for_op("30 30 m\n80 30 l\n80 80 l\n30 80 l\nb\n"));
    let no_op = render_rgba(fixture_no_overprint_for_op("30 30 m\n80 30 l\n80 80 l\n30 80 l\nb\n"));
    let (r_op, g_op, b_op) = mean_rgb(&with_op, 40, 60, 40, 60);
    let (r_no, g_no, b_no) = mean_rgb(&no_op, 40, 60, 40, 60);
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 30.0,
        "b overprint delta: expected > 30, got {delta:.1}. {}",
        HONEST_GAP_OVERPRINT_CLOSE_FILLSTROKE_NOT_WIRED
    );
}

#[test]
fn qa_round2_overprint_modulates_close_fill_stroke_evenodd_combo() {
    let with_op = render_rgba(fixture_overprint_for_op("30 30 m\n80 30 l\n80 80 l\n30 80 l\nb*\n"));
    let no_op =
        render_rgba(fixture_no_overprint_for_op("30 30 m\n80 30 l\n80 80 l\n30 80 l\nb*\n"));
    let (r_op, g_op, b_op) = mean_rgb(&with_op, 40, 60, 40, 60);
    let (r_no, g_no, b_no) = mean_rgb(&no_op, 40, 60, 40, 60);
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 30.0,
        "b* overprint delta: expected > 30, got {delta:.1}. {}",
        HONEST_GAP_OVERPRINT_CLOSE_FILLSTROKE_EVENODD_NOT_WIRED
    );
}

#[test]
fn qa_round2_overprint_modulates_fill_evenodd() {
    let with_op = render_rgba(fixture_overprint_for_op("30 30 50 50 re\nf*\n"));
    let no_op = render_rgba(fixture_no_overprint_for_op("30 30 50 50 re\nf*\n"));
    let (r_op, g_op, b_op) = mean_rgb(&with_op, 40, 60, 40, 60);
    let (r_no, g_no, b_no) = mean_rgb(&no_op, 40, 60, 40, 60);
    let delta = (r_op - r_no).abs() + (g_op - g_no).abs() + (b_op - b_no).abs();
    assert!(
        delta > 30.0,
        "f* overprint delta: expected > 30, got {delta:.1}. {}",
        HONEST_GAP_OVERPRINT_FILL_EVENODD_NOT_WIRED
    );
}

// ===========================================================================
// SMask scope through q/Q
// ===========================================================================
//
// Per §11.4.7, ExtGState /SMask is graphics-state — q pushes a copy, Q
// pops back to the prior state. After Q, any /SMask in the popped
// scope MUST be inactive. The round-2 impl rides on the
// GraphicsStateStack's `push` / `pop` (which deep-clones the state on
// push, restoring on pop). The agent flagged this as "correct but
// unprobed."

fn fixture_smask_scoped_through_q_then_paint_outside() -> Vec<u8> {
    let form_content = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        form_content.len(),
        form_content
    );
    // White background, then `q` push, /Sm gs, paint inside scope (red
    // through SMask → faded red ~(255, 128, 128)), `Q` pop, paint
    // again outside scope (red WITHOUT SMask → fully opaque red).
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   q\n\
                   /Sm gs\n\
                   1 0 0 rg\n\
                   10 10 30 30 re\nf\n\
                   Q\n\
                   1 0 0 rg\n\
                   60 60 30 30 re\nf\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >>";
    build_pdf_with_optional_output_intent(content, resources, &[&obj_5], None)
}

/// Pin: after `Q` pops the gstate that declared `/Sm gs`, the
/// subsequent paint must render WITHOUT SMask modulation. Inside the
/// scope: faded red. Outside the scope (post-Q): fully opaque red.
///
/// As-shipped this should PASS — the GraphicsStateStack pop restores
/// the prior `gs.smask = None`. If it FAILS, SMask state leaks
/// across q/Q and that's a real bug.
#[test]
fn qa_round2_smask_does_not_leak_across_q_q() {
    let rgba = render_rgba(fixture_smask_scoped_through_q_then_paint_outside());
    // Inside-scope sample: image (25, 75) (PDF y=10..40 → image y=60..90).
    let (r_in, g_in, b_in, _) = pixel_at(&rgba, 25, 75);
    // Outside-scope sample: image (75, 25) (PDF y=60..90 → image y=10..40).
    let (r_out, g_out, b_out, _) = pixel_at(&rgba, 75, 25);

    // Inside the SMask scope, red is faded by the 50% luminance
    // modulation to byte-exact (255, 127, 127) — the same reference
    // the paint-arm coverage probes hit.
    assert_eq!(
        (r_in, g_in, b_in),
        (255, 127, 127),
        "inside SMask scope (q ... /Sm gs ... paint ... Q): expected \
         byte-exact faded red (255, 127, 127); got ({r_in}, {g_in}, \
         {b_in})"
    );
    // Outside the SMask scope (post-Q), red is fully opaque. The
    // paint-arm coverage path emits byte-exact (255, 0, 0) for an
    // unmodulated red fill.
    assert_eq!(
        (r_out, g_out, b_out),
        (255, 0, 0),
        "outside SMask scope (post-Q): expected byte-exact fully \
         opaque red (255, 0, 0); got ({r_out}, {g_out}, {b_out}). If \
         this fails, SMask state leaks across q/Q boundaries — a real \
         bug."
    );
}

// ===========================================================================
// Composite overprint reconstruction loss under non-linear ICC
// ===========================================================================
//
// The round-2 composite overprint correction uses the destination RGB
// snapshot, inverts via additive-clamp (RGB→CMYK), applies the §11.7.4
// plate selection, then converts back to RGB. When the snapshot's RGB
// came from a non-trivial ICC OutputIntent, the additive-clamp
// inversion can't recover the original CMYK — the inversion is
// lossy. The probe pins the magnitude of the loss.

fn fixture_overprint_under_nonlinear_icc() -> Vec<u8> {
    let content = "0.5 0 0 0 k\n10 10 60 60 re\nf\n\
                   /OpOn gs\n\
                   0 0 1 0 k\n\
                   30 30 60 60 re\nf\n";
    let resources = "/ExtGState << /OpOn << /Type /ExtGState /op true /OP true /OPM 1 >> >>";
    let profile = build_nonlinear_cmyk_to_lab_lut8_profile();
    build_pdf_with_optional_output_intent(content, resources, &[], Some(&profile))
}

fn fixture_overprint_under_no_icc() -> Vec<u8> {
    let content = "0.5 0 0 0 k\n10 10 60 60 re\nf\n\
                   /OpOn gs\n\
                   0 0 1 0 k\n\
                   30 30 60 60 re\nf\n";
    let resources = "/ExtGState << /OpOn << /Type /ExtGState /op true /OP true /OPM 1 >> >>";
    build_pdf_with_optional_output_intent(content, resources, &[], None)
}

// ===========================================================================
// Round-3 probes — SMask coverage on PaintShading + Do paint arms
// ===========================================================================
//
// Builds on the round-2 paint-arm coverage matrix. The round-2 QA
// covered the path-painting arms (`B`, `B*`, `b`, `b*`, `f*`) and
// pinned each gap with a probe. The round-3 wiring extends the
// snapshot/apply cycle to PaintShading (`sh`) and Do (`Do`); these
// probes verify the wiring fires.
//
// Text-showing arms (`Tj`, `TJ`, `'`, `"`) are wired in the same
// round but fixture-side probing requires a font resource, which
// the synthetic-PDF builder above does not yet emit. A follow-up
// round can add font-bearing fixtures and pin the text-showing
// SMask + overprint behaviour.

/// Paint a 100×100 SMask form (50% grey ⇒ luminosity 0.5) and a
/// separate Form XObject (red 100×100 fill) so the page's `Do`
/// invocation paints opaque red modulated by the SMask.
fn fixture_smask_for_do_form_xobject() -> Vec<u8> {
    let smask_form = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        smask_form.len(),
        smask_form
    );
    let do_form = "1 0 0 rg\n20 20 60 60 re\nf\n";
    let obj_6 = format!(
        "6 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        do_form.len(),
        do_form
    );
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   /Sm gs\n\
                   /Fm1 Do\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >> \
                     /XObject << /Fm1 6 0 R >>";
    build_pdf_with_optional_output_intent(content, resources, &[&obj_5, &obj_6], None)
}

/// SMask must modulate the painted Form XObject invoked through `Do`.
/// The `Operator::Do` arm now snapshots before invoking the Form and
/// runs `apply_smask_after_paint` against that snapshot, so the
/// painted Form goes through the active soft mask. This probe pins
/// the byte-exact post-modulation pixel and guards against a
/// regression that re-introduces the pre-fix path where `Do` painted
/// opaquely through the mask.
#[test]
fn qa_round3_smask_modulates_do_form_xobject() {
    let rgba = render_rgba(fixture_smask_for_do_form_xobject());
    // Painted region (PDF 20..80, 20..80) ⇒ image (20..80, 20..80).
    // Sample centre.
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    // SMask /S /Luminosity with 50% grey form. Red painted through
    // 0.5 modulation onto white yields byte-exact (255, 127, 127).
    // Without wiring, Do paints fully opaque red ⇒ (255, 0, 0).
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "Do (Form XObject) under SMask /Luminosity 50%: expected \
         byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_DO_NOT_WIRED
    );
}

/// Axial shading from red (C0) to red (C1) — a uniform red fill, so
/// the painted output is independent of the gradient interpolator.
/// Combined with an SMask /S /Luminosity 50% grey form, the painted
/// region must modulate to ~(255, 128, 128).
fn fixture_smask_for_paint_shading() -> Vec<u8> {
    let smask_form = "0.5 g\n0 0 100 100 re\nf\n";
    let obj_5 = format!(
        "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] \
         /Resources << >> /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        smask_form.len(),
        smask_form
    );
    // Axial shading from red (1,0,0) to red (1,0,0) covering the page.
    // Uniform red — any interpolation produces red.
    let obj_6 = "6 0 obj\n<< /ShadingType 2 /ColorSpace /DeviceRGB \
                 /Coords [0 0 100 0] \
                 /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [1 0 0] /N 1 >> >>\nendobj\n";
    let content = "1 1 1 rg\n0 0 100 100 re\nf\n\
                   q\n20 20 60 60 re\nW n\n\
                   /Sm gs\n\
                   /Sh1 sh\n\
                   Q\n";
    let resources = "/ExtGState << /Sm << /Type /ExtGState \
                     /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >> \
                     /Shading << /Sh1 6 0 R >>";
    build_pdf_with_optional_output_intent(content, resources, &[&obj_5, obj_6], None)
}

/// SMask must modulate the shading paint. The shading is uniform red;
/// under a 50%-grey luminosity SMask the painted region must show
/// ~(255, 128, 128). At HEAD the `Operator::PaintShading` arm
/// bypasses the smask_snapshot / apply_smask_after_paint cycle, so
/// the shading paints through unmodulated.
#[test]
fn qa_round3_smask_modulates_paint_shading() {
    let rgba = render_rgba(fixture_smask_for_paint_shading());
    // Painted region (PDF 20..80, 20..80) ⇒ image (20..80, 20..80).
    let (r, g, b, _) = pixel_at(&rgba, 50, 50);
    assert_eq!(
        (r, g, b),
        (255, 127, 127),
        "PaintShading (sh) under SMask /Luminosity 50%: expected \
         byte-exact (255, 127, 127); got ({r}, {g}, {b}). {}",
        HONEST_GAP_SMASK_PAINT_SHADING_NOT_WIRED
    );
}

/// Composite overprint under a non-trivial ICC OutputIntent. The
/// round-2 path snapshotted post-paint RGB, inverted to CMYK via
/// additive-clamp, applied §11.7.4 plate merge, and re-converted
/// through `cmyk_to_rgb` (the additive-clamp fallback). When the
/// backdrop pixel came through a non-linear ICC, the additive-clamp
/// inversion is lossy and the re-converted RGB drifts off the
/// press-accurate value. The Priority-4 CMYK-plate-retention fix
/// keeps the backdrop CMYK quadruple resident through the page
/// composite so the overprint merge sees the real CMYK and the
/// post-merge ICC conversion lands on the press-accurate RGB.
///
/// Reference: single-paint render of CMYK(0.5, 0, 1, 0) at full
/// opacity through the same ICC. That's the OPM=1 plate merge of
/// cyan-50% backdrop and yellow-100% overprint (zero source plates
/// preserve dest; non-zero source plates replace dest); the resulting
/// CMYK quadruple, run through the OutputIntent ICC once, is what
/// the press sees.
#[test]
fn qa_round2_overprint_reconstruction_under_nonlinear_icc() {
    let rgba_icc = render_rgba(fixture_overprint_under_nonlinear_icc());
    // Press-accurate single-paint reference: OPM=1 plate merge of
    // cyan 0.5 and yellow 1.0 = CMYK(0.5, 0, 1, 0). The 5%/95% range
    // (centre of overlap) is uniformly inside the painted rect.
    let rgba_ref = render_rgba(fixture_nonlinear_icc_single_cmyk(0.5, 0.0, 1.0, 0.0));

    let (r_icc, g_icc, b_icc) = mean_rgb(&rgba_icc, 40, 60, 40, 60);
    let (r_ref, g_ref, b_ref) = mean_rgb(&rgba_ref, 40, 60, 40, 60);

    let actual = (r_icc.round() as i32, g_icc.round() as i32, b_icc.round() as i32);
    let press = (r_ref.round() as i32, g_ref.round() as i32, b_ref.round() as i32);

    // Press-accurate: actual == reference. Any delta is reconstruction
    // loss. The Priority-4 plate-retention fix drives delta to zero.
    assert_eq!(
        actual, press,
        "ISO 32000-1 §11.7.4.3 CompatibleOverprint under non-linear ICC \
         must hit the press-accurate single-paint reference; got \
         overlap={actual:?} vs reference={press:?}"
    );
}

// ===========================================================================
// Compose-first bounded loss when backdrop went through the ICC
// ===========================================================================
//
// Round-3 fix: apply_cmyk_compose_after_paint snapshots the post-paint
// RGB before the transparent paint, inverts via §10.3.5 additive clamp
// to recover CMYK, then composites + re-runs the ICC. When the backdrop
// pixel was produced by an *opaque* prior CMYK paint that ALSO went
// through the non-linear ICC, the inversion is lossy — the round-3
// agent's own commit message admits this. The probe quantifies the
// byte-delta vs the press-accurate compose-first reference (a
// single-paint render of the composed CMYK at full opacity, which is
// what a separation-backend route would produce).
//
// Fixture A: opaque backdrop CMYK(0.5, 0, 0, 0) (cyan 50%), then
// transparent CMYK(0, 0, 0.5, 0) (yellow 50%) at /ca 0.5, both under
// the non-linear ICC. The compose-first impl inverts the cyan-ICC RGB
// back through additive-clamp, composites with the yellow CMYK at
// α=0.5, then re-converts. The composed CMYK should be
// CMYK(0.25, 0, 0.25, 0).
//
// Reference: a single-paint render of CMYK(0.25, 0, 0.25, 0) at full
// opacity through the same ICC. That's the value a press-accurate
// backend (which keeps CMYK plates resident) would land on.

fn fixture_compose_first_with_icc_backdrop() -> Vec<u8> {
    // Backdrop cyan 50% at full opacity, then yellow 50% at /ca 0.5.
    let content = "0.5 0 0 0 k\n10 10 80 80 re\nf\n\
                   /Half gs\n\
                   0 0 0.5 0 k\n\
                   10 10 80 80 re\nf\n";
    let resources = "/ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >>";
    let profile = build_nonlinear_cmyk_to_lab_lut8_profile();
    build_pdf_with_optional_output_intent(content, resources, &[], Some(&profile))
}

/// Compose-first under an ICC-derived backdrop: the round-3
/// apply_cmyk_compose_after_paint inverted the post-ICC backdrop RGB
/// via §10.3.5 additive-clamp, which loses colorimetric information
/// when the backdrop went through a non-linear ICC. The Priority-4
/// CMYK-plate-retention fix keeps the backdrop CMYK quadruple resident
/// so the compose-first path reads CMYK directly instead of inverting
/// RGB.
///
/// Reference: single-paint render of the composed CMYK quadruple
/// (0.25, 0, 0.25, 0) at full opacity through the same ICC. Under the
/// fix, the two-paint render's overlap region matches byte-exact.
#[test]
fn qa_round3_compose_first_under_icc_backdrop_press_accurate() {
    let rgba_two = render_rgba(fixture_compose_first_with_icc_backdrop());
    let rgba_ref = render_rgba(fixture_nonlinear_icc_single_cmyk(0.25, 0.0, 0.25, 0.0));

    let (r_actual, g_actual, b_actual) = mean_rgb(&rgba_two, 35, 65, 35, 65);
    let (r_ref, g_ref, b_ref) = mean_rgb(&rgba_ref, 35, 65, 35, 65);

    let actual = (r_actual.round() as i32, g_actual.round() as i32, b_actual.round() as i32);
    let press = (r_ref.round() as i32, g_ref.round() as i32, b_ref.round() as i32);

    assert_eq!(
        actual, press,
        "ISO 32000-1 §11.4 compose-first under ICC backdrop must hit the \
         press-accurate single-paint reference; got overlap={actual:?} vs \
         reference={press:?}"
    );
}
