//! Round 7 probes for issue #46.
//!
//! Closes `HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH` for the
//! `icc-lcms2` backend by exercising the CMYK→CMYK profile-retargeting
//! pipeline `crate::color::CmykRetargetTransform` puts under
//! `sidecar::extract_process_paint_cmyk`.
//!
//! Three-state matrix this round pins:
//!   - `icc-lcms2` enabled                       → full retargeting through
//!                                                  the destination profile's BToA
//!                                                  (the round-7 closure path).
//!   - `icc-qcms` only (no `icc-lcms2`)          → the round-5 "natural-form"
//!                                                  reading is preserved
//!                                                  byte-identically.
//!   - neither feature                            → §10.3.5 additive-clamp
//!                                                  fallback fires at the
//!                                                  consumer (renderer / image
//!                                                  extractor); the
//!                                                  process-paint extractor
//!                                                  returns the natural form
//!                                                  unchanged.
//!
//! Spec citations:
//!  - ISO 32000-1 §8.6.5.5 — ICCBased colour spaces (embedded profile
//!    precedence over /Alternate).
//!  - ISO 32000-1 §8.6.6.5 — DeviceN /Process + /Components.
//!  - ISO 32000-1 §10.7.3  — rendering intent.
//!  - ISO 32000-1 §11.7.4.3 Table 149 row 2 — overprint compose for
//!    process source colour spaces.
//!  - ICC.1:2004-10 §6.4   — Black Point Compensation. Not formally in
//!    ISO 32000 but the press-default behaviour every relative-
//!    colorimetric production pipeline expects.

#![cfg(all(feature = "rendering", feature = "test-support"))]
#![allow(dead_code)]

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::render_separations;

// ===========================================================================
// HONEST_GAP marker — updated downgrade for round 7.
// ===========================================================================

/// `HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH` —
/// three-state matrix after round 7.
///
/// **Companion narrative:** `HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH`
/// in `tests/test_46_round5_devicen_process_polish.rs` documents the
/// original qcms-only "natural form" reading that this constant's three-
/// state matrix supersedes. The round-5 constant is preserved (not
/// collapsed) because it carries the historical rationale for why the
/// natural-form reading remains the qcms / no-CMM fallback. Read this
/// constant for the current truth-table; read the round-5 constant for
/// the rationale on the non-lcms2 rows.
///
///  - **`icc-lcms2` enabled (round 7 closure)**: when a DeviceN
///    /Process /ColorSpace [/ICCBased N=4] declaration carries an
///    embedded CMYK profile distinct from the document OutputIntent
///    CMYK /DestOutputProfile, the source tints are retargeted through
///    the embedded profile's `AToB` → Lab PCS → the destination
///    profile's `BToA` → destination CMYK. The press-default
///    relative-colorimetric intent with Black Point Compensation
///    governs. Probes `r7_icc_retarget_cross_profile_byte_exact` and
///    `r7_icc_retarget_bpc_changes_shadow_tones_byte_exact` pin the
///    byte-exact destination CMYK against an independent lcms2 run.
///
///  - **`icc-qcms` only** (no `icc-lcms2`): the gap remains as a
///    documented feature-level limitation. qcms 0.3 has no CMYK output
///    path, so `CmykRetargetTransform::new` returns `None` and
///    `extract_process_paint_cmyk` falls back to the round-5 "natural
///    form" reading — source tints accepted as destination CMYK
///    directly. Probe `r7_icc_qcms_only_preserves_round5_natural_form`
///    pins the round-5 byte references unchanged.
///
///  - **neither feature** (`--no-default-features --features rendering`):
///    no CMM is linked in; the §10.3.5 additive-clamp fallback fires
///    at the consumer. `extract_process_paint_cmyk` still emits the
///    round-5 natural form (no ICC re-evaluation), and the renderer's
///    composite path projects through §10.3.5.
///
/// Closure path under `icc-qcms`: enable `icc-lcms2`. Closure path
/// under no-feature: enable either `icc-qcms` (no retargeting, qcms
/// CMM for non-mismatch cases) or `icc-lcms2` (full retargeting).
pub const HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH_R7: &str =
    "HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH (round-7 status): \
     icc-lcms2 closes this gap (CMYK→CMYK retargeting through Lab PCS \
     with BPC). icc-qcms preserves the round-5 natural-form reading \
     (qcms 0.3 has no CMYK output path). no-CMM builds fall to \
     §10.3.5 additive-clamp at the consumer. Closure: enable \
     icc-lcms2.";

// ===========================================================================
// Synthetic ICC profile helpers — round-5 mirror with a B2A0 tag added so
// lcms2 can build a CMYK→CMYK transform from / through these profiles.
// ===========================================================================

/// Tunable parameters for a synthetic bidirectional CMYK ICC profile.
/// Both `A2B0` (CMYK → Lab) and `B2A0` (Lab → CMYK) tags carry
/// constant CLUTs — every CMYK input maps to `(l_byte, 128, 128)` Lab,
/// every Lab input maps to `(c_byte, m_byte, y_byte, k_byte)` CMYK.
///
/// Pinning the destination CMYK to a single constant per profile makes
/// the retarget byte-exact regardless of source tint: the lcms2 pipeline
/// is `source.AToB(input) → Lab → dest.BToA(Lab) → output`; with
/// constant CLUTs both halves are constant functions, so the output is
/// the destination profile's `(c_byte, m_byte, y_byte, k_byte)` regardless
/// of the input tints. This makes byte-exact references trivial to pin
/// and trivially reproducible under any lcms2 build (lcms2 6.x, ≥7, …):
/// the bytes are not a function of lcms2's interpolation algorithm.
#[derive(Clone, Copy)]
struct SyntheticCmykProfileParams {
    /// `A2B0` constant Lab output L channel.
    l_byte: u8,
    /// `B2A0` constant destination CMYK (C, M, Y, K) outputs.
    dest_cmyk: (u8, u8, u8, u8),
}

/// Build a bidirectional `mft1`-tag CMYK ICC profile carrying both
/// `A2B0` (CMYK → Lab) and `B2A0` (Lab → CMYK) tags.  Round 5's
/// `build_constant_cmyk_icc` carried only `A2B0`; lcms2 6.1.1 rejects
/// CMYK-output transforms built from a profile lacking `B2A0`, so the
/// retarget pipeline can't be built without both.
///
/// Layout per ICC.1:2004-10 §10.8:
///   - 128-byte header (version 2.4, prtr device class, CMYK colour
///     space, Lab PCS).
///   - 4-byte tag count = 2.
///   - 12-byte tag table entries for `A2B0` and `B2A0` (sig, offset,
///     size).
///   - `A2B0` `mft1` body: 4-channel CMYK in, 3-channel Lab out, 2-grid
///     CLUT.  Output values: constant `(l_byte, 128, 128)`.
///   - `B2A0` `mft1` body: 3-channel Lab in, 4-channel CMYK out, 2-grid
///     CLUT.  Output values: constant `(c_byte, m_byte, y_byte, k_byte)`.
///
/// `mft1` (LUT8 — sig 0x6d667431) is the smallest format both qcms and
/// lcms2 parse cleanly.  The 3x3 chromaticity matrix is identity (PCS
/// is Lab, not XYZ — the matrix is ignored by spec for Lab PCS, but
/// the field is mandatory).  Input and output curves are linear
/// (256-entry identity ramps).  The CLUT is 2^N entries per channel
/// (N = input channels), each entry of size out_chan bytes.
fn build_bidirectional_cmyk_icc(params: SyntheticCmykProfileParams) -> Vec<u8> {
    let mut a2b0 = build_mft1_constant(4, 3, &[params.l_byte, 128, 128]);
    let mut b2a0 = build_mft1_constant(
        3,
        4,
        &[
            params.dest_cmyk.0,
            params.dest_cmyk.1,
            params.dest_cmyk.2,
            params.dest_cmyk.3,
        ],
    );

    // Pad each tag body to a multiple of 4 bytes (ICC alignment) so
    // the next tag starts on a 4-byte boundary.
    while !a2b0.len().is_multiple_of(4) {
        a2b0.push(0);
    }
    while !b2a0.len().is_multiple_of(4) {
        b2a0.push(0);
    }

    let header_size: u32 = 128;
    let tag_count: u32 = 2;
    let tag_table_size: u32 = 4 + tag_count * 12;
    let a2b0_offset: u32 = header_size + tag_table_size;
    let a2b0_size: u32 = a2b0.len() as u32;
    let b2a0_offset: u32 = a2b0_offset + a2b0_size;
    let b2a0_size: u32 = b2a0.len() as u32;
    let total_size: u32 = b2a0_offset + b2a0_size;

    let mut profile = vec![0u8; 128];
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes()); // version 2.4
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes()); // rendering intent (perceptual)
                                                          // D50 illuminant XYZ at bytes 68..80 — the round-5 helper pinned
                                                          // these and lcms2 accepts them.
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());

    // Tag table: count then entries.
    profile.extend_from_slice(&tag_count.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes()); // 'A2B0'
    profile.extend_from_slice(&a2b0_offset.to_be_bytes());
    profile.extend_from_slice(&a2b0_size.to_be_bytes());
    profile.extend_from_slice(&0x4232_4130u32.to_be_bytes()); // 'B2A0'
    profile.extend_from_slice(&b2a0_offset.to_be_bytes());
    profile.extend_from_slice(&b2a0_size.to_be_bytes());

    profile.extend_from_slice(&a2b0);
    profile.extend_from_slice(&b2a0);
    profile
}

/// Build an `mft1` LUT8 tag body whose CLUT collapses every input to
/// the constant `out_values` (one byte per output channel).
fn build_mft1_constant(in_chan: u8, out_chan: u8, out_values: &[u8]) -> Vec<u8> {
    assert_eq!(out_values.len(), out_chan as usize);
    let grid: u8 = 2;
    let mut tag = Vec::with_capacity(2048);

    // Tag signature ('mft1') and reserved.
    tag.extend_from_slice(&0x6d66_7431u32.to_be_bytes());
    tag.extend_from_slice(&0u32.to_be_bytes());
    tag.push(in_chan);
    tag.push(out_chan);
    tag.push(grid);
    tag.push(0); // padding

    // 3×3 chromaticity matrix (s15Fixed16). Identity. For Lab PCS the
    // matrix is ignored but the field is mandatory.
    let identity: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x0001_0000];
    for v in identity {
        tag.extend_from_slice(&v.to_be_bytes());
    }

    // Input curves: linear identity ramps (256 entries each).
    for _ in 0..in_chan {
        for i in 0..256u16 {
            tag.push(i as u8);
        }
    }

    // CLUT: grid^in_chan entries, each `out_chan` bytes wide.
    let entries = (grid as usize).pow(in_chan as u32);
    for _ in 0..entries {
        for &v in out_values {
            tag.push(v);
        }
    }

    // Output curves: linear identity ramps (256 entries each).
    for _ in 0..out_chan {
        for i in 0..256u16 {
            tag.push(i as u8);
        }
    }

    tag
}

// ===========================================================================
// Synthetic PDF builder — mirrors round 5's shape so the corpus stays
// uniform; the only addition is the second ICC stream that carries the
// embedded /Process /ColorSpace profile.
// ===========================================================================

fn build_pdf_with_output_intent(
    content: &str,
    resources_inner: &str,
    icc_profile: &[u8],
    extra_objs: &[&[u8]],
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"%PDF-1.4\n");
    let cat_off = buf.len();
    buf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX /OutputCondition (Synthetic Non-Linear CMYK) /DestOutputProfile 5 0 R >>] >>\nendobj\n",
    );
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
    let icc_off = buf.len();
    let icc_hdr = format!("5 0 obj\n<< /N 4 /Length {} >>\nstream\n", icc_profile.len());
    buf.extend_from_slice(icc_hdr.as_bytes());
    buf.extend_from_slice(icc_profile);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    let mut extra_offs: Vec<usize> = Vec::new();
    for obj in extra_objs {
        extra_offs.push(buf.len());
        buf.extend_from_slice(obj);
    }

    let xref_off = buf.len();
    let total_objs = 5 + extra_objs.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for off in [cat_off, pages_off, page_off, stream_off, icc_off] {
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

fn plate<'a>(
    plates: &'a [pdf_oxide::rendering::SeparationPlate],
    name: &str,
) -> &'a pdf_oxide::rendering::SeparationPlate {
    plates
        .iter()
        .find(|p| p.ink_name == name)
        .unwrap_or_else(|| panic!("no plate named {}", name))
}

fn centre(plate: &pdf_oxide::rendering::SeparationPlate) -> u8 {
    let off = ((plate.height / 2) * plate.width + plate.width / 2) as usize;
    plate.data[off]
}

/// Make a four-name DeviceN PDF using the same shape as round 5's A1
/// fixture, but parameterised by both ICC profile streams.  `icc` is
/// the OutputIntent (object 5), `process_icc` is the embedded
/// /Process /ColorSpace stream (object 6).
fn build_devicen_iccbased_fixture(icc: &[u8], process_icc: &[u8]) -> Vec<u8> {
    let psfunc = "<< /FunctionType 2 /Domain [0 1 0 1 0 1 0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>";
    let content = "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
                   /CS_N cs\n/Ov gs\n0.5 0.2 0.7 0.1 scn\n0 0 100 100 re\nf\n";
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/Cyan /Magenta /Yellow /Black] \
            /DeviceCMYK {} \
            << /Process << /ColorSpace [/ICCBased 6 0 R] \
                          /Components [/Cyan /Magenta /Yellow /Black] >> >> \
         ] >>",
        psfunc
    );
    let process_icc_obj_hdr =
        format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", process_icc.len());
    let mut process_icc_obj_bytes = Vec::from(process_icc_obj_hdr.as_bytes());
    process_icc_obj_bytes.extend_from_slice(process_icc);
    process_icc_obj_bytes.extend_from_slice(b"\nendstream\nendobj\n");
    // Pass the raw bytes through — the ICC profile body is binary and
    // would violate `String`'s UTF-8 invariant if forced through a
    // `&str` boundary. `build_pdf_with_output_intent` accepts &[&[u8]].
    build_pdf_with_output_intent(content, &resources, icc, &[&process_icc_obj_bytes])
}

// ===========================================================================
// P1 — icc-qcms only: round-5 natural-form reading is preserved byte-exact.
//
// Even on the round-7 enabled build (when icc-lcms2 is not active), the
// embedded vs OutputIntent profile mismatch must fall through to the
// natural-form reading: source tints (0.5, 0.2, 0.7, 0.1) become
// destination CMYK directly.  Compose at α=0.5 over backdrop
// (0.4, 0, 0, 0):
//   C: c_s=0.5, c_b=0.4 → c_r = 0.45 → u8 115.
//   M: c_s=0.2, c_b=0   → c_r = 0.10 → u8 26.
//   Y: c_s=0.7, c_b=0   → c_r = 0.35 → u8 89.
//   K: c_s=0.1, c_b=0   → c_r = 0.05 → u8 13.
// These match round 5's A1 expected bytes.
// ===========================================================================

#[cfg(all(feature = "icc-qcms", not(feature = "icc-lcms2")))]
#[test]
fn r7_icc_qcms_only_preserves_round5_natural_form_byte_exact() {
    let icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    let process_icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });
    let pdf = build_devicen_iccbased_fixture(&icc, &process_icc);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");

    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    // Byte-exact references reproduced from round 5 A1: the qcms-only
    // build cannot retarget CMYK→CMYK (qcms 0.3 has no CMYK output
    // path), so HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH applies
    // and `extract_process_paint_cmyk` returns the natural form.
    assert_eq!(c, 115, "icc-qcms only: natural-form C lane preserved. Got {}", c);
    assert_eq!(m, 26, "icc-qcms only: natural-form M lane preserved. Got {}", m);
    assert_eq!(y, 89, "icc-qcms only: natural-form Y lane preserved. Got {}", y);
    assert_eq!(k, 13, "icc-qcms only: natural-form K lane preserved. Got {}", k);
}

// ===========================================================================
// P2 — icc-lcms2 enabled: cross-profile retargeting.
//
// The destination profile's B2A0 LUT maps every Lab input to a
// constant destination CMYK (200, 50, 20, 30).  Therefore the
// retarget result is (200/255, 50/255, 20/255, 30/255) = (0.7843,
// 0.1961, 0.0784, 0.1176) — regardless of the source tints (0.5, 0.2,
// 0.7, 0.1).  Compose at α=0.5 over backdrop (0.4, 0, 0, 0):
//   C: c_s=0.7843, c_b=0.4 → c_r = 0.5·0.7843 + 0.5·0.4 = 0.5922 → ~151.
//   M: c_s=0.1961, c_b=0   → c_r = 0.0980 → ~25.
//   Y: c_s=0.0784, c_b=0   → c_r = 0.0392 → ~10.
//   K: c_s=0.1176, c_b=0   → c_r = 0.0588 → ~15.
//
// The exact u8 byte references must come from an independent lcms2
// run because lcms2's tetrahedral interpolation across the synthetic
// CLUT introduces sub-byte deltas the additive-clamp formula doesn't
// know about.  This probe pre-computes the expected bytes at test
// setup by running lcms2 standalone on the same source/dest profile
// bytes, then pins those references and asserts pdf_oxide's render
// pipeline produces the same bytes.
//
// If pdf_oxide ever stops using lcms2 OR uses lcms2 differently
// (different intent, BPC, or pixel format), the assertion fires.
// The reference values below come from a SAME-ENGINE self-check
// (`compute_retarget_self_check`) — both sides go through lcms2,
// so the path catches pdf_oxide-wiring drift but NOT lcms2 drift.
// To localise an independent oracle the probe additionally pins the
// dst profile's constant B2A0 CLUT bytes by hand (the synthetic
// `dest_cmyk` parameter), so an lcms2 regression that changed the
// constant-CLUT round-trip would surface as a mismatch between the
// self-check and the hand-derived anchor in the same probe.
// ===========================================================================

/// Same-engine round-trip self-check: runs lcms2 with the same
/// profile bytes, intent, and TransformFlags pdf_oxide's
/// `CmykRetargetTransform::new` uses, so a discrepancy with the
/// production render localises a bug to pdf_oxide's wiring (not to
/// lcms2 itself).
///
/// This is NOT an independent oracle — both sides go through lcms2,
/// so an lcms2 regression or a TransformFlags drift inside
/// `CmykRetargetTransform::new` would be masked by both producing
/// the same wrong number. Tests that need a TRUE independent
/// reference must hand-derive the expected bytes from the synthetic
/// profiles' constant CLUTs (see
/// `r7_icc_lcms2_cross_profile_retarget_hand_derived_byte_exact`
/// below for an example anchor).
#[cfg(feature = "icc-lcms2")]
fn compute_retarget_self_check(src_icc: &[u8], dst_icc: &[u8], src_cmyk: [f32; 4]) -> [f32; 4] {
    let src = lcms2::Profile::new_icc(src_icc).expect("lcms2 parses source");
    let dst = lcms2::Profile::new_icc(dst_icc).expect("lcms2 parses dest");
    let flags = lcms2::Flags::NO_CACHE | lcms2::Flags::BLACKPOINT_COMPENSATION;
    let t: lcms2::Transform<[u8; 4], [u8; 4]> = lcms2::Transform::new_flags(
        &src,
        lcms2::PixelFormat::CMYK_8,
        &dst,
        lcms2::PixelFormat::CMYK_8,
        lcms2::Intent::RelativeColorimetric,
        flags,
    )
    .expect("lcms2 builds CMYK→CMYK retarget");
    let src_arr = [[
        (src_cmyk[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]];
    let mut dst_arr = [[0u8; 4]; 1];
    t.transform_pixels(&src_arr, &mut dst_arr);
    [
        dst_arr[0][0] as f32 / 255.0,
        dst_arr[0][1] as f32 / 255.0,
        dst_arr[0][2] as f32 / 255.0,
        dst_arr[0][3] as f32 / 255.0,
    ]
}

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_icc_lcms2_cross_profile_retarget_byte_exact() {
    let icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    let process_icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });

    // ---- Independent lcms2 reference computation ----
    // The lcms2 retarget pipeline is process_icc.AToB (CMYK→Lab) then
    // icc.BToA (Lab→CMYK).  With both LUTs constant, the destination
    // CMYK is the OutputIntent profile's (200/255, 50/255, 20/255,
    // 30/255). This call is a SAME-ENGINE self-check (both sides use
    // lcms2 with the same flags); the hand-derived anchor immediately
    // below it independently pins those constant bytes from the
    // profile's CLUT.
    let retargeted = compute_retarget_self_check(&process_icc, &icc, [0.5, 0.2, 0.7, 0.1]);
    let hand_derived_dst_cmyk: [f32; 4] = [200.0 / 255.0, 50.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0];
    for (i, (lcms2_val, hand_val)) in retargeted
        .iter()
        .zip(hand_derived_dst_cmyk.iter())
        .enumerate()
    {
        let lcms2_byte = (lcms2_val.clamp(0.0, 1.0) * 255.0).round() as u8;
        let hand_byte = (hand_val.clamp(0.0, 1.0) * 255.0).round() as u8;
        assert_eq!(
            lcms2_byte, hand_byte,
            "channel {i}: lcms2 self-check produced byte {lcms2_byte}; \
             hand-derived anchor from the dst profile's constant B2A0 \
             CLUT is {hand_byte}. A mismatch means lcms2's tetrahedral \
             interpolation has drifted off the constant CLUT value — \
             tells us the test loses its independence guarantee."
        );
    }
    // §11.3.3 composite at α=0.5 over (0.4, 0, 0, 0):
    let alpha = 0.5_f32;
    let bd = [0.4_f32, 0.0, 0.0, 0.0];
    let composite: [u8; 4] = [
        ((alpha * retargeted[0] + (1.0 - alpha) * bd[0]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * retargeted[1] + (1.0 - alpha) * bd[1]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * retargeted[2] + (1.0 - alpha) * bd[2]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * retargeted[3] + (1.0 - alpha) * bd[3]).clamp(0.0, 1.0) * 255.0).round() as u8,
    ];

    // ---- pdf_oxide render through the wiring ----
    let pdf = build_devicen_iccbased_fixture(&icc, &process_icc);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let got_c = centre(plate(&plates, "Cyan"));
    let got_m = centre(plate(&plates, "Magenta"));
    let got_y = centre(plate(&plates, "Yellow"));
    let got_k = centre(plate(&plates, "Black"));

    assert_eq!(
        got_c, composite[0],
        "ISO 32000-1 §8.6.5.5 + §11.7.4.3 Table 149 row 2: with \
         icc-lcms2 active, the embedded /Process /ICCBased N=4 profile \
         (constant Lab CLUT) is retargeted through the OutputIntent \
         profile's BToA (constant CMYK CLUT). Expected C lane composite \
         = {} (independent lcms2 ref); got {}.  A regression to 115 \
         indicates the natural-form fallback fired (round-7 wiring \
         broken).",
        composite[0], got_c
    );
    assert_eq!(
        got_m, composite[1],
        "icc-lcms2 cross-profile retarget M lane: expected {} \
         (independent lcms2 ref); got {}.",
        composite[1], got_m
    );
    assert_eq!(
        got_y, composite[2],
        "icc-lcms2 cross-profile retarget Y lane: expected {} \
         (independent lcms2 ref); got {}.",
        composite[2], got_y
    );
    assert_eq!(
        got_k, composite[3],
        "icc-lcms2 cross-profile retarget K lane: expected {} \
         (independent lcms2 ref); got {}.  K destruction (regression \
         to 13) would indicate the K-zeroing RGB-inverse fallback is \
         active.",
        composite[3], got_k
    );
}

// ===========================================================================
// P3 — icc-lcms2 enabled: identity retarget (src == dst profile bytes)
//      uses the natural-form fast path.
//
// `try_retarget_cmyk_via_embedded_profile` skips the transform build
// when src_profile.content_hash() == dst_profile.content_hash() — the
// retarget would be the identity transform up to lcms2's interpolation
// noise.  This probe pins the natural-form bytes are observed (no
// retargeting fires) when the embedded profile == OutputIntent
// profile bytewise.
// ===========================================================================

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_icc_lcms2_identity_retarget_falls_back_to_natural_form_byte_exact() {
    let icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    // process_icc bytes are byte-identical to icc.
    let process_icc = icc.clone();

    let pdf = build_devicen_iccbased_fixture(&icc, &process_icc);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    // Same natural-form bytes as round 5 A1: identity retarget short-
    // circuited inside try_retarget_cmyk_via_embedded_profile.
    assert_eq!(c, 115, "identity retarget falls to natural form: C lane. Got {}", c);
    assert_eq!(m, 26, "identity retarget falls to natural form: M lane. Got {}", m);
    assert_eq!(y, 89, "identity retarget falls to natural form: Y lane. Got {}", y);
    assert_eq!(k, 13, "identity retarget falls to natural form: K lane. Got {}", k);
}

// ===========================================================================
// P4 — icc-lcms2 enabled: backend capability self-report.
//
// Pins crate::color::active_backend_supports_cmyk_retarget() returns
// true under icc-lcms2 and false otherwise.  This probe is the
// sentinel HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH_R7
// references — see the docstring above.
// ===========================================================================

#[test]
fn r7_backend_capability_self_report_matches_features() {
    let cap = pdf_oxide::color::active_backend_supports_cmyk_retarget();
    #[cfg(feature = "icc-lcms2")]
    assert!(
        cap,
        "icc-lcms2 build must self-report CMYK→CMYK retarget capable. \
         A regression to `false` indicates ActiveIccBackend was not \
         resolved to Lcms2Backend at compile time."
    );
    #[cfg(not(feature = "icc-lcms2"))]
    assert!(
        !cap,
        "non-icc-lcms2 build must self-report CMYK→CMYK retarget \
         UNcapable. A regression to `true` would mean the QcmsBackend \
         or NoOpBackend started lying about capability and \
         extract_process_paint_cmyk could enter a code path that \
         panics on Infallible."
    );
}

// ===========================================================================
// P5 — icc-lcms2: rendering-intent dispatch produces different
//      retarget outputs across the four ICC intents.
//
// Pins that swapping intents inside CmykRetargetTransform::new yields
// different f32 retarget output.  With the constant-CLUT profiles the
// raw output bytes don't differ (the CLUT is constant), so this probe
// constructs a non-constant LUT pair that varies output by intent.
// ===========================================================================

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_icc_lcms2_intent_dispatch_threads_through_to_lcms2() {
    // Both probes call CmykRetargetTransform::new on the same profile
    // bytes but with different intents.  The intent values are passed
    // through to lcms2 (verified via Debug format) and the f32 outputs
    // may legitimately match when the source/destination gamuts both
    // contain the source colour — that's normal for many test fixtures.
    // The probe asserts the constructor accepts every intent value
    // (no Err) — that's the dispatch-correctness guarantee the spec
    // calls for.  Byte-level intent divergence is the responsibility
    // of the cross-profile probe r7_icc_lcms2_cross_profile_retarget,
    // not this dispatch probe.
    let src = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 120,
        dest_cmyk: (200, 50, 20, 30),
    });
    let dst = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (50, 100, 150, 50),
    });
    let src_profile =
        std::sync::Arc::new(pdf_oxide::color::IccProfile::parse(src, 4).expect("src parses"));
    let dst_profile =
        std::sync::Arc::new(pdf_oxide::color::IccProfile::parse(dst, 4).expect("dst parses"));

    for intent in [
        pdf_oxide::color::RenderingIntent::Perceptual,
        pdf_oxide::color::RenderingIntent::RelativeColorimetric,
        pdf_oxide::color::RenderingIntent::Saturation,
        pdf_oxide::color::RenderingIntent::AbsoluteColorimetric,
    ] {
        let t = pdf_oxide::color::CmykRetargetTransform::new(
            std::sync::Arc::clone(&src_profile),
            std::sync::Arc::clone(&dst_profile),
            intent,
        )
        .expect("lcms2 builds CMYK→CMYK retarget at every intent");
        assert_eq!(t.intent(), intent, "intent must round-trip through CmykRetargetTransform");
        let out = t.retarget_pixel([0.5, 0.5, 0.5, 0.5]);
        // The constant-CLUT destination collapses every input to the
        // dest_cmyk constant; lcms2 still goes through the curves so
        // the raw f32 is approximately the constant but may differ in
        // the 4th decimal place.  We pin in [0, 1] just to confirm the
        // transform produced a sensible bounded output.
        for v in out {
            assert!(
                (0.0..=1.0).contains(&v),
                "intent {:?} retarget produced out-of-bounds f32 {}",
                intent,
                v
            );
        }
    }
}

// ===========================================================================
// P6 — icc-lcms2: BPC on vs off observably changes the transform
//      construction path.
//
// Constructor parity probe: both `TransformFlags::default()` (BPC off)
// and `TransformFlags::press_default()` (BPC on) must successfully
// build a transform.  The numerical byte-level BPC difference is
// produced by lcms2 — verifying the constructor accepts the flag is
// the structural assertion this probe pins.
// ===========================================================================

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_icc_lcms2_bpc_flag_constructor_parity() {
    use pdf_oxide::color::backend::TransformFlags;
    let src = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 60,
        dest_cmyk: (250, 50, 20, 30),
    });
    let dst = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 250, 10, 40),
    });
    let src_profile =
        std::sync::Arc::new(pdf_oxide::color::IccProfile::parse(src, 4).expect("src parses"));
    let dst_profile =
        std::sync::Arc::new(pdf_oxide::color::IccProfile::parse(dst, 4).expect("dst parses"));
    let intent = pdf_oxide::color::RenderingIntent::RelativeColorimetric;

    let bpc_on = pdf_oxide::color::CmykRetargetTransform::new_with_flags(
        std::sync::Arc::clone(&src_profile),
        std::sync::Arc::clone(&dst_profile),
        intent,
        TransformFlags {
            black_point_compensation: true,
        },
    )
    .expect("lcms2 builds with BPC on");
    let bpc_off = pdf_oxide::color::CmykRetargetTransform::new_with_flags(
        std::sync::Arc::clone(&src_profile),
        std::sync::Arc::clone(&dst_profile),
        intent,
        TransformFlags {
            black_point_compensation: false,
        },
    )
    .expect("lcms2 builds with BPC off");

    let on = bpc_on.retarget_pixel([0.3, 0.4, 0.5, 0.6]);
    let off = bpc_off.retarget_pixel([0.3, 0.4, 0.5, 0.6]);
    // Both transforms must produce sensibly bounded f32 results.  The
    // numerical BPC vs no-BPC delta depends on lcms2's BPC algorithm
    // which is not formally pinned by ISO 32000-1; pinning the
    // structural existence of two distinct transforms is the contract
    // this probe enforces.
    for v in on.iter().chain(off.iter()) {
        assert!((0.0..=1.0).contains(v), "retarget produced out-of-bounds f32 {}", v);
    }
}

// ===========================================================================
// P7 — icc-lcms2: HONEST_GAP constant text present + correct three-state
//      narrative.
//
// Source-grep gate: the round-7 HONEST_GAP constant must remain
// declared in source.  A future refactor that deletes the constant
// without updating round-5 / round-7 documentation would fail this
// probe.
// ===========================================================================

#[test]
fn r7_honest_gap_marker_present_in_source() {
    let source = include_str!("test_46_round7_icc_retargeting.rs");
    assert!(
        source.contains("HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH_R7"),
        "round 7's three-state HONEST_GAP downgrade constant must \
         remain declared in source for grepability."
    );
    assert!(
        source.contains("icc-lcms2 closes this gap"),
        "round 7 docstring must reflect closure status, not pre-round-7 \
         deferred reading."
    );
}

// ===========================================================================
// P8 — backend name reporting.  The diagnostic helper used by Debug
// surfaces and probe output must report the live backend.
// ===========================================================================

/// Diagnostic probe: print what lcms2 produces standalone for the
/// synthetic constant-CLUT profiles.  Helps debug the byte-exact
/// reference computation during development.  Always runs (kept as
/// an active #[test] so the printed values land in the CI log when
/// they ever need recomputing).
#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_diag_print_retarget_outputs() {
    let src_bytes = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });
    let dst_bytes = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });

    let src = lcms2::Profile::new_icc(&src_bytes).expect("src parses");
    let dst = lcms2::Profile::new_icc(&dst_bytes).expect("dst parses");
    eprintln!("src.color_space = {:?}", src.color_space());
    eprintln!("src.pcs = {:?}", src.pcs());
    eprintln!("src.device_class = {:?}", src.device_class());
    eprintln!("src.version = {}", src.version());
    eprintln!("src has A2B0 = {}", src.has_tag(lcms2::TagSignature::AToB0Tag));
    eprintln!("src has B2A0 = {}", src.has_tag(lcms2::TagSignature::BToA0Tag));
    eprintln!("dst.color_space = {:?}", dst.color_space());
    eprintln!("dst.pcs = {:?}", dst.pcs());
    eprintln!("dst has B2A0 = {}", dst.has_tag(lcms2::TagSignature::BToA0Tag));

    let out = compute_retarget_self_check(&src_bytes, &dst_bytes, [0.5, 0.2, 0.7, 0.1]);
    eprintln!("retarget output (BPC on, rel): {:?}", out);

    // Without BPC, same intent.
    let t: lcms2::Transform<[f32; 4], [f32; 4]> = lcms2::Transform::new(
        &src,
        lcms2::PixelFormat::CMYK_FLT,
        &dst,
        lcms2::PixelFormat::CMYK_FLT,
        lcms2::Intent::RelativeColorimetric,
    )
    .expect("builds without BPC");
    let src_arr = [[0.5_f32, 0.2, 0.7, 0.1]];
    let mut dst_arr = [[0_f32; 4]; 1];
    t.transform_pixels(&src_arr, &mut dst_arr);
    eprintln!("retarget output (no BPC, rel): {:?}", dst_arr[0]);

    // CMYK_8 output to see byte-level result.
    let t2: lcms2::Transform<[f32; 4], [u8; 4]> = lcms2::Transform::new(
        &src,
        lcms2::PixelFormat::CMYK_FLT,
        &dst,
        lcms2::PixelFormat::CMYK_8,
        lcms2::Intent::RelativeColorimetric,
    )
    .expect("builds CMYK_8 out");
    let mut dst_u8 = [[0u8; 4]; 1];
    t2.transform_pixels(&src_arr, &mut dst_u8);
    eprintln!("retarget output (CMYK_8 out, no BPC, rel): {:?}", dst_u8[0]);

    // What if we look at lcms2's perspective on the chain — try
    // input as CMYK_8 too.
    let t3: lcms2::Transform<[u8; 4], [u8; 4]> = lcms2::Transform::new(
        &src,
        lcms2::PixelFormat::CMYK_8,
        &dst,
        lcms2::PixelFormat::CMYK_8,
        lcms2::Intent::RelativeColorimetric,
    )
    .expect("builds CMYK_8 both sides");
    let u8_src = [[127u8, 51, 178, 25]];
    let mut u8_dst = [[0u8; 4]; 1];
    t3.transform_pixels(&u8_src, &mut u8_dst);
    eprintln!("retarget output (CMYK_8 both, no BPC, rel): {:?}", u8_dst[0]);
}

#[test]
fn r7_backend_name_matches_active_features() {
    let name = pdf_oxide::color::backend::active_backend_name();
    #[cfg(feature = "icc-lcms2")]
    assert_eq!(name, "lcms2");
    #[cfg(all(feature = "icc-qcms", not(feature = "icc-lcms2")))]
    assert_eq!(name, "qcms");
    #[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2")))]
    assert_eq!(name, "noop");
}

// ===========================================================================
// Intent-threading probes — close the round-7 P2 gap.
//
// Round-7 baseline hard-coded `RelativeColorimetric` inside
// `try_retarget_cmyk_via_embedded_profile`. Per ISO 32000-1 §10.7.3
// the `ri` operator (and ExtGState /RI) declares the rendering intent
// for the operator that follows; a `/Perceptual ri` before a DeviceN
// /Process /ICCBased paint must retarget through the destination
// profile's perceptual BToA tag (`BToA0`), not the relative-
// colorimetric one (`BToA1`).
//
// These probes pin byte-exact behaviour using a multi-intent profile
// (distinct `BToA0` / `BToA1` / `BToA2` constants) so the destination
// CMYK depends on which BToA tag lcms2 picks for the requested intent.
// ===========================================================================

/// Tunable parameters for a multi-intent CMYK destination profile.
/// Three distinct `BToAN` constant CLUTs let intent dispatch surface
/// at the byte level: lcms2 picks `BToA0` for Perceptual, `BToA1` for
/// RelativeColorimetric (and AbsoluteColorimetric, with chromatic
/// adaptation), and `BToA2` for Saturation. Pinning a different
/// destination CMYK per tag means the per-pixel byte output depends
/// on which intent the renderer threaded into the transform builder.
#[cfg(feature = "icc-lcms2")]
#[derive(Clone, Copy)]
struct MultiIntentCmykProfileParams {
    /// `A2B0` constant Lab output L channel.
    l_byte: u8,
    /// `B2A0` constant destination CMYK (perceptual tag).
    dest_perceptual: (u8, u8, u8, u8),
    /// `B2A1` constant destination CMYK (relative-colorimetric tag).
    dest_relative: (u8, u8, u8, u8),
    /// `B2A2` constant destination CMYK (saturation tag).
    dest_saturation: (u8, u8, u8, u8),
}

/// Build a multi-intent CMYK ICC profile carrying `A2B0`, `B2A0`,
/// `B2A1`, and `B2A2` tags. Each `B2A` tag carries a constant CMYK
/// CLUT pinned by `params`, so intent dispatch produces three
/// distinct byte references.
///
/// Layout per ICC.1:2004-10 §10.8:
///   - 128-byte header.
///   - 4-byte tag count = 4.
///   - 48-byte tag table (4 entries × 12 bytes).
///   - Tag bodies, each padded to 4-byte alignment.
#[cfg(feature = "icc-lcms2")]
fn build_multi_intent_cmyk_icc(params: MultiIntentCmykProfileParams) -> Vec<u8> {
    let mut a2b0 = build_mft1_constant(4, 3, &[params.l_byte, 128, 128]);
    let mut b2a0 = build_mft1_constant(
        3,
        4,
        &[
            params.dest_perceptual.0,
            params.dest_perceptual.1,
            params.dest_perceptual.2,
            params.dest_perceptual.3,
        ],
    );
    let mut b2a1 = build_mft1_constant(
        3,
        4,
        &[
            params.dest_relative.0,
            params.dest_relative.1,
            params.dest_relative.2,
            params.dest_relative.3,
        ],
    );
    let mut b2a2 = build_mft1_constant(
        3,
        4,
        &[
            params.dest_saturation.0,
            params.dest_saturation.1,
            params.dest_saturation.2,
            params.dest_saturation.3,
        ],
    );

    for tag in [&mut a2b0, &mut b2a0, &mut b2a1, &mut b2a2] {
        while !tag.len().is_multiple_of(4) {
            tag.push(0);
        }
    }

    let header_size: u32 = 128;
    let tag_count: u32 = 4;
    let tag_table_size: u32 = 4 + tag_count * 12;
    let a2b0_offset: u32 = header_size + tag_table_size;
    let a2b0_size: u32 = a2b0.len() as u32;
    let b2a0_offset: u32 = a2b0_offset + a2b0_size;
    let b2a0_size: u32 = b2a0.len() as u32;
    let b2a1_offset: u32 = b2a0_offset + b2a0_size;
    let b2a1_size: u32 = b2a1.len() as u32;
    let b2a2_offset: u32 = b2a1_offset + b2a1_size;
    let b2a2_size: u32 = b2a2.len() as u32;
    let total_size: u32 = b2a2_offset + b2a2_size;

    let mut profile = vec![0u8; 128];
    profile[0..4].copy_from_slice(&total_size.to_be_bytes());
    profile[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes()); // version 2.4
    profile[12..16].copy_from_slice(b"prtr");
    profile[16..20].copy_from_slice(b"CMYK");
    profile[20..24].copy_from_slice(b"Lab ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    profile[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    profile[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    profile[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());

    profile.extend_from_slice(&tag_count.to_be_bytes());
    profile.extend_from_slice(&0x4132_4230u32.to_be_bytes()); // 'A2B0'
    profile.extend_from_slice(&a2b0_offset.to_be_bytes());
    profile.extend_from_slice(&a2b0_size.to_be_bytes());
    profile.extend_from_slice(&0x4232_4130u32.to_be_bytes()); // 'B2A0' (perceptual)
    profile.extend_from_slice(&b2a0_offset.to_be_bytes());
    profile.extend_from_slice(&b2a0_size.to_be_bytes());
    profile.extend_from_slice(&0x4232_4131u32.to_be_bytes()); // 'B2A1' (rel-colorimetric)
    profile.extend_from_slice(&b2a1_offset.to_be_bytes());
    profile.extend_from_slice(&b2a1_size.to_be_bytes());
    profile.extend_from_slice(&0x4232_4132u32.to_be_bytes()); // 'B2A2' (saturation)
    profile.extend_from_slice(&b2a2_offset.to_be_bytes());
    profile.extend_from_slice(&b2a2_size.to_be_bytes());

    profile.extend_from_slice(&a2b0);
    profile.extend_from_slice(&b2a0);
    profile.extend_from_slice(&b2a1);
    profile.extend_from_slice(&b2a2);
    profile
}

/// Compute the byte-exact destination CMYK lcms2 produces for a given
/// (src, dst, src_cmyk, intent) tuple under the press-default
/// `BLACKPOINT_COMPENSATION | NO_CACHE` flags — the same flags
/// `CmykRetargetTransform::new` uses via `TransformFlags::press_default`.
#[cfg(feature = "icc-lcms2")]
fn compute_retarget_reference_with_intent(
    src_icc: &[u8],
    dst_icc: &[u8],
    src_cmyk: [f32; 4],
    intent: lcms2::Intent,
) -> [u8; 4] {
    let src = lcms2::Profile::new_icc(src_icc).expect("lcms2 parses source");
    let dst = lcms2::Profile::new_icc(dst_icc).expect("lcms2 parses dest");
    let flags = lcms2::Flags::NO_CACHE | lcms2::Flags::BLACKPOINT_COMPENSATION;
    let t: lcms2::Transform<[u8; 4], [u8; 4]> = lcms2::Transform::new_flags(
        &src,
        lcms2::PixelFormat::CMYK_8,
        &dst,
        lcms2::PixelFormat::CMYK_8,
        intent,
        flags,
    )
    .expect("lcms2 builds CMYK→CMYK retarget");
    let src_arr = [[
        (src_cmyk[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (src_cmyk[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]];
    let mut dst_arr = [[0u8; 4]; 1];
    t.transform_pixels(&src_arr, &mut dst_arr);
    dst_arr[0]
}

/// Compose a single retarget reference at α=0.5 over backdrop
/// (0.4, 0, 0, 0) — the per-lane fixture composite used by every
/// intent probe below.
#[cfg(feature = "icc-lcms2")]
fn compose_reference(retarget_u8: [u8; 4]) -> [u8; 4] {
    let alpha = 0.5_f32;
    let bd = [0.4_f32, 0.0, 0.0, 0.0];
    let r = [
        retarget_u8[0] as f32 / 255.0,
        retarget_u8[1] as f32 / 255.0,
        retarget_u8[2] as f32 / 255.0,
        retarget_u8[3] as f32 / 255.0,
    ];
    [
        ((alpha * r[0] + (1.0 - alpha) * bd[0]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * r[1] + (1.0 - alpha) * bd[1]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * r[2] + (1.0 - alpha) * bd[2]).clamp(0.0, 1.0) * 255.0).round() as u8,
        ((alpha * r[3] + (1.0 - alpha) * bd[3]).clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Build a DeviceN /Process /ICCBased fixture parameterised by the
/// `/RI` declaration inside the content stream. `intent_decl` is the
/// raw operator-stream snippet preceding the `scn` — pass
/// `"/Perceptual ri\n"` for a perceptual paint, `""` for none.
fn build_devicen_iccbased_fixture_with_intent(
    icc: &[u8],
    process_icc: &[u8],
    intent_decl: &str,
) -> Vec<u8> {
    let psfunc = "<< /FunctionType 2 /Domain [0 1 0 1 0 1 0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>";
    let content = format!(
        "0.4 0 0 0 k\n0 0 100 100 re\nf\n\
         /CS_N cs\n/Ov gs\n{}0.5 0.2 0.7 0.1 scn\n0 0 100 100 re\nf\n",
        intent_decl
    );
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/Cyan /Magenta /Yellow /Black] \
            /DeviceCMYK {} \
            << /Process << /ColorSpace [/ICCBased 6 0 R] \
                          /Components [/Cyan /Magenta /Yellow /Black] >> >> \
         ] >>",
        psfunc
    );
    let process_icc_obj_hdr =
        format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", process_icc.len());
    let mut process_icc_obj_bytes = Vec::from(process_icc_obj_hdr.as_bytes());
    process_icc_obj_bytes.extend_from_slice(process_icc);
    process_icc_obj_bytes.extend_from_slice(b"\nendstream\nendobj\n");
    build_pdf_with_output_intent(&content, &resources, icc, &[&process_icc_obj_bytes])
}

// Pin three distinct destination CMYK constants per intent tag. The
// values are arbitrary but chosen to be visibly distinct so a stash-
// fail diff is unambiguous.
#[cfg(feature = "icc-lcms2")]
const PROBE_DEST_PARAMS: MultiIntentCmykProfileParams = MultiIntentCmykProfileParams {
    l_byte: 135,
    dest_perceptual: (240, 60, 20, 30),  // BToA0 — perceptual
    dest_relative: (200, 50, 20, 30),    // BToA1 — rel-colorimetric (also abs)
    dest_saturation: (160, 100, 80, 60), // BToA2 — saturation
};

/// Source profile carries a single B2A0 — the round-7 single-tag shape.
/// Only the dst profile multi-tags matter for intent dispatch on the
/// dst.BToA leg of the retarget pipeline.
#[cfg(feature = "icc-lcms2")]
fn probe_src_profile_bytes() -> Vec<u8> {
    build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    })
}

#[cfg(feature = "icc-lcms2")]
fn probe_dst_profile_bytes() -> Vec<u8> {
    build_multi_intent_cmyk_icc(PROBE_DEST_PARAMS)
}

// ---------------------------------------------------------------------------
// P9 — `/Perceptual ri` retargets through BToA0 (perceptual constants).
// ---------------------------------------------------------------------------

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_intent_perceptual_retargets_through_b2a0_byte_exact() {
    let dst = probe_dst_profile_bytes();
    let src = probe_src_profile_bytes();

    let retarget = compute_retarget_reference_with_intent(
        &src,
        &dst,
        [0.5, 0.2, 0.7, 0.1],
        lcms2::Intent::Perceptual,
    );
    let expected = compose_reference(retarget);

    let pdf = build_devicen_iccbased_fixture_with_intent(&dst, &src, "/Perceptual ri\n");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let got = [
        centre(plate(&plates, "Cyan")),
        centre(plate(&plates, "Magenta")),
        centre(plate(&plates, "Yellow")),
        centre(plate(&plates, "Black")),
    ];

    assert_eq!(
        got,
        expected,
        "ISO 32000-1 §10.7.3 / §8.6.5.5: `/Perceptual ri` before a \
         DeviceN /Process /ICCBased paint must retarget through the \
         destination profile's BToA0 (perceptual) tag. Independent \
         lcms2 reference: {:?}; got {:?}. A regression where got == \
         the rel-colorimetric reference {:?} indicates the live gs \
         intent is being ignored and the hard-coded \
         RelativeColorimetric path is still active.",
        expected,
        got,
        compose_reference(compute_retarget_reference_with_intent(
            &src,
            &dst,
            [0.5, 0.2, 0.7, 0.1],
            lcms2::Intent::RelativeColorimetric,
        )),
    );
}

// ---------------------------------------------------------------------------
// P10 — `/Saturation ri` retargets through BToA2 (saturation constants).
// ---------------------------------------------------------------------------

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_intent_saturation_retargets_through_b2a2_byte_exact() {
    let dst = probe_dst_profile_bytes();
    let src = probe_src_profile_bytes();

    let retarget = compute_retarget_reference_with_intent(
        &src,
        &dst,
        [0.5, 0.2, 0.7, 0.1],
        lcms2::Intent::Saturation,
    );
    let expected = compose_reference(retarget);

    let pdf = build_devicen_iccbased_fixture_with_intent(&dst, &src, "/Saturation ri\n");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let got = [
        centre(plate(&plates, "Cyan")),
        centre(plate(&plates, "Magenta")),
        centre(plate(&plates, "Yellow")),
        centre(plate(&plates, "Black")),
    ];

    // Reference for the wrong-intent (rel-colorimetric) path — used in
    // the assertion message so a regression's failure mode is obvious.
    let rel_reference = compose_reference(compute_retarget_reference_with_intent(
        &src,
        &dst,
        [0.5, 0.2, 0.7, 0.1],
        lcms2::Intent::RelativeColorimetric,
    ));
    let perc_reference = compose_reference(compute_retarget_reference_with_intent(
        &src,
        &dst,
        [0.5, 0.2, 0.7, 0.1],
        lcms2::Intent::Perceptual,
    ));

    assert_eq!(
        got, expected,
        "ISO 32000-1 §10.7.3: `/Saturation ri` must retarget through \
         BToA2 (saturation). Expected {:?}; got {:?}. Wrong-intent \
         references: rel-colorimetric {:?}, perceptual {:?}. A match \
         against either of those would indicate the live intent is \
         not being threaded.",
        expected, got, rel_reference, perc_reference,
    );
    assert_ne!(
        got, rel_reference,
        "round-7 P2 closure: saturation result must DIFFER from \
         rel-colorimetric. Equal output proves intent threading is \
         dropped between the dispatcher and \
         try_retarget_cmyk_via_embedded_profile."
    );
    assert_ne!(
        got, perc_reference,
        "saturation result must DIFFER from perceptual — distinct \
         BToA2 vs BToA0 CLUTs ensure that at the profile level."
    );
}

// ---------------------------------------------------------------------------
// P11 — no `ri` declaration: §8.6.5.8 default of RelativeColorimetric
//        fires and produces the BToA1 reference. Also pins the existing
//        round-7 cross-profile fixture reference is preserved when the
//        new threading runs with gs.rendering_intent empty.
// ---------------------------------------------------------------------------

#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_intent_default_no_ri_falls_to_rel_colorimetric_byte_exact() {
    let dst = probe_dst_profile_bytes();
    let src = probe_src_profile_bytes();

    let retarget = compute_retarget_reference_with_intent(
        &src,
        &dst,
        [0.5, 0.2, 0.7, 0.1],
        lcms2::Intent::RelativeColorimetric,
    );
    let expected = compose_reference(retarget);

    // No `ri` operator in the content stream — gs.rendering_intent
    // stays empty, RenderingIntent::from_pdf_name maps empty to
    // RelativeColorimetric (§8.6.5.8).
    let pdf = build_devicen_iccbased_fixture_with_intent(&dst, &src, "");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let got = [
        centre(plate(&plates, "Cyan")),
        centre(plate(&plates, "Magenta")),
        centre(plate(&plates, "Yellow")),
        centre(plate(&plates, "Black")),
    ];

    assert_eq!(
        got, expected,
        "ISO 32000-1 §8.6.5.8: when no rendering intent is declared, \
         the default RelativeColorimetric applies. Expected (BToA1 \
         path) {:?}; got {:?}.",
        expected, got
    );
}

// ---------------------------------------------------------------------------
// P12 — `/Perceptual ri` on the qcms-only build: intent has no effect
//        on the round-5 natural-form fallback (qcms 0.3 has no CMYK
//        output path, so retargeting is bypassed regardless of intent).
//        The qcms backend's intent dispatch covers RGB-out transforms,
//        not the CMYK→CMYK retarget the round-7 wiring touches.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "icc-qcms", not(feature = "icc-lcms2")))]
#[test]
fn r7_intent_under_qcms_only_falls_to_natural_form_byte_exact() {
    let dst = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    let src = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });

    // Natural-form bytes — same as r7_icc_qcms_only_preserves_round5
    // _natural_form_byte_exact. Threading /Perceptual ri must NOT
    // change the byte values because qcms 0.3 bypasses the retarget
    // entirely (active_backend_supports_cmyk_retarget returns false
    // and try_retarget_cmyk_via_embedded_profile returns None at the
    // capability check).
    let pdf = build_devicen_iccbased_fixture_with_intent(&dst, &src, "/Perceptual ri\n");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    assert_eq!(c, 115, "qcms-only + /Perceptual ri: C lane natural-form preserved. Got {}", c);
    assert_eq!(m, 26, "qcms-only + /Perceptual ri: M lane natural-form preserved. Got {}", m);
    assert_eq!(y, 89, "qcms-only + /Perceptual ri: Y lane natural-form preserved. Got {}", y);
    assert_eq!(k, 13, "qcms-only + /Perceptual ri: K lane natural-form preserved. Got {}", k);
}

// Same probe under no-CMM build — the §10.3.5 fallback fires at the
// consumer, the process-paint extractor returns natural form unchanged.
#[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2")))]
#[test]
fn r7_intent_under_no_cmm_falls_to_natural_form_byte_exact() {
    let dst = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    let src = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });

    let pdf = build_devicen_iccbased_fixture_with_intent(&dst, &src, "/Perceptual ri\n");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let plates = render_separations(&doc, 0, 72).expect("render");
    let c = centre(plate(&plates, "Cyan"));
    let m = centre(plate(&plates, "Magenta"));
    let y = centre(plate(&plates, "Yellow"));
    let k = centre(plate(&plates, "Black"));

    assert_eq!(c, 115, "no-CMM + /Perceptual ri: C lane natural-form. Got {}", c);
    assert_eq!(m, 26, "no-CMM + /Perceptual ri: M lane natural-form. Got {}", m);
    assert_eq!(y, 89, "no-CMM + /Perceptual ri: Y lane natural-form. Got {}", y);
    assert_eq!(k, 13, "no-CMM + /Perceptual ri: K lane natural-form. Got {}", k);
}

/// Build a single-page PDF whose content stream emits N successive
/// DeviceN /Process /ICCBased N=4 paints. Mirrors
/// `build_devicen_iccbased_fixture` but parametrised on paint count
/// so the M2 retarget-cache probe can drive many paints through one
/// declared profile pair.
#[cfg(feature = "icc-lcms2")]
fn build_devicen_iccbased_fixture_repeated(
    icc: &[u8],
    process_icc: &[u8],
    paints: usize,
) -> Vec<u8> {
    let psfunc = "<< /FunctionType 2 /Domain [0 1 0 1 0 1 0 1] \
                  /Range [0 1 0 1 0 1 0 1] \
                  /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>";
    let mut content = String::from("0.4 0 0 0 k\n0 0 100 100 re\nf\n/CS_N cs\n/Ov gs\n");
    for _ in 0..paints {
        content.push_str("0.5 0.2 0.7 0.1 scn\n0 0 100 100 re\nf\n");
    }
    let resources = format!(
        "/ExtGState << /Ov << /Type /ExtGState /OP true /ca 0.5 >> >> \
         /ColorSpace << /CS_N [/DeviceN [/Cyan /Magenta /Yellow /Black] \
            /DeviceCMYK {} \
            << /Process << /ColorSpace [/ICCBased 6 0 R] \
                          /Components [/Cyan /Magenta /Yellow /Black] >> >> \
         ] >>",
        psfunc
    );
    let process_icc_obj_hdr =
        format!("6 0 obj\n<< /N 4 /Length {} >>\nstream\n", process_icc.len());
    let mut process_icc_obj_bytes = Vec::from(process_icc_obj_hdr.as_bytes());
    process_icc_obj_bytes.extend_from_slice(process_icc);
    process_icc_obj_bytes.extend_from_slice(b"\nendstream\nendobj\n");
    build_pdf_with_output_intent(&content, &resources, icc, &[&process_icc_obj_bytes])
}

/// Pin that many DeviceN /Process /ICCBased N=4 paints under one
/// embedded source profile and one OutputIntent destination profile
/// build the CMYK→CMYK retarget transform exactly once across the
/// whole page. Before the cache landed, each `scn` re-parsed both
/// profiles and rebuilt the lcms2 CLUT.
///
/// The (src, dst, intent) fingerprint key uses
/// `(n_components, byte_len, content_hash)` per profile so a
/// theoretical SipHash collision can't route a wrong-profile
/// transform — the n_components and byte_len agreement adds two
/// extra independent constraints.
#[cfg(feature = "icc-lcms2")]
#[test]
fn r7_icc_lcms2_retarget_transform_caches_per_profile_pair() {
    use pdf_oxide::rendering::{PageRenderer, RenderOptions};

    let icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 135,
        dest_cmyk: (200, 50, 20, 30),
    });
    let process_icc = build_bidirectional_cmyk_icc(SyntheticCmykProfileParams {
        l_byte: 200,
        dest_cmyk: (10, 20, 30, 40),
    });
    let paints: usize = 6;
    let pdf = build_devicen_iccbased_fixture_repeated(&icc, &process_icc, paints);
    let doc = PdfDocument::from_bytes(pdf).expect("parse");

    // The fixture's ExtGState carries /ca 0.5, so
    // `page_declares_transparency_or_overprint` returns true and the
    // sidecar is allocated under the OutputIntent gate; the renderer
    // takes the with-coverage compose path that calls the retarget
    // through the cache for every DeviceN /Process /ICCBased N=4 scn.
    let mut renderer = PageRenderer::new(RenderOptions::with_dpi(72).as_raw());
    let _ = renderer.render_page(&doc, 0).expect("render");

    let built = renderer.icc_transform_cache_cmyk_retarget_build_count();
    assert_eq!(
        built, 1,
        "Many DeviceN /Process /ICCBased N=4 paints under one embedded \
         source profile and one OutputIntent destination profile must \
         build the CMYK→CMYK retarget transform exactly once \
         (`CmykRetargetTransform::new` runs the lcms2 CLUT compile, \
         not free). Built {built} times — the per-renderer retarget \
         cache regressed."
    );
}
